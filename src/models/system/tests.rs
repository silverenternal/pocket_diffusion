#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::interaction::InteractionDiagnosticProvenance;
    use crate::types::{Atom, AtomType, Ligand, Pocket};
    use crate::{config::ResearchConfig, data::synthetic_phase1_examples};
    use tch::{nn, Device, Kind, Tensor};

    fn synthetic_custom_example(
        example_id: &str,
        protein_id: &str,
        ligand_atoms: i64,
        pocket_atoms: i64,
        coord_offset: f64,
        ligand_feature_dim: i64,
    ) -> MolecularExample {
        let atoms = (0..ligand_atoms)
            .map(|index| Atom {
                coords: [
                    coord_offset + index as f64 * 0.7,
                    (index as f64 % 3.0) * 0.4,
                    (index as f64 % 2.0) * 0.2,
                ],
                atom_type: if index % 3 == 0 {
                    AtomType::Carbon
                } else if index % 3 == 1 {
                    AtomType::Nitrogen
                } else {
                    AtomType::Oxygen
                },
                index: index as usize,
            })
            .collect::<Vec<_>>();
        let bonds = if ligand_atoms > 1 {
            (0..ligand_atoms - 1)
                .map(|index| (index as usize, (index + 1) as usize))
                .collect()
        } else {
            Vec::new()
        };
        let ligand = Ligand {
            atoms,
            bond_types: vec![1; bonds.len()],
            bonds,
            fingerprint: None,
        };
        let pocket_feature_atoms = (0..pocket_atoms)
            .map(|index| Atom {
                coords: [
                    coord_offset + 0.25 + index as f64 * 0.4,
                    coord_offset + 0.15 - index as f64 * 0.2,
                    index as f64 * 0.3,
                ],
                atom_type: if index % 3 == 0 {
                    AtomType::Nitrogen
                } else if index % 3 == 1 {
                    AtomType::Carbon
                } else {
                    AtomType::Oxygen
                },
                index: index as usize,
            })
            .collect::<Vec<_>>();
        let pocket = Pocket {
            name: protein_id.to_string(),
            atoms: pocket_feature_atoms,
        };
        MolecularExample::from_legacy(example_id, protein_id, &ligand, &pocket)
            .with_pocket_feature_dim(ligand_feature_dim)
    }

    fn generation_gate_mean(summary: GenerationGateSummary) -> f64 {
        (summary.topo_from_geo
            + summary.topo_from_pocket
            + summary.geo_from_topo
            + summary.geo_from_pocket
            + summary.pocket_from_topo
            + summary.pocket_from_geo)
            / 6.0
    }

    fn assert_sync_context_matches_single(
        from_single: &ResearchForward,
        from_batch: &ResearchForward,
    ) {
        let single = &from_single.sync_context;
        let batch = &from_batch.sync_context;
        assert_eq!(single.example_id, batch.example_id);
        assert_eq!(single.protein_id, batch.protein_id);
        assert_eq!(single.ligand_atom_count, batch.ligand_atom_count);
        assert_eq!(single.pocket_atom_count, batch.pocket_atom_count);
        assert_eq!(single.topology_mask_count, batch.topology_mask_count);
        assert_eq!(single.geometry_mask_count, batch.geometry_mask_count);
        assert_eq!(single.pocket_mask_count, batch.pocket_mask_count);
        assert_eq!(single.topology_slot_count, batch.topology_slot_count);
        assert_eq!(single.geometry_slot_count, batch.geometry_slot_count);
        assert_eq!(single.pocket_slot_count, batch.pocket_slot_count);
        assert_eq!(
            single.coordinate_frame_origin,
            batch.coordinate_frame_origin
        );
        assert_eq!(single.device_kind, batch.device_kind);
    }

    fn tensor_abs_sum(tensor: &Tensor) -> f64 {
        tensor.abs().sum(Kind::Float).double_value(&[])
    }

    fn flow_full_branch_config() -> ResearchConfig {
        let mut config = ResearchConfig::default();
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.geometry_only = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = vec![
            crate::config::FlowBranchKind::Geometry,
            crate::config::FlowBranchKind::AtomType,
            crate::config::FlowBranchKind::Bond,
            crate::config::FlowBranchKind::Topology,
            crate::config::FlowBranchKind::PocketContext,
        ];
        config
    }

    fn assert_optional_f64_close(
        name: &str,
        left: Option<f64>,
        right: Option<f64>,
        tolerance: f64,
    ) {
        assert_eq!(left.is_some(), right.is_some(), "{name} presence differs");
        if let (Some(left), Some(right)) = (left, right) {
            let delta = (left - right).abs();
            assert!(
                delta <= tolerance,
                "{name} delta {delta} exceeded tolerance {tolerance}"
            );
        }
    }

    fn assert_tensor_equal(name: &str, left: &Tensor, right: &Tensor) {
        assert_eq!(
            left.size(),
            right.size(),
            "{name} shape changed under target-ligand perturbation"
        );
        if left.numel() == 0 {
            return;
        }
        let mismatches = left
            .ne_tensor(right)
            .to_kind(Kind::Int64)
            .sum(Kind::Int64)
            .int64_value(&[]);
        assert_eq!(
            mismatches, 0,
            "{name} changed under target-ligand perturbation"
        );
    }

    fn perturb_target_ligand_tensors(example: &MolecularExample) -> MolecularExample {
        let mut mutated = example.clone();
        mutated.topology.atom_types = Tensor::zeros_like(&mutated.topology.atom_types);
        mutated.topology.adjacency = Tensor::ones_like(&mutated.topology.adjacency)
            - Tensor::eye(
                mutated.topology.adjacency.size()[0],
                (Kind::Float, mutated.topology.adjacency.device()),
            );
        mutated.geometry.coords = &mutated.geometry.coords + 37.0;
        mutated.geometry.pairwise_distances = &mutated.geometry.pairwise_distances * 3.0 + 1.0;
        mutated.decoder_supervision.target_atom_types =
            Tensor::zeros_like(&mutated.decoder_supervision.target_atom_types);
        mutated.decoder_supervision.corrupted_atom_types =
            Tensor::zeros_like(&mutated.decoder_supervision.corrupted_atom_types);
        mutated.decoder_supervision.atom_corruption_mask =
            Tensor::ones_like(&mutated.decoder_supervision.atom_corruption_mask);
        mutated.decoder_supervision.target_coords = &mutated.decoder_supervision.target_coords + 19.0;
        mutated.decoder_supervision.noisy_coords = &mutated.decoder_supervision.noisy_coords - 23.0;
        mutated.decoder_supervision.coordinate_noise =
            &mutated.decoder_supervision.coordinate_noise + 11.0;
        mutated.decoder_supervision.target_pairwise_distances =
            &mutated.decoder_supervision.target_pairwise_distances * 2.0 + 0.5;
        mutated
    }

    fn permute_decoder_target_rows(example: &MolecularExample, order: &[i64]) -> MolecularExample {
        let mut permuted = example.clone();
        let order = Tensor::from_slice(order).to_kind(Kind::Int64);
        permuted.decoder_supervision.target_coords = example
            .decoder_supervision
            .target_coords
            .index_select(0, &order);
        permuted.decoder_supervision.target_atom_types = example
            .decoder_supervision
            .target_atom_types
            .index_select(0, &order);
        permuted.decoder_supervision.noisy_coords =
            example.decoder_supervision.noisy_coords.index_select(0, &order);
        permuted.decoder_supervision.coordinate_noise = example
            .decoder_supervision
            .coordinate_noise
            .index_select(0, &order);
        permuted.decoder_supervision.corrupted_atom_types = example
            .decoder_supervision
            .corrupted_atom_types
            .index_select(0, &order);
        permuted.decoder_supervision.atom_corruption_mask = example
            .decoder_supervision
            .atom_corruption_mask
            .index_select(0, &order);
        permuted.decoder_supervision.target_pairwise_distances = example
            .decoder_supervision
            .target_pairwise_distances
            .index_select(0, &order)
            .index_select(1, &order);
        permuted
    }

    fn permute_ligand_target_order(example: &MolecularExample, order: &[i64]) -> MolecularExample {
        let mut permuted = permute_decoder_target_rows(example, order);
        let order_tensor = Tensor::from_slice(order).to_kind(Kind::Int64);
        permuted.topology.atom_types = example.topology.atom_types.index_select(0, &order_tensor);
        permuted.topology.adjacency = example
            .topology
            .adjacency
            .index_select(0, &order_tensor)
            .index_select(1, &order_tensor);
        permuted.topology.chemistry_roles.role_vectors = example
            .topology
            .chemistry_roles
            .role_vectors
            .index_select(0, &order_tensor);
        permuted.topology.chemistry_roles.availability = example
            .topology
            .chemistry_roles
            .availability
            .index_select(0, &order_tensor);
        permuted.geometry.coords = example.geometry.coords.index_select(0, &order_tensor);
        permuted.geometry.pairwise_distances = example
            .geometry
            .pairwise_distances
            .index_select(0, &order_tensor)
            .index_select(1, &order_tensor);

        let mut inverse = vec![0_i64; order.len()];
        for (new_index, old_index) in order.iter().copied().enumerate() {
            if old_index >= 0 && (old_index as usize) < inverse.len() {
                inverse[old_index as usize] = new_index as i64;
            }
        }
        let edge_count = example
            .topology
            .edge_index
            .size()
            .get(1)
            .copied()
            .unwrap_or(0)
            .max(0);
        let mut remapped_src = Vec::with_capacity(edge_count as usize);
        let mut remapped_dst = Vec::with_capacity(edge_count as usize);
        for edge_ix in 0..edge_count {
            let src = example.topology.edge_index.int64_value(&[0, edge_ix]);
            let dst = example.topology.edge_index.int64_value(&[1, edge_ix]);
            remapped_src.push(inverse[src as usize]);
            remapped_dst.push(inverse[dst as usize]);
        }
        permuted.topology.edge_index = if edge_count == 0 {
            Tensor::zeros([2, 0], (Kind::Int64, example.topology.edge_index.device()))
        } else {
            Tensor::stack(
                &[
                    Tensor::from_slice(&remapped_src).to_kind(Kind::Int64),
                    Tensor::from_slice(&remapped_dst).to_kind(Kind::Int64),
                ],
                0,
            )
            .to_device(example.topology.edge_index.device())
        };
        permuted
    }

    fn translate_decoder_target_coords(
        example: &MolecularExample,
        delta: [f32; 3],
    ) -> MolecularExample {
        let mut translated = example.clone();
        let delta = Tensor::from_slice(&delta)
            .reshape([1, 3])
            .to_device(example.decoder_supervision.target_coords.device());
        translated.decoder_supervision.target_coords =
            &translated.decoder_supervision.target_coords + delta;
        translated
    }

    fn flow_velocity_loss_value(flow: &FlowMatchingTrainingRecord) -> f64 {
        let per_atom = (&flow.predicted_velocity - &flow.target_velocity)
            .pow_tensor_scalar(2.0)
            .mean_dim([1].as_slice(), false, Kind::Float);
        let denom = flow.atom_mask.sum(Kind::Float).clamp_min(1.0);
        ((per_atom * &flow.atom_mask).sum(Kind::Float) / denom).double_value(&[])
    }

    fn flow_x0_coords(flow: &FlowMatchingTrainingRecord) -> Tensor {
        &flow.sampled_coords - &flow.target_velocity * flow.t
    }

    fn atom_type_loss_value(molecular: &MolecularFlowTrainingRecord) -> f64 {
        let log_probs = molecular.atom_type_logits.log_softmax(-1, Kind::Float);
        let target = molecular
            .target_atom_types
            .to_kind(Kind::Int64)
            .clamp(
                0,
                molecular
                    .atom_type_logits
                    .size()
                    .get(1)
                    .copied()
                    .unwrap_or(1)
                    .saturating_sub(1),
            );
        let nll = -log_probs
            .gather(1, &target.unsqueeze(1), false)
            .squeeze_dim(1);
        let denom = molecular.target_atom_mask.sum(Kind::Float).clamp_min(1.0);
        ((nll * &molecular.target_atom_mask).sum(Kind::Float) / denom).double_value(&[])
    }

    fn masked_bce_value(logits: &Tensor, target: &Tensor, mask: &Tensor) -> f64 {
        let target = target.to_kind(Kind::Float);
        let per_item = logits.clamp_min(0.0) - logits * &target + (-logits.abs()).exp().log1p();
        let denom = mask.sum(Kind::Float).clamp_min(1.0);
        ((per_item * mask).sum(Kind::Float) / denom).double_value(&[])
    }

    fn masked_pair_cross_entropy_value(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> f64 {
        let size = logits.size();
        let classes = size[2];
        let flat_logits = logits.reshape([size[0] * size[1], classes]);
        let flat_targets = targets.reshape([size[0] * size[1]]).to_kind(Kind::Int64);
        let flat_mask = mask.reshape([size[0] * size[1]]).to_kind(Kind::Float);
        let per_pair = flat_logits.cross_entropy_loss::<Tensor>(
            &flat_targets,
            None,
            tch::Reduction::None,
            -100,
            0.0,
        );
        let denom = flat_mask.sum(Kind::Float).clamp_min(1.0);
        ((per_pair * flat_mask).sum(Kind::Float) / denom).double_value(&[])
    }

    fn bond_loss_value(molecular: &MolecularFlowTrainingRecord) -> f64 {
        let bond_exists = masked_bce_value(
            &molecular.bond_exists_logits,
            &molecular.target_adjacency,
            &molecular.pair_mask,
        );
        let positive_pair_mask = &molecular.pair_mask * molecular.target_adjacency.clamp(0.0, 1.0);
        let bond_type = masked_pair_cross_entropy_value(
            &molecular.bond_type_logits,
            &molecular.target_bond_types,
            &positive_pair_mask,
        );
        bond_exists + bond_type
    }

    fn topology_loss_value(molecular: &MolecularFlowTrainingRecord) -> f64 {
        masked_bce_value(
            &molecular.topology_logits,
            &molecular.target_topology,
            &molecular.pair_mask,
        )
    }

    fn assert_paths_are_per_example_and_match_single(
        single: &ResearchForward,
        from_batch: &ResearchForward,
    ) {
        assert_eq!(
            single.interaction_diagnostics.topo_from_geo.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert_eq!(
            from_batch.interaction_diagnostics.topo_from_geo.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert!(
            (single.interaction_diagnostics.topo_from_geo.gate_mean
                - from_batch.interaction_diagnostics.topo_from_geo.gate_mean)
                .abs()
                < 1e-6
        );
        assert_eq!(
            single.interaction_diagnostics.topo_from_pocket.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert_eq!(
            from_batch
                .interaction_diagnostics
                .topo_from_pocket
                .provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert!(
            (single.interaction_diagnostics.topo_from_pocket.gate_mean
                - from_batch
                    .interaction_diagnostics
                    .topo_from_pocket
                    .gate_mean)
                .abs()
                < 1e-6
        );
        assert_eq!(
            single.interaction_diagnostics.geo_from_topo.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert_eq!(
            from_batch.interaction_diagnostics.geo_from_topo.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert!(
            (single.interaction_diagnostics.geo_from_topo.gate_mean
                - from_batch.interaction_diagnostics.geo_from_topo.gate_mean)
                .abs()
                < 1e-6
        );
        assert_eq!(
            single.interaction_diagnostics.geo_from_pocket.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert_eq!(
            from_batch
                .interaction_diagnostics
                .geo_from_pocket
                .provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert!(
            (single.interaction_diagnostics.geo_from_pocket.gate_mean
                - from_batch.interaction_diagnostics.geo_from_pocket.gate_mean)
                .abs()
                < 1e-6
        );
        assert_optional_f64_close(
            "geo_from_pocket bias mean",
            single.interaction_diagnostics.geo_from_pocket.bias_mean,
            from_batch.interaction_diagnostics.geo_from_pocket.bias_mean,
            1e-6,
        );
        assert_optional_f64_close(
            "geo_from_pocket bias scale",
            single.interaction_diagnostics.geo_from_pocket.bias_scale,
            from_batch
                .interaction_diagnostics
                .geo_from_pocket
                .bias_scale,
            1e-12,
        );
        assert_optional_f64_close(
            "geo_from_pocket chemistry coverage",
            single
                .interaction_diagnostics
                .geo_from_pocket
                .chemistry_role_coverage,
            from_batch
                .interaction_diagnostics
                .geo_from_pocket
                .chemistry_role_coverage,
            1e-6,
        );
        assert_eq!(
            single.interaction_diagnostics.pocket_from_topo.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert_eq!(
            from_batch
                .interaction_diagnostics
                .pocket_from_topo
                .provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert!(
            (single.interaction_diagnostics.pocket_from_topo.gate_mean
                - from_batch
                    .interaction_diagnostics
                    .pocket_from_topo
                    .gate_mean)
                .abs()
                < 1e-6
        );
        assert_eq!(
            single.interaction_diagnostics.pocket_from_geo.provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert_eq!(
            from_batch
                .interaction_diagnostics
                .pocket_from_geo
                .provenance,
            InteractionDiagnosticProvenance::PerExample
        );
        assert!(
            (single.interaction_diagnostics.pocket_from_geo.gate_mean
                - from_batch.interaction_diagnostics.pocket_from_geo.gate_mean)
                .abs()
                < 1e-6
        );
        assert_optional_f64_close(
            "pocket_from_geo bias mean",
            single.interaction_diagnostics.pocket_from_geo.bias_mean,
            from_batch.interaction_diagnostics.pocket_from_geo.bias_mean,
            1e-6,
        );
        assert_optional_f64_close(
            "pocket_from_geo bias scale",
            single.interaction_diagnostics.pocket_from_geo.bias_scale,
            from_batch
                .interaction_diagnostics
                .pocket_from_geo
                .bias_scale,
            1e-12,
        );
        assert_optional_f64_close(
            "pocket_from_geo chemistry coverage",
            single
                .interaction_diagnostics
                .pocket_from_geo
                .chemistry_role_coverage,
            from_batch
                .interaction_diagnostics
                .pocket_from_geo
                .chemistry_role_coverage,
            1e-6,
        );
    }

    #[test]
    fn model_stack_wrappers_preserve_separate_modality_boundaries() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let encodings = system.encode_example(&example);
        let slots = system.decompose_modalities(&encodings);
        let interactions = system.interact_modalities(&slots);

        assert_eq!(encodings.topology.pooled_embedding.size()[0], config.model.hidden_dim);
        assert_eq!(encodings.geometry.pooled_embedding.size()[0], config.model.hidden_dim);
        assert_eq!(encodings.pocket.pooled_embedding.size()[0], config.model.hidden_dim);
        assert_eq!(slots.topology.slots.size()[0], config.model.num_slots);
        assert_eq!(slots.geometry.slots.size()[0], config.model.num_slots);
        assert_eq!(slots.pocket.slots.size()[0], config.model.num_slots);
        assert_eq!(
            interactions.topo_from_geo.attended_tokens.size(),
            slots.topology.slots.size()
        );
        assert_eq!(
            interactions.geo_from_pocket.attended_tokens.size(),
            slots.geometry.slots.size()
        );
        assert_eq!(
            interactions.pocket_from_topo.attended_tokens.size(),
            slots.pocket.slots.size()
        );
    }

    #[test]
    fn optimizer_forward_record_is_available_without_rollout_artifacts() {
        let config = flow_full_branch_config();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let optimizer_record = system.forward_example_optimizer_record(&example);
        assert_eq!(
            optimizer_record.generation_mode,
            crate::config::GenerationModeConfig::FlowRefinement
        );
        assert_eq!(
            optimizer_record.decoded.coordinate_deltas.size(),
            example.decoder_supervision.target_coords.size()
        );
        assert!(optimizer_record.flow_matching.is_some());
        assert!(optimizer_record.interaction_context.flow_t.is_some());
        assert_eq!(
            optimizer_record.sync_context.example_id,
            example.example_id
        );
    }

    #[test]
    fn generation_mode_contracts_expose_component_activation_boundaries() {
        for mode in crate::config::GenerationModeConfig::ALL {
            let contract = mode.compatibility_contract();
            assert_eq!(contract.generation_mode, mode);
            assert!(contract.supported);
            assert!(!contract.compatible_primary_objectives.is_empty());
            assert!(!contract.compatible_backend_families.is_empty());
            assert!(!contract.decoder_capability_label.is_empty());
        }

        let de_novo = crate::config::GenerationModeConfig::DeNovoInitialization
            .compatibility_contract();
        assert!(!de_novo.target_ligand_topology);
        assert!(!de_novo.target_ligand_geometry);
        assert_eq!(de_novo.decoder_capability_label, "pocket_conditioned_graph_flow");
        assert_eq!(
            de_novo.target_ligand_coordinate_availability,
            "target_supervision_only"
        );
    }

    fn assert_tensor_close(name: &str, left: &Tensor, right: &Tensor, tolerance: f64) {
        assert_eq!(left.size(), right.size());
        if left.numel() == 0 {
            return;
        }
        let max_delta = (left - right).abs().max().double_value(&[]);
        assert!(
            max_delta <= tolerance,
            "{name} max delta {max_delta} exceeded tolerance {tolerance}"
        );
    }

    fn assert_forward_batch_slice_matches_single(
        single: &ResearchForward,
        from_batch: &ResearchForward,
    ) {
        assert_sync_context_matches_single(single, from_batch);

        assert_tensor_close(
            "topology token embeddings",
            &single.encodings.topology.token_embeddings,
            &from_batch.encodings.topology.token_embeddings,
            1e-5,
        );
        assert_tensor_close(
            "geometry token embeddings",
            &single.encodings.geometry.token_embeddings,
            &from_batch.encodings.geometry.token_embeddings,
            1e-5,
        );
        assert_tensor_close(
            "pocket token embeddings",
            &single.encodings.pocket.token_embeddings,
            &from_batch.encodings.pocket.token_embeddings,
            1e-5,
        );
        assert_tensor_close(
            "topology slots",
            &single.slots.topology.slots,
            &from_batch.slots.topology.slots,
            1e-5,
        );
        assert_tensor_close(
            "geometry slots",
            &single.slots.geometry.slots,
            &from_batch.slots.geometry.slots,
            1e-5,
        );
        assert_tensor_close(
            "pocket slots",
            &single.slots.pocket.slots,
            &from_batch.slots.pocket.slots,
            1e-5,
        );
        assert_tensor_close(
            "geo_from_pocket gate",
            &single.interactions.geo_from_pocket.gate,
            &from_batch.interactions.geo_from_pocket.gate,
            1e-5,
        );
        assert_tensor_close(
            "pocket_from_geo attention",
            &single.interactions.pocket_from_geo.attention_weights,
            &from_batch.interactions.pocket_from_geo.attention_weights,
            5e-3,
        );
        assert_eq!(
            single.generation.state.example_id,
            from_batch.generation.state.example_id
        );
        assert_eq!(
            single.generation.state.protein_id,
            from_batch.generation.state.protein_id
        );
        assert_tensor_close(
            "generation atom types",
            &single.generation.state.partial_ligand.atom_types,
            &from_batch.generation.state.partial_ligand.atom_types,
            0.0,
        );
        assert_tensor_close(
            "generation coordinates",
            &single.generation.state.partial_ligand.coords,
            &from_batch.generation.state.partial_ligand.coords,
            1e-6,
        );
        assert_tensor_close(
            "generation atom mask",
            &single.generation.state.partial_ligand.atom_mask,
            &from_batch.generation.state.partial_ligand.atom_mask,
            0.0,
        );
        assert_eq!(
            single.generation.state.partial_ligand.step_index,
            from_batch.generation.state.partial_ligand.step_index
        );
        assert_tensor_close(
            "generation topology context",
            &single.generation.state.topology_context,
            &from_batch.generation.state.topology_context,
            1e-4,
        );
        assert_tensor_close(
            "generation geometry context",
            &single.generation.state.geometry_context,
            &from_batch.generation.state.geometry_context,
            1e-4,
        );
        assert_tensor_close(
            "generation pocket context",
            &single.generation.state.pocket_context,
            &from_batch.generation.state.pocket_context,
            1e-4,
        );
        assert_tensor_close(
            "generation topology slot mask",
            &single.generation.state.topology_slot_mask,
            &from_batch.generation.state.topology_slot_mask,
            0.0,
        );
        assert_tensor_close(
            "generation geometry slot mask",
            &single.generation.state.geometry_slot_mask,
            &from_batch.generation.state.geometry_slot_mask,
            0.0,
        );
        assert_tensor_close(
            "generation pocket slot mask",
            &single.generation.state.pocket_slot_mask,
            &from_batch.generation.state.pocket_slot_mask,
            0.0,
        );
        assert_eq!(
            single.generation.decoded.atom_type_logits.size(),
            from_batch.generation.decoded.atom_type_logits.size()
        );
        assert_tensor_close(
            "decoded atom type logits",
            &single.generation.decoded.atom_type_logits,
            &from_batch.generation.decoded.atom_type_logits,
            1e-4,
        );
        assert_tensor_close(
            "decoded coordinate deltas",
            &single.generation.decoded.coordinate_deltas,
            &from_batch.generation.decoded.coordinate_deltas,
            1e-4,
        );
        assert_tensor_close(
            "decoded stop logit",
            &single.generation.decoded.stop_logit,
            &from_batch.generation.decoded.stop_logit,
            1e-4,
        );
        assert_tensor_close(
            "decoded generation embedding",
            &single.generation.decoded.generation_embedding,
            &from_batch.generation.decoded.generation_embedding,
            1e-4,
        );
    }

    #[test]
    fn batched_backbone_matches_single_example_backbone_shapes() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = synthetic_phase1_examples()
            .into_iter()
            .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
            .collect::<Vec<_>>();

        let single = system.forward_example(&examples[0]);
        let (_, batched) = system.forward_batch(&examples[..2]);
        let from_batch = &batched[0];

        assert_eq!(
            single.encodings.topology.token_embeddings.size(),
            from_batch.encodings.topology.token_embeddings.size()
        );
        assert_eq!(
            single.slots.geometry.slots.size(),
            from_batch.slots.geometry.slots.size()
        );
        assert_eq!(
            single.interactions.geo_from_pocket.attention_weights.size(),
            from_batch
                .interactions
                .geo_from_pocket
                .attention_weights
                .size()
        );
        assert_eq!(
            from_batch.generation.decoded.atom_type_logits.size()[0],
            examples[0].topology.atom_types.size()[0]
        );
    }

    #[test]
    fn configured_encoder_paths_match_between_single_and_batched_slices() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = vec![
            synthetic_custom_example(
                "parity-ex-0",
                "protein-a",
                4,
                4,
                0.0,
                config.model.pocket_feature_dim,
            ),
            synthetic_custom_example(
                "parity-ex-1",
                "protein-b",
                3,
                6,
                1.2,
                config.model.pocket_feature_dim,
            ),
        ];

        let single = [
            system.forward_example(&examples[0]),
            system.forward_example(&examples[1]),
        ];
        let (_, batched) = system.forward_batch(&examples);

        assert_forward_batch_slice_matches_single(&single[0], &batched[0]);
        assert_forward_batch_slice_matches_single(&single[1], &batched[1]);
    }

    #[test]
    fn forward_batch_matches_single() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = vec![
            synthetic_custom_example(
                "forward-parity-0",
                "protein-a",
                4,
                5,
                0.4,
                config.model.pocket_feature_dim,
            ),
            synthetic_custom_example(
                "forward-parity-1",
                "protein-b",
                3,
                6,
                1.4,
                config.model.pocket_feature_dim,
            ),
        ];

        let single = [
            system.forward_example(&examples[0]),
            system.forward_example(&examples[1]),
        ];
        let (_, batched) = system.forward_batch(&examples);

        assert_forward_batch_slice_matches_single(&single[0], &batched[0]);
        assert_forward_batch_slice_matches_single(&single[1], &batched[1]);
        assert_paths_are_per_example_and_match_single(&single[0], &batched[0]);
        assert_paths_are_per_example_and_match_single(&single[1], &batched[1]);
    }

    #[test]
    fn flow_time_conditioned_batch_forward_preserves_per_example_semantics() {
        let mut config = ResearchConfig::default();
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config
            .model
            .temporal_interaction_policy
            .flow_time_bucket_multipliers =
            vec![crate::config::InteractionPathFlowTimeBucketMultiplier {
                path: "geo_from_pocket".to_string(),
                start_t: 0.0,
                end_t: 1.0,
                multiplier: 0.75,
            }];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = vec![
            synthetic_custom_example(
                "flow-parity-0",
                "protein-c",
                4,
                5,
                0.3,
                config.model.pocket_feature_dim,
            ),
            synthetic_custom_example(
                "flow-parity-1",
                "protein-d",
                2,
                4,
                1.7,
                config.model.pocket_feature_dim,
            ),
        ];

        let single = [
            system.forward_example(&examples[0]),
            system.forward_example(&examples[1]),
        ];
        let (_, batched) = system.forward_batch(&examples);

        assert_forward_batch_slice_matches_single(&single[0], &batched[0]);
        assert_forward_batch_slice_matches_single(&single[1], &batched[1]);
        for forward in &batched {
            assert!(forward.sync_context.flow_t.is_some());
            assert!(forward.generation.flow_matching.is_some());
            assert!(forward
                .interaction_diagnostics
                .geo_from_pocket
                .flow_t
                .is_some());
            assert!(forward
                .interaction_diagnostics
                .geo_from_pocket
                .flow_time_bucket
                .is_some());
        }
    }

    #[test]
    fn flow_matching_records_corrupted_geometry_x0_source_by_default() {
        let mut config = ResearchConfig::default();
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example(&example);
        assert_eq!(
            forward.generation.rollout.flow_x0_source.as_deref(),
            Some("target_ligand_corrupted_geometry")
        );
        assert_eq!(
            forward.generation.flow_matching.unwrap().x0_source,
            "target_ligand_corrupted_geometry"
        );
    }

    #[test]
    fn flow_matching_records_deterministic_noise_x0_source_when_configured() {
        let mut config = ResearchConfig::default();
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.use_corrupted_x0 = false;
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example(&example);
        assert_eq!(
            forward.generation.rollout.flow_x0_source.as_deref(),
            Some("deterministic_gaussian_noise")
        );
        assert_eq!(
            forward.generation.flow_matching.unwrap().x0_source,
            "deterministic_gaussian_noise"
        );
    }

    #[test]
    fn de_novo_full_molecular_flow_uses_pocket_scaffold_and_records_all_branches() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 7;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 7;
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.geometry_only = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = vec![
            crate::config::FlowBranchKind::Geometry,
            crate::config::FlowBranchKind::AtomType,
            crate::config::FlowBranchKind::Bond,
            crate::config::FlowBranchKind::Topology,
            crate::config::FlowBranchKind::PocketContext,
        ];
        config
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy =
            crate::config::FlowTargetAlignmentPolicy::HungarianDistance;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "de-novo-flow",
            "protein-de-novo",
            3,
            28,
            1.1,
            config.model.pocket_feature_dim,
        );

        let forward = system.forward_example(&example);
        assert_eq!(
            forward.generation.generation_mode,
            crate::config::GenerationModeConfig::DeNovoInitialization
        );
        assert_eq!(forward.generation.state.partial_ligand.atom_types.size(), vec![7]);
        assert_ne!(
            forward.generation.state.partial_ligand.atom_types.size(),
            example.decoder_supervision.target_atom_types.size()
        );
        assert_eq!(
            forward.generation.rollout.atom_count_source,
            "pocket_conditioned_atom_count_policy"
        );
        assert_eq!(
            forward.generation.rollout.decoder_capability,
            "pocket_conditioned_graph_flow"
        );
        let flow = forward.generation.flow_matching.as_ref().unwrap();
        assert_eq!(
            flow.flow_contract_version,
            crate::models::MOLECULAR_FLOW_CONTRACT_VERSION
        );
        assert_eq!(flow.branch_weights.geometry, 1.0);
        let molecular = flow.molecular.as_ref().unwrap();
        assert!(molecular.full_branch_set_enabled);
        assert_eq!(molecular.target_alignment_policy, "hungarian_distance");
        assert_eq!(molecular.atom_type_logits.size()[0], 7);
        assert_eq!(molecular.target_atom_mask.size(), vec![7]);
        assert_eq!(
            molecular
                .target_atom_mask
                .sum(Kind::Float)
                .double_value(&[]),
            3.0
        );
        assert_eq!(molecular.bond_exists_logits.size(), vec![7, 7]);
        assert_eq!(molecular.topology_logits.size(), vec![7, 7]);
        assert_eq!(molecular.pocket_contact_logits.size(), vec![7]);
        assert_eq!(molecular.target_pocket_contacts.size(), vec![7]);
        assert_eq!(molecular.pocket_interaction_mask.size(), vec![7]);
        assert_eq!(
            molecular.pocket_branch_target_family,
            "pocket_interaction_profile"
        );
        assert_eq!(
            molecular.pocket_context_reconstruction_role,
            "context_drift_diagnostic"
        );
        assert_eq!(molecular.pocket_context_reconstruction.size()[1], config.model.hidden_dim);
        assert_eq!(
            forward.generation.rollout.atom_count_prior_provenance,
            "pocket_volume"
        );
        assert!(forward
            .generation
            .rollout
            .steps
            .iter()
            .any(|step| step
                .flow_diagnostics
                .contains_key("molecular_flow_atom_logit_mean")));
        let final_step = forward.generation.rollout.steps.last().unwrap();
        assert_eq!(final_step.native_bonds.len(), final_step.native_bond_types.len());
        assert!(!final_step.constrained_native_bonds.is_empty());
        assert_eq!(
            final_step.constrained_native_bonds.len(),
            final_step.constrained_native_bond_types.len()
        );
        assert_eq!(
            final_step.native_graph_provenance.raw_logits_layer,
            "raw_molecular_flow_logits"
        );
        assert_eq!(
            final_step.native_graph_provenance.raw_native_extraction_layer,
            "raw_native_graph_extraction"
        );
        assert_eq!(
            final_step.native_graph_provenance.constrained_graph_layer,
            "constrained_native_graph"
        );
        assert_eq!(
            final_step.native_graph_provenance.raw_native_bond_count,
            final_step.native_bonds.len()
        );
        assert_eq!(
            final_step.native_graph_provenance.constrained_bond_count,
            final_step.constrained_native_bonds.len()
        );
        assert!(final_step
            .flow_diagnostics
            .contains_key("molecular_flow_native_component_count"));
        assert!(final_step
            .flow_diagnostics
            .contains_key("molecular_flow_native_score_threshold"));
        assert!(final_step
            .flow_diagnostics
            .contains_key("molecular_flow_native_graph_guardrail_trigger_count"));
        assert_eq!(
            final_step.flow_diagnostics
                ["molecular_flow_native_valence_violation_fraction"],
            0.0
        );
    }

    #[test]
    fn flow_matching_objective_targets_are_invariant_to_target_atom_permutation() {
        let mut config = flow_full_branch_config();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 3;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 3;
        config.generation_method.flow_matching.use_corrupted_x0 = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy =
            crate::config::FlowTargetAlignmentPolicy::HungarianDistance;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "flow-permutation",
            "protein-flow-permutation",
            3,
            10,
            2.5,
            config.model.pocket_feature_dim,
        );
        let permuted = permute_decoder_target_rows(&example, &[2, 0, 1]);

        let baseline = system.forward_example_optimizer_record(&example);
        let permuted = system.forward_example_optimizer_record(&permuted);
        let baseline_flow = baseline.flow_matching.as_ref().unwrap();
        let permuted_flow = permuted.flow_matching.as_ref().unwrap();
        assert_eq!(baseline_flow.target_matching_policy, "hungarian_distance");
        assert_eq!(permuted_flow.target_matching_policy, "hungarian_distance");
        assert_tensor_close(
            "matched geometry target velocity",
            &baseline_flow.target_velocity,
            &permuted_flow.target_velocity,
            1.0e-5,
        );
        let velocity_delta =
            (flow_velocity_loss_value(baseline_flow) - flow_velocity_loss_value(permuted_flow))
                .abs();
        assert!(
            velocity_delta <= 1.0e-6,
            "matched geometry loss changed after target permutation by {velocity_delta}"
        );

        let baseline_molecular = baseline_flow.molecular.as_ref().unwrap();
        let permuted_molecular = permuted_flow.molecular.as_ref().unwrap();
        assert_eq!(
            baseline_molecular.target_matching_policy,
            "hungarian_distance"
        );
        assert_eq!(
            permuted_molecular.target_matching_policy,
            "hungarian_distance"
        );
        assert_tensor_equal(
            "matched atom-type targets",
            &baseline_molecular.target_atom_types,
            &permuted_molecular.target_atom_types,
        );
        let atom_type_delta = (atom_type_loss_value(baseline_molecular)
            - atom_type_loss_value(permuted_molecular))
        .abs();
        assert!(
            atom_type_delta <= 1.0e-6,
            "matched atom-type loss changed after target permutation by {atom_type_delta}"
        );
    }

    #[test]
    fn bond_topology_flow_targets_are_invariant_to_target_atom_permutation() {
        let mut config = flow_full_branch_config();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 3;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 3;
        config.generation_method.flow_matching.use_corrupted_x0 = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy =
            crate::config::FlowTargetAlignmentPolicy::HungarianDistance;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "flow-square-permutation",
            "protein-flow-square-permutation",
            3,
            10,
            2.5,
            config.model.pocket_feature_dim,
        );
        let permuted = permute_ligand_target_order(&example, &[2, 0, 1]);

        let baseline = system.forward_example_optimizer_record(&example);
        let permuted = system.forward_example_optimizer_record(&permuted);
        let baseline_molecular = baseline
            .flow_matching
            .as_ref()
            .unwrap()
            .molecular
            .as_ref()
            .unwrap();
        let permuted_molecular = permuted
            .flow_matching
            .as_ref()
            .unwrap()
            .molecular
            .as_ref()
            .unwrap();

        assert_tensor_close(
            "matched adjacency target",
            &baseline_molecular.target_adjacency,
            &permuted_molecular.target_adjacency,
            1.0e-6,
        );
        assert_tensor_close(
            "matched topology target",
            &baseline_molecular.target_topology,
            &permuted_molecular.target_topology,
            1.0e-6,
        );
        assert_tensor_equal(
            "matched bond-type target",
            &baseline_molecular.target_bond_types,
            &permuted_molecular.target_bond_types,
        );
        assert_tensor_close(
            "matched pair mask",
            &baseline_molecular.pair_mask,
            &permuted_molecular.pair_mask,
            1.0e-6,
        );
        let bond_delta =
            (bond_loss_value(baseline_molecular) - bond_loss_value(permuted_molecular)).abs();
        assert!(
            bond_delta <= 1.0e-6,
            "matched bond loss changed after target permutation by {bond_delta}"
        );
        let topology_delta =
            (topology_loss_value(baseline_molecular) - topology_loss_value(permuted_molecular))
                .abs();
        assert!(
            topology_delta <= 1.0e-6,
            "matched topology loss changed after target permutation by {topology_delta}"
        );
    }

    #[test]
    fn bond_topology_flow_targets_mask_unmatched_generated_rows() {
        let mut config = flow_full_branch_config();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 5;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 5;
        config.generation_method.flow_matching.use_corrupted_x0 = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy =
            crate::config::FlowTargetAlignmentPolicy::HungarianDistance;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "flow-square-padding",
            "protein-flow-square-padding",
            3,
            10,
            2.5,
            config.model.pocket_feature_dim,
        );

        let forward = system.forward_example_optimizer_record(&example);
        let molecular = forward
            .flow_matching
            .as_ref()
            .unwrap()
            .molecular
            .as_ref()
            .unwrap();

        assert_eq!(molecular.target_adjacency.size(), vec![5, 5]);
        assert_eq!(molecular.target_bond_types.size(), vec![5, 5]);
        assert_eq!(molecular.pocket_contact_logits.size(), vec![5]);
        assert_eq!(molecular.target_pocket_contacts.size(), vec![5]);
        assert_eq!(
            molecular
                .target_atom_mask
                .sum(Kind::Float)
                .double_value(&[]),
            3.0
        );
        assert_eq!(
            molecular.pair_mask.diag(0).sum(Kind::Float).double_value(&[]),
            0.0
        );
        assert_eq!(
            molecular
                .target_adjacency
                .diag(0)
                .abs()
                .sum(Kind::Float)
                .double_value(&[]),
            0.0
        );
        assert_tensor_close(
            "matched adjacency symmetry",
            &molecular.target_adjacency,
            &molecular.target_adjacency.transpose(0, 1),
            1.0e-6,
        );
        assert_tensor_equal(
            "matched bond-type symmetry",
            &molecular.target_bond_types,
            &molecular.target_bond_types.transpose(0, 1),
        );

        for row in 0..molecular.target_atom_mask.size()[0] {
            if molecular.target_atom_mask.double_value(&[row]) > 0.5 {
                continue;
            }
            let adjacency_mass = molecular
                .target_adjacency
                .get(row)
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                + molecular
                    .target_adjacency
                    .select(1, row)
                    .abs()
                    .sum(Kind::Float)
                    .double_value(&[]);
            let bond_nonzero = molecular
                .target_bond_types
                .get(row)
                .ne(0)
                .to_kind(Kind::Int64)
                .sum(Kind::Int64)
                .int64_value(&[])
                + molecular
                    .target_bond_types
                    .select(1, row)
                    .ne(0)
                    .to_kind(Kind::Int64)
                    .sum(Kind::Int64)
                    .int64_value(&[]);
            let pair_mask_mass = molecular
                .pair_mask
                .get(row)
                .sum(Kind::Float)
                .double_value(&[])
                + molecular
                    .pair_mask
                    .select(1, row)
                    .sum(Kind::Float)
                    .double_value(&[]);
            let pocket_mask = molecular.pocket_interaction_mask.double_value(&[row]);
            let pocket_target = molecular.target_pocket_contacts.double_value(&[row]);
            assert_eq!(adjacency_mass, 0.0);
            assert_eq!(bond_nonzero, 0);
            assert_eq!(pair_mask_mass, 0.0);
            assert_eq!(pocket_mask, 0.0);
            assert_eq!(pocket_target, 0.0);
        }
    }

    #[test]
    fn pocket_interaction_profile_loss_changes_with_ligand_pocket_geometry() {
        let mut config = flow_full_branch_config();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 3;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 3;
        config.generation_method.flow_matching.use_corrupted_x0 = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy =
            crate::config::FlowTargetAlignmentPolicy::HungarianDistance;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "flow-pocket-profile",
            "protein-flow-pocket-profile",
            3,
            10,
            0.25,
            config.model.pocket_feature_dim,
        );
        let translated = translate_decoder_target_coords(&example, [50.0, 50.0, 50.0]);

        let baseline = system.forward_example_optimizer_record(&example);
        let shifted = system.forward_example_optimizer_record(&translated);
        let baseline_molecular = baseline
            .flow_matching
            .as_ref()
            .unwrap()
            .molecular
            .as_ref()
            .unwrap();
        let shifted_molecular = shifted
            .flow_matching
            .as_ref()
            .unwrap()
            .molecular
            .as_ref()
            .unwrap();

        assert_eq!(
            baseline_molecular.pocket_branch_target_family,
            "pocket_interaction_profile"
        );
        assert_eq!(
            baseline_molecular.pocket_context_reconstruction_role,
            "context_drift_diagnostic"
        );
        assert!(
            baseline_molecular
                .target_pocket_contacts
                .sum(Kind::Float)
                .double_value(&[])
                > shifted_molecular
                    .target_pocket_contacts
                    .sum(Kind::Float)
                    .double_value(&[])
        );
        let baseline_loss = masked_bce_value(
            &baseline_molecular.pocket_contact_logits,
            &baseline_molecular.target_pocket_contacts,
            &baseline_molecular.pocket_interaction_mask,
        );
        let shifted_targets_loss = masked_bce_value(
            &baseline_molecular.pocket_contact_logits,
            &shifted_molecular.target_pocket_contacts,
            &baseline_molecular.pocket_interaction_mask,
        );
        let loss_delta = (baseline_loss - shifted_targets_loss).abs();
        assert!(
            loss_delta > 1.0e-6,
            "pocket interaction-profile loss should change when ligand-pocket contacts change"
        );
    }

    #[test]
    fn de_novo_optimizer_conditioning_ignores_target_ligand_tensors() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 6;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 6;
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.geometry_only = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = crate::config::FlowBranchKind::ALL.to_vec();
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let mutated = perturb_target_ligand_tensors(&example);

        let baseline = system.forward_example_optimizer_record(&example);
        let perturbed = system.forward_example_optimizer_record(&mutated);

        assert_tensor_close(
            "de_novo topology encoding",
            &baseline.encodings.topology.token_embeddings,
            &perturbed.encodings.topology.token_embeddings,
            1.0e-6,
        );
        assert_tensor_close(
            "de_novo geometry encoding",
            &baseline.encodings.geometry.token_embeddings,
            &perturbed.encodings.geometry.token_embeddings,
            1.0e-6,
        );
        assert_tensor_close(
            "de_novo pocket encoding",
            &baseline.encodings.pocket.token_embeddings,
            &perturbed.encodings.pocket.token_embeddings,
            1.0e-6,
        );
        assert_tensor_equal(
            "de_novo initial atom types",
            &baseline.state.partial_ligand.atom_types,
            &perturbed.state.partial_ligand.atom_types,
        );
        assert_tensor_close(
            "de_novo initial coordinates",
            &baseline.state.partial_ligand.coords,
            &perturbed.state.partial_ligand.coords,
            1.0e-6,
        );
        assert_tensor_close(
            "de_novo topology context",
            &baseline.state.topology_context,
            &perturbed.state.topology_context,
            1.0e-6,
        );
        assert_tensor_close(
            "de_novo geometry context",
            &baseline.state.geometry_context,
            &perturbed.state.geometry_context,
            1.0e-6,
        );
        assert_tensor_close(
            "de_novo pocket context",
            &baseline.state.pocket_context,
            &perturbed.state.pocket_context,
            1.0e-6,
        );
        assert_tensor_close(
            "de_novo decoded atom logits",
            &baseline.decoded.atom_type_logits,
            &perturbed.decoded.atom_type_logits,
            1.0e-6,
        );
        assert_tensor_close(
            "de_novo decoded coordinate deltas",
            &baseline.decoded.coordinate_deltas,
            &perturbed.decoded.coordinate_deltas,
            1.0e-6,
        );

        let rollout = system.forward_example(&example).generation.rollout;
        assert_eq!(
            rollout.conditioning_coordinate_frame,
            "pocket_centroid_centered_conditioning_no_target_ligand_frame"
        );
    }

    #[test]
    fn de_novo_flow_targets_can_change_without_changing_conditioning() {
        let mut config = flow_full_branch_config();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 3;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 3;
        config.generation_method.flow_matching.use_corrupted_x0 = true;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "de-novo-target-audit",
            "protein-de-novo-target-audit",
            3,
            10,
            1.7,
            config.model.pocket_feature_dim,
        );
        let mutated = perturb_target_ligand_tensors(&example);

        let baseline = system.forward_example_optimizer_record(&example);
        let perturbed = system.forward_example_optimizer_record(&mutated);

        assert_tensor_equal(
            "de_novo flow initial atom types",
            &baseline.state.partial_ligand.atom_types,
            &perturbed.state.partial_ligand.atom_types,
        );
        assert_tensor_close(
            "de_novo flow initial coordinates",
            &baseline.state.partial_ligand.coords,
            &perturbed.state.partial_ligand.coords,
            1.0e-6,
        );
        let baseline_flow = baseline.flow_matching.as_ref().unwrap();
        let perturbed_flow = perturbed.flow_matching.as_ref().unwrap();
        assert_eq!(
            baseline_flow.x0_source,
            "conditioning_scaffold_deterministic_noise_no_target_ligand"
        );
        assert_eq!(
            perturbed_flow.x0_source,
            "conditioning_scaffold_deterministic_noise_no_target_ligand"
        );
        assert_tensor_close(
            "de_novo flow x0",
            &flow_x0_coords(baseline_flow),
            &flow_x0_coords(perturbed_flow),
            1.0e-5,
        );
        let target_delta = (&baseline_flow.target_velocity - &perturbed_flow.target_velocity)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            target_delta > 1.0,
            "flow training target should change when decoder supervision target coordinates change"
        );
        let loss_delta =
            (flow_velocity_loss_value(baseline_flow) - flow_velocity_loss_value(perturbed_flow))
                .abs();
        assert!(
            loss_delta > 1.0e-6,
            "flow supervision loss should change when target labels change"
        );
    }

    #[test]
    fn de_novo_rollout_refresh_and_x0_ignore_target_ligand_tensors() {
        let mut config = flow_full_branch_config();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 3;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 3;
        config.data.generation_target.context_refresh_policy =
            crate::config::InferenceContextRefreshPolicy::EveryStep;
        config.generation_method.flow_matching.use_corrupted_x0 = true;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "de-novo-rollout-audit",
            "protein-de-novo-rollout-audit",
            3,
            12,
            0.8,
            config.model.pocket_feature_dim,
        );
        let mutated = perturb_target_ligand_tensors(&example);

        let baseline = system.forward_example(&example).generation.rollout;
        let perturbed = system.forward_example(&mutated).generation.rollout;

        assert_eq!(
            baseline.conditioning_coordinate_frame,
            "pocket_centroid_centered_conditioning_no_target_ligand_frame"
        );
        assert_eq!(
            baseline.flow_x0_source.as_deref(),
            Some("conditioning_scaffold_deterministic_noise_no_target_ligand")
        );
        assert_eq!(baseline.flow_x0_source, perturbed.flow_x0_source);
        assert_eq!(baseline.context_refresh_policy, "every_step");
        assert_eq!(baseline.refresh_count, baseline.executed_steps);
        assert_eq!(baseline.executed_steps, perturbed.executed_steps);
        assert_eq!(baseline.valence_guardrail_flag, perturbed.valence_guardrail_flag);
        for (left, right) in baseline.steps.iter().zip(perturbed.steps.iter()) {
            assert_eq!(left.atom_types, right.atom_types);
            assert_eq!(left.coords, right.coords);
            assert_eq!(left.valence_guardrail_flag, right.valence_guardrail_flag);
        }
    }

    #[test]
    fn de_novo_conditioning_and_flow_x0_ignore_source_frame_translation() {
        let mut config = flow_full_branch_config();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = 3;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = 3;
        config.generation_method.flow_matching.use_corrupted_x0 = true;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "de-novo-source-frame-shift",
            "protein-source-frame-shift",
            3,
            12,
            0.8,
            config.model.pocket_feature_dim,
        );
        let mut shifted_origin = example.clone();
        shifted_origin.coordinate_frame_origin = [
            example.coordinate_frame_origin[0] + 12.5,
            example.coordinate_frame_origin[1] - 7.25,
            example.coordinate_frame_origin[2] + 3.0,
        ];

        let baseline = system.forward_example_optimizer_record(&example);
        let shifted = system.forward_example_optimizer_record(&shifted_origin);

        assert_ne!(
            baseline.sync_context.coordinate_frame_origin,
            shifted.sync_context.coordinate_frame_origin
        );
        assert_tensor_close(
            "translated-source topology encoding",
            &baseline.encodings.topology.token_embeddings,
            &shifted.encodings.topology.token_embeddings,
            1.0e-6,
        );
        assert_tensor_close(
            "translated-source geometry encoding",
            &baseline.encodings.geometry.token_embeddings,
            &shifted.encodings.geometry.token_embeddings,
            1.0e-6,
        );
        assert_tensor_close(
            "translated-source pocket encoding",
            &baseline.encodings.pocket.token_embeddings,
            &shifted.encodings.pocket.token_embeddings,
            1.0e-6,
        );
        assert_tensor_close(
            "translated-source topology context",
            &baseline.state.topology_context,
            &shifted.state.topology_context,
            1.0e-6,
        );
        assert_tensor_close(
            "translated-source geometry context",
            &baseline.state.geometry_context,
            &shifted.state.geometry_context,
            1.0e-6,
        );
        assert_tensor_close(
            "translated-source pocket context",
            &baseline.state.pocket_context,
            &shifted.state.pocket_context,
            1.0e-6,
        );
        assert_tensor_close(
            "translated-source initial coords",
            &baseline.state.partial_ligand.coords,
            &shifted.state.partial_ligand.coords,
            1.0e-6,
        );
        assert_tensor_close(
            "translated-source flow x0",
            &flow_x0_coords(baseline.flow_matching.as_ref().unwrap()),
            &flow_x0_coords(shifted.flow_matching.as_ref().unwrap()),
            1.0e-5,
        );
    }

    #[test]
    fn molecular_flow_branch_schedule_controls_effective_weights() {
        let mut config = flow_full_branch_config();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .allow_zero_weight_branch_ablation = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .atom_type
            .start_step = 10;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .bond
            .enabled = false;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example_with_interaction_context(
            &example,
            InteractionExecutionContext {
                training_step: Some(0),
                ..InteractionExecutionContext::default()
            },
        );
        let flow = forward.generation.flow_matching.as_ref().unwrap();
        let molecular = flow.molecular.as_ref().unwrap();

        assert_eq!(flow.branch_weights.atom_type, 0.0);
        assert_eq!(molecular.branch_weights.atom_type, 0.0);
        assert_eq!(molecular.branch_weights.bond, 0.0);
        assert_eq!(molecular.branch_weights.topology, 1.0);
    }

    #[test]
    fn semantic_diagnostics_are_present_and_consistent_between_single_and_batch() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = synthetic_phase1_examples()
            .into_iter()
            .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
            .collect::<Vec<_>>();

        let single = system.forward_example(&examples[0]);
        let (_, batched) = system.forward_batch(&examples[..2]);
        let from_batch = &batched[0];

        assert_eq!(
            single.diagnostics.topology.token_count,
            from_batch.diagnostics.topology.token_count
        );
        assert_eq!(
            single.diagnostics.geometry.token_count,
            from_batch.diagnostics.geometry.token_count
        );
        assert_eq!(
            single.diagnostics.pocket.token_count,
            from_batch.diagnostics.pocket.token_count
        );
        assert_eq!(
            single.diagnostics.topology.slot_count,
            from_batch.diagnostics.topology.slot_count
        );
        assert!(single.diagnostics.topology.pooled_norm.is_finite());
        assert!(single.diagnostics.geometry.slot_entropy >= 0.0);
        assert!(single.diagnostics.pocket.slot_entropy >= 0.0);
        assert!(single.diagnostics.topology.attention_visible_slot_fraction >= 0.0);
        assert!(single.diagnostics.geometry.attention_visible_slot_fraction >= 0.0);
        assert!(single.diagnostics.pocket.attention_visible_slot_fraction >= 0.0);
        assert!(single.diagnostics.topology.active_slot_count >= 0.0);
        assert!(single.diagnostics.geometry.dead_slot_count >= 0.0);
        assert!(single.diagnostics.pocket.diffuse_slot_count >= 0.0);
        assert!(single.diagnostics.pocket.mean_slot_activation.is_finite());
        assert_eq!(
            single.diagnostics.topology.reconstruction_mse.is_some(),
            from_batch.diagnostics.topology.reconstruction_mse.is_some()
        );
        assert!(single.diagnostics.topology.reconstruction_mse.unwrap() >= 0.0);
        assert!(from_batch.diagnostics.topology.reconstruction_mse.unwrap() >= 0.0);
    }

    #[test]
    fn sync_context_is_preserved_when_batch_slicing_examples_with_different_sizes() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = vec![
            synthetic_custom_example(
                "sync-ex-0",
                "protein-a",
                4,
                3,
                0.0,
                config.model.pocket_feature_dim,
            ),
            synthetic_custom_example(
                "sync-ex-1",
                "protein-b",
                2,
                5,
                1.4,
                config.model.pocket_feature_dim,
            ),
        ];

        assert_ne!(
            examples[0].topology.atom_types.size()[0],
            examples[1].topology.atom_types.size()[0]
        );
        assert_ne!(
            examples[0].pocket.coords.size()[0],
            examples[1].pocket.coords.size()[0]
        );

        let single = [
            system.forward_example(&examples[0]),
            system.forward_example(&examples[1]),
        ];
        let (_, batched) = system.forward_batch(&examples);

        assert_sync_context_matches_single(&single[0], &batched[0]);
        assert_sync_context_matches_single(&single[1], &batched[1]);
        assert_ne!(
            batched[0].sync_context.example_id,
            batched[1].sync_context.example_id
        );
        assert_ne!(
            batched[0].sync_context.protein_id,
            batched[1].sync_context.protein_id
        );
        assert_ne!(
            batched[0].sync_context.coordinate_frame_origin,
            batched[1].sync_context.coordinate_frame_origin
        );
    }

    #[test]
    fn per_example_interaction_diagnostics_align_with_single_example_slices() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = vec![
            synthetic_custom_example(
                "sync-ex-2",
                "protein-c",
                4,
                4,
                0.2,
                config.model.pocket_feature_dim,
            ),
            synthetic_custom_example(
                "sync-ex-3",
                "protein-d",
                3,
                6,
                1.8,
                config.model.pocket_feature_dim,
            ),
        ];

        let single = [
            system.forward_example(&examples[0]),
            system.forward_example(&examples[1]),
        ];
        let (_, batched) = system.forward_batch(&examples);

        assert_paths_are_per_example_and_match_single(&single[0], &batched[0]);
        assert_paths_are_per_example_and_match_single(&single[1], &batched[1]);
    }

    #[test]
    fn stochastic_rollout_sampling_is_reproducible_for_fixed_seed() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.sampling_temperature = 1.0;
        config.data.generation_target.sampling_top_k = 4;
        config.data.generation_target.sampling_top_p = 0.9;
        config.data.generation_target.coordinate_sampling_noise_std = 0.02;
        config.data.generation_target.sampling_seed = 12345;

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let left = system.forward_example(&example).generation.rollout;
        let right = system.forward_example(&example).generation.rollout;

        assert_eq!(left.steps.len(), right.steps.len());
        assert_eq!(
            left.steps.last().unwrap().atom_types,
            right.steps.last().unwrap().atom_types
        );
        assert_eq!(
            left.steps.last().unwrap().coords,
            right.steps.last().unwrap().coords
        );
    }

    #[test]
    fn conditioned_generation_request_exposes_separate_modalities() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let forward = system.forward_example(&example);

        let request = crate::models::ConditionedGenerationRequest::from_forward(
            &example,
            &forward,
            &config.data.generation_target,
        );

        assert_eq!(request.example_id, example.example_id);
        assert_eq!(request.protein_id, example.protein_id);
        assert_eq!(
            request.topology.context.size(),
            forward.generation.state.topology_context.size()
        );
        assert_eq!(
            request.geometry.context.size(),
            forward.generation.state.geometry_context.size()
        );
        assert_eq!(
            request.pocket.context.size(),
            forward.generation.state.pocket_context.size()
        );
        assert!(request.topology.active_slot_fraction >= 0.0);
        assert!(request.geometry.active_slot_fraction >= 0.0);
        assert!(request.pocket.active_slot_fraction >= 0.0);
        assert!(request.gate_summary.topo_from_geo.is_finite());
        assert_eq!(
            request.generation_config.rollout_steps,
            config.data.generation_target.rollout_steps
        );
    }

    #[test]
    fn diagnostics_bundle_is_finite_on_synthetic_forward() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example(&example);
        let bundle = forward.diagnostics_bundle();

        assert!(bundle.semantic.topology.pooled_norm.is_finite());
        assert!(bundle.semantic.geometry.slot_entropy.is_finite());
        assert!(bundle.semantic.pocket.active_slot_fraction.is_finite());

        let interaction = &bundle.interaction;
        for path in [
            &interaction.topo_from_geo,
            &interaction.topo_from_pocket,
            &interaction.geo_from_topo,
            &interaction.geo_from_pocket,
            &interaction.pocket_from_topo,
            &interaction.pocket_from_geo,
        ] {
            assert!(!path.path_name.is_empty());
            assert!(path.gate_mean.is_finite());
            assert!(path.gate_abs_mean.is_finite());
            assert!((0.0..=1.0).contains(&path.gate_closed_fraction));
            assert!((0.0..=1.0).contains(&path.gate_open_fraction));
            assert!((0.0..=1.0).contains(&path.gate_saturation_fraction));
            assert!(path.gate_gradient_proxy.is_finite());
            assert!(path.path_scale.is_finite());
            assert!(!path.gate_status.is_empty());
            assert!(path.attention_entropy.is_finite());
            assert!(path.attended_norm.is_finite());
            assert!(path.effective_update_norm.is_finite());
            if let Some(bias_mean) = path.bias_mean {
                assert!(bias_mean.is_finite());
            }
            if let Some(bias_scale) = path.bias_scale {
                assert!(bias_scale.is_finite());
            }
        }

        let generation = &bundle.generation;
        assert!(generation.last_step_mean_displacement.is_finite());
        assert!(generation.last_step_atom_change_fraction.is_finite());
        assert!(generation.mean_atom_change_fraction.is_finite());
        assert!(generation.mean_coordinate_step_scale.is_finite());
        assert_eq!(
            generation.configured_rollout_steps,
            config.data.generation_target.rollout_steps
        );
        assert!(generation.executed_rollout_steps >= 1);
        assert!(generation.executed_rollout_steps <= generation.configured_rollout_steps);

        if let Some(flow_t) = generation.flow_matching_t {
            assert!(flow_t.is_finite());
        }
        if let Some(flow_target_norm) = generation.flow_target_velocity_norm {
            assert!(flow_target_norm.is_finite());
        }
        if let Some(flow_predicted_norm) = generation.flow_predicted_velocity_norm {
            assert!(flow_predicted_norm.is_finite());
        }
    }

    #[test]
    fn modality_focus_zeroes_disabled_encodings_slots_and_probe_routes() {
        let mut config = ResearchConfig::default();
        config.model.modality_focus = crate::config::ModalityFocusConfig::TopologyOnly;
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example(&example);

        assert!(tensor_abs_sum(&forward.encodings.topology.token_embeddings) > 0.0);
        assert_eq!(tensor_abs_sum(&forward.encodings.geometry.token_embeddings), 0.0);
        assert_eq!(tensor_abs_sum(&forward.encodings.pocket.token_embeddings), 0.0);
        assert_eq!(tensor_abs_sum(&forward.slots.geometry.slots), 0.0);
        assert_eq!(tensor_abs_sum(&forward.slots.pocket.slots), 0.0);
        assert_eq!(forward.slots.geometry.active_slot_count, 0.0);
        assert_eq!(forward.slots.pocket.active_slot_count, 0.0);
        assert_eq!(forward.probes.geometry_distance_predictions.numel(), 0);
        assert_eq!(forward.probes.pocket_feature_predictions.numel(), 0);
        assert_eq!(forward.probes.topology_to_geometry_scalar_logits.numel(), 0);
        assert_eq!(forward.probes.topology_to_pocket_role_logits.numel(), 0);
        assert_eq!(forward.probes.affinity_prediction.numel(), 0);
    }

    #[test]
    fn direct_fusion_negative_control_marks_forced_open_path_provenance() {
        let mut config = ResearchConfig::default();
        config.model.interaction_mode = crate::config::CrossAttentionMode::DirectFusionNegativeControl;
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example(&example);
        for path in [
            &forward.interaction_diagnostics.topo_from_geo,
            &forward.interaction_diagnostics.topo_from_pocket,
            &forward.interaction_diagnostics.geo_from_topo,
            &forward.interaction_diagnostics.geo_from_pocket,
            &forward.interaction_diagnostics.pocket_from_topo,
            &forward.interaction_diagnostics.pocket_from_geo,
        ] {
            assert!(path.forced_open);
            assert_eq!(path.gate_open_fraction, 1.0);
        }
    }

    #[test]
    fn rollout_records_raw_and_step_bucketed_path_usage() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example(&example);
        let path_usage = &forward.generation.rollout.path_usage;

        assert!(path_usage.raw_path_means.topo_from_geo.is_finite());
        assert!(path_usage.raw_path_means.topo_from_pocket.is_finite());
        assert!(path_usage.raw_path_means.geo_from_topo.is_finite());
        assert!(path_usage.raw_path_means.geo_from_pocket.is_finite());
        assert!(path_usage.raw_path_means.pocket_from_topo.is_finite());
        assert!(path_usage.raw_path_means.pocket_from_geo.is_finite());
        assert_eq!(
            path_usage.step_bucketed_path_means.len(),
            forward.generation.rollout.executed_steps
        );

        for (expected_step, bucket) in path_usage.step_bucketed_path_means.iter().enumerate() {
            assert_eq!(bucket.start_step, expected_step);
            assert_eq!(bucket.end_step, expected_step);
            assert_eq!(
                bucket.path_means.geo_from_pocket,
                path_usage.raw_path_means.geo_from_pocket
            );
        }
    }

    #[test]
    fn refreshed_rollout_path_usage_tracks_actual_step_gates() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.rollout_steps = 3;
        config.data.generation_target.min_rollout_steps = 3;
        config.data.generation_target.context_refresh_policy =
            crate::config::InferenceContextRefreshPolicy::EveryStep;
        config
            .model
            .temporal_interaction_policy
            .rollout_bucket_multipliers =
            vec![crate::config::InteractionPathRolloutBucketMultiplier {
                path: "geo_from_pocket".to_string(),
                start_step: 0,
                end_step: 0,
                multiplier: 0.0,
            }];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let rollout = system.forward_example(&example).generation.rollout;
        let path_usage = &rollout.path_usage;

        assert_eq!(rollout.context_refresh_policy, "every_step");
        assert_eq!(rollout.refresh_count, rollout.executed_steps);
        assert!(path_usage.step_bucketed_path_means.len() >= 2);
        assert_eq!(
            path_usage.step_bucketed_path_means[0]
                .path_means
                .geo_from_pocket,
            0.0
        );
        assert!(
            path_usage.step_bucketed_path_means[1]
                .path_means
                .geo_from_pocket
                > 0.0
        );
        assert_ne!(
            path_usage.step_bucketed_path_means[0]
                .path_means
                .geo_from_pocket,
            path_usage.step_bucketed_path_means[1]
                .path_means
                .geo_from_pocket
        );
    }

    #[test]
    fn flow_rollout_static_context_conditions_on_initial_raw_gates() {
        let mut config = ResearchConfig::default();
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.steps = 3;
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let rollout = system.forward_example(&example).generation.rollout;
        let raw_gate_mean = generation_gate_mean(rollout.path_usage.raw_path_means);
        let diagnostic_gate_mean = rollout.steps[0]
            .flow_diagnostics
            .get("conditioning_gate_input_mean")
            .copied()
            .expect("flow rollout should report conditioning gate diagnostics");

        assert!(raw_gate_mean > 0.0);
        assert!((diagnostic_gate_mean - raw_gate_mean).abs() <= 1e-6);
        assert_eq!(
            rollout.path_usage.step_bucketed_path_means[0]
                .path_means
                .geo_from_pocket,
            rollout.path_usage.raw_path_means.geo_from_pocket
        );
    }

    #[test]
    fn flow_rollout_refresh_policy_updates_step_gate_usage() {
        let mut config = ResearchConfig::default();
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.steps = 3;
        config.data.generation_target.context_refresh_policy =
            crate::config::InferenceContextRefreshPolicy::EveryStep;
        config
            .model
            .temporal_interaction_policy
            .flow_time_bucket_multipliers =
            vec![crate::config::InteractionPathFlowTimeBucketMultiplier {
                path: "geo_from_pocket".to_string(),
                start_t: 0.0,
                end_t: 0.5,
                multiplier: 0.0,
            }];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let rollout = system.forward_example(&example).generation.rollout;
        let buckets = &rollout.path_usage.step_bucketed_path_means;

        assert_eq!(rollout.context_refresh_policy, "every_step");
        assert_eq!(rollout.refresh_count, rollout.executed_steps);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0].path_means.geo_from_pocket, 0.0);
        assert_eq!(buckets[1].path_means.geo_from_pocket, 0.0);
        assert!(buckets[2].path_means.geo_from_pocket > 0.0);
    }

    #[test]
    fn pocket_only_initialization_baseline_does_not_use_target_ligand_state() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline;
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::SurrogateReconstruction;
        config
            .data
            .generation_target
            .pocket_only_initialization
            .atom_count = 5;
        config
            .data
            .generation_target
            .pocket_only_initialization
            .atom_type_token = 2;
        config.validate().unwrap();

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_custom_example(
            "pocket-only-init",
            "protein-pocket-only",
            3,
            6,
            0.0,
            config.model.pocket_feature_dim,
        );

        let forward = system.forward_example(&example);
        let partial = &forward.generation.state.partial_ligand;

        assert_eq!(partial.atom_types.size(), vec![5]);
        assert_eq!(partial.coords.size(), vec![5, 3]);
        assert!(partial.atom_types.eq(2).all().int64_value(&[]) != 0);
        assert_ne!(
            partial.coords.size(),
            example.decoder_supervision.noisy_coords.size()
        );
        assert_eq!(
            forward.generation.generation_mode,
            crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline
        );
        assert_eq!(
            forward.generation.rollout.atom_count_source,
            "configured_atom_count_prior"
        );
        assert_eq!(
            forward.generation.rollout.atom_count_prior_provenance,
            "fixed"
        );
        assert_eq!(
            forward.generation.rollout.topology_source,
            "configured_uniform_atom_type_prior"
        );
        assert_eq!(
            forward.generation.rollout.geometry_source,
            "pocket_centroid_deterministic_offsets"
        );
        assert_eq!(
            forward.generation.rollout.decoder_capability,
            "fixed_atom_refinement_with_pocket_only_initialization"
        );
    }

    #[test]
    fn rollout_refresh_policy_defaults_to_static_diagnostics() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let rollout = system.forward_example(&example).generation.rollout;

        assert_eq!(rollout.context_refresh_policy, "static");
        assert_eq!(rollout.refresh_count, 0);
        assert_eq!(rollout.last_refresh_step, None);
        assert_eq!(rollout.stale_context_steps, rollout.executed_steps);
    }

    #[test]
    fn rollout_every_step_refresh_policy_reports_refreshes() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.context_refresh_policy =
            crate::config::InferenceContextRefreshPolicy::EveryStep;
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let rollout = system.forward_example(&example).generation.rollout;

        assert_eq!(rollout.context_refresh_policy, "every_step");
        assert_eq!(rollout.refresh_count, rollout.executed_steps);
        assert_eq!(
            rollout.last_refresh_step,
            rollout.executed_steps.checked_sub(1)
        );
        assert_eq!(rollout.stale_context_steps, 0);
    }

    #[test]
    fn rollout_periodic_refresh_policy_reports_stale_steps() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.context_refresh_policy =
            crate::config::InferenceContextRefreshPolicy::PeriodicN { n: 2 };
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let rollout = system.forward_example(&example).generation.rollout;
        let expected_refreshes = (0..rollout.executed_steps)
            .filter(|step| step % 2 == 0)
            .count();

        assert_eq!(rollout.context_refresh_policy, "periodic_2");
        assert_eq!(rollout.refresh_count, expected_refreshes);
        assert_eq!(
            rollout.last_refresh_step,
            (0..rollout.executed_steps).rev().find(|step| step % 2 == 0)
        );
        assert_eq!(
            rollout.stale_context_steps,
            rollout.executed_steps - expected_refreshes
        );
    }

    #[test]
    fn rollout_guardrails_mark_unclean_stop_without_changing_stop_behavior() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.rollout_steps = 1;
        config.data.generation_target.min_rollout_steps = 1;
        config.data.generation_target.stop_probability_threshold = 0.0;
        config.data.generation_target.coordinate_step_scale = 1e-9;

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let pocket_count = example.pocket.coords.size()[0].max(1);
        example.pocket.coords = example
            .decoder_supervision
            .noisy_coords
            .get(0)
            .unsqueeze(0)
            .repeat([pocket_count, 1]);

        let rollout = system.forward_example(&example).generation.rollout;

        assert_eq!(rollout.executed_steps, 1);
        assert!(rollout.stopped_early);
        assert!(rollout.steps[0].stopped);
        assert!(rollout.severe_clash_flag);
        assert!(rollout.steps[0].severe_clash_flag);
        assert!(rollout.guardrail_blockable_stop_flag);
        assert!(rollout.steps[0].guardrail_blockable_stop_flag);
    }

    #[test]
    fn rollout_guardrail_blockable_stop_keeps_legacy_alias_on_read() {
        let step_json = r#"{
            "step_index": 0,
            "stop_probability": 1.0,
            "stopped": true,
            "atom_types": [6],
            "coords": [[0.0, 0.0, 0.0]],
            "mean_displacement": 0.0,
            "atom_change_fraction": 0.0,
            "coordinate_step_scale": 1.0,
            "stop_overridden_flag": true
        }"#;
        let step: GenerationStepRecord = serde_json::from_str(step_json).unwrap();

        assert!(step.guardrail_blockable_stop_flag);

        let serialized = serde_json::to_value(&step).unwrap();
        assert!(serialized.get("guardrail_blockable_stop_flag").is_some());
        assert!(serialized.get("stop_overridden_flag").is_none());
    }

    #[test]
    fn semantic_probe_capacity_config_runs_mlp_heads() {
        let mut config = ResearchConfig::default();
        config.model.semantic_probes.hidden_layers = 2;
        config.model.semantic_probes.hidden_dim = 32;
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let forward = system.forward_example(&example);

        assert_eq!(
            forward.probes.topology_adjacency_logits.size(),
            example.topology.adjacency.size()
        );
        assert_eq!(
            forward.probes.pocket_feature_predictions.size(),
            example.pocket.atom_features.size()
        );
        assert!(forward
            .probes
            .topology_to_geometry_scalar_logits
            .double_value(&[])
            .is_finite());
    }

    #[test]
    fn rollout_valence_guardrail_detects_overvalent_hydrogen() {
        let config = ResearchConfig::default();
        let example = synthetic_custom_example(
            "guardrail-valence",
            "guardrail-pocket",
            3,
            3,
            0.0,
            config.model.pocket_feature_dim,
        );
        let hydrogen_chain = Tensor::from_slice(&[4_i64, 4, 4]).to_kind(Kind::Int64);
        let flags = rollout_guardrail_flags(&example, &hydrogen_chain, &example.geometry.coords);

        assert!(flags.valence_guardrail);
        assert!(flags.any());
    }
}
