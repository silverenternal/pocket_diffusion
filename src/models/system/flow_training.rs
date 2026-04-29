use tch::{Kind, Tensor};

use super::{
    last_refresh_step_for_policy, path_usage_summary, refresh_count_for_policy,
    stale_context_steps_for_policy, ConditionedGenerationState, CrossModalInteractions,
    FlowMatchingTrainingRecord, GenerationGateSummary, GenerationRolloutRecord,
    GenerationStepRecord, MolecularFlowBranchWeights, MolecularFlowTrainingRecord,
    NativeGraphLayerProvenance, RolloutGuardrailFlags,
};
use crate::{
    config::{
        FlowMatchingIntegrationMethod, FlowTargetAlignmentPolicy, GenerationBackendFamilyConfig,
        ResearchConfig,
    },
    data::MolecularExample,
    models::{
        interaction::InteractionExecutionContext, ConditioningState, FlowMatchingHead, FlowState,
        MolecularFlowInput, NativeGraphExtractionResult, TargetMatchingResult,
        MOLECULAR_FLOW_CONTRACT_VERSION,
    },
};

const POCKET_CONTACT_CUTOFF_ANGSTROM: f64 = 4.5;

fn native_graph_layer_provenance(
    graph: &NativeGraphExtractionResult,
) -> NativeGraphLayerProvenance {
    let raw_bond_logit_pair_count = graph
        .diagnostics
        .get("molecular_flow_raw_bond_logit_pair_count")
        .copied()
        .unwrap_or(0.0)
        .max(0.0) as usize;
    let raw_to_constrained_removed_bond_count = graph
        .diagnostics
        .get("molecular_flow_raw_to_constrained_removed_bond_count")
        .copied()
        .unwrap_or(0.0)
        .max(0.0) as usize;
    let connectivity_guardrail_added_bond_count = graph
        .diagnostics
        .get("molecular_flow_connectivity_guardrail_added_bond_count")
        .copied()
        .unwrap_or(0.0)
        .max(0.0) as usize;
    let valence_guardrail_downgrade_count = graph
        .diagnostics
        .get("molecular_flow_valence_guardrail_downgrade_count")
        .copied()
        .unwrap_or(0.0)
        .max(0.0) as usize;
    NativeGraphLayerProvenance {
        raw_logits_layer: "raw_molecular_flow_logits".to_string(),
        raw_native_extraction_layer: "raw_native_graph_extraction".to_string(),
        constrained_graph_layer: "constrained_native_graph".to_string(),
        repaired_graph_layer: "repaired_candidate_graph".to_string(),
        raw_bond_logit_pair_count,
        raw_native_bond_count: graph.raw_bonds.len(),
        constrained_bond_count: graph.bonds.len(),
        raw_to_constrained_removed_bond_count,
        connectivity_guardrail_added_bond_count,
        valence_guardrail_downgrade_count,
        guardrail_trigger_count: raw_to_constrained_removed_bond_count
            + connectivity_guardrail_added_bond_count
            + valence_guardrail_downgrade_count,
    }
}

pub(crate) fn resolved_generation_backend_family(
    config: &ResearchConfig,
) -> GenerationBackendFamilyConfig {
    config.generation_method.resolved_primary_backend_family()
}

impl super::Phase1ResearchSystem {
    pub(super) fn flow_matching_training_record(
        &self,
        example: &MolecularExample,
        generation_state: &ConditionedGenerationState,
        interactions: &CrossModalInteractions,
        interaction_execution_context: &InteractionExecutionContext,
    ) -> Option<FlowMatchingTrainingRecord> {
        if self.generation_backend_family != GenerationBackendFamilyConfig::FlowMatching {
            return None;
        }
        let branch_weights = self
            .effective_molecular_flow_branch_weights(interaction_execution_context.training_step);
        let policy = self
            .flow_matching_config
            .multi_modal
            .target_alignment_policy;
        let x0 = self.flow_matching_x0_for_generation_state(
            example,
            &generation_state.partial_ligand.coords,
        );
        let Some(matched_x1) =
            matched_target_coords(&x0, &example.decoder_supervision.target_coords, policy)
        else {
            return None;
        };
        let x1 = fill_unmatched_target_coords(&matched_x1.values, &x0, &matched_x1.matching.mask);
        let x0_source = self.flow_matching_x0_source_label_for_state(
            example,
            &generation_state.partial_ligand.coords,
        );
        if x1.size() != x0.size() {
            return None;
        }
        let t = flow_matching_t_from_example(example, Some(interaction_execution_context));
        let xt = &x0 * (1.0 - t) + &x1 * t;
        let flow_state = FlowState {
            coords: xt.shallow_clone(),
            x0_coords: x0.shallow_clone(),
            target_coords: Some(x1.shallow_clone()),
            t,
        };
        let conditioning = flow_conditioning_state(
            generation_state,
            gate_summary_from_interactions(interactions),
        );
        let predicted_velocity = self
            .generator_stack
            .flow_matching_head
            .predict_velocity(&flow_state, &conditioning)
            .ok()?
            .velocity;
        let molecular = self.molecular_flow_training_record(
            example,
            generation_state,
            &flow_state,
            &conditioning,
            branch_weights,
            &matched_x1.matching,
        );
        Some(FlowMatchingTrainingRecord {
            predicted_velocity,
            target_velocity: x1 - x0,
            sampled_coords: xt,
            t,
            x0_source: x0_source.to_string(),
            flow_contract_version: MOLECULAR_FLOW_CONTRACT_VERSION.to_string(),
            atom_mask: generation_state.partial_ligand.atom_mask.shallow_clone()
                * matched_x1.matching.mask.shallow_clone(),
            target_matching_policy: matched_x1.matching.provenance.clone(),
            target_matching_mean_cost: matched_x1.matching.cost_summary.mean_cost,
            target_matching_coverage: target_matching_coverage(&matched_x1.matching),
            target_matching_cost_summary: matched_x1.matching.cost_summary.clone(),
            branch_weights,
            molecular,
        })
    }

    pub(super) fn rollout_flow_matching(
        &self,
        example: &MolecularExample,
        initial_state: &ConditionedGenerationState,
        raw_path_means: GenerationGateSummary,
        interaction_execution_context: &InteractionExecutionContext,
    ) -> GenerationRolloutRecord {
        let configured_steps = self.flow_matching_config.steps.max(1);
        let mut coords = self
            .flow_matching_x0_for_generation_state(example, &initial_state.partial_ligand.coords);
        let x0_source = self
            .flow_matching_x0_source_label_for_state(example, &initial_state.partial_ligand.coords);
        let mut atom_types_tensor = initial_state.partial_ligand.atom_types.shallow_clone();
        let mut atom_types = tensor_to_i64_vec(&atom_types_tensor);
        let mut native_bonds = Vec::new();
        let mut native_bond_types = Vec::new();
        let mut constrained_native_bonds = Vec::new();
        let mut constrained_native_bond_types = Vec::new();
        let mut native_graph_provenance = NativeGraphLayerProvenance::default();
        let mut state = initial_state.clone();
        state.partial_ligand.coords = coords.shallow_clone();
        let mut current_gate_summary = raw_path_means;
        let mut conditioning = flow_conditioning_state(&state, current_gate_summary);
        let x0_coords = coords.shallow_clone();
        let dt = 1.0 / configured_steps as f64;
        let mut steps = Vec::with_capacity(configured_steps);
        let mut step_gate_summaries = Vec::with_capacity(configured_steps);
        let mut previous = coords.shallow_clone();
        let mut rollout_guardrails = RolloutGuardrailFlags::default();

        for step_index in 0..configured_steps {
            let t = step_index as f64 / configured_steps as f64;
            state.partial_ligand.step_index = step_index as i64;
            state.partial_ligand.coords = coords.shallow_clone();
            if self
                .generation_target
                .context_refresh_policy
                .should_refresh_at_step(step_index)
            {
                let mut refresh_context = interaction_execution_context.clone();
                refresh_context.flow_t = Some(t);
                let gate_summary = self.refresh_generation_context(
                    example,
                    &mut state,
                    step_index,
                    &refresh_context,
                );
                current_gate_summary = gate_summary;
                conditioning = flow_conditioning_state(&state, current_gate_summary);
            }
            let flow_state = FlowState {
                coords: coords.shallow_clone(),
                x0_coords: x0_coords.shallow_clone(),
                target_coords: None,
                t,
            };
            let (velocity_1, flow_diagnostics) = self
                .generator_stack
                .flow_matching_head
                .predict_velocity(&flow_state, &conditioning)
                .map(|field| (field.velocity, field.diagnostics))
                .unwrap_or_else(|_| {
                    (
                        Tensor::zeros_like(&coords),
                        std::collections::BTreeMap::new(),
                    )
                });
            let mut flow_diagnostics = flow_diagnostics;
            let update = match self.flow_matching_config.integration_method {
                FlowMatchingIntegrationMethod::Euler => velocity_1 * dt,
                FlowMatchingIntegrationMethod::Heun => {
                    let predictor = &coords + &(velocity_1.shallow_clone() * dt);
                    let predictor_state = FlowState {
                        coords: predictor,
                        x0_coords: x0_coords.shallow_clone(),
                        target_coords: None,
                        t: (t + dt).min(1.0),
                    };
                    let velocity_2 = self
                        .generator_stack
                        .flow_matching_head
                        .predict_velocity(&predictor_state, &conditioning)
                        .map(|field| field.velocity)
                        .unwrap_or_else(|_| Tensor::zeros_like(&coords));
                    (velocity_1 + velocity_2) * (0.5 * dt)
                }
            };
            coords = coords + update;
            coords = constrain_to_pocket_envelope(
                &coords,
                &example.pocket.coords,
                self.generation_target.pocket_guidance_scale,
            );
            if self.molecular_flow_rollout_enabled() {
                state.partial_ligand.atom_types = atom_types_tensor.shallow_clone();
                state.partial_ligand.coords = coords.shallow_clone();
                if let Ok(prediction) =
                    self.generator_stack
                        .molecular_flow_head
                        .predict(MolecularFlowInput {
                            atom_types: &state.partial_ligand.atom_types,
                            coords: &coords,
                            x0_coords: &x0_coords,
                            t,
                            conditioning: &conditioning,
                        })
                {
                    atom_types_tensor = prediction.atom_type_logits.argmax(-1, false);
                    atom_types = tensor_to_i64_vec(&atom_types_tensor);
                    let graph = crate::models::predicted_native_graph_from_flow(
                        &atom_types_tensor,
                        &coords,
                        &prediction.bond_exists_logits,
                        &prediction.bond_type_logits,
                        &prediction.topology_logits,
                    );
                    native_graph_provenance = native_graph_layer_provenance(&graph);
                    native_bonds = graph.raw_bonds;
                    native_bond_types = graph.raw_bond_types;
                    constrained_native_bonds = graph.bonds;
                    constrained_native_bond_types = graph.bond_types;
                    flow_diagnostics.extend(prediction.diagnostics);
                    flow_diagnostics.extend(graph.diagnostics);
                }
            }
            let delta = &coords - &previous;
            let mean_displacement = per_atom_displacement_mean(&delta);
            previous = coords.shallow_clone();
            let guardrails =
                self.rollout_guardrail_flags_for_generation(example, &atom_types_tensor, &coords);
            rollout_guardrails.merge(guardrails);
            step_gate_summaries.push(current_gate_summary);
            steps.push(GenerationStepRecord {
                step_index,
                stop_probability: 0.0,
                stopped: false,
                atom_types: atom_types.clone(),
                coords: tensor_to_coords(&coords),
                native_bonds: native_bonds.clone(),
                native_bond_types: native_bond_types.clone(),
                constrained_native_bonds: constrained_native_bonds.clone(),
                constrained_native_bond_types: constrained_native_bond_types.clone(),
                native_graph_provenance: native_graph_provenance.clone(),
                mean_displacement,
                atom_change_fraction: 0.0,
                coordinate_step_scale: dt,
                severe_clash_flag: guardrails.severe_clash,
                valence_guardrail_flag: guardrails.valence_guardrail,
                pharmacophore_conflict_flag: guardrails.pharmacophore_conflict,
                guardrail_blockable_stop_flag: false,
                flow_diagnostics,
            });
        }

        GenerationRolloutRecord {
            example_id: initial_state.example_id.clone(),
            protein_id: initial_state.protein_id.clone(),
            generation_mode: self.generation_mode.as_str().to_string(),
            decoder_capability: self.decoder_capability_label().to_string(),
            atom_count_source: self.generation_mode.atom_count_source_label().to_string(),
            atom_count_prior_provenance: self.atom_count_prior_provenance_label().to_string(),
            topology_source: self.generation_mode.topology_source_label().to_string(),
            geometry_source: self.generation_mode.geometry_source_label().to_string(),
            conditioning_coordinate_frame: self.conditioning_coordinate_frame_label().to_string(),
            flow_x0_source: Some(x0_source.to_string()),
            configured_steps,
            executed_steps: steps.len(),
            stopped_early: false,
            path_usage: path_usage_summary(raw_path_means, &step_gate_summaries),
            context_refresh_policy: self.generation_target.context_refresh_policy.label(),
            refresh_count: refresh_count_for_policy(
                &self.generation_target.context_refresh_policy,
                steps.len(),
            ),
            last_refresh_step: last_refresh_step_for_policy(
                &self.generation_target.context_refresh_policy,
                steps.len(),
            ),
            stale_context_steps: stale_context_steps_for_policy(
                &self.generation_target.context_refresh_policy,
                steps.len(),
            ),
            severe_clash_flag: rollout_guardrails.severe_clash,
            valence_guardrail_flag: rollout_guardrails.valence_guardrail,
            pharmacophore_conflict_flag: rollout_guardrails.pharmacophore_conflict,
            guardrail_blockable_stop_flag: false,
            steps,
        }
    }
}

impl super::Phase1ResearchSystem {
    fn molecular_flow_training_record(
        &self,
        example: &MolecularExample,
        generation_state: &ConditionedGenerationState,
        flow_state: &FlowState,
        conditioning: &ConditioningState,
        branch_weights: MolecularFlowBranchWeights,
        target_matching: &TargetMatchingResult,
    ) -> Option<MolecularFlowTrainingRecord> {
        if !self.molecular_flow_rollout_enabled() {
            return None;
        }
        let prediction = self
            .generator_stack
            .molecular_flow_head
            .predict(MolecularFlowInput {
                atom_types: &generation_state.partial_ligand.atom_types,
                coords: &flow_state.coords,
                x0_coords: &flow_state.x0_coords,
                t: flow_state.t,
                conditioning,
            })
            .ok()?;
        let device = generation_state.partial_ligand.atom_types.device();
        let policy = self
            .flow_matching_config
            .multi_modal
            .target_alignment_policy;
        let target_atom_types = matched_target_atom_types(
            &example.decoder_supervision.target_atom_types,
            device,
            target_matching,
        );
        let target_adjacency =
            matched_square_float_matrix(&example.topology.adjacency, device, target_matching);
        let target_bond_types = matched_square_long_matrix(
            &dense_bond_type_adjacency(example, device),
            device,
            target_matching,
        );
        let effective_atom_mask = generation_state.partial_ligand.atom_mask.shallow_clone()
            * target_matching.mask.shallow_clone();
        let pair_mask = pair_mask_from_atom_mask(&effective_atom_mask);
        let target_pocket_contacts = matched_pocket_contact_targets(
            &example.decoder_supervision.target_coords,
            &example.pocket.coords,
            device,
            target_matching,
        );
        Some(MolecularFlowTrainingRecord {
            atom_type_logits: prediction.atom_type_logits,
            target_atom_types,
            bond_exists_logits: prediction.bond_exists_logits,
            target_adjacency: target_adjacency.shallow_clone(),
            bond_type_logits: prediction.bond_type_logits,
            target_bond_types,
            topology_logits: prediction.topology_logits,
            target_topology: target_adjacency,
            pair_mask,
            pocket_contact_logits: prediction.pocket_contact_logits,
            target_pocket_contacts,
            pocket_interaction_mask: effective_atom_mask.shallow_clone(),
            pocket_branch_target_family: "pocket_interaction_profile".to_string(),
            pocket_context_reconstruction_role: "context_drift_diagnostic".to_string(),
            pocket_context_reconstruction: prediction.pocket_context_reconstruction,
            target_pocket_context: conditioning.pocket_context.detach(),
            branch_weights,
            target_alignment_policy: policy.as_str().to_string(),
            target_matching_policy: target_matching.provenance.clone(),
            target_matching_mean_cost: target_matching.cost_summary.mean_cost,
            target_matching_coverage: target_matching_coverage(target_matching),
            target_matching_cost_summary: target_matching.cost_summary.clone(),
            target_atom_mask: target_matching.mask.shallow_clone(),
            full_branch_set_enabled: self
                .flow_matching_config
                .multi_modal
                .has_full_molecular_branch_set(),
        })
    }

    fn molecular_flow_rollout_enabled(&self) -> bool {
        !self.flow_matching_config.geometry_only
            && self
                .flow_matching_config
                .multi_modal
                .enabled_branches
                .iter()
                .any(|branch| !matches!(branch, crate::config::FlowBranchKind::Geometry))
    }

    fn effective_molecular_flow_branch_weights(
        &self,
        training_step: Option<usize>,
    ) -> MolecularFlowBranchWeights {
        let effective = self
            .flow_matching_config
            .multi_modal
            .branch_schedule
            .effective_weights(
                &self.flow_matching_config.multi_modal.branch_loss_weights,
                training_step,
            );
        MolecularFlowBranchWeights {
            geometry: effective.geometry,
            atom_type: effective.atom_type,
            bond: effective.bond,
            topology: effective.topology,
            pocket_context: effective.pocket_context,
            synchronization: effective.synchronization,
        }
    }

    fn flow_matching_x0_for_generation_state(
        &self,
        example: &MolecularExample,
        reference_coords: &Tensor,
    ) -> Tensor {
        flow_matching_x0_for_state(
            example,
            reference_coords,
            self.flow_matching_config.noise_scale,
            self.flow_matching_config.use_corrupted_x0,
            self.generation_mode.uses_target_ligand_initialization(),
        )
    }

    fn flow_matching_x0_source_label_for_state(
        &self,
        example: &MolecularExample,
        reference_coords: &Tensor,
    ) -> &'static str {
        flow_matching_x0_source_label_for_state(
            example,
            reference_coords,
            self.flow_matching_config.use_corrupted_x0,
            self.generation_mode.uses_target_ligand_initialization(),
        )
    }
}

pub(crate) fn flow_matching_x0_source_label(use_corrupted_x0: bool) -> &'static str {
    if use_corrupted_x0 {
        "target_ligand_corrupted_geometry"
    } else {
        "deterministic_gaussian_noise"
    }
}

pub(crate) fn flow_matching_x0_source_label_for_state(
    example: &MolecularExample,
    reference_coords: &Tensor,
    use_corrupted_x0: bool,
    allow_target_ligand_corrupted_x0: bool,
) -> &'static str {
    if use_corrupted_x0 && allow_target_ligand_corrupted_x0 {
        if example.decoder_supervision.noisy_coords.size() == reference_coords.size() {
            flow_matching_x0_source_label(true)
        } else {
            "reference_geometry_shape_mismatch_fallback"
        }
    } else if use_corrupted_x0 {
        "conditioning_scaffold_deterministic_noise_no_target_ligand"
    } else {
        flow_matching_x0_source_label(false)
    }
}

pub(crate) fn tensor_to_i64_vec(tensor: &Tensor) -> Vec<i64> {
    let len = tensor.size().first().copied().unwrap_or(0).max(0) as usize;
    (0..len)
        .map(|index| tensor.int64_value(&[index as i64]))
        .collect()
}

pub(crate) fn tensor_to_coords(tensor: &Tensor) -> Vec<[f32; 3]> {
    let rows = tensor.size().first().copied().unwrap_or(0).max(0) as usize;
    (0..rows)
        .map(|row| {
            [
                tensor.double_value(&[row as i64, 0]) as f32,
                tensor.double_value(&[row as i64, 1]) as f32,
                tensor.double_value(&[row as i64, 2]) as f32,
            ]
        })
        .collect()
}

pub(crate) fn clip_coordinate_delta_norm(delta: &Tensor, max_norm: f64) -> Tensor {
    let norms = delta
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .sqrt();
    let scale = norms.clamp_min(max_norm).reciprocal() * max_norm;
    let mask = norms.gt(max_norm).to_kind(Kind::Float);
    let ones = Tensor::ones_like(&mask);
    let safe_scale = &mask * scale + (&ones - &mask);
    delta * safe_scale
}

pub(crate) fn per_atom_displacement_mean(delta: &Tensor) -> f64 {
    if delta.numel() == 0 {
        return 0.0;
    }
    delta
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
}

pub(crate) fn atom_change_fraction(current: &Tensor, next: &Tensor) -> f64 {
    if current.numel() == 0 || next.numel() == 0 {
        return 0.0;
    }
    current
        .ne_tensor(next)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

pub(crate) fn pocket_guidance_delta(
    current_coords: &Tensor,
    pocket_coords: &Tensor,
    coordinate_step_scale: f64,
    step_index: usize,
    rollout_steps: usize,
) -> Tensor {
    if current_coords.numel() == 0 || pocket_coords.numel() == 0 {
        return Tensor::zeros_like(current_coords);
    }

    let ligand_centroid = current_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let centroid_offset = &pocket_centroid - &ligand_centroid;
    let pocket_radius = pocket_radius_from_coords(pocket_coords, &pocket_centroid);
    let centroid_distance = centroid_offset
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt()
        .double_value(&[]);
    let overflow = (centroid_distance - pocket_radius * 0.85).max(0.0);
    if overflow <= 0.0 {
        return Tensor::zeros_like(current_coords);
    }

    let progress = (step_index + 1) as f64 / rollout_steps.max(1) as f64;
    let guidance_scale = (0.08 + 0.22 * progress + overflow.min(pocket_radius.max(1.0))).min(0.45)
        * coordinate_step_scale;
    centroid_offset.unsqueeze(0).expand_as(current_coords) * guidance_scale
}

pub(crate) fn constrain_to_pocket_envelope(
    coords: &Tensor,
    pocket_coords: &Tensor,
    pocket_guidance_scale: f64,
) -> Tensor {
    if coords.numel() == 0 || pocket_coords.numel() == 0 || pocket_guidance_scale <= 0.0 {
        return coords.shallow_clone();
    }

    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_radius = pocket_radius_from_coords(pocket_coords, &pocket_centroid).max(1.0);
    let max_radius = pocket_radius + 1.5;
    let offsets = coords - pocket_centroid.unsqueeze(0);
    let radii = offsets
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .sqrt()
        .clamp_min(1e-6);
    let scale = radii.clamp_max(max_radius) / &radii;
    let projected = pocket_centroid.unsqueeze(0) + offsets * scale;
    let outside = radii.gt(max_radius).to_kind(Kind::Float);
    let ones = Tensor::ones_like(&outside);
    &outside * projected + (&ones - &outside) * coords
}

pub(crate) fn flow_matching_x0_for_state(
    example: &MolecularExample,
    reference_coords: &Tensor,
    noise_scale: f64,
    use_corrupted_x0: bool,
    allow_target_ligand_corrupted_x0: bool,
) -> Tensor {
    if use_corrupted_x0 && allow_target_ligand_corrupted_x0 {
        if example.decoder_supervision.noisy_coords.size() == reference_coords.size() {
            return example.decoder_supervision.noisy_coords.shallow_clone();
        } else {
            return reference_coords.shallow_clone();
        }
    }

    if use_corrupted_x0 {
        return reference_coords
            + deterministic_flow_noise(
                reference_coords,
                noise_scale,
                example
                    .decoder_supervision
                    .corruption_metadata
                    .corruption_seed
                    .wrapping_add(17),
            );
    }

    deterministic_pocket_noise_x0(example, reference_coords, noise_scale)
}

fn deterministic_pocket_noise_x0(
    example: &MolecularExample,
    reference_coords: &Tensor,
    noise_scale: f64,
) -> Tensor {
    let atom_count = reference_coords.size()[0].max(0) as usize;
    let pocket_centroid = if example.pocket.coords.numel() == 0 {
        [0.0_f64, 0.0, 0.0]
    } else {
        let centroid = example
            .pocket
            .coords
            .mean_dim([0].as_slice(), false, Kind::Float);
        [
            centroid.double_value(&[0]),
            centroid.double_value(&[1]),
            centroid.double_value(&[2]),
        ]
    };
    let mut values = Vec::with_capacity(atom_count * 3);
    for atom_ix in 0..atom_count {
        for (axis, value) in pocket_centroid.iter().enumerate() {
            let unit = deterministic_flow_unit(
                example
                    .decoder_supervision
                    .corruption_metadata
                    .corruption_seed,
                atom_ix,
                axis,
            );
            let centered = unit * 2.0 - 1.0;
            values.push((*value + centered * noise_scale) as f32);
        }
    }
    Tensor::from_slice(&values)
        .reshape([atom_count as i64, 3])
        .to_device(reference_coords.device())
}

struct MatchedTargetCoords {
    values: Tensor,
    matching: TargetMatchingResult,
}

fn matched_target_coords(
    generated_coords: &Tensor,
    target: &Tensor,
    policy: FlowTargetAlignmentPolicy,
) -> Option<MatchedTargetCoords> {
    let matching = crate::models::match_molecular_targets(generated_coords, target, policy)?;
    let values = matched_float_rows(target, &matching, 3, generated_coords.device());
    Some(MatchedTargetCoords { values, matching })
}

fn matched_float_rows(
    target: &Tensor,
    matching: &TargetMatchingResult,
    cols: i64,
    device: tch::Device,
) -> Tensor {
    let source_rows = target.size().first().copied().unwrap_or(0).max(0);
    let source_cols = target.size().get(1).copied().unwrap_or(0).max(0);
    let values = matching
        .target_indices
        .iter()
        .flat_map(|target_index| {
            (0..cols).map(move |col| match target_index {
                Some(source_row)
                    if *source_row >= 0 && *source_row < source_rows && col < source_cols =>
                {
                    target.double_value(&[*source_row, col]) as f32
                }
                _ => 0.0,
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&values)
        .reshape([matching.generated_indices.len() as i64, cols])
        .to_device(device)
}

fn fill_unmatched_target_coords(
    target_values: &Tensor,
    fallback: &Tensor,
    mask: &Tensor,
) -> Tensor {
    if target_values.size() != fallback.size() {
        return target_values.shallow_clone();
    }
    let mask = mask
        .to_device(target_values.device())
        .to_kind(Kind::Float)
        .unsqueeze(1);
    let inverse_mask = Tensor::ones_like(&mask) - &mask;
    target_values * &mask + fallback * inverse_mask
}

fn matched_target_atom_types(
    target: &Tensor,
    device: tch::Device,
    matching: &TargetMatchingResult,
) -> Tensor {
    let source_count = target.size().first().copied().unwrap_or(0).max(0);
    let values = matching
        .target_indices
        .iter()
        .map(|target_index| match target_index {
            Some(source_index) if *source_index >= 0 && *source_index < source_count => {
                target.int64_value(&[*source_index])
            }
            _ => 0,
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&values).to_device(device)
}

fn matched_pocket_contact_targets(
    target_coords: &Tensor,
    pocket_coords: &Tensor,
    device: tch::Device,
    matching: &TargetMatchingResult,
) -> Tensor {
    let source_rows = target_coords.size().first().copied().unwrap_or(0).max(0);
    let pocket_rows = pocket_coords.size().first().copied().unwrap_or(0).max(0);
    if pocket_rows <= 0 {
        return Tensor::zeros(
            [matching.generated_indices.len() as i64],
            (Kind::Float, device),
        );
    }
    let values = matching
        .target_indices
        .iter()
        .map(|target_index| {
            let Some(source_row) = target_index else {
                return 0.0_f32;
            };
            if *source_row < 0 || *source_row >= source_rows {
                return 0.0_f32;
            }
            let target = [
                target_coords.double_value(&[*source_row, 0]),
                target_coords.double_value(&[*source_row, 1]),
                target_coords.double_value(&[*source_row, 2]),
            ];
            let mut min_distance_sq = f64::INFINITY;
            for pocket_row in 0..pocket_rows {
                let dx = target[0] - pocket_coords.double_value(&[pocket_row, 0]);
                let dy = target[1] - pocket_coords.double_value(&[pocket_row, 1]);
                let dz = target[2] - pocket_coords.double_value(&[pocket_row, 2]);
                min_distance_sq = min_distance_sq.min(dx * dx + dy * dy + dz * dz);
            }
            (min_distance_sq.sqrt() <= POCKET_CONTACT_CUTOFF_ANGSTROM) as i32 as f32
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&values).to_device(device)
}

fn matched_square_float_matrix(
    target: &Tensor,
    device: tch::Device,
    matching: &TargetMatchingResult,
) -> Tensor {
    let atom_count = matching.generated_indices.len() as i64;
    let rows = target.size().first().copied().unwrap_or(0).max(0);
    let cols = target.size().get(1).copied().unwrap_or(0).max(0);
    let values = (0..atom_count)
        .flat_map(|row| {
            (0..atom_count).map(move |col| {
                match matched_pair_indices(matching, row as usize, col as usize, rows, cols) {
                    Some((source_row, source_col)) => {
                        target.double_value(&[source_row, source_col]) as f32
                    }
                    _ => 0.0,
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&values)
        .reshape([atom_count, atom_count])
        .to_device(device)
}

fn matched_square_long_matrix(
    target: &Tensor,
    device: tch::Device,
    matching: &TargetMatchingResult,
) -> Tensor {
    let atom_count = matching.generated_indices.len() as i64;
    let rows = target.size().first().copied().unwrap_or(0).max(0);
    let cols = target.size().get(1).copied().unwrap_or(0).max(0);
    let values = (0..atom_count)
        .flat_map(|row| {
            (0..atom_count).map(move |col| {
                match matched_pair_indices(matching, row as usize, col as usize, rows, cols) {
                    Some((source_row, source_col)) => target.int64_value(&[source_row, source_col]),
                    _ => 0,
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&values)
        .reshape([atom_count, atom_count])
        .to_device(device)
}

fn matched_pair_indices(
    matching: &TargetMatchingResult,
    row: usize,
    col: usize,
    source_rows: i64,
    source_cols: i64,
) -> Option<(i64, i64)> {
    let source_row = matching.target_indices.get(row).copied().flatten()?;
    let source_col = matching.target_indices.get(col).copied().flatten()?;
    (source_row >= 0 && source_row < source_rows && source_col >= 0 && source_col < source_cols)
        .then_some((source_row, source_col))
}

fn target_matching_coverage(matching: &TargetMatchingResult) -> f64 {
    if matching.generated_indices.is_empty() {
        1.0
    } else {
        matching.cost_summary.matched_count as f64 / matching.generated_indices.len() as f64
    }
}

fn pair_mask_from_atom_mask(atom_mask: &Tensor) -> Tensor {
    let atom_count = atom_mask.size().first().copied().unwrap_or(0).max(0);
    if atom_count == 0 {
        return Tensor::zeros([0, 0], (Kind::Float, atom_mask.device()));
    }
    let pair_mask =
        atom_mask.to_kind(Kind::Float).unsqueeze(1) * atom_mask.to_kind(Kind::Float).unsqueeze(0);
    let eye = Tensor::eye(atom_count, (Kind::Float, atom_mask.device()));
    pair_mask * (Tensor::ones_like(&eye) - eye)
}

fn dense_bond_type_adjacency(example: &MolecularExample, device: tch::Device) -> Tensor {
    let atom_count = example
        .topology
        .atom_types
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0);
    let dense = Tensor::zeros([atom_count, atom_count], (Kind::Int64, device));
    let edge_count = example
        .topology
        .edge_index
        .size()
        .get(1)
        .copied()
        .unwrap_or(0)
        .min(
            example
                .topology
                .bond_types
                .size()
                .first()
                .copied()
                .unwrap_or(0),
        );
    for edge_ix in 0..edge_count {
        let src = example.topology.edge_index.int64_value(&[0, edge_ix]);
        let dst = example.topology.edge_index.int64_value(&[1, edge_ix]);
        let bond_type = example
            .topology
            .bond_types
            .int64_value(&[edge_ix])
            .clamp(0, 7);
        if src >= 0 && dst >= 0 && src < atom_count && dst < atom_count && bond_type > 0 {
            let _ = dense.get(src).get(dst).fill_(bond_type);
            let _ = dense.get(dst).get(src).fill_(bond_type);
        }
    }
    dense
}

pub(crate) fn flow_matching_t_from_example(
    example: &MolecularExample,
    context: Option<&InteractionExecutionContext>,
) -> f64 {
    if let Some(flow_t) = context.and_then(|context| context.flow_t) {
        return flow_t.clamp(0.05, 0.95);
    }
    let seed = flow_matching_time_seed(example, context);
    let scaled = (seed % 10_000) as f64 / 10_000.0;
    scaled.clamp(0.05, 0.95)
}

fn flow_matching_time_seed(
    example: &MolecularExample,
    context: Option<&InteractionExecutionContext>,
) -> u64 {
    let mut seed = example
        .decoder_supervision
        .corruption_metadata
        .corruption_seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);

    for byte in example.example_id.as_bytes() {
        seed ^= u64::from(*byte);
        seed = seed.rotate_left(7).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    if let Some(context) = context {
        if let Some(training_stage) = context.training_stage {
            seed ^= (training_stage as u64).wrapping_mul(0x2d35_0e35_f2b3_7d8f);
        }
        if let Some(training_step) = context.training_step {
            seed ^= (training_step as u64).wrapping_mul(0x8d5c_d3d9_6b1a_8f13);
        }
        if let Some(epoch_index) = context.epoch_index {
            seed ^= (epoch_index as u64).wrapping_mul(0x1e6a_5a35_4b3a_67f1);
        }
        if let Some(sample_order_seed) = context.sample_order_seed {
            seed ^= sample_order_seed.wrapping_mul(0x7e0f_4f5f_11ac_89cb);
        }
    }

    seed
}

pub(crate) fn gate_summary_from_interactions(
    interactions: &CrossModalInteractions,
) -> GenerationGateSummary {
    GenerationGateSummary {
        topo_from_geo: scalar_gate_from_tensor(&interactions.topo_from_geo.gate),
        topo_from_pocket: scalar_gate_from_tensor(&interactions.topo_from_pocket.gate),
        geo_from_topo: scalar_gate_from_tensor(&interactions.geo_from_topo.gate),
        geo_from_pocket: scalar_gate_from_tensor(&interactions.geo_from_pocket.gate),
        pocket_from_topo: scalar_gate_from_tensor(&interactions.pocket_from_topo.gate),
        pocket_from_geo: scalar_gate_from_tensor(&interactions.pocket_from_geo.gate),
    }
}

pub(crate) fn flow_conditioning_state(
    generation_state: &ConditionedGenerationState,
    gate_summary: GenerationGateSummary,
) -> ConditioningState {
    ConditioningState {
        topology_context: generation_state.topology_context.shallow_clone(),
        geometry_context: generation_state.geometry_context.shallow_clone(),
        pocket_context: generation_state.pocket_context.shallow_clone(),
        gate_summary,
    }
}

pub(crate) fn scalar_gate_from_tensor(gate: &Tensor) -> f64 {
    if gate.numel() == 0 {
        0.0
    } else {
        gate.mean(Kind::Float).double_value(&[])
    }
}

fn deterministic_flow_noise(coords: &Tensor, std: f64, seed: u64) -> Tensor {
    if std <= 0.0 || coords.numel() == 0 {
        return Tensor::zeros_like(coords);
    }
    let atom_count = coords.size()[0].max(0) as usize;
    let mut values = Vec::with_capacity(atom_count * 3);
    for atom_ix in 0..atom_count {
        for axis in 0..3 {
            let centered = deterministic_flow_unit(seed, atom_ix, axis) * 2.0 - 1.0;
            values.push((centered * std) as f32);
        }
    }
    Tensor::from_slice(&values)
        .reshape([atom_count as i64, 3])
        .to_device(coords.device())
}

fn deterministic_flow_unit(seed: u64, atom_ix: usize, axis: usize) -> f64 {
    let mut value = seed
        ^ ((atom_ix as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        ^ ((axis as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value as f64) / (u64::MAX as f64)
}

fn pocket_radius_from_coords(pocket_coords: &Tensor, pocket_centroid: &Tensor) -> f64 {
    if pocket_coords.numel() == 0 {
        return 0.0;
    }
    (pocket_coords - pocket_centroid.unsqueeze(0))
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
}
