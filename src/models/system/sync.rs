impl Phase1ResearchSystem {
    fn sync_context_from_example(
        &self,
        example: &MolecularExample,
        slots: &DecomposedModalities,
    ) -> ModalitySyncContext {
        ModalitySyncContext {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            ligand_atom_count: example.topology.atom_types.size()[0],
            pocket_atom_count: example.pocket.coords.size()[0],
            topology_mask_count: example.topology.atom_types.size()[0],
            geometry_mask_count: example.geometry.coords.size()[0],
            pocket_mask_count: example.pocket.coords.size()[0],
            topology_slot_count: slots.topology.slot_weights.size()[0],
            geometry_slot_count: slots.geometry.slot_weights.size()[0],
            pocket_slot_count: slots.pocket.slot_weights.size()[0],
            coordinate_frame_origin: example.coordinate_frame_origin,
            device_kind: format!("{:?}", example.topology.atom_types.device()),
            flow_t: None,
            rollout_step_index: None,
        }
    }

    pub(super) fn refresh_generation_context(
        &self,
        example: &MolecularExample,
        state: &mut ConditionedGenerationState,
        step_index: usize,
        interaction_execution_context: &InteractionExecutionContext,
    ) -> GenerationGateSummary {
        let refreshed_example = example_with_partial_ligand_state(example, &state.partial_ligand);
        let encodings =
            self.apply_modality_focus_to_encodings(self.encode_example(&refreshed_example));
        let slots = self.decompose_modalities(&encodings);
        let mut refresh_context = interaction_execution_context.clone();
        refresh_context.rollout_step_index = Some(step_index);
        let (interactions, diagnostics) = self
            .interaction_stack
            .block
            .forward_example_with_diagnostics_with_context(
                &refreshed_example,
                &slots,
                refresh_context,
            );

        state.topology_context = masked_refresh_context(
            merge_slot_contexts(
                &slots.topology.slots,
                &[&interactions.topo_from_geo, &interactions.topo_from_pocket],
            ),
            &slots.topology.active_slot_mask,
        );
        state.geometry_context = masked_refresh_context(
            merge_slot_contexts(
                &slots.geometry.slots,
                &[&interactions.geo_from_topo, &interactions.geo_from_pocket],
            ),
            &slots.geometry.active_slot_mask,
        );
        state.pocket_context = masked_refresh_context(
            merge_slot_contexts(
                &slots.pocket.slots,
                &[
                    &interactions.pocket_from_topo,
                    &interactions.pocket_from_geo,
                ],
            ),
            &slots.pocket.active_slot_mask,
        );
        state.topology_slot_mask = slots.topology.active_slot_mask.shallow_clone();
        state.geometry_slot_mask = slots.geometry.active_slot_mask.shallow_clone();
        state.pocket_slot_mask = slots.pocket.active_slot_mask.shallow_clone();

        gate_summary_from_interaction_diagnostics(&diagnostics)
    }
}

fn masked_refresh_context(context: Tensor, active_slot_mask: &Tensor) -> Tensor {
    let slot_count = context.size()[0];
    if active_slot_mask.size().as_slice() == [slot_count] {
        let device = context.device();
        context * active_slot_mask.to_device(device).unsqueeze(-1)
    } else {
        context
    }
}

fn example_with_partial_ligand_state(
    example: &MolecularExample,
    partial_ligand: &PartialLigandState,
) -> MolecularExample {
    let mut refreshed = example.clone();
    refreshed.topology.atom_types = partial_ligand.atom_types.shallow_clone();
    refreshed.topology.chemistry_roles =
        chemistry_roles_from_atom_type_tensor(&partial_ligand.atom_types);
    refreshed.geometry.coords = partial_ligand.coords.shallow_clone();
    refreshed.geometry.pairwise_distances = pairwise_distances_from_coords(&partial_ligand.coords);
    refreshed
}

fn chemistry_roles_from_atom_type_tensor(
    atom_types: &Tensor,
) -> crate::data::ChemistryRoleFeatureMatrix {
    let device = atom_types.device();
    let role_atom_types = (0..atom_types.size().first().copied().unwrap_or(0).max(0))
        .map(|index| atom_type_from_index(atom_types.int64_value(&[index])))
        .collect::<Vec<_>>();
    chemistry_role_features_from_atom_types(&role_atom_types).to_device(device)
}

fn atom_type_from_index(index: i64) -> AtomType {
    match index {
        0 => AtomType::Carbon,
        1 => AtomType::Nitrogen,
        2 => AtomType::Oxygen,
        3 => AtomType::Sulfur,
        4 => AtomType::Hydrogen,
        _ => AtomType::Other,
    }
}

fn pairwise_distances_from_coords(coords: &Tensor) -> Tensor {
    if coords.numel() == 0 {
        let atom_count = coords.size().first().copied().unwrap_or(0).max(0);
        return Tensor::zeros([atom_count, atom_count], (Kind::Float, coords.device()));
    }
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .sqrt()
}
