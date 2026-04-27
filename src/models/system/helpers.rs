fn merge_slot_contexts(base_slots: &Tensor, updates: &[&CrossAttentionOutput]) -> Tensor {
    let mut merged = base_slots.shallow_clone();
    for update in updates {
        merged = &merged + &update.attended_tokens;
    }
    merged
}

fn slice_encoded_modalities(
    batched: &BatchedEncodedModalities,
    batch_index: i64,
    example: &MolecularExample,
) -> EncodedModalities {
    let ligand_atoms = example.topology.atom_types.size()[0];
    let pocket_atoms = example.pocket.coords.size()[0];
    EncodedModalities {
        topology: slice_modality_encoding(&batched.topology, batch_index, ligand_atoms),
        geometry: slice_modality_encoding(&batched.geometry, batch_index, ligand_atoms),
        pocket: slice_modality_encoding(&batched.pocket, batch_index, pocket_atoms),
    }
}

fn slice_modality_encoding(
    batched: &BatchedModalityEncoding,
    batch_index: i64,
    token_count: i64,
) -> ModalityEncoding {
    ModalityEncoding {
        token_embeddings: batched
            .token_embeddings
            .get(batch_index)
            .narrow(0, 0, token_count)
            .shallow_clone(),
        pooled_embedding: batched.pooled_embedding.get(batch_index),
    }
}

fn slice_decomposed_modalities(
    batched: &BatchedDecomposedModalities,
    batch_index: i64,
    example: &MolecularExample,
) -> DecomposedModalities {
    let ligand_atoms = example.topology.atom_types.size()[0];
    let pocket_atoms = example.pocket.coords.size()[0];
    DecomposedModalities {
        topology: slice_slot_encoding(&batched.topology, batch_index, ligand_atoms),
        geometry: slice_slot_encoding(&batched.geometry, batch_index, ligand_atoms),
        pocket: slice_slot_encoding(&batched.pocket, batch_index, pocket_atoms),
    }
}

fn slice_slot_encoding(
    batched: &BatchedSlotEncoding,
    batch_index: i64,
    token_count: i64,
) -> SlotEncoding {
    SlotEncoding {
        slots: batched.slots.get(batch_index),
        slot_weights: batched.slot_weights.get(batch_index),
        reconstructed_tokens: batched
            .reconstructed_tokens
            .get(batch_index)
            .narrow(0, 0, token_count)
            .shallow_clone(),
    }
}

fn slice_cross_modal_interactions(
    batched: &BatchedCrossModalInteractions,
    batch_index: i64,
) -> CrossModalInteractions {
    CrossModalInteractions {
        topo_from_geo: slice_cross_attention(&batched.topo_from_geo, batch_index),
        topo_from_pocket: slice_cross_attention(&batched.topo_from_pocket, batch_index),
        geo_from_topo: slice_cross_attention(&batched.geo_from_topo, batch_index),
        geo_from_pocket: slice_cross_attention(&batched.geo_from_pocket, batch_index),
        pocket_from_topo: slice_cross_attention(&batched.pocket_from_topo, batch_index),
        pocket_from_geo: slice_cross_attention(&batched.pocket_from_geo, batch_index),
    }
}

fn slice_cross_attention(
    batched: &BatchedCrossAttentionOutput,
    batch_index: i64,
) -> CrossAttentionOutput {
    CrossAttentionOutput {
        gate: batched.gate.get(batch_index),
        attended_tokens: batched.attended_tokens.get(batch_index),
        attention_weights: batched.attention_weights.get(batch_index),
    }
}

fn ligand_pocket_slot_attention_bias(
    ligand_coords: &Tensor,
    ligand_mask: &Tensor,
    pocket_coords: &Tensor,
    pocket_mask: &Tensor,
    ligand_slots: i64,
    pocket_slots: i64,
) -> Tensor {
    let batch = ligand_coords.size()[0];
    if ligand_slots == 0 || pocket_slots == 0 {
        return Tensor::zeros(
            [batch, ligand_slots, pocket_slots],
            (Kind::Float, ligand_coords.device()),
        );
    }
    let pair_mask = ligand_mask.unsqueeze(2) * pocket_mask.unsqueeze(1);
    let diffs = ligand_coords.unsqueeze(2) - pocket_coords.unsqueeze(1);
    let distances = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([3].as_slice(), false, Kind::Float)
        .sqrt();
    let valid_distances = distances.shallow_clone() * &pair_mask;
    let denom = pair_mask
        .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        .clamp_min(1.0);
    let mean_distance =
        valid_distances.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float) / denom;
    let contact_fraction = distances.lt(4.5).to_kind(Kind::Float) * &pair_mask;
    let contact_fraction = contact_fraction.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        / pair_mask
            .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
            .clamp_min(1.0);
    let scalar_bias = contact_fraction - mean_distance.clamp_max(12.0) / 12.0;
    scalar_bias
        .reshape([batch, 1, 1])
        .expand([batch, ligand_slots, pocket_slots], true)
}

fn tensor_to_i64_vec(tensor: &Tensor) -> Vec<i64> {
    let len = tensor.size().first().copied().unwrap_or(0).max(0) as usize;
    (0..len)
        .map(|index| tensor.int64_value(&[index as i64]))
        .collect()
}

fn tensor_to_coords(tensor: &Tensor) -> Vec<[f32; 3]> {
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

fn clip_coordinate_delta_norm(delta: &Tensor, max_norm: f64) -> Tensor {
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

fn per_atom_displacement_mean(delta: &Tensor) -> f64 {
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

fn atom_change_fraction(current: &Tensor, next: &Tensor) -> f64 {
    if current.numel() == 0 || next.numel() == 0 {
        return 0.0;
    }
    current
        .ne_tensor(next)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn pocket_guidance_delta(
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

fn constrain_to_pocket_envelope(
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

fn flow_matching_x0(example: &MolecularExample, noise_scale: f64, use_corrupted_x0: bool) -> Tensor {
    let base = if use_corrupted_x0 {
        example.decoder_supervision.noisy_coords.shallow_clone()
    } else {
        let coords = &example.decoder_supervision.target_coords;
        let atom_count = coords.size()[0].max(0) as usize;
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
            for axis in 0..3 {
                let unit = deterministic_flow_unit(
                    example.decoder_supervision.corruption_metadata.corruption_seed,
                    atom_ix,
                    axis,
                );
                let centered = unit * 2.0 - 1.0;
                values.push((pocket_centroid[axis] + centered * noise_scale) as f32);
            }
        }
        Tensor::from_slice(&values)
            .reshape([atom_count as i64, 3])
            .to_device(coords.device())
    };

    if noise_scale <= 0.0 {
        base
    } else {
        &base
            + deterministic_flow_noise(
            &base,
            noise_scale,
            example
                .decoder_supervision
                .corruption_metadata
                .corruption_seed
                .wrapping_add(17),
        )
    }
}

fn flow_matching_t_from_example(example: &MolecularExample) -> f64 {
    let seed = example
        .decoder_supervision
        .corruption_metadata
        .corruption_seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let scaled = (seed % 10_000) as f64 / 10_000.0;
    scaled.clamp(0.05, 0.95)
}

fn gate_summary_from_interactions(interactions: &CrossModalInteractions) -> GenerationGateSummary {
    GenerationGateSummary {
        topo_from_geo: scalar_gate_from_tensor(&interactions.topo_from_geo.gate),
        topo_from_pocket: scalar_gate_from_tensor(&interactions.topo_from_pocket.gate),
        geo_from_topo: scalar_gate_from_tensor(&interactions.geo_from_topo.gate),
        geo_from_pocket: scalar_gate_from_tensor(&interactions.geo_from_pocket.gate),
        pocket_from_topo: scalar_gate_from_tensor(&interactions.pocket_from_topo.gate),
        pocket_from_geo: scalar_gate_from_tensor(&interactions.pocket_from_geo.gate),
    }
}

fn flow_conditioning_state(
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

fn scalar_gate_from_tensor(gate: &Tensor) -> f64 {
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

fn resolved_generation_backend_family(config: &ResearchConfig) -> GenerationBackendFamilyConfig {
    match config.generation_method.primary_backend_id() {
        "flow_matching" => GenerationBackendFamilyConfig::FlowMatching,
        "autoregressive_graph_geometry" => GenerationBackendFamilyConfig::Autoregressive,
        "energy_guided_refinement" => GenerationBackendFamilyConfig::EnergyGuidedRefinement,
        "conditioned_denoising" => GenerationBackendFamilyConfig::ConditionedDenoising,
        _ => config.generation_method.primary_backend.family,
    }
}
