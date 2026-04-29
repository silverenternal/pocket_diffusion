use crate::types::CorruptionSourceProvenance;

fn build_decoder_supervision(
    example_id: &str,
    topology: &TopologyFeatures,
    geometry: &GeometryFeatures,
    generation_target: &GenerationTargetConfig,
) -> DecoderSupervision {
    let device = topology.atom_types.device();
    let num_atoms = topology.atom_types.size()[0];
    let target_atom_types = topology.atom_types.shallow_clone();
    let target_coords = geometry.coords.shallow_clone();
    let target_pairwise_distances = geometry.pairwise_distances.shallow_clone();
    let source_is_target_ligand = generation_target
        .generation_mode
        .uses_target_ligand_initialization();
    let source_provenance = if source_is_target_ligand {
        CorruptionSourceProvenance::TargetLigand
    } else {
        CorruptionSourceProvenance::DeNovoSource
    };
    let metadata = GenerationCorruptionMetadata {
        atom_mask_ratio: generation_target.atom_mask_ratio,
        coordinate_noise_std: generation_target.coordinate_noise_std,
        corruption_seed: generation_target.corruption_seed,
        source_is_target_ligand,
        topology_source: source_provenance,
        geometry_source: source_provenance,
    };

    if num_atoms == 0 {
        let empty_long = Tensor::zeros([0], (Kind::Int64, device));
        let empty_float = Tensor::zeros([0], (Kind::Float, device));
        return DecoderSupervision {
            target_atom_types,
            corrupted_atom_types: empty_long,
            atom_corruption_mask: empty_float.shallow_clone(),
            target_coords,
            noisy_coords: Tensor::zeros([0, 3], (Kind::Float, device)),
            coordinate_noise: Tensor::zeros([0, 3], (Kind::Float, device)),
            target_pairwise_distances,
            rollout_steps: generation_target.rollout_steps,
            rollout_eval_step_weight_decay: generation_target.rollout_eval_step_weight_decay,
            corruption_metadata: metadata,
        };
    }

    let mask_values: Vec<f32> = (0..num_atoms)
        .map(|atom_ix| {
            if should_mask_atom(example_id, atom_ix as usize, generation_target) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let atom_corruption_mask = tensor_from_slice(&mask_values)
        .to_kind(Kind::Float)
        .to_device(device);
    let corrupted_atom_types =
        apply_atom_corruption(&target_atom_types, &atom_corruption_mask, generation_target);

    let noise_flat: Vec<f32> = (0..num_atoms)
        .flat_map(|atom_ix| {
            (0..3).map(move |coord_ix| {
                deterministic_noise(
                    example_id,
                    atom_ix as usize,
                    coord_ix as usize,
                    generation_target,
                )
            })
        })
        .collect();
    let coordinate_noise = tensor_from_slice(&noise_flat)
        .reshape([num_atoms, 3])
        .to_device(device);
    let noisy_coords = &target_coords + &coordinate_noise;

    DecoderSupervision {
        target_atom_types,
        corrupted_atom_types,
        atom_corruption_mask,
        target_coords,
        noisy_coords,
        coordinate_noise,
        target_pairwise_distances,
        rollout_steps: generation_target.rollout_steps,
        rollout_eval_step_weight_decay: generation_target.rollout_eval_step_weight_decay,
        corruption_metadata: metadata,
    }
}

fn should_mask_atom(
    example_id: &str,
    atom_ix: usize,
    generation_target: &GenerationTargetConfig,
) -> bool {
    if generation_target.atom_mask_ratio <= 0.0 {
        return false;
    }
    let hash = stable_atom_hash(example_id, atom_ix, generation_target.corruption_seed);
    let normalized = (hash % 10_000) as f32 / 10_000.0;
    normalized < generation_target.atom_mask_ratio
}

fn apply_atom_corruption(
    atom_types: &Tensor,
    mask: &Tensor,
    generation_target: &GenerationTargetConfig,
) -> Tensor {
    let mask_long = mask.to_kind(Kind::Int64);
    let replacement = Tensor::full_like(
        atom_types,
        ((generation_target.corruption_seed % 5) + 1) as i64,
    );
    atom_types * (1 - &mask_long) + replacement * mask_long
}

fn deterministic_noise(
    example_id: &str,
    atom_ix: usize,
    coord_ix: usize,
    generation_target: &GenerationTargetConfig,
) -> f32 {
    if generation_target.coordinate_noise_std == 0.0 {
        return 0.0;
    }
    let hash = stable_atom_hash(
        example_id,
        atom_ix * 17 + coord_ix,
        generation_target.corruption_seed,
    );
    let phase = (hash % 65_521) as f32 / 65_521.0;
    (phase * std::f32::consts::TAU).sin() * generation_target.coordinate_noise_std
}

fn stable_atom_hash(example_id: &str, atom_ix: usize, seed: u64) -> u64 {
    let mut hash = seed ^ 0x9e37_79b9_7f4a_7c15_u64;
    for byte in example_id.as_bytes() {
        hash = hash.rotate_left(7) ^ u64::from(*byte);
        hash = hash.wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    hash ^ (atom_ix as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
}
