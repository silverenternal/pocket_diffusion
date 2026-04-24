//! Primary-objective implementations for the modular research stack.

use tch::{Kind, Tensor};

use crate::{
    config::PrimaryObjectiveConfig,
    data::MolecularExample,
    models::{ResearchForward, TaskDrivenObjective},
};

/// Reconstruction-style surrogate objective over modality-specific latent paths.
#[derive(Debug, Default, Clone)]
pub struct SurrogateReconstructionObjective;

impl TaskDrivenObjective<ResearchForward> for SurrogateReconstructionObjective {
    fn name(&self) -> &'static str {
        "surrogate_reconstruction"
    }

    fn compute(&self, _example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        let topo = mse(
            &forward.slots.topology.reconstructed_tokens,
            &forward.encodings.topology.token_embeddings,
        );
        let geo = mse(
            &forward.slots.geometry.reconstructed_tokens,
            &forward.encodings.geometry.token_embeddings,
        );
        let pocket = mse(
            &forward.slots.pocket.reconstructed_tokens,
            &forward.encodings.pocket.token_embeddings,
        );
        let decoder_topology = atom_type_bootstrap_loss(
            &forward.generation.decoded.atom_type_logits,
            &forward.generation.state.partial_ligand.atom_types,
        );
        let decoded_coords = &forward.generation.state.partial_ligand.coords
            + &forward.generation.decoded.coordinate_deltas;
        let decoder_geometry = mse(
            &decoded_coords,
            &forward.generation.state.partial_ligand.coords,
        );

        topo + geo + pocket + 0.1 * (decoder_topology + decoder_geometry)
    }
}

/// Decoder-anchored corruption recovery objective for conditioned ligand generation.
#[derive(Debug, Default, Clone)]
pub struct ConditionedDenoisingObjective;

impl TaskDrivenObjective<ResearchForward> for ConditionedDenoisingObjective {
    fn name(&self) -> &'static str {
        "conditioned_denoising"
    }

    fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        let supervision = &example.decoder_supervision;
        let preserved_mask = inverse_mask(&supervision.atom_corruption_mask);
        let single_step_topology = masked_atom_type_loss(
            &forward.generation.decoded.atom_type_logits,
            &supervision.target_atom_types,
            &supervision.atom_corruption_mask,
        );
        let topology_preservation = masked_atom_type_loss(
            &forward.generation.decoded.atom_type_logits,
            &supervision.target_atom_types,
            &preserved_mask,
        );
        let single_step_predicted_coords =
            &supervision.noisy_coords + &forward.generation.decoded.coordinate_deltas;
        let single_step_geometry = masked_coord_denoising_loss(
            &single_step_predicted_coords,
            &supervision.target_coords,
            &supervision.atom_corruption_mask,
        );
        let geometry_preservation = masked_coord_denoising_loss(
            &single_step_predicted_coords,
            &supervision.target_coords,
            &preserved_mask,
        );
        let direct_noise_recovery = masked_coord_denoising_loss(
            &forward.generation.decoded.coordinate_deltas,
            &(-&supervision.coordinate_noise),
            &supervision.atom_corruption_mask,
        );
        let single_step_pairwise = pairwise_distance_recovery_loss(
            &single_step_predicted_coords,
            &supervision.target_pairwise_distances,
            &supervision.atom_corruption_mask,
        );
        let single_step_centroid =
            centroid_recovery_loss(&single_step_predicted_coords, &supervision.target_coords);
        let pocket_anchor =
            pocket_anchor_loss(&single_step_predicted_coords, &example.pocket.coords);
        let rollout = rollout_recovery_loss(supervision, &forward.generation.rollout);
        let rollout_pocket_anchor =
            rollout_pocket_anchor_loss(&forward.generation.rollout, &example.pocket.coords);

        single_step_topology
            + single_step_geometry
            + 0.25 * topology_preservation
            + 0.15 * geometry_preservation
            + 0.2 * direct_noise_recovery
            + 0.2 * single_step_pairwise
            + 0.15 * single_step_centroid
            + 0.2 * pocket_anchor
            + 0.35 * rollout
            + 0.15 * rollout_pocket_anchor
    }
}

/// Build the configured primary objective implementation.
pub(crate) fn build_primary_objective(
    config: PrimaryObjectiveConfig,
) -> Box<dyn TaskDrivenObjective<ResearchForward>> {
    match config {
        PrimaryObjectiveConfig::SurrogateReconstruction => {
            Box::new(SurrogateReconstructionObjective)
        }
        PrimaryObjectiveConfig::ConditionedDenoising => Box::new(ConditionedDenoisingObjective),
    }
}

/// Compute the configured primary objective over a mini-batch with the same
/// per-example mask semantics used by the scalar objective implementations.
pub(crate) fn compute_primary_objective_batch(
    objective: &dyn TaskDrivenObjective<ResearchForward>,
    examples: &[MolecularExample],
    forwards: &[ResearchForward],
) -> Tensor {
    debug_assert_eq!(examples.len(), forwards.len());
    let device = forwards
        .first()
        .map(|forward| forward.generation.decoded.coordinate_deltas.device())
        .or_else(|| {
            examples
                .first()
                .map(|example| example.topology.atom_types.device())
        })
        .unwrap_or(tch::Device::Cpu);
    if examples.is_empty() {
        return Tensor::zeros([1], (Kind::Float, device));
    }

    let mut total = Tensor::zeros([1], (Kind::Float, device));
    for (example, forward) in examples.iter().zip(forwards.iter()) {
        total += objective.compute(example, forward);
    }
    total / examples.len() as f64
}

fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    if pred.numel() == 0 || target.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, pred.device()))
    } else {
        (pred - target).pow_tensor_scalar(2.0).mean(Kind::Float)
    }
}

fn atom_type_bootstrap_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    if logits.numel() == 0 || targets.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, logits.device()))
    } else {
        logits.cross_entropy_for_logits(targets)
    }
}

fn masked_atom_type_loss(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0 || targets.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }

    let per_atom =
        logits.cross_entropy_loss::<Tensor>(targets, None, tch::Reduction::None, -100, 0.0);
    weighted_mean(&per_atom, mask)
}

fn masked_coord_denoising_loss(predicted: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if predicted.numel() == 0 || target.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted.device()));
    }
    let per_atom =
        (predicted - target)
            .pow_tensor_scalar(2.0)
            .mean_dim([1].as_slice(), false, Kind::Float);
    weighted_mean(&per_atom, mask)
}

fn pairwise_distance_recovery_loss(
    predicted_coords: &Tensor,
    target_pairwise_distances: &Tensor,
    mask: &Tensor,
) -> Tensor {
    if predicted_coords.numel() == 0 || target_pairwise_distances.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted_coords.device()));
    }
    let predicted_distances = pairwise_distances(predicted_coords);
    let pair_mask = pairwise_mask(mask);
    let per_pair = (predicted_distances - target_pairwise_distances).pow_tensor_scalar(2.0);
    let denom = pair_mask.sum(Kind::Float).clamp_min(1.0);
    (per_pair * pair_mask).sum(Kind::Float) / denom
}

fn pairwise_distances(coords: &Tensor) -> Tensor {
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    let squared = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float);
    let distances = squared.clamp_min(1e-12).sqrt();
    let count = coords.size().first().copied().unwrap_or(0);
    if count <= 0 {
        distances
    } else {
        let eye = Tensor::eye(count, (Kind::Float, coords.device()));
        let ones = Tensor::ones_like(&eye);
        distances * (&ones - &eye)
    }
}

fn pairwise_mask(mask: &Tensor) -> Tensor {
    let pair_mask = mask.unsqueeze(1) * mask.unsqueeze(0);
    let count = mask.size().first().copied().unwrap_or(0);
    if count <= 0 {
        pair_mask
    } else {
        let eye = Tensor::eye(count, (Kind::Float, mask.device()));
        let ones = Tensor::ones_like(&eye);
        pair_mask * (&ones - &eye)
    }
}

fn inverse_mask(mask: &Tensor) -> Tensor {
    let ones = Tensor::ones_like(mask);
    (&ones - mask).clamp(0.0, 1.0)
}

fn weighted_mean(values: &Tensor, mask: &Tensor) -> Tensor {
    let mask = mask.to_kind(Kind::Float);
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (values * mask).sum(Kind::Float) / denom
}

fn rollout_recovery_loss(
    supervision: &crate::data::DecoderSupervision,
    rollout: &crate::models::GenerationRolloutRecord,
) -> Tensor {
    if rollout.steps.is_empty() {
        return Tensor::zeros([1], (Kind::Float, supervision.target_coords.device()));
    }

    let mut total = Tensor::zeros([1], (Kind::Float, supervision.target_coords.device()));
    let mut weight_sum = 0.0_f64;
    let preserved_mask = inverse_mask(&supervision.atom_corruption_mask);
    for step in &rollout.steps {
        let coords = coords_tensor(&step.coords, supervision.target_coords.device());
        let atom_types = Tensor::from_slice(&step.atom_types)
            .to_kind(Kind::Int64)
            .to_device(supervision.target_atom_types.device());
        let weight = supervision.rollout_step_weight(step.step_index);
        let topology = masked_atom_match_loss(
            &atom_types,
            &supervision.target_atom_types,
            &supervision.atom_corruption_mask,
        );
        let topology_preservation =
            masked_atom_match_loss(&atom_types, &supervision.target_atom_types, &preserved_mask);
        let geometry = masked_coord_denoising_loss(
            &coords,
            &supervision.target_coords,
            &supervision.atom_corruption_mask,
        );
        let geometry_preservation =
            masked_coord_denoising_loss(&coords, &supervision.target_coords, &preserved_mask);
        let pairwise = pairwise_distance_recovery_loss(
            &coords,
            &supervision.target_pairwise_distances,
            &supervision.atom_corruption_mask,
        );
        let stop_target = if step.step_index + 1 >= supervision.rollout_steps {
            1.0
        } else {
            0.0
        };
        let stop_penalty = Tensor::from((step.stop_probability - stop_target).powi(2) as f32)
            .to_device(supervision.target_coords.device());
        let stability_penalty = Tensor::from(
            (step.mean_displacement.powi(2) + step.atom_change_fraction.powi(2)) as f32,
        )
        .to_device(supervision.target_coords.device());
        total += (topology
            + geometry
            + 0.2 * topology_preservation
            + 0.15 * geometry_preservation
            + 0.2 * pairwise
            + 0.05 * stop_penalty
            + 0.02 * stability_penalty)
            * weight;
        weight_sum += weight;
    }

    total / weight_sum.max(1.0)
}

fn rollout_pocket_anchor_loss(
    rollout: &crate::models::GenerationRolloutRecord,
    pocket_coords: &Tensor,
) -> Tensor {
    if rollout.steps.is_empty() || pocket_coords.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, pocket_coords.device()));
    }

    let mut total = Tensor::zeros([1], (Kind::Float, pocket_coords.device()));
    for step in &rollout.steps {
        let coords = coords_tensor(&step.coords, pocket_coords.device());
        total += pocket_anchor_loss(&coords, pocket_coords);
    }
    total / rollout.steps.len() as f64
}

fn masked_atom_match_loss(predicted: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if predicted.numel() == 0 || target.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, target.device()));
    }
    let mismatches = predicted
        .ne_tensor(target)
        .to_kind(Kind::Float)
        .to_device(target.device());
    weighted_mean(&mismatches, mask)
}

fn coords_tensor(coords: &[[f32; 3]], device: tch::Device) -> Tensor {
    if coords.is_empty() {
        Tensor::zeros([0, 3], (Kind::Float, device))
    } else {
        let flat = coords
            .iter()
            .flat_map(|coord| coord.iter().copied())
            .collect::<Vec<_>>();
        Tensor::from_slice(&flat)
            .reshape([coords.len() as i64, 3])
            .to_device(device)
    }
}

fn centroid_recovery_loss(predicted: &Tensor, target: &Tensor) -> Tensor {
    if predicted.numel() == 0 || target.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted.device()));
    }
    let predicted_centroid = predicted.mean_dim([0].as_slice(), false, Kind::Float);
    let target_centroid = target.mean_dim([0].as_slice(), false, Kind::Float);
    (predicted_centroid - target_centroid)
        .pow_tensor_scalar(2.0)
        .mean(Kind::Float)
}

fn pocket_anchor_loss(predicted_coords: &Tensor, pocket_coords: &Tensor) -> Tensor {
    if predicted_coords.numel() == 0 || pocket_coords.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted_coords.device()));
    }
    let predicted_centroid = predicted_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_radius = centered_radius(pocket_coords, &pocket_centroid);
    let centroid_distance = (&predicted_centroid - &pocket_centroid)
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt();
    (centroid_distance
        - Tensor::from((pocket_radius * 0.85) as f32).to_device(predicted_coords.device()))
    .relu()
    .pow_tensor_scalar(2.0)
}

fn centered_radius(coords: &Tensor, centroid: &Tensor) -> f64 {
    if coords.numel() == 0 {
        return 0.0;
    }
    (coords - centroid.unsqueeze(0))
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
}
