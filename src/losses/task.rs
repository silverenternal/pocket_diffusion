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
        let topology = masked_atom_type_loss(
            &forward.generation.decoded.atom_type_logits,
            &supervision.target_atom_types,
            &supervision.atom_corruption_mask,
        );
        let predicted_coords =
            &supervision.noisy_coords + &forward.generation.decoded.coordinate_deltas;
        let geometry = masked_coord_denoising_loss(
            &predicted_coords,
            &supervision.target_coords,
            &supervision.atom_corruption_mask,
        );
        let pairwise = pairwise_distance_recovery_loss(
            &predicted_coords,
            &supervision.target_pairwise_distances,
            &supervision.atom_corruption_mask,
        );
        let stop = forward
            .generation
            .decoded
            .stop_logit
            .sigmoid()
            .mean(Kind::Float);

        topology + geometry + 0.2 * pairwise + 0.05 * stop
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
    let pair_mask = mask.unsqueeze(1) * mask.unsqueeze(0);
    let per_pair = (predicted_distances - target_pairwise_distances).pow_tensor_scalar(2.0);
    let denom = pair_mask.sum(Kind::Float).clamp_min(1.0);
    (per_pair * pair_mask).sum(Kind::Float) / denom
}

fn pairwise_distances(coords: &Tensor) -> Tensor {
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .sqrt()
}

fn weighted_mean(values: &Tensor, mask: &Tensor) -> Tensor {
    let mask = mask.to_kind(Kind::Float);
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (values * mask).sum(Kind::Float) / denom
}
