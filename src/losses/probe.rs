//! Semantic probe supervision for specialized modality paths.

use tch::{Kind, Reduction, Tensor};

use crate::{data::MolecularExample, models::ResearchForward};

/// Lightweight supervision over topology, geometry, and pocket probe heads.
#[derive(Debug, Default, Clone)]
pub struct ProbeLoss;

impl ProbeLoss {
    /// Compute the semantic probe objective for one example.
    pub fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        self.compute_weighted(example, forward, 1.0)
    }

    /// Compute the semantic probe objective with an optional affinity weight.
    pub fn compute_weighted(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
        affinity_weight: f64,
    ) -> Tensor {
        let topo_loss = if forward.probes.topology_adjacency_logits.numel() == 0 {
            Tensor::zeros(
                [1],
                (
                    Kind::Float,
                    forward.probes.topology_adjacency_logits.device(),
                ),
            )
        } else {
            forward
                .probes
                .topology_adjacency_logits
                .binary_cross_entropy_with_logits::<Tensor>(
                    &example.topology.adjacency,
                    None,
                    None,
                    Reduction::Mean,
                )
        };

        let target_distances =
            example
                .geometry
                .pairwise_distances
                .mean_dim([1].as_slice(), false, Kind::Float);
        let geo_loss = if forward.probes.geometry_distance_predictions.numel() == 0 {
            Tensor::zeros([1], (Kind::Float, target_distances.device()))
        } else {
            (forward.probes.geometry_distance_predictions.shallow_clone() - target_distances)
                .pow_tensor_scalar(2.0)
                .mean(Kind::Float)
        };

        let pocket_loss = if forward.probes.pocket_feature_predictions.numel() == 0 {
            Tensor::zeros([1], (Kind::Float, example.pocket.atom_features.device()))
        } else {
            (forward.probes.pocket_feature_predictions.shallow_clone()
                - example.pocket.atom_features.shallow_clone())
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
        };

        let affinity_loss = if let Some(target_affinity) = example.targets.affinity_kcal_mol {
            (forward.probes.affinity_prediction.shallow_clone()
                - Tensor::from(target_affinity as f64)
                    .to_kind(Kind::Float)
                    .to_device(forward.probes.affinity_prediction.device()))
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
                * affinity_weight
        } else {
            Tensor::zeros(
                [1],
                (Kind::Float, forward.probes.affinity_prediction.device()),
            )
        };

        topo_loss + geo_loss + pocket_loss + affinity_loss
    }
}
