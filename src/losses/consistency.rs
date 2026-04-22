//! Topology-geometry consistency objective.

use tch::{Kind, Tensor};

use crate::{data::MolecularExample, models::ResearchForward};

/// Encourages topological proximity predictions to align with geometric locality.
#[derive(Debug, Clone)]
pub struct ConsistencyLoss {
    /// Distance cutoff used to derive geometry-based neighborhood targets.
    pub distance_cutoff: f64,
}

impl Default for ConsistencyLoss {
    fn default() -> Self {
        Self {
            distance_cutoff: 2.5,
        }
    }
}

impl ConsistencyLoss {
    /// Compute a consistency penalty between topology logits and geometry-induced proximity.
    pub fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        if forward.probes.topology_adjacency_logits.numel() == 0 {
            return Tensor::zeros(
                [1],
                (Kind::Float, example.geometry.pairwise_distances.device()),
            );
        }
        let geom_target = example
            .geometry
            .pairwise_distances
            .lt(self.distance_cutoff)
            .to_kind(Kind::Float);
        let predicted = forward.probes.topology_adjacency_logits.sigmoid();
        (predicted - geom_target)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
    }
}
