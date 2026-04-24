//! Gate sparsity regularization.

use tch::{Kind, Tensor};

use crate::models::{CrossModalInteractions, ResearchForward};

/// Penalizes excessive cross-modality gate activation.
#[derive(Debug, Default, Clone)]
pub struct GateLoss;

impl GateLoss {
    /// Compute the average gate magnitude across all directed interactions.
    pub(crate) fn compute(&self, interactions: &CrossModalInteractions) -> Tensor {
        let gates = Tensor::stack(
            &[
                interactions.topo_from_geo.gate.shallow_clone(),
                interactions.topo_from_pocket.gate.shallow_clone(),
                interactions.geo_from_topo.gate.shallow_clone(),
                interactions.geo_from_pocket.gate.shallow_clone(),
                interactions.pocket_from_topo.gate.shallow_clone(),
                interactions.pocket_from_geo.gate.shallow_clone(),
            ],
            0,
        );
        gates.abs().mean(Kind::Float)
    }

    /// Compute the mean gate sparsity penalty over a mini-batch.
    pub(crate) fn compute_batch(&self, forwards: &[ResearchForward]) -> Tensor {
        let device = forwards
            .first()
            .map(|forward| forward.interactions.topo_from_geo.gate.device())
            .unwrap_or(tch::Device::Cpu);
        if forwards.is_empty() {
            return Tensor::zeros([1], (Kind::Float, device));
        }

        let mut total = Tensor::zeros([1], (Kind::Float, device));
        for forward in forwards {
            total += self.compute(&forward.interactions);
        }
        total / forwards.len() as f64
    }
}
