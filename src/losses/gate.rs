//! Gate sparsity regularization.

use tch::{Kind, Tensor};

use crate::models::CrossModalInteractions;

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
}
