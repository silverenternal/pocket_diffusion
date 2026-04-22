//! Surrogate task objective for the Phase 3 training skeleton.

use tch::{Kind, Tensor};

use crate::models::ResearchForward;

/// Task loss based on reconstructing each modality from its structured latent path.
#[derive(Debug, Default, Clone)]
pub struct TaskLoss;

impl TaskLoss {
    /// Compute a reconstruction-style task loss over the modality-specific paths.
    pub fn compute(&self, forward: &ResearchForward) -> Tensor {
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
        topo + geo + pocket
    }
}

fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    if pred.numel() == 0 || target.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, pred.device()))
    } else {
        (pred - target).pow_tensor_scalar(2.0).mean(Kind::Float)
    }
}
