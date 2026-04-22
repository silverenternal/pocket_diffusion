//! Intra-modality redundancy reduction losses.

use tch::{Kind, Tensor};

use crate::models::DecomposedModalities;

/// Redundancy objective combining covariance decorrelation, predictability, and optional HSIC.
#[derive(Debug, Clone)]
pub struct IntraRedundancyLoss {
    /// Whether to include the HSIC-like penalty term.
    pub enable_hsic: bool,
}

impl Default for IntraRedundancyLoss {
    fn default() -> Self {
        Self { enable_hsic: true }
    }
}

impl IntraRedundancyLoss {
    /// Compute the aggregate intra-modality redundancy penalty.
    pub(crate) fn compute(&self, slots: &DecomposedModalities) -> Tensor {
        let topo = self.modality_loss(&slots.topology.slots);
        let geo = self.modality_loss(&slots.geometry.slots);
        let pocket = self.modality_loss(&slots.pocket.slots);
        (topo + geo + pocket) / 3.0
    }

    fn modality_loss(&self, slot_matrix: &Tensor) -> Tensor {
        if slot_matrix.size()[0] <= 1 {
            return Tensor::zeros([1], (Kind::Float, slot_matrix.device()));
        }

        let centered = slot_matrix - slot_matrix.mean_dim([0].as_slice(), true, Kind::Float);
        let denom = (slot_matrix.size()[0] - 1).max(1) as f64;
        let covariance = centered.transpose(0, 1).matmul(&centered) / denom;
        let dim = covariance.size()[0];
        let eye = Tensor::eye(dim, (Kind::Float, slot_matrix.device()));
        let off_diag_mask = Tensor::ones([dim, dim], (Kind::Float, slot_matrix.device())) - &eye;
        let off_diag = &covariance * off_diag_mask.shallow_clone();
        let covariance_penalty = off_diag.pow_tensor_scalar(2.0).mean(Kind::Float);

        let normalized = slot_matrix
            / slot_matrix
                .pow_tensor_scalar(2.0)
                .sum_dim_intlist([1].as_slice(), true, Kind::Float)
                .sqrt()
                .clamp_min(1e-6);
        let slot_similarity = normalized.matmul(&normalized.transpose(0, 1));
        let slot_count = slot_similarity.size()[0];
        let slot_eye = Tensor::eye(slot_count, (Kind::Float, slot_matrix.device()));
        let slot_off_diag_mask = Tensor::ones(
            [slot_count, slot_count],
            (Kind::Float, slot_matrix.device()),
        ) - slot_eye;
        let predictability_penalty = (slot_similarity * slot_off_diag_mask)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float);

        let hsic_penalty = if self.enable_hsic {
            let gram = centered.matmul(&centered.transpose(0, 1));
            let n = gram.size()[0];
            let centering = Tensor::eye(n, (Kind::Float, slot_matrix.device()))
                - Tensor::ones([n, n], (Kind::Float, slot_matrix.device())) / (n as f64);
            let centered_gram = centering.matmul(&gram).matmul(&centering);
            centered_gram.pow_tensor_scalar(2.0).mean(Kind::Float)
        } else {
            Tensor::zeros([1], (Kind::Float, slot_matrix.device()))
        };

        covariance_penalty + predictability_penalty + 0.01 * hsic_penalty
    }
}
