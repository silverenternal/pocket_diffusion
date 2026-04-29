//! Intra-modality redundancy reduction losses.

use tch::{Kind, Tensor};

use crate::models::{DecomposedModalities, ResearchForward};

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
        let topo = self.modality_loss(&slots.topology.slots, &slots.topology.active_slot_mask);
        let geo = self.modality_loss(&slots.geometry.slots, &slots.geometry.active_slot_mask);
        let pocket = self.modality_loss(&slots.pocket.slots, &slots.pocket.active_slot_mask);
        (topo + geo + pocket) / 3.0
    }

    /// Compute the mean redundancy penalty over a mini-batch.
    pub(crate) fn compute_batch(&self, forwards: &[ResearchForward]) -> Tensor {
        let device = forwards
            .first()
            .map(|forward| forward.slots.topology.slots.device())
            .unwrap_or(tch::Device::Cpu);
        if forwards.is_empty() {
            return Tensor::zeros([1], (Kind::Float, device));
        }

        let mut total = Tensor::zeros([1], (Kind::Float, device));
        for forward in forwards {
            total += self.compute(&forward.slots);
        }
        total / forwards.len() as f64
    }

    fn modality_loss(&self, slot_matrix: &Tensor, active_slot_mask: &Tensor) -> Tensor {
        if slot_matrix.size()[0] <= 1 {
            return Tensor::zeros([1], (Kind::Float, slot_matrix.device()));
        }
        let slot_matrix = active_slot_matrix(slot_matrix, active_slot_mask);
        if slot_matrix.size()[0] <= 1 {
            return Tensor::zeros([1], (Kind::Float, slot_matrix.device()));
        }

        let centered = &slot_matrix - slot_matrix.mean_dim([0].as_slice(), true, Kind::Float);
        let denom = (slot_matrix.size()[0] - 1).max(1) as f64;
        let covariance = centered.transpose(0, 1).matmul(&centered) / denom;
        let dim = covariance.size()[0];
        let eye = Tensor::eye(dim, (Kind::Float, slot_matrix.device()));
        let off_diag_mask = Tensor::ones([dim, dim], (Kind::Float, slot_matrix.device())) - &eye;
        let off_diag = &covariance * off_diag_mask.shallow_clone();
        let covariance_penalty = off_diag.pow_tensor_scalar(2.0).mean(Kind::Float);

        let normalized = &slot_matrix
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

fn active_slot_matrix(slot_matrix: &Tensor, active_slot_mask: &Tensor) -> Tensor {
    let slot_count = slot_matrix.size().first().copied().unwrap_or(0).max(0);
    if slot_count == 0 || active_slot_mask.numel() == 0 {
        return slot_matrix.shallow_clone();
    }
    if active_slot_mask.size().first().copied().unwrap_or(0) != slot_count {
        return slot_matrix.shallow_clone();
    }
    let active_indices = active_slot_mask
        .to_device(slot_matrix.device())
        .gt(0.5)
        .nonzero()
        .squeeze_dim(1);
    let active_count = active_indices.size().first().copied().unwrap_or(0);
    if active_count == 0 {
        Tensor::zeros(
            [0, slot_matrix.size().get(1).copied().unwrap_or(0)],
            (Kind::Float, slot_matrix.device()),
        )
    } else {
        slot_matrix.index_select(0, &active_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn redundancy_masks_inactive_slots_before_covariance() {
        let loss = IntraRedundancyLoss::default();
        let slots = Tensor::from_slice(&[
            1.0_f32, 0.0, //
            0.0, 1.0, //
            100.0, 100.0,
        ])
        .reshape([3, 2]);
        let active_mask = Tensor::from_slice(&[1.0_f32, 1.0, 0.0]);
        let active_only = slots.narrow(0, 0, 2);
        let all_active_mask = Tensor::ones([2], (Kind::Float, Device::Cpu));

        let masked = loss.modality_loss(&slots, &active_mask);
        let expected = loss.modality_loss(&active_only, &all_active_mask);

        let delta = (masked - expected).abs().double_value(&[]);
        assert!(delta <= 1e-6, "inactive slot changed redundancy by {delta}");
    }

    #[test]
    fn redundancy_returns_zero_for_fewer_than_two_active_slots() {
        let loss = IntraRedundancyLoss::default();
        let slots = Tensor::randn([3, 4], (Kind::Float, Device::Cpu));
        let one_active = Tensor::from_slice(&[0.0_f32, 1.0, 0.0]);
        let none_active = Tensor::zeros([3], (Kind::Float, Device::Cpu));

        assert_eq!(
            loss.modality_loss(&slots, &one_active).double_value(&[]),
            0.0
        );
        assert_eq!(
            loss.modality_loss(&slots, &none_active).double_value(&[]),
            0.0
        );
    }
}
