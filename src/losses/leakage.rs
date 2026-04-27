//! Leakage control via off-modality similarity margins.
//!
//! Current implementation is a lightweight proxy: it penalizes cross-modality slot
//! cosine similarity above a dynamic margin. This is useful as an early warning
//! signal, but low similarity alone does not prove semantic non-leakage.
//! A stronger formulation should use explicit off-modality prediction probes and
//! penalize predictive power on wrong-modality targets.

use tch::{Kind, Tensor};

use crate::{data::MolecularExample, models::ResearchForward};

/// Penalizes wrong-modality representations that become too predictive of off-modality targets.
#[derive(Debug, Clone)]
pub struct LeakageLoss {
    /// Margin under which off-modality prediction is considered excessive.
    pub margin: f64,
}

impl Default for LeakageLoss {
    fn default() -> Self {
        Self { margin: 0.25 }
    }
}

impl LeakageLoss {
    /// Compute a margin-based leakage proxy penalty.
    ///
    /// This is intentionally conservative and diagnostic-oriented rather than a
    /// strict information-theoretic leakage guarantee.
    pub(crate) fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        let (topo_geo_similarity, topo_pocket_similarity, geo_pocket_similarity) =
            pairwise_slot_similarities(forward);
        let leakage_budget = leakage_budget(example, self.margin);

        relu_scalar(topo_geo_similarity - leakage_budget)
            + relu_scalar(topo_pocket_similarity - leakage_budget)
            + relu_scalar(geo_pocket_similarity - leakage_budget)
    }

    /// Compute the mean leakage penalty over a mini-batch.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> Tensor {
        debug_assert_eq!(examples.len(), forwards.len());
        let device = forwards
            .first()
            .map(|forward| forward.slots.topology.slots.device())
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
        let mut sum_topo_geo = 0.0;
        let mut sum_topo_pocket = 0.0;
        let mut sum_geo_pocket = 0.0;
        let mut sum_budget = 0.0;
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let (topo_geo_similarity, topo_pocket_similarity, geo_pocket_similarity) =
                pairwise_slot_similarities(forward);
            let budget = leakage_budget(example, self.margin);
            sum_topo_geo += topo_geo_similarity;
            sum_topo_pocket += topo_pocket_similarity;
            sum_geo_pocket += geo_pocket_similarity;
            sum_budget += budget;
            total += self.compute(example, forward).to_device(device);
        }
        let denom = examples.len() as f64;
        log::debug!(
            "leakage diagnostics batch_mean topo_geo={:.4} topo_pocket={:.4} geo_pocket={:.4} budget={:.4}",
            sum_topo_geo / denom,
            sum_topo_pocket / denom,
            sum_geo_pocket / denom,
            sum_budget / denom,
        );
        total / examples.len() as f64
    }
}

fn pairwise_slot_similarities(forward: &ResearchForward) -> (f64, f64, f64) {
    let topo_slots = &forward.slots.topology.slots;
    let geo_slots = &forward.slots.geometry.slots;
    let pocket_slots = &forward.slots.pocket.slots;
    (
        mean_cosine_similarity(topo_slots, geo_slots),
        mean_cosine_similarity(topo_slots, pocket_slots),
        mean_cosine_similarity(geo_slots, pocket_slots),
    )
}

fn leakage_budget(example: &MolecularExample, margin: f64) -> f64 {
    let adjacency_density = if example.topology.adjacency.numel() == 0 {
        0.0
    } else {
        example
            .topology
            .adjacency
            .mean(Kind::Float)
            .double_value(&[])
    };
    let pocket_energy = if example.pocket.atom_features.numel() == 0 {
        0.0
    } else {
        example
            .pocket
            .atom_features
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
            .double_value(&[])
    };
    margin + 0.05 * adjacency_density + 0.01 * pocket_energy
}

fn mean_cosine_similarity(a: &Tensor, b: &Tensor) -> f64 {
    if a.numel() == 0 || b.numel() == 0 {
        return 0.0;
    }
    let a_mean = a.mean_dim([0].as_slice(), false, Kind::Float);
    let b_mean = b.mean_dim([0].as_slice(), false, Kind::Float);
    let dot = (&a_mean * &b_mean).sum(Kind::Float).double_value(&[]);
    let a_norm = a_mean
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt()
        .double_value(&[]);
    let b_norm = b_mean
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt()
        .double_value(&[]);
    dot / ((a_norm * b_norm).max(1e-6))
}

fn relu_scalar(value: f64) -> Tensor {
    Tensor::from(value.max(0.0)).to_kind(Kind::Float)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn relu_scalar_positive_input() {
        let result = relu_scalar(0.5);
        assert_eq!(result.double_value(&[]), 0.5);
    }

    #[test]
    fn relu_scalar_negative_input() {
        let result = relu_scalar(-0.3);
        assert_eq!(result.double_value(&[]), 0.0);
    }

    #[test]
    fn mean_cosine_similarity_identical_vectors() {
        let ones = Tensor::ones(&[10], (Kind::Float, Device::Cpu));
        let sim = mean_cosine_similarity(&ones, &ones);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical vectors should have cosine similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn mean_cosine_similarity_random_vectors() {
        let x = Tensor::randn(&[100], (Kind::Float, Device::Cpu));
        let y = Tensor::randn(&[100], (Kind::Float, Device::Cpu));
        let sim = mean_cosine_similarity(&x, &y);
        // Mean cosine similarity should be approximately in [-1, 1] (with floating point tolerance)
        assert!(
            sim >= -1.01 && sim <= 1.01,
            "Cosine similarity should be approx in [-1, 1], got {}",
            sim
        );
    }

    #[test]
    fn mean_cosine_similarity_empty_tensors() {
        let empty = Tensor::zeros(&[0], (Kind::Float, Device::Cpu));
        let sim = mean_cosine_similarity(&empty, &empty);
        assert_eq!(sim, 0.0, "Empty tensors should have 0 similarity");
    }

    #[test]
    fn leakage_budget_formula() {
        // Just verify the function exists and returns reasonable values
        let examples = crate::data::synthetic_phase1_examples();
        let budget = leakage_budget(&examples[0], 0.25);
        assert!(budget > 0.0, "Budget should be positive");
    }
}
