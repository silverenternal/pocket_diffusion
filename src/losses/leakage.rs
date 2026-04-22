//! Leakage control via off-modality similarity margins.

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
    /// Compute a simple margin-based leakage penalty.
    pub fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        let topo_slots = &forward.slots.topology.slots;
        let geo_slots = &forward.slots.geometry.slots;
        let pocket_slots = &forward.slots.pocket.slots;

        let topo_geo_similarity = mean_cosine_similarity(topo_slots, geo_slots);
        let topo_pocket_similarity = mean_cosine_similarity(topo_slots, pocket_slots);
        let geo_pocket_similarity = mean_cosine_similarity(geo_slots, pocket_slots);

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
        let leakage_budget = self.margin + 0.05 * adjacency_density + 0.01 * pocket_energy;

        relu_scalar(topo_geo_similarity - leakage_budget)
            + relu_scalar(topo_pocket_similarity - leakage_budget)
            + relu_scalar(geo_pocket_similarity - leakage_budget)
    }
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
