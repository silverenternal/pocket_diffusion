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

/// Differentiable pocket-ligand geometry objectives for contact and clash behavior.
#[derive(Debug, Clone)]
pub struct PocketGeometryAuxLoss {
    /// Target nearest pocket-atom distance for encouraging contact.
    pub contact_distance: f64,
    /// Minimum allowed pocket-ligand atom distance before clash penalty activates.
    pub clash_distance: f64,
}

impl Default for PocketGeometryAuxLoss {
    fn default() -> Self {
        Self {
            contact_distance: 4.0,
            clash_distance: 1.25,
        }
    }
}

impl PocketGeometryAuxLoss {
    /// Compute contact encouragement and clash penalty as independently weighted scalars.
    pub(crate) fn compute(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> (Tensor, Tensor) {
        let ligand_coords = &example.decoder_supervision.noisy_coords
            + &forward.generation.decoded.coordinate_deltas;
        let ligand_mask = &forward.generation.state.partial_ligand.atom_mask;
        let pocket_coords = &example.pocket.coords;
        if ligand_coords.numel() == 0 || pocket_coords.numel() == 0 {
            let zero = Tensor::zeros([1], (Kind::Float, ligand_coords.device()));
            return (zero.shallow_clone(), zero);
        }

        let distances = pairwise_distances(&ligand_coords, pocket_coords);
        let pair_mask = ligand_mask.to_kind(Kind::Float).unsqueeze(1)
            * Tensor::ones(
                [1, pocket_coords.size().first().copied().unwrap_or(0)],
                (Kind::Float, pocket_coords.device()),
            );
        let valid_distances = &distances + (Tensor::ones_like(&pair_mask) - &pair_mask) * 1.0e6;
        let nearest = valid_distances.min_dim(1, false).0;
        let contact = weighted_mean(
            &(nearest - self.contact_distance)
                .relu()
                .pow_tensor_scalar(2.0),
            ligand_mask,
        );
        let clash_values = (self.clash_distance - distances)
            .relu()
            .pow_tensor_scalar(2.0);
        let clash = weighted_pair_mean(&clash_values, &pair_mask);
        (contact, clash)
    }

    /// Compute mean contact and clash objectives over a mini-batch.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(examples.len(), forwards.len());
        let device = forwards
            .first()
            .map(|forward| forward.generation.decoded.coordinate_deltas.device())
            .or_else(|| {
                examples
                    .first()
                    .map(|example| example.topology.atom_types.device())
            })
            .unwrap_or(tch::Device::Cpu);
        if examples.is_empty() {
            let zero = Tensor::zeros([1], (Kind::Float, device));
            return (zero.shallow_clone(), zero);
        }

        let mut contact_total = Tensor::zeros([1], (Kind::Float, device));
        let mut clash_total = Tensor::zeros([1], (Kind::Float, device));
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let (contact, clash) = self.compute(example, forward);
            contact_total += contact;
            clash_total += clash;
        }
        (
            contact_total / examples.len() as f64,
            clash_total / examples.len() as f64,
        )
    }
}

fn pairwise_distances(ligand_coords: &Tensor, pocket_coords: &Tensor) -> Tensor {
    let diffs = ligand_coords.unsqueeze(1) - pocket_coords.unsqueeze(0);
    diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .clamp_min(1e-12)
        .sqrt()
}

fn weighted_mean(values: &Tensor, mask: &Tensor) -> Tensor {
    let mask = mask.to_kind(Kind::Float);
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (values * mask).sum(Kind::Float) / denom
}

fn weighted_pair_mean(values: &Tensor, mask: &Tensor) -> Tensor {
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (values * mask).sum(Kind::Float) / denom
}

impl ConsistencyLoss {
    /// Compute a consistency penalty between topology logits and geometry-induced proximity.
    pub(crate) fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
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

    /// Compute the mean topology-geometry consistency penalty over a mini-batch.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> Tensor {
        debug_assert_eq!(examples.len(), forwards.len());
        let device = forwards
            .first()
            .map(|forward| forward.probes.topology_adjacency_logits.device())
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
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            total += self.compute(example, forward);
        }
        total / examples.len() as f64
    }
}
