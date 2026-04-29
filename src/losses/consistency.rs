//! Topology-geometry consistency objective.

use tch::{Kind, Tensor};

use crate::{data::MolecularExample, models::ResearchForward};

use super::alignment::{
    align_square_matrix, align_vector, pair_mask_from_row_mask, LossTargetAlignmentPolicy,
};

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
    /// Compute contact, clash, and envelope penalties as independently weighted scalars.
    pub(crate) fn compute(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> (Tensor, Tensor, Tensor) {
        let ligand_coords = active_decoded_coords(forward);
        let ligand_mask = &forward.generation.state.partial_ligand.atom_mask;
        let pocket_coords = &example.pocket.coords;
        if ligand_coords.numel() == 0 || pocket_coords.numel() == 0 {
            let zero = Tensor::zeros([1], (Kind::Float, ligand_coords.device()));
            return (zero.shallow_clone(), zero.shallow_clone(), zero);
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
        let envelope = pocket_envelope_penalty(&ligand_coords, ligand_mask, pocket_coords);
        (contact, clash, envelope)
    }

    /// Compute mean contact, clash, and envelope objectives over a mini-batch.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> (Tensor, Tensor, Tensor) {
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
            return (zero.shallow_clone(), zero.shallow_clone(), zero);
        }

        let mut contact_total = Tensor::zeros([1], (Kind::Float, device));
        let mut clash_total = Tensor::zeros([1], (Kind::Float, device));
        let mut envelope_total = Tensor::zeros([1], (Kind::Float, device));
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let (contact, clash, envelope) = self.compute(example, forward);
            contact_total += contact;
            clash_total += clash;
            envelope_total += envelope;
        }
        (
            contact_total / examples.len() as f64,
            clash_total / examples.len() as f64,
            envelope_total / examples.len() as f64,
        )
    }
}

/// Lightweight chemistry guardrails for valence and topology-implied bond lengths.
#[derive(Debug, Clone)]
pub struct ChemistryGuardrailAuxLoss {
    /// Bond-length tolerance before a quadratic penalty activates.
    pub bond_length_tolerance: f64,
}

impl Default for ChemistryGuardrailAuxLoss {
    fn default() -> Self {
        Self {
            bond_length_tolerance: 0.25,
        }
    }
}

impl ChemistryGuardrailAuxLoss {
    /// Compute valence-overage and bond-length guardrail objectives.
    pub(crate) fn compute(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> (Tensor, Tensor) {
        let predicted_coords = active_decoded_coords(forward);
        let valence = aligned_valence_overage_loss(
            &forward.generation.decoded.atom_type_logits,
            &example.topology.atom_types,
            &example.topology.adjacency,
        );
        let bond_length = aligned_bond_length_deviation_loss(
            &predicted_coords,
            &example.topology.atom_types,
            &example.topology.adjacency,
            self.bond_length_tolerance,
        );
        (valence, bond_length)
    }

    /// Compute mean guardrail objectives over a mini-batch.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(examples.len(), forwards.len());
        let device = forwards
            .first()
            .map(|forward| forward.generation.decoded.atom_type_logits.device())
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

        let mut valence_total = Tensor::zeros([1], (Kind::Float, device));
        let mut bond_total = Tensor::zeros([1], (Kind::Float, device));
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let (valence, bond_length) = self.compute(example, forward);
            valence_total += valence;
            bond_total += bond_length;
        }
        (
            valence_total / examples.len() as f64,
            bond_total / examples.len() as f64,
        )
    }
}

fn active_decoded_coords(forward: &ResearchForward) -> Tensor {
    let state_coords = &forward.generation.state.partial_ligand.coords;
    let deltas = &forward.generation.decoded.coordinate_deltas;
    if state_coords.size() == deltas.size() {
        state_coords + deltas
    } else {
        state_coords.shallow_clone()
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

#[cfg(test)]
fn valence_overage_loss(atom_type_logits: &Tensor, adjacency: &Tensor) -> Tensor {
    let atom_count = atom_type_logits
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(adjacency.size().first().copied().unwrap_or(0))
        .max(0);
    let mask = Tensor::ones([atom_count], (Kind::Float, atom_type_logits.device()));
    valence_overage_loss_with_mask(atom_type_logits, adjacency, &mask)
}

fn aligned_valence_overage_loss(
    atom_type_logits: &Tensor,
    atom_types: &Tensor,
    adjacency: &Tensor,
) -> Tensor {
    let device = atom_type_logits.device();
    if atom_type_logits.numel() == 0 || atom_types.numel() == 0 || adjacency.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let atom_count = atom_type_logits.size().first().copied().unwrap_or(0).max(0);
    let Some(aligned_atom_types) = align_vector(
        atom_types,
        atom_count,
        Kind::Int64,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "guardrail.valence.atom_types",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let Some(aligned_adjacency) = align_square_matrix(
        adjacency,
        atom_count,
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "guardrail.valence.adjacency",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    valence_overage_loss_with_mask(
        atom_type_logits,
        &aligned_adjacency.values,
        &aligned_atom_types.mask,
    )
}

fn valence_overage_loss_with_mask(
    atom_type_logits: &Tensor,
    adjacency: &Tensor,
    row_mask: &Tensor,
) -> Tensor {
    let device = atom_type_logits.device();
    if atom_type_logits.numel() == 0 || adjacency.numel() == 0 || row_mask.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let atom_count = atom_type_logits
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(adjacency.size().first().copied().unwrap_or(0))
        .min(row_mask.size().first().copied().unwrap_or(0))
        .max(0);
    let vocab = atom_type_logits.size().get(1).copied().unwrap_or(0).max(0);
    if atom_count == 0 || vocab == 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }

    let adjacency = adjacency
        .narrow(0, 0, atom_count)
        .narrow(1, 0, atom_count)
        .to_device(device)
        .to_kind(Kind::Float);
    let degree = adjacency.sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let cap = valence_cap_tensor(vocab, device);
    let expected_cap = atom_type_logits
        .narrow(0, 0, atom_count)
        .softmax(-1, Kind::Float)
        .matmul(&cap.unsqueeze(1))
        .squeeze_dim(1);
    let overage = (degree - expected_cap).relu().pow_tensor_scalar(2.0);
    weighted_mean(
        &overage,
        &row_mask
            .narrow(0, 0, atom_count)
            .to_device(device)
            .to_kind(Kind::Float),
    )
}

#[cfg(test)]
fn bond_length_deviation_loss(
    coords: &Tensor,
    atom_types: &Tensor,
    adjacency: &Tensor,
    tolerance: f64,
) -> Tensor {
    let atom_count = coords
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(atom_types.size().first().copied().unwrap_or(0))
        .min(adjacency.size().first().copied().unwrap_or(0))
        .max(0);
    let row_mask = Tensor::ones([atom_count], (Kind::Float, coords.device()));
    let pair_mask = pair_mask_from_row_mask(&row_mask);
    bond_length_deviation_loss_with_pair_mask(coords, atom_types, adjacency, &pair_mask, tolerance)
}

fn aligned_bond_length_deviation_loss(
    coords: &Tensor,
    atom_types: &Tensor,
    adjacency: &Tensor,
    tolerance: f64,
) -> Tensor {
    let device = coords.device();
    if coords.numel() == 0 || atom_types.numel() == 0 || adjacency.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let atom_count = coords.size().first().copied().unwrap_or(0).max(0);
    let Some(aligned_atom_types) = align_vector(
        atom_types,
        atom_count,
        Kind::Int64,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "guardrail.bond_length.atom_types",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let Some(aligned_adjacency) = align_square_matrix(
        adjacency,
        atom_count,
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "guardrail.bond_length.adjacency",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let pair_mask = pair_mask_from_row_mask(&aligned_atom_types.mask);
    bond_length_deviation_loss_with_pair_mask(
        coords,
        &aligned_atom_types.values,
        &aligned_adjacency.values,
        &pair_mask,
        tolerance,
    )
}

fn bond_length_deviation_loss_with_pair_mask(
    coords: &Tensor,
    atom_types: &Tensor,
    adjacency: &Tensor,
    pair_mask: &Tensor,
    tolerance: f64,
) -> Tensor {
    let device = coords.device();
    if coords.numel() == 0
        || atom_types.numel() == 0
        || adjacency.numel() == 0
        || pair_mask.numel() == 0
    {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let atom_count = coords
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(atom_types.size().first().copied().unwrap_or(0))
        .min(adjacency.size().first().copied().unwrap_or(0))
        .min(pair_mask.size().first().copied().unwrap_or(0))
        .max(0);
    let mut total = Tensor::zeros([1], (Kind::Float, device));
    let mut count = 0usize;
    for left in 0..atom_count {
        for right in (left + 1)..atom_count {
            if adjacency.double_value(&[left, right]) <= 0.5 {
                continue;
            }
            if pair_mask.double_value(&[left, right]) <= 0.5 {
                continue;
            }
            let delta = coords.get(left) - coords.get(right);
            let distance = delta
                .pow_tensor_scalar(2.0)
                .sum(Kind::Float)
                .clamp_min(1e-12)
                .sqrt();
            let ideal = ideal_bond_length(
                atom_types.int64_value(&[left]),
                atom_types.int64_value(&[right]),
            );
            let penalty = ((distance - ideal).abs() - tolerance)
                .relu()
                .pow_tensor_scalar(2.0);
            total += penalty;
            count += 1;
        }
    }
    if count == 0 {
        Tensor::zeros([1], (Kind::Float, device))
    } else {
        total / count as f64
    }
}

fn valence_cap_tensor(vocab: i64, device: tch::Device) -> Tensor {
    let caps = (0..vocab)
        .map(max_reasonable_valence)
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    Tensor::from_slice(&caps).to_device(device)
}

fn max_reasonable_valence(atom_type: i64) -> usize {
    match atom_type {
        0 => 4,
        1 => 3,
        2 => 2,
        3 => 6,
        4 => 1,
        _ => 4,
    }
}

fn ideal_bond_length(left: i64, right: i64) -> f64 {
    covalent_radius(left) + covalent_radius(right)
}

fn covalent_radius(atom_type: i64) -> f64 {
    match atom_type {
        0 => 0.77,
        1 => 0.75,
        2 => 0.73,
        3 => 1.02,
        4 => 0.37,
        _ => 0.77,
    }
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

fn pocket_envelope_penalty(
    ligand_coords: &Tensor,
    ligand_mask: &Tensor,
    pocket_coords: &Tensor,
) -> Tensor {
    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_offsets = pocket_coords - pocket_centroid.unsqueeze(0);
    let pocket_radius = pocket_offsets
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .max()
        .double_value(&[])
        .max(1.0);
    let ligand_offsets = ligand_coords - pocket_centroid.unsqueeze(0);
    let ligand_radii = ligand_offsets
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt();
    let allowed_radius = pocket_radius + 1.5;
    weighted_mean(
        &(ligand_radii - allowed_radius)
            .relu()
            .pow_tensor_scalar(2.0),
        ligand_mask,
    )
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
        let rows = forward
            .probes
            .topology_adjacency_logits
            .size()
            .first()
            .copied()
            .unwrap_or(0)
            .max(0);
        if forward.probes.topology_adjacency_logits.dim() != 2
            || forward
                .probes
                .topology_adjacency_logits
                .size()
                .get(1)
                .copied()
                .unwrap_or(-1)
                != rows
        {
            return Tensor::zeros(
                [1],
                (
                    Kind::Float,
                    forward.probes.topology_adjacency_logits.device(),
                ),
            );
        }
        let geom_target = example
            .geometry
            .pairwise_distances
            .lt(self.distance_cutoff)
            .to_kind(Kind::Float);
        let Some(aligned) = align_square_matrix(
            &geom_target,
            rows,
            Kind::Float,
            forward.probes.topology_adjacency_logits.device(),
            LossTargetAlignmentPolicy::PadWithMask,
            "consistency.geometry_pairwise_proximity",
        ) else {
            return Tensor::zeros(
                [1],
                (
                    Kind::Float,
                    forward.probes.topology_adjacency_logits.device(),
                ),
            );
        };
        let predicted = forward.probes.topology_adjacency_logits.sigmoid();
        weighted_pair_mean(
            &(predicted - aligned.values).pow_tensor_scalar(2.0),
            &aligned.mask,
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn valence_guardrail_penalizes_overvalent_hydrogen_but_not_carbon() {
        let adjacency = Tensor::from_slice(&[
            0.0_f32, 1.0, 0.0, //
            1.0, 0.0, 1.0, //
            0.0, 1.0, 0.0,
        ])
        .reshape([3, 3]);
        let hydrogen_logits = Tensor::full([3, 6], -8.0, (Kind::Float, Device::Cpu));
        let _ = hydrogen_logits.narrow(1, 4, 1).fill_(8.0);
        let carbon_logits = Tensor::full([3, 6], -8.0, (Kind::Float, Device::Cpu));
        let _ = carbon_logits.narrow(1, 0, 1).fill_(8.0);

        let invalid = valence_overage_loss(&hydrogen_logits, &adjacency);
        let valid = valence_overage_loss(&carbon_logits, &adjacency);

        assert!(invalid.double_value(&[]) > 0.1);
        assert!(valid.double_value(&[]) < 1e-6);
    }

    #[test]
    fn bond_length_guardrail_penalizes_stretched_topology_edges() {
        let atom_types = Tensor::from_slice(&[0_i64, 0]).to_kind(Kind::Int64);
        let adjacency = Tensor::from_slice(&[0.0_f32, 1.0, 1.0, 0.0]).reshape([2, 2]);
        let valid_coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.54, 0.0, 0.0]).reshape([2, 3]);
        let stretched_coords =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 4.0, 0.0, 0.0]).reshape([2, 3]);

        let valid = bond_length_deviation_loss(&valid_coords, &atom_types, &adjacency, 0.25);
        let stretched =
            bond_length_deviation_loss(&stretched_coords, &atom_types, &adjacency, 0.25);

        assert!(valid.double_value(&[]) < 1e-6);
        assert!(stretched.double_value(&[]) > valid.double_value(&[]));
    }

    #[test]
    fn chemistry_guardrails_are_finite_for_empty_and_single_atom_inputs() {
        let empty_logits = Tensor::zeros([0, 6], (Kind::Float, Device::Cpu));
        let empty_adj = Tensor::zeros([0, 0], (Kind::Float, Device::Cpu));
        let empty_coords = Tensor::zeros([0, 3], (Kind::Float, Device::Cpu));
        let empty_atom_types = Tensor::zeros([0], (Kind::Int64, Device::Cpu));
        let single_coords = Tensor::zeros([1, 3], (Kind::Float, Device::Cpu));
        let single_atom_types = Tensor::from_slice(&[0_i64]).to_kind(Kind::Int64);
        let single_adj = Tensor::zeros([1, 1], (Kind::Float, Device::Cpu));

        assert!(valence_overage_loss(&empty_logits, &empty_adj)
            .double_value(&[])
            .is_finite());
        assert!(
            bond_length_deviation_loss(&empty_coords, &empty_atom_types, &empty_adj, 0.25)
                .double_value(&[])
                .is_finite()
        );
        assert!(
            bond_length_deviation_loss(&single_coords, &single_atom_types, &single_adj, 0.25)
                .double_value(&[])
                .is_finite()
        );
    }
}
