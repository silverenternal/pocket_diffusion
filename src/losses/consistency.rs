//! Topology-geometry consistency objective.

use tch::{Kind, Tensor};

use crate::{data::MolecularExample, models::ResearchForward};

use super::alignment::{
    align_rows, align_square_matrix, align_vector, pair_mask_from_row_mask,
    LossTargetAlignmentPolicy,
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
    /// Atom-pocket target distance bin edges in Angstrom.
    pub distance_bin_edges: [f64; 3],
    /// Tolerance around the preferred atom-pocket shell for coarse shape complementarity.
    pub shape_shell_tolerance: f64,
}

impl Default for PocketGeometryAuxLoss {
    fn default() -> Self {
        Self {
            contact_distance: 4.0,
            clash_distance: 1.25,
            distance_bin_edges: [2.0, 4.0, 6.0],
            shape_shell_tolerance: 0.75,
        }
    }
}

/// Atom-pocket distance-bin targets with explicit atom and pocket masks.
#[derive(Debug)]
pub(crate) struct AtomPocketDistanceBinTargets {
    /// Integer bin index per atom-pocket pair.
    #[allow(dead_code)] // Exposed for distance-bin construction tests and future diagnostics.
    pub bin_indices: Tensor,
    /// Scalar bin center used by the differentiable pair-distance objective.
    pub bin_centers: Tensor,
    /// Pair mask with shape `[atoms, pocket_items]`.
    pub pair_mask: Tensor,
}

/// Decomposed pocket-ligand geometry objectives.
pub(crate) struct PocketGeometryAuxOutput {
    /// Nearest-pocket contact attraction term.
    pub contact: Tensor,
    /// Steric clash margin term.
    pub clash: Tensor,
    /// Pocket-envelope containment term.
    pub envelope: Tensor,
    /// Atom-pocket pair distance-bin regression term.
    pub pair_distance: Tensor,
    /// Coarse shape-complementarity shell term.
    pub shape_complementarity: Tensor,
}

impl PocketGeometryAuxOutput {
    fn zeros(device: tch::Device) -> Self {
        Self {
            contact: Tensor::zeros([1], (Kind::Float, device)),
            clash: Tensor::zeros([1], (Kind::Float, device)),
            envelope: Tensor::zeros([1], (Kind::Float, device)),
            pair_distance: Tensor::zeros([1], (Kind::Float, device)),
            shape_complementarity: Tensor::zeros([1], (Kind::Float, device)),
        }
    }
}

impl PocketGeometryAuxLoss {
    /// Compute contact, clash, and envelope penalties as independently weighted scalars.
    #[allow(dead_code)] // Compatibility path for older callers that do not need decomposed terms.
    pub(crate) fn compute(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> (Tensor, Tensor, Tensor) {
        let components = self.compute_components(example, forward);
        (components.contact, components.clash, components.envelope)
    }

    /// Compute all pocket-geometry components for teacher-forced and optional rollout states.
    pub(crate) fn compute_components(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> PocketGeometryAuxOutput {
        let ligand_coords = active_decoded_coords(forward);
        let ligand_mask = &forward.generation.state.partial_ligand.atom_mask;
        let pocket_coords = &example.pocket.coords;
        let target_coords = &example.decoder_supervision.target_coords;
        let mut output =
            self.compute_for_coords(target_coords, &ligand_coords, ligand_mask, pocket_coords);

        let mut pair_total = output.pair_distance.shallow_clone();
        let mut shape_total = output.shape_complementarity.shallow_clone();
        let mut state_count = 1usize;
        for step in &forward.generation.rollout_training.steps {
            let rollout = self.compute_for_coords(
                target_coords,
                &step.coords,
                &step.atom_mask,
                pocket_coords,
            );
            pair_total += rollout.pair_distance;
            shape_total += rollout.shape_complementarity;
            state_count += 1;
        }

        output.pair_distance = pair_total / state_count as f64;
        output.shape_complementarity = shape_total / state_count as f64;
        output
    }

    fn compute_for_coords(
        &self,
        target_coords: &Tensor,
        ligand_coords: &Tensor,
        ligand_mask: &Tensor,
        pocket_coords: &Tensor,
    ) -> PocketGeometryAuxOutput {
        let device = ligand_coords.device();
        if ligand_coords.numel() == 0 || pocket_coords.numel() == 0 {
            return PocketGeometryAuxOutput::zeros(device);
        }
        let atom_count = ligand_coords.size().first().copied().unwrap_or(0).max(0);
        let pocket_count = pocket_coords.size().first().copied().unwrap_or(0).max(0);
        if atom_count == 0 || pocket_count == 0 {
            return PocketGeometryAuxOutput::zeros(device);
        }

        let pocket_coords = pocket_coords.to_device(device);
        let distances = pairwise_distances(ligand_coords, &pocket_coords);
        let Some(aligned_ligand_mask) = align_vector(
            ligand_mask,
            atom_count,
            Kind::Float,
            device,
            LossTargetAlignmentPolicy::PadWithMask,
            "pocket_geometry.ligand_atom_mask",
        ) else {
            return PocketGeometryAuxOutput::zeros(device);
        };
        let ligand_mask =
            (aligned_ligand_mask.values * aligned_ligand_mask.mask).to_kind(Kind::Float);
        let pocket_mask = Tensor::ones([pocket_count], (Kind::Float, device));
        let pair_mask = ligand_mask.unsqueeze(1) * pocket_mask.unsqueeze(0);
        let valid_distances = &distances + (Tensor::ones_like(&pair_mask) - &pair_mask) * 1.0e6;
        let nearest = valid_distances.min_dim(1, false).0;
        let contact = weighted_mean(
            &(&nearest - self.contact_distance)
                .relu()
                .pow_tensor_scalar(2.0),
            &ligand_mask,
        );
        let clash_values = (self.clash_distance - &distances)
            .relu()
            .pow_tensor_scalar(2.0);
        let clash = weighted_pair_mean(&clash_values, &pair_mask);
        let envelope = pocket_envelope_penalty(ligand_coords, &ligand_mask, &pocket_coords);
        let pair_distance = atom_pocket_pair_distance_loss(
            target_coords,
            ligand_coords,
            &distances,
            &ligand_mask,
            &pocket_coords,
            &pocket_mask,
            &self.distance_bin_edges,
        );
        let shape_complementarity = shape_complementarity_loss(
            &nearest,
            &ligand_mask,
            self.preferred_shape_distance(),
            self.shape_shell_tolerance,
        );
        PocketGeometryAuxOutput {
            contact,
            clash,
            envelope,
            pair_distance,
            shape_complementarity,
        }
    }

    /// Compute mean contact, clash, and envelope objectives over a mini-batch.
    #[allow(dead_code)] // Compatibility path for older callers that do not need decomposed terms.
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

    /// Compute mean decomposed pocket-geometry objectives over a mini-batch.
    pub(crate) fn compute_batch_components(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> PocketGeometryAuxOutput {
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
            return PocketGeometryAuxOutput::zeros(device);
        }

        let mut total = PocketGeometryAuxOutput::zeros(device);
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let components = self.compute_components(example, forward);
            total.contact += components.contact;
            total.clash += components.clash;
            total.envelope += components.envelope;
            total.pair_distance += components.pair_distance;
            total.shape_complementarity += components.shape_complementarity;
        }
        let denom = examples.len() as f64;
        PocketGeometryAuxOutput {
            contact: total.contact / denom,
            clash: total.clash / denom,
            envelope: total.envelope / denom,
            pair_distance: total.pair_distance / denom,
            shape_complementarity: total.shape_complementarity / denom,
        }
    }

    fn preferred_shape_distance(&self) -> f64 {
        (self.contact_distance + self.clash_distance) * 0.5
    }
}

/// Lightweight chemistry guardrails for valence and topology-implied bond lengths.
#[derive(Debug, Clone)]
pub struct ChemistryGuardrailAuxLoss {
    /// Bond-length tolerance before a quadratic penalty activates.
    pub bond_length_tolerance: f64,
    /// Minimum non-bonded distance before a clash-style quadratic penalty activates.
    pub nonbonded_distance_margin: f64,
    /// Tolerance around broad local angle bounds before the proxy activates.
    pub angle_cosine_tolerance: f64,
}

impl Default for ChemistryGuardrailAuxLoss {
    fn default() -> Self {
        Self {
            bond_length_tolerance: 0.25,
            nonbonded_distance_margin: 1.0,
            angle_cosine_tolerance: 0.05,
        }
    }
}

/// Decomposed chemistry-native auxiliary guardrail objectives.
pub(crate) struct ChemistryGuardrailAuxOutput {
    /// Total valence-budget penalty used by the scheduled valence guardrail.
    pub valence_guardrail: Tensor,
    /// Expected valence above element-specific capacity.
    pub valence_overage_guardrail: Tensor,
    /// Expected valence below a conservative element-specific lower bound.
    pub valence_underage_guardrail: Tensor,
    /// Bond-length margin term for likely generated bonds.
    pub bond_length_guardrail: Tensor,
    /// Non-bonded short-distance margin term for likely absent bonds.
    pub nonbonded_distance_guardrail: Tensor,
    /// Broad local-angle plausibility proxy for generated two-hop neighborhoods.
    pub angle_guardrail: Tensor,
}

impl ChemistryGuardrailAuxOutput {
    fn zeros(device: tch::Device) -> Self {
        Self {
            valence_guardrail: Tensor::zeros([1], (Kind::Float, device)),
            valence_overage_guardrail: Tensor::zeros([1], (Kind::Float, device)),
            valence_underage_guardrail: Tensor::zeros([1], (Kind::Float, device)),
            bond_length_guardrail: Tensor::zeros([1], (Kind::Float, device)),
            nonbonded_distance_guardrail: Tensor::zeros([1], (Kind::Float, device)),
            angle_guardrail: Tensor::zeros([1], (Kind::Float, device)),
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.valence_guardrail += other.valence_guardrail;
        self.valence_overage_guardrail += other.valence_overage_guardrail;
        self.valence_underage_guardrail += other.valence_underage_guardrail;
        self.bond_length_guardrail += other.bond_length_guardrail;
        self.nonbonded_distance_guardrail += other.nonbonded_distance_guardrail;
        self.angle_guardrail += other.angle_guardrail;
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            valence_guardrail: self.valence_guardrail * factor,
            valence_overage_guardrail: self.valence_overage_guardrail * factor,
            valence_underage_guardrail: self.valence_underage_guardrail * factor,
            bond_length_guardrail: self.bond_length_guardrail * factor,
            nonbonded_distance_guardrail: self.nonbonded_distance_guardrail * factor,
            angle_guardrail: self.angle_guardrail * factor,
        }
    }
}

impl ChemistryGuardrailAuxLoss {
    /// Compute valence-overage and bond-length guardrail objectives.
    pub(crate) fn compute(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> ChemistryGuardrailAuxOutput {
        if let Some(flow) = forward.generation.flow_matching.as_ref() {
            if let Some(molecular) = flow.molecular.as_ref() {
                let atom_count = molecular
                    .atom_type_logits
                    .size()
                    .first()
                    .copied()
                    .unwrap_or(0)
                    .max(0);
                let atom_mask =
                    active_generation_atom_mask(forward, Some(&flow.atom_mask), atom_count);
                let coords = chemistry_guardrail_coords(forward, flow);
                return native_chemistry_guardrail_loss(
                    &molecular.atom_type_logits,
                    &molecular.bond_exists_logits,
                    &molecular.bond_type_logits,
                    &coords,
                    &atom_mask,
                    self.bond_length_tolerance,
                    self.nonbonded_distance_margin,
                    self.angle_cosine_tolerance,
                );
            }
        }
        let predicted_coords = active_decoded_coords(forward);
        let overage = aligned_valence_overage_loss(
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
        let device = overage.device();
        ChemistryGuardrailAuxOutput {
            valence_guardrail: overage.shallow_clone(),
            valence_overage_guardrail: overage,
            valence_underage_guardrail: Tensor::zeros([1], (Kind::Float, device)),
            bond_length_guardrail: bond_length,
            nonbonded_distance_guardrail: Tensor::zeros([1], (Kind::Float, device)),
            angle_guardrail: Tensor::zeros([1], (Kind::Float, device)),
        }
    }

    /// Compute mean guardrail objectives over a mini-batch.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> ChemistryGuardrailAuxOutput {
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
            return ChemistryGuardrailAuxOutput::zeros(device);
        }

        let mut total = ChemistryGuardrailAuxOutput::zeros(device);
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            total.add_assign(self.compute(example, forward));
        }
        total.scale(1.0 / examples.len() as f64)
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

fn active_generation_atom_mask(
    forward: &ResearchForward,
    fallback_mask: Option<&Tensor>,
    atom_count: i64,
) -> Tensor {
    let state_mask = &forward.generation.state.partial_ligand.atom_mask;
    let device = state_mask.device();
    let source = if state_mask.numel() > 0 {
        state_mask
    } else if let Some(mask) = fallback_mask {
        mask
    } else {
        return Tensor::ones([atom_count.max(0)], (Kind::Float, device));
    };
    let device = source.device();
    let Some(aligned) = align_vector(
        source,
        atom_count.max(0),
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "guardrail.native_chemistry.atom_mask",
    ) else {
        return Tensor::zeros([atom_count.max(0)], (Kind::Float, device));
    };
    aligned.values.to_kind(Kind::Float) * aligned.mask.to_kind(Kind::Float)
}

fn chemistry_guardrail_coords(
    forward: &ResearchForward,
    flow: &crate::models::system::FlowMatchingTrainingRecord,
) -> Tensor {
    if flow.sampled_coords.size() == flow.predicted_velocity.size() {
        &flow.sampled_coords + &flow.predicted_velocity * (1.0 - flow.t).clamp(0.0, 1.0)
    } else {
        active_decoded_coords(forward)
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

pub(crate) fn atom_pocket_distance_bin_targets(
    distances: &Tensor,
    atom_mask: &Tensor,
    pocket_mask: &Tensor,
    bin_edges: &[f64],
) -> AtomPocketDistanceBinTargets {
    let device = distances.device();
    let shape = distances.size();
    let atom_count = shape.first().copied().unwrap_or(0).max(0);
    let pocket_count = shape.get(1).copied().unwrap_or(0).max(0);
    if distances.dim() != 2
        || atom_count == 0
        || pocket_count == 0
        || atom_mask.size().first().copied().unwrap_or(-1) != atom_count
        || pocket_mask.size().first().copied().unwrap_or(-1) != pocket_count
    {
        return AtomPocketDistanceBinTargets {
            bin_indices: Tensor::zeros([atom_count, pocket_count], (Kind::Int64, device)),
            bin_centers: Tensor::zeros([atom_count, pocket_count], (Kind::Float, device)),
            pair_mask: Tensor::zeros([atom_count, pocket_count], (Kind::Float, device)),
        };
    }

    let mut bin_indices = Tensor::zeros([atom_count, pocket_count], (Kind::Int64, device));
    for edge in bin_edges {
        bin_indices += distances.ge(*edge).to_kind(Kind::Int64);
    }
    let centers = distance_bin_centers(bin_edges);
    let center_tensor = Tensor::from_slice(&centers).to_device(device);
    let bin_centers = center_tensor
        .gather(0, &bin_indices.flatten(0, -1), false)
        .reshape([atom_count, pocket_count]);
    let pair_mask = atom_mask
        .to_device(device)
        .to_kind(Kind::Float)
        .unsqueeze(1)
        * pocket_mask
            .to_device(device)
            .to_kind(Kind::Float)
            .unsqueeze(0);
    AtomPocketDistanceBinTargets {
        bin_indices,
        bin_centers,
        pair_mask,
    }
}

fn distance_bin_centers(bin_edges: &[f64]) -> Vec<f32> {
    if bin_edges.is_empty() {
        return vec![1.0];
    }
    let mut centers = Vec::with_capacity(bin_edges.len() + 1);
    let mut previous = 0.0;
    for edge in bin_edges {
        centers.push(((previous + *edge) * 0.5) as f32);
        previous = *edge;
    }
    let tail_width = if bin_edges.len() >= 2 {
        (bin_edges[bin_edges.len() - 1] - bin_edges[bin_edges.len() - 2]).max(1.0)
    } else {
        bin_edges[0].max(1.0)
    };
    centers.push((previous + tail_width * 0.5) as f32);
    centers
}

fn atom_pocket_pair_distance_loss(
    target_coords: &Tensor,
    ligand_coords: &Tensor,
    predicted_distances: &Tensor,
    ligand_mask: &Tensor,
    pocket_coords: &Tensor,
    pocket_mask: &Tensor,
    bin_edges: &[f64],
) -> Tensor {
    let device = ligand_coords.device();
    let atom_count = ligand_coords.size().first().copied().unwrap_or(0).max(0);
    let Some(aligned_target_coords) = align_rows(
        target_coords,
        atom_count,
        &[3],
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "pocket_geometry.atom_pocket_target_coords",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let target_distances = pairwise_distances(&aligned_target_coords.values, pocket_coords);
    let atom_mask = ligand_mask * aligned_target_coords.mask;
    let targets =
        atom_pocket_distance_bin_targets(&target_distances, &atom_mask, pocket_mask, bin_edges);
    weighted_pair_mean(
        &(predicted_distances - targets.bin_centers).pow_tensor_scalar(2.0),
        &targets.pair_mask,
    )
}

fn shape_complementarity_loss(
    nearest_distances: &Tensor,
    ligand_mask: &Tensor,
    preferred_distance: f64,
    shell_tolerance: f64,
) -> Tensor {
    let shell = ((nearest_distances - preferred_distance).abs() - shell_tolerance.max(0.0))
        .relu()
        .pow_tensor_scalar(2.0);
    weighted_mean(&shell, ligand_mask)
}

fn native_chemistry_guardrail_loss(
    atom_type_logits: &Tensor,
    bond_exists_logits: &Tensor,
    bond_type_logits: &Tensor,
    coords: &Tensor,
    atom_mask: &Tensor,
    bond_length_tolerance: f64,
    nonbonded_distance_margin: f64,
    angle_cosine_tolerance: f64,
) -> ChemistryGuardrailAuxOutput {
    let device = atom_type_logits.device();
    if atom_type_logits.numel() == 0 || atom_type_logits.dim() != 2 {
        return ChemistryGuardrailAuxOutput::zeros(device);
    }
    let atom_count = atom_type_logits
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(bond_exists_logits.size().first().copied().unwrap_or(0))
        .min(coords.size().first().copied().unwrap_or(0))
        .min(atom_mask.size().first().copied().unwrap_or(0))
        .max(0);
    let atom_vocab = atom_type_logits.size().get(1).copied().unwrap_or(0).max(0);
    if atom_count == 0 || atom_vocab == 0 || bond_exists_logits.dim() != 2 {
        return ChemistryGuardrailAuxOutput::zeros(device);
    }

    let atom_logits = atom_type_logits.narrow(0, 0, atom_count);
    let coords = coords
        .narrow(0, 0, atom_count)
        .to_device(device)
        .to_kind(Kind::Float);
    let atom_mask = atom_mask
        .narrow(0, 0, atom_count)
        .to_device(device)
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let pair_mask = upper_triangular_pair_mask(&atom_mask);
    let bond_probabilities = bond_exists_logits
        .narrow(0, 0, atom_count)
        .narrow(1, 0, atom_count)
        .to_device(device)
        .sigmoid();
    let expected_bond_orders = expected_bond_order_matrix(&bond_probabilities, bond_type_logits);
    let valence_pair_mask = pair_mask_from_row_mask(&atom_mask);
    let expected_valence = (&expected_bond_orders * &valence_pair_mask).sum_dim_intlist(
        [1].as_slice(),
        false,
        Kind::Float,
    );
    let atom_probs = atom_logits.softmax(-1, Kind::Float);
    let max_cap = valence_cap_tensor(atom_vocab, device);
    let min_cap = valence_min_tensor(atom_vocab, device);
    let expected_max = atom_probs.matmul(&max_cap.unsqueeze(1)).squeeze_dim(1);
    let expected_min = atom_probs.matmul(&min_cap.unsqueeze(1)).squeeze_dim(1);
    let overage = (&expected_valence - &expected_max)
        .relu()
        .pow_tensor_scalar(2.0);
    let underage = (&expected_min - &expected_valence)
        .relu()
        .pow_tensor_scalar(2.0);
    let valence_overage_guardrail = weighted_mean(&overage, &atom_mask);
    let valence_underage_guardrail = weighted_mean(&underage, &atom_mask);
    let valence_guardrail =
        valence_overage_guardrail.shallow_clone() + valence_underage_guardrail.shallow_clone();

    let distances = ligand_pairwise_distances(&coords);
    let expected_radii =
        atom_probs.matmul(&covalent_radius_tensor(atom_vocab, device).unsqueeze(1));
    let ideal_lengths = &expected_radii + expected_radii.transpose(0, 1);
    let upper_bond_probabilities = &bond_probabilities * &pair_mask;
    let bond_length_values = ((&distances - &ideal_lengths).abs() - bond_length_tolerance.max(0.0))
        .relu()
        .pow_tensor_scalar(2.0);
    let bond_length_guardrail = weighted_pair_mean(&bond_length_values, &upper_bond_probabilities);
    let nonbonded_weights =
        (Tensor::ones_like(&bond_probabilities) - &bond_probabilities) * &pair_mask;
    let nonbonded_values = (nonbonded_distance_margin.max(0.0) - &distances)
        .relu()
        .pow_tensor_scalar(2.0);
    let nonbonded_distance_guardrail = weighted_pair_mean(&nonbonded_values, &nonbonded_weights);
    let angle_guardrail = local_angle_proxy_loss(
        &coords,
        &bond_probabilities,
        &atom_mask,
        angle_cosine_tolerance,
    );

    ChemistryGuardrailAuxOutput {
        valence_guardrail,
        valence_overage_guardrail,
        valence_underage_guardrail,
        bond_length_guardrail,
        nonbonded_distance_guardrail,
        angle_guardrail,
    }
}

fn expected_bond_order_matrix(bond_probabilities: &Tensor, bond_type_logits: &Tensor) -> Tensor {
    let atom_count = bond_probabilities
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0);
    let device = bond_probabilities.device();
    if bond_type_logits.dim() != 3
        || bond_type_logits.size().first().copied().unwrap_or(0) < atom_count
        || bond_type_logits.size().get(1).copied().unwrap_or(0) < atom_count
    {
        return bond_probabilities.shallow_clone();
    }
    let vocab = bond_type_logits.size().get(2).copied().unwrap_or(0).max(0);
    if vocab == 0 {
        return bond_probabilities.shallow_clone();
    }
    let orders = bond_order_tensor(vocab, device);
    let expected_order = bond_type_logits
        .narrow(0, 0, atom_count)
        .narrow(1, 0, atom_count)
        .to_device(device)
        .softmax(-1, Kind::Float)
        .matmul(&orders.unsqueeze(1))
        .squeeze_dim(-1);
    bond_probabilities * expected_order
}

fn ligand_pairwise_distances(coords: &Tensor) -> Tensor {
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .clamp_min(1e-12)
        .sqrt()
}

fn upper_triangular_pair_mask(atom_mask: &Tensor) -> Tensor {
    let atom_count = atom_mask.size().first().copied().unwrap_or(0).max(0);
    if atom_count == 0 {
        return Tensor::zeros([0, 0], (Kind::Float, atom_mask.device()));
    }
    let pair_mask = atom_mask.unsqueeze(0) * atom_mask.unsqueeze(1);
    let upper = Tensor::ones([atom_count, atom_count], (Kind::Float, atom_mask.device())).triu(1);
    pair_mask * upper
}

fn local_angle_proxy_loss(
    coords: &Tensor,
    bond_probabilities: &Tensor,
    atom_mask: &Tensor,
    tolerance: f64,
) -> Tensor {
    let device = coords.device();
    let atom_count = coords.size().first().copied().unwrap_or(0).max(0);
    if atom_count < 3 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let mut total = Tensor::zeros([1], (Kind::Float, device));
    let mut weight_total = Tensor::zeros([1], (Kind::Float, device));
    let min_cosine = -0.85 - tolerance.max(0.0);
    let max_cosine = 0.75 + tolerance.max(0.0);
    for center in 0..atom_count {
        for left in 0..atom_count {
            if left == center {
                continue;
            }
            for right in (left + 1)..atom_count {
                if right == center {
                    continue;
                }
                let weight = bond_probabilities.get(center).get(left)
                    * bond_probabilities.get(center).get(right)
                    * atom_mask.get(center)
                    * atom_mask.get(left)
                    * atom_mask.get(right);
                let left_vec = coords.get(left) - coords.get(center);
                let right_vec = coords.get(right) - coords.get(center);
                let cosine = (&left_vec * &right_vec).sum(Kind::Float)
                    / (left_vec
                        .pow_tensor_scalar(2.0)
                        .sum(Kind::Float)
                        .clamp_min(1e-12)
                        .sqrt()
                        * right_vec
                            .pow_tensor_scalar(2.0)
                            .sum(Kind::Float)
                            .clamp_min(1e-12)
                            .sqrt()
                        + 1e-12);
                let low_penalty = (Tensor::from(min_cosine as f32).to_device(device) - &cosine)
                    .relu()
                    .pow_tensor_scalar(2.0);
                let high_penalty = (&cosine - Tensor::from(max_cosine as f32).to_device(device))
                    .relu()
                    .pow_tensor_scalar(2.0);
                total += &weight * (low_penalty + high_penalty);
                weight_total += weight;
            }
        }
    }
    total / weight_total.clamp_min(1.0)
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

fn valence_min_tensor(vocab: i64, device: tch::Device) -> Tensor {
    let mins = (0..vocab)
        .map(min_reasonable_valence)
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    Tensor::from_slice(&mins).to_device(device)
}

fn bond_order_tensor(vocab: i64, device: tch::Device) -> Tensor {
    let orders = (0..vocab)
        .map(|bond_type| match bond_type {
            3 => 3.0_f32,
            2 => 2.0_f32,
            _ => 1.0_f32,
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&orders).to_device(device)
}

fn covalent_radius_tensor(vocab: i64, device: tch::Device) -> Tensor {
    let radii = (0..vocab)
        .map(covalent_radius)
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    Tensor::from_slice(&radii).to_device(device)
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

fn min_reasonable_valence(atom_type: i64) -> usize {
    match atom_type {
        4 => 1,
        0 | 1 | 2 | 3 => 1,
        _ => 1,
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

    fn logits_for_type(atom_count: i64, vocab: i64, atom_type: i64) -> Tensor {
        let logits = Tensor::full([atom_count, vocab], -8.0, (Kind::Float, Device::Cpu));
        let _ = logits
            .narrow(1, atom_type.clamp(0, vocab - 1), 1)
            .fill_(8.0);
        logits
    }

    fn dense_bond_logits(atom_count: i64, value: f64) -> Tensor {
        Tensor::full([atom_count, atom_count], value, (Kind::Float, Device::Cpu))
    }

    fn preferred_bond_type_logits(atom_count: i64, vocab: i64, bond_type: i64) -> Tensor {
        let logits = Tensor::full(
            [atom_count, atom_count, vocab],
            -8.0,
            (Kind::Float, Device::Cpu),
        );
        let _ = logits
            .narrow(2, bond_type.clamp(0, vocab - 1), 1)
            .fill_(8.0);
        logits
    }

    #[test]
    fn atom_pocket_distance_bins_respect_atom_and_pocket_masks() {
        let distances = Tensor::from_slice(&[
            0.8_f32, 2.5, 5.5, //
            1.5, 4.5, 7.5,
        ])
        .reshape([2, 3]);
        let atom_mask = Tensor::from_slice(&[1.0_f32, 0.0]);
        let pocket_mask = Tensor::from_slice(&[1.0_f32, 1.0, 0.0]);

        let targets = atom_pocket_distance_bin_targets(
            &distances,
            &atom_mask,
            &pocket_mask,
            &[2.0, 4.0, 6.0],
        );

        assert_eq!(targets.bin_indices.int64_value(&[0, 0]), 0);
        assert_eq!(targets.bin_indices.int64_value(&[0, 1]), 1);
        assert_eq!(targets.bin_indices.int64_value(&[0, 2]), 2);
        assert_eq!(targets.bin_indices.int64_value(&[1, 2]), 3);
        assert_eq!(targets.pair_mask.double_value(&[0, 0]), 1.0);
        assert_eq!(targets.pair_mask.double_value(&[0, 1]), 1.0);
        assert_eq!(targets.pair_mask.double_value(&[0, 2]), 0.0);
        assert_eq!(targets.pair_mask.double_value(&[1, 0]), 0.0);
        assert_eq!(targets.bin_centers.double_value(&[0, 1]), 3.0);
    }

    #[test]
    fn pocket_geometry_pair_distance_and_shape_losses_are_finite() {
        let loss = PocketGeometryAuxLoss::default();
        let target_coords = Tensor::from_slice(&[3.0_f32, 0.0, 0.0, 4.0, 0.0, 0.0]).reshape([2, 3]);
        let ligand_coords = Tensor::from_slice(&[3.2_f32, 0.0, 0.0, 4.2, 0.0, 0.0]).reshape([2, 3]);
        let ligand_mask = Tensor::ones([2], (Kind::Float, Device::Cpu));
        let pocket_coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0]).reshape([1, 3]);

        let output =
            loss.compute_for_coords(&target_coords, &ligand_coords, &ligand_mask, &pocket_coords);

        assert!(output.pair_distance.double_value(&[]).is_finite());
        assert!(output.shape_complementarity.double_value(&[]).is_finite());
        assert!(output.contact.double_value(&[]).is_finite());
        assert!(output.clash.double_value(&[]).is_finite());
    }

    #[test]
    fn pocket_geometry_separates_clash_shape_and_missing_pocket_noop() {
        let loss = PocketGeometryAuxLoss::default();
        let target_coords = Tensor::from_slice(&[3.0_f32, 0.0, 0.0, 3.3, 0.0, 0.0]).reshape([2, 3]);
        let ligand_mask = Tensor::ones([2], (Kind::Float, Device::Cpu));
        let pocket_coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0]).reshape([1, 3]);
        let clashing = Tensor::from_slice(&[0.2_f32, 0.0, 0.0, 0.4, 0.0, 0.0]).reshape([2, 3]);
        let separated = Tensor::from_slice(&[8.0_f32, 0.0, 0.0, 8.5, 0.0, 0.0]).reshape([2, 3]);
        let complementary = Tensor::from_slice(&[2.6_f32, 0.0, 0.0, 3.0, 0.0, 0.0]).reshape([2, 3]);
        let missing_pocket = Tensor::zeros([0, 3], (Kind::Float, Device::Cpu));

        let clashing_output =
            loss.compute_for_coords(&target_coords, &clashing, &ligand_mask, &pocket_coords);
        let separated_output =
            loss.compute_for_coords(&target_coords, &separated, &ligand_mask, &pocket_coords);
        let complementary_output =
            loss.compute_for_coords(&target_coords, &complementary, &ligand_mask, &pocket_coords);
        let noop_output =
            loss.compute_for_coords(&target_coords, &clashing, &ligand_mask, &missing_pocket);

        assert!(clashing_output.clash.double_value(&[]) > separated_output.clash.double_value(&[]));
        assert!(
            separated_output.shape_complementarity.double_value(&[])
                > complementary_output.shape_complementarity.double_value(&[])
        );
        assert_eq!(noop_output.contact.double_value(&[]), 0.0);
        assert_eq!(noop_output.clash.double_value(&[]), 0.0);
        assert_eq!(noop_output.shape_complementarity.double_value(&[]), 0.0);
    }

    #[test]
    fn pocket_geometry_pair_distance_runs_on_rollout_training_states() {
        let mut config = crate::config::ResearchConfig::default();
        config.training.rollout_training.enabled = true;
        config.training.rollout_training.rollout_steps = 2;
        let dataset = crate::data::InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let example = &dataset.examples()[0];
        let var_store = tch::nn::VarStore::new(Device::Cpu);
        let system = crate::models::Phase1ResearchSystem::new(&var_store.root(), &config);
        let forward = system.forward_example(example);

        assert_eq!(forward.generation.rollout_training.executed_steps, 2);
        let output = PocketGeometryAuxLoss::default().compute_components(example, &forward);

        assert!(output.pair_distance.double_value(&[]).is_finite());
        assert!(output.shape_complementarity.double_value(&[]).is_finite());
    }

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
    fn native_valence_budget_logs_over_under_and_respects_masks() {
        let coords = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ])
        .reshape([3, 3]);
        let hydrogen_logits = logits_for_type(3, 6, 4).set_requires_grad(true);
        let carbon_logits = logits_for_type(3, 6, 0);
        let dense_bonds = dense_bond_logits(3, 8.0).set_requires_grad(true);
        let no_bonds = dense_bond_logits(3, -8.0);
        let single_bonds = preferred_bond_type_logits(3, 4, 1).set_requires_grad(true);
        let all_mask = Tensor::ones([3], (Kind::Float, Device::Cpu));
        let center_only_mask = Tensor::from_slice(&[0.0_f32, 1.0, 0.0]);

        let over = native_chemistry_guardrail_loss(
            &hydrogen_logits,
            &dense_bonds,
            &single_bonds,
            &coords,
            &all_mask,
            0.25,
            1.0,
            0.05,
        );
        let under = native_chemistry_guardrail_loss(
            &carbon_logits,
            &no_bonds,
            &single_bonds,
            &coords,
            &all_mask,
            0.25,
            1.0,
            0.05,
        );
        let masked = native_chemistry_guardrail_loss(
            &hydrogen_logits,
            &dense_bonds,
            &single_bonds,
            &coords,
            &center_only_mask,
            0.25,
            1.0,
            0.05,
        );

        assert!(over.valence_overage_guardrail.double_value(&[]) > 0.1);
        assert!(under.valence_underage_guardrail.double_value(&[]) > 0.1);
        assert!(masked.valence_overage_guardrail.double_value(&[]) < 1e-5);
        assert!(over.valence_guardrail.requires_grad());
        assert!(over.bond_length_guardrail.requires_grad());
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
    fn native_geometry_guardrails_separate_bond_length_nonbonded_and_angle() {
        let atom_logits = logits_for_type(3, 6, 0);
        let active_mask = Tensor::ones([3], (Kind::Float, Device::Cpu));
        let plausible = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, //
            1.54, 0.0, 0.0, //
            0.0, 1.54, 0.0,
        ])
        .reshape([3, 3])
        .set_requires_grad(true);
        let short = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, //
            0.25, 0.0, 0.0, //
            0.0, 0.25, 0.0,
        ])
        .reshape([3, 3])
        .set_requires_grad(true);
        let bond_logits = Tensor::from_slice(&[
            -8.0_f32, 8.0, 8.0, //
            8.0, -8.0, -8.0, //
            8.0, -8.0, -8.0,
        ])
        .reshape([3, 3]);
        let nonbonded_clash_logits = Tensor::from_slice(&[
            -8.0_f32, -8.0, -8.0, //
            -8.0, -8.0, -8.0, //
            -8.0, -8.0, -8.0,
        ])
        .reshape([3, 3]);
        let bond_types = preferred_bond_type_logits(3, 4, 1);
        let plausible_output = native_chemistry_guardrail_loss(
            &atom_logits,
            &bond_logits,
            &bond_types,
            &plausible,
            &active_mask,
            0.25,
            1.0,
            0.05,
        );
        let short_bond_output = native_chemistry_guardrail_loss(
            &atom_logits,
            &bond_logits,
            &bond_types,
            &short,
            &active_mask,
            0.25,
            1.0,
            0.05,
        );
        let nonbonded_output = native_chemistry_guardrail_loss(
            &atom_logits,
            &nonbonded_clash_logits,
            &bond_types,
            &short,
            &active_mask,
            0.25,
            1.0,
            0.05,
        );

        assert!(
            short_bond_output.bond_length_guardrail.double_value(&[])
                > plausible_output.bond_length_guardrail.double_value(&[])
        );
        assert!(
            nonbonded_output
                .nonbonded_distance_guardrail
                .double_value(&[])
                > 0.1
        );
        assert!(plausible_output
            .angle_guardrail
            .double_value(&[])
            .is_finite());
        assert!(short_bond_output.bond_length_guardrail.requires_grad());
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
