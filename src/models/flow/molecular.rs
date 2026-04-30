//! Full molecular flow heads for de novo pocket-conditioned generation.
//!
//! The geometry branch remains continuous velocity matching. Atom type, bond,
//! topology, and pocket/context branches are discrete or representation-flow
//! heads trained from the same controlled conditioning state.

use std::collections::BTreeMap;

use tch::{nn, Kind, Tensor};

use crate::config::DecoderConditioningKind;
use crate::models::{ConditioningState, ModelError};

const PAIR_LOGIT_ROW_CHUNK_SIZE: i64 = 64;
const BOND_DISTANCE_PRIOR_CENTER_ANGSTROM: f64 = 1.50;
const BOND_DISTANCE_PRIOR_WIDTH_ANGSTROM: f64 = 0.32;
const BOND_DISTANCE_PRIOR_PEAK_LOGIT: f64 = 0.70;
const BOND_DISTANCE_PRIOR_FLOOR_LOGIT: f64 = -6.0;

/// Inputs shared by non-coordinate molecular flow branches.
pub struct MolecularFlowInput<'a> {
    /// Current atom-type tokens for the molecular draft.
    pub atom_types: &'a Tensor,
    /// Coordinates at flow time `t`.
    pub coords: &'a Tensor,
    /// Initial coordinates used by the coordinate flow branch.
    pub x0_coords: &'a Tensor,
    /// Normalized flow time.
    pub t: f64,
    /// Decomposed and gated conditioning state.
    pub conditioning: &'a ConditioningState,
}

/// Optimizer-facing predictions for all non-coordinate molecular flow branches.
#[derive(Debug)]
pub struct MolecularFlowPrediction {
    /// Atom-type categorical logits `[num_atoms, atom_vocab_size]`.
    pub atom_type_logits: Tensor,
    /// Bond existence logits `[num_atoms, num_atoms]`.
    pub bond_exists_logits: Tensor,
    /// Bond type logits `[num_atoms, num_atoms, bond_vocab_size]`.
    pub bond_type_logits: Tensor,
    /// Topology consistency logits `[num_atoms, num_atoms]`.
    pub topology_logits: Tensor,
    /// Pocket interaction/contact logits `[num_atoms]`.
    pub pocket_contact_logits: Tensor,
    /// Pocket/context representation reconstruction `[num_pocket_slots, hidden_dim]`.
    pub pocket_context_reconstruction: Tensor,
    /// Lightweight branch diagnostics.
    pub diagnostics: BTreeMap<String, f64>,
}

impl Clone for MolecularFlowPrediction {
    fn clone(&self) -> Self {
        Self {
            atom_type_logits: self.atom_type_logits.shallow_clone(),
            bond_exists_logits: self.bond_exists_logits.shallow_clone(),
            bond_type_logits: self.bond_type_logits.shallow_clone(),
            topology_logits: self.topology_logits.shallow_clone(),
            pocket_contact_logits: self.pocket_contact_logits.shallow_clone(),
            pocket_context_reconstruction: self.pocket_context_reconstruction.shallow_clone(),
            diagnostics: self.diagnostics.clone(),
        }
    }
}

/// Compact full molecular flow head with atom, bond, topology, and context branches.
#[derive(Debug)]
pub struct FullMolecularFlowHead {
    atom_embedding: nn::Embedding,
    context_projection: nn::Linear,
    local_conditioning_query_projection: nn::Linear,
    timestep_projection: nn::Linear,
    atom_projection: nn::Linear,
    atom_type_head: nn::Linear,
    pair_projection: nn::Linear,
    bond_exists_head: nn::Linear,
    bond_type_head: nn::Linear,
    topology_head: nn::Linear,
    pocket_contact_head: nn::Linear,
    pocket_context_projection: nn::Linear,
    hidden_dim: i64,
    atom_vocab_size: i64,
    bond_vocab_size: i64,
    conditioning_kind: DecoderConditioningKind,
}

impl FullMolecularFlowHead {
    /// Build all non-coordinate molecular flow branches.
    pub fn new(vs: &nn::Path, atom_vocab_size: i64, bond_vocab_size: i64, hidden_dim: i64) -> Self {
        Self::new_with_conditioning_kind(
            vs,
            atom_vocab_size,
            bond_vocab_size,
            hidden_dim,
            DecoderConditioningKind::default(),
        )
    }

    /// Build all non-coordinate molecular flow branches with an explicit conditioning mode.
    pub fn new_with_conditioning_kind(
        vs: &nn::Path,
        atom_vocab_size: i64,
        bond_vocab_size: i64,
        hidden_dim: i64,
        conditioning_kind: DecoderConditioningKind,
    ) -> Self {
        Self {
            atom_embedding: nn::embedding(
                vs / "atom_embedding",
                atom_vocab_size,
                hidden_dim,
                Default::default(),
            ),
            context_projection: nn::linear(
                vs / "context_projection",
                hidden_dim * 3,
                hidden_dim,
                Default::default(),
            ),
            local_conditioning_query_projection: nn::linear(
                vs / "local_conditioning_query_projection",
                hidden_dim + 3,
                hidden_dim,
                Default::default(),
            ),
            timestep_projection: nn::linear(
                vs / "timestep_projection",
                4,
                hidden_dim,
                Default::default(),
            ),
            atom_projection: nn::linear(
                vs / "atom_projection",
                hidden_dim * 3 + 9,
                hidden_dim,
                Default::default(),
            ),
            atom_type_head: nn::linear(
                vs / "atom_type_head",
                hidden_dim,
                atom_vocab_size,
                Default::default(),
            ),
            pair_projection: nn::linear(
                vs / "pair_projection",
                hidden_dim * 2 + 4,
                hidden_dim,
                Default::default(),
            ),
            bond_exists_head: nn::linear(
                vs / "bond_exists_head",
                hidden_dim,
                1,
                Default::default(),
            ),
            bond_type_head: nn::linear(
                vs / "bond_type_head",
                hidden_dim,
                bond_vocab_size,
                Default::default(),
            ),
            topology_head: nn::linear(vs / "topology_head", hidden_dim, 1, Default::default()),
            pocket_contact_head: nn::linear(
                vs / "pocket_contact_head",
                hidden_dim,
                1,
                Default::default(),
            ),
            pocket_context_projection: nn::linear(
                vs / "pocket_context_projection",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ),
            hidden_dim,
            atom_vocab_size,
            bond_vocab_size,
            conditioning_kind,
        }
    }

    /// Predict all non-coordinate flow branches for one molecular draft.
    pub fn predict(
        &self,
        input: MolecularFlowInput<'_>,
    ) -> Result<MolecularFlowPrediction, ModelError> {
        if input.atom_types.size().len() != 1 {
            return Err(ModelError::new(
                "molecular flow atom_types must have shape [num_atoms]",
            ));
        }
        if input.coords.size().len() != 2 || input.coords.size()[1] != 3 {
            return Err(ModelError::new(
                "molecular flow coords must have shape [num_atoms, 3]",
            ));
        }
        if input.x0_coords.size() != input.coords.size() {
            return Err(ModelError::new(
                "molecular flow x0_coords must match coords",
            ));
        }
        let num_atoms = input.atom_types.size()[0];
        if input.coords.size()[0] != num_atoms {
            return Err(ModelError::new(
                "molecular flow atom_types and coords disagree",
            ));
        }

        let device = input.coords.device();
        if num_atoms == 0 {
            return Ok(MolecularFlowPrediction {
                atom_type_logits: Tensor::zeros([0, self.atom_vocab_size], (Kind::Float, device)),
                bond_exists_logits: Tensor::zeros([0, 0], (Kind::Float, device)),
                bond_type_logits: Tensor::zeros(
                    [0, 0, self.bond_vocab_size],
                    (Kind::Float, device),
                ),
                topology_logits: Tensor::zeros([0, 0], (Kind::Float, device)),
                pocket_contact_logits: Tensor::zeros([0], (Kind::Float, device)),
                pocket_context_reconstruction: self.reconstruct_pocket_context(input.conditioning),
                diagnostics: BTreeMap::new(),
            });
        }

        let t = input.t.clamp(0.0, 1.0);
        let two_pi_t = std::f64::consts::PI * 2.0 * t;
        let timestep = Tensor::from_slice(&[
            t as f32,
            (t * t) as f32,
            two_pi_t.sin() as f32,
            two_pi_t.cos() as f32,
        ])
        .to_device(device)
        .unsqueeze(0)
        .apply(&self.timestep_projection)
        .relu()
        .squeeze_dim(0);

        let atom_tokens = input
            .atom_types
            .clamp(0, self.atom_vocab_size - 1)
            .to_kind(Kind::Int64)
            .apply(&self.atom_embedding);
        let reference_centroid = input.x0_coords.mean_dim([0].as_slice(), false, Kind::Float);
        let centered_coords = input.coords - reference_centroid.unsqueeze(0);
        let centered_x0 = input.x0_coords - reference_centroid.unsqueeze(0);
        let displacement = &centered_coords - &centered_x0;
        let atom_geometry_features =
            atom_invariant_geometry_features(&centered_coords, &centered_x0, &displacement);
        let local_conditioning_query = Tensor::cat(
            &[
                atom_tokens.shallow_clone(),
                atom_geometry_features.narrow(1, 0, 3),
            ],
            1,
        )
        .apply(&self.local_conditioning_query_projection)
        .relu();
        let repeated_context = self
            .atom_conditioning_context(&local_conditioning_query, input.conditioning)?
            .to_device(device)
            .apply(&self.context_projection)
            .relu();
        let repeated_timestep = timestep
            .unsqueeze(0)
            .expand([num_atoms, self.hidden_dim], true);
        let atom_hidden = Tensor::cat(
            &[
                atom_tokens,
                atom_geometry_features,
                repeated_context,
                repeated_timestep,
            ],
            1,
        )
        .apply(&self.atom_projection)
        .relu();
        let atom_type_logits = atom_hidden.apply(&self.atom_type_head);

        let pair_prediction =
            self.predict_pair_logits(&atom_hidden, &centered_coords, input.atom_types);
        let bond_exists_logits = pair_prediction.bond_exists_logits;
        let topology_logits = pair_prediction.topology_logits;
        let bond_type_logits = pair_prediction.bond_type_logits;
        let pocket_contact_logits = atom_hidden.apply(&self.pocket_contact_head).squeeze_dim(-1);

        let mut diagnostics = BTreeMap::new();
        diagnostics.insert(
            "molecular_flow_atom_logit_mean".to_string(),
            atom_type_logits.mean(Kind::Float).double_value(&[]),
        );
        diagnostics.insert(
            "molecular_flow_bond_probability_mean".to_string(),
            bond_exists_logits
                .sigmoid()
                .mean(Kind::Float)
                .double_value(&[]),
        );
        diagnostics.insert(
            "molecular_flow_topology_probability_mean".to_string(),
            topology_logits
                .sigmoid()
                .mean(Kind::Float)
                .double_value(&[]),
        );
        diagnostics.insert(
            "molecular_flow_pocket_contact_probability_mean".to_string(),
            pocket_contact_logits
                .sigmoid()
                .mean(Kind::Float)
                .double_value(&[]),
        );
        diagnostics.insert(
            "molecular_flow_slot_local_conditioning_enabled".to_string(),
            (self.conditioning_kind == DecoderConditioningKind::LocalAtomSlotAttention) as i32
                as f64,
        );
        diagnostics.insert(
            "molecular_flow_conditioning_mean_pooled".to_string(),
            (self.conditioning_kind == DecoderConditioningKind::MeanPooled) as i32 as f64,
        );
        diagnostics.insert(
            "molecular_flow_pair_logit_chunk_count".to_string(),
            pair_prediction.chunk_count as f64,
        );
        diagnostics.insert(
            "molecular_flow_pair_logit_max_chunk_rows".to_string(),
            pair_prediction.max_chunk_rows as f64,
        );
        diagnostics.insert(
            "molecular_flow_bond_distance_prior_mean".to_string(),
            pair_prediction.bond_distance_prior_mean,
        );

        Ok(MolecularFlowPrediction {
            atom_type_logits,
            bond_exists_logits,
            bond_type_logits,
            topology_logits,
            pocket_contact_logits,
            pocket_context_reconstruction: self.reconstruct_pocket_context(input.conditioning),
            diagnostics,
        })
    }

    fn reconstruct_pocket_context(&self, conditioning: &ConditioningState) -> Tensor {
        if conditioning.pocket_context.numel() == 0 {
            Tensor::zeros(
                [0, self.hidden_dim],
                (Kind::Float, conditioning.pocket_context.device()),
            )
        } else {
            conditioning
                .pocket_context
                .apply(&self.pocket_context_projection)
        }
    }

    fn predict_pair_logits(
        &self,
        atom_hidden: &Tensor,
        centered_coords: &Tensor,
        atom_types: &Tensor,
    ) -> PairPrediction {
        let num_atoms = atom_hidden.size()[0];
        let chunk_size = PAIR_LOGIT_ROW_CHUNK_SIZE.max(1);
        let mut bond_exists_chunks = Vec::new();
        let mut topology_chunks = Vec::new();
        let mut bond_type_chunks = Vec::new();
        let mut start = 0;
        let mut chunk_count = 0usize;
        let mut max_chunk_rows = 0usize;

        while start < num_atoms {
            let rows = (num_atoms - start).min(chunk_size);
            let atom_rows = atom_hidden.narrow(0, start, rows);
            let coord_rows = centered_coords.narrow(0, start, rows);
            let left = atom_rows
                .unsqueeze(1)
                .expand([rows, num_atoms, self.hidden_dim], true);
            let right = atom_hidden
                .unsqueeze(0)
                .expand([rows, num_atoms, self.hidden_dim], true);
            let pair_delta = coord_rows.unsqueeze(1) - centered_coords.unsqueeze(0);
            let pair_distance_sq = pair_delta
                .pow_tensor_scalar(2.0)
                .sum_dim_intlist([2].as_slice(), true, Kind::Float)
                .clamp_min(1.0e-12);
            let pair_distance = pair_distance_sq.sqrt();
            let pair_invariants = Tensor::cat(
                &[
                    pair_distance.shallow_clone(),
                    pair_distance_sq,
                    (&pair_distance + 1.0e-6).reciprocal(),
                    Tensor::ones_like(&pair_distance),
                ],
                2,
            );
            let pair_hidden = Tensor::cat(&[left, right, pair_invariants], 2)
                .apply(&self.pair_projection)
                .relu();
            bond_exists_chunks.push(pair_hidden.apply(&self.bond_exists_head).squeeze_dim(-1));
            topology_chunks.push(pair_hidden.apply(&self.topology_head).squeeze_dim(-1));
            bond_type_chunks.push(pair_hidden.apply(&self.bond_type_head));
            chunk_count += 1;
            max_chunk_rows = max_chunk_rows.max(rows as usize);
            start += rows;
        }

        let bond_distance_prior = bond_distance_logit_prior(centered_coords, atom_types);
        let bond_distance_prior_mean = bond_distance_prior.mean(Kind::Float).double_value(&[]);
        let raw_bond_exists_logits = Tensor::cat(&bond_exists_chunks, 0) + &bond_distance_prior;
        let raw_topology_logits = Tensor::cat(&topology_chunks, 0);
        let raw_bond_type_logits = Tensor::cat(&bond_type_chunks, 0);
        PairPrediction {
            bond_exists_logits: symmetrize_matrix(&raw_bond_exists_logits),
            topology_logits: symmetrize_matrix(&raw_topology_logits),
            bond_type_logits: symmetrize_pair_logits(&raw_bond_type_logits),
            chunk_count,
            max_chunk_rows,
            bond_distance_prior_mean,
        }
    }

    fn atom_conditioning_context(
        &self,
        atom_query: &Tensor,
        conditioning: &ConditioningState,
    ) -> Result<Tensor, ModelError> {
        match self.conditioning_kind {
            DecoderConditioningKind::MeanPooled => {
                Ok(
                    conditioning_summary(conditioning, self.hidden_dim, atom_query.device())
                        .unsqueeze(0)
                        .expand([atom_query.size()[0], self.hidden_dim * 3], true),
                )
            }
            DecoderConditioningKind::LocalAtomSlotAttention => {
                slot_local_conditioning(atom_query, conditioning, self.hidden_dim)
            }
        }
    }
}

struct PairPrediction {
    bond_exists_logits: Tensor,
    topology_logits: Tensor,
    bond_type_logits: Tensor,
    chunk_count: usize,
    max_chunk_rows: usize,
    bond_distance_prior_mean: f64,
}

fn conditioning_summary(
    conditioning: &ConditioningState,
    hidden_dim: i64,
    device: tch::Device,
) -> Tensor {
    Tensor::cat(
        &[
            mean_or_zeros(&conditioning.topology_context, hidden_dim, device),
            mean_or_zeros(&conditioning.geometry_context, hidden_dim, device),
            mean_or_zeros(&conditioning.pocket_context, hidden_dim, device),
        ],
        0,
    )
}

fn mean_or_zeros(tensor: &Tensor, hidden_dim: i64, device: tch::Device) -> Tensor {
    if tensor.numel() == 0 {
        Tensor::zeros([hidden_dim], (Kind::Float, device))
    } else {
        tensor
            .to_device(device)
            .mean_dim([0].as_slice(), false, Kind::Float)
    }
}

fn atom_invariant_geometry_features(
    centered_coords: &Tensor,
    centered_x0: &Tensor,
    displacement: &Tensor,
) -> Tensor {
    let coord_norm = row_norm(centered_coords);
    let x0_norm = row_norm(centered_x0);
    let displacement_norm = row_norm(displacement);
    let coord_x0_dot = row_dot(centered_coords, centered_x0);
    let coord_displacement_dot = row_dot(centered_coords, displacement);
    let x0_displacement_dot = row_dot(centered_x0, displacement);
    Tensor::cat(
        &[
            coord_norm.shallow_clone(),
            x0_norm.shallow_clone(),
            displacement_norm.shallow_clone(),
            coord_x0_dot,
            coord_displacement_dot,
            x0_displacement_dot,
            coord_norm.pow_tensor_scalar(2.0),
            x0_norm.pow_tensor_scalar(2.0),
            displacement_norm.pow_tensor_scalar(2.0),
        ],
        1,
    )
}

fn row_norm(values: &Tensor) -> Tensor {
    values
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .clamp_min(1.0e-12)
        .sqrt()
}

fn row_dot(left: &Tensor, right: &Tensor) -> Tensor {
    (left * right).sum_dim_intlist([1].as_slice(), true, Kind::Float)
}

fn slot_local_conditioning(
    atom_query: &Tensor,
    conditioning: &ConditioningState,
    hidden_dim: i64,
) -> Result<Tensor, ModelError> {
    Ok(Tensor::cat(
        &[
            slot_local_context(
                atom_query,
                &conditioning.topology_context,
                hidden_dim,
                "topology",
            )?,
            slot_local_context(
                atom_query,
                &conditioning.geometry_context,
                hidden_dim,
                "geometry",
            )?,
            slot_local_context(
                atom_query,
                &conditioning.pocket_context,
                hidden_dim,
                "pocket",
            )?,
        ],
        1,
    ))
}

fn slot_local_context(
    atom_query: &Tensor,
    slots: &Tensor,
    hidden_dim: i64,
    modality: &str,
) -> Result<Tensor, ModelError> {
    let num_atoms = atom_query.size()[0];
    let device = atom_query.device();
    if slots.numel() == 0 || slots.size().first().copied().unwrap_or(0) == 0 {
        return Ok(Tensor::zeros(
            [num_atoms, hidden_dim],
            (Kind::Float, device),
        ));
    }
    if slots.size().len() != 2 || slots.size()[1] != hidden_dim {
        return Err(ModelError::new(format!(
            "molecular flow {modality} conditioning context must have shape [num_slots, hidden_dim]"
        )));
    }
    let slots = slots.to_device(device);
    let slot_mask = slots
        .abs()
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .gt(1.0e-12)
        .to_kind(Kind::Float)
        .reshape([1, slots.size()[0]]);
    if slot_mask.sum(Kind::Float).double_value(&[]) <= 0.0 {
        return Ok(Tensor::zeros(
            [num_atoms, hidden_dim],
            (Kind::Float, device),
        ));
    }
    let scale = (hidden_dim as f64).sqrt();
    let scores = atom_query.matmul(&slots.transpose(0, 1)) / scale;
    let attention = masked_slot_attention(&scores, &slot_mask);
    Ok(attention.matmul(&slots))
}

fn masked_slot_attention(scores: &Tensor, slot_mask: &Tensor) -> Tensor {
    let mask = slot_mask.clamp(0.0, 1.0).to_device(scores.device());
    let invalid = Tensor::ones_like(&mask) - &mask;
    let weights = (scores + invalid * -1.0e9).softmax(-1, Kind::Float) * &mask;
    let denom = weights
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .clamp_min(1.0e-6);
    weights / denom
}

fn symmetrize_matrix(matrix: &Tensor) -> Tensor {
    (matrix + matrix.transpose(0, 1)) * 0.5
}

fn symmetrize_pair_logits(logits: &Tensor) -> Tensor {
    (logits + logits.transpose(0, 1)) * 0.5
}

fn bond_distance_logit_prior(centered_coords: &Tensor, atom_types: &Tensor) -> Tensor {
    let atom_count = centered_coords.size().first().copied().unwrap_or(0).max(0);
    if atom_count <= 0 {
        return Tensor::zeros([0, 0], (Kind::Float, centered_coords.device()));
    }
    let pair_delta = centered_coords.unsqueeze(1) - centered_coords.unsqueeze(0);
    let pair_distance = pair_delta
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .clamp_min(1.0e-12)
        .sqrt();
    let radii = covalent_radius_tensor(atom_types, atom_count, centered_coords.device());
    let ideal_distance = (radii.unsqueeze(1) + radii.unsqueeze(0))
        .clamp_min(BOND_DISTANCE_PRIOR_CENTER_ANGSTROM * 0.4);
    let width = (&ideal_distance * 0.25)
        .clamp_min(BOND_DISTANCE_PRIOR_WIDTH_ANGSTROM * 0.55)
        .clamp_max(BOND_DISTANCE_PRIOR_WIDTH_ANGSTROM * 1.4);
    let normalized = (pair_distance - ideal_distance).abs() / width;
    let prior = (BOND_DISTANCE_PRIOR_PEAK_LOGIT - normalized.pow_tensor_scalar(2.0)).clamp(
        BOND_DISTANCE_PRIOR_FLOOR_LOGIT,
        BOND_DISTANCE_PRIOR_PEAK_LOGIT,
    );
    let hydrogen = aligned_atom_types(atom_types, atom_count, centered_coords.device())
        .eq(4)
        .to_kind(Kind::Float);
    let hydrogen_pair = hydrogen.unsqueeze(1) * hydrogen.unsqueeze(0);
    let diagonal = Tensor::eye(atom_count, (Kind::Float, centered_coords.device()));
    let invalid_pair_mask = (hydrogen_pair + diagonal).clamp(0.0, 1.0);
    &prior * (Tensor::ones_like(&invalid_pair_mask) - &invalid_pair_mask)
        + invalid_pair_mask * BOND_DISTANCE_PRIOR_FLOOR_LOGIT
}

fn aligned_atom_types(atom_types: &Tensor, atom_count: i64, device: tch::Device) -> Tensor {
    if atom_count <= 0 {
        return Tensor::zeros([0], (Kind::Int64, device));
    }
    let available = atom_types.size().first().copied().unwrap_or(0).max(0);
    if available >= atom_count {
        atom_types.narrow(0, 0, atom_count).to_device(device)
    } else {
        let padded = Tensor::zeros([atom_count], (Kind::Int64, device));
        if available > 0 {
            padded
                .narrow(0, 0, available)
                .copy_(&atom_types.narrow(0, 0, available).to_device(device));
        }
        padded
    }
}

fn covalent_radius_tensor(atom_types: &Tensor, atom_count: i64, device: tch::Device) -> Tensor {
    let aligned = aligned_atom_types(atom_types, atom_count, device);
    let radii = (0..atom_count)
        .map(|index| covalent_radius(aligned.int64_value(&[index])) as f32)
        .collect::<Vec<_>>();
    Tensor::from_slice(&radii).to_device(device)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn same_mean_slots_a() -> Tensor {
        Tensor::from_slice(&[4.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape([2, 4])
    }

    fn same_mean_slots_b() -> Tensor {
        Tensor::from_slice(&[2.0_f32, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0]).reshape([2, 4])
    }

    fn conditioning(slots: &Tensor) -> ConditioningState {
        ConditioningState {
            topology_context: slots.shallow_clone(),
            geometry_context: slots.shallow_clone(),
            pocket_context: slots.shallow_clone(),
            gate_summary: crate::models::GenerationGateSummary::default(),
        }
    }

    fn atom_types() -> Tensor {
        Tensor::from_slice(&[0_i64, 1, 2])
    }

    fn coords() -> Tensor {
        Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.0, 0.25, 0.0, -0.5, 1.0, 0.5]).reshape([3, 3])
    }

    fn repeated_atom_types(atom_count: i64, vocab_size: i64) -> Tensor {
        let values = (0..atom_count)
            .map(|index| index % vocab_size)
            .collect::<Vec<_>>();
        Tensor::from_slice(&values)
    }

    fn line_coords(atom_count: i64) -> Tensor {
        let values = (0..atom_count)
            .flat_map(|index| [index as f32 * 0.1, (index % 7) as f32 * 0.05, 0.0])
            .collect::<Vec<_>>();
        Tensor::from_slice(&values).reshape([atom_count, 3])
    }

    fn max_abs_delta(lhs: &Tensor, rhs: &Tensor) -> f64 {
        (lhs - rhs).abs().max().double_value(&[])
    }

    fn rotate(coords: &Tensor, rotation: &Tensor) -> Tensor {
        coords.matmul(&rotation.transpose(0, 1))
    }

    #[test]
    fn bond_distance_prior_prefers_local_chemical_distances() {
        let coords =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.5, 0.0, 0.0, 4.0, 0.0, 0.0]).reshape([3, 3]);
        let atom_types = Tensor::from_slice(&[0_i64, 0, 0]);

        let prior = bond_distance_logit_prior(&coords, &atom_types);
        let near = prior.double_value(&[0, 1]);
        let far = prior.double_value(&[0, 2]);
        let diagonal = prior.double_value(&[0, 0]);

        assert!(
            near > far + 5.0,
            "near covalent-scale pairs should receive a much stronger native bond prior"
        );
        assert!(
            diagonal < near - 5.0,
            "self-pairs should not receive the local bond prior"
        );
    }

    #[test]
    fn bond_distance_prior_rejects_hydrogen_hydrogen_pairs() {
        let coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.74, 0.0, 0.0, 1.15, 0.0, 0.0])
            .reshape([3, 3]);
        let atom_types = Tensor::from_slice(&[4_i64, 4, 0]);

        let prior = bond_distance_logit_prior(&coords, &atom_types);
        let hydrogen_hydrogen = prior.double_value(&[0, 1]);
        let hydrogen_carbon = prior.double_value(&[0, 2]);

        assert!(
            hydrogen_hydrogen < hydrogen_carbon - 5.0,
            "organic ligand native graph prior should suppress hydrogen-hydrogen bonds"
        );
    }

    #[test]
    fn slot_local_context_masks_zeroed_inactive_slots() {
        let atom_query =
            Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]).reshape([2, 4]);
        let active_slot = Tensor::from_slice(&[2.0_f32, 0.0, 0.0, 0.0]).reshape([1, 4]);
        let padded_slots = Tensor::cat(
            &[
                active_slot.shallow_clone(),
                Tensor::zeros([1, 4], (Kind::Float, tch::Device::Cpu)),
            ],
            0,
        );

        let compact = slot_local_context(&atom_query, &active_slot, 4, "test").unwrap();
        let padded = slot_local_context(&atom_query, &padded_slots, 4, "test").unwrap();

        assert!(
            max_abs_delta(&compact, &padded) < 1.0e-6,
            "zeroed inactive slots should not dilute local conditioning"
        );
    }

    #[test]
    fn molecular_flow_local_slot_conditioning_changes_outputs_with_controlled_mean() {
        tch::manual_seed(19);
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let head = FullMolecularFlowHead::new_with_conditioning_kind(
            &vs.root(),
            8,
            4,
            4,
            DecoderConditioningKind::LocalAtomSlotAttention,
        );
        let atom_types = atom_types();
        let coords = coords();
        let x0_coords = Tensor::zeros([3, 3], (Kind::Float, tch::Device::Cpu));
        let conditioning_a = conditioning(&same_mean_slots_a());
        let conditioning_b = conditioning(&same_mean_slots_b());

        let prediction_a = head
            .predict(MolecularFlowInput {
                atom_types: &atom_types,
                coords: &coords,
                x0_coords: &x0_coords,
                t: 0.35,
                conditioning: &conditioning_a,
            })
            .unwrap();
        let prediction_b = head
            .predict(MolecularFlowInput {
                atom_types: &atom_types,
                coords: &coords,
                x0_coords: &x0_coords,
                t: 0.35,
                conditioning: &conditioning_b,
            })
            .unwrap();

        let output_delta = max_abs_delta(
            &prediction_a.atom_type_logits,
            &prediction_b.atom_type_logits,
        )
        .max(max_abs_delta(
            &prediction_a.bond_exists_logits,
            &prediction_b.bond_exists_logits,
        ))
        .max(max_abs_delta(
            &prediction_a.pocket_contact_logits,
            &prediction_b.pocket_contact_logits,
        ));
        assert!(
            output_delta > 1e-6,
            "local slot conditioning should affect task-critical heads, got delta {output_delta}"
        );
        assert_eq!(
            prediction_a.diagnostics["molecular_flow_slot_local_conditioning_enabled"],
            1.0
        );
        assert_eq!(
            prediction_a.diagnostics["molecular_flow_conditioning_mean_pooled"],
            0.0
        );
    }

    #[test]
    fn molecular_flow_mean_pooled_conditioning_remains_distribution_invariant_ablation() {
        tch::manual_seed(19);
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let head = FullMolecularFlowHead::new_with_conditioning_kind(
            &vs.root(),
            8,
            4,
            4,
            DecoderConditioningKind::MeanPooled,
        );
        let atom_types = atom_types();
        let coords = coords();
        let x0_coords = Tensor::zeros([3, 3], (Kind::Float, tch::Device::Cpu));
        let conditioning_a = conditioning(&same_mean_slots_a());
        let conditioning_b = conditioning(&same_mean_slots_b());

        let prediction_a = head
            .predict(MolecularFlowInput {
                atom_types: &atom_types,
                coords: &coords,
                x0_coords: &x0_coords,
                t: 0.35,
                conditioning: &conditioning_a,
            })
            .unwrap();
        let prediction_b = head
            .predict(MolecularFlowInput {
                atom_types: &atom_types,
                coords: &coords,
                x0_coords: &x0_coords,
                t: 0.35,
                conditioning: &conditioning_b,
            })
            .unwrap();

        let output_delta = max_abs_delta(
            &prediction_a.atom_type_logits,
            &prediction_b.atom_type_logits,
        )
        .max(max_abs_delta(
            &prediction_a.bond_exists_logits,
            &prediction_b.bond_exists_logits,
        ))
        .max(max_abs_delta(
            &prediction_a.pocket_contact_logits,
            &prediction_b.pocket_contact_logits,
        ));
        assert!(
            output_delta < 1e-6,
            "mean-pooled ablation should ignore slot distribution at fixed mean, got delta {output_delta}"
        );
        assert_eq!(
            prediction_a.diagnostics["molecular_flow_slot_local_conditioning_enabled"],
            0.0
        );
        assert_eq!(
            prediction_a.diagnostics["molecular_flow_conditioning_mean_pooled"],
            1.0
        );
    }

    #[test]
    fn molecular_flow_pair_logits_are_computed_in_bounded_row_chunks() {
        tch::manual_seed(23);
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let head = FullMolecularFlowHead::new_with_conditioning_kind(
            &vs.root(),
            8,
            4,
            4,
            DecoderConditioningKind::MeanPooled,
        );
        let atom_count = PAIR_LOGIT_ROW_CHUNK_SIZE + 5;
        let atom_types = repeated_atom_types(atom_count, 8);
        let coords = line_coords(atom_count);
        let x0_coords = Tensor::zeros([atom_count, 3], (Kind::Float, tch::Device::Cpu));
        let conditioning = conditioning(&same_mean_slots_a());

        let prediction = head
            .predict(MolecularFlowInput {
                atom_types: &atom_types,
                coords: &coords,
                x0_coords: &x0_coords,
                t: 0.5,
                conditioning: &conditioning,
            })
            .unwrap();

        assert_eq!(
            prediction.bond_exists_logits.size(),
            vec![atom_count, atom_count]
        );
        assert_eq!(
            prediction.bond_type_logits.size(),
            vec![atom_count, atom_count, 4]
        );
        assert!(prediction.diagnostics["molecular_flow_pair_logit_chunk_count"] > 1.0);
        assert_eq!(
            prediction.diagnostics["molecular_flow_pair_logit_max_chunk_rows"],
            PAIR_LOGIT_ROW_CHUNK_SIZE as f64
        );
        assert!(
            max_abs_delta(
                &prediction.bond_exists_logits,
                &prediction.bond_exists_logits.transpose(0, 1),
            ) < 1e-6
        );
    }

    #[test]
    fn molecular_flow_scalar_branches_are_rigid_motion_invariant() {
        tch::manual_seed(29);
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let head = FullMolecularFlowHead::new_with_conditioning_kind(
            &vs.root(),
            8,
            4,
            4,
            DecoderConditioningKind::LocalAtomSlotAttention,
        );
        let atom_types = atom_types();
        let coords = coords();
        let x0_coords = Tensor::from_slice(&[0.1_f32, -0.2, 0.0, 0.8, 0.0, 0.1, -0.6, 0.5, 0.3])
            .reshape([3, 3]);
        let conditioning = conditioning(&same_mean_slots_a());
        let rotation =
            Tensor::from_slice(&[0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([3, 3]);
        let shift = Tensor::from_slice(&[3.0_f32, -2.0, 1.0]).reshape([1, 3]);

        let base = head
            .predict(MolecularFlowInput {
                atom_types: &atom_types,
                coords: &coords,
                x0_coords: &x0_coords,
                t: 0.35,
                conditioning: &conditioning,
            })
            .unwrap();
        let transformed_coords = rotate(&coords, &rotation) + &shift;
        let transformed_x0 = rotate(&x0_coords, &rotation) + &shift;
        let transformed = head
            .predict(MolecularFlowInput {
                atom_types: &atom_types,
                coords: &transformed_coords,
                x0_coords: &transformed_x0,
                t: 0.35,
                conditioning: &conditioning,
            })
            .unwrap();

        assert!(max_abs_delta(&base.atom_type_logits, &transformed.atom_type_logits) < 1.0e-5);
        assert!(max_abs_delta(&base.bond_exists_logits, &transformed.bond_exists_logits) < 1.0e-5);
        assert!(max_abs_delta(&base.bond_type_logits, &transformed.bond_type_logits) < 1.0e-5);
        assert!(max_abs_delta(&base.topology_logits, &transformed.topology_logits) < 1.0e-5);
        assert!(
            max_abs_delta(
                &base.pocket_contact_logits,
                &transformed.pocket_contact_logits
            ) < 1.0e-5
        );
    }
}
