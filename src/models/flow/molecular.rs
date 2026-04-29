//! Full molecular flow heads for de novo pocket-conditioned generation.
//!
//! The geometry branch remains continuous velocity matching. Atom type, bond,
//! topology, and pocket/context branches are discrete or representation-flow
//! heads trained from the same controlled conditioning state.

use std::collections::BTreeMap;

use tch::{nn, Kind, Tensor};

use crate::config::DecoderConditioningKind;
use crate::models::{ConditioningState, ModelError};

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
        let repeated_context = self
            .atom_conditioning_context(&atom_tokens, input.conditioning)?
            .to_device(device)
            .apply(&self.context_projection)
            .relu();
        let reference_centroid = input.x0_coords.mean_dim([0].as_slice(), false, Kind::Float);
        let centered_coords = input.coords - reference_centroid.unsqueeze(0);
        let centered_x0 = input.x0_coords - reference_centroid.unsqueeze(0);
        let displacement = &centered_coords - &centered_x0;
        let repeated_timestep = timestep
            .unsqueeze(0)
            .expand([num_atoms, self.hidden_dim], true);
        let atom_hidden = Tensor::cat(
            &[
                atom_tokens,
                centered_coords.shallow_clone(),
                centered_x0,
                displacement,
                repeated_context,
                repeated_timestep,
            ],
            1,
        )
        .apply(&self.atom_projection)
        .relu();
        let atom_type_logits = atom_hidden.apply(&self.atom_type_head);

        let left = atom_hidden
            .unsqueeze(1)
            .expand([num_atoms, num_atoms, self.hidden_dim], true);
        let right = atom_hidden
            .unsqueeze(0)
            .expand([num_atoms, num_atoms, self.hidden_dim], true);
        let pair_delta = centered_coords.unsqueeze(1) - centered_coords.unsqueeze(0);
        let pair_distance = pair_delta
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([2].as_slice(), true, Kind::Float)
            .sqrt();
        let pair_hidden = Tensor::cat(&[left, right, pair_delta, pair_distance], 2)
            .apply(&self.pair_projection)
            .relu();
        let bond_exists_logits =
            symmetrize_matrix(&pair_hidden.apply(&self.bond_exists_head).squeeze_dim(-1));
        let topology_logits =
            symmetrize_matrix(&pair_hidden.apply(&self.topology_head).squeeze_dim(-1));
        let raw_bond_type_logits = pair_hidden.apply(&self.bond_type_head);
        let bond_type_logits = symmetrize_pair_logits(&raw_bond_type_logits);
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
    let scale = (hidden_dim as f64).sqrt();
    let attention = (atom_query.matmul(&slots.transpose(0, 1)) / scale).softmax(-1, Kind::Float);
    Ok(attention.matmul(&slots))
}

fn symmetrize_matrix(matrix: &Tensor) -> Tensor {
    (matrix + matrix.transpose(0, 1)) * 0.5
}

fn symmetrize_pair_logits(logits: &Tensor) -> Tensor {
    (logits + logits.transpose(0, 1)) * 0.5
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

    fn max_abs_delta(lhs: &Tensor, rhs: &Tensor) -> f64 {
        (lhs - rhs).abs().max().double_value(&[])
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
}
