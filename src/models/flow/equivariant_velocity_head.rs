//! EGNN-style equivariant velocity head for geometry flow matching.

use std::collections::BTreeMap;

use tch::{nn, Kind, Tensor};

use crate::models::{
    ConditioningState, FlowMatchingHead, FlowState, ModelError, PairwiseGeometryConfig,
    VelocityField,
};

/// Configuration for the equivariant geometry velocity head.
#[derive(Debug, Clone)]
pub struct EquivariantGeometryVelocityConfig {
    /// Hidden feature width for scalar message networks.
    pub hidden_dim: i64,
    /// Radius and top-k policy for bounded ligand-ligand messages.
    pub pairwise_geometry: PairwiseGeometryConfig,
}

impl Default for EquivariantGeometryVelocityConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            pairwise_geometry: PairwiseGeometryConfig::default(),
        }
    }
}

/// Geometry velocity head with scalar messages multiplying relative vectors.
#[derive(Debug)]
pub struct EquivariantGeometryVelocityHead {
    context_projection: nn::Linear,
    self_weight_projection: nn::Linear,
    pair_weight_projection: nn::Linear,
    pair_weight_out: nn::Linear,
    config: EquivariantGeometryVelocityConfig,
}

impl EquivariantGeometryVelocityHead {
    /// Create an equivariant geometry velocity head.
    pub fn new(vs: &nn::Path, config: EquivariantGeometryVelocityConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        Self {
            context_projection: nn::linear(
                vs / "context_projection",
                hidden_dim * 3 + 10,
                hidden_dim,
                Default::default(),
            ),
            self_weight_projection: nn::linear(
                vs / "self_weight_projection",
                hidden_dim + 2,
                1,
                Default::default(),
            ),
            pair_weight_projection: nn::linear(
                vs / "pair_weight_projection",
                hidden_dim + 2,
                hidden_dim,
                Default::default(),
            ),
            pair_weight_out: nn::linear(vs / "pair_weight_out", hidden_dim, 1, Default::default()),
            config,
        }
    }
}

impl FlowMatchingHead for EquivariantGeometryVelocityHead {
    fn predict_velocity(
        &self,
        state: &FlowState,
        conditioning: &ConditioningState,
    ) -> Result<VelocityField, ModelError> {
        if state.coords.size().len() != 2 || state.coords.size()[1] != 3 {
            return Err(ModelError::new(
                "flow state coords must have shape [num_atoms, 3]",
            ));
        }
        if state.x0_coords.size() != state.coords.size() {
            return Err(ModelError::new(
                "flow state x0_coords must match state.coords shape",
            ));
        }
        if let Some(target) = state.target_coords.as_ref() {
            if target.size() != state.coords.size() {
                return Err(ModelError::new(
                    "flow state target_coords must match state.coords shape when present",
                ));
            }
        }

        let num_atoms = state.coords.size()[0];
        if num_atoms == 0 {
            return Ok(VelocityField {
                velocity: Tensor::zeros_like(&state.coords),
                diagnostics: BTreeMap::new(),
            });
        }

        let device = state.coords.device();
        let context_hidden =
            invariant_conditioning_context(conditioning, self.config.hidden_dim, state.t, device)
                .apply(&self.context_projection)
                .relu();
        let displacement = &state.coords - &state.x0_coords;
        let displacement_norm = row_norm(&displacement);
        let reference_centroid = state.x0_coords.mean_dim([0].as_slice(), false, Kind::Float);
        let coord_radius = row_norm(&(&state.coords - reference_centroid.unsqueeze(0)));
        let x0_radius = row_norm(&(&state.x0_coords - reference_centroid.unsqueeze(0)));
        let self_features = Tensor::cat(
            &[
                context_hidden
                    .unsqueeze(0)
                    .expand([num_atoms, self.config.hidden_dim], true),
                displacement_norm.shallow_clone(),
                (&coord_radius - &x0_radius).abs(),
            ],
            1,
        );
        let self_weight = self_features.apply(&self.self_weight_projection).tanh();
        let self_velocity = displacement * &self_weight;

        let diffs = state.coords.unsqueeze(0) - state.coords.unsqueeze(1);
        let distances_sq = diffs
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([2].as_slice(), false, Kind::Float)
            .clamp_min(1.0e-12);
        let distances = distances_sq.sqrt();
        let neighbor_mask = local_neighbor_mask(&distances, &self.config.pairwise_geometry);
        let pair_context = context_hidden
            .unsqueeze(0)
            .unsqueeze(0)
            .expand([num_atoms, num_atoms, self.config.hidden_dim], true);
        let pair_features = Tensor::cat(
            &[
                pair_context,
                distances.unsqueeze(-1),
                distances_sq.sqrt().reciprocal().unsqueeze(-1),
            ],
            2,
        );
        let pair_weight = pair_features
            .reshape([-1, self.config.hidden_dim + 2])
            .apply(&self.pair_weight_projection)
            .relu()
            .apply(&self.pair_weight_out)
            .tanh()
            .reshape([num_atoms, num_atoms, 1])
            * neighbor_mask.unsqueeze(-1);
        let neighbor_count = neighbor_mask
            .sum_dim_intlist([1].as_slice(), true, Kind::Float)
            .clamp_min(1.0);
        let pair_velocity =
            (diffs * &pair_weight).sum_dim_intlist([1].as_slice(), false, Kind::Float)
                / neighbor_count;
        let pair_velocity = pair_velocity * self.config.pairwise_geometry.residual_scale;
        let velocity = self_velocity + pair_velocity;

        let mut diagnostics = BTreeMap::new();
        let raw_neighbor_count =
            neighbor_mask.sum(Kind::Float).double_value(&[]) / num_atoms.max(1) as f64;
        diagnostics.insert(
            "equivariant_geometry_mean_neighbor_count".to_string(),
            raw_neighbor_count,
        );
        diagnostics.insert(
            "equivariant_geometry_max_neighbors".to_string(),
            self.config.pairwise_geometry.max_neighbors as f64,
        );
        diagnostics.insert(
            "equivariant_geometry_radius".to_string(),
            self.config.pairwise_geometry.radius as f64,
        );
        diagnostics.insert(
            "equivariant_geometry_residual_scale".to_string(),
            self.config.pairwise_geometry.residual_scale,
        );
        diagnostics.insert(
            "equivariant_geometry_self_weight_abs_mean".to_string(),
            self_weight.abs().mean(Kind::Float).double_value(&[]),
        );
        diagnostics.insert(
            "equivariant_geometry_pair_weight_abs_mean".to_string(),
            pair_weight.abs().mean(Kind::Float).double_value(&[]),
        );
        Ok(VelocityField {
            velocity,
            diagnostics,
        })
    }
}

fn invariant_conditioning_context(
    conditioning: &ConditioningState,
    hidden_dim: i64,
    t: f64,
    device: tch::Device,
) -> Tensor {
    let gate_values = [
        conditioning.gate_summary.topo_from_geo as f32,
        conditioning.gate_summary.topo_from_pocket as f32,
        conditioning.gate_summary.geo_from_topo as f32,
        conditioning.gate_summary.geo_from_pocket as f32,
        conditioning.gate_summary.pocket_from_topo as f32,
        conditioning.gate_summary.pocket_from_geo as f32,
    ];
    let t = t.clamp(0.0, 1.0);
    let two_pi_t = std::f64::consts::PI * 2.0 * t;
    Tensor::cat(
        &[
            mean_or_zeros(&conditioning.topology_context, hidden_dim, device),
            mean_or_zeros(&conditioning.geometry_context, hidden_dim, device),
            mean_or_zeros(&conditioning.pocket_context, hidden_dim, device),
            Tensor::from_slice(&gate_values).to_device(device),
            Tensor::from_slice(&[
                t as f32,
                (t * t) as f32,
                two_pi_t.sin() as f32,
                two_pi_t.cos() as f32,
            ])
            .to_device(device),
        ],
        0,
    )
}

fn mean_or_zeros(tokens: &Tensor, hidden_dim: i64, device: tch::Device) -> Tensor {
    if tokens.numel() == 0 {
        Tensor::zeros([hidden_dim], (Kind::Float, device))
    } else {
        tokens
            .to_device(device)
            .mean_dim([0].as_slice(), false, Kind::Float)
    }
}

fn row_norm(values: &Tensor) -> Tensor {
    values
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .clamp_min(1.0e-12)
        .sqrt()
}

fn local_neighbor_mask(distances: &Tensor, config: &PairwiseGeometryConfig) -> Tensor {
    let atom_count = distances.size().first().copied().unwrap_or(0).max(0);
    if atom_count <= 1 || config.max_neighbors == 0 {
        return Tensor::zeros_like(distances);
    }
    let device = distances.device();
    let not_self = Tensor::ones([atom_count, atom_count], (Kind::Float, device))
        - Tensor::eye(atom_count, (Kind::Float, device));
    let radius_mask = distances.le(config.radius as f64).to_kind(Kind::Float) * &not_self;
    let retained = config.max_neighbors.min((atom_count - 1) as usize) as i64;
    if retained >= atom_count - 1 {
        return radius_mask;
    }
    let invalid = Tensor::ones_like(&radius_mask) - &radius_mask;
    let masked_distances = distances + invalid * 1.0e6;
    let (_, indices) = masked_distances.topk(retained, 1, false, true);
    Tensor::zeros_like(&radius_mask).scatter_value(1, &indices, 1.0) * radius_mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{rotation_velocity_consistency_error, GenerationGateSummary};

    fn conditioning(hidden_dim: i64) -> ConditioningState {
        ConditioningState {
            topology_context: Tensor::ones([3, hidden_dim], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([3, hidden_dim], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([5, hidden_dim], (Kind::Float, tch::Device::Cpu)),
            gate_summary: GenerationGateSummary::default(),
        }
    }

    fn state() -> FlowState {
        FlowState {
            coords: Tensor::from_slice(&[
                0.2_f32, -0.4, 1.1, 1.3, 0.7, -0.2, -0.9, 0.5, 0.3, 0.4, -1.0, 0.8,
            ])
            .reshape([4, 3]),
            x0_coords: Tensor::from_slice(&[
                0.1_f32, -0.6, 0.9, 1.0, 0.9, -0.3, -1.0, 0.2, 0.1, 0.6, -0.8, 0.5,
            ])
            .reshape([4, 3]),
            target_coords: None,
            t: 0.35,
        }
    }

    #[test]
    fn equivariant_geometry_head_predicts_atomwise_velocity() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = EquivariantGeometryVelocityHead::new(
            &var_store.root(),
            EquivariantGeometryVelocityConfig {
                hidden_dim: 16,
                pairwise_geometry: PairwiseGeometryConfig::default(),
            },
        );

        let velocity = head.predict_velocity(&state(), &conditioning(16)).unwrap();

        assert_eq!(velocity.velocity.size(), vec![4, 3]);
        assert!(velocity
            .diagnostics
            .contains_key("equivariant_geometry_mean_neighbor_count"));
    }

    #[test]
    fn equivariant_geometry_head_is_translation_invariant() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = EquivariantGeometryVelocityHead::new(
            &var_store.root(),
            EquivariantGeometryVelocityConfig {
                hidden_dim: 16,
                pairwise_geometry: PairwiseGeometryConfig::default(),
            },
        );
        let base_state = state();
        let shift = Tensor::from_slice(&[3.0_f32, -2.0, 1.0]).reshape([1, 3]);
        let shifted_state = FlowState {
            coords: &base_state.coords + &shift,
            x0_coords: &base_state.x0_coords + &shift,
            target_coords: None,
            t: base_state.t,
        };

        let base = head
            .predict_velocity(&base_state, &conditioning(16))
            .unwrap()
            .velocity;
        let shifted = head
            .predict_velocity(&shifted_state, &conditioning(16))
            .unwrap()
            .velocity;

        let max_abs = (&base - &shifted).abs().max().double_value(&[]);
        assert!(max_abs < 1.0e-5, "max translation drift was {max_abs}");
    }

    #[test]
    fn equivariant_geometry_head_rotates_velocities_consistently() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = EquivariantGeometryVelocityHead::new(
            &var_store.root(),
            EquivariantGeometryVelocityConfig {
                hidden_dim: 16,
                pairwise_geometry: PairwiseGeometryConfig::default(),
            },
        );
        let rotation =
            Tensor::from_slice(&[0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([3, 3]);

        let error =
            rotation_velocity_consistency_error(&head, &state(), &conditioning(16), &rotation)
                .unwrap();

        assert!(error < 1.0e-5, "rotation consistency error was {error}");
    }

    #[test]
    fn equivariant_geometry_neighbor_mask_respects_topk() {
        let coords = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ])
        .reshape([4, 3]);
        let distances = {
            let diffs = coords.unsqueeze(0) - coords.unsqueeze(1);
            diffs
                .pow_tensor_scalar(2.0)
                .sum_dim_intlist([2].as_slice(), false, Kind::Float)
                .sqrt()
        };
        let mask = local_neighbor_mask(
            &distances,
            &PairwiseGeometryConfig {
                radius: 10.0,
                max_neighbors: 1,
                residual_scale: 0.1,
            },
        );
        let max_neighbors = mask
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            .max()
            .double_value(&[]);

        assert!(max_neighbors <= 1.0);
    }
}
