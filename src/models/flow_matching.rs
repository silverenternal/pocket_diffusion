//! Geometry-only flow-matching head built on top of decomposed conditioning.

use std::collections::BTreeMap;
use tch::{nn, Kind, Tensor};

use super::{
    ConditioningState, FlowMatchingHead, FlowState, ModelError, PairwiseGeometryConfig,
    PairwiseGeometryMessagePassing, VelocityField,
};

/// Lightweight velocity head for geometry-only flow matching.
#[derive(Debug)]
pub struct GeometryFlowMatchingHead {
    coord_projection: nn::Linear,
    x0_projection: nn::Linear,
    displacement_projection: nn::Linear,
    timestep_projection: nn::Linear,
    conditioning_projection: nn::Linear,
    gate_projection: nn::Linear,
    fusion_projection: nn::Linear,
    residual_projection: nn::Linear,
    pairwise_projection: Option<nn::Linear>,
    pairwise_config: Option<PairwiseGeometryConfig>,
    output_norm: nn::LayerNorm,
    velocity_head: nn::Linear,
    hidden_dim: i64,
}

impl GeometryFlowMatchingHead {
    /// Create a geometry-only velocity predictor.
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        Self {
            coord_projection: nn::linear(
                vs / "coord_projection",
                3,
                hidden_dim,
                Default::default(),
            ),
            x0_projection: nn::linear(vs / "x0_projection", 3, hidden_dim, Default::default()),
            displacement_projection: nn::linear(
                vs / "displacement_projection",
                3,
                hidden_dim,
                Default::default(),
            ),
            timestep_projection: nn::linear(
                vs / "timestep_projection",
                4,
                hidden_dim,
                Default::default(),
            ),
            conditioning_projection: nn::linear(
                vs / "conditioning_projection",
                hidden_dim * 3,
                hidden_dim,
                Default::default(),
            ),
            gate_projection: nn::linear(vs / "gate_projection", 6, hidden_dim, Default::default()),
            fusion_projection: nn::linear(
                vs / "fusion_projection",
                hidden_dim * 6,
                hidden_dim,
                Default::default(),
            ),
            residual_projection: nn::linear(
                vs / "residual_projection",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ),
            pairwise_projection: None,
            pairwise_config: None,
            output_norm: nn::layer_norm(vs / "output_norm", vec![hidden_dim], Default::default()),
            velocity_head: nn::linear(vs / "velocity_head", hidden_dim, 3, Default::default()),
            hidden_dim,
        }
    }

    /// Create a geometry velocity predictor with bounded pairwise message aggregation enabled.
    pub fn new_with_pairwise(
        vs: &nn::Path,
        hidden_dim: i64,
        pairwise_config: PairwiseGeometryConfig,
    ) -> Self {
        let mut head = Self::new(vs, hidden_dim);
        head.pairwise_projection = Some(nn::linear(
            vs / "pairwise_projection",
            5,
            hidden_dim,
            Default::default(),
        ));
        head.pairwise_config = Some(pairwise_config);
        head
    }
}

impl FlowMatchingHead for GeometryFlowMatchingHead {
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
        let topology_mean = mean_or_zeros(&conditioning.topology_context, self.hidden_dim);
        let geometry_mean = mean_or_zeros(&conditioning.geometry_context, self.hidden_dim);
        let pocket_mean = mean_or_zeros(&conditioning.pocket_context, self.hidden_dim);
        let conditioning_summary = Tensor::cat(&[topology_mean, geometry_mean, pocket_mean], 0)
            .unsqueeze(0)
            .apply(&self.conditioning_projection)
            .relu()
            .squeeze_dim(0);

        let gate_values = [
            conditioning.gate_summary.topo_from_geo as f32,
            conditioning.gate_summary.topo_from_pocket as f32,
            conditioning.gate_summary.geo_from_topo as f32,
            conditioning.gate_summary.geo_from_pocket as f32,
            conditioning.gate_summary.pocket_from_topo as f32,
            conditioning.gate_summary.pocket_from_geo as f32,
        ];
        let gate_summary = Tensor::from_slice(&gate_values)
            .to_device(device)
            .unsqueeze(0)
            .apply(&self.gate_projection)
            .relu()
            .squeeze_dim(0);

        let t = state.t.clamp(0.0, 1.0);
        let two_pi_t = std::f64::consts::PI * 2.0 * t;
        let timestep_features = Tensor::from_slice(&[
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

        let reference_centroid = state.x0_coords.mean_dim([0].as_slice(), false, Kind::Float);
        let centered_coords = &state.coords - reference_centroid.unsqueeze(0);
        let centered_x0 = &state.x0_coords - reference_centroid.unsqueeze(0);
        let displacement = &centered_coords - &centered_x0;

        let coord_hidden = centered_coords.apply(&self.coord_projection).relu();
        let x0_hidden = centered_x0.apply(&self.x0_projection).relu();
        let displacement_hidden = displacement.apply(&self.displacement_projection).relu();
        let conditioning_hidden =
            Tensor::cat(&[conditioning_summary, gate_summary, timestep_features], 0)
                .unsqueeze(0)
                .expand([num_atoms, self.hidden_dim * 3], true);

        let mut fused = Tensor::cat(
            &[
                coord_hidden,
                x0_hidden,
                displacement_hidden,
                conditioning_hidden,
            ],
            1,
        )
        .apply(&self.fusion_projection)
        .relu();
        let mut diagnostics = BTreeMap::new();
        diagnostics.insert(
            "conditioning_gate_input_mean".to_string(),
            gate_values.iter().map(|value| *value as f64).sum::<f64>() / gate_values.len() as f64,
        );
        diagnostics.insert(
            "conditioning_gate_topo_from_geo".to_string(),
            conditioning.gate_summary.topo_from_geo,
        );
        diagnostics.insert(
            "conditioning_gate_topo_from_pocket".to_string(),
            conditioning.gate_summary.topo_from_pocket,
        );
        diagnostics.insert(
            "conditioning_gate_geo_from_topo".to_string(),
            conditioning.gate_summary.geo_from_topo,
        );
        diagnostics.insert(
            "conditioning_gate_geo_from_pocket".to_string(),
            conditioning.gate_summary.geo_from_pocket,
        );
        diagnostics.insert(
            "conditioning_gate_pocket_from_topo".to_string(),
            conditioning.gate_summary.pocket_from_topo,
        );
        diagnostics.insert(
            "conditioning_gate_pocket_from_geo".to_string(),
            conditioning.gate_summary.pocket_from_geo,
        );
        if let (Some(pairwise_projection), Some(pairwise_config)) =
            (&self.pairwise_projection, &self.pairwise_config)
        {
            let (pairwise_features, mean_message_count) =
                pairwise_feature_tensor(&centered_coords, pairwise_config);
            let pairwise_hidden = pairwise_features
                .to_device(device)
                .apply(pairwise_projection)
                .relu()
                * pairwise_config.residual_scale;
            fused = fused + pairwise_hidden;
            diagnostics.insert(
                "pairwise_geometry_mean_neighbor_count".to_string(),
                mean_message_count,
            );
            diagnostics.insert(
                "pairwise_geometry_max_neighbors".to_string(),
                pairwise_config.max_neighbors as f64,
            );
            diagnostics.insert(
                "pairwise_geometry_residual_scale".to_string(),
                pairwise_config.residual_scale,
            );
        }
        let skip = fused.alias_copy();
        let residual = fused.apply(&self.residual_projection).relu();
        let fused = (skip + residual).apply(&self.output_norm);
        Ok(VelocityField {
            velocity: fused.apply(&self.velocity_head),
            diagnostics,
        })
    }
}

/// Measure rotation consistency for a velocity head under a row-vector rotation matrix.
///
/// The metric compares `v(Rx)` against `R v(x)` and returns a relative mean
/// L2 error. This is an augmentation/evaluation diagnostic, not a guarantee of
/// exact E(3) equivariance.
pub fn rotation_velocity_consistency_error<H: FlowMatchingHead>(
    head: &H,
    state: &FlowState,
    conditioning: &ConditioningState,
    rotation: &Tensor,
) -> Result<f64, ModelError> {
    if rotation.size() != vec![3, 3] {
        return Err(ModelError::new("rotation matrix must have shape [3, 3]"));
    }
    let base_velocity = head.predict_velocity(state, conditioning)?.velocity;
    let rotated_state = FlowState {
        coords: rotate_coords(&state.coords, rotation),
        x0_coords: rotate_coords(&state.x0_coords, rotation),
        target_coords: state
            .target_coords
            .as_ref()
            .map(|coords| rotate_coords(coords, rotation)),
        t: state.t,
    };
    let rotated_velocity = head
        .predict_velocity(&rotated_state, conditioning)?
        .velocity;
    let expected_velocity = rotate_coords(&base_velocity, rotation);
    let error = (&rotated_velocity - &expected_velocity)
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float);
    let scale = expected_velocity
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        + 1e-8;
    Ok((error / scale).double_value(&[]))
}

fn rotate_coords(coords: &Tensor, rotation: &Tensor) -> Tensor {
    coords.matmul(&rotation.transpose(0, 1))
}

fn pairwise_feature_tensor(coords: &Tensor, config: &PairwiseGeometryConfig) -> (Tensor, f64) {
    let atom_count = coords.size().first().copied().unwrap_or(0).max(0) as usize;
    if atom_count == 0 {
        return (Tensor::zeros([0, 5], (Kind::Float, coords.device())), 0.0);
    }
    let coord_values = (0..atom_count)
        .map(|atom_ix| {
            [
                coords.double_value(&[atom_ix as i64, 0]) as f32,
                coords.double_value(&[atom_ix as i64, 1]) as f32,
                coords.double_value(&[atom_ix as i64, 2]) as f32,
            ]
        })
        .collect::<Vec<_>>();
    let builder = PairwiseGeometryMessagePassing::new(config.clone());
    let messages = builder.build_messages(&coord_values);
    let mut features = vec![[0.0_f32; 5]; atom_count];
    let mut counts = vec![0_usize; atom_count];
    for message in messages {
        let row = &mut features[message.source];
        row[0] += message.distance;
        row[1] += message.direction[0];
        row[2] += message.direction[1];
        row[3] += message.direction[2];
        row[4] += message.clash_margin;
        counts[message.source] += 1;
    }
    for (row, count) in features.iter_mut().zip(counts.iter()) {
        if *count == 0 {
            continue;
        }
        let inv = 1.0 / *count as f32;
        for value in row.iter_mut() {
            *value *= inv;
        }
    }
    let flat = features
        .into_iter()
        .flat_map(|row| row.into_iter())
        .collect::<Vec<_>>();
    let mean_count = counts.iter().sum::<usize>() as f64 / atom_count as f64;
    (
        Tensor::from_slice(&flat)
            .reshape([atom_count as i64, 5])
            .to_device(coords.device()),
        mean_count,
    )
}

fn mean_or_zeros(tokens: &Tensor, hidden_dim: i64) -> Tensor {
    if tokens.numel() == 0 {
        Tensor::zeros([hidden_dim], (Kind::Float, tokens.device()))
    } else {
        tokens.mean_dim([0].as_slice(), false, Kind::Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_flow_head_predicts_atomwise_velocity() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = GeometryFlowMatchingHead::new(&var_store.root(), 16);
        let state = FlowState {
            coords: Tensor::zeros([5, 3], (Kind::Float, tch::Device::Cpu)),
            x0_coords: Tensor::ones([5, 3], (Kind::Float, tch::Device::Cpu)),
            target_coords: None,
            t: 0.35,
        };
        let conditioning = ConditioningState {
            topology_context: Tensor::ones([4, 16], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([4, 16], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([4, 16], (Kind::Float, tch::Device::Cpu)),
            gate_summary: crate::models::GenerationGateSummary {
                topo_from_geo: 0.5,
                topo_from_pocket: 0.6,
                geo_from_topo: 0.4,
                geo_from_pocket: 0.7,
                pocket_from_topo: 0.3,
                pocket_from_geo: 0.8,
            },
        };

        let velocity = head.predict_velocity(&state, &conditioning).unwrap();

        assert_eq!(velocity.velocity.size(), vec![5, 3]);
    }

    #[test]
    fn geometry_flow_head_is_translation_invariant_for_uniform_shift() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = GeometryFlowMatchingHead::new(&var_store.root(), 16);
        let coords = Tensor::from_slice(&[
            0.2_f32, -0.4, 1.1, 1.3, 0.7, -0.2, -0.9, 0.5, 0.3, 0.8, -1.2, 0.4,
        ])
        .reshape([4, 3]);
        let x0 = Tensor::from_slice(&[
            0.1_f32, -0.6, 0.9, 1.0, 0.9, -0.3, -1.0, 0.2, 0.1, 0.5, -1.0, 0.6,
        ])
        .reshape([4, 3]);
        let base_state = FlowState {
            coords: coords.shallow_clone(),
            x0_coords: x0.shallow_clone(),
            target_coords: None,
            t: 0.42,
        };
        let shift = Tensor::from_slice(&[4.0_f32, -3.0, 1.5]).reshape([1, 3]);
        let shifted_state = FlowState {
            coords: &coords + &shift,
            x0_coords: &x0 + &shift,
            target_coords: None,
            t: 0.42,
        };
        let conditioning = ConditioningState {
            topology_context: Tensor::ones([3, 16], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([3, 16], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([3, 16], (Kind::Float, tch::Device::Cpu)),
            gate_summary: crate::models::GenerationGateSummary::default(),
        };

        let base_velocity = head
            .predict_velocity(&base_state, &conditioning)
            .unwrap()
            .velocity;
        let shifted_velocity = head
            .predict_velocity(&shifted_state, &conditioning)
            .unwrap()
            .velocity;
        let max_abs = (&base_velocity - &shifted_velocity)
            .abs()
            .max()
            .double_value(&[]);

        assert!(max_abs < 1e-5, "max translation drift was {max_abs}");
    }

    #[test]
    fn geometry_flow_head_pairwise_messages_are_bounded_and_diagnostic() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = GeometryFlowMatchingHead::new_with_pairwise(
            &var_store.root(),
            16,
            PairwiseGeometryConfig {
                radius: 100.0,
                max_neighbors: 2,
                residual_scale: 0.05,
            },
        );
        let state = FlowState {
            coords: Tensor::zeros([6, 3], (Kind::Float, tch::Device::Cpu)),
            x0_coords: Tensor::ones([6, 3], (Kind::Float, tch::Device::Cpu)),
            target_coords: None,
            t: 0.2,
        };
        let conditioning = ConditioningState {
            topology_context: Tensor::ones([4, 16], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([4, 16], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([4, 16], (Kind::Float, tch::Device::Cpu)),
            gate_summary: crate::models::GenerationGateSummary::default(),
        };

        let velocity = head.predict_velocity(&state, &conditioning).unwrap();

        assert_eq!(velocity.velocity.size(), vec![6, 3]);
        assert!(velocity.diagnostics["pairwise_geometry_mean_neighbor_count"] <= 2.0);
    }

    #[test]
    fn rotation_consistency_metric_is_finite_for_geometry_head() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = GeometryFlowMatchingHead::new(&var_store.root(), 16);
        let state = FlowState {
            coords: Tensor::from_slice(&[0.2_f32, -0.4, 1.1, 1.3, 0.7, -0.2, -0.9, 0.5, 0.3])
                .reshape([3, 3]),
            x0_coords: Tensor::from_slice(&[0.1_f32, -0.6, 0.9, 1.0, 0.9, -0.3, -1.0, 0.2, 0.1])
                .reshape([3, 3]),
            target_coords: None,
            t: 0.25,
        };
        let conditioning = ConditioningState {
            topology_context: Tensor::ones([2, 16], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([2, 16], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([2, 16], (Kind::Float, tch::Device::Cpu)),
            gate_summary: crate::models::GenerationGateSummary::default(),
        };
        let rotation =
            Tensor::from_slice(&[0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([3, 3]);

        let error =
            rotation_velocity_consistency_error(&head, &state, &conditioning, &rotation).unwrap();

        assert!(error.is_finite());
        assert!(error >= 0.0);
    }
}
