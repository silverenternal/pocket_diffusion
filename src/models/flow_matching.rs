//! Geometry-only flow-matching head built on top of decomposed conditioning.

use tch::{nn, Kind, Tensor};

use super::{ConditioningState, FlowMatchingHead, FlowState, ModelError, VelocityField};

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
            output_norm: nn::layer_norm(vs / "output_norm", vec![hidden_dim], Default::default()),
            velocity_head: nn::linear(vs / "velocity_head", hidden_dim, 3, Default::default()),
            hidden_dim,
        }
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

        let fused = Tensor::cat(
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
        let fused = (fused.shallow_clone() + fused.apply(&self.residual_projection).relu())
            .apply(&self.output_norm);
        Ok(VelocityField {
            velocity: fused.apply(&self.velocity_head),
        })
    }
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
}
