//! Gated atom-to-pocket cross-attention velocity head.

use std::collections::BTreeMap;

use tch::{nn, no_grad, Kind, Tensor};

use crate::models::{
    ConditioningState, EquivariantGeometryVelocityHead, FlowMatchingHead, FlowState,
    GeometryFlowMatchingHead, ModelError, PairwiseGeometryConfig, VelocityField,
};

/// Configuration for the gated atom-to-pocket velocity head ablation.
#[derive(Debug, Clone)]
pub struct AtomPocketCrossAttentionVelocityConfig {
    /// Enables cross-attention. When false, only the baseline geometry head is used.
    pub enabled: bool,
    /// Hidden feature width used by projections.
    pub hidden_dim: i64,
    /// Initial gate bias. Negative values make the ablation conservative.
    pub gate_initial_bias: f64,
    /// Optional bounded pairwise ligand geometry messages injected into the baseline path.
    pub pairwise_geometry: Option<PairwiseGeometryConfig>,
}

impl Default for AtomPocketCrossAttentionVelocityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hidden_dim: 128,
            gate_initial_bias: -1.0,
            pairwise_geometry: None,
        }
    }
}

/// Velocity head using `gate(atom,pocket) * Attention(Q_atom, K_pocket, V_pocket)`.
#[derive(Debug)]
pub struct AtomPocketCrossAttentionVelocityHead {
    baseline: GeometryFlowMatchingHead,
    atom_query: nn::Linear,
    pocket_key: nn::Linear,
    pocket_value: nn::Linear,
    gate_projection: nn::Linear,
    atom_projection: nn::Linear,
    fusion_projection: nn::Linear,
    velocity_delta: nn::Linear,
    config: AtomPocketCrossAttentionVelocityConfig,
}

/// Config-selectable flow velocity head used by the research system.
#[derive(Debug)]
pub enum FlowVelocityHead {
    /// Baseline pooled-conditioning geometry head.
    Geometry(Box<GeometryFlowMatchingHead>),
    /// EGNN-style equivariant geometry head.
    EquivariantGeometry(Box<EquivariantGeometryVelocityHead>),
    /// Gated atom-to-pocket cross-attention ablation head.
    AtomPocketCrossAttention(Box<AtomPocketCrossAttentionVelocityHead>),
}

impl FlowMatchingHead for FlowVelocityHead {
    fn predict_velocity(
        &self,
        state: &FlowState,
        conditioning: &ConditioningState,
    ) -> Result<VelocityField, ModelError> {
        match self {
            Self::Geometry(head) => head.predict_velocity(state, conditioning),
            Self::EquivariantGeometry(head) => head.predict_velocity(state, conditioning),
            Self::AtomPocketCrossAttention(head) => head.predict_velocity(state, conditioning),
        }
    }
}

impl AtomPocketCrossAttentionVelocityHead {
    /// Create a configurable cross-attention head while preserving the baseline head.
    pub fn new(vs: &nn::Path, config: AtomPocketCrossAttentionVelocityConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let mut gate_config = nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        gate_config.ws_init = nn::Init::Const(0.0);
        let mut head = Self {
            baseline: match config.pairwise_geometry.clone() {
                Some(pairwise) => GeometryFlowMatchingHead::new_with_pairwise(
                    &(vs / "baseline"),
                    hidden_dim,
                    pairwise,
                ),
                None => GeometryFlowMatchingHead::new(&(vs / "baseline"), hidden_dim),
            },
            atom_query: nn::linear(vs / "atom_query", 3, hidden_dim, Default::default()),
            pocket_key: nn::linear(
                vs / "pocket_key",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ),
            pocket_value: nn::linear(
                vs / "pocket_value",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ),
            gate_projection: nn::linear(vs / "gate_projection", hidden_dim * 2, 1, gate_config),
            atom_projection: nn::linear(vs / "atom_projection", 3, hidden_dim, Default::default()),
            fusion_projection: nn::linear(
                vs / "fusion_projection",
                hidden_dim * 2,
                hidden_dim,
                Default::default(),
            ),
            velocity_delta: nn::linear(vs / "velocity_delta", hidden_dim, 3, Default::default()),
            config,
        };
        if let Some(bias) = head.gate_projection.bs.as_mut() {
            no_grad(|| {
                let _ = bias.fill_(head.config.gate_initial_bias);
            });
        }
        head
    }
}

impl FlowMatchingHead for AtomPocketCrossAttentionVelocityHead {
    fn predict_velocity(
        &self,
        state: &FlowState,
        conditioning: &ConditioningState,
    ) -> Result<VelocityField, ModelError> {
        let baseline = self.baseline.predict_velocity(state, conditioning)?;
        if !self.config.enabled
            || state.coords.size()[0] == 0
            || conditioning.pocket_context.numel() == 0
        {
            return Ok(baseline);
        }
        if conditioning.pocket_context.size().len() != 2
            || conditioning.pocket_context.size()[1] != self.config.hidden_dim
        {
            return Err(ModelError::new(
                "pocket context must have shape [num_pocket_tokens, hidden_dim]",
            ));
        }

        let reference_centroid = state.x0_coords.mean_dim([0].as_slice(), false, Kind::Float);
        let centered_coords = &state.coords - reference_centroid.unsqueeze(0);
        let atom_hidden = centered_coords.apply(&self.atom_projection).relu();
        let query = centered_coords.apply(&self.atom_query);
        let key = conditioning.pocket_context.apply(&self.pocket_key);
        let value = conditioning.pocket_context.apply(&self.pocket_value);
        let scale = (self.config.hidden_dim as f64).sqrt();
        let attention = (query.matmul(&key.transpose(0, 1)) / scale).softmax(-1, Kind::Float);
        let attended = attention.matmul(&value);
        let gate = Tensor::cat(&[atom_hidden.shallow_clone(), attended.shallow_clone()], 1)
            .apply(&self.gate_projection)
            .sigmoid();
        let mut diagnostics = baseline.diagnostics;
        diagnostics.extend(attention_diagnostics(&gate, &attention));
        let gated_attention = gate * attended;
        let delta_hidden = Tensor::cat(&[atom_hidden, gated_attention], 1)
            .apply(&self.fusion_projection)
            .relu();
        Ok(VelocityField {
            velocity: baseline.velocity + delta_hidden.apply(&self.velocity_delta),
            diagnostics,
        })
    }
}

fn attention_diagnostics(gate: &Tensor, attention: &Tensor) -> BTreeMap<String, f64> {
    let mut diagnostics = BTreeMap::new();
    diagnostics.insert(
        "atom_pocket_gate_mean".to_string(),
        gate.mean(Kind::Float).double_value(&[]),
    );
    diagnostics.insert(
        "atom_pocket_gate_sparsity_lt_0_1".to_string(),
        gate.lt(0.1)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]),
    );
    let entropy = -(attention * (attention + 1e-12).log())
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
        .mean(Kind::Float)
        .double_value(&[]);
    diagnostics.insert("atom_pocket_attention_entropy".to_string(), entropy);
    let token_mean = attention.mean_dim([0].as_slice(), false, Kind::Float);
    let token_count = attention.size().get(1).copied().unwrap_or(1).max(1) as f64;
    let utilization_threshold = 0.5 / token_count;
    diagnostics.insert(
        "pocket_token_utilization".to_string(),
        token_mean
            .gt(utilization_threshold)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]),
    );
    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conditioning(hidden_dim: i64) -> ConditioningState {
        ConditioningState {
            topology_context: Tensor::ones([3, hidden_dim], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([3, hidden_dim], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([5, hidden_dim], (Kind::Float, tch::Device::Cpu)),
            gate_summary: crate::models::GenerationGateSummary::default(),
        }
    }

    #[test]
    fn cross_attention_velocity_head_is_configurable() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let config = AtomPocketCrossAttentionVelocityConfig {
            enabled: true,
            hidden_dim: 16,
            gate_initial_bias: -1.0,
            pairwise_geometry: None,
        };
        let head = AtomPocketCrossAttentionVelocityHead::new(&var_store.root(), config);
        let state = FlowState {
            coords: Tensor::zeros([4, 3], (Kind::Float, tch::Device::Cpu)),
            x0_coords: Tensor::ones([4, 3], (Kind::Float, tch::Device::Cpu)),
            target_coords: None,
            t: 0.5,
        };
        let velocity = head.predict_velocity(&state, &conditioning(16)).unwrap();
        assert_eq!(velocity.velocity.size(), vec![4, 3]);
        assert!(velocity.diagnostics.contains_key("atom_pocket_gate_mean"));
    }

    #[test]
    fn disabled_cross_attention_falls_back_to_baseline_shape() {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let head = AtomPocketCrossAttentionVelocityHead::new(
            &var_store.root(),
            AtomPocketCrossAttentionVelocityConfig {
                enabled: false,
                hidden_dim: 16,
                gate_initial_bias: -1.0,
                pairwise_geometry: None,
            },
        );
        let state = FlowState {
            coords: Tensor::zeros([2, 3], (Kind::Float, tch::Device::Cpu)),
            x0_coords: Tensor::ones([2, 3], (Kind::Float, tch::Device::Cpu)),
            target_coords: None,
            t: 0.1,
        };
        let velocity = head.predict_velocity(&state, &conditioning(16)).unwrap();
        assert_eq!(velocity.velocity.size(), vec![2, 3]);
        assert!(velocity
            .diagnostics
            .contains_key("conditioning_gate_input_mean"));
        assert!(!velocity.diagnostics.contains_key("atom_pocket_gate_mean"));
    }
}
