//! Gate sparsity regularization.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tch::{Kind, Tensor};

use crate::{
    config::InteractionPathGateRegularizationWeight,
    models::{
        interaction::InteractionPath, CrossAttentionOutput, CrossModalInteractions, ResearchForward,
    },
};

const DIRECTED_INTERACTION_PATH_COUNT: f64 = 6.0;

/// Per-path decomposition of the gate sparsity objective.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GatePathObjectiveContribution {
    /// Stable directed-path name.
    pub path_name: String,
    /// Stable chemistry-level directed-path role.
    pub path_role: String,
    /// Raw mean absolute gate value before path-specific weighting.
    pub gate_abs_mean: f64,
    /// Path-specific multiplier from model config.
    pub path_weight: f64,
    /// Fixed normalizer preserving the prior six-path average.
    pub normalized_path_factor: f64,
    /// Contribution to the unweighted `L_gate` value.
    pub objective_contribution: f64,
    /// Effective staged loss weight for `L_gate`.
    #[serde(default)]
    pub effective_loss_weight: f64,
    /// Contribution to the optimizer-facing total objective.
    #[serde(default)]
    pub optimizer_contribution: f64,
    /// Whether gate regularization was disabled for a forced-open negative control.
    #[serde(default)]
    pub forced_open: bool,
}

impl GatePathObjectiveContribution {
    pub(crate) fn with_effective_loss_weight(&self, effective_loss_weight: f64) -> Self {
        let mut updated = self.clone();
        updated.effective_loss_weight = effective_loss_weight;
        updated.optimizer_contribution = updated.objective_contribution * effective_loss_weight;
        updated
    }
}

/// Penalizes excessive cross-modality gate activation.
#[derive(Debug, Clone)]
pub struct GateLoss {
    path_weights: BTreeMap<String, f64>,
}

impl Default for GateLoss {
    fn default() -> Self {
        Self {
            path_weights: BTreeMap::new(),
        }
    }
}

impl GateLoss {
    /// Construct a gate loss with optional path-specific objective weights.
    pub(crate) fn with_path_weights(
        path_weights: Vec<InteractionPathGateRegularizationWeight>,
    ) -> Self {
        Self {
            path_weights: path_weights
                .into_iter()
                .map(|entry| (entry.path, entry.weight))
                .collect(),
        }
    }

    /// Compute the average gate magnitude across all directed interactions.
    pub(crate) fn compute(&self, interactions: &CrossModalInteractions) -> Tensor {
        if gate_objective_disabled(interactions) {
            return Tensor::zeros([1], (Kind::Float, interactions.topo_from_geo.gate.device()));
        }
        let mut total = Tensor::zeros([1], (Kind::Float, interactions.topo_from_geo.gate.device()));
        for (path, output) in interaction_outputs(interactions) {
            total += output.gate.abs().mean(Kind::Float) * self.path_weight(path);
        }
        total / DIRECTED_INTERACTION_PATH_COUNT
    }

    /// Compute the mean gate sparsity penalty over a mini-batch.
    pub(crate) fn compute_batch(&self, forwards: &[ResearchForward]) -> Tensor {
        let device = forwards
            .first()
            .map(|forward| forward.interactions.topo_from_geo.gate.device())
            .unwrap_or(tch::Device::Cpu);
        if forwards.is_empty() {
            return Tensor::zeros([1], (Kind::Float, device));
        }

        let mut total = Tensor::zeros([1], (Kind::Float, device));
        for forward in forwards {
            total += self.compute(&forward.interactions);
        }
        total / forwards.len() as f64
    }

    /// Return detached per-path contributions averaged over a mini-batch.
    pub(crate) fn path_objective_contributions_batch(
        &self,
        forwards: &[ResearchForward],
    ) -> Vec<GatePathObjectiveContribution> {
        if forwards.is_empty() {
            return Vec::new();
        }

        let mut totals = stable_path_contribution_templates(self);
        for forward in forwards {
            let per_example = self.path_objective_contributions(&forward.interactions);
            for (index, contribution) in per_example.iter().enumerate() {
                totals[index].gate_abs_mean += contribution.gate_abs_mean;
                totals[index].objective_contribution += contribution.objective_contribution;
                totals[index].forced_open |= contribution.forced_open;
            }
        }

        let count = forwards.len() as f64;
        for contribution in &mut totals {
            contribution.gate_abs_mean /= count;
            contribution.objective_contribution /= count;
        }
        totals
    }

    /// Return zero-valued records preserving stable path order and configured weights.
    pub(crate) fn zero_path_objective_contributions(&self) -> Vec<GatePathObjectiveContribution> {
        stable_path_contribution_templates(self)
    }

    fn path_objective_contributions(
        &self,
        interactions: &CrossModalInteractions,
    ) -> Vec<GatePathObjectiveContribution> {
        let disabled = gate_objective_disabled(interactions);
        interaction_outputs(interactions)
            .into_iter()
            .map(|(path, output)| {
                let gate_abs_mean = if disabled {
                    0.0
                } else {
                    output.gate.abs().mean(Kind::Float).double_value(&[])
                };
                let path_weight = self.path_weight(path);
                GatePathObjectiveContribution {
                    path_name: path.as_str().to_string(),
                    path_role: path.role().to_string(),
                    gate_abs_mean,
                    path_weight,
                    normalized_path_factor: 1.0 / DIRECTED_INTERACTION_PATH_COUNT,
                    objective_contribution: gate_abs_mean * path_weight
                        / DIRECTED_INTERACTION_PATH_COUNT,
                    effective_loss_weight: 0.0,
                    optimizer_contribution: 0.0,
                    forced_open: disabled || output.forced_open,
                }
            })
            .collect()
    }

    fn path_weight(&self, path: InteractionPath) -> f64 {
        self.path_weights.get(path.as_str()).copied().unwrap_or(1.0)
    }
}

fn stable_path_contribution_templates(loss: &GateLoss) -> Vec<GatePathObjectiveContribution> {
    stable_paths()
        .into_iter()
        .map(|path| GatePathObjectiveContribution {
            path_name: path.as_str().to_string(),
            path_role: path.role().to_string(),
            gate_abs_mean: 0.0,
            path_weight: loss.path_weight(path),
            normalized_path_factor: 1.0 / DIRECTED_INTERACTION_PATH_COUNT,
            objective_contribution: 0.0,
            effective_loss_weight: 0.0,
            optimizer_contribution: 0.0,
            forced_open: false,
        })
        .collect()
}

fn stable_paths() -> [InteractionPath; 6] {
    [
        InteractionPath::TopologyFromGeometry,
        InteractionPath::TopologyFromPocket,
        InteractionPath::GeometryFromTopology,
        InteractionPath::GeometryFromPocket,
        InteractionPath::PocketFromTopology,
        InteractionPath::PocketFromGeometry,
    ]
}

fn interaction_outputs(
    interactions: &CrossModalInteractions,
) -> [(InteractionPath, &CrossAttentionOutput); 6] {
    [
        (
            InteractionPath::TopologyFromGeometry,
            &interactions.topo_from_geo,
        ),
        (
            InteractionPath::TopologyFromPocket,
            &interactions.topo_from_pocket,
        ),
        (
            InteractionPath::GeometryFromTopology,
            &interactions.geo_from_topo,
        ),
        (
            InteractionPath::GeometryFromPocket,
            &interactions.geo_from_pocket,
        ),
        (
            InteractionPath::PocketFromTopology,
            &interactions.pocket_from_topo,
        ),
        (
            InteractionPath::PocketFromGeometry,
            &interactions.pocket_from_geo,
        ),
    ]
}

fn gate_objective_disabled(interactions: &CrossModalInteractions) -> bool {
    interaction_outputs(interactions)
        .iter()
        .any(|(_, output)| output.forced_open)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CrossAttentionOutput;
    use tch::{Device, Kind, Tensor};

    fn output(gate_value: f32, forced_open: bool) -> CrossAttentionOutput {
        CrossAttentionOutput {
            gate: Tensor::from_slice(&[gate_value]).to_device(Device::Cpu),
            forced_open,
            attended_tokens: Tensor::zeros([1, 1], (Kind::Float, Device::Cpu)),
            attention_weights: Tensor::zeros([1, 1], (Kind::Float, Device::Cpu)),
        }
    }

    fn output_with_gate(gate: Tensor) -> CrossAttentionOutput {
        CrossAttentionOutput {
            gate,
            forced_open: false,
            attended_tokens: Tensor::zeros([1, 1], (Kind::Float, Device::Cpu)),
            attention_weights: Tensor::zeros([1, 1], (Kind::Float, Device::Cpu)),
        }
    }

    #[test]
    fn gate_loss_is_disabled_for_direct_fusion_negative_control() {
        let interactions = CrossModalInteractions {
            topo_from_geo: output(1.0, true),
            topo_from_pocket: output(1.0, true),
            geo_from_topo: output(1.0, true),
            geo_from_pocket: output(1.0, true),
            pocket_from_topo: output(1.0, true),
            pocket_from_geo: output(1.0, true),
        };

        assert_eq!(
            GateLoss::default().compute(&interactions).double_value(&[]),
            0.0
        );
    }

    #[test]
    fn default_gate_loss_preserves_six_path_average_and_reports_contributions() {
        let interactions = CrossModalInteractions {
            topo_from_geo: output(0.0, false),
            topo_from_pocket: output(0.2, false),
            geo_from_topo: output(0.4, false),
            geo_from_pocket: output(0.6, false),
            pocket_from_topo: output(0.8, false),
            pocket_from_geo: output(1.0, false),
        };

        let loss = GateLoss::default();
        let value = loss.compute(&interactions).double_value(&[]);
        let contributions = loss.path_objective_contributions(&interactions);
        let contribution_sum = contributions
            .iter()
            .map(|contribution| contribution.objective_contribution)
            .sum::<f64>();

        assert!((value - 0.5).abs() < 1e-6);
        assert!((value - contribution_sum).abs() < 1e-6);
        assert_eq!(contributions.len(), 6);
        assert_eq!(contributions[3].path_name, "geo_from_pocket");
    }

    #[test]
    fn path_specific_gate_weights_scale_only_selected_contributions() {
        let interactions = CrossModalInteractions {
            topo_from_geo: output(0.0, false),
            topo_from_pocket: output(0.2, false),
            geo_from_topo: output(0.4, false),
            geo_from_pocket: output(0.6, false),
            pocket_from_topo: output(0.8, false),
            pocket_from_geo: output(1.0, false),
        };
        let loss = GateLoss::with_path_weights(vec![
            InteractionPathGateRegularizationWeight {
                path: "geo_from_pocket".to_string(),
                weight: 0.0,
            },
            InteractionPathGateRegularizationWeight {
                path: "pocket_from_geo".to_string(),
                weight: 2.0,
            },
        ]);

        let value = loss.compute(&interactions).double_value(&[]);
        let expected = (0.0 + 0.2 + 0.4 + 0.0 + 0.8 + 2.0) / 6.0;
        let contributions = loss.path_objective_contributions(&interactions);

        assert!((value - expected).abs() < 1e-6);
        assert_eq!(contributions[3].path_weight, 0.0);
        assert_eq!(contributions[5].path_weight, 2.0);
        assert_eq!(contributions[3].objective_contribution, 0.0);
    }

    #[test]
    fn gate_loss_regularizes_fine_grained_gate_tensors_by_mean_activation() {
        let interactions = CrossModalInteractions {
            topo_from_geo: output_with_gate(
                Tensor::from_slice(&[0.0_f32, 0.5, 1.0]).reshape([3, 1]),
            ),
            topo_from_pocket: output(0.0, false),
            geo_from_topo: output(0.0, false),
            geo_from_pocket: output(0.0, false),
            pocket_from_topo: output(0.0, false),
            pocket_from_geo: output(0.0, false),
        };

        let value = GateLoss::default().compute(&interactions).double_value(&[]);

        assert!((value - (0.5 / 6.0)).abs() < 1e-6);
    }
}
