//! Leakage control via off-modality similarity margins.
//!
//! Current implementation is a lightweight proxy: it penalizes cross-modality slot
//! cosine similarity above a dynamic margin. This is useful as an early warning
//! signal, but low similarity alone does not prove semantic non-leakage.
//! A stronger formulation should use explicit off-modality prediction probes and
//! penalize predictive power on wrong-modality targets.

use serde::{Deserialize, Serialize};
use tch::{Kind, Reduction, Tensor};

use crate::{
    config::types::{ExplicitLeakageProbeConfig, ExplicitLeakageProbeTrainingSemantics},
    config::PharmacophoreProbeConfig,
    data::{ChemistryRoleFeatureMatrix, MolecularExample},
    models::ResearchForward,
};

/// Role-separated leakage evidence persisted in training and evaluation artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakageEvidenceRoleReport {
    /// Schema version for the role report.
    #[serde(default = "default_leakage_evidence_role_schema_version")]
    pub schema_version: u32,
    /// Optimizer-facing leakage penalties used by the training objective.
    #[serde(default)]
    pub optimizer_penalty: LeakageOptimizerPenaltySection,
    /// Detached diagnostics computed from training-time forwards.
    #[serde(default)]
    pub detached_training_diagnostic: LeakageDetachedTrainingDiagnosticSection,
    /// Held-out frozen-probe audit, when available.
    #[serde(default)]
    pub frozen_probe_audit: LeakageFrozenProbeAuditSection,
    /// Claim boundary for interpreting these fields.
    #[serde(default)]
    pub claim_boundary: String,
}

impl Default for LeakageEvidenceRoleReport {
    fn default() -> Self {
        Self {
            schema_version: default_leakage_evidence_role_schema_version(),
            optimizer_penalty: LeakageOptimizerPenaltySection::default(),
            detached_training_diagnostic: LeakageDetachedTrainingDiagnosticSection::default(),
            frozen_probe_audit: LeakageFrozenProbeAuditSection::default(),
            claim_boundary:
                "training leakage penalties and detached diagnostics are not held-out no-leakage proof"
                    .to_string(),
        }
    }
}

impl LeakageEvidenceRoleReport {
    /// Build the training-step role contract from scalar leakage metrics.
    pub fn training_step(
        similarity_proxy_penalty: f64,
        explicit_probe_penalty: f64,
        pharmacophore_role_penalty: f64,
        explicit_probe_fit_loss: f64,
        explicit_encoder_penalty: f64,
        explicit_route_status: &str,
        similarity_proxy_diagnostic: f64,
        explicit_probe_diagnostic: f64,
        effective_delta_leak: f64,
        effective_pharmacophore_leakage: f64,
        leak_execution_mode: &str,
        pharmacophore_leakage_execution_mode: &str,
        detached_computation_when_disabled: bool,
    ) -> Self {
        Self {
            optimizer_penalty: LeakageOptimizerPenaltySection {
                active: effective_delta_leak > 0.0 || effective_pharmacophore_leakage > 0.0,
                similarity_proxy_penalty,
                explicit_probe_penalty,
                pharmacophore_role_penalty,
                explicit_probe_fit_loss,
                explicit_encoder_penalty,
                explicit_route_status: explicit_route_status.to_string(),
                effective_delta_leak,
                effective_pharmacophore_leakage,
                leak_execution_mode: leak_execution_mode.to_string(),
                pharmacophore_leakage_execution_mode: pharmacophore_leakage_execution_mode
                    .to_string(),
                interpretation: "optimizer-facing only when execution_mode=trainable and effective weights are positive".to_string(),
            },
            detached_training_diagnostic: LeakageDetachedTrainingDiagnosticSection {
                similarity_proxy_diagnostic,
                explicit_probe_diagnostic,
                computed_under_no_grad_when_disabled: detached_computation_when_disabled,
                interpretation: "diagnostic values are detached from the optimizer role and must not be read as held-out leakage estimates".to_string(),
            },
            ..Self::default()
        }
    }
}

fn default_leakage_evidence_role_schema_version() -> u32 {
    1
}

/// Optimizer-facing leakage penalty section.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakageOptimizerPenaltySection {
    /// Whether any leakage penalty can affect the current optimizer objective.
    #[serde(default)]
    pub active: bool,
    /// Similarity-proxy penalty value after training semantics are applied.
    #[serde(default)]
    pub similarity_proxy_penalty: f64,
    /// Explicit off-modality probe penalty value after training semantics are applied.
    #[serde(default)]
    pub explicit_probe_penalty: f64,
    /// Pharmacophore-role leakage penalty value after training semantics are applied.
    #[serde(default)]
    pub pharmacophore_role_penalty: f64,
    /// Probe-fitting objective value for explicit leakage probes.
    #[serde(default)]
    pub explicit_probe_fit_loss: f64,
    /// Encoder-facing penalty value for explicit leakage probes.
    #[serde(default)]
    pub explicit_encoder_penalty: f64,
    /// Route selected by the explicit leakage training semantics.
    #[serde(default)]
    pub explicit_route_status: String,
    /// Effective staged weight for core leakage.
    #[serde(default)]
    pub effective_delta_leak: f64,
    /// Effective staged weight for pharmacophore leakage.
    #[serde(default)]
    pub effective_pharmacophore_leakage: f64,
    /// Execution mode for core leakage.
    #[serde(default)]
    pub leak_execution_mode: String,
    /// Execution mode for pharmacophore leakage.
    #[serde(default)]
    pub pharmacophore_leakage_execution_mode: String,
    /// Interpretation note.
    #[serde(default)]
    pub interpretation: String,
}

impl Default for LeakageOptimizerPenaltySection {
    fn default() -> Self {
        Self {
            active: false,
            similarity_proxy_penalty: 0.0,
            explicit_probe_penalty: 0.0,
            pharmacophore_role_penalty: 0.0,
            explicit_probe_fit_loss: 0.0,
            explicit_encoder_penalty: 0.0,
            explicit_route_status: "not_run".to_string(),
            effective_delta_leak: 0.0,
            effective_pharmacophore_leakage: 0.0,
            leak_execution_mode: "not_run".to_string(),
            pharmacophore_leakage_execution_mode: "not_run".to_string(),
            interpretation: "no optimizer-facing leakage penalty recorded".to_string(),
        }
    }
}

/// Detached training-time leakage diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakageDetachedTrainingDiagnosticSection {
    /// Similarity-proxy diagnostic before training semantics are applied.
    #[serde(default)]
    pub similarity_proxy_diagnostic: f64,
    /// Explicit probe diagnostic before training semantics are applied.
    #[serde(default)]
    pub explicit_probe_diagnostic: f64,
    /// Whether disabled diagnostics were computed under no-grad semantics.
    #[serde(default)]
    pub computed_under_no_grad_when_disabled: bool,
    /// Interpretation note.
    #[serde(default)]
    pub interpretation: String,
}

impl Default for LeakageDetachedTrainingDiagnosticSection {
    fn default() -> Self {
        Self {
            similarity_proxy_diagnostic: 0.0,
            explicit_probe_diagnostic: 0.0,
            computed_under_no_grad_when_disabled: false,
            interpretation: "not_run".to_string(),
        }
    }
}

/// Held-out frozen-probe audit summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakageFrozenProbeAuditSection {
    /// Audit status: not_run, ok, or insufficient_data.
    #[serde(default)]
    pub status: String,
    /// Number of held-out routes reported.
    #[serde(default)]
    pub route_count: usize,
    /// Number of capacity-sweep rows reported.
    #[serde(default)]
    pub capacity_sweep_rows: usize,
    /// Best fractional improvement over a trivial target baseline.
    #[serde(default)]
    pub best_improvement_over_baseline: Option<f64>,
    /// Optional artifact path for the full frozen-probe sweep.
    #[serde(default)]
    pub artifact: Option<String>,
    /// Interpretation note.
    #[serde(default)]
    pub interpretation: String,
}

impl Default for LeakageFrozenProbeAuditSection {
    fn default() -> Self {
        Self {
            status: "not_run".to_string(),
            route_count: 0,
            capacity_sweep_rows: 0,
            best_improvement_over_baseline: None,
            artifact: None,
            interpretation:
                "frozen held-out leakage probe audit has not been run for this artifact".to_string(),
        }
    }
}

/// Decomposed leakage-control objective values.
pub(crate) struct LeakageLossTensors {
    /// Optimizer-facing similarity-proxy leakage objective.
    pub core: Tensor,
    /// Detached similarity-proxy diagnostic value before training semantics are applied.
    pub similarity_proxy_diagnostic: Tensor,
    /// Detached explicit leakage-probe diagnostic value before training semantics are applied.
    pub explicit_probe_diagnostic: Tensor,
    /// Explicit probe-fitting loss routed only into leakage probe heads.
    pub explicit_probe_fit_loss: Tensor,
    /// Explicit encoder penalty routed only into source encoders for frozen-probe modes.
    pub explicit_encoder_penalty: Tensor,
    /// Human-readable route selected for this leakage objective.
    pub route_status: String,
    /// Optimizer-facing total leakage objective.
    pub total: Tensor,
    /// Explicit topology->geometry leakage penalty.
    pub topology_to_geometry: Tensor,
    /// Explicit geometry->topology leakage penalty.
    pub geometry_to_topology: Tensor,
    /// Explicit pocket->geometry leakage penalty.
    pub pocket_to_geometry: Tensor,
    /// Explicit topology-to-pocket-role leakage penalty.
    pub topology_to_pocket_role: Tensor,
    /// Explicit geometry-to-pocket-role leakage penalty.
    pub geometry_to_pocket_role: Tensor,
    /// Explicit pocket-to-topology-role leakage penalty.
    pub pocket_to_topology_role: Tensor,
    /// Explicit pocket-to-ligand-role leakage penalty.
    pub pocket_to_ligand_role: Tensor,
}

/// Penalizes wrong-modality representations that become too predictive of off-modality targets.
#[derive(Debug, Clone)]
pub struct LeakageLoss {
    /// Margin under which off-modality prediction is considered excessive.
    pub margin: f64,
    /// Optional pharmacophore role leakage controls.
    pub pharmacophore: PharmacophoreProbeConfig,
    /// Explicit probe-route selection and gating.
    pub explicit: ExplicitLeakageProbeConfig,
}

impl Default for LeakageLoss {
    fn default() -> Self {
        Self {
            margin: 0.25,
            pharmacophore: PharmacophoreProbeConfig::default(),
            explicit: ExplicitLeakageProbeConfig::default(),
        }
    }
}

impl LeakageLoss {
    /// Create leakage loss wiring from training config.
    pub fn new(
        pharmacophore: PharmacophoreProbeConfig,
        explicit: ExplicitLeakageProbeConfig,
    ) -> Self {
        Self {
            pharmacophore,
            explicit,
            ..Self::default()
        }
    }

    /// Compute leakage with explicit route-level pharmacophore diagnostics.
    #[allow(dead_code)] // Compatibility wrapper for callers without trainer-step routing.
    pub(crate) fn compute_with_routes(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> LeakageLossTensors {
        self.compute_with_routes_for_step(example, forward, None)
    }

    /// Compute leakage for a concrete training step, allowing alternating semantics.
    pub(crate) fn compute_with_routes_for_step(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
        training_step: Option<usize>,
    ) -> LeakageLossTensors {
        let (topo_geo_similarity, topo_pocket_similarity, geo_pocket_similarity) =
            pairwise_slot_similarity_tensors(forward);
        let device = forward.slots.topology.slots.device();
        let leakage_budget = leakage_budget_tensor(example, self.margin, device);
        let explicit_enabled = self.explicit.enable_explicit_probes;
        let route_status =
            leakage_route_status(self.explicit.training_semantics, training_step).to_string();

        let base = (topo_geo_similarity - &leakage_budget).relu()
            + (topo_pocket_similarity - &leakage_budget).relu()
            + (geo_pocket_similarity - &leakage_budget).relu();
        let core_route = core_leakage_route(base, self.explicit.training_semantics, training_step);
        let topology_to_geometry = if explicit_enabled && self.explicit.topology_to_geometry_probe {
            scalar_leakage_route(
                &forward.probes.topology_to_geometry_scalar_logits,
                &forward
                    .probes
                    .leakage_probe_fit
                    .topology_to_geometry_scalar_logits,
                &forward
                    .probes
                    .leakage_encoder_penalty
                    .topology_to_geometry_scalar_logits,
                geometry_scalar_target(example),
                self.pharmacophore.leakage_margin,
                self.explicit.training_semantics,
                training_step,
            )
        } else {
            LeakageTrainingRoute::zero(device)
        };
        let geometry_to_topology = if explicit_enabled && self.explicit.geometry_to_topology_probe {
            scalar_leakage_route(
                &forward.probes.geometry_to_topology_scalar_logits,
                &forward
                    .probes
                    .leakage_probe_fit
                    .geometry_to_topology_scalar_logits,
                &forward
                    .probes
                    .leakage_encoder_penalty
                    .geometry_to_topology_scalar_logits,
                topology_scalar_target(example),
                self.pharmacophore.leakage_margin,
                self.explicit.training_semantics,
                training_step,
            )
        } else {
            LeakageTrainingRoute::zero(device)
        };
        let pocket_to_geometry = if explicit_enabled && self.explicit.pocket_to_geometry_probe {
            scalar_leakage_route(
                &forward.probes.pocket_to_geometry_scalar_logits,
                &forward
                    .probes
                    .leakage_probe_fit
                    .pocket_to_geometry_scalar_logits,
                &forward
                    .probes
                    .leakage_encoder_penalty
                    .pocket_to_geometry_scalar_logits,
                geometry_scalar_target(example),
                self.pharmacophore.leakage_margin,
                self.explicit.training_semantics,
                training_step,
            )
        } else {
            LeakageTrainingRoute::zero(device)
        };
        let enable_topology_to_pocket_role = (explicit_enabled
            && self.explicit.topology_to_pocket_probe)
            || self.pharmacophore.enable_topology_to_pocket_role_leakage;
        let topology_to_pocket_role = if enable_topology_to_pocket_role {
            role_leakage_route(
                &forward.probes.topology_to_pocket_role_logits,
                &forward
                    .probes
                    .leakage_probe_fit
                    .topology_to_pocket_role_logits,
                &forward
                    .probes
                    .leakage_encoder_penalty
                    .topology_to_pocket_role_logits,
                &example.pocket.chemistry_roles,
                self.pharmacophore.leakage_margin,
                self.explicit.training_semantics,
                training_step,
            )
        } else {
            LeakageTrainingRoute::zero(device)
        };
        let enable_geometry_to_pocket_role = (explicit_enabled
            && self.explicit.geometry_to_pocket_probe)
            || self.pharmacophore.enable_geometry_to_pocket_role_leakage;
        let geometry_to_pocket_role = if enable_geometry_to_pocket_role {
            role_leakage_route(
                &forward.probes.geometry_to_pocket_role_logits,
                &forward
                    .probes
                    .leakage_probe_fit
                    .geometry_to_pocket_role_logits,
                &forward
                    .probes
                    .leakage_encoder_penalty
                    .geometry_to_pocket_role_logits,
                &example.pocket.chemistry_roles,
                self.pharmacophore.leakage_margin,
                self.explicit.training_semantics,
                training_step,
            )
        } else {
            LeakageTrainingRoute::zero(device)
        };
        let pocket_to_ligand_role = if self.pharmacophore.enable_pocket_to_ligand_role_leakage {
            role_leakage_route(
                &forward.probes.pocket_to_ligand_role_logits,
                &forward
                    .probes
                    .leakage_probe_fit
                    .pocket_to_ligand_role_logits,
                &forward
                    .probes
                    .leakage_encoder_penalty
                    .pocket_to_ligand_role_logits,
                &example.topology.chemistry_roles,
                self.pharmacophore.leakage_margin,
                self.explicit.training_semantics,
                training_step,
            )
        } else {
            LeakageTrainingRoute::zero(device)
        };
        let pocket_to_topology_role = if explicit_enabled && self.explicit.pocket_to_topology_probe
        {
            role_leakage_route(
                &forward.probes.pocket_to_topology_role_logits,
                &forward
                    .probes
                    .leakage_probe_fit
                    .pocket_to_topology_role_logits,
                &forward
                    .probes
                    .leakage_encoder_penalty
                    .pocket_to_topology_role_logits,
                &example.topology.chemistry_roles,
                self.pharmacophore.leakage_margin,
                self.explicit.training_semantics,
                training_step,
            )
        } else {
            LeakageTrainingRoute::zero(device)
        };
        let explicit_probe_diagnostic = topology_to_geometry.diagnostic.shallow_clone()
            + geometry_to_topology.diagnostic.shallow_clone()
            + pocket_to_geometry.diagnostic.shallow_clone()
            + pocket_to_topology_role.diagnostic.shallow_clone()
            + topology_to_pocket_role.diagnostic.shallow_clone()
            + geometry_to_pocket_role.diagnostic.shallow_clone()
            + pocket_to_ligand_role.diagnostic.shallow_clone();
        let explicit_probe_fit_loss = core_route.probe_fit_loss.shallow_clone()
            + topology_to_geometry.probe_fit_loss.shallow_clone()
            + geometry_to_topology.probe_fit_loss.shallow_clone()
            + pocket_to_geometry.probe_fit_loss.shallow_clone()
            + pocket_to_topology_role.probe_fit_loss.shallow_clone()
            + topology_to_pocket_role.probe_fit_loss.shallow_clone()
            + geometry_to_pocket_role.probe_fit_loss.shallow_clone()
            + pocket_to_ligand_role.probe_fit_loss.shallow_clone();
        let explicit_encoder_penalty = core_route.encoder_penalty.shallow_clone()
            + topology_to_geometry.encoder_penalty.shallow_clone()
            + geometry_to_topology.encoder_penalty.shallow_clone()
            + pocket_to_geometry.encoder_penalty.shallow_clone()
            + pocket_to_topology_role.encoder_penalty.shallow_clone()
            + topology_to_pocket_role.encoder_penalty.shallow_clone()
            + geometry_to_pocket_role.encoder_penalty.shallow_clone()
            + pocket_to_ligand_role.encoder_penalty.shallow_clone();
        let core = core_route.objective.shallow_clone();
        let total = core.shallow_clone()
            + topology_to_geometry.objective.shallow_clone()
            + geometry_to_topology.objective.shallow_clone()
            + pocket_to_geometry.objective.shallow_clone()
            + pocket_to_topology_role.objective.shallow_clone()
            + topology_to_pocket_role.objective.shallow_clone()
            + geometry_to_pocket_role.objective.shallow_clone()
            + pocket_to_ligand_role.objective.shallow_clone();

        LeakageLossTensors {
            core,
            similarity_proxy_diagnostic: core_route.diagnostic,
            explicit_probe_diagnostic,
            explicit_probe_fit_loss,
            explicit_encoder_penalty,
            route_status,
            total,
            topology_to_geometry: topology_to_geometry.objective,
            geometry_to_topology: geometry_to_topology.objective,
            pocket_to_geometry: pocket_to_geometry.objective,
            topology_to_pocket_role: topology_to_pocket_role.objective,
            geometry_to_pocket_role: geometry_to_pocket_role.objective,
            pocket_to_topology_role: pocket_to_topology_role.objective,
            pocket_to_ligand_role: pocket_to_ligand_role.objective,
        }
    }

    /// Compute mean leakage objective and explicit route diagnostics over a mini-batch.
    #[allow(dead_code)] // Compatibility wrapper for callers without trainer-step routing.
    pub(crate) fn compute_batch_with_routes(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> LeakageLossTensors {
        self.compute_batch_with_routes_for_step(examples, forwards, None)
    }

    /// Compute mean leakage objective and explicit route diagnostics for a trainer step.
    pub(crate) fn compute_batch_with_routes_for_step(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
        training_step: Option<usize>,
    ) -> LeakageLossTensors {
        debug_assert_eq!(examples.len(), forwards.len());
        let device = forwards
            .first()
            .map(|forward| forward.slots.topology.slots.device())
            .or_else(|| {
                examples
                    .first()
                    .map(|example| example.topology.atom_types.device())
            })
            .unwrap_or(tch::Device::Cpu);
        let route_status =
            leakage_route_status(self.explicit.training_semantics, training_step).to_string();
        if examples.is_empty() {
            let zero = Tensor::zeros([1], (Kind::Float, device));
            return LeakageLossTensors {
                core: zero.shallow_clone(),
                similarity_proxy_diagnostic: zero.shallow_clone(),
                explicit_probe_diagnostic: zero.shallow_clone(),
                explicit_probe_fit_loss: zero.shallow_clone(),
                explicit_encoder_penalty: zero.shallow_clone(),
                route_status,
                total: zero.shallow_clone(),
                topology_to_geometry: zero.shallow_clone(),
                geometry_to_topology: zero.shallow_clone(),
                pocket_to_geometry: zero.shallow_clone(),
                topology_to_pocket_role: zero.shallow_clone(),
                geometry_to_pocket_role: zero.shallow_clone(),
                pocket_to_topology_role: zero.shallow_clone(),
                pocket_to_ligand_role: zero,
            };
        }

        let mut total = Tensor::zeros([1], (Kind::Float, device));
        let mut core = Tensor::zeros([1], (Kind::Float, device));
        let mut similarity_proxy_diagnostic = Tensor::zeros([1], (Kind::Float, device));
        let mut explicit_probe_diagnostic = Tensor::zeros([1], (Kind::Float, device));
        let mut explicit_probe_fit_loss = Tensor::zeros([1], (Kind::Float, device));
        let mut explicit_encoder_penalty = Tensor::zeros([1], (Kind::Float, device));
        let mut topology_to_geometry = Tensor::zeros([1], (Kind::Float, device));
        let mut geometry_to_topology = Tensor::zeros([1], (Kind::Float, device));
        let mut pocket_to_geometry = Tensor::zeros([1], (Kind::Float, device));
        let mut topology_to_pocket_role = Tensor::zeros([1], (Kind::Float, device));
        let mut geometry_to_pocket_role = Tensor::zeros([1], (Kind::Float, device));
        let mut pocket_to_topology_role = Tensor::zeros([1], (Kind::Float, device));
        let mut pocket_to_ligand_role = Tensor::zeros([1], (Kind::Float, device));
        let mut sum_topo_geo = 0.0;
        let mut sum_topo_pocket = 0.0;
        let mut sum_geo_pocket = 0.0;
        let mut sum_budget = 0.0;
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let (topo_geo_similarity, topo_pocket_similarity, geo_pocket_similarity) =
                pairwise_slot_similarity_diagnostics(forward);
            let budget = leakage_budget(example, self.margin);
            sum_topo_geo += topo_geo_similarity;
            sum_topo_pocket += topo_pocket_similarity;
            sum_geo_pocket += geo_pocket_similarity;
            sum_budget += budget;
            let components = self.compute_with_routes_for_step(example, forward, training_step);
            core += components.core.to_device(device);
            similarity_proxy_diagnostic += components.similarity_proxy_diagnostic.to_device(device);
            explicit_probe_diagnostic += components.explicit_probe_diagnostic.to_device(device);
            explicit_probe_fit_loss += components.explicit_probe_fit_loss.to_device(device);
            explicit_encoder_penalty += components.explicit_encoder_penalty.to_device(device);
            total += components.total.to_device(device);
            topology_to_geometry += components.topology_to_geometry.to_device(device);
            geometry_to_topology += components.geometry_to_topology.to_device(device);
            pocket_to_geometry += components.pocket_to_geometry.to_device(device);
            topology_to_pocket_role += components.topology_to_pocket_role.to_device(device);
            geometry_to_pocket_role += components.geometry_to_pocket_role.to_device(device);
            pocket_to_topology_role += components.pocket_to_topology_role.to_device(device);
            pocket_to_ligand_role += components.pocket_to_ligand_role.to_device(device);
        }
        let denom = examples.len() as f64;
        log::debug!(
            "leakage diagnostics batch_mean topo_geo={:.4} topo_pocket={:.4} geo_pocket={:.4} budget={:.4}",
            sum_topo_geo / denom,
            sum_topo_pocket / denom,
            sum_geo_pocket / denom,
            sum_budget / denom,
        );
        LeakageLossTensors {
            core: core / denom,
            similarity_proxy_diagnostic: similarity_proxy_diagnostic / denom,
            explicit_probe_diagnostic: explicit_probe_diagnostic / denom,
            explicit_probe_fit_loss: explicit_probe_fit_loss / denom,
            explicit_encoder_penalty: explicit_encoder_penalty / denom,
            route_status,
            total: total / denom,
            topology_to_geometry: topology_to_geometry / denom,
            geometry_to_topology: geometry_to_topology / denom,
            pocket_to_geometry: pocket_to_geometry / denom,
            topology_to_pocket_role: topology_to_pocket_role / denom,
            geometry_to_pocket_role: geometry_to_pocket_role / denom,
            pocket_to_topology_role: pocket_to_topology_role / denom,
            pocket_to_ligand_role: pocket_to_ligand_role / denom,
        }
    }
}

fn explicit_scalar_leakage_penalty(pred: &Tensor, target: f64, tolerance: f64) -> Tensor {
    if pred.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, pred.device()));
    }
    let pred = pred.shallow_clone().mean(Kind::Float);
    let target = Tensor::from(target)
        .to_kind(Kind::Float)
        .to_device(pred.device());
    let error = (pred - target).pow_tensor_scalar(2.0);
    (Tensor::from(tolerance).to_kind(Kind::Float) - error).relu()
}

fn explicit_scalar_prediction_loss(pred: &Tensor, target: f64) -> Tensor {
    if pred.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, pred.device()));
    }
    let pred = pred.shallow_clone().mean(Kind::Float);
    let target = Tensor::from(target)
        .to_kind(Kind::Float)
        .to_device(pred.device());
    (pred - target).pow_tensor_scalar(2.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectiveLeakageRouteSemantics {
    DetachedDiagnostic,
    ProbeFit,
    EncoderPenalty,
    AdversarialPenalty,
}

struct LeakageTrainingRoute {
    objective: Tensor,
    diagnostic: Tensor,
    probe_fit_loss: Tensor,
    encoder_penalty: Tensor,
}

impl LeakageTrainingRoute {
    fn zero(device: tch::Device) -> Self {
        let zero = Tensor::zeros([1], (Kind::Float, device));
        Self {
            objective: zero.shallow_clone(),
            diagnostic: zero.shallow_clone(),
            probe_fit_loss: zero.shallow_clone(),
            encoder_penalty: zero,
        }
    }
}

fn core_leakage_route(
    penalty: Tensor,
    semantics: ExplicitLeakageProbeTrainingSemantics,
    training_step: Option<usize>,
) -> LeakageTrainingRoute {
    let diagnostic = penalty.detach();
    let zero = Tensor::zeros_like(&diagnostic);
    match effective_leakage_route_semantics(semantics, training_step) {
        EffectiveLeakageRouteSemantics::DetachedDiagnostic
        | EffectiveLeakageRouteSemantics::ProbeFit => LeakageTrainingRoute {
            objective: zero.shallow_clone(),
            diagnostic,
            probe_fit_loss: zero.shallow_clone(),
            encoder_penalty: zero,
        },
        EffectiveLeakageRouteSemantics::EncoderPenalty
        | EffectiveLeakageRouteSemantics::AdversarialPenalty => LeakageTrainingRoute {
            objective: penalty.shallow_clone(),
            diagnostic,
            probe_fit_loss: zero,
            encoder_penalty: penalty,
        },
    }
}

fn scalar_leakage_route(
    diagnostic_pred: &Tensor,
    probe_fit_pred: &Tensor,
    encoder_penalty_pred: &Tensor,
    target: f64,
    tolerance: f64,
    semantics: ExplicitLeakageProbeTrainingSemantics,
    training_step: Option<usize>,
) -> LeakageTrainingRoute {
    let diagnostic_penalty = explicit_scalar_leakage_penalty(diagnostic_pred, target, tolerance);
    let diagnostic = diagnostic_penalty.detach();
    let zero = Tensor::zeros_like(&diagnostic);
    match effective_leakage_route_semantics(semantics, training_step) {
        EffectiveLeakageRouteSemantics::DetachedDiagnostic => LeakageTrainingRoute {
            objective: zero.shallow_clone(),
            diagnostic,
            probe_fit_loss: zero.shallow_clone(),
            encoder_penalty: zero,
        },
        EffectiveLeakageRouteSemantics::ProbeFit => {
            let probe_fit = explicit_scalar_prediction_loss(probe_fit_pred, target);
            LeakageTrainingRoute {
                objective: probe_fit.shallow_clone(),
                diagnostic,
                probe_fit_loss: probe_fit,
                encoder_penalty: zero,
            }
        }
        EffectiveLeakageRouteSemantics::EncoderPenalty => {
            let encoder_penalty =
                explicit_scalar_leakage_penalty(encoder_penalty_pred, target, tolerance);
            LeakageTrainingRoute {
                objective: encoder_penalty.shallow_clone(),
                diagnostic,
                probe_fit_loss: zero,
                encoder_penalty,
            }
        }
        EffectiveLeakageRouteSemantics::AdversarialPenalty => LeakageTrainingRoute {
            objective: diagnostic_penalty.shallow_clone(),
            diagnostic,
            probe_fit_loss: zero,
            encoder_penalty: diagnostic_penalty,
        },
    }
}

fn role_leakage_route(
    diagnostic_logits: &Tensor,
    probe_fit_logits: &Tensor,
    encoder_penalty_logits: &Tensor,
    target_roles: &ChemistryRoleFeatureMatrix,
    tolerated_bce: f64,
    semantics: ExplicitLeakageProbeTrainingSemantics,
    training_step: Option<usize>,
) -> LeakageTrainingRoute {
    let diagnostic_penalty =
        pooled_role_leakage_penalty(diagnostic_logits, target_roles, tolerated_bce);
    let diagnostic = diagnostic_penalty.detach();
    let zero = Tensor::zeros_like(&diagnostic);
    match effective_leakage_route_semantics(semantics, training_step) {
        EffectiveLeakageRouteSemantics::DetachedDiagnostic => LeakageTrainingRoute {
            objective: zero.shallow_clone(),
            diagnostic,
            probe_fit_loss: zero.shallow_clone(),
            encoder_penalty: zero,
        },
        EffectiveLeakageRouteSemantics::ProbeFit => {
            let probe_fit = pooled_role_prediction_loss(probe_fit_logits, target_roles);
            LeakageTrainingRoute {
                objective: probe_fit.shallow_clone(),
                diagnostic,
                probe_fit_loss: probe_fit,
                encoder_penalty: zero,
            }
        }
        EffectiveLeakageRouteSemantics::EncoderPenalty => {
            let encoder_penalty =
                pooled_role_leakage_penalty(encoder_penalty_logits, target_roles, tolerated_bce);
            LeakageTrainingRoute {
                objective: encoder_penalty.shallow_clone(),
                diagnostic,
                probe_fit_loss: zero,
                encoder_penalty,
            }
        }
        EffectiveLeakageRouteSemantics::AdversarialPenalty => LeakageTrainingRoute {
            objective: diagnostic_penalty.shallow_clone(),
            diagnostic,
            probe_fit_loss: zero,
            encoder_penalty: diagnostic_penalty,
        },
    }
}

fn effective_leakage_route_semantics(
    semantics: ExplicitLeakageProbeTrainingSemantics,
    training_step: Option<usize>,
) -> EffectiveLeakageRouteSemantics {
    match semantics {
        ExplicitLeakageProbeTrainingSemantics::DetachedDiagnostic => {
            EffectiveLeakageRouteSemantics::DetachedDiagnostic
        }
        ExplicitLeakageProbeTrainingSemantics::ProbeFit => EffectiveLeakageRouteSemantics::ProbeFit,
        ExplicitLeakageProbeTrainingSemantics::EncoderPenalty => {
            EffectiveLeakageRouteSemantics::EncoderPenalty
        }
        ExplicitLeakageProbeTrainingSemantics::Alternating => {
            if training_step.unwrap_or(0) % 2 == 0 {
                EffectiveLeakageRouteSemantics::ProbeFit
            } else {
                EffectiveLeakageRouteSemantics::EncoderPenalty
            }
        }
        ExplicitLeakageProbeTrainingSemantics::AdversarialPenalty => {
            EffectiveLeakageRouteSemantics::AdversarialPenalty
        }
    }
}

fn leakage_route_status(
    semantics: ExplicitLeakageProbeTrainingSemantics,
    training_step: Option<usize>,
) -> &'static str {
    match semantics {
        ExplicitLeakageProbeTrainingSemantics::DetachedDiagnostic => "detached_diagnostic",
        ExplicitLeakageProbeTrainingSemantics::ProbeFit => "probe_fit",
        ExplicitLeakageProbeTrainingSemantics::EncoderPenalty => "encoder_penalty",
        ExplicitLeakageProbeTrainingSemantics::AdversarialPenalty => "adversarial_penalty",
        ExplicitLeakageProbeTrainingSemantics::Alternating => {
            if effective_leakage_route_semantics(semantics, training_step)
                == EffectiveLeakageRouteSemantics::ProbeFit
            {
                "alternating_probe_fit"
            } else {
                "alternating_encoder_penalty"
            }
        }
    }
}

fn geometry_scalar_target(example: &MolecularExample) -> f64 {
    if example.geometry.pairwise_distances.numel() == 0 {
        0.0
    } else {
        example
            .geometry
            .pairwise_distances
            .mean(Kind::Float)
            .double_value(&[])
    }
}

fn topology_scalar_target(example: &MolecularExample) -> f64 {
    if example.topology.adjacency.numel() == 0 {
        0.0
    } else {
        example
            .topology
            .adjacency
            .mean(Kind::Float)
            .double_value(&[])
    }
}

fn pooled_role_leakage_penalty(
    logits: &Tensor,
    target_roles: &ChemistryRoleFeatureMatrix,
    tolerated_bce: f64,
) -> Tensor {
    let device = logits.device();
    if logits.numel() == 0
        || target_roles.role_vectors.numel() == 0
        || target_roles.availability.numel() == 0
    {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let (target_profile, available) = available_role_profile(target_roles, device);
    if !available {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let cols = logits
        .size()
        .last()
        .copied()
        .unwrap_or(0)
        .min(target_profile.size().first().copied().unwrap_or(0));
    if cols <= 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let logits = logits.flatten(0, -1).narrow(0, 0, cols);
    let target = target_profile.narrow(0, 0, cols);
    let bce =
        logits.binary_cross_entropy_with_logits::<Tensor>(&target, None, None, Reduction::Mean);
    (Tensor::from(tolerated_bce)
        .to_kind(Kind::Float)
        .to_device(device)
        - bce)
        .relu()
}

fn pooled_role_prediction_loss(
    logits: &Tensor,
    target_roles: &ChemistryRoleFeatureMatrix,
) -> Tensor {
    let device = logits.device();
    if logits.numel() == 0
        || target_roles.role_vectors.numel() == 0
        || target_roles.availability.numel() == 0
    {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let (target_profile, available) = available_role_profile(target_roles, device);
    if !available {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let cols = logits
        .size()
        .last()
        .copied()
        .unwrap_or(0)
        .min(target_profile.size().first().copied().unwrap_or(0));
    if cols <= 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let logits = logits.flatten(0, -1).narrow(0, 0, cols);
    let target = target_profile.narrow(0, 0, cols);
    logits.binary_cross_entropy_with_logits::<Tensor>(&target, None, None, Reduction::Mean)
}

fn available_role_profile(
    target_roles: &ChemistryRoleFeatureMatrix,
    device: tch::Device,
) -> (Tensor, bool) {
    let role_vectors = target_roles.role_vectors.to_device(device);
    let availability = target_roles
        .availability
        .to_device(device)
        .to_kind(Kind::Float);
    let available = availability.sum(Kind::Float).double_value(&[]);
    if available <= 0.0 {
        let cols = role_vectors.size().get(1).copied().unwrap_or(0).max(1);
        return (Tensor::zeros([cols], (Kind::Float, device)), false);
    }
    let mask = availability.unsqueeze(-1);
    let profile =
        (role_vectors * &mask).sum_dim_intlist([0].as_slice(), false, Kind::Float) / available;
    (profile, true)
}

fn pairwise_slot_similarity_tensors(forward: &ResearchForward) -> (Tensor, Tensor, Tensor) {
    let topo_slots = &forward.slots.topology.slots;
    let geo_slots = &forward.slots.geometry.slots;
    let pocket_slots = &forward.slots.pocket.slots;
    (
        mean_cosine_similarity_tensor(topo_slots, geo_slots),
        mean_cosine_similarity_tensor(topo_slots, pocket_slots),
        mean_cosine_similarity_tensor(geo_slots, pocket_slots),
    )
}

fn pairwise_slot_similarity_diagnostics(forward: &ResearchForward) -> (f64, f64, f64) {
    let topo_slots = &forward.slots.topology.slots;
    let geo_slots = &forward.slots.geometry.slots;
    let pocket_slots = &forward.slots.pocket.slots;
    (
        mean_cosine_similarity_diagnostic(topo_slots, geo_slots),
        mean_cosine_similarity_diagnostic(topo_slots, pocket_slots),
        mean_cosine_similarity_diagnostic(geo_slots, pocket_slots),
    )
}

fn leakage_budget_tensor(example: &MolecularExample, margin: f64, device: tch::Device) -> Tensor {
    let margin = Tensor::from(margin).to_kind(Kind::Float).to_device(device);
    let adjacency_density = if example.topology.adjacency.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, device))
    } else {
        example
            .topology
            .adjacency
            .to_device(device)
            .mean(Kind::Float)
            .reshape([1])
    };
    let pocket_energy = if example.pocket.atom_features.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, device))
    } else {
        example
            .pocket
            .atom_features
            .to_device(device)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
            .reshape([1])
    };
    (margin + adjacency_density * 0.05 + pocket_energy * 0.01).reshape([1])
}

fn leakage_budget(example: &MolecularExample, margin: f64) -> f64 {
    let adjacency_density = if example.topology.adjacency.numel() == 0 {
        0.0
    } else {
        example
            .topology
            .adjacency
            .mean(Kind::Float)
            .double_value(&[])
    };
    let pocket_energy = if example.pocket.atom_features.numel() == 0 {
        0.0
    } else {
        example
            .pocket
            .atom_features
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
            .double_value(&[])
    };
    margin + 0.05 * adjacency_density + 0.01 * pocket_energy
}

fn mean_cosine_similarity_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    let device = a.device();
    if a.numel() == 0 || b.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let a_mean = a.mean_dim([0].as_slice(), false, Kind::Float);
    let b_mean = b.mean_dim([0].as_slice(), false, Kind::Float);
    let dot = (&a_mean * &b_mean).sum(Kind::Float);
    let a_norm = a_mean.pow_tensor_scalar(2.0).sum(Kind::Float).sqrt();
    let b_norm = b_mean.pow_tensor_scalar(2.0).sum(Kind::Float).sqrt();
    (dot / (&a_norm * &b_norm).clamp_min(1e-6)).reshape([1])
}

fn mean_cosine_similarity_diagnostic(a: &Tensor, b: &Tensor) -> f64 {
    if a.numel() == 0 || b.numel() == 0 {
        return 0.0;
    }
    let a_mean = a.mean_dim([0].as_slice(), false, Kind::Float);
    let b_mean = b.mean_dim([0].as_slice(), false, Kind::Float);
    let dot = (&a_mean * &b_mean).sum(Kind::Float).double_value(&[]);
    let a_norm = a_mean
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt()
        .double_value(&[]);
    let b_norm = b_mean
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt()
        .double_value(&[]);
    dot / ((a_norm * b_norm).max(1e-6))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Tensor};

    use crate::{
        config::{
            types::{ExplicitLeakageProbeConfig, ExplicitLeakageProbeTrainingSemantics},
            PharmacophoreProbeConfig, ResearchConfig,
        },
        data::{synthetic_phase1_examples, InMemoryDataset},
        models::Phase1ResearchSystem,
    };

    #[test]
    fn tensor_relu_positive_input() {
        let result = Tensor::from(0.5_f32).relu();
        assert_eq!(result.double_value(&[]), 0.5);
    }

    #[test]
    fn tensor_relu_negative_input() {
        let result = Tensor::from(-0.3_f32).relu();
        assert_eq!(result.double_value(&[]), 0.0);
    }

    #[test]
    fn mean_cosine_similarity_tensor_identical_vectors_requires_grad() {
        let ones = Tensor::ones(&[10], (Kind::Float, Device::Cpu));
        let left = ones.shallow_clone().set_requires_grad(true);
        let sim = mean_cosine_similarity_tensor(&left, &ones);
        assert!(
            (sim.double_value(&[]) - 1.0).abs() < 1e-5,
            "Identical vectors should have cosine similarity ~1.0, got {}",
            sim.double_value(&[])
        );
        assert!(sim.requires_grad());
    }

    #[test]
    fn mean_cosine_similarity_diagnostic_random_vectors() {
        let x = Tensor::randn(&[100], (Kind::Float, Device::Cpu));
        let y = Tensor::randn(&[100], (Kind::Float, Device::Cpu));
        let sim = mean_cosine_similarity_diagnostic(&x, &y);
        // Mean cosine similarity should be approximately in [-1, 1] (with floating point tolerance)
        assert!(
            (-1.01..=1.01).contains(&sim),
            "Cosine similarity should be approx in [-1, 1], got {}",
            sim
        );
    }

    #[test]
    fn mean_cosine_similarity_tensor_empty_tensors() {
        let empty = Tensor::zeros(&[0], (Kind::Float, Device::Cpu));
        let sim = mean_cosine_similarity_tensor(&empty, &empty);
        assert_eq!(
            sim.double_value(&[]),
            0.0,
            "Empty tensors should have 0 similarity"
        );
    }

    #[test]
    fn leakage_budget_formula() {
        // Just verify the function exists and returns reasonable values
        let examples = crate::data::synthetic_phase1_examples();
        let budget = leakage_budget(&examples[0], 0.25);
        assert!(budget > 0.0, "Budget should be positive");
    }

    #[test]
    fn pharmacophore_leakage_routes_are_reported_independently() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..2];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let loss = LeakageLoss::new(
            PharmacophoreProbeConfig {
                enable_topology_to_pocket_role_leakage: true,
                enable_geometry_to_pocket_role_leakage: true,
                enable_pocket_to_ligand_role_leakage: true,
                leakage_margin: 1.0,
                ..PharmacophoreProbeConfig::default()
            },
            ExplicitLeakageProbeConfig::default(),
        );

        let components = loss.compute_batch_with_routes(examples, &forwards);

        assert!(components
            .similarity_proxy_diagnostic
            .double_value(&[])
            .is_finite());
        assert!(components
            .explicit_probe_diagnostic
            .double_value(&[])
            .is_finite());
        assert!(components.total.double_value(&[]).is_finite());
        assert!(components
            .topology_to_pocket_role
            .double_value(&[])
            .is_finite());
        assert!(components
            .geometry_to_pocket_role
            .double_value(&[])
            .is_finite());
        assert!(components
            .pocket_to_ligand_role
            .double_value(&[])
            .is_finite());
    }

    #[test]
    fn explicit_off_modality_routes_are_activated_by_config() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..2];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);

        let loss = LeakageLoss::new(
            PharmacophoreProbeConfig {
                leakage_margin: 1.0,
                ..PharmacophoreProbeConfig::default()
            },
            ExplicitLeakageProbeConfig {
                enable_explicit_probes: true,
                topology_to_geometry_probe: true,
                geometry_to_topology_probe: true,
                pocket_to_geometry_probe: true,
                pocket_to_topology_probe: true,
                ..ExplicitLeakageProbeConfig::default()
            },
        );

        let components = loss.compute_batch_with_routes(examples, &forwards);
        assert!(components
            .similarity_proxy_diagnostic
            .double_value(&[])
            .is_finite());
        assert!(components
            .explicit_probe_diagnostic
            .double_value(&[])
            .is_finite());
        assert!(components
            .topology_to_geometry
            .double_value(&[])
            .is_finite());
        assert!(components
            .geometry_to_topology
            .double_value(&[])
            .is_finite());
        assert!(components.pocket_to_geometry.double_value(&[]).is_finite());
        assert!(components
            .pocket_to_topology_role
            .double_value(&[])
            .is_finite());
        assert!(components
            .topology_to_pocket_role
            .double_value(&[])
            .is_finite());
        assert!(components
            .geometry_to_pocket_role
            .double_value(&[])
            .is_finite());
        assert!(components
            .pocket_to_ligand_role
            .double_value(&[])
            .is_finite());
        assert!(components.total.double_value(&[]).is_finite());
    }

    #[test]
    fn explicit_leakage_semantics_can_detach_diagnostic_penalties() {
        let diagnostic_pred = Tensor::from_slice(&[0.25_f32]).set_requires_grad(true);
        let probe_fit_pred = Tensor::from_slice(&[0.10_f32]).set_requires_grad(true);
        let encoder_penalty_pred = Tensor::from_slice(&[0.40_f32]).set_requires_grad(true);
        let adversarial = scalar_leakage_route(
            &diagnostic_pred,
            &probe_fit_pred,
            &encoder_penalty_pred,
            0.25,
            1.0,
            ExplicitLeakageProbeTrainingSemantics::AdversarialPenalty,
            None,
        );
        let detached = scalar_leakage_route(
            &diagnostic_pred,
            &probe_fit_pred,
            &encoder_penalty_pred,
            0.25,
            1.0,
            ExplicitLeakageProbeTrainingSemantics::DetachedDiagnostic,
            None,
        );
        let probe_fit = scalar_leakage_route(
            &diagnostic_pred,
            &probe_fit_pred,
            &encoder_penalty_pred,
            0.25,
            1.0,
            ExplicitLeakageProbeTrainingSemantics::ProbeFit,
            None,
        );
        let encoder_penalty = scalar_leakage_route(
            &diagnostic_pred,
            &probe_fit_pred,
            &encoder_penalty_pred,
            0.25,
            1.0,
            ExplicitLeakageProbeTrainingSemantics::EncoderPenalty,
            None,
        );

        assert!(adversarial.objective.requires_grad());
        assert!(!adversarial.diagnostic.requires_grad());
        assert!(!detached.objective.requires_grad());
        assert!(!detached.diagnostic.requires_grad());
        assert!(probe_fit.objective.requires_grad());
        assert!(probe_fit.probe_fit_loss.requires_grad());
        assert!(!probe_fit.encoder_penalty.requires_grad());
        assert!(encoder_penalty.objective.requires_grad());
        assert!(!encoder_penalty.probe_fit_loss.requires_grad());
        assert!(encoder_penalty.encoder_penalty.requires_grad());
        assert!(adversarial.objective.double_value(&[]) > 0.0);
        assert_eq!(detached.objective.double_value(&[]), 0.0);
        assert_eq!(
            adversarial.objective.double_value(&[]),
            detached.diagnostic.double_value(&[])
        );
    }

    #[test]
    fn core_similarity_leakage_reaches_modality_gradients_when_adversarial() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let mut loss = LeakageLoss::new(
            PharmacophoreProbeConfig::default(),
            ExplicitLeakageProbeConfig {
                training_semantics: ExplicitLeakageProbeTrainingSemantics::AdversarialPenalty,
                ..ExplicitLeakageProbeConfig::default()
            },
        );
        loss.margin = -2.0;

        let components = loss.compute_batch_with_routes(examples, &forwards);
        assert!(components.core.requires_grad());
        assert!(!components.similarity_proxy_diagnostic.requires_grad());
        components.core.backward();

        assert_gradient_active(&var_store, "topology", &["topology.", "topology/"]);
        assert_gradient_active(&var_store, "geometry", &["geometry.", "geometry/"]);
        assert_gradient_active(&var_store, "pocket", &["pocket.", "pocket/"]);
        assert_gradient_active(
            &var_store,
            "topology slots",
            &["slot_topology.", "slot_topology/"],
        );
        assert_gradient_active(
            &var_store,
            "geometry slots",
            &["slot_geometry.", "slot_geometry/"],
        );
        assert_gradient_active(
            &var_store,
            "pocket slots",
            &["slot_pocket.", "slot_pocket/"],
        );
    }

    #[test]
    fn explicit_leakage_probes_reach_probe_and_modality_gradients_when_active() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let mut loss = LeakageLoss::new(
            PharmacophoreProbeConfig {
                enable_topology_to_pocket_role_leakage: true,
                enable_geometry_to_pocket_role_leakage: true,
                enable_pocket_to_ligand_role_leakage: true,
                leakage_margin: 1_000_000.0,
                ..PharmacophoreProbeConfig::default()
            },
            ExplicitLeakageProbeConfig {
                enable_explicit_probes: true,
                training_semantics: ExplicitLeakageProbeTrainingSemantics::AdversarialPenalty,
                topology_to_geometry_probe: true,
                topology_to_pocket_probe: true,
                geometry_to_topology_probe: true,
                geometry_to_pocket_probe: true,
                pocket_to_topology_probe: true,
                pocket_to_geometry_probe: true,
            },
        );
        loss.margin = -2.0;

        let components = loss.compute_batch_with_routes(examples, &forwards);
        assert!(components.total.requires_grad());
        assert!(components.explicit_probe_diagnostic.double_value(&[]) > 0.0);
        components.total.backward();

        assert_gradient_active(&var_store, "topology", &["topology.", "topology/"]);
        assert_gradient_active(&var_store, "geometry", &["geometry.", "geometry/"]);
        assert_gradient_active(&var_store, "pocket", &["pocket.", "pocket/"]);
        assert_gradient_active(&var_store, "probe heads", &["probes.", "probes/"]);
    }

    #[test]
    fn probe_fit_leakage_route_updates_probe_heads_without_modality_gradients() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let loss = all_explicit_leakage_loss(ExplicitLeakageProbeTrainingSemantics::ProbeFit);

        let components = loss.compute_batch_with_routes_for_step(examples, &forwards, Some(0));
        assert_eq!(components.route_status, "probe_fit");
        assert!(components.total.requires_grad());
        assert!(components.explicit_probe_fit_loss.requires_grad());
        assert!(!components.explicit_encoder_penalty.requires_grad());
        assert!(components
            .explicit_probe_fit_loss
            .double_value(&[])
            .is_finite());
        components.total.backward();

        assert_gradient_active(&var_store, "probe heads", &["probes.", "probes/"]);
        assert_gradient_inactive(&var_store, "topology", &["topology.", "topology/"]);
        assert_gradient_inactive(&var_store, "geometry", &["geometry.", "geometry/"]);
        assert_gradient_inactive(&var_store, "pocket", &["pocket.", "pocket/"]);
        assert_gradient_inactive(
            &var_store,
            "topology slots",
            &["slot_topology.", "slot_topology/"],
        );
        assert_gradient_inactive(
            &var_store,
            "geometry slots",
            &["slot_geometry.", "slot_geometry/"],
        );
        assert_gradient_inactive(
            &var_store,
            "pocket slots",
            &["slot_pocket.", "slot_pocket/"],
        );
    }

    #[test]
    fn encoder_penalty_leakage_route_updates_modalities_without_probe_head_gradients() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let loss = all_explicit_leakage_loss(ExplicitLeakageProbeTrainingSemantics::EncoderPenalty);

        let components = loss.compute_batch_with_routes_for_step(examples, &forwards, Some(1));
        assert_eq!(components.route_status, "encoder_penalty");
        assert!(components.total.requires_grad());
        assert!(components.explicit_encoder_penalty.requires_grad());
        assert!(!components.explicit_probe_fit_loss.requires_grad());
        assert!(components.explicit_encoder_penalty.double_value(&[]) > 0.0);
        components.total.backward();

        assert_gradient_active(&var_store, "topology", &["topology.", "topology/"]);
        assert_gradient_active(&var_store, "geometry", &["geometry.", "geometry/"]);
        assert_gradient_active(&var_store, "pocket", &["pocket.", "pocket/"]);
        assert_gradient_inactive(&var_store, "probe heads", &["probes.", "probes/"]);
    }

    #[test]
    fn alternating_leakage_route_switches_probe_fit_and_encoder_penalty_by_step() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let loss = all_explicit_leakage_loss(ExplicitLeakageProbeTrainingSemantics::Alternating);

        let probe_step = loss.compute_batch_with_routes_for_step(examples, &forwards, Some(0));
        let encoder_step = loss.compute_batch_with_routes_for_step(examples, &forwards, Some(1));

        assert_eq!(probe_step.route_status, "alternating_probe_fit");
        assert!(probe_step.explicit_probe_fit_loss.double_value(&[]) > 0.0);
        assert_eq!(probe_step.explicit_encoder_penalty.double_value(&[]), 0.0);
        assert_eq!(encoder_step.route_status, "alternating_encoder_penalty");
        assert!(encoder_step.explicit_encoder_penalty.double_value(&[]) > 0.0);
        assert_eq!(encoder_step.explicit_probe_fit_loss.double_value(&[]), 0.0);
    }

    #[test]
    fn detached_leakage_mode_reports_diagnostics_without_optimizer_objective() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let mut loss = LeakageLoss::new(
            PharmacophoreProbeConfig {
                enable_topology_to_pocket_role_leakage: true,
                enable_geometry_to_pocket_role_leakage: true,
                enable_pocket_to_ligand_role_leakage: true,
                leakage_margin: 1_000_000.0,
                ..PharmacophoreProbeConfig::default()
            },
            ExplicitLeakageProbeConfig {
                enable_explicit_probes: true,
                training_semantics: ExplicitLeakageProbeTrainingSemantics::DetachedDiagnostic,
                topology_to_geometry_probe: true,
                topology_to_pocket_probe: true,
                geometry_to_topology_probe: true,
                geometry_to_pocket_probe: true,
                pocket_to_topology_probe: true,
                pocket_to_geometry_probe: true,
            },
        );
        loss.margin = -2.0;

        let components = loss.compute_batch_with_routes(examples, &forwards);
        assert_eq!(components.core.double_value(&[]), 0.0);
        assert_eq!(components.total.double_value(&[]), 0.0);
        assert!(components.similarity_proxy_diagnostic.double_value(&[]) > 0.0);
        assert!(components.explicit_probe_diagnostic.double_value(&[]) > 0.0);
        assert_eq!(components.route_status, "detached_diagnostic");
        assert!(!components.total.requires_grad());
    }

    fn all_explicit_leakage_loss(semantics: ExplicitLeakageProbeTrainingSemantics) -> LeakageLoss {
        let mut loss = LeakageLoss::new(
            PharmacophoreProbeConfig {
                enable_topology_to_pocket_role_leakage: true,
                enable_geometry_to_pocket_role_leakage: true,
                enable_pocket_to_ligand_role_leakage: true,
                leakage_margin: 1_000_000.0,
                ..PharmacophoreProbeConfig::default()
            },
            ExplicitLeakageProbeConfig {
                enable_explicit_probes: true,
                training_semantics: semantics,
                topology_to_geometry_probe: true,
                topology_to_pocket_probe: true,
                geometry_to_topology_probe: true,
                geometry_to_pocket_probe: true,
                pocket_to_topology_probe: true,
                pocket_to_geometry_probe: true,
            },
        );
        loss.margin = 1_000_000.0;
        loss
    }

    fn assert_gradient_active(var_store: &nn::VarStore, label: &str, prefixes: &[&str]) {
        let (matched, l2_norm) = gradient_l2_for_prefixes(var_store, prefixes);
        assert!(matched > 0, "no parameters matched {label}");
        assert!(
            l2_norm > 0.0 && l2_norm.is_finite(),
            "expected active finite gradient for {label}, got {l2_norm}"
        );
    }

    fn assert_gradient_inactive(var_store: &nn::VarStore, label: &str, prefixes: &[&str]) {
        let (matched, l2_norm) = gradient_l2_for_prefixes(var_store, prefixes);
        assert!(matched > 0, "no parameters matched {label}");
        assert!(
            l2_norm <= 1.0e-12,
            "expected inactive gradient for {label}, got {l2_norm}"
        );
    }

    fn gradient_l2_for_prefixes(var_store: &nn::VarStore, prefixes: &[&str]) -> (usize, f64) {
        let mut matched = 0;
        let mut sq_sum = 0.0;
        for (name, tensor) in var_store.variables() {
            if !prefixes.iter().any(|prefix| name.starts_with(prefix)) {
                continue;
            }
            matched += 1;
            let grad = tensor.grad();
            if !grad.defined() || grad.numel() == 0 {
                continue;
            }
            let grad_sq_sum = (&grad * &grad).sum(Kind::Float).double_value(&[]);
            if grad_sq_sum.is_finite() && grad_sq_sum > 0.0 {
                sq_sum += grad_sq_sum;
            }
        }
        (matched, sq_sum.sqrt())
    }
}
