//! Auxiliary objective block for staged semantic decoupling training.
//!
//! This module keeps the paper-facing auxiliary objectives in one ablatable
//! block while leaving the trainer responsible only for scheduling and optimizer
//! execution.

use tch::{no_grad, Kind, Tensor};

use crate::{
    config::types::ExplicitLeakageProbeConfig,
    config::{InteractionPathGateRegularizationWeight, PharmacophoreProbeConfig},
    data::MolecularExample,
    models::{ResearchForward, SlotEncoding},
    training::{
        AuxiliaryLossMetrics, AuxiliaryObjectiveFamily, AuxiliaryObjectiveReport,
        AuxiliaryObjectiveReportEntry, EffectiveLossWeights,
    },
};

use super::{
    leakage::LeakageLossTensors, probe::ProbeLossTensors, ChemistryGuardrailAuxLoss,
    ConsistencyLoss, GateLoss, GatePathObjectiveContribution, IntraRedundancyLoss, LeakageLoss,
    MutualInformationMonitor, PocketGeometryAuxLoss, ProbeLoss,
};

/// Execution mode selected for an auxiliary objective family after staged weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxiliaryObjectiveExecution {
    /// Build the normal gradient-carrying objective graph.
    Trainable,
    /// Compute a detached diagnostic value under `no_grad`.
    DetachedDiagnostic,
    /// Return an explicit zero tensor without evaluating the objective.
    SkippedZeroWeight,
}

impl AuxiliaryObjectiveExecution {
    fn from_weight(effective_weight: f64, detached_diagnostic_when_disabled: bool) -> Self {
        if effective_weight.is_finite() && effective_weight > 0.0 {
            Self::Trainable
        } else if detached_diagnostic_when_disabled {
            Self::DetachedDiagnostic
        } else {
            Self::SkippedZeroWeight
        }
    }

    fn computes(self) -> bool {
        matches!(self, Self::Trainable | Self::DetachedDiagnostic)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Trainable => "trainable",
            Self::DetachedDiagnostic => "detached_diagnostic",
            Self::SkippedZeroWeight => "skipped_zero_weight",
        }
    }
}

/// Per-family auxiliary objective execution plan.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AuxiliaryObjectiveExecutionPlan {
    pub intra_red: AuxiliaryObjectiveExecution,
    pub probe: AuxiliaryObjectiveExecution,
    pub pharmacophore_probe: AuxiliaryObjectiveExecution,
    pub leak: AuxiliaryObjectiveExecution,
    pub pharmacophore_leakage: AuxiliaryObjectiveExecution,
    pub gate: AuxiliaryObjectiveExecution,
    pub slot: AuxiliaryObjectiveExecution,
    pub consistency: AuxiliaryObjectiveExecution,
    pub pocket_contact: AuxiliaryObjectiveExecution,
    pub pocket_clash: AuxiliaryObjectiveExecution,
    pub pocket_envelope: AuxiliaryObjectiveExecution,
    pub valence_guardrail: AuxiliaryObjectiveExecution,
    pub bond_length_guardrail: AuxiliaryObjectiveExecution,
}

/// Compact execution-mode counts for staged auxiliary smoke reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(dead_code)] // Used by the smoke-report test and documented benchmark command.
pub(crate) struct AuxiliaryObjectiveExecutionCounts {
    pub trainable: usize,
    pub detached_diagnostic: usize,
    pub skipped_zero_weight: usize,
}

impl AuxiliaryObjectiveExecutionPlan {
    /// Build the training execution plan from effective staged weights.
    pub(crate) fn from_effective_weights(weights: &EffectiveLossWeights) -> Self {
        Self::from_effective_weights_with_detached_diagnostics(weights, false)
    }

    /// Build a plan that optionally computes disabled families as detached diagnostics.
    pub(crate) fn from_effective_weights_with_detached_diagnostics(
        weights: &EffectiveLossWeights,
        detached_diagnostic_when_disabled: bool,
    ) -> Self {
        let mode = |weight| {
            AuxiliaryObjectiveExecution::from_weight(weight, detached_diagnostic_when_disabled)
        };
        Self {
            intra_red: mode(weights.intra_red),
            probe: mode(weights.probe),
            pharmacophore_probe: mode(weights.pharmacophore_probe),
            leak: mode(weights.leak),
            pharmacophore_leakage: mode(weights.pharmacophore_leakage),
            gate: mode(weights.gate),
            slot: mode(weights.slot),
            consistency: mode(weights.consistency),
            pocket_contact: mode(weights.pocket_contact),
            pocket_clash: mode(weights.pocket_clash),
            pocket_envelope: mode(weights.pocket_envelope),
            valence_guardrail: mode(weights.valence_guardrail),
            bond_length_guardrail: mode(weights.bond_length_guardrail),
        }
    }

    #[allow(dead_code)] // Compatibility path for callers that intentionally compute all auxiliaries.
    fn all_trainable() -> Self {
        Self {
            intra_red: AuxiliaryObjectiveExecution::Trainable,
            probe: AuxiliaryObjectiveExecution::Trainable,
            pharmacophore_probe: AuxiliaryObjectiveExecution::Trainable,
            leak: AuxiliaryObjectiveExecution::Trainable,
            pharmacophore_leakage: AuxiliaryObjectiveExecution::Trainable,
            gate: AuxiliaryObjectiveExecution::Trainable,
            slot: AuxiliaryObjectiveExecution::Trainable,
            consistency: AuxiliaryObjectiveExecution::Trainable,
            pocket_contact: AuxiliaryObjectiveExecution::Trainable,
            pocket_clash: AuxiliaryObjectiveExecution::Trainable,
            pocket_envelope: AuxiliaryObjectiveExecution::Trainable,
            valence_guardrail: AuxiliaryObjectiveExecution::Trainable,
            bond_length_guardrail: AuxiliaryObjectiveExecution::Trainable,
        }
    }

    /// Count objective families by execution mode.
    #[allow(dead_code)] // Used by the smoke-report test and documented benchmark command.
    pub(crate) fn execution_counts(&self) -> AuxiliaryObjectiveExecutionCounts {
        let mut counts = AuxiliaryObjectiveExecutionCounts::default();
        for mode in self.modes() {
            match mode {
                AuxiliaryObjectiveExecution::Trainable => counts.trainable += 1,
                AuxiliaryObjectiveExecution::DetachedDiagnostic => counts.detached_diagnostic += 1,
                AuxiliaryObjectiveExecution::SkippedZeroWeight => counts.skipped_zero_weight += 1,
            }
        }
        counts
    }

    #[allow(dead_code)] // Helper for execution-count reporting.
    fn modes(&self) -> [AuxiliaryObjectiveExecution; 13] {
        [
            self.intra_red,
            self.probe,
            self.pharmacophore_probe,
            self.leak,
            self.pharmacophore_leakage,
            self.gate,
            self.slot,
            self.consistency,
            self.pocket_contact,
            self.pocket_clash,
            self.pocket_envelope,
            self.valence_guardrail,
            self.bond_length_guardrail,
        ]
    }
}

/// Unweighted auxiliary objective tensors for one mini-batch.
pub(crate) struct AuxiliaryObjectiveTensors {
    /// Intra-modality redundancy objective.
    pub intra_red: Tensor,
    /// Semantic probe objective.
    pub probe: Tensor,
    /// Core semantic probe objective before optional pharmacophore role terms.
    pub probe_core: Tensor,
    /// Ligand pharmacophore role probe subterm.
    pub probe_ligand_pharmacophore: Tensor,
    /// Pocket pharmacophore role probe subterm.
    pub probe_pocket_pharmacophore: Tensor,
    /// Leakage-control objective.
    pub leak: Tensor,
    /// Optimizer-facing similarity-proxy leakage objective.
    pub leak_core: Tensor,
    /// Detached similarity-proxy leakage diagnostic before training semantics are applied.
    pub leak_similarity_proxy_diagnostic: Tensor,
    /// Detached explicit leakage-probe diagnostic before training semantics are applied.
    pub leak_explicit_probe_diagnostic: Tensor,
    /// Explicit probe-fitting loss routed into leakage probe heads.
    pub leak_probe_fit_loss: Tensor,
    /// Explicit encoder penalty routed into source encoders.
    pub leak_encoder_penalty: Tensor,
    /// Active explicit leakage route status.
    pub leak_route_status: String,
    /// Explicit topology-to-geometry leakage subterm.
    pub leak_topology_to_geometry: Tensor,
    /// Explicit geometry-to-topology leakage subterm.
    pub leak_geometry_to_topology: Tensor,
    /// Explicit pocket-to-geometry leakage subterm.
    pub leak_pocket_to_geometry: Tensor,
    /// Explicit topology-to-pocket-role leakage subterm.
    pub leak_topology_to_pocket_role: Tensor,
    /// Explicit geometry-to-pocket-role leakage subterm.
    pub leak_geometry_to_pocket_role: Tensor,
    /// Explicit pocket-to-ligand-role leakage subterm.
    pub leak_pocket_to_ligand_role: Tensor,
    /// Gate sparsity objective.
    pub gate: Tensor,
    /// Per-path decomposition of the gate sparsity objective.
    pub gate_path_contributions: Vec<GatePathObjectiveContribution>,
    /// Slot sparsity and balance objective.
    pub slot: Tensor,
    /// Topology-geometry consistency objective.
    pub consistency: Tensor,
    /// Pocket-ligand contact encouragement objective.
    pub pocket_contact: Tensor,
    /// Pocket-ligand steric-clash penalty objective.
    pub pocket_clash: Tensor,
    /// Pocket-envelope containment objective.
    pub pocket_envelope: Tensor,
    /// Conservative valence overage objective.
    pub valence_guardrail: Tensor,
    /// Topology-implied bond-length objective.
    pub bond_length_guardrail: Tensor,
    /// Diagnostic topology-geometry dependence estimate.
    pub mi_topo_geo: f64,
    /// Diagnostic topology-pocket dependence estimate.
    pub mi_topo_pocket: f64,
    /// Diagnostic geometry-pocket dependence estimate.
    pub mi_geo_pocket: f64,
}

impl AuxiliaryObjectiveTensors {
    /// Convert raw tensors into flat metrics plus per-family diagnostic report.
    #[allow(dead_code)] // Kept for compatibility with callers that compute all auxiliaries.
    pub(crate) fn to_metrics_with_weights(
        &self,
        scalar_or_nan: impl Fn(&Tensor) -> f64,
        effective_weights: &EffectiveLossWeights,
    ) -> (AuxiliaryLossMetrics, AuxiliaryObjectiveReport) {
        self.to_metrics_with_weights_and_execution_plan(
            scalar_or_nan,
            effective_weights,
            &AuxiliaryObjectiveExecutionPlan::all_trainable(),
        )
    }

    /// Convert raw tensors into metrics with explicit execution-mode provenance.
    pub(crate) fn to_metrics_with_weights_and_execution_plan(
        &self,
        scalar_or_nan: impl Fn(&Tensor) -> f64,
        effective_weights: &EffectiveLossWeights,
        execution_plan: &AuxiliaryObjectiveExecutionPlan,
    ) -> (AuxiliaryLossMetrics, AuxiliaryObjectiveReport) {
        let unweighted = self.unweighted_values(&scalar_or_nan);
        let report = Self::build_report(&unweighted, effective_weights, execution_plan);
        let explicit_probe_penalty = unweighted.leak_topology_to_geometry
            + unweighted.leak_geometry_to_topology
            + unweighted.leak_pocket_to_geometry;
        let pharmacophore_role_penalty = unweighted.leak_topology_to_pocket_role
            + unweighted.leak_geometry_to_pocket_role
            + unweighted.leak_pocket_to_ligand_role;
        let metrics = AuxiliaryLossMetrics {
            intra_red: unweighted.intra_red,
            probe: unweighted.probe,
            probe_ligand_pharmacophore: unweighted.probe_ligand_pharmacophore,
            probe_pocket_pharmacophore: unweighted.probe_pocket_pharmacophore,
            leak: unweighted.leak,
            leak_core: unweighted.leak_core,
            leak_similarity_proxy_diagnostic: unweighted.leak_similarity_proxy_diagnostic,
            leak_explicit_probe_diagnostic: unweighted.leak_explicit_probe_diagnostic,
            leak_probe_fit_loss: unweighted.leak_probe_fit_loss,
            leak_encoder_penalty: unweighted.leak_encoder_penalty,
            leak_route_status: self.leak_route_status.clone(),
            leak_topology_to_geometry: unweighted.leak_topology_to_geometry,
            leak_geometry_to_topology: unweighted.leak_geometry_to_topology,
            leak_pocket_to_geometry: unweighted.leak_pocket_to_geometry,
            leak_topology_to_pocket_role: unweighted.leak_topology_to_pocket_role,
            leak_geometry_to_pocket_role: unweighted.leak_geometry_to_pocket_role,
            leak_pocket_to_ligand_role: unweighted.leak_pocket_to_ligand_role,
            leakage_roles: crate::losses::LeakageEvidenceRoleReport::training_step(
                unweighted.leak_core,
                explicit_probe_penalty,
                pharmacophore_role_penalty,
                unweighted.leak_probe_fit_loss,
                unweighted.leak_encoder_penalty,
                &self.leak_route_status,
                unweighted.leak_similarity_proxy_diagnostic,
                unweighted.leak_explicit_probe_diagnostic,
                effective_weights.leak,
                effective_weights.pharmacophore_leakage,
                execution_plan.leak.as_str(),
                execution_plan.pharmacophore_leakage.as_str(),
                execution_plan.leak == AuxiliaryObjectiveExecution::DetachedDiagnostic
                    || execution_plan.pharmacophore_leakage
                        == AuxiliaryObjectiveExecution::DetachedDiagnostic,
            ),
            gate: unweighted.gate,
            gate_path_contributions: self
                .gate_path_contributions
                .iter()
                .map(|contribution| contribution.with_effective_loss_weight(effective_weights.gate))
                .collect(),
            slot: unweighted.slot,
            consistency: unweighted.consistency,
            pocket_contact: unweighted.pocket_contact,
            pocket_clash: unweighted.pocket_clash,
            pocket_envelope: unweighted.pocket_envelope,
            valence_guardrail: unweighted.valence_guardrail,
            bond_length_guardrail: unweighted.bond_length_guardrail,
            mi_topo_geo: self.mi_topo_geo,
            mi_topo_pocket: self.mi_topo_pocket,
            mi_geo_pocket: self.mi_geo_pocket,
            auxiliary_objective_report: report.clone(),
        };
        (metrics, report)
    }

    fn unweighted_values(
        &self,
        scalar_or_nan: impl Fn(&Tensor) -> f64,
    ) -> AuxiliaryObjectiveUnweightedValues {
        AuxiliaryObjectiveUnweightedValues {
            intra_red: scalar_or_nan(&self.intra_red),
            probe: scalar_or_nan(&self.probe),
            probe_core: scalar_or_nan(&self.probe_core),
            probe_ligand_pharmacophore: scalar_or_nan(&self.probe_ligand_pharmacophore),
            probe_pocket_pharmacophore: scalar_or_nan(&self.probe_pocket_pharmacophore),
            leak: scalar_or_nan(&self.leak),
            leak_core: scalar_or_nan(&self.leak_core),
            leak_similarity_proxy_diagnostic: scalar_or_nan(&self.leak_similarity_proxy_diagnostic),
            leak_explicit_probe_diagnostic: scalar_or_nan(&self.leak_explicit_probe_diagnostic),
            leak_probe_fit_loss: scalar_or_nan(&self.leak_probe_fit_loss),
            leak_encoder_penalty: scalar_or_nan(&self.leak_encoder_penalty),
            leak_topology_to_geometry: scalar_or_nan(&self.leak_topology_to_geometry),
            leak_geometry_to_topology: scalar_or_nan(&self.leak_geometry_to_topology),
            leak_pocket_to_geometry: scalar_or_nan(&self.leak_pocket_to_geometry),
            leak_topology_to_pocket_role: scalar_or_nan(&self.leak_topology_to_pocket_role),
            leak_geometry_to_pocket_role: scalar_or_nan(&self.leak_geometry_to_pocket_role),
            leak_pocket_to_ligand_role: scalar_or_nan(&self.leak_pocket_to_ligand_role),
            gate: scalar_or_nan(&self.gate),
            slot: scalar_or_nan(&self.slot),
            consistency: scalar_or_nan(&self.consistency),
            pocket_contact: scalar_or_nan(&self.pocket_contact),
            pocket_clash: scalar_or_nan(&self.pocket_clash),
            pocket_envelope: scalar_or_nan(&self.pocket_envelope),
            valence_guardrail: scalar_or_nan(&self.valence_guardrail),
            bond_length_guardrail: scalar_or_nan(&self.bond_length_guardrail),
        }
    }

    fn build_report(
        unweighted: &AuxiliaryObjectiveUnweightedValues,
        effective_weights: &EffectiveLossWeights,
        execution_plan: &AuxiliaryObjectiveExecutionPlan,
    ) -> AuxiliaryObjectiveReport {
        AuxiliaryObjectiveReport {
            entries: vec![
                objective_entry(
                    AuxiliaryObjectiveFamily::IntraRed,
                    unweighted.intra_red,
                    effective_weights.intra_red,
                    execution_plan.intra_red,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::Probe,
                    unweighted.probe_core,
                    effective_weights.probe,
                    execution_plan.probe,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::PharmacophoreProbe,
                    unweighted.probe_ligand_pharmacophore + unweighted.probe_pocket_pharmacophore,
                    effective_weights.pharmacophore_probe,
                    execution_plan.pharmacophore_probe,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::Leak,
                    unweighted.leak_core
                        + unweighted.leak_topology_to_geometry
                        + unweighted.leak_geometry_to_topology
                        + unweighted.leak_pocket_to_geometry,
                    effective_weights.leak,
                    execution_plan.leak,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::PharmacophoreLeakage,
                    unweighted.leak_topology_to_pocket_role
                        + unweighted.leak_geometry_to_pocket_role
                        + unweighted.leak_pocket_to_ligand_role,
                    effective_weights.pharmacophore_leakage,
                    execution_plan.pharmacophore_leakage,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::Gate,
                    unweighted.gate,
                    effective_weights.gate,
                    execution_plan.gate,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::Slot,
                    unweighted.slot,
                    effective_weights.slot,
                    execution_plan.slot,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::Consistency,
                    unweighted.consistency,
                    effective_weights.consistency,
                    execution_plan.consistency,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::PocketContact,
                    unweighted.pocket_contact,
                    effective_weights.pocket_contact,
                    execution_plan.pocket_contact,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::PocketClash,
                    unweighted.pocket_clash,
                    effective_weights.pocket_clash,
                    execution_plan.pocket_clash,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::PocketEnvelope,
                    unweighted.pocket_envelope,
                    effective_weights.pocket_envelope,
                    execution_plan.pocket_envelope,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::ValenceGuardrail,
                    unweighted.valence_guardrail,
                    effective_weights.valence_guardrail,
                    execution_plan.valence_guardrail,
                ),
                objective_entry(
                    AuxiliaryObjectiveFamily::BondLengthGuardrail,
                    unweighted.bond_length_guardrail,
                    effective_weights.bond_length_guardrail,
                    execution_plan.bond_length_guardrail,
                ),
            ],
        }
    }
}

#[derive(Debug)]
struct AuxiliaryObjectiveUnweightedValues {
    intra_red: f64,
    probe: f64,
    probe_core: f64,
    probe_ligand_pharmacophore: f64,
    probe_pocket_pharmacophore: f64,
    leak: f64,
    leak_core: f64,
    leak_similarity_proxy_diagnostic: f64,
    leak_explicit_probe_diagnostic: f64,
    leak_probe_fit_loss: f64,
    leak_encoder_penalty: f64,
    leak_topology_to_geometry: f64,
    leak_geometry_to_topology: f64,
    leak_pocket_to_geometry: f64,
    leak_topology_to_pocket_role: f64,
    leak_geometry_to_pocket_role: f64,
    leak_pocket_to_ligand_role: f64,
    gate: f64,
    slot: f64,
    consistency: f64,
    pocket_contact: f64,
    pocket_clash: f64,
    pocket_envelope: f64,
    valence_guardrail: f64,
    bond_length_guardrail: f64,
}

fn objective_entry(
    family: AuxiliaryObjectiveFamily,
    unweighted_value: f64,
    effective_weight: f64,
    execution: AuxiliaryObjectiveExecution,
) -> AuxiliaryObjectiveReportEntry {
    let enabled = effective_weight.is_finite() && effective_weight > 0.0;
    let weighted_value = unweighted_value * effective_weight;
    let (status, warning) =
        objective_status(unweighted_value, effective_weight, weighted_value, enabled);
    AuxiliaryObjectiveReportEntry {
        family,
        unweighted_value,
        effective_weight,
        weighted_value,
        enabled,
        execution_mode: execution.as_str().to_string(),
        status,
        warning,
    }
}

fn objective_status(
    unweighted_value: f64,
    effective_weight: f64,
    weighted_value: f64,
    enabled: bool,
) -> (String, Option<String>) {
    if !unweighted_value.is_finite() || !effective_weight.is_finite() || !weighted_value.is_finite()
    {
        return (
            "nonfinite".to_string(),
            Some("objective value or effective weight is non-finite".to_string()),
        );
    }
    if !enabled {
        return ("inactive_zero_weight".to_string(), None);
    }
    if weighted_value.abs() <= 1.0e-12 {
        return (
            "active_zero_weighted_value".to_string(),
            Some(
                "enabled objective has zero weighted value; check disabled or detached subterms"
                    .to_string(),
            ),
        );
    }
    ("active".to_string(), None)
}

/// Slot utilization control with sparsity and anti-collapse balance terms.
#[derive(Debug, Clone)]
pub struct SlotControlLoss {
    /// Relative sparsity term weight inside the slot-control objective.
    slot_sparsity_weight: f64,
    /// Relative balance term weight inside the slot-control objective.
    slot_balance_weight: f64,
}

impl SlotControlLoss {
    /// Create a slot-control loss with explicit relative term weights.
    pub(crate) fn with_weights(slot_sparsity_weight: f64, slot_balance_weight: f64) -> Self {
        Self {
            slot_sparsity_weight,
            slot_balance_weight,
        }
    }

    /// Compute the slot-control objective for one forward pass.
    pub(crate) fn compute(&self, forward: &ResearchForward) -> Tensor {
        let sparsity_weight = self.slot_sparsity_weight.max(0.0);
        let balance_weight = self.slot_balance_weight.max(0.0);
        let device = forward.slots.topology.slot_activations.device();
        let mut total = Tensor::zeros([1], (Kind::Float, device));
        let mut active_modalities = 0.0;

        for slots in [
            &forward.slots.topology,
            &forward.slots.geometry,
            &forward.slots.pocket,
        ] {
            if let Some(penalty) =
                active_slot_modality_penalty(slots, sparsity_weight, balance_weight)
            {
                total += penalty;
                active_modalities += 1.0;
            }
        }

        if active_modalities == 0.0 {
            total
        } else {
            total / active_modalities
        }
    }

    /// Compute the mean slot-control objective over a mini-batch.
    pub(crate) fn compute_batch(
        &self,
        forwards: &[ResearchForward],
        device: tch::Device,
    ) -> Tensor {
        if forwards.is_empty() {
            return Tensor::zeros([1], (Kind::Float, device));
        }
        let mut total = Tensor::zeros([1], (Kind::Float, device));
        for forward in forwards {
            total += self.compute(forward);
        }
        total / forwards.len() as f64
    }
}

fn active_slot_modality_penalty(
    slots: &SlotEncoding,
    sparsity_weight: f64,
    balance_weight: f64,
) -> Option<Tensor> {
    if slots.slot_weights.numel() == 0
        || slots.slot_weights.abs().sum(Kind::Float).double_value(&[]) <= 1.0e-12
    {
        return None;
    }
    let (sparsity, activation_balance) = slot_penalties(&slots.slot_activations);
    let mass_balance = slot_mass_balance_penalty(&slots.slot_weights);
    Some(sparsity * sparsity_weight + (activation_balance + mass_balance) * balance_weight)
}

impl Default for SlotControlLoss {
    fn default() -> Self {
        Self::with_weights(1.0, 1.0)
    }
}

/// Paper-facing auxiliary objective group.
#[derive(Debug, Clone)]
pub struct AuxiliaryObjectiveBlock {
    /// Intra-modality redundancy reduction.
    pub(crate) redundancy: IntraRedundancyLoss,
    /// Lightweight semantic probe supervision.
    pub(crate) probe: ProbeLoss,
    /// Off-modality leakage control.
    pub(crate) leakage: LeakageLoss,
    /// Sparse gated cross-modality interaction control.
    pub(crate) gate: GateLoss,
    /// Slot utilization control.
    pub(crate) slot: SlotControlLoss,
    /// Topology-geometry consistency regularization.
    pub(crate) consistency: ConsistencyLoss,
    /// Pocket-ligand geometry contact and clash regularization.
    pub(crate) pocket_geometry: PocketGeometryAuxLoss,
    /// Lightweight chemistry guardrails.
    pub(crate) chemistry_guardrails: ChemistryGuardrailAuxLoss,
    /// Diagnostic dependence monitor; not back-propagated.
    pub(crate) mi_monitor: MutualInformationMonitor,
}

impl Default for AuxiliaryObjectiveBlock {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl AuxiliaryObjectiveBlock {
    /// Construct a block with explicit slot-control relative term weights.
    pub(crate) fn new(slot_sparsity_weight: f64, slot_balance_weight: f64) -> Self {
        Self::new_with_pharmacophore_config(
            slot_sparsity_weight,
            slot_balance_weight,
            PharmacophoreProbeConfig::default(),
            ExplicitLeakageProbeConfig::default(),
        )
    }

    /// Construct a block with explicit pharmacophore role-probe configuration.
    pub(crate) fn new_with_pharmacophore_config(
        slot_sparsity_weight: f64,
        slot_balance_weight: f64,
        pharmacophore: PharmacophoreProbeConfig,
        explicit_leakage_probes: ExplicitLeakageProbeConfig,
    ) -> Self {
        Self::new_with_pharmacophore_and_gate_config(
            slot_sparsity_weight,
            slot_balance_weight,
            pharmacophore,
            explicit_leakage_probes,
            Vec::new(),
        )
    }

    /// Construct a block with explicit pharmacophore and interaction gate configuration.
    pub(crate) fn new_with_pharmacophore_and_gate_config(
        slot_sparsity_weight: f64,
        slot_balance_weight: f64,
        pharmacophore: PharmacophoreProbeConfig,
        explicit_leakage_probes: ExplicitLeakageProbeConfig,
        gate_path_weights: Vec<InteractionPathGateRegularizationWeight>,
    ) -> Self {
        Self {
            redundancy: IntraRedundancyLoss::default(),
            probe: ProbeLoss::new(pharmacophore.clone()),
            leakage: LeakageLoss::new(pharmacophore, explicit_leakage_probes),
            gate: GateLoss::with_path_weights(gate_path_weights),
            slot: SlotControlLoss::with_weights(slot_sparsity_weight, slot_balance_weight),
            consistency: ConsistencyLoss::default(),
            pocket_geometry: PocketGeometryAuxLoss::default(),
            chemistry_guardrails: ChemistryGuardrailAuxLoss::default(),
            mi_monitor: MutualInformationMonitor::default(),
        }
    }
}

impl AuxiliaryObjectiveBlock {
    /// Compute all unweighted auxiliary objectives and diagnostics for a batch.
    #[allow(dead_code)] // Kept for compatibility with non-staged diagnostic callers.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
        affinity_weight_for: impl Fn(&MolecularExample) -> f64,
        device: tch::Device,
    ) -> AuxiliaryObjectiveTensors {
        self.compute_batch_with_execution_plan(
            examples,
            forwards,
            affinity_weight_for,
            device,
            &AuxiliaryObjectiveExecutionPlan::all_trainable(),
        )
    }

    /// Compute auxiliary objectives according to an explicit staged execution plan.
    pub(crate) fn compute_batch_with_execution_plan(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
        affinity_weight_for: impl Fn(&MolecularExample) -> f64,
        device: tch::Device,
        execution_plan: &AuxiliaryObjectiveExecutionPlan,
    ) -> AuxiliaryObjectiveTensors {
        self.compute_batch_with_execution_plan_and_step(
            examples,
            forwards,
            affinity_weight_for,
            device,
            execution_plan,
            None,
        )
    }

    /// Compute auxiliary objectives with a concrete trainer step for route schedules.
    pub(crate) fn compute_batch_with_execution_plan_and_step(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
        affinity_weight_for: impl Fn(&MolecularExample) -> f64,
        device: tch::Device,
        execution_plan: &AuxiliaryObjectiveExecutionPlan,
        leakage_training_step: Option<usize>,
    ) -> AuxiliaryObjectiveTensors {
        let intra_red = tensor_for_execution(execution_plan.intra_red, device, || {
            self.redundancy.compute_batch(forwards)
        });
        let probe_components = probe_for_execution(
            execution_plan.probe,
            execution_plan.pharmacophore_probe,
            device,
            || {
                self.probe.compute_batch_weighted_components(
                    examples,
                    forwards,
                    affinity_weight_for,
                )
            },
        );
        let leak_components = leakage_for_execution(
            execution_plan.leak,
            execution_plan.pharmacophore_leakage,
            device,
            || {
                self.leakage.compute_batch_with_routes_for_step(
                    examples,
                    forwards,
                    leakage_training_step,
                )
            },
        );
        let gate = tensor_for_execution(execution_plan.gate, device, || {
            self.gate.compute_batch(forwards)
        });
        let gate_path_contributions = if execution_plan.gate.computes() {
            self.gate.path_objective_contributions_batch(forwards)
        } else {
            self.gate.zero_path_objective_contributions()
        };
        let slot = tensor_for_execution(execution_plan.slot, device, || {
            self.slot.compute_batch(forwards, device)
        });
        let consistency = tensor_for_execution(execution_plan.consistency, device, || {
            self.consistency.compute_batch(examples, forwards)
        });
        let (pocket_contact, pocket_clash, pocket_envelope) = pocket_geometry_for_execution(
            execution_plan.pocket_contact,
            execution_plan.pocket_clash,
            execution_plan.pocket_envelope,
            device,
            || self.pocket_geometry.compute_batch(examples, forwards),
        );
        let (valence_guardrail, bond_length_guardrail) = chemistry_guardrails_for_execution(
            execution_plan.valence_guardrail,
            execution_plan.bond_length_guardrail,
            device,
            || self.chemistry_guardrails.compute_batch(examples, forwards),
        );
        let (mi_topo_geo, mi_topo_pocket, mi_geo_pocket) = average_mi(&self.mi_monitor, forwards);

        AuxiliaryObjectiveTensors {
            intra_red,
            probe: probe_components.total,
            probe_core: probe_components.core,
            probe_ligand_pharmacophore: probe_components.ligand_pharmacophore,
            probe_pocket_pharmacophore: probe_components.pocket_pharmacophore,
            leak: leak_components.total,
            leak_core: leak_components.core,
            leak_similarity_proxy_diagnostic: leak_components.similarity_proxy_diagnostic,
            leak_explicit_probe_diagnostic: leak_components.explicit_probe_diagnostic,
            leak_probe_fit_loss: leak_components.explicit_probe_fit_loss,
            leak_encoder_penalty: leak_components.explicit_encoder_penalty,
            leak_route_status: leak_components.route_status,
            leak_topology_to_geometry: leak_components.topology_to_geometry,
            leak_geometry_to_topology: leak_components.geometry_to_topology,
            leak_pocket_to_geometry: leak_components.pocket_to_geometry,
            leak_topology_to_pocket_role: leak_components.topology_to_pocket_role,
            leak_geometry_to_pocket_role: leak_components.geometry_to_pocket_role,
            leak_pocket_to_ligand_role: leak_components.pocket_to_ligand_role,
            gate,
            gate_path_contributions,
            slot,
            consistency,
            pocket_contact,
            pocket_clash,
            pocket_envelope,
            valence_guardrail,
            bond_length_guardrail,
            mi_topo_geo,
            mi_topo_pocket,
            mi_geo_pocket,
        }
    }
}

fn zero_tensor(device: tch::Device) -> Tensor {
    Tensor::zeros([1], (Kind::Float, device))
}

fn tensor_for_execution(
    execution: AuxiliaryObjectiveExecution,
    device: tch::Device,
    compute: impl FnOnce() -> Tensor,
) -> Tensor {
    match execution {
        AuxiliaryObjectiveExecution::Trainable => compute(),
        AuxiliaryObjectiveExecution::DetachedDiagnostic => no_grad(compute),
        AuxiliaryObjectiveExecution::SkippedZeroWeight => zero_tensor(device),
    }
}

fn probe_for_execution(
    probe_execution: AuxiliaryObjectiveExecution,
    pharmacophore_execution: AuxiliaryObjectiveExecution,
    device: tch::Device,
    compute: impl FnOnce() -> ProbeLossTensors,
) -> ProbeLossTensors {
    if !probe_execution.computes() && !pharmacophore_execution.computes() {
        return zero_probe_tensors(device);
    }
    let all_detached = probe_execution != AuxiliaryObjectiveExecution::Trainable
        && pharmacophore_execution != AuxiliaryObjectiveExecution::Trainable;
    let mut tensors = if all_detached {
        no_grad(compute)
    } else {
        compute()
    };
    if probe_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        tensors.core = zero_tensor(device);
    }
    if pharmacophore_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        tensors.ligand_pharmacophore = zero_tensor(device);
        tensors.pocket_pharmacophore = zero_tensor(device);
    }
    tensors.total = tensors.core.shallow_clone()
        + tensors.ligand_pharmacophore.shallow_clone()
        + tensors.pocket_pharmacophore.shallow_clone();
    tensors
}

fn zero_probe_tensors(device: tch::Device) -> ProbeLossTensors {
    let zero = zero_tensor(device);
    ProbeLossTensors {
        core: zero.shallow_clone(),
        total: zero.shallow_clone(),
        ligand_pharmacophore: zero.shallow_clone(),
        pocket_pharmacophore: zero,
    }
}

fn leakage_for_execution(
    leak_execution: AuxiliaryObjectiveExecution,
    pharmacophore_execution: AuxiliaryObjectiveExecution,
    device: tch::Device,
    compute: impl FnOnce() -> LeakageLossTensors,
) -> LeakageLossTensors {
    if !leak_execution.computes() && !pharmacophore_execution.computes() {
        return zero_leakage_tensors(device);
    }
    let all_detached = leak_execution != AuxiliaryObjectiveExecution::Trainable
        && pharmacophore_execution != AuxiliaryObjectiveExecution::Trainable;
    let mut tensors = if all_detached {
        no_grad(compute)
    } else {
        compute()
    };
    if leak_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        tensors.core = zero_tensor(device);
        tensors.topology_to_geometry = zero_tensor(device);
        tensors.geometry_to_topology = zero_tensor(device);
        tensors.pocket_to_geometry = zero_tensor(device);
    }
    if pharmacophore_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        tensors.topology_to_pocket_role = zero_tensor(device);
        tensors.geometry_to_pocket_role = zero_tensor(device);
        tensors.pocket_to_topology_role = zero_tensor(device);
        tensors.pocket_to_ligand_role = zero_tensor(device);
    }
    tensors.total = tensors.core.shallow_clone()
        + tensors.topology_to_geometry.shallow_clone()
        + tensors.geometry_to_topology.shallow_clone()
        + tensors.pocket_to_geometry.shallow_clone()
        + tensors.pocket_to_topology_role.shallow_clone()
        + tensors.topology_to_pocket_role.shallow_clone()
        + tensors.geometry_to_pocket_role.shallow_clone()
        + tensors.pocket_to_ligand_role.shallow_clone();
    refresh_leakage_route_summary(&mut tensors, device);
    tensors
}

fn refresh_leakage_route_summary(tensors: &mut LeakageLossTensors, device: tch::Device) {
    match tensors.route_status.as_str() {
        "probe_fit" | "alternating_probe_fit" => {
            tensors.explicit_probe_fit_loss = tensors.total.shallow_clone();
            tensors.explicit_encoder_penalty = zero_tensor(device);
        }
        "encoder_penalty" | "alternating_encoder_penalty" | "adversarial_penalty" => {
            tensors.explicit_probe_fit_loss = zero_tensor(device);
            tensors.explicit_encoder_penalty = tensors.total.shallow_clone();
        }
        "detached_diagnostic" | "skipped_zero_weight" => {
            tensors.explicit_probe_fit_loss = zero_tensor(device);
            tensors.explicit_encoder_penalty = zero_tensor(device);
        }
        _ => {}
    }
}

fn zero_leakage_tensors(device: tch::Device) -> LeakageLossTensors {
    let zero = zero_tensor(device);
    LeakageLossTensors {
        core: zero.shallow_clone(),
        similarity_proxy_diagnostic: zero.shallow_clone(),
        explicit_probe_diagnostic: zero.shallow_clone(),
        explicit_probe_fit_loss: zero.shallow_clone(),
        explicit_encoder_penalty: zero.shallow_clone(),
        route_status: "skipped_zero_weight".to_string(),
        total: zero.shallow_clone(),
        topology_to_geometry: zero.shallow_clone(),
        geometry_to_topology: zero.shallow_clone(),
        pocket_to_geometry: zero.shallow_clone(),
        topology_to_pocket_role: zero.shallow_clone(),
        geometry_to_pocket_role: zero.shallow_clone(),
        pocket_to_topology_role: zero.shallow_clone(),
        pocket_to_ligand_role: zero,
    }
}

fn pocket_geometry_for_execution(
    contact_execution: AuxiliaryObjectiveExecution,
    clash_execution: AuxiliaryObjectiveExecution,
    envelope_execution: AuxiliaryObjectiveExecution,
    device: tch::Device,
    compute: impl FnOnce() -> (Tensor, Tensor, Tensor),
) -> (Tensor, Tensor, Tensor) {
    if !contact_execution.computes()
        && !clash_execution.computes()
        && !envelope_execution.computes()
    {
        return (
            zero_tensor(device),
            zero_tensor(device),
            zero_tensor(device),
        );
    }
    let all_detached = contact_execution != AuxiliaryObjectiveExecution::Trainable
        && clash_execution != AuxiliaryObjectiveExecution::Trainable
        && envelope_execution != AuxiliaryObjectiveExecution::Trainable;
    let (mut contact, mut clash, mut envelope) = if all_detached {
        no_grad(compute)
    } else {
        compute()
    };
    if contact_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        contact = zero_tensor(device);
    }
    if clash_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        clash = zero_tensor(device);
    }
    if envelope_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        envelope = zero_tensor(device);
    }
    (contact, clash, envelope)
}

fn chemistry_guardrails_for_execution(
    valence_execution: AuxiliaryObjectiveExecution,
    bond_execution: AuxiliaryObjectiveExecution,
    device: tch::Device,
    compute: impl FnOnce() -> (Tensor, Tensor),
) -> (Tensor, Tensor) {
    if !valence_execution.computes() && !bond_execution.computes() {
        return (zero_tensor(device), zero_tensor(device));
    }
    let all_detached = valence_execution != AuxiliaryObjectiveExecution::Trainable
        && bond_execution != AuxiliaryObjectiveExecution::Trainable;
    let (mut valence, mut bond) = if all_detached {
        no_grad(compute)
    } else {
        compute()
    };
    if valence_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        valence = zero_tensor(device);
    }
    if bond_execution == AuxiliaryObjectiveExecution::SkippedZeroWeight {
        bond = zero_tensor(device);
    }
    (valence, bond)
}

fn average_mi(
    mi_monitor: &MutualInformationMonitor,
    forwards: &[ResearchForward],
) -> (f64, f64, f64) {
    if forwards.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut mi_topo_geo = 0.0;
    let mut mi_topo_pocket = 0.0;
    let mut mi_geo_pocket = 0.0;
    for forward in forwards {
        let (tg, tp, gp) = mi_monitor.compute_all_mi(forward);
        mi_topo_geo += tg;
        mi_topo_pocket += tp;
        mi_geo_pocket += gp;
    }
    let denom = forwards.len() as f64;
    (
        mi_topo_geo / denom,
        mi_topo_pocket / denom,
        mi_geo_pocket / denom,
    )
}

fn slot_penalties(slot_activations: &Tensor) -> (Tensor, Tensor) {
    if slot_activations.numel() == 0 {
        let zeros = Tensor::zeros([1], (Kind::Float, slot_activations.device()));
        return (zeros.shallow_clone(), zeros);
    }
    let activations = slot_activations.clamp(0.0, 1.0);
    let sparsity = activations.mean(Kind::Float);
    let mean_activation = activations.mean(Kind::Float);
    let slot_count = activations.size().first().copied().unwrap_or(0).max(0);
    let dead_floor = Tensor::full([], 0.05, (Kind::Float, slot_activations.device()));
    let saturation_ceiling = Tensor::full([], 0.95, (Kind::Float, slot_activations.device()));
    let dead_collapse = (dead_floor - &mean_activation)
        .relu()
        .pow_tensor_scalar(2.0);
    let saturated_collapse = (&mean_activation - saturation_ceiling)
        .relu()
        .pow_tensor_scalar(2.0);
    let utilization_imbalance = if slot_count <= 1 {
        Tensor::zeros([], (Kind::Float, slot_activations.device()))
    } else {
        let normalized = &activations / activations.sum(Kind::Float).clamp_min(1e-6);
        let entropy = -(&normalized * normalized.clamp_min(1e-8).log()).sum(Kind::Float);
        let max_entropy = (slot_count as f64).ln().max(1e-6);
        (Tensor::from(1.0).to_device(slot_activations.device()) - entropy / max_entropy)
            .clamp(0.0, 1.0)
    };
    let balance = dead_collapse + saturated_collapse + utilization_imbalance;
    (sparsity, balance)
}

fn slot_mass_balance_penalty(slot_weights: &Tensor) -> Tensor {
    if slot_weights.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, slot_weights.device()));
    }
    let weights = slot_weights.flatten(0, -1).clamp_min(0.0);
    let total_mass = weights.sum(Kind::Float);
    if total_mass.double_value(&[]) <= 1.0e-12 {
        return Tensor::zeros([1], (Kind::Float, slot_weights.device()));
    }
    let slot_count = weights.size().first().copied().unwrap_or(0).max(0);
    if slot_count <= 1 {
        return Tensor::zeros([1], (Kind::Float, slot_weights.device()));
    }
    let normalized = &weights / total_mass.clamp_min(1.0e-6);
    let entropy = -(&normalized * normalized.clamp_min(1.0e-8).log()).sum(Kind::Float);
    let max_entropy = (slot_count as f64).ln().max(1.0e-6);
    (Tensor::from(1.0).to_device(slot_weights.device()) - entropy / max_entropy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Instant;

    use tch::{nn, Device, Kind, Tensor};

    use crate::{
        config::{AffinityWeighting, PrimaryObjectiveConfig, ResearchConfig},
        data::InMemoryDataset,
        models::{Phase1ResearchSystem, SlotEncoding},
        training::EffectiveLossWeights,
    };

    #[test]
    fn auxiliary_block_matches_component_batch_aggregation() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 3;
        config.training.primary_objective = PrimaryObjectiveConfig::ConditionedDenoising;
        config.training.affinity_weighting = AffinityWeighting::InverseFrequency;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..3];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let block = AuxiliaryObjectiveBlock::default();
        let weights = measurement_weights_for_test(examples, config.training.affinity_weighting);

        let batch = block.compute_batch(
            examples,
            &forwards,
            |example| {
                example
                    .targets
                    .affinity_measurement_type
                    .as_deref()
                    .and_then(|measurement| weights.get(measurement))
                    .copied()
                    .unwrap_or(1.0)
            },
            var_store.device(),
        );

        let denom = examples.len() as f64;
        let mut probe_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut redundancy_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut leakage_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut gate_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut slot_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut consistency_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut contact_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut clash_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut envelope_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut valence_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut bond_length_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let affinity_weight = example
                .targets
                .affinity_measurement_type
                .as_deref()
                .and_then(|measurement| weights.get(measurement))
                .copied()
                .unwrap_or(1.0);
            probe_manual += block
                .probe
                .compute_weighted(example, forward, affinity_weight);
            redundancy_manual += block.redundancy.compute(&forward.slots);
            leakage_manual += block.leakage.compute_with_routes(example, forward).total;
            gate_manual += block.gate.compute(&forward.interactions);
            slot_manual += block.slot.compute(forward);
            consistency_manual += block.consistency.compute(example, forward);
            let (contact, clash, envelope) = block.pocket_geometry.compute(example, forward);
            contact_manual += contact;
            clash_manual += clash;
            envelope_manual += envelope;
            let (valence, bond_length) = block.chemistry_guardrails.compute(example, forward);
            valence_manual += valence;
            bond_length_manual += bond_length;
        }

        assert_close(&batch.probe, &(probe_manual / denom));
        assert_eq!(batch.probe_ligand_pharmacophore.double_value(&[]), 0.0);
        assert_eq!(batch.probe_pocket_pharmacophore.double_value(&[]), 0.0);
        assert_close(&batch.intra_red, &(redundancy_manual / denom));
        assert_close(&batch.leak, &(leakage_manual / denom));
        assert_eq!(batch.leak_topology_to_geometry.double_value(&[]), 0.0);
        assert_eq!(batch.leak_geometry_to_topology.double_value(&[]), 0.0);
        assert_eq!(batch.leak_pocket_to_geometry.double_value(&[]), 0.0);
        assert_eq!(batch.leak_topology_to_pocket_role.double_value(&[]), 0.0);
        assert_eq!(batch.leak_geometry_to_pocket_role.double_value(&[]), 0.0);
        assert_eq!(batch.leak_pocket_to_ligand_role.double_value(&[]), 0.0);
        assert_close(&batch.gate, &(gate_manual / denom));
        assert_close(&batch.slot, &(slot_manual / denom));
        assert_close(&batch.consistency, &(consistency_manual / denom));
        assert_close(&batch.pocket_contact, &(contact_manual / denom));
        assert_close(&batch.pocket_clash, &(clash_manual / denom));
        assert_close(&batch.pocket_envelope, &(envelope_manual / denom));
        assert_close(&batch.valence_guardrail, &(valence_manual / denom));
        assert_close(&batch.bond_length_guardrail, &(bond_length_manual / denom));
    }

    #[test]
    fn auxiliary_objective_report_exposes_stable_families_with_weight_metadata() {
        let tensors = AuxiliaryObjectiveTensors {
            intra_red: Tensor::full([1], 0.2, (Kind::Float, Device::Cpu)),
            probe: Tensor::full([1], 0.4, (Kind::Float, Device::Cpu)),
            probe_core: Tensor::full([1], 0.31, (Kind::Float, Device::Cpu)),
            probe_ligand_pharmacophore: Tensor::full([1], 0.04, (Kind::Float, Device::Cpu)),
            probe_pocket_pharmacophore: Tensor::full([1], 0.05, (Kind::Float, Device::Cpu)),
            leak: Tensor::full([1], 0.6, (Kind::Float, Device::Cpu)),
            leak_core: Tensor::full([1], 0.39, (Kind::Float, Device::Cpu)),
            leak_similarity_proxy_diagnostic: Tensor::full([1], 0.41, (Kind::Float, Device::Cpu)),
            leak_explicit_probe_diagnostic: Tensor::full([1], 0.42, (Kind::Float, Device::Cpu)),
            leak_probe_fit_loss: Tensor::full([1], 0.43, (Kind::Float, Device::Cpu)),
            leak_encoder_penalty: Tensor::full([1], 0.44, (Kind::Float, Device::Cpu)),
            leak_route_status: "encoder_penalty".to_string(),
            leak_topology_to_geometry: Tensor::full([1], 0.31, (Kind::Float, Device::Cpu)),
            leak_geometry_to_topology: Tensor::full([1], 0.32, (Kind::Float, Device::Cpu)),
            leak_pocket_to_geometry: Tensor::full([1], 0.33, (Kind::Float, Device::Cpu)),
            leak_topology_to_pocket_role: Tensor::full([1], 0.06, (Kind::Float, Device::Cpu)),
            leak_geometry_to_pocket_role: Tensor::full([1], 0.07, (Kind::Float, Device::Cpu)),
            leak_pocket_to_ligand_role: Tensor::full([1], 0.08, (Kind::Float, Device::Cpu)),
            gate: Tensor::full([1], 0.8, (Kind::Float, Device::Cpu)),
            gate_path_contributions: Vec::new(),
            slot: Tensor::full([1], 1.0, (Kind::Float, Device::Cpu)),
            consistency: Tensor::full([1], 1.2, (Kind::Float, Device::Cpu)),
            pocket_contact: Tensor::full([1], 1.4, (Kind::Float, Device::Cpu)),
            pocket_clash: Tensor::full([1], 1.6, (Kind::Float, Device::Cpu)),
            pocket_envelope: Tensor::full([1], 1.8, (Kind::Float, Device::Cpu)),
            valence_guardrail: Tensor::full([1], 2.0, (Kind::Float, Device::Cpu)),
            bond_length_guardrail: Tensor::full([1], 2.2, (Kind::Float, Device::Cpu)),
            mi_topo_geo: 0.7,
            mi_topo_pocket: 0.8,
            mi_geo_pocket: 0.9,
        };
        let weights = EffectiveLossWeights {
            primary: 1.0,
            intra_red: 0.25,
            probe: 0.0,
            pharmacophore_probe: 0.2,
            leak: 0.5,
            pharmacophore_leakage: 0.3,
            gate: 0.0,
            slot: 0.75,
            consistency: 0.9,
            pocket_contact: 0.0,
            pocket_clash: 1.1,
            pocket_envelope: 0.0,
            valence_guardrail: 0.6,
            bond_length_guardrail: 0.0,
        };

        let (metrics, report) =
            tensors.to_metrics_with_weights(|tensor| tensor.double_value(&[]), &weights);

        assert_eq!(report.entries.len(), 13);
        assert!((metrics.leak_probe_fit_loss - 0.43).abs() < 1.0e-6);
        assert!((metrics.leak_encoder_penalty - 0.44).abs() < 1.0e-6);
        assert_eq!(metrics.leak_route_status, "encoder_penalty");
        assert!(metrics.leakage_roles.optimizer_penalty.active);
        assert_eq!(
            metrics
                .leakage_roles
                .optimizer_penalty
                .explicit_route_status,
            "encoder_penalty"
        );
        assert_eq!(
            metrics.leakage_roles.optimizer_penalty.leak_execution_mode,
            "trainable"
        );
        assert!(metrics
            .leakage_roles
            .detached_training_diagnostic
            .interpretation
            .contains("held-out leakage estimates"));
        assert_eq!(metrics.leakage_roles.frozen_probe_audit.status, "not_run");
        let families = report
            .entries
            .iter()
            .map(|entry| entry.family.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            families,
            vec![
                "intra_red",
                "probe",
                "pharmacophore_probe",
                "leak",
                "pharmacophore_leakage",
                "gate",
                "slot",
                "consistency",
                "pocket_contact",
                "pocket_clash",
                "pocket_envelope",
                "valence_guardrail",
                "bond_length_guardrail",
            ]
        );
        let expected_weights = [
            0.25f64, 0.0, 0.2, 0.5, 0.3, 0.0, 0.75, 0.9, 0.0, 1.1, 0.0, 0.6, 0.0,
        ];
        for (entry, expected_weight) in report.entries.iter().zip(expected_weights) {
            assert_eq!(entry.effective_weight, expected_weight);
            assert_eq!(
                entry.weighted_value,
                entry.unweighted_value * expected_weight
            );
            assert_eq!(entry.enabled, expected_weight > 0.0);
            assert_eq!(entry.execution_mode, "trainable");
            assert!(!entry.status.is_empty());
        }
    }

    #[test]
    fn auxiliary_execution_plan_skips_zero_weight_graphs_and_reports_disabled_status() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..2];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let block = AuxiliaryObjectiveBlock::default();
        let weights = EffectiveLossWeights {
            primary: 1.0,
            intra_red: 0.0,
            probe: 0.0,
            pharmacophore_probe: 0.0,
            leak: 0.0,
            pharmacophore_leakage: 0.0,
            gate: 0.0,
            slot: 0.0,
            consistency: 0.2,
            pocket_contact: 0.0,
            pocket_clash: 0.0,
            pocket_envelope: 0.0,
            valence_guardrail: 0.0,
            bond_length_guardrail: 0.0,
        };
        let plan = AuxiliaryObjectiveExecutionPlan::from_effective_weights(&weights);

        let tensors = block.compute_batch_with_execution_plan(
            examples,
            &forwards,
            |_| 1.0,
            var_store.device(),
            &plan,
        );
        assert_eq!(tensors.intra_red.double_value(&[]), 0.0);
        assert_eq!(tensors.slot.double_value(&[]), 0.0);
        assert_eq!(tensors.gate.double_value(&[]), 0.0);
        assert!(!tensors.intra_red.requires_grad());
        assert!(!tensors.slot.requires_grad());
        assert!(!tensors.gate.requires_grad());

        let (_, report) = tensors.to_metrics_with_weights_and_execution_plan(
            |tensor| tensor.double_value(&[]),
            &weights,
            &plan,
        );
        let slot = report
            .entries
            .iter()
            .find(|entry| entry.family == AuxiliaryObjectiveFamily::Slot)
            .unwrap();
        assert!(!slot.enabled);
        assert_eq!(slot.status, "inactive_zero_weight");
        assert_eq!(slot.execution_mode, "skipped_zero_weight");
        let consistency = report
            .entries
            .iter()
            .find(|entry| entry.family == AuxiliaryObjectiveFamily::Consistency)
            .unwrap();
        assert!(consistency.enabled);
        assert_eq!(consistency.execution_mode, "trainable");
    }

    #[test]
    fn auxiliary_execution_plan_can_request_detached_disabled_diagnostics() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..2];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let block = AuxiliaryObjectiveBlock::default();
        let weights = EffectiveLossWeights {
            primary: 1.0,
            intra_red: 0.0,
            probe: 0.0,
            pharmacophore_probe: 0.0,
            leak: 0.0,
            pharmacophore_leakage: 0.0,
            gate: 0.0,
            slot: 0.0,
            consistency: 0.0,
            pocket_contact: 0.0,
            pocket_clash: 0.0,
            pocket_envelope: 0.0,
            valence_guardrail: 0.0,
            bond_length_guardrail: 0.0,
        };
        let plan =
            AuxiliaryObjectiveExecutionPlan::from_effective_weights_with_detached_diagnostics(
                &weights, true,
            );

        let tensors = block.compute_batch_with_execution_plan(
            examples,
            &forwards,
            |_| 1.0,
            var_store.device(),
            &plan,
        );

        assert!(tensors.slot.double_value(&[]).is_finite());
        assert!(tensors.slot.double_value(&[]) > 0.0);
        assert!(!tensors.slot.requires_grad());
        let (_, report) = tensors.to_metrics_with_weights_and_execution_plan(
            |tensor| tensor.double_value(&[]),
            &weights,
            &plan,
        );
        assert!(report
            .entries
            .iter()
            .all(|entry| entry.execution_mode == "detached_diagnostic"));
        assert!(report.entries.iter().all(|entry| !entry.enabled));
    }

    #[test]
    fn staged_auxiliary_execution_plan_reports_stage_dependent_proxy_work() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..2];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let block = AuxiliaryObjectiveBlock::default();
        let stage1_weights = EffectiveLossWeights {
            primary: 1.0,
            intra_red: 0.0,
            probe: 0.0,
            pharmacophore_probe: 0.0,
            leak: 0.0,
            pharmacophore_leakage: 0.0,
            gate: 0.0,
            slot: 0.0,
            consistency: 0.1,
            pocket_contact: 0.0,
            pocket_clash: 0.0,
            pocket_envelope: 0.0,
            valence_guardrail: 0.0,
            bond_length_guardrail: 0.0,
        };
        let stage4_weights = EffectiveLossWeights {
            intra_red: 0.1,
            probe: 0.2,
            leak: 0.05,
            gate: 0.05,
            slot: 0.05,
            ..stage1_weights
        };

        let stage1_plan = AuxiliaryObjectiveExecutionPlan::from_effective_weights(&stage1_weights);
        let stage4_plan = AuxiliaryObjectiveExecutionPlan::from_effective_weights(&stage4_weights);
        let stage1_start = Instant::now();
        let stage1 = block.compute_batch_with_execution_plan(
            examples,
            &forwards,
            |_| 1.0,
            var_store.device(),
            &stage1_plan,
        );
        let stage1_elapsed = stage1_start.elapsed();
        let stage4_start = Instant::now();
        let stage4 = block.compute_batch_with_execution_plan(
            examples,
            &forwards,
            |_| 1.0,
            var_store.device(),
            &stage4_plan,
        );
        let stage4_elapsed = stage4_start.elapsed();

        let stage1_counts = stage1_plan.execution_counts();
        let stage4_counts = stage4_plan.execution_counts();
        println!(
            "aux_execution_smoke stage1_trainable={} stage1_skipped={} stage1_us={} stage4_trainable={} stage4_skipped={} stage4_us={}",
            stage1_counts.trainable,
            stage1_counts.skipped_zero_weight,
            stage1_elapsed.as_micros(),
            stage4_counts.trainable,
            stage4_counts.skipped_zero_weight,
            stage4_elapsed.as_micros()
        );

        assert!(stage4_counts.trainable > stage1_counts.trainable);
        assert!(stage1_counts.skipped_zero_weight > stage4_counts.skipped_zero_weight);
        assert_eq!(stage1.slot.double_value(&[]), 0.0);
        assert!(stage4.slot.double_value(&[]).is_finite());
    }

    #[test]
    fn slot_control_loss_supports_configurable_sparsity_balance_weighting() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);

        let mut forward = forwards[0].clone();
        forward.slots.topology.slot_activations =
            Tensor::zeros_like(&forward.slots.topology.slot_activations);
        forward.slots.geometry.slot_activations =
            Tensor::ones_like(&forward.slots.geometry.slot_activations);
        forward.slots.pocket.slot_activations =
            Tensor::ones_like(&forward.slots.pocket.slot_activations) * 0.5;

        let (topo_sparsity, topo_balance) = slot_penalties(&Tensor::zeros_like(
            &forwards[0].slots.topology.slot_activations,
        ));
        let (geo_sparsity, geo_balance) = slot_penalties(&Tensor::ones_like(
            &forwards[0].slots.geometry.slot_activations,
        ));
        let (pocket_sparsity, pocket_balance) =
            slot_penalties(&(Tensor::ones_like(&forwards[0].slots.pocket.slot_activations) * 0.5));
        let topo_mass_balance = slot_mass_balance_penalty(&forward.slots.topology.slot_weights);
        let geo_mass_balance = slot_mass_balance_penalty(&forward.slots.geometry.slot_weights);
        let pocket_mass_balance = slot_mass_balance_penalty(&forward.slots.pocket.slot_weights);

        let sparse_only = SlotControlLoss::with_weights(1.0, 0.0).compute(&forward);
        let balance_only = SlotControlLoss::with_weights(0.0, 1.0).compute(&forward);
        let combined = SlotControlLoss::with_weights(1.0, 1.0).compute(&forward);

        let topo_sparsity_v = topo_sparsity.double_value(&[]);
        let topo_balance_v = topo_balance.double_value(&[]);
        let geo_sparsity_v = geo_sparsity.double_value(&[]);
        let geo_balance_v = geo_balance.double_value(&[]);
        let pocket_sparsity_v = pocket_sparsity.double_value(&[]);
        let pocket_balance_v = pocket_balance.double_value(&[]);
        let topo_mass_balance_v = topo_mass_balance.double_value(&[]);
        let geo_mass_balance_v = geo_mass_balance.double_value(&[]);
        let pocket_mass_balance_v = pocket_mass_balance.double_value(&[]);

        let expected_sparse = (topo_sparsity_v + geo_sparsity_v + pocket_sparsity_v) / 3.0;
        let expected_balance = (topo_balance_v
            + geo_balance_v
            + pocket_balance_v
            + topo_mass_balance_v
            + geo_mass_balance_v
            + pocket_mass_balance_v)
            / 3.0;
        let expected_combined = (topo_sparsity_v
            + geo_sparsity_v
            + pocket_sparsity_v
            + topo_balance_v
            + geo_balance_v
            + pocket_balance_v
            + topo_mass_balance_v
            + geo_mass_balance_v
            + pocket_mass_balance_v)
            / 3.0;

        let expected_sparse = Tensor::from(expected_sparse);
        let expected_balance = Tensor::from(expected_balance);
        let expected_combined = Tensor::from(expected_combined);

        assert_close(&sparse_only, &expected_sparse);
        assert_close(&balance_only, &expected_balance);
        assert_close(&combined, &expected_combined);
        assert!(sparse_only.double_value(&[]).is_finite());
        assert!(balance_only.double_value(&[]).is_finite());
        assert!(combined.double_value(&[]).is_finite());
    }

    #[test]
    fn slot_control_loss_ignores_disabled_modalities_and_normalizes_active_scale() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let mut full = forwards[0].clone();
        full.slots.topology.slot_activations =
            Tensor::ones_like(&full.slots.topology.slot_activations) * 0.5;
        full.slots.geometry.slot_activations =
            Tensor::ones_like(&full.slots.geometry.slot_activations) * 0.5;
        full.slots.pocket.slot_activations =
            Tensor::ones_like(&full.slots.pocket.slot_activations) * 0.5;
        set_uniform_slot_weights(&mut full.slots.topology);
        set_uniform_slot_weights(&mut full.slots.geometry);
        set_uniform_slot_weights(&mut full.slots.pocket);

        let mut topology_only = full.clone();
        zero_test_slot_modality(&mut topology_only.slots.geometry);
        zero_test_slot_modality(&mut topology_only.slots.pocket);

        let mut all_disabled = topology_only.clone();
        zero_test_slot_modality(&mut all_disabled.slots.topology);

        let loss = SlotControlLoss::with_weights(1.0, 1.0);
        let full_value = loss.compute(&full);
        let topology_only_value = loss.compute(&topology_only);
        let all_disabled_value = loss.compute(&all_disabled);

        assert_close(&full_value, &topology_only_value);
        assert_eq!(all_disabled_value.double_value(&[]), 0.0);
    }

    #[test]
    fn slot_penalties_use_activation_gates_without_uniform_per_sample_pressure() {
        let dead = Tensor::zeros([4], (Kind::Float, Device::Cpu));
        let moderate = Tensor::ones([4], (Kind::Float, Device::Cpu)) * 0.5;
        let saturated = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let one_hot = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let same_mean_balanced = Tensor::ones([4], (Kind::Float, Device::Cpu)) * 0.25;

        let (dead_sparsity, dead_balance) = slot_penalties(&dead);
        let (moderate_sparsity, moderate_balance) = slot_penalties(&moderate);
        let (saturated_sparsity, saturated_balance) = slot_penalties(&saturated);
        let (_, one_hot_balance) = slot_penalties(&one_hot);
        let (_, same_mean_balanced_balance) = slot_penalties(&same_mean_balanced);

        assert!(dead_sparsity.double_value(&[]) < moderate_sparsity.double_value(&[]));
        assert!(moderate_sparsity.double_value(&[]) < saturated_sparsity.double_value(&[]));
        assert!(dead_balance.double_value(&[]) > moderate_balance.double_value(&[]));
        assert!(saturated_balance.double_value(&[]) > moderate_balance.double_value(&[]));
        assert!(one_hot_balance.double_value(&[]) > same_mean_balanced_balance.double_value(&[]));
        assert_eq!(moderate_balance.double_value(&[]), 0.0);
    }

    #[test]
    fn slot_mass_balance_penalty_detects_concentrated_assignment_mass() {
        let concentrated = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let balanced = Tensor::ones([4], (Kind::Float, Device::Cpu)) / 4.0;

        assert!(
            slot_mass_balance_penalty(&concentrated).double_value(&[])
                > slot_mass_balance_penalty(&balanced).double_value(&[])
        );
        assert!(slot_mass_balance_penalty(&balanced).double_value(&[]) <= 1.0e-7);
    }

    fn assert_close(left: &Tensor, right: &Tensor) {
        let delta = (left - right).abs().double_value(&[]);
        assert!(
            delta <= 1e-5,
            "loss mismatch: left={} right={} delta={delta}",
            left.double_value(&[]),
            right.double_value(&[])
        );
    }

    fn zero_test_slot_modality(slots: &mut SlotEncoding) {
        slots.slots = Tensor::zeros_like(&slots.slots);
        slots.slot_weights = Tensor::zeros_like(&slots.slot_weights);
        slots.token_assignments = Tensor::zeros_like(&slots.token_assignments);
        slots.slot_activation_logits = Tensor::zeros_like(&slots.slot_activation_logits);
        slots.slot_activations = Tensor::zeros_like(&slots.slot_activations);
        slots.active_slot_mask = Tensor::zeros_like(&slots.active_slot_mask);
        slots.active_slot_count = 0.0;
        slots.reconstructed_tokens = Tensor::zeros_like(&slots.reconstructed_tokens);
    }

    fn set_uniform_slot_weights(slots: &mut SlotEncoding) {
        let slot_count = slots
            .slot_weights
            .size()
            .first()
            .copied()
            .unwrap_or(0)
            .max(1) as f64;
        slots.slot_weights = Tensor::ones_like(&slots.slot_weights) / slot_count;
    }

    fn measurement_weights_for_test(
        examples: &[crate::data::MolecularExample],
        strategy: AffinityWeighting,
    ) -> BTreeMap<String, f64> {
        if strategy == AffinityWeighting::None {
            return BTreeMap::new();
        }

        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for example in examples {
            if example.targets.affinity_kcal_mol.is_some() {
                let measurement = example
                    .targets
                    .affinity_measurement_type
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                *counts.entry(measurement).or_default() += 1;
            }
        }

        let total = counts.values().sum::<usize>() as f64;
        let families = counts.len() as f64;
        counts
            .into_iter()
            .map(|(measurement, count)| (measurement, total / (families * count as f64)))
            .collect()
    }
}
