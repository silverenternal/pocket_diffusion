/// Named loss values emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryObjectiveMetrics {
    /// Name of the active primary objective.
    pub objective_name: String,
    /// Weighted scalar value used as the primary optimization anchor.
    pub primary_value: f64,
    /// Effective staged weight applied to the primary objective.
    #[serde(default)]
    pub effective_weight: f64,
    /// Weighted primary contribution to the total optimizer objective.
    #[serde(default)]
    pub weighted_value: f64,
    /// Whether the primary objective contributes to this optimizer step.
    #[serde(default)]
    pub enabled: bool,
    /// Whether the primary objective is decoder-anchored.
    pub decoder_anchored: bool,
    /// Optional primary objective component decomposition.
    #[serde(default)]
    pub components: PrimaryObjectiveComponentMetrics,
    /// Per-component provenance labels separating trainable objective terms from diagnostics.
    #[serde(default)]
    pub component_provenance: Vec<PrimaryObjectiveComponentProvenance>,
    /// Reporting-only normalized primary component scale diagnostics.
    #[serde(default)]
    pub component_scale_report: PrimaryObjectiveComponentScaleReport,
    /// Effective primary branch schedule weights for flow-style objectives.
    #[serde(default)]
    pub branch_schedule: PrimaryBranchScheduleReport,
}

/// Provenance contract for one primary-objective component.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrimaryObjectiveComponentProvenance {
    /// Stable component field name.
    pub component_name: String,
    /// High-level source family for the component.
    pub anchor: String,
    /// Whether this value remains connected to differentiable tensor operations.
    pub differentiable: bool,
    /// Whether this value is allowed to contribute to the optimizer-facing total.
    pub optimizer_facing: bool,
    /// Human-readable interpretation of the component role.
    pub role: String,
    /// Effective branch weight when this component is owned by a scheduled flow branch.
    #[serde(default)]
    pub effective_branch_weight: Option<f64>,
    /// Schedule/config source that produced `effective_branch_weight`.
    #[serde(default)]
    pub branch_schedule_source: Option<String>,
}

/// Per-component primary objective scale diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrimaryObjectiveComponentScaleReport {
    /// Per-component records in stable component order.
    #[serde(default)]
    pub entries: Vec<PrimaryObjectiveComponentScaleRecord>,
    /// Ratio between the largest and smallest positive normalized absolute values.
    #[serde(default)]
    pub max_to_min_normalized_ratio: f64,
    /// Number of component entries with warnings.
    #[serde(default)]
    pub warning_count: usize,
    /// Configured warning ratio used to build this report.
    #[serde(default)]
    pub warning_ratio: f64,
    /// Normalization source: unit_reference or running_abs_ema.
    #[serde(default)]
    pub normalization_source: String,
}

/// Scalar scale record for one primary component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryObjectiveComponentScaleRecord {
    /// Stable primary component name.
    pub component_name: String,
    /// Component value before the staged primary weight is applied.
    pub unweighted_value: f64,
    /// Effective staged primary weight.
    pub objective_weight: f64,
    /// Component contribution after the staged primary weight is applied.
    pub weighted_value: f64,
    /// Positive normalization denominator used only for diagnostics.
    pub normalization_scale: f64,
    /// Component value divided by `normalization_scale`.
    pub normalized_value: f64,
    /// Whether this component contributes to the optimizer-facing primary objective.
    pub optimizer_facing: bool,
    /// Compact scale status.
    pub status: String,
    /// Optional scale warning.
    #[serde(default)]
    pub warning: Option<String>,
}

/// Effective branch-schedule report for the primary flow objective family.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrimaryBranchScheduleReport {
    /// Whether a flow branch schedule was observed for this step.
    #[serde(default)]
    pub observed: bool,
    /// Global training step used to resolve the schedule.
    #[serde(default)]
    pub training_step: Option<usize>,
    /// One-based training stage index used for the surrounding staged trainer.
    #[serde(default)]
    pub stage_index: Option<usize>,
    /// Source config/contract for the branch weights.
    #[serde(default)]
    pub source: String,
    /// Per-branch effective weights.
    #[serde(default)]
    pub entries: Vec<PrimaryBranchWeightRecord>,
}

/// Effective weight for one primary flow branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryBranchWeightRecord {
    /// Stable branch name.
    pub branch_name: String,
    /// Branch loss before branch scheduling is applied.
    #[serde(default)]
    pub unweighted_value: f64,
    /// Effective branch weight after primary branch scheduling.
    pub effective_weight: f64,
    /// Effective schedule multiplier applied to the static branch weight.
    #[serde(default)]
    pub schedule_multiplier: f64,
    /// Branch contribution after branch scheduling but before the staged primary weight.
    #[serde(default)]
    pub weighted_value: f64,
    /// Whether this branch is active for optimizer-facing losses.
    pub optimizer_facing: bool,
    /// Flow contract version or objective provenance label.
    pub provenance: String,
    /// Atom-row matching policy when this branch consumes matched molecular targets.
    #[serde(default)]
    pub target_matching_policy: Option<String>,
    /// Mean matching cost for matched atom rows when target matching is active.
    #[serde(default)]
    pub target_matching_mean_cost: Option<f64>,
    /// Maximum single-row matching cost when target matching is active.
    #[serde(default)]
    pub target_matching_max_cost: Option<f64>,
    /// Total matching cost over matched atom rows.
    #[serde(default)]
    pub target_matching_total_cost: Option<f64>,
    /// Fraction of generated atom rows backed by matched target supervision.
    #[serde(default)]
    pub target_matching_coverage: Option<f64>,
    /// Number of generated rows assigned to target rows.
    #[serde(default)]
    pub target_matching_matched_count: Option<usize>,
    /// Number of generated rows left unmatched and masked out.
    #[serde(default)]
    pub target_matching_unmatched_generated_count: Option<usize>,
    /// Number of target rows not selected by the matching policy.
    #[serde(default)]
    pub target_matching_unmatched_target_count: Option<usize>,
    /// Whether the assignment was solved exactly for the selected policy.
    #[serde(default)]
    pub target_matching_exact_assignment: Option<bool>,
}

/// Optional per-subterm decomposition for the active primary objective.
///
/// Fields without an `eval_` prefix are values from tensor-preserving paths
/// that can contribute to backpropagation when included by the active
/// objective. Fields with an `eval_` prefix are detached diagnostics derived
/// from sampled, argmaxed, vectorized, or scalar rollout records and must not
/// be interpreted as optimizer-driving losses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrimaryObjectiveComponentMetrics {
    /// Topology-related gradient-bearing reconstruction/consistency contribution.
    #[serde(default)]
    pub topology: Option<f64>,
    /// Geometry-related gradient-bearing reconstruction/consistency contribution.
    #[serde(default)]
    pub geometry: Option<f64>,
    /// Pocket-anchor gradient-bearing contribution.
    #[serde(default)]
    pub pocket_anchor: Option<f64>,
    /// Reserved for future tensor-preserving multi-step rollout refinement.
    #[serde(default)]
    pub rollout: Option<f64>,
    /// Evaluation-only recovery score computed from sampled rollout records.
    #[serde(default)]
    pub rollout_eval_recovery: Option<f64>,
    /// Evaluation-only pocket-anchor score computed from sampled rollout records.
    #[serde(default)]
    pub rollout_eval_pocket_anchor: Option<f64>,
    /// Evaluation-only stop-policy score computed from sampled rollout records.
    #[serde(default)]
    pub rollout_eval_stop: Option<f64>,
    /// Flow velocity gradient-bearing consistency contribution.
    #[serde(default)]
    pub flow_velocity: Option<f64>,
    /// Flow endpoint gradient-bearing consistency contribution.
    #[serde(default)]
    pub flow_endpoint: Option<f64>,
    /// Atom-type categorical flow contribution.
    #[serde(default)]
    pub flow_atom_type: Option<f64>,
    /// Bond existence/type flow contribution.
    #[serde(default)]
    pub flow_bond: Option<f64>,
    /// Topology synchronization flow contribution.
    #[serde(default)]
    pub flow_topology: Option<f64>,
    /// Pocket/context representation flow contribution.
    #[serde(default)]
    pub flow_pocket_context: Option<f64>,
    /// Cross-branch molecular flow synchronization contribution.
    #[serde(default)]
    pub flow_synchronization: Option<f64>,
}

impl PrimaryObjectiveComponentMetrics {
    /// Return provenance records for components observed in this metric bundle.
    pub fn provenance_records(&self) -> Vec<PrimaryObjectiveComponentProvenance> {
        let mut records = Vec::new();
        push_primary_component_provenance(
            &mut records,
            "topology",
            self.topology,
            "decoder_or_surrogate_topology",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "geometry",
            self.geometry,
            "decoder_or_surrogate_geometry",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "pocket_anchor",
            self.pocket_anchor,
            "decoder_pocket_anchor",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "rollout",
            self.rollout,
            "tensor_preserving_rollout",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "rollout_eval_recovery",
            self.rollout_eval_recovery,
            "sampled_rollout_record",
            false,
            false,
        );
        push_primary_component_provenance(
            &mut records,
            "rollout_eval_pocket_anchor",
            self.rollout_eval_pocket_anchor,
            "sampled_rollout_record",
            false,
            false,
        );
        push_primary_component_provenance(
            &mut records,
            "rollout_eval_stop",
            self.rollout_eval_stop,
            "sampled_rollout_stop_policy",
            false,
            false,
        );
        push_primary_component_provenance(
            &mut records,
            "flow_velocity",
            self.flow_velocity,
            "flow_matching_velocity",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "flow_endpoint",
            self.flow_endpoint,
            "flow_matching_endpoint",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "flow_atom_type",
            self.flow_atom_type,
            "molecular_flow_atom_type",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "flow_bond",
            self.flow_bond,
            "molecular_flow_bond",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "flow_topology",
            self.flow_topology,
            "molecular_flow_topology",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "flow_pocket_context",
            self.flow_pocket_context,
            "molecular_flow_pocket_context",
            true,
            true,
        );
        push_primary_component_provenance(
            &mut records,
            "flow_synchronization",
            self.flow_synchronization,
            "molecular_flow_branch_synchronization",
            true,
            true,
        );
        records
    }

    /// Return observed component names and values in stable report order.
    pub fn observed_component_values(&self) -> Vec<(&'static str, f64)> {
        let mut values = Vec::new();
        push_observed_component(&mut values, "topology", self.topology);
        push_observed_component(&mut values, "geometry", self.geometry);
        push_observed_component(&mut values, "pocket_anchor", self.pocket_anchor);
        push_observed_component(&mut values, "rollout", self.rollout);
        push_observed_component(
            &mut values,
            "rollout_eval_recovery",
            self.rollout_eval_recovery,
        );
        push_observed_component(
            &mut values,
            "rollout_eval_pocket_anchor",
            self.rollout_eval_pocket_anchor,
        );
        push_observed_component(&mut values, "rollout_eval_stop", self.rollout_eval_stop);
        push_observed_component(&mut values, "flow_velocity", self.flow_velocity);
        push_observed_component(&mut values, "flow_endpoint", self.flow_endpoint);
        push_observed_component(&mut values, "flow_atom_type", self.flow_atom_type);
        push_observed_component(&mut values, "flow_bond", self.flow_bond);
        push_observed_component(&mut values, "flow_topology", self.flow_topology);
        push_observed_component(
            &mut values,
            "flow_pocket_context",
            self.flow_pocket_context,
        );
        push_observed_component(
            &mut values,
            "flow_synchronization",
            self.flow_synchronization,
        );
        values
    }

    /// Build reporting-only component scale diagnostics.
    pub fn scale_report(
        &self,
        objective_weight: f64,
        warning_ratio: f64,
        epsilon: f64,
        running_scales: Option<&BTreeMap<String, f64>>,
    ) -> PrimaryObjectiveComponentScaleReport {
        let normalization_source = if running_scales.is_some() {
            "running_abs_ema"
        } else {
            "unit_reference"
        }
        .to_string();
        let mut entries = self
            .observed_component_values()
            .into_iter()
            .map(|(component_name, value)| {
                let normalization_scale = running_scales
                    .and_then(|scales| scales.get(component_name).copied())
                    .unwrap_or(1.0)
                    .abs()
                    .max(epsilon);
                let normalized_value = value / normalization_scale;
                PrimaryObjectiveComponentScaleRecord {
                    component_name: component_name.to_string(),
                    unweighted_value: value,
                    objective_weight,
                    weighted_value: value * objective_weight,
                    normalization_scale,
                    normalized_value,
                    optimizer_facing: !component_name.starts_with("rollout_eval_"),
                    status: "ok".to_string(),
                    warning: None,
                }
            })
            .collect::<Vec<_>>();

        let positive_min = entries
            .iter()
            .filter_map(|entry| {
                let abs = entry.normalized_value.abs();
                (abs > epsilon && abs.is_finite()).then_some(abs)
            })
            .fold(f64::INFINITY, f64::min);
        let positive_max = entries
            .iter()
            .filter_map(|entry| {
                let abs = entry.normalized_value.abs();
                (abs > epsilon && abs.is_finite()).then_some(abs)
            })
            .fold(0.0, f64::max);
        let max_to_min_normalized_ratio =
            if positive_min.is_finite() && positive_min > epsilon {
                positive_max / positive_min
            } else {
                0.0
            };
        if max_to_min_normalized_ratio > warning_ratio {
            for entry in &mut entries {
                let abs = entry.normalized_value.abs();
                if abs.is_finite() && abs > positive_min * warning_ratio {
                    entry.status = "dominant".to_string();
                    entry.warning = Some(format!(
                        "normalized primary component exceeds the smallest nonzero component by more than {warning_ratio:.3}x"
                    ));
                }
            }
        }
        let warning_count = entries
            .iter()
            .filter(|entry| entry.warning.is_some())
            .count();

        PrimaryObjectiveComponentScaleReport {
            entries,
            max_to_min_normalized_ratio,
            warning_count,
            warning_ratio,
            normalization_source,
        }
    }
}

fn push_observed_component(
    values: &mut Vec<(&'static str, f64)>,
    component_name: &'static str,
    value: Option<f64>,
) {
    if let Some(value) = value {
        values.push((component_name, value));
    }
}

fn push_primary_component_provenance(
    records: &mut Vec<PrimaryObjectiveComponentProvenance>,
    component_name: &str,
    value: Option<f64>,
    anchor: &str,
    differentiable: bool,
    optimizer_facing: bool,
) {
    if value.is_none() {
        return;
    }
    records.push(PrimaryObjectiveComponentProvenance {
        component_name: component_name.to_string(),
        anchor: anchor.to_string(),
        differentiable,
        optimizer_facing,
        role: if optimizer_facing {
            "trainable_objective".to_string()
        } else {
            "evaluation_only".to_string()
        },
        effective_branch_weight: None,
        branch_schedule_source: None,
    });
}

/// Auxiliary losses emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxiliaryLossMetrics {
    /// Intra-modality redundancy objective.
    pub intra_red: f64,
    /// Semantic probe objective.
    pub probe: f64,
    /// Ligand pharmacophore role probe subterm.
    #[serde(default)]
    pub probe_ligand_pharmacophore: f64,
    /// Pocket pharmacophore role probe subterm.
    #[serde(default)]
    pub probe_pocket_pharmacophore: f64,
    /// Total optimizer-facing leakage objective.
    pub leak: f64,
    /// Optimizer-facing similarity-proxy leakage objective.
    #[serde(default)]
    pub leak_core: f64,
    /// Detached similarity-proxy diagnostic before training semantics are applied.
    #[serde(default)]
    pub leak_similarity_proxy_diagnostic: f64,
    /// Detached explicit leakage-probe diagnostic before training semantics are applied.
    #[serde(default)]
    pub leak_explicit_probe_diagnostic: f64,
    /// Explicit probe-fitting loss routed into leakage probe heads.
    #[serde(default)]
    pub leak_probe_fit_loss: f64,
    /// Explicit encoder penalty routed into source encoders.
    #[serde(default)]
    pub leak_encoder_penalty: f64,
    /// Active explicit leakage route status.
    #[serde(default)]
    pub leak_route_status: String,
    /// Explicit topology->geometry leakage penalty.
    #[serde(default)]
    pub leak_topology_to_geometry: f64,
    /// Explicit geometry->topology leakage penalty.
    #[serde(default)]
    pub leak_geometry_to_topology: f64,
    /// Explicit pocket->geometry leakage penalty.
    #[serde(default)]
    pub leak_pocket_to_geometry: f64,
    /// Explicit topology-to-pocket-role leakage penalty.
    #[serde(default)]
    pub leak_topology_to_pocket_role: f64,
    /// Explicit geometry-to-pocket-role leakage penalty.
    #[serde(default)]
    pub leak_geometry_to_pocket_role: f64,
    /// Explicit pocket-to-ligand-role leakage penalty.
    #[serde(default)]
    pub leak_pocket_to_ligand_role: f64,
    /// Role-separated leakage evidence contract.
    #[serde(default)]
    pub leakage_roles: crate::losses::LeakageEvidenceRoleReport,
    /// Gate regularization objective.
    pub gate: f64,
    /// Per-path contribution breakdown for the gate regularization objective.
    #[serde(default)]
    pub gate_path_contributions: Vec<crate::losses::GatePathObjectiveContribution>,
    /// Slot sparsity and balance objective.
    pub slot: f64,
    /// Topology-geometry consistency objective.
    pub consistency: f64,
    /// Pocket-ligand contact encouragement objective.
    #[serde(default)]
    pub pocket_contact: f64,
    /// Pocket-ligand steric-clash penalty objective.
    #[serde(default)]
    pub pocket_clash: f64,
    /// Pocket-envelope containment objective.
    #[serde(default)]
    pub pocket_envelope: f64,
    /// Conservative valence overage objective.
    #[serde(default)]
    pub valence_guardrail: f64,
    /// Topology-implied bond-length objective.
    #[serde(default)]
    pub bond_length_guardrail: f64,
    /// Mutual information between topology and geometry (decoupling indicator).
    #[serde(default)]
    pub mi_topo_geo: f64,
    /// Mutual information between topology and pocket (decoupling indicator).
    #[serde(default)]
    pub mi_topo_pocket: f64,
    /// Mutual information between geometry and pocket (decoupling indicator).
    #[serde(default)]
    pub mi_geo_pocket: f64,
    /// Family-level auxiliary objective diagnostics.
    #[serde(default)]
    pub auxiliary_objective_report: AuxiliaryObjectiveReport,
}

/// Stable objective family name used for auxiliary objective reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuxiliaryObjectiveFamily {
    /// Intra-modality redundancy reduction.
    IntraRed,
    /// Semantic probe regularization.
    Probe,
    /// Pharmacophore role-probe subterms.
    PharmacophoreProbe,
    /// Off-modality leakage penalty.
    Leak,
    /// Pharmacophore role-leakage subterms.
    PharmacophoreLeakage,
    /// Interaction gate penalty.
    Gate,
    /// Slot utilization and balance regularization.
    Slot,
    /// Topology-geometry consistency objective.
    Consistency,
    /// Pocket-ligand contact encouragement.
    PocketContact,
    /// Pocket-ligand steric-clash penalty.
    PocketClash,
    /// Pocket-envelope containment.
    PocketEnvelope,
    /// Conservative valence overage.
    ValenceGuardrail,
    /// Topology-implied bond-length deviation.
    BondLengthGuardrail,
}

impl AuxiliaryObjectiveFamily {
    /// Stable external representation for configuration and logging.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IntraRed => "intra_red",
            Self::Probe => "probe",
            Self::PharmacophoreProbe => "pharmacophore_probe",
            Self::Leak => "leak",
            Self::PharmacophoreLeakage => "pharmacophore_leakage",
            Self::Gate => "gate",
            Self::Slot => "slot",
            Self::Consistency => "consistency",
            Self::PocketContact => "pocket_contact",
            Self::PocketClash => "pocket_clash",
            Self::PocketEnvelope => "pocket_envelope",
            Self::ValenceGuardrail => "valence_guardrail",
            Self::BondLengthGuardrail => "bond_length_guardrail",
        }
    }
}

/// Scalar record for one auxiliary objective family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxiliaryObjectiveReportEntry {
    /// Stable auxiliary-family identifier.
    pub family: AuxiliaryObjectiveFamily,
    /// Unweighted objective value emitted from the raw computation.
    pub unweighted_value: f64,
    /// Effective (staged + warmed) weight used at this step.
    pub effective_weight: f64,
    /// Weighted contribution to the total objective.
    #[serde(default)]
    pub weighted_value: f64,
    /// Whether this family contributes to the total objective at this step.
    pub enabled: bool,
    /// Execution path used for this family: trainable, detached_diagnostic, or skipped_zero_weight.
    #[serde(default = "default_auxiliary_execution_mode")]
    pub execution_mode: String,
    /// Compact objective-scale status label.
    #[serde(default)]
    pub status: String,
    /// Optional numerical-scale warning for this objective family.
    #[serde(default)]
    pub warning: Option<String>,
}

fn default_auxiliary_execution_mode() -> String {
    "unknown".to_string()
}

/// Report for all auxiliary objective families for one optimizer step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuxiliaryObjectiveReport {
    /// Per-family objective records in stable family order.
    pub entries: Vec<AuxiliaryObjectiveReportEntry>,
}

/// Named loss values emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossBreakdown {
    /// Primary objective metrics.
    pub primary: PrimaryObjectiveMetrics,
    /// Auxiliary regularizer metrics.
    pub auxiliaries: AuxiliaryLossMetrics,
    /// Weighted total objective.
    pub total: f64,
}

/// Stage progress and active objective-family summary for one step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageProgressMetrics {
    /// One-based stage index.
    #[serde(default)]
    pub stage_index: usize,
    /// Fixed-schedule stage index before optional adaptive guarding.
    #[serde(default)]
    pub fixed_stage_index: usize,
    /// Linear ramp value inside the current stage.
    #[serde(default)]
    pub stage_ramp: f64,
    /// Objective family names with positive effective weights.
    #[serde(default)]
    pub active_objective_families: Vec<String>,
    /// Whether adaptive stage readiness checks were enabled.
    #[serde(default)]
    pub adaptive_stage_enabled: bool,
    /// Whether the fixed schedule was held at an earlier effective stage.
    #[serde(default)]
    pub adaptive_stage_hold: bool,
    /// Compact readiness status: disabled, ready, warning, or held.
    #[serde(default)]
    pub readiness_status: String,
    /// Deterministic reasons used to advance, warn, or hold.
    #[serde(default)]
    pub readiness_reasons: Vec<String>,
    /// Per-step objective execution counts for profiling staged graph cost.
    #[serde(default)]
    pub objective_execution_counts: ObjectiveExecutionCountMetrics,
}

/// Objective execution-mode counts for one step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObjectiveExecutionCountMetrics {
    /// Primary objective count when enabled.
    #[serde(default)]
    pub primary_enabled_count: usize,
    /// Auxiliary families evaluated with gradients.
    #[serde(default)]
    pub trainable_auxiliary_count: usize,
    /// Auxiliary families evaluated as detached diagnostics.
    #[serde(default)]
    pub detached_diagnostic_count: usize,
    /// Auxiliary families skipped because effective weight is zero.
    #[serde(default)]
    pub skipped_zero_weight_count: usize,
    /// Total active optimizer-facing objective-family count.
    #[serde(default)]
    pub optimizer_facing_count: usize,
}

/// Runtime and memory profile for one training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRuntimeProfileMetrics {
    /// Wall-clock step runtime in milliseconds.
    #[serde(default)]
    pub step_time_ms: f64,
    /// Number of examples processed per second for this training step.
    #[serde(default)]
    pub examples_per_second: f64,
    /// Batch size used by this step.
    #[serde(default)]
    pub batch_size: usize,
    /// Number of batched model forward calls used by this step.
    #[serde(default)]
    pub forward_batch_count: usize,
    /// Number of per-example forward calls used by this step.
    #[serde(default)]
    pub per_example_forward_count: usize,
    /// Compact execution-mode label for runtime grouping.
    #[serde(default)]
    pub forward_execution_mode: String,
    /// Correctness reason when de novo execution deliberately uses per-example forwards.
    #[serde(default)]
    pub de_novo_per_example_reason: Option<String>,
    /// Whether sampled rollout diagnostic traces were requested for this step.
    #[serde(default)]
    pub rollout_diagnostics_built: bool,
    /// Number of sampled rollout diagnostic traces executed for this step.
    #[serde(default)]
    pub rollout_diagnostic_execution_count: usize,
    /// Whether executed rollout diagnostic traces were built under no-grad.
    #[serde(default)]
    pub rollout_diagnostics_no_grad: bool,
    /// System used-memory estimate before the step, in MiB.
    #[serde(default)]
    pub memory_before_mb: f64,
    /// System used-memory estimate after the step, in MiB.
    #[serde(default)]
    pub memory_after_mb: f64,
    /// Signed used-memory delta for this step, in MiB.
    #[serde(default)]
    pub memory_delta_mb: f64,
    /// Objective execution counts duplicated here for per-stage runtime grouping.
    #[serde(default)]
    pub objective_execution_counts: ObjectiveExecutionCountMetrics,
}

impl Default for TrainingRuntimeProfileMetrics {
    fn default() -> Self {
        Self {
            step_time_ms: 0.0,
            examples_per_second: 0.0,
            batch_size: 0,
            forward_batch_count: 0,
            per_example_forward_count: 0,
            forward_execution_mode: "unavailable".to_string(),
            de_novo_per_example_reason: None,
            rollout_diagnostics_built: false,
            rollout_diagnostic_execution_count: 0,
            rollout_diagnostics_no_grad: false,
            memory_before_mb: 0.0,
            memory_after_mb: 0.0,
            memory_delta_mb: 0.0,
            objective_execution_counts: ObjectiveExecutionCountMetrics::default(),
        }
    }
}

/// Gradient-health summary for one trainable module group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientModuleMetrics {
    /// Stable module bucket name.
    pub module_name: String,
    /// Whether this module is expected to receive gradients for the current step/configuration.
    #[serde(default = "default_gradient_expected_active")]
    pub expected_active: bool,
    /// Number of trainable tensors assigned to this bucket.
    pub trainable_tensor_count: usize,
    /// Number of tensors with a defined gradient after backward.
    pub gradient_tensor_count: usize,
    /// Number of gradient tensors containing NaN or infinity.
    #[serde(default)]
    pub nonfinite_gradient_tensors: usize,
    /// L2 norm of finite gradients in this bucket after any configured clipping.
    pub grad_l2_norm: f64,
    /// Maximum absolute finite gradient value in this bucket after any configured clipping.
    pub grad_abs_max: f64,
    /// Compact status label: active, zero_gradient, no_gradient, or nonfinite_gradient.
    pub status: String,
    /// True when an inactive module is inactive because the active ablation/objective disables it.
    #[serde(default)]
    pub inactive_expected: bool,
    /// True when a module expected to be trainable for this step received no effective gradient.
    #[serde(default)]
    pub inactive_unexpected: bool,
}

impl Default for GradientModuleMetrics {
    fn default() -> Self {
        Self {
            module_name: String::new(),
            expected_active: default_gradient_expected_active(),
            trainable_tensor_count: 0,
            gradient_tensor_count: 0,
            nonfinite_gradient_tensors: 0,
            grad_l2_norm: 0.0,
            grad_abs_max: 0.0,
            status: "no_gradient".to_string(),
            inactive_expected: false,
            inactive_unexpected: false,
        }
    }
}

fn default_gradient_expected_active() -> bool {
    true
}

/// Optimizer-step gradient and numerical-health diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientHealthMetrics {
    /// Per-module gradient summaries.
    #[serde(default)]
    pub modules: Vec<GradientModuleMetrics>,
    /// Global gradient L2 norm before clipping.
    pub pre_clip_global_grad_l2_norm: f64,
    /// Global gradient L2 norm after clipping.
    pub global_grad_l2_norm: f64,
    /// Maximum absolute finite gradient value after clipping.
    pub global_grad_abs_max: f64,
    /// Number of gradient tensors containing NaN or infinity.
    #[serde(default)]
    pub nonfinite_gradient_tensors: usize,
    /// Whether global-norm clipping was configured.
    #[serde(default)]
    pub clipping_enabled: bool,
    /// Configured global-norm clip threshold, when present.
    #[serde(default)]
    pub clip_global_norm: Option<f64>,
    /// Whether the pre-clip norm exceeded the configured threshold.
    #[serde(default)]
    pub clipped: bool,
    /// Whether the optimizer step was skipped for numerical health.
    #[serde(default)]
    pub optimizer_step_skipped: bool,
    /// Non-finite loss terms grouped by objective family.
    #[serde(default)]
    pub nonfinite_loss_terms: Vec<String>,
    /// Sparse objective-family gradient contribution diagnostics.
    #[serde(default)]
    pub objective_families: ObjectiveGradientDiagnostics,
}

impl Default for GradientHealthMetrics {
    fn default() -> Self {
        Self {
            modules: Vec::new(),
            pre_clip_global_grad_l2_norm: 0.0,
            global_grad_l2_norm: 0.0,
            global_grad_abs_max: 0.0,
            nonfinite_gradient_tensors: 0,
            clipping_enabled: false,
            clip_global_norm: None,
            clipped: false,
            optimizer_step_skipped: false,
            nonfinite_loss_terms: Vec::new(),
            objective_families: ObjectiveGradientDiagnostics::default(),
        }
    }
}

/// Objective-family gradient contribution diagnostics for one step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObjectiveGradientDiagnostics {
    /// Whether diagnostics are enabled in configuration.
    #[serde(default)]
    pub enabled: bool,
    /// Whether this step was sampled.
    #[serde(default)]
    pub sampled: bool,
    /// Configured sampling interval.
    #[serde(default)]
    pub sample_every_steps: usize,
    /// Diagnostic mode used to keep runtime bounded.
    #[serde(default)]
    pub sampling_mode: String,
    /// Per-objective-family records.
    #[serde(default)]
    pub entries: Vec<ObjectiveGradientFamilyMetrics>,
}

/// Gradient contribution record for one objective family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveGradientFamilyMetrics {
    /// Stable family label.
    pub family_name: String,
    /// Weighted loss contribution used to allocate the gradient proxy.
    pub weighted_value: f64,
    /// Estimated family-level L2 gradient norm.
    pub grad_l2_norm: f64,
    /// Estimated fraction of global gradient norm.
    pub grad_norm_fraction: f64,
    /// Compact status label.
    pub status: String,
    /// Component or objective provenance label.
    pub provenance: String,
    /// Optional anomaly label.
    #[serde(default)]
    pub anomaly: Option<String>,
}

/// Per-path interaction usage recorded for one optimization step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPathStepMetrics {
    /// Stable directed-path name.
    pub path_name: String,
    /// Stable chemistry-level directed-path role.
    pub path_role: String,
    /// Mean gate activation for this path.
    pub gate_mean: f64,
    /// Mean absolute gate activation for this path.
    pub gate_abs_mean: f64,
    /// Conservative sparsity summary derived from absolute gate activation.
    pub gate_sparsity: f64,
    /// Fraction of gate elements effectively closed.
    #[serde(default)]
    pub gate_closed_fraction: f64,
    /// Fraction of gate elements effectively open.
    #[serde(default)]
    pub gate_open_fraction: f64,
    /// Fraction of gate elements near either saturation boundary.
    #[serde(default)]
    pub gate_saturation_fraction: f64,
    /// Number of gate elements summarized by this path.
    #[serde(default)]
    pub gate_element_count: usize,
    /// Entropy of normalized gate mass for scalar versus fine-grained gates.
    #[serde(default)]
    pub gate_entropy: f64,
    /// Mean sigmoid derivative proxy for gate gradient health.
    #[serde(default)]
    pub gate_gradient_proxy: f64,
    /// Whether this path was forced open by a negative-control ablation.
    #[serde(default)]
    pub forced_open: bool,
    /// Temporal or staged multiplier applied to the path update.
    #[serde(default)]
    pub path_scale: f64,
    /// Compact gate health label.
    #[serde(default)]
    pub gate_status: String,
    /// Optional warning for always-open, always-closed, or saturated gates.
    #[serde(default)]
    pub gate_warning: Option<String>,
    /// Mean attention entropy for this path.
    pub attention_entropy: f64,
    /// Mean norm of the effective path update after temporal scaling.
    #[serde(default)]
    pub effective_update_norm: f64,
    /// Optional staged-training index used by the interaction policy.
    #[serde(default)]
    pub training_stage_index: Option<usize>,
    /// Optional rollout step used by the interaction policy.
    #[serde(default)]
    pub rollout_step_index: Option<usize>,
    /// Optional flow time used by the interaction policy.
    #[serde(default)]
    pub flow_t: Option<f64>,
    /// Optional coarse flow-time bucket used for grouped diagnostics.
    #[serde(default)]
    pub flow_time_bucket: Option<String>,
}

impl Default for InteractionPathStepMetrics {
    fn default() -> Self {
        Self {
            path_name: String::new(),
            path_role: String::new(),
            gate_mean: 0.0,
            gate_abs_mean: 0.0,
            gate_sparsity: 1.0,
            gate_closed_fraction: 1.0,
            gate_open_fraction: 0.0,
            gate_saturation_fraction: 1.0,
            gate_element_count: 0,
            gate_entropy: 0.0,
            gate_gradient_proxy: 0.0,
            forced_open: false,
            path_scale: 1.0,
            gate_status: "always_closed".to_string(),
            gate_warning: None,
            attention_entropy: 0.0,
            effective_update_norm: 0.0,
            training_stage_index: None,
            rollout_step_index: None,
            flow_t: None,
            flow_time_bucket: None,
        }
    }
}

/// Gate usage summary for one directed path inside a coarse flow-time bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionFlowTimeBucketMetrics {
    /// Coarse flow-time bucket: low, mid, or high.
    pub bucket: String,
    /// Stable directed-path name.
    pub path_name: String,
    /// Number of examples contributing to this bucket/path pair.
    pub sample_count: usize,
    /// Mean gate activation for the bucket/path pair.
    pub mean_gate: f64,
    /// Mean gate sparsity for the bucket/path pair.
    pub mean_gate_sparsity: f64,
}

/// Interaction diagnostics aggregated into one training step record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionStepMetrics {
    /// Current staged-training phase, duplicated here for grouped diagnostics.
    #[serde(default)]
    pub stage: Option<TrainingStage>,
    /// Numeric stage index consumed by temporal interaction policies.
    #[serde(default)]
    pub stage_index: Option<usize>,
    /// One record per directed interaction path.
    #[serde(default)]
    pub paths: Vec<InteractionPathStepMetrics>,
    /// Flow-time bucket summaries for flow-matching interaction diagnostics.
    #[serde(default)]
    pub flow_time_buckets: Vec<InteractionFlowTimeBucketMetrics>,
    /// Mean gate activation across all directed paths.
    pub mean_gate: f64,
    /// Mean gate sparsity across all directed paths.
    pub mean_gate_sparsity: f64,
    /// Mean attention entropy across all directed paths.
    pub mean_attention_entropy: f64,
}

impl Default for InteractionStepMetrics {
    fn default() -> Self {
        Self {
            stage: None,
            stage_index: None,
            paths: Vec::new(),
            flow_time_buckets: Vec::new(),
            mean_gate: 0.0,
            mean_gate_sparsity: 1.0,
            mean_attention_entropy: 0.0,
        }
    }
}

impl InteractionStepMetrics {
    pub(crate) fn from_forwards(
        stage: TrainingStage,
        stage_index: Option<usize>,
        forwards: &[crate::models::system::ResearchForward],
    ) -> Self {
        if forwards.is_empty() {
            return Self {
                stage: Some(stage),
                stage_index,
                flow_time_buckets: Vec::new(),
                ..Self::default()
            };
        }

        let mut paths = Vec::with_capacity(6);
        for path_index in 0..6 {
            let first =
                interaction_path_metric_at(&forwards[0].interaction_diagnostics, path_index);
            let mut gate_mean = 0.0;
            let mut gate_abs_mean = 0.0;
            let mut gate_closed_fraction = 0.0;
            let mut gate_open_fraction = 0.0;
            let mut gate_saturation_fraction = 0.0;
            let mut gate_element_count = 0usize;
            let mut gate_entropy = 0.0;
            let mut gate_gradient_proxy = 0.0;
            let mut forced_open = first.forced_open;
            let mut path_scale = 0.0;
            let mut attention_entropy = 0.0;
            let mut effective_update_norm = 0.0;
            let mut training_stage_index = first.training_stage;
            let mut rollout_step_index = first.rollout_step_index;
            let mut flow_t = first.flow_t;
            let mut flow_time_bucket = first.flow_time_bucket.clone();

            for forward in forwards {
                let path = interaction_path_metric_at(&forward.interaction_diagnostics, path_index);
                gate_mean += path.gate_mean;
                gate_abs_mean += path.gate_abs_mean;
                gate_closed_fraction += path.gate_closed_fraction;
                gate_open_fraction += path.gate_open_fraction;
                gate_saturation_fraction += path.gate_saturation_fraction;
                gate_element_count += path.gate_element_count;
                gate_entropy += path.gate_entropy;
                gate_gradient_proxy += path.gate_gradient_proxy;
                forced_open |= path.forced_open;
                path_scale += path.path_scale;
                attention_entropy += path.attention_entropy;
                effective_update_norm += path.effective_update_norm;
                training_stage_index = training_stage_index.or(path.training_stage);
                rollout_step_index = rollout_step_index.or(path.rollout_step_index);
                flow_t = flow_t.or(path.flow_t);
                flow_time_bucket = flow_time_bucket.or_else(|| path.flow_time_bucket.clone());
            }

            let count = forwards.len() as f64;
            let gate_mean = gate_mean / count;
            let gate_abs_mean = gate_abs_mean / count;
            let gate_closed_fraction = gate_closed_fraction / count;
            let gate_open_fraction = gate_open_fraction / count;
            let gate_saturation_fraction = gate_saturation_fraction / count;
            let gate_entropy = gate_entropy / count;
            let gate_gradient_proxy = gate_gradient_proxy / count;
            let path_scale = path_scale / count;
            let attention_entropy = attention_entropy / count;
            let effective_update_norm = effective_update_norm / count;
            let gate_sparsity = (1.0 - gate_abs_mean.clamp(0.0, 1.0)).clamp(0.0, 1.0);
            let (gate_status, gate_warning) = gate_health_status(
                gate_closed_fraction,
                gate_open_fraction,
                gate_saturation_fraction,
                gate_gradient_proxy,
            );
            paths.push(InteractionPathStepMetrics {
                path_name: first.path_name.clone(),
                path_role: first.path_role.to_string(),
                gate_mean,
                gate_abs_mean,
                gate_sparsity,
                gate_closed_fraction,
                gate_open_fraction,
                gate_saturation_fraction,
                gate_element_count,
                gate_entropy,
                gate_gradient_proxy,
                forced_open,
                path_scale,
                gate_status,
                gate_warning,
                attention_entropy,
                effective_update_norm,
                training_stage_index,
                rollout_step_index,
                flow_t,
                flow_time_bucket,
            });
        }

        let flow_time_buckets = flow_time_bucket_metrics(forwards);
        let count = paths.len().max(1) as f64;
        let mean_gate = paths.iter().map(|path| path.gate_mean).sum::<f64>() / count;
        let mean_gate_sparsity = paths.iter().map(|path| path.gate_sparsity).sum::<f64>() / count;
        let mean_attention_entropy =
            paths.iter().map(|path| path.attention_entropy).sum::<f64>() / count;

        Self {
            stage: Some(stage),
            stage_index,
            paths,
            flow_time_buckets,
            mean_gate,
            mean_gate_sparsity,
            mean_attention_entropy,
        }
    }
}

fn flow_time_bucket_metrics(
    forwards: &[crate::models::system::ResearchForward],
) -> Vec<InteractionFlowTimeBucketMetrics> {
    let mut summaries = Vec::new();
    for bucket in ["low", "mid", "high"] {
        for path_index in 0..6 {
            let mut path_name = String::new();
            let mut gate_sum = 0.0;
            let mut sparsity_sum = 0.0;
            let mut count = 0_usize;

            for forward in forwards {
                let path = interaction_path_metric_at(&forward.interaction_diagnostics, path_index);
                let path_bucket = path
                    .flow_time_bucket
                    .as_deref()
                    .or_else(|| path.flow_t.map(flow_time_bucket_label));
                if path_bucket != Some(bucket) {
                    continue;
                }

                if path_name.is_empty() {
                    path_name = path.path_name.clone();
                }
                gate_sum += path.gate_mean;
                sparsity_sum += (1.0 - path.gate_abs_mean.clamp(0.0, 1.0)).clamp(0.0, 1.0);
                count += 1;
            }

            if count > 0 {
                summaries.push(InteractionFlowTimeBucketMetrics {
                    bucket: bucket.to_string(),
                    path_name,
                    sample_count: count,
                    mean_gate: gate_sum / count as f64,
                    mean_gate_sparsity: sparsity_sum / count as f64,
                });
            }
        }
    }
    summaries
}

fn flow_time_bucket_label(flow_t: f64) -> &'static str {
    if flow_t < (1.0 / 3.0) {
        "low"
    } else if flow_t < (2.0 / 3.0) {
        "mid"
    } else {
        "high"
    }
}

fn gate_health_status(
    closed_fraction: f64,
    open_fraction: f64,
    saturation_fraction: f64,
    gradient_proxy: f64,
) -> (String, Option<String>) {
    if closed_fraction >= 0.98 {
        (
            "always_closed".to_string(),
            Some("gate is effectively always closed for this path".to_string()),
        )
    } else if open_fraction >= 0.98 {
        (
            "always_open".to_string(),
            Some("gate is effectively always open for this path".to_string()),
        )
    } else if saturation_fraction >= 0.80 || gradient_proxy <= 0.02 {
        (
            "saturated".to_string(),
            Some("gate is near a saturation boundary; gradient signal may be weak".to_string()),
        )
    } else {
        ("active".to_string(), None)
    }
}

fn interaction_path_metric_at(
    diagnostics: &crate::models::interaction::CrossModalInteractionDiagnostics,
    index: usize,
) -> &crate::models::interaction::CrossModalInteractionPathDiagnostics {
    match index {
        0 => &diagnostics.topo_from_geo,
        1 => &diagnostics.topo_from_pocket,
        2 => &diagnostics.geo_from_topo,
        3 => &diagnostics.geo_from_pocket,
        4 => &diagnostics.pocket_from_topo,
        _ => &diagnostics.pocket_from_geo,
    }
}

/// Cheap scalar synchronization-health diagnostics for one optimizer step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationHealthMetrics {
    /// Number of examples whose topology/geometry/pocket mask counts disagree with atom counts.
    pub mask_count_mismatch: usize,
    /// Number of examples whose modality slot counts disagree with the fixed upper bound.
    pub slot_count_mismatch: usize,
    /// Number of examples with an invalid shared coordinate-frame origin.
    pub coordinate_frame_mismatch: usize,
    /// Maximum stale context steps observed across rollout records in this step.
    pub stale_context_steps: usize,
    /// Maximum context refresh count observed across rollout records in this step.
    pub refresh_count: usize,
    /// Whether per-example batch slices retained synchronized ids, counts, and diagnostics.
    pub batch_slice_sync_pass: bool,
}

impl Default for SynchronizationHealthMetrics {
    fn default() -> Self {
        Self {
            mask_count_mismatch: 0,
            slot_count_mismatch: 0,
            coordinate_frame_mismatch: 0,
            stale_context_steps: 0,
            refresh_count: 0,
            batch_slice_sync_pass: true,
        }
    }
}

impl SynchronizationHealthMetrics {
    pub(crate) fn from_forwards(forwards: &[crate::models::system::ResearchForward]) -> Self {
        let mut metrics = Self::default();
        let mut batch_slice_sync_pass = true;

        for forward in forwards {
            let sync = &forward.sync_context;
            if mask_counts_mismatch(sync) {
                metrics.mask_count_mismatch += 1;
            }
            if slot_counts_mismatch(sync) {
                metrics.slot_count_mismatch += 1;
            }
            if coordinate_frame_mismatch(sync) {
                metrics.coordinate_frame_mismatch += 1;
            }

            let rollout = &forward.generation.rollout;
            metrics.stale_context_steps =
                metrics.stale_context_steps.max(rollout.stale_context_steps);
            metrics.refresh_count = metrics.refresh_count.max(rollout.refresh_count);

            if sync.example_id.is_empty()
                || sync.protein_id.is_empty()
                || sync.example_id != rollout.example_id
                || sync.protein_id != rollout.protein_id
                || !all_path_diagnostics_are_per_example(forward)
            {
                batch_slice_sync_pass = false;
            }
        }

        metrics.batch_slice_sync_pass = batch_slice_sync_pass
            && metrics.mask_count_mismatch == 0
            && metrics.slot_count_mismatch == 0
            && metrics.coordinate_frame_mismatch == 0;
        metrics
    }
}

fn mask_counts_mismatch(sync: &crate::models::system::ModalitySyncContext) -> bool {
    sync.ligand_atom_count < 0
        || sync.pocket_atom_count < 0
        || sync.topology_mask_count != sync.ligand_atom_count
        || sync.geometry_mask_count != sync.ligand_atom_count
        || sync.pocket_mask_count != sync.pocket_atom_count
}

fn slot_counts_mismatch(sync: &crate::models::system::ModalitySyncContext) -> bool {
    sync.topology_slot_count <= 0
        || sync.geometry_slot_count <= 0
        || sync.pocket_slot_count <= 0
        || sync.topology_slot_count != sync.geometry_slot_count
        || sync.topology_slot_count != sync.pocket_slot_count
}

fn coordinate_frame_mismatch(sync: &crate::models::system::ModalitySyncContext) -> bool {
    !sync
        .coordinate_frame_origin
        .iter()
        .all(|value| value.is_finite())
}

fn all_path_diagnostics_are_per_example(forward: &crate::models::system::ResearchForward) -> bool {
    (0..6).all(|index| {
        interaction_path_metric_at(&forward.interaction_diagnostics, index).provenance
            == crate::models::interaction::InteractionDiagnosticProvenance::PerExample
    })
}

/// Compact per-modality slot signature summary for one optimization step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotSignatureStepSummary {
    /// Topology, geometry, or pocket.
    pub modality: String,
    /// Optional staged-training index attached by the trainer.
    #[serde(default)]
    pub stage_index: Option<usize>,
    /// Number of examples contributing to this modality summary.
    pub sample_count: usize,
    /// Fixed upper-bound slot count observed for this modality.
    pub slot_count: usize,
    /// Mean independent activation fraction for active examples.
    pub active_slot_fraction: f64,
    /// Mean attention-visible slot fraction after masking.
    pub attention_visible_slot_fraction: f64,
    /// Mean assignment entropy over retained slot weights.
    pub assignment_entropy: f64,
    /// Lightweight semantic-probe alignment proxy for the same modality.
    pub semantic_probe_alignment: f64,
    /// Permutation-aware matching score when a multi-seed caller supplies one.
    #[serde(default)]
    pub cross_seed_matching_score: Option<f64>,
    /// Local proxy scope used when no cross-seed comparison is available.
    #[serde(default)]
    pub matching_scope: String,
    /// Local first/last example signature cosine proxy.
    #[serde(default)]
    pub matching_score: f64,
    /// Bounded mean slot signature vector for summary artifacts.
    #[serde(default)]
    pub signature: Vec<f64>,
}

impl Default for SlotSignatureStepSummary {
    fn default() -> Self {
        Self {
            modality: String::new(),
            stage_index: None,
            sample_count: 0,
            slot_count: 0,
            active_slot_fraction: 0.0,
            attention_visible_slot_fraction: 0.0,
            assignment_entropy: 0.0,
            semantic_probe_alignment: 0.0,
            cross_seed_matching_score: None,
            matching_scope: "unavailable".to_string(),
            matching_score: 0.0,
            signature: Vec::new(),
        }
    }
}

/// Compact slot-utilization diagnostics for one optimization step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotUtilizationStepMetrics {
    /// Optional staged-training index used for stage-aware collapse diagnostics.
    #[serde(default)]
    pub stage_index: Option<usize>,
    /// Mean active slot count across modalities and examples.
    pub mean_active_slot_count: f64,
    /// Mean independent activation fraction across modalities and examples.
    pub mean_active_slot_fraction: f64,
    /// Mean attention-visible slot fraction after active-slot masking.
    pub mean_attention_visible_slot_fraction: f64,
    /// Mean slot-utilization entropy across modalities and examples.
    pub mean_slot_entropy: f64,
    /// Fraction of slots whose activation gate is effectively off.
    pub dead_slot_fraction: f64,
    /// Number of effectively dead slots by mean activation bucket.
    #[serde(default)]
    pub dead_slot_count: usize,
    /// Number of weakly used diffuse slots by mean activation bucket.
    #[serde(default)]
    pub diffuse_slot_count: usize,
    /// Number of nearly always-active slots by mean activation bucket.
    #[serde(default)]
    pub saturated_slot_count: usize,
    /// Number of modality/example diagnostics with collapse-like slot usage.
    pub collapse_warning_count: usize,
    /// Compact warning labels suitable for JSON summaries.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Per-modality slot signatures for active, non-disabled modality branches.
    #[serde(default)]
    pub slot_signatures: Vec<SlotSignatureStepSummary>,
}

impl Default for SlotUtilizationStepMetrics {
    fn default() -> Self {
        Self {
            stage_index: None,
            mean_active_slot_count: 0.0,
            mean_active_slot_fraction: 0.0,
            mean_attention_visible_slot_fraction: 0.0,
            mean_slot_entropy: 0.0,
            dead_slot_fraction: 0.0,
            dead_slot_count: 0,
            diffuse_slot_count: 0,
            saturated_slot_count: 0,
            collapse_warning_count: 0,
            warnings: Vec::new(),
            slot_signatures: Vec::new(),
        }
    }
}

impl SlotUtilizationStepMetrics {
    pub(crate) fn from_forwards(forwards: &[crate::models::system::ResearchForward]) -> Self {
        if forwards.is_empty() {
            return Self::default();
        }

        let mut active_count = 0.0;
        let mut active_fraction = 0.0;
        let mut visible_fraction = 0.0;
        let mut entropy = 0.0;
        let mut dead_slots = 0.0;
        let mut slot_total = 0.0;
        let mut dead_slot_count = 0usize;
        let mut diffuse_slot_count = 0usize;
        let mut saturated_slot_count = 0usize;
        let mut collapse_warning_count = 0usize;
        let mut warnings = Vec::new();
        let mut observations = 0.0_f64;
        let mut signature_accumulators = [
            SlotSignatureAccumulator::new("topology"),
            SlotSignatureAccumulator::new("geometry"),
            SlotSignatureAccumulator::new("pocket"),
        ];

        for forward in forwards {
            observe_slot_utilization_branch(
                &forward.diagnostics.topology,
                &forward.slots.topology,
                slot_probe_alignment("topology", forward, &forward.slots.topology),
                &mut signature_accumulators[0],
                &mut active_count,
                &mut active_fraction,
                &mut visible_fraction,
                &mut entropy,
                &mut dead_slots,
                &mut slot_total,
                &mut dead_slot_count,
                &mut diffuse_slot_count,
                &mut saturated_slot_count,
                &mut collapse_warning_count,
                &mut warnings,
                &mut observations,
            );
            observe_slot_utilization_branch(
                &forward.diagnostics.geometry,
                &forward.slots.geometry,
                slot_probe_alignment("geometry", forward, &forward.slots.geometry),
                &mut signature_accumulators[1],
                &mut active_count,
                &mut active_fraction,
                &mut visible_fraction,
                &mut entropy,
                &mut dead_slots,
                &mut slot_total,
                &mut dead_slot_count,
                &mut diffuse_slot_count,
                &mut saturated_slot_count,
                &mut collapse_warning_count,
                &mut warnings,
                &mut observations,
            );
            observe_slot_utilization_branch(
                &forward.diagnostics.pocket,
                &forward.slots.pocket,
                slot_probe_alignment("pocket", forward, &forward.slots.pocket),
                &mut signature_accumulators[2],
                &mut active_count,
                &mut active_fraction,
                &mut visible_fraction,
                &mut entropy,
                &mut dead_slots,
                &mut slot_total,
                &mut dead_slot_count,
                &mut diffuse_slot_count,
                &mut saturated_slot_count,
                &mut collapse_warning_count,
                &mut warnings,
                &mut observations,
            );
        }

        let denom = observations.max(1.0);
        Self {
            stage_index: None,
            mean_active_slot_count: active_count / denom,
            mean_active_slot_fraction: active_fraction / denom,
            mean_attention_visible_slot_fraction: visible_fraction / denom,
            mean_slot_entropy: entropy / denom,
            dead_slot_fraction: dead_slots / slot_total.max(1.0),
            dead_slot_count,
            diffuse_slot_count,
            saturated_slot_count,
            collapse_warning_count,
            warnings,
            slot_signatures: signature_accumulators
                .into_iter()
                .filter_map(SlotSignatureAccumulator::finish)
                .collect(),
        }
    }

    pub(crate) fn with_stage_index(mut self, stage_index: usize) -> Self {
        self.stage_index = Some(stage_index);
        for warning in &mut self.warnings {
            if !warning.starts_with("stage") {
                *warning = format!("stage{stage_index}_{warning}");
            }
        }
        for signature in &mut self.slot_signatures {
            signature.stage_index = Some(stage_index);
        }
        self
    }
}

#[allow(clippy::too_many_arguments)]
fn observe_slot_utilization_branch(
    branch: &crate::models::semantic::SemanticBranchDiagnostics,
    slots_encoding: &crate::models::SlotEncoding,
    probe_alignment: f64,
    accumulator: &mut SlotSignatureAccumulator,
    active_count: &mut f64,
    active_fraction: &mut f64,
    visible_fraction: &mut f64,
    entropy: &mut f64,
    dead_slots: &mut f64,
    slot_total: &mut f64,
    dead_slot_count: &mut usize,
    diffuse_slot_count: &mut usize,
    saturated_slot_count: &mut usize,
    collapse_warning_count: &mut usize,
    warnings: &mut Vec<String>,
    observations: &mut f64,
) {
    if slot_modality_disabled_by_ablation(slots_encoding) {
        return;
    }
    let slots = branch.slot_count.max(0) as f64;
    *active_count += branch.active_slot_count;
    *active_fraction += branch.active_slot_fraction;
    *visible_fraction += branch.attention_visible_slot_fraction;
    *entropy += branch.slot_entropy;
    *dead_slots += branch.dead_slot_count;
    *slot_total += slots;
    *observations += 1.0;
    accumulator.observe(branch, slots_encoding, probe_alignment);
    let bucket_counts = slot_activation_bucket_counts(&slots_encoding.slot_activations);
    *dead_slot_count += bucket_counts.dead;
    *diffuse_slot_count += bucket_counts.diffuse;
    *saturated_slot_count += bucket_counts.saturated;

    let collapsed = slots > 0.0
        && (branch.active_slot_count <= 0.0
            || branch.dead_slot_count >= slots
            || branch.slot_entropy <= 1.0e-6
            || bucket_counts.dead >= slots as usize
            || bucket_counts.saturated >= slots as usize);
    if collapsed {
        *collapse_warning_count += 1;
        if warnings.len() < 8 {
            warnings.push(format!("{}_slot_collapse", branch.modality));
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SlotActivationBucketCounts {
    dead: usize,
    diffuse: usize,
    saturated: usize,
}

fn slot_activation_bucket_counts(slot_activations: &tch::Tensor) -> SlotActivationBucketCounts {
    let mut counts = SlotActivationBucketCounts::default();
    let slot_count = slot_activations
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize;
    for slot in 0..slot_count.min(slot_activations.numel()) {
        let activation = slot_activations
            .double_value(&[slot as i64])
            .clamp(0.0, 1.0);
        if activation <= 0.05 {
            counts.dead += 1;
        } else if activation >= 0.95 {
            counts.saturated += 1;
        } else if activation < 0.50 {
            counts.diffuse += 1;
        }
    }
    counts
}

struct SlotSignatureAccumulator {
    modality: &'static str,
    sample_count: usize,
    slot_count: usize,
    active_slot_fraction_sum: f64,
    attention_visible_slot_fraction_sum: f64,
    assignment_entropy_sum: f64,
    semantic_probe_alignment_sum: f64,
    signature_sum: Vec<f64>,
    first_signature: Option<Vec<f64>>,
    last_signature: Option<Vec<f64>>,
}

impl SlotSignatureAccumulator {
    fn new(modality: &'static str) -> Self {
        Self {
            modality,
            sample_count: 0,
            slot_count: 0,
            active_slot_fraction_sum: 0.0,
            attention_visible_slot_fraction_sum: 0.0,
            assignment_entropy_sum: 0.0,
            semantic_probe_alignment_sum: 0.0,
            signature_sum: Vec::new(),
            first_signature: None,
            last_signature: None,
        }
    }

    fn observe(
        &mut self,
        branch: &crate::models::semantic::SemanticBranchDiagnostics,
        slots: &crate::models::SlotEncoding,
        semantic_probe_alignment: f64,
    ) {
        let signature = compact_slot_signature(&slots.slots);
        if self.signature_sum.len() < signature.len() {
            self.signature_sum.resize(signature.len(), 0.0);
        }
        for (index, value) in signature.iter().enumerate() {
            self.signature_sum[index] += *value;
        }
        if self.first_signature.is_none() {
            self.first_signature = Some(signature.clone());
        }
        self.last_signature = Some(signature);
        self.sample_count += 1;
        self.slot_count = self.slot_count.max(branch.slot_count.max(0) as usize);
        self.active_slot_fraction_sum += branch.active_slot_fraction;
        self.attention_visible_slot_fraction_sum += branch.attention_visible_slot_fraction;
        self.assignment_entropy_sum += slot_weight_entropy(&slots.slot_weights);
        self.semantic_probe_alignment_sum += semantic_probe_alignment;
    }

    fn finish(self) -> Option<SlotSignatureStepSummary> {
        if self.sample_count == 0 {
            return None;
        }
        let denom = self.sample_count as f64;
        let signature = self
            .signature_sum
            .into_iter()
            .map(|value| value / denom)
            .collect::<Vec<_>>();
        let matching_score = match (&self.first_signature, &self.last_signature) {
            (Some(first), Some(last)) => vector_cosine_similarity(first, last),
            _ => 0.0,
        };
        Some(SlotSignatureStepSummary {
            modality: self.modality.to_string(),
            stage_index: None,
            sample_count: self.sample_count,
            slot_count: self.slot_count,
            active_slot_fraction: self.active_slot_fraction_sum / denom,
            attention_visible_slot_fraction: self.attention_visible_slot_fraction_sum / denom,
            assignment_entropy: self.assignment_entropy_sum / denom,
            semantic_probe_alignment: self.semantic_probe_alignment_sum / denom,
            cross_seed_matching_score: None,
            matching_scope: "within_step_repeated_signature_proxy".to_string(),
            matching_score,
            signature,
        })
    }
}

fn slot_modality_disabled_by_ablation(slots: &crate::models::SlotEncoding) -> bool {
    slots.slot_weights.numel() == 0
        || (slots.slot_weights.abs().sum(tch::Kind::Float).double_value(&[]) <= 1.0e-12
            && slots.slots.abs().sum(tch::Kind::Float).double_value(&[]) <= 1.0e-12
            && slots
                .slot_activations
                .abs()
                .sum(tch::Kind::Float)
                .double_value(&[])
                <= 1.0e-12)
}

fn compact_slot_signature(slots: &tch::Tensor) -> Vec<f64> {
    if slots.numel() == 0 || slots.size().len() != 2 {
        return Vec::new();
    }
    let hidden_dim = slots.size().get(1).copied().unwrap_or(0).max(0).min(8) as usize;
    if hidden_dim == 0 {
        return Vec::new();
    }
    let mean = slots.mean_dim([0].as_slice(), false, tch::Kind::Float);
    (0..hidden_dim)
        .map(|index| mean.double_value(&[index as i64]))
        .collect()
}

fn slot_weight_entropy(weights: &tch::Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    let normalized = (weights / weights.sum(tch::Kind::Float).clamp_min(1e-12)).clamp_min(1e-12);
    (-(&normalized * normalized.log()).sum(tch::Kind::Float)).double_value(&[])
}

fn slot_probe_alignment(
    modality: &str,
    forward: &crate::models::system::ResearchForward,
    slots: &crate::models::SlotEncoding,
) -> f64 {
    match modality {
        "topology" if forward.probes.topology_adjacency_logits.numel() > 0 => {
            let probe = forward
                .probes
                .topology_adjacency_logits
                .sigmoid()
                .mean(tch::Kind::Float)
                .double_value(&[]);
            probe * active_slot_fraction_tensor(&slots.slot_activations)
        }
        "geometry" if forward.probes.geometry_distance_predictions.numel() > 0 => {
            let probe = 1.0
                / (1.0
                    + forward
                        .probes
                        .geometry_distance_predictions
                        .abs()
                        .mean(tch::Kind::Float)
                        .double_value(&[]));
            probe * active_slot_fraction_tensor(&slots.slot_activations)
        }
        "pocket" if forward.probes.pocket_feature_predictions.numel() > 0 => {
            let probe = 1.0
                / (1.0
                    + forward
                        .probes
                        .pocket_feature_predictions
                        .abs()
                        .mean(tch::Kind::Float)
                        .double_value(&[]));
            probe * active_slot_fraction_tensor(&slots.slot_activations)
        }
        _ => 0.0,
    }
}

fn active_slot_fraction_tensor(values: &tch::Tensor) -> f64 {
    if values.numel() == 0 {
        return 0.0;
    }
    values
        .gt(0.05)
        .to_kind(tch::Kind::Float)
        .mean(tch::Kind::Float)
        .double_value(&[])
}

fn vector_cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    let len = left.len().min(right.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for index in 0..len {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
    }
    dot / (left_norm.sqrt() * right_norm.sqrt()).max(1e-6)
}

/// One trainer step record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Global optimization step.
    pub step: usize,
    /// Explicit generation-mode contract active for this optimizer step.
    #[serde(default = "default_step_generation_mode")]
    pub generation_mode: String,
    /// Zero-based sampler epoch used for this step.
    #[serde(default)]
    pub epoch_index: usize,
    /// Effective sample-order seed used for this epoch.
    #[serde(default)]
    pub sample_order_seed: u64,
    /// Source example indices included in this mini-batch.
    #[serde(default)]
    pub batch_sample_indices: Vec<usize>,
    /// Current staged-training phase.
    pub stage: TrainingStage,
    /// Stage ramp and active objective-family summary.
    #[serde(default)]
    pub stage_progress: StageProgressMetrics,
    /// Loss values for this step.
    pub losses: LossBreakdown,
    /// Stage-aware directed interaction path diagnostics.
    #[serde(default)]
    pub interaction: InteractionStepMetrics,
    /// Scalar synchronization-health diagnostics for modality and rollout alignment.
    #[serde(default)]
    pub synchronization: SynchronizationHealthMetrics,
    /// Compact slot utilization and collapse diagnostics.
    #[serde(default)]
    pub slot_utilization: SlotUtilizationStepMetrics,
    /// Gradient and numerical-health diagnostics for this optimizer step.
    #[serde(default)]
    pub gradient_health: GradientHealthMetrics,
    /// Per-step training runtime and memory profile.
    #[serde(default)]
    pub runtime_profile: TrainingRuntimeProfileMetrics,
}

fn default_step_generation_mode() -> String {
    crate::config::GenerationModeConfig::TargetLigandDenoising
        .as_str()
        .to_string()
}

/// Coarse training stage aligned with the research schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingStage {
    /// Stage 1: task and consistency only.
    Stage1,
    /// Stage 2: add redundancy reduction.
    Stage2,
    /// Stage 3: add probes and leakage.
    Stage3,
    /// Stage 4: add gate and slot control.
    Stage4,
}

impl TrainingStage {
    /// Numeric index used by temporal interaction policy configs.
    pub const fn index(self) -> usize {
        match self {
            Self::Stage1 => 0,
            Self::Stage2 => 1,
            Self::Stage3 => 2,
            Self::Stage4 => 3,
        }
    }
}
