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
    /// Target/provenance source consumed by this component.
    #[serde(default = "default_primary_component_target_source")]
    pub target_source: String,
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

/// Stable metadata for one primary-objective component name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimaryObjectiveComponentDescriptor {
    /// High-level source family for the component.
    pub anchor: &'static str,
    /// Target/provenance source consumed by this component.
    pub target_source: &'static str,
    /// Whether this value remains connected to differentiable tensor operations.
    pub differentiable: bool,
    /// Whether this value is allowed to contribute to the optimizer-facing total.
    pub optimizer_facing: bool,
    /// Human-readable interpretation of the component role.
    pub role: &'static str,
    /// Flow branch that owns this component when branch schedules are reported.
    pub branch_name: Option<&'static str>,
    /// Short audit boundary used by objective coverage artifacts.
    pub claim_boundary: &'static str,
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

/// Audit summary for primary components owned by one scheduled flow branch.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PrimaryBranchComponentAudit {
    /// Stable branch name.
    #[serde(default)]
    pub branch_name: String,
    /// Observed primary component names assigned to this branch by the component registry.
    #[serde(default)]
    pub observed_component_names: Vec<String>,
    /// Number of observed primary components assigned to this branch.
    #[serde(default)]
    pub observed_component_count: usize,
    /// Number of observed components allowed to contribute to optimizer-facing totals.
    #[serde(default)]
    pub optimizer_facing_component_count: usize,
    /// Number of observed diagnostic-only components assigned to this branch.
    #[serde(default)]
    pub diagnostic_component_count: usize,
    /// Sum of emitted component scalar values for this branch; audit-only because some subterms are nested decompositions.
    #[serde(default)]
    pub observed_component_value_sum: f64,
    /// Sum of emitted optimizer-facing component scalar values for this branch.
    #[serde(default)]
    pub optimizer_facing_component_value_sum: f64,
    /// Sum of emitted diagnostic-only component scalar values for this branch.
    #[serde(default)]
    pub diagnostic_component_value_sum: f64,
}

impl PrimaryBranchComponentAudit {
    /// Build an empty audit record for a scheduled branch.
    pub fn for_branch(branch_name: impl Into<String>) -> Self {
        Self {
            branch_name: branch_name.into(),
            ..Self::default()
        }
    }

    fn observe_component(
        &mut self,
        component_name: &'static str,
        value: f64,
        descriptor: PrimaryObjectiveComponentDescriptor,
    ) {
        self.observed_component_names
            .push(component_name.to_string());
        self.observed_component_count += 1;
        self.observed_component_value_sum += value;
        if descriptor.optimizer_facing {
            self.optimizer_facing_component_count += 1;
            self.optimizer_facing_component_value_sum += value;
        } else {
            self.diagnostic_component_count += 1;
            self.diagnostic_component_value_sum += value;
        }
    }
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
    /// Registry-derived audit of primary components owned by this branch.
    #[serde(default)]
    pub component_audit: PrimaryBranchComponentAudit,
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
    /// Sparse negative-class density calibration subterm inside `flow_bond`.
    #[serde(default)]
    pub flow_bond_sparse_negative_rate: Option<f64>,
    /// Bounded native graph confidence pressure subterm inside `flow_bond`.
    #[serde(default)]
    pub flow_bond_confidence_pressure: Option<f64>,
    /// Expected-degree alignment subterm inside `flow_bond`.
    #[serde(default)]
    pub flow_bond_degree_alignment: Option<f64>,
    /// Topology synchronization flow contribution.
    #[serde(default)]
    pub flow_topology: Option<f64>,
    /// Sparse negative-class density calibration subterm inside `flow_topology`.
    #[serde(default)]
    pub flow_topology_sparse_negative_rate: Option<f64>,
    /// Bounded native graph confidence pressure subterm inside `flow_topology`.
    #[serde(default)]
    pub flow_topology_confidence_pressure: Option<f64>,
    /// Expected-degree alignment subterm inside `flow_topology`.
    #[serde(default)]
    pub flow_topology_degree_alignment: Option<f64>,
    /// Joint bond/topology native extraction-score calibration contribution.
    #[serde(default)]
    pub flow_native_score_calibration: Option<f64>,
    /// Uncapped native score calibration diagnostic before the max-loss safety cap.
    #[serde(default)]
    pub flow_native_score_calibration_uncapped_raw: Option<f64>,
    /// Ratio between capped and uncapped native score calibration raw loss.
    #[serde(default)]
    pub flow_native_score_calibration_cap_scale: Option<f64>,
    /// Native score calibration subterm for negative pairs above the extraction ceiling.
    #[serde(default)]
    pub flow_native_score_calibration_false_positive_margin: Option<f64>,
    /// Native score calibration subterm for positive pairs below the extraction floor.
    #[serde(default)]
    pub flow_native_score_calibration_false_negative_margin: Option<f64>,
    /// Native score calibration subterm for dense score mass above the target budget.
    #[serde(default)]
    pub flow_native_score_calibration_density_budget: Option<f64>,
    /// Native score calibration subterm for soft-thresholded positive target misses.
    #[serde(default)]
    pub flow_native_score_calibration_soft_positive_miss: Option<f64>,
    /// Native score calibration subterm for soft-thresholded negative-pair extraction.
    #[serde(default)]
    pub flow_native_score_calibration_soft_negative_extraction: Option<f64>,
    /// Native score calibration subterm for soft extraction mass above the target budget.
    #[serde(default)]
    pub flow_native_score_calibration_soft_extraction_budget: Option<f64>,
    /// Native score calibration subterm for per-atom score-degree mismatch.
    #[serde(default)]
    pub flow_native_score_calibration_degree_alignment: Option<f64>,
    /// Native score calibration subterm for positive-vs-negative score separation.
    #[serde(default)]
    pub flow_native_score_calibration_score_separation: Option<f64>,
    /// Ligand-pocket contact interaction-profile flow contribution.
    #[serde(default)]
    pub flow_pocket_context: Option<f64>,
    /// Cross-branch molecular flow synchronization contribution.
    #[serde(default)]
    pub flow_synchronization: Option<f64>,
}

impl PrimaryObjectiveComponentMetrics {
    /// Add observed component values from another metric bundle in-place.
    pub(crate) fn add_assign(&mut self, other: &Self) {
        other.for_each_observed_component(|component_name, value| {
            self.add_component_value(component_name, value);
        });
    }

    /// Return a metric bundle with every observed component multiplied by `factor`.
    pub(crate) fn scale(&self, factor: f64) -> Self {
        let mut scaled = Self::default();
        self.for_each_observed_component(|component_name, value| {
            scaled.add_component_value(component_name, value * factor);
        });
        scaled
    }

    /// Return provenance records for components observed in this metric bundle.
    pub fn provenance_records(&self) -> Vec<PrimaryObjectiveComponentProvenance> {
        let mut records = Vec::new();
        self.for_each_observed_component(|component_name, _| {
            records.push(primary_objective_component_provenance_record(component_name));
        });
        records
    }

    /// Return observed component names and values in stable report order.
    pub fn observed_component_values(&self) -> Vec<(&'static str, f64)> {
        let mut values = Vec::new();
        self.for_each_observed_component(|component_name, value| {
            values.push((component_name, value));
        });
        values
    }

    /// Group observed primary components by registry-owned flow branch.
    pub fn branch_component_audits(&self) -> Vec<PrimaryBranchComponentAudit> {
        let mut audits: BTreeMap<&'static str, PrimaryBranchComponentAudit> = BTreeMap::new();
        self.for_each_observed_component(|component_name, value| {
            let descriptor = primary_objective_component_descriptor(component_name);
            let Some(branch_name) = descriptor.branch_name else {
                return;
            };
            audits
                .entry(branch_name)
                .or_insert_with(|| PrimaryBranchComponentAudit::for_branch(branch_name))
                .observe_component(component_name, value, descriptor);
        });
        audits.into_values().collect()
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
                    optimizer_facing: primary_objective_component_descriptor(component_name)
                        .optimizer_facing,
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

    fn for_each_observed_component(&self, mut visit: impl FnMut(&'static str, f64)) {
        macro_rules! visit_component {
            ($field:ident) => {
                if let Some(value) = self.$field {
                    visit(stringify!($field), value);
                }
            };
        }

        visit_component!(topology);
        visit_component!(geometry);
        visit_component!(pocket_anchor);
        visit_component!(rollout);
        visit_component!(rollout_eval_recovery);
        visit_component!(rollout_eval_pocket_anchor);
        visit_component!(rollout_eval_stop);
        visit_component!(flow_velocity);
        visit_component!(flow_endpoint);
        visit_component!(flow_atom_type);
        visit_component!(flow_bond);
        visit_component!(flow_bond_sparse_negative_rate);
        visit_component!(flow_bond_confidence_pressure);
        visit_component!(flow_bond_degree_alignment);
        visit_component!(flow_topology);
        visit_component!(flow_topology_sparse_negative_rate);
        visit_component!(flow_topology_confidence_pressure);
        visit_component!(flow_topology_degree_alignment);
        visit_component!(flow_native_score_calibration);
        visit_component!(flow_native_score_calibration_uncapped_raw);
        visit_component!(flow_native_score_calibration_cap_scale);
        visit_component!(flow_native_score_calibration_false_positive_margin);
        visit_component!(flow_native_score_calibration_false_negative_margin);
        visit_component!(flow_native_score_calibration_density_budget);
        visit_component!(flow_native_score_calibration_soft_positive_miss);
        visit_component!(flow_native_score_calibration_soft_negative_extraction);
        visit_component!(flow_native_score_calibration_soft_extraction_budget);
        visit_component!(flow_native_score_calibration_degree_alignment);
        visit_component!(flow_native_score_calibration_score_separation);
        visit_component!(flow_pocket_context);
        visit_component!(flow_synchronization);
    }

    fn add_component_value(&mut self, component_name: &'static str, value: f64) {
        macro_rules! add_component {
            ($field:ident) => {{
                self.$field = Some(self.$field.unwrap_or(0.0) + value);
            }};
        }

        match component_name {
            "topology" => add_component!(topology),
            "geometry" => add_component!(geometry),
            "pocket_anchor" => add_component!(pocket_anchor),
            "rollout" => add_component!(rollout),
            "rollout_eval_recovery" => add_component!(rollout_eval_recovery),
            "rollout_eval_pocket_anchor" => add_component!(rollout_eval_pocket_anchor),
            "rollout_eval_stop" => add_component!(rollout_eval_stop),
            "flow_velocity" => add_component!(flow_velocity),
            "flow_endpoint" => add_component!(flow_endpoint),
            "flow_atom_type" => add_component!(flow_atom_type),
            "flow_bond" => add_component!(flow_bond),
            "flow_bond_sparse_negative_rate" => add_component!(flow_bond_sparse_negative_rate),
            "flow_bond_confidence_pressure" => add_component!(flow_bond_confidence_pressure),
            "flow_bond_degree_alignment" => add_component!(flow_bond_degree_alignment),
            "flow_topology" => add_component!(flow_topology),
            "flow_topology_sparse_negative_rate" => {
                add_component!(flow_topology_sparse_negative_rate)
            }
            "flow_topology_confidence_pressure" => {
                add_component!(flow_topology_confidence_pressure)
            }
            "flow_topology_degree_alignment" => add_component!(flow_topology_degree_alignment),
            "flow_native_score_calibration" => add_component!(flow_native_score_calibration),
            "flow_native_score_calibration_uncapped_raw" => {
                add_component!(flow_native_score_calibration_uncapped_raw)
            }
            "flow_native_score_calibration_cap_scale" => {
                add_component!(flow_native_score_calibration_cap_scale)
            }
            "flow_native_score_calibration_false_positive_margin" => {
                add_component!(flow_native_score_calibration_false_positive_margin)
            }
            "flow_native_score_calibration_false_negative_margin" => {
                add_component!(flow_native_score_calibration_false_negative_margin)
            }
            "flow_native_score_calibration_density_budget" => {
                add_component!(flow_native_score_calibration_density_budget)
            }
            "flow_native_score_calibration_soft_positive_miss" => {
                add_component!(flow_native_score_calibration_soft_positive_miss)
            }
            "flow_native_score_calibration_soft_negative_extraction" => {
                add_component!(flow_native_score_calibration_soft_negative_extraction)
            }
            "flow_native_score_calibration_soft_extraction_budget" => {
                add_component!(flow_native_score_calibration_soft_extraction_budget)
            }
            "flow_native_score_calibration_degree_alignment" => {
                add_component!(flow_native_score_calibration_degree_alignment)
            }
            "flow_native_score_calibration_score_separation" => {
                add_component!(flow_native_score_calibration_score_separation)
            }
            "flow_pocket_context" => add_component!(flow_pocket_context),
            "flow_synchronization" => add_component!(flow_synchronization),
            _ => debug_assert!(false, "unknown primary component {component_name}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PrimaryObjectiveComponentSpec {
    component_name: &'static str,
    descriptor: PrimaryObjectiveComponentDescriptor,
}

impl PrimaryObjectiveComponentSpec {
    const fn trainable(
        component_name: &'static str,
        anchor: &'static str,
        target_source: &'static str,
        branch_name: Option<&'static str>,
        claim_boundary: &'static str,
    ) -> Self {
        Self {
            component_name,
            descriptor: PrimaryObjectiveComponentDescriptor {
                anchor,
                target_source,
                differentiable: true,
                optimizer_facing: true,
                role: "trainable_objective",
                branch_name,
                claim_boundary,
            },
        }
    }

    const fn diagnostic(
        component_name: &'static str,
        anchor: &'static str,
        target_source: &'static str,
        branch_name: Option<&'static str>,
        claim_boundary: &'static str,
    ) -> Self {
        Self {
            component_name,
            descriptor: PrimaryObjectiveComponentDescriptor {
                anchor,
                target_source,
                differentiable: false,
                optimizer_facing: false,
                role: "evaluation_only",
                branch_name,
                claim_boundary,
            },
        }
    }
}

const TENSOR_PRIMARY_COMPONENT_BOUNDARY: &str = "tensor-preserving primary objective component";
const ROLLOUT_EVAL_COMPONENT_BOUNDARY: &str =
    "detached sampled-rollout diagnostic; not optimizer-facing unless a future tensor-preserving trainable_rollout_* objective is implemented";
const FLOW_PRIMARY_COMPONENT_BOUNDARY: &str =
    "flow component is optimizer-facing only for flow-compatible primary objectives";
const NATIVE_SCORE_CAP_AUDIT_BOUNDARY: &str =
    "native-score calibration cap audit; diagnostic only and excluded from optimizer-facing totals";

const PRIMARY_OBJECTIVE_COMPONENT_SPECS: &[PrimaryObjectiveComponentSpec] = &[
    PrimaryObjectiveComponentSpec::trainable(
        "topology",
        "decoder_or_surrogate_topology",
        "reference_ligand",
        None,
        TENSOR_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "geometry",
        "decoder_or_surrogate_geometry",
        "reference_ligand",
        None,
        TENSOR_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "pocket_anchor",
        "decoder_pocket_anchor",
        "pocket_geometry",
        None,
        TENSOR_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "rollout",
        "tensor_preserving_rollout",
        "generated_rollout_state",
        None,
        "reserved for future tensor-preserving rollout objective",
    ),
    PrimaryObjectiveComponentSpec::diagnostic(
        "rollout_eval_recovery",
        "sampled_rollout_record",
        "generated_rollout_state",
        None,
        ROLLOUT_EVAL_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::diagnostic(
        "rollout_eval_pocket_anchor",
        "sampled_rollout_record",
        "generated_rollout_state",
        None,
        ROLLOUT_EVAL_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::diagnostic(
        "rollout_eval_stop",
        "sampled_rollout_stop_policy",
        "generated_rollout_state",
        None,
        ROLLOUT_EVAL_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_velocity",
        "flow_matching_velocity",
        "reference_ligand",
        Some("geometry"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_endpoint",
        "flow_matching_velocity_derived_endpoint_consistency",
        "reference_ligand",
        Some("geometry"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_atom_type",
        "molecular_flow_atom_type",
        "reference_ligand",
        Some("atom_type"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_bond",
        "molecular_flow_bond",
        "reference_ligand",
        Some("bond"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_bond_sparse_negative_rate",
        "molecular_flow_bond_sparse_calibration",
        "reference_ligand",
        Some("bond"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_bond_confidence_pressure",
        "molecular_flow_bond_native_graph_confidence",
        "reference_ligand",
        Some("bond"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_bond_degree_alignment",
        "molecular_flow_bond_expected_degree_alignment",
        "reference_ligand",
        Some("bond"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_topology",
        "molecular_flow_topology",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_topology_sparse_negative_rate",
        "molecular_flow_topology_sparse_calibration",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_topology_confidence_pressure",
        "molecular_flow_topology_native_graph_confidence",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_topology_degree_alignment",
        "molecular_flow_topology_expected_degree_alignment",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration",
        "molecular_flow_native_extraction_score_calibration",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::diagnostic(
        "flow_native_score_calibration_uncapped_raw",
        "molecular_flow_native_score_cap_audit",
        "optimizer_loss_audit",
        Some("topology"),
        NATIVE_SCORE_CAP_AUDIT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::diagnostic(
        "flow_native_score_calibration_cap_scale",
        "molecular_flow_native_score_cap_audit",
        "optimizer_loss_audit",
        Some("topology"),
        NATIVE_SCORE_CAP_AUDIT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_false_positive_margin",
        "molecular_flow_native_score_false_positive_margin",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_false_negative_margin",
        "molecular_flow_native_score_false_negative_margin",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_density_budget",
        "molecular_flow_native_score_density_budget",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_soft_positive_miss",
        "molecular_flow_native_score_soft_positive_miss",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_soft_negative_extraction",
        "molecular_flow_native_score_soft_negative_extraction",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_soft_extraction_budget",
        "molecular_flow_native_score_soft_extraction_budget",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_degree_alignment",
        "molecular_flow_native_score_degree_alignment",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_native_score_calibration_score_separation",
        "molecular_flow_native_score_separation",
        "reference_ligand",
        Some("topology"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_pocket_context",
        "molecular_flow_pocket_interaction_profile",
        "pocket_ligand_pair",
        Some("pocket_context"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
    PrimaryObjectiveComponentSpec::trainable(
        "flow_synchronization",
        "molecular_flow_branch_synchronization",
        "generated_rollout_state",
        Some("synchronization"),
        FLOW_PRIMARY_COMPONENT_BOUNDARY,
    ),
];

fn primary_objective_component_spec(
    component_name: &str,
) -> Option<&'static PrimaryObjectiveComponentSpec> {
    PRIMARY_OBJECTIVE_COMPONENT_SPECS
        .iter()
        .find(|spec| spec.component_name == component_name)
}

/// Return stable primary component names covered by the audit registry.
pub fn primary_objective_component_names() -> impl Iterator<Item = &'static str> {
    PRIMARY_OBJECTIVE_COMPONENT_SPECS
        .iter()
        .map(|spec| spec.component_name)
}

/// Return stable metadata for a primary-objective component.
pub fn primary_objective_component_descriptor(
    component_name: &str,
) -> PrimaryObjectiveComponentDescriptor {
    primary_objective_component_spec(component_name)
        .map(|spec| spec.descriptor)
        .unwrap_or_else(legacy_primary_component_descriptor)
}

/// Return the flow branch that owns a primary component, if any.
pub fn primary_objective_component_branch_name(component_name: &str) -> Option<&'static str> {
    primary_objective_component_spec(component_name).and_then(|spec| spec.descriptor.branch_name)
}

/// Build a provenance record from the shared component registry.
pub fn primary_objective_component_provenance_record(
    component_name: &str,
) -> PrimaryObjectiveComponentProvenance {
    let descriptor = primary_objective_component_descriptor(component_name);
    PrimaryObjectiveComponentProvenance {
        component_name: component_name.to_string(),
        anchor: descriptor.anchor.to_string(),
        target_source: descriptor.target_source.to_string(),
        differentiable: descriptor.differentiable,
        optimizer_facing: descriptor.optimizer_facing,
        role: descriptor.role.to_string(),
        effective_branch_weight: None,
        branch_schedule_source: None,
    }
}

fn legacy_primary_component_descriptor() -> PrimaryObjectiveComponentDescriptor {
    PrimaryObjectiveComponentDescriptor {
        anchor: "legacy_unknown",
        target_source: "legacy_unknown",
        differentiable: false,
        optimizer_facing: false,
        role: "evaluation_only",
        branch_name: None,
        claim_boundary: "diagnostic primary objective component",
    }
}

#[cfg(test)]
mod primary_component_descriptor_tests {
    use super::*;

    #[test]
    fn descriptor_marks_native_score_cap_audit_as_non_optimizer_topology_component() {
        let descriptor =
            primary_objective_component_descriptor("flow_native_score_calibration_cap_scale");

        assert_eq!(descriptor.branch_name, Some("topology"));
        assert_eq!(descriptor.target_source, "optimizer_loss_audit");
        assert!(!descriptor.optimizer_facing);
        assert!(!descriptor.differentiable);
        assert!(descriptor.claim_boundary.contains("diagnostic only"));
    }

    #[test]
    fn descriptor_routes_flow_subterms_to_branch_schedule_owners() {
        assert_eq!(
            primary_objective_component_branch_name("flow_bond_confidence_pressure"),
            Some("bond")
        );
        assert_eq!(
            primary_objective_component_branch_name(
                "flow_native_score_calibration_false_positive_margin"
            ),
            Some("topology")
        );
        assert_eq!(
            primary_objective_component_branch_name("flow_pocket_context"),
            Some("pocket_context")
        );
    }

    #[test]
    fn descriptor_registry_names_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for component_name in primary_objective_component_names() {
            assert!(
                seen.insert(component_name),
                "duplicate primary component descriptor for {component_name}"
            );
        }
    }

    #[test]
    fn observed_components_are_all_descriptor_backed() {
        let components = PrimaryObjectiveComponentMetrics {
            topology: Some(1.0),
            geometry: Some(1.0),
            pocket_anchor: Some(1.0),
            rollout: Some(1.0),
            rollout_eval_recovery: Some(1.0),
            rollout_eval_pocket_anchor: Some(1.0),
            rollout_eval_stop: Some(1.0),
            flow_velocity: Some(1.0),
            flow_endpoint: Some(1.0),
            flow_atom_type: Some(1.0),
            flow_bond: Some(1.0),
            flow_bond_sparse_negative_rate: Some(1.0),
            flow_bond_confidence_pressure: Some(1.0),
            flow_bond_degree_alignment: Some(1.0),
            flow_topology: Some(1.0),
            flow_topology_sparse_negative_rate: Some(1.0),
            flow_topology_confidence_pressure: Some(1.0),
            flow_topology_degree_alignment: Some(1.0),
            flow_native_score_calibration: Some(1.0),
            flow_native_score_calibration_uncapped_raw: Some(1.0),
            flow_native_score_calibration_cap_scale: Some(1.0),
            flow_native_score_calibration_false_positive_margin: Some(1.0),
            flow_native_score_calibration_false_negative_margin: Some(1.0),
            flow_native_score_calibration_density_budget: Some(1.0),
            flow_native_score_calibration_soft_positive_miss: Some(1.0),
            flow_native_score_calibration_soft_negative_extraction: Some(1.0),
            flow_native_score_calibration_soft_extraction_budget: Some(1.0),
            flow_native_score_calibration_degree_alignment: Some(1.0),
            flow_native_score_calibration_score_separation: Some(1.0),
            flow_pocket_context: Some(1.0),
            flow_synchronization: Some(1.0),
        };

        let observed = components.observed_component_values();
        let provenance = components.provenance_records();

        assert_eq!(observed.len(), provenance.len());
        for ((component_name, _), record) in observed.iter().zip(provenance.iter()) {
            assert_eq!(*component_name, record.component_name);
            assert_ne!(record.anchor, "legacy_unknown");
            assert_ne!(record.target_source, "legacy_unknown");
        }
    }

    #[test]
    fn branch_component_audits_group_observed_flow_terms_by_registry_owner() {
        let components = PrimaryObjectiveComponentMetrics {
            flow_bond: Some(2.0),
            flow_bond_confidence_pressure: Some(0.5),
            flow_native_score_calibration: Some(1.5),
            flow_native_score_calibration_cap_scale: Some(0.25),
            flow_pocket_context: Some(3.0),
            ..PrimaryObjectiveComponentMetrics::default()
        };

        let audits = components.branch_component_audits();
        let audit = |name: &str| {
            audits
                .iter()
                .find(|audit| audit.branch_name == name)
                .unwrap()
        };

        assert_eq!(audit("bond").observed_component_count, 2);
        assert_eq!(audit("bond").optimizer_facing_component_count, 2);
        assert_eq!(audit("bond").diagnostic_component_count, 0);
        assert_eq!(audit("topology").observed_component_count, 2);
        assert_eq!(audit("topology").optimizer_facing_component_count, 1);
        assert_eq!(audit("topology").diagnostic_component_count, 1);
        assert_eq!(audit("pocket_context").observed_component_count, 1);
    }

    #[test]
    fn component_add_assign_and_scale_preserve_observed_registry_values() {
        let mut components = PrimaryObjectiveComponentMetrics {
            topology: Some(1.0),
            flow_native_score_calibration_cap_scale: Some(0.25),
            ..PrimaryObjectiveComponentMetrics::default()
        };
        let other = PrimaryObjectiveComponentMetrics {
            topology: Some(2.0),
            flow_bond: Some(3.0),
            flow_native_score_calibration_cap_scale: Some(0.5),
            ..PrimaryObjectiveComponentMetrics::default()
        };

        components.add_assign(&other);
        let scaled = components.scale(2.0);

        assert_eq!(components.topology, Some(3.0));
        assert_eq!(components.flow_bond, Some(3.0));
        assert_eq!(components.flow_native_score_calibration_cap_scale, Some(0.75));
        assert_eq!(scaled.topology, Some(6.0));
        assert_eq!(scaled.flow_bond, Some(6.0));
        assert_eq!(scaled.flow_native_score_calibration_cap_scale, Some(1.5));
        assert_eq!(scaled.geometry, None);
    }
}

fn default_primary_component_target_source() -> String {
    "legacy_unknown".to_string()
}

/// Auxiliary losses emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxiliaryLossMetrics {
    /// Intra-modality redundancy objective.
    pub intra_red: f64,
    /// Semantic probe objective.
    pub probe: f64,
    /// Internal sparse negative-class calibration inside topology adjacency probing.
    #[serde(default)]
    pub probe_topology_sparse_negative_rate: f64,
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
    /// Explicit pocket-to-topology-role leakage penalty.
    #[serde(default)]
    pub leak_pocket_to_topology_role: f64,
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
    /// Atom-pocket pair distance-bin objective.
    #[serde(default)]
    pub pocket_pair_distance: f64,
    /// Pocket-ligand steric-clash penalty objective.
    #[serde(default)]
    pub pocket_clash: f64,
    /// Coarse pocket-ligand shape-complementarity objective.
    #[serde(default)]
    pub pocket_shape_complementarity: f64,
    /// Pocket-envelope containment objective.
    #[serde(default)]
    pub pocket_envelope: f64,
    /// Explicit pocket-conditioned size and composition prior objective.
    #[serde(default)]
    pub pocket_prior: f64,
    /// Atom-count classification subterm for the pocket prior.
    #[serde(default)]
    pub pocket_prior_atom_count: f64,
    /// Composition-distribution subterm for the pocket prior.
    #[serde(default)]
    pub pocket_prior_composition: f64,
    /// Mean absolute atom-count prediction error for the pocket prior.
    #[serde(default)]
    pub pocket_prior_atom_count_mae: f64,
    /// Conservative valence overage objective.
    #[serde(default)]
    pub valence_guardrail: f64,
    /// Expected valence above element-specific capacity.
    #[serde(default)]
    pub valence_overage_guardrail: f64,
    /// Expected valence below a conservative lower bound.
    #[serde(default)]
    pub valence_underage_guardrail: f64,
    /// Topology-implied bond-length objective.
    #[serde(default)]
    pub bond_length_guardrail: f64,
    /// Generated non-bonded short-distance margin objective.
    #[serde(default)]
    pub nonbonded_distance_guardrail: f64,
    /// Generated local-angle plausibility objective.
    #[serde(default)]
    pub angle_guardrail: f64,
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
    /// Atom-pocket pair distance-bin supervision.
    PocketPairDistance,
    /// Pocket-ligand steric-clash penalty.
    PocketClash,
    /// Coarse pocket-ligand shape-complementarity penalty.
    PocketShapeComplementarity,
    /// Pocket-envelope containment.
    PocketEnvelope,
    /// Explicit pocket-conditioned size and composition priors.
    PocketPrior,
    /// Conservative valence overage.
    ValenceGuardrail,
    /// Topology-implied bond-length deviation.
    BondLengthGuardrail,
    /// Generated non-bonded short-distance margin.
    NonbondedDistanceGuardrail,
    /// Generated local-angle plausibility proxy.
    AngleGuardrail,
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
            Self::PocketPairDistance => "pocket_pair_distance",
            Self::PocketClash => "pocket_clash",
            Self::PocketShapeComplementarity => "pocket_shape_complementarity",
            Self::PocketEnvelope => "pocket_envelope",
            Self::PocketPrior => "pocket_prior",
            Self::ValenceGuardrail => "valence_guardrail",
            Self::BondLengthGuardrail => "bond_length_guardrail",
            Self::NonbondedDistanceGuardrail => "nonbonded_distance_guardrail",
            Self::AngleGuardrail => "angle_guardrail",
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

/// Aggregated optimizer objective-family scale record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveFamilyBudgetEntry {
    /// Stable high-level family label, e.g. task, rollout, chemistry.
    pub family: String,
    /// Aggregate unweighted value before staged family weights are applied.
    pub unweighted_value: f64,
    /// Aggregate effective weight implied by the weighted and unweighted totals.
    pub effective_weight: f64,
    /// Final weighted contribution after optional budget clamping.
    #[serde(default)]
    pub weighted_value: f64,
    /// Weighted contribution before optional budget clamping.
    #[serde(default)]
    pub raw_weighted_value: f64,
    /// Fraction of the final total absolute weighted objective owned by this family.
    #[serde(default)]
    pub percentage_of_total: f64,
    /// Optional configured non-primary family budget cap.
    #[serde(default)]
    pub budget_cap_fraction: Option<f64>,
    /// Configured budget behavior: none, warn, or clamp.
    #[serde(default = "default_objective_family_budget_action")]
    pub budget_action: String,
    /// Whether this family was scaled down by budget clamping.
    #[serde(default)]
    pub budget_clamped: bool,
    /// Whether this family contributes to the optimizer objective.
    pub enabled: bool,
    /// Compact scale status.
    #[serde(default)]
    pub status: String,
    /// Optional scale or budget warning.
    #[serde(default)]
    pub warning: Option<String>,
}

impl ObjectiveFamilyBudgetEntry {
    /// Build one high-level objective-family budget entry.
    pub fn new(
        family: impl Into<String>,
        unweighted_value: f64,
        effective_weight: f64,
        weighted_value: f64,
        enabled: bool,
    ) -> Self {
        let enabled = enabled && effective_weight.is_finite() && effective_weight > 0.0;
        let status = if !unweighted_value.is_finite()
            || !effective_weight.is_finite()
            || !weighted_value.is_finite()
        {
            "nonfinite".to_string()
        } else if enabled {
            "active".to_string()
        } else {
            "inactive_zero_weight".to_string()
        };
        Self {
            family: family.into(),
            unweighted_value,
            effective_weight,
            weighted_value,
            raw_weighted_value: weighted_value,
            percentage_of_total: 0.0,
            budget_cap_fraction: None,
            budget_action: default_objective_family_budget_action(),
            budget_clamped: false,
            enabled,
            status,
            warning: None,
        }
    }
}

fn default_objective_family_budget_action() -> String {
    "none".to_string()
}

/// Aggregated objective-family budget report for one optimizer step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObjectiveFamilyBudgetReport {
    /// Per-family records in stable budget-family order.
    #[serde(default)]
    pub entries: Vec<ObjectiveFamilyBudgetEntry>,
    /// Absolute weighted primary/task contribution.
    #[serde(default)]
    pub primary_weighted_abs_value: f64,
    /// Sum of final absolute weighted family contributions.
    #[serde(default)]
    pub total_weighted_abs_value: f64,
    /// Sum of raw absolute weighted family contributions before clamping.
    #[serde(default)]
    pub raw_total_weighted_abs_value: f64,
    /// Configured non-primary family cap, if any.
    #[serde(default)]
    pub budget_cap_fraction: Option<f64>,
    /// Configured budget behavior: none, warn, or clamp.
    #[serde(default = "default_objective_family_budget_action")]
    pub budget_action: String,
    /// Number of families that exceeded a warning-only budget cap.
    #[serde(default)]
    pub budget_warning_count: usize,
    /// Number of families clamped by the configured budget cap.
    #[serde(default)]
    pub budget_clamped_count: usize,
}

impl ObjectiveFamilyBudgetReport {
    /// Populate percentages without applying a budget cap.
    pub fn refresh_percentages(&mut self, epsilon: f64) {
        for entry in &mut self.entries {
            entry.raw_weighted_value = entry.weighted_value;
            entry.percentage_of_total = 0.0;
            entry.budget_cap_fraction = None;
            entry.budget_action = default_objective_family_budget_action();
            entry.budget_clamped = false;
        }
        self.budget_cap_fraction = None;
        self.budget_action = default_objective_family_budget_action();
        self.budget_warning_count = 0;
        self.budget_clamped_count = 0;
        self.recompute_totals_and_percentages(epsilon);
    }

    /// Apply optional non-primary family caps and recompute final percentages.
    pub fn apply_budget_caps(
        &mut self,
        budget_cap_fraction: Option<f64>,
        budget_action: &str,
        dominance_warning_ratio: f64,
        epsilon: f64,
    ) {
        let epsilon = positive_or_default(epsilon, 1.0e-12);
        let dominance_warning_ratio = positive_or_default(dominance_warning_ratio, 10.0);
        let budget_cap_fraction =
            budget_cap_fraction.filter(|cap| cap.is_finite() && *cap > 0.0);
        let budget_action = if budget_cap_fraction.is_some() {
            budget_action
        } else {
            "none"
        };

        for entry in &mut self.entries {
            entry.raw_weighted_value = entry.weighted_value;
            entry.percentage_of_total = 0.0;
            entry.budget_cap_fraction = if objective_family_budget_applies(&entry.family) {
                budget_cap_fraction
            } else {
                None
            };
            entry.budget_action = if entry.budget_cap_fraction.is_some() {
                budget_action.to_string()
            } else {
                default_objective_family_budget_action()
            };
            entry.budget_clamped = false;
        }

        let raw_total_abs = self
            .entries
            .iter()
            .filter(|entry| entry.enabled && entry.raw_weighted_value.is_finite())
            .map(|entry| entry.raw_weighted_value.abs())
            .sum::<f64>()
            .max(epsilon);
        let primary_abs = self
            .entries
            .iter()
            .find(|entry| entry.family == "task")
            .filter(|entry| entry.enabled && entry.raw_weighted_value.is_finite())
            .map(|entry| entry.raw_weighted_value.abs())
            .unwrap_or(0.0)
            .max(epsilon);

        self.budget_warning_count = 0;
        self.budget_clamped_count = 0;
        for entry in &mut self.entries {
            if !entry.enabled || !entry.raw_weighted_value.is_finite() {
                continue;
            }
            let raw_abs = entry.raw_weighted_value.abs();
            if raw_abs <= epsilon {
                continue;
            }
            if let Some(cap) = entry.budget_cap_fraction {
                let raw_percentage = raw_abs / raw_total_abs;
                if raw_percentage > cap {
                    if budget_action == "clamp" {
                        let limit =
                            objective_family_budget_limit(raw_abs, raw_total_abs, cap, epsilon);
                        if raw_abs > limit + epsilon {
                            let sign = if entry.raw_weighted_value.is_sign_negative() {
                                -1.0
                            } else {
                                1.0
                            };
                            entry.weighted_value = sign * limit;
                            entry.budget_clamped = true;
                            entry.status = "budget_clamped".to_string();
                            entry.warning = Some(format!(
                                "weighted objective family clamped to configured budget cap {:.4}",
                                cap
                            ));
                            self.budget_clamped_count += 1;
                            continue;
                        }
                    } else {
                        entry.status = "budget_cap_warning".to_string();
                        entry.warning = Some(format!(
                            "weighted objective family exceeds configured budget cap {:.4}",
                            cap
                        ));
                        self.budget_warning_count += 1;
                        continue;
                    }
                }
            }

            if objective_family_budget_applies(&entry.family)
                && raw_abs > primary_abs * dominance_warning_ratio
            {
                entry.status = "dominant".to_string();
                entry.warning = Some(format!(
                    "weighted objective family exceeds {:.4}x the weighted primary objective",
                    dominance_warning_ratio
                ));
            }
        }

        self.budget_cap_fraction = budget_cap_fraction;
        self.budget_action = budget_action.to_string();
        self.recompute_totals_and_percentages(epsilon);
    }

    fn recompute_totals_and_percentages(&mut self, epsilon: f64) {
        let epsilon = positive_or_default(epsilon, 1.0e-12);
        self.primary_weighted_abs_value = self
            .entries
            .iter()
            .find(|entry| entry.family == "task")
            .filter(|entry| entry.enabled && entry.weighted_value.is_finite())
            .map(|entry| entry.weighted_value.abs())
            .unwrap_or(0.0);
        self.total_weighted_abs_value = self
            .entries
            .iter()
            .filter(|entry| entry.enabled && entry.weighted_value.is_finite())
            .map(|entry| entry.weighted_value.abs())
            .sum::<f64>();
        self.raw_total_weighted_abs_value = self
            .entries
            .iter()
            .filter(|entry| entry.enabled && entry.raw_weighted_value.is_finite())
            .map(|entry| entry.raw_weighted_value.abs())
            .sum::<f64>();
        let denominator = self.total_weighted_abs_value.max(epsilon);
        for entry in &mut self.entries {
            entry.percentage_of_total = if entry.enabled && entry.weighted_value.is_finite() {
                entry.weighted_value.abs() / denominator
            } else {
                0.0
            };
        }
    }
}

fn positive_or_default(value: f64, default_value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        default_value
    }
}

fn objective_family_budget_applies(family: &str) -> bool {
    family != "task"
}

fn objective_family_budget_limit(
    raw_abs: f64,
    raw_total_abs: f64,
    cap: f64,
    epsilon: f64,
) -> f64 {
    if cap >= 1.0 {
        return raw_abs;
    }
    let other_abs = (raw_total_abs - raw_abs).max(0.0);
    if other_abs <= epsilon {
        0.0
    } else {
        (cap * other_abs / (1.0 - cap)).max(0.0)
    }
}

/// Optimizer-facing short-rollout loss and scheduled-sampling diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutTrainingLossMetrics {
    /// Whether `training.rollout_training.enabled` was true.
    #[serde(default)]
    pub enabled: bool,
    /// Whether this step is at or past the configured rollout warmup step.
    #[serde(default)]
    pub active: bool,
    /// Configured warmup step for rollout-state losses.
    #[serde(default)]
    pub warmup_step: usize,
    /// Configured bounded rollout step count.
    #[serde(default)]
    pub configured_steps: usize,
    /// Mean emitted differentiable rollout records per contributing example.
    #[serde(default)]
    pub executed_steps_mean: f64,
    /// Maximum examples allowed to contribute rollout-state loss.
    #[serde(default)]
    pub max_batch_examples: usize,
    /// Number of examples that contributed rollout-state loss.
    #[serde(default)]
    pub contributing_examples: usize,
    /// Detach policy used between generated states.
    #[serde(default)]
    pub detach_policy: String,
    /// Target/evidence source for rollout-state losses.
    #[serde(default)]
    pub target_source: String,
    /// Teacher-forced primary loss observed on the same step.
    #[serde(default)]
    pub teacher_forced_loss: f64,
    /// Weighted rollout-state loss added to the optimizer objective.
    #[serde(default)]
    pub rollout_state_loss: f64,
    /// Divergence between rollout-state loss and teacher-forced primary loss.
    #[serde(default)]
    pub teacher_rollout_divergence: f64,
    /// Compact generated-state validity proxy in `[0, 1]`.
    #[serde(default)]
    pub generated_state_validity: f64,
    /// Weighted atom-validity contribution.
    #[serde(default)]
    pub atom_validity: f64,
    /// Weighted bond-consistency contribution.
    #[serde(default)]
    pub bond_consistency: f64,
    /// Weighted sparse negative-class calibration inside rollout bond consistency.
    #[serde(default)]
    pub bond_sparse_negative_rate: f64,
    /// Weighted pocket-contact contribution.
    #[serde(default)]
    pub pocket_contact: f64,
    /// Weighted clash-margin contribution.
    #[serde(default)]
    pub clash_margin: f64,
    /// Weighted endpoint-consistency contribution.
    #[serde(default)]
    pub endpoint_consistency: f64,
    /// Memory-control note for audit reports.
    #[serde(default)]
    pub memory_control: String,
}

impl Default for RolloutTrainingLossMetrics {
    fn default() -> Self {
        Self {
            enabled: false,
            active: false,
            warmup_step: 0,
            configured_steps: 0,
            executed_steps_mean: 0.0,
            max_batch_examples: 0,
            contributing_examples: 0,
            detach_policy: "disabled".to_string(),
            target_source: "generated_rollout_state".to_string(),
            teacher_forced_loss: 0.0,
            rollout_state_loss: 0.0,
            teacher_rollout_divergence: 0.0,
            generated_state_validity: 0.0,
            atom_validity: 0.0,
            bond_consistency: 0.0,
            bond_sparse_negative_rate: 0.0,
            pocket_contact: 0.0,
            clash_margin: 0.0,
            endpoint_consistency: 0.0,
            memory_control: "disabled".to_string(),
        }
    }
}

/// Named loss values emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossBreakdown {
    /// Primary objective metrics.
    pub primary: PrimaryObjectiveMetrics,
    /// Auxiliary regularizer metrics.
    pub auxiliaries: AuxiliaryLossMetrics,
    /// Optional optimizer-facing short-rollout objective diagnostics.
    #[serde(default)]
    pub rollout_training: RolloutTrainingLossMetrics,
    /// High-level objective-family scale and budget report.
    #[serde(default)]
    pub objective_family_budget_report: ObjectiveFamilyBudgetReport,
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
    /// Explicit promotion-gate decision: fixed_schedule, promoted, held_previous_stage, etc.
    #[serde(default)]
    pub promotion_gate_decision: String,
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
    /// Effective scaffold/x0 samples evaluated per pocket for this step.
    #[serde(default = "default_generation_sample_count")]
    pub generation_sample_count: usize,
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
            generation_sample_count: default_generation_sample_count(),
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

fn default_generation_sample_count() -> usize {
    1
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
    /// Configured dominant-family fraction threshold for this step.
    #[serde(default)]
    pub dominance_fraction_threshold: f64,
    /// Number of sampled families marked as dominant.
    #[serde(default)]
    pub dominant_family_count: usize,
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
    /// Mean maximum assignment-mass fraction across modalities and examples.
    #[serde(default)]
    pub mean_slot_mass_max_fraction: f64,
    /// Mean effective number of assignment-mass slots, computed as exp(entropy).
    #[serde(default)]
    pub mean_slot_mass_effective_count: f64,
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
    /// Number of modality/example diagnostics with highly concentrated slot mass.
    #[serde(default)]
    pub mass_concentration_warning_count: usize,
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
            mean_slot_mass_max_fraction: 0.0,
            mean_slot_mass_effective_count: 0.0,
            dead_slot_fraction: 0.0,
            dead_slot_count: 0,
            diffuse_slot_count: 0,
            saturated_slot_count: 0,
            collapse_warning_count: 0,
            mass_concentration_warning_count: 0,
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
        let mut mass_max_fraction = 0.0;
        let mut mass_effective_count = 0.0;
        let mut dead_slots = 0.0;
        let mut slot_total = 0.0;
        let mut dead_slot_count = 0usize;
        let mut diffuse_slot_count = 0usize;
        let mut saturated_slot_count = 0usize;
        let mut collapse_warning_count = 0usize;
        let mut mass_concentration_warning_count = 0usize;
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
                &mut mass_max_fraction,
                &mut mass_effective_count,
                &mut dead_slots,
                &mut slot_total,
                &mut dead_slot_count,
                &mut diffuse_slot_count,
                &mut saturated_slot_count,
                &mut collapse_warning_count,
                &mut mass_concentration_warning_count,
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
                &mut mass_max_fraction,
                &mut mass_effective_count,
                &mut dead_slots,
                &mut slot_total,
                &mut dead_slot_count,
                &mut diffuse_slot_count,
                &mut saturated_slot_count,
                &mut collapse_warning_count,
                &mut mass_concentration_warning_count,
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
                &mut mass_max_fraction,
                &mut mass_effective_count,
                &mut dead_slots,
                &mut slot_total,
                &mut dead_slot_count,
                &mut diffuse_slot_count,
                &mut saturated_slot_count,
                &mut collapse_warning_count,
                &mut mass_concentration_warning_count,
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
            mean_slot_mass_max_fraction: mass_max_fraction / denom,
            mean_slot_mass_effective_count: mass_effective_count / denom,
            dead_slot_fraction: dead_slots / slot_total.max(1.0),
            dead_slot_count,
            diffuse_slot_count,
            saturated_slot_count,
            collapse_warning_count,
            mass_concentration_warning_count,
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
    mass_max_fraction: &mut f64,
    mass_effective_count: &mut f64,
    dead_slots: &mut f64,
    slot_total: &mut f64,
    dead_slot_count: &mut usize,
    diffuse_slot_count: &mut usize,
    saturated_slot_count: &mut usize,
    collapse_warning_count: &mut usize,
    mass_concentration_warning_count: &mut usize,
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
    let max_mass_fraction = slot_mass_max_fraction(&slots_encoding.slot_weights);
    let effective_mass_count = slot_mass_effective_count(&slots_encoding.slot_weights);
    *mass_max_fraction += max_mass_fraction;
    *mass_effective_count += effective_mass_count;
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
    let mass_concentrated = slots > 1.0 && (max_mass_fraction >= 0.90 || effective_mass_count <= 1.25);
    if mass_concentrated {
        *mass_concentration_warning_count += 1;
        if warnings.len() < 8 {
            warnings.push(format!("{}_slot_mass_concentrated", branch.modality));
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

fn slot_mass_max_fraction(weights: &tch::Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    let total = weights.sum(tch::Kind::Float).double_value(&[]);
    if !total.is_finite() || total.abs() <= 1.0e-12 {
        return 0.0;
    }
    (weights / total)
        .max()
        .double_value(&[])
        .clamp(0.0, 1.0)
}

fn slot_mass_effective_count(weights: &tch::Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    let entropy = slot_weight_entropy(weights);
    if entropy.is_finite() {
        entropy.exp().min(weights.numel() as f64)
    } else {
        0.0
    }
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
