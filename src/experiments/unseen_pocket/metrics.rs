/// Backend-backed generation metrics for chemistry, docking, and pocket scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealGenerationMetrics {
    /// Chemistry validity backend slot.
    pub chemistry_validity: ReservedBackendMetrics,
    /// Docking or affinity rescoring backend slot.
    pub docking_affinity: ReservedBackendMetrics,
    /// Downstream pocket compatibility backend slot.
    pub pocket_compatibility: ReservedBackendMetrics,
}

/// Backend schema entry used for active, disabled, or unavailable evaluation hooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservedBackendMetrics {
    /// Whether this backend has been integrated for the current run.
    pub available: bool,
    /// Backend identifier when available.
    pub backend_name: Option<String>,
    /// Reserved metrics map emitted by the backend.
    pub metrics: BTreeMap<String, f64>,
    /// Explanation of backend availability or fallback status.
    pub status: String,
}

/// Versioned contract that defines claim-facing drug-discovery metric evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugLevelClaimContract {
    /// Schema version for compatibility checks.
    pub schema_version: u32,
    /// Stable contract identifier.
    pub contract_name: String,
    /// Allowed status labels for metric evidence.
    pub status_values: Vec<String>,
    /// Canonical generation layers used by drug-level claim reports.
    pub layers: Vec<String>,
    /// Mapping from capability class to layer names.
    pub capability_groups: BTreeMap<String, Vec<String>>,
    /// Required claim metric groups.
    pub required_metric_groups: Vec<String>,
    /// Required provenance fields for every backend-scored metric group.
    pub required_backend_fields: Vec<String>,
    /// Metric requirements keyed by group name.
    pub metric_groups: BTreeMap<String, DrugLevelMetricGroupContract>,
}

/// Required and optional metrics for one drug-level evidence group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugLevelMetricGroupContract {
    /// Metrics that must be present when the group is observed.
    pub required_metrics: Vec<String>,
    /// Metrics that may be emitted when the corresponding backend is available.
    #[serde(default)]
    pub optional_metrics: Vec<String>,
}

/// Aggregate evaluation metrics for one split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationDiagnostics {
    /// Fraction of samples producing finite outputs.
    pub finite_forward_fraction: f64,
    /// Fraction of unique protein-ligand ids in the split.
    pub unique_complex_fraction: f64,
    /// Fraction of proteins not seen in the training set.
    pub unseen_protein_fraction: f64,
    /// RMSE between distance-probe predictions and target pairwise distances.
    pub distance_probe_rmse: f64,
    /// Cross-modal cosine alignment between topology and pocket latents.
    pub topology_pocket_cosine_alignment: f64,
    /// Mean topology reconstruction error across the split.
    pub topology_reconstruction_mse: f64,
    /// Mean active slot fraction.
    pub slot_activation_mean: f64,
    /// Mean normalized assignment-mass entropy across modality slots.
    #[serde(default)]
    pub slot_assignment_entropy_mean: f64,
    /// Mean raw slot activation probability across modality slots.
    #[serde(default)]
    pub slot_activation_probability_mean: f64,
    /// Mean fraction of slots visible to attention after active-slot masking.
    #[serde(default)]
    pub attention_visible_slot_fraction: f64,
    /// Mean gate activation.
    pub gate_activation_mean: f64,
    /// Mean leakage proxy.
    pub leakage_proxy_mean: f64,
}

/// Proxy task metrics derived from lightweight probe heads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyTaskMetrics {
    /// MAE of affinity prediction on labeled examples.
    pub affinity_probe_mae: f64,
    /// RMSE of affinity prediction on labeled examples.
    pub affinity_probe_rmse: f64,
    /// Fraction of examples in the split with affinity labels.
    pub labeled_fraction: f64,
    /// Affinity error summarized per measurement type.
    pub affinity_by_measurement: Vec<MeasurementMetrics>,
    /// Probe loss comparisons against trivial target-only baselines.
    #[serde(default)]
    pub probe_baselines: Vec<ProbeBaselineMetric>,
}

/// One semantic/leakage probe comparison against a target-only trivial baseline.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProbeBaselineMetric {
    /// Stable probe target label.
    pub target: String,
    /// Loss type used for this comparison.
    pub loss_kind: String,
    /// Observed probe loss when the target is available.
    #[serde(default)]
    pub observed_loss: Option<f64>,
    /// Loss from a target-only trivial predictor on the same split.
    #[serde(default)]
    pub trivial_baseline_loss: Option<f64>,
    /// Whether the observed probe improves on the trivial baseline.
    #[serde(default)]
    pub improves_over_trivial: Option<bool>,
    /// Supervision availability status for this target.
    #[serde(default)]
    pub supervision_status: String,
    /// Number of available examples or rows used by this comparison.
    #[serde(default)]
    pub available_count: usize,
    /// Interpretation note for leakage/specialization review.
    #[serde(default)]
    pub interpretation: String,
    /// Mean target positive rate for binary probe targets.
    #[serde(default)]
    pub target_positive_rate: Option<f64>,
    /// Mean predicted probability for binary probe targets.
    #[serde(default)]
    pub prediction_positive_rate: Option<f64>,
    /// Absolute gap between predicted and target positive rates.
    #[serde(default)]
    pub positive_rate_gap: Option<f64>,
    /// BCE contribution on positive labels for binary probe targets.
    #[serde(default)]
    pub positive_observed_loss: Option<f64>,
    /// BCE contribution on negative labels for binary probe targets.
    #[serde(default)]
    pub negative_observed_loss: Option<f64>,
    /// Mean scalar target value for regression probe targets.
    #[serde(default)]
    pub scalar_target_mean: Option<f64>,
    /// Mean scalar prediction value for regression probe targets.
    #[serde(default)]
    pub scalar_prediction_mean: Option<f64>,
    /// Mean signed scalar prediction error for regression probe targets.
    #[serde(default)]
    pub scalar_mean_error: Option<f64>,
}

/// Split-level counts needed to interpret evaluation outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitContextMetrics {
    /// Number of examples evaluated.
    pub example_count: usize,
    /// Number of unique complex identifiers in the split.
    pub unique_complex_count: usize,
    /// Number of unique proteins in the split.
    pub unique_protein_count: usize,
    /// Number of training proteins used as the unseen-pocket reference set.
    pub train_reference_protein_count: usize,
    /// Histogram of ligand atom-count bins represented by this split.
    #[serde(default)]
    pub ligand_atom_count_bins: BTreeMap<String, usize>,
    /// Histogram of pocket atom-count bins represented by this split.
    #[serde(default)]
    pub pocket_atom_count_bins: BTreeMap<String, usize>,
    /// Histogram of measurement families represented by this split.
    #[serde(default)]
    pub measurement_family_histogram: BTreeMap<String, usize>,
}

/// Runtime resource measurements for one evaluation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageMetrics {
    /// Average process memory delta in MB during evaluation.
    pub memory_usage_mb: f64,
    /// Elapsed evaluation time in milliseconds.
    pub evaluation_time_ms: f64,
    /// Coarse evaluated examples per second.
    #[serde(default)]
    pub examples_per_second: f64,
    /// Configured evaluation batch size used for forward passes.
    #[serde(default)]
    pub evaluation_batch_size: usize,
    /// Number of forward batches executed for this split.
    #[serde(default)]
    pub forward_batch_count: usize,
    /// Number of explicit per-example forward calls represented by this split.
    #[serde(default)]
    pub per_example_forward_count: usize,
    /// Whether evaluation forward passes ran under `tch::no_grad`.
    #[serde(default)]
    pub no_grad: bool,
    /// Whether evaluation used the batched forward path.
    #[serde(default)]
    pub batched_forward: bool,
    /// Correctness reason when de novo evaluation intentionally falls back to per-example forwards.
    #[serde(default)]
    pub de_novo_per_example_reason: Option<String>,
    /// Average ligand atom count observed in the evaluation split.
    #[serde(default)]
    pub average_ligand_atoms: f64,
    /// Average pocket atom count observed in the evaluation split.
    #[serde(default)]
    pub average_pocket_atoms: f64,
}

/// Focused model-design diagnostics for unseen-pocket interpretation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDesignEvaluationMetrics {
    /// Fraction of evaluated examples whose protein id was absent from train.
    pub heldout_unseen_protein_fraction: f64,
    /// Fraction of forward passes with finite modality embeddings.
    pub finite_forward_fraction: f64,
    /// Ligand topology reconstruction error from slot reconstruction.
    pub ligand_topology_reconstruction_mse: f64,
    /// Distance-probe RMSE used as a geometry-consistency diagnostic.
    pub geometry_distance_probe_rmse: f64,
    /// Bounded inverse-error geometry score.
    pub geometry_consistency_score: f64,
    /// Pocket contact fraction on the preferred processed candidate layer.
    pub local_pocket_contact_fraction: f64,
    /// Pocket clash fraction on the preferred processed candidate layer.
    pub local_pocket_clash_fraction: f64,
    /// Valid fraction for raw model-native rollout candidates.
    pub raw_model_valid_fraction: f64,
    /// Pocket contact fraction for raw model-native rollout candidates.
    pub raw_model_pocket_contact_fraction: f64,
    /// Clash fraction for raw model-native rollout candidates.
    pub raw_model_clash_fraction: f64,
    /// Mean final rollout displacement for raw model-native candidates.
    pub raw_model_mean_displacement: f64,
    /// Valid fraction for the preferred processed candidate layer.
    pub processed_valid_fraction: f64,
    /// Pocket contact fraction for the preferred processed candidate layer.
    pub processed_pocket_contact_fraction: f64,
    /// Clash fraction for the preferred processed candidate layer.
    pub processed_clash_fraction: f64,
    /// Processed minus raw valid-fraction delta.
    pub processing_valid_fraction_delta: f64,
    /// Processed minus raw pocket-contact delta.
    pub processing_pocket_contact_delta: f64,
    /// Processed minus raw clash-fraction delta; negative is better.
    pub processing_clash_delta: f64,
    /// Coarse evaluation throughput.
    pub examples_per_second: f64,
    /// Process memory delta in MB during evaluation.
    pub memory_usage_mb: f64,
    /// Mean active slot fraction.
    pub slot_activation_mean: f64,
    /// Mean slot-signature similarity across topology, geometry, and pocket.
    pub slot_signature_similarity_mean: f64,
    /// Mean directed gate activation.
    pub gate_activation_mean: f64,
    /// Mean directed gate saturation fraction.
    pub gate_saturation_fraction: f64,
    /// Mean fraction of gate elements effectively closed.
    #[serde(default)]
    pub gate_closed_fraction_mean: f64,
    /// Mean fraction of gate elements effectively open.
    #[serde(default)]
    pub gate_open_fraction_mean: f64,
    /// Mean sigmoid-derivative proxy for gate gradient health.
    #[serde(default)]
    pub gate_gradient_proxy_mean: f64,
    /// Mean norm of effective gated interaction updates.
    #[serde(default)]
    pub gate_effective_update_norm_mean: f64,
    /// Number of directed-path gate warnings observed across examples.
    pub gate_warning_count: usize,
    /// Compact audit note for interpreting gate health diagnostics.
    #[serde(default)]
    pub gate_audit_note: String,
    /// Mean leakage proxy from cross-modality slot similarity.
    pub leakage_proxy_mean: f64,
    /// Canonical layer interpreted as raw model-native output.
    pub raw_model_layer: String,
    /// Canonical processed layer used for processed quality fields.
    pub processed_layer: String,
    /// Ordered postprocessing or selection chain for the processed layer.
    #[serde(default)]
    pub processed_postprocessor_chain: Vec<String>,
    /// Claim-boundary note for the processed layer.
    #[serde(default)]
    pub processed_claim_boundary: String,
    /// Raw native rollout mean bond count before repair or pruning.
    #[serde(default)]
    pub raw_native_bond_count_mean: f64,
    /// Raw native rollout mean connected-component count before repair or pruning.
    #[serde(default)]
    pub raw_native_component_count_mean: f64,
    /// Raw native rollout valence violation fraction before repair or pruning.
    #[serde(default)]
    pub raw_native_valence_violation_fraction: f64,
    /// Raw native rollout fraction with synchronized cached and explicit bond payloads.
    #[serde(default)]
    pub raw_native_topology_bond_sync_fraction: f64,
    /// Raw native rollout atom-type entropy.
    #[serde(default)]
    pub raw_native_atom_type_entropy: f64,
    /// Raw native rollout graph validity fraction before repair or pruning.
    #[serde(default)]
    pub raw_native_graph_valid_fraction: f64,
    /// Claim-boundary note separating raw model quality from constrained or repaired quality.
    pub raw_vs_processed_note: String,
}

impl Default for ModelDesignEvaluationMetrics {
    fn default() -> Self {
        Self {
            heldout_unseen_protein_fraction: 0.0,
            finite_forward_fraction: 0.0,
            ligand_topology_reconstruction_mse: 0.0,
            geometry_distance_probe_rmse: 0.0,
            geometry_consistency_score: 0.0,
            local_pocket_contact_fraction: 0.0,
            local_pocket_clash_fraction: 0.0,
            raw_model_valid_fraction: 0.0,
            raw_model_pocket_contact_fraction: 0.0,
            raw_model_clash_fraction: 0.0,
            raw_model_mean_displacement: 0.0,
            processed_valid_fraction: 0.0,
            processed_pocket_contact_fraction: 0.0,
            processed_clash_fraction: 0.0,
            processing_valid_fraction_delta: 0.0,
            processing_pocket_contact_delta: 0.0,
            processing_clash_delta: 0.0,
            examples_per_second: 0.0,
            memory_usage_mb: 0.0,
            slot_activation_mean: 0.0,
            slot_signature_similarity_mean: 0.0,
            gate_activation_mean: 0.0,
            gate_saturation_fraction: 0.0,
            gate_closed_fraction_mean: 0.0,
            gate_open_fraction_mean: 0.0,
            gate_gradient_proxy_mean: 0.0,
            gate_effective_update_norm_mean: 0.0,
            gate_warning_count: 0,
            gate_audit_note: String::new(),
            leakage_proxy_mean: 0.0,
            raw_model_layer: "raw_rollout".to_string(),
            processed_layer: "unavailable".to_string(),
            processed_postprocessor_chain: Vec::new(),
            processed_claim_boundary: String::new(),
            raw_native_bond_count_mean: 0.0,
            raw_native_component_count_mean: 0.0,
            raw_native_valence_violation_fraction: 0.0,
            raw_native_topology_bond_sync_fraction: 0.0,
            raw_native_atom_type_entropy: 0.0,
            raw_native_graph_valid_fraction: 0.0,
            raw_vs_processed_note:
                "raw model-native quality is unavailable until candidates are generated".to_string(),
        }
    }
}

/// Aggregate evaluation metrics for one split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Representation-level diagnostics for the modular stack.
    pub representation_diagnostics: RepresentationDiagnostics,
    /// Proxy task metrics produced by lightweight probe heads.
    pub proxy_task_metrics: ProxyTaskMetrics,
    /// Split-level context counts.
    pub split_context: SplitContextMetrics,
    /// Runtime resource measurements.
    pub resource_usage: ResourceUsageMetrics,
    /// Focused model-design diagnostics for unseen-pocket interpretation.
    #[serde(default)]
    pub model_design: ModelDesignEvaluationMetrics,
    /// Reserved section for chemistry/docking/pocket compatibility backends.
    pub real_generation_metrics: RealGenerationMetrics,
    /// Candidate-quality metrics split by generation and postprocessing layer.
    pub layered_generation_metrics: LayeredGenerationMetrics,
    /// Method-aware comparison summary for this split.
    #[serde(default)]
    pub method_comparison: MethodComparisonSummary,
    /// Audit tying reported evaluation metrics to optimizer-facing terms, candidate layers, and backend coverage.
    #[serde(default)]
    pub train_eval_alignment: TrainEvalAlignmentReport,
    /// Chemistry-interpretable collaboration diagnostics with explicit provenance.
    #[serde(default)]
    pub chemistry_collaboration: ChemistryCollaborationMetrics,
    /// Held-out frozen leakage-probe calibration for this evaluation split.
    #[serde(default)]
    pub frozen_leakage_probe_calibration: FrozenLeakageProbeCalibrationReport,
    /// Stable comparison-friendly summary fields for cross-run reporting.
    pub comparison_summary: GenerationQualitySummary,
    /// Lightweight slot semantic-stability diagnostics for this split.
    #[serde(default)]
    pub slot_stability: SlotStabilityMetrics,
    /// Deterministic per-stratum summaries for claim review.
    #[serde(default)]
    pub strata: Vec<StratumEvaluationMetrics>,
}

/// Lightweight summary for one deterministic data stratum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StratumEvaluationMetrics {
    /// Stratum axis, such as `ligand_atoms`, `pocket_atoms`, or `measurement`.
    pub axis: String,
    /// Stable bin label.
    pub bin: String,
    /// Number of examples in this stratum.
    pub example_count: usize,
    /// Fraction of examples in the stratum that are unseen relative to training proteins.
    pub unseen_protein_fraction: f64,
    /// Labeled fraction for this stratum.
    pub labeled_fraction: f64,
    /// Mean ligand atom count for this stratum.
    pub average_ligand_atoms: f64,
    /// Mean pocket atom count for this stratum.
    pub average_pocket_atoms: f64,
}

/// Claim-facing evaluation matrix spanning seen, unseen, test, and stress surfaces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationMatrixReport {
    /// Schema version for compatibility checks.
    #[serde(default = "default_evaluation_matrix_schema_version")]
    pub schema_version: u32,
    /// Dataset-level pocket identity policy used for unseen-pocket interpretation.
    #[serde(default)]
    pub pocket_identity_policy: String,
    /// Rows in stable claim-review order.
    #[serde(default)]
    pub rows: Vec<EvaluationMatrixRow>,
}

fn default_evaluation_matrix_schema_version() -> u32 {
    1
}

/// One row of the unseen-pocket evaluation matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationMatrixRow {
    /// Stable row label.
    pub split_label: String,
    /// Split family: seen_pocket_validation, unseen_pocket_validation, unseen_pocket_test, or stress_pocket.
    pub split_type: String,
    /// Pocket identity handling for this row.
    pub pocket_identity_handling: String,
    /// evaluated, configured_not_evaluated, or not_configured.
    pub evaluation_status: String,
    /// Number of examples represented by this row.
    pub example_count: usize,
    /// Quality metrics used by generation claims.
    pub quality: EvaluationMatrixQualityMetrics,
    /// Efficiency metrics used by runtime and scaling claims.
    pub efficiency: EvaluationMatrixEfficiencyMetrics,
}

/// Quality slice copied into the evaluation matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationMatrixQualityMetrics {
    /// Raw model-native valid fraction.
    #[serde(default)]
    pub raw_valid_fraction: Option<f64>,
    /// Raw model-native pocket contact fraction.
    #[serde(default)]
    pub raw_pocket_contact_fraction: Option<f64>,
    /// Raw model-native clash fraction.
    #[serde(default)]
    pub raw_clash_fraction: Option<f64>,
    /// Preferred processed-layer valid fraction.
    #[serde(default)]
    pub processed_valid_fraction: Option<f64>,
    /// Diversity/uniqueness signal when available.
    #[serde(default)]
    pub diversity_unique_fraction: Option<f64>,
    /// Mean active slot fraction.
    #[serde(default)]
    pub slot_activation_mean: Option<f64>,
    /// Mean directed gate activation.
    #[serde(default)]
    pub gate_activation_mean: Option<f64>,
    /// Mean leakage proxy.
    #[serde(default)]
    pub leakage_proxy_mean: Option<f64>,
}

/// Efficiency slice copied into the evaluation matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationMatrixEfficiencyMetrics {
    /// Elapsed evaluation time in milliseconds.
    #[serde(default)]
    pub evaluation_time_ms: Option<f64>,
    /// Evaluated examples per second.
    #[serde(default)]
    pub examples_per_second: Option<f64>,
    /// Configured evaluation batch size.
    #[serde(default)]
    pub evaluation_batch_size: Option<usize>,
    /// Number of forward batches.
    #[serde(default)]
    pub forward_batch_count: Option<usize>,
    /// Whether evaluation ran without gradient tracking.
    #[serde(default)]
    pub no_grad: Option<bool>,
}

/// Split-level audit that separates optimizer-facing training terms from detached evaluation evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainEvalAlignmentReport {
    /// Schema version for this alignment report.
    pub schema_version: u32,
    /// Metric rows with explicit training/evaluation role attribution.
    #[serde(default)]
    pub metric_rows: Vec<TrainEvalAlignmentMetricRow>,
    /// Backend coverage rows for claim gates and reviewer tooling.
    #[serde(default)]
    pub backend_coverage: Vec<BackendCoverageContractRow>,
    /// Review of the configured validation best metric.
    #[serde(default)]
    pub best_metric_review: BestMetricReview,
    /// Compact decision for claim-facing interpretation.
    pub decision: String,
}

impl Default for TrainEvalAlignmentReport {
    fn default() -> Self {
        Self {
            schema_version: 1,
            metric_rows: Vec::new(),
            backend_coverage: Vec::new(),
            best_metric_review: BestMetricReview::default(),
            decision: "unavailable: train-eval alignment report not emitted".to_string(),
        }
    }
}

/// One metric attribution row in the train-eval alignment report.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainEvalAlignmentMetricRow {
    /// Stable metric name.
    pub metric_name: String,
    /// Coarse metric family, such as `representation`, `raw_candidate`, or `backend`.
    pub metric_family: String,
    /// Target or evidence source that produced this metric row.
    #[serde(default = "default_train_eval_target_source")]
    pub target_source: String,
    /// Evidence role, such as `optimizer_term`, `detached_diagnostic`, or `claim_backend`.
    pub evidence_role: String,
    /// Observed scalar value when available.
    #[serde(default)]
    pub observed_value: Option<f64>,
    /// Optimizer-facing objective terms that may use this metric family.
    #[serde(default)]
    pub optimizer_facing_terms: Vec<String>,
    /// Whether this metric is currently optimizer-facing under the active config.
    pub optimizer_facing: bool,
    /// Whether this metric is detached diagnostic evidence only.
    pub detached_diagnostic: bool,
    /// Candidate layer that owns this metric when applicable.
    #[serde(default)]
    pub candidate_layer: Option<String>,
    /// Whether the candidate layer is raw model-native evidence.
    pub model_native_raw: bool,
    /// Active method family when this row describes method-comparison evidence.
    #[serde(default)]
    pub method_family: Option<String>,
    /// Backend slot when this row depends on external/backend evidence.
    #[serde(default)]
    pub backend_slot: Option<String>,
    /// Backend identifier when available.
    #[serde(default)]
    pub backend_name: Option<String>,
    /// Whether the backend slot was available.
    #[serde(default)]
    pub backend_available: Option<bool>,
    /// Coverage fraction reported or inferred for the backend slot.
    #[serde(default)]
    pub backend_coverage_fraction: Option<f64>,
    /// Claim boundary attached to this metric row.
    pub claim_boundary: String,
}

fn default_train_eval_target_source() -> String {
    "legacy_unknown".to_string()
}

/// Backend coverage contract row used by claim gates.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendCoverageContractRow {
    /// Logical backend slot, such as `chemistry_validity`.
    pub backend_slot: String,
    /// Backend implementation name when available.
    #[serde(default)]
    pub backend_name: Option<String>,
    /// Whether the backend was available for this split.
    pub available: bool,
    /// Number or fraction of examples scored when reported.
    #[serde(default)]
    pub examples_scored: Option<f64>,
    /// Number or fraction of candidates scored when reported.
    #[serde(default)]
    pub candidates_scored: Option<f64>,
    /// Fraction of records missing required structure when reported.
    #[serde(default)]
    pub missing_structure_fraction: Option<f64>,
    /// General backend coverage fraction when reported.
    #[serde(default)]
    pub coverage_fraction: Option<f64>,
    /// Whether heuristic fallback is present or the backend is unavailable.
    pub fallback_status: String,
    /// Whether the row contains explicitly labeled heuristic fallback metrics.
    pub heuristic_fallback_labeled: bool,
    /// Claim boundary for interpreting the backend row.
    pub claim_boundary: String,
}

/// Review of `training.best_metric` for smoke versus claim-bearing use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestMetricReview {
    /// Configured best metric string.
    pub configured_best_metric: String,
    /// Normalized metric name without the `validation.` prefix.
    pub normalized_best_metric: String,
    /// Whether the metric is available on this evaluation artifact.
    pub metric_available: bool,
    /// Whether the metric is quality-aware rather than a smoke/runtime diagnostic.
    pub quality_aware: bool,
    /// Whether this metric is recommended for claim-bearing checkpoint selection.
    pub claim_bearing_recommended: bool,
    /// Availability requirement attached to this metric.
    pub availability_requirement: String,
    /// Stable status label.
    pub status: String,
    /// Human-readable warning or failure reason.
    #[serde(default)]
    pub warning: Option<String>,
}

impl Default for BestMetricReview {
    fn default() -> Self {
        Self {
            configured_best_metric: String::new(),
            normalized_best_metric: String::new(),
            metric_available: false,
            quality_aware: false,
            claim_bearing_recommended: false,
            availability_requirement: "unavailable".to_string(),
            status: "unavailable".to_string(),
            warning: Some("best metric review not emitted".to_string()),
        }
    }
}

/// Deterministic slot-stability diagnostics over one evaluated split.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotStabilityMetrics {
    /// Mean active-slot fraction for topology slots.
    pub topology_activation_mean: f64,
    /// Mean active-slot fraction for geometry slots.
    pub geometry_activation_mean: f64,
    /// Mean active-slot fraction for pocket slots.
    pub pocket_activation_mean: f64,
    /// Average pairwise similarity of topology slot signatures across examples.
    pub topology_signature_similarity: f64,
    /// Average pairwise similarity of geometry slot signatures across examples.
    pub geometry_signature_similarity: f64,
    /// Average pairwise similarity of pocket slot signatures across examples.
    pub pocket_signature_similarity: f64,
    /// Mean per-slot probe-alignment proxy for topology slots.
    pub topology_probe_alignment: f64,
    /// Mean per-slot probe-alignment proxy for geometry slots.
    pub geometry_probe_alignment: f64,
    /// Mean per-slot probe-alignment proxy for pocket slots.
    pub pocket_probe_alignment: f64,
    /// Per-slot topology semantic-alignment summary.
    #[serde(default)]
    pub topology_slot_alignment: Vec<f64>,
    /// Per-slot geometry semantic-alignment summary.
    #[serde(default)]
    pub geometry_slot_alignment: Vec<f64>,
    /// Per-slot pocket semantic-alignment summary.
    #[serde(default)]
    pub pocket_slot_alignment: Vec<f64>,
    /// Permutation-aware slot-signature matching reports.
    #[serde(default)]
    pub signature_matching: Vec<SlotSignatureMatchReport>,
    /// Per-modality slot-collapse alarms.
    #[serde(default)]
    pub collapse_warnings: Vec<SlotCollapseWarning>,
    /// Per-modality usage buckets and semantic-enrichment summaries.
    #[serde(default)]
    pub modality_usage: Vec<SlotModalityUsageReport>,
    /// Number of richer slot-collapse warnings available to adaptive stage guards.
    #[serde(default)]
    pub stage_guard_warning_count: usize,
}

/// Per-modality slot usage and semantic-enrichment summary for one split.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotModalityUsageReport {
    /// Modality label.
    pub modality: String,
    /// Number of examples contributing to the report.
    pub sample_count: usize,
    /// Fixed upper-bound slot count observed for this modality.
    pub slot_count: usize,
    /// Mean hard active-slot fraction.
    pub active_slot_fraction: f64,
    /// Mean attention-visible slot fraction after masking.
    pub attention_visible_fraction: f64,
    /// Mean slot assignment entropy.
    pub assignment_entropy: f64,
    /// Mean dominant slot assignment mass.
    pub dominant_slot_fraction: f64,
    /// Number of effectively dead slots by mean activation.
    pub dead_slot_count: usize,
    /// Number of weakly used diffuse slots by mean activation.
    pub diffuse_slot_count: usize,
    /// Number of nearly always-active slots by mean activation.
    pub saturated_slot_count: usize,
    /// Fraction of slots classified as dead.
    pub dead_slot_fraction: f64,
    /// Fraction of slots classified as diffuse.
    pub diffuse_slot_fraction: f64,
    /// Fraction of slots classified as saturated.
    pub saturated_slot_fraction: f64,
    /// Whether this modality contributes a collapse warning to readiness checks.
    pub stage_guard_collapse_warning: bool,
    /// Lightweight semantic-enrichment summary for this modality.
    pub semantic_enrichment: SlotSemanticEnrichmentSummary,
}

/// Compact semantic-enrichment summary derived from per-slot target alignment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotSemanticEnrichmentSummary {
    /// Target family used for this modality summary.
    pub target_family: String,
    /// Mean per-slot target-alignment proxy.
    pub mean_alignment: f64,
    /// Maximum per-slot target-alignment proxy.
    pub max_alignment: f64,
    /// Number of slots above the modality enrichment threshold.
    pub enriched_slot_count: usize,
    /// Entropy of the normalized per-slot alignment mass.
    pub enrichment_entropy: f64,
    /// Max-over-mean enrichment ratio; higher suggests specialization.
    pub role_enrichment_score: f64,
}

/// Persisted slot-semantic report emitted alongside experiment summaries.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotSemanticReportArtifact {
    /// Schema version for this report artifact.
    pub schema_version: u32,
    /// Validation split slot-stability metrics.
    pub validation: SlotStabilityMetrics,
    /// Test split slot-stability metrics.
    pub test: SlotStabilityMetrics,
    /// Latest training-step slot readiness snapshot, when training history is available.
    #[serde(default)]
    pub latest_training: Option<SlotSemanticTrainingSnapshot>,
    /// Optional cross-seed matching rows for multi-seed callers.
    #[serde(default)]
    pub cross_seed_matching: Vec<SlotSignatureMatchReport>,
}

/// Training-side slot readiness snapshot stored in the semantic report artifact.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotSemanticTrainingSnapshot {
    /// Global optimization step.
    pub step: usize,
    /// Effective stage index for this training step.
    pub stage_index: Option<usize>,
    /// Collapse warnings consumed by adaptive stage readiness.
    pub collapse_warning_count: usize,
    /// Warning labels emitted by slot-utilization diagnostics.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Per-modality slot signatures from the latest step.
    #[serde(default)]
    pub slot_signatures: Vec<crate::training::SlotSignatureStepSummary>,
}

/// Permutation-aware slot signature matching between two run/example groups.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotSignatureMatchReport {
    /// Modality label.
    pub modality: String,
    /// Comparison scope, such as cross_seed or within_split_repeated_signature_proxy.
    pub comparison_scope: String,
    /// Number of matched slots above threshold.
    pub matched_slot_count: usize,
    /// Number of unmatched slots in the left/reference group.
    pub unmatched_left_slots: usize,
    /// Number of unmatched slots in the right/comparison group.
    pub unmatched_right_slots: usize,
    /// Mean similarity over matched slots.
    pub mean_matched_similarity: f64,
    /// Explicit cross-seed matching score when the comparison scope is cross-seed.
    #[serde(default)]
    pub cross_seed_matching_score: f64,
    /// Whether the report suggests slot collapse or unstable matching.
    pub collapse_warning: bool,
}

/// Slot-collapse alarm for one modality.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotCollapseWarning {
    /// Modality label.
    pub modality: String,
    /// Compact status: balanced, dead, saturated, or single_slot_dominated.
    pub status: String,
    /// Mean hard active-slot fraction.
    pub active_slot_fraction: f64,
    /// Mean attention-visible slot fraction.
    pub attention_visible_fraction: f64,
    /// Mean slot assignment entropy.
    pub assignment_entropy: f64,
    /// Mean dominant slot mass.
    pub dominant_slot_fraction: f64,
    /// Human-readable warning.
    pub warning: String,
}

/// Candidate metrics split by raw model state and downstream postprocessing stages.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayeredGenerationMetrics {
    /// Canonical raw model output before repair, bond constraints, and reranking.
    #[serde(default)]
    pub raw_flow: CandidateLayerMetrics,
    /// Canonical constrained flow layer after bond/valence constraints but before final selection.
    #[serde(default)]
    pub constrained_flow: CandidateLayerMetrics,
    /// Canonical repaired layer after geometry repair but before bond-constrained selection.
    #[serde(default)]
    pub repaired: CandidateLayerMetrics,
    /// Direct final rollout state before repair and bond inference.
    pub raw_rollout: CandidateLayerMetrics,
    /// Geometry-repaired candidate state before bond inference.
    pub repaired_candidates: CandidateLayerMetrics,
    /// Repaired candidates after bond inference and valence pruning.
    pub inferred_bond_candidates: CandidateLayerMetrics,
    /// Proxy-reranked candidates selected from inferred-bond candidates.
    #[serde(default)]
    pub reranked_candidates: CandidateLayerMetrics,
    /// Previous fixed-weight deterministic proxy selector for reranker comparison.
    #[serde(default)]
    pub deterministic_proxy_candidates: CandidateLayerMetrics,
    /// Active bounded-reranker calibration metadata.
    #[serde(default)]
    pub reranker_calibration: RerankerCalibrationReport,
    /// Backend-scored metrics copied from the active chemistry/docking/pocket layer.
    pub backend_scored_candidates: BTreeMap<String, BTreeMap<String, f64>>,
    /// Method-aware comparison summary aligned to this layered artifact.
    #[serde(default)]
    pub method_comparison: MethodComparisonSummary,
    /// Active flow-head ablation identity for compression-risk audits.
    #[serde(default)]
    pub flow_head_ablation: FlowHeadAblationDiagnostics,
    /// Aggregated scalar diagnostics emitted by flow velocity heads.
    #[serde(default)]
    pub flow_head_diagnostics: BTreeMap<String, f64>,
    /// Explicit contract separating native, constrained, repaired, reranked, and backend layers.
    #[serde(default)]
    pub generation_path_contract: Vec<GenerationPathContractRow>,
    /// Raw-vs-repaired failure-case audit for postprocessing claim boundaries.
    #[serde(default)]
    pub repair_case_audit: RepairCaseAuditReport,
}

/// Stable contract row for one claim-facing generation path/layer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationPathContractRow {
    /// Canonical claim-facing layer name.
    pub canonical_layer: String,
    /// Legacy field name used in persisted artifacts.
    pub legacy_field_name: String,
    /// Coarse generation path class.
    pub generation_path_class: String,
    /// Whether this row is raw model-native output.
    pub model_native_raw: bool,
    /// Whether this layer should be interpreted as backend-supported.
    pub backend_supported: bool,
    /// Ordered postprocessor or selection steps represented by the layer.
    pub postprocessor_chain: Vec<String>,
    /// Claim boundary for this layer.
    pub claim_boundary: String,
}

/// Canonical generation path contract emitted with layered metrics.
pub fn canonical_generation_path_contract() -> Vec<GenerationPathContractRow> {
    let mut rows = Vec::new();
    rows.push(GenerationPathContractRow {
        canonical_layer: "raw_molecular_flow_logits".to_string(),
        legacy_field_name: "flow_head_logits".to_string(),
        generation_path_class: "model_native_logits".to_string(),
        model_native_raw: true,
        backend_supported: false,
        postprocessor_chain: Vec::new(),
        claim_boundary:
            "raw flow-head logits before native graph extraction, constraints, repair, or reranking"
                .to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: "raw_native_graph_extraction".to_string(),
        legacy_field_name: "native_bonds".to_string(),
        generation_path_class: CandidateLayerKind::RawRollout
            .generation_path_class()
            .to_string(),
        model_native_raw: true,
        backend_supported: false,
        postprocessor_chain: vec!["thresholded_native_graph_extraction".to_string()],
        claim_boundary:
            "thresholded model-native graph extraction before connectivity, density, valence, repair, or reranking constraints"
                .to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: "constrained_native_graph".to_string(),
        legacy_field_name: "constrained_native_bonds".to_string(),
        generation_path_class: CandidateLayerKind::InferredBond
            .generation_path_class()
            .to_string(),
        model_native_raw: false,
        backend_supported: false,
        postprocessor_chain: vec![
            "native_graph_connectivity_guardrail".to_string(),
            "native_graph_density_limit".to_string(),
            "native_graph_valence_guardrail".to_string(),
        ],
        claim_boundary:
            "native graph after deterministic connectivity, density, and valence guardrails; do not cite as raw graph quality"
                .to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: "raw_flow".to_string(),
        legacy_field_name: "raw_flow".to_string(),
        generation_path_class: CandidateLayerKind::RawRollout
            .generation_path_class()
            .to_string(),
        model_native_raw: true,
        backend_supported: false,
        postprocessor_chain: Vec::new(),
        claim_boundary: CandidateLayerKind::RawRollout.claim_boundary().to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: CandidateLayerKind::RawRollout
            .canonical_generation_layer()
            .to_string(),
        legacy_field_name: CandidateLayerKind::RawRollout
            .legacy_field_name()
            .to_string(),
        generation_path_class: CandidateLayerKind::RawRollout
            .generation_path_class()
            .to_string(),
        model_native_raw: CandidateLayerKind::RawRollout.is_model_native_raw(),
        backend_supported: false,
        postprocessor_chain: Vec::new(),
        claim_boundary: CandidateLayerKind::RawRollout.claim_boundary().to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: "constrained_flow".to_string(),
        legacy_field_name: "constrained_flow".to_string(),
        generation_path_class: CandidateLayerKind::InferredBond
            .generation_path_class()
            .to_string(),
        model_native_raw: false,
        backend_supported: false,
        postprocessor_chain: vec![
            "bond_inference".to_string(),
            "valence_pruning".to_string(),
        ],
        claim_boundary: CandidateLayerKind::InferredBond
            .claim_boundary()
            .to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: CandidateLayerKind::InferredBond
            .canonical_generation_layer()
            .to_string(),
        legacy_field_name: CandidateLayerKind::InferredBond
            .legacy_field_name()
            .to_string(),
        generation_path_class: CandidateLayerKind::InferredBond
            .generation_path_class()
            .to_string(),
        model_native_raw: CandidateLayerKind::InferredBond.is_model_native_raw(),
        backend_supported: false,
        postprocessor_chain: vec![
            "bond_inference".to_string(),
            "valence_pruning".to_string(),
        ],
        claim_boundary: CandidateLayerKind::InferredBond
            .claim_boundary()
            .to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: CandidateLayerKind::Repaired
            .canonical_generation_layer()
            .to_string(),
        legacy_field_name: "repaired".to_string(),
        generation_path_class: CandidateLayerKind::Repaired
            .generation_path_class()
            .to_string(),
        model_native_raw: false,
        backend_supported: false,
        postprocessor_chain: vec!["geometry_repair".to_string()],
        claim_boundary: CandidateLayerKind::Repaired.claim_boundary().to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: CandidateLayerKind::Repaired
            .canonical_generation_layer()
            .to_string(),
        legacy_field_name: CandidateLayerKind::Repaired
            .legacy_field_name()
            .to_string(),
        generation_path_class: CandidateLayerKind::Repaired
            .generation_path_class()
            .to_string(),
        model_native_raw: CandidateLayerKind::Repaired.is_model_native_raw(),
        backend_supported: false,
        postprocessor_chain: vec!["geometry_repair".to_string()],
        claim_boundary: CandidateLayerKind::Repaired.claim_boundary().to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: CandidateLayerKind::DeterministicProxy
            .canonical_generation_layer()
            .to_string(),
        legacy_field_name: CandidateLayerKind::DeterministicProxy
            .legacy_field_name()
            .to_string(),
        generation_path_class: CandidateLayerKind::DeterministicProxy
            .generation_path_class()
            .to_string(),
        model_native_raw: CandidateLayerKind::DeterministicProxy.is_model_native_raw(),
        backend_supported: false,
        postprocessor_chain: vec!["deterministic_proxy_selection".to_string()],
        claim_boundary: CandidateLayerKind::DeterministicProxy
            .claim_boundary()
            .to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: CandidateLayerKind::Reranked
            .canonical_generation_layer()
            .to_string(),
        legacy_field_name: CandidateLayerKind::Reranked
            .legacy_field_name()
            .to_string(),
        generation_path_class: CandidateLayerKind::Reranked
            .generation_path_class()
            .to_string(),
        model_native_raw: CandidateLayerKind::Reranked.is_model_native_raw(),
        backend_supported: false,
        postprocessor_chain: vec![
            "bond_inference".to_string(),
            "valence_pruning".to_string(),
            "calibrated_reranking".to_string(),
        ],
        claim_boundary: CandidateLayerKind::Reranked.claim_boundary().to_string(),
    });
    rows.push(GenerationPathContractRow {
        canonical_layer: "external_scored_candidates".to_string(),
        legacy_field_name: "backend_scored_candidates".to_string(),
        generation_path_class: "external_backend".to_string(),
        model_native_raw: false,
        backend_supported: true,
        postprocessor_chain: vec!["external_backend_scoring".to_string()],
        claim_boundary:
            "external backend scoring evidence; supports downstream scoring claims, not raw-native model quality by itself"
                .to_string(),
    });
    rows
}

/// Flow-head identity and diagnostic availability for ablation reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowHeadAblationDiagnostics {
    /// Schema version for this diagnostics block.
    pub schema_version: u32,
    /// Configured velocity-head kind.
    pub head_kind: String,
    /// Whether local atom-to-pocket attention is selected.
    pub local_atom_pocket_attention: bool,
    /// Whether the EGNN-style equivariant geometry head is selected.
    #[serde(default)]
    pub equivariant_geometry_head: bool,
    /// Whether pairwise ligand geometry messages are enabled.
    pub pairwise_geometry_enabled: bool,
    /// Stable ablation label used in configs and reports.
    pub ablation_label: String,
    /// Decoder conditioning family used by atom and coordinate decoder heads.
    #[serde(default = "default_decoder_conditioning_label")]
    pub decoder_conditioning_kind: String,
    /// Full molecular-flow conditioning family used by atom, bond, topology, and pocket heads.
    #[serde(default = "default_decoder_conditioning_label")]
    pub molecular_flow_conditioning_kind: String,
    /// Whether slot-local conditioning is active for generation heads in this run.
    #[serde(default = "default_slot_local_conditioning_enabled")]
    pub slot_local_conditioning_enabled: bool,
    /// Whether this run is the explicit mean-pooled conditioning ablation.
    #[serde(default)]
    pub mean_pooled_conditioning_ablation: bool,
    /// Whether scalar flow-head diagnostics were observed.
    pub diagnostics_available: bool,
    /// Claim boundary for this ablation.
    pub claim_boundary: String,
    /// Flow branch ids enabled by the current config.
    #[serde(default)]
    pub enabled_flow_branches: Vec<String>,
    /// Required full-flow branches not enabled by the current config.
    #[serde(default)]
    pub disabled_flow_branches: Vec<String>,
    /// Whether this run may claim full molecular flow.
    #[serde(default)]
    pub full_molecular_flow_claim_allowed: bool,
    /// Stable reason for full-flow claim gating.
    #[serde(default)]
    pub claim_gate_reason: String,
    /// Configured target matching/alignment policy for de novo flow supervision.
    #[serde(default = "default_target_alignment_policy_label")]
    pub target_alignment_policy: String,
    /// Whether the matching policy is non-index and safe for full-flow de novo claims.
    #[serde(default)]
    pub target_matching_claim_safe: bool,
    /// Artifact fields that carry optimizer-step matching provenance.
    #[serde(default)]
    pub target_matching_artifact_fields: Vec<String>,
}

impl Default for FlowHeadAblationDiagnostics {
    fn default() -> Self {
        Self {
            schema_version: 1,
            head_kind: "geometry".to_string(),
            local_atom_pocket_attention: false,
            equivariant_geometry_head: false,
            pairwise_geometry_enabled: false,
            ablation_label: "geometry_mean_pooling".to_string(),
            decoder_conditioning_kind: default_decoder_conditioning_label(),
            molecular_flow_conditioning_kind: default_decoder_conditioning_label(),
            slot_local_conditioning_enabled: default_slot_local_conditioning_enabled(),
            mean_pooled_conditioning_ablation: false,
            diagnostics_available: false,
            claim_boundary: "geometry-first coordinate velocity only; no topology or bond flow"
                .to_string(),
            enabled_flow_branches: vec!["geometry".to_string()],
            disabled_flow_branches: vec![
                "atom_type".to_string(),
                "bond".to_string(),
                "topology".to_string(),
                "pocket_context".to_string(),
            ],
            full_molecular_flow_claim_allowed: false,
            claim_gate_reason: "claim_not_requested".to_string(),
            target_alignment_policy: "pad_with_mask".to_string(),
            target_matching_claim_safe: false,
            target_matching_artifact_fields: vec![
                "training_history[].losses.primary.branch_schedule.entries[].target_matching_policy"
                    .to_string(),
                "training_history[].losses.primary.branch_schedule.entries[].target_matching_coverage"
                    .to_string(),
                "training_history[].losses.primary.branch_schedule.entries[].target_matching_mean_cost"
                    .to_string(),
            ],
        }
    }
}

fn default_decoder_conditioning_label() -> String {
    "local_atom_slot_attention".to_string()
}

fn default_slot_local_conditioning_enabled() -> bool {
    true
}

/// Method-aware comparison summary persisted alongside layered candidate metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MethodComparisonSummary {
    /// Active method metadata for the current surface.
    #[serde(default)]
    pub active_method: Option<PocketGenerationMethodMetadata>,
    /// Method family of the active row, duplicated for quick report scans.
    #[serde(default)]
    pub active_method_family: Option<String>,
    /// Raw-native model evidence presented before processed method layers.
    #[serde(default)]
    pub raw_native_evidence: ClaimRawNativeEvidenceSummary,
    /// Additive processed/repaired/reranked evidence kept separate from raw-native metrics.
    #[serde(default)]
    pub processed_generation_evidence: ClaimProcessedGenerationEvidenceSummary,
    /// Family-aware metric layer selected for the active method row.
    #[serde(default)]
    pub active_selected_metric_layer: Option<String>,
    /// One comparison row per executed or declared method.
    #[serde(default)]
    pub methods: Vec<MethodComparisonRow>,
    /// Planned backend-agnostic metric interfaces reserved for future work.
    #[serde(default)]
    pub planned_metric_interfaces: Vec<PlannedMetricInterface>,
    /// Additive preference-alignment artifact summary.
    #[serde(default)]
    pub preference_alignment: PreferenceAlignmentSummary,
    /// Flow-matching focused layer metrics extracted from method comparison outputs.
    #[serde(default)]
    pub flow_metrics: FlowMethodMetrics,
}

/// Focused flow-matching layer metrics and deltas against conditioned denoising.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FlowMethodMetrics {
    /// Raw flow output quality metrics when flow backend is available.
    #[serde(default)]
    pub raw_output: Option<CandidateLayerMetrics>,
    /// Repaired flow output quality metrics when flow backend is available.
    #[serde(default)]
    pub repaired_output: Option<CandidateLayerMetrics>,
    /// Flow-vs-denoising deltas (flow minus denoising) for shared metrics.
    #[serde(default)]
    pub versus_conditioned_denoising: Option<FlowVsDenoisingDelta>,
}

/// Compact flow-vs-denoising delta report.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FlowVsDenoisingDelta {
    /// Delta in native valid fraction (flow - denoising).
    pub native_valid_fraction_delta: f64,
    /// Delta in native pocket-contact fraction (flow - denoising).
    pub native_pocket_contact_fraction_delta: f64,
    /// Delta in native clash fraction (flow - denoising); lower-is-better in absolute terms.
    pub native_clash_fraction_delta: f64,
}

/// Raw-native claim gate evaluated before processed, repaired, or reranked metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawNativeClaimGateReport {
    /// Stable gate label.
    #[serde(default = "default_raw_native_claim_gate_name")]
    pub gate_name: String,
    /// Whether the raw-native checks pass without consulting processed metrics.
    #[serde(default)]
    pub passed: bool,
    /// Per-metric threshold checks applied to raw-native model evidence.
    #[serde(default)]
    pub checks: BTreeMap<String, BackendThresholdCheck>,
    /// Failed raw-native gate reasons.
    #[serde(default)]
    pub failed_reasons: Vec<String>,
    /// Whether processed/repaired/reranked metrics are deliberately excluded from this gate.
    #[serde(default = "default_true")]
    pub processed_metrics_excluded: bool,
    /// Reviewer-facing gate decision.
    #[serde(default)]
    pub decision: String,
}

impl Default for RawNativeClaimGateReport {
    fn default() -> Self {
        Self {
            gate_name: default_raw_native_claim_gate_name(),
            passed: false,
            checks: BTreeMap::new(),
            failed_reasons: Vec::new(),
            processed_metrics_excluded: true,
            decision: "raw-native claim gate unavailable".to_string(),
        }
    }
}

fn default_raw_native_claim_gate_name() -> String {
    "raw_native_claim_gate".to_string()
}

fn default_true() -> bool {
    true
}

/// First claim-facing section for model-native evidence before any downstream processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimRawNativeEvidenceSummary {
    /// Schema version for the raw-native evidence block.
    #[serde(default = "default_claim_evidence_schema_version")]
    pub schema_version: u32,
    /// Evidence role attached to this block.
    #[serde(default = "default_raw_native_evidence_role")]
    pub evidence_role: String,
    /// Candidate layer used for raw model-native evidence.
    #[serde(default)]
    pub raw_model_layer: String,
    /// Whether the selected layer is contractually model-native raw output.
    #[serde(default)]
    pub model_native_raw: bool,
    /// Candidate count in the raw layer.
    #[serde(default)]
    pub candidate_count: usize,
    /// Raw model-native valid fraction.
    #[serde(default)]
    pub valid_fraction: Option<f64>,
    /// Raw native graph-valid fraction before repair or inferred-bond constraints.
    #[serde(default)]
    pub native_graph_valid_fraction: Option<f64>,
    /// Raw native mean bond count.
    #[serde(default)]
    pub native_bond_count_mean: Option<f64>,
    /// Raw native mean connected-component count.
    #[serde(default)]
    pub native_component_count_mean: Option<f64>,
    /// Raw native valence-violation fraction.
    #[serde(default)]
    pub native_valence_violation_fraction: Option<f64>,
    /// Raw native bond-payload synchronization fraction.
    #[serde(default)]
    pub topology_bond_sync_fraction: Option<f64>,
    /// Raw mean centroid offset from the pocket center.
    #[serde(default)]
    pub mean_centroid_offset: Option<f64>,
    /// Raw mean rollout displacement.
    #[serde(default)]
    pub mean_displacement: Option<f64>,
    /// Raw clash fraction.
    #[serde(default)]
    pub clash_fraction: Option<f64>,
    /// Raw pocket-contact fraction.
    #[serde(default)]
    pub pocket_contact_fraction: Option<f64>,
    /// Raw compact strict pocket-fit proxy.
    #[serde(default)]
    pub strict_pocket_fit_score: Option<f64>,
    /// Enabled molecular-flow branches for branch-claim review.
    #[serde(default)]
    pub enabled_flow_branches: Vec<String>,
    /// Required full-flow branches that are disabled.
    #[serde(default)]
    pub disabled_flow_branches: Vec<String>,
    /// Whether full molecular-flow wording is allowed by the branch contract.
    #[serde(default)]
    pub full_molecular_flow_claim_allowed: bool,
    /// Stable branch-gate reason for full molecular-flow wording.
    #[serde(default)]
    pub branch_claim_gate_reason: String,
    /// Whether de novo target matching is claim-safe.
    #[serde(default)]
    pub target_matching_claim_safe: bool,
    /// Slot activation evidence carried alongside raw model evidence.
    #[serde(default)]
    pub slot_activation_mean: Option<f64>,
    /// Controlled-interaction gate activation evidence.
    #[serde(default)]
    pub gate_activation_mean: Option<f64>,
    /// Leakage evidence used to bound specialization claims.
    #[serde(default)]
    pub leakage_proxy_mean: Option<f64>,
    /// Claim boundary for interpreting this raw layer.
    #[serde(default)]
    pub claim_boundary: String,
    /// Independent gate for raw-native regressions.
    #[serde(default)]
    pub raw_native_gate: RawNativeClaimGateReport,
}

impl Default for ClaimRawNativeEvidenceSummary {
    fn default() -> Self {
        Self {
            schema_version: default_claim_evidence_schema_version(),
            evidence_role: default_raw_native_evidence_role(),
            raw_model_layer: "unavailable".to_string(),
            model_native_raw: false,
            candidate_count: 0,
            valid_fraction: None,
            native_graph_valid_fraction: None,
            native_bond_count_mean: None,
            native_component_count_mean: None,
            native_valence_violation_fraction: None,
            topology_bond_sync_fraction: None,
            mean_centroid_offset: None,
            mean_displacement: None,
            clash_fraction: None,
            pocket_contact_fraction: None,
            strict_pocket_fit_score: None,
            enabled_flow_branches: Vec::new(),
            disabled_flow_branches: Vec::new(),
            full_molecular_flow_claim_allowed: false,
            branch_claim_gate_reason: String::new(),
            target_matching_claim_safe: false,
            slot_activation_mean: None,
            gate_activation_mean: None,
            leakage_proxy_mean: None,
            claim_boundary: "raw-native evidence unavailable".to_string(),
            raw_native_gate: RawNativeClaimGateReport::default(),
        }
    }
}

fn default_claim_evidence_schema_version() -> u32 {
    1
}

fn default_raw_native_evidence_role() -> String {
    "model_native_raw_first".to_string()
}

/// Additive claim-facing section for constrained, repaired, selected, or reranked metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimProcessedGenerationEvidenceSummary {
    /// Schema version for the processed evidence block.
    #[serde(default = "default_claim_evidence_schema_version")]
    pub schema_version: u32,
    /// Evidence role attached to this block.
    #[serde(default = "default_processed_evidence_role")]
    pub evidence_role: String,
    /// Processed layer selected for additive performance interpretation.
    #[serde(default)]
    pub processed_layer: String,
    /// Whether the processed layer is still model-native raw output.
    #[serde(default)]
    pub model_native_raw: bool,
    /// Candidate count in the processed layer.
    #[serde(default)]
    pub candidate_count: usize,
    /// Processed valid fraction.
    #[serde(default)]
    pub valid_fraction: Option<f64>,
    /// Processed pocket-contact fraction.
    #[serde(default)]
    pub pocket_contact_fraction: Option<f64>,
    /// Processed compact strict pocket-fit proxy.
    #[serde(default)]
    pub strict_pocket_fit_score: Option<f64>,
    /// Processed mean centroid offset.
    #[serde(default)]
    pub mean_centroid_offset: Option<f64>,
    /// Processed clash fraction.
    #[serde(default)]
    pub clash_fraction: Option<f64>,
    /// Processed minus raw valid-fraction delta.
    #[serde(default)]
    pub processing_valid_fraction_delta: Option<f64>,
    /// Processed minus raw pocket-contact delta.
    #[serde(default)]
    pub processing_pocket_contact_delta: Option<f64>,
    /// Processed minus raw clash-fraction delta.
    #[serde(default)]
    pub processing_clash_delta: Option<f64>,
    /// Ordered postprocessing or selection chain.
    #[serde(default)]
    pub postprocessor_chain: Vec<String>,
    /// Claim boundary for interpreting this processed layer.
    #[serde(default)]
    pub claim_boundary: String,
    /// Reviewer-facing note keeping processed metrics additive rather than raw-native.
    #[serde(default)]
    pub additive_interpretation: String,
}

impl Default for ClaimProcessedGenerationEvidenceSummary {
    fn default() -> Self {
        Self {
            schema_version: default_claim_evidence_schema_version(),
            evidence_role: default_processed_evidence_role(),
            processed_layer: "unavailable".to_string(),
            model_native_raw: false,
            candidate_count: 0,
            valid_fraction: None,
            pocket_contact_fraction: None,
            strict_pocket_fit_score: None,
            mean_centroid_offset: None,
            clash_fraction: None,
            processing_valid_fraction_delta: None,
            processing_pocket_contact_delta: None,
            processing_clash_delta: None,
            postprocessor_chain: Vec::new(),
            claim_boundary: "processed evidence unavailable".to_string(),
            additive_interpretation: String::new(),
        }
    }
}

fn default_processed_evidence_role() -> String {
    "additive_processed_or_reranked_evidence".to_string()
}

/// Additive summary for interaction-profile and preference-pair artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceAlignmentSummary {
    /// Schema version for this summary section.
    pub schema_version: u32,
    /// Whether interaction-profile extraction was enabled.
    pub profile_extraction_enabled: bool,
    /// Whether preference-pair construction was enabled.
    pub pair_construction_enabled: bool,
    /// Number of profiles emitted when artifacts are present.
    pub profile_count: usize,
    /// Number of preference pairs emitted when artifacts are present.
    pub preference_pair_count: usize,
    /// Expected profile artifact path relative to the run directory.
    #[serde(default)]
    pub profile_artifact: Option<String>,
    /// Expected pair artifact path relative to the run directory.
    #[serde(default)]
    pub preference_pair_artifact: Option<String>,
    /// Expected reranker-summary artifact path relative to the run directory.
    #[serde(default)]
    pub reranker_summary_artifact: Option<String>,
    /// Missing artifacts mean unavailable evidence, not failed alignment.
    pub missing_artifacts_mean_unavailable: bool,
}

impl Default for PreferenceAlignmentSummary {
    fn default() -> Self {
        Self {
            schema_version: crate::models::PREFERENCE_RERANKER_SCHEMA_VERSION,
            profile_extraction_enabled: false,
            pair_construction_enabled: false,
            profile_count: 0,
            preference_pair_count: 0,
            profile_artifact: None,
            preference_pair_artifact: None,
            reranker_summary_artifact: None,
            missing_artifacts_mean_unavailable: true,
        }
    }
}

/// Planned backend-agnostic metric surface.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlannedMetricInterface {
    /// Stable interface key.
    pub interface_id: String,
    /// Human-readable description.
    pub description: String,
    /// Whether the interface is intentionally backend-agnostic.
    pub backend_agnostic: bool,
}

/// Compact quality summary for a homogeneous candidate layer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CandidateLayerMetrics {
    /// Canonical candidate layer name represented by this homogeneous summary.
    #[serde(default = "default_candidate_layer_label")]
    pub layer_name: String,
    /// Provenance label for pocket-interaction proxy metrics in this layer.
    #[serde(default = "default_pocket_interaction_provenance")]
    pub pocket_interaction_provenance: String,
    /// Explicit generation-mode contract shared by candidates in this layer.
    #[serde(default = "default_generation_mode_label")]
    pub generation_mode: String,
    /// Number of candidate records in the layer.
    pub candidate_count: usize,
    /// Fraction with nonempty finite atom and coordinate payloads.
    pub valid_fraction: f64,
    /// Fraction with at least one atom near the pocket envelope.
    pub pocket_contact_fraction: f64,
    /// Coarse atom-pocket distance-bin fit proxy for this layer.
    #[serde(default)]
    pub pocket_distance_bin_accuracy: f64,
    /// Contact precision proxy: contacted candidates discounted by clash burden.
    #[serde(default)]
    pub pocket_contact_precision_proxy: f64,
    /// Contact recall proxy: average fraction of atoms close to the pocket envelope.
    #[serde(default)]
    pub pocket_contact_recall_proxy: f64,
    /// Optional role-compatibility proxy from ligand atom roles and pocket contact geometry.
    #[serde(default)]
    pub pocket_role_compatibility_proxy: f64,
    /// Average centroid offset from the pocket centroid.
    pub mean_centroid_offset: f64,
    /// Mean non-bonded clash proxy; lower is better.
    pub clash_fraction: f64,
    /// Mean final rollout coordinate displacement when available.
    #[serde(default)]
    pub mean_displacement: f64,
    /// Mean final rollout atom-change fraction when available.
    #[serde(default)]
    pub atom_change_fraction: f64,
    /// Fraction of unique validity-eligible equivalence classes among all candidates.
    ///
    /// Invalid candidates do not increase this fraction. The equivalence class
    /// is permutation-invariant over atom order and uses atom composition,
    /// typed bonds, pairwise distances, and pocket-radial distance buckets.
    pub uniqueness_proxy_fraction: f64,
    /// Fraction of candidates eligible for diversity accounting.
    ///
    /// Eligible candidates have finite atom/coordinate payloads, synchronized
    /// native bond indices, and no cached valence violations.
    #[serde(default)]
    pub diversity_eligible_fraction: f64,
    /// Unique equivalence-class fraction among eligible candidates only.
    #[serde(default)]
    pub validity_conditioned_unique_fraction: f64,
    /// Fraction of eligible candidates that collapse into duplicate equivalence classes.
    #[serde(default)]
    pub equivalence_duplicate_fraction: f64,
    /// Fraction of candidates excluded from diversity accounting by raw validity checks.
    #[serde(default)]
    pub invalid_diversity_excluded_fraction: f64,
    /// Provenance for the conservative diversity metric.
    #[serde(default)]
    pub diversity_metric_source: String,
    /// Fraction of unique atom-type sequences in the candidate layer.
    #[serde(default)]
    pub atom_type_sequence_diversity: f64,
    /// Fraction of unique inferred-bond topology signatures in the candidate layer.
    #[serde(default)]
    pub bond_topology_diversity: f64,
    /// Fraction of unique coarse coordinate-shape signatures in the candidate layer.
    #[serde(default)]
    pub coordinate_shape_diversity: f64,
    /// Fraction of atom-type signatures not present in the training-reference set.
    #[serde(default)]
    pub novel_atom_type_sequence_fraction: f64,
    /// Fraction of bond-topology signatures not present in the training-reference set.
    #[serde(default)]
    pub novel_bond_topology_fraction: f64,
    /// Fraction of coarse coordinate-shape signatures not present in the training-reference set.
    #[serde(default)]
    pub novel_coordinate_shape_fraction: f64,
    /// Fraction of scaffold signatures not present in the split-local training references.
    #[serde(default)]
    pub scaffold_novelty_fraction: f64,
    /// Fraction of unique scaffold signatures in this layer.
    #[serde(default)]
    pub unique_scaffold_fraction: f64,
    /// Mean pairwise fingerprint Tanimoto similarity within this layer.
    #[serde(default)]
    pub pairwise_tanimoto_mean: f64,
    /// Mean nearest-neighbor fingerprint similarity to split-local training references.
    #[serde(default)]
    pub nearest_train_similarity: f64,
    /// Fraction of candidates with usable scaffold/fingerprint evidence.
    #[serde(default)]
    pub scaffold_metric_coverage_fraction: f64,
    /// Proxy hydrogen-bond interaction score for this layer.
    #[serde(default)]
    pub hydrogen_bond_proxy: f64,
    /// Proxy hydrophobic-contact interaction score for this layer.
    #[serde(default)]
    pub hydrophobic_contact_proxy: f64,
    /// Mean residue-level contact count when residue identities are available.
    #[serde(default)]
    pub residue_contact_count: f64,
    /// Fraction of candidates carrying source pocket residue identity provenance.
    #[serde(default)]
    pub residue_identity_coverage_fraction: f64,
    /// Fraction of contacted key residues over key residues observed in the pocket.
    #[serde(default)]
    pub key_residue_contact_coverage: f64,
    /// Clash burden aligned to interaction-profile naming; lower is better.
    #[serde(default)]
    pub clash_burden: f64,
    /// Balance between broad pocket contact and overpacked clash avoidance.
    #[serde(default)]
    pub contact_balance: f64,
    /// Fraction of candidates with usable interaction-profile proxy evidence.
    #[serde(default)]
    pub interaction_profile_coverage_fraction: f64,
    /// Mean native bond count for this candidate layer.
    #[serde(default)]
    pub native_bond_count_mean: f64,
    /// Mean connected-component count for the native graph payload.
    #[serde(default)]
    pub native_component_count_mean: f64,
    /// Mean atom-level valence violation fraction for the native graph payload.
    #[serde(default)]
    pub native_valence_violation_fraction: f64,
    /// Fraction of candidates whose native graph has more than one disconnected fragment.
    #[serde(default)]
    pub native_disconnected_fragment_fraction: f64,
    /// Fraction of candidates with inconsistent native bond-order/type provenance.
    #[serde(default)]
    pub native_bond_order_conflict_fraction: f64,
    /// Mean number of graph guardrail/repair actions separating raw native and constrained layers.
    #[serde(default)]
    pub native_graph_repair_delta_mean: f64,
    /// Mean raw native bonds removed by constrained graph extraction.
    #[serde(default)]
    pub native_raw_to_constrained_removed_bond_count_mean: f64,
    /// Mean connectivity guardrail bonds added to keep the native graph connected.
    #[serde(default)]
    pub native_connectivity_guardrail_added_bond_count_mean: f64,
    /// Mean valence downgrade actions applied during native graph extraction.
    #[serde(default)]
    pub native_valence_guardrail_downgrade_count_mean: f64,
    /// Fraction whose cached bond count and explicit bond list are synchronized.
    #[serde(default)]
    pub topology_bond_sync_fraction: f64,
    /// Entropy of native atom-type tokens in this layer.
    #[serde(default)]
    pub atom_type_entropy: f64,
    /// Fraction passing native graph validity checks before downstream repair.
    #[serde(default)]
    pub native_graph_valid_fraction: f64,
    /// Boundary label explaining whether these native graph metrics are raw or postprocessed.
    #[serde(default)]
    pub native_graph_metric_source: String,
}

/// Standalone report focused on raw model-native generation evidence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawNativeGenerationReport {
    /// Schema version for compatibility checks.
    pub schema_version: u32,
    /// Human-readable report role.
    pub report_role: String,
    /// Claim-facing split used for this report.
    pub split_label: String,
    /// Raw model-native layer summary.
    pub raw_native: RawNativeLayerSummary,
    /// Additive processed/repaired layer summary.
    pub processed: RawNativeLayerSummary,
    /// Rollout diagnostics tied to raw generation.
    pub rollout_diagnostics: RawNativeRolloutDiagnostics,
    /// Latest objective-family budget report from training, when available.
    #[serde(default)]
    pub objective_families: Vec<crate::training::ObjectiveFamilyBudgetEntry>,
    /// Claim eligibility summary separating supported from unsupported statements.
    pub claim_eligibility: RawNativeClaimEligibility,
}

/// Compact layer summary for raw-native generation reporting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawNativeLayerSummary {
    /// Candidate layer label.
    pub layer_name: String,
    /// Whether this layer is model-native raw output.
    pub model_native_raw: bool,
    /// Number of candidates represented.
    pub candidate_count: usize,
    /// Valid candidate fraction.
    pub valid_fraction: f64,
    /// Pocket-contact fraction.
    pub pocket_contact_fraction: f64,
    /// Clash fraction.
    pub clash_fraction: f64,
    /// Diversity fraction among validity-eligible candidates.
    pub validity_conditioned_unique_fraction: f64,
    /// Claim boundary for this layer.
    pub claim_boundary: String,
}

/// Raw rollout diagnostics copied into the standalone report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawNativeRolloutDiagnostics {
    /// Generation mode label.
    pub generation_mode: String,
    /// Mean raw rollout displacement.
    pub raw_model_mean_displacement: f64,
    /// Latest optimizer-facing rollout training enabled flag.
    #[serde(default)]
    pub latest_rollout_training_enabled: bool,
    /// Latest optimizer-facing rollout training active flag.
    #[serde(default)]
    pub latest_rollout_training_active: bool,
    /// Latest generated-state validity proxy from rollout training.
    #[serde(default)]
    pub latest_generated_state_validity: Option<f64>,
}

/// Claim eligibility section for raw-native generation reporting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawNativeClaimEligibility {
    /// support label: supported, weakly_supported, or unsupported.
    pub status: String,
    /// Whether repaired/processed evidence is additive rather than raw-native evidence.
    pub processed_evidence_additive_only: bool,
    /// Reasons preventing stronger raw-native claims.
    #[serde(default)]
    pub unsupported_reasons: Vec<String>,
}

/// Compact per-candidate metric snapshot for raw-vs-repaired repair audits.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepairCandidateMetricSnapshot {
    /// Whether the candidate has finite atom and coordinate payloads.
    #[serde(default)]
    pub valid: bool,
    /// Whether native graph payloads pass pre-repair validity checks.
    #[serde(default)]
    pub native_graph_valid: bool,
    /// Whether this candidate has at least one pocket contact.
    #[serde(default)]
    pub pocket_contact: bool,
    /// Centroid offset from the pocket center.
    #[serde(default)]
    pub centroid_offset: f64,
    /// Non-bonded clash proxy.
    #[serde(default)]
    pub clash_fraction: f64,
    /// Compact strict pocket-fit proxy.
    #[serde(default)]
    pub strict_pocket_fit_score: f64,
    /// Candidate bond count.
    #[serde(default)]
    pub bond_count: usize,
    /// Connected-component count.
    #[serde(default)]
    pub component_count: f64,
    /// Valence violation fraction.
    #[serde(default)]
    pub valence_violation_fraction: f64,
}

/// Representative raw-vs-repaired candidate pair for postprocessing review.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepairCaseRecord {
    /// Case role: `repair_helps`, `repair_harms`, `repair_neutral`, or `raw_failure`.
    #[serde(default)]
    pub case_role: String,
    /// Stable example id.
    #[serde(default)]
    pub example_id: String,
    /// Protein id for pocket-level review.
    #[serde(default)]
    pub protein_id: String,
    /// Raw candidate index in the retained artifact layer.
    #[serde(default)]
    pub raw_candidate_index: usize,
    /// Repaired candidate index in the retained artifact layer.
    #[serde(default)]
    pub repaired_candidate_index: usize,
    /// Repaired strict-fit minus raw strict-fit.
    #[serde(default)]
    pub strict_pocket_fit_delta: f64,
    /// Raw candidate metrics.
    #[serde(default)]
    pub raw_metrics: RepairCandidateMetricSnapshot,
    /// Repaired candidate metrics.
    #[serde(default)]
    pub repaired_metrics: RepairCandidateMetricSnapshot,
    /// Ordered postprocessing steps applied to the repaired candidate.
    #[serde(default)]
    pub postprocessor_chain: Vec<String>,
    /// Interpretation note for claim review.
    #[serde(default)]
    pub interpretation: String,
}

/// Layer-level raw-vs-repaired deltas for postprocessing review.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepairLayerDeltaSummary {
    /// Raw layer name.
    #[serde(default)]
    pub raw_layer: String,
    /// Repaired layer name.
    #[serde(default)]
    pub repaired_layer: String,
    /// Raw candidate count.
    #[serde(default)]
    pub raw_candidate_count: usize,
    /// Repaired candidate count.
    #[serde(default)]
    pub repaired_candidate_count: usize,
    /// Repaired minus raw valid-fraction delta.
    #[serde(default)]
    pub valid_fraction_delta: f64,
    /// Repaired minus raw pocket-contact delta.
    #[serde(default)]
    pub pocket_contact_fraction_delta: f64,
    /// Repaired minus raw strict-pocket-fit delta.
    #[serde(default)]
    pub strict_pocket_fit_score_delta: f64,
    /// Repaired minus raw centroid-offset delta; negative is better.
    #[serde(default)]
    pub mean_centroid_offset_delta: f64,
    /// Repaired minus raw clash-fraction delta; negative is better.
    #[serde(default)]
    pub clash_fraction_delta: f64,
    /// Repaired minus raw native graph-valid delta.
    #[serde(default)]
    pub native_graph_valid_fraction_delta: f64,
}

/// No-repair ablation baseline for repair/postprocessing claims.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoRepairAblationMetrics {
    /// Whether candidate repair was enabled for this run.
    #[serde(default)]
    pub repair_enabled: bool,
    /// Layer interpreted as the no-repair baseline.
    #[serde(default)]
    pub no_repair_layer: String,
    /// Candidate metrics for the no-repair baseline.
    #[serde(default)]
    pub no_repair_metrics: CandidateLayerMetrics,
    /// Interpretation note for no-repair comparisons.
    #[serde(default)]
    pub interpretation: String,
}

/// Claim-facing repair audit separating raw generation from postprocessing evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairCaseAuditReport {
    /// Schema version for this repair audit block.
    #[serde(default = "default_repair_case_audit_schema_version")]
    pub schema_version: u32,
    /// Split label that produced this audit.
    #[serde(default)]
    pub split_label: String,
    /// Raw-vs-repaired delta summary.
    #[serde(default)]
    pub raw_vs_repaired_delta: RepairLayerDeltaSummary,
    /// No-repair ablation baseline metrics.
    #[serde(default)]
    pub no_repair_ablation: NoRepairAblationMetrics,
    /// Number of retained cases by role.
    #[serde(default)]
    pub case_counts: BTreeMap<String, usize>,
    /// Representative cases where repair improves strict pocket fit.
    #[serde(default)]
    pub repair_helps: Vec<RepairCaseRecord>,
    /// Representative cases where repair degrades strict pocket fit.
    #[serde(default)]
    pub repair_harms: Vec<RepairCaseRecord>,
    /// Representative neutral cases.
    #[serde(default)]
    pub repair_neutral: Vec<RepairCaseRecord>,
    /// Representative raw failures before repair.
    #[serde(default)]
    pub raw_failures: Vec<RepairCaseRecord>,
    /// Optional persisted artifact path relative to the checkpoint directory.
    #[serde(default)]
    pub artifact_name: Option<String>,
    /// Claim boundary for repaired-layer interpretation.
    #[serde(default)]
    pub claim_boundary: String,
}

impl Default for RepairCaseAuditReport {
    fn default() -> Self {
        Self {
            schema_version: default_repair_case_audit_schema_version(),
            split_label: String::new(),
            raw_vs_repaired_delta: RepairLayerDeltaSummary::default(),
            no_repair_ablation: NoRepairAblationMetrics::default(),
            case_counts: BTreeMap::new(),
            repair_helps: Vec::new(),
            repair_harms: Vec::new(),
            repair_neutral: Vec::new(),
            raw_failures: Vec::new(),
            artifact_name: None,
            claim_boundary:
                "repair audit unavailable; do not cite repaired layers as raw generation evidence"
                    .to_string(),
        }
    }
}

fn default_repair_case_audit_schema_version() -> u32 {
    1
}

/// Provenance label for chemistry collaboration diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChemistryMetricProvenance {
    /// Deterministic in-repo heuristic or model diagnostic.
    Heuristic,
    /// Value supplied or directly supported by an external chemistry backend.
    BackendSupported,
    /// Value supplied or directly supported by docking or score-only backends.
    DockingSupported,
    /// Value derived from experimental measurements.
    Experimental,
    /// Explicitly unavailable for the current artifact.
    Unavailable,
}

impl Default for ChemistryMetricProvenance {
    fn default() -> Self {
        Self::Unavailable
    }
}

/// Optional scalar chemistry metric plus evidence provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChemistryCollaborationMetric {
    /// Metric value. `None` means unavailable and must not be interpreted as zero.
    #[serde(default)]
    pub value: Option<f64>,
    /// Claim-safe provenance label for this value.
    #[serde(default)]
    pub provenance: ChemistryMetricProvenance,
    /// Compact status text for reports and validation artifacts.
    #[serde(default)]
    pub status: String,
}

impl ChemistryCollaborationMetric {
    /// Build an available scalar if it is finite; otherwise keep it unavailable.
    pub fn available(value: f64, provenance: ChemistryMetricProvenance, status: &str) -> Self {
        if value.is_finite() {
            Self {
                value: Some(value),
                provenance,
                status: status.to_string(),
            }
        } else {
            Self::unavailable("non-finite metric value")
        }
    }

    /// Build an explicitly unavailable scalar.
    pub fn unavailable(status: &str) -> Self {
        Self {
            value: None,
            provenance: ChemistryMetricProvenance::Unavailable,
            status: status.to_string(),
        }
    }
}

impl Default for ChemistryCollaborationMetric {
    fn default() -> Self {
        Self::unavailable("unavailable")
    }
}

/// Gate usage grouped by chemistry-level directed-path role.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChemistryRoleGateUsage {
    /// Stable chemical role attached to the directed interaction path.
    pub chemical_role: String,
    /// Mean gate usage for this role, when interaction diagnostics are available.
    pub gate_mean: ChemistryCollaborationMetric,
    /// Number of path diagnostics that contributed to this aggregate.
    pub path_count: usize,
}

/// Chemistry-interpretable collaboration diagnostics for a split or claim summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChemistryCollaborationMetrics {
    /// Mean gate usage by chemical role across directed interaction paths.
    #[serde(default)]
    pub gate_usage_by_chemical_role: Vec<ChemistryRoleGateUsage>,
    /// Attention-weighted topology-pocket pharmacophore compatibility coverage.
    #[serde(default)]
    pub pharmacophore_role_coverage: ChemistryCollaborationMetric,
    /// Attention-weighted topology-pocket pharmacophore conflict rate.
    #[serde(default)]
    pub role_conflict_rate: ChemistryCollaborationMetric,
    /// Fraction of evaluated rollouts with severe ligand-pocket clash diagnostics.
    #[serde(default)]
    pub severe_clash_fraction: ChemistryCollaborationMetric,
    /// Fraction of evaluated rollouts with conservative valence violations.
    #[serde(default)]
    pub valence_violation_fraction: ChemistryCollaborationMetric,
    /// Mean topology-implied bond-length guardrail objective.
    #[serde(default)]
    pub bond_length_guardrail_mean: ChemistryCollaborationMetric,
    /// Key-residue contact coverage when residue identities exist.
    #[serde(default)]
    pub key_residue_contact_coverage: ChemistryCollaborationMetric,
}

impl Default for ChemistryCollaborationMetrics {
    fn default() -> Self {
        Self {
            gate_usage_by_chemical_role: Vec::new(),
            pharmacophore_role_coverage: ChemistryCollaborationMetric::unavailable(
                "no pharmacophore role diagnostics",
            ),
            role_conflict_rate: ChemistryCollaborationMetric::unavailable(
                "no pharmacophore role diagnostics",
            ),
            severe_clash_fraction: ChemistryCollaborationMetric::unavailable(
                "no rollout diagnostics",
            ),
            valence_violation_fraction: ChemistryCollaborationMetric::unavailable(
                "no rollout diagnostics",
            ),
            bond_length_guardrail_mean: ChemistryCollaborationMetric::unavailable(
                "no bond-length guardrail evaluation",
            ),
            key_residue_contact_coverage: ChemistryCollaborationMetric::unavailable(
                "residue identities unavailable",
            ),
        }
    }
}

/// Comparison-friendly summary values extracted from a split evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationQualitySummary {
    /// Explicit generation-mode contract for this evaluation summary.
    #[serde(default = "default_generation_mode_label")]
    pub generation_mode: String,
    /// Primary objective label used by this run.
    pub primary_objective: String,
    /// Provenance of the primary objective signal used for claim interpretation.
    #[serde(default = "default_primary_objective_provenance")]
    pub primary_objective_provenance: String,
    /// Claim boundary attached to the selected primary objective.
    #[serde(default = "default_primary_objective_claim_boundary")]
    pub primary_objective_claim_boundary: String,
    /// Optional ablation variant label.
    pub variant_label: Option<String>,
    /// Controlled cross-modality interaction style used by this run.
    pub interaction_mode: String,
    /// Chemistry-valid candidate fraction when available.
    pub candidate_valid_fraction: Option<f64>,
    /// Pocket-contact-style downstream score when available.
    pub pocket_contact_fraction: Option<f64>,
    /// Pocket-compatibility score when available.
    pub pocket_compatibility_fraction: Option<f64>,
    /// Lower-is-better centroid deviation from the target pocket center.
    pub mean_centroid_offset: Option<f64>,
    /// Higher-is-better compact strict pocket-fit score.
    pub strict_pocket_fit_score: Option<f64>,
    /// Higher-is-better unique chemistry fraction from backend or heuristic reporting.
    pub unique_smiles_fraction: Option<f64>,
    /// Held-out unseen-protein fraction for this split.
    pub unseen_protein_fraction: f64,
    /// Topology specialization score derived from adjacency probe confidence.
    pub topology_specialization_score: f64,
    /// Geometry specialization score derived from inverse distance-probe error.
    pub geometry_specialization_score: f64,
    /// Pocket specialization score derived from inverse pocket-feature reconstruction error.
    pub pocket_specialization_score: f64,
    /// Mean active slot fraction carried into the claim surface.
    pub slot_activation_mean: f64,
    /// Mean directed gate activation carried into the claim surface.
    pub gate_activation_mean: f64,
    /// Mean leakage proxy carried into the claim surface.
    pub leakage_proxy_mean: f64,
    /// Chemistry-interpretable collaboration diagnostics with provenance labels.
    #[serde(default)]
    pub chemistry_collaboration: ChemistryCollaborationMetrics,
}

/// Train/validation/test experiment summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnseenPocketExperimentSummary {
    /// Applied experiment configuration.
    pub config: UnseenPocketExperimentConfig,
    /// Machine-readable dataset validation artifact for the run.
    pub dataset_validation: crate::data::DatasetValidationReport,
    /// Coordinate-frame provenance persisted explicitly for audit and replay.
    #[serde(default)]
    pub coordinate_frame: crate::training::CoordinateFrameProvenance,
    /// Split-distribution and leakage audit.
    pub split_report: SplitReport,
    /// Reproducibility and schema metadata for this run.
    pub reproducibility: crate::training::ReproducibilityMetadata,
    /// Training-step history collected during the run.
    pub training_history: Vec<StepMetrics>,
    /// Validation metrics on unseen-pocket split.
    pub validation: EvaluationMetrics,
    /// Test metrics on unseen-pocket split.
    pub test: EvaluationMetrics,
    /// Explicit seen/unseen/test/stress evaluation matrix for claim review.
    #[serde(default)]
    pub evaluation_matrix: EvaluationMatrixReport,
    /// Optional ablation-matrix summary emitted alongside the base run.
    pub ablation_matrix: Option<AblationMatrixSummary>,
    /// Optional performance regression gate report.
    #[serde(default)]
    pub performance_gates: PerformanceGateReport,
}

/// Performance regression gate report persisted with experiment and claim artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceGateReport {
    /// Whether all configured gates passed.
    pub passed: bool,
    /// Human-readable failed gate reasons.
    pub failed_reasons: Vec<String>,
    /// Validation examples/sec observed for this run.
    pub validation_examples_per_second: f64,
    /// Test examples/sec observed for this run.
    pub test_examples_per_second: f64,
    /// Validation evaluation memory delta in MB.
    pub validation_memory_mb: f64,
    /// Test evaluation memory delta in MB.
    pub test_memory_mb: f64,
}

/// One reviewer-facing backend threshold check.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendThresholdCheck {
    /// Observed value, when emitted by the backend surface.
    #[serde(default)]
    pub value: Option<f64>,
    /// Threshold applied during reviewer validation.
    #[serde(default)]
    pub threshold: f64,
    /// Whether the observed value clears the threshold.
    #[serde(default)]
    pub passed: bool,
    /// Comparison direction, such as `min` or `max`.
    #[serde(default)]
    pub direction: String,
}

/// Persisted backend-review policy for a claim surface.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendReviewReport {
    /// Stable policy label for reviewer tooling.
    #[serde(default)]
    pub policy_label: String,
    /// Human-readable backend review status: `pass`, `fail`, or `not_applicable`.
    #[serde(default)]
    pub reviewer_status: String,
    /// Machine-readable review verdict.
    #[serde(default)]
    pub reviewer_passed: bool,
    /// Whether this surface is intended to support claim-bearing backend review.
    #[serde(default)]
    pub claim_bearing_surface: bool,
    /// Whether the persisted artifact currently clears the claim-bearing backend policy.
    #[serde(default)]
    pub claim_bearing_ready: bool,
    /// Explicit requirements for claim-bearing backend interpretation.
    #[serde(default)]
    pub claim_bearing_requirements: Vec<String>,
    /// Concrete pass/fail reasons recorded with the artifact.
    #[serde(default)]
    pub reviewer_reasons: Vec<String>,
    /// Whether chemistry-validity evidence is currently available enough for review.
    #[serde(default)]
    pub chemistry_validity_ready: bool,
    /// Whether the stronger docking backend is currently available.
    #[serde(default)]
    pub docking_backend_available: bool,
    /// Fraction of candidates with complete Vina-ready inputs when the stronger backend is configured.
    #[serde(default)]
    pub docking_input_completeness_fraction: Option<f64>,
    /// Fraction of candidates successfully scored by the stronger docking backend.
    #[serde(default)]
    pub docking_score_coverage_fraction: Option<f64>,
}

/// Persisted ablation matrix summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationMatrixSummary {
    /// Base checkpoint directory that owns the matrix artifact.
    pub artifact_dir: std::path::PathBuf,
    /// One row per executed ablation variant.
    pub variants: Vec<AblationRunSummary>,
}

/// Comparison summary for one ablation variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationRunSummary {
    /// Stable variant label.
    pub variant_label: String,
    /// Validation comparison summary.
    pub validation: GenerationQualitySummary,
    /// Test comparison summary.
    pub test: GenerationQualitySummary,
    /// Explicit raw-native generation quality slice for claim-safe ablation review.
    #[serde(default)]
    pub raw_generation_quality: AblationRawGenerationQuality,
    /// Runtime slice used to compare generation-alignment costs.
    #[serde(default)]
    pub runtime: AblationRuntimeSummary,
    /// Latest optimizer objective-family behavior, when this variant ran training.
    #[serde(default)]
    pub objective_families: Vec<crate::training::ObjectiveFamilyBudgetEntry>,
}

/// Raw model-native quality fields copied into ablation matrix rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AblationRawGenerationQuality {
    /// Split represented by this quality slice.
    #[serde(default)]
    pub split_label: String,
    /// Canonical raw model-native layer name.
    #[serde(default)]
    pub raw_layer: String,
    /// Raw model-native validity fraction.
    #[serde(default)]
    pub raw_valid_fraction: f64,
    /// Raw model-native pocket-contact fraction.
    #[serde(default)]
    pub raw_pocket_contact_fraction: f64,
    /// Raw model-native clash fraction.
    #[serde(default)]
    pub raw_clash_fraction: f64,
    /// Mean final raw rollout displacement.
    #[serde(default)]
    pub raw_mean_displacement: f64,
    /// Validity-conditioned uniqueness for raw rollout candidates.
    #[serde(default)]
    pub raw_validity_conditioned_unique_fraction: f64,
}

/// Runtime fields copied into ablation matrix rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AblationRuntimeSummary {
    /// Elapsed evaluation time in milliseconds.
    #[serde(default)]
    pub evaluation_time_ms: f64,
    /// Evaluated examples per second.
    #[serde(default)]
    pub examples_per_second: f64,
    /// Process memory delta in MB during evaluation.
    #[serde(default)]
    pub memory_usage_mb: f64,
    /// Configured evaluation batch size.
    #[serde(default)]
    pub evaluation_batch_size: usize,
    /// Number of batched forward passes.
    #[serde(default)]
    pub forward_batch_count: usize,
    /// Number of explicit per-example forward calls.
    #[serde(default)]
    pub per_example_forward_count: usize,
    /// Whether evaluation ran without gradient tracking.
    #[serde(default)]
    pub no_grad: bool,
}

/// Compact claim-bearing report persisted for quick scientific review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimReport {
    /// Artifact directory that owns the report.
    pub artifact_dir: std::path::PathBuf,
    /// Primary objective and variant identity for the base run.
    pub run_label: String,
    /// Raw-native model evidence, intentionally placed before processed sections.
    #[serde(default)]
    pub raw_native_evidence: ClaimRawNativeEvidenceSummary,
    /// Additive processed, repaired, selected, or reranked evidence.
    #[serde(default)]
    pub processed_generation_evidence: ClaimProcessedGenerationEvidenceSummary,
    /// Representative raw-vs-repaired cases and no-repair ablation metrics.
    #[serde(default)]
    pub postprocessing_repair_audit: RepairCaseAuditReport,
    /// High-signal validation summary.
    pub validation: GenerationQualitySummary,
    /// High-signal test summary.
    pub test: GenerationQualitySummary,
    /// Direct backend metric snapshots used to support the claim surface.
    pub backend_metrics: RealGenerationMetrics,
    /// Reviewer-facing backend threshold checks derived directly from the persisted backend metrics.
    #[serde(default)]
    pub backend_thresholds: BTreeMap<String, BackendThresholdCheck>,
    /// Reviewer-facing backend policy for claim-bearing interpretation.
    #[serde(default)]
    pub backend_review: BackendReviewReport,
    /// Layered raw/repaired/scored generation metrics for attribution review.
    #[serde(default)]
    pub layered_generation_metrics: LayeredGenerationMetrics,
    /// Model-design raw-versus-processed metrics copied from the claim-bearing test split.
    #[serde(default)]
    pub model_design: ModelDesignEvaluationMetrics,
    /// Claim-surface audit that checks raw model metrics and processed metrics carry layer provenance.
    #[serde(default)]
    pub layer_provenance_audit: ClaimLayerProvenanceAudit,
    /// Reviewer-facing chemistry novelty and diversity summary.
    #[serde(default)]
    pub chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary,
    /// Reviewer-facing chemistry collaboration diagnostics with provenance labels.
    #[serde(default)]
    pub chemistry_collaboration: ChemistryCollaborationMetrics,
    /// Reviewer-facing experiment context, including whether the surface is real-backend-backed.
    #[serde(default)]
    pub claim_context: ClaimContext,
    /// Backend environment fingerprint attached to this claim surface.
    #[serde(default)]
    pub backend_environment: Option<BackendEnvironmentReport>,
    /// Main ablation deltas relative to the base test summary.
    pub ablation_deltas: Vec<ClaimDeltaSummary>,
    /// Lightweight reranker evidence before considering adversarial training.
    #[serde(default)]
    pub reranker_report: RerankerReport,
    /// Slot-stability diagnostics copied from the claim-bearing test split.
    #[serde(default)]
    pub slot_stability: SlotStabilityMetrics,
    /// Leakage calibration recommendation derived from claim-bearing ablations.
    #[serde(default)]
    pub leakage_calibration: LeakageCalibrationReport,
    /// Coarse performance gate report for claim review.
    #[serde(default)]
    pub performance_gates: PerformanceGateReport,
    /// Stronger baseline/control rows collected for claim review.
    #[serde(default)]
    pub baseline_comparisons: Vec<BaselineComparisonRow>,
    /// Method-aware comparison artifact for the active claim surface.
    #[serde(default)]
    pub method_comparison: MethodComparisonSummary,
    /// Train/eval metric attribution audit copied from the claim-bearing test split.
    #[serde(default)]
    pub train_eval_alignment: TrainEvalAlignmentReport,
}

/// Claim-surface raw-versus-processed layer provenance audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimLayerProvenanceAudit {
    /// Layer used for raw model-design fields.
    #[serde(default)]
    pub raw_model_layer: String,
    /// Layer used for processed model-design fields.
    #[serde(default)]
    pub processed_layer: String,
    /// Whether the raw model-design layer is contractually model-native raw output.
    #[serde(default)]
    pub raw_layer_model_native: bool,
    /// Whether the processed layer has an explicit generation path contract.
    #[serde(default)]
    pub processed_layer_has_contract: bool,
    /// Canonical processed layer name when a contract is present.
    #[serde(default)]
    pub processed_canonical_layer: String,
    /// Generation path class for the processed layer.
    #[serde(default)]
    pub processed_generation_path_class: String,
    /// Ordered postprocessing chain attached to the processed layer.
    #[serde(default)]
    pub processed_postprocessor_chain: Vec<String>,
    /// Whether processed metrics would be cited as raw model-native evidence.
    #[serde(default)]
    pub processed_metrics_cited_as_raw: bool,
    /// Whether this claim surface passes the raw-versus-processed provenance audit.
    #[serde(default)]
    pub claim_safe: bool,
    /// Reviewer-facing audit decision.
    #[serde(default)]
    pub decision: String,
}

impl Default for ClaimLayerProvenanceAudit {
    fn default() -> Self {
        Self {
            raw_model_layer: "unavailable".to_string(),
            processed_layer: "unavailable".to_string(),
            raw_layer_model_native: false,
            processed_layer_has_contract: false,
            processed_canonical_layer: "unavailable".to_string(),
            processed_generation_path_class: "unavailable".to_string(),
            processed_postprocessor_chain: Vec::new(),
            processed_metrics_cited_as_raw: false,
            claim_safe: false,
            decision: "layer provenance audit unavailable".to_string(),
        }
    }
}

/// Compact row for heuristic or ablation baseline/control comparisons.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BaselineComparisonRow {
    /// Stable baseline label.
    pub label: String,
    /// Artifact source for this row, such as `generation_layer` or `ablation_matrix`.
    pub source: String,
    /// Candidate-layer metrics when the baseline is a generation-layer proxy.
    #[serde(default)]
    pub candidate_layer: Option<CandidateLayerMetrics>,
    /// Test-summary metrics when the baseline comes from an ablation/control run.
    #[serde(default)]
    pub test_summary: Option<GenerationQualitySummary>,
    /// Interpretation note for claim review.
    pub interpretation: String,
}

/// Compact comparison of baseline deterministic selection and proxy reranking.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RerankerReport {
    /// Metrics before lightweight chemistry/pocket proxy reranking.
    pub baseline: CandidateLayerMetrics,
    /// Metrics after selecting calibrated reranked candidates.
    pub reranked: CandidateLayerMetrics,
    /// Metrics after the previous deterministic proxy selector.
    #[serde(default)]
    pub deterministic_proxy: CandidateLayerMetrics,
    /// Bounded linear calibration used by the active reranker.
    #[serde(default)]
    pub calibration: RerankerCalibrationReport,
    /// Raw-flow strict pocket-fit proxy before repair or reranking.
    #[serde(default)]
    pub raw_strict_pocket_fit: Option<f64>,
    /// Raw-flow docking score when layer-attributed backend scoring is available.
    #[serde(default)]
    pub raw_docking_score: Option<f64>,
    /// Raw-flow QED when layer-attributed chemistry scoring is available.
    #[serde(default)]
    pub raw_qed: Option<f64>,
    /// Raw-flow synthetic accessibility score when layer-attributed chemistry scoring is available.
    #[serde(default)]
    pub raw_sa: Option<f64>,
    /// Positive quality lift from raw_flow to repaired, bounded to [0, 1].
    #[serde(default)]
    pub repair_dependency_score: f64,
    /// Metric-wise gains from constrained_flow to reranked under the same candidate pool.
    #[serde(default)]
    pub reranker_gain: BTreeMap<String, f64>,
    /// Native quality attributed only to raw_flow.
    #[serde(default)]
    pub flow_native_quality: Option<f64>,
    /// Interpretation note identifying whether gains come from repair, selection, or backend scoring.
    #[serde(default)]
    pub layer_attribution_note: String,
    /// Decision text for whether reranking remains sufficient.
    pub decision: String,
}

/// Persisted bounded-reranker calibration metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RerankerCalibrationReport {
    /// Calibration method name.
    pub method: String,
    /// Feature coefficients after non-negative normalization.
    pub coefficients: BTreeMap<String, f64>,
    /// Mean backend-compatible target used for the split-local calibration.
    pub target_mean: f64,
    /// Number of candidates used to fit coefficients.
    pub fitted_candidate_count: usize,
}

/// Reviewer-facing chemistry novelty/diversity summary derived from persisted candidate layers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChemistryNoveltyDiversitySummary {
    /// Layer chosen for claim-bearing chemistry interpretation.
    pub review_layer: String,
    /// Explicit uniqueness metric carried into review summaries.
    pub unique_smiles_fraction: Option<f64>,
    /// Diversity of atom-type sequences within the chosen layer.
    pub atom_type_sequence_diversity: f64,
    /// Diversity of inferred bond topologies within the chosen layer.
    pub bond_topology_diversity: f64,
    /// Diversity of coarse coordinate shapes within the chosen layer.
    pub coordinate_shape_diversity: f64,
    /// Novelty of atom-type sequences relative to training references.
    pub novel_atom_type_sequence_fraction: f64,
    /// Novelty of bond-topology signatures relative to training references.
    pub novel_bond_topology_fraction: f64,
    /// Novelty of coarse coordinate-shape signatures relative to training references.
    pub novel_coordinate_shape_fraction: f64,
    /// Human-readable reviewer note for interpretation.
    pub interpretation: String,
    /// Stronger reviewer-facing chemistry evidence beyond signature-only summaries.
    #[serde(default)]
    pub benchmark_evidence: ChemistryBenchmarkEvidence,
}

/// Reviewer-facing claim context for a persisted surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimContext {
    /// Stable configured surface label when present.
    #[serde(default)]
    pub surface_label: Option<String>,
    /// Whether active chemistry and pocket backends were enabled on this surface.
    #[serde(default)]
    pub real_backend_backed: bool,
    /// Human-readable note describing how to interpret the surface.
    #[serde(default)]
    pub evidence_mode: String,
    /// Explicit generation-mode contract for claim-boundary checks.
    #[serde(default = "default_generation_mode_label")]
    pub generation_mode: String,
    /// Whether de novo generation wording is allowed for this surface.
    #[serde(default)]
    pub de_novo_claim_allowed: bool,
    /// Configured target matching/alignment policy for molecular de novo flow.
    #[serde(default = "default_target_alignment_policy_label")]
    pub target_alignment_policy: String,
    /// Whether target matching is non-index and claim-safe for de novo full-flow wording.
    #[serde(default)]
    pub target_matching_claim_safe: bool,
    /// Whether full molecular flow wording is allowed by branch and matching gates.
    #[serde(default)]
    pub full_molecular_flow_claim_allowed: bool,
    /// Claim-boundary note for full molecular flow wording.
    #[serde(default)]
    pub full_molecular_flow_claim_boundary: String,
    /// Configured cross-modality interaction mode for this surface.
    #[serde(default = "default_interaction_mode_label")]
    pub interaction_mode: String,
    /// Whether this surface uses the direct-fusion negative-control ablation.
    #[serde(default)]
    pub direct_fusion_negative_control: bool,
    /// Whether preferred controlled-interaction architecture wording is allowed.
    #[serde(default = "default_preferred_architecture_claim_allowed")]
    pub preferred_architecture_claim_allowed: bool,
    /// Compact boundary note for interaction architecture claims.
    #[serde(default)]
    pub interaction_claim_boundary: String,
}

impl Default for ClaimContext {
    fn default() -> Self {
        Self {
            surface_label: None,
            real_backend_backed: false,
            evidence_mode: String::new(),
            generation_mode: default_generation_mode_label(),
            de_novo_claim_allowed: false,
            target_alignment_policy: "pad_with_mask".to_string(),
            target_matching_claim_safe: false,
            full_molecular_flow_claim_allowed: false,
            full_molecular_flow_claim_boundary: String::new(),
            interaction_mode: default_interaction_mode_label(),
            direct_fusion_negative_control: false,
            preferred_architecture_claim_allowed: default_preferred_architecture_claim_allowed(),
            interaction_claim_boundary: String::new(),
        }
    }
}

fn default_generation_mode_label() -> String {
    crate::config::GenerationModeConfig::TargetLigandDenoising
        .as_str()
        .to_string()
}

fn default_candidate_layer_label() -> String {
    "unavailable".to_string()
}

fn default_pocket_interaction_provenance() -> String {
    "unavailable".to_string()
}

fn default_interaction_mode_label() -> String {
    "transformer".to_string()
}

fn default_target_alignment_policy_label() -> String {
    "pad_with_mask".to_string()
}

fn default_preferred_architecture_claim_allowed() -> bool {
    true
}

fn default_primary_objective_provenance() -> String {
    "legacy_or_unknown_objective_provenance".to_string()
}

fn default_primary_objective_claim_boundary() -> String {
    "interpret_with_persisted_metric_schema_and_component_provenance".to_string()
}

/// Compact chemistry evidence that goes beyond uniqueness and signature novelty alone.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChemistryBenchmarkEvidence {
    /// Whether this summary is derived from active external chemistry tooling.
    #[serde(default)]
    pub backend_backed: bool,
    /// Fraction of candidates sanitized by the chemistry backend.
    #[serde(default)]
    pub sanitized_fraction: Option<f64>,
    /// Fraction of backend-unique molecules on the reviewed layer.
    #[serde(default)]
    pub unique_smiles_fraction: Option<f64>,
    /// Candidate count on the chemistry review layer used to build the summary.
    #[serde(default)]
    pub review_candidate_count: usize,
    /// Aggregate chemistry quality score from backend-backed validity evidence.
    #[serde(default)]
    pub validity_quality_score: Option<f64>,
    /// Aggregate novelty/diversity score from held-out review-layer candidates.
    #[serde(default)]
    pub novelty_diversity_score: f64,
    /// Compact tier label separating proxy-only chemistry from benchmark-style evidence.
    #[serde(default)]
    pub evidence_tier: String,
    /// Whether the surface clears the stronger reviewer benchmark gate for chemistry evidence.
    #[serde(default)]
    pub stronger_reviewer_benchmark: bool,
    /// Whether the surface also clears the explicit external benchmark-dataset chemistry tier.
    #[serde(default)]
    pub external_benchmark_backed: bool,
    /// Named external benchmark dataset or baseline family supporting the stronger tier.
    #[serde(default)]
    pub external_benchmark_dataset: Option<String>,
    /// Minimum review-layer candidate count required for the stronger reviewer chemistry tier.
    #[serde(default)]
    pub stronger_review_candidate_threshold: usize,
    /// Explicit backend quality metrics required by the stronger reviewer chemistry tier.
    #[serde(default)]
    pub stronger_required_backend_metrics: Vec<String>,
    /// Fraction of stronger-tier chemistry checks satisfied by the current surface.
    #[serde(default)]
    pub stronger_benchmark_support_score: Option<f64>,
    /// Fraction of explicit external-benchmark checks satisfied by the current surface.
    #[serde(default)]
    pub external_benchmark_support_score: Option<f64>,
    /// Human-readable note describing why the stronger tier was or was not reached.
    #[serde(default)]
    pub stronger_benchmark_note: String,
    /// Explicit dataset and baseline checks required by the external benchmark tier.
    #[serde(default)]
    pub external_required_checks: Vec<String>,
    /// Human-readable note describing why the external benchmark tier was or was not reached.
    #[serde(default)]
    pub external_benchmark_note: String,
    /// Explicit metrics included in the benchmark-style chemistry summary.
    #[serde(default)]
    pub benchmark_components: Vec<String>,
    /// Benchmark-style reviewer note explaining the evidence strength.
    #[serde(default)]
    pub interpretation: String,
}

/// Persisted backend command environment for reviewer-facing reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendEnvironmentReport {
    /// Stable hash of the backend command configuration.
    pub config_fingerprint: String,
    /// Whether this surface is intended to be real-backend-backed.
    pub real_backend_backed: bool,
    /// Minimal prerequisites required to revalidate the backend surface locally.
    #[serde(default)]
    pub prerequisites: Vec<String>,
    /// Chemistry backend command and runtime status.
    pub chemistry_backend: BackendCommandReport,
    /// Docking backend command and runtime status.
    pub docking_backend: BackendCommandReport,
    /// Pocket-compatibility backend command and runtime status.
    pub pocket_backend: BackendCommandReport,
}

/// Persisted command identity and backend runtime status.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendCommandReport {
    /// Logical reviewer-facing backend slot name.
    pub logical_name: String,
    /// Whether this backend was enabled in the experiment config.
    pub enabled: bool,
    /// Configured executable path or command.
    #[serde(default)]
    pub executable: Option<String>,
    /// Static backend command arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Stable hash of the concrete command identity.
    pub command_fingerprint: String,
    /// Whether the runtime metrics indicate the backend was available enough to score examples.
    pub runtime_available: bool,
    /// Persisted backend adapter name when present.
    #[serde(default)]
    pub backend_name: Option<String>,
    /// Human-readable backend status copied from the evaluation report.
    pub status: String,
    /// Schema version emitted by the backend when available.
    #[serde(default)]
    pub schema_version: Option<f64>,
    /// Number of examples scored by the backend when reported.
    #[serde(default)]
    pub backend_examples_scored: Option<f64>,
}

/// Leakage-weight calibration summary for preserving useful cross-modality dependence.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LeakageCalibrationReport {
    /// Recommended default leakage weight for claim-bearing configs.
    pub recommended_delta_leak: f64,
    /// Number of claim-delta rows considered.
    pub evaluated_variants: usize,
    /// Preferred upper bound for reviewer-facing leakage proxy.
    #[serde(default = "default_preferred_leakage_proxy_threshold")]
    pub preferred_max_leakage_proxy_mean: f64,
    /// Absolute upper bound beyond which reviewer wording should fail.
    #[serde(default = "default_hard_leakage_proxy_threshold")]
    pub hard_max_leakage_proxy_mean: f64,
    /// Maximum tolerated leakage regression relative to the base run.
    #[serde(default = "default_max_leakage_regression_threshold")]
    pub max_leakage_proxy_regression: f64,
    /// Capacity metadata for semantic and explicit leakage probe heads.
    #[serde(default)]
    pub probe_capacity: ProbeCapacityReport,
    /// Test-split probe comparisons against trivial target-only baselines.
    #[serde(default)]
    pub probe_baseline_comparisons: Vec<ProbeBaselineMetric>,
    /// Held-out leakage probe calibration on frozen representations, when run.
    #[serde(default)]
    pub frozen_probe_calibration: FrozenLeakageProbeCalibrationReport,
    /// Role-separated leakage evidence contract for claim review.
    #[serde(default)]
    pub leakage_roles: crate::losses::LeakageEvidenceRoleReport,
    /// Optional artifact path for a capacity sweep over frozen leakage probes.
    #[serde(default)]
    pub capacity_sweep_artifact: Option<String>,
    /// Whether current probe evidence is strong enough to support no-leakage wording.
    #[serde(default)]
    pub no_leakage_claim_supported: bool,
    /// Reviewer-facing status: `pass`, `caution`, or `fail`.
    #[serde(default)]
    pub reviewer_status: String,
    /// Machine-readable review verdict.
    #[serde(default)]
    pub reviewer_passed: bool,
    /// Concrete reasons behind the reviewer verdict.
    #[serde(default)]
    pub reviewer_reasons: Vec<String>,
    /// Human-readable calibration decision.
    pub decision: String,
}

/// Held-out calibration report for separately trained leakage probes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenLeakageProbeCalibrationReport {
    /// Schema version for frozen leakage probe calibration artifacts.
    #[serde(default = "default_frozen_leakage_probe_schema_version")]
    pub schema_version: u32,
    /// Overall calibration status: `not_run`, `ok`, or `insufficient_data`.
    #[serde(default)]
    pub calibration_status: String,
    /// Evaluation split used for this held-out frozen-probe audit.
    #[serde(default)]
    pub split_name: String,
    /// Training-time signal this calibration should be compared against.
    #[serde(default)]
    pub training_time_signal: String,
    /// Frozen representation source used by the calibration probes.
    #[serde(default)]
    pub representation_source: String,
    /// Whether optimizer-facing leakage penalties are explicitly separated from this report.
    #[serde(default)]
    pub optimizer_penalty_separated: bool,
    /// Trivial baseline persisted for every route/sweep row.
    #[serde(default)]
    pub trivial_baseline: String,
    /// Route-level held-out performance summaries.
    #[serde(default)]
    pub routes: Vec<FrozenLeakageProbeRouteReport>,
    /// Full capacity/regularization sweep rows.
    #[serde(default)]
    pub capacity_sweep: Vec<FrozenLeakageProbeSweepRow>,
    /// Claim boundary for interpreting the report.
    #[serde(default)]
    pub claim_boundary: String,
}

impl Default for FrozenLeakageProbeCalibrationReport {
    fn default() -> Self {
        Self {
            schema_version: default_frozen_leakage_probe_schema_version(),
            calibration_status: "not_run".to_string(),
            split_name: "not_run".to_string(),
            training_time_signal: "training-time leakage penalty/proxy only".to_string(),
            representation_source: "not_run".to_string(),
            optimizer_penalty_separated: false,
            trivial_baseline: "heldout_target_mean_mse".to_string(),
            routes: Vec::new(),
            capacity_sweep: Vec::new(),
            claim_boundary: "held-out frozen-probe predictability is diagnostic evidence, not proof of absence of leakage".to_string(),
        }
    }
}

/// Best held-out calibration row for one off-modality route.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FrozenLeakageProbeRouteReport {
    /// Stable route label, such as `topology_to_geometry`.
    pub route: String,
    /// Evaluation split used for this route audit.
    #[serde(default)]
    pub split_name: String,
    /// Source representation branch.
    pub source_modality: String,
    /// Target family predicted from the frozen source representation.
    pub target: String,
    /// Number of training examples used for the separate probe.
    pub train_count: usize,
    /// Number of held-out examples used for evaluation.
    pub heldout_count: usize,
    /// Best hidden/capacity setting in the sweep.
    pub best_capacity: usize,
    /// Best ridge regularization setting in the sweep.
    pub best_regularization: f64,
    /// Held-out mean-squared error for the best row.
    pub heldout_mse: f64,
    /// Target-mean baseline MSE on the same held-out examples.
    pub baseline_mse: f64,
    /// Fractional MSE improvement over the target-mean baseline.
    pub improvement_over_baseline: f64,
    /// Whether the held-out probe beats the baseline by a positive margin.
    pub predicts_off_modality_target: bool,
}

/// One capacity/regularization row from the frozen leakage probe sweep.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FrozenLeakageProbeSweepRow {
    /// Stable route label.
    pub route: String,
    /// Evaluation split used for this sweep row.
    #[serde(default)]
    pub split_name: String,
    /// Probe capacity, interpreted as number of frozen feature dimensions used.
    pub capacity: usize,
    /// Ridge regularization value used by the separate probe fit.
    pub regularization: f64,
    /// Number of training examples.
    pub train_count: usize,
    /// Number of held-out examples.
    pub heldout_count: usize,
    /// Held-out mean-squared error.
    pub heldout_mse: f64,
    /// Held-out target-mean baseline MSE.
    pub baseline_mse: f64,
    /// Fractional improvement over the target-mean baseline.
    pub improvement_over_baseline: f64,
}

fn default_frozen_leakage_probe_schema_version() -> u32 {
    1
}

/// Probe-head capacity metadata persisted with leakage calibration artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProbeCapacityReport {
    /// Configured semantic probe hidden width.
    pub hidden_dim: i64,
    /// Configured number of hidden layers before each probe output head.
    pub hidden_layers: usize,
    /// Human-readable probe family label.
    pub architecture: String,
    /// Whether this capacity is only the linear diagnostic baseline.
    pub linear_baseline_only: bool,
    /// Interpretation note used to bound leakage claims.
    pub interpretation: String,
}

fn default_preferred_leakage_proxy_threshold() -> f64 {
    0.08
}

fn default_hard_leakage_proxy_threshold() -> f64 {
    0.12
}

fn default_max_leakage_regression_threshold() -> f64 {
    0.03
}

/// Compact delta row between the base run and one ablation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimDeltaSummary {
    /// Stable ablation label.
    pub variant_label: String,
    /// Delta in candidate-valid fraction when present.
    pub candidate_valid_fraction_delta: Option<f64>,
    /// Delta in pocket-contact fraction when present.
    pub pocket_contact_fraction_delta: Option<f64>,
    /// Delta in pocket-compatibility fraction when present.
    pub pocket_compatibility_fraction_delta: Option<f64>,
    /// Delta in mean centroid offset when present. Positive means worse than baseline.
    pub mean_centroid_offset_delta: Option<f64>,
    /// Delta in strict pocket-fit score when present.
    pub strict_pocket_fit_score_delta: Option<f64>,
    /// Delta in unique smiles fraction when present.
    pub unique_smiles_fraction_delta: Option<f64>,
    /// Delta in topology specialization score.
    pub topology_specialization_score_delta: f64,
    /// Delta in geometry specialization score.
    pub geometry_specialization_score_delta: f64,
    /// Delta in pocket specialization score.
    pub pocket_specialization_score_delta: f64,
    /// Delta in mean slot activation.
    pub slot_activation_mean_delta: f64,
    /// Delta in mean gate activation.
    pub gate_activation_mean_delta: f64,
    /// Delta in mean leakage proxy.
    pub leakage_proxy_mean_delta: f64,
}

/// Compact review artifact comparing lightweight and Transformer interaction modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionModeReview {
    /// Root directory containing the reviewed surfaces.
    pub review_root: std::path::PathBuf,
    /// One row per reviewed artifact surface.
    pub surfaces: Vec<InteractionSurfaceReview>,
    /// Aggregate test-set win/loss/tie counts across all reviewed surfaces.
    pub aggregate_test_tally: InteractionWinLossTally,
    /// Roadmap recommendation derived from the reviewed surfaces.
    pub recommendation: String,
}

/// One surface-level review comparing lightweight and Transformer interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionSurfaceReview {
    /// Human-readable surface label.
    pub surface_label: String,
    /// Artifact directory that owns the reviewed summaries.
    pub artifact_dir: std::path::PathBuf,
    /// Validation comparison review.
    pub validation: InteractionSplitReview,
    /// Test comparison review.
    pub test: InteractionSplitReview,
}

/// Per-split compact review grouped by metric family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionSplitReview {
    /// Lightweight interaction summary for this split.
    pub lightweight: GenerationQualitySummary,
    /// Transformer interaction summary for this split.
    pub transformer: GenerationQualitySummary,
    /// Aggregate tally over all reviewed metrics for this split.
    pub tally: InteractionWinLossTally,
    /// Geometric-fit and regression-gate metrics.
    pub geometric_fit: Vec<InteractionMetricVerdict>,
    /// Specialization and leakage metrics.
    pub specialization: Vec<InteractionMetricVerdict>,
    /// Slot and gate usage metrics.
    pub utilization: Vec<InteractionMetricVerdict>,
}

/// Compact win/loss/tie tally.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InteractionWinLossTally {
    /// Number of metrics won by lightweight interaction.
    pub lightweight_wins: usize,
    /// Number of metrics won by Transformer interaction.
    pub transformer_wins: usize,
    /// Number of tied metrics.
    pub ties: usize,
}

/// One reviewed metric verdict between lightweight and Transformer interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionMetricVerdict {
    /// Stable metric name.
    pub metric: String,
    /// Whether larger or smaller values are preferred.
    pub direction: MetricDirection,
    /// Lightweight value when available.
    pub lightweight: Option<f64>,
    /// Transformer value when available.
    pub transformer: Option<f64>,
    /// Winner for this metric.
    pub winner: MetricWinner,
    /// Absolute delta computed in the preferred direction.
    pub preferred_delta: f64,
}

/// Optimization direction for a reviewed metric.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// Higher values are better.
    HigherIsBetter,
    /// Lower values are better.
    LowerIsBetter,
}

/// Winner label for a reviewed metric.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetricWinner {
    /// Lightweight interaction wins.
    Lightweight,
    /// Transformer interaction wins.
    Transformer,
    /// Values are effectively tied.
    Tie,
}

/// Error summary for one affinity measurement family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMetrics {
    /// Measurement family label, such as `Kd`, `Ki`, `IC50`, or `dG`.
    pub measurement_type: String,
    /// Number of labeled examples for this measurement family.
    pub count: usize,
    /// MAE on this measurement family.
    pub mae: f64,
    /// RMSE on this measurement family.
    pub rmse: f64,
}
