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
    /// Average ligand atom count observed in the evaluation split.
    #[serde(default)]
    pub average_ligand_atoms: f64,
    /// Average pocket atom count observed in the evaluation split.
    #[serde(default)]
    pub average_pocket_atoms: f64,
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
    /// Reserved section for chemistry/docking/pocket compatibility backends.
    pub real_generation_metrics: RealGenerationMetrics,
    /// Candidate-quality metrics split by generation and postprocessing layer.
    pub layered_generation_metrics: LayeredGenerationMetrics,
    /// Method-aware comparison summary for this split.
    #[serde(default)]
    pub method_comparison: MethodComparisonSummary,
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
}

/// Method-aware comparison summary persisted alongside layered candidate metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MethodComparisonSummary {
    /// Active method metadata for the current surface.
    #[serde(default)]
    pub active_method: Option<PocketGenerationMethodMetadata>,
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
    /// Number of candidate records in the layer.
    pub candidate_count: usize,
    /// Fraction with nonempty finite atom and coordinate payloads.
    pub valid_fraction: f64,
    /// Fraction with at least one atom near the pocket envelope.
    pub pocket_contact_fraction: f64,
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
    /// Fraction of unique atom-type sequences and coarse coordinate buckets.
    pub uniqueness_proxy_fraction: f64,
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
}

/// Comparison-friendly summary values extracted from a split evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationQualitySummary {
    /// Primary objective label used by this run.
    pub primary_objective: String,
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
}

/// Train/validation/test experiment summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnseenPocketExperimentSummary {
    /// Applied experiment configuration.
    pub config: UnseenPocketExperimentConfig,
    /// Machine-readable dataset validation artifact for the run.
    pub dataset_validation: crate::data::DatasetValidationReport,
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
}

/// Compact claim-bearing report persisted for quick scientific review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimReport {
    /// Artifact directory that owns the report.
    pub artifact_dir: std::path::PathBuf,
    /// Primary objective and variant identity for the base run.
    pub run_label: String,
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
    /// Reviewer-facing chemistry novelty and diversity summary.
    #[serde(default)]
    pub chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary,
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
