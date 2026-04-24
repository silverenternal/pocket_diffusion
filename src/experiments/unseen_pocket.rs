//! Unseen-pocket experiment loop, ablations, and evaluation summaries.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tch::nn;

use crate::{
    config::{
        AutomatedSearchConfig, AutomatedSearchHardGateConfig, AutomatedSearchScoreWeightConfig,
        AutomatedSearchSpaceConfig, AutomatedSearchStrategy, CrossAttentionMode,
        ExternalBackendCommandConfig, ResearchConfig,
    },
    data::InMemoryDataset,
    models::{
        generate_layered_candidates_with_options, report_to_metrics, CandidateGenerationLayers,
        ChemistryValidityEvaluator, CommandChemistryValidityEvaluator, CommandDockingEvaluator,
        CommandPocketCompatibilityEvaluator, DockingEvaluator, GeneratedCandidateRecord,
        HeuristicChemistryValidityEvaluator, HeuristicDockingEvaluator,
        HeuristicPocketCompatibilityEvaluator, Phase1ResearchSystem, PocketCompatibilityEvaluator,
        ResearchForward,
    },
    runtime::parse_runtime_device,
    training::{
        reproducibility_metadata, stable_json_hash, ResearchTrainer, RunArtifactBundle,
        RunArtifactPaths, RunKind, SplitReport, StepMetrics,
    },
};

/// Toggles used to disable parts of the model or objective for ablation studies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    /// Disable slot decomposition metrics and slot loss reporting.
    pub disable_slots: bool,
    /// Disable gated cross-modal interaction metrics.
    pub disable_cross_attention: bool,
    /// Disable semantic probe metrics.
    pub disable_probes: bool,
    /// Disable leakage reporting.
    pub disable_leakage: bool,
    /// Disable ligand-pocket geometry bias inside controlled cross-attention.
    #[serde(default)]
    pub disable_geometry_interaction_bias: bool,
    /// Disable decoder-time pocket guidance during rollout.
    #[serde(default)]
    pub disable_rollout_pocket_guidance: bool,
    /// Disable candidate geometry repair before bond inference and scoring.
    #[serde(default)]
    pub disable_candidate_repair: bool,
    /// Override the primary objective used by this experiment variant.
    pub primary_objective_override: Option<crate::config::PrimaryObjectiveConfig>,
    /// Override the cross-modal interaction block style for controlled interaction ablations.
    pub interaction_mode_override: Option<CrossAttentionMode>,
    /// Human-readable variant label for persisted comparison artifacts.
    pub variant_label: Option<String>,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            disable_slots: false,
            disable_cross_attention: false,
            disable_probes: false,
            disable_leakage: false,
            disable_geometry_interaction_bias: false,
            disable_rollout_pocket_guidance: false,
            disable_candidate_repair: false,
            primary_objective_override: None,
            interaction_mode_override: None,
            variant_label: None,
        }
    }
}

/// Executable backend configuration for chemistry, docking, and pocket scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalEvaluationConfig {
    /// External chemistry-validity command adapter.
    #[serde(default)]
    pub chemistry_backend: ExternalBackendCommandConfig,
    /// External docking or pocket-affinity command adapter.
    #[serde(default)]
    pub docking_backend: ExternalBackendCommandConfig,
    /// External pocket-compatibility command adapter.
    #[serde(default)]
    pub pocket_backend: ExternalBackendCommandConfig,
    /// Persist compact raw/repaired/scored candidate artifacts for regression review.
    #[serde(default = "default_persist_generation_artifacts")]
    pub persist_generation_artifacts: bool,
    /// Maximum number of examples represented in per-split candidate artifacts.
    #[serde(default = "default_generation_artifact_example_limit")]
    pub generation_artifact_example_limit: usize,
    /// Maximum number of candidate records retained per layer in artifacts.
    #[serde(default = "default_generation_artifact_candidate_limit")]
    pub generation_artifact_candidate_limit: usize,
}

impl Default for ExternalEvaluationConfig {
    fn default() -> Self {
        Self {
            chemistry_backend: ExternalBackendCommandConfig::default(),
            docking_backend: ExternalBackendCommandConfig::default(),
            pocket_backend: ExternalBackendCommandConfig::default(),
            persist_generation_artifacts: default_persist_generation_artifacts(),
            generation_artifact_example_limit: default_generation_artifact_example_limit(),
            generation_artifact_candidate_limit: default_generation_artifact_candidate_limit(),
        }
    }
}

impl ExternalEvaluationConfig {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.chemistry_backend
            .validate("external_evaluation.chemistry_backend")?;
        self.docking_backend
            .validate("external_evaluation.docking_backend")?;
        self.pocket_backend
            .validate("external_evaluation.pocket_backend")?;
        if self.persist_generation_artifacts
            && (self.generation_artifact_example_limit == 0
                || self.generation_artifact_candidate_limit == 0)
        {
            return Err(
                "external_evaluation generation artifact limits must be positive when enabled"
                    .into(),
            );
        }
        Ok(())
    }
}

fn default_persist_generation_artifacts() -> bool {
    true
}

fn default_generation_artifact_example_limit() -> usize {
    8
}

fn default_generation_artifact_candidate_limit() -> usize {
    32
}

/// Lightweight config-driven ablation matrix over major architectural controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationMatrixConfig {
    /// Whether to execute the matrix after the base experiment.
    pub enabled: bool,
    /// Whether to include a surrogate reconstruction baseline.
    pub include_surrogate_objective: bool,
    /// Whether to include a conditioned-denoising baseline.
    pub include_conditioned_denoising: bool,
    /// Whether to include a variant with slots disabled in reporting.
    pub include_disable_slots: bool,
    /// Whether to include a variant with cross-attention disabled in reporting.
    pub include_disable_cross_attention: bool,
    /// Whether to include a variant with probes disabled in reporting.
    pub include_disable_probes: bool,
    /// Whether to include a lightweight controlled-interaction baseline.
    #[serde(default = "default_include_lightweight_interaction")]
    pub include_lightweight_interaction: bool,
    /// Whether to include a Transformer-style controlled-interaction variant.
    #[serde(default = "default_include_transformer_interaction")]
    pub include_transformer_interaction: bool,
    /// Whether to include the no geometry-biased controlled-attention variant.
    #[serde(default = "default_include_geometry_bias_ablation")]
    pub include_disable_geometry_interaction_bias: bool,
    /// Whether to include the rollout-only pocket-guidance disabled variant.
    #[serde(default = "default_include_rollout_guidance_ablation")]
    pub include_disable_rollout_pocket_guidance: bool,
    /// Whether to include the no candidate-repair variant.
    #[serde(default = "default_include_candidate_repair_ablation")]
    pub include_disable_candidate_repair: bool,
}

impl Default for AblationMatrixConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            include_surrogate_objective: true,
            include_conditioned_denoising: true,
            include_disable_slots: true,
            include_disable_cross_attention: true,
            include_disable_probes: false,
            include_lightweight_interaction: true,
            include_transformer_interaction: true,
            include_disable_geometry_interaction_bias: default_include_geometry_bias_ablation(),
            include_disable_rollout_pocket_guidance: default_include_rollout_guidance_ablation(),
            include_disable_candidate_repair: default_include_candidate_repair_ablation(),
        }
    }
}

fn default_include_geometry_bias_ablation() -> bool {
    true
}

fn default_include_rollout_guidance_ablation() -> bool {
    true
}

fn default_include_candidate_repair_ablation() -> bool {
    true
}

fn default_include_lightweight_interaction() -> bool {
    true
}

fn default_include_transformer_interaction() -> bool {
    true
}

/// High-level experiment configuration for unseen-pocket generalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnseenPocketExperimentConfig {
    /// Core model/training/runtime configuration.
    pub research: ResearchConfig,
    /// Stable label used by shared cross-surface reporting.
    pub surface_label: Option<String>,
    /// Optional reviewer benchmark metadata used for claim-bearing evidence tiers.
    #[serde(default)]
    pub reviewer_benchmark: ReviewerBenchmarkConfig,
    /// Ablation toggles.
    pub ablation: AblationConfig,
    /// Optional external chemistry/docking/pocket backends.
    #[serde(default)]
    pub external_evaluation: ExternalEvaluationConfig,
    /// Optional ablation matrix executed from the same config.
    #[serde(default)]
    pub ablation_matrix: AblationMatrixConfig,
    /// Optional cross-surface automated search loop rooted at this config.
    #[serde(default)]
    pub automated_search: AutomatedSearchConfig,
    /// Optional multi-seed runner settings for claim-bearing uncertainty reports.
    #[serde(default)]
    pub multi_seed: MultiSeedExperimentConfig,
    /// Optional coarse performance regression thresholds.
    #[serde(default)]
    pub performance_gates: PerformanceGateConfig,
}

impl UnseenPocketExperimentConfig {
    /// Validate a config before allocating runtime state.
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.research.validate()?;
        self.automated_search.validate()?;
        self.external_evaluation.validate()?;
        self.performance_gates.validate()?;
        if self.multi_seed.enabled && self.multi_seed.seeds.is_empty() {
            return Err("multi_seed.seeds must be non-empty when multi_seed.enabled=true".into());
        }
        Ok(())
    }
}

impl Default for UnseenPocketExperimentConfig {
    fn default() -> Self {
        Self {
            research: ResearchConfig::default(),
            surface_label: None,
            reviewer_benchmark: ReviewerBenchmarkConfig::default(),
            ablation: AblationConfig::default(),
            external_evaluation: ExternalEvaluationConfig::default(),
            ablation_matrix: AblationMatrixConfig::default(),
            automated_search: AutomatedSearchConfig::default(),
            multi_seed: MultiSeedExperimentConfig::default(),
            performance_gates: PerformanceGateConfig::default(),
        }
    }
}

/// Optional metadata describing which reviewer-facing benchmark contract a surface targets.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReviewerBenchmarkConfig {
    /// Stable benchmark dataset label surfaced in claim summaries and reviewer bundles.
    #[serde(default)]
    pub dataset: Option<String>,
}

/// Optional coarse performance gates for claim and medium-profile runs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceGateConfig {
    /// Minimum validation examples/sec required when set.
    #[serde(default)]
    pub min_validation_examples_per_second: Option<f64>,
    /// Minimum test examples/sec required when set.
    #[serde(default)]
    pub min_test_examples_per_second: Option<f64>,
    /// Maximum validation memory delta in MB allowed when set.
    #[serde(default)]
    pub max_validation_memory_mb: Option<f64>,
    /// Maximum test memory delta in MB allowed when set.
    #[serde(default)]
    pub max_test_memory_mb: Option<f64>,
}

impl PerformanceGateConfig {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        for (name, value) in [
            (
                "performance_gates.min_validation_examples_per_second",
                self.min_validation_examples_per_second,
            ),
            (
                "performance_gates.min_test_examples_per_second",
                self.min_test_examples_per_second,
            ),
            (
                "performance_gates.max_validation_memory_mb",
                self.max_validation_memory_mb,
            ),
            (
                "performance_gates.max_test_memory_mb",
                self.max_test_memory_mb,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value < 0.0 {
                    return Err(
                        format!("{name} must be omitted or a non-negative finite number").into(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Config for deterministic repeated runs over split, corruption, and sampling seeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSeedExperimentConfig {
    /// Whether this config is intended for the multi-seed entrypoint.
    #[serde(default)]
    pub enabled: bool,
    /// Base seeds. Each seed drives split, corruption, and sampling seeds with stable offsets.
    #[serde(default = "default_multi_seed_values")]
    pub seeds: Vec<u64>,
    /// Directory that owns per-seed artifacts and the aggregate report.
    #[serde(default = "default_multi_seed_artifact_root")]
    pub artifact_root_dir: PathBuf,
}

impl Default for MultiSeedExperimentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            seeds: default_multi_seed_values(),
            artifact_root_dir: default_multi_seed_artifact_root(),
        }
    }
}

fn default_multi_seed_values() -> Vec<u64> {
    vec![17, 42, 101]
}

fn default_multi_seed_artifact_root() -> PathBuf {
    PathBuf::from("./checkpoints/multi_seed")
}

/// Persisted automated multi-surface search summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchSummary {
    /// Root directory that owns the search artifacts.
    pub artifact_root: PathBuf,
    /// Search strategy used for this run.
    pub strategy: AutomatedSearchStrategy,
    /// Shared hard regression gates enforced across surfaces.
    pub hard_gates: AutomatedSearchHardGateConfig,
    /// Multi-objective score weights used after hard-gate filtering.
    pub score_weights: AutomatedSearchScoreWeightConfig,
    /// Executed candidates in ranked order.
    pub ranked_candidates: Vec<AutomatedSearchCandidateSummary>,
    /// Winning candidate identifier when at least one candidate survives gating.
    pub winning_candidate_id: Option<String>,
    /// Aggregate roadmap recommendation after the search cycle.
    pub roadmap_decision: String,
}

/// Persisted aggregate over deterministic seed repeats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSeedExperimentSummary {
    /// Root directory containing per-seed artifacts.
    pub artifact_root: PathBuf,
    /// Source config used to launch the run.
    pub source_config: PathBuf,
    /// One row per seed.
    pub seed_runs: Vec<MultiSeedRunSummary>,
    /// Aggregate claim and stability metrics across seeds.
    pub aggregates: MultiSeedAggregateReport,
    /// Decision text for whether this surface is stable enough for claim review.
    pub stability_decision: String,
}

impl MultiSeedExperimentSummary {
    fn with_aggregates(mut self) -> Self {
        self.aggregates = MultiSeedAggregateReport {
            candidate_valid_fraction: self.aggregate(|run| run.candidate_valid_fraction),
            strict_pocket_fit_score: self.aggregate(|run| run.strict_pocket_fit_score),
            unique_smiles_fraction: self.aggregate(|run| run.unique_smiles_fraction),
            rdkit_unique_smiles_fraction: self.aggregate(|run| run.rdkit_unique_smiles_fraction),
            slot_activation_mean: self.aggregate(|run| run.slot_activation_mean),
            gate_activation_mean: self.aggregate(|run| run.gate_activation_mean),
            leakage_proxy_mean: self.aggregate(|run| run.leakage_proxy_mean),
            topology_signature_similarity: self.aggregate(|run| run.topology_signature_similarity),
            geometry_signature_similarity: self.aggregate(|run| run.geometry_signature_similarity),
            pocket_signature_similarity: self.aggregate(|run| run.pocket_signature_similarity),
            examples_per_second: self.aggregate(|run| run.examples_per_second),
            test_examples_per_second: self.aggregate(|run| run.test_examples_per_second),
        };
        self.stability_decision = self.stability_decision();
        self
    }

    /// Aggregate one metric across seeds.
    pub fn aggregate(&self, metric: fn(&MultiSeedRunSummary) -> f64) -> MultiSeedMetricAggregate {
        let values = self.seed_runs.iter().map(metric).collect::<Vec<_>>();
        MultiSeedMetricAggregate::from_values(&values)
    }

    /// Stability decision combining slot, leakage, gate, geometry, and quality signals.
    pub fn stability_decision(&self) -> String {
        let validity = self.aggregate(|run| run.candidate_valid_fraction);
        let leakage = self.aggregate(|run| run.leakage_proxy_mean);
        let slot = self.aggregate(|run| run.slot_activation_mean);
        let pocket = self.aggregate(|run| run.strict_pocket_fit_score);
        let real_backend_backed = self.seed_runs.iter().all(|run| run.real_backend_backed);
        if self.seed_runs.len() < 2 {
            return "insufficient seeds for claim-bearing stability; run at least two seeds"
                .to_string();
        }
        if validity.min < 0.2 || pocket.min < 0.05 {
            return "unstable for claim use: at least one seed has weak validity or pocket fit"
                .to_string();
        }
        if leakage.max > 0.95 && pocket.mean < 0.2 {
            return "unstable for claim use: leakage control coincides with weak pocket fit"
                .to_string();
        }
        if slot.mean <= 0.0 {
            return "unstable for claim use: slot activation collapsed across seeds".to_string();
        }
        if real_backend_backed {
            "stable enough for larger-data real-backend claim review on the held-out pocket surface"
                .to_string()
        } else {
            "stable enough for larger-data held-out-pocket review; upgrade this surface with real backends before stronger reviewer claims"
                .to_string()
        }
    }
}

/// Aggregate report persisted by the multi-seed runner.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiSeedAggregateReport {
    /// Aggregate candidate validity fraction.
    pub candidate_valid_fraction: MultiSeedMetricAggregate,
    /// Aggregate strict pocket-fit score.
    pub strict_pocket_fit_score: MultiSeedMetricAggregate,
    /// Aggregate unique chemistry fraction.
    pub unique_smiles_fraction: MultiSeedMetricAggregate,
    /// Aggregate toolkit-backed unique chemistry fraction on the test split.
    #[serde(default)]
    pub rdkit_unique_smiles_fraction: MultiSeedMetricAggregate,
    /// Aggregate slot activation.
    pub slot_activation_mean: MultiSeedMetricAggregate,
    /// Aggregate gate activation.
    pub gate_activation_mean: MultiSeedMetricAggregate,
    /// Aggregate leakage proxy.
    pub leakage_proxy_mean: MultiSeedMetricAggregate,
    /// Aggregate topology slot-signature similarity.
    pub topology_signature_similarity: MultiSeedMetricAggregate,
    /// Aggregate geometry slot-signature similarity.
    pub geometry_signature_similarity: MultiSeedMetricAggregate,
    /// Aggregate pocket slot-signature similarity.
    pub pocket_signature_similarity: MultiSeedMetricAggregate,
    /// Aggregate evaluation throughput.
    pub examples_per_second: MultiSeedMetricAggregate,
    /// Aggregate test-split throughput for reviewer-facing summaries.
    #[serde(default)]
    pub test_examples_per_second: MultiSeedMetricAggregate,
}

/// Per-seed claim-surface summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSeedRunSummary {
    /// Base seed used for this run.
    pub seed: u64,
    /// Artifact directory for this seed.
    pub artifact_dir: PathBuf,
    /// Config hash persisted by the seed run.
    pub config_hash: String,
    /// Candidate validity fraction on the test split.
    pub candidate_valid_fraction: f64,
    /// Strict pocket-fit score on the test split.
    pub strict_pocket_fit_score: f64,
    /// Unique chemistry fraction on the test split.
    pub unique_smiles_fraction: f64,
    /// Toolkit-backed unique chemistry fraction on the test split.
    #[serde(default)]
    pub rdkit_unique_smiles_fraction: f64,
    /// Mean active slot fraction on the test split.
    pub slot_activation_mean: f64,
    /// Mean directed gate activation on the test split.
    pub gate_activation_mean: f64,
    /// Mean leakage proxy on the test split.
    pub leakage_proxy_mean: f64,
    /// Mean topology slot signature similarity.
    pub topology_signature_similarity: f64,
    /// Mean geometry slot signature similarity.
    pub geometry_signature_similarity: f64,
    /// Mean pocket slot signature similarity.
    pub pocket_signature_similarity: f64,
    /// Evaluation throughput for the test split.
    pub examples_per_second: f64,
    /// Explicit alias for test-split throughput in reviewer-facing summaries.
    #[serde(default)]
    pub test_examples_per_second: f64,
    /// Whether this seed run used active real backends on the claim surface.
    #[serde(default)]
    pub real_backend_backed: bool,
}

impl MultiSeedRunSummary {
    fn from_experiment(seed: u64, summary: &UnseenPocketExperimentSummary) -> Self {
        let test = &summary.test.comparison_summary;
        let slot = &summary.test.slot_stability;
        Self {
            seed,
            artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
            config_hash: summary.reproducibility.config_hash.clone(),
            candidate_valid_fraction: test.candidate_valid_fraction.unwrap_or(0.0),
            strict_pocket_fit_score: test.strict_pocket_fit_score.unwrap_or(0.0),
            unique_smiles_fraction: test.unique_smiles_fraction.unwrap_or(0.0),
            rdkit_unique_smiles_fraction: test.unique_smiles_fraction.unwrap_or(0.0),
            slot_activation_mean: test.slot_activation_mean,
            gate_activation_mean: test.gate_activation_mean,
            leakage_proxy_mean: test.leakage_proxy_mean,
            topology_signature_similarity: slot.topology_signature_similarity,
            geometry_signature_similarity: slot.geometry_signature_similarity,
            pocket_signature_similarity: slot.pocket_signature_similarity,
            examples_per_second: summary.test.resource_usage.examples_per_second,
            test_examples_per_second: summary.test.resource_usage.examples_per_second,
            real_backend_backed: claim_is_real_backend_backed(summary),
        }
    }
}

/// Mean/std/min/max aggregate for one seed-level metric.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiSeedMetricAggregate {
    /// Number of finite seed values included in the aggregate.
    pub count: usize,
    /// Mean value.
    pub mean: f64,
    /// Population standard deviation.
    pub std: f64,
    /// Standard error of the mean.
    pub standard_error: f64,
    /// Lower bound of a deterministic 95% t-style confidence interval for the mean.
    pub confidence95_low: f64,
    /// Upper bound of a deterministic 95% t-style confidence interval for the mean.
    pub confidence95_high: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
}

impl MultiSeedMetricAggregate {
    fn from_values(values: &[f64]) -> Self {
        let finite_values = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if finite_values.is_empty() {
            return Self::default();
        }
        let count = finite_values.len();
        let mean = finite_values.iter().sum::<f64>() / count as f64;
        let std = (finite_values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count as f64)
            .sqrt();
        let sample_std = if count > 1 {
            (finite_values
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (count - 1) as f64)
                .sqrt()
        } else {
            0.0
        };
        let standard_error = if count > 0 {
            sample_std / (count as f64).sqrt()
        } else {
            0.0
        };
        let half_width = t_critical_95(count.saturating_sub(1)) * standard_error;
        let min = finite_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = finite_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Self {
            count,
            mean,
            std,
            standard_error,
            confidence95_low: mean - half_width,
            confidence95_high: mean + half_width,
            min,
            max,
        }
    }
}

fn t_critical_95(degrees_of_freedom: usize) -> f64 {
    match degrees_of_freedom {
        0 => 0.0,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20..=29 => 2.045,
        30..=39 => 2.023,
        40..=59 => 2.000,
        60..=119 => 1.980,
        _ => 1.960,
    }
}

/// Persisted summary for one automated-search candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchCandidateSummary {
    /// Stable candidate identifier.
    pub candidate_id: String,
    /// Human-readable override labels applied to this candidate.
    pub overrides: Vec<String>,
    /// Artifact directory that owns this candidate.
    pub artifact_dir: PathBuf,
    /// Surface-level claim reports collected for this candidate.
    pub surfaces: Vec<AutomatedSearchSurfaceSummary>,
    /// Shared interaction review built from all surface artifacts when available.
    pub aggregate_interaction_review: Option<InteractionModeReview>,
    /// Hard-gate evaluation result.
    pub gate_result: AutomatedSearchGateResult,
    /// Multi-objective score when the candidate survives hard gates.
    pub score: Option<f64>,
}

/// Compact persisted summary for one scored surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchSurfaceSummary {
    /// Surface label used in reports.
    pub surface_label: String,
    /// Source config path for this surface.
    pub source_config: PathBuf,
    /// Artifact directory produced for this surface.
    pub artifact_dir: PathBuf,
    /// Compact claim report emitted by the surface run.
    pub claim_report: ClaimReport,
}

/// Hard-gate result attached to each candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchGateResult {
    /// Whether the candidate survives all hard gates.
    pub passed: bool,
    /// Explicit blocking reasons when the candidate fails.
    pub blocked_reasons: Vec<String>,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    id: String,
    overrides: Vec<SearchOverride>,
}

#[derive(Debug, Clone)]
enum SearchOverride {
    GateTemperature(f64),
    GateBias(f64),
    AttentionResidualScale(f64),
    FfnResidualScale(f64),
    RolloutSteps(usize),
    MinRolloutSteps(usize),
    StopProbabilityThreshold(f64),
    CoordinateStepScale(f64),
    TrainingStepWeightDecay(f64),
    CoordinateMomentum(f64),
    AtomMomentum(f64),
    AtomCommitTemperature(f64),
    MaxCoordinateDeltaNorm(f64),
    StopDeltaThreshold(f64),
    StopPatience(usize),
    BetaIntraRed(f64),
    GammaProbe(f64),
    DeltaLeak(f64),
    EtaGate(f64),
    MuSlot(f64),
}

/// Load an experiment configuration from JSON.
pub fn load_experiment_config(
    path: impl AsRef<std::path::Path>,
) -> Result<UnseenPocketExperimentConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

/// Execute the bounded multi-surface automated search rooted at one config.
pub fn run_automated_search(
    orchestrator_path: impl AsRef<Path>,
) -> Result<AutomatedSearchSummary, Box<dyn std::error::Error>> {
    let orchestrator_path = orchestrator_path.as_ref();
    let orchestrator_dir = orchestrator_path.parent().unwrap_or_else(|| Path::new("."));
    let config = load_experiment_config(orchestrator_path)?;
    if !config.automated_search.enabled {
        return Err("automated_search.enabled must be true for the search entrypoint".into());
    }
    config.validate()?;
    let search = &config.automated_search;
    for surface_path in &search.surface_configs {
        let resolved = resolve_relative_path(orchestrator_dir, surface_path);
        if !resolved.is_file() {
            return Err(format!("search surface config not found: {}", resolved.display()).into());
        }
    }
    let artifact_root = resolve_relative_path(orchestrator_dir, &search.artifact_root_dir);
    fs::create_dir_all(&artifact_root)?;

    let candidates = build_search_candidates(&config.research, search)?;
    let mut summaries = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        summaries.push(run_search_candidate(
            orchestrator_dir,
            &artifact_root,
            search,
            &candidate,
        )?);
    }
    summaries.sort_by(search_candidate_rank_cmp);
    let winning_candidate_id = summaries
        .iter()
        .find(|candidate| candidate.gate_result.passed)
        .map(|candidate| candidate.candidate_id.clone());
    let roadmap_decision = summarize_search_roadmap(&summaries);
    let summary = AutomatedSearchSummary {
        artifact_root: artifact_root.clone(),
        strategy: search.strategy,
        hard_gates: search.hard_gates.clone(),
        score_weights: search.score_weights.clone(),
        ranked_candidates: summaries,
        winning_candidate_id,
        roadmap_decision,
    };
    fs::write(
        artifact_root.join("search_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

/// Execute repeated experiment runs and persist aggregate seed-level uncertainty.
pub fn run_multi_seed_experiment(
    path: impl AsRef<Path>,
) -> Result<MultiSeedExperimentSummary, Box<dyn std::error::Error>> {
    let source_path = path.as_ref();
    let source_dir = source_path.parent().unwrap_or_else(|| Path::new("."));
    let config = load_experiment_config(source_path)?;
    if !config.multi_seed.enabled {
        return Err("multi_seed.enabled must be true for the multi-seed entrypoint".into());
    }
    config.validate()?;
    let artifact_root = resolve_relative_path(source_dir, &config.multi_seed.artifact_root_dir);
    fs::create_dir_all(&artifact_root)?;

    let mut seed_runs = Vec::with_capacity(config.multi_seed.seeds.len());
    for seed in &config.multi_seed.seeds {
        let mut seed_config = config.clone();
        seed_config.multi_seed.enabled = false;
        seed_config.research.data.split_seed = *seed;
        seed_config.research.data.generation_target.corruption_seed = seed.saturating_add(10_000);
        seed_config.research.data.generation_target.sampling_seed = seed.saturating_add(20_000);
        seed_config.research.training.checkpoint_dir = artifact_root.join(format!("seed_{seed}"));
        let run_summary = UnseenPocketExperiment::run_with_options(seed_config, false)?;
        seed_runs.push(MultiSeedRunSummary::from_experiment(*seed, &run_summary));
    }
    let summary = MultiSeedExperimentSummary {
        artifact_root: artifact_root.clone(),
        source_config: source_path.to_path_buf(),
        seed_runs,
        aggregates: MultiSeedAggregateReport::default(),
        stability_decision: String::new(),
    }
    .with_aggregates();
    fs::write(
        artifact_root.join("multi_seed_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

fn run_search_candidate(
    orchestrator_dir: &Path,
    artifact_root: &Path,
    search: &AutomatedSearchConfig,
    candidate: &SearchCandidate,
) -> Result<AutomatedSearchCandidateSummary, Box<dyn std::error::Error>> {
    let candidate_dir = artifact_root.join(&candidate.id);
    fs::create_dir_all(&candidate_dir)?;

    let mut surface_summaries = Vec::with_capacity(search.surface_configs.len());
    for surface_path in &search.surface_configs {
        let source_config = resolve_relative_path(orchestrator_dir, surface_path);
        let mut surface_config = load_experiment_config(&source_config)?;
        apply_search_candidate(
            &mut surface_config,
            candidate,
            &candidate_dir,
            &source_config,
        );
        let summary = UnseenPocketExperiment::run_with_options(surface_config, false)?;
        let claim_path = summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("claim_summary.json");
        let claim_report: ClaimReport = serde_json::from_str(&fs::read_to_string(&claim_path)?)?;
        surface_summaries.push(AutomatedSearchSurfaceSummary {
            surface_label: surface_label_from_config(&summary.config, &source_config),
            source_config: source_config.clone(),
            artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
            claim_report,
        });
    }

    let review_path = candidate_dir.join("interaction_mode_review.json");
    let aggregate_interaction_review = if review_path.exists() {
        Some(serde_json::from_str(&fs::read_to_string(review_path)?)?)
    } else {
        None
    };
    let gate_result = evaluate_search_gates(&surface_summaries, &search.hard_gates);
    let score = if gate_result.passed {
        Some(score_search_candidate(
            &surface_summaries,
            aggregate_interaction_review.as_ref(),
            &search.score_weights,
        ))
    } else {
        None
    };
    let candidate_summary = AutomatedSearchCandidateSummary {
        candidate_id: candidate.id.clone(),
        overrides: candidate
            .overrides
            .iter()
            .map(SearchOverride::label)
            .collect(),
        artifact_dir: candidate_dir.clone(),
        surfaces: surface_summaries,
        aggregate_interaction_review,
        gate_result,
        score,
    };
    fs::write(
        candidate_dir.join("candidate_summary.json"),
        serde_json::to_string_pretty(&candidate_summary)?,
    )?;
    Ok(candidate_summary)
}

fn build_search_candidates(
    base: &ResearchConfig,
    search: &AutomatedSearchConfig,
) -> Result<Vec<SearchCandidate>, Box<dyn std::error::Error>> {
    let axes = search_axes(base, &search.search_space);
    if axes.is_empty() {
        return Err("automated_search.search_space produced no candidate axes".into());
    }

    let mut candidates = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    if search.include_base_candidate {
        let base_candidate = SearchCandidate {
            id: "candidate_000_base".to_string(),
            overrides: Vec::new(),
        };
        seen.insert(candidate_signature(&base_candidate));
        candidates.push(base_candidate);
    }

    let mut enumerated = match search.strategy {
        AutomatedSearchStrategy::Grid => build_grid_candidates(&axes, search.max_candidates),
        AutomatedSearchStrategy::Random => {
            build_random_candidates(&axes, search.max_candidates, search.random_seed)
        }
    };
    for overrides in enumerated.drain(..) {
        let next = SearchCandidate {
            id: format!("candidate_{:03}", candidates.len()),
            overrides,
        };
        if seen.insert(candidate_signature(&next)) {
            candidates.push(next);
        }
        if candidates.len() >= search.max_candidates {
            break;
        }
    }
    if candidates.is_empty() {
        return Err("automated search failed to generate any candidate settings".into());
    }
    Ok(candidates)
}

fn search_axes(
    base: &ResearchConfig,
    search_space: &AutomatedSearchSpaceConfig,
) -> Vec<Vec<SearchOverride>> {
    let mut axes = Vec::new();
    push_f64_axis(&mut axes, &search_space.gate_temperature, |value| {
        SearchOverride::GateTemperature(value)
    });
    push_f64_axis(&mut axes, &search_space.gate_bias, |value| {
        SearchOverride::GateBias(value)
    });
    push_f64_axis(&mut axes, &search_space.attention_residual_scale, |value| {
        SearchOverride::AttentionResidualScale(value)
    });
    push_f64_axis(&mut axes, &search_space.ffn_residual_scale, |value| {
        SearchOverride::FfnResidualScale(value)
    });
    push_usize_axis(&mut axes, &search_space.rollout_steps, |value| {
        SearchOverride::RolloutSteps(value)
    });
    push_usize_axis(&mut axes, &search_space.min_rollout_steps, |value| {
        SearchOverride::MinRolloutSteps(value)
    });
    push_f64_axis(
        &mut axes,
        &search_space.stop_probability_threshold,
        |value| SearchOverride::StopProbabilityThreshold(value),
    );
    push_f64_axis(&mut axes, &search_space.coordinate_step_scale, |value| {
        SearchOverride::CoordinateStepScale(value)
    });
    push_f64_axis(
        &mut axes,
        &search_space.training_step_weight_decay,
        |value| SearchOverride::TrainingStepWeightDecay(value),
    );
    push_f64_axis(&mut axes, &search_space.coordinate_momentum, |value| {
        SearchOverride::CoordinateMomentum(value)
    });
    push_f64_axis(&mut axes, &search_space.atom_momentum, |value| {
        SearchOverride::AtomMomentum(value)
    });
    push_f64_axis(&mut axes, &search_space.atom_commit_temperature, |value| {
        SearchOverride::AtomCommitTemperature(value)
    });
    push_f64_axis(
        &mut axes,
        &search_space.max_coordinate_delta_norm,
        |value| SearchOverride::MaxCoordinateDeltaNorm(value),
    );
    push_f64_axis(&mut axes, &search_space.stop_delta_threshold, |value| {
        SearchOverride::StopDeltaThreshold(value)
    });
    push_usize_axis(&mut axes, &search_space.stop_patience, |value| {
        SearchOverride::StopPatience(value)
    });
    push_f64_axis(&mut axes, &search_space.beta_intra_red, |value| {
        SearchOverride::BetaIntraRed(value)
    });
    push_f64_axis(&mut axes, &search_space.gamma_probe, |value| {
        SearchOverride::GammaProbe(value)
    });
    push_f64_axis(&mut axes, &search_space.delta_leak, |value| {
        SearchOverride::DeltaLeak(value)
    });
    push_f64_axis(&mut axes, &search_space.eta_gate, |value| {
        SearchOverride::EtaGate(value)
    });
    push_f64_axis(&mut axes, &search_space.mu_slot, |value| {
        SearchOverride::MuSlot(value)
    });

    let base_overrides = vec![
        SearchOverride::GateTemperature(base.model.interaction_tuning.gate_temperature),
        SearchOverride::GateBias(base.model.interaction_tuning.gate_bias),
        SearchOverride::AttentionResidualScale(
            base.model.interaction_tuning.attention_residual_scale,
        ),
        SearchOverride::FfnResidualScale(base.model.interaction_tuning.ffn_residual_scale),
        SearchOverride::RolloutSteps(base.data.generation_target.rollout_steps),
        SearchOverride::MinRolloutSteps(base.data.generation_target.min_rollout_steps),
        SearchOverride::StopProbabilityThreshold(
            base.data.generation_target.stop_probability_threshold,
        ),
        SearchOverride::CoordinateStepScale(base.data.generation_target.coordinate_step_scale),
        SearchOverride::TrainingStepWeightDecay(
            base.data.generation_target.training_step_weight_decay,
        ),
        SearchOverride::CoordinateMomentum(base.data.generation_target.coordinate_momentum),
        SearchOverride::AtomMomentum(base.data.generation_target.atom_momentum),
        SearchOverride::AtomCommitTemperature(base.data.generation_target.atom_commit_temperature),
        SearchOverride::MaxCoordinateDeltaNorm(
            base.data.generation_target.max_coordinate_delta_norm,
        ),
        SearchOverride::StopDeltaThreshold(base.data.generation_target.stop_delta_threshold),
        SearchOverride::StopPatience(base.data.generation_target.stop_patience),
        SearchOverride::BetaIntraRed(base.training.loss_weights.beta_intra_red),
        SearchOverride::GammaProbe(base.training.loss_weights.gamma_probe),
        SearchOverride::DeltaLeak(base.training.loss_weights.delta_leak),
        SearchOverride::EtaGate(base.training.loss_weights.eta_gate),
        SearchOverride::MuSlot(base.training.loss_weights.mu_slot),
    ];
    axes.retain(|axis| {
        let Some(first) = axis.first() else {
            return false;
        };
        !base_overrides.iter().any(|base_override| {
            base_override.same_axis(first) && axis.len() == 1 && base_override.same_value(first)
        })
    });
    axes
}

fn build_grid_candidates(
    axes: &[Vec<SearchOverride>],
    max_candidates: usize,
) -> Vec<Vec<SearchOverride>> {
    let mut candidates = Vec::new();
    let mut current = Vec::new();
    build_grid_candidates_recursive(axes, 0, &mut current, &mut candidates, max_candidates);
    candidates
}

fn build_grid_candidates_recursive(
    axes: &[Vec<SearchOverride>],
    axis_index: usize,
    current: &mut Vec<SearchOverride>,
    out: &mut Vec<Vec<SearchOverride>>,
    max_candidates: usize,
) {
    if out.len() >= max_candidates {
        return;
    }
    if axis_index == axes.len() {
        out.push(current.clone());
        return;
    }
    for choice in &axes[axis_index] {
        current.push(choice.clone());
        build_grid_candidates_recursive(axes, axis_index + 1, current, out, max_candidates);
        current.pop();
        if out.len() >= max_candidates {
            break;
        }
    }
}

fn build_random_candidates(
    axes: &[Vec<SearchOverride>],
    max_candidates: usize,
    seed: u64,
) -> Vec<Vec<SearchOverride>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut results = Vec::new();
    let sample_count = max_candidates.saturating_mul(4).max(max_candidates);
    for _ in 0..sample_count {
        let candidate = axes
            .iter()
            .filter_map(|axis| axis.choose(&mut rng).cloned())
            .collect::<Vec<_>>();
        results.push(candidate);
        if results.len() >= max_candidates {
            break;
        }
    }
    results
}

fn push_f64_axis(
    axes: &mut Vec<Vec<SearchOverride>>,
    values: &[f64],
    build: impl Fn(f64) -> SearchOverride,
) {
    if values.is_empty() {
        return;
    }
    axes.push(values.iter().copied().map(build).collect());
}

fn push_usize_axis(
    axes: &mut Vec<Vec<SearchOverride>>,
    values: &[usize],
    build: impl Fn(usize) -> SearchOverride,
) {
    if values.is_empty() {
        return;
    }
    axes.push(values.iter().copied().map(build).collect());
}

fn candidate_signature(candidate: &SearchCandidate) -> String {
    candidate
        .overrides
        .iter()
        .map(SearchOverride::label)
        .collect::<Vec<_>>()
        .join("|")
}

fn apply_search_candidate(
    config: &mut UnseenPocketExperimentConfig,
    candidate: &SearchCandidate,
    candidate_dir: &Path,
    source_config: &Path,
) {
    for override_value in &candidate.overrides {
        override_value.apply(config);
    }
    let surface_label = surface_label_from_config(config, source_config);
    config.research.training.checkpoint_dir = candidate_dir.join(surface_label);
}

fn resolve_relative_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn evaluate_search_gates(
    surfaces: &[AutomatedSearchSurfaceSummary],
    gates: &AutomatedSearchHardGateConfig,
) -> AutomatedSearchGateResult {
    let mut blocked_reasons = Vec::new();
    for surface in surfaces {
        let label = &surface.surface_label;
        check_min_gate(
            &mut blocked_reasons,
            label,
            "candidate_valid_fraction",
            surface.claim_report.test.candidate_valid_fraction,
            gates.minimum_candidate_valid_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "rdkit_sanitized_fraction",
            backend_metric(
                &surface.claim_report.backend_metrics.chemistry_validity,
                "rdkit_sanitized_fraction",
            )
            .or(surface.claim_report.test.candidate_valid_fraction),
            gates.minimum_sanitized_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "unique_smiles_fraction",
            surface
                .claim_report
                .test
                .unique_smiles_fraction
                .or_else(|| {
                    backend_metric(
                        &surface.claim_report.backend_metrics.chemistry_validity,
                        "rdkit_unique_smiles_fraction",
                    )
                }),
            gates.minimum_unique_smiles_fraction,
        );
        check_max_gate(
            &mut blocked_reasons,
            label,
            "clash_fraction",
            backend_metric(
                &surface.claim_report.backend_metrics.pocket_compatibility,
                "clash_fraction",
            )
            .or_else(|| {
                backend_metric(
                    &surface.claim_report.backend_metrics.docking_affinity,
                    "clash_fraction",
                )
            }),
            gates.maximum_clash_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "strict_pocket_fit_score",
            surface.claim_report.test.strict_pocket_fit_score,
            gates.minimum_strict_pocket_fit_score,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "pocket_contact_fraction",
            surface.claim_report.test.pocket_contact_fraction,
            gates.minimum_pocket_contact_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "pocket_compatibility_fraction",
            surface.claim_report.test.pocket_compatibility_fraction,
            gates.minimum_pocket_compatibility_fraction,
        );
        let raw = &surface.claim_report.layered_generation_metrics.raw_rollout;
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_centroid_offset",
            Some(raw.mean_centroid_offset),
            gates.maximum_raw_centroid_offset,
        );
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_clash_fraction",
            Some(raw.clash_fraction),
            gates.maximum_raw_clash_fraction,
        );
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_mean_displacement",
            Some(raw.mean_displacement),
            gates.maximum_raw_mean_displacement,
        );
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_atom_change_fraction",
            Some(raw.atom_change_fraction),
            gates.maximum_raw_atom_change_fraction,
        );
        check_optional_min_gate(
            &mut blocked_reasons,
            label,
            "raw_uniqueness_proxy_fraction",
            Some(raw.uniqueness_proxy_fraction),
            gates.minimum_raw_uniqueness_proxy_fraction,
        );
    }
    AutomatedSearchGateResult {
        passed: blocked_reasons.is_empty(),
        blocked_reasons,
    }
}

fn check_optional_min_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: Option<f64>,
) {
    if let Some(threshold) = threshold {
        check_min_gate(blocked_reasons, surface_label, metric, value, threshold);
    }
}

fn check_optional_max_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: Option<f64>,
) {
    if let Some(threshold) = threshold {
        check_max_gate(blocked_reasons, surface_label, metric, value, threshold);
    }
}

fn check_min_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: f64,
) {
    match value {
        Some(value) if value + 1e-12 >= threshold => {}
        Some(value) => blocked_reasons.push(format!(
            "{surface_label}:{metric}={value:.4} below minimum {threshold:.4}"
        )),
        None => blocked_reasons.push(format!(
            "{surface_label}:{metric} missing; minimum {threshold:.4} required"
        )),
    }
}

fn check_max_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: f64,
) {
    match value {
        Some(value) if value <= threshold + 1e-12 => {}
        Some(value) => blocked_reasons.push(format!(
            "{surface_label}:{metric}={value:.4} above maximum {threshold:.4}"
        )),
        None => blocked_reasons.push(format!(
            "{surface_label}:{metric} missing; maximum {threshold:.4} required"
        )),
    }
}

fn score_search_candidate(
    surfaces: &[AutomatedSearchSurfaceSummary],
    review: Option<&InteractionModeReview>,
    weights: &AutomatedSearchScoreWeightConfig,
) -> f64 {
    let surface_count = surfaces.len().max(1) as f64;
    let mut chemistry = 0.0;
    let mut uniqueness = 0.0;
    let mut geometry = 0.0;
    let mut pocket = 0.0;
    let mut specialization = 0.0;
    let mut utilization = 0.0;

    for surface in surfaces {
        let test = &surface.claim_report.test;
        chemistry += mean_present(&[
            test.candidate_valid_fraction,
            backend_metric(
                &surface.claim_report.backend_metrics.chemistry_validity,
                "rdkit_sanitized_fraction",
            ),
        ]);
        uniqueness += mean_present(&[test.unique_smiles_fraction]);
        geometry += mean_present(&[
            test.strict_pocket_fit_score,
            test.mean_centroid_offset.map(|offset| 1.0 / (1.0 + offset)),
        ]);
        pocket += mean_present(&[
            test.pocket_contact_fraction,
            test.pocket_compatibility_fraction,
        ]);
        specialization += mean_present(&[
            Some(test.topology_specialization_score),
            Some(test.geometry_specialization_score),
            Some(test.pocket_specialization_score),
            Some(1.0 / (1.0 + test.leakage_proxy_mean.max(0.0))),
        ]);
        utilization += mean_present(&[
            Some(1.0 / (1.0 + test.slot_activation_mean.max(0.0))),
            Some(1.0 / (1.0 + test.gate_activation_mean.max(0.0))),
        ]);
    }

    chemistry /= surface_count;
    uniqueness /= surface_count;
    geometry /= surface_count;
    pocket /= surface_count;
    specialization /= surface_count;
    utilization /= surface_count;
    let interaction_review = review.map(interaction_review_score).unwrap_or(0.0);
    weights.chemistry * chemistry
        + weights.uniqueness * uniqueness
        + weights.geometry * geometry
        + weights.pocket * pocket
        + weights.specialization * specialization
        + weights.utilization * utilization
        + weights.interaction_review * interaction_review
}

fn interaction_review_score(review: &InteractionModeReview) -> f64 {
    let tally = &review.aggregate_test_tally;
    let total = tally.lightweight_wins + tally.transformer_wins + tally.ties;
    if total == 0 {
        return 0.0;
    }
    (tally.transformer_wins as f64 - tally.lightweight_wins as f64) / total as f64
}

fn mean_present(values: &[Option<f64>]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().flatten() {
        if value.is_finite() {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn backend_metric(metrics: &ReservedBackendMetrics, key: &str) -> Option<f64> {
    metrics
        .metrics
        .get(key)
        .copied()
        .filter(|value| value.is_finite())
}

fn search_candidate_rank_cmp(
    left: &AutomatedSearchCandidateSummary,
    right: &AutomatedSearchCandidateSummary,
) -> std::cmp::Ordering {
    match (left.gate_result.passed, right.gate_result.passed) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id)),
    }
}

fn summarize_search_roadmap(candidates: &[AutomatedSearchCandidateSummary]) -> String {
    let survived = candidates
        .iter()
        .filter(|candidate| candidate.gate_result.passed)
        .count();
    if survived > 0 {
        return format!(
            "{survived} candidate(s) survived chemistry, uniqueness, clash, and strict pocket-fit gates; keep automated search plus lightweight reranking as the mainline before considering adversarial training."
        );
    }
    let blocked = candidates.len();
    format!(
        "all {blocked} candidate(s) were blocked by hard regression gates; inspect blocked_reasons and expand bounded search or add a reranker before attempting higher-risk adversarial training."
    )
}

fn surface_label_from_config(
    config: &UnseenPocketExperimentConfig,
    source_config: &Path,
) -> String {
    config
        .surface_label
        .clone()
        .or_else(|| config.ablation.variant_label.clone())
        .unwrap_or_else(|| {
            source_config
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("surface")
                .trim_start_matches("unseen_pocket_")
                .to_string()
        })
}

impl SearchOverride {
    fn apply(&self, config: &mut UnseenPocketExperimentConfig) {
        match self {
            SearchOverride::GateTemperature(value) => {
                config.research.model.interaction_tuning.gate_temperature = *value;
            }
            SearchOverride::GateBias(value) => {
                config.research.model.interaction_tuning.gate_bias = *value;
            }
            SearchOverride::AttentionResidualScale(value) => {
                config
                    .research
                    .model
                    .interaction_tuning
                    .attention_residual_scale = *value;
            }
            SearchOverride::FfnResidualScale(value) => {
                config.research.model.interaction_tuning.ffn_residual_scale = *value;
            }
            SearchOverride::RolloutSteps(value) => {
                config.research.data.generation_target.rollout_steps = *value;
            }
            SearchOverride::MinRolloutSteps(value) => {
                config.research.data.generation_target.min_rollout_steps = *value;
            }
            SearchOverride::StopProbabilityThreshold(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .stop_probability_threshold = *value;
            }
            SearchOverride::CoordinateStepScale(value) => {
                config.research.data.generation_target.coordinate_step_scale = *value;
            }
            SearchOverride::TrainingStepWeightDecay(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .training_step_weight_decay = *value;
            }
            SearchOverride::CoordinateMomentum(value) => {
                config.research.data.generation_target.coordinate_momentum = *value;
            }
            SearchOverride::AtomMomentum(value) => {
                config.research.data.generation_target.atom_momentum = *value;
            }
            SearchOverride::AtomCommitTemperature(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .atom_commit_temperature = *value;
            }
            SearchOverride::MaxCoordinateDeltaNorm(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .max_coordinate_delta_norm = *value;
            }
            SearchOverride::StopDeltaThreshold(value) => {
                config.research.data.generation_target.stop_delta_threshold = *value;
            }
            SearchOverride::StopPatience(value) => {
                config.research.data.generation_target.stop_patience = *value;
            }
            SearchOverride::BetaIntraRed(value) => {
                config.research.training.loss_weights.beta_intra_red = *value;
            }
            SearchOverride::GammaProbe(value) => {
                config.research.training.loss_weights.gamma_probe = *value;
            }
            SearchOverride::DeltaLeak(value) => {
                config.research.training.loss_weights.delta_leak = *value;
            }
            SearchOverride::EtaGate(value) => {
                config.research.training.loss_weights.eta_gate = *value;
            }
            SearchOverride::MuSlot(value) => {
                config.research.training.loss_weights.mu_slot = *value;
            }
        }
    }

    fn label(&self) -> String {
        match self {
            SearchOverride::GateTemperature(value) => format!("gate_temperature={value:.6}"),
            SearchOverride::GateBias(value) => format!("gate_bias={value:.6}"),
            SearchOverride::AttentionResidualScale(value) => {
                format!("attention_residual_scale={value:.6}")
            }
            SearchOverride::FfnResidualScale(value) => format!("ffn_residual_scale={value:.6}"),
            SearchOverride::RolloutSteps(value) => format!("rollout_steps={value}"),
            SearchOverride::MinRolloutSteps(value) => format!("min_rollout_steps={value}"),
            SearchOverride::StopProbabilityThreshold(value) => {
                format!("stop_probability_threshold={value:.6}")
            }
            SearchOverride::CoordinateStepScale(value) => {
                format!("coordinate_step_scale={value:.6}")
            }
            SearchOverride::TrainingStepWeightDecay(value) => {
                format!("training_step_weight_decay={value:.6}")
            }
            SearchOverride::CoordinateMomentum(value) => format!("coordinate_momentum={value:.6}"),
            SearchOverride::AtomMomentum(value) => format!("atom_momentum={value:.6}"),
            SearchOverride::AtomCommitTemperature(value) => {
                format!("atom_commit_temperature={value:.6}")
            }
            SearchOverride::MaxCoordinateDeltaNorm(value) => {
                format!("max_coordinate_delta_norm={value:.6}")
            }
            SearchOverride::StopDeltaThreshold(value) => {
                format!("stop_delta_threshold={value:.6}")
            }
            SearchOverride::StopPatience(value) => format!("stop_patience={value}"),
            SearchOverride::BetaIntraRed(value) => format!("beta_intra_red={value:.6}"),
            SearchOverride::GammaProbe(value) => format!("gamma_probe={value:.6}"),
            SearchOverride::DeltaLeak(value) => format!("delta_leak={value:.6}"),
            SearchOverride::EtaGate(value) => format!("eta_gate={value:.6}"),
            SearchOverride::MuSlot(value) => format!("mu_slot={value:.6}"),
        }
    }

    fn same_axis(&self, other: &SearchOverride) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    fn same_value(&self, other: &SearchOverride) -> bool {
        self.label() == other.label()
    }
}

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

/// Runs a compact end-to-end unseen-pocket experiment on the current stack.
pub struct UnseenPocketExperiment;

impl UnseenPocketExperiment {
    /// Execute the experiment using the configured dataset loader.
    pub fn run(
        config: UnseenPocketExperimentConfig,
    ) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
        Self::run_with_options(config, false)
    }

    /// Execute the experiment with optional checkpoint resume.
    pub fn run_with_options(
        mut config: UnseenPocketExperimentConfig,
        resume_from_latest: bool,
    ) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
        config.validate()?;
        if let Some(mode) = config.ablation.interaction_mode_override {
            config.research.model.interaction_mode = mode;
        }
        if config.ablation.disable_geometry_interaction_bias {
            config
                .research
                .model
                .interaction_tuning
                .geometry_attention_bias_scale = 0.0;
        }
        if config.ablation.disable_rollout_pocket_guidance {
            config.research.data.generation_target.pocket_guidance_scale = 0.0;
        }
        let loaded = InMemoryDataset::load_from_config(&config.research.data)?;
        let dataset = loaded
            .dataset
            .with_pocket_feature_dim(config.research.model.pocket_feature_dim);
        let splits = dataset.split_by_protein_fraction_with_options(
            config.research.data.val_fraction,
            config.research.data.test_fraction,
            config.research.data.split_seed,
            config.research.data.stratify_by_measurement,
        );

        if splits.train.examples().is_empty() {
            return Err(
                "training split is empty; adjust val/test fractions or dataset size".into(),
            );
        }

        let device = parse_runtime_device(&config.research.runtime.device)?;
        let mut var_store = nn::VarStore::new(device);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config.research);
        let mut trainer = ResearchTrainer::new(&var_store, config.research.clone())?;
        trainer.set_dataset_validation_fingerprint(stable_json_hash(&loaded.validation));
        let mut resumed_checkpoint_metadata = None;
        if resume_from_latest {
            if let Some(checkpoint) = trainer.resume_from_latest(&mut var_store)? {
                resumed_checkpoint_metadata = Some(checkpoint.metadata.clone());
                if let Some(history) = load_experiment_history(
                    &config.research.training.checkpoint_dir,
                    checkpoint.metadata.step,
                )? {
                    trainer.replace_history(history);
                }
                log::info!(
                    "resumed experiment training from step {} at {}",
                    checkpoint.metadata.step,
                    checkpoint.weights_path.display()
                );
            }
        }
        let train_examples: Vec<_> = splits
            .train
            .examples()
            .iter()
            .map(|example| example.to_device(device))
            .collect();
        trainer.fit(&var_store, &system, &train_examples)?;

        let train_proteins: std::collections::BTreeSet<&str> = splits
            .train
            .examples()
            .iter()
            .map(|example| example.protein_id.as_str())
            .collect();

        let validation = evaluate_split(
            &system,
            splits.val.examples(),
            splits.train.examples(),
            &train_proteins,
            &config.research,
            config.ablation.clone(),
            &config.external_evaluation,
            "validation",
            device,
        );
        let test = evaluate_split(
            &system,
            splits.test.examples(),
            splits.train.examples(),
            &train_proteins,
            &config.research,
            config.ablation.clone(),
            &config.external_evaluation,
            "test",
            device,
        );

        let reproducibility = reproducibility_metadata(
            &config.research,
            &loaded.validation,
            resumed_checkpoint_metadata.as_ref(),
        );
        let dataset_validation = loaded.validation;
        let mut summary = UnseenPocketExperimentSummary {
            config,
            dataset_validation,
            split_report: SplitReport::from_datasets(&splits.train, &splits.val, &splits.test),
            reproducibility,
            training_history: trainer.history().to_vec(),
            validation,
            test,
            ablation_matrix: None,
            performance_gates: PerformanceGateReport::default(),
        };
        summary.performance_gates = build_performance_gate_report(
            &summary.config.performance_gates,
            &summary.validation.resource_usage,
            &summary.test.resource_usage,
        );
        if summary.config.ablation_matrix.enabled {
            let matrix = run_ablation_matrix(&summary.config)?;
            persist_ablation_matrix(&summary.config.research.training.checkpoint_dir, &matrix)?;
            summary.ablation_matrix = Some(matrix);
        }
        persist_experiment_summary(&summary)?;
        Ok(summary)
    }
}

fn persist_experiment_summary(
    summary: &UnseenPocketExperimentSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(&summary.config.research.training.checkpoint_dir)?;
    let artifact_dir = &summary.config.research.training.checkpoint_dir;
    let config_snapshot = artifact_dir.join("config.snapshot.json");
    let validation_path = artifact_dir.join("dataset_validation_report.json");
    let validation_alias_path = artifact_dir.join("dataset_validation.json");
    let split_report_path = artifact_dir.join("split_report.json");
    let summary_path = artifact_dir.join("experiment_summary.json");
    let claim_summary_path = artifact_dir.join("claim_summary.json");
    let bundle_path = artifact_dir.join("run_artifacts.json");
    fs::write(
        &config_snapshot,
        serde_json::to_string_pretty(&summary.config)?,
    )?;
    let validation_json = serde_json::to_string_pretty(&summary.dataset_validation)?;
    fs::write(&validation_path, &validation_json)?;
    fs::write(&validation_alias_path, validation_json)?;
    fs::write(
        &split_report_path,
        serde_json::to_string_pretty(&summary.split_report)?,
    )?;
    fs::write(&summary_path, serde_json::to_string_pretty(summary)?)?;
    fs::write(
        &claim_summary_path,
        serde_json::to_string_pretty(&build_claim_report(summary))?,
    )?;
    let bundle = RunArtifactBundle {
        schema_version: summary.reproducibility.artifact_bundle_schema_version,
        run_kind: RunKind::Experiment,
        artifact_dir: artifact_dir.clone(),
        config_hash: summary.reproducibility.config_hash.clone(),
        dataset_validation_fingerprint: summary
            .reproducibility
            .dataset_validation_fingerprint
            .clone(),
        metric_schema_version: summary.reproducibility.metric_schema_version,
        backend_environment: Some(build_backend_environment_report(summary)),
        paths: RunArtifactPaths {
            config_snapshot,
            dataset_validation_report: validation_path,
            split_report: split_report_path,
            run_summary: summary_path,
            run_bundle: bundle_path.clone(),
            latest_checkpoint: Some(artifact_dir.join("latest.ot")),
        },
    };
    fs::write(bundle_path, serde_json::to_string_pretty(&bundle)?)?;
    persist_interaction_reviews(summary)?;
    Ok(())
}

fn build_performance_gate_report(
    config: &PerformanceGateConfig,
    validation: &ResourceUsageMetrics,
    test: &ResourceUsageMetrics,
) -> PerformanceGateReport {
    let mut failed_reasons = Vec::new();
    if let Some(minimum) = config.min_validation_examples_per_second {
        if validation.examples_per_second < minimum {
            failed_reasons.push(format!(
                "validation examples/sec {:.4} below minimum {:.4}",
                validation.examples_per_second, minimum
            ));
        }
    }
    if let Some(minimum) = config.min_test_examples_per_second {
        if test.examples_per_second < minimum {
            failed_reasons.push(format!(
                "test examples/sec {:.4} below minimum {:.4}",
                test.examples_per_second, minimum
            ));
        }
    }
    if let Some(maximum) = config.max_validation_memory_mb {
        if validation.memory_usage_mb > maximum {
            failed_reasons.push(format!(
                "validation memory delta {:.4} MB above maximum {:.4}",
                validation.memory_usage_mb, maximum
            ));
        }
    }
    if let Some(maximum) = config.max_test_memory_mb {
        if test.memory_usage_mb > maximum {
            failed_reasons.push(format!(
                "test memory delta {:.4} MB above maximum {:.4}",
                test.memory_usage_mb, maximum
            ));
        }
    }
    PerformanceGateReport {
        passed: failed_reasons.is_empty(),
        failed_reasons,
        validation_examples_per_second: validation.examples_per_second,
        test_examples_per_second: test.examples_per_second,
        validation_memory_mb: validation.memory_usage_mb,
        test_memory_mb: test.memory_usage_mb,
    }
}

fn persist_interaction_reviews(
    summary: &UnseenPocketExperimentSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(matrix) = summary.ablation_matrix.as_ref() else {
        return Ok(());
    };
    let Some(surface_review) = build_surface_interaction_review(summary, matrix) else {
        return Ok(());
    };
    let artifact_dir = &summary.config.research.training.checkpoint_dir;
    fs::write(
        artifact_dir.join("interaction_mode_review.json"),
        serde_json::to_string_pretty(&surface_review)?,
    )?;
    if let Some(shared_review) = build_shared_interaction_review(artifact_dir.parent())? {
        fs::write(
            shared_review
                .review_root
                .join("interaction_mode_review.json"),
            serde_json::to_string_pretty(&shared_review)?,
        )?;
    }
    Ok(())
}

fn build_surface_interaction_review(
    summary: &UnseenPocketExperimentSummary,
    matrix: &AblationMatrixSummary,
) -> Option<InteractionSurfaceReview> {
    let lightweight = find_interaction_summary(summary, matrix, CrossAttentionMode::Lightweight)?;
    let transformer = find_interaction_summary(summary, matrix, CrossAttentionMode::Transformer)?;
    Some(InteractionSurfaceReview {
        surface_label: surface_label_from_dir(&summary.config.research.training.checkpoint_dir),
        artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
        validation: build_split_review(&lightweight.validation, &transformer.validation),
        test: build_split_review(&lightweight.test, &transformer.test),
    })
}

fn build_shared_interaction_review(
    root_dir: Option<&std::path::Path>,
) -> Result<Option<InteractionModeReview>, Box<dyn std::error::Error>> {
    let Some(root_dir) = root_dir else {
        return Ok(None);
    };
    let mut surfaces = Vec::new();
    for surface_name in ["claim_matrix", "harder_pressure", "tight_geometry_pressure"] {
        let artifact_dir = root_dir.join(surface_name);
        let claim_path = artifact_dir.join("claim_summary.json");
        let matrix_path = artifact_dir.join("ablation_matrix_summary.json");
        if !claim_path.exists() || !matrix_path.exists() {
            continue;
        }
        let claim: ClaimReport = serde_json::from_str(&fs::read_to_string(claim_path)?)?;
        let matrix: AblationMatrixSummary =
            serde_json::from_str(&fs::read_to_string(matrix_path)?)?;
        if let Some(surface_review) =
            build_surface_interaction_review_from_artifacts(artifact_dir.clone(), claim, matrix)
        {
            surfaces.push(surface_review);
        }
    }
    if surfaces.is_empty() {
        return Ok(None);
    }

    let mut aggregate_test_tally = InteractionWinLossTally::default();
    for surface in &surfaces {
        accumulate_tally(&mut aggregate_test_tally, &surface.test.tally);
    }
    let recommendation = summarize_interaction_recommendation(&surfaces, &aggregate_test_tally);
    Ok(Some(InteractionModeReview {
        review_root: root_dir.to_path_buf(),
        surfaces,
        aggregate_test_tally,
        recommendation,
    }))
}

fn build_surface_interaction_review_from_artifacts(
    artifact_dir: std::path::PathBuf,
    claim: ClaimReport,
    matrix: AblationMatrixSummary,
) -> Option<InteractionSurfaceReview> {
    let lightweight =
        find_interaction_summary_from_artifacts(&claim, &matrix, CrossAttentionMode::Lightweight)?;
    let transformer =
        find_interaction_summary_from_artifacts(&claim, &matrix, CrossAttentionMode::Transformer)?;
    Some(InteractionSurfaceReview {
        surface_label: surface_label_from_dir(&artifact_dir),
        artifact_dir,
        validation: build_split_review(&lightweight.validation, &transformer.validation),
        test: build_split_review(&lightweight.test, &transformer.test),
    })
}

fn build_split_review(
    lightweight: &GenerationQualitySummary,
    transformer: &GenerationQualitySummary,
) -> InteractionSplitReview {
    let geometric_fit = vec![
        metric_verdict(
            "candidate_valid_fraction",
            lightweight.candidate_valid_fraction,
            transformer.candidate_valid_fraction,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "pocket_contact_fraction",
            lightweight.pocket_contact_fraction,
            transformer.pocket_contact_fraction,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "pocket_compatibility_fraction",
            lightweight.pocket_compatibility_fraction,
            transformer.pocket_compatibility_fraction,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "mean_centroid_offset",
            lightweight.mean_centroid_offset,
            transformer.mean_centroid_offset,
            MetricDirection::LowerIsBetter,
        ),
        metric_verdict(
            "strict_pocket_fit_score",
            lightweight.strict_pocket_fit_score,
            transformer.strict_pocket_fit_score,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "unique_smiles_fraction",
            lightweight.unique_smiles_fraction,
            transformer.unique_smiles_fraction,
            MetricDirection::HigherIsBetter,
        ),
    ];
    let specialization = vec![
        metric_verdict(
            "topology_specialization_score",
            Some(lightweight.topology_specialization_score),
            Some(transformer.topology_specialization_score),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "geometry_specialization_score",
            Some(lightweight.geometry_specialization_score),
            Some(transformer.geometry_specialization_score),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "pocket_specialization_score",
            Some(lightweight.pocket_specialization_score),
            Some(transformer.pocket_specialization_score),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "leakage_proxy_mean",
            Some(lightweight.leakage_proxy_mean),
            Some(transformer.leakage_proxy_mean),
            MetricDirection::LowerIsBetter,
        ),
    ];
    let utilization = vec![
        metric_verdict(
            "slot_activation_mean",
            Some(lightweight.slot_activation_mean),
            Some(transformer.slot_activation_mean),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "gate_activation_mean",
            Some(lightweight.gate_activation_mean),
            Some(transformer.gate_activation_mean),
            MetricDirection::HigherIsBetter,
        ),
    ];
    let mut tally = InteractionWinLossTally::default();
    for verdict in geometric_fit
        .iter()
        .chain(specialization.iter())
        .chain(utilization.iter())
    {
        update_tally(&mut tally, verdict.winner);
    }
    InteractionSplitReview {
        lightweight: lightweight.clone(),
        transformer: transformer.clone(),
        tally,
        geometric_fit,
        specialization,
        utilization,
    }
}

fn metric_verdict(
    metric: &str,
    lightweight: Option<f64>,
    transformer: Option<f64>,
    direction: MetricDirection,
) -> InteractionMetricVerdict {
    const EPSILON: f64 = 1e-6;
    let (winner, preferred_delta) = match (lightweight, transformer) {
        (Some(lightweight), Some(transformer)) => {
            let signed_delta = match direction {
                MetricDirection::HigherIsBetter => lightweight - transformer,
                MetricDirection::LowerIsBetter => transformer - lightweight,
            };
            if signed_delta.abs() <= EPSILON {
                (MetricWinner::Tie, 0.0)
            } else if signed_delta > 0.0 {
                (MetricWinner::Lightweight, signed_delta)
            } else {
                (MetricWinner::Transformer, -signed_delta)
            }
        }
        _ => (MetricWinner::Tie, 0.0),
    };
    InteractionMetricVerdict {
        metric: metric.to_string(),
        direction,
        lightweight,
        transformer,
        winner,
        preferred_delta,
    }
}

fn update_tally(tally: &mut InteractionWinLossTally, winner: MetricWinner) {
    match winner {
        MetricWinner::Lightweight => tally.lightweight_wins += 1,
        MetricWinner::Transformer => tally.transformer_wins += 1,
        MetricWinner::Tie => tally.ties += 1,
    }
}

fn accumulate_tally(target: &mut InteractionWinLossTally, source: &InteractionWinLossTally) {
    target.lightweight_wins += source.lightweight_wins;
    target.transformer_wins += source.transformer_wins;
    target.ties += source.ties;
}

fn summarize_interaction_recommendation(
    surfaces: &[InteractionSurfaceReview],
    aggregate_test_tally: &InteractionWinLossTally,
) -> String {
    let lightweight_surface_wins = surfaces
        .iter()
        .filter(|surface| surface.test.tally.lightweight_wins > surface.test.tally.transformer_wins)
        .count();
    let transformer_surface_wins = surfaces
        .iter()
        .filter(|surface| surface.test.tally.transformer_wins > surface.test.tally.lightweight_wins)
        .count();
    if lightweight_surface_wins >= 2 && transformer_surface_wins == 0 {
        "keep interaction tuning active; lightweight still wins most reviewed test surfaces, so stronger claim-oriented interpretation work is premature".to_string()
    } else if transformer_surface_wins >= 2 && lightweight_surface_wins == 0 {
        "Transformer interaction now has a consistent reviewed advantage across test surfaces; claim-oriented interpretation work can start cautiously while keeping regression gates live".to_string()
    } else if aggregate_test_tally.transformer_wins > aggregate_test_tally.lightweight_wins {
        "interaction tuning is still active, but the tuned Transformer path now has the broader reviewed metric lead; require one more clean surface refresh before escalating claim scope".to_string()
    } else {
        "interaction tuning should remain the roadmap focus; reviewed surfaces still show a bounded tradeoff instead of a clean interaction-mode winner".to_string()
    }
}

fn surface_label_from_dir(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown_surface")
        .to_string()
}

fn find_interaction_summary<'a>(
    summary: &UnseenPocketExperimentSummary,
    matrix: &AblationMatrixSummary,
    mode: CrossAttentionMode,
) -> Option<AblationRunSummary> {
    match mode {
        CrossAttentionMode::Lightweight => matrix
            .variants
            .iter()
            .find(|variant| variant.test.interaction_mode == "lightweight")
            .cloned(),
        CrossAttentionMode::Transformer => {
            if summary.test.comparison_summary.interaction_mode == "transformer" {
                Some(AblationRunSummary {
                    variant_label: summary
                        .config
                        .ablation
                        .variant_label
                        .clone()
                        .unwrap_or_else(|| "base_run".to_string()),
                    validation: summary.validation.comparison_summary.clone(),
                    test: summary.test.comparison_summary.clone(),
                })
            } else {
                matrix
                    .variants
                    .iter()
                    .find(|variant| variant.test.interaction_mode == "transformer")
                    .cloned()
            }
        }
    }
}

fn find_interaction_summary_from_artifacts(
    claim: &ClaimReport,
    matrix: &AblationMatrixSummary,
    mode: CrossAttentionMode,
) -> Option<AblationRunSummary> {
    match mode {
        CrossAttentionMode::Lightweight => matrix
            .variants
            .iter()
            .find(|variant| variant.test.interaction_mode == "lightweight")
            .cloned(),
        CrossAttentionMode::Transformer => {
            if claim.test.interaction_mode == "transformer" {
                Some(AblationRunSummary {
                    variant_label: claim.run_label.clone(),
                    validation: claim.validation.clone(),
                    test: claim.test.clone(),
                })
            } else {
                matrix
                    .variants
                    .iter()
                    .find(|variant| variant.test.interaction_mode == "transformer")
                    .cloned()
            }
        }
    }
}

fn load_experiment_history(
    checkpoint_dir: &std::path::Path,
    resumed_step: usize,
) -> Result<Option<Vec<StepMetrics>>, Box<dyn std::error::Error>> {
    let path = checkpoint_dir.join("experiment_summary.json");
    if !path.exists() {
        return Ok(None);
    }

    let summary: UnseenPocketExperimentSummary = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(Some(
        summary
            .training_history
            .into_iter()
            .filter(|metrics| metrics.step <= resumed_step)
            .collect(),
    ))
}

/// Evaluate a dataset split using the trained modular research system.
pub fn evaluate_split(
    system: &Phase1ResearchSystem,
    examples: &[crate::data::MolecularExample],
    train_examples: &[crate::data::MolecularExample],
    train_proteins: &std::collections::BTreeSet<&str>,
    research: &ResearchConfig,
    ablation: AblationConfig,
    external_evaluation: &ExternalEvaluationConfig,
    split_label: &str,
    device: tch::Device,
) -> EvaluationMetrics {
    let start = Instant::now();
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();
    let memory_before = sys.used_memory() as f64 / (1024.0 * 1024.0);

    if examples.is_empty() {
        return EvaluationMetrics {
            representation_diagnostics: RepresentationDiagnostics {
                finite_forward_fraction: 0.0,
                unique_complex_fraction: 0.0,
                unseen_protein_fraction: 0.0,
                distance_probe_rmse: 0.0,
                topology_pocket_cosine_alignment: 0.0,
                topology_reconstruction_mse: 0.0,
                slot_activation_mean: 0.0,
                gate_activation_mean: 0.0,
                leakage_proxy_mean: 0.0,
            },
            proxy_task_metrics: ProxyTaskMetrics {
                affinity_probe_mae: 0.0,
                affinity_probe_rmse: 0.0,
                labeled_fraction: 0.0,
                affinity_by_measurement: Vec::new(),
            },
            split_context: SplitContextMetrics {
                example_count: 0,
                unique_complex_count: 0,
                unique_protein_count: 0,
                train_reference_protein_count: train_proteins.len(),
                ligand_atom_count_bins: BTreeMap::new(),
                pocket_atom_count_bins: BTreeMap::new(),
                measurement_family_histogram: BTreeMap::new(),
            },
            resource_usage: ResourceUsageMetrics {
                memory_usage_mb: memory_before,
                evaluation_time_ms: 0.0,
                examples_per_second: 0.0,
                average_ligand_atoms: 0.0,
                average_pocket_atoms: 0.0,
            },
            real_generation_metrics: disabled_real_generation_metrics(),
            layered_generation_metrics: empty_layered_generation_metrics(),
            comparison_summary: GenerationQualitySummary {
                primary_objective: primary_objective_label(research.training.primary_objective),
                variant_label: ablation.variant_label.clone(),
                interaction_mode: interaction_mode_label(
                    ablation
                        .interaction_mode_override
                        .unwrap_or(research.model.interaction_mode),
                ),
                candidate_valid_fraction: None,
                pocket_contact_fraction: None,
                pocket_compatibility_fraction: None,
                mean_centroid_offset: None,
                strict_pocket_fit_score: None,
                unique_smiles_fraction: None,
                unseen_protein_fraction: 0.0,
                topology_specialization_score: 0.0,
                geometry_specialization_score: 0.0,
                pocket_specialization_score: 0.0,
                slot_activation_mean: 0.0,
                gate_activation_mean: 0.0,
                leakage_proxy_mean: 0.0,
            },
            slot_stability: SlotStabilityMetrics::default(),
            strata: Vec::new(),
        };
    }

    let forwards: Vec<ResearchForward> = examples
        .iter()
        .map(|example| system.forward_example(&example.to_device(device)))
        .collect();

    sys.refresh_memory();
    let memory_after = sys.used_memory() as f64 / (1024.0 * 1024.0);

    let finite_forward_fraction = forwards
        .iter()
        .filter(|forward| {
            tensor_is_finite(&forward.encodings.topology.pooled_embedding)
                && tensor_is_finite(&forward.encodings.geometry.pooled_embedding)
                && tensor_is_finite(&forward.encodings.pocket.pooled_embedding)
        })
        .count() as f64
        / examples.len() as f64;

    let unique_ids = examples
        .iter()
        .map(|example| format!("{}::{}", example.protein_id, example.example_id))
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64;
    let unique_complex_fraction = unique_ids / examples.len() as f64;

    let unseen_protein_fraction = examples
        .iter()
        .filter(|example| !train_proteins.contains(example.protein_id.as_str()))
        .count() as f64
        / examples.len() as f64;

    let distance_probe_rmse = (examples
        .iter()
        .zip(forwards.iter())
        .map(|(example, forward)| {
            let target = example
                .geometry
                .pairwise_distances
                .mean(tch::Kind::Float)
                .double_value(&[]);
            let pred = forward
                .probes
                .geometry_distance_predictions
                .mean(tch::Kind::Float)
                .double_value(&[]);
            let error = pred - target;
            error * error
        })
        .sum::<f64>()
        / examples.len() as f64)
        .sqrt();

    let topology_pocket_cosine_alignment = forwards
        .iter()
        .map(|forward| {
            let topo = &forward.encodings.topology.pooled_embedding;
            let pocket = &forward.encodings.pocket.pooled_embedding;
            (topo * pocket).sum(tch::Kind::Float).double_value(&[])
                / (topo.norm().double_value(&[]) * pocket.norm().double_value(&[])).max(1e-6)
        })
        .sum::<f64>()
        / examples.len() as f64;

    let labeled_examples: Vec<(&crate::data::MolecularExample, &ResearchForward)> = examples
        .iter()
        .zip(forwards.iter())
        .filter(|(example, _)| example.targets.affinity_kcal_mol.is_some())
        .collect();
    let labeled_fraction = labeled_examples.len() as f64 / examples.len() as f64;
    let affinity_probe_mae = if labeled_examples.is_empty() {
        0.0
    } else {
        labeled_examples
            .iter()
            .map(|(example, forward)| {
                let target = example.targets.affinity_kcal_mol.unwrap() as f64;
                let pred = forward.probes.affinity_prediction.double_value(&[]);
                (pred - target).abs()
            })
            .sum::<f64>()
            / labeled_examples.len() as f64
    };
    let affinity_probe_rmse = if labeled_examples.is_empty() {
        0.0
    } else {
        (labeled_examples
            .iter()
            .map(|(example, forward)| {
                let target = example.targets.affinity_kcal_mol.unwrap() as f64;
                let pred = forward.probes.affinity_prediction.double_value(&[]);
                let error = pred - target;
                error * error
            })
            .sum::<f64>()
            / labeled_examples.len() as f64)
            .sqrt()
    };
    let affinity_by_measurement = measurement_breakdown(&labeled_examples);

    let topology_reconstruction_mse = forwards
        .iter()
        .map(|forward| {
            (forward.slots.topology.reconstructed_tokens.shallow_clone()
                - forward.encodings.topology.token_embeddings.shallow_clone())
            .pow_tensor_scalar(2.0)
            .mean(tch::Kind::Float)
            .double_value(&[])
        })
        .sum::<f64>()
        / examples.len() as f64;
    let topology_specialization_score = forwards
        .iter()
        .map(|forward| {
            if forward.probes.topology_adjacency_logits.numel() == 0 {
                0.0
            } else {
                forward
                    .probes
                    .topology_adjacency_logits
                    .sigmoid()
                    .mean(tch::Kind::Float)
                    .double_value(&[])
            }
        })
        .sum::<f64>()
        / examples.len() as f64;
    let geometry_specialization_score = 1.0 / (1.0 + distance_probe_rmse);
    let pocket_feature_rmse = (examples
        .iter()
        .zip(forwards.iter())
        .map(|(example, forward)| {
            let predicted = &forward.probes.pocket_feature_predictions;
            let target = &example.pocket.atom_features;
            if predicted.numel() == 0 || target.numel() == 0 {
                0.0
            } else {
                (predicted - target)
                    .pow_tensor_scalar(2.0)
                    .mean(tch::Kind::Float)
                    .double_value(&[])
            }
        })
        .sum::<f64>()
        / examples.len() as f64)
        .sqrt();
    let pocket_specialization_score = 1.0 / (1.0 + pocket_feature_rmse);

    let slot_activation_mean = if ablation.disable_slots {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let slot_means = [
                    active_slot_fraction(&forward.slots.topology.slot_weights),
                    active_slot_fraction(&forward.slots.geometry.slot_weights),
                    active_slot_fraction(&forward.slots.pocket.slot_weights),
                ];
                slot_means.iter().sum::<f64>() / slot_means.len() as f64
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let gate_activation_mean = if ablation.disable_cross_attention {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                [
                    forward.interactions.topo_from_geo.gate.double_value(&[0]),
                    forward
                        .interactions
                        .topo_from_pocket
                        .gate
                        .double_value(&[0]),
                    forward.interactions.geo_from_topo.gate.double_value(&[0]),
                    forward.interactions.geo_from_pocket.gate.double_value(&[0]),
                    forward
                        .interactions
                        .pocket_from_topo
                        .gate
                        .double_value(&[0]),
                    forward.interactions.pocket_from_geo.gate.double_value(&[0]),
                ]
                .iter()
                .sum::<f64>()
                    / 6.0
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let leakage_proxy_mean = if ablation.disable_leakage {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let topo = mean_slot(&forward.slots.topology.slots);
                let geo = mean_slot(&forward.slots.geometry.slots);
                let pocket = mean_slot(&forward.slots.pocket.slots);
                cosine_similarity(&topo, &geo).abs()
                    + cosine_similarity(&topo, &pocket).abs()
                    + cosine_similarity(&geo, &pocket).abs()
            })
            .sum::<f64>()
            / (examples.len() as f64 * 3.0)
    };
    let slot_stability = if ablation.disable_slots {
        SlotStabilityMetrics::default()
    } else {
        compute_slot_stability(&forwards)
    };

    let unique_protein_count = examples
        .iter()
        .map(|example| example.protein_id.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    let (real_generation_metrics, layered_generation_metrics) = evaluate_real_generation_metrics(
        examples,
        train_examples,
        &forwards,
        research,
        &ablation,
        external_evaluation,
        split_label,
    );
    let evaluation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (ligand_atom_count_bins, pocket_atom_count_bins, measurement_family_histogram) =
        split_histograms(examples);
    let average_ligand_atoms = average_ligand_atoms(examples);
    let average_pocket_atoms = average_pocket_atoms(examples);

    EvaluationMetrics {
        representation_diagnostics: RepresentationDiagnostics {
            finite_forward_fraction,
            unique_complex_fraction,
            unseen_protein_fraction,
            distance_probe_rmse,
            topology_pocket_cosine_alignment,
            topology_reconstruction_mse,
            slot_activation_mean,
            gate_activation_mean,
            leakage_proxy_mean,
        },
        proxy_task_metrics: ProxyTaskMetrics {
            affinity_probe_mae,
            affinity_probe_rmse,
            labeled_fraction,
            affinity_by_measurement,
        },
        split_context: SplitContextMetrics {
            example_count: examples.len(),
            unique_complex_count: unique_ids as usize,
            unique_protein_count,
            train_reference_protein_count: train_proteins.len(),
            ligand_atom_count_bins,
            pocket_atom_count_bins,
            measurement_family_histogram,
        },
        resource_usage: ResourceUsageMetrics {
            memory_usage_mb: (memory_after - memory_before).max(0.0),
            evaluation_time_ms,
            examples_per_second: if evaluation_time_ms > 0.0 {
                examples.len() as f64 / (evaluation_time_ms / 1000.0)
            } else {
                0.0
            },
            average_ligand_atoms,
            average_pocket_atoms,
        },
        real_generation_metrics: real_generation_metrics.clone(),
        layered_generation_metrics,
        comparison_summary: build_comparison_summary(
            research,
            &ablation,
            unseen_protein_fraction,
            topology_specialization_score,
            geometry_specialization_score,
            pocket_specialization_score,
            slot_activation_mean,
            gate_activation_mean,
            leakage_proxy_mean,
            &real_generation_metrics,
        ),
        slot_stability,
        strata: build_stratum_metrics(examples, train_proteins),
    }
}

fn disabled_real_generation_metrics() -> RealGenerationMetrics {
    RealGenerationMetrics {
        chemistry_validity: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "chemistry-validity backend unavailable for this evaluation".to_string(),
        },
        docking_affinity: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "docking or affinity backend unavailable for this evaluation".to_string(),
        },
        pocket_compatibility: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "pocket-compatibility backend unavailable for this evaluation".to_string(),
        },
    }
}

fn empty_layered_generation_metrics() -> LayeredGenerationMetrics {
    LayeredGenerationMetrics {
        raw_rollout: summarize_candidate_layer(&[], &NoveltyReferenceSignatures::default()),
        repaired_candidates: summarize_candidate_layer(&[], &NoveltyReferenceSignatures::default()),
        inferred_bond_candidates: summarize_candidate_layer(
            &[],
            &NoveltyReferenceSignatures::default(),
        ),
        reranked_candidates: summarize_candidate_layer(&[], &NoveltyReferenceSignatures::default()),
        deterministic_proxy_candidates: summarize_candidate_layer(
            &[],
            &NoveltyReferenceSignatures::default(),
        ),
        reranker_calibration: RerankerCalibrationReport::default(),
        backend_scored_candidates: BTreeMap::new(),
    }
}

fn evaluate_real_generation_metrics(
    examples: &[crate::data::MolecularExample],
    train_examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    research: &ResearchConfig,
    ablation: &AblationConfig,
    external_evaluation: &ExternalEvaluationConfig,
    split_label: &str,
) -> (RealGenerationMetrics, LayeredGenerationMetrics) {
    let layers = examples
        .iter()
        .zip(forwards.iter())
        .take(external_evaluation.generation_artifact_example_limit)
        .map(|(example, forward)| {
            generate_layered_candidates_with_options(
                example,
                forward,
                3,
                !ablation.disable_candidate_repair,
            )
        })
        .fold(
            CandidateGenerationLayers {
                raw_rollout: Vec::new(),
                repaired: Vec::new(),
                inferred_bond: Vec::new(),
            },
            |mut acc, mut next| {
                acc.raw_rollout.append(&mut next.raw_rollout);
                acc.repaired.append(&mut next.repaired);
                acc.inferred_bond.append(&mut next.inferred_bond);
                acc
            },
        );
    let candidates = layers
        .inferred_bond
        .iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .cloned()
        .collect::<Vec<_>>();
    let raw_rollout = layers
        .raw_rollout
        .iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .cloned()
        .collect::<Vec<_>>();
    let repaired = layers
        .repaired
        .iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .cloned()
        .collect::<Vec<_>>();
    let proxy_reranked = proxy_rerank_candidates(&candidates);
    let calibrated_reranker = CalibratedReranker::fit(&candidates);
    let reranked = calibrated_reranker.rerank(&candidates);
    let backend_candidates = final_backend_candidate_layer(&candidates, &reranked, &proxy_reranked);
    let novelty_reference = novelty_reference_signatures(train_examples);

    let mut layered = LayeredGenerationMetrics {
        raw_rollout: summarize_candidate_layer(&raw_rollout, &novelty_reference),
        repaired_candidates: summarize_candidate_layer(&repaired, &novelty_reference),
        inferred_bond_candidates: summarize_candidate_layer(&candidates, &novelty_reference),
        reranked_candidates: summarize_candidate_layer(&reranked, &novelty_reference),
        deterministic_proxy_candidates: summarize_candidate_layer(
            &proxy_reranked,
            &novelty_reference,
        ),
        reranker_calibration: calibrated_reranker.report(),
        backend_scored_candidates: BTreeMap::new(),
    };
    apply_raw_rollout_stability(&mut layered.raw_rollout, forwards);

    if candidates.is_empty() {
        let disabled = disabled_real_generation_metrics();
        maybe_persist_generation_artifacts(
            research,
            external_evaluation,
            split_label,
            &raw_rollout,
            &repaired,
            &candidates,
            &reranked,
            &proxy_reranked,
            &disabled,
            &layered,
        );
        return (disabled, layered);
    }

    let heuristic_chemistry =
        HeuristicChemistryValidityEvaluator.evaluate_chemistry(&backend_candidates);
    let chemistry = if external_evaluation.chemistry_backend.enabled {
        CommandChemistryValidityEvaluator {
            config: external_evaluation.chemistry_backend.clone(),
        }
        .evaluate_chemistry(&backend_candidates)
    } else {
        heuristic_chemistry.clone()
    };
    let heuristic_docking = HeuristicDockingEvaluator.evaluate_docking(&backend_candidates);
    let docking = if external_evaluation.docking_backend.enabled {
        CommandDockingEvaluator {
            config: external_evaluation.docking_backend.clone(),
        }
        .evaluate_docking(&backend_candidates)
    } else {
        heuristic_docking.clone()
    };
    let heuristic_pocket =
        HeuristicPocketCompatibilityEvaluator.evaluate_pocket_compatibility(&backend_candidates);
    let pocket = if external_evaluation.pocket_backend.enabled {
        CommandPocketCompatibilityEvaluator {
            config: external_evaluation.pocket_backend.clone(),
        }
        .evaluate_pocket_compatibility(&backend_candidates)
    } else {
        heuristic_pocket.clone()
    };

    let real_generation = RealGenerationMetrics {
        chemistry_validity: merge_backend_reports(
            chemistry,
            heuristic_chemistry,
            external_evaluation.chemistry_backend.enabled,
            backend_status(
                external_evaluation.chemistry_backend.enabled,
                "external chemistry-validity backend on modular rollout candidates",
                "active heuristic chemistry-validity backend on modular rollout candidates",
            ),
        ),
        docking_affinity: merge_backend_reports(
            docking,
            heuristic_docking,
            external_evaluation.docking_backend.enabled,
            backend_status(
                external_evaluation.docking_backend.enabled,
                "external docking backend on modular rollout candidates",
                "active heuristic docking-oriented hook on modular rollout candidates",
            ),
        ),
        pocket_compatibility: merge_backend_reports(
            pocket,
            heuristic_pocket,
            external_evaluation.pocket_backend.enabled,
            backend_status(
                external_evaluation.pocket_backend.enabled,
                "external pocket-compatibility backend on modular rollout candidates",
                "active heuristic pocket-compatibility hook on modular rollout candidates",
            ),
        ),
    };
    layered.backend_scored_candidates = backend_metric_layers(&real_generation);
    maybe_persist_generation_artifacts(
        research,
        external_evaluation,
        split_label,
        &raw_rollout,
        &repaired,
        &candidates,
        &reranked,
        &proxy_reranked,
        &real_generation,
        &layered,
    );
    (real_generation, layered)
}

fn final_backend_candidate_layer(
    inferred_bond: &[GeneratedCandidateRecord],
    reranked: &[GeneratedCandidateRecord],
    deterministic_proxy: &[GeneratedCandidateRecord],
) -> Vec<GeneratedCandidateRecord> {
    best_backend_compatible_candidate(reranked, 0)
        .into_iter()
        .chain(best_backend_compatible_candidate(deterministic_proxy, 1))
        .chain(best_backend_compatible_candidate(inferred_bond, 2))
        .max_by(|left, right| {
            left.selection_score
                .partial_cmp(&right.selection_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.layer_priority.cmp(&left.layer_priority))
        })
        .map(|candidate| vec![candidate.record])
        .or_else(|| {
            if !reranked.is_empty() {
                Some(reranked.to_vec())
            } else if !deterministic_proxy.is_empty() {
                Some(deterministic_proxy.to_vec())
            } else if !inferred_bond.is_empty() {
                Some(inferred_bond.to_vec())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

#[derive(Debug)]
struct BackendSelectionCandidate {
    record: GeneratedCandidateRecord,
    selection_score: f64,
    layer_priority: usize,
}

fn best_backend_compatible_candidate(
    candidates: &[GeneratedCandidateRecord],
    layer_priority: usize,
) -> Option<BackendSelectionCandidate> {
    if candidates.is_empty() {
        return None;
    }
    candidates
        .iter()
        .filter_map(|candidate| {
            let clash_fraction = candidate_backend_pocket_clash_fraction(candidate)?;
            (clash_fraction <= 0.0).then(|| BackendSelectionCandidate {
                record: candidate.clone(),
                selection_score: backend_claim_selection_score(candidate),
                layer_priority,
            })
        })
        .max_by(|left, right| {
            left.selection_score
                .partial_cmp(&right.selection_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn backend_status(external_enabled: bool, external: &str, heuristic: &str) -> String {
    if external_enabled {
        external.to_string()
    } else {
        heuristic.to_string()
    }
}

fn merge_backend_reports(
    primary: crate::models::ExternalEvaluationReport,
    heuristic: crate::models::ExternalEvaluationReport,
    external_enabled: bool,
    enabled_status: impl Into<String>,
) -> ReservedBackendMetrics {
    let mut merged = report_to_metrics(primary, enabled_status);
    if external_enabled {
        for metric in heuristic.metrics {
            merged
                .metrics
                .insert(format!("heuristic_{}", metric.metric_name), metric.value);
        }
    }
    merged
}

fn backend_metric_layers(
    metrics: &RealGenerationMetrics,
) -> BTreeMap<String, BTreeMap<String, f64>> {
    BTreeMap::from([
        (
            "chemistry_validity".to_string(),
            metrics.chemistry_validity.metrics.clone(),
        ),
        (
            "docking_affinity".to_string(),
            metrics.docking_affinity.metrics.clone(),
        ),
        (
            "pocket_compatibility".to_string(),
            metrics.pocket_compatibility.metrics.clone(),
        ),
    ])
}

fn summarize_candidate_layer(
    candidates: &[GeneratedCandidateRecord],
    novelty_reference: &NoveltyReferenceSignatures,
) -> CandidateLayerMetrics {
    if candidates.is_empty() {
        return CandidateLayerMetrics {
            candidate_count: 0,
            valid_fraction: 0.0,
            pocket_contact_fraction: 0.0,
            mean_centroid_offset: 0.0,
            clash_fraction: 0.0,
            mean_displacement: 0.0,
            atom_change_fraction: 0.0,
            uniqueness_proxy_fraction: 0.0,
            atom_type_sequence_diversity: 0.0,
            bond_topology_diversity: 0.0,
            coordinate_shape_diversity: 0.0,
            novel_atom_type_sequence_fraction: 0.0,
            novel_bond_topology_fraction: 0.0,
            novel_coordinate_shape_fraction: 0.0,
        };
    }
    let total = candidates.len() as f64;
    let valid_fraction = candidates
        .iter()
        .filter(|candidate| candidate_is_valid(candidate))
        .count() as f64
        / total;
    let pocket_contact_fraction = candidates
        .iter()
        .filter(|candidate| candidate_has_pocket_contact(candidate))
        .count() as f64
        / total;
    let mean_centroid_offset = candidates
        .iter()
        .map(candidate_centroid_offset)
        .filter(|value| value.is_finite())
        .sum::<f64>()
        / total;
    let clash_fraction = candidates.iter().map(candidate_clash_fraction).sum::<f64>() / total;
    let uniqueness_proxy_fraction = candidates
        .iter()
        .map(candidate_uniqueness_signature)
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64
        / total;
    let atom_type_sequence_diversity = diversity_fraction(candidates, candidate_atom_signature);
    let bond_topology_diversity = diversity_fraction(candidates, candidate_bond_signature);
    let coordinate_shape_diversity = diversity_fraction(candidates, candidate_shape_signature);
    let novel_atom_type_sequence_fraction = novelty_fraction(
        candidates,
        candidate_atom_signature,
        &novelty_reference.atom_signatures,
    );
    let novel_bond_topology_fraction = novelty_fraction(
        candidates,
        candidate_bond_signature,
        &novelty_reference.bond_signatures,
    );
    let novel_coordinate_shape_fraction = novelty_fraction(
        candidates,
        candidate_shape_signature,
        &novelty_reference.shape_signatures,
    );

    CandidateLayerMetrics {
        candidate_count: candidates.len(),
        valid_fraction,
        pocket_contact_fraction,
        mean_centroid_offset,
        clash_fraction,
        mean_displacement: 0.0,
        atom_change_fraction: 0.0,
        uniqueness_proxy_fraction,
        atom_type_sequence_diversity,
        bond_topology_diversity,
        coordinate_shape_diversity,
        novel_atom_type_sequence_fraction,
        novel_bond_topology_fraction,
        novel_coordinate_shape_fraction,
    }
}

fn diversity_fraction(
    candidates: &[GeneratedCandidateRecord],
    signature: fn(&GeneratedCandidateRecord) -> String,
) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates
        .iter()
        .map(signature)
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64
        / candidates.len() as f64
}

#[derive(Debug, Clone, Default)]
struct NoveltyReferenceSignatures {
    atom_signatures: std::collections::BTreeSet<String>,
    bond_signatures: std::collections::BTreeSet<String>,
    shape_signatures: std::collections::BTreeSet<String>,
}

fn novelty_reference_signatures(
    train_examples: &[crate::data::MolecularExample],
) -> NoveltyReferenceSignatures {
    NoveltyReferenceSignatures {
        atom_signatures: train_examples.iter().map(example_atom_signature).collect(),
        bond_signatures: train_examples.iter().map(example_bond_signature).collect(),
        shape_signatures: train_examples.iter().map(example_shape_signature).collect(),
    }
}

fn novelty_fraction(
    candidates: &[GeneratedCandidateRecord],
    signature: fn(&GeneratedCandidateRecord) -> String,
    references: &std::collections::BTreeSet<String>,
) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates
        .iter()
        .map(signature)
        .filter(|value| !references.contains(value))
        .count() as f64
        / candidates.len() as f64
}

fn proxy_rerank_candidates(
    candidates: &[GeneratedCandidateRecord],
) -> Vec<GeneratedCandidateRecord> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|left, right| {
        proxy_rerank_score(right)
            .partial_cmp(&proxy_rerank_score(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = (ranked.len() / 2).max(1).min(ranked.len());
    ranked.truncate(keep);
    ranked
}

fn proxy_rerank_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let valid = if candidate_is_valid(candidate) {
        1.0
    } else {
        0.0
    };
    let contact = if candidate_has_pocket_contact(candidate) {
        1.0
    } else {
        0.0
    };
    let centroid = 1.0 / (1.0 + candidate_centroid_offset(candidate).max(0.0));
    let clash = 1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0);
    let valence = if valence_sane_proxy(candidate) {
        1.0
    } else {
        0.0
    };
    0.25 * valid + 0.25 * contact + 0.2 * centroid + 0.2 * clash + 0.1 * valence
}

#[derive(Debug, Clone)]
struct CalibratedReranker {
    coefficients: BTreeMap<String, f64>,
    target_mean: f64,
    fitted_candidate_count: usize,
}

impl CalibratedReranker {
    fn fit(candidates: &[GeneratedCandidateRecord]) -> Self {
        let feature_names = reranker_feature_names();
        if candidates.is_empty() {
            return Self {
                coefficients: default_reranker_coefficients(),
                target_mean: 0.0,
                fitted_candidate_count: 0,
            };
        }

        let features = candidates.iter().map(reranker_features).collect::<Vec<_>>();
        let targets = candidates
            .iter()
            .map(backend_compatible_rerank_target)
            .collect::<Vec<_>>();
        let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let mut means = vec![0.0; feature_names.len()];
        for row in &features {
            for (index, value) in row.iter().enumerate() {
                means[index] += value;
            }
        }
        for mean in &mut means {
            *mean /= features.len() as f64;
        }

        let mut weights = vec![0.0; feature_names.len()];
        for (row, target) in features.iter().zip(targets.iter()) {
            for (index, value) in row.iter().enumerate() {
                weights[index] += (value - means[index]) * (target - target_mean);
            }
        }
        for weight in &mut weights {
            *weight = weight.max(0.0);
        }
        let sum = weights.iter().sum::<f64>();
        let coefficients = if sum <= 1e-12 {
            default_reranker_coefficients()
        } else {
            feature_names
                .iter()
                .zip(weights.iter())
                .map(|(name, weight)| ((*name).to_string(), weight / sum))
                .collect()
        };

        Self {
            coefficients,
            target_mean,
            fitted_candidate_count: candidates.len(),
        }
    }

    fn rerank(&self, candidates: &[GeneratedCandidateRecord]) -> Vec<GeneratedCandidateRecord> {
        let mut ranked = candidates.to_vec();
        ranked.sort_by(|left, right| {
            self.score(right)
                .partial_cmp(&self.score(left))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.source.cmp(&right.source))
        });
        let keep = (ranked.len() / 2).max(1).min(ranked.len());
        ranked.truncate(keep);
        ranked
    }

    fn score(&self, candidate: &GeneratedCandidateRecord) -> f64 {
        reranker_feature_names()
            .iter()
            .zip(reranker_features(candidate).iter())
            .map(|(name, value)| self.coefficients.get(*name).copied().unwrap_or(0.0) * value)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    fn report(&self) -> RerankerCalibrationReport {
        RerankerCalibrationReport {
            method: "bounded_nonnegative_feature_target_covariance_v1".to_string(),
            coefficients: self.coefficients.clone(),
            target_mean: self.target_mean,
            fitted_candidate_count: self.fitted_candidate_count,
        }
    }
}

fn reranker_feature_names() -> [&'static str; 6] {
    [
        "valid",
        "valence_sane",
        "pocket_contact",
        "centroid_fit",
        "clash_free",
        "bond_density_fit",
    ]
}

fn reranker_features(candidate: &GeneratedCandidateRecord) -> Vec<f64> {
    vec![
        if candidate_is_valid(candidate) {
            1.0
        } else {
            0.0
        },
        if valence_sane_proxy(candidate) {
            1.0
        } else {
            0.0
        },
        if candidate_has_pocket_contact(candidate) {
            1.0
        } else {
            0.0
        },
        centroid_fit_feature(candidate),
        1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0),
        bond_density_fit_feature(candidate),
    ]
}

fn default_reranker_coefficients() -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("valid".to_string(), 0.22),
        ("valence_sane".to_string(), 0.18),
        ("pocket_contact".to_string(), 0.20),
        ("centroid_fit".to_string(), 0.18),
        ("clash_free".to_string(), 0.17),
        ("bond_density_fit".to_string(), 0.05),
    ])
}

fn backend_compatible_rerank_target(candidate: &GeneratedCandidateRecord) -> f64 {
    let features = reranker_features(candidate);
    (0.24 * features[0]
        + 0.18 * features[1]
        + 0.20 * features[2]
        + 0.20 * features[3]
        + 0.15 * features[4]
        + 0.03 * features[5])
        .clamp(0.0, 1.0)
}

fn centroid_fit_feature(candidate: &GeneratedCandidateRecord) -> f64 {
    let radius = (candidate.pocket_radius as f64).max(1.0);
    (1.0 - candidate_centroid_offset(candidate) / (radius + 2.0)).clamp(0.0, 1.0)
}

fn bond_density_fit_feature(candidate: &GeneratedCandidateRecord) -> f64 {
    let atoms = candidate.atom_types.len();
    if atoms < 2 {
        return 0.0;
    }
    let density = candidate.inferred_bonds.len() as f64 / atoms as f64;
    (1.0 - (density - 1.05).abs() / 1.05).clamp(0.0, 1.0)
}

fn valence_sane_proxy(candidate: &GeneratedCandidateRecord) -> bool {
    if candidate.atom_types.is_empty() {
        return false;
    }
    let mut degrees = vec![0usize; candidate.atom_types.len()];
    for &(left, right) in &candidate.inferred_bonds {
        if left < degrees.len() && right < degrees.len() {
            degrees[left] += 1;
            degrees[right] += 1;
        }
    }
    degrees
        .iter()
        .zip(candidate.atom_types.iter())
        .all(|(degree, atom_type)| *degree <= max_reasonable_valence(*atom_type))
}

fn max_reasonable_valence(atom_type: i64) -> usize {
    match atom_type {
        1 => 1,
        6 => 4,
        7 => 4,
        8 => 3,
        9 | 17 | 35 | 53 => 1,
        15 => 5,
        16 => 6,
        _ => 4,
    }
}

fn apply_raw_rollout_stability(layer: &mut CandidateLayerMetrics, forwards: &[ResearchForward]) {
    let final_steps = forwards
        .iter()
        .filter_map(|forward| forward.generation.rollout.steps.last())
        .collect::<Vec<_>>();
    if final_steps.is_empty() {
        return;
    }
    let denom = final_steps.len() as f64;
    layer.mean_displacement = final_steps
        .iter()
        .map(|step| step.mean_displacement)
        .sum::<f64>()
        / denom;
    layer.atom_change_fraction = final_steps
        .iter()
        .map(|step| step.atom_change_fraction)
        .sum::<f64>()
        / denom;
}

#[derive(Debug, Serialize)]
struct LayeredGenerationArtifact<'a> {
    schema_version: u32,
    split_label: &'a str,
    layered_metrics: &'a LayeredGenerationMetrics,
    raw_rollout_candidates: &'a [GeneratedCandidateRecord],
    repaired_candidates: &'a [GeneratedCandidateRecord],
    inferred_bond_candidates: &'a [GeneratedCandidateRecord],
    deterministic_proxy_candidates: &'a [GeneratedCandidateRecord],
    reranked_candidates: &'a [GeneratedCandidateRecord],
    backend_metrics: &'a RealGenerationMetrics,
    backend_failure_examples: Vec<BackendFailureExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendFailureExample {
    backend: String,
    example_id: String,
    protein_id: String,
    candidate_source: String,
    reason: String,
    backend_status: String,
    source_pocket_path: Option<String>,
    source_ligand_path: Option<String>,
}

fn maybe_persist_generation_artifacts(
    research: &ResearchConfig,
    external_evaluation: &ExternalEvaluationConfig,
    split_label: &str,
    raw_rollout: &[GeneratedCandidateRecord],
    repaired: &[GeneratedCandidateRecord],
    inferred_bond: &[GeneratedCandidateRecord],
    reranked: &[GeneratedCandidateRecord],
    deterministic_proxy: &[GeneratedCandidateRecord],
    backend_metrics: &RealGenerationMetrics,
    layered_metrics: &LayeredGenerationMetrics,
) {
    if !external_evaluation.persist_generation_artifacts {
        return;
    }
    if fs::create_dir_all(&research.training.checkpoint_dir).is_err() {
        return;
    }
    let artifact = LayeredGenerationArtifact {
        schema_version: 2,
        split_label,
        layered_metrics,
        raw_rollout_candidates: raw_rollout,
        repaired_candidates: repaired,
        inferred_bond_candidates: inferred_bond,
        deterministic_proxy_candidates: deterministic_proxy,
        reranked_candidates: reranked,
        backend_metrics,
        backend_failure_examples: backend_failure_examples(inferred_bond, backend_metrics, 16),
    };
    if let Ok(content) = serde_json::to_string_pretty(&artifact) {
        let path = research
            .training
            .checkpoint_dir
            .join(format!("generation_layers_{split_label}.json"));
        if let Err(error) = fs::write(&path, content) {
            log::warn!(
                "failed to persist generation layer artifact {}: {}",
                path.display(),
                error
            );
        }
    }
}

fn backend_failure_examples(
    candidates: &[GeneratedCandidateRecord],
    metrics: &RealGenerationMetrics,
    limit: usize,
) -> Vec<BackendFailureExample> {
    let mut examples = Vec::new();
    collect_backend_command_failures(
        &mut examples,
        "chemistry_validity",
        &metrics.chemistry_validity,
        candidates,
        limit,
    );
    collect_backend_command_failures(
        &mut examples,
        "docking_affinity",
        &metrics.docking_affinity,
        candidates,
        limit,
    );
    collect_backend_command_failures(
        &mut examples,
        "pocket_compatibility",
        &metrics.pocket_compatibility,
        candidates,
        limit,
    );

    for candidate in candidates {
        if examples.len() >= limit {
            break;
        }
        if !candidate_is_valid(candidate) {
            examples.push(failure_example(
                "chemistry_validity",
                candidate,
                "candidate has inconsistent atom/coordinate lengths or non-finite coordinates",
                &metrics.chemistry_validity.status,
            ));
        } else if !valence_sane_proxy(candidate) {
            examples.push(failure_example(
                "chemistry_validity",
                candidate,
                "candidate exceeds lightweight valence sanity proxy",
                &metrics.chemistry_validity.status,
            ));
        }
        if examples.len() >= limit {
            break;
        }
        if candidate.source_pocket_path.is_none() {
            examples.push(failure_example(
                "pocket_compatibility",
                candidate,
                "candidate is missing source pocket structure provenance",
                &metrics.pocket_compatibility.status,
            ));
        } else if candidate_clash_fraction(candidate) > 0.0 {
            examples.push(failure_example(
                "pocket_compatibility",
                candidate,
                "candidate has nonzero lightweight pocket clash fraction",
                &metrics.pocket_compatibility.status,
            ));
        }
    }
    examples
}

fn collect_backend_command_failures(
    failures: &mut Vec<BackendFailureExample>,
    backend: &str,
    metrics: &ReservedBackendMetrics,
    candidates: &[GeneratedCandidateRecord],
    limit: usize,
) {
    if failures.len() >= limit {
        return;
    }
    let reason = backend_command_failure_reason(metrics).or_else(|| {
        let missing = metrics
            .metrics
            .get("backend_missing_structure_fraction")
            .copied()
            .unwrap_or(0.0);
        (missing > 0.0).then(|| {
            format!("backend_missing_structure_fraction={missing:.4} for scored candidates")
        })
    });
    let Some(reason) = reason else {
        return;
    };
    for candidate in candidates.iter().take(limit.saturating_sub(failures.len())) {
        failures.push(failure_example(
            backend,
            candidate,
            &reason,
            &metrics.status,
        ));
    }
}

fn backend_command_failure_reason(metrics: &ReservedBackendMetrics) -> Option<String> {
    if let Some((name, _)) = metrics
        .metrics
        .iter()
        .find(|(name, value)| name.ends_with("_available") && **value <= 0.0)
    {
        return Some(format!("{name}=0"));
    }
    [
        "backend_command_failed",
        "backend_command_spawn_error",
        "backend_output_parse_error",
        "backend_missing_schema_version",
        "backend_missing_examples_scored",
        "backend_missing_executable",
        "backend_tempfile_error",
        "backend_input_write_error",
    ]
    .iter()
    .find(|name| metrics.metrics.get(**name).copied().unwrap_or(0.0) > 0.0)
    .map(|name| format!("{name}=1"))
}

fn failure_example(
    backend: &str,
    candidate: &GeneratedCandidateRecord,
    reason: &str,
    backend_status: &str,
) -> BackendFailureExample {
    BackendFailureExample {
        backend: backend.to_string(),
        example_id: candidate.example_id.clone(),
        protein_id: candidate.protein_id.clone(),
        candidate_source: candidate.source.clone(),
        reason: reason.to_string(),
        backend_status: backend_status.to_string(),
        source_pocket_path: candidate.source_pocket_path.clone(),
        source_ligand_path: candidate.source_ligand_path.clone(),
    }
}

fn candidate_is_valid(candidate: &GeneratedCandidateRecord) -> bool {
    !candidate.atom_types.is_empty()
        && candidate.atom_types.len() == candidate.coords.len()
        && candidate
            .coords
            .iter()
            .all(|coord| coord.iter().all(|value| value.is_finite()))
}

fn candidate_has_pocket_contact(candidate: &GeneratedCandidateRecord) -> bool {
    candidate.coords.iter().any(|coord| {
        coord_distance(coord, &candidate.pocket_centroid) <= (candidate.pocket_radius + 2.0) as f64
    })
}

fn candidate_centroid_offset(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return f64::INFINITY;
    }
    let mut centroid = [0.0_f64; 3];
    for coord in &candidate.coords {
        centroid[0] += coord[0] as f64;
        centroid[1] += coord[1] as f64;
        centroid[2] += coord[2] as f64;
    }
    let denom = candidate.coords.len() as f64;
    let centroid = [
        (centroid[0] / denom) as f32,
        (centroid[1] / denom) as f32,
        (centroid[2] / denom) as f32,
    ];
    coord_distance(&centroid, &candidate.pocket_centroid)
}

fn candidate_clash_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.len() < 2 {
        return 0.0;
    }
    let bonds = candidate
        .inferred_bonds
        .iter()
        .map(|&(left, right)| {
            if left < right {
                (left, right)
            } else {
                (right, left)
            }
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut total = 0_usize;
    let mut clashing = 0_usize;
    for left in 0..candidate.coords.len() {
        for right in (left + 1)..candidate.coords.len() {
            if bonds.contains(&(left, right)) {
                continue;
            }
            total += 1;
            if coord_distance(&candidate.coords[left], &candidate.coords[right]) < 0.9 {
                clashing += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        clashing as f64 / total as f64
    }
}

fn candidate_backend_pocket_clash_fraction(candidate: &GeneratedCandidateRecord) -> Option<f64> {
    let pocket_path = candidate.source_pocket_path.as_deref()?;
    let pocket_coords = parse_pdb_coords(pocket_path)?;
    let candidate_coords = backend_candidate_coords(candidate);
    if pocket_coords.is_empty() || candidate_coords.is_empty() {
        return None;
    }
    let clashing = candidate_coords
        .iter()
        .filter(|coord| {
            pocket_coords
                .iter()
                .any(|pocket| coord_distance(coord, pocket) < 1.2)
        })
        .count();
    Some(clashing as f64 / candidate_coords.len() as f64)
}

fn candidate_backend_atom_coverage_fraction(candidate: &GeneratedCandidateRecord) -> Option<f64> {
    let pocket_path = candidate.source_pocket_path.as_deref()?;
    let pocket_coords = parse_pdb_coords(pocket_path)?;
    let candidate_coords = backend_candidate_coords(candidate);
    if pocket_coords.is_empty() || candidate_coords.is_empty() {
        return None;
    }
    let covered = candidate_coords
        .iter()
        .filter(|coord| {
            pocket_coords
                .iter()
                .any(|pocket| coord_distance(coord, pocket) <= 3.5)
        })
        .count();
    Some(covered as f64 / candidate_coords.len() as f64)
}

fn backend_claim_selection_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let clash_penalty = candidate_backend_pocket_clash_fraction(candidate).unwrap_or(1.0);
    let coverage = candidate_backend_atom_coverage_fraction(candidate).unwrap_or(0.0);
    let centroid_fit = 1.0 / (1.0 + candidate_centroid_offset(candidate).max(0.0));
    let valence_bonus = if valence_sane_proxy(candidate) {
        1.0
    } else {
        0.98
    };
    ((coverage * centroid_fit * valence_bonus) - clash_penalty).clamp(0.0, 1.0)
}

fn backend_candidate_coords(candidate: &GeneratedCandidateRecord) -> Vec<[f32; 3]> {
    let origin = candidate.coordinate_frame_origin;
    candidate
        .coords
        .iter()
        .map(|coord| {
            [
                coord[0] + origin[0],
                coord[1] + origin[1],
                coord[2] + origin[2],
            ]
        })
        .collect()
}

fn parse_pdb_coords(path: &str) -> Option<Vec<[f32; 3]>> {
    let content = fs::read_to_string(path).ok()?;
    let mut coords = Vec::new();
    for line in content.lines() {
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        let x = line.get(30..38)?.trim().parse::<f32>().ok()?;
        let y = line.get(38..46)?.trim().parse::<f32>().ok()?;
        let z = line.get(46..54)?.trim().parse::<f32>().ok()?;
        coords.push([x, y, z]);
    }
    Some(coords)
}

fn candidate_uniqueness_signature(candidate: &GeneratedCandidateRecord) -> String {
    format!(
        "{}::{}::{}",
        candidate_atom_signature(candidate),
        candidate_bond_signature(candidate),
        candidate_shape_signature(candidate)
    )
}

fn candidate_atom_signature(candidate: &GeneratedCandidateRecord) -> String {
    format!("{:?}", candidate.atom_types)
}

fn example_atom_signature(example: &crate::data::MolecularExample) -> String {
    let atom_count = ligand_atom_count(example);
    let mut atom_types = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        atom_types.push(example.topology.atom_types.int64_value(&[index as i64]));
    }
    format!("{:?}", atom_types)
}

fn candidate_bond_signature(candidate: &GeneratedCandidateRecord) -> String {
    let mut bonds = candidate
        .inferred_bonds
        .iter()
        .map(|(left, right)| {
            let (low, high) = if left <= right {
                (*left, *right)
            } else {
                (*right, *left)
            };
            format!("{low}-{high}")
        })
        .collect::<Vec<_>>();
    bonds.sort();
    bonds.join("|")
}

fn example_bond_signature(example: &crate::data::MolecularExample) -> String {
    let atom_count = ligand_atom_count(example);
    let mut bonds = Vec::new();
    for left in 0..atom_count {
        for right in (left + 1)..atom_count {
            if example
                .topology
                .adjacency
                .double_value(&[left as i64, right as i64])
                > 0.5
            {
                bonds.push(format!("{left}-{right}"));
            }
        }
    }
    bonds.sort();
    bonds.join("|")
}

fn candidate_shape_signature(candidate: &GeneratedCandidateRecord) -> String {
    let coord_buckets = candidate
        .coords
        .iter()
        .map(|coord| {
            format!(
                "{:.1}:{:.1}:{:.1}",
                coord[0] as f64, coord[1] as f64, coord[2] as f64
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    coord_buckets
}

fn example_shape_signature(example: &crate::data::MolecularExample) -> String {
    let atom_count = ligand_atom_count(example);
    let mut coord_buckets = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        let x = example.geometry.coords.double_value(&[index as i64, 0]) as f32;
        let y = example.geometry.coords.double_value(&[index as i64, 1]) as f32;
        let z = example.geometry.coords.double_value(&[index as i64, 2]) as f32;
        coord_buckets.push(format!("{:.1}:{:.1}:{:.1}", x as f64, y as f64, z as f64));
    }
    coord_buckets.join("|")
}

fn coord_distance(left: &[f32; 3], right: &[f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn build_comparison_summary(
    research: &ResearchConfig,
    ablation: &AblationConfig,
    unseen_protein_fraction: f64,
    topology_specialization_score: f64,
    geometry_specialization_score: f64,
    pocket_specialization_score: f64,
    slot_activation_mean: f64,
    gate_activation_mean: f64,
    leakage_proxy_mean: f64,
    metrics: &RealGenerationMetrics,
) -> GenerationQualitySummary {
    GenerationQualitySummary {
        primary_objective: primary_objective_label(
            ablation
                .primary_objective_override
                .unwrap_or(research.training.primary_objective),
        ),
        variant_label: ablation.variant_label.clone(),
        interaction_mode: interaction_mode_label(
            ablation
                .interaction_mode_override
                .unwrap_or(research.model.interaction_mode),
        ),
        candidate_valid_fraction: metric_value_with_heuristic_fallback(
            &metrics.chemistry_validity,
            "valid_fraction",
        ),
        pocket_contact_fraction: metric_value_with_heuristic_fallback(
            &metrics.docking_affinity,
            "pocket_contact_fraction",
        )
        .or_else(|| {
            metric_value_with_heuristic_fallback(&metrics.docking_affinity, "contact_fraction")
        }),
        pocket_compatibility_fraction: metric_value_with_heuristic_fallback(
            &metrics.pocket_compatibility,
            "centroid_inside_fraction",
        )
        .or_else(|| {
            metric_value_with_heuristic_fallback(
                &metrics.pocket_compatibility,
                "atom_coverage_fraction",
            )
        }),
        mean_centroid_offset: metric_value_with_heuristic_fallback(
            &metrics.docking_affinity,
            "mean_centroid_offset",
        ),
        strict_pocket_fit_score: metric_value_with_heuristic_fallback(
            &metrics.pocket_compatibility,
            "strict_pocket_fit_score",
        )
        .or_else(|| {
            let coverage = metric_value_with_heuristic_fallback(
                &metrics.pocket_compatibility,
                "atom_coverage_fraction",
            )?;
            let centroid_fit = metric_value_with_heuristic_fallback(
                &metrics.docking_affinity,
                "centroid_fit_score",
            )
            .or_else(|| {
                metric_value_with_heuristic_fallback(
                    &metrics.docking_affinity,
                    "mean_centroid_offset",
                )
                .map(|offset| 1.0 / (1.0 + offset))
            })?;
            Some(coverage * centroid_fit)
        }),
        unique_smiles_fraction: metric_value_with_heuristic_fallback(
            &metrics.chemistry_validity,
            "rdkit_unique_smiles_fraction",
        )
        .or_else(|| {
            metric_value_with_heuristic_fallback(
                &metrics.chemistry_validity,
                "unique_smiles_fraction",
            )
        }),
        unseen_protein_fraction,
        topology_specialization_score,
        geometry_specialization_score,
        pocket_specialization_score,
        slot_activation_mean,
        gate_activation_mean,
        leakage_proxy_mean,
    }
}

fn metric_value(metrics: &ReservedBackendMetrics, name: &str) -> Option<f64> {
    metrics.metrics.get(name).copied()
}

fn metric_value_with_heuristic_fallback(
    metrics: &ReservedBackendMetrics,
    name: &str,
) -> Option<f64> {
    metric_value(metrics, name).or_else(|| metric_value(metrics, &format!("heuristic_{name}")))
}

fn split_histograms(
    examples: &[crate::data::MolecularExample],
) -> (
    BTreeMap<String, usize>,
    BTreeMap<String, usize>,
    BTreeMap<String, usize>,
) {
    let mut ligand_bins = BTreeMap::new();
    let mut pocket_bins = BTreeMap::new();
    let mut measurements = BTreeMap::new();
    for example in examples {
        *ligand_bins
            .entry(atom_count_bin(ligand_atom_count(example)))
            .or_default() += 1;
        *pocket_bins
            .entry(atom_count_bin(pocket_atom_count(example)))
            .or_default() += 1;
        *measurements.entry(measurement_family(example)).or_default() += 1;
    }
    (ligand_bins, pocket_bins, measurements)
}

fn build_stratum_metrics(
    examples: &[crate::data::MolecularExample],
    train_proteins: &std::collections::BTreeSet<&str>,
) -> Vec<StratumEvaluationMetrics> {
    let mut strata = Vec::new();
    let mut axes: BTreeMap<(String, String), Vec<&crate::data::MolecularExample>> = BTreeMap::new();
    for example in examples {
        axes.entry((
            "ligand_atoms".to_string(),
            atom_count_bin(ligand_atom_count(example)),
        ))
        .or_default()
        .push(example);
        axes.entry((
            "pocket_atoms".to_string(),
            atom_count_bin(pocket_atom_count(example)),
        ))
        .or_default()
        .push(example);
        axes.entry(("measurement".to_string(), measurement_family(example)))
            .or_default()
            .push(example);
    }
    for ((axis, bin), bucket) in axes {
        let example_count = bucket.len();
        let labeled = bucket
            .iter()
            .filter(|example| example.targets.affinity_kcal_mol.is_some())
            .count();
        let unseen = bucket
            .iter()
            .filter(|example| !train_proteins.contains(example.protein_id.as_str()))
            .count();
        let ligand_atoms = bucket
            .iter()
            .map(|example| ligand_atom_count(example))
            .sum::<usize>();
        let pocket_atoms = bucket
            .iter()
            .map(|example| pocket_atom_count(example))
            .sum::<usize>();
        strata.push(StratumEvaluationMetrics {
            axis,
            bin,
            example_count,
            unseen_protein_fraction: fraction(unseen, example_count),
            labeled_fraction: fraction(labeled, example_count),
            average_ligand_atoms: fraction(ligand_atoms, example_count),
            average_pocket_atoms: fraction(pocket_atoms, example_count),
        });
    }
    strata
}

fn ligand_atom_count(example: &crate::data::MolecularExample) -> usize {
    example
        .topology
        .atom_types
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize
}

fn pocket_atom_count(example: &crate::data::MolecularExample) -> usize {
    example
        .pocket
        .coords
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize
}

fn average_ligand_atoms(examples: &[crate::data::MolecularExample]) -> f64 {
    fraction(
        examples.iter().map(ligand_atom_count).sum::<usize>(),
        examples.len(),
    )
}

fn average_pocket_atoms(examples: &[crate::data::MolecularExample]) -> f64 {
    fraction(
        examples.iter().map(pocket_atom_count).sum::<usize>(),
        examples.len(),
    )
}

fn measurement_family(example: &crate::data::MolecularExample) -> String {
    example
        .targets
        .affinity_measurement_type
        .as_deref()
        .unwrap_or("unknown")
        .to_string()
}

fn atom_count_bin(count: usize) -> String {
    match count {
        0 => "0".to_string(),
        1..=8 => "1-8".to_string(),
        9..=16 => "9-16".to_string(),
        17..=32 => "17-32".to_string(),
        33..=64 => "33-64".to_string(),
        65..=128 => "65-128".to_string(),
        129..=256 => "129-256".to_string(),
        _ => ">256".to_string(),
    }
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn primary_objective_label(config: crate::config::PrimaryObjectiveConfig) -> String {
    match config {
        crate::config::PrimaryObjectiveConfig::SurrogateReconstruction => {
            "surrogate_reconstruction".to_string()
        }
        crate::config::PrimaryObjectiveConfig::ConditionedDenoising => {
            "conditioned_denoising".to_string()
        }
    }
}

fn interaction_mode_label(mode: CrossAttentionMode) -> String {
    match mode {
        CrossAttentionMode::Lightweight => "lightweight".to_string(),
        CrossAttentionMode::Transformer => "transformer".to_string(),
    }
}

fn run_ablation_matrix(
    config: &UnseenPocketExperimentConfig,
) -> Result<AblationMatrixSummary, Box<dyn std::error::Error>> {
    let mut variants = Vec::new();
    for ablation in ablation_variants(config) {
        let mut variant_config = config.clone();
        variant_config.ablation = ablation.clone();
        variant_config.ablation_matrix.enabled = false;
        if let Some(primary) = ablation.primary_objective_override {
            variant_config.research.training.primary_objective = primary;
        }
        if let Some(mode) = ablation.interaction_mode_override {
            variant_config.research.model.interaction_mode = mode;
        }
        if let Some(label) = &ablation.variant_label {
            variant_config.research.training.checkpoint_dir = config
                .research
                .training
                .checkpoint_dir
                .join("ablations")
                .join(label);
        }
        let summary = UnseenPocketExperiment::run_with_options(variant_config, false)?;
        variants.push(AblationRunSummary {
            variant_label: ablation
                .variant_label
                .clone()
                .unwrap_or_else(|| "unnamed_variant".to_string()),
            validation: summary.validation.comparison_summary,
            test: summary.test.comparison_summary,
        });
    }

    Ok(AblationMatrixSummary {
        artifact_dir: config.research.training.checkpoint_dir.join("ablations"),
        variants,
    })
}

fn ablation_variants(config: &UnseenPocketExperimentConfig) -> Vec<AblationConfig> {
    let mut variants = Vec::new();
    if config.ablation_matrix.include_surrogate_objective
        && config.research.training.primary_objective
            != crate::config::PrimaryObjectiveConfig::SurrogateReconstruction
    {
        variants.push(AblationConfig {
            primary_objective_override: Some(
                crate::config::PrimaryObjectiveConfig::SurrogateReconstruction,
            ),
            variant_label: Some("objective_surrogate".to_string()),
            ..AblationConfig::default()
        });
    }
    if config.ablation_matrix.include_conditioned_denoising
        && config.research.training.primary_objective
            != crate::config::PrimaryObjectiveConfig::ConditionedDenoising
    {
        variants.push(AblationConfig {
            primary_objective_override: Some(
                crate::config::PrimaryObjectiveConfig::ConditionedDenoising,
            ),
            variant_label: Some("objective_conditioned_denoising".to_string()),
            ..AblationConfig::default()
        });
    }
    if config.ablation_matrix.include_disable_slots {
        variants.push(AblationConfig {
            disable_slots: true,
            variant_label: Some("disable_slots".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_disable_cross_attention {
        variants.push(AblationConfig {
            disable_cross_attention: true,
            variant_label: Some("disable_cross_attention".to_string()),
            ..config.ablation.clone()
        });
    }
    if config
        .ablation_matrix
        .include_disable_geometry_interaction_bias
    {
        variants.push(AblationConfig {
            disable_geometry_interaction_bias: true,
            variant_label: Some("disable_geometry_interaction_bias".to_string()),
            ..config.ablation.clone()
        });
    }
    if config
        .ablation_matrix
        .include_disable_rollout_pocket_guidance
    {
        variants.push(AblationConfig {
            disable_rollout_pocket_guidance: true,
            variant_label: Some("disable_rollout_pocket_guidance".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_disable_candidate_repair {
        variants.push(AblationConfig {
            disable_candidate_repair: true,
            variant_label: Some("disable_candidate_repair".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_disable_probes {
        variants.push(AblationConfig {
            disable_probes: true,
            variant_label: Some("disable_probes".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_lightweight_interaction
        && config.research.model.interaction_mode != CrossAttentionMode::Lightweight
    {
        variants.push(AblationConfig {
            interaction_mode_override: Some(CrossAttentionMode::Lightweight),
            variant_label: Some("interaction_lightweight".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_transformer_interaction
        && config.research.model.interaction_mode != CrossAttentionMode::Transformer
    {
        variants.push(AblationConfig {
            interaction_mode_override: Some(CrossAttentionMode::Transformer),
            variant_label: Some("interaction_transformer".to_string()),
            ..config.ablation.clone()
        });
    }
    variants
}

fn persist_ablation_matrix(
    checkpoint_dir: &std::path::Path,
    matrix: &AblationMatrixSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(checkpoint_dir)?;
    fs::write(
        checkpoint_dir.join("ablation_matrix_summary.json"),
        serde_json::to_string_pretty(matrix)?,
    )?;
    Ok(())
}

fn build_claim_report(summary: &UnseenPocketExperimentSummary) -> ClaimReport {
    let ablation_deltas = summary
        .ablation_matrix
        .as_ref()
        .map(|matrix| {
            matrix
                .variants
                .iter()
                .map(|variant| ClaimDeltaSummary {
                    variant_label: variant.variant_label.clone(),
                    candidate_valid_fraction_delta: subtract_optional(
                        variant.test.candidate_valid_fraction,
                        summary.test.comparison_summary.candidate_valid_fraction,
                    ),
                    pocket_contact_fraction_delta: subtract_optional(
                        variant.test.pocket_contact_fraction,
                        summary.test.comparison_summary.pocket_contact_fraction,
                    ),
                    pocket_compatibility_fraction_delta: subtract_optional(
                        variant.test.pocket_compatibility_fraction,
                        summary
                            .test
                            .comparison_summary
                            .pocket_compatibility_fraction,
                    ),
                    mean_centroid_offset_delta: subtract_optional(
                        variant.test.mean_centroid_offset,
                        summary.test.comparison_summary.mean_centroid_offset,
                    ),
                    strict_pocket_fit_score_delta: subtract_optional(
                        variant.test.strict_pocket_fit_score,
                        summary.test.comparison_summary.strict_pocket_fit_score,
                    ),
                    unique_smiles_fraction_delta: subtract_optional(
                        variant.test.unique_smiles_fraction,
                        summary.test.comparison_summary.unique_smiles_fraction,
                    ),
                    topology_specialization_score_delta: variant.test.topology_specialization_score
                        - summary
                            .test
                            .comparison_summary
                            .topology_specialization_score,
                    geometry_specialization_score_delta: variant.test.geometry_specialization_score
                        - summary
                            .test
                            .comparison_summary
                            .geometry_specialization_score,
                    pocket_specialization_score_delta: variant.test.pocket_specialization_score
                        - summary.test.comparison_summary.pocket_specialization_score,
                    slot_activation_mean_delta: variant.test.slot_activation_mean
                        - summary.test.comparison_summary.slot_activation_mean,
                    gate_activation_mean_delta: variant.test.gate_activation_mean
                        - summary.test.comparison_summary.gate_activation_mean,
                    leakage_proxy_mean_delta: variant.test.leakage_proxy_mean
                        - summary.test.comparison_summary.leakage_proxy_mean,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    ClaimReport {
        artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
        run_label: summary
            .config
            .ablation
            .variant_label
            .clone()
            .unwrap_or_else(|| "base_run".to_string()),
        validation: summary.validation.comparison_summary.clone(),
        test: summary.test.comparison_summary.clone(),
        backend_metrics: summary.test.real_generation_metrics.clone(),
        backend_thresholds: build_backend_thresholds(summary),
        backend_review: build_backend_review(summary),
        layered_generation_metrics: summary.test.layered_generation_metrics.clone(),
        chemistry_novelty_diversity: build_chemistry_novelty_diversity(summary),
        claim_context: build_claim_context(summary),
        backend_environment: Some(build_backend_environment_report(summary)),
        reranker_report: build_reranker_report(&summary.test.layered_generation_metrics),
        slot_stability: summary.test.slot_stability.clone(),
        leakage_calibration: build_leakage_calibration_report(summary, &ablation_deltas),
        performance_gates: summary.performance_gates.clone(),
        baseline_comparisons: build_baseline_comparisons(summary),
        ablation_deltas,
    }
}

fn backend_threshold_check(
    value: Option<f64>,
    threshold: f64,
    direction: &str,
) -> BackendThresholdCheck {
    let passed = match (value, direction) {
        (Some(observed), "min") => observed >= threshold,
        (Some(observed), "max") => observed <= threshold,
        _ => false,
    };
    BackendThresholdCheck {
        value,
        threshold,
        passed,
        direction: direction.to_string(),
    }
}

fn build_backend_thresholds(
    summary: &UnseenPocketExperimentSummary,
) -> BTreeMap<String, BackendThresholdCheck> {
    let chemistry = &summary.test.real_generation_metrics.chemistry_validity;
    let docking = &summary.test.real_generation_metrics.docking_affinity;
    let pocket = &summary.test.real_generation_metrics.pocket_compatibility;
    let strict_fit = metric_value_with_heuristic_fallback(pocket, "strict_pocket_fit_score");
    let pocket_contact = metric_value_with_heuristic_fallback(docking, "pocket_contact_fraction")
        .or_else(|| metric_value_with_heuristic_fallback(docking, "contact_fraction"));
    BTreeMap::from([
        (
            "rdkit_available".to_string(),
            backend_threshold_check(backend_metric(chemistry, "rdkit_available"), 1.0, "min"),
        ),
        (
            "rdkit_sanitized_fraction".to_string(),
            backend_threshold_check(
                backend_metric(chemistry, "rdkit_sanitized_fraction"),
                0.95,
                "min",
            ),
        ),
        (
            "rdkit_unique_smiles_fraction".to_string(),
            backend_threshold_check(
                backend_metric(chemistry, "rdkit_unique_smiles_fraction").or_else(|| {
                    metric_value_with_heuristic_fallback(chemistry, "unique_smiles_fraction")
                }),
                0.5,
                "min",
            ),
        ),
        (
            "backend_missing_structure_fraction".to_string(),
            backend_threshold_check(
                backend_metric(pocket, "backend_missing_structure_fraction"),
                0.0,
                "max",
            ),
        ),
        (
            "clash_fraction".to_string(),
            backend_threshold_check(
                metric_value_with_heuristic_fallback(pocket, "clash_fraction"),
                0.1,
                "max",
            ),
        ),
        (
            "strict_pocket_fit_score".to_string(),
            backend_threshold_check(strict_fit, 0.35, "min"),
        ),
        (
            "pocket_contact_fraction".to_string(),
            backend_threshold_check(pocket_contact, 0.8, "min"),
        ),
    ])
}

fn is_vina_backend_surface(summary: &UnseenPocketExperimentSummary) -> bool {
    summary
        .config
        .external_evaluation
        .docking_backend
        .args
        .iter()
        .any(|arg| arg.contains("vina_score_backend.py"))
}

fn build_backend_review(summary: &UnseenPocketExperimentSummary) -> BackendReviewReport {
    let backend_thresholds = build_backend_thresholds(summary);
    let chemistry_ready = [
        "rdkit_available",
        "rdkit_sanitized_fraction",
        "rdkit_unique_smiles_fraction",
    ]
    .iter()
    .all(|name| {
        backend_thresholds
            .get(*name)
            .map(|result| result.passed)
            .unwrap_or(false)
    });
    let pocket_ready = [
        "backend_missing_structure_fraction",
        "clash_fraction",
        "strict_pocket_fit_score",
        "pocket_contact_fraction",
    ]
    .iter()
    .all(|name| {
        backend_thresholds
            .get(*name)
            .map(|result| result.passed)
            .unwrap_or(false)
    });
    if !is_vina_backend_surface(summary) {
        return BackendReviewReport {
            policy_label: "repository_real_backend_policy".to_string(),
            reviewer_status: if chemistry_ready && pocket_ready {
                "pass".to_string()
            } else {
                "fail".to_string()
            },
            reviewer_passed: chemistry_ready && pocket_ready,
            claim_bearing_surface: claim_is_real_backend_backed(summary),
            claim_bearing_ready: chemistry_ready && pocket_ready,
            claim_bearing_requirements: vec![
                "Keep chemistry validity, pocket compatibility, clash, and strict pocket-fit metrics above the shared backend thresholds.".to_string(),
                "Keep backend_missing_structure_fraction at or below 0.0 on claim-bearing reviewer surfaces.".to_string(),
            ],
            reviewer_reasons: Vec::new(),
            chemistry_validity_ready: chemistry_ready,
            docking_backend_available: true,
            docking_input_completeness_fraction: None,
            docking_score_coverage_fraction: None,
        };
    }

    let docking = &summary.test.real_generation_metrics.docking_affinity;
    let vina_available = backend_metric(docking, "vina_available").unwrap_or(0.0) >= 1.0;
    let docking_input_completeness_fraction =
        backend_metric(docking, "candidate_with_complete_vina_inputs_fraction");
    let docking_score_coverage_fraction = backend_metric(docking, "vina_score_success_fraction")
        .or_else(|| backend_metric(docking, "backend_examples_scored"));
    let docking_ready = vina_available
        && docking_input_completeness_fraction
            .map(|value| value >= 1.0)
            .unwrap_or(false)
        && docking_score_coverage_fraction
            .map(|value| value >= 1.0)
            .unwrap_or(false);

    let mut reviewer_reasons = Vec::new();
    if !chemistry_ready {
        reviewer_reasons.push(
            "chemistry validity backend does not yet clear the shared RDKit availability, sanitization, or uniqueness thresholds".to_string(),
        );
    }
    if !pocket_ready {
        reviewer_reasons.push(
            "pocket compatibility metrics do not yet clear the shared missing-structure, clash, strict-fit, or contact thresholds".to_string(),
        );
    }
    if !vina_available {
        reviewer_reasons.push(
            "AutoDock Vina is unavailable on this machine, so the stronger docking companion cannot become claim-bearing".to_string(),
        );
    }
    if docking_input_completeness_fraction
        .map(|value| value < 1.0)
        .unwrap_or(true)
    {
        reviewer_reasons.push(
            "not all candidates provide complete Vina-ready receptor/ligand PDBQT inputs"
                .to_string(),
        );
    }
    if docking_score_coverage_fraction
        .map(|value| value < 1.0)
        .unwrap_or(true)
    {
        reviewer_reasons.push(
            "the stronger docking backend did not score every reviewed candidate".to_string(),
        );
    }

    let claim_bearing_ready = chemistry_ready && pocket_ready && docking_ready;
    BackendReviewReport {
        policy_label: "vina_claim_bearing_companion_policy".to_string(),
        reviewer_status: if claim_bearing_ready {
            "pass".to_string()
        } else {
            "fail".to_string()
        },
        reviewer_passed: claim_bearing_ready,
        claim_bearing_surface: true,
        claim_bearing_ready,
        claim_bearing_requirements: vec![
            "Chemistry validity must clear the shared RDKit availability, sanitization, and uniqueness thresholds.".to_string(),
            "Pocket compatibility must clear the shared missing-structure, clash, strict pocket-fit, and pocket-contact thresholds.".to_string(),
            "AutoDock Vina must be available and every reviewed candidate must provide Vina-ready receptor and ligand PDBQT inputs.".to_string(),
            "The stronger docking backend must score every reviewed candidate before this surface is described as claim-bearing backend evidence.".to_string(),
        ],
        reviewer_reasons,
        chemistry_validity_ready: chemistry_ready,
        docking_backend_available: vina_available,
        docking_input_completeness_fraction,
        docking_score_coverage_fraction,
    }
}

fn build_chemistry_novelty_diversity(
    summary: &UnseenPocketExperimentSummary,
) -> ChemistryNoveltyDiversitySummary {
    let (review_layer, layer) = if summary
        .test
        .layered_generation_metrics
        .reranked_candidates
        .candidate_count
        > 0
    {
        (
            "reranked_candidates",
            &summary.test.layered_generation_metrics.reranked_candidates,
        )
    } else {
        (
            "inferred_bond_candidates",
            &summary
                .test
                .layered_generation_metrics
                .inferred_bond_candidates,
        )
    };
    ChemistryNoveltyDiversitySummary {
        review_layer: review_layer.to_string(),
        unique_smiles_fraction: summary.test.comparison_summary.unique_smiles_fraction,
        atom_type_sequence_diversity: layer.atom_type_sequence_diversity,
        bond_topology_diversity: layer.bond_topology_diversity,
        coordinate_shape_diversity: layer.coordinate_shape_diversity,
        novel_atom_type_sequence_fraction: layer.novel_atom_type_sequence_fraction,
        novel_bond_topology_fraction: layer.novel_bond_topology_fraction,
        novel_coordinate_shape_fraction: layer.novel_coordinate_shape_fraction,
        interpretation: format!(
            "Review chemistry novelty/diversity on `{review_layer}`; novelty is measured against training-reference structural signatures instead of relying only on within-layer uniqueness."
        ),
        benchmark_evidence: build_chemistry_benchmark_evidence(summary),
    }
}

fn build_claim_context(summary: &UnseenPocketExperimentSummary) -> ClaimContext {
    let real_backend_backed = claim_is_real_backend_backed(summary);
    let evidence_mode = if real_backend_backed {
        "real-backend-backed held-out pocket evidence".to_string()
    } else {
        "heuristic-only held-out pocket evidence".to_string()
    };
    ClaimContext {
        surface_label: summary.config.surface_label.clone(),
        real_backend_backed,
        evidence_mode,
    }
}

fn claim_is_real_backend_backed(summary: &UnseenPocketExperimentSummary) -> bool {
    backend_config_enabled(&summary.config.external_evaluation.chemistry_backend)
        && backend_config_enabled(&summary.config.external_evaluation.pocket_backend)
}

fn backend_config_enabled(config: &ExternalBackendCommandConfig) -> bool {
    config.enabled
        && config
            .executable
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
}

fn build_chemistry_benchmark_evidence(
    summary: &UnseenPocketExperimentSummary,
) -> ChemistryBenchmarkEvidence {
    let chemistry = &summary.test.real_generation_metrics.chemistry_validity;
    let layer = if summary
        .test
        .layered_generation_metrics
        .reranked_candidates
        .candidate_count
        > 0
    {
        &summary.test.layered_generation_metrics.reranked_candidates
    } else {
        &summary
            .test
            .layered_generation_metrics
            .inferred_bond_candidates
    };
    let sanitized_fraction = chemistry.metrics.get("rdkit_sanitized_fraction").copied();
    let parseable_fraction = chemistry.metrics.get("rdkit_parseable_fraction").copied();
    let finite_conformer_fraction = chemistry
        .metrics
        .get("rdkit_finite_conformer_fraction")
        .copied();
    let unique_smiles_fraction = chemistry
        .metrics
        .get("rdkit_unique_smiles_fraction")
        .copied()
        .or(summary.test.comparison_summary.unique_smiles_fraction);
    let validity_quality_score = match (sanitized_fraction, unique_smiles_fraction) {
        (Some(sanitized), Some(unique)) => Some((sanitized + unique) / 2.0),
        (Some(sanitized), None) => Some(sanitized),
        (None, Some(unique)) => Some(unique),
        (None, None) => None,
    };
    let novelty_diversity_score = [
        layer.atom_type_sequence_diversity,
        layer.bond_topology_diversity,
        layer.coordinate_shape_diversity,
        layer.novel_atom_type_sequence_fraction,
        layer.novel_bond_topology_fraction,
        layer.novel_coordinate_shape_fraction,
    ]
    .into_iter()
    .sum::<f64>()
        / 6.0;
    let backend_backed = claim_is_real_backend_backed(summary) && sanitized_fraction.is_some();
    let stronger_candidate_threshold = 8;
    let stronger_required_backend_metrics = vec![
        "rdkit_parseable_fraction".to_string(),
        "rdkit_finite_conformer_fraction".to_string(),
        "rdkit_sanitized_fraction".to_string(),
        "rdkit_unique_smiles_fraction".to_string(),
    ];
    let stronger_checks = [
        parseable_fraction.map(|value| value >= 0.95),
        finite_conformer_fraction.map(|value| value >= 0.95),
        sanitized_fraction.map(|value| value >= 0.95),
        unique_smiles_fraction.map(|value| value >= 0.5),
        Some(layer.candidate_count >= stronger_candidate_threshold),
        Some(novelty_diversity_score >= 0.75),
    ];
    let stronger_check_count = stronger_checks.len();
    let stronger_passed = stronger_checks
        .iter()
        .all(|result| matches!(result, Some(true)));
    let stronger_support_score = Some(
        stronger_checks
            .iter()
            .filter(|result| matches!(result, Some(true)))
            .count() as f64
            / stronger_check_count as f64,
    );
    let val_family_count = summary
        .split_report
        .val
        .protein_family_proxy_histogram
        .len();
    let test_family_count = summary
        .split_report
        .test
        .protein_family_proxy_histogram
        .len();
    let parsed_examples = summary.dataset_validation.parsed_examples;
    let retained_label_coverage = summary.dataset_validation.retained_label_coverage;
    let surface_label = summary
        .config
        .surface_label
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let configured_external_benchmark = summary
        .config
        .reviewer_benchmark
        .dataset
        .clone()
        .or_else(|| {
            if surface_label.contains("pdbbindpp")
                || summary
                    .config
                    .research
                    .training
                    .checkpoint_dir
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .contains("pdbbindpp")
            {
                Some("pdbbindpp-2020".to_string())
            } else {
                None
            }
        });
    let external_benchmark_label = configured_external_benchmark
        .clone()
        .unwrap_or_else(|| "configured_external_benchmark".to_string());
    let external_required_checks = vec![
        format!(
            "reviewer_benchmark.dataset is configured for {}",
            external_benchmark_label
        ),
        "dataset_validation.parsed_examples >= 100".to_string(),
        "dataset_validation.retained_label_coverage >= 0.8".to_string(),
        "val protein-family count >= 10".to_string(),
        "test protein-family count >= 10".to_string(),
        "reviewer benchmark-plus chemistry gate already passed".to_string(),
    ];
    let external_checks = [
        configured_external_benchmark.is_some(),
        parsed_examples >= 100,
        retained_label_coverage >= 0.8,
        val_family_count >= 10,
        test_family_count >= 10,
        stronger_passed,
    ];
    let external_benchmark_backed = external_checks.iter().all(|passed| *passed);
    let external_benchmark_support_score = Some(
        external_checks.iter().filter(|passed| **passed).count() as f64
            / external_checks.len() as f64,
    );
    let benchmark_components = if backend_backed {
        let mut components = vec![
            "rdkit_sanitized_fraction".to_string(),
            "rdkit_unique_smiles_fraction".to_string(),
            "atom_type_sequence_diversity".to_string(),
            "bond_topology_diversity".to_string(),
            "coordinate_shape_diversity".to_string(),
            "novel_atom_type_sequence_fraction".to_string(),
            "novel_bond_topology_fraction".to_string(),
            "novel_coordinate_shape_fraction".to_string(),
        ];
        if stronger_passed {
            components.extend([
                "rdkit_parseable_fraction".to_string(),
                "rdkit_finite_conformer_fraction".to_string(),
                "review_candidate_count".to_string(),
                "stronger_benchmark_support_score".to_string(),
            ]);
        }
        if external_benchmark_backed {
            components.extend([
                "external_benchmark_dataset".to_string(),
                "parsed_examples".to_string(),
                "retained_label_coverage".to_string(),
                "val_protein_family_count".to_string(),
                "test_protein_family_count".to_string(),
                "external_benchmark_support_score".to_string(),
            ]);
        }
        components
    } else {
        vec![
            "atom_type_sequence_diversity".to_string(),
            "bond_topology_diversity".to_string(),
            "coordinate_shape_diversity".to_string(),
            "novel_atom_type_sequence_fraction".to_string(),
            "novel_bond_topology_fraction".to_string(),
            "novel_coordinate_shape_fraction".to_string(),
        ]
    };
    let evidence_tier = if backend_backed && external_benchmark_backed {
        "external_benchmark_backed".to_string()
    } else if backend_backed && stronger_passed {
        "reviewer_benchmark_plus".to_string()
    } else if backend_backed {
        "local_benchmark_style".to_string()
    } else {
        "proxy_only".to_string()
    };
    let stronger_benchmark_note = if !backend_backed {
        "Stronger reviewer chemistry evidence is unavailable without an active backend-backed chemistry surface.".to_string()
    } else if stronger_passed {
        format!(
            "This surface clears the stronger reviewer chemistry gate because parseable, finite-conformer, sanitized, and unique-SMILES fractions all pass with review_candidate_count={} and novelty_diversity_score={:.4}.",
            layer.candidate_count,
            novelty_diversity_score,
        )
    } else {
        format!(
            "This surface stays at local benchmark-style chemistry evidence because the stronger reviewer gate is only {:.3} supported; it requires parseable, finite-conformer, sanitized, and unique-SMILES backend quality plus at least {} review-layer candidates and novelty_diversity_score >= 0.75.",
            stronger_support_score.unwrap_or(0.0),
            stronger_candidate_threshold,
        )
    };
    let external_benchmark_note = if !backend_backed {
        "External benchmark-backed chemistry evidence is unavailable without an active backend-backed chemistry surface.".to_string()
    } else if external_benchmark_backed {
        format!(
            "This surface clears the explicit external benchmark-dataset chemistry tier for {} with parsed_examples={}, retained_label_coverage={:.4}, val_family_count={}, test_family_count={}, and reviewer benchmark-plus chemistry already passing.",
            external_benchmark_label,
            parsed_examples,
            retained_label_coverage,
            val_family_count,
            test_family_count,
        )
    } else {
        format!(
            "This surface does not yet clear the explicit external benchmark-dataset chemistry tier; support is {:.3} and requires a configured benchmark dataset label, parsed_examples>=100, retained_label_coverage>=0.8, held-out family counts>=10 on validation/test, and reviewer benchmark-plus chemistry already passing.",
            external_benchmark_support_score.unwrap_or(0.0),
        )
    };
    let interpretation = if backend_backed && external_benchmark_backed {
        format!(
            "Combines backend-backed chemistry quality, held-out-pocket novelty/diversity aggregates, reviewer benchmark-plus checks, and an explicit external benchmark-dataset layer on {} (validity_quality_score={:.4}, novelty_diversity_score={:.4}, reviewer_support_score={:.4}, external_support_score={:.4}).",
            external_benchmark_label,
            validity_quality_score.unwrap_or(0.0),
            novelty_diversity_score,
            stronger_support_score.unwrap_or(0.0),
            external_benchmark_support_score.unwrap_or(0.0),
        )
    } else if backend_backed && stronger_passed {
        format!(
            "Combines backend-backed chemistry quality, held-out-pocket novelty/diversity aggregates, and explicit reviewer benchmark checks (validity_quality_score={:.4}, novelty_diversity_score={:.4}, support_score={:.4}) for a stronger reviewer benchmark-plus chemistry summary.",
            validity_quality_score.unwrap_or(0.0),
            novelty_diversity_score,
            stronger_support_score.unwrap_or(0.0),
        )
    } else if backend_backed {
        format!(
            "Combines backend-measured sanitization and unique-SMILES chemistry quality with held-out-pocket novelty/diversity aggregates (validity_quality_score={:.4}, novelty_diversity_score={:.4}) for a local benchmark-style chemistry summary.",
            validity_quality_score.unwrap_or(0.0),
            novelty_diversity_score,
        )
    } else {
        format!(
            "No active chemistry backend was attached, so chemistry evidence remains proxy-only; the novelty/diversity aggregate ({:.4}) is structural-signature-based rather than backend benchmark-backed.",
            novelty_diversity_score,
        )
    };
    ChemistryBenchmarkEvidence {
        backend_backed,
        sanitized_fraction,
        unique_smiles_fraction,
        review_candidate_count: layer.candidate_count,
        validity_quality_score,
        novelty_diversity_score,
        evidence_tier,
        stronger_reviewer_benchmark: stronger_passed,
        external_benchmark_backed,
        external_benchmark_dataset: if external_benchmark_backed {
            configured_external_benchmark
        } else {
            None
        },
        stronger_review_candidate_threshold: stronger_candidate_threshold,
        stronger_required_backend_metrics,
        stronger_benchmark_support_score: stronger_support_score,
        external_benchmark_support_score,
        stronger_benchmark_note,
        external_required_checks,
        external_benchmark_note,
        benchmark_components,
        interpretation,
    }
}

fn build_backend_environment_report(
    summary: &UnseenPocketExperimentSummary,
) -> BackendEnvironmentReport {
    let config = &summary.config.external_evaluation;
    let real_backend_backed = claim_is_real_backend_backed(summary);
    let fingerprint_source = (
        &config.chemistry_backend,
        &config.docking_backend,
        &config.pocket_backend,
    );
    BackendEnvironmentReport {
        config_fingerprint: stable_json_hash(&fingerprint_source),
        real_backend_backed,
        prerequisites: vec![
            "Use the canonical experiment config that enables the external chemistry, docking, and pocket backends.".to_string(),
            "Run `python3 tools/reviewer_env_check.py --config <experiment-config>` before reviewer revalidation on a fresh machine.".to_string(),
            "Ensure `python3` can execute `tools/rdkit_validity_backend.py` and `tools/pocket_contact_backend.py` from the repository root.".to_string(),
            "Keep source protein and ligand structure provenance available in the configured dataset so backend scoring does not degrade into missing-structure examples.".to_string(),
        ],
        chemistry_backend: build_backend_command_report(
            "chemistry_validity",
            &config.chemistry_backend,
            &summary.test.real_generation_metrics.chemistry_validity,
        ),
        docking_backend: build_backend_command_report(
            "docking_affinity",
            &config.docking_backend,
            &summary.test.real_generation_metrics.docking_affinity,
        ),
        pocket_backend: build_backend_command_report(
            "pocket_compatibility",
            &config.pocket_backend,
            &summary.test.real_generation_metrics.pocket_compatibility,
        ),
    }
}

fn build_backend_command_report(
    logical_name: &str,
    config: &ExternalBackendCommandConfig,
    metrics: &ReservedBackendMetrics,
) -> BackendCommandReport {
    let command_identity = (&config.executable, &config.args);
    let backend_examples_scored = metrics.metrics.get("backend_examples_scored").copied();
    let schema_version = metrics.metrics.get("schema_version").copied();
    let spawn_failed = metrics
        .metrics
        .get("backend_command_spawn_error")
        .copied()
        .unwrap_or(0.0)
        > 0.0;
    let command_failed = metrics
        .metrics
        .get("backend_command_failed")
        .copied()
        .unwrap_or(0.0)
        > 0.0;
    let runtime_available = backend_config_enabled(config)
        && !spawn_failed
        && !command_failed
        && backend_examples_scored.unwrap_or(0.0) > 0.0;
    BackendCommandReport {
        logical_name: logical_name.to_string(),
        enabled: config.enabled,
        executable: config.executable.clone(),
        args: config.args.clone(),
        command_fingerprint: stable_json_hash(&command_identity),
        runtime_available,
        backend_name: metrics.backend_name.clone(),
        status: metrics.status.clone(),
        schema_version,
        backend_examples_scored,
    }
}

fn build_baseline_comparisons(
    summary: &UnseenPocketExperimentSummary,
) -> Vec<BaselineComparisonRow> {
    let layers = &summary.test.layered_generation_metrics;
    let mut rows = vec![
        BaselineComparisonRow {
            label: "heuristic_raw_rollout_no_repair".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.raw_rollout.clone()),
            test_summary: None,
            interpretation:
                "Raw decoder rollout before geometry repair, bond inference, or reranking."
                    .to_string(),
        },
        BaselineComparisonRow {
            label: "pocket_centroid_repair_proxy".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.repaired_candidates.clone()),
            test_summary: None,
            interpretation:
                "Geometry-repaired candidates expose how much pocket-centroid postprocessing helps."
                    .to_string(),
        },
        BaselineComparisonRow {
            label: "deterministic_proxy_reranker".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.deterministic_proxy_candidates.clone()),
            test_summary: None,
            interpretation:
                "Current deterministic selection proxy before learned reranker calibration."
                    .to_string(),
        },
        BaselineComparisonRow {
            label: "calibrated_reranker".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.reranked_candidates.clone()),
            test_summary: None,
            interpretation:
                "Active bounded calibrated reranker used on the claim-bearing selection path."
                    .to_string(),
        },
    ];

    if let Some(matrix) = &summary.ablation_matrix {
        for wanted in [
            "objective_surrogate",
            "disable_slots",
            "disable_rollout_pocket_guidance",
            "disable_cross_attention",
        ] {
            if let Some(variant) = matrix
                .variants
                .iter()
                .find(|variant| variant.variant_label == wanted)
            {
                rows.push(BaselineComparisonRow {
                    label: wanted.to_string(),
                    source: "ablation_matrix".to_string(),
                    candidate_layer: None,
                    test_summary: Some(variant.test.clone()),
                    interpretation: match wanted {
                        "objective_surrogate" => {
                            "Surrogate reconstruction objective control.".to_string()
                        }
                        "disable_slots" => "No-slot controlled-interaction control.".to_string(),
                        "disable_rollout_pocket_guidance" => {
                            "No rollout-time pocket-guidance control.".to_string()
                        }
                        "disable_cross_attention" => {
                            "No cross-modal interaction control, not a replacement architecture."
                                .to_string()
                        }
                        _ => "Ablation control.".to_string(),
                    },
                });
            }
        }
    }
    rows
}

fn subtract_optional(candidate: Option<f64>, baseline: Option<f64>) -> Option<f64> {
    candidate
        .zip(baseline)
        .map(|(candidate, baseline)| candidate - baseline)
}

fn build_reranker_report(metrics: &LayeredGenerationMetrics) -> RerankerReport {
    let baseline = metrics.inferred_bond_candidates.clone();
    let reranked = metrics.reranked_candidates.clone();
    let deterministic_proxy = metrics.deterministic_proxy_candidates.clone();
    let calibration = metrics.reranker_calibration.clone();
    let validity_delta = reranked.valid_fraction - baseline.valid_fraction;
    let pocket_delta = reranked.pocket_contact_fraction - baseline.pocket_contact_fraction;
    let clash_delta = baseline.clash_fraction - reranked.clash_fraction;
    let decision = if validity_delta >= -1e-6 && pocket_delta >= -1e-6 && clash_delta >= -1e-6 {
        "bounded calibrated reranking is sufficient to keep adversarial training out of the mainline for this surface; confirm coefficients on larger backend-scored held-out pockets".to_string()
    } else {
        "calibrated reranking did not dominate deterministic selection; expand backend-scored calibration evidence before considering adversarial training".to_string()
    };
    RerankerReport {
        baseline,
        deterministic_proxy,
        reranked,
        calibration,
        decision,
    }
}

fn build_leakage_calibration_report(
    summary: &UnseenPocketExperimentSummary,
    deltas: &[ClaimDeltaSummary],
) -> LeakageCalibrationReport {
    let preferred = default_preferred_leakage_proxy_threshold();
    let hard = default_hard_leakage_proxy_threshold();
    let regression_limit = default_max_leakage_regression_threshold();
    let test_leakage = summary.test.comparison_summary.leakage_proxy_mean;
    let split_checks = &summary.split_report.leakage_checks;
    let max_regression = deltas
        .iter()
        .map(|delta| delta.leakage_proxy_mean_delta)
        .fold(0.0_f64, f64::max);
    let blockers = deltas
        .iter()
        .filter(|delta| {
            delta
                .strict_pocket_fit_score_delta
                .is_some_and(|value| value < -0.05)
                || delta
                    .candidate_valid_fraction_delta
                    .is_some_and(|value| value < -0.05)
        })
        .count();
    let mut reasons = Vec::new();
    if test_leakage > hard {
        reasons.push(format!(
            "test leakage proxy {:.4} exceeds the hard reviewer bound {:.4}",
            test_leakage, hard
        ));
    } else if test_leakage > preferred {
        reasons.push(format!(
            "test leakage proxy {:.4} exceeds the preferred reviewer bound {:.4}",
            test_leakage, preferred
        ));
    }
    if max_regression > regression_limit {
        reasons.push(format!(
            "worst reviewed leakage regression {:.4} exceeds the allowed delta {:.4}",
            max_regression, regression_limit
        ));
    }
    if split_checks.protein_overlap_detected || split_checks.duplicate_example_ids_detected {
        reasons.push(
            "split leakage audit detected cross-split overlap or duplicated example identifiers"
                .to_string(),
        );
    }

    let (reviewer_status, reviewer_passed) = if test_leakage > hard
        || split_checks.protein_overlap_detected
        || split_checks.duplicate_example_ids_detected
    {
        ("fail".to_string(), false)
    } else if test_leakage > preferred || max_regression > regression_limit {
        ("caution".to_string(), true)
    } else {
        ("pass".to_string(), true)
    };
    let decision = if blockers == 0 && reviewer_status == "pass" {
        "current leakage weight passes reviewer bounds and preserves physically necessary cross-modality dependence on reviewed variants".to_string()
    } else if blockers == 0 && reviewer_status == "caution" {
        "current leakage weight remains usable, but reviewed ablations or the base run sit above the preferred reviewer band; keep leakage interpretation explicit in claim-facing notes".to_string()
    } else if reviewer_status == "caution" {
        format!(
            "{blockers} reviewed variant(s) regress pocket fit or chemistry; base leakage remains bounded, so keep claim wording cautious and cite the ablation deltas explicitly"
        )
    } else {
        format!(
            "{blockers} reviewed variant(s) regress pocket fit or chemistry; keep leakage at or below the current default until a sweep clears these blockers"
        )
    };
    LeakageCalibrationReport {
        recommended_delta_leak: summary.config.research.training.loss_weights.delta_leak,
        evaluated_variants: deltas.len(),
        preferred_max_leakage_proxy_mean: preferred,
        hard_max_leakage_proxy_mean: hard,
        max_leakage_proxy_regression: regression_limit,
        reviewer_status,
        reviewer_passed,
        reviewer_reasons: reasons,
        decision,
    }
}

fn measurement_breakdown(
    labeled_examples: &[(&crate::data::MolecularExample, &ResearchForward)],
) -> Vec<MeasurementMetrics> {
    let mut grouped: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for (example, forward) in labeled_examples {
        let measurement = example
            .targets
            .affinity_measurement_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        grouped.entry(measurement).or_default().push((
            example.targets.affinity_kcal_mol.unwrap() as f64,
            forward.probes.affinity_prediction.double_value(&[]),
        ));
    }

    grouped
        .into_iter()
        .map(|(measurement_type, pairs)| {
            let count = pairs.len();
            let mae = pairs
                .iter()
                .map(|(target, pred)| (pred - target).abs())
                .sum::<f64>()
                / count as f64;
            let rmse = (pairs
                .iter()
                .map(|(target, pred)| {
                    let error = pred - target;
                    error * error
                })
                .sum::<f64>()
                / count as f64)
                .sqrt();
            MeasurementMetrics {
                measurement_type,
                count,
                mae,
                rmse,
            }
        })
        .collect()
}

fn tensor_is_finite(tensor: &tch::Tensor) -> bool {
    tensor
        .isfinite()
        .all()
        .to_kind(tch::Kind::Int64)
        .int64_value(&[])
        != 0
}

fn active_slot_fraction(weights: &tch::Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    weights
        .gt(0.05)
        .to_kind(tch::Kind::Float)
        .mean(tch::Kind::Float)
        .double_value(&[])
}

fn mean_slot(slots: &tch::Tensor) -> tch::Tensor {
    if slots.numel() == 0 {
        tch::Tensor::zeros([1], (tch::Kind::Float, slots.device()))
    } else {
        slots.mean_dim([0].as_slice(), false, tch::Kind::Float)
    }
}

fn cosine_similarity(a: &tch::Tensor, b: &tch::Tensor) -> f64 {
    let dot = (a * b).sum(tch::Kind::Float).double_value(&[]);
    let a_norm = a.norm().double_value(&[]);
    let b_norm = b.norm().double_value(&[]);
    dot / (a_norm * b_norm).max(1e-6)
}

fn compute_slot_stability(forwards: &[ResearchForward]) -> SlotStabilityMetrics {
    if forwards.is_empty() {
        return SlotStabilityMetrics::default();
    }
    SlotStabilityMetrics {
        topology_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.topology.slot_weights)
        }),
        geometry_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.geometry.slot_weights)
        }),
        pocket_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.pocket.slot_weights)
        }),
        topology_signature_similarity: slot_signature_similarity(forwards, |forward| {
            &forward.slots.topology.slots
        }),
        geometry_signature_similarity: slot_signature_similarity(forwards, |forward| {
            &forward.slots.geometry.slots
        }),
        pocket_signature_similarity: slot_signature_similarity(forwards, |forward| {
            &forward.slots.pocket.slots
        }),
        topology_probe_alignment: mean_by(forwards, |forward| {
            if forward.probes.topology_adjacency_logits.numel() == 0 {
                0.0
            } else {
                let probe = forward
                    .probes
                    .topology_adjacency_logits
                    .sigmoid()
                    .mean(tch::Kind::Float);
                let activity = forward.slots.topology.slot_weights.mean(tch::Kind::Float);
                (&probe * &activity).double_value(&[])
            }
        }),
        geometry_probe_alignment: mean_by(forwards, |forward| {
            if forward.probes.geometry_distance_predictions.numel() == 0 {
                0.0
            } else {
                let probe = 1.0
                    / (1.0
                        + forward
                            .probes
                            .geometry_distance_predictions
                            .abs()
                            .mean(tch::Kind::Float)
                            .double_value(&[]));
                probe * active_slot_fraction(&forward.slots.geometry.slot_weights)
            }
        }),
        pocket_probe_alignment: mean_by(forwards, |forward| {
            if forward.probes.pocket_feature_predictions.numel() == 0 {
                0.0
            } else {
                let probe = 1.0
                    / (1.0
                        + forward
                            .probes
                            .pocket_feature_predictions
                            .abs()
                            .mean(tch::Kind::Float)
                            .double_value(&[]));
                probe * active_slot_fraction(&forward.slots.pocket.slot_weights)
            }
        }),
    }
}

fn mean_by(forwards: &[ResearchForward], f: impl Fn(&ResearchForward) -> f64) -> f64 {
    forwards.iter().map(f).sum::<f64>() / forwards.len().max(1) as f64
}

fn slot_signature_similarity<'a>(
    forwards: &'a [ResearchForward],
    slots: impl Fn(&'a ResearchForward) -> &'a tch::Tensor,
) -> f64 {
    let signatures = forwards
        .iter()
        .map(|forward| mean_slot(slots(forward)))
        .collect::<Vec<_>>();
    if signatures.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for left in 0..signatures.len() {
        for right in (left + 1)..signatures.len() {
            total += cosine_similarity(&signatures[left], &signatures[right]).abs();
            count += 1;
        }
    }
    total / count.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn unseen_pocket_experiment_smoke_test() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = UnseenPocketExperimentConfig::default();
        config.research.training.max_steps = 2;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 1;
        config.research.training.schedule.stage3_steps = 2;
        config.research.training.checkpoint_every = 100;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = temp.path().join("checkpoints");

        let summary = UnseenPocketExperiment::run(config).unwrap();
        assert_eq!(summary.training_history.len(), 2);
        assert!(
            summary
                .validation
                .representation_diagnostics
                .finite_forward_fraction
                >= 0.0
        );
        assert!(
            summary
                .test
                .representation_diagnostics
                .finite_forward_fraction
                >= 0.0
        );
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("experiment_summary.json")
            .exists());
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("dataset_validation_report.json")
            .exists());
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("claim_summary.json")
            .exists());
        assert_eq!(summary.dataset_validation.parsed_examples, 4);
    }

    #[test]
    fn resumed_experiment_preserves_prior_history() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint_dir = temp.path().join("checkpoints");

        let mut config = UnseenPocketExperimentConfig::default();
        config.research.training.max_steps = 2;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 1;
        config.research.training.schedule.stage3_steps = 2;
        config.research.training.checkpoint_every = 1;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = checkpoint_dir;

        let _ = UnseenPocketExperiment::run(config.clone()).unwrap();

        config.research.training.max_steps = 4;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 2;
        config.research.training.schedule.stage3_steps = 3;
        let summary = UnseenPocketExperiment::run_with_options(config, true).unwrap();

        assert_eq!(summary.training_history.len(), 4);
        assert_eq!(summary.training_history[0].step, 0);
        assert_eq!(summary.training_history[1].step, 1);
        assert_eq!(summary.training_history[2].step, 2);
        assert_eq!(summary.training_history[3].step, 3);
        assert!(!summary.split_report.leakage_checks.protein_overlap_detected);
    }

    #[test]
    fn comparison_summary_surfaces_strict_generation_metrics() {
        let mut docking = ReservedBackendMetrics {
            available: true,
            backend_name: Some("heuristic_docking_hook_v1".to_string()),
            metrics: BTreeMap::new(),
            status: "test".to_string(),
        };
        docking
            .metrics
            .insert("mean_centroid_offset".to_string(), 1.75);
        docking
            .metrics
            .insert("centroid_fit_score".to_string(), 0.36);
        let mut pocket = ReservedBackendMetrics {
            available: true,
            backend_name: Some("heuristic_pocket_compatibility_v1".to_string()),
            metrics: BTreeMap::new(),
            status: "test".to_string(),
        };
        pocket
            .metrics
            .insert("strict_pocket_fit_score".to_string(), 0.31);
        let metrics = RealGenerationMetrics {
            chemistry_validity: ReservedBackendMetrics {
                available: true,
                backend_name: Some("heuristic_validity_v1".to_string()),
                metrics: BTreeMap::from([("valid_fraction".to_string(), 1.0)]),
                status: "test".to_string(),
            },
            docking_affinity: docking,
            pocket_compatibility: pocket,
        };

        let summary = build_comparison_summary(
            &ResearchConfig::default(),
            &AblationConfig::default(),
            1.0,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            0.05,
            &metrics,
        );

        assert_eq!(summary.mean_centroid_offset, Some(1.75));
        assert_eq!(summary.strict_pocket_fit_score, Some(0.31));
        assert_eq!(summary.unique_smiles_fraction, None);
    }

    #[test]
    fn comparison_summary_surfaces_uniqueness_from_rdkit_metrics() {
        let metrics = RealGenerationMetrics {
            chemistry_validity: ReservedBackendMetrics {
                available: true,
                backend_name: Some("external_command_chemistry".to_string()),
                metrics: BTreeMap::from([
                    ("valid_fraction".to_string(), 1.0),
                    ("rdkit_unique_smiles_fraction".to_string(), 0.67),
                ]),
                status: "test".to_string(),
            },
            docking_affinity: ReservedBackendMetrics {
                available: false,
                backend_name: None,
                metrics: BTreeMap::new(),
                status: "test".to_string(),
            },
            pocket_compatibility: ReservedBackendMetrics {
                available: false,
                backend_name: None,
                metrics: BTreeMap::new(),
                status: "test".to_string(),
            },
        };

        let summary = build_comparison_summary(
            &ResearchConfig::default(),
            &AblationConfig::default(),
            1.0,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            0.05,
            &metrics,
        );

        assert_eq!(summary.unique_smiles_fraction, Some(0.67));
    }

    #[test]
    fn automated_search_generates_bounded_candidates() {
        let mut search = AutomatedSearchConfig {
            enabled: true,
            max_candidates: 3,
            include_base_candidate: true,
            ..AutomatedSearchConfig::default()
        };
        search.search_space.gate_temperature = vec![1.0, 1.2];
        search.search_space.coordinate_step_scale = vec![0.65, 0.8];

        let candidates = build_search_candidates(&ResearchConfig::default(), &search).unwrap();

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].id, "candidate_000_base");
        assert!(candidates
            .iter()
            .any(|candidate| !candidate.overrides.is_empty()));
    }

    #[test]
    fn automated_search_hard_gates_block_cross_surface_regression() {
        let surface = AutomatedSearchSurfaceSummary {
            surface_label: "tight_geometry_pressure".to_string(),
            source_config: "configs/unseen_pocket_tight_geometry_pressure.json".into(),
            artifact_dir: "checkpoints/tight_geometry_pressure".into(),
            claim_report: ClaimReport {
                artifact_dir: "checkpoints/tight_geometry_pressure".into(),
                run_label: "test".to_string(),
                validation: test_generation_quality_summary(),
                test: GenerationQualitySummary {
                    strict_pocket_fit_score: Some(0.35),
                    unique_smiles_fraction: Some(0.33),
                    ..test_generation_quality_summary()
                },
                backend_metrics: RealGenerationMetrics {
                    chemistry_validity: ReservedBackendMetrics {
                        available: true,
                        backend_name: Some("rdkit".to_string()),
                        metrics: BTreeMap::from([
                            ("valid_fraction".to_string(), 1.0),
                            ("rdkit_sanitized_fraction".to_string(), 1.0),
                        ]),
                        status: "test".to_string(),
                    },
                    docking_affinity: ReservedBackendMetrics {
                        available: true,
                        backend_name: Some("pocket".to_string()),
                        metrics: BTreeMap::new(),
                        status: "test".to_string(),
                    },
                    pocket_compatibility: ReservedBackendMetrics {
                        available: true,
                        backend_name: Some("pocket".to_string()),
                        metrics: BTreeMap::from([("clash_fraction".to_string(), 0.0)]),
                        status: "test".to_string(),
                    },
                },
                backend_thresholds: BTreeMap::new(),
                backend_review: BackendReviewReport::default(),
                layered_generation_metrics: empty_layered_generation_metrics(),
                chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary::default(),
                claim_context: ClaimContext::default(),
                backend_environment: None,
                ablation_deltas: Vec::new(),
                reranker_report: RerankerReport::default(),
                slot_stability: SlotStabilityMetrics::default(),
                leakage_calibration: LeakageCalibrationReport::default(),
                performance_gates: PerformanceGateReport::default(),
                baseline_comparisons: Vec::new(),
            },
        };

        let gate_result =
            evaluate_search_gates(&[surface], &AutomatedSearchHardGateConfig::default());

        assert!(!gate_result.passed);
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("unique_smiles_fraction")));
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("strict_pocket_fit_score")));
    }

    #[test]
    fn automated_search_can_block_raw_model_regressions() {
        let mut surface = AutomatedSearchSurfaceSummary {
            surface_label: "raw_gate_surface".to_string(),
            source_config: "configs/raw_gate.json".into(),
            artifact_dir: "checkpoints/raw_gate".into(),
            claim_report: test_claim_report(),
        };
        surface.claim_report.layered_generation_metrics.raw_rollout = CandidateLayerMetrics {
            candidate_count: 3,
            valid_fraction: 1.0,
            pocket_contact_fraction: 1.0,
            mean_centroid_offset: 9.5,
            clash_fraction: 0.4,
            mean_displacement: 0.8,
            atom_change_fraction: 0.5,
            uniqueness_proxy_fraction: 0.2,
            atom_type_sequence_diversity: 0.2,
            bond_topology_diversity: 0.2,
            coordinate_shape_diversity: 0.2,
            novel_atom_type_sequence_fraction: 0.2,
            novel_bond_topology_fraction: 0.2,
            novel_coordinate_shape_fraction: 0.2,
        };
        let gates = AutomatedSearchHardGateConfig {
            maximum_raw_centroid_offset: Some(3.0),
            maximum_raw_clash_fraction: Some(0.1),
            maximum_raw_mean_displacement: Some(0.5),
            maximum_raw_atom_change_fraction: Some(0.25),
            minimum_raw_uniqueness_proxy_fraction: Some(0.5),
            ..AutomatedSearchHardGateConfig::default()
        };

        let gate_result = evaluate_search_gates(&[surface], &gates);

        assert!(!gate_result.passed);
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("raw_centroid_offset")));
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("raw_uniqueness_proxy_fraction")));
    }

    #[test]
    fn chemistry_benchmark_evidence_uses_configured_external_dataset() {
        let mut summary: UnseenPocketExperimentSummary = serde_json::from_str(
            &std::fs::read_to_string("checkpoints/pdbbindpp_real_backends/experiment_summary.json")
                .expect("existing experiment summary should be readable for benchmark-evidence tests"),
        )
        .expect("existing experiment summary should deserialize");
        summary.config.surface_label = Some("lp_pdbbind_refined_real_backends".to_string());
        summary.config.reviewer_benchmark.dataset = Some("lp_pdbbind_refined".to_string());
        summary.config.research.training.checkpoint_dir =
            PathBuf::from("./checkpoints/lp_pdbbind_refined_real_backends");
        summary.dataset_validation.parsed_examples = 5048;
        summary.dataset_validation.retained_label_coverage = 1.0;
        summary.split_report.val.protein_family_proxy_histogram = BTreeMap::from_iter(
            (0..12).map(|ix| (format!("val_family_{ix}"), 1usize)),
        );
        summary.split_report.test.protein_family_proxy_histogram = BTreeMap::from_iter(
            (0..12).map(|ix| (format!("test_family_{ix}"), 1usize)),
        );
        summary.test.comparison_summary.candidate_valid_fraction = Some(1.0);
        summary.test.comparison_summary.unique_smiles_fraction = Some(1.0);
        summary.test.comparison_summary.strict_pocket_fit_score = Some(0.6);
        summary.test.comparison_summary.leakage_proxy_mean = 0.05;
        summary.test.representation_diagnostics.leakage_proxy_mean = 0.05;
        summary.test.layered_generation_metrics.reranked_candidates = CandidateLayerMetrics {
            candidate_count: 8,
            valid_fraction: 1.0,
            pocket_contact_fraction: 1.0,
            mean_centroid_offset: 0.4,
            clash_fraction: 0.0,
            mean_displacement: 0.1,
            atom_change_fraction: 0.1,
            uniqueness_proxy_fraction: 1.0,
            atom_type_sequence_diversity: 1.0,
            bond_topology_diversity: 1.0,
            coordinate_shape_diversity: 1.0,
            novel_atom_type_sequence_fraction: 1.0,
            novel_bond_topology_fraction: 1.0,
            novel_coordinate_shape_fraction: 1.0,
        };
        summary.test.real_generation_metrics.chemistry_validity.metrics = BTreeMap::from([
            ("rdkit_parseable_fraction".to_string(), 1.0),
            ("rdkit_finite_conformer_fraction".to_string(), 1.0),
            ("rdkit_sanitized_fraction".to_string(), 1.0),
            ("rdkit_unique_smiles_fraction".to_string(), 1.0),
        ]);

        let evidence = build_chemistry_benchmark_evidence(&summary);

        assert!(evidence.external_benchmark_backed);
        assert_eq!(
            evidence.external_benchmark_dataset.as_deref(),
            Some("lp_pdbbind_refined")
        );
        assert!(evidence
            .external_benchmark_note
            .contains("lp_pdbbind_refined"));
    }

    #[test]
    fn split_review_counts_wins_losses_and_ties() {
        let lightweight = GenerationQualitySummary {
            primary_objective: "conditioned_denoising".to_string(),
            variant_label: Some("interaction_lightweight".to_string()),
            interaction_mode: "lightweight".to_string(),
            candidate_valid_fraction: Some(1.0),
            pocket_contact_fraction: Some(1.0),
            pocket_compatibility_fraction: Some(1.0),
            mean_centroid_offset: Some(0.8),
            strict_pocket_fit_score: Some(0.55),
            unique_smiles_fraction: Some(0.8),
            unseen_protein_fraction: 1.0,
            topology_specialization_score: 0.2,
            geometry_specialization_score: 0.4,
            pocket_specialization_score: 0.5,
            slot_activation_mean: 0.7,
            gate_activation_mean: 0.3,
            leakage_proxy_mean: 0.08,
        };
        let transformer = GenerationQualitySummary {
            primary_objective: "conditioned_denoising".to_string(),
            variant_label: Some("interaction_transformer".to_string()),
            interaction_mode: "transformer".to_string(),
            candidate_valid_fraction: Some(1.0),
            pocket_contact_fraction: Some(1.0),
            pocket_compatibility_fraction: Some(1.0),
            mean_centroid_offset: Some(0.9),
            strict_pocket_fit_score: Some(0.5),
            unique_smiles_fraction: Some(0.8),
            unseen_protein_fraction: 1.0,
            topology_specialization_score: 0.6,
            geometry_specialization_score: 0.7,
            pocket_specialization_score: 0.55,
            slot_activation_mean: 0.8,
            gate_activation_mean: 0.4,
            leakage_proxy_mean: 0.04,
        };

        let review = build_split_review(&lightweight, &transformer);

        assert_eq!(review.tally.lightweight_wins, 2);
        assert_eq!(review.tally.transformer_wins, 6);
        assert_eq!(review.tally.ties, 4);
        assert_eq!(review.geometric_fit[3].winner, MetricWinner::Lightweight);
        assert_eq!(review.specialization[0].winner, MetricWinner::Transformer);
        assert_eq!(review.utilization[0].winner, MetricWinner::Transformer);
    }

    #[test]
    fn multi_seed_metric_aggregate_reports_deterministic_confidence_interval() {
        let aggregate = MultiSeedMetricAggregate::from_values(&[1.0, 2.0, 3.0]);

        assert_eq!(aggregate.count, 3);
        assert!((aggregate.mean - 2.0).abs() < 1e-12);
        assert!((aggregate.std - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((aggregate.standard_error - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((aggregate.confidence95_low - (2.0 - 4.303 / 3.0_f64.sqrt())).abs() < 1e-12);
        assert!((aggregate.confidence95_high - (2.0 + 4.303 / 3.0_f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn performance_gate_report_tracks_threshold_failures() {
        let config = PerformanceGateConfig {
            min_validation_examples_per_second: Some(10.0),
            min_test_examples_per_second: Some(20.0),
            max_validation_memory_mb: Some(5.0),
            max_test_memory_mb: Some(5.0),
        };
        let validation = ResourceUsageMetrics {
            memory_usage_mb: 6.0,
            evaluation_time_ms: 100.0,
            examples_per_second: 9.0,
            average_ligand_atoms: 5.0,
            average_pocket_atoms: 10.0,
        };
        let test = ResourceUsageMetrics {
            memory_usage_mb: 1.0,
            evaluation_time_ms: 100.0,
            examples_per_second: 30.0,
            average_ligand_atoms: 5.0,
            average_pocket_atoms: 10.0,
        };

        let report = build_performance_gate_report(&config, &validation, &test);

        assert!(!report.passed);
        assert_eq!(report.failed_reasons.len(), 2);
    }

    #[test]
    fn candidate_layer_reports_diversity_proxies() {
        let mut first = test_candidate(vec![6, 6], vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
        first.inferred_bonds = vec![(0, 1)];
        let mut second = test_candidate(vec![6, 8], vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        second.inferred_bonds = vec![(0, 1)];

        let metrics =
            summarize_candidate_layer(&[first, second], &NoveltyReferenceSignatures::default());

        assert_eq!(metrics.candidate_count, 2);
        assert_eq!(metrics.atom_type_sequence_diversity, 1.0);
        assert_eq!(metrics.bond_topology_diversity, 0.5);
        assert_eq!(metrics.coordinate_shape_diversity, 1.0);
        assert_eq!(metrics.novel_atom_type_sequence_fraction, 1.0);
        assert_eq!(metrics.novel_bond_topology_fraction, 1.0);
        assert_eq!(metrics.novel_coordinate_shape_fraction, 1.0);
    }

    #[test]
    fn candidate_layer_reports_novelty_against_reference_examples() {
        let reference = crate::data::MolecularExample::from_legacy(
            "ref",
            "protein",
            &crate::types::Ligand {
                atoms: vec![
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [0.0, 0.0, 0.0],
                        index: 0,
                    },
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [1.5, 0.0, 0.0],
                        index: 1,
                    },
                ],
                bonds: vec![(0, 1)],
                fingerprint: None,
            },
            &crate::types::Pocket {
                name: "pocket".to_string(),
                atoms: vec![
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [0.0, 0.0, 0.0],
                        index: 0,
                    },
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [2.0, 0.0, 0.0],
                        index: 1,
                    },
                ],
            },
        );
        let novelty_reference = novelty_reference_signatures(&[reference]);
        let mut seen = test_candidate(vec![0, 0], vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
        seen.inferred_bonds = vec![(0, 1)];
        let mut novel = test_candidate(vec![0, 2], vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        novel.inferred_bonds = vec![(0, 1)];

        let metrics = summarize_candidate_layer(&[seen, novel], &novelty_reference);

        assert_eq!(metrics.novel_atom_type_sequence_fraction, 0.5);
        assert_eq!(metrics.novel_bond_topology_fraction, 0.0);
        assert_eq!(metrics.novel_coordinate_shape_fraction, 1.0);
    }

    #[test]
    fn backend_metrics_choose_best_clean_candidate_across_layers() {
        let temp = tempfile::tempdir().unwrap();
        let pocket_path = temp.path().join("pocket.pdb");
        std::fs::write(
            &pocket_path,
            concat!(
                "ATOM      1  C   LIG A   1       0.000   0.000   0.000\n",
                "ATOM      2  C   LIG A   1       0.200   0.000   0.000\n",
                "ATOM      3  C   LIG A   1       0.400   0.000   0.000\n",
                "ATOM      4  C   LIG A   1       0.600   0.000   0.000\n",
                "ATOM      5  C   LIG A   1       0.800   0.000   0.000\n",
            ),
        )
        .unwrap();

        let mut inferred = test_candidate(vec![6, 6], vec![[2.1, 0.0, 0.0], [2.5, 0.0, 0.0]]);
        inferred.source_pocket_path = Some(pocket_path.display().to_string());
        let mut reranked = test_candidate(vec![6, 6], vec![[4.0, 0.0, 0.0], [4.4, 0.0, 0.0]]);
        reranked.source_pocket_path = Some(pocket_path.display().to_string());
        let reranked = vec![reranked];
        let proxy = vec![test_candidate(
            vec![6, 6],
            vec![[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )];

        let selected = final_backend_candidate_layer(&[inferred], &reranked, &proxy);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [2.1, 0.0, 0.0]);
    }

    #[test]
    fn backend_metrics_fall_back_to_proxy_when_reranked_is_empty() {
        let inferred = vec![test_candidate(
            vec![6, 6],
            vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        )];
        let proxy = vec![test_candidate(
            vec![6, 6],
            vec![[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )];

        let selected = final_backend_candidate_layer(&inferred, &[], &proxy);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn backend_metrics_skip_reranked_candidate_with_pocket_clash() {
        let temp = tempfile::tempdir().unwrap();
        let pocket_path = temp.path().join("pocket.pdb");
        std::fs::write(
            &pocket_path,
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000\n",
        )
        .unwrap();

        let inferred = vec![test_candidate(
            vec![6, 6],
            vec![[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        )];
        let mut clashing = test_candidate(vec![6, 6], vec![[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]);
        clashing.source_pocket_path = Some(pocket_path.display().to_string());
        let mut clean = test_candidate(vec![6, 6], vec![[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]]);
        clean.source_pocket_path = Some(pocket_path.display().to_string());
        let reranked = vec![clashing, clean];

        let selected = final_backend_candidate_layer(&inferred, &reranked, &[]);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [2.0, 0.0, 0.0]);
    }

    #[test]
    fn backend_metrics_fall_back_when_all_reranked_candidates_clash() {
        let temp = tempfile::tempdir().unwrap();
        let pocket_path = temp.path().join("pocket.pdb");
        std::fs::write(
            &pocket_path,
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000\n",
        )
        .unwrap();

        let mut inferred = test_candidate(vec![6, 6], vec![[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]]);
        inferred.source_pocket_path = Some(pocket_path.display().to_string());
        let mut clashing = test_candidate(vec![6, 6], vec![[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]);
        clashing.source_pocket_path = Some(pocket_path.display().to_string());

        let inferred = [inferred];
        let reranked = [clashing];
        let selected = final_backend_candidate_layer(&inferred, &reranked, &[]);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [2.0, 0.0, 0.0]);
    }

    fn test_generation_quality_summary() -> GenerationQualitySummary {
        GenerationQualitySummary {
            primary_objective: "conditioned_denoising".to_string(),
            variant_label: Some("test".to_string()),
            interaction_mode: "transformer".to_string(),
            candidate_valid_fraction: Some(1.0),
            pocket_contact_fraction: Some(1.0),
            pocket_compatibility_fraction: Some(1.0),
            mean_centroid_offset: Some(0.5),
            strict_pocket_fit_score: Some(0.6),
            unique_smiles_fraction: Some(0.8),
            unseen_protein_fraction: 1.0,
            topology_specialization_score: 0.5,
            geometry_specialization_score: 0.5,
            pocket_specialization_score: 0.5,
            slot_activation_mean: 0.4,
            gate_activation_mean: 0.3,
            leakage_proxy_mean: 0.05,
        }
    }

    fn test_claim_report() -> ClaimReport {
        ClaimReport {
            artifact_dir: "checkpoints/test".into(),
            run_label: "test".to_string(),
            validation: test_generation_quality_summary(),
            test: test_generation_quality_summary(),
            backend_metrics: RealGenerationMetrics {
                chemistry_validity: ReservedBackendMetrics {
                    available: true,
                    backend_name: Some("chemistry".to_string()),
                    metrics: BTreeMap::from([
                        ("valid_fraction".to_string(), 1.0),
                        ("rdkit_sanitized_fraction".to_string(), 1.0),
                    ]),
                    status: "test".to_string(),
                },
                docking_affinity: ReservedBackendMetrics {
                    available: true,
                    backend_name: Some("docking".to_string()),
                    metrics: BTreeMap::new(),
                    status: "test".to_string(),
                },
                pocket_compatibility: ReservedBackendMetrics {
                    available: true,
                    backend_name: Some("pocket".to_string()),
                    metrics: BTreeMap::from([("clash_fraction".to_string(), 0.0)]),
                    status: "test".to_string(),
                },
            },
            backend_thresholds: BTreeMap::new(),
            backend_review: BackendReviewReport::default(),
            layered_generation_metrics: empty_layered_generation_metrics(),
            chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary::default(),
            claim_context: ClaimContext::default(),
            backend_environment: None,
            ablation_deltas: Vec::new(),
            reranker_report: RerankerReport::default(),
            slot_stability: SlotStabilityMetrics::default(),
            leakage_calibration: LeakageCalibrationReport::default(),
            performance_gates: PerformanceGateReport::default(),
            baseline_comparisons: Vec::new(),
        }
    }

    fn test_candidate(atom_types: Vec<i64>, coords: Vec<[f32; 3]>) -> GeneratedCandidateRecord {
        GeneratedCandidateRecord {
            example_id: "example".to_string(),
            protein_id: "protein".to_string(),
            molecular_representation: None,
            atom_types,
            coords,
            inferred_bonds: Vec::new(),
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 6.0,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: "test".to_string(),
            source_pocket_path: Some("pocket.pdb".to_string()),
            source_ligand_path: Some("ligand.sdf".to_string()),
        }
    }
}
