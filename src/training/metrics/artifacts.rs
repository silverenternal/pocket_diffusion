/// Persisted summary of a config-driven modular training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunSummary {
    /// Research configuration used for the run.
    pub config: ResearchConfig,
    /// Machine-readable dataset validation artifact for the run.
    pub dataset_validation: DatasetValidationReport,
    /// Dataset sizes observed by the run.
    pub splits: DatasetSplitSizes,
    /// Split-distribution and leakage audit.
    pub split_report: SplitReport,
    /// Optional step that the run resumed from.
    pub resumed_from_step: Option<usize>,
    /// Reproducibility and schema metadata for this run.
    pub reproducibility: ReproducibilityMetadata,
    /// All collected training metrics.
    pub training_history: Vec<StepMetrics>,
    /// Validation metrics evaluated after training.
    pub validation: EvaluationMetrics,
    /// Test metrics evaluated after training.
    pub test: EvaluationMetrics,
}

/// Shared reproducibility metadata persisted across train and experiment runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityMetadata {
    /// Stable hash of the effective config JSON.
    pub config_hash: String,
    /// Stable hash of the dataset validation artifact JSON.
    pub dataset_validation_fingerprint: String,
    /// Metric schema version used for evaluation artifacts.
    pub metric_schema_version: u32,
    /// Shared bundle schema version used on disk.
    pub artifact_bundle_schema_version: u32,
    /// Explicit deterministic runtime knobs and seeds that affect replay.
    #[serde(default)]
    pub determinism_controls: DeterminismControls,
    /// Bounded replay tolerance used by reviewer-surface drift checks.
    #[serde(default)]
    pub replay_tolerance: ReplayTolerance,
    /// Explicit resume semantics for this run family.
    pub resume_contract: ResumeContract,
    /// Whether the run resumed from a previous checkpoint.
    pub resume_provenance: ResumeProvenance,
}

/// Runtime and seed controls that materially affect replay behavior.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeterminismControls {
    /// Protein split seed.
    #[serde(default)]
    pub split_seed: u64,
    /// Decoder corruption seed.
    #[serde(default)]
    pub corruption_seed: u64,
    /// Decoder rollout sampling seed.
    #[serde(default)]
    pub sampling_seed: u64,
    /// Runtime device string.
    #[serde(default)]
    pub device: String,
    /// Configured Rust-side data workers.
    #[serde(default)]
    pub data_workers: usize,
    /// Optional libtorch intra-op thread count.
    #[serde(default)]
    pub tch_intra_op_threads: Option<i32>,
    /// Optional libtorch inter-op thread count.
    #[serde(default)]
    pub tch_inter_op_threads: Option<i32>,
}

/// Reviewer-facing tolerances for bounded replay and drift checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTolerance {
    /// Maximum absolute drift allowed for leakage proxy mean.
    pub leakage_proxy_mean_abs: f64,
    /// Maximum absolute drift allowed for strict pocket fit.
    pub strict_pocket_fit_score_abs: f64,
    /// Maximum absolute drift allowed for clash fraction.
    pub clash_fraction_abs: f64,
    /// Maximum absolute drift allowed for chemistry quality metrics.
    pub chemistry_quality_abs: f64,
}

impl Default for ReplayTolerance {
    fn default() -> Self {
        Self {
            leakage_proxy_mean_abs: 0.01,
            strict_pocket_fit_score_abs: 0.03,
            clash_fraction_abs: 0.02,
            chemistry_quality_abs: 0.05,
        }
    }
}

/// Reviewer-facing continuity class for resumed runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResumeContinuityMode {
    /// Fresh run with no prior checkpoint applied.
    #[default]
    FreshRun,
    /// Weights, step, history, and metadata were restored, but optimizer moments were not.
    MetadataOnlyContinuation,
    /// Full optimizer/scheduler internal state was restored, enabling strict replay claims.
    FullOptimizerContinuation,
}

/// Explicit statement of what a resume operation restores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeContract {
    /// Stable version identifier for the contract.
    pub version: String,
    /// Whether model weights are restored.
    pub restores_model_weights: bool,
    /// Whether the step counter is restored.
    pub restores_step: bool,
    /// Whether training-history summaries are restored.
    pub restores_history: bool,
    /// Whether optimizer state is restored.
    pub restores_optimizer_state: bool,
    /// Reviewer-facing label for the strongest continuity this contract can support.
    #[serde(default)]
    pub continuity_mode: ResumeContinuityMode,
    /// Whether the run family supports strict deterministic replay when resumed.
    #[serde(default)]
    pub supports_strict_replay: bool,
    /// Human-readable note describing the limits of reproducibility.
    pub notes: String,
}

/// Summary of whether and how the current run resumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeProvenance {
    /// Whether this run resumed from a prior checkpoint.
    pub resumed: bool,
    /// Step loaded from the checkpoint, if any.
    pub resumed_from_step: Option<usize>,
    /// Config hash recorded in the loaded checkpoint, if any.
    pub checkpoint_config_hash: Option<String>,
    /// Dataset validation fingerprint recorded in the loaded checkpoint, if any.
    pub checkpoint_dataset_fingerprint: Option<String>,
    /// Whether optimizer-state metadata was present in the loaded checkpoint.
    #[serde(default)]
    pub restored_optimizer_state_metadata: bool,
    /// Whether scheduler-state metadata was present in the loaded checkpoint.
    #[serde(default)]
    pub restored_scheduler_state_metadata: bool,
    /// Reviewer-facing label for the continuity actually achieved by this run.
    #[serde(default)]
    pub continuity_mode: ResumeContinuityMode,
    /// Whether this specific resumed run qualifies as a strict replay.
    #[serde(default)]
    pub strict_replay_achieved: bool,
}

/// Shared on-disk artifact bundle for training and experiment workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunArtifactBundle {
    /// Bundle schema version.
    pub schema_version: u32,
    /// Run kind stored in this bundle.
    pub run_kind: RunKind,
    /// Base directory for persisted outputs.
    pub artifact_dir: PathBuf,
    /// Config hash used to produce the run.
    pub config_hash: String,
    /// Dataset validation fingerprint associated with the run.
    pub dataset_validation_fingerprint: String,
    /// Metric schema version used by summaries.
    pub metric_schema_version: u32,
    /// Reviewer-facing backend environment fingerprint when external backends are configured.
    #[serde(default)]
    pub backend_environment: Option<crate::experiments::BackendEnvironmentReport>,
    /// Paths to individual artifacts.
    pub paths: RunArtifactPaths,
}

/// Research workflow variant represented by a bundle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    /// Config-driven training workflow.
    Training,
    /// Config-driven unseen-pocket experiment workflow.
    Experiment,
}

/// Concrete files persisted for a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunArtifactPaths {
    /// Persisted config snapshot.
    pub config_snapshot: PathBuf,
    /// Dataset validation report.
    pub dataset_validation_report: PathBuf,
    /// Split audit report.
    pub split_report: PathBuf,
    /// Primary run summary file.
    pub run_summary: PathBuf,
    /// Shared run bundle description.
    pub run_bundle: PathBuf,
    /// Latest checkpoint weights path, if any.
    pub latest_checkpoint: Option<PathBuf>,
}
