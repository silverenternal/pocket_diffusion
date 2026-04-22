//! Staged training utilities for the modular research stack.

pub mod checkpoint;
pub mod demos;
pub mod entrypoints;
pub mod metrics;
pub mod reporting;
pub mod scheduler;
pub mod trainer;

pub use checkpoint::{CheckpointManager, CheckpointMetadata, LoadedCheckpoint};
pub use demos::{run_phase1_demo, run_phase3_training_demo, run_phase4_experiment_demo};
pub(crate) use entrypoints::DatasetInspection;
pub use entrypoints::{inspect_dataset_from_config, run_training_from_config};
pub(crate) use metrics::reproducibility_metadata;
pub use metrics::{
    stable_json_hash, AuxiliaryLossMetrics, DatasetSplitSizes, LossBreakdown,
    PrimaryObjectiveMetrics, ReproducibilityMetadata, ResumeContract, ResumeProvenance,
    RunArtifactBundle, RunArtifactPaths, RunKind, SplitLeakageChecks, SplitReport, SplitStats,
    StepMetrics, TrainingRunSummary, TrainingStage, ARTIFACT_BUNDLE_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION, RESUME_CONTRACT_VERSION,
};
pub use reporting::{
    print_dataset_inspection, print_eval_metrics, print_experiment_run, print_step_metrics,
    print_training_run,
};
pub use scheduler::{EffectiveLossWeights, StageScheduler};
pub use trainer::ResearchTrainer;
