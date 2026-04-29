//! Staged training utilities for the modular research stack.

pub mod checkpoint;
pub mod demos;
pub mod entrypoints;
pub mod metrics;
pub mod reporting;
pub mod scheduler;
pub mod trainer;

pub use checkpoint::{
    BackendTrainingMetadata, BestCheckpointMetadata, CheckpointManager, CheckpointMetadata,
    LoadedCheckpoint, OptimizerStateMetadata, ResumeMode, SchedulerStateMetadata,
};
pub use demos::{run_phase1_demo, run_phase3_training_demo, run_phase4_experiment_demo};
pub(crate) use entrypoints::DatasetInspection;
pub use entrypoints::{inspect_dataset_from_config, run_training_from_config};
pub(crate) use metrics::{determinism_controls_from_config, reproducibility_metadata};
pub use metrics::{
    stable_json_hash, training_summary_checkpoint_replay_compatibility, AuxiliaryLossMetrics,
    AuxiliaryObjectiveFamily, AuxiliaryObjectiveReport, AuxiliaryObjectiveReportEntry,
    BestCheckpointSummary, CoordinateFrameProvenance, DatasetSplitSizes, DeterminismControls,
    EarlyStoppingSummary, GradientHealthMetrics, GradientModuleMetrics,
    InteractionFlowTimeBucketMetrics, InteractionPathStepMetrics, InteractionStepMetrics,
    LossBreakdown, ObjectiveCoverageRecord, ObjectiveCoverageReport,
    ObjectiveExecutionCountMetrics, ObjectiveGradientDiagnostics, ObjectiveGradientFamilyMetrics,
    PrimaryBranchScheduleReport, PrimaryBranchWeightRecord, PrimaryObjectiveComponentMetrics,
    PrimaryObjectiveComponentProvenance, PrimaryObjectiveComponentScaleRecord,
    PrimaryObjectiveComponentScaleReport, PrimaryObjectiveMetrics, ReplayCompatibilityClass,
    ReplayCompatibilityMismatch, ReplayCompatibilityReport, ReplayTolerance,
    ReproducibilityMetadata, ResumeContinuityMode, ResumeContract, ResumeProvenance,
    RunArtifactBundle, RunArtifactPaths, RunKind, SlotSignatureStepSummary,
    SlotUtilizationStepMetrics, SplitLeakageChecks, SplitReport, SplitStats, StageProgressMetrics,
    StepMetrics, SynchronizationHealthMetrics, TrainingRunSummary, TrainingRuntimeProfileMetrics,
    TrainingStage, ValidationHistoryEntry, ARTIFACT_BUNDLE_SCHEMA_VERSION, METRIC_SCHEMA_VERSION,
    RESUME_CONTRACT_VERSION,
};
pub use reporting::{
    print_automated_search, print_dataset_inspection, print_eval_metrics, print_experiment_run,
    print_step_metrics, print_training_run,
};
pub use scheduler::{EffectiveLossWeights, StageScheduler};
pub use trainer::ResearchTrainer;
