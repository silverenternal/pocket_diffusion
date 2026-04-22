//! Training metrics and persisted run summaries.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::config::ResearchConfig;
use crate::{
    data::{DatasetValidationReport, InMemoryDataset},
    experiments::EvaluationMetrics,
};

/// Version tag for the persisted metric schema.
pub const METRIC_SCHEMA_VERSION: u32 = 3;
/// Version tag for the shared run artifact bundle schema.
pub const ARTIFACT_BUNDLE_SCHEMA_VERSION: u32 = 1;
/// Human-readable resume contract identifier for the current research path.
pub const RESUME_CONTRACT_VERSION: &str = "weights+history+step";

/// Named loss values emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryObjectiveMetrics {
    /// Name of the active primary objective.
    pub objective_name: String,
    /// Weighted scalar value used as the primary optimization anchor.
    pub primary_value: f64,
    /// Whether the primary objective is decoder-anchored.
    pub decoder_anchored: bool,
}

/// Auxiliary losses emitted by each training step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AuxiliaryLossMetrics {
    /// Intra-modality redundancy objective.
    pub intra_red: f64,
    /// Semantic probe objective.
    pub probe: f64,
    /// Leakage objective.
    pub leak: f64,
    /// Gate regularization objective.
    pub gate: f64,
    /// Slot sparsity and balance objective.
    pub slot: f64,
    /// Topology-geometry consistency objective.
    pub consistency: f64,
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

/// One trainer step record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Global optimization step.
    pub step: usize,
    /// Current staged-training phase.
    pub stage: TrainingStage,
    /// Loss values for this step.
    pub losses: LossBreakdown,
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

/// Dataset split sizes used by a training or experiment run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplitSizes {
    /// Number of examples before splitting.
    pub total: usize,
    /// Number of training examples.
    pub train: usize,
    /// Number of validation examples.
    pub val: usize,
    /// Number of test examples.
    pub test: usize,
}

/// Split-level audit artifact for unseen-pocket experiments and training runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitReport {
    /// Training split audit.
    pub train: SplitStats,
    /// Validation split audit.
    pub val: SplitStats,
    /// Test split audit.
    pub test: SplitStats,
    /// Cross-split leakage checks.
    pub leakage_checks: SplitLeakageChecks,
}

/// Per-split statistics needed to audit unseen-pocket claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitStats {
    /// Number of examples in the split.
    pub example_count: usize,
    /// Number of unique protein ids in the split.
    pub unique_protein_count: usize,
    /// Number of labeled examples in the split.
    pub labeled_example_count: usize,
    /// Fraction of examples in the split with labels.
    pub labeled_fraction: f64,
    /// Histogram of dominant measurement families in the split.
    pub dominant_measurement_histogram: BTreeMap<String, usize>,
}

/// Explicit leakage checks across train/val/test partitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitLeakageChecks {
    /// Whether protein ids overlap across splits.
    pub protein_overlap_detected: bool,
    /// Whether example ids overlap across splits.
    pub duplicate_example_ids_detected: bool,
    /// Number of train/val protein overlaps.
    pub train_val_protein_overlap: usize,
    /// Number of train/test protein overlaps.
    pub train_test_protein_overlap: usize,
    /// Number of val/test protein overlaps.
    pub val_test_protein_overlap: usize,
    /// Number of duplicated example ids across all splits.
    pub duplicated_example_ids: usize,
}

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
    /// Explicit resume semantics for this run family.
    pub resume_contract: ResumeContract,
    /// Whether the run resumed from a previous checkpoint.
    pub resume_provenance: ResumeProvenance,
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

impl SplitReport {
    /// Build a split report from three dataset partitions.
    pub fn from_datasets(
        train: &InMemoryDataset,
        val: &InMemoryDataset,
        test: &InMemoryDataset,
    ) -> Self {
        let train_stats = SplitStats::from_dataset(train);
        let val_stats = SplitStats::from_dataset(val);
        let test_stats = SplitStats::from_dataset(test);

        let train_proteins = protein_set(train);
        let val_proteins = protein_set(val);
        let test_proteins = protein_set(test);

        let train_ids = example_id_set(train);
        let val_ids = example_id_set(val);
        let test_ids = example_id_set(test);

        let train_val_protein_overlap = train_proteins.intersection(&val_proteins).count();
        let train_test_protein_overlap = train_proteins.intersection(&test_proteins).count();
        let val_test_protein_overlap = val_proteins.intersection(&test_proteins).count();

        let duplicated_example_ids = train_ids.intersection(&val_ids).count()
            + train_ids.intersection(&test_ids).count()
            + val_ids.intersection(&test_ids).count();

        Self {
            train: train_stats,
            val: val_stats,
            test: test_stats,
            leakage_checks: SplitLeakageChecks {
                protein_overlap_detected: train_val_protein_overlap > 0
                    || train_test_protein_overlap > 0
                    || val_test_protein_overlap > 0,
                duplicate_example_ids_detected: duplicated_example_ids > 0,
                train_val_protein_overlap,
                train_test_protein_overlap,
                val_test_protein_overlap,
                duplicated_example_ids,
            },
        }
    }
}

impl SplitStats {
    fn from_dataset(dataset: &InMemoryDataset) -> Self {
        let example_count = dataset.examples().len();
        let unique_protein_count = protein_set(dataset).len();
        let labeled_example_count = dataset
            .examples()
            .iter()
            .filter(|example| example.targets.affinity_kcal_mol.is_some())
            .count();
        let labeled_fraction = if example_count == 0 {
            0.0
        } else {
            labeled_example_count as f64 / example_count as f64
        };
        let mut dominant_measurement_histogram = BTreeMap::new();
        for example in dataset.examples() {
            let measurement = example
                .targets
                .affinity_measurement_type
                .as_deref()
                .unwrap_or("unknown")
                .to_string();
            *dominant_measurement_histogram
                .entry(measurement)
                .or_default() += 1;
        }
        Self {
            example_count,
            unique_protein_count,
            labeled_example_count,
            labeled_fraction,
            dominant_measurement_histogram,
        }
    }
}

fn protein_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<&str> {
    dataset
        .examples()
        .iter()
        .map(|example| example.protein_id.as_str())
        .collect()
}

fn example_id_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<&str> {
    dataset
        .examples()
        .iter()
        .map(|example| example.example_id.as_str())
        .collect()
}

/// Build the reproducibility metadata for a config-driven run.
pub fn reproducibility_metadata(
    config: &ResearchConfig,
    dataset_validation: &DatasetValidationReport,
    checkpoint_metadata: Option<&crate::training::CheckpointMetadata>,
) -> ReproducibilityMetadata {
    ReproducibilityMetadata {
        config_hash: stable_json_hash(config),
        dataset_validation_fingerprint: stable_json_hash(dataset_validation),
        metric_schema_version: METRIC_SCHEMA_VERSION,
        artifact_bundle_schema_version: ARTIFACT_BUNDLE_SCHEMA_VERSION,
        resume_contract: ResumeContract {
            version: RESUME_CONTRACT_VERSION.to_string(),
            restores_model_weights: true,
            restores_step: true,
            restores_history: true,
            restores_optimizer_state: false,
            notes: "Resume restores model weights, step index, and prior persisted training history. Optimizer state is not restored, so restart behavior is a convenience continuation rather than a strict deterministic replay.".to_string(),
        },
        resume_provenance: ResumeProvenance {
            resumed: checkpoint_metadata.is_some(),
            resumed_from_step: checkpoint_metadata.map(|meta| meta.step),
            checkpoint_config_hash: checkpoint_metadata.and_then(|meta| meta.config_hash.clone()),
            checkpoint_dataset_fingerprint: checkpoint_metadata
                .and_then(|meta| meta.dataset_validation_fingerprint.clone()),
        },
    }
}

/// Compute a stable short hash for a serializable structure.
pub fn stable_json_hash<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_string(value).unwrap_or_else(|_| "<serde-error>".to_string());
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
