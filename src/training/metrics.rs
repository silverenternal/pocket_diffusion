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
pub const METRIC_SCHEMA_VERSION: u32 = 5;
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
    /// Pocket-ligand contact encouragement objective.
    #[serde(default)]
    pub pocket_contact: f64,
    /// Pocket-ligand steric-clash penalty objective.
    #[serde(default)]
    pub pocket_clash: f64,
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
    /// Conservative split-quality warnings for claim-bearing unseen-pocket runs.
    #[serde(default)]
    pub quality_checks: SplitQualityChecks,
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
    /// Histogram of ligand atom-count bins in the split.
    #[serde(default)]
    pub ligand_atom_count_bins: BTreeMap<String, usize>,
    /// Histogram of pocket atom-count bins in the split.
    #[serde(default)]
    pub pocket_atom_count_bins: BTreeMap<String, usize>,
    /// Lightweight protein-family proxy histogram derived from stable protein id prefixes.
    #[serde(default)]
    pub protein_family_proxy_histogram: BTreeMap<String, usize>,
    /// Average ligand atom count in the split.
    #[serde(default)]
    pub average_ligand_atoms: f64,
    /// Average pocket atom count in the split.
    #[serde(default)]
    pub average_pocket_atoms: f64,
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

/// Conservative split-quality checks that make weak held-out surfaces explicit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitQualityChecks {
    /// Whether validation has too few proxy protein families for claim-bearing use.
    pub weak_val_family_count: bool,
    /// Whether test has too few proxy protein families for claim-bearing use.
    pub weak_test_family_count: bool,
    /// Whether atom-count distributions are severely skewed against train.
    pub severe_atom_count_skew_detected: bool,
    /// Whether measurement-family coverage differs substantially across splits.
    pub measurement_family_skew_detected: bool,
    /// Human-readable warnings suitable for persisted audit artifacts.
    pub warnings: Vec<String>,
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
        let quality_checks = SplitQualityChecks::from_stats(&train_stats, &val_stats, &test_stats);

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
            quality_checks,
        }
    }
}

impl SplitQualityChecks {
    fn from_stats(train: &SplitStats, val: &SplitStats, test: &SplitStats) -> Self {
        const MIN_HELDOUT_FAMILIES: usize = 3;
        let weak_val_family_count = val.protein_family_proxy_histogram.len() < MIN_HELDOUT_FAMILIES;
        let weak_test_family_count =
            test.protein_family_proxy_histogram.len() < MIN_HELDOUT_FAMILIES;
        let severe_atom_count_skew_detected =
            severe_ratio_skew(train.average_ligand_atoms, val.average_ligand_atoms)
                || severe_ratio_skew(train.average_ligand_atoms, test.average_ligand_atoms)
                || severe_ratio_skew(train.average_pocket_atoms, val.average_pocket_atoms)
                || severe_ratio_skew(train.average_pocket_atoms, test.average_pocket_atoms);
        let measurement_family_skew_detected = !histogram_keys_cover(
            &train.dominant_measurement_histogram,
            &val.dominant_measurement_histogram,
        ) || !histogram_keys_cover(
            &train.dominant_measurement_histogram,
            &test.dominant_measurement_histogram,
        );

        let mut warnings = Vec::new();
        if weak_val_family_count {
            warnings.push(format!(
                "validation split has {} proxy protein families; claim-bearing runs should have at least {MIN_HELDOUT_FAMILIES}",
                val.protein_family_proxy_histogram.len()
            ));
        }
        if weak_test_family_count {
            warnings.push(format!(
                "test split has {} proxy protein families; claim-bearing runs should have at least {MIN_HELDOUT_FAMILIES}",
                test.protein_family_proxy_histogram.len()
            ));
        }
        if severe_atom_count_skew_detected {
            warnings.push(
                "held-out ligand or pocket atom-count averages differ from train by more than 3x"
                    .to_string(),
            );
        }
        if measurement_family_skew_detected {
            warnings.push(
                "validation or test measurement-family labels are not covered by the train split"
                    .to_string(),
            );
        }

        Self {
            weak_val_family_count,
            weak_test_family_count,
            severe_atom_count_skew_detected,
            measurement_family_skew_detected,
            warnings,
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
        let mut ligand_atom_count_bins = BTreeMap::new();
        let mut pocket_atom_count_bins = BTreeMap::new();
        let mut protein_family_proxy_histogram = BTreeMap::new();
        let mut ligand_atoms_total = 0usize;
        let mut pocket_atoms_total = 0usize;
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
            let ligand_atoms = example
                .topology
                .atom_types
                .size()
                .first()
                .copied()
                .unwrap_or(0)
                .max(0) as usize;
            let pocket_atoms = example
                .pocket
                .coords
                .size()
                .first()
                .copied()
                .unwrap_or(0)
                .max(0) as usize;
            ligand_atoms_total += ligand_atoms;
            pocket_atoms_total += pocket_atoms;
            *ligand_atom_count_bins
                .entry(atom_count_bin(ligand_atoms))
                .or_default() += 1;
            *pocket_atom_count_bins
                .entry(atom_count_bin(pocket_atoms))
                .or_default() += 1;
            *protein_family_proxy_histogram
                .entry(protein_family_proxy(&example.protein_id))
                .or_default() += 1;
        }
        let average_ligand_atoms = if example_count == 0 {
            0.0
        } else {
            ligand_atoms_total as f64 / example_count as f64
        };
        let average_pocket_atoms = if example_count == 0 {
            0.0
        } else {
            pocket_atoms_total as f64 / example_count as f64
        };
        Self {
            example_count,
            unique_protein_count,
            labeled_example_count,
            labeled_fraction,
            dominant_measurement_histogram,
            ligand_atom_count_bins,
            pocket_atom_count_bins,
            protein_family_proxy_histogram,
            average_ligand_atoms,
            average_pocket_atoms,
        }
    }
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

fn severe_ratio_skew(reference: f64, candidate: f64) -> bool {
    if reference <= 0.0 || candidate <= 0.0 {
        return false;
    }
    let ratio = if reference > candidate {
        reference / candidate
    } else {
        candidate / reference
    };
    ratio > 3.0
}

fn histogram_keys_cover(
    train: &BTreeMap<String, usize>,
    heldout: &BTreeMap<String, usize>,
) -> bool {
    heldout
        .keys()
        .filter(|key| key.as_str() != "unknown")
        .all(|key| train.contains_key(key))
}

fn protein_family_proxy(protein_id: &str) -> String {
    protein_id
        .split(|ch: char| ch == '_' || ch == '-' || ch == ':' || ch == '.')
        .next()
        .filter(|part| !part.is_empty())
        .unwrap_or("unknown")
        .to_string()
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
        determinism_controls: DeterminismControls {
            split_seed: config.data.split_seed,
            corruption_seed: config.data.generation_target.corruption_seed,
            sampling_seed: config.data.generation_target.sampling_seed,
            device: config.runtime.device.clone(),
            data_workers: config.runtime.data_workers,
            tch_intra_op_threads: config.runtime.tch_intra_op_threads,
            tch_inter_op_threads: config.runtime.tch_inter_op_threads,
        },
        replay_tolerance: ReplayTolerance::default(),
        resume_contract: ResumeContract {
            version: RESUME_CONTRACT_VERSION.to_string(),
            restores_model_weights: true,
            restores_step: true,
            restores_history: true,
            restores_optimizer_state: true,
            continuity_mode: ResumeContinuityMode::MetadataOnlyContinuation,
            supports_strict_replay: false,
            notes: "Resume restores model weights, step index, prior persisted training history, and any checkpointed optimizer/scheduler metadata. The underlying tch Adam moment buffers are not persisted, so this remains a bounded reproducibility aid rather than a strict deterministic replay.".to_string(),
        },
        resume_provenance: ResumeProvenance {
            resumed: checkpoint_metadata.is_some(),
            resumed_from_step: checkpoint_metadata.map(|meta| meta.step),
            checkpoint_config_hash: checkpoint_metadata.and_then(|meta| meta.config_hash.clone()),
            checkpoint_dataset_fingerprint: checkpoint_metadata
                .and_then(|meta| meta.dataset_validation_fingerprint.clone()),
            restored_optimizer_state_metadata: checkpoint_metadata
                .and_then(|meta| meta.optimizer_state.as_ref())
                .is_some(),
            restored_scheduler_state_metadata: checkpoint_metadata
                .and_then(|meta| meta.scheduler_state.as_ref())
                .is_some(),
            continuity_mode: if checkpoint_metadata.is_none() {
                ResumeContinuityMode::FreshRun
            } else {
                ResumeContinuityMode::MetadataOnlyContinuation
            },
            strict_replay_achieved: false,
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
