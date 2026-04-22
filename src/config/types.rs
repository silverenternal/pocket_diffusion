//! Strongly typed configuration for Phase 1 of the research framework.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Top-level configuration for pocket-conditioned molecular generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// Data loading and split configuration.
    pub data: DataConfig,
    /// Model architecture configuration.
    pub model: ModelConfig,
    /// Training and optimization configuration.
    pub training: TrainingConfig,
    /// Runtime and device preferences.
    pub runtime: RuntimeConfig,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            data: DataConfig::default(),
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }
}

/// Dataset and split configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Root directory for data assets.
    pub root_dir: PathBuf,
    /// Dataset source format used by the real-data loader.
    pub dataset_format: DatasetFormat,
    /// Optional manifest path for explicit sample enumeration.
    pub manifest_path: Option<PathBuf>,
    /// Optional CSV/TSV label table used to attach affinity targets.
    pub label_table_path: Option<PathBuf>,
    /// Maximum ligand atoms retained in a batch.
    pub max_ligand_atoms: usize,
    /// Maximum pocket atoms retained in a batch.
    pub max_pocket_atoms: usize,
    /// Pocket extraction cutoff radius in angstroms.
    pub pocket_cutoff_angstrom: f32,
    /// Optional limit for quick debugging runs.
    pub max_examples: Option<usize>,
    /// Batch size used by the training loader.
    pub batch_size: usize,
    /// Unseen-pocket split seed.
    pub split_seed: u64,
    /// Fraction of examples used for validation.
    pub val_fraction: f32,
    /// Fraction of examples used for test.
    pub test_fraction: f32,
    /// Whether to stratify protein-level splits by dominant affinity measurement type.
    pub stratify_by_measurement: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("./data"),
            dataset_format: DatasetFormat::Synthetic,
            manifest_path: None,
            label_table_path: None,
            max_ligand_atoms: 64,
            max_pocket_atoms: 256,
            pocket_cutoff_angstrom: 6.0,
            max_examples: None,
            batch_size: 4,
            split_seed: 42,
            val_fraction: 0.1,
            test_fraction: 0.1,
            stratify_by_measurement: false,
        }
    }
}

/// Encoder and latent architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Shared hidden size used by modality encoders.
    pub hidden_dim: i64,
    /// Slot count upper bound per modality.
    pub num_slots: i64,
    /// Atom-type vocabulary size.
    pub atom_vocab_size: i64,
    /// Bond-type vocabulary size.
    pub bond_vocab_size: i64,
    /// Input pocket feature width.
    pub pocket_feature_dim: i64,
    /// Pairwise geometric feature width.
    pub pair_feature_dim: i64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_slots: 8,
            atom_vocab_size: 32,
            bond_vocab_size: 8,
            pocket_feature_dim: 24,
            pair_feature_dim: 8,
        }
    }
}

/// Runtime preferences that affect execution but not model semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Preferred device string, such as `cpu` or `cuda:0`.
    pub device: String,
    /// Number of worker threads intended for data processing.
    pub data_workers: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: "cpu".to_string(),
            data_workers: 0,
        }
    }
}

/// End-to-end training configuration including staged-loss weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate for the optimizer.
    pub learning_rate: f64,
    /// Number of optimization steps to run in example trainers.
    pub max_steps: usize,
    /// Stage schedule.
    pub schedule: StageScheduleConfig,
    /// Loss weights used after warmup ramps are applied.
    pub loss_weights: LossWeightConfig,
    /// Checkpoint directory for trainer skeleton output.
    pub checkpoint_dir: PathBuf,
    /// Checkpoint interval in steps.
    pub checkpoint_every: usize,
    /// Logging interval in steps.
    pub log_every: usize,
    /// Weighting strategy for labeled affinity supervision.
    pub affinity_weighting: AffinityWeighting,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            max_steps: 8,
            schedule: StageScheduleConfig::default(),
            loss_weights: LossWeightConfig::default(),
            checkpoint_dir: PathBuf::from("./checkpoints"),
            checkpoint_every: 10,
            log_every: 1,
            affinity_weighting: AffinityWeighting::None,
        }
    }
}

/// Stage boundaries for gradual regularizer activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageScheduleConfig {
    /// End step of stage 1.
    pub stage1_steps: usize,
    /// End step of stage 2.
    pub stage2_steps: usize,
    /// End step of stage 3.
    pub stage3_steps: usize,
}

impl Default for StageScheduleConfig {
    fn default() -> Self {
        Self {
            stage1_steps: 2,
            stage2_steps: 4,
            stage3_steps: 6,
        }
    }
}

/// Final loss weights before scheduler warmup scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossWeightConfig {
    /// Main task objective.
    pub alpha_task: f64,
    /// Intra-modality redundancy objective.
    pub beta_intra_red: f64,
    /// Semantic probe supervision.
    pub gamma_probe: f64,
    /// Leakage control.
    pub delta_leak: f64,
    /// Gate sparsity objective.
    pub eta_gate: f64,
    /// Slot sparsity and balance objective.
    pub mu_slot: f64,
    /// Topology-geometry consistency objective.
    pub nu_consistency: f64,
}

impl Default for LossWeightConfig {
    fn default() -> Self {
        Self {
            alpha_task: 1.0,
            beta_intra_red: 0.1,
            gamma_probe: 0.2,
            delta_leak: 0.05,
            eta_gate: 0.05,
            mu_slot: 0.05,
            nu_consistency: 0.1,
        }
    }
}

/// Backing format used by the dataset loader.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFormat {
    /// Built-in synthetic fallback used for smoke tests.
    Synthetic,
    /// Explicit JSON manifest listing pocket and ligand file pairs.
    ManifestJson,
    /// Scan a PDBbind-like directory tree with per-complex subdirectories.
    PdbbindLikeDir,
}

/// Weighting mode for affinity supervision across mixed measurement families.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AffinityWeighting {
    /// Apply no extra weighting across measurement families.
    None,
    /// Weight each labeled example by the inverse frequency of its measurement family.
    InverseFrequency,
}

/// Load a research configuration from a JSON file.
pub fn load_research_config(
    path: impl Into<PathBuf>,
) -> Result<ResearchConfig, Box<dyn std::error::Error>> {
    let path = path.into();
    let content = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&content)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_research_config_from_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        fs::write(
            &path,
            r#"{
                "data": {
                    "root_dir": "./data",
                    "dataset_format": "synthetic",
                    "manifest_path": null,
                    "label_table_path": null,
                    "max_ligand_atoms": 64,
                    "max_pocket_atoms": 256,
                    "pocket_cutoff_angstrom": 6.0,
                    "max_examples": 2,
                    "batch_size": 3,
                    "split_seed": 7,
                    "val_fraction": 0.2,
                    "test_fraction": 0.2,
                    "stratify_by_measurement": false
                },
                "model": {
                    "hidden_dim": 32,
                    "num_slots": 4,
                    "atom_vocab_size": 16,
                    "bond_vocab_size": 4,
                    "pocket_feature_dim": 12,
                    "pair_feature_dim": 8
                },
                "training": {
                    "learning_rate": 0.001,
                    "max_steps": 5,
                    "schedule": {
                        "stage1_steps": 1,
                        "stage2_steps": 2,
                        "stage3_steps": 3
                    },
                    "loss_weights": {
                        "alpha_task": 1.0,
                        "beta_intra_red": 0.1,
                        "gamma_probe": 0.2,
                        "delta_leak": 0.05,
                        "eta_gate": 0.05,
                        "mu_slot": 0.05,
                        "nu_consistency": 0.1
                    },
                    "checkpoint_dir": "./checkpoints",
                    "checkpoint_every": 2,
                    "log_every": 1,
                    "affinity_weighting": "none"
                },
                "runtime": {
                    "device": "cpu",
                    "data_workers": 0
                }
            }"#,
        )
        .unwrap();

        let config = load_research_config(&path).unwrap();
        assert_eq!(config.data.batch_size, 3);
        assert_eq!(config.model.pocket_feature_dim, 12);
        assert_eq!(config.runtime.device, "cpu");
    }
}
