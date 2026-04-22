//! Strongly typed configuration for Phase 1 of the research framework.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Validation error raised before launching a config-driven run.
#[derive(Debug, Error)]
#[error("invalid research config: {message}")]
pub struct ConfigValidationError {
    message: String,
}

impl ConfigValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

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

impl ResearchConfig {
    /// Validate that the config encodes a runnable research workflow.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        self.data.validate()?;
        self.model.validate()?;
        self.training.validate()?;
        self.runtime.validate()?;
        self.validate_cross_section_invariants()?;
        Ok(())
    }

    fn validate_cross_section_invariants(&self) -> Result<(), ConfigValidationError> {
        if self.training.max_steps < self.training.schedule.stage3_steps {
            return Err(ConfigValidationError::new(format!(
                "training.max_steps={} must be >= schedule.stage3_steps={}",
                self.training.max_steps, self.training.schedule.stage3_steps
            )));
        }
        if self.data.dataset_format == DatasetFormat::Synthetic
            && self.data.stratify_by_measurement
            && self.data.max_examples == Some(0)
        {
            return Err(ConfigValidationError::new(
                "synthetic runs cannot request stratified measurement splits with max_examples=0",
            ));
        }
        Ok(())
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
    /// Lightweight or strict parsing behavior for on-disk assets.
    #[serde(default)]
    pub parsing_mode: ParsingMode,
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
    /// Decoder-side corruption and denoising target generation.
    #[serde(default)]
    pub generation_target: GenerationTargetConfig,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("./data"),
            dataset_format: DatasetFormat::Synthetic,
            manifest_path: None,
            label_table_path: None,
            parsing_mode: ParsingMode::Lightweight,
            max_ligand_atoms: 64,
            max_pocket_atoms: 256,
            pocket_cutoff_angstrom: 6.0,
            max_examples: None,
            batch_size: 4,
            split_seed: 42,
            val_fraction: 0.1,
            test_fraction: 0.1,
            stratify_by_measurement: false,
            generation_target: GenerationTargetConfig::default(),
        }
    }
}

impl DataConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.max_ligand_atoms == 0 {
            return Err(ConfigValidationError::new(
                "data.max_ligand_atoms must be greater than zero",
            ));
        }
        if self.max_pocket_atoms == 0 {
            return Err(ConfigValidationError::new(
                "data.max_pocket_atoms must be greater than zero",
            ));
        }
        if self.batch_size == 0 {
            return Err(ConfigValidationError::new(
                "data.batch_size must be greater than zero",
            ));
        }
        if self.pocket_cutoff_angstrom <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.pocket_cutoff_angstrom must be positive",
            ));
        }
        if let Some(limit) = self.max_examples {
            if limit == 0 {
                return Err(ConfigValidationError::new(
                    "data.max_examples must be omitted or greater than zero",
                ));
            }
        }
        validate_split_fractions(self.val_fraction, self.test_fraction)?;
        self.generation_target.validate()?;
        match self.dataset_format {
            DatasetFormat::Synthetic => {}
            DatasetFormat::ManifestJson => {
                let manifest_path = self.manifest_path.as_deref().ok_or_else(|| {
                    ConfigValidationError::new(
                        "data.manifest_path is required when data.dataset_format=manifest_json",
                    )
                })?;
                ensure_file_exists(manifest_path, "data.manifest_path")?;
            }
            DatasetFormat::PdbbindLikeDir => {
                ensure_directory_exists(&self.root_dir, "data.root_dir")?;
            }
        }
        if let Some(label_table_path) = self.label_table_path.as_deref() {
            ensure_file_exists(label_table_path, "data.label_table_path")?;
        }
        Ok(())
    }
}

/// Configurable corruption process used to derive decoder-side supervision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationTargetConfig {
    /// Fraction of ligand atoms masked for corruption recovery.
    pub atom_mask_ratio: f32,
    /// Deterministic coordinate noise scale applied to ligand atoms.
    pub coordinate_noise_std: f32,
    /// Seed used for deterministic corruption and denoising target construction.
    pub corruption_seed: u64,
}

impl Default for GenerationTargetConfig {
    fn default() -> Self {
        Self {
            atom_mask_ratio: 0.15,
            coordinate_noise_std: 0.08,
            corruption_seed: 1337,
        }
    }
}

impl GenerationTargetConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.atom_mask_ratio.is_finite() || !(0.0..=1.0).contains(&self.atom_mask_ratio) {
            return Err(ConfigValidationError::new(
                "data.generation_target.atom_mask_ratio must be finite and in [0, 1]",
            ));
        }
        if !self.coordinate_noise_std.is_finite() || self.coordinate_noise_std < 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.coordinate_noise_std must be finite and non-negative",
            ));
        }
        Ok(())
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

impl ModelConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.hidden_dim <= 0 {
            return Err(ConfigValidationError::new(
                "model.hidden_dim must be greater than zero",
            ));
        }
        if self.num_slots <= 0 {
            return Err(ConfigValidationError::new(
                "model.num_slots must be greater than zero",
            ));
        }
        if self.atom_vocab_size <= 0 || self.bond_vocab_size <= 0 {
            return Err(ConfigValidationError::new(
                "model.atom_vocab_size and model.bond_vocab_size must be greater than zero",
            ));
        }
        if self.pocket_feature_dim <= 0 || self.pair_feature_dim <= 0 {
            return Err(ConfigValidationError::new(
                "model.pocket_feature_dim and model.pair_feature_dim must be greater than zero",
            ));
        }
        Ok(())
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

impl RuntimeConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.device.trim().is_empty() {
            return Err(ConfigValidationError::new(
                "runtime.device must not be empty",
            ));
        }
        Ok(())
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
    /// Configured primary objective implementation.
    #[serde(default = "default_primary_objective")]
    pub primary_objective: PrimaryObjectiveConfig,
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
            primary_objective: PrimaryObjectiveConfig::SurrogateReconstruction,
            affinity_weighting: AffinityWeighting::None,
        }
    }
}

impl TrainingConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(ConfigValidationError::new(
                "training.learning_rate must be a finite positive value",
            ));
        }
        if self.max_steps == 0 {
            return Err(ConfigValidationError::new(
                "training.max_steps must be greater than zero",
            ));
        }
        self.schedule.validate(self.max_steps)?;
        self.loss_weights.validate()?;
        if self.checkpoint_dir.as_os_str().is_empty() {
            return Err(ConfigValidationError::new(
                "training.checkpoint_dir must not be empty",
            ));
        }
        if self.checkpoint_dir.exists() && !self.checkpoint_dir.is_dir() {
            return Err(ConfigValidationError::new(format!(
                "training.checkpoint_dir={} must be a directory when it already exists",
                self.checkpoint_dir.display()
            )));
        }
        Ok(())
    }
}

fn default_primary_objective() -> PrimaryObjectiveConfig {
    PrimaryObjectiveConfig::SurrogateReconstruction
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

impl StageScheduleConfig {
    fn validate(&self, max_steps: usize) -> Result<(), ConfigValidationError> {
        if self.stage1_steps > self.stage2_steps || self.stage2_steps > self.stage3_steps {
            return Err(ConfigValidationError::new(
                "training.schedule must satisfy stage1_steps <= stage2_steps <= stage3_steps",
            ));
        }
        if self.stage3_steps > max_steps {
            return Err(ConfigValidationError::new(format!(
                "training.schedule.stage3_steps={} must be <= training.max_steps={max_steps}",
                self.stage3_steps
            )));
        }
        Ok(())
    }
}

/// Final loss weights before scheduler warmup scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossWeightConfig {
    /// Primary objective weight.
    #[serde(alias = "alpha_task")]
    pub alpha_primary: f64,
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
            alpha_primary: 1.0,
            beta_intra_red: 0.1,
            gamma_probe: 0.2,
            delta_leak: 0.05,
            eta_gate: 0.05,
            mu_slot: 0.05,
            nu_consistency: 0.1,
        }
    }
}

impl LossWeightConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        for (name, value) in [
            ("training.loss_weights.alpha_primary", self.alpha_primary),
            ("training.loss_weights.beta_intra_red", self.beta_intra_red),
            ("training.loss_weights.gamma_probe", self.gamma_probe),
            ("training.loss_weights.delta_leak", self.delta_leak),
            ("training.loss_weights.eta_gate", self.eta_gate),
            ("training.loss_weights.mu_slot", self.mu_slot),
            ("training.loss_weights.nu_consistency", self.nu_consistency),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

/// Active primary objective for the staged trainer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PrimaryObjectiveConfig {
    /// Reconstruction-style surrogate objective over modality token paths.
    SurrogateReconstruction,
    /// Decoder-side corruption recovery over ligand atom identities and coordinates.
    ConditionedDenoising,
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

/// Parsing strictness for lightweight real-data ingestion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParsingMode {
    /// Convenience-oriented parsing with nearest-atom pocket fallback and permissive file picking.
    #[default]
    Lightweight,
    /// Fail-fast parsing that rejects ambiguous discovery and fallback pocket extraction.
    Strict,
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

fn validate_split_fractions(
    val_fraction: f32,
    test_fraction: f32,
) -> Result<(), ConfigValidationError> {
    if !val_fraction.is_finite() || !test_fraction.is_finite() {
        return Err(ConfigValidationError::new(
            "data.val_fraction and data.test_fraction must be finite",
        ));
    }
    if val_fraction < 0.0 || test_fraction < 0.0 {
        return Err(ConfigValidationError::new(
            "data.val_fraction and data.test_fraction must be non-negative",
        ));
    }
    if val_fraction + test_fraction >= 1.0 {
        return Err(ConfigValidationError::new(format!(
            "data.val_fraction + data.test_fraction must be < 1.0 (got {:.4})",
            val_fraction + test_fraction
        )));
    }
    Ok(())
}

fn ensure_file_exists(path: &Path, field_name: &str) -> Result<(), ConfigValidationError> {
    if !path.exists() {
        return Err(ConfigValidationError::new(format!(
            "{field_name}={} does not exist",
            path.display()
        )));
    }
    if !path.is_file() {
        return Err(ConfigValidationError::new(format!(
            "{field_name}={} must point to a file",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_directory_exists(path: &Path, field_name: &str) -> Result<(), ConfigValidationError> {
    if !path.exists() {
        return Err(ConfigValidationError::new(format!(
            "{field_name}={} does not exist",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(ConfigValidationError::new(format!(
            "{field_name}={} must point to a directory",
            path.display()
        )));
    }
    Ok(())
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

    #[test]
    fn validate_rejects_invalid_split_fractions() {
        let mut config = ResearchConfig::default();
        config.data.val_fraction = 0.6;
        config.data.test_fraction = 0.4;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("data.val_fraction + data.test_fraction must be < 1.0"));
    }

    #[test]
    fn validate_rejects_manifest_mode_without_manifest_path() {
        let mut config = ResearchConfig::default();
        config.data.dataset_format = DatasetFormat::ManifestJson;
        config.training.max_steps = 8;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("data.manifest_path is required when data.dataset_format=manifest_json"));
    }
}
