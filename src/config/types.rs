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
    /// Optional bounded automated search over interaction, rollout, and loss controls.
    #[serde(default)]
    pub automated_search: AutomatedSearchConfig,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            data: DataConfig::default(),
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            runtime: RuntimeConfig::default(),
            automated_search: AutomatedSearchConfig::default(),
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
        self.automated_search.validate()?;
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
    /// Optional inclusion/exclusion filters for real-data evidence surfaces.
    #[serde(default)]
    pub quality_filters: DataQualityFilterConfig,
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
            quality_filters: DataQualityFilterConfig::default(),
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
        self.quality_filters.validate()?;
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

/// Optional dataset quality filters used to make real-data inclusion criteria reproducible.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataQualityFilterConfig {
    /// Minimum retained labeled fraction required after filtering.
    #[serde(default)]
    pub min_label_coverage: Option<f32>,
    /// Maximum allowed pocket-fallback fraction before the load is rejected.
    #[serde(default)]
    pub max_fallback_fraction: Option<f32>,
    /// Optional atom-count exclusion threshold for parsed ligands.
    #[serde(default)]
    pub max_ligand_atoms: Option<usize>,
    /// Optional atom-count exclusion threshold for parsed pockets.
    #[serde(default)]
    pub max_pocket_atoms: Option<usize>,
    /// Require source protein and ligand structure paths to be retained on examples.
    #[serde(default)]
    pub require_source_structure_provenance: bool,
    /// Require labeled examples to retain measurement-family and normalization metadata.
    #[serde(default)]
    pub require_affinity_metadata: bool,
    /// Maximum retained fraction of approximate measurement families such as `IC50` or `EC50`.
    #[serde(default)]
    pub max_approximate_label_fraction: Option<f32>,
    /// Minimum retained coverage of normalization provenance on labeled examples.
    #[serde(default)]
    pub min_normalization_provenance_coverage: Option<f32>,
}

impl DataQualityFilterConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_optional_fraction(
            self.min_label_coverage,
            "data.quality_filters.min_label_coverage",
        )?;
        validate_optional_fraction(
            self.max_fallback_fraction,
            "data.quality_filters.max_fallback_fraction",
        )?;
        validate_optional_fraction(
            self.max_approximate_label_fraction,
            "data.quality_filters.max_approximate_label_fraction",
        )?;
        validate_optional_fraction(
            self.min_normalization_provenance_coverage,
            "data.quality_filters.min_normalization_provenance_coverage",
        )?;
        if self.max_ligand_atoms == Some(0) {
            return Err(ConfigValidationError::new(
                "data.quality_filters.max_ligand_atoms must be omitted or greater than zero",
            ));
        }
        if self.max_pocket_atoms == Some(0) {
            return Err(ConfigValidationError::new(
                "data.quality_filters.max_pocket_atoms must be omitted or greater than zero",
            ));
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
    /// Number of iterative decoder refinement steps used by active generation paths.
    pub rollout_steps: usize,
    /// Minimum number of refinement steps before the stop logit may terminate rollout.
    pub min_rollout_steps: usize,
    /// Sigmoid threshold used to terminate iterative generation early.
    pub stop_probability_threshold: f64,
    /// Scalar applied to decoder coordinate updates during iterative refinement.
    pub coordinate_step_scale: f64,
    /// Geometric decay used to weight later rollout steps during training.
    pub training_step_weight_decay: f64,
    /// Rollout update semantics used by the iterative decoder.
    #[serde(default)]
    pub rollout_mode: GenerationRolloutMode,
    /// Momentum used to smooth coordinate updates in stronger rollout mode.
    #[serde(default = "default_coordinate_momentum")]
    pub coordinate_momentum: f64,
    /// Momentum used to smooth atom-type logits in stronger rollout mode.
    #[serde(default = "default_atom_momentum")]
    pub atom_momentum: f64,
    /// Temperature applied before committing atom-type updates in stronger rollout mode.
    #[serde(default = "default_atom_commit_temperature")]
    pub atom_commit_temperature: f64,
    /// Maximum L2 norm allowed for one atom coordinate delta before scaling.
    #[serde(default = "default_max_coordinate_delta_norm")]
    pub max_coordinate_delta_norm: f64,
    /// Stability threshold used by adaptive stopping in stronger rollout mode.
    #[serde(default = "default_stop_delta_threshold")]
    pub stop_delta_threshold: f64,
    /// Number of consecutive stable steps required before adaptive stopping may trigger.
    #[serde(default = "default_stop_patience")]
    pub stop_patience: usize,
    /// Seed used by reproducible rollout sampling controls.
    #[serde(default = "default_sampling_seed")]
    pub sampling_seed: u64,
    /// Atom sampling temperature; zero preserves deterministic argmax commits.
    #[serde(default)]
    pub sampling_temperature: f64,
    /// Optional top-k truncation for stochastic atom commits; zero disables top-k filtering.
    #[serde(default)]
    pub sampling_top_k: usize,
    /// Top-p nucleus threshold for stochastic atom commits.
    #[serde(default = "default_sampling_top_p")]
    pub sampling_top_p: f64,
    /// Deterministic coordinate noise scale added during stochastic rollout.
    #[serde(default)]
    pub coordinate_sampling_noise_std: f64,
    /// Multiplier for decoder-time pocket-centroid guidance during rollout.
    #[serde(default = "default_pocket_guidance_scale")]
    pub pocket_guidance_scale: f64,
}

impl Default for GenerationTargetConfig {
    fn default() -> Self {
        Self {
            atom_mask_ratio: 0.15,
            coordinate_noise_std: 0.08,
            corruption_seed: 1337,
            rollout_steps: 4,
            min_rollout_steps: 2,
            stop_probability_threshold: 0.82,
            coordinate_step_scale: 0.8,
            training_step_weight_decay: 0.9,
            rollout_mode: GenerationRolloutMode::default(),
            coordinate_momentum: default_coordinate_momentum(),
            atom_momentum: default_atom_momentum(),
            atom_commit_temperature: default_atom_commit_temperature(),
            max_coordinate_delta_norm: default_max_coordinate_delta_norm(),
            stop_delta_threshold: default_stop_delta_threshold(),
            stop_patience: default_stop_patience(),
            sampling_seed: default_sampling_seed(),
            sampling_temperature: 0.0,
            sampling_top_k: 0,
            sampling_top_p: default_sampling_top_p(),
            coordinate_sampling_noise_std: 0.0,
            pocket_guidance_scale: default_pocket_guidance_scale(),
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
        if self.rollout_steps == 0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.rollout_steps must be greater than zero",
            ));
        }
        if self.min_rollout_steps > self.rollout_steps {
            return Err(ConfigValidationError::new(
                "data.generation_target.min_rollout_steps must be <= data.generation_target.rollout_steps",
            ));
        }
        if !self.stop_probability_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.stop_probability_threshold)
        {
            return Err(ConfigValidationError::new(
                "data.generation_target.stop_probability_threshold must be finite and in [0, 1]",
            ));
        }
        if !self.coordinate_step_scale.is_finite() || self.coordinate_step_scale <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.coordinate_step_scale must be finite and positive",
            ));
        }
        if !self.training_step_weight_decay.is_finite() || self.training_step_weight_decay <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.training_step_weight_decay must be finite and positive",
            ));
        }
        if !(0.0..1.0).contains(&self.coordinate_momentum) {
            return Err(ConfigValidationError::new(
                "data.generation_target.coordinate_momentum must be in [0, 1)",
            ));
        }
        if !(0.0..1.0).contains(&self.atom_momentum) {
            return Err(ConfigValidationError::new(
                "data.generation_target.atom_momentum must be in [0, 1)",
            ));
        }
        if !self.atom_commit_temperature.is_finite() || self.atom_commit_temperature <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.atom_commit_temperature must be finite and positive",
            ));
        }
        if !self.max_coordinate_delta_norm.is_finite() || self.max_coordinate_delta_norm <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.max_coordinate_delta_norm must be finite and positive",
            ));
        }
        if !self.stop_delta_threshold.is_finite() || self.stop_delta_threshold < 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.stop_delta_threshold must be finite and non-negative",
            ));
        }
        if self.stop_patience == 0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.stop_patience must be greater than zero",
            ));
        }
        if !self.sampling_temperature.is_finite() || self.sampling_temperature < 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.sampling_temperature must be finite and non-negative",
            ));
        }
        if !self.sampling_top_p.is_finite() || !(0.0..=1.0).contains(&self.sampling_top_p) {
            return Err(ConfigValidationError::new(
                "data.generation_target.sampling_top_p must be finite and in [0, 1]",
            ));
        }
        if self.sampling_temperature > 0.0 && self.sampling_top_p <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.sampling_top_p must be > 0 when sampling_temperature > 0",
            ));
        }
        if !self.coordinate_sampling_noise_std.is_finite()
            || self.coordinate_sampling_noise_std < 0.0
        {
            return Err(ConfigValidationError::new(
                "data.generation_target.coordinate_sampling_noise_std must be finite and non-negative",
            ));
        }
        if !self.pocket_guidance_scale.is_finite() || self.pocket_guidance_scale < 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.pocket_guidance_scale must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Rollout update semantics for the iterative conditioned generator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GenerationRolloutMode {
    /// Preserve the original direct-update refinement loop as an ablation baseline.
    #[default]
    Lightweight,
    /// Use momentum-smoothed atom/geometry updates with stability-aware stopping.
    MomentumRefine,
}

fn default_coordinate_momentum() -> f64 {
    0.55
}

fn default_atom_momentum() -> f64 {
    0.4
}

fn default_atom_commit_temperature() -> f64 {
    0.9
}

fn default_max_coordinate_delta_norm() -> f64 {
    1.5
}

fn default_stop_delta_threshold() -> f64 {
    0.02
}

fn default_stop_patience() -> usize {
    2
}

fn default_sampling_seed() -> u64 {
    2027
}

fn default_sampling_top_p() -> f64 {
    1.0
}

fn default_pocket_guidance_scale() -> f64 {
    1.0
}

/// Command-line backend adapter configuration used for external chemistry or docking hooks.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExternalBackendCommandConfig {
    /// Whether the backend should be invoked for the current run.
    pub enabled: bool,
    /// Executable path or command name.
    pub executable: Option<String>,
    /// Static argument list appended after the generated input/output paths.
    #[serde(default)]
    pub args: Vec<String>,
}

impl ExternalBackendCommandConfig {
    /// Validate that enabled backends provide an executable.
    pub fn validate(&self, logical_name: &str) -> Result<(), ConfigValidationError> {
        if self.enabled
            && self
                .executable
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
        {
            return Err(ConfigValidationError::new(format!(
                "{logical_name}.executable is required when {logical_name}.enabled=true"
            )));
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
    /// Cross-modality interaction block style.
    #[serde(default)]
    pub interaction_mode: CrossAttentionMode,
    /// Feed-forward expansion factor used by the Transformer-style interaction block.
    #[serde(default = "default_interaction_ff_multiplier")]
    pub interaction_ff_multiplier: i64,
    /// Compact tuning knobs for the Transformer-style interaction refinement path.
    #[serde(default)]
    pub interaction_tuning: InteractionTuningConfig,
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
            interaction_mode: CrossAttentionMode::default(),
            interaction_ff_multiplier: default_interaction_ff_multiplier(),
            interaction_tuning: InteractionTuningConfig::default(),
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
        if self.interaction_ff_multiplier <= 0 {
            return Err(ConfigValidationError::new(
                "model.interaction_ff_multiplier must be greater than zero",
            ));
        }
        self.interaction_tuning.validate()?;
        Ok(())
    }
}

/// Controlled cross-modality interaction style used by the modular stack.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CrossAttentionMode {
    /// Preserve the original gated single-path interaction as the main ablation baseline.
    Lightweight,
    /// Add normalization, residual structure, and feed-forward refinement around gated attention.
    #[default]
    Transformer,
}

fn default_interaction_ff_multiplier() -> i64 {
    2
}

/// Compact tuning controls for the Transformer-style controlled-interaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionTuningConfig {
    /// Temperature applied to gate logits before the sigmoid.
    #[serde(default = "default_interaction_gate_temperature")]
    pub gate_temperature: f64,
    /// Additive bias applied to gate logits before temperature scaling.
    #[serde(default = "default_interaction_gate_bias")]
    pub gate_bias: f64,
    /// Residual scale applied to the gated attention update.
    #[serde(default = "default_attention_residual_scale")]
    pub attention_residual_scale: f64,
    /// Residual scale applied to the Transformer-style feed-forward refinement.
    #[serde(default = "default_ffn_residual_scale")]
    pub ffn_residual_scale: f64,
    /// Whether the refinement feed-forward block should use pre-normalized inputs.
    #[serde(default = "default_transformer_pre_norm")]
    pub transformer_pre_norm: bool,
    /// Multiplier for ligand-pocket geometry bias injected into controlled attention.
    #[serde(default = "default_geometry_attention_bias_scale")]
    pub geometry_attention_bias_scale: f64,
}

impl Default for InteractionTuningConfig {
    fn default() -> Self {
        Self {
            gate_temperature: default_interaction_gate_temperature(),
            gate_bias: default_interaction_gate_bias(),
            attention_residual_scale: default_attention_residual_scale(),
            ffn_residual_scale: default_ffn_residual_scale(),
            transformer_pre_norm: default_transformer_pre_norm(),
            geometry_attention_bias_scale: default_geometry_attention_bias_scale(),
        }
    }
}

impl InteractionTuningConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.gate_temperature.is_finite() || self.gate_temperature <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.gate_temperature must be finite and positive",
            ));
        }
        if !self.gate_bias.is_finite() {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.gate_bias must be finite",
            ));
        }
        if !self.attention_residual_scale.is_finite() || self.attention_residual_scale <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.attention_residual_scale must be finite and positive",
            ));
        }
        if !self.ffn_residual_scale.is_finite() || self.ffn_residual_scale <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.ffn_residual_scale must be finite and positive",
            ));
        }
        if !self.geometry_attention_bias_scale.is_finite()
            || self.geometry_attention_bias_scale < 0.0
        {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.geometry_attention_bias_scale must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

fn default_interaction_gate_temperature() -> f64 {
    1.35
}

fn default_interaction_gate_bias() -> f64 {
    -0.1
}

fn default_attention_residual_scale() -> f64 {
    0.7
}

fn default_ffn_residual_scale() -> f64 {
    0.35
}

fn default_transformer_pre_norm() -> bool {
    true
}

fn default_geometry_attention_bias_scale() -> f64 {
    1.0
}

/// Runtime preferences that affect execution but not model semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Preferred device string, such as `cpu` or `cuda:0`.
    pub device: String,
    /// Number of worker threads intended for data processing.
    pub data_workers: usize,
    /// Optional libtorch intra-op thread count for tensor kernels.
    #[serde(default)]
    pub tch_intra_op_threads: Option<i32>,
    /// Optional libtorch inter-op thread count for parallel operator scheduling.
    #[serde(default)]
    pub tch_inter_op_threads: Option<i32>,
}

/// Bounded cross-surface search strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AutomatedSearchStrategy {
    /// Exhaust the bounded value fragments until `max_candidates` is reached.
    #[default]
    Grid,
    /// Sample bounded value fragments uniformly at random.
    Random,
}

/// Config-driven automated search over interaction, rollout, and loss controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchConfig {
    /// Whether the automated search entrypoint should be active for this config.
    #[serde(default)]
    pub enabled: bool,
    /// Search strategy used to enumerate candidate settings.
    #[serde(default)]
    pub strategy: AutomatedSearchStrategy,
    /// Whether to always include the base config as a scored candidate.
    #[serde(default = "default_include_base_candidate")]
    pub include_base_candidate: bool,
    /// Hard cap on generated candidates.
    #[serde(default = "default_search_max_candidates")]
    pub max_candidates: usize,
    /// Seed used by random search.
    #[serde(default = "default_search_seed")]
    pub random_seed: u64,
    /// Surface configs scored together during one search cycle.
    #[serde(default)]
    pub surface_configs: Vec<PathBuf>,
    /// Root directory for ranked search artifacts.
    #[serde(default = "default_search_artifact_root")]
    pub artifact_root_dir: PathBuf,
    /// Search-time hard regression gates.
    #[serde(default)]
    pub hard_gates: AutomatedSearchHardGateConfig,
    /// Multi-objective weights used after hard gates pass.
    #[serde(default)]
    pub score_weights: AutomatedSearchScoreWeightConfig,
    /// Bounded search space over explainable knobs.
    #[serde(default)]
    pub search_space: AutomatedSearchSpaceConfig,
}

impl Default for AutomatedSearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: AutomatedSearchStrategy::default(),
            include_base_candidate: default_include_base_candidate(),
            max_candidates: default_search_max_candidates(),
            random_seed: default_search_seed(),
            surface_configs: Vec::new(),
            artifact_root_dir: default_search_artifact_root(),
            hard_gates: AutomatedSearchHardGateConfig::default(),
            score_weights: AutomatedSearchScoreWeightConfig::default(),
            search_space: AutomatedSearchSpaceConfig::default(),
        }
    }
}

impl AutomatedSearchConfig {
    /// Validate the search config before launching a search cycle.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.enabled {
            return Ok(());
        }
        if self.max_candidates == 0 {
            return Err(ConfigValidationError::new(
                "automated_search.max_candidates must be greater than zero when enabled",
            ));
        }
        if self.surface_configs.is_empty() {
            return Err(ConfigValidationError::new(
                "automated_search.surface_configs must not be empty when enabled",
            ));
        }
        if self.artifact_root_dir.as_os_str().is_empty() {
            return Err(ConfigValidationError::new(
                "automated_search.artifact_root_dir must not be empty when enabled",
            ));
        }
        self.hard_gates.validate()?;
        self.score_weights.validate()?;
        self.search_space.validate()?;
        Ok(())
    }
}

/// Hard regression gates applied before multi-objective ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchHardGateConfig {
    /// Minimum chemistry-valid fraction across all surfaces.
    #[serde(default = "default_min_candidate_valid_fraction")]
    pub minimum_candidate_valid_fraction: f64,
    /// Minimum sanitization fraction across all surfaces.
    #[serde(default = "default_min_sanitized_fraction")]
    pub minimum_sanitized_fraction: f64,
    /// Minimum uniqueness fraction across all surfaces.
    #[serde(default = "default_min_unique_smiles_fraction")]
    pub minimum_unique_smiles_fraction: f64,
    /// Maximum clash fraction across all surfaces.
    #[serde(default = "default_max_clash_fraction")]
    pub maximum_clash_fraction: f64,
    /// Minimum strict pocket-fit score across all surfaces.
    #[serde(default = "default_min_strict_pocket_fit_score")]
    pub minimum_strict_pocket_fit_score: f64,
    /// Minimum pocket-contact fraction across all surfaces.
    #[serde(default = "default_min_pocket_contact_fraction")]
    pub minimum_pocket_contact_fraction: f64,
    /// Minimum pocket-compatibility fraction across all surfaces.
    #[serde(default = "default_min_pocket_compatibility_fraction")]
    pub minimum_pocket_compatibility_fraction: f64,
    /// Optional maximum raw-rollout centroid offset before repair/scoring.
    #[serde(default)]
    pub maximum_raw_centroid_offset: Option<f64>,
    /// Optional maximum raw-rollout clash proxy before repair/scoring.
    #[serde(default)]
    pub maximum_raw_clash_fraction: Option<f64>,
    /// Optional maximum final raw-rollout coordinate displacement.
    #[serde(default)]
    pub maximum_raw_mean_displacement: Option<f64>,
    /// Optional maximum final raw-rollout atom-change fraction.
    #[serde(default)]
    pub maximum_raw_atom_change_fraction: Option<f64>,
    /// Optional minimum raw-rollout uniqueness proxy.
    #[serde(default)]
    pub minimum_raw_uniqueness_proxy_fraction: Option<f64>,
}

impl Default for AutomatedSearchHardGateConfig {
    fn default() -> Self {
        Self {
            minimum_candidate_valid_fraction: default_min_candidate_valid_fraction(),
            minimum_sanitized_fraction: default_min_sanitized_fraction(),
            minimum_unique_smiles_fraction: default_min_unique_smiles_fraction(),
            maximum_clash_fraction: default_max_clash_fraction(),
            minimum_strict_pocket_fit_score: default_min_strict_pocket_fit_score(),
            minimum_pocket_contact_fraction: default_min_pocket_contact_fraction(),
            minimum_pocket_compatibility_fraction: default_min_pocket_compatibility_fraction(),
            maximum_raw_centroid_offset: None,
            maximum_raw_clash_fraction: None,
            maximum_raw_mean_displacement: None,
            maximum_raw_atom_change_fraction: None,
            minimum_raw_uniqueness_proxy_fraction: None,
        }
    }
}

impl AutomatedSearchHardGateConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        for (name, value) in [
            (
                "automated_search.hard_gates.minimum_candidate_valid_fraction",
                self.minimum_candidate_valid_fraction,
            ),
            (
                "automated_search.hard_gates.minimum_sanitized_fraction",
                self.minimum_sanitized_fraction,
            ),
            (
                "automated_search.hard_gates.minimum_unique_smiles_fraction",
                self.minimum_unique_smiles_fraction,
            ),
            (
                "automated_search.hard_gates.minimum_strict_pocket_fit_score",
                self.minimum_strict_pocket_fit_score,
            ),
            (
                "automated_search.hard_gates.minimum_pocket_contact_fraction",
                self.minimum_pocket_contact_fraction,
            ),
            (
                "automated_search.hard_gates.minimum_pocket_compatibility_fraction",
                self.minimum_pocket_compatibility_fraction,
            ),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigValidationError::new(format!(
                    "{name} must be finite and in [0, 1]"
                )));
            }
        }
        if !self.maximum_clash_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_clash_fraction)
        {
            return Err(ConfigValidationError::new(
                "automated_search.hard_gates.maximum_clash_fraction must be finite and in [0, 1]",
            ));
        }
        for (name, value) in [
            (
                "automated_search.hard_gates.maximum_raw_clash_fraction",
                self.maximum_raw_clash_fraction,
            ),
            (
                "automated_search.hard_gates.maximum_raw_atom_change_fraction",
                self.maximum_raw_atom_change_fraction,
            ),
            (
                "automated_search.hard_gates.minimum_raw_uniqueness_proxy_fraction",
                self.minimum_raw_uniqueness_proxy_fraction,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    return Err(ConfigValidationError::new(format!(
                        "{name} must be finite and in [0, 1] when set"
                    )));
                }
            }
        }
        for (name, value) in [
            (
                "automated_search.hard_gates.maximum_raw_centroid_offset",
                self.maximum_raw_centroid_offset,
            ),
            (
                "automated_search.hard_gates.maximum_raw_mean_displacement",
                self.maximum_raw_mean_displacement,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value < 0.0 {
                    return Err(ConfigValidationError::new(format!(
                        "{name} must be finite and non-negative when set"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Multi-objective scoring weights used after hard-gate filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchScoreWeightConfig {
    /// Weight for chemistry-valid and sanitized quality.
    #[serde(default = "default_chemistry_score_weight")]
    pub chemistry: f64,
    /// Weight for uniqueness and diversity.
    #[serde(default = "default_uniqueness_score_weight")]
    pub uniqueness: f64,
    /// Weight for geometric fit.
    #[serde(default = "default_geometry_score_weight")]
    pub geometry: f64,
    /// Weight for pocket-contact and compatibility behavior.
    #[serde(default = "default_pocket_score_weight")]
    pub pocket: f64,
    /// Weight for representation specialization quality.
    #[serde(default = "default_specialization_score_weight")]
    pub specialization: f64,
    /// Weight for slot and gate utilization quality.
    #[serde(default = "default_utilization_score_weight")]
    pub utilization: f64,
    /// Weight for aggregate cross-surface interaction review margin.
    #[serde(default = "default_interaction_review_score_weight")]
    pub interaction_review: f64,
}

impl Default for AutomatedSearchScoreWeightConfig {
    fn default() -> Self {
        Self {
            chemistry: default_chemistry_score_weight(),
            uniqueness: default_uniqueness_score_weight(),
            geometry: default_geometry_score_weight(),
            pocket: default_pocket_score_weight(),
            specialization: default_specialization_score_weight(),
            utilization: default_utilization_score_weight(),
            interaction_review: default_interaction_review_score_weight(),
        }
    }
}

impl AutomatedSearchScoreWeightConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        for (name, value) in [
            ("automated_search.score_weights.chemistry", self.chemistry),
            ("automated_search.score_weights.uniqueness", self.uniqueness),
            ("automated_search.score_weights.geometry", self.geometry),
            ("automated_search.score_weights.pocket", self.pocket),
            (
                "automated_search.score_weights.specialization",
                self.specialization,
            ),
            (
                "automated_search.score_weights.utilization",
                self.utilization,
            ),
            (
                "automated_search.score_weights.interaction_review",
                self.interaction_review,
            ),
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

/// Bounded search space over explainable interaction, rollout, and loss knobs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomatedSearchSpaceConfig {
    /// Candidate temperatures for gate sharpening or smoothing.
    #[serde(default)]
    pub gate_temperature: Vec<f64>,
    /// Candidate additive gate biases.
    #[serde(default)]
    pub gate_bias: Vec<f64>,
    /// Candidate residual scales for controlled attention updates.
    #[serde(default)]
    pub attention_residual_scale: Vec<f64>,
    /// Candidate residual scales for Transformer feed-forward refinement.
    #[serde(default)]
    pub ffn_residual_scale: Vec<f64>,
    /// Candidate rollout lengths.
    #[serde(default)]
    pub rollout_steps: Vec<usize>,
    /// Candidate minimum rollout lengths before stopping.
    #[serde(default)]
    pub min_rollout_steps: Vec<usize>,
    /// Candidate early-stop thresholds.
    #[serde(default)]
    pub stop_probability_threshold: Vec<f64>,
    /// Candidate coordinate update scales.
    #[serde(default)]
    pub coordinate_step_scale: Vec<f64>,
    /// Candidate per-step weighting decay values.
    #[serde(default)]
    pub training_step_weight_decay: Vec<f64>,
    /// Candidate coordinate momentum values.
    #[serde(default)]
    pub coordinate_momentum: Vec<f64>,
    /// Candidate atom-logit momentum values.
    #[serde(default)]
    pub atom_momentum: Vec<f64>,
    /// Candidate atom commit temperatures.
    #[serde(default)]
    pub atom_commit_temperature: Vec<f64>,
    /// Candidate coordinate delta clipping norms.
    #[serde(default)]
    pub max_coordinate_delta_norm: Vec<f64>,
    /// Candidate stability thresholds for adaptive stopping.
    #[serde(default)]
    pub stop_delta_threshold: Vec<f64>,
    /// Candidate stop patience values.
    #[serde(default)]
    pub stop_patience: Vec<usize>,
    /// Candidate intra-modality redundancy weights.
    #[serde(default)]
    pub beta_intra_red: Vec<f64>,
    /// Candidate semantic probe weights.
    #[serde(default)]
    pub gamma_probe: Vec<f64>,
    /// Candidate leakage weights.
    #[serde(default)]
    pub delta_leak: Vec<f64>,
    /// Candidate gate sparsity weights.
    #[serde(default)]
    pub eta_gate: Vec<f64>,
    /// Candidate slot sparsity weights.
    #[serde(default)]
    pub mu_slot: Vec<f64>,
}

impl AutomatedSearchSpaceConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        let mut configured_axes = 0usize;
        for (name, values, positive_required, unit_interval_required) in [
            (
                "automated_search.search_space.gate_temperature",
                &self.gate_temperature,
                true,
                false,
            ),
            (
                "automated_search.search_space.gate_bias",
                &self.gate_bias,
                false,
                false,
            ),
            (
                "automated_search.search_space.attention_residual_scale",
                &self.attention_residual_scale,
                true,
                false,
            ),
            (
                "automated_search.search_space.ffn_residual_scale",
                &self.ffn_residual_scale,
                true,
                false,
            ),
            (
                "automated_search.search_space.stop_probability_threshold",
                &self.stop_probability_threshold,
                false,
                true,
            ),
            (
                "automated_search.search_space.coordinate_step_scale",
                &self.coordinate_step_scale,
                true,
                false,
            ),
            (
                "automated_search.search_space.training_step_weight_decay",
                &self.training_step_weight_decay,
                true,
                false,
            ),
            (
                "automated_search.search_space.coordinate_momentum",
                &self.coordinate_momentum,
                false,
                true,
            ),
            (
                "automated_search.search_space.atom_momentum",
                &self.atom_momentum,
                false,
                true,
            ),
            (
                "automated_search.search_space.atom_commit_temperature",
                &self.atom_commit_temperature,
                true,
                false,
            ),
            (
                "automated_search.search_space.max_coordinate_delta_norm",
                &self.max_coordinate_delta_norm,
                true,
                false,
            ),
            (
                "automated_search.search_space.stop_delta_threshold",
                &self.stop_delta_threshold,
                false,
                false,
            ),
            (
                "automated_search.search_space.beta_intra_red",
                &self.beta_intra_red,
                false,
                false,
            ),
            (
                "automated_search.search_space.gamma_probe",
                &self.gamma_probe,
                false,
                false,
            ),
            (
                "automated_search.search_space.delta_leak",
                &self.delta_leak,
                false,
                false,
            ),
            (
                "automated_search.search_space.eta_gate",
                &self.eta_gate,
                false,
                false,
            ),
            (
                "automated_search.search_space.mu_slot",
                &self.mu_slot,
                false,
                false,
            ),
        ] {
            configured_axes += validate_f64_search_values(
                name,
                values,
                positive_required,
                unit_interval_required,
            )?;
        }
        for (name, values, positive_required) in [
            (
                "automated_search.search_space.rollout_steps",
                &self.rollout_steps,
                true,
            ),
            (
                "automated_search.search_space.min_rollout_steps",
                &self.min_rollout_steps,
                false,
            ),
            (
                "automated_search.search_space.stop_patience",
                &self.stop_patience,
                true,
            ),
        ] {
            configured_axes += validate_usize_search_values(name, values, positive_required)?;
        }
        if configured_axes == 0 {
            return Err(ConfigValidationError::new(
                "automated_search.search_space must configure at least one bounded axis when enabled",
            ));
        }
        Ok(())
    }
}

fn default_include_base_candidate() -> bool {
    true
}

fn default_search_max_candidates() -> usize {
    16
}

fn default_search_seed() -> u64 {
    20260423
}

fn default_search_artifact_root() -> PathBuf {
    PathBuf::from("./checkpoints/automated_search")
}

fn default_min_candidate_valid_fraction() -> f64 {
    1.0
}

fn default_min_sanitized_fraction() -> f64 {
    1.0
}

fn default_min_unique_smiles_fraction() -> f64 {
    0.5
}

fn default_max_clash_fraction() -> f64 {
    0.0
}

fn default_min_strict_pocket_fit_score() -> f64 {
    0.4
}

fn default_min_pocket_contact_fraction() -> f64 {
    1.0
}

fn default_min_pocket_compatibility_fraction() -> f64 {
    1.0
}

fn default_chemistry_score_weight() -> f64 {
    2.0
}

fn default_uniqueness_score_weight() -> f64 {
    1.5
}

fn default_geometry_score_weight() -> f64 {
    2.0
}

fn default_pocket_score_weight() -> f64 {
    1.5
}

fn default_specialization_score_weight() -> f64 {
    0.75
}

fn default_utilization_score_weight() -> f64 {
    0.5
}

fn default_interaction_review_score_weight() -> f64 {
    1.0
}

fn validate_f64_search_values(
    name: &str,
    values: &[f64],
    positive_required: bool,
    unit_interval_required: bool,
) -> Result<usize, ConfigValidationError> {
    if values.is_empty() {
        return Ok(0);
    }
    for value in values {
        if !value.is_finite() {
            return Err(ConfigValidationError::new(format!(
                "{name} entries must be finite"
            )));
        }
        if positive_required && *value <= 0.0 {
            return Err(ConfigValidationError::new(format!(
                "{name} entries must be greater than zero"
            )));
        }
        if unit_interval_required && !(0.0..=1.0).contains(value) {
            return Err(ConfigValidationError::new(format!(
                "{name} entries must be in [0, 1]"
            )));
        }
    }
    Ok(1)
}

fn validate_usize_search_values(
    name: &str,
    values: &[usize],
    positive_required: bool,
) -> Result<usize, ConfigValidationError> {
    if values.is_empty() {
        return Ok(0);
    }
    if positive_required && values.iter().any(|value| *value == 0) {
        return Err(ConfigValidationError::new(format!(
            "{name} entries must be greater than zero"
        )));
    }
    Ok(1)
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: "cpu".to_string(),
            data_workers: 0,
            tch_intra_op_threads: None,
            tch_inter_op_threads: None,
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
        for (name, value) in [
            ("runtime.tch_intra_op_threads", self.tch_intra_op_threads),
            ("runtime.tch_inter_op_threads", self.tch_inter_op_threads),
        ] {
            if let Some(value) = value {
                if value <= 0 {
                    return Err(ConfigValidationError::new(format!(
                        "{name} must be omitted or greater than zero"
                    )));
                }
            }
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
    /// Pocket-ligand contact encouragement objective.
    #[serde(default)]
    pub rho_pocket_contact: f64,
    /// Pocket-ligand steric-clash penalty objective.
    #[serde(default)]
    pub sigma_pocket_clash: f64,
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
            rho_pocket_contact: 0.0,
            sigma_pocket_clash: 0.0,
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
            (
                "training.loss_weights.rho_pocket_contact",
                self.rho_pocket_contact,
            ),
            (
                "training.loss_weights.sigma_pocket_clash",
                self.sigma_pocket_clash,
            ),
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

fn validate_optional_fraction(
    value: Option<f32>,
    field_name: &str,
) -> Result<(), ConfigValidationError> {
    if let Some(value) = value {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ConfigValidationError::new(format!(
                "{field_name} must be a finite fraction in [0, 1]"
            )));
        }
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
