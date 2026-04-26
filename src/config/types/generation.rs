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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalBackendCommandConfig {
    /// Whether the backend should be invoked for the current run.
    pub enabled: bool,
    /// Executable path or command name.
    pub executable: Option<String>,
    /// Static argument list appended after the generated input/output paths.
    #[serde(default)]
    pub args: Vec<String>,
    /// Maximum backend runtime before the process is killed.
    #[serde(default = "default_external_backend_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for ExternalBackendCommandConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            executable: None,
            args: Vec::new(),
            timeout_ms: default_external_backend_timeout_ms(),
        }
    }
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
        if self.enabled && self.timeout_ms == 0 {
            return Err(ConfigValidationError::new(format!(
                "{logical_name}.timeout_ms must be greater than zero"
            )));
        }
        Ok(())
    }
}

fn default_external_backend_timeout_ms() -> u64 {
    30_000
}
