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
    /// Candidate detached rollout-evaluation per-step weighting decay values.
    #[serde(default, alias = "training_step_weight_decay")]
    pub rollout_eval_step_weight_decay: Vec<f64>,
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
                "automated_search.search_space.rollout_eval_step_weight_decay",
                &self.rollout_eval_step_weight_decay,
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
    if positive_required && values.contains(&0) {
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
