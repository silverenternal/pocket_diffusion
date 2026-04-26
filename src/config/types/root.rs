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
    /// Method selection and comparison configuration.
    #[serde(default)]
    pub generation_method: GenerationMethodConfig,
    /// Additive interaction preference-alignment controls.
    #[serde(default)]
    pub preference_alignment: PreferenceAlignmentConfig,
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
            generation_method: GenerationMethodConfig::default(),
            preference_alignment: PreferenceAlignmentConfig::default(),
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
        self.generation_method.validate()?;
        self.preference_alignment.validate()?;
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

/// Generation-method selection and fair comparison configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMethodConfig {
    /// Active method id used by the primary claim-bearing run.
    #[serde(default = "default_active_generation_method")]
    pub active_method: String,
    /// Backend-neutral primary generator selection.
    #[serde(default)]
    pub primary_backend: GenerationBackendConfig,
    /// Additional methods evaluated on shared splits for fair comparison.
    #[serde(default = "default_comparison_generation_methods")]
    pub comparison_methods: Vec<String>,
    /// Backend-neutral comparison generator selections.
    #[serde(default = "default_comparison_generation_backends")]
    pub comparison_backends: Vec<GenerationBackendConfig>,
    /// Candidate count requested from each method execution.
    #[serde(default = "default_generation_method_candidate_count")]
    pub candidate_count: usize,
    /// Whether the shared comparison runner should execute auxiliary methods.
    #[serde(default = "default_enable_method_comparison")]
    pub enable_comparison_runner: bool,
}

impl Default for GenerationMethodConfig {
    fn default() -> Self {
        Self {
            active_method: default_active_generation_method(),
            primary_backend: GenerationBackendConfig::default(),
            comparison_methods: default_comparison_generation_methods(),
            comparison_backends: default_comparison_generation_backends(),
            candidate_count: default_generation_method_candidate_count(),
            enable_comparison_runner: default_enable_method_comparison(),
        }
    }
}

impl GenerationMethodConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.active_method.trim().is_empty() {
            return Err(ConfigValidationError::new(
                "generation_method.active_method must be non-empty",
            ));
        }
        self.primary_backend
            .validate("generation_method.primary_backend")?;
        if self.candidate_count == 0 {
            return Err(ConfigValidationError::new(
                "generation_method.candidate_count must be greater than zero",
            ));
        }
        if self
            .comparison_methods
            .iter()
            .any(|method_id| method_id.trim().is_empty())
        {
            return Err(ConfigValidationError::new(
                "generation_method.comparison_methods may not contain empty identifiers",
            ));
        }
        for (index, backend) in self.comparison_backends.iter().enumerate() {
            backend.validate(&format!("generation_method.comparison_backends[{index}]"))?;
        }
        Ok(())
    }

    /// Primary backend id after resolving the compatibility string field.
    pub fn primary_backend_id(&self) -> &str {
        if self.primary_backend.backend_id == default_active_generation_method()
            && self.active_method != default_active_generation_method()
        {
            &self.active_method
        } else {
            &self.primary_backend.backend_id
        }
    }

    /// Comparison backend ids after merging compatibility and backend-neutral fields.
    pub fn comparison_backend_ids(&self) -> Vec<String> {
        let mut ids = self.comparison_methods.clone();
        ids.extend(
            self.comparison_backends
                .iter()
                .map(|backend| backend.backend_id.clone()),
        );
        let mut seen = std::collections::BTreeSet::new();
        ids.into_iter()
            .filter(|method_id| seen.insert(method_id.clone()))
            .collect()
    }
}

fn default_active_generation_method() -> String {
    "conditioned_denoising".to_string()
}

fn default_comparison_generation_methods() -> Vec<String> {
    vec![
        "heuristic_raw_rollout_no_repair".to_string(),
        "pocket_centroid_repair_proxy".to_string(),
        "deterministic_proxy_reranker".to_string(),
        "calibrated_reranker".to_string(),
        "flow_matching".to_string(),
        "autoregressive_graph_geometry".to_string(),
        "energy_guided_refinement".to_string(),
    ]
}

fn default_comparison_generation_backends() -> Vec<GenerationBackendConfig> {
    Vec::new()
}

fn default_generation_method_candidate_count() -> usize {
    3
}

fn default_enable_method_comparison() -> bool {
    true
}

/// Additive controls for interaction profile extraction and preference evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceAlignmentConfig {
    /// Enable interaction profile extraction from generated candidate layers.
    #[serde(default)]
    pub enable_profile_extraction: bool,
    /// Enable rule/backend-based preference-pair construction.
    #[serde(default)]
    pub enable_pair_construction: bool,
    /// Enable preference-aware reranking as an additive candidate layer.
    #[serde(default)]
    pub enable_preference_reranking: bool,
    /// Maximum pairs emitted per example before future trainer ingestion.
    #[serde(default = "default_preference_max_pairs_per_example")]
    pub max_pairs_per_example: usize,
    /// Minimum soft-score margin needed to persist a preference pair.
    #[serde(default = "default_preference_min_soft_margin")]
    pub min_soft_margin: f64,
    /// Hard clash threshold for rule-based preferences.
    #[serde(default = "default_preference_max_clash_fraction")]
    pub max_clash_fraction: f64,
    /// Hard strict-pocket-fit threshold for rule-based preferences.
    #[serde(default = "default_preference_min_strict_pocket_fit_score")]
    pub min_strict_pocket_fit_score: f64,
    /// Whether missing preference artifacts mean unavailable evidence, not failure.
    #[serde(default = "default_missing_preference_artifacts_unavailable")]
    pub missing_artifacts_mean_unavailable: bool,
}

impl Default for PreferenceAlignmentConfig {
    fn default() -> Self {
        Self {
            enable_profile_extraction: false,
            enable_pair_construction: false,
            enable_preference_reranking: false,
            max_pairs_per_example: default_preference_max_pairs_per_example(),
            min_soft_margin: default_preference_min_soft_margin(),
            max_clash_fraction: default_preference_max_clash_fraction(),
            min_strict_pocket_fit_score: default_preference_min_strict_pocket_fit_score(),
            missing_artifacts_mean_unavailable: default_missing_preference_artifacts_unavailable(),
        }
    }
}

impl PreferenceAlignmentConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.max_pairs_per_example == 0
            && (self.enable_pair_construction || self.enable_preference_reranking)
        {
            return Err(ConfigValidationError::new(
                "preference_alignment.max_pairs_per_example must be > 0 when pair construction or preference reranking is enabled",
            ));
        }
        if !self.min_soft_margin.is_finite() || self.min_soft_margin < 0.0 {
            return Err(ConfigValidationError::new(
                "preference_alignment.min_soft_margin must be finite and non-negative",
            ));
        }
        if !self.max_clash_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_clash_fraction)
        {
            return Err(ConfigValidationError::new(
                "preference_alignment.max_clash_fraction must be finite and in [0, 1]",
            ));
        }
        if !self.min_strict_pocket_fit_score.is_finite()
            || !(0.0..=1.0).contains(&self.min_strict_pocket_fit_score)
        {
            return Err(ConfigValidationError::new(
                "preference_alignment.min_strict_pocket_fit_score must be finite and in [0, 1]",
            ));
        }
        if self.enable_pair_construction && !self.enable_profile_extraction {
            return Err(ConfigValidationError::new(
                "preference_alignment.enable_pair_construction requires enable_profile_extraction",
            ));
        }
        if self.enable_preference_reranking && !self.enable_profile_extraction {
            return Err(ConfigValidationError::new(
                "preference_alignment.enable_preference_reranking requires enable_profile_extraction",
            ));
        }
        Ok(())
    }
}

fn default_preference_max_pairs_per_example() -> usize {
    256
}

fn default_preference_min_soft_margin() -> f64 {
    0.05
}

fn default_preference_max_clash_fraction() -> f64 {
    0.10
}

fn default_preference_min_strict_pocket_fit_score() -> f64 {
    0.35
}

fn default_missing_preference_artifacts_unavailable() -> bool {
    true
}

/// Backend family declared by config before model-registry compatibility checks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationBackendFamilyConfig {
    /// Shared conditioned denoising rollout over decomposed topology/geometry/context states.
    ConditionedDenoising,
    /// Flow-matching style continuous transport backend.
    FlowMatching,
    /// Diffusion or score-based generation backend.
    Diffusion,
    /// Sequential graph and coordinate construction backend.
    Autoregressive,
    /// Energy or proxy-guided refinement backend.
    EnergyGuidedRefinement,
    /// Lightweight heuristic or diagnostic backend.
    Heuristic,
    /// Repair-only backend.
    RepairOnly,
    /// Reranking-only backend.
    RerankerOnly,
    /// External executable wrapper backend.
    ExternalWrapper,
}

/// Backend-neutral model switch configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationBackendConfig {
    /// Stable registry id for the backend.
    #[serde(default = "default_active_generation_method")]
    pub backend_id: String,
    /// Declared family used for compatibility checks and artifact metadata.
    #[serde(default)]
    pub family: GenerationBackendFamilyConfig,
    /// Whether this backend owns trainable parameters.
    #[serde(default = "default_generation_backend_trainable")]
    pub trainable: bool,
    /// Optional checkpoint path for this backend family.
    #[serde(default)]
    pub checkpoint_path: Option<PathBuf>,
    /// Backend-specific sampling step count when it differs from rollout_steps.
    #[serde(default)]
    pub sampling_steps: Option<usize>,
    /// Backend-specific sampling temperature.
    #[serde(default)]
    pub sampling_temperature: Option<f64>,
    /// Optional external wrapper command when family=external_wrapper.
    #[serde(default)]
    pub external_wrapper: ExternalBackendCommandConfig,
}

impl Default for GenerationBackendConfig {
    fn default() -> Self {
        Self {
            backend_id: default_active_generation_method(),
            family: GenerationBackendFamilyConfig::default(),
            trainable: default_generation_backend_trainable(),
            checkpoint_path: None,
            sampling_steps: None,
            sampling_temperature: None,
            external_wrapper: ExternalBackendCommandConfig::default(),
        }
    }
}

impl Default for GenerationBackendFamilyConfig {
    fn default() -> Self {
        Self::ConditionedDenoising
    }
}

impl GenerationBackendConfig {
    fn validate(&self, logical_name: &str) -> Result<(), ConfigValidationError> {
        if self.backend_id.trim().is_empty() {
            return Err(ConfigValidationError::new(format!(
                "{logical_name}.backend_id must be non-empty"
            )));
        }
        if self.sampling_steps == Some(0) {
            return Err(ConfigValidationError::new(format!(
                "{logical_name}.sampling_steps must be positive when provided"
            )));
        }
        if let Some(temperature) = self.sampling_temperature {
            if !temperature.is_finite() || temperature < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "{logical_name}.sampling_temperature must be finite and non-negative"
                )));
            }
        }
        self.external_wrapper
            .validate(&format!("{logical_name}.external_wrapper"))?;
        if self.family != GenerationBackendFamilyConfig::ExternalWrapper
            && self.external_wrapper.enabled
        {
            return Err(ConfigValidationError::new(format!(
                "{logical_name}.external_wrapper may only be enabled for family=external_wrapper"
            )));
        }
        Ok(())
    }
}

fn default_generation_backend_trainable() -> bool {
    true
}
