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
    /// Weight applied to the flow-matching objective when enabled.
    #[serde(default = "default_flow_matching_loss_weight")]
    pub flow_matching_loss_weight: f64,
    /// Weight applied to denoising term in the hybrid denoising+flow objective.
    #[serde(default = "default_hybrid_denoising_weight")]
    pub hybrid_denoising_weight: f64,
    /// Weight applied to flow term in the hybrid denoising+flow objective.
    #[serde(default = "default_hybrid_flow_weight")]
    pub hybrid_flow_weight: f64,
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
            flow_matching_loss_weight: default_flow_matching_loss_weight(),
            hybrid_denoising_weight: default_hybrid_denoising_weight(),
            hybrid_flow_weight: default_hybrid_flow_weight(),
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
        for (name, value) in [
            (
                "training.flow_matching_loss_weight",
                self.flow_matching_loss_weight,
            ),
            (
                "training.hybrid_denoising_weight",
                self.hybrid_denoising_weight,
            ),
            ("training.hybrid_flow_weight", self.hybrid_flow_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if self.primary_objective == PrimaryObjectiveConfig::DenoisingFlowMatching
            && (self.hybrid_denoising_weight + self.hybrid_flow_weight) <= 0.0
        {
            return Err(ConfigValidationError::new(
                "training.hybrid_denoising_weight + training.hybrid_flow_weight must be > 0 for denoising_flow_matching",
            ));
        }
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
    /// Geometry-only flow-matching objective over velocity prediction.
    FlowMatching,
    /// Hybrid objective combining conditioned denoising and flow matching.
    DenoisingFlowMatching,
}

fn default_flow_matching_loss_weight() -> f64 {
    1.0
}

fn default_hybrid_denoising_weight() -> f64 {
    0.5
}

fn default_hybrid_flow_weight() -> f64 {
    0.5
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
