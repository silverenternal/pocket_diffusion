/// End-to-end training configuration including staged-loss weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate for the optimizer.
    pub learning_rate: f64,
    /// Number of optimization steps to run in example trainers.
    pub max_steps: usize,
    /// Deterministic data-order and epoch controls.
    #[serde(default)]
    pub data_order: TrainingDataOrderConfig,
    /// Resume policy for checkpoint restoration.
    #[serde(default)]
    pub resume: TrainingResumeConfig,
    /// Gradient clipping and gradient-health diagnostics.
    #[serde(default)]
    pub gradient_clipping: GradientClippingConfig,
    /// Stage schedule.
    pub schedule: StageScheduleConfig,
    /// Optional readiness guard around fixed staged training.
    #[serde(default)]
    pub adaptive_stage_guard: AdaptiveStageGuardConfig,
    /// Loss weights used after warmup ramps are applied.
    pub loss_weights: LossWeightConfig,
    /// Optional leakage-probe scaffolding for future off-modality experiments.
    #[serde(default)]
    pub explicit_leakage_probes: ExplicitLeakageProbeConfig,
    /// Optional pharmacophore role probes and role-leakage controls.
    #[serde(default)]
    pub pharmacophore_probes: PharmacophoreProbeConfig,
    /// Warmup stage controls for chemistry-aware auxiliary objectives.
    #[serde(default)]
    pub chemistry_warmup: ChemistryObjectiveWarmupConfig,
    /// Reporting-only normalization and scale checks for primary objective components.
    #[serde(default)]
    pub objective_scale_diagnostics: ObjectiveScaleDiagnosticsConfig,
    /// Sparse objective-family gradient diagnostics. Disabled by default.
    #[serde(default)]
    pub objective_gradient_diagnostics: ObjectiveGradientDiagnosticsConfig,
    /// Checkpoint directory for trainer skeleton output.
    pub checkpoint_dir: PathBuf,
    /// Checkpoint interval in steps.
    pub checkpoint_every: usize,
    /// Validation interval in optimization steps; 0 disables periodic validation.
    #[serde(default)]
    pub validation_every: usize,
    /// Validation metric used for best-checkpoint selection.
    ///
    /// Use `auto` to select a profile-specific metric from the full
    /// `ResearchConfig`.
    #[serde(default = "default_best_validation_metric")]
    pub best_metric: String,
    /// Number of validation checks without improvement before stopping early.
    #[serde(default)]
    pub early_stopping_patience: Option<usize>,
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
    /// Reserved switch for a future tensor-preserving rollout loss.
    ///
    /// The current rollout path emits detached `rollout_eval_*` diagnostics
    /// only, so enabling this is rejected during config validation.
    #[serde(default)]
    pub enable_trainable_rollout_loss: bool,
    /// Whether training steps should build detached rollout diagnostic traces.
    ///
    /// Optimizer-facing losses do not require these sampled traces; disabling
    /// them keeps rollout evaluation components at their zero-step diagnostic
    /// defaults while avoiding diagnostic graph and memory work.
    #[serde(default = "default_build_rollout_diagnostics")]
    pub build_rollout_diagnostics: bool,
    /// Weighting strategy for labeled affinity supervision.
    pub affinity_weighting: AffinityWeighting,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            max_steps: 8,
            data_order: TrainingDataOrderConfig::default(),
            resume: TrainingResumeConfig::default(),
            gradient_clipping: GradientClippingConfig::default(),
            schedule: StageScheduleConfig::default(),
            adaptive_stage_guard: AdaptiveStageGuardConfig::default(),
            loss_weights: LossWeightConfig::default(),
            explicit_leakage_probes: ExplicitLeakageProbeConfig::default(),
            pharmacophore_probes: PharmacophoreProbeConfig::default(),
            chemistry_warmup: ChemistryObjectiveWarmupConfig::default(),
            objective_scale_diagnostics: ObjectiveScaleDiagnosticsConfig::default(),
            objective_gradient_diagnostics: ObjectiveGradientDiagnosticsConfig::default(),
            checkpoint_dir: PathBuf::from("./checkpoints"),
            checkpoint_every: 10,
            validation_every: 0,
            best_metric: default_best_validation_metric(),
            early_stopping_patience: None,
            log_every: 1,
            primary_objective: PrimaryObjectiveConfig::ConditionedDenoising,
            flow_matching_loss_weight: default_flow_matching_loss_weight(),
            hybrid_denoising_weight: default_hybrid_denoising_weight(),
            hybrid_flow_weight: default_hybrid_flow_weight(),
            enable_trainable_rollout_loss: false,
            build_rollout_diagnostics: default_build_rollout_diagnostics(),
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
        self.adaptive_stage_guard.validate()?;
        self.data_order.validate()?;
        self.resume.validate()?;
        self.gradient_clipping.validate()?;
        self.loss_weights.validate()?;
        self.explicit_leakage_probes.validate()?;
        self.pharmacophore_probes.validate()?;
        self.chemistry_warmup.validate()?;
        self.objective_scale_diagnostics.validate()?;
        self.objective_gradient_diagnostics.validate()?;
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
        if self.enable_trainable_rollout_loss {
            return Err(ConfigValidationError::new(
                "training.enable_trainable_rollout_loss is not implemented; rollout metrics are currently detached rollout_eval diagnostics only",
            ));
        }
        if self.checkpoint_dir.as_os_str().is_empty() {
            return Err(ConfigValidationError::new(
                "training.checkpoint_dir must not be empty",
            ));
        }
        if is_configs_checkpoint_tree(&self.checkpoint_dir) {
            return Err(ConfigValidationError::new(
                "training.checkpoint_dir must not be inside configs/checkpoints; use checkpoints/ as the generated checkpoint root",
            ));
        }
        if self.checkpoint_dir.exists() && !self.checkpoint_dir.is_dir() {
            return Err(ConfigValidationError::new(format!(
                "training.checkpoint_dir={} must be a directory when it already exists",
                self.checkpoint_dir.display()
            )));
        }
        if self.best_metric.trim().is_empty() {
            return Err(ConfigValidationError::new(
                "training.best_metric must not be empty",
            ));
        }
        if !is_supported_best_validation_metric(&self.best_metric) {
            return Err(ConfigValidationError::new(format!(
                "training.best_metric={} is not supported",
                self.best_metric
            )));
        }
        if let Some(patience) = self.early_stopping_patience {
            if patience == 0 {
                return Err(ConfigValidationError::new(
                    "training.early_stopping_patience must be omitted or greater than zero",
                ));
            }
            if self.validation_every == 0 {
                return Err(ConfigValidationError::new(
                    "training.validation_every must be greater than zero when early_stopping_patience is set",
                ));
            }
        }
        Ok(())
    }
}

fn default_best_validation_metric() -> String {
    "auto".to_string()
}

fn default_build_rollout_diagnostics() -> bool {
    true
}

fn is_supported_best_validation_metric(metric: &str) -> bool {
    matches!(
        normalize_best_validation_metric(metric).as_str(),
        "auto"
            | "finite_forward_fraction"
            | "strict_pocket_fit_score"
            | "candidate_valid_fraction"
            | "leakage_proxy_mean"
            | "distance_probe_rmse"
            | "affinity_probe_mae"
            | "examples_per_second"
    )
}

fn normalize_best_validation_metric(metric: &str) -> String {
    metric
        .trim()
        .strip_prefix("validation.")
        .unwrap_or_else(|| metric.trim())
        .to_string()
}

fn is_configs_checkpoint_tree(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    let mut parts = Vec::new();
    for part in normalized.split('/') {
        if part == "." || part == ".." || part.is_empty() {
            continue;
        }
        parts.push(part);
    }
    for window in parts.windows(2) {
        if window[0] == "configs" && window[1] == "checkpoints" {
            return true;
        }
    }
    false
}

/// Controls for checkpoint resume semantics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingResumeConfig {
    /// Require exact optimizer-internal-state restoration when resuming.
    ///
    /// The current tch optimizer wrapper does not expose Adam moment-buffer
    /// serialization, so setting this to true intentionally rejects
    /// weights-only checkpoints instead of silently advertising exact replay.
    #[serde(default)]
    pub require_optimizer_exact: bool,
}

impl TrainingResumeConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        Ok(())
    }
}

/// Optional optimizer-time gradient clipping.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GradientClippingConfig {
    /// Clip the global gradient L2 norm to this value when provided.
    #[serde(default)]
    pub global_norm: Option<f64>,
}

impl GradientClippingConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if let Some(global_norm) = self.global_norm {
            if !global_norm.is_finite() || global_norm <= 0.0 {
                return Err(ConfigValidationError::new(
                    "training.gradient_clipping.global_norm must be omitted or finite and positive",
                ));
            }
        }
        Ok(())
    }
}

/// Deterministic mini-batch sampling and epoch controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataOrderConfig {
    /// Shuffle example order independently for each epoch.
    #[serde(default)]
    pub shuffle: bool,
    /// Base seed used to derive each epoch's sample order.
    #[serde(default = "default_sampler_seed")]
    pub sampler_seed: u64,
    /// Drop the final short batch in each epoch.
    #[serde(default)]
    pub drop_last: bool,
    /// Optional hard cap on full sampler epochs.
    #[serde(default)]
    pub max_epochs: Option<usize>,
}

impl Default for TrainingDataOrderConfig {
    fn default() -> Self {
        Self {
            shuffle: false,
            sampler_seed: default_sampler_seed(),
            drop_last: false,
            max_epochs: None,
        }
    }
}

impl TrainingDataOrderConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.max_epochs == Some(0) {
            return Err(ConfigValidationError::new(
                "training.data_order.max_epochs must be omitted or greater than zero",
            ));
        }
        Ok(())
    }
}

fn default_sampler_seed() -> u64 {
    0
}

fn default_primary_objective() -> PrimaryObjectiveConfig {
    PrimaryObjectiveConfig::ConditionedDenoising
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
    /// Number of Stage 4 steps used to ramp gate and slot objectives.
    #[serde(default = "default_stage4_warmup_steps")]
    pub stage4_warmup_steps: usize,
}

impl Default for StageScheduleConfig {
    fn default() -> Self {
        Self {
            stage1_steps: 2,
            stage2_steps: 4,
            stage3_steps: 6,
            stage4_warmup_steps: default_stage4_warmup_steps(),
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

fn default_stage4_warmup_steps() -> usize {
    2
}

/// Reporting-only primary objective scale diagnostics.
///
/// These controls never rescale the optimizer objective. They only determine
/// how trainer metrics normalize component values and when a scale warning is
/// emitted for inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveScaleDiagnosticsConfig {
    /// Emit per-component primary scale records.
    #[serde(default = "default_objective_scale_diagnostics_enabled")]
    pub enabled: bool,
    /// Warn when one normalized component exceeds another nonzero component by this ratio.
    #[serde(default = "default_objective_scale_warning_ratio")]
    pub warning_ratio: f64,
    /// Small positive floor used for denominator stability.
    #[serde(default = "default_objective_scale_epsilon")]
    pub epsilon: f64,
    /// Optional EMA momentum for normalizing by recent absolute component magnitudes.
    #[serde(default)]
    pub running_scale_momentum: Option<f64>,
}

impl Default for ObjectiveScaleDiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: default_objective_scale_diagnostics_enabled(),
            warning_ratio: default_objective_scale_warning_ratio(),
            epsilon: default_objective_scale_epsilon(),
            running_scale_momentum: None,
        }
    }
}

impl ObjectiveScaleDiagnosticsConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.warning_ratio.is_finite() || self.warning_ratio <= 0.0 {
            return Err(ConfigValidationError::new(
                "training.objective_scale_diagnostics.warning_ratio must be finite and positive",
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(ConfigValidationError::new(
                "training.objective_scale_diagnostics.epsilon must be finite and positive",
            ));
        }
        if let Some(momentum) = self.running_scale_momentum {
            if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
                return Err(ConfigValidationError::new(
                    "training.objective_scale_diagnostics.running_scale_momentum must be omitted or finite in [0, 1)",
                ));
            }
        }
        Ok(())
    }
}

fn default_objective_scale_diagnostics_enabled() -> bool {
    true
}

fn default_objective_scale_warning_ratio() -> f64 {
    10.0
}

fn default_objective_scale_epsilon() -> f64 {
    1.0e-12
}

/// Objective-family gradient diagnostic backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveGradientSamplingMode {
    /// Run exact sampled autograd queries before the optimizer backward pass.
    #[default]
    ExactSampled,
    /// Use the legacy post-backward weighted-loss-share allocation proxy.
    LossShareProxy,
}

/// Sparse objective-family gradient diagnostics.
///
/// Exact sampling is disabled by default and only runs on the configured
/// interval when explicitly enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveGradientDiagnosticsConfig {
    /// Enable objective-family gradient contribution diagnostics.
    #[serde(default)]
    pub enabled: bool,
    /// Sample every N global optimizer steps when enabled.
    #[serde(default = "default_objective_gradient_sample_every_steps")]
    pub sample_every_steps: usize,
    /// Diagnostic backend used on sampled steps.
    #[serde(default)]
    pub sampling_mode: ObjectiveGradientSamplingMode,
    /// Include auxiliary families in addition to the active primary objective.
    #[serde(default = "default_objective_gradient_include_auxiliary")]
    pub include_auxiliary: bool,
    /// Optional allow-list of objective families to sample.
    ///
    /// Empty means all active families selected by `include_auxiliary`.
    /// Accepted labels are `primary`, `primary:<objective>`,
    /// `<auxiliary_family>`, or `auxiliary:<auxiliary_family>`.
    #[serde(default)]
    pub included_families: Vec<String>,
}

impl Default for ObjectiveGradientDiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_every_steps: default_objective_gradient_sample_every_steps(),
            sampling_mode: ObjectiveGradientSamplingMode::ExactSampled,
            include_auxiliary: default_objective_gradient_include_auxiliary(),
            included_families: Vec::new(),
        }
    }
}

impl ObjectiveGradientDiagnosticsConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.enabled && self.sample_every_steps == 0 {
            return Err(ConfigValidationError::new(
                "training.objective_gradient_diagnostics.sample_every_steps must be greater than zero when enabled",
            ));
        }
        for family in &self.included_families {
            let normalized = family.trim();
            if normalized.is_empty() || !is_supported_objective_gradient_family(normalized) {
                return Err(ConfigValidationError::new(format!(
                    "training.objective_gradient_diagnostics.included_families contains unsupported family '{family}'"
                )));
            }
        }
        Ok(())
    }
}

fn default_objective_gradient_sample_every_steps() -> usize {
    100
}

fn default_objective_gradient_include_auxiliary() -> bool {
    true
}

fn is_supported_objective_gradient_family(family: &str) -> bool {
    matches!(
        family,
        "primary"
            | "primary:conditioned_denoising"
            | "primary:surrogate_reconstruction"
            | "primary:flow_matching"
            | "primary:denoising_flow_matching"
            | "intra_red"
            | "auxiliary:intra_red"
            | "probe"
            | "auxiliary:probe"
            | "pharmacophore_probe"
            | "auxiliary:pharmacophore_probe"
            | "leak"
            | "auxiliary:leak"
            | "pharmacophore_leakage"
            | "auxiliary:pharmacophore_leakage"
            | "gate"
            | "auxiliary:gate"
            | "slot"
            | "auxiliary:slot"
            | "consistency"
            | "auxiliary:consistency"
            | "pocket_contact"
            | "auxiliary:pocket_contact"
            | "pocket_clash"
            | "auxiliary:pocket_clash"
            | "pocket_envelope"
            | "auxiliary:pocket_envelope"
            | "valence_guardrail"
            | "auxiliary:valence_guardrail"
            | "bond_length_guardrail"
            | "auxiliary:bond_length_guardrail"
    )
}

/// Optional deterministic readiness guard around fixed staged training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStageGuardConfig {
    /// Enable readiness evaluation. Disabled keeps the fixed schedule exactly.
    #[serde(default)]
    pub enabled: bool,
    /// Hold the effective stage at the previous stage when readiness fails.
    ///
    /// When false, readiness failures are warnings only and fixed schedule
    /// weights remain active.
    #[serde(default)]
    pub hold_stages: bool,
    /// Number of recent step records used for readiness evaluation.
    #[serde(default = "default_adaptive_stage_readiness_window")]
    pub readiness_window: usize,
    /// Minimum fraction of recent steps with finite primary and total losses.
    #[serde(default = "default_adaptive_stage_min_finite_step_fraction")]
    pub min_finite_step_fraction: f64,
    /// Maximum allowed non-finite gradient tensor count on the latest step.
    #[serde(default)]
    pub max_nonfinite_gradient_tensors: usize,
    /// Require the latest optimizer step not to be skipped.
    #[serde(default = "default_adaptive_stage_require_no_optimizer_skip")]
    pub require_no_optimizer_skip: bool,
    /// Maximum allowed slot collapse warning count on the latest step.
    #[serde(default = "default_adaptive_stage_max_slot_collapse_warnings")]
    pub max_slot_collapse_warnings: usize,
    /// Maximum allowed mean gate saturation fraction on the latest step.
    #[serde(default = "default_adaptive_stage_max_gate_saturation_fraction")]
    pub max_gate_saturation_fraction: f64,
    /// Optional minimum per-modality slot signature matching score on the latest step.
    #[serde(default)]
    pub min_slot_signature_matching_score: Option<f64>,
    /// Optional maximum detached leakage diagnostic before advancing stages.
    #[serde(default)]
    pub max_leakage_diagnostic: Option<f64>,
    /// Minimum relative primary-loss improvement across the readiness window.
    #[serde(default)]
    pub min_primary_loss_improvement_fraction: f64,
}

impl Default for AdaptiveStageGuardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hold_stages: false,
            readiness_window: default_adaptive_stage_readiness_window(),
            min_finite_step_fraction: default_adaptive_stage_min_finite_step_fraction(),
            max_nonfinite_gradient_tensors: 0,
            require_no_optimizer_skip: default_adaptive_stage_require_no_optimizer_skip(),
            max_slot_collapse_warnings: default_adaptive_stage_max_slot_collapse_warnings(),
            max_gate_saturation_fraction: default_adaptive_stage_max_gate_saturation_fraction(),
            min_slot_signature_matching_score: None,
            max_leakage_diagnostic: None,
            min_primary_loss_improvement_fraction: 0.0,
        }
    }
}

impl AdaptiveStageGuardConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.readiness_window == 0 {
            return Err(ConfigValidationError::new(
                "training.adaptive_stage_guard.readiness_window must be greater than zero",
            ));
        }
        if !self.min_finite_step_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.min_finite_step_fraction)
        {
            return Err(ConfigValidationError::new(
                "training.adaptive_stage_guard.min_finite_step_fraction must be finite and between 0 and 1",
            ));
        }
        if !self.max_gate_saturation_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_gate_saturation_fraction)
        {
            return Err(ConfigValidationError::new(
                "training.adaptive_stage_guard.max_gate_saturation_fraction must be finite and between 0 and 1",
            ));
        }
        if let Some(threshold) = self.min_slot_signature_matching_score {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                return Err(ConfigValidationError::new(
                    "training.adaptive_stage_guard.min_slot_signature_matching_score must be finite and between 0 and 1",
                ));
            }
        }
        if let Some(threshold) = self.max_leakage_diagnostic {
            if !threshold.is_finite() || threshold < 0.0 {
                return Err(ConfigValidationError::new(
                    "training.adaptive_stage_guard.max_leakage_diagnostic must be finite and non-negative",
                ));
            }
        }
        if !self.min_primary_loss_improvement_fraction.is_finite()
            || self.min_primary_loss_improvement_fraction < 0.0
        {
            return Err(ConfigValidationError::new(
                "training.adaptive_stage_guard.min_primary_loss_improvement_fraction must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

fn default_adaptive_stage_readiness_window() -> usize {
    2
}

fn default_adaptive_stage_min_finite_step_fraction() -> f64 {
    1.0
}

fn default_adaptive_stage_require_no_optimizer_skip() -> bool {
    true
}

fn default_adaptive_stage_max_slot_collapse_warnings() -> usize {
    0
}

fn default_adaptive_stage_max_gate_saturation_fraction() -> f64 {
    0.95
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
    /// Relative weight for slot sparsity term inside `L_slot`.
    #[serde(default = "default_slot_sparsity_weight")]
    pub slot_sparsity_weight: f64,
    /// Relative weight for slot-balance term inside `L_slot`.
    #[serde(default = "default_slot_balance_weight")]
    pub slot_balance_weight: f64,
    /// Pocket-ligand contact encouragement objective.
    #[serde(default)]
    pub rho_pocket_contact: f64,
    /// Pocket-ligand steric-clash penalty objective.
    #[serde(default)]
    pub sigma_pocket_clash: f64,
    /// Pocket-envelope containment objective.
    #[serde(default)]
    pub tau_pocket_envelope: f64,
    /// Conservative valence overage objective.
    #[serde(default)]
    pub upsilon_valence_guardrail: f64,
    /// Topology-implied bond-length objective.
    #[serde(default)]
    pub phi_bond_length_guardrail: f64,
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
            slot_sparsity_weight: default_slot_sparsity_weight(),
            slot_balance_weight: default_slot_balance_weight(),
            rho_pocket_contact: 0.0,
            sigma_pocket_clash: 0.0,
            tau_pocket_envelope: 0.0,
            upsilon_valence_guardrail: 0.0,
            phi_bond_length_guardrail: 0.0,
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
                "training.loss_weights.slot_sparsity_weight",
                self.slot_sparsity_weight,
            ),
            (
                "training.loss_weights.slot_balance_weight",
                self.slot_balance_weight,
            ),
            (
                "training.loss_weights.rho_pocket_contact",
                self.rho_pocket_contact,
            ),
            (
                "training.loss_weights.sigma_pocket_clash",
                self.sigma_pocket_clash,
            ),
            (
                "training.loss_weights.tau_pocket_envelope",
                self.tau_pocket_envelope,
            ),
            (
                "training.loss_weights.upsilon_valence_guardrail",
                self.upsilon_valence_guardrail,
            ),
            (
                "training.loss_weights.phi_bond_length_guardrail",
                self.phi_bond_length_guardrail,
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

/// Training semantics for explicit off-modality leakage probes.
///
/// See `docs/q14_leakage_training_contract.md` for the claim boundary and for
/// contracted future modes that require additional gradient-routing tests.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExplicitLeakageProbeTrainingSemantics {
    /// Treat explicit probe values as detached diagnostics only.
    DetachedDiagnostic,
    /// Train explicit leakage probes against detached source features.
    ProbeFit,
    /// Penalize source encoders through explicit probes with probe parameters detached.
    EncoderPenalty,
    /// Alternate probe-fit and encoder-penalty phases by trainer step.
    Alternating,
    /// Backpropagate an adversarial penalty when wrong-modality predictions are too accurate.
    #[default]
    AdversarialPenalty,
}

/// Off-modality leakage-probe scaffolding flags (default disabled).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplicitLeakageProbeConfig {
    /// Enable explicit leakage-probe objective wiring.
    ///
    /// Default `false` keeps current behavior based on the similarity-margin proxy.
    #[serde(default)]
    pub enable_explicit_probes: bool,
    /// Gradient semantics for enabled explicit probe routes.
    #[serde(default)]
    pub training_semantics: ExplicitLeakageProbeTrainingSemantics,
    /// Probe geometry targets from topology inputs.
    #[serde(default)]
    pub topology_to_geometry_probe: bool,
    /// Probe pocket features from topology inputs.
    #[serde(default)]
    pub topology_to_pocket_probe: bool,
    /// Probe topology targets from geometry inputs.
    #[serde(default)]
    pub geometry_to_topology_probe: bool,
    /// Probe pocket features from geometry inputs.
    #[serde(default)]
    pub geometry_to_pocket_probe: bool,
    /// Probe topology targets from pocket inputs.
    #[serde(default)]
    pub pocket_to_topology_probe: bool,
    /// Probe geometry targets from pocket inputs.
    #[serde(default)]
    pub pocket_to_geometry_probe: bool,
}

impl Default for ExplicitLeakageProbeConfig {
    fn default() -> Self {
        Self {
            enable_explicit_probes: false,
            training_semantics: ExplicitLeakageProbeTrainingSemantics::default(),
            topology_to_geometry_probe: false,
            topology_to_pocket_probe: false,
            geometry_to_topology_probe: false,
            geometry_to_pocket_probe: false,
            pocket_to_topology_probe: false,
            pocket_to_geometry_probe: false,
        }
    }
}

impl ExplicitLeakageProbeConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.enable_explicit_probes {
            return Ok(());
        }

        let any_target = self.topology_to_geometry_probe
            || self.topology_to_pocket_probe
            || self.geometry_to_topology_probe
            || self.geometry_to_pocket_probe
            || self.pocket_to_topology_probe
            || self.pocket_to_geometry_probe;
        if !any_target {
            return Err(ConfigValidationError::new(
                "training.explicit_leakage_probes.enable_explicit_probes is true but all probe targets are disabled",
            ));
        }
        Ok(())
    }
}

/// Optional pharmacophore-role probe and leakage-control wiring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmacophoreProbeConfig {
    /// Supervise ligand pharmacophore role prediction from ligand/topology tokens.
    #[serde(default)]
    pub enable_ligand_role_probe: bool,
    /// Supervise pocket pharmacophore role prediction from pocket/context tokens.
    #[serde(default)]
    pub enable_pocket_role_probe: bool,
    /// Penalize topology representations that become too predictive of pocket roles.
    #[serde(default)]
    pub enable_topology_to_pocket_role_leakage: bool,
    /// Penalize geometry representations that become too predictive of pocket roles.
    #[serde(default)]
    pub enable_geometry_to_pocket_role_leakage: bool,
    /// Penalize pocket representations that become too predictive of ligand roles.
    #[serde(default)]
    pub enable_pocket_to_ligand_role_leakage: bool,
    /// Minimum tolerated role-prediction BCE before an off-modality route is penalized.
    #[serde(default = "default_pharmacophore_leakage_margin")]
    pub leakage_margin: f64,
}

impl Default for PharmacophoreProbeConfig {
    fn default() -> Self {
        Self {
            enable_ligand_role_probe: false,
            enable_pocket_role_probe: false,
            enable_topology_to_pocket_role_leakage: false,
            enable_geometry_to_pocket_role_leakage: false,
            enable_pocket_to_ligand_role_leakage: false,
            leakage_margin: default_pharmacophore_leakage_margin(),
        }
    }
}

impl PharmacophoreProbeConfig {
    /// Whether any supervised same-modality pharmacophore role probe is active.
    pub const fn role_probe_enabled(&self) -> bool {
        self.enable_ligand_role_probe || self.enable_pocket_role_probe
    }

    /// Whether any explicit pharmacophore role leakage route is active.
    pub const fn role_leakage_enabled(&self) -> bool {
        self.enable_topology_to_pocket_role_leakage
            || self.enable_geometry_to_pocket_role_leakage
            || self.enable_pocket_to_ligand_role_leakage
    }

    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.leakage_margin.is_finite() || self.leakage_margin < 0.0 {
            return Err(ConfigValidationError::new(
                "training.pharmacophore_probes.leakage_margin must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

fn default_pharmacophore_leakage_margin() -> f64 {
    0.35
}

/// Stage number where chemistry-aware objective families become eligible.
///
/// Values are one-based training stages in `[1, 4]`. Defaults preserve the
/// original four-stage progression: pocket geometry and chemistry guardrails
/// start in stage 2, while probe/leakage semantics start in stage 3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemistryObjectiveWarmupConfig {
    /// Stage where the pocket-envelope containment loss may become active.
    #[serde(default = "default_stage2_start")]
    pub pocket_envelope_start_stage: usize,
    /// Stage where the valence guardrail loss may become active.
    #[serde(default = "default_stage2_start")]
    pub valence_guardrail_start_stage: usize,
    /// Stage where the bond-length guardrail loss may become active.
    #[serde(default = "default_stage2_start")]
    pub bond_length_guardrail_start_stage: usize,
    /// Stage where pharmacophore role probe subterms may become active.
    #[serde(default = "default_stage3_start")]
    pub pharmacophore_probe_start_stage: usize,
    /// Stage where pharmacophore role leakage subterms may become active.
    #[serde(default = "default_stage3_start")]
    pub pharmacophore_leakage_start_stage: usize,
}

impl Default for ChemistryObjectiveWarmupConfig {
    fn default() -> Self {
        Self {
            pocket_envelope_start_stage: default_stage2_start(),
            valence_guardrail_start_stage: default_stage2_start(),
            bond_length_guardrail_start_stage: default_stage2_start(),
            pharmacophore_probe_start_stage: default_stage3_start(),
            pharmacophore_leakage_start_stage: default_stage3_start(),
        }
    }
}

impl ChemistryObjectiveWarmupConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        for (name, value) in [
            (
                "training.chemistry_warmup.pocket_envelope_start_stage",
                self.pocket_envelope_start_stage,
            ),
            (
                "training.chemistry_warmup.valence_guardrail_start_stage",
                self.valence_guardrail_start_stage,
            ),
            (
                "training.chemistry_warmup.bond_length_guardrail_start_stage",
                self.bond_length_guardrail_start_stage,
            ),
            (
                "training.chemistry_warmup.pharmacophore_probe_start_stage",
                self.pharmacophore_probe_start_stage,
            ),
            (
                "training.chemistry_warmup.pharmacophore_leakage_start_stage",
                self.pharmacophore_leakage_start_stage,
            ),
        ] {
            if !(1..=4).contains(&value) {
                return Err(ConfigValidationError::new(format!(
                    "{name} must be a one-based training stage in [1, 4]"
                )));
            }
        }
        Ok(())
    }
}

fn default_stage2_start() -> usize {
    2
}

fn default_stage3_start() -> usize {
    3
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

impl PrimaryObjectiveConfig {
    /// Stable config and artifact label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SurrogateReconstruction => "surrogate_reconstruction",
            Self::ConditionedDenoising => "conditioned_denoising",
            Self::FlowMatching => "flow_matching",
            Self::DenoisingFlowMatching => "denoising_flow_matching",
        }
    }
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

fn default_slot_sparsity_weight() -> f64 {
    1.0
}

fn default_slot_balance_weight() -> f64 {
    1.0
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
