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

    /// Best-checkpoint validation metric after resolving profile-specific defaults.
    pub fn resolved_best_metric(&self) -> String {
        let configured = self
            .training
            .best_metric
            .trim()
            .strip_prefix("validation.")
            .unwrap_or_else(|| self.training.best_metric.trim());
        if configured != "auto" {
            return configured.to_string();
        }
        self.default_best_metric_for_profile().to_string()
    }

    fn default_best_metric_for_profile(&self) -> &'static str {
        let backend_family = self.generation_method.resolved_primary_backend_family();
        let resolved_mode = self
            .data
            .generation_target
            .generation_mode
            .resolved_for_backend(backend_family);
        let full_flow_claim = self
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow;
        if full_flow_claim || resolved_mode == GenerationModeConfig::DeNovoInitialization {
            return "strict_pocket_fit_score";
        }
        if matches!(
            self.training.primary_objective,
            PrimaryObjectiveConfig::FlowMatching | PrimaryObjectiveConfig::DenoisingFlowMatching
        ) {
            return "strict_pocket_fit_score";
        }
        if resolved_mode == GenerationModeConfig::PocketOnlyInitializationBaseline {
            return "candidate_valid_fraction";
        }
        "distance_probe_rmse"
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
        if self
            .data
            .generation_target
            .pocket_only_initialization
            .atom_type_token
            >= self.model.atom_vocab_size
        {
            return Err(ConfigValidationError::new(
                "data.generation_target.pocket_only_initialization.atom_type_token must be less than model.atom_vocab_size",
            ));
        }
        let backend_family = self.generation_method.resolved_primary_backend_family();
        let requested_mode = self.data.generation_target.generation_mode;
        let resolved_mode = requested_mode.resolved_for_backend(backend_family);
        let contract = resolved_mode.compatibility_contract();
        if !contract.supported {
            return Err(ConfigValidationError::new(format!(
                "data.generation_target.generation_mode={} is unsupported: {}",
                resolved_mode.as_str(),
                contract
                    .unsupported_reason
                    .unwrap_or("no compatible implementation is registered")
            )));
        }
        if !contract.supports_backend_family(backend_family) {
            return Err(ConfigValidationError::new(format!(
                "data.generation_target.generation_mode={} is incompatible with generation_method.primary_backend.family={}; choose one of {:?}",
                resolved_mode.as_str(),
                backend_family_label(backend_family),
                contract
                    .compatible_backend_families
                    .iter()
                    .map(|family| backend_family_label(*family))
                    .collect::<Vec<_>>()
            )));
        }
        if !contract.supports_primary_objective(self.training.primary_objective) {
            return Err(ConfigValidationError::new(format!(
                "training.primary_objective={} is incompatible with data.generation_target.generation_mode={} after backend resolution; compatible objectives are {:?}. For pocket_only_initialization_baseline use surrogate_reconstruction unless a future shape-safe pocket-only objective is added.",
                self.training.primary_objective.as_str(),
                resolved_mode.as_str(),
                contract
                    .compatible_primary_objectives
                    .iter()
                    .map(|objective| objective.as_str())
                    .collect::<Vec<_>>()
            )));
        }
        if resolved_mode == GenerationModeConfig::DeNovoInitialization {
            if backend_family != GenerationBackendFamilyConfig::FlowMatching {
                return Err(ConfigValidationError::new(
                    "de_novo_initialization requires generation_method.primary_backend.family=flow_matching",
                ));
            }
            if self.generation_method.flow_matching.geometry_only {
                return Err(ConfigValidationError::new(
                    "de_novo_initialization requires generation_method.flow_matching.geometry_only=false",
                ));
            }
            if !self
                .generation_method
                .flow_matching
                .multi_modal
                .has_full_molecular_branch_set()
            {
                return Err(ConfigValidationError::new(
                    "de_novo_initialization requires full molecular flow branches: geometry, atom_type, bond, topology, pocket_context",
                ));
            }
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
    /// Flow-matching-specific geometry transport controls.
    #[serde(default)]
    pub flow_matching: FlowMatchingConfig,
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
            flow_matching: FlowMatchingConfig::default(),
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
        self.flow_matching.validate()?;
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

    /// Primary backend family after resolving compatibility method ids.
    pub fn resolved_primary_backend_family(&self) -> GenerationBackendFamilyConfig {
        match self.primary_backend_id() {
            "flow_matching" => GenerationBackendFamilyConfig::FlowMatching,
            "autoregressive_graph_geometry" => GenerationBackendFamilyConfig::Autoregressive,
            "energy_guided_refinement" => GenerationBackendFamilyConfig::EnergyGuidedRefinement,
            "conditioned_denoising" => GenerationBackendFamilyConfig::ConditionedDenoising,
            _ => self.primary_backend.family,
        }
    }
}

fn backend_family_label(family: GenerationBackendFamilyConfig) -> &'static str {
    match family {
        GenerationBackendFamilyConfig::ConditionedDenoising => "conditioned_denoising",
        GenerationBackendFamilyConfig::FlowMatching => "flow_matching",
        GenerationBackendFamilyConfig::Diffusion => "diffusion",
        GenerationBackendFamilyConfig::Autoregressive => "autoregressive",
        GenerationBackendFamilyConfig::EnergyGuidedRefinement => "energy_guided_refinement",
        GenerationBackendFamilyConfig::Heuristic => "heuristic",
        GenerationBackendFamilyConfig::RepairOnly => "repair_only",
        GenerationBackendFamilyConfig::RerankerOnly => "reranker_only",
        GenerationBackendFamilyConfig::ExternalWrapper => "external_wrapper",
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

/// Integration method used by geometry flow-matching rollout.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FlowMatchingIntegrationMethod {
    /// First-order Euler integration.
    #[default]
    Euler,
    /// Two-stage Heun integration.
    Heun,
}

/// Flow-matching configuration for molecular generator rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMatchingConfig {
    /// Integration step count used during flow rollout.
    pub steps: usize,
    /// Initial-noise scale used to construct x0.
    pub noise_scale: f64,
    /// Integration update method.
    #[serde(default)]
    pub integration_method: FlowMatchingIntegrationMethod,
    /// Restrict transport to coordinates only. Set false with all multi-modal
    /// branches enabled for de novo molecular flow.
    #[serde(default = "default_flow_matching_geometry_only")]
    pub geometry_only: bool,
    /// Whether x0 should start from decoder corruption (`true`) or Gaussian init (`false`).
    #[serde(default = "default_flow_matching_use_corrupted_x0")]
    pub use_corrupted_x0: bool,
    /// Multi-modal branch contract for molecular flow.
    #[serde(default)]
    pub multi_modal: MultiModalFlowConfig,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            noise_scale: 0.15,
            integration_method: FlowMatchingIntegrationMethod::default(),
            geometry_only: default_flow_matching_geometry_only(),
            use_corrupted_x0: default_flow_matching_use_corrupted_x0(),
            multi_modal: MultiModalFlowConfig::default(),
        }
    }
}

impl FlowMatchingConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.steps == 0 {
            return Err(ConfigValidationError::new(
                "generation_method.flow_matching.steps must be greater than zero",
            ));
        }
        if !self.noise_scale.is_finite() || self.noise_scale < 0.0 {
            return Err(ConfigValidationError::new(
                "generation_method.flow_matching.noise_scale must be finite and non-negative",
            ));
        }
        self.multi_modal.validate()?;
        if !self.geometry_only && !self.multi_modal.has_full_molecular_branch_set() {
            return Err(ConfigValidationError::new(
                "generation_method.flow_matching.geometry_only=false requires geometry, atom_type, bond, topology, and pocket_context branches",
            ));
        }
        Ok(())
    }
}

/// Flow branch identifiers used by the multi-modal flow roadmap.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FlowBranchKind {
    /// Current coordinate velocity branch.
    Geometry,
    /// Categorical atom-type branch.
    AtomType,
    /// Edge-existence and bond-order branch.
    Bond,
    /// Topology consistency coordinator over atom and bond branches.
    Topology,
    /// Pocket/context conditional representation branch.
    PocketContext,
}

impl FlowBranchKind {
    /// All optimizer-facing molecular flow branches.
    pub const ALL: [Self; 5] = [
        Self::Geometry,
        Self::AtomType,
        Self::Bond,
        Self::Topology,
        Self::PocketContext,
    ];

    /// Stable config/artifact label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Geometry => "geometry",
            Self::AtomType => "atom_type",
            Self::Bond => "bond",
            Self::Topology => "topology",
            Self::PocketContext => "pocket_context",
        }
    }

    /// Whether this branch has an optimizer-facing implementation today.
    pub fn implemented(self) -> bool {
        matches!(
            self,
            Self::Geometry
                | Self::AtomType
                | Self::Bond
                | Self::Topology
                | Self::PocketContext
        )
    }
}

/// Target-alignment policy used when de novo atom count differs from ligand supervision.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FlowTargetAlignmentPolicy {
    /// Require exact generated and target row counts and preserve target order.
    IndexExact,
    /// Truncate target tensors to generated atom count; missing rows are masked out.
    Truncate,
    /// Index-preserving truncate/pad policy with explicit masks.
    MaskedTruncate,
    /// Pad missing target rows and mask padded targets out of gradient reductions.
    #[default]
    PadWithMask,
    /// Match generated rows to target rows by minimum coordinate distance.
    HungarianDistance,
    /// Lightweight OT-style distance matching surface for future larger sweeps.
    LightweightOptimalTransport,
    /// Deterministic contiguous subgraph smoke policy for shape-mismatch tests.
    SampledSubgraph,
    /// Reject mismatched atom counts and omit molecular branch losses.
    RejectMismatch,
    /// Legacy modulo repetition, allowed only for explicit smoke/debug configs.
    SmokeOnlyModuloRepeat,
}

impl FlowTargetAlignmentPolicy {
    /// Stable config and artifact label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndexExact => "index_exact",
            Self::Truncate => "truncate",
            Self::MaskedTruncate => "masked_truncate",
            Self::PadWithMask => "pad_with_mask",
            Self::HungarianDistance => "hungarian_distance",
            Self::LightweightOptimalTransport => "lightweight_optimal_transport",
            Self::SampledSubgraph => "sampled_subgraph",
            Self::RejectMismatch => "reject_mismatch",
            Self::SmokeOnlyModuloRepeat => "smoke_only_modulo_repeat",
        }
    }

    /// Whether this policy is suitable for claim-bearing de novo molecular flow.
    pub const fn claim_safe_for_de_novo(self) -> bool {
        matches!(
            self,
            Self::HungarianDistance | Self::LightweightOptimalTransport
        )
    }
}

/// Loss weights reserved for modality-specific flow branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowBranchLossWeights {
    /// Weight for coordinate velocity and endpoint losses.
    #[serde(default = "default_flow_branch_weight")]
    pub geometry: f64,
    /// Weight for atom-type categorical flow loss.
    #[serde(default = "default_flow_branch_weight")]
    pub atom_type: f64,
    /// Weight for bond-existence/type flow loss.
    #[serde(default = "default_flow_branch_weight")]
    pub bond: f64,
    /// Weight for topology consistency flow loss.
    #[serde(default = "default_flow_branch_weight")]
    pub topology: f64,
    /// Weight for pocket/context representation flow loss.
    #[serde(default = "default_flow_branch_weight")]
    pub pocket_context: f64,
    /// Weight for joint synchronization losses across branches.
    #[serde(default = "default_flow_synchronization_weight")]
    pub synchronization: f64,
}

impl Default for FlowBranchLossWeights {
    fn default() -> Self {
        Self {
            geometry: default_flow_branch_weight(),
            atom_type: default_flow_branch_weight(),
            bond: default_flow_branch_weight(),
            topology: default_flow_branch_weight(),
            pocket_context: default_flow_branch_weight(),
            synchronization: default_flow_synchronization_weight(),
        }
    }
}

impl FlowBranchLossWeights {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        for (name, value) in [
            ("geometry", self.geometry),
            ("atom_type", self.atom_type),
            ("bond", self.bond),
            ("topology", self.topology),
            ("pocket_context", self.pocket_context),
            ("synchronization", self.synchronization),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "generation_method.flow_matching.multi_modal.branch_loss_weights.{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    fn weight_for_branch(&self, branch: FlowBranchKind) -> f64 {
        match branch {
            FlowBranchKind::Geometry => self.geometry,
            FlowBranchKind::AtomType => self.atom_type,
            FlowBranchKind::Bond => self.bond,
            FlowBranchKind::Topology => self.topology,
            FlowBranchKind::PocketContext => self.pocket_context,
        }
    }
}

fn default_flow_branch_weight() -> f64 {
    1.0
}

fn default_flow_synchronization_weight() -> f64 {
    0.25
}

/// Per-branch primary-flow warm-start schedule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FlowBranchScheduleEntry {
    /// Whether this branch contributes to the optimizer objective.
    #[serde(default = "default_flow_branch_schedule_enabled")]
    pub enabled: bool,
    /// First global training step where this branch may become nonzero.
    #[serde(default)]
    pub start_step: usize,
    /// Linear warmup length after `start_step`.
    #[serde(default)]
    pub warmup_steps: usize,
    /// Final multiplier applied to the static branch weight.
    #[serde(default = "default_flow_branch_schedule_multiplier")]
    pub final_weight_multiplier: f64,
}

impl Default for FlowBranchScheduleEntry {
    fn default() -> Self {
        Self {
            enabled: default_flow_branch_schedule_enabled(),
            start_step: 0,
            warmup_steps: 0,
            final_weight_multiplier: default_flow_branch_schedule_multiplier(),
        }
    }
}

impl FlowBranchScheduleEntry {
    fn validate(&self, name: &str) -> Result<(), ConfigValidationError> {
        if !self.final_weight_multiplier.is_finite() || self.final_weight_multiplier < 0.0 {
            return Err(ConfigValidationError::new(format!(
                "generation_method.flow_matching.multi_modal.branch_schedule.{name}.final_weight_multiplier must be finite and non-negative"
            )));
        }
        Ok(())
    }

    /// Effective multiplier at a global training step.
    pub fn effective_multiplier(&self, training_step: Option<usize>) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        let Some(step) = training_step else {
            return self.final_weight_multiplier;
        };
        if step < self.start_step {
            return 0.0;
        }
        if self.warmup_steps == 0 {
            return self.final_weight_multiplier;
        }
        let progress = (step - self.start_step + 1) as f64 / self.warmup_steps as f64;
        self.final_weight_multiplier * progress.clamp(0.0, 1.0)
    }
}

fn default_flow_branch_schedule_enabled() -> bool {
    true
}

fn default_flow_branch_schedule_multiplier() -> f64 {
    1.0
}

/// Independent warm-start controls for each primary molecular-flow branch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FlowBranchScheduleConfig {
    /// Coordinate velocity and endpoint branch.
    #[serde(default)]
    pub geometry: FlowBranchScheduleEntry,
    /// Atom-type categorical branch.
    #[serde(default)]
    pub atom_type: FlowBranchScheduleEntry,
    /// Bond existence/type branch.
    #[serde(default)]
    pub bond: FlowBranchScheduleEntry,
    /// Topology consistency branch.
    #[serde(default)]
    pub topology: FlowBranchScheduleEntry,
    /// Pocket/context reconstruction branch.
    #[serde(default)]
    pub pocket_context: FlowBranchScheduleEntry,
    /// Cross-branch synchronization branch.
    #[serde(default)]
    pub synchronization: FlowBranchScheduleEntry,
}

impl FlowBranchScheduleConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        self.geometry.validate("geometry")?;
        self.atom_type.validate("atom_type")?;
        self.bond.validate("bond")?;
        self.topology.validate("topology")?;
        self.pocket_context.validate("pocket_context")?;
        self.synchronization.validate("synchronization")?;
        Ok(())
    }

    /// Apply schedule multipliers to static branch weights.
    pub fn effective_weights(
        &self,
        weights: &FlowBranchLossWeights,
        training_step: Option<usize>,
    ) -> FlowBranchLossWeights {
        FlowBranchLossWeights {
            geometry: weights.geometry * self.geometry.effective_multiplier(training_step),
            atom_type: weights.atom_type * self.atom_type.effective_multiplier(training_step),
            bond: weights.bond * self.bond.effective_multiplier(training_step),
            topology: weights.topology * self.topology.effective_multiplier(training_step),
            pocket_context: weights.pocket_context
                * self.pocket_context.effective_multiplier(training_step),
            synchronization: weights.synchronization
                * self.synchronization.effective_multiplier(training_step),
        }
    }
}

/// Multi-modal flow controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalFlowConfig {
    /// Enabled branch ids for the flow objective family.
    #[serde(default = "default_enabled_flow_branches")]
    pub enabled_branches: Vec<FlowBranchKind>,
    /// Branch-specific loss weights.
    #[serde(default)]
    pub branch_loss_weights: FlowBranchLossWeights,
    /// Branch-specific warm-start and ablation schedule.
    #[serde(default)]
    pub branch_schedule: FlowBranchScheduleConfig,
    /// Explicit target alignment policy for de novo generated atom-count mismatches.
    #[serde(default)]
    pub target_alignment_policy: FlowTargetAlignmentPolicy,
    /// Number of warm-start steps before future joint synchronization may activate.
    #[serde(default)]
    pub warm_start_steps: usize,
    /// Allow present branches to have zero optimizer weight as explicit ablations.
    #[serde(default)]
    pub allow_zero_weight_branch_ablation: bool,
    /// Whether artifacts are allowed to request a full molecular flow claim.
    #[serde(default)]
    pub claim_full_molecular_flow: bool,
}

impl Default for MultiModalFlowConfig {
    fn default() -> Self {
        Self {
            enabled_branches: default_enabled_flow_branches(),
            branch_loss_weights: FlowBranchLossWeights::default(),
            branch_schedule: FlowBranchScheduleConfig::default(),
            target_alignment_policy: FlowTargetAlignmentPolicy::default(),
            warm_start_steps: 0,
            allow_zero_weight_branch_ablation: false,
            claim_full_molecular_flow: false,
        }
    }
}

impl MultiModalFlowConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.enabled_branches.is_empty() {
            return Err(ConfigValidationError::new(
                "generation_method.flow_matching.multi_modal.enabled_branches must include at least geometry",
            ));
        }
        if !self.enabled_branches.contains(&FlowBranchKind::Geometry) {
            return Err(ConfigValidationError::new(
                "generation_method.flow_matching.multi_modal.enabled_branches must include geometry",
            ));
        }
        self.branch_loss_weights.validate()?;
        self.branch_schedule.validate()?;
        if self.claim_full_molecular_flow && !self.has_full_molecular_branch_set() {
            return Err(ConfigValidationError::new(
                "full_molecular_flow claims require geometry, atom_type, bond, topology, and pocket_context branches",
            ));
        }
        if self.claim_full_molecular_flow {
            for branch in FlowBranchKind::ALL {
                if !branch.implemented() {
                    return Err(ConfigValidationError::new(format!(
                        "full_molecular_flow claim configs require implemented branch {}",
                        branch.as_str()
                    )));
                }
                if self.branch_loss_weights.weight_for_branch(branch) <= 0.0 {
                    return Err(ConfigValidationError::new(format!(
                        "full_molecular_flow claim configs require branch_loss_weights.{} to be positive",
                        branch.as_str()
                    )));
                }
                let entry = self.branch_schedule.entry_for_branch(branch);
                if !entry.enabled || entry.final_weight_multiplier <= 0.0 {
                    return Err(ConfigValidationError::new(format!(
                        "full_molecular_flow claim configs require branch_schedule.{} to be enabled with positive final_weight_multiplier",
                        branch.as_str()
                    )));
                }
            }
            let active_at_initial_step = FlowBranchKind::ALL.iter().any(|branch| {
                self.branch_schedule
                    .entry_for_branch(*branch)
                    .effective_multiplier(Some(0))
                    > 0.0
            });
            if !active_at_initial_step {
                return Err(ConfigValidationError::new(
                    "full_molecular_flow claim configs require at least one branch_schedule entry to be active at step 0",
                ));
            }
        }
        if self.claim_full_molecular_flow && !self.target_alignment_policy.claim_safe_for_de_novo()
        {
            return Err(ConfigValidationError::new(
                "full_molecular_flow claim configs require non-index target matching such as hungarian_distance or lightweight_optimal_transport",
            ));
        }
        let scheduled_enabled_branch_count = self
            .enabled_branches
            .iter()
            .filter(|branch| {
                let entry = self.branch_schedule.entry_for_branch(**branch);
                entry.enabled
                    && entry.final_weight_multiplier > 0.0
                    && self.branch_loss_weights.weight_for_branch(**branch) > 0.0
            })
            .count();
        if scheduled_enabled_branch_count == 0 {
            return Err(ConfigValidationError::new(
                "generation_method.flow_matching.multi_modal.branch_schedule must leave at least one enabled branch with positive final_weight_multiplier and positive branch_loss_weight",
            ));
        }
        let zero_weight_branches = self
            .enabled_branches
            .iter()
            .filter(|branch| {
                let entry = self.branch_schedule.entry_for_branch(**branch);
                !entry.enabled
                    || entry.final_weight_multiplier == 0.0
                    || self.branch_loss_weights.weight_for_branch(**branch) == 0.0
            })
            .map(|branch| branch.as_str())
            .collect::<Vec<_>>();
        if !zero_weight_branches.is_empty() && !self.allow_zero_weight_branch_ablation {
            return Err(ConfigValidationError::new(format!(
                "generation_method.flow_matching.multi_modal has present branch(es) [{}] with zero optimizer weight or disabled final schedule; set allow_zero_weight_branch_ablation=true only for explicit ablation configs",
                zero_weight_branches.join(", ")
            )));
        }
        Ok(())
    }

    /// Whether all branches required for a full molecular flow claim are enabled.
    pub fn has_full_molecular_branch_set(&self) -> bool {
        [
            FlowBranchKind::Geometry,
            FlowBranchKind::AtomType,
            FlowBranchKind::Bond,
            FlowBranchKind::Topology,
            FlowBranchKind::PocketContext,
        ]
        .iter()
        .all(|branch| self.enabled_branches.contains(branch))
    }
}

impl FlowBranchScheduleConfig {
    fn entry_for_branch(&self, branch: FlowBranchKind) -> &FlowBranchScheduleEntry {
        match branch {
            FlowBranchKind::Geometry => &self.geometry,
            FlowBranchKind::AtomType => &self.atom_type,
            FlowBranchKind::Bond => &self.bond,
            FlowBranchKind::Topology => &self.topology,
            FlowBranchKind::PocketContext => &self.pocket_context,
        }
    }
}

fn default_enabled_flow_branches() -> Vec<FlowBranchKind> {
    vec![FlowBranchKind::Geometry]
}

fn default_flow_matching_geometry_only() -> bool {
    true
}

fn default_flow_matching_use_corrupted_x0() -> bool {
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
