/// Explicit generation semantics attached to training, rollout, and artifacts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GenerationModeConfig {
    /// Decoder recovers a target-derived corrupted ligand state.
    #[default]
    TargetLigandDenoising,
    /// Decoder refines an existing ligand-like initialization without claiming de novo sampling.
    LigandRefinement,
    /// Flow-matching transport refines target-derived or ligand-like coordinates.
    FlowRefinement,
    /// Conservative pocket-only initialization baseline with configured atom-count prior.
    PocketOnlyInitializationBaseline,
    /// Pocket-conditioned de novo initialization with an internally predicted atom count.
    DeNovoInitialization,
}

impl GenerationModeConfig {
    /// All generation modes, including reserved unsupported modes.
    pub const ALL: [Self; 5] = [
        Self::TargetLigandDenoising,
        Self::LigandRefinement,
        Self::FlowRefinement,
        Self::PocketOnlyInitializationBaseline,
        Self::DeNovoInitialization,
    ];

    /// Stable artifact label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TargetLigandDenoising => "target_ligand_denoising",
            Self::LigandRefinement => "ligand_refinement",
            Self::FlowRefinement => "flow_refinement",
            Self::PocketOnlyInitializationBaseline => "pocket_only_initialization_baseline",
            Self::DeNovoInitialization => "de_novo_initialization",
        }
    }

    /// Resolve legacy backend selections into explicit generation semantics.
    pub fn resolved_for_backend(self, family: GenerationBackendFamilyConfig) -> Self {
        match (self, family) {
            (Self::TargetLigandDenoising, GenerationBackendFamilyConfig::FlowMatching) => {
                Self::FlowRefinement
            }
            (mode, _) => mode,
        }
    }

    /// Whether this mode can support de novo generation wording.
    pub fn permits_de_novo_claims(self) -> bool {
        matches!(self, Self::DeNovoInitialization)
    }

    /// Whether decoder initialization is derived from target ligand atoms and coordinates.
    pub fn uses_target_ligand_initialization(self) -> bool {
        !matches!(
            self,
            Self::PocketOnlyInitializationBaseline | Self::DeNovoInitialization
        )
    }

    /// Stable atom-count source label for rollout and artifact diagnostics.
    pub fn atom_count_source_label(self) -> &'static str {
        match self {
            Self::PocketOnlyInitializationBaseline => "configured_atom_count_prior",
            Self::DeNovoInitialization => "pocket_conditioned_atom_count_policy",
            Self::TargetLigandDenoising | Self::LigandRefinement | Self::FlowRefinement => {
                "target_ligand_atom_count"
            }
        }
    }

    /// Stable topology-source label for decoder initialization diagnostics.
    pub fn topology_source_label(self) -> &'static str {
        match self {
            Self::PocketOnlyInitializationBaseline => "configured_uniform_atom_type_prior",
            Self::DeNovoInitialization => "pocket_conditioned_topology_flow",
            Self::TargetLigandDenoising | Self::LigandRefinement | Self::FlowRefinement => {
                "target_ligand_topology"
            }
        }
    }

    /// Stable geometry-source label for decoder initialization diagnostics.
    pub fn geometry_source_label(self) -> &'static str {
        match self {
            Self::PocketOnlyInitializationBaseline => "pocket_centroid_deterministic_offsets",
            Self::DeNovoInitialization => "pocket_conditioned_coordinate_flow",
            Self::TargetLigandDenoising | Self::LigandRefinement | Self::FlowRefinement => {
                "target_ligand_corrupted_geometry"
            }
        }
    }

    /// Machine-readable compatibility contract for this generation mode.
    pub fn compatibility_contract(self) -> GenerationModeCompatibilityContract {
        match self {
            Self::TargetLigandDenoising => GenerationModeCompatibilityContract {
                generation_mode: self,
                claim_label: "target_ligand_denoising",
                supported: true,
                unsupported_reason: None,
                target_ligand_topology: true,
                target_ligand_geometry: true,
                fixed_atom_count: true,
                pocket_only_initialization: false,
                graph_growth: false,
                atom_count_source: self.atom_count_source_label(),
                topology_source: self.topology_source_label(),
                geometry_source: self.geometry_source_label(),
                pocket_context_availability: "inference_available",
                target_ligand_atom_type_availability: "target_supervision_only",
                target_ligand_topology_availability: "target_supervision_only",
                target_ligand_coordinate_availability: "target_supervision_only",
                postprocessing_layer_availability: "postprocessing_only",
                decoder_capability_label: "fixed_atom_refinement",
                compatible_backend_families: &[
                    GenerationBackendFamilyConfig::ConditionedDenoising,
                    GenerationBackendFamilyConfig::Heuristic,
                    GenerationBackendFamilyConfig::RepairOnly,
                    GenerationBackendFamilyConfig::RerankerOnly,
                    GenerationBackendFamilyConfig::ExternalWrapper,
                ],
                compatible_primary_objectives: &[
                    PrimaryObjectiveConfig::SurrogateReconstruction,
                    PrimaryObjectiveConfig::ConditionedDenoising,
                ],
            },
            Self::LigandRefinement => GenerationModeCompatibilityContract {
                generation_mode: self,
                claim_label: "ligand_refinement",
                supported: true,
                unsupported_reason: None,
                target_ligand_topology: true,
                target_ligand_geometry: true,
                fixed_atom_count: true,
                pocket_only_initialization: false,
                graph_growth: false,
                atom_count_source: self.atom_count_source_label(),
                topology_source: self.topology_source_label(),
                geometry_source: self.geometry_source_label(),
                pocket_context_availability: "inference_available",
                target_ligand_atom_type_availability: "target_supervision_only",
                target_ligand_topology_availability: "target_supervision_only",
                target_ligand_coordinate_availability: "target_supervision_only",
                postprocessing_layer_availability: "postprocessing_only",
                decoder_capability_label: "fixed_atom_refinement",
                compatible_backend_families: &[
                    GenerationBackendFamilyConfig::ConditionedDenoising,
                    GenerationBackendFamilyConfig::EnergyGuidedRefinement,
                    GenerationBackendFamilyConfig::Heuristic,
                    GenerationBackendFamilyConfig::RepairOnly,
                    GenerationBackendFamilyConfig::RerankerOnly,
                    GenerationBackendFamilyConfig::ExternalWrapper,
                ],
                compatible_primary_objectives: &[
                    PrimaryObjectiveConfig::SurrogateReconstruction,
                    PrimaryObjectiveConfig::ConditionedDenoising,
                ],
            },
            Self::FlowRefinement => GenerationModeCompatibilityContract {
                generation_mode: self,
                claim_label: "flow_refinement",
                supported: true,
                unsupported_reason: None,
                target_ligand_topology: true,
                target_ligand_geometry: true,
                fixed_atom_count: true,
                pocket_only_initialization: false,
                graph_growth: false,
                atom_count_source: self.atom_count_source_label(),
                topology_source: self.topology_source_label(),
                geometry_source: self.geometry_source_label(),
                pocket_context_availability: "inference_available",
                target_ligand_atom_type_availability: "target_supervision_only",
                target_ligand_topology_availability: "target_supervision_only",
                target_ligand_coordinate_availability: "target_supervision_only",
                postprocessing_layer_availability: "postprocessing_only",
                decoder_capability_label: "fixed_atom_refinement",
                compatible_backend_families: &[GenerationBackendFamilyConfig::FlowMatching],
                compatible_primary_objectives: &[
                    PrimaryObjectiveConfig::SurrogateReconstruction,
                    PrimaryObjectiveConfig::FlowMatching,
                    PrimaryObjectiveConfig::DenoisingFlowMatching,
                ],
            },
            Self::PocketOnlyInitializationBaseline => GenerationModeCompatibilityContract {
                generation_mode: self,
                claim_label: "pocket_only_initialization_baseline",
                supported: true,
                unsupported_reason: None,
                target_ligand_topology: false,
                target_ligand_geometry: false,
                fixed_atom_count: true,
                pocket_only_initialization: true,
                graph_growth: false,
                atom_count_source: self.atom_count_source_label(),
                topology_source: self.topology_source_label(),
                geometry_source: self.geometry_source_label(),
                pocket_context_availability: "inference_available",
                target_ligand_atom_type_availability: "target_supervision_only",
                target_ligand_topology_availability: "target_supervision_only",
                target_ligand_coordinate_availability: "target_supervision_only",
                postprocessing_layer_availability: "postprocessing_only",
                decoder_capability_label: "fixed_atom_refinement_with_pocket_only_initialization",
                compatible_backend_families: &[
                    GenerationBackendFamilyConfig::ConditionedDenoising,
                    GenerationBackendFamilyConfig::Heuristic,
                    GenerationBackendFamilyConfig::ExternalWrapper,
                ],
                compatible_primary_objectives: &[PrimaryObjectiveConfig::SurrogateReconstruction],
            },
            Self::DeNovoInitialization => GenerationModeCompatibilityContract {
                generation_mode: self,
                claim_label: "de_novo_pocket_conditioned_molecular_flow",
                supported: true,
                unsupported_reason: None,
                target_ligand_topology: false,
                target_ligand_geometry: false,
                fixed_atom_count: false,
                pocket_only_initialization: true,
                graph_growth: true,
                atom_count_source: self.atom_count_source_label(),
                topology_source: self.topology_source_label(),
                geometry_source: self.geometry_source_label(),
                pocket_context_availability: "inference_available",
                target_ligand_atom_type_availability: "target_supervision_only",
                target_ligand_topology_availability: "target_supervision_only",
                target_ligand_coordinate_availability: "target_supervision_only",
                postprocessing_layer_availability: "postprocessing_only",
                decoder_capability_label: "pocket_conditioned_graph_flow",
                compatible_backend_families: &[GenerationBackendFamilyConfig::FlowMatching],
                compatible_primary_objectives: &[
                    PrimaryObjectiveConfig::FlowMatching,
                    PrimaryObjectiveConfig::DenoisingFlowMatching,
                ],
            },
        }
    }
}

/// Generation-mode compatibility matrix consumed by validation and artifacts.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct GenerationModeCompatibilityContract {
    /// Generation mode described by this row.
    pub generation_mode: GenerationModeConfig,
    /// Claim-safe label for reports and reviewer artifacts.
    pub claim_label: &'static str,
    /// Whether the current implementation supports executing this mode.
    pub supported: bool,
    /// Explicit reason for unsupported reserved rows.
    pub unsupported_reason: Option<&'static str>,
    /// Whether target ligand topology is used by initialization or loss semantics.
    pub target_ligand_topology: bool,
    /// Whether target ligand geometry is used by initialization or loss semantics.
    pub target_ligand_geometry: bool,
    /// Whether the decoder preserves an externally supplied atom count.
    pub fixed_atom_count: bool,
    /// Whether the mode initializes ligand state from pocket-only priors.
    pub pocket_only_initialization: bool,
    /// Whether the mode grows molecular graphs and predicts stop/atom counts.
    pub graph_growth: bool,
    /// Stable atom-count source label.
    pub atom_count_source: &'static str,
    /// Stable topology source label.
    pub topology_source: &'static str,
    /// Stable geometry source label.
    pub geometry_source: &'static str,
    /// Pocket/context feature availability label for inference contracts.
    pub pocket_context_availability: &'static str,
    /// Target-ligand atom-type availability label.
    pub target_ligand_atom_type_availability: &'static str,
    /// Target-ligand topology availability label.
    pub target_ligand_topology_availability: &'static str,
    /// Target-ligand coordinate availability label.
    pub target_ligand_coordinate_availability: &'static str,
    /// Availability label for postprocessing-only candidate layers.
    pub postprocessing_layer_availability: &'static str,
    /// Required decoder capability label.
    pub decoder_capability_label: &'static str,
    /// Backend families that can execute this generation contract.
    pub compatible_backend_families: &'static [GenerationBackendFamilyConfig],
    /// Primary objectives that are shape-safe and semantically compatible.
    pub compatible_primary_objectives: &'static [PrimaryObjectiveConfig],
}

impl GenerationModeCompatibilityContract {
    /// Whether this row supports the selected backend family.
    pub fn supports_backend_family(self, family: GenerationBackendFamilyConfig) -> bool {
        self.compatible_backend_families.contains(&family)
    }

    /// Whether this row supports the selected primary objective.
    pub fn supports_primary_objective(self, objective: PrimaryObjectiveConfig) -> bool {
        self.compatible_primary_objectives.contains(&objective)
    }
}

/// Conservative initialization controls for the pocket-only baseline mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketOnlyInitializationConfig {
    /// Fixed atom-count prior used by the baseline initializer.
    #[serde(default = "default_pocket_only_atom_count")]
    pub atom_count: usize,
    /// Atom-type token used for every initialized atom.
    #[serde(default)]
    pub atom_type_token: i64,
    /// Fraction of pocket radius used for deterministic coordinate offsets.
    #[serde(default = "default_pocket_only_radius_fraction")]
    pub radius_fraction: f64,
    /// Seed used by deterministic coordinate offset construction.
    #[serde(default = "default_pocket_only_coordinate_seed")]
    pub coordinate_seed: u64,
}

impl Default for PocketOnlyInitializationConfig {
    fn default() -> Self {
        Self {
            atom_count: default_pocket_only_atom_count(),
            atom_type_token: 0,
            radius_fraction: default_pocket_only_radius_fraction(),
            coordinate_seed: default_pocket_only_coordinate_seed(),
        }
    }
}

impl PocketOnlyInitializationConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.atom_count == 0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.pocket_only_initialization.atom_count must be greater than zero",
            ));
        }
        if self.atom_type_token < 0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.pocket_only_initialization.atom_type_token must be non-negative",
            ));
        }
        if !self.radius_fraction.is_finite() || self.radius_fraction <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.pocket_only_initialization.radius_fraction must be finite and positive",
            ));
        }
        Ok(())
    }
}

fn default_pocket_only_atom_count() -> usize {
    16
}

fn default_pocket_only_radius_fraction() -> f64 {
    0.35
}

fn default_pocket_only_coordinate_seed() -> u64 {
    4242
}

/// Pocket-conditioned initializer controls for true de novo molecular flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeNovoInitializationConfig {
    /// Minimum atom count proposed from pocket context.
    #[serde(default = "default_de_novo_min_atom_count")]
    pub min_atom_count: usize,
    /// Maximum atom count proposed from pocket context.
    #[serde(default = "default_de_novo_max_atom_count")]
    pub max_atom_count: usize,
    /// Approximate pocket atoms represented by one ligand atom in the count policy.
    #[serde(default = "default_de_novo_pocket_atom_divisor")]
    pub pocket_atom_divisor: f64,
    /// Optional dataset-calibrated deterministic atom count for claim/debug sweeps.
    #[serde(default)]
    pub dataset_calibrated_atom_count: Option<usize>,
    /// Fraction of pocket radius used for the initial scaffold envelope.
    #[serde(default = "default_de_novo_radius_fraction")]
    pub radius_fraction: f64,
    /// Seed used by deterministic de novo atom-type and coordinate priors.
    #[serde(default = "default_de_novo_seed")]
    pub seed: u64,
}

impl Default for DeNovoInitializationConfig {
    fn default() -> Self {
        Self {
            min_atom_count: default_de_novo_min_atom_count(),
            max_atom_count: default_de_novo_max_atom_count(),
            pocket_atom_divisor: default_de_novo_pocket_atom_divisor(),
            dataset_calibrated_atom_count: None,
            radius_fraction: default_de_novo_radius_fraction(),
            seed: default_de_novo_seed(),
        }
    }
}

impl DeNovoInitializationConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.min_atom_count == 0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.de_novo_initialization.min_atom_count must be greater than zero",
            ));
        }
        if self.max_atom_count < self.min_atom_count {
            return Err(ConfigValidationError::new(
                "data.generation_target.de_novo_initialization.max_atom_count must be >= min_atom_count",
            ));
        }
        if !self.pocket_atom_divisor.is_finite() || self.pocket_atom_divisor <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.de_novo_initialization.pocket_atom_divisor must be finite and positive",
            ));
        }
        if let Some(atom_count) = self.dataset_calibrated_atom_count {
            if atom_count == 0 {
                return Err(ConfigValidationError::new(
                    "data.generation_target.de_novo_initialization.dataset_calibrated_atom_count must be greater than zero",
                ));
            }
            if atom_count < self.min_atom_count || atom_count > self.max_atom_count {
                return Err(ConfigValidationError::new(
                    "data.generation_target.de_novo_initialization.dataset_calibrated_atom_count must be within min_atom_count..=max_atom_count",
                ));
            }
        }
        if !self.radius_fraction.is_finite() || self.radius_fraction <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.generation_target.de_novo_initialization.radius_fraction must be finite and positive",
            ));
        }
        Ok(())
    }
}

fn default_de_novo_min_atom_count() -> usize {
    8
}

fn default_de_novo_max_atom_count() -> usize {
    64
}

fn default_de_novo_pocket_atom_divisor() -> f64 {
    6.0
}

fn default_de_novo_radius_fraction() -> f64 {
    0.45
}

fn default_de_novo_seed() -> u64 {
    9_001
}

/// Configurable corruption process used to derive decoder-side supervision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationTargetConfig {
    /// Explicit generation-mode contract for artifacts and claim boundaries.
    #[serde(default)]
    pub generation_mode: GenerationModeConfig,
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
    /// Geometric decay used only by detached rollout evaluation diagnostics.
    #[serde(
        default = "default_rollout_eval_step_weight_decay",
        alias = "training_step_weight_decay"
    )]
    pub rollout_eval_step_weight_decay: f64,
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
    /// Explicit policy for refreshing cross-modal inference context during rollout.
    #[serde(default)]
    pub context_refresh_policy: InferenceContextRefreshPolicy,
    /// Conservative pocket-only initializer used by the baseline generation mode.
    #[serde(default)]
    pub pocket_only_initialization: PocketOnlyInitializationConfig,
    /// Pocket-conditioned initializer used by `de_novo_initialization`.
    #[serde(default)]
    pub de_novo_initialization: DeNovoInitializationConfig,
}

impl Default for GenerationTargetConfig {
    fn default() -> Self {
        Self {
            generation_mode: GenerationModeConfig::default(),
            atom_mask_ratio: 0.15,
            coordinate_noise_std: 0.08,
            corruption_seed: 1337,
            rollout_steps: 4,
            min_rollout_steps: 2,
            stop_probability_threshold: 0.82,
            coordinate_step_scale: 0.8,
            rollout_eval_step_weight_decay: default_rollout_eval_step_weight_decay(),
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
            context_refresh_policy: InferenceContextRefreshPolicy::default(),
            pocket_only_initialization: PocketOnlyInitializationConfig::default(),
            de_novo_initialization: DeNovoInitializationConfig::default(),
        }
    }
}

impl GenerationTargetConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        self.pocket_only_initialization.validate()?;
        self.de_novo_initialization.validate()?;
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
        if !self.rollout_eval_step_weight_decay.is_finite()
            || self.rollout_eval_step_weight_decay <= 0.0
        {
            return Err(ConfigValidationError::new(
                "data.generation_target.rollout_eval_step_weight_decay must be finite and positive",
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
        self.context_refresh_policy.validate()?;
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

/// Inference-time policy for refreshing cross-modal generation context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum InferenceContextRefreshPolicy {
    /// Preserve the current static context for compatibility.
    #[default]
    Static,
    /// Refresh context at every rollout step.
    EveryStep,
    /// Refresh context at step 0 and then every `n` steps.
    PeriodicN { n: usize },
}

impl InferenceContextRefreshPolicy {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if let Self::PeriodicN { n } = self {
            if *n == 0 {
                return Err(ConfigValidationError::new(
                    "data.generation_target.context_refresh_policy.periodic_n.n must be greater than zero",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn should_refresh_at_step(&self, step_index: usize) -> bool {
        match self {
            Self::Static => false,
            Self::EveryStep => true,
            Self::PeriodicN { n } => step_index % *n == 0,
        }
    }

    pub(crate) fn label(&self) -> String {
        match self {
            Self::Static => "static".to_string(),
            Self::EveryStep => "every_step".to_string(),
            Self::PeriodicN { n } => format!("periodic_{n}"),
        }
    }
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

fn default_rollout_eval_step_weight_decay() -> f64 {
    0.9
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
