/// Toggles used to disable parts of the model or objective for ablation studies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    /// Disable slot decomposition metrics and slot loss reporting.
    pub disable_slots: bool,
    /// Disable gated cross-modal interaction metrics.
    pub disable_cross_attention: bool,
    /// Disable semantic probe metrics.
    pub disable_probes: bool,
    /// Disable leakage reporting.
    pub disable_leakage: bool,
    /// Disable ligand-pocket geometry bias inside controlled cross-attention.
    #[serde(default)]
    pub disable_geometry_interaction_bias: bool,
    /// Disable decoder-time pocket guidance during rollout.
    #[serde(default)]
    pub disable_rollout_pocket_guidance: bool,
    /// Disable candidate geometry repair before bond inference and scoring.
    #[serde(default)]
    pub disable_candidate_repair: bool,
    /// Override the primary objective used by this experiment variant.
    pub primary_objective_override: Option<crate::config::PrimaryObjectiveConfig>,
    /// Override the active generation backend for model-switch ablations.
    #[serde(default)]
    pub generation_backend_override: Option<crate::config::GenerationBackendConfig>,
    /// Override explicit generation semantics for generation-mode boundary ablations.
    #[serde(default)]
    pub generation_mode_override: Option<crate::config::GenerationModeConfig>,
    /// Override flow-matching transport controls for generation-mode boundary ablations.
    #[serde(default)]
    pub flow_matching_override: Option<crate::config::FlowMatchingConfig>,
    /// Override the geometry-flow velocity head for generation-alignment ablations.
    #[serde(default)]
    pub flow_velocity_head_override: Option<crate::config::FlowVelocityHeadConfig>,
    /// Override pairwise geometry messages used by flow velocity heads.
    #[serde(default)]
    pub pairwise_geometry_override: Option<crate::config::PairwiseGeometryConfig>,
    /// Override optimizer-facing short-rollout training controls.
    #[serde(default)]
    pub rollout_training_override: Option<crate::config::RolloutTrainingConfig>,
    /// Override staged loss weights for objective-family ablations.
    #[serde(default)]
    pub loss_weights_override: Option<crate::config::LossWeightConfig>,
    /// Override pharmacophore role probes for rich pocket-interaction ablations.
    #[serde(default)]
    pub pharmacophore_probes_override: Option<crate::config::PharmacophoreProbeConfig>,
    /// Override slot count for representation-capacity ablations.
    #[serde(default)]
    pub num_slots_override: Option<i64>,
    /// Override active-slot masking for slot masking ablations.
    #[serde(default)]
    pub slot_attention_masking_override: Option<bool>,
    /// Override gate sparsity weight for controlled-interaction ablations.
    #[serde(default)]
    pub eta_gate_override: Option<f64>,
    /// Override interaction gate temperature for gate-scale ablations.
    #[serde(default)]
    pub interaction_gate_temperature_override: Option<f64>,
    /// Override leakage penalty weight for specialization ablations.
    #[serde(default)]
    pub delta_leak_override: Option<f64>,
    /// Disable intra-modality redundancy regularization for objective ablations.
    #[serde(default)]
    pub disable_redundancy: bool,
    /// Collapse staged training gates so every configured objective is active from step zero.
    #[serde(default)]
    pub disable_staged_schedule: bool,
    /// Override downstream modality visibility for topology/geometry/pocket-only controls.
    #[serde(default)]
    pub modality_focus_override: Option<crate::config::ModalityFocusConfig>,
    /// Override topology encoder family for encoder-capacity ablations.
    #[serde(default)]
    pub topology_encoder_kind_override: Option<crate::config::TopologyEncoderKind>,
    /// Override geometry operator family for geometry-representation ablations.
    #[serde(default)]
    pub geometry_operator_override: Option<crate::config::GeometryOperatorKind>,
    /// Override pocket encoder family for context-encoder ablations.
    #[serde(default)]
    pub pocket_encoder_kind_override: Option<crate::config::PocketEncoderKind>,
    /// Override decoder conditioning path for local-versus-global conditioning ablations.
    #[serde(default)]
    pub decoder_conditioning_override: Option<crate::config::DecoderConditioningKind>,
    /// Override the cross-modal interaction block style for controlled interaction ablations.
    pub interaction_mode_override: Option<CrossAttentionMode>,
    /// Human-readable variant label for persisted comparison artifacts.
    pub variant_label: Option<String>,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            disable_slots: false,
            disable_cross_attention: false,
            disable_probes: false,
            disable_leakage: false,
            disable_geometry_interaction_bias: false,
            disable_rollout_pocket_guidance: false,
            disable_candidate_repair: false,
            primary_objective_override: None,
            generation_backend_override: None,
            generation_mode_override: None,
            flow_matching_override: None,
            flow_velocity_head_override: None,
            pairwise_geometry_override: None,
            rollout_training_override: None,
            loss_weights_override: None,
            pharmacophore_probes_override: None,
            num_slots_override: None,
            slot_attention_masking_override: None,
            eta_gate_override: None,
            interaction_gate_temperature_override: None,
            delta_leak_override: None,
            disable_redundancy: false,
            disable_staged_schedule: false,
            modality_focus_override: None,
            topology_encoder_kind_override: None,
            geometry_operator_override: None,
            pocket_encoder_kind_override: None,
            decoder_conditioning_override: None,
            interaction_mode_override: None,
            variant_label: None,
        }
    }
}

/// Executable backend configuration for chemistry, docking, and pocket scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalEvaluationConfig {
    /// External chemistry-validity command adapter.
    #[serde(default)]
    pub chemistry_backend: ExternalBackendCommandConfig,
    /// External docking or pocket-affinity command adapter.
    #[serde(default)]
    pub docking_backend: ExternalBackendCommandConfig,
    /// External pocket-compatibility command adapter.
    #[serde(default)]
    pub pocket_backend: ExternalBackendCommandConfig,
    /// Persist compact raw/repaired/scored candidate artifacts for regression review.
    #[serde(default = "default_persist_generation_artifacts")]
    pub persist_generation_artifacts: bool,
    /// Maximum number of examples represented in per-split candidate artifacts.
    #[serde(default = "default_generation_artifact_example_limit")]
    pub generation_artifact_example_limit: usize,
    /// Maximum number of candidate records retained per layer in artifacts.
    #[serde(default = "default_generation_artifact_candidate_limit")]
    pub generation_artifact_candidate_limit: usize,
}

impl Default for ExternalEvaluationConfig {
    fn default() -> Self {
        Self {
            chemistry_backend: ExternalBackendCommandConfig::default(),
            docking_backend: ExternalBackendCommandConfig::default(),
            pocket_backend: ExternalBackendCommandConfig::default(),
            persist_generation_artifacts: default_persist_generation_artifacts(),
            generation_artifact_example_limit: default_generation_artifact_example_limit(),
            generation_artifact_candidate_limit: default_generation_artifact_candidate_limit(),
        }
    }
}

impl ExternalEvaluationConfig {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.chemistry_backend
            .validate("external_evaluation.chemistry_backend")?;
        self.docking_backend
            .validate("external_evaluation.docking_backend")?;
        self.pocket_backend
            .validate("external_evaluation.pocket_backend")?;
        if self.persist_generation_artifacts
            && (self.generation_artifact_example_limit == 0
                || self.generation_artifact_candidate_limit == 0)
        {
            return Err(
                "external_evaluation generation artifact limits must be positive when enabled"
                    .into(),
            );
        }
        Ok(())
    }
}

fn default_persist_generation_artifacts() -> bool {
    true
}

fn default_generation_artifact_example_limit() -> usize {
    8
}

fn default_generation_artifact_candidate_limit() -> usize {
    32
}

/// Lightweight config-driven ablation matrix over major architectural controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationMatrixConfig {
    /// Whether to execute the matrix after the base experiment.
    pub enabled: bool,
    /// Whether to include a surrogate reconstruction baseline.
    pub include_surrogate_objective: bool,
    /// Whether to include a conditioned-denoising baseline.
    pub include_conditioned_denoising: bool,
    /// Whether to include executable generation-mode boundary variants.
    #[serde(default = "default_include_generation_mode_ablation")]
    pub include_generation_mode_ablation: bool,
    /// Whether to include a variant with slots disabled in reporting.
    pub include_disable_slots: bool,
    /// Whether to include a variant with cross-attention disabled in reporting.
    pub include_disable_cross_attention: bool,
    /// Whether to include a variant with probes disabled in reporting.
    pub include_disable_probes: bool,
    /// Whether to include a variant with leakage diagnostics disabled.
    #[serde(default = "default_include_disable_leakage")]
    pub include_disable_leakage: bool,
    /// Whether to include the lightweight topology encoder baseline.
    #[serde(default = "default_include_topology_encoder_ablation")]
    pub include_topology_encoder_ablation: bool,
    /// Whether to include the raw-coordinate geometry operator baseline.
    #[serde(default = "default_include_geometry_operator_ablation")]
    pub include_geometry_operator_ablation: bool,
    /// Whether to include the feature-projection pocket encoder baseline.
    #[serde(default = "default_include_pocket_encoder_ablation")]
    pub include_pocket_encoder_ablation: bool,
    /// Whether to include mean-pooled decoder conditioning against local atom-slot conditioning.
    #[serde(default = "default_include_decoder_conditioning_ablation")]
    pub include_decoder_conditioning_ablation: bool,
    /// Whether to include a lightweight controlled-interaction baseline.
    #[serde(default = "default_include_lightweight_interaction")]
    pub include_lightweight_interaction: bool,
    /// Whether to include a Transformer-style controlled-interaction variant.
    #[serde(default = "default_include_transformer_interaction")]
    pub include_transformer_interaction: bool,
    /// Whether to include a direct-fusion negative control.
    #[serde(default = "default_include_direct_fusion_negative_control")]
    pub include_direct_fusion_negative_control: bool,
    /// Whether to include the no geometry-biased controlled-attention variant.
    #[serde(default = "default_include_geometry_bias_ablation")]
    pub include_disable_geometry_interaction_bias: bool,
    /// Whether to include the rollout-only pocket-guidance disabled variant.
    #[serde(default = "default_include_rollout_guidance_ablation")]
    pub include_disable_rollout_pocket_guidance: bool,
    /// Whether to include the no candidate-repair variant.
    #[serde(default = "default_include_candidate_repair_ablation")]
    pub include_disable_candidate_repair: bool,
    /// Whether to include backend-family comparison variants.
    #[serde(default = "default_include_backend_family_ablation")]
    pub include_backend_family_ablation: bool,
    /// Whether to include a reduced slot-count variant.
    #[serde(default = "default_include_slot_count_ablation")]
    pub include_slot_count_ablation: bool,
    /// Whether to include active-slot attention masking as a direct ablation.
    #[serde(default = "default_include_slot_attention_masking_ablation")]
    pub include_slot_attention_masking_ablation: bool,
    /// Whether to include a gate-sparsity disabled variant.
    #[serde(default = "default_include_gate_sparsity_ablation")]
    pub include_gate_sparsity_ablation: bool,
    /// Whether to include gate-temperature scale variants.
    #[serde(default = "default_include_gate_scale_ablation")]
    pub include_gate_scale_ablation: bool,
    /// Whether to include a leakage-penalty disabled variant.
    #[serde(default = "default_include_leakage_penalty_ablation")]
    pub include_leakage_penalty_ablation: bool,
    /// Whether to include a redundancy-loss disabled variant.
    #[serde(default = "default_include_redundancy_ablation")]
    pub include_redundancy_ablation: bool,
    /// Whether to include topology-only, geometry-only, and pocket-only variants.
    #[serde(default = "default_include_modality_focus_ablation")]
    pub include_modality_focus_ablation: bool,
    /// Whether to include a no-staged-schedule variant.
    #[serde(default = "default_include_staged_schedule_ablation")]
    pub include_staged_schedule_ablation: bool,
    /// Whether to include MLP-vs-equivariant geometry-flow head variants.
    #[serde(default = "default_include_generation_alignment_flow_head_ablation")]
    pub include_generation_alignment_flow_head_ablation: bool,
    /// Whether to include optimizer-facing rollout-training on/off variants.
    #[serde(default = "default_include_generation_alignment_rollout_training_ablation")]
    pub include_generation_alignment_rollout_training_ablation: bool,
    /// Whether to include native chemistry-constraint on/off variants.
    #[serde(default = "default_include_generation_alignment_chemistry_ablation")]
    pub include_generation_alignment_chemistry_ablation: bool,
    /// Whether to include thin-vs-rich pocket-interaction loss variants.
    #[serde(default = "default_include_generation_alignment_pocket_interaction_ablation")]
    pub include_generation_alignment_pocket_interaction_ablation: bool,
}

impl Default for AblationMatrixConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            include_surrogate_objective: true,
            include_conditioned_denoising: true,
            include_generation_mode_ablation: default_include_generation_mode_ablation(),
            include_disable_slots: true,
            include_disable_cross_attention: true,
            include_disable_probes: false,
            include_disable_leakage: default_include_disable_leakage(),
            include_topology_encoder_ablation: default_include_topology_encoder_ablation(),
            include_geometry_operator_ablation: default_include_geometry_operator_ablation(),
            include_pocket_encoder_ablation: default_include_pocket_encoder_ablation(),
            include_decoder_conditioning_ablation: default_include_decoder_conditioning_ablation(),
            include_lightweight_interaction: true,
            include_transformer_interaction: true,
            include_direct_fusion_negative_control:
                default_include_direct_fusion_negative_control(),
            include_disable_geometry_interaction_bias: default_include_geometry_bias_ablation(),
            include_disable_rollout_pocket_guidance: default_include_rollout_guidance_ablation(),
            include_disable_candidate_repair: default_include_candidate_repair_ablation(),
            include_backend_family_ablation: default_include_backend_family_ablation(),
            include_slot_count_ablation: default_include_slot_count_ablation(),
            include_slot_attention_masking_ablation:
                default_include_slot_attention_masking_ablation(),
            include_gate_sparsity_ablation: default_include_gate_sparsity_ablation(),
            include_gate_scale_ablation: default_include_gate_scale_ablation(),
            include_leakage_penalty_ablation: default_include_leakage_penalty_ablation(),
            include_redundancy_ablation: default_include_redundancy_ablation(),
            include_modality_focus_ablation: default_include_modality_focus_ablation(),
            include_staged_schedule_ablation: default_include_staged_schedule_ablation(),
            include_generation_alignment_flow_head_ablation:
                default_include_generation_alignment_flow_head_ablation(),
            include_generation_alignment_rollout_training_ablation:
                default_include_generation_alignment_rollout_training_ablation(),
            include_generation_alignment_chemistry_ablation:
                default_include_generation_alignment_chemistry_ablation(),
            include_generation_alignment_pocket_interaction_ablation:
                default_include_generation_alignment_pocket_interaction_ablation(),
        }
    }
}

fn default_include_backend_family_ablation() -> bool {
    true
}

fn default_include_slot_count_ablation() -> bool {
    true
}

fn default_include_generation_mode_ablation() -> bool {
    false
}

fn default_include_slot_attention_masking_ablation() -> bool {
    false
}

fn default_include_gate_sparsity_ablation() -> bool {
    true
}

fn default_include_gate_scale_ablation() -> bool {
    false
}

fn default_include_direct_fusion_negative_control() -> bool {
    false
}

fn default_include_leakage_penalty_ablation() -> bool {
    true
}

fn default_include_redundancy_ablation() -> bool {
    true
}

fn default_include_modality_focus_ablation() -> bool {
    false
}

fn default_include_staged_schedule_ablation() -> bool {
    false
}

fn default_include_generation_alignment_flow_head_ablation() -> bool {
    false
}

fn default_include_generation_alignment_rollout_training_ablation() -> bool {
    false
}

fn default_include_generation_alignment_chemistry_ablation() -> bool {
    false
}

fn default_include_generation_alignment_pocket_interaction_ablation() -> bool {
    false
}

fn default_include_disable_leakage() -> bool {
    false
}

fn default_include_topology_encoder_ablation() -> bool {
    false
}

fn default_include_geometry_operator_ablation() -> bool {
    false
}

fn default_include_pocket_encoder_ablation() -> bool {
    false
}

fn default_include_decoder_conditioning_ablation() -> bool {
    false
}

fn default_include_geometry_bias_ablation() -> bool {
    true
}

fn default_include_rollout_guidance_ablation() -> bool {
    true
}

fn default_include_candidate_repair_ablation() -> bool {
    true
}

fn default_include_lightweight_interaction() -> bool {
    true
}

fn default_include_transformer_interaction() -> bool {
    true
}

/// High-level experiment configuration for unseen-pocket generalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnseenPocketExperimentConfig {
    /// Core model/training/runtime configuration.
    pub research: ResearchConfig,
    /// Stable label used by shared cross-surface reporting.
    pub surface_label: Option<String>,
    /// Optional reviewer benchmark metadata used for claim-bearing evidence tiers.
    #[serde(default)]
    pub reviewer_benchmark: ReviewerBenchmarkConfig,
    /// Ablation toggles.
    pub ablation: AblationConfig,
    /// Optional external chemistry/docking/pocket backends.
    #[serde(default)]
    pub external_evaluation: ExternalEvaluationConfig,
    /// Optional ablation matrix executed from the same config.
    #[serde(default)]
    pub ablation_matrix: AblationMatrixConfig,
    /// Optional cross-surface automated search loop rooted at this config.
    #[serde(default)]
    pub automated_search: AutomatedSearchConfig,
    /// Optional multi-seed runner settings for claim-bearing uncertainty reports.
    #[serde(default)]
    pub multi_seed: MultiSeedExperimentConfig,
    /// Optional coarse performance regression thresholds.
    #[serde(default)]
    pub performance_gates: PerformanceGateConfig,
}

impl UnseenPocketExperimentConfig {
    /// Validate a config before allocating runtime state.
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.research.validate()?;
        if !PocketGenerationMethodRegistry::contains(self.research.generation_method.primary_backend_id()) {
            let family = format!(
                "{:?}",
                self.research.generation_method.primary_backend.family
            )
            .to_ascii_lowercase();
            return Err(format!(
                "unknown generation_method.active_method `{}` (family={family}, config_path=unavailable)",
                self.research.generation_method.primary_backend_id(),
            )
            .into());
        }
        PocketGenerationMethodRegistry::validate_backend_config(
            &self.research.generation_method.primary_backend,
        )
        .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;
        for backend in &self.research.generation_method.comparison_backends {
            PocketGenerationMethodRegistry::validate_backend_config(backend)
                .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;
        }
        for method_id in self.research.generation_method.comparison_backend_ids() {
            if !PocketGenerationMethodRegistry::contains(&method_id) {
                return Err(format!(
                    "unknown generation_method.comparison_methods entry `{method_id}` (family=unknown, config_path=unavailable)"
                )
                .into());
            }
        }
        self.automated_search.validate()?;
        self.external_evaluation.validate()?;
        self.performance_gates.validate()?;
        if self.multi_seed.enabled && self.multi_seed.seeds.is_empty() {
            return Err("multi_seed.seeds must be non-empty when multi_seed.enabled=true".into());
        }
        Ok(())
    }
}

/// Validate an experiment config and attach the source path to fail-fast errors.
pub fn validate_experiment_config_with_source(
    config: &UnseenPocketExperimentConfig,
    source_path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    config.validate().map_err(|error| {
        format!(
            "{}; config_path={}",
            error,
            source_path.as_ref().display()
        )
        .into()
    })
}

impl Default for UnseenPocketExperimentConfig {
    fn default() -> Self {
        Self {
            research: ResearchConfig::default(),
            surface_label: None,
            reviewer_benchmark: ReviewerBenchmarkConfig::default(),
            ablation: AblationConfig::default(),
            external_evaluation: ExternalEvaluationConfig::default(),
            ablation_matrix: AblationMatrixConfig::default(),
            automated_search: AutomatedSearchConfig::default(),
            multi_seed: MultiSeedExperimentConfig::default(),
            performance_gates: PerformanceGateConfig::default(),
        }
    }
}

/// Optional metadata describing which reviewer-facing benchmark contract a surface targets.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReviewerBenchmarkConfig {
    /// Stable benchmark dataset label surfaced in claim summaries and reviewer bundles.
    #[serde(default)]
    pub dataset: Option<String>,
}

/// Optional coarse performance gates for claim and medium-profile runs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceGateConfig {
    /// Minimum validation examples/sec required when set.
    #[serde(default)]
    pub min_validation_examples_per_second: Option<f64>,
    /// Minimum test examples/sec required when set.
    #[serde(default)]
    pub min_test_examples_per_second: Option<f64>,
    /// Maximum validation memory delta in MB allowed when set.
    #[serde(default)]
    pub max_validation_memory_mb: Option<f64>,
    /// Maximum test memory delta in MB allowed when set.
    #[serde(default)]
    pub max_test_memory_mb: Option<f64>,
    /// Minimum raw-native test valid fraction required when set.
    #[serde(default)]
    pub min_test_raw_model_valid_fraction: Option<f64>,
    /// Minimum raw-native test pocket-contact fraction required when set.
    #[serde(default)]
    pub min_test_raw_model_pocket_contact_fraction: Option<f64>,
    /// Maximum raw-native test clash fraction allowed when set.
    #[serde(default)]
    pub max_test_raw_model_clash_fraction: Option<f64>,
    /// Minimum raw-native graph validity fraction required when set.
    #[serde(default)]
    pub min_test_raw_native_graph_valid_fraction: Option<f64>,
}

impl PerformanceGateConfig {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        for (name, value) in [
            (
                "performance_gates.min_validation_examples_per_second",
                self.min_validation_examples_per_second,
            ),
            (
                "performance_gates.min_test_examples_per_second",
                self.min_test_examples_per_second,
            ),
            (
                "performance_gates.max_validation_memory_mb",
                self.max_validation_memory_mb,
            ),
            (
                "performance_gates.max_test_memory_mb",
                self.max_test_memory_mb,
            ),
            (
                "performance_gates.min_test_raw_model_valid_fraction",
                self.min_test_raw_model_valid_fraction,
            ),
            (
                "performance_gates.min_test_raw_model_pocket_contact_fraction",
                self.min_test_raw_model_pocket_contact_fraction,
            ),
            (
                "performance_gates.max_test_raw_model_clash_fraction",
                self.max_test_raw_model_clash_fraction,
            ),
            (
                "performance_gates.min_test_raw_native_graph_valid_fraction",
                self.min_test_raw_native_graph_valid_fraction,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value < 0.0 {
                    return Err(
                        format!("{name} must be omitted or a non-negative finite number").into(),
                    );
                }
            }
        }
        for (name, value) in [
            (
                "performance_gates.min_test_raw_model_valid_fraction",
                self.min_test_raw_model_valid_fraction,
            ),
            (
                "performance_gates.min_test_raw_model_pocket_contact_fraction",
                self.min_test_raw_model_pocket_contact_fraction,
            ),
            (
                "performance_gates.max_test_raw_model_clash_fraction",
                self.max_test_raw_model_clash_fraction,
            ),
            (
                "performance_gates.min_test_raw_native_graph_valid_fraction",
                self.min_test_raw_native_graph_valid_fraction,
            ),
        ] {
            if let Some(value) = value {
                if value > 1.0 {
                    return Err(format!("{name} must be in [0, 1] when set").into());
                }
            }
        }
        Ok(())
    }
}

/// Config for deterministic repeated runs over split, corruption, and sampling seeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSeedExperimentConfig {
    /// Whether this config is intended for the multi-seed entrypoint.
    #[serde(default)]
    pub enabled: bool,
    /// Base seeds. Each seed drives split, corruption, and sampling seeds with stable offsets.
    #[serde(default = "default_multi_seed_values")]
    pub seeds: Vec<u64>,
    /// Directory that owns per-seed artifacts and the aggregate report.
    #[serde(default = "default_multi_seed_artifact_root")]
    pub artifact_root_dir: PathBuf,
}

impl Default for MultiSeedExperimentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            seeds: default_multi_seed_values(),
            artifact_root_dir: default_multi_seed_artifact_root(),
        }
    }
}

fn default_multi_seed_values() -> Vec<u64> {
    vec![17, 42, 101]
}

fn default_multi_seed_artifact_root() -> PathBuf {
    PathBuf::from("./checkpoints/multi_seed")
}

/// Persisted automated multi-surface search summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchSummary {
    /// Root directory that owns the search artifacts.
    pub artifact_root: PathBuf,
    /// Search strategy used for this run.
    pub strategy: AutomatedSearchStrategy,
    /// Shared hard regression gates enforced across surfaces.
    pub hard_gates: AutomatedSearchHardGateConfig,
    /// Multi-objective score weights used after hard-gate filtering.
    pub score_weights: AutomatedSearchScoreWeightConfig,
    /// Executed candidates in ranked order.
    pub ranked_candidates: Vec<AutomatedSearchCandidateSummary>,
    /// Winning candidate identifier when at least one candidate survives gating.
    pub winning_candidate_id: Option<String>,
    /// Aggregate roadmap recommendation after the search cycle.
    pub roadmap_decision: String,
}

/// Persisted aggregate over deterministic seed repeats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSeedExperimentSummary {
    /// Root directory containing per-seed artifacts.
    pub artifact_root: PathBuf,
    /// Source config used to launch the run.
    pub source_config: PathBuf,
    /// One row per seed.
    pub seed_runs: Vec<MultiSeedRunSummary>,
    /// Aggregate claim and stability metrics across seeds.
    pub aggregates: MultiSeedAggregateReport,
    /// Decision text for whether this surface is stable enough for claim review.
    pub stability_decision: String,
}

impl MultiSeedExperimentSummary {
    fn with_aggregates(mut self) -> Self {
        self.aggregates = MultiSeedAggregateReport {
            candidate_valid_fraction: self.aggregate(|run| run.candidate_valid_fraction),
            strict_pocket_fit_score: self.aggregate(|run| run.strict_pocket_fit_score),
            unique_smiles_fraction: self.aggregate(|run| run.unique_smiles_fraction),
            rdkit_unique_smiles_fraction: self.aggregate(|run| run.rdkit_unique_smiles_fraction),
            slot_activation_mean: self.aggregate(|run| run.slot_activation_mean),
            gate_activation_mean: self.aggregate(|run| run.gate_activation_mean),
            leakage_proxy_mean: self.aggregate(|run| run.leakage_proxy_mean),
            topology_signature_similarity: self.aggregate(|run| run.topology_signature_similarity),
            geometry_signature_similarity: self.aggregate(|run| run.geometry_signature_similarity),
            pocket_signature_similarity: self.aggregate(|run| run.pocket_signature_similarity),
            examples_per_second: self.aggregate(|run| run.examples_per_second),
            test_examples_per_second: self.aggregate(|run| run.test_examples_per_second),
        };
        self.stability_decision = self.stability_decision();
        self
    }

    /// Aggregate one metric across seeds.
    pub fn aggregate(&self, metric: fn(&MultiSeedRunSummary) -> f64) -> MultiSeedMetricAggregate {
        let values = self.seed_runs.iter().map(metric).collect::<Vec<_>>();
        MultiSeedMetricAggregate::from_values(&values)
    }

    /// Stability decision combining slot, leakage, gate, geometry, and quality signals.
    pub fn stability_decision(&self) -> String {
        let validity = self.aggregate(|run| run.candidate_valid_fraction);
        let leakage = self.aggregate(|run| run.leakage_proxy_mean);
        let slot = self.aggregate(|run| run.slot_activation_mean);
        let pocket = self.aggregate(|run| run.strict_pocket_fit_score);
        let real_backend_backed = self.seed_runs.iter().all(|run| run.real_backend_backed);
        if self.seed_runs.len() < 2 {
            return "insufficient seeds for claim-bearing stability; run at least two seeds"
                .to_string();
        }
        if validity.min < 0.2 || pocket.min < 0.05 {
            return "unstable for claim use: at least one seed has weak validity or pocket fit"
                .to_string();
        }
        if leakage.max > 0.95 && pocket.mean < 0.2 {
            return "unstable for claim use: leakage control coincides with weak pocket fit"
                .to_string();
        }
        if slot.mean <= 0.0 {
            return "unstable for claim use: slot activation collapsed across seeds".to_string();
        }
        if real_backend_backed {
            "stable enough for larger-data real-backend claim review on the held-out pocket surface"
                .to_string()
        } else {
            "stable enough for larger-data held-out-pocket review; upgrade this surface with real backends before stronger reviewer claims"
                .to_string()
        }
    }
}

/// Aggregate report persisted by the multi-seed runner.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiSeedAggregateReport {
    /// Aggregate candidate validity fraction.
    pub candidate_valid_fraction: MultiSeedMetricAggregate,
    /// Aggregate strict pocket-fit score.
    pub strict_pocket_fit_score: MultiSeedMetricAggregate,
    /// Aggregate unique chemistry fraction.
    pub unique_smiles_fraction: MultiSeedMetricAggregate,
    /// Aggregate toolkit-backed unique chemistry fraction on the test split.
    #[serde(default)]
    pub rdkit_unique_smiles_fraction: MultiSeedMetricAggregate,
    /// Aggregate slot activation.
    pub slot_activation_mean: MultiSeedMetricAggregate,
    /// Aggregate gate activation.
    pub gate_activation_mean: MultiSeedMetricAggregate,
    /// Aggregate leakage proxy.
    pub leakage_proxy_mean: MultiSeedMetricAggregate,
    /// Aggregate topology slot-signature similarity.
    pub topology_signature_similarity: MultiSeedMetricAggregate,
    /// Aggregate geometry slot-signature similarity.
    pub geometry_signature_similarity: MultiSeedMetricAggregate,
    /// Aggregate pocket slot-signature similarity.
    pub pocket_signature_similarity: MultiSeedMetricAggregate,
    /// Aggregate evaluation throughput.
    pub examples_per_second: MultiSeedMetricAggregate,
    /// Aggregate test-split throughput for reviewer-facing summaries.
    #[serde(default)]
    pub test_examples_per_second: MultiSeedMetricAggregate,
}

/// Per-seed claim-surface summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSeedRunSummary {
    /// Base seed used for this run.
    pub seed: u64,
    /// Artifact directory for this seed.
    pub artifact_dir: PathBuf,
    /// Config hash persisted by the seed run.
    pub config_hash: String,
    /// Candidate validity fraction on the test split.
    pub candidate_valid_fraction: f64,
    /// Strict pocket-fit score on the test split.
    pub strict_pocket_fit_score: f64,
    /// Unique chemistry fraction on the test split.
    pub unique_smiles_fraction: f64,
    /// Toolkit-backed unique chemistry fraction on the test split.
    #[serde(default)]
    pub rdkit_unique_smiles_fraction: f64,
    /// Mean active slot fraction on the test split.
    pub slot_activation_mean: f64,
    /// Mean directed gate activation on the test split.
    pub gate_activation_mean: f64,
    /// Mean leakage proxy on the test split.
    pub leakage_proxy_mean: f64,
    /// Mean topology slot signature similarity.
    pub topology_signature_similarity: f64,
    /// Mean geometry slot signature similarity.
    pub geometry_signature_similarity: f64,
    /// Mean pocket slot signature similarity.
    pub pocket_signature_similarity: f64,
    /// Evaluation throughput for the test split.
    pub examples_per_second: f64,
    /// Explicit alias for test-split throughput in reviewer-facing summaries.
    #[serde(default)]
    pub test_examples_per_second: f64,
    /// Whether this seed run used active real backends on the claim surface.
    #[serde(default)]
    pub real_backend_backed: bool,
}

impl MultiSeedRunSummary {
    fn from_experiment(seed: u64, summary: &UnseenPocketExperimentSummary) -> Self {
        let test = &summary.test.comparison_summary;
        let slot = &summary.test.slot_stability;
        Self {
            seed,
            artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
            config_hash: summary.reproducibility.config_hash.clone(),
            candidate_valid_fraction: test.candidate_valid_fraction.unwrap_or(0.0),
            strict_pocket_fit_score: test.strict_pocket_fit_score.unwrap_or(0.0),
            unique_smiles_fraction: test.unique_smiles_fraction.unwrap_or(0.0),
            rdkit_unique_smiles_fraction: test.unique_smiles_fraction.unwrap_or(0.0),
            slot_activation_mean: test.slot_activation_mean,
            gate_activation_mean: test.gate_activation_mean,
            leakage_proxy_mean: test.leakage_proxy_mean,
            topology_signature_similarity: slot.topology_signature_similarity,
            geometry_signature_similarity: slot.geometry_signature_similarity,
            pocket_signature_similarity: slot.pocket_signature_similarity,
            examples_per_second: summary.test.resource_usage.examples_per_second,
            test_examples_per_second: summary.test.resource_usage.examples_per_second,
            real_backend_backed: claim_is_real_backend_backed(summary),
        }
    }
}

/// Mean/std/min/max aggregate for one seed-level metric.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiSeedMetricAggregate {
    /// Number of finite seed values included in the aggregate.
    pub count: usize,
    /// Mean value.
    pub mean: f64,
    /// Population standard deviation.
    pub std: f64,
    /// Standard error of the mean.
    pub standard_error: f64,
    /// Lower bound of a deterministic 95% t-style confidence interval for the mean.
    pub confidence95_low: f64,
    /// Upper bound of a deterministic 95% t-style confidence interval for the mean.
    pub confidence95_high: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
}

impl MultiSeedMetricAggregate {
    fn from_values(values: &[f64]) -> Self {
        let finite_values = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if finite_values.is_empty() {
            return Self::default();
        }
        let count = finite_values.len();
        let mean = finite_values.iter().sum::<f64>() / count as f64;
        let std = (finite_values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count as f64)
            .sqrt();
        let sample_std = if count > 1 {
            (finite_values
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (count - 1) as f64)
                .sqrt()
        } else {
            0.0
        };
        let standard_error = if count > 0 {
            sample_std / (count as f64).sqrt()
        } else {
            0.0
        };
        let half_width = t_critical_95(count.saturating_sub(1)) * standard_error;
        let min = finite_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = finite_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Self {
            count,
            mean,
            std,
            standard_error,
            confidence95_low: mean - half_width,
            confidence95_high: mean + half_width,
            min,
            max,
        }
    }
}

fn t_critical_95(degrees_of_freedom: usize) -> f64 {
    match degrees_of_freedom {
        0 => 0.0,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20..=29 => 2.045,
        30..=39 => 2.023,
        40..=59 => 2.000,
        60..=119 => 1.980,
        _ => 1.960,
    }
}

/// Persisted summary for one automated-search candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchCandidateSummary {
    /// Stable candidate identifier.
    pub candidate_id: String,
    /// Human-readable override labels applied to this candidate.
    pub overrides: Vec<String>,
    /// Artifact directory that owns this candidate.
    pub artifact_dir: PathBuf,
    /// Surface-level claim reports collected for this candidate.
    pub surfaces: Vec<AutomatedSearchSurfaceSummary>,
    /// Shared interaction review built from all surface artifacts when available.
    pub aggregate_interaction_review: Option<InteractionModeReview>,
    /// Hard-gate evaluation result.
    pub gate_result: AutomatedSearchGateResult,
    /// Multi-objective score when the candidate survives hard gates.
    pub score: Option<f64>,
}

/// Compact persisted summary for one scored surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchSurfaceSummary {
    /// Surface label used in reports.
    pub surface_label: String,
    /// Source config path for this surface.
    pub source_config: PathBuf,
    /// Artifact directory produced for this surface.
    pub artifact_dir: PathBuf,
    /// Compact claim report emitted by the surface run.
    pub claim_report: ClaimReport,
}

/// Hard-gate result attached to each candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSearchGateResult {
    /// Whether the candidate survives all hard gates.
    pub passed: bool,
    /// Explicit blocking reasons when the candidate fails.
    pub blocked_reasons: Vec<String>,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    id: String,
    overrides: Vec<SearchOverride>,
}

#[derive(Debug, Clone)]
enum SearchOverride {
    GateTemperature(f64),
    GateBias(f64),
    AttentionResidualScale(f64),
    FfnResidualScale(f64),
    RolloutSteps(usize),
    MinRolloutSteps(usize),
    StopProbabilityThreshold(f64),
    CoordinateStepScale(f64),
    RolloutEvalStepWeightDecay(f64),
    CoordinateMomentum(f64),
    AtomMomentum(f64),
    AtomCommitTemperature(f64),
    MaxCoordinateDeltaNorm(f64),
    StopDeltaThreshold(f64),
    StopPatience(usize),
    BetaIntraRed(f64),
    GammaProbe(f64),
    DeltaLeak(f64),
    EtaGate(f64),
    MuSlot(f64),
}
