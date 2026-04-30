//! Configuration types for the modular research framework.

pub mod types;

pub use types::{
    load_research_config, AdaptiveStageGuardConfig, AffinityWeighting, AutomatedSearchConfig,
    AutomatedSearchHardGateConfig, AutomatedSearchScoreWeightConfig, AutomatedSearchSpaceConfig,
    AutomatedSearchStrategy, BondRefinementConfig, ChemistryObjectiveWarmupConfig,
    ConfigValidationError, CrossAttentionMode, DataConfig, DataQualityFilterConfig, DatasetFormat,
    DecoderConditioningConfig, DecoderConditioningKind, ExplicitLeakageProbeTrainingSemantics,
    ExternalBackendCommandConfig, FlowBranchKind, FlowBranchLossWeights, FlowBranchScheduleConfig,
    FlowBranchScheduleEntry, FlowMatchingConfig, FlowMatchingIntegrationMethod,
    FlowTargetAlignmentPolicy, FlowVelocityHeadConfig, FlowVelocityHeadKind,
    GenerationBackendConfig, GenerationBackendFamilyConfig, GenerationMethodConfig,
    GenerationModeCompatibilityContract, GenerationModeConfig, GenerationRolloutMode,
    GenerationTargetConfig, GeometryEncoderConfig, GeometryOperatorKind,
    InferenceContextRefreshPolicy, InteractionGateMode, InteractionPathFlowTimeBucketMultiplier,
    InteractionPathGateRegularizationWeight, InteractionPathRolloutBucketMultiplier,
    InteractionPathStageMultiplier, InteractionTuningConfig, LossWeightConfig, ModalityFocusConfig,
    ModelConfig, MultiModalFlowConfig, MultiSampleInitializationConfig, ObjectiveBudgetAction,
    ObjectiveGradientDiagnosticsConfig, ObjectiveGradientSamplingMode,
    ObjectiveScaleDiagnosticsConfig, PairwiseGeometryConfig, ParsingMode, PharmacophoreProbeConfig,
    PocketEncoderConfig, PocketEncoderKind, PocketOnlyInitializationConfig, PrimaryObjectiveConfig,
    ResearchConfig, RolloutTrainingConfig, RolloutTrainingDetachPolicy, RuntimeConfig,
    SemanticProbeConfig, SlotDecompositionConfig, SparseTopologyCalibrationConfig,
    StageScheduleConfig, TemporalInteractionPolicyConfig, TopologyEncoderConfig,
    TopologyEncoderKind, TrainingConfig, TrainingResumeConfig,
};
