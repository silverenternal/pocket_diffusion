//! Configuration types for the modular research framework.

pub mod types;

pub use types::{
    load_research_config, AffinityWeighting, AutomatedSearchConfig, AutomatedSearchHardGateConfig,
    AutomatedSearchScoreWeightConfig, AutomatedSearchSpaceConfig, AutomatedSearchStrategy,
    ConfigValidationError, CrossAttentionMode, DataConfig, DataQualityFilterConfig, DatasetFormat,
    ExternalBackendCommandConfig, GenerationRolloutMode, GenerationTargetConfig,
    InteractionTuningConfig, LossWeightConfig, ModelConfig, ParsingMode, PrimaryObjectiveConfig,
    ResearchConfig, RuntimeConfig, StageScheduleConfig, TrainingConfig,
};
