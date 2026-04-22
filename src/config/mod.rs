//! Configuration types for the modular research framework.

pub mod types;

pub use types::{
    load_research_config, AffinityWeighting, ConfigValidationError, DataConfig, DatasetFormat,
    GenerationTargetConfig, LossWeightConfig, ModelConfig, ParsingMode, PrimaryObjectiveConfig,
    ResearchConfig, RuntimeConfig, StageScheduleConfig, TrainingConfig,
};
