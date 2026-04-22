//! Experiment orchestration for the modular research stack.

mod entrypoints;
pub mod unseen_pocket;

pub use entrypoints::{
    run_experiment_from_config, run_generation_demo_from_config, ResearchGenerationDemoSummary,
};
pub(crate) use unseen_pocket::evaluate_split;
pub use unseen_pocket::{
    load_experiment_config, AblationConfig, EvaluationMetrics, MeasurementMetrics,
    ProxyTaskMetrics, RealGenerationMetrics, RepresentationDiagnostics, ReservedBackendMetrics,
    ResourceUsageMetrics, SplitContextMetrics, UnseenPocketExperiment,
    UnseenPocketExperimentConfig, UnseenPocketExperimentSummary,
};
