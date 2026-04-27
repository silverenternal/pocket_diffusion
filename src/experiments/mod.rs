//! Experiment orchestration for the modular research stack.

mod entrypoints;
pub mod unseen_pocket;

pub use entrypoints::{
    run_ablation_matrix_from_config, run_automated_search_from_config, run_experiment_from_config,
    run_generation_demo_from_config, run_multi_seed_experiment_from_config,
    ResearchGenerationDemoSummary,
};
pub(crate) use unseen_pocket::evaluate_split;
pub use unseen_pocket::{
    load_experiment_config, AblationConfig, AblationMatrixConfig, AblationMatrixSummary,
    AblationRunSummary, AutomatedSearchCandidateSummary, AutomatedSearchGateResult,
    AutomatedSearchSummary, AutomatedSearchSurfaceSummary, BackendCommandReport,
    BackendEnvironmentReport, CandidateLayerMetrics, ChemistryBenchmarkEvidence,
    ChemistryNoveltyDiversitySummary, ClaimContext, ClaimReport, DrugLevelClaimContract,
    DrugLevelMetricGroupContract, EvaluationMetrics, ExternalEvaluationConfig,
    GenerationQualitySummary, LayeredGenerationMetrics, MeasurementMetrics,
    MethodComparisonSummary, MultiSeedAggregateReport, MultiSeedExperimentConfig,
    MultiSeedExperimentSummary, MultiSeedMetricAggregate, MultiSeedRunSummary,
    PlannedMetricInterface, ProxyTaskMetrics, RealGenerationMetrics, RepresentationDiagnostics,
    RerankerCalibrationReport, RerankerReport, ReservedBackendMetrics, ResourceUsageMetrics,
    SlotStabilityMetrics, SplitContextMetrics, StratumEvaluationMetrics, UnseenPocketExperiment,
    UnseenPocketExperimentConfig, UnseenPocketExperimentSummary,
};
