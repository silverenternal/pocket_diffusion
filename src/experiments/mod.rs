//! Experiment orchestration for the modular research stack.

mod entrypoints;
pub mod unseen_pocket;

pub use entrypoints::{
    run_ablation_matrix_from_config, run_automated_search_from_config, run_experiment_from_config,
    run_generation_demo_from_config, run_generation_layers_from_config,
    run_multi_seed_experiment_from_config, LayeredGenerationRunSummary,
    ResearchGenerationDemoSummary,
};
pub(crate) use unseen_pocket::evaluate_split;
pub use unseen_pocket::{
    canonical_generation_path_contract, load_experiment_config,
    validate_experiment_config_with_source, AblationConfig, AblationMatrixConfig,
    AblationMatrixSummary, AblationRunSummary, AutomatedSearchCandidateSummary,
    AutomatedSearchGateResult, AutomatedSearchSummary, AutomatedSearchSurfaceSummary,
    BackendCommandReport, BackendCoverageContractRow, BackendEnvironmentReport, BestMetricReview,
    CandidateLayerMetrics, ChemistryBenchmarkEvidence, ChemistryCollaborationMetric,
    ChemistryCollaborationMetrics, ChemistryMetricProvenance, ChemistryNoveltyDiversitySummary,
    ChemistryRoleGateUsage, ClaimContext, ClaimReport, DrugLevelClaimContract,
    DrugLevelMetricGroupContract, EvaluationMetrics, ExternalEvaluationConfig,
    FlowHeadAblationDiagnostics, FrozenLeakageProbeCalibrationReport, GenerationPathContractRow,
    GenerationQualitySummary, LayeredGenerationMetrics, MeasurementMetrics,
    MethodComparisonSummary, ModelDesignEvaluationMetrics, MultiSeedAggregateReport,
    MultiSeedExperimentConfig, MultiSeedExperimentSummary, MultiSeedMetricAggregate,
    MultiSeedRunSummary, NoRepairAblationMetrics, PlannedMetricInterface, ProxyTaskMetrics,
    RealGenerationMetrics, RepairCandidateMetricSnapshot, RepairCaseAuditReport, RepairCaseRecord,
    RepairLayerDeltaSummary, RepresentationDiagnostics, RerankerCalibrationReport, RerankerReport,
    ReservedBackendMetrics, ResourceUsageMetrics, SlotStabilityMetrics, SplitContextMetrics,
    StratumEvaluationMetrics, TrainEvalAlignmentMetricRow, TrainEvalAlignmentReport,
    UnseenPocketExperiment, UnseenPocketExperimentConfig, UnseenPocketExperimentSummary,
};
