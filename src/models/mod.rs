//! Model interfaces and Phase 1 encoder skeletons.

pub mod cross_attention;
pub mod decoder;
pub mod evaluation;
pub mod flow;
pub mod flow_matching;
pub mod geo_encoder;
pub mod geometry;
pub mod interaction;
pub mod methods;
pub mod pocket_encoder;
pub mod preference;
pub mod probe_heads;
pub mod semantic;
pub mod slot_decomposition;
pub mod system;
pub mod target_matching;
pub mod topo_encoder;
pub mod traits;

pub use cross_attention::GatedCrossAttention;
pub use decoder::ModularLigandDecoder;
pub(crate) use evaluation::{
    candidate_records_to_legacy, generate_layered_candidates_from_forward, report_to_metrics,
};
pub use evaluation::{
    CommandChemistryValidityEvaluator, CommandDockingEvaluator,
    CommandPocketCompatibilityEvaluator, DrugMetricDirection, DrugMetricDomain,
    DrugMetricObservation, DrugMetricPanel, DrugMetricSpec, HeuristicChemistryValidityEvaluator,
    HeuristicDockingEvaluator, HeuristicPocketCompatibilityEvaluator,
};
pub use flow::{
    current_molecular_flow_state_contract, current_multimodal_flow_contract,
    AtomPocketCrossAttentionVelocityConfig, AtomPocketCrossAttentionVelocityHead,
    FlowBranchSupportStatus, FlowVelocityHead, FullMolecularFlowHead, MolecularFlowInput,
    MolecularFlowPrediction, MolecularFlowStateContract, MolecularFlowStateVariableContract,
    MultiModalFlowBranchRecord, MultiModalFlowContract, MOLECULAR_FLOW_CONTRACT_VERSION,
    REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES,
};
pub use flow::{
    predicted_native_graph_from_flow, predicted_native_graph_from_flow_with_config,
    NativeGraphExtractionConfig, NativeGraphExtractionResult, NATIVE_GRAPH_EXTRACTOR_VERSION,
};
pub use flow_matching::GeometryFlowMatchingHead;
pub use geo_encoder::GeometryEncoderImpl;
pub use geometry::{
    PairwiseGeometryConfig, PairwiseGeometryMessage, PairwiseGeometryMessagePassing,
};
pub use interaction::CrossModalInteractionBlock;
pub use methods::{
    flatten_layered_output, summarize_method_output, MethodComparisonRow,
    PocketGenerationMethodRegistry,
};
pub use pocket_encoder::PocketEncoderImpl;
pub use preference::{
    build_bounded_preference_pairs, extract_interaction_profiles, BackendPreferenceCandidateRef,
    BackendPreferenceClass, BackendPreferencePair, BackendPreferencePairArtifact,
    HeuristicInteractionProfileExtractor, InteractionFeatureProvenance, InteractionFeatureValue,
    InteractionProfile, InteractionProfileExtractionContext, InteractionProfileExtractor,
    PreferenceConstructionConfig, PreferenceDatasetBuilder, PreferencePair, PreferencePairArtifact,
    PreferenceProfileArtifact, PreferenceReasonCode, PreferenceReranker,
    PreferenceRerankerSummaryArtifact, PreferenceSource, PreferenceTrainer,
    RuleBasedPreferenceDatasetBuilder, RuleBasedPreferenceReranker,
    BACKEND_PREFERENCE_PAIR_SCHEMA_VERSION, INTERACTION_PROFILE_SCHEMA_VERSION,
    PREFERENCE_PAIR_SCHEMA_VERSION, PREFERENCE_RERANKER_SCHEMA_VERSION,
};
pub use probe_heads::{ProbeOutputs, SemanticProbeHeads};
pub use semantic::{GeometrySemanticBranch, PocketSemanticBranch, TopologySemanticBranch};
pub use slot_decomposition::SoftSlotDecomposer;
pub use system::Phase1ResearchSystem;
pub(crate) use system::{CrossModalInteractions, DecomposedModalities, ResearchForward};
pub use target_matching::{
    match_molecular_targets, TargetMatchingCostSummary, TargetMatchingResult,
};
pub use topo_encoder::TopologyEncoderImpl;
pub use traits::{
    AtomCountPrior, CandidateLayerKind, CandidateLayerOutput, CandidateLayerProvenance,
    ChemistryValidityEvaluator, ConditionedGenerationRequest, ConditionedGenerationState,
    ConditionedLigandDecoder, ConditioningState, CrossAttentionOutput, CrossModalInteractor,
    DeNovoScaffoldInitializer, DecoderCapabilityDescriptor, DecoderOutput, DockingEvaluator,
    Encoder, ExternalConditioningSummary, ExternalEvaluationReport,
    ExternalGenerationRequestRecord, ExternalGenerationResponseRecord, ExternalMetricRecord,
    FixedAtomCountPrior, FlowMatchingHead, FlowState, GeneratedCandidateRecord,
    GenerationEvidenceRole, GenerationExecutionMode, GenerationGateSummary,
    GenerationMethodCapability, GenerationModalityConditioning, GenerationPathUsageSummary,
    GenerationRolloutRecord, GenerationStepPathUsageSummary, GenerationStepRecord, GeometryEncoder,
    LayeredGenerationOutput, LossTerm, ModalityEncoding, ModelError, NativeGraphLayerProvenance,
    PartialLigandState, PocketCentroidScaffoldInitializer, PocketCompatibilityEvaluator,
    PocketEncoder, PocketGenerationContext, PocketGenerationMethod, PocketGenerationMethodFamily,
    PocketGenerationMethodMetadata, PocketInitializationContext, PocketVolumeAtomCountPrior,
    ScaffoldInitializer, SlotDecomposer, SlotEncoding, TaskDrivenObjective, TopologyEncoder,
    TrainerHook, VelocityField,
};
pub(crate) use traits::{
    BatchedCrossAttentionOutput, BatchedModalityEncoding, BatchedSlotEncoding,
};
