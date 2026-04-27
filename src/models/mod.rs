//! Model interfaces and Phase 1 encoder skeletons.

pub mod cross_attention;
pub mod decoder;
pub mod evaluation;
pub mod flow;
pub mod flow_matching;
pub mod geo_encoder;
pub mod geometry;
pub mod methods;
pub mod pocket_encoder;
pub mod preference;
pub mod probe_heads;
pub mod slot_decomposition;
pub mod system;
pub mod topo_encoder;
pub mod traits;

pub use cross_attention::GatedCrossAttention;
pub use decoder::ModularLigandDecoder;
pub(crate) use evaluation::{
    candidate_records_to_legacy, generate_candidates_from_forward, report_to_metrics,
};
pub use evaluation::{
    CommandChemistryValidityEvaluator, CommandDockingEvaluator,
    CommandPocketCompatibilityEvaluator, HeuristicChemistryValidityEvaluator,
    HeuristicDockingEvaluator, HeuristicPocketCompatibilityEvaluator,
};
pub use flow::{AtomPocketCrossAttentionVelocityConfig, AtomPocketCrossAttentionVelocityHead};
pub use flow_matching::GeometryFlowMatchingHead;
pub use geo_encoder::GeometryEncoderImpl;
pub use geometry::{
    PairwiseGeometryConfig, PairwiseGeometryMessage, PairwiseGeometryMessagePassing,
};
pub use methods::{
    flatten_layered_output, summarize_method_output, MethodComparisonRow,
    PocketGenerationMethodRegistry,
};
pub use pocket_encoder::PocketEncoderImpl;
pub use preference::{
    build_bounded_preference_pairs, extract_interaction_profiles,
    HeuristicInteractionProfileExtractor, InteractionFeatureProvenance, InteractionFeatureValue,
    InteractionProfile, InteractionProfileExtractionContext, InteractionProfileExtractor,
    PreferenceConstructionConfig, PreferenceDatasetBuilder, PreferencePair, PreferencePairArtifact,
    PreferenceProfileArtifact, PreferenceReasonCode, PreferenceReranker,
    PreferenceRerankerSummaryArtifact, PreferenceSource, PreferenceTrainer,
    RuleBasedPreferenceDatasetBuilder, RuleBasedPreferenceReranker,
    INTERACTION_PROFILE_SCHEMA_VERSION, PREFERENCE_PAIR_SCHEMA_VERSION,
    PREFERENCE_RERANKER_SCHEMA_VERSION,
};
pub use probe_heads::{ProbeOutputs, SemanticProbeHeads};
pub use slot_decomposition::SoftSlotDecomposer;
pub use system::Phase1ResearchSystem;
pub(crate) use system::{CrossModalInteractions, DecomposedModalities, ResearchForward};
pub use topo_encoder::TopologyEncoderImpl;
pub(crate) use traits::{
    BatchedCrossAttentionOutput, BatchedModalityEncoding, BatchedSlotEncoding,
};
pub use traits::{
    CandidateLayerKind, CandidateLayerOutput, CandidateLayerProvenance, ChemistryValidityEvaluator,
    ConditionedGenerationRequest, ConditionedGenerationState, ConditionedLigandDecoder,
    ConditioningState, CrossAttentionOutput, CrossModalInteractor, DecoderOutput, DockingEvaluator,
    Encoder, ExternalConditioningSummary, ExternalEvaluationReport,
    ExternalGenerationRequestRecord, ExternalGenerationResponseRecord, ExternalMetricRecord,
    FlowMatchingHead, FlowState, GeneratedCandidateRecord, GenerationEvidenceRole,
    GenerationExecutionMode, GenerationGateSummary, GenerationMethodCapability,
    GenerationModalityConditioning, GenerationRolloutRecord, GenerationStepRecord, GeometryEncoder,
    LayeredGenerationOutput, LossTerm, ModalityEncoding, ModelError, PartialLigandState,
    PocketCompatibilityEvaluator, PocketEncoder, PocketGenerationContext, PocketGenerationMethod,
    PocketGenerationMethodFamily, PocketGenerationMethodMetadata, SlotDecomposer, SlotEncoding,
    TaskDrivenObjective, TopologyEncoder, TrainerHook, VelocityField,
};
