//! Model interfaces and Phase 1 encoder skeletons.

pub mod cross_attention;
pub mod decoder;
pub mod evaluation;
pub mod geo_encoder;
pub mod pocket_encoder;
pub mod probe_heads;
pub mod slot_decomposition;
pub mod system;
pub mod topo_encoder;
pub mod traits;

pub use cross_attention::GatedCrossAttention;
pub use decoder::ModularLigandDecoder;
pub(crate) use evaluation::{
    candidate_records_to_legacy, generate_candidates_from_forward,
    generate_layered_candidates_with_options, report_to_metrics, CandidateGenerationLayers,
};
pub use evaluation::{
    CommandChemistryValidityEvaluator, CommandDockingEvaluator,
    CommandPocketCompatibilityEvaluator, HeuristicChemistryValidityEvaluator,
    HeuristicDockingEvaluator, HeuristicPocketCompatibilityEvaluator,
};
pub use geo_encoder::GeometryEncoderImpl;
pub use pocket_encoder::PocketEncoderImpl;
pub use probe_heads::{ProbeOutputs, SemanticProbeHeads};
pub use slot_decomposition::SoftSlotDecomposer;
pub use system::Phase1ResearchSystem;
pub(crate) use system::{CrossModalInteractions, DecomposedModalities, ResearchForward};
pub use topo_encoder::TopologyEncoderImpl;
pub(crate) use traits::{
    BatchedCrossAttentionOutput, BatchedModalityEncoding, BatchedSlotEncoding,
};
pub use traits::{
    ChemistryValidityEvaluator, ConditionedGenerationState, ConditionedLigandDecoder,
    CrossAttentionOutput, CrossModalInteractor, DecoderOutput, DockingEvaluator, Encoder,
    ExternalEvaluationReport, ExternalMetricRecord, GeneratedCandidateRecord,
    GenerationRolloutRecord, GenerationStepRecord, GeometryEncoder, LossTerm, ModalityEncoding,
    PartialLigandState, PocketCompatibilityEvaluator, PocketEncoder, SlotDecomposer, SlotEncoding,
    TaskDrivenObjective, TopologyEncoder, TrainerHook,
};
