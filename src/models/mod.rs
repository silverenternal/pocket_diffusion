//! Model interfaces and Phase 1 encoder skeletons.

pub mod cross_attention;
pub mod geo_encoder;
pub mod pocket_encoder;
pub mod probe_heads;
pub mod slot_decomposition;
pub mod system;
pub mod topo_encoder;
pub mod traits;

pub use cross_attention::GatedCrossAttention;
pub use geo_encoder::GeometryEncoderImpl;
pub use pocket_encoder::PocketEncoderImpl;
pub use probe_heads::{ProbeOutputs, SemanticProbeHeads};
pub use slot_decomposition::SoftSlotDecomposer;
pub use system::Phase1ResearchSystem;
pub(crate) use system::{CrossModalInteractions, DecomposedModalities, ResearchForward};
pub use topo_encoder::TopologyEncoderImpl;
pub use traits::{
    ChemistryValidityEvaluator, CrossAttentionOutput, CrossModalInteractor, DockingEvaluator,
    Encoder, ExternalEvaluationReport, ExternalMetricRecord, GeneratedCandidateRecord,
    GeometryEncoder, LossTerm, ModalityEncoding, PocketCompatibilityEvaluator, PocketEncoder,
    SlotDecomposer, SlotEncoding, TaskDrivenObjective, TopologyEncoder, TrainerHook,
};
