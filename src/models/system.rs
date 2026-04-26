//! Phase 2 system wiring for separate encoders, slots, controlled interaction, and probes.
//!
//! The implementation is split into type bundles, system methods, rollout
//! helpers, slicing helpers, and tests while preserving the original module API.

use tch::{nn, Kind, Tensor};

use super::{
    BatchedCrossAttentionOutput, BatchedModalityEncoding, BatchedSlotEncoding,
    ConditionedGenerationState, ConditionedLigandDecoder, CrossAttentionOutput, DecoderOutput,
    Encoder, GatedCrossAttention, GenerationRolloutRecord, GenerationStepRecord,
    GeometryEncoderImpl, ModalityEncoding, ModularLigandDecoder, PartialLigandState,
    PocketEncoderImpl, ProbeOutputs, SemanticProbeHeads, SlotDecomposer, SlotEncoding,
    SoftSlotDecomposer, TopologyEncoderImpl,
};
use crate::{
    config::{GenerationRolloutMode, GenerationTargetConfig, ResearchConfig},
    data::{MolecularBatch, MolecularExample},
};

include!("system/types.rs");
include!("system/impl.rs");
include!("system/rollout.rs");
include!("system/helpers.rs");
include!("system/tests.rs");
