/// Output bundle for the separate modality encoders.
#[derive(Debug, Clone)]
pub(crate) struct EncodedModalities {
    /// Topology encoding.
    pub topology: ModalityEncoding,
    /// Geometry encoding.
    pub geometry: ModalityEncoding,
    /// Pocket/context encoding.
    pub pocket: ModalityEncoding,
}

/// Batched output bundle for the separate modality encoders.
#[derive(Debug, Clone)]
pub(crate) struct BatchedEncodedModalities {
    /// Topology encoding.
    pub topology: BatchedModalityEncoding,
    /// Geometry encoding.
    pub geometry: BatchedModalityEncoding,
    /// Pocket/context encoding.
    pub pocket: BatchedModalityEncoding,
}

/// Slot-decomposed outputs for each modality.
#[derive(Debug, Clone)]
pub(crate) struct DecomposedModalities {
    /// Topology slots.
    pub topology: SlotEncoding,
    /// Geometry slots.
    pub geometry: SlotEncoding,
    /// Pocket/context slots.
    pub pocket: SlotEncoding,
}

/// Batched slot-decomposed outputs for each modality.
#[derive(Debug, Clone)]
pub(crate) struct BatchedDecomposedModalities {
    /// Topology slots.
    pub topology: BatchedSlotEncoding,
    /// Geometry slots.
    pub geometry: BatchedSlotEncoding,
    /// Pocket/context slots.
    pub pocket: BatchedSlotEncoding,
}

/// Directed gated cross-modality interactions.
#[derive(Debug, Clone)]
pub(crate) struct CrossModalInteractions {
    /// Topology receiving information from geometry.
    pub topo_from_geo: CrossAttentionOutput,
    /// Topology receiving information from pocket context.
    pub topo_from_pocket: CrossAttentionOutput,
    /// Geometry receiving information from topology.
    pub geo_from_topo: CrossAttentionOutput,
    /// Geometry receiving information from pocket context.
    pub geo_from_pocket: CrossAttentionOutput,
    /// Pocket receiving information from topology.
    pub pocket_from_topo: CrossAttentionOutput,
    /// Pocket receiving information from geometry.
    pub pocket_from_geo: CrossAttentionOutput,
}

/// Batched directed gated cross-modality interactions.
#[derive(Debug, Clone)]
pub(crate) struct BatchedCrossModalInteractions {
    /// Topology receiving information from geometry.
    pub topo_from_geo: BatchedCrossAttentionOutput,
    /// Topology receiving information from pocket context.
    pub topo_from_pocket: BatchedCrossAttentionOutput,
    /// Geometry receiving information from topology.
    pub geo_from_topo: BatchedCrossAttentionOutput,
    /// Geometry receiving information from pocket context.
    pub geo_from_pocket: BatchedCrossAttentionOutput,
    /// Pocket receiving information from topology.
    pub pocket_from_topo: BatchedCrossAttentionOutput,
    /// Pocket receiving information from geometry.
    pub pocket_from_geo: BatchedCrossAttentionOutput,
}

/// Decoder-facing generation bundle produced by the modular backbone.
#[derive(Debug, Clone)]
pub(crate) struct GenerationForward {
    /// Explicit decoder input state with separated modality conditioning.
    pub state: ConditionedGenerationState,
    /// Decoder output for the current ligand draft.
    pub decoded: DecoderOutput,
    /// Iterative rollout trace aligned with the active generation semantics.
    pub rollout: GenerationRolloutRecord,
}

/// Full Phase 2 forward-pass bundle.
#[derive(Debug, Clone)]
pub(crate) struct ResearchForward {
    /// Pre-decomposition modality encodings.
    pub encodings: EncodedModalities,
    /// Slot-decomposed modality encodings.
    pub slots: DecomposedModalities,
    /// Directed gated interactions.
    pub interactions: CrossModalInteractions,
    /// Semantic probe predictions.
    pub probes: ProbeOutputs,
    /// Decoder-facing conditioned generation path.
    pub generation: GenerationForward,
}

