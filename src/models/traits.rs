//! Replaceable model interfaces for research ablations.

use tch::Tensor;

use crate::data::{GeometryFeatures, PocketFeatures, TopologyFeatures};

/// Generic encoder interface.
pub trait Encoder<Input, Output> {
    /// Encode an input modality into a latent representation.
    fn encode(&self, input: &Input) -> Output;
}

/// Shared latent representation emitted by each modality encoder.
#[derive(Debug)]
pub struct ModalityEncoding {
    /// Per-token hidden states.
    pub token_embeddings: Tensor,
    /// Pooled summary representation.
    pub pooled_embedding: Tensor,
}

impl Clone for ModalityEncoding {
    fn clone(&self) -> Self {
        Self {
            token_embeddings: self.token_embeddings.shallow_clone(),
            pooled_embedding: self.pooled_embedding.shallow_clone(),
        }
    }
}

/// Trait alias style abstraction for topology encoders.
pub trait TopologyEncoder: Encoder<TopologyFeatures, ModalityEncoding> {}

/// Trait alias style abstraction for geometry encoders.
pub trait GeometryEncoder: Encoder<GeometryFeatures, ModalityEncoding> {}

/// Trait alias style abstraction for pocket encoders.
pub trait PocketEncoder: Encoder<PocketFeatures, ModalityEncoding> {}

/// Placeholder for later slot decomposition implementations.
pub trait SlotDecomposer<Input, Output> {
    /// Decompose a modality embedding into structured latent slots.
    fn decompose(&self, input: &Input) -> Output;
}

/// Placeholder for later gated cross-modal interaction modules.
pub trait CrossModalInteractor<Input, Output> {
    /// Exchange information across modalities in a controlled way.
    fn interact(&self, input: &Input) -> Output;
}

/// Placeholder for later loss implementations.
pub trait LossTerm<State> {
    /// Compute a scalar loss contribution for the provided state.
    fn compute(&self, state: &State) -> Tensor;
}

/// Hooks that observe trainer lifecycle events.
pub trait TrainerHook<State> {
    /// Called after a training step completes.
    fn on_step_end(&mut self, _state: &State) {}
}

/// Slot decomposition output for one modality.
#[derive(Debug)]
pub struct SlotEncoding {
    /// Slot tensors with shape `[num_slots, hidden_dim]`.
    pub slots: Tensor,
    /// Soft activation weights with shape `[num_slots]`.
    pub slot_weights: Tensor,
    /// Reconstruction of token features with shape `[num_tokens, hidden_dim]`.
    pub reconstructed_tokens: Tensor,
}

impl Clone for SlotEncoding {
    fn clone(&self) -> Self {
        Self {
            slots: self.slots.shallow_clone(),
            slot_weights: self.slot_weights.shallow_clone(),
            reconstructed_tokens: self.reconstructed_tokens.shallow_clone(),
        }
    }
}

/// Output for one directed gated cross-modal attention path.
#[derive(Debug)]
pub struct CrossAttentionOutput {
    /// Gate scalar in `[0, 1]`.
    pub gate: Tensor,
    /// Attention-weighted update with shape `[query_len, hidden_dim]`.
    pub attended_tokens: Tensor,
    /// Attention weights with shape `[query_len, key_len]`.
    pub attention_weights: Tensor,
}

impl Clone for CrossAttentionOutput {
    fn clone(&self) -> Self {
        Self {
            gate: self.gate.shallow_clone(),
            attended_tokens: self.attended_tokens.shallow_clone(),
            attention_weights: self.attention_weights.shallow_clone(),
        }
    }
}
