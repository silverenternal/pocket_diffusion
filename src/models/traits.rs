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

/// Pluggable primary objective interface for the modular training stack.
pub trait TaskDrivenObjective<State> {
    /// Stable schema-facing objective name.
    fn name(&self) -> &'static str;

    /// Compute the primary scalar optimization target.
    fn compute(&self, state: &State) -> Tensor;
}

/// Hooks that observe trainer lifecycle events.
pub trait TrainerHook<State> {
    /// Called after a training step completes.
    fn on_step_end(&mut self, _state: &State) {}
}

/// Candidate payload reserved for future chemistry and docking evaluation backends.
#[derive(Debug, Clone)]
pub struct GeneratedCandidateRecord {
    /// Stable example identifier that produced the candidate.
    pub example_id: String,
    /// Protein identifier for pocket-conditioned evaluation.
    pub protein_id: String,
    /// Optional backend-specific molecular representation.
    pub molecular_representation: Option<String>,
}

/// Named metrics emitted by an external chemistry or docking backend.
#[derive(Debug, Clone)]
pub struct ExternalMetricRecord {
    /// Stable metric name, such as `valid_fraction` or `best_docking_score`.
    pub metric_name: String,
    /// Numeric metric value.
    pub value: f64,
}

/// Backend response for future chemistry-grade evaluation.
#[derive(Debug, Clone)]
pub struct ExternalEvaluationReport {
    /// Backend identifier.
    pub backend_name: String,
    /// Metrics emitted by the backend.
    pub metrics: Vec<ExternalMetricRecord>,
}

/// Extension point for future chemistry validity evaluation.
pub trait ChemistryValidityEvaluator {
    /// Stable backend identifier.
    fn backend_name(&self) -> &'static str;

    /// Evaluate generated candidates for chemistry validity.
    fn evaluate_chemistry(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport;
}

/// Extension point for future docking or affinity rescoring.
pub trait DockingEvaluator {
    /// Stable backend identifier.
    fn backend_name(&self) -> &'static str;

    /// Evaluate generated candidates against pockets with a docking backend.
    fn evaluate_docking(&self, candidates: &[GeneratedCandidateRecord])
        -> ExternalEvaluationReport;
}

/// Extension point for future pocket-compatibility validation.
pub trait PocketCompatibilityEvaluator {
    /// Stable backend identifier.
    fn backend_name(&self) -> &'static str;

    /// Evaluate whether generated candidates are pocket-compatible downstream.
    fn evaluate_pocket_compatibility(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport;
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
