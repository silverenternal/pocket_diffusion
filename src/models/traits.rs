//! Replaceable model interfaces for research ablations.

use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::data::{GeometryFeatures, MolecularExample, PocketFeatures, TopologyFeatures};

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

/// Padded latent representation emitted by a batched modality encoder.
#[derive(Debug)]
pub(crate) struct BatchedModalityEncoding {
    /// Per-token hidden states with shape `[batch, max_tokens, hidden_dim]`.
    pub token_embeddings: Tensor,
    /// Mask for active tokens with shape `[batch, max_tokens]`.
    pub token_mask: Tensor,
    /// Pooled summary representation with shape `[batch, hidden_dim]`.
    pub pooled_embedding: Tensor,
}

impl Clone for BatchedModalityEncoding {
    fn clone(&self) -> Self {
        Self {
            token_embeddings: self.token_embeddings.shallow_clone(),
            token_mask: self.token_mask.shallow_clone(),
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
    fn compute(&self, example: &MolecularExample, state: &State) -> Tensor;
}

/// Hooks that observe trainer lifecycle events.
pub trait TrainerHook<State> {
    /// Called after a training step completes.
    fn on_step_end(&mut self, _state: &State) {}
}

/// Partial ligand state consumed by a decoder or iterative sampler.
#[derive(Debug)]
pub struct PartialLigandState {
    /// Current atom-type tokens with shape `[num_atoms]`.
    pub atom_types: Tensor,
    /// Current Cartesian coordinates with shape `[num_atoms, 3]`.
    pub coords: Tensor,
    /// Active atom mask with shape `[num_atoms]`.
    pub atom_mask: Tensor,
    /// Decoder or sampler step index for iterative generation.
    pub step_index: i64,
}

impl Clone for PartialLigandState {
    fn clone(&self) -> Self {
        Self {
            atom_types: self.atom_types.shallow_clone(),
            coords: self.coords.shallow_clone(),
            atom_mask: self.atom_mask.shallow_clone(),
            step_index: self.step_index,
        }
    }
}

/// Explicit decoder-facing state that keeps topology, geometry, and pocket conditioning separate.
#[derive(Debug)]
pub struct ConditionedGenerationState {
    /// Stable example identifier.
    pub example_id: String,
    /// Stable protein identifier for pocket-conditioned generation.
    pub protein_id: String,
    /// Current partial ligand draft.
    pub partial_ligand: PartialLigandState,
    /// Topology-side conditioning slots after controlled interaction.
    pub topology_context: Tensor,
    /// Geometry-side conditioning slots after controlled interaction.
    pub geometry_context: Tensor,
    /// Pocket/context conditioning slots after controlled interaction.
    pub pocket_context: Tensor,
}

impl Clone for ConditionedGenerationState {
    fn clone(&self) -> Self {
        Self {
            example_id: self.example_id.clone(),
            protein_id: self.protein_id.clone(),
            partial_ligand: self.partial_ligand.clone(),
            topology_context: self.topology_context.shallow_clone(),
            geometry_context: self.geometry_context.shallow_clone(),
            pocket_context: self.pocket_context.shallow_clone(),
        }
    }
}

/// Decoder outputs for one conditioned ligand-generation step.
#[derive(Debug)]
pub struct DecoderOutput {
    /// Per-atom topology logits with shape `[num_atoms, atom_vocab_size]`.
    pub atom_type_logits: Tensor,
    /// Per-atom coordinate updates with shape `[num_atoms, 3]`.
    pub coordinate_deltas: Tensor,
    /// Scalar stop/update logit for iterative decoding.
    pub stop_logit: Tensor,
    /// Joint generation embedding reserved for later task objectives and samplers.
    pub generation_embedding: Tensor,
}

impl Clone for DecoderOutput {
    fn clone(&self) -> Self {
        Self {
            atom_type_logits: self.atom_type_logits.shallow_clone(),
            coordinate_deltas: self.coordinate_deltas.shallow_clone(),
            stop_logit: self.stop_logit.shallow_clone(),
            generation_embedding: self.generation_embedding.shallow_clone(),
        }
    }
}

/// Replaceable decoder contract for conditioned ligand construction.
pub trait ConditionedLigandDecoder {
    /// Decode one modular generation state into topology and geometry updates.
    fn decode(&self, state: &ConditionedGenerationState) -> DecoderOutput;
}

/// One iterative refinement step emitted by the modular decoder rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStepRecord {
    /// Zero-based refinement index.
    pub step_index: usize,
    /// Mean stop probability at this refinement step.
    pub stop_probability: f64,
    /// Whether rollout terminated immediately after this step.
    pub stopped: bool,
    /// Atom types after applying this step.
    pub atom_types: Vec<i64>,
    /// Coordinates after applying this step.
    pub coords: Vec<[f32; 3]>,
    /// Mean per-atom displacement applied at this step.
    pub mean_displacement: f64,
    /// Fraction of atoms whose committed identity changed at this step.
    pub atom_change_fraction: f64,
    /// Effective coordinate step scale applied by the rollout controller.
    pub coordinate_step_scale: f64,
}

/// Full iterative rollout trace for one conditioned ligand generation example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRolloutRecord {
    /// Stable example identifier.
    pub example_id: String,
    /// Stable protein identifier.
    pub protein_id: String,
    /// Configured step budget for this rollout.
    pub configured_steps: usize,
    /// Actual number of executed refinement steps.
    pub executed_steps: usize,
    /// Whether rollout terminated due to the learned stop head.
    pub stopped_early: bool,
    /// Per-step state trace.
    pub steps: Vec<GenerationStepRecord>,
}

/// Candidate payload reserved for future chemistry and docking evaluation backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCandidateRecord {
    /// Stable example identifier that produced the candidate.
    pub example_id: String,
    /// Protein identifier for pocket-conditioned evaluation.
    pub protein_id: String,
    /// Optional backend-specific molecular representation.
    pub molecular_representation: Option<String>,
    /// Predicted atom types for this candidate.
    pub atom_types: Vec<i64>,
    /// Predicted coordinates for this candidate.
    pub coords: Vec<[f32; 3]>,
    /// Distance-inferred bond list.
    pub inferred_bonds: Vec<(usize, usize)>,
    /// Pocket centroid used for downstream compatibility heuristics.
    pub pocket_centroid: [f32; 3],
    /// Pocket radius summary used for downstream compatibility heuristics.
    pub pocket_radius: f32,
    /// Translation from ligand-centered model coordinates back to source structure coordinates.
    pub coordinate_frame_origin: [f32; 3],
    /// Generator path that produced this candidate.
    pub source: String,
    /// Optional source protein structure path used for downstream scoring workflows.
    pub source_pocket_path: Option<String>,
    /// Optional source ligand path associated with the conditioning example.
    pub source_ligand_path: Option<String>,
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

/// Batched slot decomposition output for one modality.
#[derive(Debug)]
pub(crate) struct BatchedSlotEncoding {
    /// Slot tensors with shape `[batch, num_slots, hidden_dim]`.
    pub slots: Tensor,
    /// Soft activation weights with shape `[batch, num_slots]`.
    pub slot_weights: Tensor,
    /// Reconstruction of token features with shape `[batch, max_tokens, hidden_dim]`.
    pub reconstructed_tokens: Tensor,
    /// Active token mask with shape `[batch, max_tokens]`.
    pub token_mask: Tensor,
}

impl Clone for BatchedSlotEncoding {
    fn clone(&self) -> Self {
        Self {
            slots: self.slots.shallow_clone(),
            slot_weights: self.slot_weights.shallow_clone(),
            reconstructed_tokens: self.reconstructed_tokens.shallow_clone(),
            token_mask: self.token_mask.shallow_clone(),
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

/// Batched output for one directed gated cross-modal attention path.
#[derive(Debug)]
pub(crate) struct BatchedCrossAttentionOutput {
    /// Gate scalar per batch item with shape `[batch, 1]`.
    pub gate: Tensor,
    /// Attention-weighted update with shape `[batch, query_len, hidden_dim]`.
    pub attended_tokens: Tensor,
    /// Attention weights with shape `[batch, query_len, key_len]`.
    pub attention_weights: Tensor,
}

impl Clone for BatchedCrossAttentionOutput {
    fn clone(&self) -> Self {
        Self {
            gate: self.gate.shallow_clone(),
            attended_tokens: self.attended_tokens.shallow_clone(),
            attention_weights: self.attention_weights.shallow_clone(),
        }
    }
}
