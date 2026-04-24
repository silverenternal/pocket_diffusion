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

/// Stable method family used by the comparison platform.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PocketGenerationMethodFamily {
    /// Full conditioned denoising with the shared disentangled backbone.
    ConditionedDenoising,
    /// Lightweight heuristic generation or selection path.
    Heuristic,
    /// Repair-first or repair-only postprocessing path.
    RepairOnly,
    /// Candidate reranking-only selection path.
    RerankerOnly,
    /// Reserved flow-matching generator family.
    FlowMatching,
    /// Reserved diffusion generator family.
    Diffusion,
    /// Reserved autoregressive generator family.
    Autoregressive,
    /// Wrapper around an external executable or service.
    ExternalWrapper,
}

/// Claim-review semantics for one method.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationEvidenceRole {
    /// Method is on a claim-bearing path for the active reviewer surface.
    ClaimBearing,
    /// Method is intended as a fair comparison baseline or control.
    ComparisonOnly,
    /// Method is diagnostic or exploratory only.
    DiagnosticOnly,
    /// Method delegates generation outside the native Rust stack.
    ExternalWrapper,
}

/// How a method is expected to execute inside the comparison runner.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationExecutionMode {
    /// Native per-example generation.
    PerExample,
    /// Native batched generation.
    Batched,
    /// Delegates to an external process or service.
    ExternalCommand,
    /// Placeholder registration that documents a future method family.
    Stub,
}

/// Stable candidate layer identifiers shared across methods and artifacts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CandidateLayerKind {
    /// Direct method-native rollout output.
    RawRollout,
    /// Geometry-repaired output.
    Repaired,
    /// Bond-inferred output.
    InferredBond,
    /// Deterministic proxy reranker output.
    DeterministicProxy,
    /// Active calibrated reranker output.
    Reranked,
}

impl CandidateLayerKind {
    /// Legacy field name used by persisted reviewer artifacts.
    pub fn legacy_field_name(self) -> &'static str {
        match self {
            Self::RawRollout => "raw_rollout",
            Self::Repaired => "repaired_candidates",
            Self::InferredBond => "inferred_bond_candidates",
            Self::DeterministicProxy => "deterministic_proxy_candidates",
            Self::Reranked => "reranked_candidates",
        }
    }
}

/// Capability flags surfaced by one generation method.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationMethodCapability {
    /// Whether the method has trainable parameters on its own path.
    pub trainable: bool,
    /// Whether the method can generate in batches.
    pub batched_generation: bool,
    /// Whether the method emits native raw candidates.
    pub method_native_generation: bool,
    /// Whether the method depends on shared postprocessors.
    pub uses_postprocessors: bool,
    /// Whether the method supports explicit repair layering.
    pub repair_layer: bool,
    /// Whether the method supports explicit bond-inference layering.
    pub inferred_bond_layer: bool,
    /// Whether the method supports explicit reranked layering.
    pub reranked_layer: bool,
    /// Whether the method delegates execution outside the native binary.
    pub external_wrapper: bool,
    /// Whether the registration is a future stub rather than an active implementation.
    pub stub: bool,
}

/// Stable metadata exposed by every registered generation method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketGenerationMethodMetadata {
    /// Stable machine-readable method identifier.
    pub method_id: String,
    /// Human-readable method name.
    pub method_name: String,
    /// Method family used for reporting and compatibility checks.
    pub method_family: PocketGenerationMethodFamily,
    /// Capability summary for the method.
    pub capability: GenerationMethodCapability,
    /// Stable layered output support declaration.
    pub layered_output_support: Vec<CandidateLayerKind>,
    /// Claim-review role for the method.
    pub evidence_role: GenerationEvidenceRole,
    /// Expected execution mode for the method.
    pub execution_mode: GenerationExecutionMode,
}

/// Provenance for one candidate layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateLayerProvenance {
    /// Method identifier that owns the layer.
    pub source_method_id: String,
    /// Method name that owns the layer.
    pub source_method_name: String,
    /// Method family that owns the layer.
    pub source_method_family: PocketGenerationMethodFamily,
    /// Stable layer identifier.
    pub layer_kind: CandidateLayerKind,
    /// Backward-compatible persisted field name.
    pub legacy_field_name: String,
    /// Whether the layer is method-native rather than derived.
    pub method_native: bool,
    /// Ordered postprocessor chain that produced this layer.
    pub postprocessor_chain: Vec<String>,
    /// Whether the layer is available for this method.
    pub available: bool,
}

/// Candidate collection plus provenance for one generation layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateLayerOutput {
    /// Provenance attached to the layer.
    pub provenance: CandidateLayerProvenance,
    /// Candidate payloads for this layer.
    pub candidates: Vec<GeneratedCandidateRecord>,
}

/// Stable layered output schema used by all generation methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredGenerationOutput {
    /// Method metadata that produced these layers.
    pub metadata: PocketGenerationMethodMetadata,
    /// Raw rollout layer when available.
    #[serde(default)]
    pub raw_rollout: Option<CandidateLayerOutput>,
    /// Geometry-repaired layer when available.
    #[serde(default)]
    pub repaired: Option<CandidateLayerOutput>,
    /// Bond-inferred layer when available.
    #[serde(default)]
    pub inferred_bond: Option<CandidateLayerOutput>,
    /// Deterministic proxy selection layer when available.
    #[serde(default)]
    pub deterministic_proxy: Option<CandidateLayerOutput>,
    /// Calibrated reranked layer when available.
    #[serde(default)]
    pub reranked: Option<CandidateLayerOutput>,
}

impl LayeredGenerationOutput {
    /// Construct an empty layered output for a method registration.
    pub fn empty(metadata: PocketGenerationMethodMetadata) -> Self {
        Self {
            metadata,
            raw_rollout: None,
            repaired: None,
            inferred_bond: None,
            deterministic_proxy: None,
            reranked: None,
        }
    }
}

/// Per-example method execution context.
#[derive(Debug, Clone)]
pub struct PocketGenerationContext {
    /// Input example used for conditioned generation.
    pub example: MolecularExample,
    /// Optional shared backbone forward state.
    pub(crate) forward: Option<crate::models::system::ResearchForward>,
    /// Requested candidate count for this execution.
    pub candidate_limit: usize,
    /// Whether shared repair postprocessing is enabled for this execution.
    pub enable_repair: bool,
}

/// Stable generation-method contract for comparison runners and demos.
pub trait PocketGenerationMethod {
    /// Return stable method metadata for registry, artifacts, and claim review.
    fn metadata(&self) -> PocketGenerationMethodMetadata;

    /// Generate layered outputs for one example.
    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput;

    /// Generate layered outputs for a batch.
    fn generate_batch(&self, contexts: Vec<PocketGenerationContext>) -> Vec<LayeredGenerationOutput> {
        contexts
            .into_iter()
            .map(|context| self.generate_for_example(context))
            .collect()
    }
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
