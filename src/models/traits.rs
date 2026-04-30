//! Replaceable model interfaces for research ablations.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

use crate::{
    config::{GenerationModeConfig, GenerationTargetConfig},
    data::{GeometryFeatures, MolecularExample, PocketFeatures, TopologyFeatures},
};

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
    /// Attention-visible topology slot mask with shape `[slots]`.
    pub topology_slot_mask: Tensor,
    /// Attention-visible geometry slot mask with shape `[slots]`.
    pub geometry_slot_mask: Tensor,
    /// Attention-visible pocket/context slot mask with shape `[slots]`.
    pub pocket_slot_mask: Tensor,
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
            topology_slot_mask: self.topology_slot_mask.shallow_clone(),
            geometry_slot_mask: self.geometry_slot_mask.shallow_clone(),
            pocket_slot_mask: self.pocket_slot_mask.shallow_clone(),
        }
    }
}

/// Backend-facing context for one decomposed modality.
#[derive(Debug)]
pub struct GenerationModalityConditioning {
    /// Controlled-interaction context consumed by the backend.
    pub context: Tensor,
    /// Original decomposed slots for this modality.
    pub slots: Tensor,
    /// Legacy assignment-mass weights for utilization and compatibility checks.
    pub slot_weights: Tensor,
    /// Independent learned slot activation gates.
    pub slot_activations: Tensor,
    /// Binary attention-visible slot mask used by local attention paths.
    pub active_slot_mask: Tensor,
    /// Mean slot activation for backend-agnostic reporting.
    pub active_slot_fraction: f64,
    /// Fraction of slots visible to attention after masking.
    pub attention_visible_slot_fraction: f64,
}

impl Clone for GenerationModalityConditioning {
    fn clone(&self) -> Self {
        Self {
            context: self.context.shallow_clone(),
            slots: self.slots.shallow_clone(),
            slot_weights: self.slot_weights.shallow_clone(),
            slot_activations: self.slot_activations.shallow_clone(),
            active_slot_mask: self.active_slot_mask.shallow_clone(),
            active_slot_fraction: self.active_slot_fraction,
            attention_visible_slot_fraction: self.attention_visible_slot_fraction,
        }
    }
}

/// Directed gate summary attached to a backend-neutral generation request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct GenerationGateSummary {
    /// Topology receiving information from geometry.
    pub topo_from_geo: f64,
    /// Topology receiving information from pocket context.
    pub topo_from_pocket: f64,
    /// Geometry receiving information from topology.
    pub geo_from_topo: f64,
    /// Geometry receiving information from pocket context.
    pub geo_from_pocket: f64,
    /// Pocket/context receiving information from topology.
    pub pocket_from_topo: f64,
    /// Pocket/context receiving information from geometry.
    pub pocket_from_geo: f64,
}

/// Step-bucketed directed interaction path usage for inference summaries.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationStepPathUsageSummary {
    /// Inclusive start step represented by this bucket.
    pub start_step: usize,
    /// Inclusive end step represented by this bucket.
    pub end_step: usize,
    /// Mean gate activation for each directed path in this step bucket.
    pub path_means: GenerationGateSummary,
}

/// Directed interaction path usage attached to a rollout trace.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationPathUsageSummary {
    /// Mean gate activation for each directed path before rollout begins.
    pub raw_path_means: GenerationGateSummary,
    /// Rollout-step buckets carrying path means used by the inference summary.
    #[serde(default)]
    pub step_bucketed_path_means: Vec<GenerationStepPathUsageSummary>,
}

/// Backend-neutral request that exposes decomposed conditioning explicitly.
#[derive(Debug, Clone)]
pub struct ConditionedGenerationRequest {
    /// Stable example identifier.
    pub example_id: String,
    /// Stable protein identifier.
    pub protein_id: String,
    /// Topology-side conditioning.
    pub topology: GenerationModalityConditioning,
    /// Geometry-side conditioning.
    pub geometry: GenerationModalityConditioning,
    /// Pocket/context-side conditioning.
    pub pocket: GenerationModalityConditioning,
    /// Directed controlled-interaction gate summaries.
    pub gate_summary: GenerationGateSummary,
    /// Decoder-facing state that keeps modality contexts separate.
    pub generation_state: ConditionedGenerationState,
    /// Generation/sampling config selected for this request.
    pub generation_config: GenerationTargetConfig,
    /// Explicit generation-mode contract for this request.
    pub generation_mode: GenerationModeConfig,
}

impl ConditionedGenerationRequest {
    pub(crate) fn from_forward(
        example: &MolecularExample,
        forward: &crate::models::system::ResearchForward,
        generation_config: &GenerationTargetConfig,
    ) -> Self {
        Self {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            topology: GenerationModalityConditioning {
                context: forward.generation.state.topology_context.shallow_clone(),
                slots: forward.slots.topology.slots.shallow_clone(),
                slot_weights: forward.slots.topology.slot_weights.shallow_clone(),
                slot_activations: forward.slots.topology.slot_activations.shallow_clone(),
                active_slot_mask: forward.slots.topology.active_slot_mask.shallow_clone(),
                active_slot_fraction: active_slot_fraction(
                    &forward.slots.topology.slot_activations,
                ),
                attention_visible_slot_fraction: active_slot_fraction(
                    &forward.slots.topology.active_slot_mask,
                ),
            },
            geometry: GenerationModalityConditioning {
                context: forward.generation.state.geometry_context.shallow_clone(),
                slots: forward.slots.geometry.slots.shallow_clone(),
                slot_weights: forward.slots.geometry.slot_weights.shallow_clone(),
                slot_activations: forward.slots.geometry.slot_activations.shallow_clone(),
                active_slot_mask: forward.slots.geometry.active_slot_mask.shallow_clone(),
                active_slot_fraction: active_slot_fraction(
                    &forward.slots.geometry.slot_activations,
                ),
                attention_visible_slot_fraction: active_slot_fraction(
                    &forward.slots.geometry.active_slot_mask,
                ),
            },
            pocket: GenerationModalityConditioning {
                context: forward.generation.state.pocket_context.shallow_clone(),
                slots: forward.slots.pocket.slots.shallow_clone(),
                slot_weights: forward.slots.pocket.slot_weights.shallow_clone(),
                slot_activations: forward.slots.pocket.slot_activations.shallow_clone(),
                active_slot_mask: forward.slots.pocket.active_slot_mask.shallow_clone(),
                active_slot_fraction: active_slot_fraction(&forward.slots.pocket.slot_activations),
                attention_visible_slot_fraction: active_slot_fraction(
                    &forward.slots.pocket.active_slot_mask,
                ),
            },
            gate_summary: GenerationGateSummary {
                topo_from_geo: scalar_gate(&forward.interactions.topo_from_geo.gate),
                topo_from_pocket: scalar_gate(&forward.interactions.topo_from_pocket.gate),
                geo_from_topo: scalar_gate(&forward.interactions.geo_from_topo.gate),
                geo_from_pocket: scalar_gate(&forward.interactions.geo_from_pocket.gate),
                pocket_from_topo: scalar_gate(&forward.interactions.pocket_from_topo.gate),
                pocket_from_geo: scalar_gate(&forward.interactions.pocket_from_geo.gate),
            },
            generation_state: forward.generation.state.clone(),
            generation_config: generation_config.clone(),
            generation_mode: forward.generation.generation_mode,
        }
    }
}

fn active_slot_fraction(weights: &Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    weights
        .gt(0.05)
        .to_kind(tch::Kind::Float)
        .mean(tch::Kind::Float)
        .double_value(&[])
}

fn scalar_gate(gate: &Tensor) -> f64 {
    if gate.numel() == 0 {
        0.0
    } else {
        gate.mean(tch::Kind::Float).double_value(&[])
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
    /// Decoder capability contract used to validate generation modes and reports.
    fn capability(&self) -> DecoderCapabilityDescriptor {
        DecoderCapabilityDescriptor::fixed_atom_refinement()
    }

    /// Decode one modular generation state into topology and geometry updates.
    fn decode(&self, state: &ConditionedGenerationState) -> DecoderOutput;
}

/// Explicit decoder capability descriptor for generation-mode validation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecoderCapabilityDescriptor {
    /// Decoder preserves an externally supplied atom count.
    pub fixed_atom_refinement: bool,
    /// Decoder can consume a scaffold initializer.
    pub scaffold_conditioned: bool,
    /// Decoder predicts atom counts internally.
    pub atom_count_prediction: bool,
    /// Decoder grows molecular graphs with stop decisions.
    pub graph_growth: bool,
    /// Decoder can run from a pocket-only initialization baseline.
    pub pocket_only_initialization: bool,
}

impl DecoderCapabilityDescriptor {
    /// Capability contract for the current modular fixed-atom decoder.
    pub const fn fixed_atom_refinement() -> Self {
        Self {
            fixed_atom_refinement: true,
            scaffold_conditioned: false,
            atom_count_prediction: false,
            graph_growth: false,
            pocket_only_initialization: true,
        }
    }

    /// Capability contract for pocket-conditioned de novo graph flow.
    pub const fn pocket_conditioned_graph_flow() -> Self {
        Self {
            fixed_atom_refinement: true,
            scaffold_conditioned: true,
            atom_count_prediction: true,
            graph_growth: true,
            pocket_only_initialization: true,
        }
    }

    /// Stable label used in rollout and checkpoint diagnostics.
    pub const fn label(self) -> &'static str {
        if self.graph_growth {
            "graph_growth"
        } else if self.atom_count_prediction {
            "atom_count_prediction"
        } else if self.scaffold_conditioned {
            "scaffold_conditioned_fixed_atom_refinement"
        } else if self.pocket_only_initialization {
            "fixed_atom_refinement_with_pocket_only_initialization"
        } else if self.fixed_atom_refinement {
            "fixed_atom_refinement"
        } else {
            "unsupported"
        }
    }

    /// Whether the capability can execute the requested mode.
    pub fn supports_mode(self, mode: GenerationModeConfig) -> bool {
        match mode {
            GenerationModeConfig::TargetLigandDenoising
            | GenerationModeConfig::LigandRefinement
            | GenerationModeConfig::FlowRefinement => self.fixed_atom_refinement,
            GenerationModeConfig::PocketOnlyInitializationBaseline => {
                self.fixed_atom_refinement && self.pocket_only_initialization
            }
            GenerationModeConfig::DeNovoInitialization => {
                self.atom_count_prediction && self.graph_growth
            }
        }
    }
}

/// Pocket-conditioned context passed to atom-count and scaffold initializers.
pub struct PocketInitializationContext<'a> {
    /// Stable example identifier.
    pub example_id: &'a str,
    /// Stable protein identifier.
    pub protein_id: &'a str,
    /// Pocket coordinates with shape `[num_pocket_atoms, 3]`.
    pub pocket_coords: &'a Tensor,
}

/// Replaceable atom-count prior for initialization-only baselines.
pub trait AtomCountPrior {
    /// Propose an atom count without reading target ligand topology.
    fn propose_atom_count(&self, context: &PocketInitializationContext<'_>) -> usize;
    /// Stable provenance label for artifact and evaluation summaries.
    fn provenance(&self) -> &'static str;
}

/// Fixed atom-count prior used by the conservative pocket-only baseline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct FixedAtomCountPrior {
    /// Atom count emitted for every pocket.
    pub atom_count: usize,
}

impl AtomCountPrior for FixedAtomCountPrior {
    fn propose_atom_count(&self, _context: &PocketInitializationContext<'_>) -> usize {
        self.atom_count.max(1)
    }

    fn provenance(&self) -> &'static str {
        "fixed"
    }
}

/// Pocket-size atom-count prior used by the de novo molecular flow path.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PocketVolumeAtomCountPrior {
    /// Minimum emitted atom count.
    pub min_atom_count: usize,
    /// Maximum emitted atom count.
    pub max_atom_count: usize,
    /// Approximate pocket atoms represented by one ligand atom.
    pub pocket_atom_divisor: f64,
}

impl AtomCountPrior for PocketVolumeAtomCountPrior {
    fn propose_atom_count(&self, context: &PocketInitializationContext<'_>) -> usize {
        let pocket_atoms = context
            .pocket_coords
            .size()
            .first()
            .copied()
            .unwrap_or(0)
            .max(0) as f64;
        let centroid = pocket_centroid_or_zero(context.pocket_coords);
        let radius = pocket_radius_or_default(context.pocket_coords, &centroid);
        let divisor = self.pocket_atom_divisor.max(1.0e-6);
        let raw = (pocket_atoms / divisor + (radius / 2.5)).round() as usize;
        raw.clamp(self.min_atom_count.max(1), self.max_atom_count.max(1))
    }

    fn provenance(&self) -> &'static str {
        "pocket_volume"
    }
}

/// Replaceable scaffold initializer for generation modes that own initialization.
pub trait ScaffoldInitializer {
    /// Build an initial partial ligand state without reading target ligand atoms.
    fn initialize(&self, context: &PocketInitializationContext<'_>) -> PartialLigandState;
}

/// Deterministic pocket-centroid initializer used as a low-claim baseline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PocketCentroidScaffoldInitializer {
    /// Fixed atom-count prior.
    pub atom_count_prior: FixedAtomCountPrior,
    /// Atom-type token assigned to every initialized atom.
    pub atom_type_token: i64,
    /// Fraction of pocket radius used for deterministic offsets.
    pub radius_fraction: f64,
    /// Seed used by deterministic offset construction.
    pub coordinate_seed: u64,
}

impl ScaffoldInitializer for PocketCentroidScaffoldInitializer {
    fn initialize(&self, context: &PocketInitializationContext<'_>) -> PartialLigandState {
        let device = context.pocket_coords.device();
        let atom_count = self.atom_count_prior.propose_atom_count(context).max(1) as i64;
        let atom_types = Tensor::full([atom_count], self.atom_type_token, (Kind::Int64, device));
        let centroid = pocket_centroid_or_zero(context.pocket_coords);
        let radius = pocket_radius_or_default(context.pocket_coords, &centroid)
            * self.radius_fraction.max(1.0e-6);
        let offsets = deterministic_pocket_offsets(
            context.example_id,
            context.protein_id,
            atom_count as usize,
            radius as f32,
            self.coordinate_seed,
        );
        let coords = centroid.unsqueeze(0)
            + Tensor::from_slice(&offsets)
                .reshape([atom_count, 3])
                .to_device(device);

        PartialLigandState {
            atom_types,
            coords,
            atom_mask: Tensor::ones([atom_count], (Kind::Float, device)),
            step_index: 0,
        }
    }
}

/// Pocket-conditioned scaffold initializer for de novo molecular flow.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DeNovoScaffoldInitializer {
    /// Pocket-derived atom-count policy.
    pub atom_count_prior: PocketVolumeAtomCountPrior,
    /// Atom vocabulary size used to keep deterministic priors in range.
    pub atom_vocab_size: i64,
    /// Fraction of pocket radius used for deterministic coordinate offsets.
    pub radius_fraction: f64,
    /// Seed used by deterministic atom-type and coordinate construction.
    pub seed: u64,
}

impl ScaffoldInitializer for DeNovoScaffoldInitializer {
    fn initialize(&self, context: &PocketInitializationContext<'_>) -> PartialLigandState {
        let device = context.pocket_coords.device();
        let atom_count = self.atom_count_prior.propose_atom_count(context).max(1);
        let atom_types = deterministic_de_novo_atom_types(
            context.example_id,
            context.protein_id,
            atom_count,
            self.atom_vocab_size,
            self.seed,
            device,
        );
        let centroid = pocket_centroid_or_zero(context.pocket_coords);
        let radius = pocket_radius_or_default(context.pocket_coords, &centroid)
            * self.radius_fraction.max(1.0e-6);
        let offsets = deterministic_pocket_offsets(
            context.example_id,
            context.protein_id,
            atom_count,
            radius as f32,
            self.seed,
        );
        let coords = centroid.unsqueeze(0)
            + Tensor::from_slice(&offsets)
                .reshape([atom_count as i64, 3])
                .to_device(device);

        PartialLigandState {
            atom_types,
            coords,
            atom_mask: Tensor::ones([atom_count as i64], (Kind::Float, device)),
            step_index: 0,
        }
    }
}

fn deterministic_de_novo_atom_types(
    example_id: &str,
    protein_id: &str,
    atom_count: usize,
    atom_vocab_size: i64,
    seed: u64,
    device: tch::Device,
) -> Tensor {
    let usable_vocab = atom_vocab_size.max(1);
    let heavy_template = [0_i64, 0, 0, 1, 2, 3];
    let values = (0..atom_count)
        .map(|atom_ix| {
            let hash = stable_initialization_hash(example_id, protein_id, atom_ix, seed);
            let template_value = heavy_template[(hash as usize) % heavy_template.len()];
            template_value.min(usable_vocab - 1).max(0)
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&values).to_device(device)
}

fn pocket_centroid_or_zero(pocket_coords: &Tensor) -> Tensor {
    if pocket_coords.numel() == 0 {
        Tensor::zeros([3], (Kind::Float, pocket_coords.device()))
    } else {
        pocket_coords.mean_dim([0].as_slice(), false, Kind::Float)
    }
}

fn pocket_radius_or_default(pocket_coords: &Tensor, centroid: &Tensor) -> f64 {
    if pocket_coords.numel() == 0 {
        return 1.0;
    }
    (pocket_coords - centroid.unsqueeze(0))
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
        .max(1.0)
}

fn deterministic_pocket_offsets(
    example_id: &str,
    protein_id: &str,
    atom_count: usize,
    radius: f32,
    seed: u64,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(atom_count * 3);
    for atom_ix in 0..atom_count {
        let hash = stable_initialization_hash(example_id, protein_id, atom_ix, seed);
        let phase = (hash % 65_521) as f32 / 65_521.0 * std::f32::consts::TAU;
        let height = ((hash.rotate_left(11) % 10_000) as f32 / 10_000.0 - 0.5) * radius;
        values.push(phase.cos() * radius);
        values.push(phase.sin() * radius);
        values.push(height);
    }
    values
}

fn stable_initialization_hash(
    example_id: &str,
    protein_id: &str,
    atom_ix: usize,
    seed: u64,
) -> u64 {
    let mut hash = seed ^ (atom_ix as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for byte in example_id.bytes().chain(protein_id.bytes()) {
        hash = hash.rotate_left(9) ^ u64::from(byte);
        hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    }
    hash
}

/// Continuous coordinate state used by the geometry branch of flow-matching generators.
#[derive(Debug)]
pub struct FlowState {
    /// Coordinates at the current integration time `t` with shape `[num_atoms, 3]`.
    pub coords: Tensor,
    /// Initial coordinates used to build the transport path.
    pub x0_coords: Tensor,
    /// Optional target coordinates used during training only.
    pub target_coords: Option<Tensor>,
    /// Scalar normalized timestep in `[0, 1]`.
    pub t: f64,
}

impl Clone for FlowState {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.shallow_clone(),
            x0_coords: self.x0_coords.shallow_clone(),
            target_coords: self.target_coords.as_ref().map(Tensor::shallow_clone),
            t: self.t,
        }
    }
}

/// Explicit conditioning bundle consumed by flow-matching velocity heads.
#[derive(Debug)]
pub struct ConditioningState {
    /// Topology conditioning context with shape `[slots, hidden_dim]`.
    pub topology_context: Tensor,
    /// Geometry conditioning context with shape `[slots, hidden_dim]`.
    pub geometry_context: Tensor,
    /// Pocket conditioning context with shape `[slots, hidden_dim]`.
    pub pocket_context: Tensor,
    /// Directed gate summary from controlled cross-modal interaction.
    pub gate_summary: GenerationGateSummary,
}

impl Clone for ConditioningState {
    fn clone(&self) -> Self {
        Self {
            topology_context: self.topology_context.shallow_clone(),
            geometry_context: self.geometry_context.shallow_clone(),
            pocket_context: self.pocket_context.shallow_clone(),
            gate_summary: self.gate_summary,
        }
    }
}

/// Predicted per-atom flow velocity field.
#[derive(Debug)]
pub struct VelocityField {
    /// Predicted velocity vectors with shape `[num_atoms, 3]`.
    pub velocity: Tensor,
    /// Optional scalar diagnostics emitted by ablation heads.
    pub diagnostics: BTreeMap<String, f64>,
}

impl Clone for VelocityField {
    fn clone(&self) -> Self {
        Self {
            velocity: self.velocity.shallow_clone(),
            diagnostics: self.diagnostics.clone(),
        }
    }
}

/// Structured model error surface for replaceable generator heads.
#[derive(Debug, Clone)]
pub struct ModelError {
    /// Human-readable message.
    pub message: String,
}

impl ModelError {
    /// Create a model error with a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ModelError {}

/// Replaceable geometry flow-matching prediction head.
pub trait FlowMatchingHead {
    /// Predict velocity for one flow state under decomposed conditioning.
    fn predict_velocity(
        &self,
        state: &FlowState,
        conditioning: &ConditioningState,
    ) -> Result<VelocityField, ModelError>;
}

/// Rollout-level provenance separating raw logits, raw native graph extraction, and constrained graph evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeGraphLayerProvenance {
    /// Layer that owns the raw flow-head logits before graph extraction.
    pub raw_logits_layer: String,
    /// Layer that owns thresholded native graph extraction before valence or repair constraints.
    pub raw_native_extraction_layer: String,
    /// Layer that owns connectivity, density, and valence constrained graph payloads.
    pub constrained_graph_layer: String,
    /// Layer that owns geometry-repaired graph payloads when downstream candidate repair is enabled.
    pub repaired_graph_layer: String,
    /// Number of pairwise raw bond logits considered by the extractor.
    pub raw_bond_logit_pair_count: usize,
    /// Number of thresholded raw native bonds before graph constraints.
    pub raw_native_bond_count: usize,
    /// Number of bonds retained by the constrained native graph.
    pub constrained_bond_count: usize,
    /// Number of thresholded raw native bonds removed by graph guardrails.
    pub raw_to_constrained_removed_bond_count: usize,
    /// Number of low-score connectivity edges inserted by the graph extractor.
    pub connectivity_guardrail_added_bond_count: usize,
    /// Number of bond-type downgrades caused by conservative valence checks.
    pub valence_guardrail_downgrade_count: usize,
    /// Aggregate count of graph guardrail actions observed during extraction.
    pub guardrail_trigger_count: usize,
}

impl Default for NativeGraphLayerProvenance {
    fn default() -> Self {
        Self {
            raw_logits_layer: "unavailable".to_string(),
            raw_native_extraction_layer: "unavailable".to_string(),
            constrained_graph_layer: "unavailable".to_string(),
            repaired_graph_layer: "unavailable".to_string(),
            raw_bond_logit_pair_count: 0,
            raw_native_bond_count: 0,
            constrained_bond_count: 0,
            raw_to_constrained_removed_bond_count: 0,
            connectivity_guardrail_added_bond_count: 0,
            valence_guardrail_downgrade_count: 0,
            guardrail_trigger_count: 0,
        }
    }
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
    /// Native model-predicted bonds after applying this step.
    #[serde(default)]
    pub native_bonds: Vec<(usize, usize)>,
    /// Native model-predicted bond types aligned with `native_bonds`.
    #[serde(default)]
    pub native_bond_types: Vec<i64>,
    /// Native graph after connectivity, density, and valence constraints.
    #[serde(default)]
    pub constrained_native_bonds: Vec<(usize, usize)>,
    /// Constrained native bond types aligned with `constrained_native_bonds`.
    #[serde(default)]
    pub constrained_native_bond_types: Vec<i64>,
    /// Layer-level provenance for raw logits, native extraction, constrained graph, and repair.
    #[serde(default)]
    pub native_graph_provenance: NativeGraphLayerProvenance,
    /// Mean per-atom displacement applied at this step.
    pub mean_displacement: f64,
    /// Fraction of atoms whose committed identity changed at this step.
    pub atom_change_fraction: f64,
    /// Effective coordinate step scale applied by the rollout controller.
    pub coordinate_step_scale: f64,
    /// Whether this step produced a severe ligand-pocket steric clash diagnostic.
    #[serde(default)]
    pub severe_clash_flag: bool,
    /// Whether this step violated conservative topology valence guardrails.
    #[serde(default)]
    pub valence_guardrail_flag: bool,
    /// Whether this step showed a close-range ligand-pocket pharmacophore conflict.
    #[serde(default)]
    pub pharmacophore_conflict_flag: bool,
    /// Whether chemistry guardrails would have blocked this stop if enforcement were enabled.
    #[serde(default, alias = "stop_overridden_flag")]
    pub guardrail_blockable_stop_flag: bool,
    /// Optional flow-head diagnostics such as atom-pocket gate usage and pairwise message stats.
    #[serde(default)]
    pub flow_diagnostics: BTreeMap<String, f64>,
}

/// Full iterative rollout trace for one conditioned ligand generation example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRolloutRecord {
    /// Stable example identifier.
    pub example_id: String,
    /// Stable protein identifier.
    pub protein_id: String,
    /// Stable pocket identifier used for generation reports.
    #[serde(default)]
    pub pocket_id: String,
    /// Zero-based sample index for multi-sample de novo initialization.
    #[serde(default)]
    pub sample_index: usize,
    /// Bounded per-pocket sample count used by this rollout.
    #[serde(default = "default_generation_sample_count")]
    pub sample_count: usize,
    /// Deterministic seed used by the scaffold/x0 initializer when available.
    #[serde(default)]
    pub sample_seed: Option<u64>,
    /// Report-facing provenance for the sample seed.
    #[serde(default = "default_generation_sample_seed_provenance")]
    pub sample_seed_provenance: String,
    /// Explicit generation-mode contract for this rollout.
    #[serde(default = "default_generation_mode_label")]
    pub generation_mode: String,
    /// Decoder capability contract active for this rollout.
    #[serde(default = "default_decoder_capability_label")]
    pub decoder_capability: String,
    /// Atom-count source used by the initial ligand state.
    #[serde(default = "default_atom_count_source_label")]
    pub atom_count_source: String,
    /// Provenance family for the atom-count prior.
    #[serde(default = "default_atom_count_prior_provenance_label")]
    pub atom_count_prior_provenance: String,
    /// Topology source used by the initial ligand state.
    #[serde(default = "default_topology_source_label")]
    pub topology_source: String,
    /// Geometry source used by the initial ligand state.
    #[serde(default = "default_geometry_source_label")]
    pub geometry_source: String,
    /// Coordinate-frame provenance used to build conditioning inputs.
    #[serde(default = "default_conditioning_coordinate_frame_label")]
    pub conditioning_coordinate_frame: String,
    /// Flow-specific x0 source used when a coordinate flow rollout is active.
    #[serde(default)]
    pub flow_x0_source: Option<String>,
    /// Configured step budget for this rollout.
    pub configured_steps: usize,
    /// Actual number of executed refinement steps.
    pub executed_steps: usize,
    /// Whether rollout terminated due to the learned stop head.
    pub stopped_early: bool,
    /// Directed interaction path usage observed by the inference path.
    #[serde(default)]
    pub path_usage: GenerationPathUsageSummary,
    /// Context refresh policy used for this rollout.
    #[serde(default)]
    pub context_refresh_policy: String,
    /// Number of rollout steps where context refresh was scheduled.
    #[serde(default)]
    pub refresh_count: usize,
    /// Last rollout step where context refresh was scheduled.
    #[serde(default)]
    pub last_refresh_step: Option<usize>,
    /// Number of executed steps that reused stale static or periodic context.
    #[serde(default)]
    pub stale_context_steps: usize,
    /// Whether any rollout step produced a severe ligand-pocket steric clash diagnostic.
    #[serde(default)]
    pub severe_clash_flag: bool,
    /// Whether any rollout step violated conservative topology valence guardrails.
    #[serde(default)]
    pub valence_guardrail_flag: bool,
    /// Whether any rollout step showed a close-range ligand-pocket pharmacophore conflict.
    #[serde(default)]
    pub pharmacophore_conflict_flag: bool,
    /// Whether chemistry guardrails would have blocked at least one stop if enforcement were enabled.
    #[serde(default, alias = "stop_overridden_flag")]
    pub guardrail_blockable_stop_flag: bool,
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
    /// Cached bond count for artifact filtering and chemistry reports.
    #[serde(default)]
    pub bond_count: usize,
    /// Cached count of atoms whose inferred degree exceeds a conservative valence cap.
    #[serde(default)]
    pub valence_violation_count: usize,
    /// Pocket centroid used for downstream compatibility heuristics.
    pub pocket_centroid: [f32; 3],
    /// Pocket radius summary used for downstream compatibility heuristics.
    pub pocket_radius: f32,
    /// Translation from ligand-centered model coordinates back to source structure coordinates.
    pub coordinate_frame_origin: [f32; 3],
    /// Generator path that produced this candidate.
    pub source: String,
    /// Explicit generation-mode contract for this candidate.
    #[serde(default = "default_generation_mode_label")]
    pub generation_mode: String,
    /// Canonical metric layer used when aggregating this candidate.
    #[serde(default = "default_generation_layer_label")]
    pub generation_layer: String,
    /// Coarse generation path class for raw-vs-processed attribution.
    #[serde(default = "default_generation_path_class_label")]
    pub generation_path_class: String,
    /// Whether this candidate is raw model-native evidence before postprocessing.
    #[serde(default)]
    pub model_native_raw: bool,
    /// Ordered postprocessing steps already applied to this candidate.
    #[serde(default)]
    pub postprocessor_chain: Vec<String>,
    /// Claim-boundary note for interpreting this candidate layer.
    #[serde(default)]
    pub claim_boundary: String,
    /// Optional source protein structure path used for downstream scoring workflows.
    pub source_pocket_path: Option<String>,
    /// Optional source ligand path associated with the conditioning example.
    pub source_ligand_path: Option<String>,
}

/// JSONL request schema for external molecular generator wrappers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalGenerationRequestRecord {
    /// Schema version for wrapper compatibility.
    pub schema_version: u32,
    /// Stable request id.
    pub request_id: String,
    /// Example id being generated.
    pub example_id: String,
    /// Protein id for pocket conditioning.
    pub protein_id: String,
    /// Requested candidate count.
    pub candidate_limit: usize,
    /// Explicit generation-mode contract for this request.
    #[serde(default = "default_generation_mode_label")]
    pub generation_mode: String,
    /// Explicit decomposed conditioning summary.
    pub conditioning: ExternalConditioningSummary,
}

/// Compact decomposed conditioning summary safe for external wrappers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalConditioningSummary {
    /// Mean active topology slot fraction.
    pub topology_slot_activation: f64,
    /// Mean active geometry slot fraction.
    pub geometry_slot_activation: f64,
    /// Mean active pocket/context slot fraction.
    pub pocket_slot_activation: f64,
    /// Directed gate summary.
    pub gates: GenerationGateSummary,
}

/// JSONL response schema for external molecular generator wrappers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalGenerationResponseRecord {
    /// Schema version for wrapper compatibility.
    pub schema_version: u32,
    /// Request id echoed from the request.
    pub request_id: String,
    /// Candidate payloads emitted by the wrapper.
    pub candidates: Vec<GeneratedCandidateRecord>,
    /// Wrapper status such as `ok`, `unavailable`, or `error`.
    pub status: String,
    /// Optional wrapper version string.
    pub wrapper_version: Option<String>,
    /// Optional environment fingerprint for auditability.
    pub environment_fingerprint: Option<String>,
    /// Optional failure reason.
    pub failure_reason: Option<String>,
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
    /// Raw geometry before any bond payload refinement.
    RawGeometry,
    /// Direct method-native rollout output.
    RawRollout,
    /// Coordinate-preserving bond-logit or distance-bond refinement.
    BondLogitsRefined,
    /// Coordinate-preserving valence-constrained bond refinement.
    ValenceRefined,
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
            Self::RawGeometry => "raw_geometry_candidates",
            Self::RawRollout => "raw_rollout",
            Self::BondLogitsRefined => "bond_logits_refined_candidates",
            Self::ValenceRefined => "valence_refined_candidates",
            Self::Repaired => "repaired_candidates",
            Self::InferredBond => "inferred_bond_candidates",
            Self::DeterministicProxy => "deterministic_proxy_candidates",
            Self::Reranked => "reranked_candidates",
        }
    }

    /// Canonical claim-facing generation layer name.
    pub fn canonical_generation_layer(self) -> &'static str {
        match self {
            Self::RawGeometry | Self::RawRollout => "raw_flow",
            Self::BondLogitsRefined | Self::ValenceRefined | Self::InferredBond => {
                "constrained_flow"
            }
            Self::Repaired => "repaired",
            Self::DeterministicProxy => "deterministic_proxy",
            Self::Reranked => "reranked",
        }
    }

    /// Coarse path class used to separate native generation from postprocessing.
    pub fn generation_path_class(self) -> &'static str {
        match self {
            Self::RawGeometry | Self::RawRollout => "model_native_raw",
            Self::BondLogitsRefined | Self::ValenceRefined | Self::InferredBond => "constrained",
            Self::Repaired => "repaired",
            Self::DeterministicProxy | Self::Reranked => "reranked",
        }
    }

    /// Whether the layer is raw model-native output before shared postprocessors.
    pub fn is_model_native_raw(self) -> bool {
        matches!(self, Self::RawGeometry | Self::RawRollout)
    }

    /// Short claim boundary for this layer.
    pub fn claim_boundary(self) -> &'static str {
        match self {
            Self::RawGeometry | Self::RawRollout => {
                "raw model-native decoder output before repair, reranking, or backend scoring"
            }
            Self::BondLogitsRefined | Self::ValenceRefined | Self::InferredBond => {
                "constraint-supported candidate after bond or valence postprocessing"
            }
            Self::Repaired => {
                "geometry-repaired candidate; improvements are not model-native alone"
            }
            Self::DeterministicProxy => {
                "deterministic selector output; selection contribution must be separated"
            }
            Self::Reranked => "reranked candidate; final quality includes selection contribution",
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
    /// Canonical claim-facing generation layer name.
    #[serde(default)]
    pub canonical_generation_layer: String,
    /// Coarse generation path class.
    #[serde(default)]
    pub generation_path_class: String,
    /// Explicit generation-mode contract for this layer.
    #[serde(default = "default_generation_mode_label")]
    pub generation_mode: String,
    /// Whether the layer is method-native rather than derived.
    pub method_native: bool,
    /// Ordered postprocessor chain that produced this layer.
    pub postprocessor_chain: Vec<String>,
    /// Whether this layer already includes backend-scored evidence.
    #[serde(default)]
    pub backend_supported: bool,
    /// Claim boundary for interpreting this layer.
    #[serde(default)]
    pub claim_boundary: String,
    /// Whether the layer is available for this method.
    pub available: bool,
}

fn default_generation_mode_label() -> String {
    GenerationModeConfig::TargetLigandDenoising
        .as_str()
        .to_string()
}

fn default_decoder_capability_label() -> String {
    DecoderCapabilityDescriptor::fixed_atom_refinement()
        .label()
        .to_string()
}

fn default_generation_sample_count() -> usize {
    1
}

fn default_generation_sample_seed_provenance() -> String {
    "single_sample_default".to_string()
}

fn default_atom_count_source_label() -> String {
    GenerationModeConfig::TargetLigandDenoising
        .atom_count_source_label()
        .to_string()
}

fn default_atom_count_prior_provenance_label() -> String {
    "target_ligand".to_string()
}

fn default_topology_source_label() -> String {
    GenerationModeConfig::TargetLigandDenoising
        .topology_source_label()
        .to_string()
}

fn default_geometry_source_label() -> String {
    GenerationModeConfig::TargetLigandDenoising
        .geometry_source_label()
        .to_string()
}

fn default_conditioning_coordinate_frame_label() -> String {
    "ligand_centered_model_frame".to_string()
}

fn default_generation_layer_label() -> String {
    "unassigned".to_string()
}

fn default_generation_path_class_label() -> String {
    "unassigned".to_string()
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
    /// Raw geometry layer before any coordinate or bond refinement.
    #[serde(default)]
    pub raw_geometry: Option<CandidateLayerOutput>,
    /// Coordinate-preserving bond-logit refinement layer.
    #[serde(default)]
    pub bond_logits_refined: Option<CandidateLayerOutput>,
    /// Coordinate-preserving valence-constrained refinement layer.
    #[serde(default)]
    pub valence_refined: Option<CandidateLayerOutput>,
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
            raw_geometry: None,
            bond_logits_refined: None,
            valence_refined: None,
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
    /// Explicit decomposed backend request.
    pub conditioned_request: Option<ConditionedGenerationRequest>,
    /// Optional shared backbone forward state.
    pub(crate) forward: Option<crate::models::system::ResearchForward>,
    /// Requested candidate count for this execution.
    pub candidate_limit: usize,
    /// Whether shared repair postprocessing is enabled for this execution.
    pub enable_repair: bool,
}

impl PocketGenerationContext {
    /// Attach explicit decomposed conditioning derived from the shared backbone.
    pub(crate) fn with_conditioned_request(
        mut self,
        generation_config: &GenerationTargetConfig,
    ) -> Self {
        self.conditioned_request = self.forward.as_ref().map(|forward| {
            ConditionedGenerationRequest::from_forward(&self.example, forward, generation_config)
        });
        self
    }
}

/// Stable generation-method contract for comparison runners and demos.
pub trait PocketGenerationMethod {
    /// Return stable method metadata for registry, artifacts, and claim review.
    fn metadata(&self) -> PocketGenerationMethodMetadata;

    /// Generate layered outputs for one example.
    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput;

    /// Generate layered outputs for a batch.
    fn generate_batch(
        &self,
        contexts: Vec<PocketGenerationContext>,
    ) -> Vec<LayeredGenerationOutput> {
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
    /// Normalized assignment mass retained for compatibility with existing reports.
    pub slot_weights: Tensor,
    /// Per-token slot assignment probabilities with shape `[num_tokens, num_slots]`.
    pub token_assignments: Tensor,
    /// Independent learned slot activation logits with shape `[num_slots]`.
    pub slot_activation_logits: Tensor,
    /// Independent learned slot activation gates with shape `[num_slots]`.
    pub slot_activations: Tensor,
    /// Binary attention-visible slot mask with shape `[num_slots]`.
    pub active_slot_mask: Tensor,
    /// Number of slots whose activation gate is above the active threshold.
    pub active_slot_count: f64,
    /// Reconstruction of token features with shape `[num_tokens, hidden_dim]`.
    pub reconstructed_tokens: Tensor,
}

impl Clone for SlotEncoding {
    fn clone(&self) -> Self {
        Self {
            slots: self.slots.shallow_clone(),
            slot_weights: self.slot_weights.shallow_clone(),
            token_assignments: self.token_assignments.shallow_clone(),
            slot_activation_logits: self.slot_activation_logits.shallow_clone(),
            slot_activations: self.slot_activations.shallow_clone(),
            active_slot_mask: self.active_slot_mask.shallow_clone(),
            active_slot_count: self.active_slot_count,
            reconstructed_tokens: self.reconstructed_tokens.shallow_clone(),
        }
    }
}

/// Batched slot decomposition output for one modality.
#[derive(Debug)]
pub(crate) struct BatchedSlotEncoding {
    /// Slot tensors with shape `[batch, num_slots, hidden_dim]`.
    pub slots: Tensor,
    /// Normalized assignment mass retained for compatibility with existing reports.
    pub slot_weights: Tensor,
    /// Per-token slot assignment probabilities with shape `[batch, max_tokens, num_slots]`.
    pub token_assignments: Tensor,
    /// Independent learned slot activation logits with shape `[batch, num_slots]`.
    pub slot_activation_logits: Tensor,
    /// Independent learned slot activation gates with shape `[batch, num_slots]`.
    pub slot_activations: Tensor,
    /// Binary attention-visible slot mask with shape `[batch, num_slots]`.
    pub active_slot_mask: Tensor,
    /// Active slot counts per example with shape `[batch]`.
    pub active_slot_count: Tensor,
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
            token_assignments: self.token_assignments.shallow_clone(),
            slot_activation_logits: self.slot_activation_logits.shallow_clone(),
            slot_activations: self.slot_activations.shallow_clone(),
            active_slot_mask: self.active_slot_mask.shallow_clone(),
            active_slot_count: self.active_slot_count.shallow_clone(),
            reconstructed_tokens: self.reconstructed_tokens.shallow_clone(),
            token_mask: self.token_mask.shallow_clone(),
        }
    }
}

/// Output for one directed gated cross-modal attention path.
#[derive(Debug)]
pub struct CrossAttentionOutput {
    /// Gate tensor in `[0, 1]`; scalar path gates have shape `[1]`, fine-grained gates may be per target slot.
    pub gate: Tensor,
    /// Whether the gate value was forced open by a negative-control ablation.
    pub forced_open: bool,
    /// Attention-weighted update with shape `[query_len, hidden_dim]`.
    pub attended_tokens: Tensor,
    /// Attention weights with shape `[query_len, key_len]`.
    pub attention_weights: Tensor,
}

impl Clone for CrossAttentionOutput {
    fn clone(&self) -> Self {
        Self {
            gate: self.gate.shallow_clone(),
            forced_open: self.forced_open,
            attended_tokens: self.attended_tokens.shallow_clone(),
            attention_weights: self.attention_weights.shallow_clone(),
        }
    }
}

/// Batched output for one directed gated cross-modal attention path.
#[derive(Debug)]
pub(crate) struct BatchedCrossAttentionOutput {
    /// Gate tensor per batch item; scalar path gates have shape `[batch, 1]`, fine-grained gates may be per target slot.
    pub gate: Tensor,
    /// Whether the gate value was forced open by a negative-control ablation.
    pub forced_open: bool,
    /// Attention-weighted update with shape `[batch, query_len, hidden_dim]`.
    pub attended_tokens: Tensor,
    /// Attention weights with shape `[batch, query_len, key_len]`.
    pub attention_weights: Tensor,
}

impl Clone for BatchedCrossAttentionOutput {
    fn clone(&self) -> Self {
        Self {
            gate: self.gate.shallow_clone(),
            forced_open: self.forced_open,
            attended_tokens: self.attended_tokens.shallow_clone(),
            attention_weights: self.attention_weights.shallow_clone(),
        }
    }
}
