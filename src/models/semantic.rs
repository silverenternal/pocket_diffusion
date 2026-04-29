//! Semantic modality branches for topology, geometry, and pocket context.
//!
//! Each branch owns its encoder and slot decomposer so the model surface mirrors
//! the three semantic factors used by the paper objective. Cross-modality
//! exchange remains outside these branches and is handled only by explicit
//! gated interaction modules.

use tch::{nn, Kind, Tensor};

use super::{
    BatchedModalityEncoding, BatchedSlotEncoding, Encoder, GeometryEncoderImpl, ModalityEncoding,
    PocketEncoderImpl, SlotDecomposer, SlotEncoding, SoftSlotDecomposer, TopologyEncoderImpl,
};
use crate::data::{GeometryFeatures, PocketFeatures, TopologyFeatures};

/// Compact branch-level diagnostics emitted by semantic branches.
#[allow(dead_code)] // Diagnostics are serialized/inspected by audit tooling outside core lib paths.
#[derive(Debug, Clone)]
pub(crate) struct SemanticBranchDiagnostics {
    /// Branch name used by diagnostics and ablation reports.
    pub modality: String,
    /// Number of tokens entering the branch encoder.
    pub token_count: i64,
    /// Number of active slots in the branch decomposition.
    pub slot_count: i64,
    /// Fraction of slots with non-trivial activation.
    pub active_slot_fraction: f64,
    /// Fraction of slots visible to attention after active-slot masking.
    pub attention_visible_slot_fraction: f64,
    /// Number of slots whose independent activation gate is above threshold.
    pub active_slot_count: f64,
    /// Number of slots whose independent activation gate is effectively off.
    pub dead_slot_count: f64,
    /// Number of slots with mid-range diffuse activation.
    pub diffuse_slot_count: f64,
    /// Mean independent slot activation.
    pub mean_slot_activation: f64,
    /// Entropy of slot utilization distribution.
    pub slot_entropy: f64,
    /// Norm of the pooled branch embedding.
    pub pooled_norm: f64,
    /// Reconstruction mean-squared error from branch slots to tokens when available.
    pub reconstruction_mse: Option<f64>,
}

impl SemanticBranchDiagnostics {
    /// Build diagnostics from one example's modality encoding and slots.
    pub(crate) fn from_modalities(
        modality: impl Into<String>,
        encoding: &ModalityEncoding,
        slots: &SlotEncoding,
    ) -> Self {
        let modality = modality.into();
        Self {
            modality,
            token_count: encoding.token_embeddings.size()[0],
            slot_count: slots.slot_weights.size()[0],
            active_slot_fraction: active_slot_fraction(&slots.slot_activations),
            attention_visible_slot_fraction: attention_visible_slot_fraction(
                &slots.active_slot_mask,
            ),
            active_slot_count: slots.active_slot_count,
            dead_slot_count: dead_slot_count(&slots.slot_activations),
            diffuse_slot_count: diffuse_slot_count(&slots.slot_activations),
            mean_slot_activation: mean_slot_activation(&slots.slot_activations),
            slot_entropy: slot_entropy(&slots.slot_weights),
            pooled_norm: tensor_norm_f64(&encoding.pooled_embedding),
            reconstruction_mse: Some(reconstruction_mse(
                &encoding.token_embeddings,
                &slots.reconstructed_tokens,
            )),
        }
    }

    /// Build batched diagnostics from batched modality encoding and slots.
    #[allow(dead_code)] // Retained for evaluation hooks that operate on batch-level diagnostics.
    pub(crate) fn from_batched_modalities(
        modality: impl Into<String>,
        encoding: &BatchedModalityEncoding,
        slots: &BatchedSlotEncoding,
    ) -> Self {
        let modality = modality.into();
        Self {
            modality,
            token_count: encoding
                .token_mask
                .sum(Kind::Float)
                .to_kind(Kind::Int64)
                .int64_value(&[]),
            slot_count: slots.slot_weights.size()[1],
            active_slot_fraction: batched_active_slot_fraction(&slots.slot_activations),
            attention_visible_slot_fraction: batched_attention_visible_slot_fraction(
                &slots.active_slot_mask,
            ),
            active_slot_count: slots.active_slot_count.mean(Kind::Float).double_value(&[]),
            dead_slot_count: batched_dead_slot_count(&slots.slot_activations),
            diffuse_slot_count: batched_diffuse_slot_count(&slots.slot_activations),
            mean_slot_activation: slots.slot_activations.mean(Kind::Float).double_value(&[]),
            slot_entropy: batched_slot_entropy(&slots.slot_weights),
            pooled_norm: batched_pooled_norm(&encoding.pooled_embedding),
            reconstruction_mse: batched_reconstruction_mse(
                &encoding.token_embeddings,
                &slots.reconstructed_tokens,
                &encoding.token_mask,
            ),
        }
    }
}

/// Diagnostics across topology, geometry, and pocket branches.
#[allow(dead_code)] // Bundle fields are deliberately report-facing even when tests inspect subsets.
#[derive(Debug, Clone)]
pub(crate) struct SemanticDiagnosticsBundle {
    /// Topology branch diagnostics.
    pub topology: SemanticBranchDiagnostics,
    /// Geometry branch diagnostics.
    pub geometry: SemanticBranchDiagnostics,
    /// Pocket/context branch diagnostics.
    pub pocket: SemanticBranchDiagnostics,
}

impl SemanticDiagnosticsBundle {
    /// Build a diagnostics bundle from one example.
    pub(crate) fn from_modalities(
        topology: (&ModalityEncoding, &SlotEncoding),
        geometry: (&ModalityEncoding, &SlotEncoding),
        pocket: (&ModalityEncoding, &SlotEncoding),
    ) -> Self {
        Self {
            topology: SemanticBranchDiagnostics::from_modalities(
                "topology", topology.0, topology.1,
            ),
            geometry: SemanticBranchDiagnostics::from_modalities(
                "geometry", geometry.0, geometry.1,
            ),
            pocket: SemanticBranchDiagnostics::from_modalities("pocket", pocket.0, pocket.1),
        }
    }

    /// Build a diagnostics bundle from batched tensors.
    #[allow(dead_code)] // Retained for batch-reporting without changing the forward API.
    pub(crate) fn from_batched_modalities(
        topology: (&BatchedModalityEncoding, &BatchedSlotEncoding),
        geometry: (&BatchedModalityEncoding, &BatchedSlotEncoding),
        pocket: (&BatchedModalityEncoding, &BatchedSlotEncoding),
    ) -> Self {
        Self {
            topology: SemanticBranchDiagnostics::from_batched_modalities(
                "topology", topology.0, topology.1,
            ),
            geometry: SemanticBranchDiagnostics::from_batched_modalities(
                "geometry", geometry.0, geometry.1,
            ),
            pocket: SemanticBranchDiagnostics::from_batched_modalities(
                "pocket", pocket.0, pocket.1,
            ),
        }
    }
}

fn active_slot_fraction(weights: &Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    weights
        .gt(0.05)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn attention_visible_slot_fraction(mask: &Tensor) -> f64 {
    if mask.numel() == 0 {
        return 0.0;
    }
    mask.to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn dead_slot_count(activations: &Tensor) -> f64 {
    if activations.numel() == 0 {
        return 0.0;
    }
    activations
        .lt(0.05)
        .to_kind(Kind::Float)
        .sum(Kind::Float)
        .double_value(&[])
}

fn diffuse_slot_count(activations: &Tensor) -> f64 {
    if activations.numel() == 0 {
        return 0.0;
    }
    let lower = activations.gt(0.05).to_kind(Kind::Float);
    let upper = activations.lt(0.5).to_kind(Kind::Float);
    (lower * upper).sum(Kind::Float).double_value(&[])
}

fn mean_slot_activation(activations: &Tensor) -> f64 {
    if activations.numel() == 0 {
        return 0.0;
    }
    activations.mean(Kind::Float).double_value(&[])
}

#[allow(dead_code)] // Helper for the retained batched semantic diagnostics path.
fn batched_active_slot_fraction(weights: &Tensor) -> f64 {
    if weights.size().len() != 2 || weights.numel() == 0 {
        return 0.0;
    }
    let active = weights.gt(0.05).to_kind(Kind::Float);
    active
        .mean_dim([1].as_slice(), true, Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

#[allow(dead_code)] // Helper for the retained batched semantic diagnostics path.
fn batched_attention_visible_slot_fraction(mask: &Tensor) -> f64 {
    if mask.size().len() != 2 || mask.numel() == 0 {
        return 0.0;
    }
    mask.to_kind(Kind::Float)
        .mean_dim([1].as_slice(), true, Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn batched_dead_slot_count(activations: &Tensor) -> f64 {
    if activations.numel() == 0 {
        return 0.0;
    }
    activations
        .lt(0.05)
        .to_kind(Kind::Float)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn batched_diffuse_slot_count(activations: &Tensor) -> f64 {
    if activations.numel() == 0 {
        return 0.0;
    }
    let lower = activations.gt(0.05).to_kind(Kind::Float);
    let upper = activations.lt(0.5).to_kind(Kind::Float);
    (lower * upper)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn slot_entropy(weights: &Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    let normalized = (weights / weights.sum(Kind::Float).clamp_min(1e-12)).clamp_min(1e-12);
    let entropy = -(&normalized * normalized.log()).sum(Kind::Float);
    entropy.double_value(&[])
}

#[allow(dead_code)] // Helper for the retained batched semantic diagnostics path.
fn batched_slot_entropy(weights: &Tensor) -> f64 {
    if weights.size().len() != 2 || weights.numel() == 0 {
        return 0.0;
    }
    let row_sums = weights.sum_dim_intlist([1].as_slice(), true, Kind::Float);
    let normalized = (weights / row_sums.clamp_min(1e-12)).clamp_min(1e-12);
    let entropy =
        -(&normalized * normalized.log()).sum_dim_intlist([1].as_slice(), false, Kind::Float);
    entropy.mean(Kind::Float).double_value(&[])
}

fn tensor_norm_f64(values: &Tensor) -> f64 {
    if values.numel() == 0 {
        0.0
    } else {
        values
            .pow_tensor_scalar(2.0)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[])
    }
}

#[allow(dead_code)] // Helper for the retained batched semantic diagnostics path.
fn batched_pooled_norm(pooled: &Tensor) -> f64 {
    if pooled.numel() == 0 {
        0.0
    } else {
        let norms = pooled
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            .sqrt();
        norms.mean(Kind::Float).double_value(&[])
    }
}

fn reconstruction_mse(original: &Tensor, reconstructed: &Tensor) -> f64 {
    if original.numel() == 0 || reconstructed.numel() == 0 {
        return 0.0;
    }
    if original.size().ne(&reconstructed.size()) {
        return 0.0;
    }
    (original - reconstructed)
        .pow_tensor_scalar(2.0)
        .mean(Kind::Float)
        .double_value(&[])
}

#[allow(dead_code)] // Helper for the retained batched semantic diagnostics path.
fn batched_reconstruction_mse(
    original: &Tensor,
    reconstructed: &Tensor,
    mask: &Tensor,
) -> Option<f64> {
    if original.numel() == 0 || reconstructed.numel() == 0 || mask.numel() == 0 {
        return None;
    }
    if original.size().len() != 3 || reconstructed.size().len() != 3 || mask.size().len() != 2 {
        return None;
    }
    if original.size() != reconstructed.size() || original.size()[0] != mask.size()[0] {
        return None;
    }
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    if denom.double_value(&[]) <= 0.0 {
        return None;
    }
    let mask = mask.unsqueeze(-1);
    let diff = original - reconstructed;
    let per_token = diff
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float);
    let total = (per_token * mask).sum(Kind::Float);
    Some((total / denom).double_value(&[]))
}

/// Topology semantic branch: atom/bond structure encoder plus topology slots.
#[derive(Debug)]
pub struct TopologySemanticBranch {
    /// Topology encoder.
    pub encoder: TopologyEncoderImpl,
    /// Topology slot decomposer.
    pub slots: SoftSlotDecomposer,
}

impl TopologySemanticBranch {
    /// Wrap an existing topology encoder and slot decomposer.
    pub fn from_parts(encoder: TopologyEncoderImpl, slots: SoftSlotDecomposer) -> Self {
        Self { encoder, slots }
    }

    /// Create the topology branch with its own encoder and slot namespace.
    pub fn new(vs: &nn::Path, atom_vocab_size: i64, hidden_dim: i64, num_slots: i64) -> Self {
        Self {
            encoder: TopologyEncoderImpl::new(&(vs / "encoder"), atom_vocab_size, hidden_dim),
            slots: SoftSlotDecomposer::new(&(vs / "slots"), hidden_dim, num_slots),
        }
    }

    /// Encode one topology feature set.
    pub(crate) fn encode(&self, input: &TopologyFeatures) -> ModalityEncoding {
        self.encoder.encode(input)
    }

    /// Encode padded topology tensors.
    pub(crate) fn encode_batch(
        &self,
        atom_types: &Tensor,
        adjacency: &Tensor,
        bond_type_adjacency: &Tensor,
        mask: &Tensor,
    ) -> BatchedModalityEncoding {
        self.encoder
            .encode_batch(atom_types, adjacency, bond_type_adjacency, mask)
    }

    /// Decompose topology encoding into learned slots.
    pub(crate) fn decompose(&self, encoding: &ModalityEncoding) -> SlotEncoding {
        self.slots.decompose(encoding)
    }

    /// Decompose padded topology encoding into learned slots.
    pub(crate) fn decompose_batch(
        &self,
        encoding: &BatchedModalityEncoding,
    ) -> BatchedSlotEncoding {
        self.slots.decompose_batch(encoding)
    }
}

/// Geometry semantic branch: coordinate/distance encoder plus geometry slots.
#[derive(Debug)]
pub struct GeometrySemanticBranch {
    /// Geometry encoder.
    pub encoder: GeometryEncoderImpl,
    /// Geometry slot decomposer.
    pub slots: SoftSlotDecomposer,
}

impl GeometrySemanticBranch {
    /// Wrap an existing geometry encoder and slot decomposer.
    pub fn from_parts(encoder: GeometryEncoderImpl, slots: SoftSlotDecomposer) -> Self {
        Self { encoder, slots }
    }

    /// Create the geometry branch with its own encoder and slot namespace.
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_slots: i64) -> Self {
        Self {
            encoder: GeometryEncoderImpl::new(&(vs / "encoder"), hidden_dim),
            slots: SoftSlotDecomposer::new(&(vs / "slots"), hidden_dim, num_slots),
        }
    }

    /// Encode one geometry feature set.
    pub(crate) fn encode(&self, input: &GeometryFeatures) -> ModalityEncoding {
        self.encoder.encode(input)
    }

    /// Encode padded geometry tensors.
    pub(crate) fn encode_batch(
        &self,
        coords: &Tensor,
        pairwise_distances: &Tensor,
        mask: &Tensor,
    ) -> BatchedModalityEncoding {
        self.encoder.encode_batch(coords, pairwise_distances, mask)
    }

    /// Decompose geometry encoding into learned slots.
    pub(crate) fn decompose(&self, encoding: &ModalityEncoding) -> SlotEncoding {
        self.slots.decompose(encoding)
    }

    /// Decompose padded geometry encoding into learned slots.
    pub(crate) fn decompose_batch(
        &self,
        encoding: &BatchedModalityEncoding,
    ) -> BatchedSlotEncoding {
        self.slots.decompose_batch(encoding)
    }
}

/// Pocket semantic branch: local pocket/context encoder plus pocket slots.
#[derive(Debug)]
pub struct PocketSemanticBranch {
    /// Pocket/context encoder.
    pub encoder: PocketEncoderImpl,
    /// Pocket/context slot decomposer.
    pub slots: SoftSlotDecomposer,
}

impl PocketSemanticBranch {
    /// Wrap an existing pocket encoder and slot decomposer.
    pub fn from_parts(encoder: PocketEncoderImpl, slots: SoftSlotDecomposer) -> Self {
        Self { encoder, slots }
    }

    /// Create the pocket branch with its own encoder and slot namespace.
    pub fn new(vs: &nn::Path, pocket_feature_dim: i64, hidden_dim: i64, num_slots: i64) -> Self {
        Self {
            encoder: PocketEncoderImpl::new(&(vs / "encoder"), pocket_feature_dim, hidden_dim),
            slots: SoftSlotDecomposer::new(&(vs / "slots"), hidden_dim, num_slots),
        }
    }

    /// Encode one pocket feature set.
    pub(crate) fn encode(&self, input: &PocketFeatures) -> ModalityEncoding {
        self.encoder.encode(input)
    }

    /// Encode padded pocket tensors.
    pub(crate) fn encode_batch(
        &self,
        atom_features: &Tensor,
        coords: &Tensor,
        pooled_features: &Tensor,
        mask: &Tensor,
    ) -> BatchedModalityEncoding {
        self.encoder
            .encode_batch(atom_features, coords, pooled_features, mask)
    }

    /// Decompose pocket encoding into learned slots.
    pub(crate) fn decompose(&self, encoding: &ModalityEncoding) -> SlotEncoding {
        self.slots.decompose(encoding)
    }

    /// Decompose padded pocket encoding into learned slots.
    pub(crate) fn decompose_batch(
        &self,
        encoding: &BatchedModalityEncoding,
    ) -> BatchedSlotEncoding {
        self.slots.decompose_batch(encoding)
    }
}
