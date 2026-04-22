//! Lightweight semantic probe heads for modality specialization checks.

use tch::{nn, Tensor};

use super::{ModalityEncoding, SlotEncoding};

/// Probe outputs used for later semantic supervision and leakage analysis.
#[derive(Debug)]
pub struct ProbeOutputs {
    /// Topology-specialized adjacency score matrix.
    pub topology_adjacency_logits: Tensor,
    /// Geometry-specialized per-atom distance estimate.
    pub geometry_distance_predictions: Tensor,
    /// Pocket-specialized atom feature reconstruction.
    pub pocket_feature_predictions: Tensor,
    /// Complex-level affinity prediction from pooled multi-modal latents.
    pub affinity_prediction: Tensor,
}

impl Clone for ProbeOutputs {
    fn clone(&self) -> Self {
        Self {
            topology_adjacency_logits: self.topology_adjacency_logits.shallow_clone(),
            geometry_distance_predictions: self.geometry_distance_predictions.shallow_clone(),
            pocket_feature_predictions: self.pocket_feature_predictions.shallow_clone(),
            affinity_prediction: self.affinity_prediction.shallow_clone(),
        }
    }
}

/// A group of lightweight probe heads attached to the specialized modality paths.
#[derive(Debug)]
pub struct SemanticProbeHeads {
    topo_pair_head: nn::Linear,
    geo_distance_head: nn::Linear,
    pocket_feature_head: nn::Linear,
    affinity_head: nn::Linear,
}

impl SemanticProbeHeads {
    /// Create probe heads from the shared hidden size.
    pub fn new(vs: &nn::Path, hidden_dim: i64, pocket_feature_dim: i64) -> Self {
        let topo_pair_head =
            nn::linear(vs / "topo_pair_head", hidden_dim * 2, 1, Default::default());
        let geo_distance_head =
            nn::linear(vs / "geo_distance_head", hidden_dim, 1, Default::default());
        let pocket_feature_head = nn::linear(
            vs / "pocket_feature_head",
            hidden_dim,
            pocket_feature_dim,
            Default::default(),
        );
        let affinity_head = nn::linear(vs / "affinity_head", hidden_dim * 3, 1, Default::default());
        Self {
            topo_pair_head,
            geo_distance_head,
            pocket_feature_head,
            affinity_head,
        }
    }

    /// Compute probe outputs from modality encodings and slot summaries.
    pub fn forward(
        &self,
        topology: &ModalityEncoding,
        geometry: &ModalityEncoding,
        pocket: &ModalityEncoding,
        topology_slots: &SlotEncoding,
        geometry_slots: &SlotEncoding,
        pocket_slots: &SlotEncoding,
    ) -> ProbeOutputs {
        let topology_tokens = if topology.token_embeddings.size()[0] == 0 {
            topology_slots.reconstructed_tokens.shallow_clone()
        } else {
            topology.token_embeddings.shallow_clone()
        };
        let geometry_tokens = if geometry.token_embeddings.size()[0] == 0 {
            geometry_slots.reconstructed_tokens.shallow_clone()
        } else {
            geometry.token_embeddings.shallow_clone()
        };
        let pocket_tokens = if pocket.token_embeddings.size()[0] == 0 {
            pocket_slots.reconstructed_tokens.shallow_clone()
        } else {
            pocket.token_embeddings.shallow_clone()
        };

        let num_topo = topology_tokens.size()[0];
        let topology_adjacency_logits = if num_topo == 0 {
            Tensor::zeros([0, 0], (tch::Kind::Float, topology_tokens.device()))
        } else {
            let left = topology_tokens.unsqueeze(1).repeat([1, num_topo, 1]);
            let right = topology_tokens.unsqueeze(0).repeat([num_topo, 1, 1]);
            Tensor::cat(&[left, right], -1)
                .apply(&self.topo_pair_head)
                .squeeze_dim(-1)
        };

        let geometry_distance_predictions = geometry_tokens
            .apply(&self.geo_distance_head)
            .squeeze_dim(-1);
        let pocket_feature_predictions = pocket_tokens.apply(&self.pocket_feature_head);
        let affinity_prediction = Tensor::cat(
            &[
                topology.pooled_embedding.shallow_clone(),
                geometry.pooled_embedding.shallow_clone(),
                pocket.pooled_embedding.shallow_clone(),
            ],
            -1,
        )
        .apply(&self.affinity_head)
        .squeeze_dim(-1);

        ProbeOutputs {
            topology_adjacency_logits,
            geometry_distance_predictions,
            pocket_feature_predictions,
            affinity_prediction,
        }
    }
}
