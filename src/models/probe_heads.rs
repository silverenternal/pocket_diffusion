//! Lightweight semantic probe heads for modality specialization checks.

use tch::{nn, Tensor};

use crate::{config::SemanticProbeConfig, data::CHEMISTRY_ROLE_FEATURE_DIM};

use super::{ModalityEncoding, SlotEncoding};

/// Explicit leakage-probe outputs for one gradient-routing variant.
#[derive(Debug)]
pub struct LeakageProbeOutputs {
    /// Topology predicts geometry target summary.
    pub topology_to_geometry_scalar_logits: Tensor,
    /// Geometry predicts topology target summary.
    pub geometry_to_topology_scalar_logits: Tensor,
    /// Pocket predicts geometry target summary.
    pub pocket_to_geometry_scalar_logits: Tensor,
    /// Pooled topology-to-pocket role leakage probe logits.
    pub topology_to_pocket_role_logits: Tensor,
    /// Pooled geometry-to-pocket role leakage probe logits.
    pub geometry_to_pocket_role_logits: Tensor,
    /// Pooled pocket-to-ligand role leakage probe logits.
    pub pocket_to_ligand_role_logits: Tensor,
    /// Pooled pocket-to-topology role leakage probe logits.
    pub pocket_to_topology_role_logits: Tensor,
}

impl Clone for LeakageProbeOutputs {
    fn clone(&self) -> Self {
        Self {
            topology_to_geometry_scalar_logits: self
                .topology_to_geometry_scalar_logits
                .shallow_clone(),
            geometry_to_topology_scalar_logits: self
                .geometry_to_topology_scalar_logits
                .shallow_clone(),
            pocket_to_geometry_scalar_logits: self.pocket_to_geometry_scalar_logits.shallow_clone(),
            topology_to_pocket_role_logits: self.topology_to_pocket_role_logits.shallow_clone(),
            geometry_to_pocket_role_logits: self.geometry_to_pocket_role_logits.shallow_clone(),
            pocket_to_ligand_role_logits: self.pocket_to_ligand_role_logits.shallow_clone(),
            pocket_to_topology_role_logits: self.pocket_to_topology_role_logits.shallow_clone(),
        }
    }
}

/// Probe outputs used for later semantic supervision and leakage analysis.
#[derive(Debug)]
pub struct ProbeOutputs {
    /// Topology-specialized adjacency score matrix.
    pub topology_adjacency_logits: Tensor,
    /// Geometry-specialized per-atom distance estimate.
    pub geometry_distance_predictions: Tensor,
    /// Pocket-specialized atom feature reconstruction.
    pub pocket_feature_predictions: Tensor,
    /// Topology predicts geometry target summary.
    pub topology_to_geometry_scalar_logits: Tensor,
    /// Geometry predicts topology target summary.
    pub geometry_to_topology_scalar_logits: Tensor,
    /// Pocket predicts geometry target summary.
    pub pocket_to_geometry_scalar_logits: Tensor,
    /// Ligand pharmacophore role logits from ligand/topology-specialized tokens.
    pub ligand_pharmacophore_role_logits: Tensor,
    /// Pocket pharmacophore role logits from pocket/context-specialized tokens.
    pub pocket_pharmacophore_role_logits: Tensor,
    /// Pooled topology-to-pocket role leakage probe logits.
    pub topology_to_pocket_role_logits: Tensor,
    /// Pooled geometry-to-pocket role leakage probe logits.
    pub geometry_to_pocket_role_logits: Tensor,
    /// Pooled pocket-to-ligand role leakage probe logits.
    pub pocket_to_ligand_role_logits: Tensor,
    /// Pooled pocket-to-topology role leakage probe logits.
    pub pocket_to_topology_role_logits: Tensor,
    /// Explicit leakage probes computed with detached source features for probe fitting.
    pub leakage_probe_fit: LeakageProbeOutputs,
    /// Explicit leakage probes computed with detached probe parameters for encoder penalty.
    pub leakage_encoder_penalty: LeakageProbeOutputs,
    /// Complex-level affinity prediction from pooled multi-modal latents.
    pub affinity_prediction: Tensor,
}

impl Clone for ProbeOutputs {
    fn clone(&self) -> Self {
        Self {
            topology_adjacency_logits: self.topology_adjacency_logits.shallow_clone(),
            geometry_distance_predictions: self.geometry_distance_predictions.shallow_clone(),
            pocket_feature_predictions: self.pocket_feature_predictions.shallow_clone(),
            topology_to_geometry_scalar_logits: self
                .topology_to_geometry_scalar_logits
                .shallow_clone(),
            geometry_to_topology_scalar_logits: self
                .geometry_to_topology_scalar_logits
                .shallow_clone(),
            pocket_to_geometry_scalar_logits: self.pocket_to_geometry_scalar_logits.shallow_clone(),
            ligand_pharmacophore_role_logits: self.ligand_pharmacophore_role_logits.shallow_clone(),
            pocket_pharmacophore_role_logits: self.pocket_pharmacophore_role_logits.shallow_clone(),
            topology_to_pocket_role_logits: self.topology_to_pocket_role_logits.shallow_clone(),
            geometry_to_pocket_role_logits: self.geometry_to_pocket_role_logits.shallow_clone(),
            pocket_to_ligand_role_logits: self.pocket_to_ligand_role_logits.shallow_clone(),
            pocket_to_topology_role_logits: self.pocket_to_topology_role_logits.shallow_clone(),
            leakage_probe_fit: self.leakage_probe_fit.clone(),
            leakage_encoder_penalty: self.leakage_encoder_penalty.clone(),
            affinity_prediction: self.affinity_prediction.shallow_clone(),
        }
    }
}

/// A group of lightweight probe heads attached to the specialized modality paths.
#[derive(Debug)]
pub struct SemanticProbeHeads {
    topo_pair_head: ProbeProjection,
    geo_distance_head: ProbeProjection,
    pocket_feature_head: ProbeProjection,
    ligand_pharmacophore_role_head: ProbeProjection,
    pocket_pharmacophore_role_head: ProbeProjection,
    topology_to_geometry_head: ProbeProjection,
    geometry_to_topology_head: ProbeProjection,
    pocket_to_geometry_head: ProbeProjection,
    topology_to_pocket_role_head: ProbeProjection,
    geometry_to_pocket_role_head: ProbeProjection,
    pocket_to_ligand_role_head: ProbeProjection,
    pocket_to_topology_role_head: ProbeProjection,
    affinity_head: ProbeProjection,
}

impl SemanticProbeHeads {
    /// Create probe heads from the shared hidden size.
    pub fn new(vs: &nn::Path, hidden_dim: i64, pocket_feature_dim: i64) -> Self {
        Self::new_with_config(
            vs,
            hidden_dim,
            pocket_feature_dim,
            &SemanticProbeConfig::default(),
        )
    }

    /// Create probe heads with explicit capacity controls.
    pub fn new_with_config(
        vs: &nn::Path,
        hidden_dim: i64,
        pocket_feature_dim: i64,
        config: &SemanticProbeConfig,
    ) -> Self {
        let topo_pair_head = ProbeProjection::new(vs, "topo_pair_head", hidden_dim * 2, 1, config);
        let geo_distance_head =
            ProbeProjection::new(vs, "geo_distance_head", hidden_dim, 1, config);
        let pocket_feature_head = ProbeProjection::new(
            vs,
            "pocket_feature_head",
            hidden_dim,
            pocket_feature_dim,
            config,
        );
        let ligand_pharmacophore_role_head = ProbeProjection::new(
            vs,
            "ligand_pharmacophore_role_head",
            hidden_dim,
            CHEMISTRY_ROLE_FEATURE_DIM,
            config,
        );
        let pocket_pharmacophore_role_head = ProbeProjection::new(
            vs,
            "pocket_pharmacophore_role_head",
            hidden_dim,
            CHEMISTRY_ROLE_FEATURE_DIM,
            config,
        );
        let topology_to_geometry_head =
            ProbeProjection::new(vs, "topology_to_geometry_head", hidden_dim, 1, config);
        let geometry_to_topology_head =
            ProbeProjection::new(vs, "geometry_to_topology_head", hidden_dim, 1, config);
        let pocket_to_geometry_head =
            ProbeProjection::new(vs, "pocket_to_geometry_head", hidden_dim, 1, config);
        let topology_to_pocket_role_head = ProbeProjection::new(
            vs,
            "topology_to_pocket_role_head",
            hidden_dim,
            CHEMISTRY_ROLE_FEATURE_DIM,
            config,
        );
        let geometry_to_pocket_role_head = ProbeProjection::new(
            vs,
            "geometry_to_pocket_role_head",
            hidden_dim,
            CHEMISTRY_ROLE_FEATURE_DIM,
            config,
        );
        let pocket_to_ligand_role_head = ProbeProjection::new(
            vs,
            "pocket_to_ligand_role_head",
            hidden_dim,
            CHEMISTRY_ROLE_FEATURE_DIM,
            config,
        );
        let pocket_to_topology_role_head = ProbeProjection::new(
            vs,
            "pocket_to_topology_role_head",
            hidden_dim,
            CHEMISTRY_ROLE_FEATURE_DIM,
            config,
        );
        let affinity_head = ProbeProjection::new(vs, "affinity_head", hidden_dim * 3, 1, config);
        Self {
            topo_pair_head,
            geo_distance_head,
            pocket_feature_head,
            ligand_pharmacophore_role_head,
            pocket_pharmacophore_role_head,
            topology_to_geometry_head,
            geometry_to_topology_head,
            pocket_to_geometry_head,
            topology_to_pocket_role_head,
            geometry_to_pocket_role_head,
            pocket_to_ligand_role_head,
            pocket_to_topology_role_head,
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
                .apply_probe(&self.topo_pair_head)
                .squeeze_dim(-1)
        };

        let geometry_distance_predictions = geometry_tokens
            .apply_probe(&self.geo_distance_head)
            .squeeze_dim(-1);
        let pocket_feature_predictions = pocket_tokens.apply_probe(&self.pocket_feature_head);
        let topology_to_geometry_scalar_logits = topology
            .pooled_embedding
            .apply_probe(&self.topology_to_geometry_head);
        let geometry_to_topology_scalar_logits = geometry
            .pooled_embedding
            .apply_probe(&self.geometry_to_topology_head);
        let pocket_to_geometry_scalar_logits = pocket
            .pooled_embedding
            .apply_probe(&self.pocket_to_geometry_head);
        let ligand_pharmacophore_role_logits =
            topology_tokens.apply_probe(&self.ligand_pharmacophore_role_head);
        let pocket_pharmacophore_role_logits =
            pocket_tokens.apply_probe(&self.pocket_pharmacophore_role_head);
        let topology_to_pocket_role_logits = topology
            .pooled_embedding
            .apply_probe(&self.topology_to_pocket_role_head);
        let geometry_to_pocket_role_logits = geometry
            .pooled_embedding
            .apply_probe(&self.geometry_to_pocket_role_head);
        let pocket_to_ligand_role_logits = pocket
            .pooled_embedding
            .apply_probe(&self.pocket_to_ligand_role_head);
        let pocket_to_topology_role_logits = pocket
            .pooled_embedding
            .apply_probe(&self.pocket_to_topology_role_head);
        let leakage_probe_fit =
            self.leakage_outputs_with_detached_sources(topology, geometry, pocket);
        let leakage_encoder_penalty =
            self.leakage_outputs_with_detached_probe_parameters(topology, geometry, pocket);
        let affinity_prediction = Tensor::cat(
            &[
                topology.pooled_embedding.shallow_clone(),
                geometry.pooled_embedding.shallow_clone(),
                pocket.pooled_embedding.shallow_clone(),
            ],
            -1,
        )
        .apply_probe(&self.affinity_head)
        .squeeze_dim(-1);

        ProbeOutputs {
            topology_adjacency_logits,
            geometry_distance_predictions,
            pocket_feature_predictions,
            topology_to_geometry_scalar_logits,
            geometry_to_topology_scalar_logits,
            pocket_to_geometry_scalar_logits,
            ligand_pharmacophore_role_logits,
            pocket_pharmacophore_role_logits,
            topology_to_pocket_role_logits,
            geometry_to_pocket_role_logits,
            pocket_to_ligand_role_logits,
            pocket_to_topology_role_logits,
            leakage_probe_fit,
            leakage_encoder_penalty,
            affinity_prediction,
        }
    }

    fn leakage_outputs_with_detached_sources(
        &self,
        topology: &ModalityEncoding,
        geometry: &ModalityEncoding,
        pocket: &ModalityEncoding,
    ) -> LeakageProbeOutputs {
        LeakageProbeOutputs {
            topology_to_geometry_scalar_logits: self
                .topology_to_geometry_head
                .forward(&topology.pooled_embedding.detach()),
            geometry_to_topology_scalar_logits: self
                .geometry_to_topology_head
                .forward(&geometry.pooled_embedding.detach()),
            pocket_to_geometry_scalar_logits: self
                .pocket_to_geometry_head
                .forward(&pocket.pooled_embedding.detach()),
            topology_to_pocket_role_logits: self
                .topology_to_pocket_role_head
                .forward(&topology.pooled_embedding.detach()),
            geometry_to_pocket_role_logits: self
                .geometry_to_pocket_role_head
                .forward(&geometry.pooled_embedding.detach()),
            pocket_to_ligand_role_logits: self
                .pocket_to_ligand_role_head
                .forward(&pocket.pooled_embedding.detach()),
            pocket_to_topology_role_logits: self
                .pocket_to_topology_role_head
                .forward(&pocket.pooled_embedding.detach()),
        }
    }

    fn leakage_outputs_with_detached_probe_parameters(
        &self,
        topology: &ModalityEncoding,
        geometry: &ModalityEncoding,
        pocket: &ModalityEncoding,
    ) -> LeakageProbeOutputs {
        LeakageProbeOutputs {
            topology_to_geometry_scalar_logits: self
                .topology_to_geometry_head
                .forward_with_detached_parameters(&topology.pooled_embedding),
            geometry_to_topology_scalar_logits: self
                .geometry_to_topology_head
                .forward_with_detached_parameters(&geometry.pooled_embedding),
            pocket_to_geometry_scalar_logits: self
                .pocket_to_geometry_head
                .forward_with_detached_parameters(&pocket.pooled_embedding),
            topology_to_pocket_role_logits: self
                .topology_to_pocket_role_head
                .forward_with_detached_parameters(&topology.pooled_embedding),
            geometry_to_pocket_role_logits: self
                .geometry_to_pocket_role_head
                .forward_with_detached_parameters(&geometry.pooled_embedding),
            pocket_to_ligand_role_logits: self
                .pocket_to_ligand_role_head
                .forward_with_detached_parameters(&pocket.pooled_embedding),
            pocket_to_topology_role_logits: self
                .pocket_to_topology_role_head
                .forward_with_detached_parameters(&pocket.pooled_embedding),
        }
    }
}

#[derive(Debug)]
struct ProbeProjection {
    hidden_layers: Vec<nn::Linear>,
    output: nn::Linear,
}

impl ProbeProjection {
    fn new(
        vs: &nn::Path,
        name: &str,
        input_dim: i64,
        output_dim: i64,
        config: &SemanticProbeConfig,
    ) -> Self {
        if config.hidden_layers == 0 {
            return Self {
                hidden_layers: Vec::new(),
                output: nn::linear(vs / name, input_dim, output_dim, Default::default()),
            };
        }

        let mut hidden_layers = Vec::with_capacity(config.hidden_layers);
        let mut current_dim = input_dim;
        for index in 0..config.hidden_layers {
            let layer_name = format!("{name}_hidden_{index}");
            hidden_layers.push(nn::linear(
                vs / layer_name,
                current_dim,
                config.hidden_dim,
                Default::default(),
            ));
            current_dim = config.hidden_dim;
        }
        let output_name = format!("{name}_output");
        let output = nn::linear(
            vs / output_name,
            current_dim,
            output_dim,
            Default::default(),
        );
        Self {
            hidden_layers,
            output,
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut hidden = input.shallow_clone();
        for layer in &self.hidden_layers {
            hidden = hidden.apply(layer).relu();
        }
        hidden.apply(&self.output)
    }

    fn forward_with_detached_parameters(&self, input: &Tensor) -> Tensor {
        let mut hidden = input.shallow_clone();
        for layer in &self.hidden_layers {
            hidden = linear_with_detached_parameters(&hidden, layer).relu();
        }
        linear_with_detached_parameters(&hidden, &self.output)
    }
}

trait ApplyProbe {
    fn apply_probe(&self, projection: &ProbeProjection) -> Tensor;
}

impl ApplyProbe for Tensor {
    fn apply_probe(&self, projection: &ProbeProjection) -> Tensor {
        projection.forward(self)
    }
}

fn linear_with_detached_parameters(input: &Tensor, layer: &nn::Linear) -> Tensor {
    let detached_bias = layer.bs.as_ref().map(|bias| bias.detach());
    input.linear(&layer.ws.detach(), detached_bias.as_ref())
}
