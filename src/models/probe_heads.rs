//! Lightweight semantic probe heads for modality specialization checks.

use tch::{nn, Kind, Tensor};

use crate::{config::SemanticProbeConfig, data::CHEMISTRY_ROLE_FEATURE_DIM};

use super::{ModalityEncoding, SlotEncoding};

const GEOMETRY_DISTANCE_PRIOR_INTERCEPT_ANGSTROM: f64 = -0.39;
const GEOMETRY_DISTANCE_PRIOR_SQRT_ATOM_SLOPE: f64 = 0.85;
const GEOMETRY_DISTANCE_RESIDUAL_RANGE_ANGSTROM: f64 = 2.0;
const GEOMETRY_DISTANCE_RESIDUAL_TEMPERATURE: f64 = 0.25;
const TOPOLOGY_EXPECTED_DIRECTED_DEGREE: f64 = 1.75;
const TOPOLOGY_EDGE_PROBABILITY_MIN: f64 = 0.01;
const TOPOLOGY_EDGE_PROBABILITY_MAX: f64 = 0.25;
const TOPOLOGY_EDGE_RESIDUAL_TEMPERATURE: f64 = 0.15;
const TOPOLOGY_EDGE_RESIDUAL_RANGE_LOGIT: f64 = 1.25;
const TOPOLOGY_SELF_EDGE_LOGIT: f64 = -12.0;
const SPARSE_ROLE_LOGIT_PRIOR: f64 = -1.12;
const SPARSE_ROLE_RESIDUAL_TEMPERATURE: f64 = 0.25;
const SPARSE_ROLE_RESIDUAL_RANGE_LOGIT: f64 = 1.5;
const POCKET_FEATURE_LOGIT_PRIOR: f64 = -3.2;
const POCKET_FEATURE_RESIDUAL_TEMPERATURE: f64 = 0.25;
const AFFINITY_PRIOR_KCAL_MOL: f64 = -7.0;
const AFFINITY_RESIDUAL_RANGE_KCAL_MOL: f64 = 4.0;
const AFFINITY_RESIDUAL_TEMPERATURE: f64 = 0.25;
const AFFINITY_DESCRIPTOR_DIM: i64 = 7;

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
        let topo_pair_head = ProbeProjection::new(vs, "topo_pair_head", hidden_dim * 4, 1, config);
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
        let affinity_head = ProbeProjection::new(
            vs,
            "affinity_head",
            hidden_dim * 3 + AFFINITY_DESCRIPTOR_DIM,
            1,
            config,
        );
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
        let topology_tokens = topology_probe_tokens(topology, topology_slots);
        let topology_role_tokens = slot_conditioned_probe_tokens(topology, topology_slots);
        let geometry_tokens = slot_conditioned_probe_tokens(geometry, geometry_slots);
        let pocket_tokens = slot_conditioned_probe_tokens(pocket, pocket_slots);

        let num_topo = topology_tokens.size()[0];
        let topology_adjacency_logits = if num_topo == 0 {
            Tensor::zeros([0, 0], (tch::Kind::Float, topology_tokens.device()))
        } else {
            let pair_logits = topology_pair_features(&topology_tokens)
                .apply_probe(&self.topo_pair_head)
                .squeeze_dim(-1);
            topology_adjacency_logits_with_sparse_prior(pair_logits)
        };

        let geometry_distance_prior =
            geometry_distance_size_prior(geometry_tokens.size().first().copied().unwrap_or(0))
                .to_device(geometry_tokens.device());
        let geometry_distance_residual = geometry_tokens
            .apply_probe(&self.geo_distance_head)
            .squeeze_dim(-1)
            * GEOMETRY_DISTANCE_RESIDUAL_TEMPERATURE;
        let geometry_distance_predictions = (geometry_distance_residual.tanh()
            * GEOMETRY_DISTANCE_RESIDUAL_RANGE_ANGSTROM
            + geometry_distance_prior)
            .clamp_min(0.0);
        let pocket_feature_predictions =
            sparse_pocket_feature_predictions(pocket_tokens.apply_probe(&self.pocket_feature_head));
        let topology_to_geometry_scalar_logits = topology
            .pooled_embedding
            .apply_probe(&self.topology_to_geometry_head);
        let geometry_to_topology_scalar_logits = geometry
            .pooled_embedding
            .apply_probe(&self.geometry_to_topology_head);
        let pocket_to_geometry_scalar_logits = pocket
            .pooled_embedding
            .apply_probe(&self.pocket_to_geometry_head);
        let ligand_pharmacophore_role_logits = sparse_role_logits(
            topology_role_tokens.apply_probe(&self.ligand_pharmacophore_role_head),
        );
        let pocket_pharmacophore_role_logits =
            sparse_role_logits(pocket_tokens.apply_probe(&self.pocket_pharmacophore_role_head));
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
        let affinity_residual = Tensor::cat(
            &[
                topology.pooled_embedding.shallow_clone(),
                geometry.pooled_embedding.shallow_clone(),
                pocket.pooled_embedding.shallow_clone(),
                affinity_condition_features(topology, geometry, pocket),
            ],
            -1,
        )
        .apply_probe(&self.affinity_head)
        .squeeze_dim(-1);
        let affinity_prediction = affinity_prediction_with_prior(affinity_residual);

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

fn topology_probe_tokens(encoding: &ModalityEncoding, slots: &SlotEncoding) -> Tensor {
    if encoding.token_embeddings.numel() > 0 {
        encoding.token_embeddings.shallow_clone()
    } else {
        slots.reconstructed_tokens.shallow_clone()
    }
}

fn slot_conditioned_probe_tokens(encoding: &ModalityEncoding, slots: &SlotEncoding) -> Tensor {
    if slots.reconstructed_tokens.numel() > 0 {
        slots.reconstructed_tokens.shallow_clone()
    } else {
        encoding.token_embeddings.shallow_clone()
    }
}

fn topology_pair_features(tokens: &Tensor) -> Tensor {
    let rows = tokens.size().first().copied().unwrap_or(0).max(0);
    if tokens.numel() == 0 || tokens.dim() != 2 || rows == 0 {
        return Tensor::zeros([0, 0, 0], (Kind::Float, tokens.device()));
    }
    let hidden = tokens.size().get(1).copied().unwrap_or(0).max(0);
    let left = tokens.unsqueeze(1).expand([rows, rows, hidden], true);
    let right = tokens.unsqueeze(0).expand([rows, rows, hidden], true);
    let difference = (&left - &right).abs();
    let product = &left * &right;
    Tensor::cat(&[left, right, difference, product], -1)
}

fn affinity_condition_features(
    topology: &ModalityEncoding,
    geometry: &ModalityEncoding,
    pocket: &ModalityEncoding,
) -> Tensor {
    let device = topology.pooled_embedding.device();
    let topology_count = token_count(&topology.token_embeddings);
    let geometry_count = token_count(&geometry.token_embeddings);
    let ligand_count = topology_count.max(geometry_count) as f32;
    let pocket_count = token_count(&pocket.token_embeddings) as f32;
    let total_count = (ligand_count + pocket_count).max(1.0);
    let count_features = Tensor::from_slice(&[
        (ligand_count + 1.0).ln() / 5.0,
        (pocket_count + 1.0).ln() / 7.0,
        ligand_count / total_count,
        ((pocket_count + 1.0).ln() - (ligand_count + 1.0).ln()) / 7.0,
    ])
    .to_device(device);
    Tensor::cat(
        &[
            count_features,
            normalized_embedding_rms(&topology.pooled_embedding),
            normalized_embedding_rms(&geometry.pooled_embedding),
            normalized_embedding_rms(&pocket.pooled_embedding),
        ],
        0,
    )
}

fn token_count(tokens: &Tensor) -> i64 {
    if tokens.numel() == 0 || tokens.dim() == 0 {
        0
    } else {
        tokens.size().first().copied().unwrap_or(0).max(0)
    }
}

fn normalized_embedding_rms(embedding: &Tensor) -> Tensor {
    if embedding.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, embedding.device()))
    } else {
        embedding
            .to_kind(Kind::Float)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
            .sqrt()
            .unsqueeze(0)
            .clamp(0.0, 10.0)
    }
}

fn geometry_distance_size_prior(atom_count: i64) -> Tensor {
    if atom_count <= 0 {
        return Tensor::zeros([0], (tch::Kind::Float, tch::Device::Cpu));
    }
    let count = atom_count.max(1) as f64;
    let prior = GEOMETRY_DISTANCE_PRIOR_INTERCEPT_ANGSTROM
        + GEOMETRY_DISTANCE_PRIOR_SQRT_ATOM_SLOPE * count.sqrt();
    Tensor::from_slice(&[prior as f32]).expand([atom_count], true)
}

fn topology_adjacency_logits_with_sparse_prior(pair_logits: Tensor) -> Tensor {
    if pair_logits.numel() == 0 || pair_logits.dim() != 2 {
        return pair_logits;
    }
    let rows = pair_logits.size().first().copied().unwrap_or(0).max(0);
    let cols = pair_logits.size().get(1).copied().unwrap_or(0).max(0);
    if rows == 0 || rows != cols {
        return (pair_logits * TOPOLOGY_EDGE_RESIDUAL_TEMPERATURE).tanh()
            * TOPOLOGY_EDGE_RESIDUAL_RANGE_LOGIT
            + topology_edge_logit_prior(rows.max(cols));
    }
    let symmetrized =
        (&pair_logits + pair_logits.transpose(0, 1)) * (0.5 * TOPOLOGY_EDGE_RESIDUAL_TEMPERATURE);
    let symmetrized =
        symmetrized.tanh() * TOPOLOGY_EDGE_RESIDUAL_RANGE_LOGIT + topology_edge_logit_prior(rows);
    let eye = Tensor::eye(rows, (Kind::Float, pair_logits.device()));
    let off_diagonal = Tensor::ones([rows, rows], (Kind::Float, pair_logits.device())) - &eye;
    symmetrized * off_diagonal + eye * TOPOLOGY_SELF_EDGE_LOGIT
}

fn topology_edge_logit_prior(atom_count: i64) -> f64 {
    let denominator = (atom_count - 1).max(1) as f64;
    let probability = (TOPOLOGY_EXPECTED_DIRECTED_DEGREE / denominator)
        .clamp(TOPOLOGY_EDGE_PROBABILITY_MIN, TOPOLOGY_EDGE_PROBABILITY_MAX);
    (probability / (1.0 - probability)).ln()
}

fn sparse_role_logits(raw_logits: Tensor) -> Tensor {
    (raw_logits * SPARSE_ROLE_RESIDUAL_TEMPERATURE).tanh() * SPARSE_ROLE_RESIDUAL_RANGE_LOGIT
        + SPARSE_ROLE_LOGIT_PRIOR
}

fn sparse_pocket_feature_predictions(raw_predictions: Tensor) -> Tensor {
    (raw_predictions * POCKET_FEATURE_RESIDUAL_TEMPERATURE + POCKET_FEATURE_LOGIT_PRIOR).sigmoid()
}

fn affinity_prediction_with_prior(raw_prediction: Tensor) -> Tensor {
    (raw_prediction * AFFINITY_RESIDUAL_TEMPERATURE).tanh() * AFFINITY_RESIDUAL_RANGE_KCAL_MOL
        + AFFINITY_PRIOR_KCAL_MOL
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

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn topology_probe_tokens_prefer_encoder_tokens() {
        let encoding = ModalityEncoding {
            token_embeddings: Tensor::full([2, 4], 2.0, (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        };
        let slots = SlotEncoding {
            slots: Tensor::zeros([2, 4], (Kind::Float, Device::Cpu)),
            slot_weights: Tensor::ones([2], (Kind::Float, Device::Cpu)) / 2.0,
            token_assignments: Tensor::ones([2, 2], (Kind::Float, Device::Cpu)) / 2.0,
            slot_activation_logits: Tensor::zeros([2], (Kind::Float, Device::Cpu)),
            slot_activations: Tensor::ones([2], (Kind::Float, Device::Cpu)),
            active_slot_mask: Tensor::ones([2], (Kind::Float, Device::Cpu)),
            active_slot_count: 2.0,
            reconstructed_tokens: Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
        };

        let selected = topology_probe_tokens(&encoding, &slots);

        assert_eq!(selected.sum(Kind::Float).double_value(&[]), 16.0);
    }

    #[test]
    fn slot_conditioned_probe_tokens_prefer_reconstructions() {
        let encoding = ModalityEncoding {
            token_embeddings: Tensor::zeros([2, 4], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        };
        let slots = SlotEncoding {
            slots: Tensor::zeros([2, 4], (Kind::Float, Device::Cpu)),
            slot_weights: Tensor::ones([2], (Kind::Float, Device::Cpu)) / 2.0,
            token_assignments: Tensor::ones([2, 2], (Kind::Float, Device::Cpu)) / 2.0,
            slot_activation_logits: Tensor::zeros([2], (Kind::Float, Device::Cpu)),
            slot_activations: Tensor::ones([2], (Kind::Float, Device::Cpu)),
            active_slot_mask: Tensor::ones([2], (Kind::Float, Device::Cpu)),
            active_slot_count: 2.0,
            reconstructed_tokens: Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
        };

        let selected = slot_conditioned_probe_tokens(&encoding, &slots);

        assert_eq!(selected.sum(Kind::Float).double_value(&[]), 8.0);
    }

    #[test]
    fn topology_pair_features_include_relation_terms() {
        let tokens = Tensor::from_slice(&[
            1.0_f32, 2.0, //
            3.0, 5.0,
        ])
        .reshape([2, 2]);

        let features = topology_pair_features(&tokens);

        assert_eq!(features.size(), vec![2, 2, 8]);
        assert_eq!(features.double_value(&[0, 1, 0]), 1.0);
        assert_eq!(features.double_value(&[0, 1, 2]), 3.0);
        assert_eq!(features.double_value(&[0, 1, 4]), 2.0);
        assert_eq!(features.double_value(&[0, 1, 6]), 3.0);
    }

    #[test]
    fn affinity_condition_features_include_scaled_counts_and_norms() {
        let topology = ModalityEncoding {
            token_embeddings: Tensor::zeros([4, 3], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::ones([3], (Kind::Float, Device::Cpu)),
        };
        let geometry = ModalityEncoding {
            token_embeddings: Tensor::zeros([5, 3], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::ones([3], (Kind::Float, Device::Cpu)) * 2.0,
        };
        let pocket = ModalityEncoding {
            token_embeddings: Tensor::zeros([10, 3], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::ones([3], (Kind::Float, Device::Cpu)) * 3.0,
        };

        let features = affinity_condition_features(&topology, &geometry, &pocket);

        assert_eq!(features.size(), vec![AFFINITY_DESCRIPTOR_DIM]);
        assert!(features.double_value(&[0]) > 0.0);
        assert!(features.double_value(&[1]) > 0.0);
        assert_eq!(features.double_value(&[4]), 1.0);
        assert_eq!(features.double_value(&[5]), 2.0);
        assert_eq!(features.double_value(&[6]), 3.0);
        for index in 0..AFFINITY_DESCRIPTOR_DIM {
            assert!(features.double_value(&[index]).is_finite());
        }
    }

    #[test]
    fn geometry_distance_probe_predictions_are_positive_and_scaled() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let probes = SemanticProbeHeads::new(&var_store.root(), 4, 3);
        let encoding = ModalityEncoding {
            token_embeddings: Tensor::zeros([2, 4], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        };
        let slots = SlotEncoding {
            slots: Tensor::zeros([2, 4], (Kind::Float, Device::Cpu)),
            slot_weights: Tensor::ones([2], (Kind::Float, Device::Cpu)) / 2.0,
            token_assignments: Tensor::ones([2, 2], (Kind::Float, Device::Cpu)) / 2.0,
            slot_activation_logits: Tensor::zeros([2], (Kind::Float, Device::Cpu)),
            slot_activations: Tensor::ones([2], (Kind::Float, Device::Cpu)),
            active_slot_mask: Tensor::ones([2], (Kind::Float, Device::Cpu)),
            active_slot_count: 2.0,
            reconstructed_tokens: Tensor::zeros([2, 4], (Kind::Float, Device::Cpu)),
        };

        let outputs = probes.forward(&encoding, &encoding, &encoding, &slots, &slots, &slots);
        let min_prediction = outputs
            .geometry_distance_predictions
            .min()
            .double_value(&[]);
        let mean_prediction = outputs
            .geometry_distance_predictions
            .mean(Kind::Float)
            .double_value(&[]);

        assert!(min_prediction > 0.0);
        assert!(mean_prediction > 0.0);
        assert!(mean_prediction < 10.0);
    }

    #[test]
    fn sparse_probe_priors_match_chemistry_target_scales() {
        let pair_logits = Tensor::zeros([3, 3], (Kind::Float, Device::Cpu));
        let topology_logits = topology_adjacency_logits_with_sparse_prior(pair_logits);
        assert_eq!(topology_logits.size(), vec![3, 3]);
        assert!(topology_logits.double_value(&[0, 0]) < -10.0);
        assert!(topology_logits.double_value(&[0, 1]) < 0.0);
        assert_eq!(
            topology_logits.double_value(&[0, 1]),
            topology_logits.double_value(&[1, 0])
        );

        let role_logits = sparse_role_logits(Tensor::zeros(
            [2, CHEMISTRY_ROLE_FEATURE_DIM],
            (Kind::Float, Device::Cpu),
        ));
        assert!(role_logits.mean(Kind::Float).double_value(&[]) < -1.0);

        let pocket_features =
            sparse_pocket_feature_predictions(Tensor::zeros([2, 3], (Kind::Float, Device::Cpu)));
        let pocket_mean = pocket_features.mean(Kind::Float).double_value(&[]);
        assert!(pocket_mean > 0.0);
        assert!(pocket_mean < 0.1);

        let affinity =
            affinity_prediction_with_prior(Tensor::zeros([], (Kind::Float, Device::Cpu)));
        assert_eq!(affinity.double_value(&[]), AFFINITY_PRIOR_KCAL_MOL);
    }
}
