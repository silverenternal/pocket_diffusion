//! Topology encoder skeleton with explicit graph-aware inputs.

use tch::{nn, Kind, Tensor};

use super::{BatchedModalityEncoding, Encoder, ModalityEncoding, TopologyEncoder};
use crate::{
    config::{TopologyEncoderConfig, TopologyEncoderKind},
    data::TopologyFeatures,
};

/// Topology encoder that can run either the legacy degree projection baseline
/// or residual graph message passing over dense ligand adjacency.
#[derive(Debug)]
pub struct TopologyEncoderImpl {
    atom_embedding: nn::Embedding,
    bond_type_embedding: nn::Embedding,
    atom_projection: nn::Linear,
    message_layers: Vec<nn::Linear>,
    kind: TopologyEncoderKind,
    residual_scale: f64,
    normalize_messages: bool,
    bond_type_vocab_size: i64,
}

impl TopologyEncoderImpl {
    /// Create a topology encoder for categorical atom inputs.
    pub fn new(vs: &nn::Path, atom_vocab_size: i64, hidden_dim: i64) -> Self {
        Self::new_with_config(
            vs,
            atom_vocab_size,
            hidden_dim,
            &TopologyEncoderConfig::default(),
        )
    }

    /// Create a topology encoder with explicit ablation controls.
    pub fn new_with_config(
        vs: &nn::Path,
        atom_vocab_size: i64,
        hidden_dim: i64,
        config: &TopologyEncoderConfig,
    ) -> Self {
        let atom_embedding = nn::embedding(
            vs / "atom_embed",
            atom_vocab_size,
            hidden_dim,
            Default::default(),
        );
        let bond_type_embedding = nn::embedding(
            vs / "bond_type_embed",
            config.bond_type_vocab_size,
            hidden_dim,
            Default::default(),
        );
        let atom_projection = nn::linear(
            vs / "atom_proj",
            hidden_dim + 1,
            hidden_dim,
            Default::default(),
        );
        let message_layers = (0..config.message_passing_layers)
            .map(|layer_ix| {
                nn::linear(
                    vs / format!("message_layer_{layer_ix}"),
                    hidden_dim * 3 + 1,
                    hidden_dim,
                    Default::default(),
                )
            })
            .collect();
        Self {
            atom_embedding,
            bond_type_embedding,
            atom_projection,
            message_layers,
            kind: config.kind,
            residual_scale: config.residual_scale,
            normalize_messages: config.normalize_messages,
            bond_type_vocab_size: config.bond_type_vocab_size,
        }
    }

    /// Encode padded topology tensors without iterating over examples.
    pub(crate) fn encode_batch(
        &self,
        atom_types: &Tensor,
        adjacency: &Tensor,
        bond_type_adjacency: &Tensor,
        mask: &Tensor,
    ) -> BatchedModalityEncoding {
        let atom_emb = atom_types.apply(&self.atom_embedding);
        let degree = adjacency.sum_dim_intlist([2].as_slice(), true, Kind::Float);
        let token_embeddings = Tensor::cat(&[atom_emb, degree.shallow_clone()], -1)
            .apply(&self.atom_projection)
            .relu()
            * mask.unsqueeze(-1);
        let token_embeddings = self.apply_message_passing_batch(
            token_embeddings,
            adjacency,
            bond_type_adjacency,
            &degree,
            mask,
        );
        let denom = mask
            .sum_dim_intlist([1].as_slice(), true, Kind::Float)
            .clamp_min(1.0);
        let pooled_embedding =
            token_embeddings.sum_dim_intlist([1].as_slice(), false, Kind::Float) / denom;

        BatchedModalityEncoding {
            token_embeddings,
            token_mask: mask.shallow_clone(),
            pooled_embedding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn topology_message_passing_preserves_batch_shapes_and_masks() {
        let vs = nn::VarStore::new(Device::Cpu);
        let encoder = TopologyEncoderImpl::new_with_config(
            &vs.root(),
            8,
            6,
            &TopologyEncoderConfig {
                kind: TopologyEncoderKind::MessagePassing,
                message_passing_layers: 2,
                residual_scale: 0.25,
                normalize_messages: true,
                bond_type_vocab_size: 8,
            },
        );
        let atom_types = Tensor::from_slice(&[1_i64, 2, 0]).reshape([1, 3]);
        let adjacency = Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .reshape([1, 3, 3]);
        let bond_type_adjacency =
            Tensor::from_slice(&[0_i64, 1, 0, 1, 0, 0, 0, 0, 0]).reshape([1, 3, 3]);
        let mask = Tensor::from_slice(&[1.0_f32, 1.0, 0.0]).reshape([1, 3]);

        let encoded = encoder.encode_batch(&atom_types, &adjacency, &bond_type_adjacency, &mask);

        assert_eq!(encoded.token_embeddings.size(), vec![1, 3, 6]);
        assert_eq!(encoded.pooled_embedding.size(), vec![1, 6]);
        let padded_norm = encoded
            .token_embeddings
            .get(0)
            .get(2)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert_eq!(padded_norm, 0.0);
    }

    #[test]
    fn lightweight_topology_encoder_remains_selectable() {
        let vs = nn::VarStore::new(Device::Cpu);
        let encoder = TopologyEncoderImpl::new_with_config(
            &vs.root(),
            8,
            6,
            &TopologyEncoderConfig {
                kind: TopologyEncoderKind::Lightweight,
                message_passing_layers: 2,
                residual_scale: 0.25,
                normalize_messages: true,
                bond_type_vocab_size: 8,
            },
        );

        assert_eq!(encoder.kind, TopologyEncoderKind::Lightweight);
    }

    #[test]
    fn typed_topology_encoder_uses_bond_type_inputs() {
        let vs = nn::VarStore::new(Device::Cpu);
        let encoder = TopologyEncoderImpl::new_with_config(
            &vs.root(),
            8,
            6,
            &TopologyEncoderConfig {
                kind: TopologyEncoderKind::TypedMessagePassing,
                message_passing_layers: 1,
                residual_scale: 0.5,
                normalize_messages: true,
                bond_type_vocab_size: 8,
            },
        );
        let topology = TopologyFeatures {
            atom_types: Tensor::from_slice(&[1_i64, 2, 3]),
            edge_index: Tensor::from_slice(&[0_i64, 1, 1, 2]).reshape([2, 2]),
            bond_types: Tensor::from_slice(&[1_i64, 2]),
            adjacency: Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .reshape([3, 3]),
            chemistry_roles: crate::data::features::chemistry_role_features_from_atom_types(&[
                crate::types::AtomType::Carbon,
                crate::types::AtomType::Nitrogen,
                crate::types::AtomType::Oxygen,
            ]),
        };
        let mut changed = topology.clone();
        changed.bond_types = Tensor::from_slice(&[3_i64, 3]);

        let encoded = encoder.encode(&topology);
        let changed_encoded = encoder.encode(&changed);
        let diff = (&encoded.token_embeddings - &changed_encoded.token_embeddings)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);

        assert!(
            diff > 1e-8,
            "typed bond ids should affect topology encoding"
        );
    }

    #[test]
    fn typed_topology_encoder_zero_bond_types_fall_back_to_binary_messages() {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let typed = TopologyEncoderImpl::new_with_config(
            &(root.clone() / "typed"),
            8,
            6,
            &TopologyEncoderConfig {
                kind: TopologyEncoderKind::TypedMessagePassing,
                message_passing_layers: 1,
                residual_scale: 0.5,
                normalize_messages: true,
                bond_type_vocab_size: 8,
            },
        );
        let binary = TopologyEncoderImpl::new_with_config(
            &(root / "binary"),
            8,
            6,
            &TopologyEncoderConfig {
                kind: TopologyEncoderKind::MessagePassing,
                message_passing_layers: 1,
                residual_scale: 0.5,
                normalize_messages: true,
                bond_type_vocab_size: 8,
            },
        );
        let hidden = Tensor::ones([3, 6], (Kind::Float, Device::Cpu));
        let adjacency =
            Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).reshape([3, 3]);
        let bond_type_adjacency = Tensor::zeros([3, 3], (Kind::Int64, Device::Cpu));

        let typed_messages =
            typed.neighbor_messages_single(&hidden, &adjacency, &bond_type_adjacency);
        let binary_messages =
            binary.neighbor_messages_single(&hidden, &adjacency, &bond_type_adjacency);

        assert!(
            (&typed_messages - &binary_messages)
                .abs()
                .max()
                .double_value(&[])
                < 1e-8
        );
    }
}

impl Encoder<TopologyFeatures, ModalityEncoding> for TopologyEncoderImpl {
    fn encode(&self, input: &TopologyFeatures) -> ModalityEncoding {
        let atom_emb = input.atom_types.apply(&self.atom_embedding);
        let degree = input
            .adjacency
            .sum_dim_intlist([1].as_slice(), true, Kind::Float);
        let token_embeddings = Tensor::cat(&[atom_emb, degree.shallow_clone()], 1)
            .apply(&self.atom_projection)
            .relu();
        let bond_type_adjacency = dense_bond_type_adjacency(input, input.atom_types.size()[0]);
        let token_embeddings = self.apply_message_passing_single(
            token_embeddings,
            &input.adjacency,
            &bond_type_adjacency,
            &degree,
        );
        let pooled_embedding = if token_embeddings.size()[0] == 0 {
            Tensor::zeros(
                [self.atom_projection.ws.size()[0]],
                (Kind::Float, token_embeddings.device()),
            )
        } else {
            token_embeddings.mean_dim([0].as_slice(), false, Kind::Float)
        };

        ModalityEncoding {
            token_embeddings,
            pooled_embedding,
        }
    }
}

impl TopologyEncoder for TopologyEncoderImpl {}

impl TopologyEncoderImpl {
    fn apply_message_passing_single(
        &self,
        mut hidden: Tensor,
        adjacency: &Tensor,
        bond_type_adjacency: &Tensor,
        degree: &Tensor,
    ) -> Tensor {
        if self.kind == TopologyEncoderKind::Lightweight || self.message_layers.is_empty() {
            return hidden;
        }
        for layer in &self.message_layers {
            let neighbor_messages =
                self.neighbor_messages_single(&hidden, adjacency, bond_type_adjacency);
            let neighbor_messages = if self.normalize_messages {
                neighbor_messages / degree.clamp_min(1.0)
            } else {
                neighbor_messages
            };
            let update = Tensor::cat(
                &[
                    hidden.shallow_clone(),
                    neighbor_messages.shallow_clone(),
                    hidden.shallow_clone() * neighbor_messages,
                    degree.shallow_clone(),
                ],
                1,
            )
            .apply(layer)
            .relu();
            hidden = hidden + update * self.residual_scale;
        }
        hidden
    }

    fn apply_message_passing_batch(
        &self,
        mut hidden: Tensor,
        adjacency: &Tensor,
        bond_type_adjacency: &Tensor,
        degree: &Tensor,
        mask: &Tensor,
    ) -> Tensor {
        if self.kind == TopologyEncoderKind::Lightweight || self.message_layers.is_empty() {
            return hidden;
        }
        let expanded_mask = mask.unsqueeze(-1);
        for layer in &self.message_layers {
            let neighbor_messages =
                self.neighbor_messages_batch(&hidden, adjacency, bond_type_adjacency);
            let neighbor_messages = if self.normalize_messages {
                neighbor_messages / degree.clamp_min(1.0)
            } else {
                neighbor_messages
            };
            let update = Tensor::cat(
                &[
                    hidden.shallow_clone(),
                    neighbor_messages.shallow_clone(),
                    hidden.shallow_clone() * neighbor_messages,
                    degree.shallow_clone(),
                ],
                -1,
            )
            .apply(layer)
            .relu();
            hidden = (hidden + update * self.residual_scale) * &expanded_mask;
        }
        hidden
    }

    fn neighbor_messages_single(
        &self,
        hidden: &Tensor,
        adjacency: &Tensor,
        bond_type_adjacency: &Tensor,
    ) -> Tensor {
        if self.kind != TopologyEncoderKind::TypedMessagePassing {
            return adjacency.matmul(hidden);
        }
        let typed = typed_bond_embeddings(
            bond_type_adjacency,
            &self.bond_type_embedding,
            self.bond_type_vocab_size,
        );
        ((hidden.unsqueeze(0) + typed) * adjacency.unsqueeze(-1)).sum_dim_intlist(
            [1].as_slice(),
            false,
            Kind::Float,
        )
    }

    fn neighbor_messages_batch(
        &self,
        hidden: &Tensor,
        adjacency: &Tensor,
        bond_type_adjacency: &Tensor,
    ) -> Tensor {
        if self.kind != TopologyEncoderKind::TypedMessagePassing {
            return adjacency.matmul(hidden);
        }
        let typed = typed_bond_embeddings(
            bond_type_adjacency,
            &self.bond_type_embedding,
            self.bond_type_vocab_size,
        );
        ((hidden.unsqueeze(1) + typed) * adjacency.unsqueeze(-1)).sum_dim_intlist(
            [2].as_slice(),
            false,
            Kind::Float,
        )
    }
}

fn typed_bond_embeddings(
    bond_type_adjacency: &Tensor,
    embedding: &nn::Embedding,
    bond_type_vocab_size: i64,
) -> Tensor {
    let safe_types = bond_type_adjacency
        .clamp(0, bond_type_vocab_size - 1)
        .to_kind(Kind::Int64);
    let known_mask = safe_types.gt(0).to_kind(Kind::Float).unsqueeze(-1);
    safe_types.apply(embedding) * known_mask
}

fn dense_bond_type_adjacency(input: &TopologyFeatures, atom_count: i64) -> Tensor {
    let device = input.atom_types.device();
    let dense = Tensor::zeros([atom_count, atom_count], (Kind::Int64, device));
    if atom_count <= 0 || input.edge_index.size().len() != 2 {
        return dense;
    }
    let edge_count = input
        .edge_index
        .size()
        .get(1)
        .copied()
        .unwrap_or(0)
        .min(input.bond_types.size().first().copied().unwrap_or(0));
    for edge_ix in 0..edge_count {
        let src = input.edge_index.int64_value(&[0, edge_ix]);
        let dst = input.edge_index.int64_value(&[1, edge_ix]);
        let bond_type = input.bond_types.int64_value(&[edge_ix]).clamp(0, 7);
        if src < 0 || dst < 0 || src >= atom_count || dst >= atom_count || bond_type == 0 {
            continue;
        }
        let _ = dense.get(src).get(dst).fill_(bond_type);
        let _ = dense.get(dst).get(src).fill_(bond_type);
    }
    dense
}
