//! Topology encoder skeleton with explicit graph-aware inputs.

use tch::{nn, Kind, Tensor};

use super::{Encoder, ModalityEncoding, TopologyEncoder};
use crate::data::TopologyFeatures;

/// Minimal topology encoder that mixes atom embeddings with graph degree signals.
#[derive(Debug)]
pub struct TopologyEncoderImpl {
    atom_embedding: nn::Embedding,
    atom_projection: nn::Linear,
}

impl TopologyEncoderImpl {
    /// Create a topology encoder for categorical atom inputs.
    pub fn new(vs: &nn::Path, atom_vocab_size: i64, hidden_dim: i64) -> Self {
        let atom_embedding = nn::embedding(
            vs / "atom_embed",
            atom_vocab_size,
            hidden_dim,
            Default::default(),
        );
        let atom_projection = nn::linear(
            vs / "atom_proj",
            hidden_dim + 1,
            hidden_dim,
            Default::default(),
        );
        Self {
            atom_embedding,
            atom_projection,
        }
    }
}

impl Encoder<TopologyFeatures, ModalityEncoding> for TopologyEncoderImpl {
    fn encode(&self, input: &TopologyFeatures) -> ModalityEncoding {
        let atom_emb = input.atom_types.apply(&self.atom_embedding);
        let degree = input
            .adjacency
            .sum_dim_intlist([1].as_slice(), true, Kind::Float);
        let token_embeddings = Tensor::cat(&[atom_emb, degree], 1)
            .apply(&self.atom_projection)
            .relu();
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
