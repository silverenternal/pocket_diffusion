//! Pocket encoder skeleton with separate context processing.

use tch::{nn, Kind, Tensor};

use super::{BatchedModalityEncoding, Encoder, ModalityEncoding, PocketEncoder};
use crate::data::PocketFeatures;

/// Minimal pocket encoder that combines per-atom and pooled context features.
#[derive(Debug)]
pub struct PocketEncoderImpl {
    atom_projection: nn::Linear,
    pooled_projection: nn::Linear,
}

impl PocketEncoderImpl {
    /// Create a pocket encoder for local context features.
    pub fn new(vs: &nn::Path, pocket_feature_dim: i64, hidden_dim: i64) -> Self {
        let atom_projection = nn::linear(
            vs / "atom_proj",
            pocket_feature_dim + 3,
            hidden_dim,
            Default::default(),
        );
        let pooled_projection = nn::linear(
            vs / "pooled_proj",
            pocket_feature_dim,
            hidden_dim,
            Default::default(),
        );
        Self {
            atom_projection,
            pooled_projection,
        }
    }

    /// Encode padded pocket tensors without iterating over examples.
    pub(crate) fn encode_batch(
        &self,
        atom_features: &Tensor,
        coords: &Tensor,
        pooled_features: &Tensor,
        mask: &Tensor,
    ) -> BatchedModalityEncoding {
        let local_context =
            Tensor::cat(&[atom_features.shallow_clone(), coords.shallow_clone()], -1)
                .apply(&self.atom_projection)
                .relu()
                * mask.unsqueeze(-1);
        let pooled_embedding = pooled_features.apply(&self.pooled_projection).relu();

        BatchedModalityEncoding {
            token_embeddings: local_context,
            token_mask: mask.shallow_clone(),
            pooled_embedding,
        }
    }
}

impl Encoder<PocketFeatures, ModalityEncoding> for PocketEncoderImpl {
    fn encode(&self, input: &PocketFeatures) -> ModalityEncoding {
        let local_context = Tensor::cat(
            &[
                input.atom_features.shallow_clone(),
                input.coords.shallow_clone(),
            ],
            1,
        )
        .apply(&self.atom_projection)
        .relu();
        let pooled_embedding = input
            .pooled_features
            .unsqueeze(0)
            .apply(&self.pooled_projection)
            .relu()
            .squeeze_dim(0);
        let token_embeddings = if local_context.size()[0] == 0 {
            Tensor::zeros(
                [0, self.atom_projection.ws.size()[0]],
                (Kind::Float, input.coords.device()),
            )
        } else {
            local_context
        };

        ModalityEncoding {
            token_embeddings,
            pooled_embedding,
        }
    }
}

impl PocketEncoder for PocketEncoderImpl {}
