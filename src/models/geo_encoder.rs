//! Geometry encoder skeleton with distance-aware coordinate processing.

use tch::{nn, Kind, Tensor};

use super::{BatchedModalityEncoding, Encoder, GeometryEncoder, ModalityEncoding};
use crate::data::GeometryFeatures;

/// Minimal geometry encoder that projects coordinates and local distance statistics.
#[derive(Debug)]
pub struct GeometryEncoderImpl {
    coord_projection: nn::Linear,
    output_projection: nn::Linear,
}

impl GeometryEncoderImpl {
    /// Create a geometry encoder for coordinate-based features.
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        let coord_projection = nn::linear(vs / "coord_proj", 3, hidden_dim, Default::default());
        let output_projection = nn::linear(
            vs / "out_proj",
            hidden_dim + 1,
            hidden_dim,
            Default::default(),
        );
        Self {
            coord_projection,
            output_projection,
        }
    }

    /// Encode padded geometry tensors without iterating over examples.
    pub(crate) fn encode_batch(
        &self,
        coords: &Tensor,
        pairwise_distances: &Tensor,
        mask: &Tensor,
    ) -> BatchedModalityEncoding {
        let coord_hidden = coords.apply(&self.coord_projection).relu();
        let pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2);
        let denom = pair_mask
            .sum_dim_intlist([2].as_slice(), true, Kind::Float)
            .clamp_min(1.0);
        let mean_distances =
            (pairwise_distances * &pair_mask).sum_dim_intlist([2].as_slice(), true, Kind::Float)
                / denom;
        let token_embeddings = Tensor::cat(&[coord_hidden, mean_distances], -1)
            .apply(&self.output_projection)
            .relu()
            * mask.unsqueeze(-1);
        let pooled_denom = mask
            .sum_dim_intlist([1].as_slice(), true, Kind::Float)
            .clamp_min(1.0);
        let pooled_embedding =
            token_embeddings.sum_dim_intlist([1].as_slice(), false, Kind::Float) / pooled_denom;

        BatchedModalityEncoding {
            token_embeddings,
            token_mask: mask.shallow_clone(),
            pooled_embedding,
        }
    }
}

impl Encoder<GeometryFeatures, ModalityEncoding> for GeometryEncoderImpl {
    fn encode(&self, input: &GeometryFeatures) -> ModalityEncoding {
        let coord_hidden = input.coords.apply(&self.coord_projection).relu();
        let mean_distances = if input.pairwise_distances.size()[0] == 0 {
            Tensor::zeros([0, 1], (Kind::Float, input.coords.device()))
        } else {
            input
                .pairwise_distances
                .mean_dim([1].as_slice(), true, Kind::Float)
        };
        let token_embeddings = Tensor::cat(&[coord_hidden, mean_distances], 1)
            .apply(&self.output_projection)
            .relu();
        let pooled_embedding = if token_embeddings.size()[0] == 0 {
            Tensor::zeros(
                [self.output_projection.ws.size()[0]],
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

impl GeometryEncoder for GeometryEncoderImpl {}
