//! Geometry encoder skeleton with distance-aware coordinate processing.

use tch::{nn, Kind, Tensor};

use super::{Encoder, GeometryEncoder, ModalityEncoding};
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
