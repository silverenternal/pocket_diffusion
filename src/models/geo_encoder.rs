//! Geometry encoder with configurable ligand geometry operators.
//!
//! The default operator augments centroid-centered coordinates with aggregated
//! pair-distance kernels. A stronger local-frame variant projects coordinates and
//! pair directions into a deterministic ligand-local frame before aggregation.
//! The local-frame path is invariant to global translation and proper rotations
//! for non-degenerate anchor atoms, while raw coordinate projection remains
//! selectable as the conservative baseline.

use tch::{nn, Kind, Tensor};

use super::{BatchedModalityEncoding, Encoder, GeometryEncoder, ModalityEncoding};
use crate::config::{GeometryEncoderConfig, GeometryOperatorKind};
use crate::data::GeometryFeatures;

/// Geometry encoder that projects coordinates and optional pair-distance kernels.
#[derive(Debug)]
pub struct GeometryEncoderImpl {
    coord_projection: nn::Linear,
    operator_projection: nn::Linear,
    output_projection: nn::Linear,
    config: GeometryEncoderConfig,
}

impl GeometryEncoderImpl {
    /// Create a geometry encoder for coordinate-based features.
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        Self::new_with_config(vs, hidden_dim, &GeometryEncoderConfig::default())
    }

    /// Create a geometry encoder with an explicit operator-family config.
    pub fn new_with_config(vs: &nn::Path, hidden_dim: i64, config: &GeometryEncoderConfig) -> Self {
        let coord_projection = nn::linear(vs / "coord_proj", 3, hidden_dim, Default::default());
        let operator_projection = nn::linear(
            vs / "operator_proj",
            operator_feature_dim(config),
            hidden_dim,
            Default::default(),
        );
        let output_projection = nn::linear(
            vs / "out_proj",
            hidden_dim * 2 + 1,
            hidden_dim,
            Default::default(),
        );
        Self {
            coord_projection,
            operator_projection,
            output_projection,
            config: config.clone(),
        }
    }

    /// Encode padded geometry tensors without iterating over examples.
    pub(crate) fn encode_batch(
        &self,
        coords: &Tensor,
        pairwise_distances: &Tensor,
        mask: &Tensor,
    ) -> BatchedModalityEncoding {
        let coord_inputs = self.coordinate_inputs_batch(coords, mask);
        let coord_hidden = coord_inputs.apply(&self.coord_projection).relu();
        let pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2);
        let denom = pair_mask
            .sum_dim_intlist([2].as_slice(), true, Kind::Float)
            .clamp_min(1.0);
        let mean_distances =
            (pairwise_distances * &pair_mask).sum_dim_intlist([2].as_slice(), true, Kind::Float)
                / denom;
        let operator_hidden =
            self.operator_hidden_batch(coords, pairwise_distances, &pair_mask, mask);
        let token_embeddings = Tensor::cat(&[coord_hidden, mean_distances, operator_hidden], -1)
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

    fn coordinate_inputs_single(&self, coords: &Tensor) -> Tensor {
        match self.config.operator {
            GeometryOperatorKind::LocalFramePairMessage => local_frame_coords_single(coords),
            GeometryOperatorKind::RawCoordinateProjection
            | GeometryOperatorKind::PairDistanceKernel => center_coords(coords),
        }
    }

    fn coordinate_inputs_batch(&self, coords: &Tensor, mask: &Tensor) -> Tensor {
        match self.config.operator {
            GeometryOperatorKind::LocalFramePairMessage => local_frame_coords_batch(coords, mask),
            GeometryOperatorKind::RawCoordinateProjection
            | GeometryOperatorKind::PairDistanceKernel => center_batched_coords(coords, mask),
        }
    }

    fn operator_hidden_single(&self, coords: &Tensor, pairwise_distances: &Tensor) -> Tensor {
        let atom_count = pairwise_distances.size().first().copied().unwrap_or(0);
        match self.config.operator {
            GeometryOperatorKind::RawCoordinateProjection => Tensor::zeros(
                [atom_count, self.coord_projection.ws.size()[0]],
                (Kind::Float, pairwise_distances.device()),
            ),
            GeometryOperatorKind::PairDistanceKernel => {
                let pair_mask = pairwise_distances.gt(0.0).to_kind(Kind::Float);
                let kernels =
                    distance_kernel_features(pairwise_distances, &pair_mask, &self.config);
                kernels.apply(&self.operator_projection).relu() * self.config.residual_scale
            }
            GeometryOperatorKind::LocalFramePairMessage => {
                let pair_mask = pairwise_distances.gt(0.0).to_kind(Kind::Float);
                let features = local_frame_pair_features_single(
                    coords,
                    pairwise_distances,
                    &pair_mask,
                    &self.config,
                );
                features.apply(&self.operator_projection).relu() * self.config.residual_scale
            }
        }
    }

    fn operator_hidden_batch(
        &self,
        coords: &Tensor,
        pairwise_distances: &Tensor,
        pair_mask: &Tensor,
        mask: &Tensor,
    ) -> Tensor {
        let size = pairwise_distances.size();
        let batch_size = size.first().copied().unwrap_or(0);
        let atom_count = size.get(1).copied().unwrap_or(0);
        match self.config.operator {
            GeometryOperatorKind::RawCoordinateProjection => Tensor::zeros(
                [batch_size, atom_count, self.coord_projection.ws.size()[0]],
                (Kind::Float, pairwise_distances.device()),
            ),
            GeometryOperatorKind::PairDistanceKernel => {
                let non_self_pair_mask =
                    pair_mask * pairwise_distances.gt(0.0).to_kind(Kind::Float);
                let kernels =
                    distance_kernel_features(pairwise_distances, &non_self_pair_mask, &self.config);
                kernels.apply(&self.operator_projection).relu() * self.config.residual_scale
            }
            GeometryOperatorKind::LocalFramePairMessage => {
                let non_self_pair_mask =
                    pair_mask * pairwise_distances.gt(0.0).to_kind(Kind::Float);
                let features = local_frame_pair_features_batch(
                    coords,
                    pairwise_distances,
                    &non_self_pair_mask,
                    mask,
                    &self.config,
                );
                features.apply(&self.operator_projection).relu() * self.config.residual_scale
            }
        }
    }
}

impl Encoder<GeometryFeatures, ModalityEncoding> for GeometryEncoderImpl {
    fn encode(&self, input: &GeometryFeatures) -> ModalityEncoding {
        let coord_inputs = self.coordinate_inputs_single(&input.coords);
        let coord_hidden = coord_inputs.apply(&self.coord_projection).relu();
        let mean_distances = if input.pairwise_distances.size()[0] == 0 {
            Tensor::zeros([0, 1], (Kind::Float, input.coords.device()))
        } else {
            input
                .pairwise_distances
                .mean_dim([1].as_slice(), true, Kind::Float)
        };
        let operator_hidden = self.operator_hidden_single(&input.coords, &input.pairwise_distances);
        let token_embeddings = Tensor::cat(&[coord_hidden, mean_distances, operator_hidden], 1)
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

fn center_coords(coords: &Tensor) -> Tensor {
    if coords.size().first().copied().unwrap_or(0) == 0 {
        return coords.shallow_clone();
    }
    let centroid = coords.mean_dim([0].as_slice(), true, Kind::Float);
    coords - centroid
}

fn center_batched_coords(coords: &Tensor, mask: &Tensor) -> Tensor {
    let denom = mask
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .clamp_min(1.0)
        .unsqueeze(-1);
    let centroid =
        (coords * mask.unsqueeze(-1)).sum_dim_intlist([1].as_slice(), true, Kind::Float) / denom;
    (coords - centroid) * mask.unsqueeze(-1)
}

fn operator_feature_dim(config: &GeometryEncoderConfig) -> i64 {
    match config.operator {
        GeometryOperatorKind::LocalFramePairMessage => config.distance_kernel_count + 3,
        GeometryOperatorKind::RawCoordinateProjection
        | GeometryOperatorKind::PairDistanceKernel => config.distance_kernel_count,
    }
}

fn local_frame_coords_single(coords: &Tensor) -> Tensor {
    let atom_count = coords.size().first().copied().unwrap_or(0);
    if atom_count == 0 {
        return coords.shallow_clone();
    }
    let mask = Tensor::ones([1, atom_count], (Kind::Float, coords.device()));
    local_frame_coords_batch(&coords.unsqueeze(0), &mask).squeeze_dim(0)
}

fn local_frame_coords_batch(coords: &Tensor, mask: &Tensor) -> Tensor {
    let centered = center_batched_coords(coords, mask);
    let frame = local_frame_axes_batch(&centered);
    centered.matmul(&frame.transpose(1, 2)) * mask.unsqueeze(-1)
}

fn local_frame_pair_features_single(
    coords: &Tensor,
    pairwise_distances: &Tensor,
    pair_mask: &Tensor,
    config: &GeometryEncoderConfig,
) -> Tensor {
    let atom_count = coords.size().first().copied().unwrap_or(0);
    if atom_count == 0 {
        return Tensor::zeros(
            [0, operator_feature_dim(config)],
            (Kind::Float, coords.device()),
        );
    }
    let mask = Tensor::ones([1, atom_count], (Kind::Float, coords.device()));
    local_frame_pair_features_batch(
        &coords.unsqueeze(0),
        &pairwise_distances.unsqueeze(0),
        &pair_mask.unsqueeze(0),
        &mask,
        config,
    )
    .squeeze_dim(0)
}

fn local_frame_pair_features_batch(
    coords: &Tensor,
    pairwise_distances: &Tensor,
    pair_mask: &Tensor,
    mask: &Tensor,
    config: &GeometryEncoderConfig,
) -> Tensor {
    let radial = distance_kernel_features(pairwise_distances, pair_mask, config);
    let centered = center_batched_coords(coords, mask);
    let frame = local_frame_axes_batch(&centered);
    let local_frame = frame.transpose(1, 2).unsqueeze(1);
    let relative = centered.unsqueeze(1) - centered.unsqueeze(2);
    let local_relative = relative.matmul(&local_frame);
    let directions = local_relative / pairwise_distances.clamp_min(1.0e-6).unsqueeze(-1);
    let denom = pair_mask
        .sum_dim_intlist([2].as_slice(), true, Kind::Float)
        .clamp_min(1.0);
    let directional =
        (directions * pair_mask.unsqueeze(-1)).sum_dim_intlist([2].as_slice(), false, Kind::Float)
            / denom;
    Tensor::cat(&[radial, directional], -1)
}

fn local_frame_axes_batch(centered_coords: &Tensor) -> Tensor {
    let size = centered_coords.size();
    let batch_size = size.first().copied().unwrap_or(0);
    let atom_count = size.get(1).copied().unwrap_or(0);
    let device = centered_coords.device();
    let fallback_x = Tensor::from_slice(&[1.0_f32, 0.0, 0.0])
        .to_device(device)
        .reshape([1, 3])
        .expand([batch_size, 3], true);
    let fallback_y = Tensor::from_slice(&[0.0_f32, 1.0, 0.0])
        .to_device(device)
        .reshape([1, 3])
        .expand([batch_size, 3], true);

    let first = if atom_count > 0 {
        centered_coords.select(1, 0)
    } else {
        Tensor::zeros([batch_size, 3], (Kind::Float, device))
    };
    let e1 = normalize_with_fallback_batch(&first, &fallback_x);

    let second = if atom_count > 1 {
        centered_coords.select(1, 1)
    } else {
        fallback_y.shallow_clone()
    };
    let second_orthogonal = orthogonal_component(&second, &e1);
    let y_orthogonal = orthogonal_component(&fallback_y, &e1);
    let x_orthogonal = orthogonal_component(&fallback_x, &e1);
    let y_norm = vector_norm_batch(&y_orthogonal);
    let use_y = y_norm.gt(1.0e-6).to_kind(Kind::Float);
    let fallback_orthogonal =
        &y_orthogonal * &use_y + &x_orthogonal * (Tensor::ones_like(&use_y) - &use_y);
    let e2 = normalize_with_fallback_batch(&second_orthogonal, &fallback_orthogonal);
    let e3 = cross_3d_batch(&e1, &e2);

    Tensor::stack(&[e1, e2, e3], 1)
}

fn orthogonal_component(vector: &Tensor, basis: &Tensor) -> Tensor {
    vector - (vector * basis).sum_dim_intlist([1].as_slice(), true, Kind::Float) * basis
}

fn normalize_with_fallback_batch(vector: &Tensor, fallback: &Tensor) -> Tensor {
    let norm = vector_norm_batch(vector);
    let active = norm.gt(1.0e-6).to_kind(Kind::Float);
    let candidate = vector * &active + fallback * (Tensor::ones_like(&active) - &active);
    &candidate / vector_norm_batch(&candidate).clamp_min(1.0e-6)
}

fn vector_norm_batch(vector: &Tensor) -> Tensor {
    vector
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .sqrt()
}

fn cross_3d_batch(left: &Tensor, right: &Tensor) -> Tensor {
    let lx = left.narrow(1, 0, 1);
    let ly = left.narrow(1, 1, 1);
    let lz = left.narrow(1, 2, 1);
    let rx = right.narrow(1, 0, 1);
    let ry = right.narrow(1, 1, 1);
    let rz = right.narrow(1, 2, 1);
    Tensor::cat(
        &[
            &ly * &rz - &lz * &ry,
            &lz * &rx - &lx * &rz,
            &lx * &ry - &ly * &rx,
        ],
        1,
    )
}

fn distance_kernel_features(
    pairwise_distances: &Tensor,
    pair_mask: &Tensor,
    config: &GeometryEncoderConfig,
) -> Tensor {
    let centers = distance_kernel_centers(config, pairwise_distances.device());
    let radial = ((pairwise_distances.unsqueeze(-1) - centers).pow_tensor_scalar(2.0)
        * -config.distance_kernel_gamma)
        .exp();
    let masked = radial * pair_mask.unsqueeze(-1);
    let denom = pair_mask
        .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
        .clamp_min(1.0)
        .unsqueeze(-1);
    masked.sum_dim_intlist([-2].as_slice(), false, Kind::Float) / denom.squeeze_dim(-1)
}

fn distance_kernel_centers(config: &GeometryEncoderConfig, device: tch::Device) -> Tensor {
    let count = config.distance_kernel_count.max(1) as usize;
    let step = if count == 1 {
        0.0
    } else {
        config.distance_kernel_max_distance as f32 / (count - 1) as f32
    };
    let centers = (0..count).map(|idx| idx as f32 * step).collect::<Vec<_>>();
    Tensor::from_slice(&centers).to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    fn pairwise(coords: &Tensor) -> Tensor {
        let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
        diffs
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([2].as_slice(), false, Kind::Float)
            .sqrt()
    }

    #[test]
    fn pair_distance_kernel_preserves_single_and_batch_shapes() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = GeometryEncoderConfig::default();
        let encoder = GeometryEncoderImpl::new_with_config(&var_store.root(), 12, &config);
        let coords =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).reshape([3, 3]);
        let distances = pairwise(&coords);
        let single = encoder.encode(&GeometryFeatures {
            coords: coords.shallow_clone(),
            pairwise_distances: distances.shallow_clone(),
        });
        assert_eq!(single.token_embeddings.size(), vec![3, 12]);
        assert_eq!(single.pooled_embedding.size(), vec![12]);

        let mask = Tensor::from_slice(&[1.0_f32, 1.0, 1.0]).reshape([1, 3]);
        let batch = encoder.encode_batch(&coords.unsqueeze(0), &distances.unsqueeze(0), &mask);
        assert_eq!(batch.token_embeddings.size(), vec![1, 3, 12]);
        assert_eq!(batch.pooled_embedding.size(), vec![1, 12]);
    }

    #[test]
    fn raw_geometry_operator_remains_selectable() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = GeometryEncoderConfig {
            operator: GeometryOperatorKind::RawCoordinateProjection,
            ..GeometryEncoderConfig::default()
        };
        let encoder = GeometryEncoderImpl::new_with_config(&var_store.root(), 8, &config);
        let coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.5, 0.0, 0.0]).reshape([2, 3]);
        let output = encoder.encode(&GeometryFeatures {
            coords: coords.shallow_clone(),
            pairwise_distances: pairwise(&coords),
        });
        assert_eq!(output.token_embeddings.size(), vec![2, 8]);
    }

    #[test]
    fn local_frame_pair_message_preserves_rotation_and_translation() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = GeometryEncoderConfig {
            operator: GeometryOperatorKind::LocalFramePairMessage,
            ..GeometryEncoderConfig::default()
        };
        let encoder = GeometryEncoderImpl::new_with_config(&var_store.root(), 10, &config);
        let coords =
            Tensor::from_slice(&[1.0_f32, 0.0, 0.0, -0.2, 1.4, 0.1, 0.3, 0.5, 1.2]).reshape([3, 3]);
        let rotation =
            Tensor::from_slice(&[0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([3, 3]);
        let shifted =
            coords.matmul(&rotation) + Tensor::from_slice(&[4.0_f32, -3.0, 2.0]).reshape([1, 3]);

        let base = encoder.encode(&GeometryFeatures {
            coords: coords.shallow_clone(),
            pairwise_distances: pairwise(&coords),
        });
        let transformed = encoder.encode(&GeometryFeatures {
            coords: shifted.shallow_clone(),
            pairwise_distances: pairwise(&shifted),
        });
        let token_delta = (base.token_embeddings - transformed.token_embeddings)
            .abs()
            .max()
            .double_value(&[]);
        let pooled_delta = (base.pooled_embedding - transformed.pooled_embedding)
            .abs()
            .max()
            .double_value(&[]);
        assert!(token_delta < 1e-5, "token delta was {token_delta}");
        assert!(pooled_delta < 1e-5, "pooled delta was {pooled_delta}");

        let mask = Tensor::from_slice(&[1.0_f32, 1.0, 1.0]).reshape([1, 3]);
        let batched =
            encoder.encode_batch(&coords.unsqueeze(0), &pairwise(&coords).unsqueeze(0), &mask);
        assert_eq!(batched.token_embeddings.size(), vec![1, 3, 10]);
        assert_eq!(batched.pooled_embedding.size(), vec![1, 10]);
    }

    #[test]
    fn distance_kernel_features_are_translation_invariant() {
        let config = GeometryEncoderConfig::default();
        let coords =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).reshape([3, 3]);
        let shifted = &coords + Tensor::from_slice(&[4.0_f32, -3.0, 2.0]).reshape([1, 3]);
        let base_distances = pairwise(&coords);
        let shifted_distances = pairwise(&shifted);
        let mask = base_distances.gt(0.0).to_kind(Kind::Float);
        let base = distance_kernel_features(&base_distances, &mask, &config);
        let shifted_mask = shifted_distances.gt(0.0).to_kind(Kind::Float);
        let next = distance_kernel_features(&shifted_distances, &shifted_mask, &config);
        let max_delta = (base - next).abs().max().double_value(&[]);
        assert!(max_delta < 1e-5);
    }
}
