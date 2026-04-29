//! Pocket encoder with configurable local context aggregation.
//!
//! The default local-message operator aggregates only inference-available
//! pocket atom/residue features, pocket coordinates, and masks. The stronger
//! ligand-relative local-frame variant uses the same inputs after projecting
//! pocket coordinates and neighbor directions into a deterministic frame around
//! the ligand-centered pocket origin. Neither path consumes decoder targets or
//! training-only ligand labels, keeping context encoding separate from topology
//! and geometry.

use tch::{nn, Kind, Tensor};

use super::{BatchedModalityEncoding, Encoder, ModalityEncoding, PocketEncoder};
use crate::config::{PocketEncoderConfig, PocketEncoderKind};
use crate::data::PocketFeatures;

/// Pocket encoder that combines per-atom features, coordinates, and local context.
#[derive(Debug)]
pub struct PocketEncoderImpl {
    atom_projection: nn::Linear,
    pooled_projection: nn::Linear,
    message_layers: Vec<nn::Linear>,
    config: PocketEncoderConfig,
}

impl PocketEncoderImpl {
    /// Create a pocket encoder for local context features.
    pub fn new(vs: &nn::Path, pocket_feature_dim: i64, hidden_dim: i64) -> Self {
        Self::new_with_config(
            vs,
            pocket_feature_dim,
            hidden_dim,
            &PocketEncoderConfig::default(),
        )
    }

    /// Create a pocket encoder with an explicit local-context config.
    pub fn new_with_config(
        vs: &nn::Path,
        pocket_feature_dim: i64,
        hidden_dim: i64,
        config: &PocketEncoderConfig,
    ) -> Self {
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
        let message_input_dim = message_input_dim(hidden_dim, config);
        let message_layers = (0..config.message_passing_layers)
            .map(|layer_ix| {
                nn::linear(
                    vs / format!("local_message_{layer_ix}"),
                    message_input_dim,
                    hidden_dim,
                    Default::default(),
                )
            })
            .collect();
        Self {
            atom_projection,
            pooled_projection,
            message_layers,
            config: config.clone(),
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
        let coord_inputs = self.coordinate_inputs_batch(coords, mask);
        let projected = Tensor::cat(&[atom_features.shallow_clone(), coord_inputs], -1)
            .apply(&self.atom_projection)
            .relu()
            * mask.unsqueeze(-1);
        let local_context = self.apply_local_messages_batch(projected, coords, mask);
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
        let coord_inputs = self.coordinate_inputs_single(&input.coords);
        let local_context = Tensor::cat(&[input.atom_features.shallow_clone(), coord_inputs], 1)
            .apply(&self.atom_projection)
            .relu();
        let local_context = self.apply_local_messages_single(local_context, &input.coords);
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

impl PocketEncoderImpl {
    fn coordinate_inputs_single(&self, coords: &Tensor) -> Tensor {
        match self.config.kind {
            PocketEncoderKind::LigandRelativeLocalFrame => local_frame_coords_single(coords),
            PocketEncoderKind::FeatureProjection | PocketEncoderKind::LocalMessagePassing => {
                coords.shallow_clone()
            }
        }
    }

    fn coordinate_inputs_batch(&self, coords: &Tensor, mask: &Tensor) -> Tensor {
        match self.config.kind {
            PocketEncoderKind::LigandRelativeLocalFrame => local_frame_coords_batch(coords, mask),
            PocketEncoderKind::FeatureProjection | PocketEncoderKind::LocalMessagePassing => {
                coords.shallow_clone() * mask.unsqueeze(-1)
            }
        }
    }

    fn apply_local_messages_single(&self, mut hidden: Tensor, coords: &Tensor) -> Tensor {
        if self.config.kind == PocketEncoderKind::FeatureProjection
            || self.message_layers.is_empty()
        {
            return hidden;
        }
        let distances = pairwise_distances(coords);
        let message_mask = local_message_mask(&distances, self.config.neighbor_radius);
        let message_summary = self.message_summary_single(coords, &distances, &message_mask);
        for layer in &self.message_layers {
            let denom = message_mask
                .sum_dim_intlist([1].as_slice(), true, Kind::Float)
                .clamp_min(1.0);
            let weights = &message_mask / denom;
            let neighbor_hidden = weights.matmul(&hidden);
            let update = Tensor::cat(
                &[
                    hidden.shallow_clone(),
                    neighbor_hidden,
                    message_summary.shallow_clone(),
                ],
                1,
            )
            .apply(layer)
            .relu();
            hidden = hidden + update * self.config.residual_scale;
        }
        hidden
    }

    fn apply_local_messages_batch(
        &self,
        mut hidden: Tensor,
        coords: &Tensor,
        mask: &Tensor,
    ) -> Tensor {
        if self.config.kind == PocketEncoderKind::FeatureProjection
            || self.message_layers.is_empty()
        {
            return hidden * mask.unsqueeze(-1);
        }
        let distances = batched_pairwise_distances(coords);
        let message_mask = local_message_mask(&distances, self.config.neighbor_radius)
            * mask.unsqueeze(1)
            * mask.unsqueeze(2);
        let message_summary = self.message_summary_batch(coords, &distances, &message_mask, mask);
        for layer in &self.message_layers {
            let denom = message_mask
                .sum_dim_intlist([2].as_slice(), true, Kind::Float)
                .clamp_min(1.0);
            let weights = &message_mask / denom;
            let neighbor_hidden = weights.matmul(&hidden);
            let update = Tensor::cat(
                &[
                    hidden.shallow_clone(),
                    neighbor_hidden,
                    message_summary.shallow_clone(),
                ],
                -1,
            )
            .apply(layer)
            .relu();
            hidden = (hidden + update * self.config.residual_scale) * mask.unsqueeze(-1);
        }
        hidden
    }

    fn message_summary_single(
        &self,
        coords: &Tensor,
        distances: &Tensor,
        message_mask: &Tensor,
    ) -> Tensor {
        match self.config.kind {
            PocketEncoderKind::LigandRelativeLocalFrame => {
                local_frame_message_summary_single(coords, distances, message_mask)
            }
            PocketEncoderKind::FeatureProjection | PocketEncoderKind::LocalMessagePassing => {
                masked_mean_distance(distances, message_mask)
            }
        }
    }

    fn message_summary_batch(
        &self,
        coords: &Tensor,
        distances: &Tensor,
        message_mask: &Tensor,
        mask: &Tensor,
    ) -> Tensor {
        match self.config.kind {
            PocketEncoderKind::LigandRelativeLocalFrame => {
                local_frame_message_summary_batch(coords, distances, message_mask, mask)
            }
            PocketEncoderKind::FeatureProjection | PocketEncoderKind::LocalMessagePassing => {
                masked_mean_distance(distances, message_mask)
            }
        }
    }
}

fn message_input_dim(hidden_dim: i64, config: &PocketEncoderConfig) -> i64 {
    match config.kind {
        PocketEncoderKind::LigandRelativeLocalFrame => hidden_dim * 2 + 4,
        PocketEncoderKind::FeatureProjection | PocketEncoderKind::LocalMessagePassing => {
            hidden_dim * 2 + 1
        }
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
    let masked_coords = coords * mask.unsqueeze(-1);
    let frame = local_frame_axes_batch(&masked_coords);
    coords.matmul(&frame.transpose(1, 2)) * mask.unsqueeze(-1)
}

fn local_frame_message_summary_single(
    coords: &Tensor,
    distances: &Tensor,
    message_mask: &Tensor,
) -> Tensor {
    let atom_count = coords.size().first().copied().unwrap_or(0);
    if atom_count == 0 {
        return Tensor::zeros([0, 4], (Kind::Float, coords.device()));
    }
    let mask = Tensor::ones([1, atom_count], (Kind::Float, coords.device()));
    local_frame_message_summary_batch(
        &coords.unsqueeze(0),
        &distances.unsqueeze(0),
        &message_mask.unsqueeze(0),
        &mask,
    )
    .squeeze_dim(0)
}

fn local_frame_message_summary_batch(
    coords: &Tensor,
    distances: &Tensor,
    message_mask: &Tensor,
    mask: &Tensor,
) -> Tensor {
    let distance_summary = masked_mean_distance(distances, message_mask);
    let masked_coords = coords * mask.unsqueeze(-1);
    let frame = local_frame_axes_batch(&masked_coords);
    let local_frame = frame.transpose(1, 2).unsqueeze(1);
    let relative = coords.unsqueeze(1) - coords.unsqueeze(2);
    let local_relative = relative.matmul(&local_frame);
    let directions = local_relative / distances.clamp_min(1.0e-6).unsqueeze(-1);
    let denom = message_mask
        .sum_dim_intlist([2].as_slice(), true, Kind::Float)
        .clamp_min(1.0);
    let directional = (directions * message_mask.unsqueeze(-1)).sum_dim_intlist(
        [2].as_slice(),
        false,
        Kind::Float,
    ) / denom;
    Tensor::cat(&[distance_summary, directional], -1)
}

fn local_frame_axes_batch(coords: &Tensor) -> Tensor {
    let size = coords.size();
    let batch_size = size.first().copied().unwrap_or(0);
    let atom_count = size.get(1).copied().unwrap_or(0);
    let device = coords.device();
    let fallback_x = Tensor::from_slice(&[1.0_f32, 0.0, 0.0])
        .to_device(device)
        .reshape([1, 3])
        .expand([batch_size, 3], true);
    let fallback_y = Tensor::from_slice(&[0.0_f32, 1.0, 0.0])
        .to_device(device)
        .reshape([1, 3])
        .expand([batch_size, 3], true);

    let first = if atom_count > 0 {
        coords.select(1, 0)
    } else {
        Tensor::zeros([batch_size, 3], (Kind::Float, device))
    };
    let e1 = normalize_with_fallback_batch(&first, &fallback_x);

    let second = if atom_count > 1 {
        coords.select(1, 1)
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

fn pairwise_distances(coords: &Tensor) -> Tensor {
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .sqrt()
}

fn batched_pairwise_distances(coords: &Tensor) -> Tensor {
    let diffs = coords.unsqueeze(2) - coords.unsqueeze(1);
    diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([3].as_slice(), false, Kind::Float)
        .sqrt()
}

fn local_message_mask(distances: &Tensor, radius: f64) -> Tensor {
    distances.gt(0.0).to_kind(Kind::Float) * distances.le(radius).to_kind(Kind::Float)
}

fn masked_mean_distance(distances: &Tensor, mask: &Tensor) -> Tensor {
    let neighbor_dim = distances.size().len() as i64 - 1;
    let denom = mask
        .sum_dim_intlist([neighbor_dim].as_slice(), true, Kind::Float)
        .clamp_min(1.0);
    (distances * mask).sum_dim_intlist([neighbor_dim].as_slice(), true, Kind::Float) / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn pocket_local_message_passing_preserves_batch_shapes_and_masks() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let encoder = PocketEncoderImpl::new_with_config(
            &var_store.root(),
            5,
            12,
            &PocketEncoderConfig::default(),
        );
        let atom_features = Tensor::ones([1, 3, 5], (Kind::Float, Device::Cpu));
        let coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 9.0, 0.0, 0.0])
            .reshape([1, 3, 3]);
        let pooled_features = Tensor::ones([1, 5], (Kind::Float, Device::Cpu));
        let mask = Tensor::from_slice(&[1.0_f32, 1.0, 0.0]).reshape([1, 3]);

        let output = encoder.encode_batch(&atom_features, &coords, &pooled_features, &mask);

        assert_eq!(output.token_embeddings.size(), vec![1, 3, 12]);
        assert_eq!(output.pooled_embedding.size(), vec![1, 12]);
        let padded_norm = output.token_embeddings.get(0).get(2).abs().sum(Kind::Float);
        assert_eq!(padded_norm.double_value(&[]), 0.0);
    }

    #[test]
    fn feature_projection_pocket_encoder_remains_selectable() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = PocketEncoderConfig {
            kind: PocketEncoderKind::FeatureProjection,
            ..PocketEncoderConfig::default()
        };
        let encoder = PocketEncoderImpl::new_with_config(&var_store.root(), 4, 8, &config);
        let features = PocketFeatures {
            atom_features: Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            coords: Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]).reshape([2, 3]),
            pooled_features: Tensor::ones([4], (Kind::Float, Device::Cpu)),
            chemistry_roles: crate::data::features::ChemistryRoleFeatureMatrix {
                role_vectors: Tensor::zeros(
                    [2, crate::data::features::CHEMISTRY_ROLE_FEATURE_DIM],
                    (Kind::Float, Device::Cpu),
                ),
                availability: Tensor::zeros([2], (Kind::Float, Device::Cpu)),
                provenance: crate::data::features::ChemistryRoleFeatureProvenance::Unavailable,
            },
        };
        let output = encoder.encode(&features);
        assert_eq!(output.token_embeddings.size(), vec![2, 8]);
        assert_eq!(output.pooled_embedding.size(), vec![8]);
    }

    #[test]
    fn local_message_mask_respects_neighbor_radius() {
        let coords =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.5, 0.0, 0.0, 5.0, 0.0, 0.0]).reshape([3, 3]);
        let distances = pairwise_distances(&coords);
        let tight = local_message_mask(&distances, 2.0);
        let wide = local_message_mask(&distances, 6.0);

        assert_eq!(tight.get(0).sum(Kind::Float).double_value(&[]), 1.0);
        assert_eq!(tight.get(2).sum(Kind::Float).double_value(&[]), 0.0);
        assert_eq!(wide.get(0).sum(Kind::Float).double_value(&[]), 2.0);
    }

    #[test]
    fn ligand_relative_local_frame_preserves_masks_and_coordinate_frame() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = PocketEncoderConfig {
            kind: PocketEncoderKind::LigandRelativeLocalFrame,
            neighbor_radius: 4.0,
            ..PocketEncoderConfig::default()
        };
        let encoder = PocketEncoderImpl::new_with_config(&var_store.root(), 4, 10, &config);
        let atom_features = Tensor::ones([1, 3, 4], (Kind::Float, Device::Cpu));
        let coords = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, -0.2, 1.4, 0.1, 0.3, 0.5, 1.2])
            .reshape([1, 3, 3]);
        let pooled_features = Tensor::ones([1, 4], (Kind::Float, Device::Cpu));
        let mask = Tensor::from_slice(&[1.0_f32, 1.0, 0.0]).reshape([1, 3]);

        let output = encoder.encode_batch(&atom_features, &coords, &pooled_features, &mask);
        assert_eq!(output.token_embeddings.size(), vec![1, 3, 10]);
        let padded_norm = output.token_embeddings.get(0).get(2).abs().sum(Kind::Float);
        assert_eq!(padded_norm.double_value(&[]), 0.0);

        let all_active_mask = Tensor::ones([1, 3], (Kind::Float, Device::Cpu));
        let rotation =
            Tensor::from_slice(&[0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([3, 3]);
        let rotated = coords.get(0).matmul(&rotation).unsqueeze(0);
        let base =
            encoder.encode_batch(&atom_features, &coords, &pooled_features, &all_active_mask);
        let next =
            encoder.encode_batch(&atom_features, &rotated, &pooled_features, &all_active_mask);
        let token_delta = (base.token_embeddings - next.token_embeddings)
            .abs()
            .max()
            .double_value(&[]);
        assert!(token_delta < 1e-5, "token delta was {token_delta}");
    }

    #[test]
    fn ligand_relative_local_frame_requires_upstream_ligand_centering() {
        let coords = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, -0.2, 1.4, 0.1, 0.3, 0.5, 1.2])
            .reshape([1, 3, 3]);
        let mask = Tensor::ones([1, 3], (Kind::Float, Device::Cpu));
        let shift = Tensor::from_slice(&[6.0_f32, -4.0, 2.5]).reshape([1, 1, 3]);

        let base = local_frame_coords_batch(&coords, &mask);
        let translated = local_frame_coords_batch(&(coords + shift), &mask);
        let delta = (base - translated).abs().max().double_value(&[]);

        assert!(
            delta > 1.0,
            "local-frame coordinates unexpectedly ignored upstream translation: {delta}"
        );
    }
}
