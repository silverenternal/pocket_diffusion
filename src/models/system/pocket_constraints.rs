//! Pocket-frame geometric constraints used by rollout and flow refinement.
//!
//! This module intentionally keeps pocket-envelope operations separate from
//! optimizer rollout code so geometry guidance can be audited and ablated
//! without changing molecular-flow logic.

use tch::{Kind, Tensor};

const POCKET_CENTROID_RELAXATION_FRACTION: f64 = 0.10;
const MAX_POCKET_CENTROID_RELAXATION_FRACTION: f64 = 0.15;
const POCKET_ENVELOPE_MARGIN_ANGSTROM: f64 = 1.5;

pub(crate) fn constrain_to_pocket_envelope(
    coords: &Tensor,
    pocket_coords: &Tensor,
    pocket_guidance_scale: f64,
) -> Tensor {
    if coords.numel() == 0 || pocket_coords.numel() == 0 || pocket_guidance_scale <= 0.0 {
        return coords.shallow_clone();
    }

    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_radius = pocket_radius_from_coords(pocket_coords, &pocket_centroid).max(1.0);
    let max_radius = pocket_radius + POCKET_ENVELOPE_MARGIN_ANGSTROM;
    let projected = project_to_pocket_radius(coords, &pocket_centroid, max_radius);
    let pull_fraction = (POCKET_CENTROID_RELAXATION_FRACTION
        * pocket_guidance_scale.clamp(0.0, 1.5))
    .min(MAX_POCKET_CENTROID_RELAXATION_FRACTION);
    if pull_fraction <= 0.0 {
        return projected;
    }
    let ligand_centroid = projected.mean_dim([0].as_slice(), false, Kind::Float);
    let centroid_delta = (&pocket_centroid - ligand_centroid)
        .unsqueeze(0)
        .expand_as(&projected)
        * pull_fraction;
    project_to_pocket_radius(&(projected + centroid_delta), &pocket_centroid, max_radius)
}

pub(crate) fn pocket_radius_from_coords(pocket_coords: &Tensor, pocket_centroid: &Tensor) -> f64 {
    if pocket_coords.numel() == 0 {
        return 0.0;
    }
    (pocket_coords - pocket_centroid.unsqueeze(0))
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
}

fn project_to_pocket_radius(coords: &Tensor, pocket_centroid: &Tensor, max_radius: f64) -> Tensor {
    let offsets = coords - pocket_centroid.unsqueeze(0);
    let radii = offsets
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .sqrt()
        .clamp_min(1e-6);
    let scale = radii.clamp_max(max_radius) / &radii;
    let projected = pocket_centroid.unsqueeze(0) + offsets * scale;
    let outside = radii.gt(max_radius).to_kind(Kind::Float);
    let ones = Tensor::ones_like(&outside);
    &outside * projected + (&ones - &outside) * coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn pocket_envelope_relaxes_centroid_without_changing_internal_distances() {
        let coords = Tensor::from_slice(&[3.0_f32, 0.0, 0.0, 5.0, 0.0, 0.0]).reshape([2, 3]);
        let pocket = Tensor::from_slice(&[
            -10.0_f32, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0,
        ])
        .reshape([4, 3]);

        let constrained = constrain_to_pocket_envelope(&coords, &pocket, 1.0);
        let before_offset = coords
            .mean_dim([0].as_slice(), false, Kind::Float)
            .pow_tensor_scalar(2.0)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[]);
        let after_offset = constrained
            .mean_dim([0].as_slice(), false, Kind::Float)
            .pow_tensor_scalar(2.0)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[]);
        let before_distance = (&coords.get(0) - &coords.get(1))
            .pow_tensor_scalar(2.0)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[]);
        let after_distance = (&constrained.get(0) - &constrained.get(1))
            .pow_tensor_scalar(2.0)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[]);

        assert!(after_offset < before_offset);
        assert!((before_distance - after_distance).abs() < 1.0e-6);
    }

    #[test]
    fn pocket_envelope_respects_zero_guidance_scale() {
        let coords = Tensor::from_slice(&[3.0_f32, 0.0, 0.0, 5.0, 0.0, 0.0]).reshape([2, 3]);
        let pocket = Tensor::from_slice(&[
            -10.0_f32, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0,
        ])
        .reshape([4, 3]);

        let constrained = constrain_to_pocket_envelope(&coords, &pocket, 0.0);

        assert!(
            (&coords - constrained).abs().max().double_value(&[]) < 1.0e-6,
            "disabled pocket guidance should leave in-envelope coordinates unchanged"
        );
    }

    #[test]
    fn empty_pocket_radius_is_zero() {
        let empty = Tensor::zeros([0, 3], (Kind::Float, Device::Cpu));
        let centroid = Tensor::zeros([3], (Kind::Float, Device::Cpu));

        assert_eq!(pocket_radius_from_coords(&empty, &centroid), 0.0);
    }
}
