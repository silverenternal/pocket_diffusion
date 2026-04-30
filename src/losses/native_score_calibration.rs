//! Native extraction-score calibration subterms shared by flow objectives.

use tch::{Kind, Tensor};

use crate::chemistry::native_score::{NATIVE_SCORE_BOND_WEIGHT, NATIVE_SCORE_TOPOLOGY_WEIGHT};

use super::classification::masked_positive_negative_score_separation_loss;

/// Component tensors for native extraction-score calibration.
pub(crate) struct NativeScoreCalibrationComponents {
    /// Negative pairs above the extraction score ceiling.
    pub false_positive_margin: Tensor,
    /// Positive pairs below the extraction score floor.
    pub false_negative_margin: Tensor,
    /// Dense learned score mass above the target budget.
    pub density_budget: Tensor,
    /// Soft-thresholded positive target pairs not selected by extraction.
    pub soft_positive_miss: Tensor,
    /// Soft-thresholded negative pairs selected by extraction.
    pub soft_negative_extraction: Tensor,
    /// Soft-thresholded extraction mass above the target budget.
    pub soft_extraction_budget: Tensor,
    /// Per-atom score-degree mismatch.
    pub degree_alignment: Tensor,
    /// Positive-vs-negative score ranking margin.
    pub score_separation: Tensor,
}

impl NativeScoreCalibrationComponents {
    pub(crate) fn zeros(device: tch::Device) -> Self {
        Self {
            false_positive_margin: Tensor::zeros([1], (Kind::Float, device)),
            false_negative_margin: Tensor::zeros([1], (Kind::Float, device)),
            density_budget: Tensor::zeros([1], (Kind::Float, device)),
            soft_positive_miss: Tensor::zeros([1], (Kind::Float, device)),
            soft_negative_extraction: Tensor::zeros([1], (Kind::Float, device)),
            soft_extraction_budget: Tensor::zeros([1], (Kind::Float, device)),
            degree_alignment: Tensor::zeros([1], (Kind::Float, device)),
            score_separation: Tensor::zeros([1], (Kind::Float, device)),
        }
    }

    pub(crate) fn scale_for_reporting(self, scale: &Tensor, effective_weight: f64) -> Self {
        Self {
            false_positive_margin: self.false_positive_margin * scale * effective_weight,
            false_negative_margin: self.false_negative_margin * scale * effective_weight,
            density_budget: self.density_budget * scale * effective_weight,
            soft_positive_miss: self.soft_positive_miss * scale * effective_weight,
            soft_negative_extraction: self.soft_negative_extraction * scale * effective_weight,
            soft_extraction_budget: self.soft_extraction_budget * scale * effective_weight,
            degree_alignment: self.degree_alignment * scale * effective_weight,
            score_separation: self.score_separation * scale * effective_weight,
        }
    }
}

/// Raw native score calibration total plus weighted subterm decomposition.
pub(crate) struct NativeScoreCalibrationRawLoss {
    /// Uncapped raw loss before schedule/scoping weights.
    pub total: Tensor,
    /// Uncapped raw component values whose sum equals `total`.
    pub components: NativeScoreCalibrationComponents,
}

impl NativeScoreCalibrationRawLoss {
    fn zeros(device: tch::Device) -> Self {
        Self {
            total: Tensor::zeros([1], (Kind::Float, device)),
            components: NativeScoreCalibrationComponents::zeros(device),
        }
    }
}

/// Compute raw paired bond/topology extraction-score calibration.
pub(crate) fn native_score_calibration_raw_loss(
    bond_logits: &Tensor,
    topology_logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    extraction_threshold: f64,
) -> NativeScoreCalibrationRawLoss {
    if bond_logits.numel() == 0
        || topology_logits.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || bond_logits.size() != topology_logits.size()
        || bond_logits.size() != target.size()
        || bond_logits.size() != mask.size()
        || bond_logits.size().len() != 2
    {
        return NativeScoreCalibrationRawLoss::zeros(bond_logits.device());
    }

    let target = target
        .to_device(bond_logits.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let mask = mask
        .to_device(bond_logits.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let positive = &target * &mask;
    let negative = (&mask - &positive).clamp(0.0, 1.0);
    let positive_count = positive.sum(Kind::Float);
    let negative_count = negative.sum(Kind::Float);
    if positive_count.double_value(&[]) <= 0.0 || negative_count.double_value(&[]) <= 0.0 {
        return NativeScoreCalibrationRawLoss::zeros(bond_logits.device());
    }

    let learned_score = bond_logits.sigmoid() * NATIVE_SCORE_BOND_WEIGHT
        + topology_logits.sigmoid() * NATIVE_SCORE_TOPOLOGY_WEIGHT;
    let negative_ceiling = (extraction_threshold - 0.05).clamp(0.05, 0.90);
    let positive_floor = extraction_threshold.clamp(0.05, 0.90);
    let false_positive_margin = ((&learned_score - negative_ceiling).relu() * &negative)
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        / negative_count.clamp_min(1.0);
    let false_negative_margin = ((positive_floor - &learned_score).relu() * &positive)
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        / positive_count.clamp_min(1.0);
    let predicted_mass = (&learned_score * &mask).sum(Kind::Float);
    let target_mass = positive_count.shallow_clone();
    let density_budget = ((predicted_mass - &target_mass * 1.15).relu()
        / target_mass.clamp_min(1.0))
    .pow_tensor_scalar(2.0);
    let extraction_threshold_tensor =
        Tensor::from(extraction_threshold as f32).to_device(bond_logits.device());
    let soft_extracted = ((&learned_score - extraction_threshold_tensor) / 0.04).sigmoid() * &mask;
    let soft_positive_miss = ((Tensor::ones_like(&soft_extracted) - &soft_extracted) * &positive)
        .sum(Kind::Float)
        / positive_count.clamp_min(1.0);
    let soft_negative_extraction =
        (&soft_extracted * &negative).sum(Kind::Float) / negative_count.clamp_min(1.0);
    let soft_extracted_mass = soft_extracted.sum(Kind::Float);
    let soft_extraction_budget = ((soft_extracted_mass - &target_mass * 1.05).relu()
        / target_mass.clamp_min(1.0))
    .pow_tensor_scalar(2.0);
    let target_degree = positive.sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let score_degree = (&learned_score * &mask).sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let active_atoms = target_degree.gt(0.0).to_kind(Kind::Float);
    let score_degree_alignment = if active_atoms.sum(Kind::Float).double_value(&[]) > 0.0 {
        let normalization = (&target_degree + 1.0).detach().pow_tensor_scalar(2.0);
        ((score_degree - &target_degree).pow_tensor_scalar(2.0) / normalization.clamp_min(1.0)
            * &active_atoms)
            .sum(Kind::Float)
            / active_atoms.sum(Kind::Float).clamp_min(1.0)
    } else {
        Tensor::zeros([1], (Kind::Float, bond_logits.device()))
    };
    let score_separation =
        masked_positive_negative_score_separation_loss(&learned_score, &target, &mask, 0.08, 2.0);

    let components = NativeScoreCalibrationComponents {
        false_positive_margin: false_positive_margin * 1.75,
        false_negative_margin: false_negative_margin * 1.50,
        density_budget: density_budget * 0.75,
        soft_positive_miss: soft_positive_miss * 1.25,
        soft_negative_extraction: soft_negative_extraction * 1.25,
        soft_extraction_budget: soft_extraction_budget * 0.75,
        degree_alignment: score_degree_alignment * 0.45,
        score_separation: score_separation * 0.60,
    };
    let total = &components.false_positive_margin
        + &components.false_negative_margin
        + &components.density_budget
        + &components.soft_positive_miss
        + &components.soft_negative_extraction
        + &components.soft_extraction_budget
        + &components.degree_alignment
        + &components.score_separation;
    NativeScoreCalibrationRawLoss { total, components }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn raw_native_score_components_sum_to_total() {
        let target =
            Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape([3, 3]);
        let mask = Tensor::ones([3, 3], (Kind::Float, Device::Cpu))
            - Tensor::eye(3, (Kind::Float, Device::Cpu));
        let logits = Tensor::from_slice(&[0.0_f32, -1.5, 3.5, -1.5, 0.0, 3.5, 3.5, 3.5, 0.0])
            .reshape([3, 3]);

        let loss = native_score_calibration_raw_loss(&logits, &logits, &target, &mask, 0.55);
        let component_sum = &loss.components.false_positive_margin
            + &loss.components.false_negative_margin
            + &loss.components.density_budget
            + &loss.components.soft_positive_miss
            + &loss.components.soft_negative_extraction
            + &loss.components.soft_extraction_budget
            + &loss.components.degree_alignment
            + &loss.components.score_separation;

        let delta = (loss.total - component_sum).abs().double_value(&[]);
        assert!(delta < 1.0e-6);
    }
}
