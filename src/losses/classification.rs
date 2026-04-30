//! Masked classification losses shared by task and probe objectives.

use tch::{Kind, Tensor};

/// Binary cross-entropy with logits over masked items.
pub(crate) fn masked_bce_with_logits(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || logits.size() != target.size()
        || logits.size() != mask.size()
    {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let target = target.to_device(logits.device()).to_kind(Kind::Float);
    let mask = mask.to_device(logits.device()).to_kind(Kind::Float);
    let per_item = logits.clamp_min(0.0) - logits * &target + (-logits.abs()).exp().log1p();
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (per_item * mask).sum(Kind::Float) / denom
}

/// Class-balanced BCE with logits over masked items.
///
/// Positive examples receive enough weight to match the total negative mass,
/// capped to avoid unstable gradients on extremely sparse mini-batches.
pub(crate) fn masked_balanced_bce_with_logits(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
) -> Tensor {
    if logits.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || logits.size() != target.size()
        || logits.size() != mask.size()
    {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let target = target
        .to_device(logits.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let mask = mask.to_device(logits.device()).to_kind(Kind::Float);
    let positive = (&target * &mask).sum(Kind::Float);
    let negative = ((Tensor::ones_like(&target) - &target) * &mask).sum(Kind::Float);
    let positive_count = positive.double_value(&[]);
    let negative_count = negative.double_value(&[]);
    if positive_count <= 0.0 || negative_count <= 0.0 {
        return masked_bce_with_logits(logits, &target, &mask);
    }

    let positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0);
    let item_weights = &mask * (&target * positive_weight + (Tensor::ones_like(&target) - &target));
    let per_item = logits.clamp_min(0.0) - logits * &target + (-logits.abs()).exp().log1p();
    (per_item * &item_weights).sum(Kind::Float) / item_weights.sum(Kind::Float).clamp_min(1.0)
}

/// Penalize negative-class probability mass for sparse binary targets.
///
/// Class-balanced BCE keeps rare positives learnable, but it can leave sparse
/// graph targets over-dense. This term is intentionally aggregate-only and
/// target-aware: it only constrains the average probability on negative labels,
/// avoiding direct pressure against true positive edges.
#[allow(dead_code)] // Compatibility wrapper for call sites that use the default sparse-rate floor.
pub(crate) fn masked_sparse_negative_rate_loss(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
) -> Tensor {
    masked_sparse_negative_rate_loss_with_floor(logits, target, mask, 0.05)
}

/// Penalize negative-class probability mass with a configurable sparse-rate floor.
pub(crate) fn masked_sparse_negative_rate_loss_with_floor(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    min_rate_scale: f64,
) -> Tensor {
    if logits.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || logits.size() != target.size()
        || logits.size() != mask.size()
    {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let target = target
        .to_device(logits.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let mask = mask
        .to_device(logits.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    let target_rate = (&target * &mask).sum(Kind::Float) / &denom;
    let negative_mask = (Tensor::ones_like(&target) - &target) * &mask;
    let negative_count = negative_mask.sum(Kind::Float);
    if negative_count.double_value(&[]) <= 0.0 {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let negative_prediction_rate =
        (logits.sigmoid() * negative_mask).sum(Kind::Float) / negative_count.clamp_min(1.0);
    let scale = target_rate.detach().clamp_min(min_rate_scale.max(1.0e-6));
    ((negative_prediction_rate - target_rate.detach()).relu() / scale).pow_tensor_scalar(2.0)
}

/// Penalize weak separation between positive and negative masked scores.
///
/// Sparse graph heads can satisfy aggregate density constraints while still
/// assigning similar scores to true and false edges. This ranking-style term is
/// intentionally mean-based rather than sample-mined so it remains stable on
/// tiny batches and very sparse adjacency targets.
pub(crate) fn masked_positive_negative_score_separation_loss(
    scores: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    margin: f64,
    max_loss: f64,
) -> Tensor {
    if scores.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || scores.size() != target.size()
        || scores.size() != mask.size()
    {
        return Tensor::zeros([1], (Kind::Float, scores.device()));
    }
    let scores = scores.to_kind(Kind::Float);
    let target = target
        .to_device(scores.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let mask = mask
        .to_device(scores.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let positive_mask = &target * &mask;
    let negative_mask = (Tensor::ones_like(&target) - &target) * &mask;
    let positive_count = positive_mask.sum(Kind::Float);
    let negative_count = negative_mask.sum(Kind::Float);
    if positive_count.double_value(&[]) <= 0.0 || negative_count.double_value(&[]) <= 0.0 {
        return Tensor::zeros([1], (Kind::Float, scores.device()));
    }

    let positive_mean = (&scores * &positive_mask).sum(Kind::Float) / positive_count.clamp_min(1.0);
    let negative_mean = (&scores * &negative_mask).sum(Kind::Float) / negative_count.clamp_min(1.0);
    let margin = Tensor::from(margin.max(0.0) as f32).to_device(scores.device());
    (negative_mean + margin - positive_mean)
        .relu()
        .pow_tensor_scalar(2.0)
        .clamp_max(max_loss.max(1.0e-6))
}

/// Penalize positive masked scores that fall below a target floor.
///
/// This is useful for sparse binary probes where global BCE can look good by
/// predicting almost all negatives while missing the rare positive class.
pub(crate) fn masked_positive_score_floor_loss(
    scores: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    floor: f64,
    max_loss: f64,
) -> Tensor {
    if scores.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || scores.size() != target.size()
        || scores.size() != mask.size()
    {
        return Tensor::zeros([1], (Kind::Float, scores.device()));
    }
    let scores = scores.to_kind(Kind::Float);
    let target = target
        .to_device(scores.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let mask = mask
        .to_device(scores.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let positive_mask = &target * &mask;
    let positive_count = positive_mask.sum(Kind::Float);
    if positive_count.double_value(&[]) <= 0.0 {
        return Tensor::zeros([1], (Kind::Float, scores.device()));
    }

    let floor = Tensor::from(floor as f32).to_device(scores.device());
    (((floor - scores).relu() * positive_mask)
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        / positive_count.clamp_min(1.0))
    .clamp_max(max_loss.max(1.0e-6))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn balanced_bce_upweights_rare_positive_errors() {
        let logits = Tensor::from_slice(&[-2.0_f32, 0.0, 0.0, 0.0]);
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let mask = Tensor::ones([4], (Kind::Float, Device::Cpu));

        let plain = masked_bce_with_logits(&logits, &target, &mask).double_value(&[]);
        let balanced = masked_balanced_bce_with_logits(&logits, &target, &mask).double_value(&[]);

        assert!(
            balanced > plain,
            "balanced BCE should emphasize rare positive errors"
        );
    }

    #[test]
    fn balanced_bce_falls_back_when_only_one_class_is_observed() {
        let logits = Tensor::from_slice(&[0.5_f32, -0.5]);
        let target = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([2], (Kind::Float, Device::Cpu));

        let plain = masked_bce_with_logits(&logits, &target, &mask).double_value(&[]);
        let balanced = masked_balanced_bce_with_logits(&logits, &target, &mask).double_value(&[]);

        assert!((balanced - plain).abs() < 1.0e-8);
    }

    #[test]
    fn sparse_negative_rate_penalizes_false_positive_density() {
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let mask = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let calibrated = Tensor::from_slice(&[8.0_f32, -8.0, -8.0, -8.0]);
        let over_dense = Tensor::zeros([4], (Kind::Float, Device::Cpu));

        let calibrated_loss =
            masked_sparse_negative_rate_loss(&calibrated, &target, &mask).double_value(&[]);
        let over_dense_loss =
            masked_sparse_negative_rate_loss(&over_dense, &target, &mask).double_value(&[]);

        assert!(over_dense_loss > calibrated_loss);
    }

    #[test]
    fn score_separation_penalizes_unranked_sparse_scores() {
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let mask = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let separated = Tensor::from_slice(&[0.8_f32, 0.2, 0.1, 0.0]);
        let unranked = Tensor::from_slice(&[0.1_f32, 0.4, 0.3, 0.2]);

        let separated_loss =
            masked_positive_negative_score_separation_loss(&separated, &target, &mask, 0.1, 2.0)
                .double_value(&[]);
        let unranked_loss =
            masked_positive_negative_score_separation_loss(&unranked, &target, &mask, 0.1, 2.0)
                .double_value(&[]);

        assert!(unranked_loss > separated_loss + 0.01);
    }

    #[test]
    fn positive_score_floor_penalizes_missed_sparse_positives() {
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let mask = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let covered = Tensor::from_slice(&[-0.5_f32, -4.0, -4.0, -4.0]);
        let missed = Tensor::from_slice(&[-6.0_f32, -4.0, -4.0, -4.0]);

        let covered_loss = masked_positive_score_floor_loss(&covered, &target, &mask, -1.25, 4.0)
            .double_value(&[]);
        let missed_loss =
            masked_positive_score_floor_loss(&missed, &target, &mask, -1.25, 4.0).double_value(&[]);

        assert!(missed_loss > covered_loss + 1.0);
    }
}
