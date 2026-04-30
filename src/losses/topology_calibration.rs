//! Sparse topology calibration shared by probe, flow, and rollout objectives.

use tch::{Kind, Tensor};

use crate::config::SparseTopologyCalibrationConfig;

use super::{
    classification::masked_sparse_negative_rate_loss_with_floor,
    native_score_calibration::{
        native_score_calibration_raw_loss, NativeScoreCalibrationComponents,
    },
};

/// Objective scope receiving sparse topology calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SparseTopologyCalibrationScope {
    /// Same-modality topology semantic probe.
    Probe,
    /// Teacher-forced molecular flow bond/topology heads.
    Flow,
    /// Optimizer-facing generated rollout bond consistency.
    Rollout,
}

/// Tensor bundle for one topology calibration evaluation.
pub(crate) struct SparseTopologyCalibrationLoss {
    /// Raw penalty after the configured safety cap and before the scope weight.
    #[allow(dead_code)] // Kept for audit-oriented callers that need cap diagnostics.
    pub raw: Tensor,
    /// Raw penalty before the configured safety cap.
    #[allow(dead_code)] // Kept for audit-oriented callers that need cap diagnostics.
    pub uncapped_raw: Tensor,
    /// Penalty after the internal scope weight and warmup multiplier.
    pub weighted: Tensor,
    /// Effective scalar weight used for this step and scope.
    #[allow(dead_code)] // Serialized metrics currently report weighted terms.
    pub effective_weight: f64,
    /// Configured cap applied to the raw term.
    #[allow(dead_code)] // Kept for audit-oriented callers that need cap diagnostics.
    pub max_raw_loss: f64,
    /// Optional optimizer-facing native-score subterm decomposition after cap scaling.
    pub native_score_components: Option<NativeScoreCalibrationComponents>,
}

/// Configured sparse binary calibration for over-dense graph predictions.
#[derive(Debug, Clone)]
pub(crate) struct SparseTopologyCalibration {
    config: SparseTopologyCalibrationConfig,
}

impl Default for SparseTopologyCalibration {
    fn default() -> Self {
        Self::new(SparseTopologyCalibrationConfig::default())
    }
}

impl SparseTopologyCalibration {
    /// Construct calibration from training config.
    pub(crate) fn new(config: SparseTopologyCalibrationConfig) -> Self {
        Self { config }
    }

    /// Compute a scoped calibration loss for one binary topology tensor.
    pub(crate) fn loss(
        &self,
        scope: SparseTopologyCalibrationScope,
        step: Option<usize>,
        logits: &Tensor,
        target: &Tensor,
        mask: &Tensor,
    ) -> SparseTopologyCalibrationLoss {
        let effective_weight = self.effective_weight(scope, step);
        let device = logits.device();
        if effective_weight <= 0.0 {
            return zero_calibration_loss(device, effective_weight, self.config.max_raw_loss);
        }
        let uncapped_raw = masked_sparse_negative_rate_loss_with_floor(
            logits,
            target,
            mask,
            self.config.min_rate_scale,
        );
        let raw = bounded_scalar_loss(&uncapped_raw, self.config.max_raw_loss);
        let weighted = &raw * effective_weight;
        SparseTopologyCalibrationLoss {
            raw,
            uncapped_raw,
            weighted,
            effective_weight,
            max_raw_loss: self.config.max_raw_loss,
            native_score_components: None,
        }
    }

    /// Compute bounded native graph confidence pressure for flow graph heads.
    pub(crate) fn confidence_pressure_loss(
        &self,
        scope: SparseTopologyCalibrationScope,
        step: Option<usize>,
        logits: &Tensor,
        target: &Tensor,
        mask: &Tensor,
    ) -> SparseTopologyCalibrationLoss {
        let effective_weight = match scope {
            SparseTopologyCalibrationScope::Flow => {
                self.config.confidence_pressure_weight * self.config.schedule_multiplier(step)
            }
            SparseTopologyCalibrationScope::Probe | SparseTopologyCalibrationScope::Rollout => 0.0,
        };
        let device = logits.device();
        if effective_weight <= 0.0 {
            return zero_calibration_loss(
                device,
                effective_weight,
                self.config.confidence_pressure_max_loss,
            );
        }

        let uncapped_raw = native_graph_confidence_pressure_raw_loss(logits, target, mask);
        let raw = bounded_scalar_loss(&uncapped_raw, self.config.confidence_pressure_max_loss);
        let weighted = &raw * effective_weight;
        SparseTopologyCalibrationLoss {
            raw,
            uncapped_raw,
            weighted,
            effective_weight,
            max_raw_loss: self.config.confidence_pressure_max_loss,
            native_score_components: None,
        }
    }

    /// Compute bounded target-degree alignment for flow graph heads.
    pub(crate) fn degree_alignment_loss(
        &self,
        scope: SparseTopologyCalibrationScope,
        step: Option<usize>,
        logits: &Tensor,
        target: &Tensor,
        mask: &Tensor,
    ) -> SparseTopologyCalibrationLoss {
        let effective_weight = match scope {
            SparseTopologyCalibrationScope::Flow => {
                self.config.degree_alignment_weight * self.config.schedule_multiplier(step)
            }
            SparseTopologyCalibrationScope::Probe => {
                self.config.probe_degree_alignment_weight * self.config.schedule_multiplier(step)
            }
            SparseTopologyCalibrationScope::Rollout => 0.0,
        };
        let device = logits.device();
        if effective_weight <= 0.0 {
            return zero_calibration_loss(
                device,
                effective_weight,
                self.config.degree_alignment_max_loss,
            );
        }

        let uncapped_raw = expected_degree_alignment_raw_loss(logits, target, mask);
        let raw = bounded_scalar_loss(&uncapped_raw, self.config.degree_alignment_max_loss);
        let weighted = &raw * effective_weight;
        SparseTopologyCalibrationLoss {
            raw,
            uncapped_raw,
            weighted,
            effective_weight,
            max_raw_loss: self.config.degree_alignment_max_loss,
            native_score_components: None,
        }
    }

    /// Compute bounded extraction-score calibration for the paired native graph heads.
    pub(crate) fn native_score_calibration_loss(
        &self,
        scope: SparseTopologyCalibrationScope,
        step: Option<usize>,
        bond_logits: &Tensor,
        topology_logits: &Tensor,
        target: &Tensor,
        mask: &Tensor,
    ) -> SparseTopologyCalibrationLoss {
        let effective_weight = match scope {
            SparseTopologyCalibrationScope::Flow => {
                self.config.native_score_calibration_weight * self.config.schedule_multiplier(step)
            }
            SparseTopologyCalibrationScope::Probe | SparseTopologyCalibrationScope::Rollout => 0.0,
        };
        let device = bond_logits.device();
        if effective_weight <= 0.0 {
            return zero_calibration_loss(
                device,
                effective_weight,
                self.config.native_score_calibration_max_loss,
            );
        }

        let raw_loss = native_score_calibration_raw_loss(
            bond_logits,
            topology_logits,
            target,
            mask,
            self.config.native_score_threshold,
        );
        let uncapped_raw = raw_loss.total;
        let raw = bounded_scalar_loss(&uncapped_raw, self.config.native_score_calibration_max_loss);
        let component_scale = bounded_component_scale(&raw, &uncapped_raw);
        let weighted = &raw * effective_weight;
        let native_score_components = Some(
            raw_loss
                .components
                .scale_for_reporting(&component_scale, effective_weight),
        );
        SparseTopologyCalibrationLoss {
            raw,
            uncapped_raw,
            weighted,
            effective_weight,
            max_raw_loss: self.config.native_score_calibration_max_loss,
            native_score_components,
        }
    }

    fn effective_weight(&self, scope: SparseTopologyCalibrationScope, step: Option<usize>) -> f64 {
        match scope {
            SparseTopologyCalibrationScope::Probe => self.config.effective_probe_weight(step),
            SparseTopologyCalibrationScope::Flow => self.config.effective_flow_weight(step),
            SparseTopologyCalibrationScope::Rollout => self.config.effective_rollout_weight(step),
        }
    }
}

fn zero_calibration_loss(
    device: tch::Device,
    effective_weight: f64,
    max_raw_loss: f64,
) -> SparseTopologyCalibrationLoss {
    let zero = Tensor::zeros([1], (Kind::Float, device));
    SparseTopologyCalibrationLoss {
        raw: zero.shallow_clone(),
        uncapped_raw: zero.shallow_clone(),
        weighted: zero,
        effective_weight,
        max_raw_loss,
        native_score_components: None,
    }
}

fn bounded_scalar_loss(raw: &Tensor, max_raw_loss: f64) -> Tensor {
    raw.clamp_max(max_raw_loss.max(1.0e-6))
}

fn bounded_component_scale(raw: &Tensor, uncapped_raw: &Tensor) -> Tensor {
    let denominator = uncapped_raw.detach().clamp_min(1.0e-6);
    (raw.detach() / denominator).clamp(0.0, 1.0)
}

fn native_graph_confidence_pressure_raw_loss(
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
    let mask = mask
        .to_device(logits.device())
        .to_kind(Kind::Float)
        .clamp(0.0, 1.0);
    let positive_pairs = &target * &mask;
    let positive_count = positive_pairs.sum(Kind::Float);
    if positive_count.double_value(&[]) <= 0.0 {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }

    let probabilities = logits.sigmoid();
    let positive_logit_floor = Tensor::from(1.0f32).to_device(logits.device());
    let degree_floor = Tensor::from(1.0f32).to_device(logits.device());
    let negative_logit_ceiling = Tensor::from(-1.0f32).to_device(logits.device());
    let positive_margin_loss = ((&positive_logit_floor - logits).relu() * &positive_pairs)
        .sum(Kind::Float)
        / positive_count.clamp_min(1.0);

    let target_degree = positive_pairs.sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let active_atoms = target_degree.gt(0.0).to_kind(Kind::Float);
    let supported_degree =
        (&probabilities * &positive_pairs).sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let degree_floor_loss = ((&degree_floor - supported_degree).relu() * &active_atoms)
        .sum(Kind::Float)
        / active_atoms.sum(Kind::Float).clamp_min(1.0);

    let negative_pairs = (&mask - &positive_pairs).clamp(0.0, 1.0);
    let negative_count = negative_pairs.sum(Kind::Float);
    let density_margin_loss = if negative_count.double_value(&[]) > 0.0 {
        ((logits - &negative_logit_ceiling).relu() * &negative_pairs).sum(Kind::Float)
            / negative_count.clamp_min(1.0)
    } else {
        Tensor::zeros([1], (Kind::Float, logits.device()))
    };
    let target_bond_mass = &positive_count * 0.5;
    let predicted_bond_mass = (&probabilities * &mask).sum(Kind::Float) * 0.5;
    let density_budget_loss = ((predicted_bond_mass - &target_bond_mass * 1.25).relu()
        / target_bond_mass.clamp_min(1.0))
    .pow_tensor_scalar(2.0);

    0.5 * positive_margin_loss
        + degree_floor_loss
        + 0.1 * density_margin_loss
        + 0.25 * density_budget_loss
}

fn expected_degree_alignment_raw_loss(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || logits.size() != target.size()
        || logits.size() != mask.size()
        || logits.size().len() != 2
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
    let probabilities = logits.sigmoid() * &mask;
    let target_edges = &target * &mask;
    let target_degree = target_edges.sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let predicted_degree = probabilities.sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let active_atoms = target_degree.gt(0.0).to_kind(Kind::Float);
    if active_atoms.sum(Kind::Float).double_value(&[]) <= 0.0 {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }

    let normalization = (&target_degree + 1.0).detach().pow_tensor_scalar(2.0);
    let per_atom =
        (&predicted_degree - &target_degree).pow_tensor_scalar(2.0) / normalization.clamp_min(1.0);
    (per_atom * &active_atoms).sum(Kind::Float) / active_atoms.sum(Kind::Float).clamp_min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn sparse_topology_calibration_warmup_controls_effective_weight() {
        let calibration = SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 2,
            warmup_steps: 4,
            probe_weight: 0.08,
            flow_weight: 0.2,
            rollout_weight: 0.1,
            min_rate_scale: 0.05,
            max_raw_loss: 8.0,
            confidence_pressure_weight: 0.5,
            confidence_pressure_max_loss: 8.0,
            degree_alignment_weight: 0.1,
            probe_degree_alignment_weight: 0.02,
            degree_alignment_max_loss: 6.0,
            native_score_calibration_weight: 0.15,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        });
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let mask = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let logits = Tensor::zeros([4], (Kind::Float, Device::Cpu));

        let before = calibration.loss(
            SparseTopologyCalibrationScope::Flow,
            Some(1),
            &logits,
            &target,
            &mask,
        );
        let ramp = calibration.loss(
            SparseTopologyCalibrationScope::Flow,
            Some(2),
            &logits,
            &target,
            &mask,
        );
        let final_step = calibration.loss(
            SparseTopologyCalibrationScope::Flow,
            Some(8),
            &logits,
            &target,
            &mask,
        );

        assert_eq!(before.effective_weight, 0.0);
        assert!((ramp.effective_weight - 0.05).abs() < 1.0e-12);
        assert!((final_step.effective_weight - 0.2).abs() < 1.0e-12);
        assert!(final_step.weighted.double_value(&[]) > ramp.weighted.double_value(&[]));
    }

    #[test]
    fn sparse_topology_calibration_caps_unweighted_density_loss() {
        let calibration = SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 0,
            warmup_steps: 0,
            probe_weight: 0.0,
            flow_weight: 1.0,
            rollout_weight: 0.0,
            min_rate_scale: 0.01,
            max_raw_loss: 0.25,
            confidence_pressure_weight: 0.5,
            confidence_pressure_max_loss: 8.0,
            degree_alignment_weight: 0.1,
            probe_degree_alignment_weight: 0.02,
            degree_alignment_max_loss: 6.0,
            native_score_calibration_weight: 0.15,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        });
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let mask = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let logits = Tensor::full([4], 20.0, (Kind::Float, Device::Cpu));

        let loss = calibration.loss(
            SparseTopologyCalibrationScope::Flow,
            Some(0),
            &logits,
            &target,
            &mask,
        );

        assert!(loss.uncapped_raw.double_value(&[]) > 0.25);
        assert!((loss.raw.double_value(&[]) - 0.25).abs() < 1.0e-8);
        assert!((loss.weighted.double_value(&[]) - 0.25).abs() < 1.0e-8);
    }

    #[test]
    fn native_graph_confidence_pressure_is_scheduled_and_capped() {
        let calibration = SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 2,
            warmup_steps: 2,
            probe_weight: 0.0,
            flow_weight: 0.0,
            rollout_weight: 0.0,
            min_rate_scale: 0.05,
            max_raw_loss: 8.0,
            confidence_pressure_weight: 0.5,
            confidence_pressure_max_loss: 0.2,
            degree_alignment_weight: 0.1,
            probe_degree_alignment_weight: 0.02,
            degree_alignment_max_loss: 6.0,
            native_score_calibration_weight: 0.15,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        });
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]).reshape([2, 2]);
        let mask = Tensor::ones([2, 2], (Kind::Float, Device::Cpu));
        let logits = Tensor::full([2, 2], 20.0, (Kind::Float, Device::Cpu));

        let before = calibration.confidence_pressure_loss(
            SparseTopologyCalibrationScope::Flow,
            Some(1),
            &logits,
            &target,
            &mask,
        );
        let active = calibration.confidence_pressure_loss(
            SparseTopologyCalibrationScope::Flow,
            Some(4),
            &logits,
            &target,
            &mask,
        );

        assert_eq!(before.effective_weight, 0.0);
        assert!(active.uncapped_raw.double_value(&[]) > 0.2);
        assert!((active.raw.double_value(&[]) - 0.2).abs() < 1.0e-8);
        assert!((active.weighted.double_value(&[]) - 0.1).abs() < 1.0e-8);
    }

    #[test]
    fn expected_degree_alignment_penalizes_underconnected_predictions() {
        let calibration = SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 0,
            warmup_steps: 0,
            probe_weight: 0.0,
            flow_weight: 0.0,
            rollout_weight: 0.0,
            min_rate_scale: 0.05,
            max_raw_loss: 8.0,
            confidence_pressure_weight: 0.0,
            confidence_pressure_max_loss: 8.0,
            degree_alignment_weight: 0.5,
            probe_degree_alignment_weight: 0.02,
            degree_alignment_max_loss: 6.0,
            native_score_calibration_weight: 0.15,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        });
        let target =
            Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).reshape([3, 3]);
        let mask = Tensor::ones([3, 3], (Kind::Float, Device::Cpu))
            - Tensor::eye(3, (Kind::Float, Device::Cpu));
        let underconnected = Tensor::full([3, 3], -8.0, (Kind::Float, Device::Cpu));
        let aligned = &target * 12.0 - 6.0;

        let under_loss = calibration
            .degree_alignment_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &underconnected,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);
        let aligned_loss = calibration
            .degree_alignment_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &aligned,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);

        assert!(under_loss > aligned_loss + 0.1);
    }

    #[test]
    fn expected_degree_alignment_is_scheduled_and_capped() {
        let calibration = SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 2,
            warmup_steps: 2,
            probe_weight: 0.0,
            flow_weight: 0.0,
            rollout_weight: 0.0,
            min_rate_scale: 0.05,
            max_raw_loss: 8.0,
            confidence_pressure_weight: 0.0,
            confidence_pressure_max_loss: 8.0,
            degree_alignment_weight: 0.5,
            probe_degree_alignment_weight: 0.02,
            degree_alignment_max_loss: 0.2,
            native_score_calibration_weight: 0.15,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        });
        let target = Tensor::from_slice(&[0.0_f32, 1.0, 1.0, 0.0]).reshape([2, 2]);
        let mask = Tensor::ones([2, 2], (Kind::Float, Device::Cpu))
            - Tensor::eye(2, (Kind::Float, Device::Cpu));
        let logits = Tensor::full([2, 2], -20.0, (Kind::Float, Device::Cpu));

        let before = calibration.degree_alignment_loss(
            SparseTopologyCalibrationScope::Flow,
            Some(1),
            &logits,
            &target,
            &mask,
        );
        let active = calibration.degree_alignment_loss(
            SparseTopologyCalibrationScope::Flow,
            Some(4),
            &logits,
            &target,
            &mask,
        );

        assert_eq!(before.effective_weight, 0.0);
        assert!(active.uncapped_raw.double_value(&[]) > 0.2);
        assert!((active.raw.double_value(&[]) - 0.2).abs() < 1.0e-8);
        assert!((active.weighted.double_value(&[]) - 0.1).abs() < 1.0e-8);
    }

    #[test]
    fn native_score_calibration_penalizes_false_positive_extraction_scores() {
        let calibration = SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 0,
            warmup_steps: 0,
            probe_weight: 0.0,
            flow_weight: 0.0,
            rollout_weight: 0.0,
            min_rate_scale: 0.05,
            max_raw_loss: 8.0,
            confidence_pressure_weight: 0.0,
            confidence_pressure_max_loss: 8.0,
            degree_alignment_weight: 0.0,
            probe_degree_alignment_weight: 0.0,
            degree_alignment_max_loss: 6.0,
            native_score_calibration_weight: 0.5,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        });
        let target =
            Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape([3, 3]);
        let mask = Tensor::ones([3, 3], (Kind::Float, Device::Cpu))
            - Tensor::eye(3, (Kind::Float, Device::Cpu));
        let calibrated = &target * 12.0 - 6.0;
        let over_dense = Tensor::full([3, 3], 8.0, (Kind::Float, Device::Cpu));

        let calibrated_loss = calibration
            .native_score_calibration_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &calibrated,
                &calibrated,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);
        let over_dense = calibration.native_score_calibration_loss(
            SparseTopologyCalibrationScope::Flow,
            Some(0),
            &over_dense,
            &over_dense,
            &target,
            &mask,
        );
        let over_dense_loss = over_dense.raw.double_value(&[]);
        let components = over_dense.native_score_components.as_ref().unwrap();

        assert!(over_dense_loss > calibrated_loss + 0.1);
        assert!(components.false_positive_margin.double_value(&[]) > 0.0);
        assert!(components.soft_negative_extraction.double_value(&[]) > 0.0);
    }

    #[test]
    fn native_score_calibration_penalizes_unranked_extraction_scores() {
        let calibration = SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 0,
            warmup_steps: 0,
            probe_weight: 0.0,
            flow_weight: 0.0,
            rollout_weight: 0.0,
            min_rate_scale: 0.05,
            max_raw_loss: 8.0,
            confidence_pressure_weight: 0.0,
            confidence_pressure_max_loss: 8.0,
            degree_alignment_weight: 0.0,
            probe_degree_alignment_weight: 0.0,
            degree_alignment_max_loss: 6.0,
            native_score_calibration_weight: 0.5,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        });
        let target =
            Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape([3, 3]);
        let mask = Tensor::ones([3, 3], (Kind::Float, Device::Cpu))
            - Tensor::eye(3, (Kind::Float, Device::Cpu));
        let separated = &target * 10.0 - 5.0;
        let unranked = Tensor::from_slice(&[0.0_f32, -1.5, 3.5, -1.5, 0.0, 3.5, 3.5, 3.5, 0.0])
            .reshape([3, 3]);

        let separated_loss = calibration
            .native_score_calibration_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &separated,
                &separated,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);
        let unranked_loss = calibration
            .native_score_calibration_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &unranked,
                &unranked,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);

        assert!(unranked_loss > separated_loss + 0.02);
    }
}
