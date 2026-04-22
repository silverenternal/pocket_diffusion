//! Schedules staged training weights and phase transitions.

use crate::config::{LossWeightConfig, StageScheduleConfig};

use super::TrainingStage;

/// Effective weights after stage gating and warmup scaling.
#[derive(Debug, Clone, Copy)]
pub struct EffectiveLossWeights {
    /// Primary objective.
    pub primary: f64,
    /// Intra-modality redundancy objective.
    pub intra_red: f64,
    /// Semantic probe objective.
    pub probe: f64,
    /// Leakage objective.
    pub leak: f64,
    /// Gate regularization objective.
    pub gate: f64,
    /// Slot control objective.
    pub slot: f64,
    /// Topology-geometry consistency objective.
    pub consistency: f64,
}

/// Maps optimization steps to staged loss activation.
#[derive(Debug, Clone)]
pub struct StageScheduler {
    schedule: StageScheduleConfig,
    weights: LossWeightConfig,
}

impl StageScheduler {
    /// Create a scheduler from config.
    pub fn new(schedule: StageScheduleConfig, weights: LossWeightConfig) -> Self {
        Self { schedule, weights }
    }

    /// Determine the training stage for a step.
    pub fn stage_for_step(&self, step: usize) -> TrainingStage {
        if step < self.schedule.stage1_steps {
            TrainingStage::Stage1
        } else if step < self.schedule.stage2_steps {
            TrainingStage::Stage2
        } else if step < self.schedule.stage3_steps {
            TrainingStage::Stage3
        } else {
            TrainingStage::Stage4
        }
    }

    /// Compute effective weights with gradual warmup inside each stage.
    pub fn weights_for_step(&self, step: usize) -> EffectiveLossWeights {
        let stage = self.stage_for_step(step);
        let ramp = self.stage_ramp(step, stage);
        let w = &self.weights;
        EffectiveLossWeights {
            primary: w.alpha_primary,
            intra_red: if matches!(stage, TrainingStage::Stage1) {
                0.0
            } else {
                w.beta_intra_red * ramp
            },
            probe: if matches!(stage, TrainingStage::Stage1 | TrainingStage::Stage2) {
                0.0
            } else {
                w.gamma_probe * ramp
            },
            leak: if matches!(stage, TrainingStage::Stage1 | TrainingStage::Stage2) {
                0.0
            } else {
                w.delta_leak * ramp
            },
            gate: if stage == TrainingStage::Stage4 {
                w.eta_gate * ramp
            } else {
                0.0
            },
            slot: if stage == TrainingStage::Stage4 {
                w.mu_slot * ramp
            } else {
                0.0
            },
            consistency: w.nu_consistency,
        }
    }

    fn stage_ramp(&self, step: usize, stage: TrainingStage) -> f64 {
        let (start, end) = match stage {
            TrainingStage::Stage1 => (0, self.schedule.stage1_steps.max(1)),
            TrainingStage::Stage2 => (
                self.schedule.stage1_steps,
                self.schedule
                    .stage2_steps
                    .max(self.schedule.stage1_steps + 1),
            ),
            TrainingStage::Stage3 => (
                self.schedule.stage2_steps,
                self.schedule
                    .stage3_steps
                    .max(self.schedule.stage2_steps + 1),
            ),
            TrainingStage::Stage4 => (
                self.schedule.stage3_steps,
                (self.schedule.stage3_steps + 2).max(self.schedule.stage3_steps + 1),
            ),
        };
        let progress =
            step.saturating_sub(start) as f64 / (end.saturating_sub(start).max(1) as f64);
        progress.clamp(0.1, 1.0)
    }
}
