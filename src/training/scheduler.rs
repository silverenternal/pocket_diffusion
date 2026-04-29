//! Schedules staged training weights and phase transitions.

use crate::config::{ChemistryObjectiveWarmupConfig, LossWeightConfig, StageScheduleConfig};

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
    /// Pharmacophore role-probe subterms.
    pub pharmacophore_probe: f64,
    /// Leakage objective.
    pub leak: f64,
    /// Pharmacophore role-leakage subterms.
    pub pharmacophore_leakage: f64,
    /// Gate regularization objective.
    pub gate: f64,
    /// Slot control objective.
    pub slot: f64,
    /// Topology-geometry consistency objective.
    pub consistency: f64,
    /// Pocket-ligand contact encouragement objective.
    pub pocket_contact: f64,
    /// Pocket-ligand steric-clash penalty objective.
    pub pocket_clash: f64,
    /// Pocket-envelope containment objective.
    pub pocket_envelope: f64,
    /// Conservative valence overage objective.
    pub valence_guardrail: f64,
    /// Topology-implied bond-length objective.
    pub bond_length_guardrail: f64,
}

/// Maps optimization steps to staged loss activation.
#[derive(Debug, Clone)]
pub struct StageScheduler {
    schedule: StageScheduleConfig,
    weights: LossWeightConfig,
    chemistry_warmup: ChemistryObjectiveWarmupConfig,
}

impl StageScheduler {
    /// Create a scheduler from config.
    pub fn new(schedule: StageScheduleConfig, weights: LossWeightConfig) -> Self {
        Self::new_with_chemistry_warmup(
            schedule,
            weights,
            ChemistryObjectiveWarmupConfig::default(),
        )
    }

    /// Create a scheduler with explicit chemistry objective warmup controls.
    pub fn new_with_chemistry_warmup(
        schedule: StageScheduleConfig,
        weights: LossWeightConfig,
        chemistry_warmup: ChemistryObjectiveWarmupConfig,
    ) -> Self {
        Self {
            schedule,
            weights,
            chemistry_warmup,
        }
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
            probe: self.weight_after_stage(stage, ramp, 3, w.gamma_probe),
            pharmacophore_probe: self.weight_after_stage(
                stage,
                ramp,
                self.chemistry_warmup.pharmacophore_probe_start_stage,
                w.gamma_probe,
            ),
            leak: self.weight_after_stage(stage, ramp, 3, w.delta_leak),
            pharmacophore_leakage: self.weight_after_stage(
                stage,
                ramp,
                self.chemistry_warmup.pharmacophore_leakage_start_stage,
                w.delta_leak,
            ),
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
            pocket_contact: if matches!(stage, TrainingStage::Stage1) {
                0.0
            } else {
                w.rho_pocket_contact * ramp
            },
            pocket_clash: if matches!(stage, TrainingStage::Stage1) {
                0.0
            } else {
                w.sigma_pocket_clash * ramp
            },
            pocket_envelope: self.weight_after_stage(
                stage,
                ramp,
                self.chemistry_warmup.pocket_envelope_start_stage,
                w.tau_pocket_envelope,
            ),
            valence_guardrail: self.weight_after_stage(
                stage,
                ramp,
                self.chemistry_warmup.valence_guardrail_start_stage,
                w.upsilon_valence_guardrail,
            ),
            bond_length_guardrail: self.weight_after_stage(
                stage,
                ramp,
                self.chemistry_warmup.bond_length_guardrail_start_stage,
                w.phi_bond_length_guardrail,
            ),
        }
    }

    /// Linear ramp value inside the active training stage.
    pub fn ramp_for_step(&self, step: usize) -> f64 {
        let stage = self.stage_for_step(step);
        self.stage_ramp(step, stage)
    }

    fn weight_after_stage(
        &self,
        stage: TrainingStage,
        ramp: f64,
        start_stage: usize,
        final_weight: f64,
    ) -> f64 {
        if stage.index() + 1 < start_stage {
            0.0
        } else {
            final_weight * ramp
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
                self.schedule.stage3_steps + self.schedule.stage4_warmup_steps.max(1),
            ),
        };
        let progress =
            step.saturating_sub(start) as f64 / (end.saturating_sub(start).max(1) as f64);
        progress.clamp(0.1, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compact_schedule() -> StageScheduleConfig {
        StageScheduleConfig {
            stage1_steps: 1,
            stage2_steps: 2,
            stage3_steps: 3,
            stage4_warmup_steps: 2,
        }
    }

    fn nonzero_chemistry_weights() -> LossWeightConfig {
        let mut weights = LossWeightConfig::default();
        weights.gamma_probe = 0.2;
        weights.delta_leak = 0.3;
        weights.tau_pocket_envelope = 0.4;
        weights.upsilon_valence_guardrail = 0.5;
        weights.phi_bond_length_guardrail = 0.6;
        weights
    }

    #[test]
    fn default_chemistry_warmup_preserves_existing_stage_gates() {
        let scheduler = StageScheduler::new(compact_schedule(), nonzero_chemistry_weights());

        let stage1 = scheduler.weights_for_step(0);
        assert_eq!(stage1.pocket_envelope, 0.0);
        assert_eq!(stage1.valence_guardrail, 0.0);
        assert_eq!(stage1.bond_length_guardrail, 0.0);
        assert_eq!(stage1.pharmacophore_probe, 0.0);
        assert_eq!(stage1.pharmacophore_leakage, 0.0);

        let stage2 = scheduler.weights_for_step(1);
        assert!(stage2.pocket_envelope > 0.0);
        assert!(stage2.valence_guardrail > 0.0);
        assert!(stage2.bond_length_guardrail > 0.0);
        assert_eq!(stage2.pharmacophore_probe, 0.0);
        assert_eq!(stage2.pharmacophore_leakage, 0.0);

        let stage3 = scheduler.weights_for_step(2);
        assert!(stage3.pharmacophore_probe > 0.0);
        assert!(stage3.pharmacophore_leakage > 0.0);
    }

    #[test]
    fn chemistry_warmup_can_delay_objectives_until_configured_stage() {
        let scheduler = StageScheduler::new_with_chemistry_warmup(
            compact_schedule(),
            nonzero_chemistry_weights(),
            ChemistryObjectiveWarmupConfig {
                pocket_envelope_start_stage: 4,
                valence_guardrail_start_stage: 4,
                bond_length_guardrail_start_stage: 4,
                pharmacophore_probe_start_stage: 4,
                pharmacophore_leakage_start_stage: 4,
            },
        );

        for step in 0..3 {
            let weights = scheduler.weights_for_step(step);
            assert_eq!(weights.pocket_envelope, 0.0);
            assert_eq!(weights.valence_guardrail, 0.0);
            assert_eq!(weights.bond_length_guardrail, 0.0);
            assert_eq!(weights.pharmacophore_probe, 0.0);
            assert_eq!(weights.pharmacophore_leakage, 0.0);
        }

        let stage4 = scheduler.weights_for_step(3);
        assert!(stage4.pocket_envelope > 0.0);
        assert!(stage4.valence_guardrail > 0.0);
        assert!(stage4.bond_length_guardrail > 0.0);
        assert!(stage4.pharmacophore_probe > 0.0);
        assert!(stage4.pharmacophore_leakage > 0.0);
    }

    #[test]
    fn stage4_gate_and_slot_warmup_is_configurable() {
        let mut schedule = compact_schedule();
        schedule.stage4_warmup_steps = 10;
        let mut weights = nonzero_chemistry_weights();
        weights.eta_gate = 0.5;
        weights.mu_slot = 0.7;
        let eta_gate = weights.eta_gate;
        let mu_slot = weights.mu_slot;
        let scheduler = StageScheduler::new(schedule, weights);

        let start = scheduler.weights_for_step(3);
        let mid = scheduler.weights_for_step(8);
        let end = scheduler.weights_for_step(13);

        assert!(start.gate > 0.0);
        assert!(start.gate < mid.gate);
        assert!(mid.gate < end.gate);
        assert_eq!(end.gate, eta_gate);
        assert_eq!(end.slot, mu_slot);
    }
}
