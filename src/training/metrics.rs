//! Training metrics and loss breakdowns.

use serde::{Deserialize, Serialize};

/// Named loss values emitted by each training step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LossBreakdown {
    /// Main task objective.
    pub task: f64,
    /// Intra-modality redundancy objective.
    pub intra_red: f64,
    /// Semantic probe objective.
    pub probe: f64,
    /// Leakage objective.
    pub leak: f64,
    /// Gate regularization objective.
    pub gate: f64,
    /// Slot sparsity and balance objective.
    pub slot: f64,
    /// Topology-geometry consistency objective.
    pub consistency: f64,
    /// Weighted total objective.
    pub total: f64,
}

/// One trainer step record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Global optimization step.
    pub step: usize,
    /// Current staged-training phase.
    pub stage: TrainingStage,
    /// Loss values for this step.
    pub losses: LossBreakdown,
}

/// Coarse training stage aligned with the research schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingStage {
    /// Stage 1: task and consistency only.
    Stage1,
    /// Stage 2: add redundancy reduction.
    Stage2,
    /// Stage 3: add probes and leakage.
    Stage3,
    /// Stage 4: add gate and slot control.
    Stage4,
}
