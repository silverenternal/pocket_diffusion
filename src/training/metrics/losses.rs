/// Named loss values emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryObjectiveMetrics {
    /// Name of the active primary objective.
    pub objective_name: String,
    /// Weighted scalar value used as the primary optimization anchor.
    pub primary_value: f64,
    /// Whether the primary objective is decoder-anchored.
    pub decoder_anchored: bool,
}

/// Auxiliary losses emitted by each training step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AuxiliaryLossMetrics {
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
    /// Pocket-ligand contact encouragement objective.
    #[serde(default)]
    pub pocket_contact: f64,
    /// Pocket-ligand steric-clash penalty objective.
    #[serde(default)]
    pub pocket_clash: f64,
}

/// Named loss values emitted by each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossBreakdown {
    /// Primary objective metrics.
    pub primary: PrimaryObjectiveMetrics,
    /// Auxiliary regularizer metrics.
    pub auxiliaries: AuxiliaryLossMetrics,
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
