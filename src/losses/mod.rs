//! Ablatable loss modules for the modular research stack.

pub(crate) mod alignment;
pub mod auxiliary;
pub(crate) mod classification;
pub mod consistency;
pub mod gate;
pub mod leakage;
pub mod mutual_information;
pub(crate) mod native_score_calibration;
pub mod pocket_prior;
pub mod probe;
pub mod redundancy;
pub mod task;
pub(crate) mod topology_calibration;

pub use auxiliary::{AuxiliaryObjectiveBlock, SlotControlLoss};
pub use consistency::{ChemistryGuardrailAuxLoss, ConsistencyLoss, PocketGeometryAuxLoss};
pub use gate::{GateLoss, GatePathObjectiveContribution};
pub use leakage::{
    LeakageDetachedTrainingDiagnosticSection, LeakageEvidenceRoleReport,
    LeakageFrozenProbeAuditSection, LeakageLoss, LeakageOptimizerPenaltySection,
};
pub use mutual_information::{DecouplingQualityReport, MutualInformationMonitor};
pub use pocket_prior::PocketPriorAuxLoss;
pub use probe::ProbeLoss;
pub use redundancy::IntraRedundancyLoss;
#[cfg(test)]
pub(crate) use task::compute_primary_objective_batch_with_components;
pub use task::SurrogateReconstructionObjective;
pub(crate) use task::{
    build_primary_objective, compute_primary_objective_batch_with_components_at_step,
    compute_rollout_training_loss, PrimaryObjectiveWithComponents,
};
