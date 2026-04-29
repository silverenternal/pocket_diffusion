//! Ablatable loss modules for the modular research stack.

pub(crate) mod alignment;
pub mod auxiliary;
pub mod consistency;
pub mod gate;
pub mod leakage;
pub mod mutual_information;
pub mod probe;
pub mod redundancy;
pub mod task;

pub use auxiliary::{AuxiliaryObjectiveBlock, SlotControlLoss};
pub use consistency::{ChemistryGuardrailAuxLoss, ConsistencyLoss, PocketGeometryAuxLoss};
pub use gate::{GateLoss, GatePathObjectiveContribution};
pub use leakage::{
    LeakageDetachedTrainingDiagnosticSection, LeakageEvidenceRoleReport,
    LeakageFrozenProbeAuditSection, LeakageLoss, LeakageOptimizerPenaltySection,
};
pub use mutual_information::{DecouplingQualityReport, MutualInformationMonitor};
pub use probe::ProbeLoss;
pub use redundancy::IntraRedundancyLoss;
pub use task::SurrogateReconstructionObjective;
pub(crate) use task::{
    build_primary_objective, compute_primary_objective_batch_with_components,
    PrimaryObjectiveWithComponents,
};
