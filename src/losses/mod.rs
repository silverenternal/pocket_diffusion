//! Ablatable loss modules for the modular research stack.

pub mod consistency;
pub mod gate;
pub mod leakage;
pub mod mutual_information;
pub mod probe;
pub mod redundancy;
pub mod task;

pub use consistency::{ConsistencyLoss, PocketGeometryAuxLoss};
pub use gate::GateLoss;
pub use leakage::LeakageLoss;
pub use mutual_information::{DecouplingQualityReport, MutualInformationMonitor};
pub use probe::ProbeLoss;
pub use redundancy::IntraRedundancyLoss;
pub use task::SurrogateReconstructionObjective;
pub(crate) use task::{build_primary_objective, compute_primary_objective_batch};
