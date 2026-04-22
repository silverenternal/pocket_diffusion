//! Ablatable loss modules for the modular research stack.

pub mod consistency;
pub mod gate;
pub mod leakage;
pub mod probe;
pub mod redundancy;
pub mod task;

pub use consistency::ConsistencyLoss;
pub use gate::GateLoss;
pub use leakage::LeakageLoss;
pub use probe::ProbeLoss;
pub use redundancy::IntraRedundancyLoss;
pub(crate) use task::build_primary_objective;
pub use task::SurrogateReconstructionObjective;
