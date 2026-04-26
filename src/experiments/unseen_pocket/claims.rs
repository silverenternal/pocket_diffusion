// Claim, ablation, backend, and diagnostic helpers are split by responsibility
// while retaining the original module API and private helper visibility.
include!("claims/ablation.rs");
include!("claims/report.rs");
include!("claims/chemistry.rs");
include!("claims/backend.rs");
include!("claims/diagnostics.rs");
