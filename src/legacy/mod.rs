//! Legacy compatibility surface.
//!
//! This module gathers the pre-research-stack APIs that are still useful for
//! demos, comparisons, and backwards compatibility. New config-driven research
//! code should prefer `crate::config`, `crate::data`, `crate::models`,
//! `crate::training`, and `crate::experiments`.
//!
//! Legacy items stay discoverable here, rather than through the crate root.

#[allow(deprecated)]
pub mod comparison;
pub mod demo;

#[allow(deprecated)]
pub use crate::dataset::{
    DatasetDownloader, DatasetError, LigandReader, PDBbindConfig, PocketReader,
};
#[allow(deprecated)]
pub use crate::experiment::{
    ComparisonExperiment, ComparisonResult, ExperimentConfig, MethodResult,
};
pub use crate::pocket::create_example_prrsv_pocket;
pub use crate::types::{CandidateMolecule, GenerationResult};
#[allow(deprecated)]
pub use crate::PocketDiffusionPipeline;
pub use comparison::run_comparison_experiment;
pub use demo::run_legacy_demo;
