//! Legacy compatibility surface.
//!
//! This module gathers the pre-research-stack APIs that are still useful for
//! demos, comparisons, and backwards compatibility. New config-driven research
//! code should prefer `crate::config`, `crate::data`, `crate::models`,
//! `crate::training`, and `crate::experiments`.
//!
//! The crate continues to re-export many of these items at the root for
//! compatibility, but `crate::legacy::*` is the clearer namespace for old flows.

pub mod comparison;
pub mod demo;

pub use crate::dataset::{
    DatasetDownloader, DatasetError, LigandReader, PDBbindConfig, PocketReader,
};
pub use crate::experiment::{
    ComparisonExperiment, ComparisonResult, ExperimentConfig, MethodResult,
};
pub use crate::pocket::create_example_prrsv_pocket;
pub use crate::{CandidateMolecule, GenerationResult, PocketDiffusionPipeline};
pub use comparison::*;
pub use demo::*;
