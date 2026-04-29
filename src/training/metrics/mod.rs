//! Training metrics and persisted run summaries.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::config::ResearchConfig;
use crate::training::checkpoint::ResumeMode;
use crate::{
    data::{DatasetValidationReport, InMemoryDataset},
    experiments::EvaluationMetrics,
};

/// Version tag for the persisted metric schema.
pub const METRIC_SCHEMA_VERSION: u32 = 14;
/// Version tag for the shared run artifact bundle schema.
pub const ARTIFACT_BUNDLE_SCHEMA_VERSION: u32 = 1;
/// Human-readable resume contract identifier for the current research path.
pub const RESUME_CONTRACT_VERSION: &str = "weights+history+step";

include!("losses.rs");
include!("split_types.rs");
include!("artifacts.rs");
include!("split_impl.rs");
include!("reproducibility.rs");
