//! Strongly typed configuration for Phase 1 of the research framework.
//!
//! The public `crate::config::types` surface is preserved while config domains
//! are split into smaller implementation files.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

include!("root.rs");
include!("data.rs");
include!("generation.rs");
include!("model.rs");
include!("search_runtime.rs");
include!("training.rs");
include!("tests.rs");
