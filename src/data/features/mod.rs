//! Feature structures and builders used by the modular research pipeline.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use crate::{
    config::GenerationTargetConfig,
    types::{tensor_from_slice, AtomType, GenerationCorruptionMetadata, Ligand, Pocket},
};

include!("types.rs");
include!("example.rs");
include!("device.rs");
include!("builders.rs");
include!("supervision.rs");
include!("tests.rs");
