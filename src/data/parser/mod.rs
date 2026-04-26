//! Parsers and discovery helpers for molecular dataset ingestion.
//!
//! This facade keeps the original parser API stable while separating manifest,
//! affinity-label, structure-file, and fixture/test code.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{ExampleTargets, MolecularExample};
use crate::config::ParsingMode;
use crate::types::{Atom, AtomType, Ligand, Pocket};

include!("manifest.rs");
include!("affinity.rs");
include!("discovery.rs");
include!("sdf.rs");
include!("pdb.rs");
include!("tests.rs");
