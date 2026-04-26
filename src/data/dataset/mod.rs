//! Dataset traits and simple in-memory implementations.
//!
//! Loading, filtering, splitting, and tests are kept in separate files under a
//! compatibility facade.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use super::{
    apply_affinity_labels, discover_pdbbind_like_entries, load_affinity_labels, load_manifest,
    load_manifest_entry, synthetic_phase1_examples, DataParseError, DatasetValidationReport,
    MolecularExample,
};
use crate::config::{DataConfig, DataQualityFilterConfig, DatasetFormat};

include!("core.rs");
include!("quality.rs");
include!("splits.rs");
include!("tests.rs");
