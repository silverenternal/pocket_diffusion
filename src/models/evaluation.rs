//! Heuristic candidate generation and evaluation helpers for the modular research path.
//!
//! Candidate generation, evaluator adapters, scoring helpers, and tests are
//! split under this facade to preserve existing imports.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tch::{Kind, Tensor};

use serde::{Deserialize, Serialize};

use super::{
    ChemistryValidityEvaluator, DockingEvaluator, ExternalEvaluationReport, ExternalMetricRecord,
    GeneratedCandidateRecord, PocketCompatibilityEvaluator, ResearchForward,
};
use crate::config::ExternalBackendCommandConfig;
use crate::{
    data::MolecularExample,
    types::{Atom, AtomType, CandidateMolecule, Ligand},
};

include!("evaluation/evaluators.rs");
include!("evaluation/candidates.rs");
include!("evaluation/scoring.rs");
include!("evaluation/tests.rs");
