//! Heuristic candidate generation and evaluation helpers for the modular research path.
//!
//! Candidate generation, evaluator adapters, scoring helpers, and drug-metric
//! interpretation are split into explicit submodules to keep the evaluation
//! surface ablation-friendly.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

use super::{
    ChemistryValidityEvaluator, DockingEvaluator, ExternalEvaluationReport, ExternalMetricRecord,
    GeneratedCandidateRecord, PocketCompatibilityEvaluator, ResearchForward,
};
use crate::config::ExternalBackendCommandConfig;
use crate::{
    data::MolecularExample,
    types::{Atom, AtomType, CandidateMolecule, Ligand},
};

mod candidates;
mod drug_metrics;
mod evaluators;
mod scoring;
#[cfg(test)]
mod tests;

pub(crate) use candidates::{candidate_records_to_legacy, report_to_metrics};
pub use drug_metrics::{
    classify_drug_metric, DrugMetricDirection, DrugMetricDomain, DrugMetricDomainSummary,
    DrugMetricGuardrailFailure, DrugMetricObservation, DrugMetricPanel, DrugMetricSpec,
};
pub(crate) use evaluators::{
    generate_claim_facing_candidates_from_forward, generate_layered_candidates_from_forward,
    generate_layered_candidates_with_options, CandidateGenerationLayers,
};
pub use evaluators::{
    CommandChemistryValidityEvaluator, CommandDockingEvaluator,
    CommandPocketCompatibilityEvaluator, HeuristicChemistryValidityEvaluator,
    HeuristicDockingEvaluator, HeuristicPocketCompatibilityEvaluator,
};
