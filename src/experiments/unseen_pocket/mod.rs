//! Unseen-pocket experiment loop, ablations, and evaluation summaries.
//!
//! This facade preserves the original `experiments::unseen_pocket` API while
//! keeping the major implementation regions in separate files.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tch::nn;

use crate::{
    config::{
        AutomatedSearchConfig, AutomatedSearchHardGateConfig, AutomatedSearchScoreWeightConfig,
        AutomatedSearchSpaceConfig, AutomatedSearchStrategy, CrossAttentionMode,
        ExternalBackendCommandConfig, ResearchConfig,
    },
    data::InMemoryDataset,
    models::{
        build_bounded_preference_pairs, extract_interaction_profiles, flatten_layered_output,
        report_to_metrics, summarize_method_output, ChemistryValidityEvaluator,
        CommandChemistryValidityEvaluator, CommandDockingEvaluator,
        CommandPocketCompatibilityEvaluator, DockingEvaluator, ExternalEvaluationReport,
        ExternalMetricRecord, GeneratedCandidateRecord, HeuristicChemistryValidityEvaluator,
        HeuristicDockingEvaluator, HeuristicPocketCompatibilityEvaluator, LayeredGenerationOutput,
        MethodComparisonRow, Phase1ResearchSystem, PocketCompatibilityEvaluator,
        PocketGenerationContext, PocketGenerationMethodMetadata, PocketGenerationMethodRegistry,
        PreferenceConstructionConfig, PreferencePairArtifact, PreferenceProfileArtifact,
        PreferenceReranker, PreferenceRerankerSummaryArtifact, ResearchForward,
        RuleBasedPreferenceReranker,
    },
    runtime::parse_runtime_device,
    training::{
        reproducibility_metadata, stable_json_hash, ResearchTrainer, RunArtifactBundle,
        RunArtifactPaths, RunKind, SplitReport, StepMetrics,
    },
};

include!("config.rs");
include!("search.rs");
include!("metrics.rs");
include!("run.rs");
include!("evaluation.rs");
include!("claims.rs");
include!("tests.rs");
