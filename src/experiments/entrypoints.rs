//! Config-driven entrypoints for modular research experiments.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::experiments::unseen_pocket::{run_automated_search, run_multi_seed_experiment};
use crate::experiments::{
    load_experiment_config, AblationMatrixSummary, AutomatedSearchSummary,
    MultiSeedExperimentSummary, RealGenerationMetrics, UnseenPocketExperiment,
    UnseenPocketExperimentSummary,
};
use crate::{
    config::load_research_config,
    data::InMemoryDataset,
    models::{
        generate_layered_candidates_from_forward,
        generate_layered_candidates_from_generation_samples, report_to_metrics,
        ChemistryValidityEvaluator, DockingEvaluator, HeuristicChemistryValidityEvaluator,
        HeuristicDockingEvaluator, HeuristicPocketCompatibilityEvaluator, Phase1ResearchSystem,
        PocketCompatibilityEvaluator,
    },
    training::CheckpointManager,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tch::nn;

/// Execute the unseen-pocket experiment from a JSON config path.
pub fn run_experiment_from_config(
    path: impl AsRef<Path>,
    resume: bool,
) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
    let config_path = path.as_ref().to_path_buf();
    let config = load_experiment_config(&config_path)?;
    crate::experiments::unseen_pocket::validate_experiment_config_with_source(
        &config,
        &config_path,
    )?;
    UnseenPocketExperiment::run_with_options(config, resume)
}

/// Execute the configured ablation matrix and return the persisted comparison summary.
pub fn run_ablation_matrix_from_config(
    path: impl AsRef<Path>,
) -> Result<AblationMatrixSummary, Box<dyn std::error::Error>> {
    let config_path = path.as_ref().to_path_buf();
    let mut config = load_experiment_config(&config_path)?;
    config.ablation_matrix.enabled = true;
    crate::experiments::unseen_pocket::validate_experiment_config_with_source(
        &config,
        &config_path,
    )?;
    let summary = UnseenPocketExperiment::run_with_options(config, false)?;
    summary
        .ablation_matrix
        .ok_or_else(|| "ablation matrix was enabled but no summary was emitted".into())
}

/// Execute automated cross-surface tuning from a JSON experiment config path.
pub fn run_automated_search_from_config(
    path: impl AsRef<Path>,
) -> Result<AutomatedSearchSummary, Box<dyn std::error::Error>> {
    run_automated_search(path)
}

/// Execute the configured multi-seed experiment runner from a JSON config path.
pub fn run_multi_seed_experiment_from_config(
    path: impl AsRef<Path>,
) -> Result<MultiSeedExperimentSummary, Box<dyn std::error::Error>> {
    run_multi_seed_experiment(path)
}

/// Summary for the config-driven modular generation demo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchGenerationDemoSummary {
    /// Example identifier used for conditioned generation.
    pub example_id: String,
    /// Protein identifier used for conditioned generation.
    pub protein_id: String,
    /// Number of constrained-flow demo candidates emitted by the generation path.
    pub candidate_count: usize,
    /// Number of raw-flow candidates persisted for the same example.
    pub raw_candidate_count: usize,
    /// Number of rollout steps executed for the selected example.
    pub rollout_steps: usize,
    /// Full per-step refinement trace for the selected example.
    pub rollout: crate::models::GenerationRolloutRecord,
    /// Whether a latest checkpoint was loaded before generation.
    pub loaded_checkpoint: bool,
    /// Path to persisted raw-flow demo candidates.
    pub raw_candidate_path: PathBuf,
    /// Path to persisted constrained-flow demo candidates.
    pub constrained_candidate_path: PathBuf,
    /// Chemistry, docking, and pocket-compatibility metrics over raw-flow candidates.
    pub raw_generation_metrics: DemoGenerationMetrics,
    /// Chemistry, docking, and pocket-compatibility metrics over constrained-flow candidates.
    pub constrained_generation_metrics: DemoGenerationMetrics,
}

/// Candidate-metric block with explicit layer and evidence-role attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoGenerationMetric {
    /// Candidate layer used to compute this metric block.
    pub candidate_layer: String,
    /// Evidence role for this metric block.
    pub evidence_role: String,
    /// Whether this metric backend is enabled for the run.
    pub available: bool,
    /// Backend identifier if available.
    pub backend_name: Option<String>,
    /// Metric payload emitted by the backend/heuristic.
    pub metrics: BTreeMap<String, f64>,
    /// Human-readable status for compatibility provenance.
    pub status: String,
}

impl DemoGenerationMetric {
    fn from_reserved(
        metric: crate::experiments::ReservedBackendMetrics,
        candidate_layer: &str,
        evidence_role: &str,
    ) -> Self {
        Self {
            candidate_layer: candidate_layer.to_string(),
            evidence_role: evidence_role.to_string(),
            available: metric.available,
            backend_name: metric.backend_name,
            metrics: metric.metrics,
            status: metric.status,
        }
    }

    pub fn to_reserved(&self) -> crate::experiments::ReservedBackendMetrics {
        crate::experiments::ReservedBackendMetrics {
            available: self.available,
            backend_name: self.backend_name.clone(),
            metrics: self.metrics.clone(),
            status: self.status.clone(),
        }
    }
}

/// Layer-bound demo metrics with explicit evidence attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoGenerationMetrics {
    pub chemistry_validity: DemoGenerationMetric,
    pub docking_affinity: DemoGenerationMetric,
    pub pocket_compatibility: DemoGenerationMetric,
}

impl DemoGenerationMetrics {
    pub fn to_reserved(&self) -> RealGenerationMetrics {
        RealGenerationMetrics {
            chemistry_validity: self.chemistry_validity.to_reserved(),
            docking_affinity: self.docking_affinity.to_reserved(),
            pocket_compatibility: self.pocket_compatibility.to_reserved(),
        }
    }
}

/// Summary for an all-example layered generation artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredGenerationRunSummary {
    /// Path of the persisted generation layer artifact.
    pub artifact_path: PathBuf,
    /// Number of conditioning examples processed.
    pub example_count: usize,
    /// Number of raw flow candidates persisted.
    pub raw_flow_candidate_count: usize,
    /// Number of constrained flow candidates persisted.
    pub constrained_flow_candidate_count: usize,
    /// Whether model weights were loaded from the configured latest checkpoint.
    pub loaded_checkpoint: bool,
}

/// Run a small config-driven modular generation demo.
pub fn run_generation_demo_from_config(
    path: impl AsRef<Path>,
    resume: bool,
    example_id: Option<&str>,
    num_candidates: usize,
) -> Result<ResearchGenerationDemoSummary, Box<dyn std::error::Error>> {
    let config = load_research_config(path.as_ref().to_path_buf())?;
    config.validate()?;
    config.runtime.apply_tch_thread_settings();

    let loaded = InMemoryDataset::load_from_config(&config.data)?;
    let dataset = loaded
        .dataset
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    let splits = dataset.split_by_protein_fraction_with_options(
        config.data.val_fraction,
        config.data.test_fraction,
        config.data.split_seed,
        config.data.stratify_by_measurement,
    );

    let example = select_demo_example(
        splits.test.examples(),
        splits.val.examples(),
        splits.train.examples(),
        example_id,
    )?
    .to_device(config.runtime.resolve_device()?);

    let mut var_store = nn::VarStore::new(config.runtime.resolve_device()?);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let checkpoint_manager = CheckpointManager::new(config.training.checkpoint_dir.clone());
    let loaded_checkpoint = if resume {
        checkpoint_manager.load_latest(&mut var_store)?.is_some()
    } else {
        false
    };

    let forwards = system.forward_example_generation_samples(&example);
    let forward = forwards
        .first()
        .expect("generation sample forward should be present");
    let layers = if forwards.len() > 1 {
        generate_layered_candidates_from_generation_samples(
            &example,
            &forwards,
            num_candidates.max(1),
        )
    } else {
        generate_layered_candidates_from_forward(&example, forward, num_candidates.max(1))
    };
    let raw_generation_metrics =
        evaluate_demo_metrics(&layers.raw_rollout, "raw_flow", "raw_model_native");
    let constrained_generation_metrics = evaluate_demo_metrics(
        &layers.inferred_bond,
        "constrained_flow",
        "constrained_sampling",
    );

    let raw_candidate_path = config
        .training
        .checkpoint_dir
        .join("generation_demo_candidates_raw.json");
    let constrained_candidate_path = config
        .training
        .checkpoint_dir
        .join("generation_demo_candidates_constrained_flow.json");
    let compatibility_candidate_path = config
        .training
        .checkpoint_dir
        .join("generation_demo_candidates.json");
    let raw_candidates = enrich_candidates(
        layers.raw_rollout,
        "raw_flow",
        "raw_flow",
        true,
        false,
        &[],
        "raw_model_capability",
    )?;
    let constrained_candidates = enrich_candidates(
        layers.inferred_bond,
        "constrained_flow",
        "raw_flow",
        false,
        true,
        &["geometry_repair", "bond_inference"],
        "constrained_sampling_only",
    )?;

    fs::create_dir_all(&config.training.checkpoint_dir)?;
    fs::write(
        &raw_candidate_path,
        serde_json::to_string_pretty(&raw_candidates)?,
    )?;
    fs::write(
        &constrained_candidate_path,
        serde_json::to_string_pretty(&constrained_candidates)?,
    )?;
    // Keep a compatibility path for existing tooling. This file contains post-processed
    // constrained-flow candidates to preserve the old demo UX.
    fs::write(
        &compatibility_candidate_path,
        serde_json::to_string_pretty(&constrained_candidates)?,
    )?;

    let summary = ResearchGenerationDemoSummary {
        example_id: example.example_id.clone(),
        protein_id: example.protein_id.clone(),
        candidate_count: constrained_candidates.len(),
        raw_candidate_count: raw_candidates.len(),
        rollout_steps: forward.generation.rollout.executed_steps,
        rollout: forward.generation.rollout.clone(),
        loaded_checkpoint,
        raw_candidate_path,
        constrained_candidate_path,
        raw_generation_metrics,
        constrained_generation_metrics,
    };

    fs::write(
        config
            .training
            .checkpoint_dir
            .join("generation_demo_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

fn evaluate_demo_metrics(
    candidates: &[crate::models::GeneratedCandidateRecord],
    candidate_layer: &str,
    evidence_role: &str,
) -> DemoGenerationMetrics {
    DemoGenerationMetrics {
        chemistry_validity: DemoGenerationMetric::from_reserved(
            report_to_metrics(
                HeuristicChemistryValidityEvaluator.evaluate_chemistry(candidates),
                &format!(
                    "active heuristic chemistry-validity hook on {candidate_layer} demo candidates"
                ),
            ),
            candidate_layer,
            evidence_role,
        ),
        docking_affinity: DemoGenerationMetric::from_reserved(
            report_to_metrics(
                HeuristicDockingEvaluator.evaluate_docking(candidates),
                &format!(
                    "active heuristic docking-oriented hook on {candidate_layer} demo candidates"
                ),
            ),
            candidate_layer,
            evidence_role,
        ),
        pocket_compatibility: DemoGenerationMetric::from_reserved(
            report_to_metrics(
                HeuristicPocketCompatibilityEvaluator.evaluate_pocket_compatibility(candidates),
                &format!(
                    "active heuristic pocket-compatibility hook on {candidate_layer} demo candidates"
                ),
            ),
            candidate_layer,
            evidence_role,
        ),
    }
}

/// Run one native forward pass for every configured example and persist Q2 layers.
pub fn run_generation_layers_from_config(
    path: impl AsRef<Path>,
    resume: bool,
    split_label: &str,
    num_candidates: usize,
) -> Result<LayeredGenerationRunSummary, Box<dyn std::error::Error>> {
    let config_path = path.as_ref().to_path_buf();
    let config = load_research_config(config_path.clone())?;
    config.validate()?;
    config.runtime.apply_tch_thread_settings();

    let loaded = InMemoryDataset::load_from_config(&config.data)?;
    let dataset = loaded
        .dataset
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    let device = config.runtime.resolve_device()?;
    let mut var_store = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let checkpoint_manager = CheckpointManager::new(config.training.checkpoint_dir.clone());
    let loaded_checkpoint = if resume {
        checkpoint_manager.load_latest(&mut var_store)?.is_some()
    } else {
        false
    };

    let candidate_limit = num_candidates.max(1);
    let mut raw_flow = Vec::new();
    let mut constrained_flow = Vec::new();
    for example in dataset.examples() {
        let example = example.to_device(device);
        let forwards = system.forward_example_generation_samples(&example);
        let layers = if forwards.len() > 1 {
            generate_layered_candidates_from_generation_samples(
                &example,
                &forwards,
                candidate_limit,
            )
        } else {
            generate_layered_candidates_from_forward(&example, &forwards[0], candidate_limit)
        };
        raw_flow.extend(enrich_candidates(
            layers.raw_rollout,
            "raw_flow",
            "raw_flow",
            true,
            false,
            &[],
            "raw_model_capability",
        )?);
        constrained_flow.extend(enrich_candidates(
            layers.inferred_bond,
            "constrained_flow",
            "raw_flow",
            false,
            true,
            &["geometry_repair", "bond_inference"],
            "constrained_sampling_only",
        )?);
    }

    fs::create_dir_all(&config.training.checkpoint_dir)?;
    let artifact_path = config
        .training
        .checkpoint_dir
        .join(format!("generation_layers_{split_label}.json"));
    let artifact = json!({
        "schema_version": 3,
        "artifact_name": "q2_ours_public100_generation_layers",
        "split_label": split_label,
        "method_id": "flow_matching",
        "source_config": config_path.display().to_string(),
        "source_manifest": config.data.manifest_path.as_ref().map(|path| path.display().to_string()),
        "loaded_checkpoint": loaded_checkpoint,
        "candidate_budget_per_example": candidate_limit,
        "example_count": dataset.examples().len(),
        "raw_flow_candidates": raw_flow,
        "constrained_flow_candidates": constrained_flow,
        "claim_boundary": "raw_flow is native geometry-first flow evidence; constrained_flow includes deterministic geometry repair and bond inference and must not be used as raw model-native evidence."
    });
    fs::write(&artifact_path, serde_json::to_string_pretty(&artifact)?)?;

    Ok(LayeredGenerationRunSummary {
        artifact_path,
        example_count: dataset.examples().len(),
        raw_flow_candidate_count: artifact["raw_flow_candidates"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0),
        constrained_flow_candidate_count: artifact["constrained_flow_candidates"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0),
        loaded_checkpoint,
    })
}

fn enrich_candidates(
    candidates: Vec<crate::models::GeneratedCandidateRecord>,
    layer: &str,
    source_layer: &str,
    model_native: bool,
    constrained_sampling: bool,
    postprocessing_steps: &[&str],
    claim_allowed: &str,
) -> Result<Vec<Value>, serde_json::Error> {
    candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            let mut value = serde_json::to_value(&candidate)?;
            let candidate_id = format!("flow_matching:{layer}:{}:{index}", candidate.example_id);
            if let Some(object) = value.as_object_mut() {
                object.insert("candidate_id".to_string(), json!(candidate_id));
                object.insert("method_id".to_string(), json!("flow_matching"));
                object.insert("layer".to_string(), json!(layer));
                object.insert("source_layer".to_string(), json!(source_layer));
                object.insert("model_native".to_string(), json!(model_native));
                object.insert(
                    "coordinate_frame_contract".to_string(),
                    json!("candidate.coords are ligand-centered model-frame coordinates; coordinate_frame_origin reconstructs source-frame coordinates"),
                );
                object.insert(
                    "constrained_sampling".to_string(),
                    json!(constrained_sampling),
                );
                object.insert(
                    "postprocessing_steps".to_string(),
                    json!(postprocessing_steps),
                );
                object.insert("claim_allowed".to_string(), json!(claim_allowed));
                object.insert(
                    "transformation_provenance".to_string(),
                    json!({
                        "algorithm": "native_rust_generation_layers",
                        "source_layer": source_layer,
                        "source_candidate_id": if layer == "raw_flow" {
                            candidate_id.clone()
                        } else {
                            format!("flow_matching:{source_layer}:{}:{index}", candidate.example_id)
                        },
                        "transformations": postprocessing_steps,
                    }),
                );
            }
            Ok(value)
        })
        .collect()
}

fn select_demo_example<'a>(
    test: &'a [crate::data::MolecularExample],
    val: &'a [crate::data::MolecularExample],
    train: &'a [crate::data::MolecularExample],
    requested_example_id: Option<&str>,
) -> Result<&'a crate::data::MolecularExample, Box<dyn std::error::Error>> {
    let all = test
        .iter()
        .chain(val.iter())
        .chain(train.iter())
        .collect::<Vec<_>>();
    if let Some(example_id) = requested_example_id {
        return all
            .into_iter()
            .find(|example| example.example_id == example_id)
            .ok_or_else(|| {
                format!("example_id `{example_id}` not found in configured dataset").into()
            });
    }

    test.first()
        .or_else(|| val.first())
        .or_else(|| train.first())
        .ok_or_else(|| "configured dataset is empty".into())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{config::ResearchConfig, experiments::UnseenPocketExperimentConfig};

    #[test]
    fn config_driven_experiment_writes_summary() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = UnseenPocketExperimentConfig::default();
        config.research.runtime.device = "cpu".to_string();
        config.research.data.batch_size = 2;
        config.research.training.max_steps = 2;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 1;
        config.research.training.schedule.stage3_steps = 2;
        config.research.training.log_every = 100;
        config.research.training.checkpoint_every = 100;
        config.research.training.checkpoint_dir = temp.path().join("checkpoints");

        let config_path = temp.path().join("experiment_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let summary = run_experiment_from_config(&config_path, false).unwrap();

        assert_eq!(summary.training_history.len(), 2);
        assert!(
            summary
                .validation
                .representation_diagnostics
                .finite_forward_fraction
                >= 0.0
        );
        assert!(
            summary
                .test
                .representation_diagnostics
                .finite_forward_fraction
                >= 0.0
        );
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("experiment_summary.json")
            .exists());
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("claim_summary.json")
            .exists());
    }

    #[test]
    fn conditioned_denoising_experiment_reports_finite_forward_summaries() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = UnseenPocketExperimentConfig::default();
        config.research.runtime.device = "cpu".to_string();
        config.research.data.root_dir = "./examples/datasets/mini_pdbbind".into();
        config.research.data.dataset_format = crate::config::DatasetFormat::ManifestJson;
        config.research.data.manifest_path =
            Some("./examples/datasets/mini_pdbbind/manifest.json".into());
        config.research.data.label_table_path =
            Some("./examples/datasets/mini_pdbbind/affinity_labels.csv".into());
        config.research.data.batch_size = 2;
        config.research.data.val_fraction = 0.25;
        config.research.data.test_fraction = 0.25;
        config.research.data.stratify_by_measurement = true;
        config.research.training.max_steps = 3;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 2;
        config.research.training.schedule.stage3_steps = 3;
        config.research.training.log_every = 100;
        config.research.training.checkpoint_every = 100;
        config.research.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising;
        config.research.training.checkpoint_dir = temp.path().join("checkpoints");

        let config_path = temp.path().join("conditioned_experiment.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let summary = run_experiment_from_config(&config_path, false).unwrap();

        assert_eq!(
            summary.config.research.training.primary_objective,
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising
        );
        assert_eq!(
            summary
                .training_history
                .last()
                .unwrap()
                .losses
                .primary
                .objective_name,
            "conditioned_denoising"
        );
        assert_eq!(
            summary
                .validation
                .representation_diagnostics
                .finite_forward_fraction,
            1.0
        );
        assert_eq!(
            summary
                .test
                .representation_diagnostics
                .finite_forward_fraction,
            1.0
        );
    }

    #[test]
    fn generation_demo_writes_candidate_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = ResearchConfig::default();
        config.runtime.device = "cpu".to_string();
        config.data.batch_size = 2;
        config.training.checkpoint_dir = temp.path().join("checkpoints");

        let config_path = temp.path().join("generation_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let summary = run_generation_demo_from_config(&config_path, false, None, 3).unwrap();

        assert_eq!(summary.candidate_count, 3);
        assert_eq!(summary.raw_candidate_count, 3);
        assert_eq!(
            summary
                .raw_generation_metrics
                .chemistry_validity
                .candidate_layer,
            "raw_flow"
        );
        assert_eq!(
            summary
                .raw_generation_metrics
                .chemistry_validity
                .evidence_role,
            "raw_model_native"
        );
        assert_eq!(
            summary
                .constrained_generation_metrics
                .chemistry_validity
                .candidate_layer,
            "constrained_flow"
        );
        assert_eq!(
            summary
                .constrained_generation_metrics
                .chemistry_validity
                .evidence_role,
            "constrained_sampling"
        );
        assert!(summary.rollout_steps >= 1);
        assert!(summary.raw_candidate_path.exists());
        assert!(summary.constrained_candidate_path.exists());
        let raw_candidate_payload: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&summary.raw_candidate_path).unwrap())
                .unwrap();
        assert!(raw_candidate_payload
            .as_array()
            .unwrap()
            .iter()
            .all(|candidate| candidate["coordinate_frame_contract"]
                .as_str()
                .unwrap()
                .contains("coordinate_frame_origin")));
        assert!(config
            .training
            .checkpoint_dir
            .join("generation_demo_candidates.json")
            .exists());
        assert!(config
            .training
            .checkpoint_dir
            .join("generation_demo_summary.json")
            .exists());
    }
}
