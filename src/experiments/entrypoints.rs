//! Config-driven entrypoints for modular research experiments.

use std::fs;
use std::path::Path;

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
        generate_candidates_from_forward, report_to_metrics, ChemistryValidityEvaluator,
        DockingEvaluator, HeuristicChemistryValidityEvaluator, HeuristicDockingEvaluator,
        HeuristicPocketCompatibilityEvaluator, Phase1ResearchSystem, PocketCompatibilityEvaluator,
    },
    training::CheckpointManager,
};
use serde::{Deserialize, Serialize};
use tch::nn;

/// Execute the unseen-pocket experiment from a JSON config path.
pub fn run_experiment_from_config(
    path: impl AsRef<Path>,
    resume: bool,
) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
    let config = load_experiment_config(path)?;
    UnseenPocketExperiment::run_with_options(config, resume)
}

/// Execute the configured ablation matrix and return the persisted comparison summary.
pub fn run_ablation_matrix_from_config(
    path: impl AsRef<Path>,
) -> Result<AblationMatrixSummary, Box<dyn std::error::Error>> {
    let mut config = load_experiment_config(path)?;
    config.ablation_matrix.enabled = true;
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
    /// Number of candidates emitted by the modular decoder path.
    pub candidate_count: usize,
    /// Number of rollout steps executed for the selected example.
    pub rollout_steps: usize,
    /// Full per-step refinement trace for the selected example.
    pub rollout: crate::models::GenerationRolloutRecord,
    /// Whether a latest checkpoint was loaded before generation.
    pub loaded_checkpoint: bool,
    /// Chemistry, docking, and pocket-compatibility metrics over emitted candidates.
    pub real_generation_metrics: RealGenerationMetrics,
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

    let forward = system.forward_example(&example);
    let candidates = generate_candidates_from_forward(&example, &forward, num_candidates.max(1));
    let chemistry = report_to_metrics(
        HeuristicChemistryValidityEvaluator.evaluate_chemistry(&candidates),
        "active heuristic chemistry-validity backend on modular rollout generation demo",
    );
    let docking = report_to_metrics(
        HeuristicDockingEvaluator.evaluate_docking(&candidates),
        "active heuristic docking-oriented hook on modular rollout generation demo",
    );
    let pocket = report_to_metrics(
        HeuristicPocketCompatibilityEvaluator.evaluate_pocket_compatibility(&candidates),
        "active heuristic pocket-compatibility hook on modular rollout generation demo",
    );

    fs::create_dir_all(&config.training.checkpoint_dir)?;
    fs::write(
        config
            .training
            .checkpoint_dir
            .join("generation_demo_candidates.json"),
        serde_json::to_string_pretty(&candidates)?,
    )?;

    let summary = ResearchGenerationDemoSummary {
        example_id: example.example_id.clone(),
        protein_id: example.protein_id.clone(),
        candidate_count: candidates.len(),
        rollout_steps: forward.generation.rollout.executed_steps,
        rollout: forward.generation.rollout.clone(),
        loaded_checkpoint,
        real_generation_metrics: RealGenerationMetrics {
            chemistry_validity: chemistry,
            docking_affinity: docking,
            pocket_compatibility: pocket,
        },
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
        assert!(summary.rollout_steps >= 1);
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
