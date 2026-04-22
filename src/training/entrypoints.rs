//! Config-driven entrypoints for the modular research training stack.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use tch::nn;

use crate::{
    config::{load_research_config, DatasetFormat},
    data::{Dataset, InMemoryDataset},
    experiments::{evaluate_split, AblationConfig, EvaluationMetrics},
    models::Phase1ResearchSystem,
};

use super::{DatasetSplitSizes, ResearchTrainer, StepMetrics, TrainingRunSummary};

/// One sample row emitted by dataset inspection.
#[derive(Debug, Clone)]
pub struct InspectionExample {
    /// Example identifier.
    pub example_id: String,
    /// Protein split key used for unseen-pocket splitting.
    pub protein_id: String,
    /// Number of ligand atoms in this example.
    pub ligand_atoms: i64,
    /// Number of pocket atoms in this example.
    pub pocket_atoms: i64,
    /// Width of the padded/truncated pocket atom features.
    pub pocket_feature_dim: i64,
    /// Optional normalized affinity target.
    pub affinity_kcal_mol: Option<f32>,
    /// Optional affinity measurement family.
    pub affinity_measurement_type: Option<String>,
    /// Optional raw affinity value.
    pub affinity_raw_value: Option<f32>,
    /// Optional raw affinity unit.
    pub affinity_raw_unit: Option<String>,
}

/// Compact inspection summary for a config-driven dataset.
#[derive(Debug, Clone)]
pub struct DatasetInspection {
    /// Configured dataset format.
    pub dataset_format: DatasetFormat,
    /// Total examples before splitting.
    pub total_examples: usize,
    /// Number of training examples.
    pub train_examples: usize,
    /// Number of validation examples.
    pub val_examples: usize,
    /// Number of test examples.
    pub test_examples: usize,
    /// Configured pocket feature width used to normalize examples.
    pub pocket_feature_dim: i64,
    /// Example previews for CLI inspection.
    pub examples: Vec<InspectionExample>,
}

/// Outputs produced by a config-driven training run.
#[derive(Debug, Clone)]
pub struct TrainingRunOutput {
    /// Persisted training summary.
    pub summary: TrainingRunSummary,
    /// Validation metrics evaluated after training.
    pub validation: EvaluationMetrics,
    /// Test metrics evaluated after training.
    pub test: EvaluationMetrics,
    /// Rolling latest checkpoint path for resume.
    pub latest_checkpoint_path: PathBuf,
}

/// Inspect a dataset using the modular research config path.
pub fn inspect_dataset_from_config(
    path: impl AsRef<Path>,
) -> Result<DatasetInspection, Box<dyn std::error::Error>> {
    let config = load_research_config(path.as_ref())?;
    let dataset = InMemoryDataset::from_data_config(&config.data)?
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    let splits = dataset.split_by_protein_fraction_with_options(
        config.data.val_fraction,
        config.data.test_fraction,
        config.data.split_seed,
        config.data.stratify_by_measurement,
    );

    let examples = dataset
        .examples()
        .iter()
        .take(3)
        .map(|example| InspectionExample {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            ligand_atoms: example.geometry.coords.size()[0],
            pocket_atoms: example.pocket.coords.size()[0],
            pocket_feature_dim: example.pocket.atom_features.size()[1],
            affinity_kcal_mol: example.targets.affinity_kcal_mol,
            affinity_measurement_type: example
                .targets
                .affinity_measurement_type
                .as_ref()
                .map(ToOwned::to_owned),
            affinity_raw_value: example.targets.affinity_raw_value,
            affinity_raw_unit: example
                .targets
                .affinity_raw_unit
                .as_ref()
                .map(ToOwned::to_owned),
        })
        .collect();

    Ok(DatasetInspection {
        dataset_format: config.data.dataset_format,
        total_examples: dataset.len(),
        train_examples: splits.train.len(),
        val_examples: splits.val.len(),
        test_examples: splits.test.len(),
        pocket_feature_dim: config.model.pocket_feature_dim,
        examples,
    })
}

/// Execute config-driven modular training with optional checkpoint resume.
pub fn run_training_from_config(
    path: impl AsRef<Path>,
    resume: bool,
) -> Result<TrainingRunOutput, Box<dyn std::error::Error>> {
    let config = load_research_config(path.as_ref())?;
    let dataset = InMemoryDataset::from_data_config(&config.data)?
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    let splits = dataset.split_by_protein_fraction_with_options(
        config.data.val_fraction,
        config.data.test_fraction,
        config.data.split_seed,
        config.data.stratify_by_measurement,
    );

    let device = config.runtime.resolve_device()?;
    let mut var_store = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let mut trainer = ResearchTrainer::new(&var_store, config.clone())?;
    let mut resumed_from_step = None;
    if resume {
        if let Some(checkpoint) = trainer.resume_from_latest(&mut var_store)? {
            resumed_from_step = Some(checkpoint.metadata.step);
            if let Some(history) =
                load_training_history(&config.training.checkpoint_dir, checkpoint.metadata.step)?
            {
                trainer.replace_history(history);
            }
            log::info!(
                "resumed training from step {} at {}",
                checkpoint.metadata.step,
                checkpoint.weights_path.display()
            );
        } else {
            log::info!(
                "no checkpoint found under {}; starting fresh",
                config.training.checkpoint_dir.display()
            );
        }
    }

    let train_examples: Vec<_> = splits
        .train
        .examples()
        .iter()
        .map(|example| example.to_device(device))
        .collect();
    trainer.fit(&var_store, &system, &train_examples)?;

    let train_proteins: BTreeSet<&str> = splits
        .train
        .examples()
        .iter()
        .map(|example| example.protein_id.as_str())
        .collect();
    let validation = evaluate_split(
        &system,
        splits.val.examples(),
        &train_proteins,
        AblationConfig::default(),
        device,
    );
    let test = evaluate_split(
        &system,
        splits.test.examples(),
        &train_proteins,
        AblationConfig::default(),
        device,
    );

    let summary = TrainingRunSummary {
        config: config.clone(),
        splits: DatasetSplitSizes {
            total: dataset.len(),
            train: splits.train.len(),
            val: splits.val.len(),
            test: splits.test.len(),
        },
        resumed_from_step,
        summary_path: config.training.checkpoint_dir.join("training_summary.json"),
        training_history: trainer.history().to_vec(),
        validation: validation.clone(),
        test: test.clone(),
    };
    write_training_summary(&summary)?;

    Ok(TrainingRunOutput {
        latest_checkpoint_path: config.training.checkpoint_dir.join("latest.ot"),
        summary,
        validation,
        test,
    })
}

fn write_training_summary(
    summary: &TrainingRunSummary,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(&summary.config.training.checkpoint_dir)?;
    fs::write(
        &summary.summary_path,
        serde_json::to_string_pretty(summary)?,
    )?;
    Ok(summary.summary_path.clone())
}

fn load_training_history(
    checkpoint_dir: &Path,
    resumed_step: usize,
) -> Result<Option<Vec<StepMetrics>>, Box<dyn std::error::Error>> {
    let summary_path = checkpoint_dir.join("training_summary.json");
    if !summary_path.exists() {
        return Ok(None);
    }

    let summary: TrainingRunSummary = serde_json::from_str(&fs::read_to_string(summary_path)?)?;
    Ok(Some(
        summary
            .training_history
            .into_iter()
            .filter(|metrics| metrics.step <= resumed_step)
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ResearchConfig;

    #[test]
    fn inspect_dataset_reports_configured_pocket_dim() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = ResearchConfig::default();
        config.data.dataset_format = DatasetFormat::Synthetic;
        config.model.pocket_feature_dim = 11;

        let config_path = temp.path().join("research_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let inspection = inspect_dataset_from_config(&config_path).unwrap();

        assert_eq!(inspection.pocket_feature_dim, 11);
        assert!(!inspection.examples.is_empty());
        assert_eq!(inspection.examples[0].pocket_feature_dim, 11);
    }

    #[test]
    fn config_driven_training_writes_summary_with_eval_metrics() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = ResearchConfig::default();
        config.runtime.device = "cpu".to_string();
        config.data.dataset_format = DatasetFormat::Synthetic;
        config.data.batch_size = 2;
        config.data.val_fraction = 0.25;
        config.data.test_fraction = 0.25;
        config.training.max_steps = 2;
        config.training.log_every = 100;
        config.training.checkpoint_every = 100;
        config.training.checkpoint_dir = temp.path().join("checkpoints");

        let config_path = temp.path().join("research_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let output = run_training_from_config(&config_path, false).unwrap();

        let summary: TrainingRunSummary =
            serde_json::from_str(&fs::read_to_string(&output.summary.summary_path).unwrap())
                .unwrap();

        assert_eq!(summary.training_history.len(), 2);
        assert_eq!(summary.splits.total, 4);
        assert_eq!(summary.splits.val, 1);
        assert_eq!(summary.splits.test, 1);
        assert!(summary.validation.validity >= 0.0);
        assert!(summary.test.validity >= 0.0);
        assert_eq!(
            output.latest_checkpoint_path,
            config.training.checkpoint_dir.join("latest.ot")
        );
    }

    #[test]
    fn resumed_training_preserves_prior_history_in_summary() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint_dir = temp.path().join("checkpoints");

        let mut config = ResearchConfig::default();
        config.runtime.device = "cpu".to_string();
        config.data.dataset_format = DatasetFormat::Synthetic;
        config.data.batch_size = 2;
        config.data.val_fraction = 0.25;
        config.data.test_fraction = 0.25;
        config.training.max_steps = 2;
        config.training.log_every = 100;
        config.training.checkpoint_every = 1;
        config.training.checkpoint_dir = checkpoint_dir.clone();

        let first_config_path = temp.path().join("research_config_first.json");
        fs::write(
            &first_config_path,
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
        let _ = run_training_from_config(&first_config_path, false).unwrap();

        config.training.max_steps = 4;
        let resumed_config_path = temp.path().join("research_config_resumed.json");
        fs::write(
            &resumed_config_path,
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        let output = run_training_from_config(&resumed_config_path, true).unwrap();
        let summary: TrainingRunSummary =
            serde_json::from_str(&fs::read_to_string(&output.summary.summary_path).unwrap())
                .unwrap();

        assert_eq!(summary.resumed_from_step, Some(1));
        assert_eq!(summary.training_history.len(), 4);
        assert_eq!(summary.training_history[0].step, 0);
        assert_eq!(summary.training_history[1].step, 1);
        assert_eq!(summary.training_history[2].step, 2);
        assert_eq!(summary.training_history[3].step, 3);
    }
}
