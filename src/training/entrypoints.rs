//! Config-driven entrypoints for the modular research training stack.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use tch::nn;

use crate::{
    config::{load_research_config, DatasetFormat},
    data::{Dataset, DatasetValidationReport, InMemoryDataset},
    experiments::{evaluate_split, AblationConfig},
    models::Phase1ResearchSystem,
    training::{
        reproducibility_metadata, stable_json_hash, RunArtifactBundle, RunArtifactPaths, RunKind,
    },
};

use super::{DatasetSplitSizes, ResearchTrainer, SplitReport, StepMetrics, TrainingRunSummary};

/// One sample row emitted by dataset inspection.
#[derive(Debug, Clone)]
pub(crate) struct InspectionExample {
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
    /// Optional normalization provenance.
    pub affinity_normalization_provenance: Option<String>,
    /// Whether the affinity target is approximate.
    pub affinity_is_approximate: bool,
    /// Optional normalization warning.
    pub affinity_normalization_warning: Option<String>,
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
    pub(crate) examples: Vec<InspectionExample>,
    /// Machine-readable dataset validation artifact.
    pub validation: DatasetValidationReport,
    /// Persisted path for the dataset validation report.
    pub validation_report_path: PathBuf,
    /// Split-distribution and leakage audit.
    pub split_report: SplitReport,
}

/// Outputs produced by a config-driven training run.
/// Inspect a dataset using the modular research config path.
pub fn inspect_dataset_from_config(
    path: impl AsRef<Path>,
) -> Result<DatasetInspection, Box<dyn std::error::Error>> {
    let config = load_research_config(path.as_ref())?;
    config.validate()?;
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
            affinity_normalization_provenance: example
                .targets
                .affinity_normalization_provenance
                .as_ref()
                .map(ToOwned::to_owned),
            affinity_is_approximate: example.targets.affinity_is_approximate,
            affinity_normalization_warning: example
                .targets
                .affinity_normalization_warning
                .as_ref()
                .map(ToOwned::to_owned),
        })
        .collect();

    let split_report = SplitReport::from_datasets(&splits.train, &splits.val, &splits.test);
    let validation_report_path = write_dataset_validation_report(
        &config.training.checkpoint_dir,
        "dataset_validation_report.json",
        &loaded.validation,
    )?;

    Ok(DatasetInspection {
        dataset_format: config.data.dataset_format,
        total_examples: dataset.len(),
        train_examples: splits.train.len(),
        val_examples: splits.val.len(),
        test_examples: splits.test.len(),
        pocket_feature_dim: config.model.pocket_feature_dim,
        examples,
        validation: loaded.validation,
        validation_report_path,
        split_report,
    })
}

/// Execute config-driven modular training with optional checkpoint resume.
pub fn run_training_from_config(
    path: impl AsRef<Path>,
    resume: bool,
) -> Result<TrainingRunSummary, Box<dyn std::error::Error>> {
    let config = load_research_config(path.as_ref())?;
    config.validate()?;
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

    let device = config.runtime.resolve_device()?;
    let mut var_store = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let mut trainer = ResearchTrainer::new(&var_store, config.clone())?;
    trainer.set_dataset_validation_fingerprint(stable_json_hash(&loaded.validation));
    let mut resumed_from_step = None;
    let mut resumed_checkpoint_metadata = None;
    if resume {
        if let Some(checkpoint) = trainer.resume_from_latest(&mut var_store)? {
            resumed_from_step = Some(checkpoint.metadata.step);
            resumed_checkpoint_metadata = Some(checkpoint.metadata.clone());
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
        dataset_validation: loaded.validation.clone(),
        splits: DatasetSplitSizes {
            total: dataset.len(),
            train: splits.train.len(),
            val: splits.val.len(),
            test: splits.test.len(),
        },
        split_report: SplitReport::from_datasets(&splits.train, &splits.val, &splits.test),
        resumed_from_step,
        reproducibility: reproducibility_metadata(
            &config,
            &loaded.validation,
            resumed_checkpoint_metadata.as_ref(),
        ),
        training_history: trainer.history().to_vec(),
        validation: validation.clone(),
        test: test.clone(),
    };
    persist_training_artifacts(&summary)?;

    Ok(summary)
}

fn persist_training_artifacts(
    summary: &TrainingRunSummary,
) -> Result<RunArtifactBundle, Box<dyn std::error::Error>> {
    let artifact_dir = &summary.config.training.checkpoint_dir;
    fs::create_dir_all(artifact_dir)?;
    let config_snapshot = artifact_dir.join("config.snapshot.json");
    let dataset_validation_report = artifact_dir.join("dataset_validation_report.json");
    let split_report = artifact_dir.join("split_report.json");
    let run_summary = artifact_dir.join("training_summary.json");
    let run_bundle = artifact_dir.join("run_artifacts.json");

    fs::write(
        &config_snapshot,
        serde_json::to_string_pretty(&summary.config)?,
    )?;
    fs::write(
        &dataset_validation_report,
        serde_json::to_string_pretty(&summary.dataset_validation)?,
    )?;
    fs::write(
        &split_report,
        serde_json::to_string_pretty(&summary.split_report)?,
    )?;
    fs::write(&run_summary, serde_json::to_string_pretty(summary)?)?;

    let bundle = RunArtifactBundle {
        schema_version: summary.reproducibility.artifact_bundle_schema_version,
        run_kind: RunKind::Training,
        artifact_dir: artifact_dir.clone(),
        config_hash: summary.reproducibility.config_hash.clone(),
        dataset_validation_fingerprint: summary
            .reproducibility
            .dataset_validation_fingerprint
            .clone(),
        metric_schema_version: summary.reproducibility.metric_schema_version,
        paths: RunArtifactPaths {
            config_snapshot,
            dataset_validation_report,
            split_report,
            run_summary,
            run_bundle: run_bundle.clone(),
            latest_checkpoint: Some(artifact_dir.join("latest.ot")),
        },
    };
    fs::write(&run_bundle, serde_json::to_string_pretty(&bundle)?)?;
    Ok(bundle)
}

fn write_dataset_validation_report(
    checkpoint_dir: &Path,
    file_name: &str,
    report: &DatasetValidationReport,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(checkpoint_dir)?;
    let path = checkpoint_dir.join(file_name);
    fs::write(&path, serde_json::to_string_pretty(report)?)?;
    Ok(path)
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
        assert_eq!(inspection.validation.discovered_complexes, 4);
        assert!(inspection.validation_report_path.exists());
        assert!(
            !inspection
                .split_report
                .leakage_checks
                .protein_overlap_detected
        );
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
        config.training.schedule.stage1_steps = 1;
        config.training.schedule.stage2_steps = 1;
        config.training.schedule.stage3_steps = 2;
        config.training.log_every = 100;
        config.training.checkpoint_every = 100;
        config.training.checkpoint_dir = temp.path().join("checkpoints");

        let config_path = temp.path().join("research_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let summary = run_training_from_config(&config_path, false).unwrap();

        let summary: TrainingRunSummary = serde_json::from_str(
            &fs::read_to_string(
                summary
                    .config
                    .training
                    .checkpoint_dir
                    .join("training_summary.json"),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(summary.training_history.len(), 2);
        assert_eq!(summary.splits.total, 4);
        assert_eq!(summary.splits.val, 1);
        assert_eq!(summary.splits.test, 1);
        assert_eq!(summary.dataset_validation.parsed_examples, 4);
        assert!(summary
            .config
            .training
            .checkpoint_dir
            .join("dataset_validation_report.json")
            .exists());
        assert_eq!(summary.reproducibility.metric_schema_version, 3);
        assert!(summary
            .config
            .training
            .checkpoint_dir
            .join("run_artifacts.json")
            .exists());
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
        assert!(config.training.checkpoint_dir.join("latest.ot").exists());
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
        config.training.schedule.stage1_steps = 1;
        config.training.schedule.stage2_steps = 1;
        config.training.schedule.stage3_steps = 2;
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
        config.training.schedule.stage1_steps = 1;
        config.training.schedule.stage2_steps = 2;
        config.training.schedule.stage3_steps = 3;
        let resumed_config_path = temp.path().join("research_config_resumed.json");
        fs::write(
            &resumed_config_path,
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        let output = run_training_from_config(&resumed_config_path, true).unwrap();
        let summary: TrainingRunSummary = serde_json::from_str(
            &fs::read_to_string(
                output
                    .config
                    .training
                    .checkpoint_dir
                    .join("training_summary.json"),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(summary.resumed_from_step, Some(1));
        assert_eq!(summary.training_history.len(), 4);
        assert_eq!(summary.training_history[0].step, 0);
        assert_eq!(summary.training_history[1].step, 1);
        assert_eq!(summary.training_history[2].step, 2);
        assert_eq!(summary.training_history[3].step, 3);
        assert!(summary
            .config
            .training
            .checkpoint_dir
            .join("dataset_validation_report.json")
            .exists());
        assert!(summary.reproducibility.resume_provenance.resumed);
    }
}
