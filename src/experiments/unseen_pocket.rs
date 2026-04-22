//! Unseen-pocket experiment loop, ablations, and evaluation summaries.

use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tch::nn;

use crate::{
    config::ResearchConfig,
    data::InMemoryDataset,
    models::{
        generate_candidates_from_forward, report_to_metrics, ChemistryValidityEvaluator,
        DockingEvaluator, HeuristicChemistryValidityEvaluator, HeuristicDockingEvaluator,
        HeuristicPocketCompatibilityEvaluator, Phase1ResearchSystem, PocketCompatibilityEvaluator,
        ResearchForward,
    },
    runtime::parse_runtime_device,
    training::{
        reproducibility_metadata, stable_json_hash, ResearchTrainer, RunArtifactBundle,
        RunArtifactPaths, RunKind, SplitReport, StepMetrics,
    },
};

/// Toggles used to disable parts of the model or objective for ablation studies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AblationConfig {
    /// Disable slot decomposition metrics and slot loss reporting.
    pub disable_slots: bool,
    /// Disable gated cross-modal interaction metrics.
    pub disable_cross_attention: bool,
    /// Disable semantic probe metrics.
    pub disable_probes: bool,
    /// Disable leakage reporting.
    pub disable_leakage: bool,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            disable_slots: false,
            disable_cross_attention: false,
            disable_probes: false,
            disable_leakage: false,
        }
    }
}

/// High-level experiment configuration for unseen-pocket generalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnseenPocketExperimentConfig {
    /// Core model/training/runtime configuration.
    pub research: ResearchConfig,
    /// Ablation toggles.
    pub ablation: AblationConfig,
}

impl UnseenPocketExperimentConfig {
    /// Validate a config before allocating runtime state.
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.research.validate()?;
        Ok(())
    }
}

impl Default for UnseenPocketExperimentConfig {
    fn default() -> Self {
        Self {
            research: ResearchConfig::default(),
            ablation: AblationConfig::default(),
        }
    }
}

/// Load an experiment configuration from JSON.
pub fn load_experiment_config(
    path: impl AsRef<std::path::Path>,
) -> Result<UnseenPocketExperimentConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

/// Placeholder section reserved for chemistry-grade evaluation backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealGenerationMetrics {
    /// Chemistry validity backend slot.
    pub chemistry_validity: ReservedBackendMetrics,
    /// Docking or affinity rescoring backend slot.
    pub docking_affinity: ReservedBackendMetrics,
    /// Downstream pocket compatibility backend slot.
    pub pocket_compatibility: ReservedBackendMetrics,
}

/// Reserved backend schema entry used before integrating external toolkits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservedBackendMetrics {
    /// Whether this backend has been integrated for the current run.
    pub available: bool,
    /// Backend identifier when available.
    pub backend_name: Option<String>,
    /// Reserved metrics map emitted by the backend.
    pub metrics: BTreeMap<String, f64>,
    /// Explanation when the backend is not yet active.
    pub status: String,
}

/// Aggregate evaluation metrics for one split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationDiagnostics {
    /// Fraction of samples producing finite outputs.
    pub finite_forward_fraction: f64,
    /// Fraction of unique protein-ligand ids in the split.
    pub unique_complex_fraction: f64,
    /// Fraction of proteins not seen in the training set.
    pub unseen_protein_fraction: f64,
    /// RMSE between distance-probe predictions and target pairwise distances.
    pub distance_probe_rmse: f64,
    /// Cross-modal cosine alignment between topology and pocket latents.
    pub topology_pocket_cosine_alignment: f64,
    /// Mean topology reconstruction error across the split.
    pub topology_reconstruction_mse: f64,
    /// Mean active slot fraction.
    pub slot_activation_mean: f64,
    /// Mean gate activation.
    pub gate_activation_mean: f64,
    /// Mean leakage proxy.
    pub leakage_proxy_mean: f64,
}

/// Proxy task metrics derived from lightweight probe heads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyTaskMetrics {
    /// MAE of affinity prediction on labeled examples.
    pub affinity_probe_mae: f64,
    /// RMSE of affinity prediction on labeled examples.
    pub affinity_probe_rmse: f64,
    /// Fraction of examples in the split with affinity labels.
    pub labeled_fraction: f64,
    /// Affinity error summarized per measurement type.
    pub affinity_by_measurement: Vec<MeasurementMetrics>,
}

/// Split-level counts needed to interpret evaluation outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitContextMetrics {
    /// Number of examples evaluated.
    pub example_count: usize,
    /// Number of unique complex identifiers in the split.
    pub unique_complex_count: usize,
    /// Number of unique proteins in the split.
    pub unique_protein_count: usize,
    /// Number of training proteins used as the unseen-pocket reference set.
    pub train_reference_protein_count: usize,
}

/// Runtime resource measurements for one evaluation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageMetrics {
    /// Average process memory delta in MB during evaluation.
    pub memory_usage_mb: f64,
    /// Elapsed evaluation time in milliseconds.
    pub evaluation_time_ms: f64,
}

/// Aggregate evaluation metrics for one split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Representation-level diagnostics for the modular stack.
    pub representation_diagnostics: RepresentationDiagnostics,
    /// Proxy task metrics produced by lightweight probe heads.
    pub proxy_task_metrics: ProxyTaskMetrics,
    /// Split-level context counts.
    pub split_context: SplitContextMetrics,
    /// Runtime resource measurements.
    pub resource_usage: ResourceUsageMetrics,
    /// Reserved section for chemistry/docking/pocket compatibility backends.
    pub real_generation_metrics: RealGenerationMetrics,
}

/// Train/validation/test experiment summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnseenPocketExperimentSummary {
    /// Applied experiment configuration.
    pub config: UnseenPocketExperimentConfig,
    /// Machine-readable dataset validation artifact for the run.
    pub dataset_validation: crate::data::DatasetValidationReport,
    /// Split-distribution and leakage audit.
    pub split_report: SplitReport,
    /// Reproducibility and schema metadata for this run.
    pub reproducibility: crate::training::ReproducibilityMetadata,
    /// Training-step history collected during the run.
    pub training_history: Vec<StepMetrics>,
    /// Validation metrics on unseen-pocket split.
    pub validation: EvaluationMetrics,
    /// Test metrics on unseen-pocket split.
    pub test: EvaluationMetrics,
}

/// Error summary for one affinity measurement family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMetrics {
    /// Measurement family label, such as `Kd`, `Ki`, `IC50`, or `dG`.
    pub measurement_type: String,
    /// Number of labeled examples for this measurement family.
    pub count: usize,
    /// MAE on this measurement family.
    pub mae: f64,
    /// RMSE on this measurement family.
    pub rmse: f64,
}

/// Runs a compact end-to-end unseen-pocket experiment on the current stack.
pub struct UnseenPocketExperiment;

impl UnseenPocketExperiment {
    /// Execute the experiment using the configured dataset loader.
    pub fn run(
        config: UnseenPocketExperimentConfig,
    ) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
        Self::run_with_options(config, false)
    }

    /// Execute the experiment with optional checkpoint resume.
    pub fn run_with_options(
        config: UnseenPocketExperimentConfig,
        resume_from_latest: bool,
    ) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
        config.validate()?;
        let loaded = InMemoryDataset::load_from_config(&config.research.data)?;
        let dataset = loaded
            .dataset
            .with_pocket_feature_dim(config.research.model.pocket_feature_dim);
        let splits = dataset.split_by_protein_fraction_with_options(
            config.research.data.val_fraction,
            config.research.data.test_fraction,
            config.research.data.split_seed,
            config.research.data.stratify_by_measurement,
        );

        if splits.train.examples().is_empty() {
            return Err(
                "training split is empty; adjust val/test fractions or dataset size".into(),
            );
        }

        let device = parse_runtime_device(&config.research.runtime.device)?;
        let mut var_store = nn::VarStore::new(device);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config.research);
        let mut trainer = ResearchTrainer::new(&var_store, config.research.clone())?;
        trainer.set_dataset_validation_fingerprint(stable_json_hash(&loaded.validation));
        let mut resumed_checkpoint_metadata = None;
        if resume_from_latest {
            if let Some(checkpoint) = trainer.resume_from_latest(&mut var_store)? {
                resumed_checkpoint_metadata = Some(checkpoint.metadata.clone());
                if let Some(history) = load_experiment_history(
                    &config.research.training.checkpoint_dir,
                    checkpoint.metadata.step,
                )? {
                    trainer.replace_history(history);
                }
                log::info!(
                    "resumed experiment training from step {} at {}",
                    checkpoint.metadata.step,
                    checkpoint.weights_path.display()
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

        let train_proteins: std::collections::BTreeSet<&str> = splits
            .train
            .examples()
            .iter()
            .map(|example| example.protein_id.as_str())
            .collect();

        let validation = evaluate_split(
            &system,
            splits.val.examples(),
            &train_proteins,
            config.ablation,
            device,
        );
        let test = evaluate_split(
            &system,
            splits.test.examples(),
            &train_proteins,
            config.ablation,
            device,
        );

        let reproducibility = reproducibility_metadata(
            &config.research,
            &loaded.validation,
            resumed_checkpoint_metadata.as_ref(),
        );
        let dataset_validation = loaded.validation;
        let summary = UnseenPocketExperimentSummary {
            config,
            dataset_validation,
            split_report: SplitReport::from_datasets(&splits.train, &splits.val, &splits.test),
            reproducibility,
            training_history: trainer.history().to_vec(),
            validation,
            test,
        };
        persist_experiment_summary(&summary)?;
        Ok(summary)
    }
}

fn persist_experiment_summary(
    summary: &UnseenPocketExperimentSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(&summary.config.research.training.checkpoint_dir)?;
    let artifact_dir = &summary.config.research.training.checkpoint_dir;
    let config_snapshot = artifact_dir.join("config.snapshot.json");
    let validation_path = artifact_dir.join("dataset_validation_report.json");
    let split_report_path = artifact_dir.join("split_report.json");
    let summary_path = artifact_dir.join("experiment_summary.json");
    let bundle_path = artifact_dir.join("run_artifacts.json");
    fs::write(
        &config_snapshot,
        serde_json::to_string_pretty(&summary.config)?,
    )?;
    fs::write(
        &validation_path,
        serde_json::to_string_pretty(&summary.dataset_validation)?,
    )?;
    fs::write(
        &split_report_path,
        serde_json::to_string_pretty(&summary.split_report)?,
    )?;
    fs::write(&summary_path, serde_json::to_string_pretty(summary)?)?;
    let bundle = RunArtifactBundle {
        schema_version: summary.reproducibility.artifact_bundle_schema_version,
        run_kind: RunKind::Experiment,
        artifact_dir: artifact_dir.clone(),
        config_hash: summary.reproducibility.config_hash.clone(),
        dataset_validation_fingerprint: summary
            .reproducibility
            .dataset_validation_fingerprint
            .clone(),
        metric_schema_version: summary.reproducibility.metric_schema_version,
        paths: RunArtifactPaths {
            config_snapshot,
            dataset_validation_report: validation_path,
            split_report: split_report_path,
            run_summary: summary_path,
            run_bundle: bundle_path.clone(),
            latest_checkpoint: Some(artifact_dir.join("latest.ot")),
        },
    };
    fs::write(bundle_path, serde_json::to_string_pretty(&bundle)?)?;
    Ok(())
}

fn load_experiment_history(
    checkpoint_dir: &std::path::Path,
    resumed_step: usize,
) -> Result<Option<Vec<StepMetrics>>, Box<dyn std::error::Error>> {
    let path = checkpoint_dir.join("experiment_summary.json");
    if !path.exists() {
        return Ok(None);
    }

    let summary: UnseenPocketExperimentSummary = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(Some(
        summary
            .training_history
            .into_iter()
            .filter(|metrics| metrics.step <= resumed_step)
            .collect(),
    ))
}

/// Evaluate a dataset split using the trained modular research system.
pub fn evaluate_split(
    system: &Phase1ResearchSystem,
    examples: &[crate::data::MolecularExample],
    train_proteins: &std::collections::BTreeSet<&str>,
    ablation: AblationConfig,
    device: tch::Device,
) -> EvaluationMetrics {
    let start = Instant::now();
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();
    let memory_before = sys.used_memory() as f64 / (1024.0 * 1024.0);

    if examples.is_empty() {
        return EvaluationMetrics {
            representation_diagnostics: RepresentationDiagnostics {
                finite_forward_fraction: 0.0,
                unique_complex_fraction: 0.0,
                unseen_protein_fraction: 0.0,
                distance_probe_rmse: 0.0,
                topology_pocket_cosine_alignment: 0.0,
                topology_reconstruction_mse: 0.0,
                slot_activation_mean: 0.0,
                gate_activation_mean: 0.0,
                leakage_proxy_mean: 0.0,
            },
            proxy_task_metrics: ProxyTaskMetrics {
                affinity_probe_mae: 0.0,
                affinity_probe_rmse: 0.0,
                labeled_fraction: 0.0,
                affinity_by_measurement: Vec::new(),
            },
            split_context: SplitContextMetrics {
                example_count: 0,
                unique_complex_count: 0,
                unique_protein_count: 0,
                train_reference_protein_count: train_proteins.len(),
            },
            resource_usage: ResourceUsageMetrics {
                memory_usage_mb: memory_before,
                evaluation_time_ms: 0.0,
            },
            real_generation_metrics: disabled_real_generation_metrics(),
        };
    }

    let forwards: Vec<ResearchForward> = examples
        .iter()
        .map(|example| system.forward_example(&example.to_device(device)))
        .collect();

    sys.refresh_memory();
    let memory_after = sys.used_memory() as f64 / (1024.0 * 1024.0);

    let finite_forward_fraction = forwards
        .iter()
        .filter(|forward| {
            tensor_is_finite(&forward.encodings.topology.pooled_embedding)
                && tensor_is_finite(&forward.encodings.geometry.pooled_embedding)
                && tensor_is_finite(&forward.encodings.pocket.pooled_embedding)
        })
        .count() as f64
        / examples.len() as f64;

    let unique_ids = examples
        .iter()
        .map(|example| format!("{}::{}", example.protein_id, example.example_id))
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64;
    let unique_complex_fraction = unique_ids / examples.len() as f64;

    let unseen_protein_fraction = examples
        .iter()
        .filter(|example| !train_proteins.contains(example.protein_id.as_str()))
        .count() as f64
        / examples.len() as f64;

    let distance_probe_rmse = (examples
        .iter()
        .zip(forwards.iter())
        .map(|(example, forward)| {
            let target = example
                .geometry
                .pairwise_distances
                .mean(tch::Kind::Float)
                .double_value(&[]);
            let pred = forward
                .probes
                .geometry_distance_predictions
                .mean(tch::Kind::Float)
                .double_value(&[]);
            let error = pred - target;
            error * error
        })
        .sum::<f64>()
        / examples.len() as f64)
        .sqrt();

    let topology_pocket_cosine_alignment = forwards
        .iter()
        .map(|forward| {
            let topo = &forward.encodings.topology.pooled_embedding;
            let pocket = &forward.encodings.pocket.pooled_embedding;
            (topo * pocket).sum(tch::Kind::Float).double_value(&[])
                / (topo.norm().double_value(&[]) * pocket.norm().double_value(&[])).max(1e-6)
        })
        .sum::<f64>()
        / examples.len() as f64;

    let labeled_examples: Vec<(&crate::data::MolecularExample, &ResearchForward)> = examples
        .iter()
        .zip(forwards.iter())
        .filter(|(example, _)| example.targets.affinity_kcal_mol.is_some())
        .collect();
    let labeled_fraction = labeled_examples.len() as f64 / examples.len() as f64;
    let affinity_probe_mae = if labeled_examples.is_empty() {
        0.0
    } else {
        labeled_examples
            .iter()
            .map(|(example, forward)| {
                let target = example.targets.affinity_kcal_mol.unwrap() as f64;
                let pred = forward.probes.affinity_prediction.double_value(&[]);
                (pred - target).abs()
            })
            .sum::<f64>()
            / labeled_examples.len() as f64
    };
    let affinity_probe_rmse = if labeled_examples.is_empty() {
        0.0
    } else {
        (labeled_examples
            .iter()
            .map(|(example, forward)| {
                let target = example.targets.affinity_kcal_mol.unwrap() as f64;
                let pred = forward.probes.affinity_prediction.double_value(&[]);
                let error = pred - target;
                error * error
            })
            .sum::<f64>()
            / labeled_examples.len() as f64)
            .sqrt()
    };
    let affinity_by_measurement = measurement_breakdown(&labeled_examples);

    let topology_reconstruction_mse = forwards
        .iter()
        .map(|forward| {
            (forward.slots.topology.reconstructed_tokens.shallow_clone()
                - forward.encodings.topology.token_embeddings.shallow_clone())
            .pow_tensor_scalar(2.0)
            .mean(tch::Kind::Float)
            .double_value(&[])
        })
        .sum::<f64>()
        / examples.len() as f64;

    let slot_activation_mean = if ablation.disable_slots {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let slot_means = [
                    active_slot_fraction(&forward.slots.topology.slot_weights),
                    active_slot_fraction(&forward.slots.geometry.slot_weights),
                    active_slot_fraction(&forward.slots.pocket.slot_weights),
                ];
                slot_means.iter().sum::<f64>() / slot_means.len() as f64
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let gate_activation_mean = if ablation.disable_cross_attention {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                [
                    forward.interactions.topo_from_geo.gate.double_value(&[0]),
                    forward
                        .interactions
                        .topo_from_pocket
                        .gate
                        .double_value(&[0]),
                    forward.interactions.geo_from_topo.gate.double_value(&[0]),
                    forward.interactions.geo_from_pocket.gate.double_value(&[0]),
                    forward
                        .interactions
                        .pocket_from_topo
                        .gate
                        .double_value(&[0]),
                    forward.interactions.pocket_from_geo.gate.double_value(&[0]),
                ]
                .iter()
                .sum::<f64>()
                    / 6.0
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let leakage_proxy_mean = if ablation.disable_leakage {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let topo = mean_slot(&forward.slots.topology.slots);
                let geo = mean_slot(&forward.slots.geometry.slots);
                let pocket = mean_slot(&forward.slots.pocket.slots);
                cosine_similarity(&topo, &geo).abs()
                    + cosine_similarity(&topo, &pocket).abs()
                    + cosine_similarity(&geo, &pocket).abs()
            })
            .sum::<f64>()
            / (examples.len() as f64 * 3.0)
    };

    let unique_protein_count = examples
        .iter()
        .map(|example| example.protein_id.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    EvaluationMetrics {
        representation_diagnostics: RepresentationDiagnostics {
            finite_forward_fraction,
            unique_complex_fraction,
            unseen_protein_fraction,
            distance_probe_rmse,
            topology_pocket_cosine_alignment,
            topology_reconstruction_mse,
            slot_activation_mean,
            gate_activation_mean,
            leakage_proxy_mean,
        },
        proxy_task_metrics: ProxyTaskMetrics {
            affinity_probe_mae,
            affinity_probe_rmse,
            labeled_fraction,
            affinity_by_measurement,
        },
        split_context: SplitContextMetrics {
            example_count: examples.len(),
            unique_complex_count: unique_ids as usize,
            unique_protein_count,
            train_reference_protein_count: train_proteins.len(),
        },
        resource_usage: ResourceUsageMetrics {
            memory_usage_mb: (memory_after - memory_before).max(0.0),
            evaluation_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        },
        real_generation_metrics: evaluate_real_generation_metrics(examples, &forwards),
    }
}

fn disabled_real_generation_metrics() -> RealGenerationMetrics {
    RealGenerationMetrics {
        chemistry_validity: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "reserved for a future chemistry-validity backend adapter".to_string(),
        },
        docking_affinity: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "reserved for a future docking or affinity rescoring backend adapter"
                .to_string(),
        },
        pocket_compatibility: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "reserved for a future downstream pocket-compatibility backend adapter"
                .to_string(),
        },
    }
}

fn evaluate_real_generation_metrics(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> RealGenerationMetrics {
    let candidates = examples
        .iter()
        .zip(forwards.iter())
        .flat_map(|(example, forward)| generate_candidates_from_forward(example, forward, 3))
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return disabled_real_generation_metrics();
    }

    let chemistry = HeuristicChemistryValidityEvaluator.evaluate_chemistry(&candidates);
    let docking = HeuristicDockingEvaluator.evaluate_docking(&candidates);
    let pocket = HeuristicPocketCompatibilityEvaluator.evaluate_pocket_compatibility(&candidates);

    RealGenerationMetrics {
        chemistry_validity: report_to_metrics(
            chemistry,
            "active heuristic chemistry-validity backend on modular decoder candidates",
        ),
        docking_affinity: report_to_metrics(
            docking,
            "active heuristic docking-oriented hook on modular decoder candidates",
        ),
        pocket_compatibility: report_to_metrics(
            pocket,
            "active heuristic pocket-compatibility hook on modular decoder candidates",
        ),
    }
}

fn measurement_breakdown(
    labeled_examples: &[(&crate::data::MolecularExample, &ResearchForward)],
) -> Vec<MeasurementMetrics> {
    let mut grouped: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for (example, forward) in labeled_examples {
        let measurement = example
            .targets
            .affinity_measurement_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        grouped.entry(measurement).or_default().push((
            example.targets.affinity_kcal_mol.unwrap() as f64,
            forward.probes.affinity_prediction.double_value(&[]),
        ));
    }

    grouped
        .into_iter()
        .map(|(measurement_type, pairs)| {
            let count = pairs.len();
            let mae = pairs
                .iter()
                .map(|(target, pred)| (pred - target).abs())
                .sum::<f64>()
                / count as f64;
            let rmse = (pairs
                .iter()
                .map(|(target, pred)| {
                    let error = pred - target;
                    error * error
                })
                .sum::<f64>()
                / count as f64)
                .sqrt();
            MeasurementMetrics {
                measurement_type,
                count,
                mae,
                rmse,
            }
        })
        .collect()
}

fn tensor_is_finite(tensor: &tch::Tensor) -> bool {
    tensor
        .isfinite()
        .all()
        .to_kind(tch::Kind::Int64)
        .int64_value(&[])
        != 0
}

fn active_slot_fraction(weights: &tch::Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    weights
        .gt(0.05)
        .to_kind(tch::Kind::Float)
        .mean(tch::Kind::Float)
        .double_value(&[])
}

fn mean_slot(slots: &tch::Tensor) -> tch::Tensor {
    if slots.numel() == 0 {
        tch::Tensor::zeros([1], (tch::Kind::Float, slots.device()))
    } else {
        slots.mean_dim([0].as_slice(), false, tch::Kind::Float)
    }
}

fn cosine_similarity(a: &tch::Tensor, b: &tch::Tensor) -> f64 {
    let dot = (a * b).sum(tch::Kind::Float).double_value(&[]);
    let a_norm = a.norm().double_value(&[]);
    let b_norm = b.norm().double_value(&[]);
    dot / (a_norm * b_norm).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unseen_pocket_experiment_smoke_test() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = UnseenPocketExperimentConfig::default();
        config.research.training.max_steps = 2;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 1;
        config.research.training.schedule.stage3_steps = 2;
        config.research.training.checkpoint_every = 100;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = temp.path().join("checkpoints");

        let summary = UnseenPocketExperiment::run(config).unwrap();
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
            .join("dataset_validation_report.json")
            .exists());
        assert_eq!(summary.dataset_validation.parsed_examples, 4);
    }

    #[test]
    fn resumed_experiment_preserves_prior_history() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint_dir = temp.path().join("checkpoints");

        let mut config = UnseenPocketExperimentConfig::default();
        config.research.training.max_steps = 2;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 1;
        config.research.training.schedule.stage3_steps = 2;
        config.research.training.checkpoint_every = 1;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = checkpoint_dir;

        let _ = UnseenPocketExperiment::run(config.clone()).unwrap();

        config.research.training.max_steps = 4;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 2;
        config.research.training.schedule.stage3_steps = 3;
        let summary = UnseenPocketExperiment::run_with_options(config, true).unwrap();

        assert_eq!(summary.training_history.len(), 4);
        assert_eq!(summary.training_history[0].step, 0);
        assert_eq!(summary.training_history[1].step, 1);
        assert_eq!(summary.training_history[2].step, 2);
        assert_eq!(summary.training_history[3].step, 3);
        assert!(!summary.split_report.leakage_checks.protein_overlap_detected);
    }
}
