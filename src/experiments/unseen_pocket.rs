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
    models::{Phase1ResearchSystem, ResearchForward},
    runtime::parse_runtime_device,
    training::{ResearchTrainer, StepMetrics},
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

/// Aggregate evaluation metrics for one split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Fraction of samples producing finite outputs.
    pub validity: f64,
    /// Fraction of unique protein-ligand ids in the split.
    pub uniqueness: f64,
    /// Fraction of proteins not seen in the training set.
    pub novelty: f64,
    /// RMSE between predicted and target pairwise distances.
    pub distance_rmse: f64,
    /// Cross-modal cosine alignment between topology and pocket latents.
    pub affinity_alignment: f64,
    /// MAE of affinity prediction on labeled examples.
    pub affinity_mae: f64,
    /// RMSE of affinity prediction on labeled examples.
    pub affinity_rmse: f64,
    /// Fraction of examples in the split with affinity labels.
    pub labeled_fraction: f64,
    /// Affinity error summarized per measurement type.
    pub affinity_by_measurement: Vec<MeasurementMetrics>,
    /// Average process memory in MB during evaluation.
    pub memory_usage_mb: f64,
    /// Elapsed evaluation time in milliseconds.
    pub evaluation_time_ms: f64,
    /// Mean topology reconstruction error across the split.
    pub reconstruction_mse: f64,
    /// Mean active slot fraction.
    pub slot_usage_mean: f64,
    /// Mean gate activation.
    pub gate_usage_mean: f64,
    /// Mean leakage proxy.
    pub leakage_mean: f64,
}

/// Train/validation/test experiment summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnseenPocketExperimentSummary {
    /// Applied experiment configuration.
    pub config: UnseenPocketExperimentConfig,
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
        let dataset = InMemoryDataset::from_data_config(&config.research.data)?
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
        if resume_from_latest {
            if let Some(checkpoint) = trainer.resume_from_latest(&mut var_store)? {
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

        let summary = UnseenPocketExperimentSummary {
            config,
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
    let path = summary
        .config
        .research
        .training
        .checkpoint_dir
        .join("experiment_summary.json");
    fs::write(path, serde_json::to_string_pretty(summary)?)?;
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
            validity: 0.0,
            uniqueness: 0.0,
            novelty: 0.0,
            distance_rmse: 0.0,
            affinity_alignment: 0.0,
            affinity_mae: 0.0,
            affinity_rmse: 0.0,
            labeled_fraction: 0.0,
            affinity_by_measurement: Vec::new(),
            memory_usage_mb: memory_before,
            evaluation_time_ms: 0.0,
            reconstruction_mse: 0.0,
            slot_usage_mean: 0.0,
            gate_usage_mean: 0.0,
            leakage_mean: 0.0,
        };
    }

    let forwards: Vec<ResearchForward> = examples
        .iter()
        .map(|example| system.forward_example(&example.to_device(device)))
        .collect();

    sys.refresh_memory();
    let memory_after = sys.used_memory() as f64 / (1024.0 * 1024.0);

    let validity = forwards
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
    let uniqueness = unique_ids / examples.len() as f64;

    let novelty = examples
        .iter()
        .filter(|example| !train_proteins.contains(example.protein_id.as_str()))
        .count() as f64
        / examples.len() as f64;

    let distance_rmse = (examples
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

    let affinity_alignment = forwards
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
    let affinity_mae = if labeled_examples.is_empty() {
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
    let affinity_rmse = if labeled_examples.is_empty() {
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

    let reconstruction_mse = forwards
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

    let slot_usage_mean = if ablation.disable_slots {
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

    let gate_usage_mean = if ablation.disable_cross_attention {
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

    let leakage_mean = if ablation.disable_leakage {
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

    EvaluationMetrics {
        validity,
        uniqueness,
        novelty,
        distance_rmse,
        affinity_alignment,
        affinity_mae,
        affinity_rmse,
        labeled_fraction,
        affinity_by_measurement,
        memory_usage_mb: (memory_after - memory_before).max(0.0),
        evaluation_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        reconstruction_mse,
        slot_usage_mean,
        gate_usage_mean,
        leakage_mean,
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
        config.research.training.checkpoint_every = 100;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = temp.path().join("checkpoints");

        let summary = UnseenPocketExperiment::run(config).unwrap();
        assert_eq!(summary.training_history.len(), 2);
        assert!(summary.validation.validity >= 0.0);
        assert!(summary.test.validity >= 0.0);
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("experiment_summary.json")
            .exists());
    }

    #[test]
    fn resumed_experiment_preserves_prior_history() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint_dir = temp.path().join("checkpoints");

        let mut config = UnseenPocketExperimentConfig::default();
        config.research.training.max_steps = 2;
        config.research.training.checkpoint_every = 1;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = checkpoint_dir;

        let _ = UnseenPocketExperiment::run(config.clone()).unwrap();

        config.research.training.max_steps = 4;
        let summary = UnseenPocketExperiment::run_with_options(config, true).unwrap();

        assert_eq!(summary.training_history.len(), 4);
        assert_eq!(summary.training_history[0].step, 0);
        assert_eq!(summary.training_history[1].step, 1);
        assert_eq!(summary.training_history[2].step, 2);
        assert_eq!(summary.training_history[3].step, 3);
    }
}
