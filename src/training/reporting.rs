//! Terminal reporting helpers for config-driven research runs.

use crate::{
    experiments::{EvaluationMetrics, UnseenPocketExperimentSummary},
    training::{DatasetInspection, StepMetrics, TrainingRunOutput},
};

/// Print a compact dataset inspection report.
pub fn print_dataset_inspection(inspection: &DatasetInspection) {
    println!("================================================");
    println!("  Dataset Inspection");
    println!("================================================");
    println!("dataset format: {:?}", inspection.dataset_format);
    println!("total examples: {}", inspection.total_examples);
    println!("train examples: {}", inspection.train_examples);
    println!("val examples: {}", inspection.val_examples);
    println!("test examples: {}", inspection.test_examples);
    println!(
        "configured pocket feature dim: {}",
        inspection.pocket_feature_dim
    );

    for example in &inspection.examples {
        println!(
            "  {} | protein={} | ligand_atoms={} | pocket_atoms={} | pocket_dim={} | affinity={:?} | measurement={:?} {:?} {:?}",
            example.example_id,
            example.protein_id,
            example.ligand_atoms,
            example.pocket_atoms,
            example.pocket_feature_dim,
            example.affinity_kcal_mol,
            example.affinity_measurement_type,
            example.affinity_raw_value,
            example.affinity_raw_unit,
        );
    }
}

/// Print a config-driven training report.
pub fn print_training_run(output: &TrainingRunOutput) {
    let summary = &output.summary;

    println!("================================================");
    println!("  Config-Driven Training Run");
    println!("================================================");
    println!("dataset:");
    println!("  total examples: {}", summary.splits.total);
    println!("  train: {}", summary.splits.train);
    println!("  val: {}", summary.splits.val);
    println!("  test: {}", summary.splits.test);
    println!("runtime:");
    println!("  device: {}", summary.config.runtime.device);
    println!("  batch size: {}", summary.config.data.batch_size);
    println!(
        "  checkpoint dir: {}",
        summary.config.training.checkpoint_dir.display()
    );
    if let Some(step) = summary.resumed_from_step {
        println!("resume:");
        println!("  resumed from step {}", step);
    }
    for metric in &summary.training_history {
        print_step_metrics(metric);
    }
    if let Some(last) = summary.training_history.last() {
        println!(
            "completed {} step(s); final stage={:?}, total loss={:.4}",
            summary.training_history.len(),
            last.stage,
            last.losses.total
        );
    }
    println!("validation:");
    print_eval_metrics(&output.validation);
    println!("test:");
    print_eval_metrics(&output.test);
    println!("artifacts:");
    println!(
        "  latest checkpoint: {}",
        output.latest_checkpoint_path.display()
    );
    println!("  training summary: {}", summary.summary_path.display());
}

/// Print a config-driven unseen-pocket experiment report.
pub fn print_experiment_run(summary: &UnseenPocketExperimentSummary) {
    println!("================================================");
    println!("  Config-Driven Unseen-Pocket Experiment");
    println!("================================================");
    println!("runtime:");
    println!("  device: {}", summary.config.research.runtime.device);
    println!("  batch size: {}", summary.config.research.data.batch_size);
    println!(
        "  checkpoint dir: {}",
        summary.config.research.training.checkpoint_dir.display()
    );
    println!("training:");
    println!("  steps: {}", summary.training_history.len());
    if let Some(last) = summary.training_history.last() {
        println!(
            "  last stage: {:?}, last total loss: {:.4}",
            last.stage, last.losses.total
        );
    }
    println!("validation:");
    print_eval_metrics(&summary.validation);
    println!("test:");
    print_eval_metrics(&summary.test);
    println!("artifacts:");
    println!(
        "  latest checkpoint: {}",
        summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("latest.ot")
            .display()
    );
    println!(
        "  experiment summary: {}",
        summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("experiment_summary.json")
            .display()
    );
}

/// Print one training step record.
pub fn print_step_metrics(metrics: &StepMetrics) {
    println!(
        "step {} [{:?}] total={:.4} task={:.4} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4}",
        metrics.step,
        metrics.stage,
        metrics.losses.total,
        metrics.losses.task,
        metrics.losses.intra_red,
        metrics.losses.probe,
        metrics.losses.leak,
        metrics.losses.gate,
        metrics.losses.slot,
        metrics.losses.consistency,
    );
}

/// Print evaluation metrics for validation/test reporting.
pub fn print_eval_metrics(metrics: &EvaluationMetrics) {
    println!("  validity: {:.4}", metrics.validity);
    println!("  uniqueness: {:.4}", metrics.uniqueness);
    println!("  novelty: {:.4}", metrics.novelty);
    println!("  distance rmse: {:.4}", metrics.distance_rmse);
    println!("  affinity alignment: {:.4}", metrics.affinity_alignment);
    println!("  affinity mae: {:.4}", metrics.affinity_mae);
    println!("  affinity rmse: {:.4}", metrics.affinity_rmse);
    println!("  labeled fraction: {:.4}", metrics.labeled_fraction);
    for group in &metrics.affinity_by_measurement {
        println!(
            "  affinity [{}]: count={} mae={:.4} rmse={:.4}",
            group.measurement_type, group.count, group.mae, group.rmse
        );
    }
    println!("  memory usage mb: {:.4}", metrics.memory_usage_mb);
    println!("  eval time ms: {:.4}", metrics.evaluation_time_ms);
    println!("  reconstruction mse: {:.4}", metrics.reconstruction_mse);
    println!("  slot usage mean: {:.4}", metrics.slot_usage_mean);
    println!("  gate usage mean: {:.4}", metrics.gate_usage_mean);
    println!("  leakage mean: {:.4}", metrics.leakage_mean);
}
