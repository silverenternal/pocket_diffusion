//! Terminal reporting helpers for config-driven research runs.

use crate::{
    experiments::{EvaluationMetrics, UnseenPocketExperimentSummary},
    training::{DatasetInspection, StepMetrics, TrainingRunSummary},
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
    println!("dataset validation:");
    print_dataset_validation(&inspection.validation);
    println!("split audit:");
    print_split_report(&inspection.split_report);
    println!(
        "validation artifact: {}",
        inspection.validation_report_path.display()
    );

    for example in &inspection.examples {
        println!(
            "  {} | protein={} | ligand_atoms={} | pocket_atoms={} | pocket_dim={} | affinity={:?} | measurement={:?} {:?} {:?} | provenance={:?} | approximate={} | warning={:?}",
            example.example_id,
            example.protein_id,
            example.ligand_atoms,
            example.pocket_atoms,
            example.pocket_feature_dim,
            example.affinity_kcal_mol,
            example.affinity_measurement_type,
            example.affinity_raw_value,
            example.affinity_raw_unit,
            example.affinity_normalization_provenance,
            example.affinity_is_approximate,
            example.affinity_normalization_warning,
        );
    }
}

/// Print a config-driven training report.
pub fn print_training_run(summary: &TrainingRunSummary) {
    println!("================================================");
    println!("  Config-Driven Training Run");
    println!("================================================");
    println!("dataset:");
    println!("  total examples: {}", summary.splits.total);
    println!("  train: {}", summary.splits.train);
    println!("  val: {}", summary.splits.val);
    println!("  test: {}", summary.splits.test);
    println!("dataset validation:");
    print_dataset_validation(&summary.dataset_validation);
    println!("split audit:");
    print_split_report(&summary.split_report);
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
    print_eval_metrics(&summary.validation);
    println!("test:");
    print_eval_metrics(&summary.test);
    println!("artifacts:");
    println!(
        "  latest checkpoint: {}",
        summary
            .config
            .training
            .checkpoint_dir
            .join("latest.ot")
            .display()
    );
    println!(
        "  dataset validation: {}",
        summary
            .config
            .training
            .checkpoint_dir
            .join("dataset_validation_report.json")
            .display()
    );
    println!(
        "  training summary: {}",
        summary
            .config
            .training
            .checkpoint_dir
            .join("training_summary.json")
            .display()
    );
    println!(
        "  run bundle: {}",
        summary
            .config
            .training
            .checkpoint_dir
            .join("run_artifacts.json")
            .display()
    );
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
    println!("dataset validation:");
    print_dataset_validation(&summary.dataset_validation);
    println!("split audit:");
    print_split_report(&summary.split_report);
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
    println!(
        "  dataset validation: {}",
        summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("dataset_validation_report.json")
            .display()
    );
}

/// Print one training step record.
pub fn print_step_metrics(metrics: &StepMetrics) {
    println!(
        "step {} [{:?}] total={:.4} primary:{}={:.4} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4}",
        metrics.step,
        metrics.stage,
        metrics.losses.total,
        metrics.losses.primary.objective_name,
        metrics.losses.primary.surrogate_reconstruction,
        metrics.losses.auxiliaries.intra_red,
        metrics.losses.auxiliaries.probe,
        metrics.losses.auxiliaries.leak,
        metrics.losses.auxiliaries.gate,
        metrics.losses.auxiliaries.slot,
        metrics.losses.auxiliaries.consistency,
    );
}

/// Print evaluation metrics for validation/test reporting.
pub fn print_eval_metrics(metrics: &EvaluationMetrics) {
    println!("  representation diagnostics:");
    println!(
        "    finite-forward fraction: {:.4}",
        metrics.representation_diagnostics.finite_forward_fraction
    );
    println!(
        "    unique-complex fraction: {:.4}",
        metrics.representation_diagnostics.unique_complex_fraction
    );
    println!(
        "    unseen-protein fraction: {:.4}",
        metrics.representation_diagnostics.unseen_protein_fraction
    );
    println!(
        "    topology reconstruction mse: {:.4}",
        metrics
            .representation_diagnostics
            .topology_reconstruction_mse
    );
    println!(
        "    topology-pocket cosine alignment: {:.4}",
        metrics
            .representation_diagnostics
            .topology_pocket_cosine_alignment
    );
    println!(
        "    slot activation mean: {:.4}",
        metrics.representation_diagnostics.slot_activation_mean
    );
    println!(
        "    gate activation mean: {:.4}",
        metrics.representation_diagnostics.gate_activation_mean
    );
    println!(
        "    leakage proxy mean: {:.4}",
        metrics.representation_diagnostics.leakage_proxy_mean
    );
    println!("  proxy task metrics:");
    println!(
        "    affinity probe mae: {:.4}",
        metrics.proxy_task_metrics.affinity_probe_mae
    );
    println!(
        "    affinity probe rmse: {:.4}",
        metrics.proxy_task_metrics.affinity_probe_rmse
    );
    println!(
        "    labeled fraction: {:.4}",
        metrics.proxy_task_metrics.labeled_fraction
    );
    for group in &metrics.proxy_task_metrics.affinity_by_measurement {
        println!(
            "    affinity [{}]: count={} mae={:.4} rmse={:.4}",
            group.measurement_type, group.count, group.mae, group.rmse
        );
    }
    println!("  split context:");
    println!("    examples: {}", metrics.split_context.example_count);
    println!(
        "    unique complexes: {}",
        metrics.split_context.unique_complex_count
    );
    println!(
        "    unique proteins: {}",
        metrics.split_context.unique_protein_count
    );
    println!(
        "    train reference proteins: {}",
        metrics.split_context.train_reference_protein_count
    );
    println!("  resource usage:");
    println!(
        "    memory usage mb: {:.4}",
        metrics.resource_usage.memory_usage_mb
    );
    println!(
        "    eval time ms: {:.4}",
        metrics.resource_usage.evaluation_time_ms
    );
    println!("  reserved real-generation metrics:");
    print_reserved_backend(
        "chemistry",
        &metrics.real_generation_metrics.chemistry_validity,
    );
    print_reserved_backend("docking", &metrics.real_generation_metrics.docking_affinity);
    print_reserved_backend(
        "pocket compatibility",
        &metrics.real_generation_metrics.pocket_compatibility,
    );
}

fn print_dataset_validation(report: &crate::data::DatasetValidationReport) {
    println!("  discovered complexes: {}", report.discovered_complexes);
    println!("  parsed examples: {}", report.parsed_examples);
    println!("  parsed ligands: {}", report.parsed_ligands);
    println!("  parsed pockets: {}", report.parsed_pockets);
    println!("  attached labels: {}", report.attached_labels);
    println!("  unlabeled examples: {}", report.unlabeled_examples);
    println!(
        "  label matches: example_id={} protein_id={}",
        report.example_id_label_matches, report.protein_id_label_matches
    );
    println!(
        "  pocket fallback extractions: {}",
        report.fallback_pocket_extractions
    );
    println!("  truncated examples: {}", report.truncated_examples);
    println!("  loaded label rows: {}", report.loaded_label_rows);
    println!(
        "  approximate affinity labels: {}",
        report.approximate_affinity_labels
    );
    println!(
        "  normalization warnings: {}",
        report.affinity_normalization_warnings
    );
    if !report.normalization_warning_messages.is_empty() {
        println!(
            "  warning messages: {:?}",
            report.normalization_warning_messages
        );
    }
    println!("  parsing mode: {}", report.parsing_mode);
}

fn print_split_report(report: &crate::training::SplitReport) {
    print_split_stats("train", &report.train);
    print_split_stats("val", &report.val);
    print_split_stats("test", &report.test);
    println!(
        "  leakage checks: protein_overlap={} duplicate_example_ids={}",
        report.leakage_checks.protein_overlap_detected,
        report.leakage_checks.duplicate_example_ids_detected
    );
    println!(
        "  overlap counts: train/val={} train/test={} val/test={} duplicated_ids={}",
        report.leakage_checks.train_val_protein_overlap,
        report.leakage_checks.train_test_protein_overlap,
        report.leakage_checks.val_test_protein_overlap,
        report.leakage_checks.duplicated_example_ids
    );
}

fn print_reserved_backend(label: &str, backend: &crate::experiments::ReservedBackendMetrics) {
    println!(
        "    {}: available={} backend={:?} status={}",
        label, backend.available, backend.backend_name, backend.status
    );
}

fn print_split_stats(name: &str, stats: &crate::training::SplitStats) {
    println!(
        "  {}: examples={} proteins={} labeled={} ({:.4}) measurements={:?}",
        name,
        stats.example_count,
        stats.unique_protein_count,
        stats.labeled_example_count,
        stats.labeled_fraction,
        stats.dominant_measurement_histogram
    );
}
