//! Terminal reporting helpers for config-driven research runs.

use crate::{
    experiments::{AutomatedSearchSummary, EvaluationMetrics, UnseenPocketExperimentSummary},
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
    println!("reproducibility:");
    println!(
        "  continuity mode: {:?}",
        summary.reproducibility.resume_provenance.continuity_mode
    );
    println!(
        "  strict replay achieved: {}",
        summary
            .reproducibility
            .resume_provenance
            .strict_replay_achieved
    );
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
    println!("reproducibility:");
    println!(
        "  continuity mode: {:?}",
        summary.reproducibility.resume_provenance.continuity_mode
    );
    println!(
        "  strict replay achieved: {}",
        summary
            .reproducibility
            .resume_provenance
            .strict_replay_achieved
    );
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
        "  claim summary: {}",
        summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("claim_summary.json")
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
    if let Some(matrix) = &summary.ablation_matrix {
        println!("ablation matrix:");
        for variant in &matrix.variants {
            println!(
                "  {} [{}]: valid={:?} pocket={:?} compatible={:?} unseen={:.4}",
                variant.variant_label,
                variant.test.interaction_mode,
                variant.test.candidate_valid_fraction,
                variant.test.pocket_contact_fraction,
                variant.test.pocket_compatibility_fraction,
                variant.test.unseen_protein_fraction
            );
        }
    }
}

/// Print a compact automated search report.
pub fn print_automated_search(summary: &AutomatedSearchSummary) {
    println!("================================================");
    println!("  Automated Cross-Surface Search");
    println!("================================================");
    println!("artifact root: {}", summary.artifact_root.display());
    println!("strategy: {:?}", summary.strategy);
    println!(
        "winner: {}",
        summary
            .winning_candidate_id
            .as_deref()
            .unwrap_or("none; all candidates blocked")
    );
    for candidate in &summary.ranked_candidates {
        println!(
            "{} | passed={} | score={:?} | surfaces={}",
            candidate.candidate_id,
            candidate.gate_result.passed,
            candidate.score,
            candidate.surfaces.len()
        );
        if !candidate.overrides.is_empty() {
            println!("  overrides: {}", candidate.overrides.join(", "));
        }
        for reason in &candidate.gate_result.blocked_reasons {
            println!("  blocked: {reason}");
        }
    }
    println!("roadmap:");
    println!("  {}", summary.roadmap_decision);
    println!(
        "search summary: {}",
        summary.artifact_root.join("search_summary.json").display()
    );
}

/// Print one training step record.
pub fn print_step_metrics(metrics: &StepMetrics) {
    println!(
        "step {} [{:?}] total={:.4} primary:{}={:.4} decoder_anchor={} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4} pocket_contact={:.4} pocket_clash={:.4}",
        metrics.step,
        metrics.stage,
        metrics.losses.total,
        metrics.losses.primary.objective_name,
        metrics.losses.primary.primary_value,
        metrics.losses.primary.decoder_anchored,
        metrics.losses.auxiliaries.intra_red,
        metrics.losses.auxiliaries.probe,
        metrics.losses.auxiliaries.leak,
        metrics.losses.auxiliaries.gate,
        metrics.losses.auxiliaries.slot,
        metrics.losses.auxiliaries.consistency,
        metrics.losses.auxiliaries.pocket_contact,
        metrics.losses.auxiliaries.pocket_clash,
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
    if !metrics.split_context.ligand_atom_count_bins.is_empty() {
        println!(
            "    ligand atom bins: {:?}",
            metrics.split_context.ligand_atom_count_bins
        );
    }
    if !metrics.split_context.pocket_atom_count_bins.is_empty() {
        println!(
            "    pocket atom bins: {:?}",
            metrics.split_context.pocket_atom_count_bins
        );
    }
    println!("  resource usage:");
    println!(
        "    memory usage mb: {:.4}",
        metrics.resource_usage.memory_usage_mb
    );
    println!(
        "    eval time ms: {:.4}",
        metrics.resource_usage.evaluation_time_ms
    );
    println!(
        "    examples/sec: {:.4}",
        metrics.resource_usage.examples_per_second
    );
    println!(
        "    avg atoms ligand/pocket: {:.2}/{:.2}",
        metrics.resource_usage.average_ligand_atoms, metrics.resource_usage.average_pocket_atoms
    );
    println!("  real-generation metrics:");
    print_reserved_backend(
        "chemistry",
        &metrics.real_generation_metrics.chemistry_validity,
    );
    print_reserved_backend("docking", &metrics.real_generation_metrics.docking_affinity);
    print_reserved_backend(
        "pocket compatibility",
        &metrics.real_generation_metrics.pocket_compatibility,
    );
    println!("  layered generation metrics:");
    print_candidate_layer(
        "raw rollout",
        &metrics.layered_generation_metrics.raw_rollout,
    );
    print_candidate_layer(
        "repaired",
        &metrics.layered_generation_metrics.repaired_candidates,
    );
    print_candidate_layer(
        "inferred bonds",
        &metrics.layered_generation_metrics.inferred_bond_candidates,
    );
    print_candidate_layer(
        "proxy reranked",
        &metrics.layered_generation_metrics.reranked_candidates,
    );
    println!("  slot stability:");
    println!(
        "    activation topo={:.4} geo={:.4} pocket={:.4}",
        metrics.slot_stability.topology_activation_mean,
        metrics.slot_stability.geometry_activation_mean,
        metrics.slot_stability.pocket_activation_mean
    );
    println!(
        "    signature similarity topo={:.4} geo={:.4} pocket={:.4}",
        metrics.slot_stability.topology_signature_similarity,
        metrics.slot_stability.geometry_signature_similarity,
        metrics.slot_stability.pocket_signature_similarity
    );
    println!("  comparison summary:");
    println!(
        "    primary objective: {}",
        metrics.comparison_summary.primary_objective
    );
    println!(
        "    variant label: {:?}",
        metrics.comparison_summary.variant_label
    );
    println!(
        "    interaction mode: {}",
        metrics.comparison_summary.interaction_mode
    );
    println!(
        "    candidate valid fraction: {:?}",
        metrics.comparison_summary.candidate_valid_fraction
    );
    println!(
        "    pocket contact fraction: {:?}",
        metrics.comparison_summary.pocket_contact_fraction
    );
    println!(
        "    pocket compatibility fraction: {:?}",
        metrics.comparison_summary.pocket_compatibility_fraction
    );
    println!(
        "    mean centroid offset: {:?}",
        metrics.comparison_summary.mean_centroid_offset
    );
    println!(
        "    strict pocket-fit score: {:?}",
        metrics.comparison_summary.strict_pocket_fit_score
    );
    println!(
        "    unique smiles fraction: {:?}",
        metrics.comparison_summary.unique_smiles_fraction
    );
    println!(
        "    unseen-protein fraction: {:.4}",
        metrics.comparison_summary.unseen_protein_fraction
    );
    println!(
        "    specialization: topology={:.4} geometry={:.4} pocket={:.4}",
        metrics.comparison_summary.topology_specialization_score,
        metrics.comparison_summary.geometry_specialization_score,
        metrics.comparison_summary.pocket_specialization_score
    );
    println!(
        "    utilization: slots={:.4} gates={:.4} leakage={:.4}",
        metrics.comparison_summary.slot_activation_mean,
        metrics.comparison_summary.gate_activation_mean,
        metrics.comparison_summary.leakage_proxy_mean
    );
}

fn print_candidate_layer(label: &str, layer: &crate::experiments::CandidateLayerMetrics) {
    println!(
        "    {label}: count={} valid={:.4} contact={:.4} centroid_offset={:.4} clash={:.4} displacement={:.4} atom_change={:.4} unique_proxy={:.4}",
        layer.candidate_count,
        layer.valid_fraction,
        layer.pocket_contact_fraction,
        layer.mean_centroid_offset,
        layer.clash_fraction,
        layer.mean_displacement,
        layer.atom_change_fraction,
        layer.uniqueness_proxy_fraction,
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
    println!(
        "  quality filtered examples: {}",
        report.quality_filtered_examples
    );
    println!(
        "  quality filter detail: unlabeled={} ligand_atoms={} pocket_atoms={} missing_source={}",
        report.quality_filtered_unlabeled_examples,
        report.quality_filtered_ligand_atom_limit,
        report.quality_filtered_pocket_atom_limit,
        report.quality_filtered_missing_source_provenance
    );
    println!(
        "  quality filter metadata detail: missing_affinity_metadata={}",
        report.quality_filtered_missing_affinity_metadata
    );
    println!(
        "  retained label coverage: {:.4}",
        report.retained_label_coverage
    );
    println!(
        "  retained source provenance coverage: {:.4}",
        report.retained_source_provenance_coverage
    );
    println!(
        "  observed fallback fraction: {:.4}",
        report.observed_fallback_fraction
    );
    println!("  truncated examples: {}", report.truncated_examples);
    println!("  loaded label rows: {}", report.loaded_label_rows);
    println!(
        "  label table rows: seen={} blank={} comment={}",
        report.label_table_rows_seen,
        report.label_table_blank_rows,
        report.label_table_comment_rows
    );
    println!(
        "  approximate affinity labels: {}",
        report.approximate_affinity_labels
    );
    if !report.loaded_label_measurement_family_histogram.is_empty() {
        println!(
            "  loaded label families: {:?}",
            report.loaded_label_measurement_family_histogram
        );
    }
    println!(
        "  label attachment detail: duplicate_example={} duplicate_protein={} unmatched_example={} unmatched_protein={}",
        report.duplicate_example_id_label_rows,
        report.duplicate_protein_id_label_rows,
        report.unmatched_example_id_label_rows,
        report.unmatched_protein_id_label_rows
    );
    println!(
        "  retained approximate labels: {} ({:.4})",
        report.retained_approximate_affinity_labels,
        report.retained_approximate_label_fraction
    );
    println!(
        "  normalization warnings: {}",
        report.affinity_normalization_warnings
    );
    println!(
        "  retained normalization provenance coverage: {:.4}",
        report.retained_normalization_provenance_coverage
    );
    println!(
        "  retained metadata gaps: measurement_type={} normalization_provenance={}",
        report.retained_missing_measurement_type,
        report.retained_missing_normalization_provenance
    );
    if !report.retained_measurement_family_histogram.is_empty() {
        println!(
            "  retained measurement families ({}): {:?}",
            report.retained_measurement_family_count,
            report.retained_measurement_family_histogram
        );
    }
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
    if !backend.metrics.is_empty() {
        println!("      metrics: {:?}", backend.metrics);
    }
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
