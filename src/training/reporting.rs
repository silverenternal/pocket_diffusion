//! Terminal reporting helpers for config-driven research runs.

use crate::{
    experiments::{
        AutomatedSearchSummary, ChemistryCollaborationMetric, EvaluationMetrics,
        UnseenPocketExperimentSummary,
    },
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
        "  resume mode: {}",
        summary
            .reproducibility
            .resume_provenance
            .resume_mode
            .as_str()
    );
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
    println!("validation checkpoints:");
    println!("  validation checks: {}", summary.validation_history.len());
    if let Some(best) = &summary.best_checkpoint {
        println!(
            "  best step: {} ({}={:.4}, higher_is_better={})",
            best.step, best.metric_name, best.metric_value, best.higher_is_better
        );
        println!("  best checkpoint: {}", best.weights_path.display());
    }
    if summary.early_stopping.enabled {
        println!(
            "  early stopping: stopped={} stop_step={:?} patience={:?}",
            summary.early_stopping.stopped_early,
            summary.early_stopping.stop_step,
            summary.early_stopping.patience
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
        "  resume mode: {}",
        summary
            .reproducibility
            .resume_provenance
            .resume_mode
            .as_str()
    );
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
        "step {} [{:?}] total={:.4} primary:{}={:.4} decoder_anchor={} intra_red={:.4} probe={:.4} probe_topology_sparse_negative_rate={:.4} probe_ligand_pharmacophore={:.4} probe_pocket_pharmacophore={:.4} leak={:.4} leak_core={:.4} leak_similarity_proxy_diagnostic={:.4} leak_explicit_probe_diagnostic={:.4} leak_topology_to_geometry={:.4} leak_geometry_to_topology={:.4} leak_pocket_to_geometry={:.4} leak_topology_to_pocket_role={:.4} leak_geometry_to_pocket_role={:.4} leak_pocket_to_topology_role={:.4} leak_pocket_to_ligand_role={:.4} gate={:.4} slot={:.4} consistency={:.4} pocket_contact={:.4} pocket_pair_distance={:.4} pocket_clash={:.4} pocket_shape_complementarity={:.4} pocket_envelope={:.4} pocket_prior={:.4} pocket_prior_atom_count={:.4} pocket_prior_composition={:.4} pocket_prior_atom_count_mae={:.4} valence_guardrail={:.4} valence_overage_guardrail={:.4} valence_underage_guardrail={:.4} bond_length_guardrail={:.4} nonbonded_distance_guardrail={:.4} angle_guardrail={:.4} interaction_gate={:.4} interaction_sparsity={:.4} interaction_entropy={:.4} grad_norm={:.4} grad_nonfinite={} grad_clipped={} optimizer_step_skipped={} sync_mask_mismatch={} sync_slot_mismatch={} sync_frame_mismatch={} stale_context_steps={} refresh_count={} batch_slice_sync_pass={}",
        metrics.step,
        metrics.stage,
        metrics.losses.total,
        metrics.losses.primary.objective_name,
        metrics.losses.primary.primary_value,
        metrics.losses.primary.decoder_anchored,
        metrics.losses.auxiliaries.intra_red,
        metrics.losses.auxiliaries.probe,
        metrics
            .losses
            .auxiliaries
            .probe_topology_sparse_negative_rate,
        metrics.losses.auxiliaries.probe_ligand_pharmacophore,
        metrics.losses.auxiliaries.probe_pocket_pharmacophore,
        metrics.losses.auxiliaries.leak,
        metrics.losses.auxiliaries.leak_core,
        metrics
            .losses
            .auxiliaries
            .leak_similarity_proxy_diagnostic,
        metrics.losses.auxiliaries.leak_explicit_probe_diagnostic,
        metrics.losses.auxiliaries.leak_topology_to_geometry,
        metrics.losses.auxiliaries.leak_geometry_to_topology,
        metrics.losses.auxiliaries.leak_pocket_to_geometry,
        metrics
            .losses
            .auxiliaries
            .leak_topology_to_pocket_role,
        metrics
            .losses
            .auxiliaries
            .leak_geometry_to_pocket_role,
        metrics
            .losses
            .auxiliaries
            .leak_pocket_to_topology_role,
        metrics.losses.auxiliaries.leak_pocket_to_ligand_role,
        metrics.losses.auxiliaries.gate,
        metrics.losses.auxiliaries.slot,
        metrics.losses.auxiliaries.consistency,
        metrics.losses.auxiliaries.pocket_contact,
        metrics.losses.auxiliaries.pocket_pair_distance,
        metrics.losses.auxiliaries.pocket_clash,
        metrics.losses.auxiliaries.pocket_shape_complementarity,
        metrics.losses.auxiliaries.pocket_envelope,
        metrics.losses.auxiliaries.pocket_prior,
        metrics.losses.auxiliaries.pocket_prior_atom_count,
        metrics.losses.auxiliaries.pocket_prior_composition,
        metrics.losses.auxiliaries.pocket_prior_atom_count_mae,
        metrics.losses.auxiliaries.valence_guardrail,
        metrics.losses.auxiliaries.valence_overage_guardrail,
        metrics.losses.auxiliaries.valence_underage_guardrail,
        metrics.losses.auxiliaries.bond_length_guardrail,
        metrics.losses.auxiliaries.nonbonded_distance_guardrail,
        metrics.losses.auxiliaries.angle_guardrail,
        metrics.interaction.mean_gate,
        metrics.interaction.mean_gate_sparsity,
        metrics.interaction.mean_attention_entropy,
        metrics.gradient_health.global_grad_l2_norm,
        metrics.gradient_health.nonfinite_gradient_tensors,
        metrics.gradient_health.clipped,
        metrics.gradient_health.optimizer_step_skipped,
        metrics.synchronization.mask_count_mismatch,
        metrics.synchronization.slot_count_mismatch,
        metrics.synchronization.coordinate_frame_mismatch,
        metrics.synchronization.stale_context_steps,
        metrics.synchronization.refresh_count,
        metrics.synchronization.batch_slice_sync_pass,
    );
    for record in &metrics.losses.primary.component_provenance {
        println!(
            "  primary component {} anchor={} target_source={} role={} differentiable={} optimizer_facing={}",
            record.component_name,
            record.anchor,
            record.target_source,
            record.role,
            record.differentiable,
            record.optimizer_facing
        );
    }
    println!(
        "  stage ramp={:.4} promotion_gate={} active_objectives={}",
        metrics.stage_progress.stage_ramp,
        metrics.stage_progress.promotion_gate_decision,
        metrics.stage_progress.active_objective_families.join(",")
    );
    println!(
        "  slot utilization active_count={:.4} active_fraction={:.4} visible_fraction={:.4} entropy={:.4} mass_max={:.4} mass_effective={:.4} dead_fraction={:.4} collapse_warnings={} mass_warnings={}",
        metrics.slot_utilization.mean_active_slot_count,
        metrics.slot_utilization.mean_active_slot_fraction,
        metrics.slot_utilization.mean_attention_visible_slot_fraction,
        metrics.slot_utilization.mean_slot_entropy,
        metrics.slot_utilization.mean_slot_mass_max_fraction,
        metrics.slot_utilization.mean_slot_mass_effective_count,
        metrics.slot_utilization.dead_slot_fraction,
        metrics.slot_utilization.collapse_warning_count,
        metrics.slot_utilization.mass_concentration_warning_count
    );
    println!(
        "  primary weight={:.4} weighted={:.4} enabled={}",
        metrics.losses.primary.effective_weight,
        metrics.losses.primary.weighted_value,
        metrics.losses.primary.enabled
    );
    println!(
        "  rollout training enabled={} active={} steps={:.2} teacher_forced={:.4} rollout_state={:.4} divergence={:.4} generated_validity={:.4} bond_sparse_negative_rate={:.4}",
        metrics.losses.rollout_training.enabled,
        metrics.losses.rollout_training.active,
        metrics.losses.rollout_training.executed_steps_mean,
        metrics.losses.rollout_training.teacher_forced_loss,
        metrics.losses.rollout_training.rollout_state_loss,
        metrics.losses.rollout_training.teacher_rollout_divergence,
        metrics.losses.rollout_training.generated_state_validity,
        metrics.losses.rollout_training.bond_sparse_negative_rate
    );
    println!(
        "  objective gradients enabled={} sampled={} mode={} dominant_count={} threshold={:.4}",
        metrics.gradient_health.objective_families.enabled,
        metrics.gradient_health.objective_families.sampled,
        metrics.gradient_health.objective_families.sampling_mode,
        metrics
            .gradient_health
            .objective_families
            .dominant_family_count,
        metrics
            .gradient_health
            .objective_families
            .dominance_fraction_threshold
    );
    for entry in &metrics.gradient_health.objective_families.entries {
        println!(
            "  objective_gradient_family {} weighted={:.4} grad_l2={:.4} grad_fraction={:.4} status={} provenance={} anomaly={}",
            entry.family_name,
            entry.weighted_value,
            entry.grad_l2_norm,
            entry.grad_norm_fraction,
            entry.status,
            entry.provenance,
            entry.anomaly.as_deref().unwrap_or("")
        );
    }
    for entry in &metrics.losses.objective_family_budget_report.entries {
        println!(
            "  objective_family {} unweighted={:.4} weight={:.4} weighted={:.4} raw_weighted={:.4} pct_total={:.4} enabled={} status={} budget_cap={:?} budget_action={} clamped={} warning={}",
            entry.family,
            entry.unweighted_value,
            entry.effective_weight,
            entry.weighted_value,
            entry.raw_weighted_value,
            entry.percentage_of_total,
            entry.enabled,
            entry.status,
            entry.budget_cap_fraction,
            entry.budget_action,
            entry.budget_clamped,
            entry.warning.as_deref().unwrap_or("")
        );
    }
    for entry in &metrics
        .losses
        .auxiliaries
        .auxiliary_objective_report
        .entries
    {
        println!(
            "  objective {} unweighted={:.4} weight={:.4} weighted={:.4} enabled={} status={} warning={}",
            entry.family.as_str(),
            entry.unweighted_value,
            entry.effective_weight,
            entry.weighted_value,
            entry.enabled,
            entry.status,
            entry.warning.as_deref().unwrap_or("")
        );
    }
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
        "    slot assignment entropy mean: {:.4}",
        metrics
            .representation_diagnostics
            .slot_assignment_entropy_mean
    );
    println!(
        "    slot activation probability mean: {:.4}",
        metrics
            .representation_diagnostics
            .slot_activation_probability_mean
    );
    println!(
        "    attention-visible slot fraction: {:.4}",
        metrics
            .representation_diagnostics
            .attention_visible_slot_fraction
    );
    println!(
        "    gate activation mean: {:.4}",
        metrics.representation_diagnostics.gate_activation_mean
    );
    println!(
        "    leakage proxy mean: {:.4}",
        metrics.representation_diagnostics.leakage_proxy_mean
    );
    println!("  chemistry collaboration:");
    println!(
        "    pharmacophore role coverage: {}",
        format_chemistry_metric(&metrics.chemistry_collaboration.pharmacophore_role_coverage)
    );
    println!(
        "    role conflict rate: {}",
        format_chemistry_metric(&metrics.chemistry_collaboration.role_conflict_rate)
    );
    println!(
        "    severe clash fraction: {}",
        format_chemistry_metric(&metrics.chemistry_collaboration.severe_clash_fraction)
    );
    println!(
        "    valence violation fraction: {}",
        format_chemistry_metric(&metrics.chemistry_collaboration.valence_violation_fraction)
    );
    println!(
        "    bond length guardrail mean: {}",
        format_chemistry_metric(&metrics.chemistry_collaboration.bond_length_guardrail_mean)
    );
    println!(
        "    key residue contact coverage: {}",
        format_chemistry_metric(&metrics.chemistry_collaboration.key_residue_contact_coverage)
    );
    for role in &metrics.chemistry_collaboration.gate_usage_by_chemical_role {
        println!(
            "    gate {}: {}",
            role.chemical_role,
            format_chemistry_metric(&role.gate_mean)
        );
    }
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
    for baseline in &metrics.proxy_task_metrics.probe_baselines {
        println!(
            "    probe [{}] {}: observed={} trivial={} improves={} status={} count={} target_pos={} pred_pos={} pos_gap={} pos_loss={} neg_loss={} target_mean={} pred_mean={} mean_error={}",
            baseline.target,
            baseline.loss_kind,
            format_optional_f64(baseline.observed_loss),
            format_optional_f64(baseline.trivial_baseline_loss),
            format_optional_bool(baseline.improves_over_trivial),
            baseline.supervision_status,
            baseline.available_count,
            format_optional_f64(baseline.target_positive_rate),
            format_optional_f64(baseline.prediction_positive_rate),
            format_optional_f64(baseline.positive_rate_gap),
            format_optional_f64(baseline.positive_observed_loss),
            format_optional_f64(baseline.negative_observed_loss),
            format_optional_f64(baseline.scalar_target_mean),
            format_optional_f64(baseline.scalar_prediction_mean),
            format_optional_f64(baseline.scalar_mean_error)
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
        "    eval batches: size={} count={} no_grad={} batched={}",
        metrics.resource_usage.evaluation_batch_size,
        metrics.resource_usage.forward_batch_count,
        metrics.resource_usage.no_grad,
        metrics.resource_usage.batched_forward
    );
    println!(
        "    avg atoms ligand/pocket: {:.2}/{:.2}",
        metrics.resource_usage.average_ligand_atoms, metrics.resource_usage.average_pocket_atoms
    );
    println!("  model-design diagnostics:");
    println!(
        "    heldout unseen={:.4} finite_forward={:.4} geometry_score={:.4}",
        metrics.model_design.heldout_unseen_protein_fraction,
        metrics.model_design.finite_forward_fraction,
        metrics.model_design.geometry_consistency_score
    );
    println!(
        "    raw [{}]: valid={:.4} contact={:.4} clash={:.4}",
        metrics.model_design.raw_model_layer,
        metrics.model_design.raw_model_valid_fraction,
        metrics.model_design.raw_model_pocket_contact_fraction,
        metrics.model_design.raw_model_clash_fraction
    );
    println!(
        "    processed [{}]: valid={:.4} contact={:.4} clash={:.4}",
        metrics.model_design.processed_layer,
        metrics.model_design.processed_valid_fraction,
        metrics.model_design.processed_pocket_contact_fraction,
        metrics.model_design.processed_clash_fraction
    );
    println!(
        "    slot={:.4} gate={:.4} gate_saturation={:.4} gate_closed={:.4} gate_grad={:.4} leakage={:.4}",
        metrics.model_design.slot_activation_mean,
        metrics.model_design.gate_activation_mean,
        metrics.model_design.gate_saturation_fraction,
        metrics.model_design.gate_closed_fraction_mean,
        metrics.model_design.gate_gradient_proxy_mean,
        metrics.model_design.leakage_proxy_mean
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
        "    primary objective provenance: {}",
        metrics.comparison_summary.primary_objective_provenance
    );
    println!(
        "    primary objective claim boundary: {}",
        metrics.comparison_summary.primary_objective_claim_boundary
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

fn format_chemistry_metric(metric: &ChemistryCollaborationMetric) -> String {
    match metric.value {
        Some(value) => format!("{value:.4} [{:?}; {}]", metric.provenance, metric.status),
        None => format!("NA [{:?}; {}]", metric.provenance, metric.status),
    }
}

fn print_candidate_layer(label: &str, layer: &crate::experiments::CandidateLayerMetrics) {
    println!(
        "    {label}: count={} valid={:.4} contact={:.4} distance_bin={:.4} precision={:.4} recall={:.4} role={:.4} centroid_offset={:.4} clash={:.4} displacement={:.4} atom_change={:.4} unique_proxy={:.4} graph_valid={:.4} repair_delta={:.2} raw_removed={:.2} conn_added={:.2} valence_down={:.2} provenance={}",
        layer.candidate_count,
        layer.valid_fraction,
        layer.pocket_contact_fraction,
        layer.pocket_distance_bin_accuracy,
        layer.pocket_contact_precision_proxy,
        layer.pocket_contact_recall_proxy,
        layer.pocket_role_compatibility_proxy,
        layer.mean_centroid_offset,
        layer.clash_fraction,
        layer.mean_displacement,
        layer.atom_change_fraction,
        layer.uniqueness_proxy_fraction,
        layer.native_graph_valid_fraction,
        layer.native_graph_repair_delta_mean,
        layer.native_raw_to_constrained_removed_bond_count_mean,
        layer.native_connectivity_guardrail_added_bond_count_mean,
        layer.native_valence_guardrail_downgrade_count_mean,
        layer.pocket_interaction_provenance,
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
        report.retained_approximate_affinity_labels, report.retained_approximate_label_fraction
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
        report.retained_missing_measurement_type, report.retained_missing_normalization_provenance
    );
    if !report.retained_measurement_family_histogram.is_empty() {
        println!(
            "  retained measurement families ({}): {:?}",
            report.retained_measurement_family_count, report.retained_measurement_family_histogram
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

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "unavailable".to_string())
}

fn format_optional_bool(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn print_split_stats(name: &str, stats: &crate::training::SplitStats) {
    println!(
        "  {}: examples={} proteins={} labeled={} ({:.4}) measurements={:?} protein_families={:?} pocket_families={:?} ligand_scaffolds={:?} unavailable={:?}",
        name,
        stats.example_count,
        stats.unique_protein_count,
        stats.labeled_example_count,
        stats.labeled_fraction,
        stats.affinity_measurement_family_histogram,
        stats.protein_family_proxy_histogram,
        stats.pocket_family_proxy_histogram,
        stats.ligand_scaffold_proxy_histogram,
        stats.metadata_availability.unavailable_fields
    );
}
