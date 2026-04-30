fn build_comparison_summary(
    research: &ResearchConfig,
    ablation: &AblationConfig,
    unseen_protein_fraction: f64,
    topology_specialization_score: f64,
    geometry_specialization_score: f64,
    pocket_specialization_score: f64,
    slot_activation_mean: f64,
    gate_activation_mean: f64,
    leakage_proxy_mean: f64,
    metrics: &RealGenerationMetrics,
    chemistry_collaboration: &ChemistryCollaborationMetrics,
) -> GenerationQualitySummary {
    let active_primary_objective = ablation
        .primary_objective_override
        .unwrap_or(research.training.primary_objective);
    GenerationQualitySummary {
        generation_mode: generation_mode_label(research),
        primary_objective: primary_objective_label(active_primary_objective),
        primary_objective_provenance: primary_objective_provenance_label(active_primary_objective),
        primary_objective_claim_boundary: primary_objective_claim_boundary_label(
            active_primary_objective,
        ),
        variant_label: ablation.variant_label.clone(),
        interaction_mode: interaction_mode_label(
            ablation
                .interaction_mode_override
                .unwrap_or(research.model.interaction_mode),
        ),
        candidate_valid_fraction: metric_value_with_heuristic_fallback(
            &metrics.chemistry_validity,
            "valid_fraction",
        ),
        pocket_contact_fraction: metric_value_with_heuristic_fallback(
            &metrics.docking_affinity,
            "pocket_contact_fraction",
        )
        .or_else(|| {
            metric_value_with_heuristic_fallback(&metrics.docking_affinity, "contact_fraction")
        }),
        pocket_compatibility_fraction: metric_value_with_heuristic_fallback(
            &metrics.pocket_compatibility,
            "centroid_inside_fraction",
        )
        .or_else(|| {
            metric_value_with_heuristic_fallback(
                &metrics.pocket_compatibility,
                "atom_coverage_fraction",
            )
        }),
        mean_centroid_offset: metric_value_with_heuristic_fallback(
            &metrics.docking_affinity,
            "mean_centroid_offset",
        ),
        strict_pocket_fit_score: metric_value_with_heuristic_fallback(
            &metrics.pocket_compatibility,
            "strict_pocket_fit_score",
        )
        .or_else(|| {
            let coverage = metric_value_with_heuristic_fallback(
                &metrics.pocket_compatibility,
                "atom_coverage_fraction",
            )?;
            let centroid_fit = metric_value_with_heuristic_fallback(
                &metrics.docking_affinity,
                "centroid_fit_score",
            )
            .or_else(|| {
                metric_value_with_heuristic_fallback(
                    &metrics.docking_affinity,
                    "mean_centroid_offset",
                )
                .map(|offset| 1.0 / (1.0 + offset))
            })?;
            Some(coverage * centroid_fit)
        }),
        unique_smiles_fraction: metric_value_with_heuristic_fallback(
            &metrics.chemistry_validity,
            "rdkit_unique_smiles_fraction",
        )
        .or_else(|| {
            metric_value_with_heuristic_fallback(
                &metrics.chemistry_validity,
                "unique_smiles_fraction",
            )
        }),
        unseen_protein_fraction,
        topology_specialization_score,
        geometry_specialization_score,
        pocket_specialization_score,
        slot_activation_mean,
        gate_activation_mean,
        leakage_proxy_mean,
        chemistry_collaboration: chemistry_collaboration.clone(),
    }
}

fn build_train_eval_alignment_report(
    research: &ResearchConfig,
    real_generation_metrics: &RealGenerationMetrics,
    method_comparison: &MethodComparisonSummary,
    model_design: &ModelDesignEvaluationMetrics,
    finite_forward_fraction: f64,
    distance_probe_rmse: f64,
    leakage_proxy_mean: f64,
    affinity_probe_mae: f64,
    labeled_fraction: f64,
    examples_per_second: f64,
    comparison_summary: &GenerationQualitySummary,
) -> TrainEvalAlignmentReport {
    let probe_active = research.training.loss_weights.gamma_probe > 0.0;
    let leakage_active = research.training.loss_weights.delta_leak > 0.0;
    let backend_coverage = vec![
        backend_coverage_contract_row(
            "chemistry_validity",
            &real_generation_metrics.chemistry_validity,
            "chemistry validity metrics require explicit chemistry backend availability or labeled heuristic fallback",
        ),
        backend_coverage_contract_row(
            "docking_affinity",
            &real_generation_metrics.docking_affinity,
            "docking/contact metrics require explicit docking backend availability or labeled heuristic fallback",
        ),
        backend_coverage_contract_row(
            "pocket_compatibility",
            &real_generation_metrics.pocket_compatibility,
            "pocket-fit metrics require explicit pocket backend availability or labeled heuristic fallback",
        ),
    ];

    let chemistry_coverage = backend_coverage
        .iter()
        .find(|row| row.backend_slot == "chemistry_validity")
        .and_then(|row| row.coverage_fraction);
    let docking_coverage = backend_coverage
        .iter()
        .find(|row| row.backend_slot == "docking_affinity")
        .and_then(|row| row.coverage_fraction);
    let pocket_coverage = backend_coverage
        .iter()
        .find(|row| row.backend_slot == "pocket_compatibility")
        .and_then(|row| row.coverage_fraction);

    let active_method_family = method_comparison.active_method_family.clone();
    let active_selected_metric_layer = method_comparison.active_selected_metric_layer.clone();
    let active_method_row = method_comparison.methods.iter().find(|row| {
        Some(row.method_id.as_str())
            == method_comparison
                .active_method
                .as_ref()
                .map(|method| method.method_id.as_str())
    });
    let active_selected_layer_raw = active_method_row
        .map(|row| row.selected_metric_layer_model_native_raw)
        .unwrap_or_else(|| {
            active_selected_metric_layer
                .as_deref()
                .map(layer_name_is_model_native_raw)
                .unwrap_or(false)
        });
    let backend_selection =
        backend_metric_selection_context(research, method_comparison, active_selected_metric_layer.as_deref());

    let mut metric_rows = vec![
        TrainEvalAlignmentMetricRow {
            metric_name: "finite_forward_fraction".to_string(),
            metric_family: "representation_health".to_string(),
            target_source: "model_forward_state".to_string(),
            evidence_role: "smoke_diagnostic".to_string(),
            observed_value: Some(finite_forward_fraction),
            optimizer_facing_terms: Vec::new(),
            optimizer_facing: false,
            detached_diagnostic: true,
            candidate_layer: None,
            model_native_raw: false,
            method_family: None,
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "finite forward fraction is a smoke metric, not a model-quality objective"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "distance_probe_rmse".to_string(),
            metric_family: "geometry_probe".to_string(),
            target_source: "auxiliary_probe_target".to_string(),
            evidence_role: if probe_active {
                "optimizer_term"
            } else {
                "disabled_probe_diagnostic"
            }
            .to_string(),
            observed_value: Some(distance_probe_rmse),
            optimizer_facing_terms: vec!["L_probe".to_string()],
            optimizer_facing: probe_active,
            detached_diagnostic: !probe_active,
            candidate_layer: None,
            model_native_raw: false,
            method_family: None,
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "geometry probe quality is optimizer-facing only when gamma_probe is active"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "leakage_proxy_mean".to_string(),
            metric_family: "leakage_control".to_string(),
            target_source: "off_modality_probe_target".to_string(),
            evidence_role: if leakage_active {
                "optimizer_term"
            } else {
                "disabled_leakage_diagnostic"
            }
            .to_string(),
            observed_value: Some(leakage_proxy_mean),
            optimizer_facing_terms: vec!["L_leak".to_string()],
            optimizer_facing: leakage_active,
            detached_diagnostic: !leakage_active,
            candidate_layer: None,
            model_native_raw: false,
            method_family: None,
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "leakage proxy is a specialization risk signal and does not prove no leakage alone"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "affinity_probe_mae".to_string(),
            metric_family: "context_probe".to_string(),
            target_source: "auxiliary_probe_target".to_string(),
            evidence_role: if probe_active {
                "optimizer_term"
            } else {
                "disabled_probe_diagnostic"
            }
            .to_string(),
            observed_value: if labeled_fraction > 0.0 {
                Some(affinity_probe_mae)
            } else {
                None
            },
            optimizer_facing_terms: vec!["L_probe".to_string()],
            optimizer_facing: probe_active && labeled_fraction > 0.0,
            detached_diagnostic: !(probe_active && labeled_fraction > 0.0),
            candidate_layer: None,
            model_native_raw: false,
            method_family: None,
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "affinity probe evidence requires labeled examples and active probe weighting"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "rollout_eval_recovery".to_string(),
            metric_family: "rollout_eval".to_string(),
            target_source: "generated_rollout_state".to_string(),
            evidence_role: "detached_diagnostic".to_string(),
            observed_value: Some(model_design.raw_model_valid_fraction),
            optimizer_facing_terms: Vec::new(),
            optimizer_facing: false,
            detached_diagnostic: true,
            candidate_layer: Some("raw_rollout".to_string()),
            model_native_raw: true,
            method_family: active_method_family.clone(),
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "rollout_eval metrics are detached rollout diagnostics unless trainable rollout loss is implemented"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "raw_model_valid_fraction".to_string(),
            metric_family: "raw_candidate".to_string(),
            target_source: "generated_rollout_state".to_string(),
            evidence_role: "model_native_raw".to_string(),
            observed_value: Some(model_design.raw_model_valid_fraction),
            optimizer_facing_terms: Vec::new(),
            optimizer_facing: false,
            detached_diagnostic: true,
            candidate_layer: Some(model_design.raw_model_layer.clone()),
            model_native_raw: true,
            method_family: active_method_family.clone(),
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "raw model quality is measured before repair, constraints, reranking, or backend scoring"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "raw_native_graph_valid_fraction".to_string(),
            metric_family: "raw_native_graph".to_string(),
            target_source: "generated_rollout_state".to_string(),
            evidence_role: "model_native_raw".to_string(),
            observed_value: Some(model_design.raw_native_graph_valid_fraction),
            optimizer_facing_terms: Vec::new(),
            optimizer_facing: false,
            detached_diagnostic: true,
            candidate_layer: Some(model_design.raw_model_layer.clone()),
            model_native_raw: true,
            method_family: active_method_family.clone(),
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "native graph validity is computed on raw rollout atom and bond payloads before repair or inferred-bond layers"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "processed_valid_fraction".to_string(),
            metric_family: "processed_candidate".to_string(),
            target_source: "repair_layer".to_string(),
            evidence_role: "postprocessed_candidate".to_string(),
            observed_value: Some(model_design.processed_valid_fraction),
            optimizer_facing_terms: Vec::new(),
            optimizer_facing: false,
            detached_diagnostic: true,
            candidate_layer: Some(model_design.processed_layer.clone()),
            model_native_raw: layer_name_is_model_native_raw(&model_design.processed_layer),
            method_family: active_method_family.clone(),
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "processed candidate quality may include repair, constraints, or selection and must not overwrite raw fields"
                    .to_string(),
        },
        TrainEvalAlignmentMetricRow {
            metric_name: "method_comparison.active_selected_metric_layer".to_string(),
            metric_family: "method_comparison".to_string(),
            target_source: "generation_layer_selection".to_string(),
            evidence_role: "family_aware_metric_layer".to_string(),
            observed_value: None,
            optimizer_facing_terms: Vec::new(),
            optimizer_facing: false,
            detached_diagnostic: true,
            candidate_layer: active_selected_metric_layer,
            model_native_raw: active_selected_layer_raw,
            method_family: active_method_family,
            backend_slot: None,
            backend_name: None,
            backend_available: None,
            backend_coverage_fraction: None,
            claim_boundary:
                "method-level metrics must use the selected layer appropriate for the method family"
                    .to_string(),
        },
    ];

    metric_rows.push(backend_metric_alignment_row(
        "candidate_valid_fraction",
        "chemistry_validity",
        &real_generation_metrics.chemistry_validity,
        comparison_summary.candidate_valid_fraction,
        chemistry_coverage,
        "claim_backend",
        "chemistry validity may be backend-backed or explicitly labeled heuristic fallback",
        &backend_selection,
    ));
    metric_rows.push(backend_metric_alignment_row(
        "unique_smiles_fraction",
        "chemistry_validity",
        &real_generation_metrics.chemistry_validity,
        comparison_summary.unique_smiles_fraction,
        chemistry_coverage,
        "claim_backend",
        "unique chemistry evidence must preserve backend or heuristic provenance",
        &backend_selection,
    ));
    metric_rows.push(backend_metric_alignment_row(
        "pocket_contact_fraction",
        "docking_affinity",
        &real_generation_metrics.docking_affinity,
        comparison_summary.pocket_contact_fraction,
        docking_coverage,
        "claim_backend",
        "pocket contact evidence must not be treated as optimizer-facing unless a differentiable objective exists",
        &backend_selection,
    ));
    metric_rows.push(backend_metric_alignment_row(
        "strict_pocket_fit_score",
        "pocket_compatibility",
        &real_generation_metrics.pocket_compatibility,
        comparison_summary.strict_pocket_fit_score,
        pocket_coverage,
        "claim_backend",
        "strict pocket-fit evidence requires pocket backend coverage or explicitly labeled heuristic fallback",
        &backend_selection,
    ));
    metric_rows.push(TrainEvalAlignmentMetricRow {
        metric_name: "examples_per_second".to_string(),
        metric_family: "efficiency".to_string(),
        target_source: "runtime_profile".to_string(),
        evidence_role: "runtime_diagnostic".to_string(),
        observed_value: Some(examples_per_second),
        optimizer_facing_terms: Vec::new(),
        optimizer_facing: false,
        detached_diagnostic: true,
        candidate_layer: None,
        model_native_raw: false,
        method_family: None,
        backend_slot: None,
        backend_name: None,
        backend_available: None,
        backend_coverage_fraction: None,
        claim_boundary: "runtime metric for efficiency comparison, not candidate-quality evidence"
            .to_string(),
    });

    let best_metric_review = best_metric_review(
        research,
        finite_forward_fraction,
        distance_probe_rmse,
        leakage_proxy_mean,
        affinity_probe_mae,
        labeled_fraction,
        examples_per_second,
        comparison_summary,
    );
    let backend_rows_ready = backend_coverage.iter().all(|row| {
        !row.available || row.examples_scored.unwrap_or(0.0) > 0.0 || row.coverage_fraction.is_some()
    });
    let raw_processed_separated = metric_rows.iter().any(|row| {
        row.metric_name == "raw_model_valid_fraction"
            && row.model_native_raw
            && row.candidate_layer.as_deref() == Some("raw_rollout")
    }) && metric_rows.iter().any(|row| {
        row.metric_name == "processed_valid_fraction"
            && row.candidate_layer.as_deref() != Some("raw_rollout")
            && !row.model_native_raw
    });
    let rollout_eval_detached = metric_rows.iter().any(|row| {
        row.metric_name == "rollout_eval_recovery"
            && !row.optimizer_facing
            && row.detached_diagnostic
    });
    let decision = if backend_rows_ready && raw_processed_separated && rollout_eval_detached {
        "pass: metrics are attributed to optimizer terms, raw/processed layers, and backend coverage"
    } else if rollout_eval_detached {
        "caution: alignment emitted, but processed layer or backend coverage is incomplete"
    } else {
        "fail: rollout_eval metrics are not safely separated from optimizer-facing terms"
    }
    .to_string();

    TrainEvalAlignmentReport {
        schema_version: 1,
        metric_rows,
        backend_coverage,
        best_metric_review,
        decision,
    }
}

fn backend_metric_alignment_row(
    metric_name: &str,
    backend_slot: &str,
    metrics: &ReservedBackendMetrics,
    observed_value: Option<f64>,
    backend_coverage_fraction: Option<f64>,
    evidence_role: &str,
    claim_boundary: &str,
    selection: &BackendMetricSelectionContext,
) -> TrainEvalAlignmentMetricRow {
    let claim_boundary = if let Some(layer) = &selection.candidate_layer {
        format!(
            "{claim_boundary}; evaluated_candidate_layer={layer}; selection_reason={}",
            selection.selection_reason
        )
    } else {
        claim_boundary.to_string()
    };
    TrainEvalAlignmentMetricRow {
        metric_name: metric_name.to_string(),
        metric_family: "backend_candidate".to_string(),
        target_source: selection.target_source.clone(),
        evidence_role: evidence_role.to_string(),
        observed_value,
        optimizer_facing_terms: Vec::new(),
        optimizer_facing: false,
        detached_diagnostic: true,
        candidate_layer: selection.candidate_layer.clone(),
        model_native_raw: selection.model_native_raw,
        method_family: None,
        backend_slot: Some(backend_slot.to_string()),
        backend_name: metrics.backend_name.clone(),
        backend_available: Some(metrics.available),
        backend_coverage_fraction,
        claim_boundary,
    }
}

#[derive(Debug, Clone)]
struct BackendMetricSelectionContext {
    target_source: String,
    candidate_layer: Option<String>,
    model_native_raw: bool,
    selection_reason: String,
}

fn backend_metric_selection_context(
    research: &ResearchConfig,
    method_comparison: &MethodComparisonSummary,
    active_selected_metric_layer: Option<&str>,
) -> BackendMetricSelectionContext {
    let flow_contract =
        crate::models::flow::current_multimodal_flow_contract(&research.generation_method.flow_matching);
    let full_branch_runtime = !research.generation_method.flow_matching.geometry_only
        && flow_contract.disabled_branches.is_empty();
    if full_branch_runtime && method_comparison.raw_native_evidence.model_native_raw {
        return BackendMetricSelectionContext {
            target_source: "external_backend".to_string(),
            candidate_layer: Some(method_comparison.raw_native_evidence.raw_model_layer.clone()),
            model_native_raw: true,
            selection_reason: if flow_contract.full_molecular_flow_claim_allowed {
                "full_molecular_flow_raw_rollout_preferred".to_string()
            } else {
                format!(
                    "full_branch_raw_rollout_preferred_but_claim_gate_blocked_by_{}",
                    flow_contract.claim_gate_reason
                )
            },
        };
    }
    BackendMetricSelectionContext {
        target_source: "external_backend".to_string(),
        candidate_layer: active_selected_metric_layer.map(str::to_string),
        model_native_raw: active_selected_metric_layer
            .map(layer_name_is_model_native_raw)
            .unwrap_or(false),
        selection_reason: if research.generation_method.flow_matching.geometry_only {
            "geometry_only_or_non_flow_metric_layer".to_string()
        } else {
            format!(
                "processed_or_selected_layer_used_because_full_branch_runtime={full_branch_runtime}"
            )
        },
    }
}

fn backend_coverage_contract_row(
    backend_slot: &str,
    metrics: &ReservedBackendMetrics,
    claim_boundary: &str,
) -> BackendCoverageContractRow {
    let heuristic_fallback_labeled = metrics
        .metrics
        .keys()
        .any(|key| key.starts_with("heuristic_"));
    let examples_scored =
        first_metric_value(metrics, &["backend_examples_scored", "examples_scored"]);
    let candidates_scored =
        first_metric_value(metrics, &["backend_candidates_scored", "candidates_scored"]);
    let missing_structure_fraction = first_metric_value(
        metrics,
        &[
            "backend_missing_structure_fraction",
            "missing_structure_fraction",
        ],
    );
    let coverage_fraction = first_metric_value(
        metrics,
        &[
            "backend_coverage_fraction",
            "coverage_fraction",
            "rdkit_sanitized_fraction",
            "vina_score_success_fraction",
            "atom_coverage_fraction",
            "centroid_inside_fraction",
            "backend_examples_scored",
        ],
    );
    let fallback_status = if !metrics.available {
        "unavailable".to_string()
    } else if heuristic_fallback_labeled {
        "heuristic_fallback_present".to_string()
    } else {
        "backend_reported".to_string()
    };

    BackendCoverageContractRow {
        backend_slot: backend_slot.to_string(),
        backend_name: metrics.backend_name.clone(),
        available: metrics.available,
        examples_scored,
        candidates_scored,
        missing_structure_fraction,
        coverage_fraction,
        fallback_status,
        heuristic_fallback_labeled,
        claim_boundary: claim_boundary.to_string(),
    }
}

fn first_metric_value(metrics: &ReservedBackendMetrics, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| backend_metric(metrics, key))
}

fn layer_name_is_model_native_raw(layer: &str) -> bool {
    matches!(layer, "raw_flow" | "raw_rollout" | "raw_geometry_candidates")
}

fn best_metric_review(
    research: &ResearchConfig,
    finite_forward_fraction: f64,
    distance_probe_rmse: f64,
    leakage_proxy_mean: f64,
    affinity_probe_mae: f64,
    labeled_fraction: f64,
    examples_per_second: f64,
    comparison_summary: &GenerationQualitySummary,
) -> BestMetricReview {
    let configured_best_metric = research.training.best_metric.clone();
    let normalized_best_metric = research.resolved_best_metric();
    let metric_available = match normalized_best_metric.as_str() {
        "finite_forward_fraction" => finite_forward_fraction.is_finite(),
        "strict_pocket_fit_score" => comparison_summary.strict_pocket_fit_score.is_some(),
        "candidate_valid_fraction" => comparison_summary.candidate_valid_fraction.is_some(),
        "leakage_proxy_mean" => leakage_proxy_mean.is_finite(),
        "distance_probe_rmse" => distance_probe_rmse.is_finite(),
        "affinity_probe_mae" => labeled_fraction > 0.0 && affinity_probe_mae.is_finite(),
        "examples_per_second" => examples_per_second.is_finite() && examples_per_second > 0.0,
        _ => false,
    };
    let quality_aware = matches!(
        normalized_best_metric.as_str(),
        "strict_pocket_fit_score"
            | "candidate_valid_fraction"
            | "leakage_proxy_mean"
            | "distance_probe_rmse"
            | "affinity_probe_mae"
    );
    let availability_requirement = match normalized_best_metric.as_str() {
        "finite_forward_fraction" => "requires finite model forwards only",
        "strict_pocket_fit_score" => "requires pocket compatibility backend or labeled heuristic fallback",
        "candidate_valid_fraction" => "requires chemistry validity backend or labeled heuristic fallback",
        "leakage_proxy_mean" => "requires leakage diagnostics emitted by the model",
        "distance_probe_rmse" => "requires geometry probe targets and predictions",
        "affinity_probe_mae" => "requires labeled affinity examples",
        "examples_per_second" => "requires measured evaluation runtime",
        _ => "unsupported validation metric",
    }
    .to_string();
    let claim_bearing_recommended = metric_available && quality_aware;
    let (status, warning) = if !metric_available {
        (
            "unavailable".to_string(),
            Some(format!(
                "best metric `{normalized_best_metric}` is unavailable under current evaluation coverage"
            )),
        )
    } else if normalized_best_metric == "finite_forward_fraction" {
        (
            "smoke_default".to_string(),
            Some(
                "finite_forward_fraction is acceptable for smoke configs but is not quality-aware"
                    .to_string(),
            ),
        )
    } else if quality_aware {
        ("claim_metric_candidate".to_string(), None)
    } else {
        (
            "diagnostic_only".to_string(),
            Some(format!(
                "best metric `{normalized_best_metric}` is diagnostic, not claim-quality evidence"
            )),
        )
    };

    BestMetricReview {
        configured_best_metric,
        normalized_best_metric,
        metric_available,
        quality_aware,
        claim_bearing_recommended,
        availability_requirement,
        status,
        warning,
    }
}

fn build_model_design_evaluation_metrics(
    finite_forward_fraction: f64,
    unseen_protein_fraction: f64,
    topology_reconstruction_mse: f64,
    distance_probe_rmse: f64,
    slot_activation_mean: f64,
    gate_activation_mean: f64,
    leakage_proxy_mean: f64,
    slot_stability: &SlotStabilityMetrics,
    forwards: &[ResearchForward],
    layered: &LayeredGenerationMetrics,
    examples_per_second: f64,
    memory_usage_mb: f64,
) -> ModelDesignEvaluationMetrics {
    let raw = &layered.raw_rollout;
    let (processed_layer, processed) = preferred_model_design_processed_layer(layered);
    let processed_contract = generation_path_contract_for_layer(layered, processed_layer);
    let gate_health = aggregate_gate_health(forwards);
    let slot_signature_similarity_mean = (slot_stability.topology_signature_similarity
        + slot_stability.geometry_signature_similarity
        + slot_stability.pocket_signature_similarity)
        / 3.0;

    ModelDesignEvaluationMetrics {
        heldout_unseen_protein_fraction: unseen_protein_fraction,
        finite_forward_fraction,
        ligand_topology_reconstruction_mse: topology_reconstruction_mse,
        geometry_distance_probe_rmse: distance_probe_rmse,
        geometry_consistency_score: 1.0 / (1.0 + distance_probe_rmse.max(0.0)),
        local_pocket_contact_fraction: processed.pocket_contact_fraction,
        local_pocket_clash_fraction: processed.clash_fraction,
        raw_model_valid_fraction: raw.valid_fraction,
        raw_model_pocket_contact_fraction: raw.pocket_contact_fraction,
        raw_model_clash_fraction: raw.clash_fraction,
        raw_model_mean_displacement: raw.mean_displacement,
        processed_valid_fraction: processed.valid_fraction,
        processed_pocket_contact_fraction: processed.pocket_contact_fraction,
        processed_clash_fraction: processed.clash_fraction,
        processing_valid_fraction_delta: processed.valid_fraction - raw.valid_fraction,
        processing_pocket_contact_delta: processed.pocket_contact_fraction
            - raw.pocket_contact_fraction,
        processing_clash_delta: processed.clash_fraction - raw.clash_fraction,
        examples_per_second,
        memory_usage_mb,
        slot_activation_mean,
        slot_signature_similarity_mean,
        gate_activation_mean,
        gate_saturation_fraction: gate_health.saturation_fraction,
        gate_closed_fraction_mean: gate_health.closed_fraction,
        gate_open_fraction_mean: gate_health.open_fraction,
        gate_gradient_proxy_mean: gate_health.gradient_proxy,
        gate_effective_update_norm_mean: gate_health.effective_update_norm,
        gate_warning_count: gate_health.warning_count,
        gate_audit_note: gate_health.audit_note,
        leakage_proxy_mean,
        raw_model_layer: "raw_rollout".to_string(),
        processed_layer: processed_layer.to_string(),
        processed_postprocessor_chain: processed_contract
            .as_ref()
            .map(|row| row.postprocessor_chain.clone())
            .unwrap_or_default(),
        processed_claim_boundary: processed_contract
            .as_ref()
            .map(|row| row.claim_boundary.clone())
            .unwrap_or_else(|| "processed layer claim boundary unavailable".to_string()),
        raw_native_bond_count_mean: raw.native_bond_count_mean,
        raw_native_component_count_mean: raw.native_component_count_mean,
        raw_native_valence_violation_fraction: raw.native_valence_violation_fraction,
        raw_native_topology_bond_sync_fraction: raw.topology_bond_sync_fraction,
        raw_native_atom_type_entropy: raw.atom_type_entropy,
        raw_native_graph_valid_fraction: raw.native_graph_valid_fraction,
        raw_vs_processed_note:
            "raw_* fields are model-native rollout quality; processed_* fields may include repair, bond inference, or reranking and must not be cited as raw model quality"
                .to_string(),
    }
}

fn generation_path_contract_for_layer(
    layered: &LayeredGenerationMetrics,
    layer: &str,
) -> Option<GenerationPathContractRow> {
    layered.generation_path_contract.iter().find_map(|row| {
        (row.legacy_field_name == layer
            || row.canonical_layer == layer
            || (layer == "raw_flow" && row.legacy_field_name == "raw_rollout"))
        .then(|| row.clone())
    })
}

fn preferred_model_design_processed_layer(
    layered: &LayeredGenerationMetrics,
) -> (&'static str, &CandidateLayerMetrics) {
    if layered.reranked_candidates.candidate_count > 0 {
        ("reranked_candidates", &layered.reranked_candidates)
    } else if layered.inferred_bond_candidates.candidate_count > 0 {
        (
            "inferred_bond_candidates",
            &layered.inferred_bond_candidates,
        )
    } else if layered.repaired_candidates.candidate_count > 0 {
        ("repaired_candidates", &layered.repaired_candidates)
    } else if layered.constrained_flow.candidate_count > 0 {
        ("constrained_flow", &layered.constrained_flow)
    } else if layered.repaired.candidate_count > 0 {
        ("repaired", &layered.repaired)
    } else {
        ("raw_rollout", &layered.raw_rollout)
    }
}

struct GateHealthAggregate {
    saturation_fraction: f64,
    closed_fraction: f64,
    open_fraction: f64,
    gradient_proxy: f64,
    effective_update_norm: f64,
    warning_count: usize,
    audit_note: String,
}

fn aggregate_gate_health(forwards: &[ResearchForward]) -> GateHealthAggregate {
    let mut sum = 0.0;
    let mut closed = 0.0;
    let mut open = 0.0;
    let mut gradient = 0.0;
    let mut update_norm = 0.0;
    let mut count = 0usize;
    let mut warning_count = 0usize;
    for forward in forwards {
        for path_index in 0..6 {
            let path = interaction_path_diagnostic_at(forward, path_index);
            if path.gate_saturation_fraction.is_finite()
                && path.gate_closed_fraction.is_finite()
                && path.gate_open_fraction.is_finite()
                && path.gate_gradient_proxy.is_finite()
                && path.effective_update_norm.is_finite()
            {
                sum += path.gate_saturation_fraction;
                closed += path.gate_closed_fraction;
                open += path.gate_open_fraction;
                gradient += path.gate_gradient_proxy;
                update_norm += path.effective_update_norm;
                count += 1;
            }
            if path.gate_warning.is_some() {
                warning_count += 1;
            }
        }
    }
    let saturation_fraction = fraction_f64(sum, count);
    let closed_fraction = fraction_f64(closed, count);
    let open_fraction = fraction_f64(open, count);
    let gradient_proxy = fraction_f64(gradient, count);
    let effective_update_norm = fraction_f64(update_norm, count);
    let audit_note = if count == 0 {
        "gate diagnostics unavailable; no directed interaction paths were summarized".to_string()
    } else if closed_fraction >= 0.8 {
        "most directed gates are effectively closed; inspect gate bias, temperature, and sparsity schedule before interpreting low cross-modal usage as learned independence".to_string()
    } else if saturation_fraction >= 0.8 {
        "most directed gates are saturated; interaction gradients may be weak even when gate mean is nonzero".to_string()
    } else {
        "gate diagnostics summarize all six explicit directed paths; processed candidate quality remains separated from raw model-native quality".to_string()
    };
    GateHealthAggregate {
        saturation_fraction,
        closed_fraction,
        open_fraction,
        gradient_proxy,
        effective_update_norm,
        warning_count,
        audit_note,
    }
}

fn metric_value(metrics: &ReservedBackendMetrics, name: &str) -> Option<f64> {
    metrics.metrics.get(name).copied()
}

fn metric_value_with_heuristic_fallback(
    metrics: &ReservedBackendMetrics,
    name: &str,
) -> Option<f64> {
    metric_value(metrics, name).or_else(|| metric_value(metrics, &format!("heuristic_{name}")))
}

fn compute_chemistry_collaboration_metrics(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    layered: &LayeredGenerationMetrics,
) -> ChemistryCollaborationMetrics {
    if forwards.is_empty() {
        return ChemistryCollaborationMetrics::default();
    }

    let gate_usage_by_chemical_role = chemistry_role_gate_usage(forwards);
    let (pharmacophore_role_coverage, role_conflict_rate) =
        pharmacophore_collaboration_metrics(forwards);
    let rollout_count = forwards.len() as f64;
    let severe_clash_fraction = forwards
        .iter()
        .filter(|forward| rollout_has_severe_clash(&forward.generation.rollout))
        .count() as f64
        / rollout_count;
    let valence_violation_fraction = forwards
        .iter()
        .filter(|forward| rollout_has_valence_violation(&forward.generation.rollout))
        .count() as f64
        / rollout_count;
    let chemistry_guardrails =
        crate::losses::ChemistryGuardrailAuxLoss::default().compute_batch(examples, forwards);
    let bond_length_guardrail_mean = chemistry_guardrails
        .bond_length_guardrail
        .double_value(&[]);
    let key_residue_contact_coverage = key_residue_contact_collaboration_metric(layered);

    ChemistryCollaborationMetrics {
        gate_usage_by_chemical_role,
        pharmacophore_role_coverage,
        role_conflict_rate,
        severe_clash_fraction: ChemistryCollaborationMetric::available(
            severe_clash_fraction,
            ChemistryMetricProvenance::Heuristic,
            "rollout guardrail diagnostic",
        ),
        valence_violation_fraction: ChemistryCollaborationMetric::available(
            valence_violation_fraction,
            ChemistryMetricProvenance::Heuristic,
            "rollout guardrail diagnostic",
        ),
        bond_length_guardrail_mean: ChemistryCollaborationMetric::available(
            bond_length_guardrail_mean,
            ChemistryMetricProvenance::Heuristic,
            "topology-implied bond-length guardrail",
        ),
        key_residue_contact_coverage,
    }
}

fn chemistry_role_gate_usage(forwards: &[ResearchForward]) -> Vec<ChemistryRoleGateUsage> {
    let mut by_role: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for forward in forwards {
        for path_index in 0..6 {
            let path = interaction_path_diagnostic_at(forward, path_index);
            let entry = by_role
                .entry(path.path_role.to_string())
                .or_insert((0.0, 0));
            entry.0 += path.gate_mean;
            entry.1 += 1;
        }
    }

    by_role
        .into_iter()
        .map(|(chemical_role, (sum, count))| ChemistryRoleGateUsage {
            chemical_role,
            gate_mean: ChemistryCollaborationMetric::available(
                sum / count.max(1) as f64,
                ChemistryMetricProvenance::Heuristic,
                "directed cross-modal gate diagnostic",
            ),
            path_count: count,
        })
        .collect()
}

fn pharmacophore_collaboration_metrics(
    forwards: &[ResearchForward],
) -> (ChemistryCollaborationMetric, ChemistryCollaborationMetric) {
    let mut coverage_sum = 0.0;
    let mut conflict_sum = 0.0;
    let mut count = 0_usize;
    let mut provenance_labels = Vec::new();

    for forward in forwards {
        for path_index in 0..6 {
            let path = interaction_path_diagnostic_at(forward, path_index);
            let Some(coverage) = path.pharmacophore_role_coverage else {
                continue;
            };
            let Some(conflict_rate) = path.pharmacophore_role_conflict_rate else {
                continue;
            };
            if !coverage.is_finite() || !conflict_rate.is_finite() {
                continue;
            }
            if path.pharmacophore_role_provenance.as_deref() == Some("unavailable") {
                continue;
            }
            coverage_sum += coverage;
            conflict_sum += conflict_rate;
            count += 1;
            if let Some(provenance) = &path.pharmacophore_role_provenance {
                provenance_labels.push(provenance.as_str());
            }
        }
    }

    if count == 0 {
        return (
            ChemistryCollaborationMetric::unavailable("no pharmacophore role diagnostics"),
            ChemistryCollaborationMetric::unavailable("no pharmacophore role diagnostics"),
        );
    }

    let provenance = chemistry_provenance_from_labels(&provenance_labels);
    (
        ChemistryCollaborationMetric::available(
            coverage_sum / count as f64,
            provenance,
            "topology-pocket pharmacophore diagnostic",
        ),
        ChemistryCollaborationMetric::available(
            conflict_sum / count as f64,
            provenance,
            "topology-pocket pharmacophore diagnostic",
        ),
    )
}

fn chemistry_provenance_from_labels(labels: &[&str]) -> ChemistryMetricProvenance {
    if labels
        .iter()
        .any(|label| label.eq_ignore_ascii_case("experimental"))
    {
        ChemistryMetricProvenance::Experimental
    } else if labels
        .iter()
        .any(|label| label.to_ascii_lowercase().contains("docking"))
    {
        ChemistryMetricProvenance::DockingSupported
    } else if labels.iter().any(|label| {
        let lower = label.to_ascii_lowercase();
        lower.contains("backend") || lower.contains("external")
    }) {
        ChemistryMetricProvenance::BackendSupported
    } else if labels
        .iter()
        .any(|label| label.eq_ignore_ascii_case("heuristic"))
    {
        ChemistryMetricProvenance::Heuristic
    } else {
        ChemistryMetricProvenance::Unavailable
    }
}

fn key_residue_contact_collaboration_metric(
    layered: &LayeredGenerationMetrics,
) -> ChemistryCollaborationMetric {
    let layer = preferred_chemistry_collaboration_layer(layered);
    if layer.candidate_count == 0 || layer.residue_identity_coverage_fraction <= 0.0 {
        ChemistryCollaborationMetric::unavailable("residue identities unavailable")
    } else {
        ChemistryCollaborationMetric::available(
            layer.key_residue_contact_coverage,
            ChemistryMetricProvenance::Heuristic,
            "residue-aware interaction-profile diagnostic",
        )
    }
}

fn preferred_chemistry_collaboration_layer(
    layered: &LayeredGenerationMetrics,
) -> &CandidateLayerMetrics {
    if layered.reranked_candidates.candidate_count > 0 {
        &layered.reranked_candidates
    } else if layered.inferred_bond_candidates.candidate_count > 0 {
        &layered.inferred_bond_candidates
    } else if layered.repaired_candidates.candidate_count > 0 {
        &layered.repaired_candidates
    } else {
        &layered.raw_rollout
    }
}

fn rollout_has_severe_clash(rollout: &crate::models::GenerationRolloutRecord) -> bool {
    rollout.severe_clash_flag || rollout.steps.iter().any(|step| step.severe_clash_flag)
}

fn rollout_has_valence_violation(rollout: &crate::models::GenerationRolloutRecord) -> bool {
    rollout.valence_guardrail_flag || rollout.steps.iter().any(|step| step.valence_guardrail_flag)
}

fn interaction_path_diagnostic_at(
    forward: &ResearchForward,
    index: usize,
) -> &crate::models::interaction::CrossModalInteractionPathDiagnostics {
    match index {
        0 => &forward.interaction_diagnostics.topo_from_geo,
        1 => &forward.interaction_diagnostics.topo_from_pocket,
        2 => &forward.interaction_diagnostics.geo_from_topo,
        3 => &forward.interaction_diagnostics.geo_from_pocket,
        4 => &forward.interaction_diagnostics.pocket_from_topo,
        _ => &forward.interaction_diagnostics.pocket_from_geo,
    }
}

fn split_histograms(
    examples: &[crate::data::MolecularExample],
) -> (
    BTreeMap<String, usize>,
    BTreeMap<String, usize>,
    BTreeMap<String, usize>,
) {
    let mut ligand_bins = BTreeMap::new();
    let mut pocket_bins = BTreeMap::new();
    let mut measurements = BTreeMap::new();
    for example in examples {
        *ligand_bins
            .entry(atom_count_bin(ligand_atom_count(example)))
            .or_default() += 1;
        *pocket_bins
            .entry(atom_count_bin(pocket_atom_count(example)))
            .or_default() += 1;
        *measurements.entry(measurement_family(example)).or_default() += 1;
    }
    (ligand_bins, pocket_bins, measurements)
}

fn build_stratum_metrics(
    examples: &[crate::data::MolecularExample],
    train_proteins: &std::collections::BTreeSet<&str>,
) -> Vec<StratumEvaluationMetrics> {
    let mut strata = Vec::new();
    let mut axes: BTreeMap<(String, String), Vec<&crate::data::MolecularExample>> = BTreeMap::new();
    for example in examples {
        axes.entry((
            "ligand_atoms".to_string(),
            atom_count_bin(ligand_atom_count(example)),
        ))
        .or_default()
        .push(example);
        axes.entry((
            "pocket_atoms".to_string(),
            atom_count_bin(pocket_atom_count(example)),
        ))
        .or_default()
        .push(example);
        axes.entry(("measurement".to_string(), measurement_family(example)))
            .or_default()
            .push(example);
    }
    for ((axis, bin), bucket) in axes {
        let example_count = bucket.len();
        let labeled = bucket
            .iter()
            .filter(|example| example.targets.affinity_kcal_mol.is_some())
            .count();
        let unseen = bucket
            .iter()
            .filter(|example| !train_proteins.contains(example.protein_id.as_str()))
            .count();
        let ligand_atoms = bucket
            .iter()
            .map(|example| ligand_atom_count(example))
            .sum::<usize>();
        let pocket_atoms = bucket
            .iter()
            .map(|example| pocket_atom_count(example))
            .sum::<usize>();
        strata.push(StratumEvaluationMetrics {
            axis,
            bin,
            example_count,
            unseen_protein_fraction: fraction(unseen, example_count),
            labeled_fraction: fraction(labeled, example_count),
            average_ligand_atoms: fraction(ligand_atoms, example_count),
            average_pocket_atoms: fraction(pocket_atoms, example_count),
        });
    }
    strata
}

fn ligand_atom_count(example: &crate::data::MolecularExample) -> usize {
    example
        .topology
        .atom_types
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize
}

fn pocket_atom_count(example: &crate::data::MolecularExample) -> usize {
    example
        .pocket
        .coords
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize
}

fn average_ligand_atoms(examples: &[crate::data::MolecularExample]) -> f64 {
    fraction(
        examples.iter().map(ligand_atom_count).sum::<usize>(),
        examples.len(),
    )
}

fn average_pocket_atoms(examples: &[crate::data::MolecularExample]) -> f64 {
    fraction(
        examples.iter().map(pocket_atom_count).sum::<usize>(),
        examples.len(),
    )
}

fn measurement_family(example: &crate::data::MolecularExample) -> String {
    example
        .targets
        .affinity_measurement_type
        .as_deref()
        .unwrap_or("unknown")
        .to_string()
}

fn atom_count_bin(count: usize) -> String {
    match count {
        0 => "0".to_string(),
        1..=8 => "1-8".to_string(),
        9..=16 => "9-16".to_string(),
        17..=32 => "17-32".to_string(),
        33..=64 => "33-64".to_string(),
        65..=128 => "65-128".to_string(),
        129..=256 => "129-256".to_string(),
        _ => ">256".to_string(),
    }
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn fraction_f64(numerator: f64, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}

fn primary_objective_label(config: crate::config::PrimaryObjectiveConfig) -> String {
    match config {
        crate::config::PrimaryObjectiveConfig::SurrogateReconstruction => {
            "surrogate_reconstruction".to_string()
        }
        crate::config::PrimaryObjectiveConfig::ConditionedDenoising => {
            "conditioned_denoising".to_string()
        }
        crate::config::PrimaryObjectiveConfig::FlowMatching => "flow_matching".to_string(),
        crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching => {
            "denoising_flow_matching".to_string()
        }
    }
}

fn primary_objective_provenance_label(config: crate::config::PrimaryObjectiveConfig) -> String {
    match config {
        crate::config::PrimaryObjectiveConfig::SurrogateReconstruction => {
            "bootstrap_debug_shape_safe_surrogate".to_string()
        }
        crate::config::PrimaryObjectiveConfig::ConditionedDenoising => {
            "decoder_anchored_conditioned_denoising".to_string()
        }
        crate::config::PrimaryObjectiveConfig::FlowMatching => {
            "tensor_preserving_flow_matching_velocity_endpoint".to_string()
        }
        crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching => {
            "composite_training_objective_conditioned_denoising_plus_flow_matching".to_string()
        }
    }
}

fn primary_objective_claim_boundary_label(config: crate::config::PrimaryObjectiveConfig) -> String {
    match config {
        crate::config::PrimaryObjectiveConfig::SurrogateReconstruction => {
            "bootstrap_debug_or_shape_safe_baseline_not_generation_quality".to_string()
        }
        crate::config::PrimaryObjectiveConfig::ConditionedDenoising => {
            "target_ligand_denoising_or_refinement_training_signal".to_string()
        }
        crate::config::PrimaryObjectiveConfig::FlowMatching => {
            "flow_refinement_training_signal_requires_flow_backend_records".to_string()
        }
        crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching => {
            "hybrid_training_composition_not_a_separate_generation_mode".to_string()
        }
    }
}

fn generation_mode_label(research: &ResearchConfig) -> String {
    let family = if research.generation_method.primary_backend_id() == "flow_matching" {
        crate::config::GenerationBackendFamilyConfig::FlowMatching
    } else {
        research.generation_method.primary_backend.family
    };
    research
        .data
        .generation_target
        .generation_mode
        .resolved_for_backend(family)
        .as_str()
        .to_string()
}

fn interaction_mode_label(mode: CrossAttentionMode) -> String {
    match mode {
        CrossAttentionMode::Lightweight => "lightweight".to_string(),
        CrossAttentionMode::Transformer => "transformer".to_string(),
        CrossAttentionMode::DirectFusionNegativeControl => {
            "direct_fusion_negative_control".to_string()
        }
    }
}
