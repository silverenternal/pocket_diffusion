fn build_claim_report(summary: &UnseenPocketExperimentSummary) -> ClaimReport {
    let ablation_deltas = summary
        .ablation_matrix
        .as_ref()
        .map(|matrix| {
            matrix
                .variants
                .iter()
                .map(|variant| ClaimDeltaSummary {
                    variant_label: variant.variant_label.clone(),
                    candidate_valid_fraction_delta: subtract_optional(
                        variant.test.candidate_valid_fraction,
                        summary.test.comparison_summary.candidate_valid_fraction,
                    ),
                    pocket_contact_fraction_delta: subtract_optional(
                        variant.test.pocket_contact_fraction,
                        summary.test.comparison_summary.pocket_contact_fraction,
                    ),
                    pocket_compatibility_fraction_delta: subtract_optional(
                        variant.test.pocket_compatibility_fraction,
                        summary
                            .test
                            .comparison_summary
                            .pocket_compatibility_fraction,
                    ),
                    mean_centroid_offset_delta: subtract_optional(
                        variant.test.mean_centroid_offset,
                        summary.test.comparison_summary.mean_centroid_offset,
                    ),
                    strict_pocket_fit_score_delta: subtract_optional(
                        variant.test.strict_pocket_fit_score,
                        summary.test.comparison_summary.strict_pocket_fit_score,
                    ),
                    unique_smiles_fraction_delta: subtract_optional(
                        variant.test.unique_smiles_fraction,
                        summary.test.comparison_summary.unique_smiles_fraction,
                    ),
                    topology_specialization_score_delta: variant.test.topology_specialization_score
                        - summary
                            .test
                            .comparison_summary
                            .topology_specialization_score,
                    geometry_specialization_score_delta: variant.test.geometry_specialization_score
                        - summary
                            .test
                            .comparison_summary
                            .geometry_specialization_score,
                    pocket_specialization_score_delta: variant.test.pocket_specialization_score
                        - summary.test.comparison_summary.pocket_specialization_score,
                    slot_activation_mean_delta: variant.test.slot_activation_mean
                        - summary.test.comparison_summary.slot_activation_mean,
                    gate_activation_mean_delta: variant.test.gate_activation_mean
                        - summary.test.comparison_summary.gate_activation_mean,
                    leakage_proxy_mean_delta: variant.test.leakage_proxy_mean
                        - summary.test.comparison_summary.leakage_proxy_mean,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let layer_provenance_audit = build_layer_provenance_audit(summary);
    let raw_native_evidence = build_raw_native_evidence(summary, &layer_provenance_audit);
    let processed_generation_evidence =
        build_processed_generation_evidence(summary, &layer_provenance_audit);
    let mut method_comparison = summary.test.method_comparison.clone();
    method_comparison.raw_native_evidence = raw_native_evidence.clone();
    method_comparison.processed_generation_evidence = processed_generation_evidence.clone();

    ClaimReport {
        artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
        run_label: summary
            .config
            .ablation
            .variant_label
            .clone()
            .unwrap_or_else(|| "base_run".to_string()),
        raw_native_evidence,
        processed_generation_evidence,
        postprocessing_repair_audit: summary.test.layered_generation_metrics.repair_case_audit.clone(),
        validation: summary.validation.comparison_summary.clone(),
        test: summary.test.comparison_summary.clone(),
        backend_metrics: summary.test.real_generation_metrics.clone(),
        backend_thresholds: build_backend_thresholds(summary),
        backend_review: build_backend_review(summary),
        layered_generation_metrics: summary.test.layered_generation_metrics.clone(),
        model_design: summary.test.model_design.clone(),
        layer_provenance_audit,
        chemistry_novelty_diversity: build_chemistry_novelty_diversity(summary),
        chemistry_collaboration: summary
            .test
            .comparison_summary
            .chemistry_collaboration
            .clone(),
        claim_context: build_claim_context(summary),
        backend_environment: Some(build_backend_environment_report(summary)),
        reranker_report: build_reranker_report(summary),
        slot_stability: summary.test.slot_stability.clone(),
        leakage_calibration: build_leakage_calibration_report(summary, &ablation_deltas),
        performance_gates: summary.performance_gates.clone(),
        baseline_comparisons: build_baseline_comparisons(summary),
        method_comparison,
        train_eval_alignment: summary.test.train_eval_alignment.clone(),
        ablation_deltas,
    }
}

fn build_raw_native_evidence(
    summary: &UnseenPocketExperimentSummary,
    audit: &ClaimLayerProvenanceAudit,
) -> ClaimRawNativeEvidenceSummary {
    let model = &summary.test.model_design;
    let raw = layer_metrics_by_name(&summary.test.layered_generation_metrics, &model.raw_model_layer)
        .unwrap_or(&summary.test.layered_generation_metrics.raw_rollout);
    let branch = &summary
        .test
        .layered_generation_metrics
        .flow_head_ablation;
    let claim_boundary = find_generation_path_contract(
        &summary
            .test
            .layered_generation_metrics
            .generation_path_contract,
        &model.raw_model_layer,
    )
    .map(|row| row.claim_boundary.clone())
    .unwrap_or_else(|| model.raw_vs_processed_note.clone());

    ClaimRawNativeEvidenceSummary {
        schema_version: default_claim_evidence_schema_version(),
        evidence_role: "model_native_raw_first".to_string(),
        raw_model_layer: model.raw_model_layer.clone(),
        model_native_raw: audit.raw_layer_model_native,
        candidate_count: raw.candidate_count,
        valid_fraction: Some(model.raw_model_valid_fraction),
        native_graph_valid_fraction: Some(model.raw_native_graph_valid_fraction),
        native_bond_count_mean: Some(model.raw_native_bond_count_mean),
        native_component_count_mean: Some(model.raw_native_component_count_mean),
        native_valence_violation_fraction: Some(model.raw_native_valence_violation_fraction),
        topology_bond_sync_fraction: Some(model.raw_native_topology_bond_sync_fraction),
        mean_centroid_offset: Some(raw.mean_centroid_offset),
        mean_displacement: Some(model.raw_model_mean_displacement),
        clash_fraction: Some(model.raw_model_clash_fraction),
        pocket_contact_fraction: Some(model.raw_model_pocket_contact_fraction),
        strict_pocket_fit_score: Some(layer_native_quality(raw)),
        enabled_flow_branches: branch.enabled_flow_branches.clone(),
        disabled_flow_branches: branch.disabled_flow_branches.clone(),
        full_molecular_flow_claim_allowed: branch.full_molecular_flow_claim_allowed,
        branch_claim_gate_reason: branch.claim_gate_reason.clone(),
        target_matching_claim_safe: branch.target_matching_claim_safe,
        slot_activation_mean: Some(model.slot_activation_mean),
        gate_activation_mean: Some(model.gate_activation_mean),
        leakage_proxy_mean: Some(model.leakage_proxy_mean),
        claim_boundary,
        raw_native_gate: build_raw_native_claim_gate(summary),
    }
}

fn build_processed_generation_evidence(
    summary: &UnseenPocketExperimentSummary,
    audit: &ClaimLayerProvenanceAudit,
) -> ClaimProcessedGenerationEvidenceSummary {
    let model = &summary.test.model_design;
    let processed = layer_metrics_by_name(
        &summary.test.layered_generation_metrics,
        &model.processed_layer,
    )
    .unwrap_or(&summary.test.layered_generation_metrics.raw_rollout);
    ClaimProcessedGenerationEvidenceSummary {
        schema_version: default_claim_evidence_schema_version(),
        evidence_role: "additive_processed_or_reranked_evidence".to_string(),
        processed_layer: model.processed_layer.clone(),
        model_native_raw: layer_name_is_model_native_raw(&model.processed_layer),
        candidate_count: processed.candidate_count,
        valid_fraction: Some(model.processed_valid_fraction),
        pocket_contact_fraction: Some(model.processed_pocket_contact_fraction),
        strict_pocket_fit_score: Some(layer_native_quality(processed)),
        mean_centroid_offset: Some(processed.mean_centroid_offset),
        clash_fraction: Some(model.processed_clash_fraction),
        processing_valid_fraction_delta: Some(model.processing_valid_fraction_delta),
        processing_pocket_contact_delta: Some(model.processing_pocket_contact_delta),
        processing_clash_delta: Some(model.processing_clash_delta),
        postprocessor_chain: audit.processed_postprocessor_chain.clone(),
        claim_boundary: model.processed_claim_boundary.clone(),
        additive_interpretation:
            "processed/repaired/reranked metrics are additive evidence and do not override raw-native gate outcomes"
                .to_string(),
    }
}

fn build_raw_native_claim_gate(summary: &UnseenPocketExperimentSummary) -> RawNativeClaimGateReport {
    let config = &summary.config.performance_gates;
    let model = &summary.test.model_design;
    let mut checks = BTreeMap::new();
    insert_optional_gate_check(
        &mut checks,
        "raw_model_valid_fraction",
        Some(model.raw_model_valid_fraction),
        config.min_test_raw_model_valid_fraction,
        "min",
    );
    insert_optional_gate_check(
        &mut checks,
        "raw_model_pocket_contact_fraction",
        Some(model.raw_model_pocket_contact_fraction),
        config.min_test_raw_model_pocket_contact_fraction,
        "min",
    );
    insert_optional_gate_check(
        &mut checks,
        "raw_model_clash_fraction",
        Some(model.raw_model_clash_fraction),
        config.max_test_raw_model_clash_fraction,
        "max",
    );
    insert_optional_gate_check(
        &mut checks,
        "raw_native_graph_valid_fraction",
        Some(model.raw_native_graph_valid_fraction),
        config.min_test_raw_native_graph_valid_fraction,
        "min",
    );

    let failed_reasons = checks
        .iter()
        .filter_map(|(metric, check)| {
            (!check.passed).then(|| {
                format!(
                    "raw-native claim gate failed for {metric}: observed {:?} required {} {:.4}",
                    check.value, check.direction, check.threshold
                )
            })
        })
        .collect::<Vec<_>>();
    let passed = failed_reasons.is_empty();
    let decision = if checks.is_empty() {
        "not_configured: raw-native thresholds omitted; raw metrics still lead claim review"
            .to_string()
    } else if passed {
        "pass: raw-native metrics clear configured gates before processed metrics are considered"
            .to_string()
    } else {
        "fail: raw-native regression gate failed independent of processed/repaired/reranked metrics"
            .to_string()
    };

    RawNativeClaimGateReport {
        gate_name: "raw_native_claim_gate".to_string(),
        passed,
        checks,
        failed_reasons,
        processed_metrics_excluded: true,
        decision,
    }
}

fn insert_optional_gate_check(
    checks: &mut BTreeMap<String, BackendThresholdCheck>,
    metric: &str,
    value: Option<f64>,
    threshold: Option<f64>,
    direction: &str,
) {
    if let Some(threshold) = threshold {
        checks.insert(
            metric.to_string(),
            backend_threshold_check(value, threshold, direction),
        );
    }
}

fn layer_metrics_by_name<'a>(
    layered: &'a LayeredGenerationMetrics,
    layer: &str,
) -> Option<&'a CandidateLayerMetrics> {
    match layer {
        "raw_flow" => Some(&layered.raw_flow),
        "constrained_flow" => Some(&layered.constrained_flow),
        "repaired" => Some(&layered.repaired),
        "raw_rollout" => Some(&layered.raw_rollout),
        "repaired_candidates" => Some(&layered.repaired_candidates),
        "inferred_bond_candidates" => Some(&layered.inferred_bond_candidates),
        "reranked_candidates" => Some(&layered.reranked_candidates),
        "deterministic_proxy_candidates" => Some(&layered.deterministic_proxy_candidates),
        _ => None,
    }
}

fn build_layer_provenance_audit(summary: &UnseenPocketExperimentSummary) -> ClaimLayerProvenanceAudit {
    let raw_model_layer = summary.test.model_design.raw_model_layer.clone();
    let processed_layer = summary.test.model_design.processed_layer.clone();
    let raw_contract = find_generation_path_contract(
        &summary.test.layered_generation_metrics.generation_path_contract,
        &raw_model_layer,
    );
    let processed_contract = find_generation_path_contract(
        &summary.test.layered_generation_metrics.generation_path_contract,
        &processed_layer,
    );
    let processed_layer_unavailable =
        processed_layer.trim().is_empty() || processed_layer == "unavailable";
    let raw_layer_model_native = raw_contract
        .map(|row| row.model_native_raw && row.generation_path_class == "model_native_raw")
        .unwrap_or(false);
    let processed_layer_has_contract = processed_contract.is_some();
    let processed_metrics_cited_as_raw = processed_contract
        .map(|row| !row.model_native_raw && raw_model_layer == processed_layer)
        .unwrap_or_else(|| raw_model_layer == processed_layer && raw_model_layer != "raw_rollout");
    let processed_canonical_layer = processed_contract
        .map(|row| row.canonical_layer.clone())
        .unwrap_or_else(|| "unavailable".to_string());
    let processed_generation_path_class = processed_contract
        .map(|row| row.generation_path_class.clone())
        .unwrap_or_else(|| "unavailable".to_string());
    let processed_postprocessor_chain = processed_contract
        .map(|row| row.postprocessor_chain.clone())
        .unwrap_or_default();
    let processed_requires_provenance = processed_contract
        .map(|row| !row.model_native_raw)
        .unwrap_or(!processed_layer_unavailable && processed_layer != raw_model_layer);
    let processed_has_required_provenance = !processed_requires_provenance
        || processed_layer_unavailable
        || (processed_layer_has_contract
            && (!processed_postprocessor_chain.is_empty()
                || processed_generation_path_class != "model_native_raw"));
    let claim_safe =
        raw_layer_model_native && processed_has_required_provenance && !processed_metrics_cited_as_raw;
    let decision = if claim_safe {
        "pass: raw model-design fields use model-native raw layer and processed fields carry explicit layer provenance".to_string()
    } else {
        "fail: raw/processed model-design fields lack sufficient layer provenance for claim use"
            .to_string()
    };

    ClaimLayerProvenanceAudit {
        raw_model_layer,
        processed_layer,
        raw_layer_model_native,
        processed_layer_has_contract,
        processed_canonical_layer,
        processed_generation_path_class,
        processed_postprocessor_chain,
        processed_metrics_cited_as_raw,
        claim_safe,
        decision,
    }
}

fn find_generation_path_contract<'a>(
    rows: &'a [GenerationPathContractRow],
    layer: &str,
) -> Option<&'a GenerationPathContractRow> {
    rows.iter().find(|row| {
        row.legacy_field_name == layer
            || row.canonical_layer == layer
            || (layer == "raw_flow" && row.legacy_field_name == "raw_rollout")
    })
}

fn backend_threshold_check(
    value: Option<f64>,
    threshold: f64,
    direction: &str,
) -> BackendThresholdCheck {
    let passed = match (value, direction) {
        (Some(observed), "min") => observed >= threshold,
        (Some(observed), "max") => observed <= threshold,
        _ => false,
    };
    BackendThresholdCheck {
        value,
        threshold,
        passed,
        direction: direction.to_string(),
    }
}

fn build_backend_thresholds(
    summary: &UnseenPocketExperimentSummary,
) -> BTreeMap<String, BackendThresholdCheck> {
    let chemistry = &summary.test.real_generation_metrics.chemistry_validity;
    let docking = &summary.test.real_generation_metrics.docking_affinity;
    let pocket = &summary.test.real_generation_metrics.pocket_compatibility;
    let strict_fit = metric_value_with_heuristic_fallback(pocket, "strict_pocket_fit_score");
    let pocket_contact = metric_value_with_heuristic_fallback(docking, "pocket_contact_fraction")
        .or_else(|| metric_value_with_heuristic_fallback(docking, "contact_fraction"));
    BTreeMap::from([
        (
            "rdkit_available".to_string(),
            backend_threshold_check(backend_metric(chemistry, "rdkit_available"), 1.0, "min"),
        ),
        (
            "rdkit_sanitized_fraction".to_string(),
            backend_threshold_check(
                backend_metric(chemistry, "rdkit_sanitized_fraction"),
                0.95,
                "min",
            ),
        ),
        (
            "rdkit_unique_smiles_fraction".to_string(),
            backend_threshold_check(
                backend_metric(chemistry, "rdkit_unique_smiles_fraction").or_else(|| {
                    metric_value_with_heuristic_fallback(chemistry, "unique_smiles_fraction")
                }),
                0.5,
                "min",
            ),
        ),
        (
            "backend_missing_structure_fraction".to_string(),
            backend_threshold_check(
                backend_metric(pocket, "backend_missing_structure_fraction"),
                0.0,
                "max",
            ),
        ),
        (
            "clash_fraction".to_string(),
            backend_threshold_check(
                metric_value_with_heuristic_fallback(pocket, "clash_fraction"),
                0.1,
                "max",
            ),
        ),
        (
            "strict_pocket_fit_score".to_string(),
            backend_threshold_check(strict_fit, 0.35, "min"),
        ),
        (
            "pocket_contact_fraction".to_string(),
            backend_threshold_check(pocket_contact, 0.8, "min"),
        ),
    ])
}

fn is_vina_backend_surface(summary: &UnseenPocketExperimentSummary) -> bool {
    summary
        .config
        .external_evaluation
        .docking_backend
        .args
        .iter()
        .any(|arg| arg.contains("vina_score_backend.py"))
}

fn build_backend_review(summary: &UnseenPocketExperimentSummary) -> BackendReviewReport {
    let backend_thresholds = build_backend_thresholds(summary);
    let chemistry_ready = [
        "rdkit_available",
        "rdkit_sanitized_fraction",
        "rdkit_unique_smiles_fraction",
    ]
    .iter()
    .all(|name| {
        backend_thresholds
            .get(*name)
            .map(|result| result.passed)
            .unwrap_or(false)
    });
    let pocket_ready = [
        "backend_missing_structure_fraction",
        "clash_fraction",
        "strict_pocket_fit_score",
        "pocket_contact_fraction",
    ]
    .iter()
    .all(|name| {
        backend_thresholds
            .get(*name)
            .map(|result| result.passed)
            .unwrap_or(false)
    });
    if !is_vina_backend_surface(summary) {
        return BackendReviewReport {
            policy_label: "repository_real_backend_policy".to_string(),
            reviewer_status: if chemistry_ready && pocket_ready {
                "pass".to_string()
            } else {
                "fail".to_string()
            },
            reviewer_passed: chemistry_ready && pocket_ready,
            claim_bearing_surface: claim_is_real_backend_backed(summary),
            claim_bearing_ready: chemistry_ready && pocket_ready,
            claim_bearing_requirements: vec![
                "Keep chemistry validity, pocket compatibility, clash, and strict pocket-fit metrics above the shared backend thresholds.".to_string(),
                "Keep backend_missing_structure_fraction at or below 0.0 on claim-bearing reviewer surfaces.".to_string(),
            ],
            reviewer_reasons: Vec::new(),
            chemistry_validity_ready: chemistry_ready,
            docking_backend_available: true,
            docking_input_completeness_fraction: None,
            docking_score_coverage_fraction: None,
        };
    }

    let docking = &summary.test.real_generation_metrics.docking_affinity;
    let vina_available = backend_metric(docking, "vina_available").unwrap_or(0.0) >= 1.0;
    let docking_input_completeness_fraction =
        backend_metric(docking, "candidate_with_complete_vina_inputs_fraction");
    let docking_score_coverage_fraction = backend_metric(docking, "vina_score_success_fraction")
        .or_else(|| backend_metric(docking, "backend_examples_scored"));
    let docking_ready = vina_available
        && docking_input_completeness_fraction
            .map(|value| value >= 1.0)
            .unwrap_or(false)
        && docking_score_coverage_fraction
            .map(|value| value >= 1.0)
            .unwrap_or(false);

    let mut reviewer_reasons = Vec::new();
    if !chemistry_ready {
        reviewer_reasons.push(
            "chemistry validity backend does not yet clear the shared RDKit availability, sanitization, or uniqueness thresholds".to_string(),
        );
    }
    if !pocket_ready {
        reviewer_reasons.push(
            "pocket compatibility metrics do not yet clear the shared missing-structure, clash, strict-fit, or contact thresholds".to_string(),
        );
    }
    if !vina_available {
        reviewer_reasons.push(
            "AutoDock Vina is unavailable on this machine, so the stronger docking companion cannot become claim-bearing".to_string(),
        );
    }
    if docking_input_completeness_fraction
        .map(|value| value < 1.0)
        .unwrap_or(true)
    {
        reviewer_reasons.push(
            "not all candidates provide complete Vina-ready receptor/ligand PDBQT inputs"
                .to_string(),
        );
    }
    if docking_score_coverage_fraction
        .map(|value| value < 1.0)
        .unwrap_or(true)
    {
        reviewer_reasons.push(
            "the stronger docking backend did not score every reviewed candidate".to_string(),
        );
    }

    let claim_bearing_ready = chemistry_ready && pocket_ready && docking_ready;
    BackendReviewReport {
        policy_label: "vina_claim_bearing_companion_policy".to_string(),
        reviewer_status: if claim_bearing_ready {
            "pass".to_string()
        } else {
            "fail".to_string()
        },
        reviewer_passed: claim_bearing_ready,
        claim_bearing_surface: true,
        claim_bearing_ready,
        claim_bearing_requirements: vec![
            "Chemistry validity must clear the shared RDKit availability, sanitization, and uniqueness thresholds.".to_string(),
            "Pocket compatibility must clear the shared missing-structure, clash, strict pocket-fit, and pocket-contact thresholds.".to_string(),
            "AutoDock Vina must be available and every reviewed candidate must provide Vina-ready receptor and ligand PDBQT inputs.".to_string(),
            "The stronger docking backend must score every reviewed candidate before this surface is described as claim-bearing backend evidence.".to_string(),
        ],
        reviewer_reasons,
        chemistry_validity_ready: chemistry_ready,
        docking_backend_available: vina_available,
        docking_input_completeness_fraction,
        docking_score_coverage_fraction,
    }
}

fn build_chemistry_novelty_diversity(
    summary: &UnseenPocketExperimentSummary,
) -> ChemistryNoveltyDiversitySummary {
    let (review_layer, layer) = if summary
        .test
        .layered_generation_metrics
        .reranked_candidates
        .candidate_count
        > 0
    {
        (
            "reranked_candidates",
            &summary.test.layered_generation_metrics.reranked_candidates,
        )
    } else {
        (
            "inferred_bond_candidates",
            &summary
                .test
                .layered_generation_metrics
                .inferred_bond_candidates,
        )
    };
    ChemistryNoveltyDiversitySummary {
        review_layer: review_layer.to_string(),
        unique_smiles_fraction: summary.test.comparison_summary.unique_smiles_fraction,
        atom_type_sequence_diversity: layer.atom_type_sequence_diversity,
        bond_topology_diversity: layer.bond_topology_diversity,
        coordinate_shape_diversity: layer.coordinate_shape_diversity,
        novel_atom_type_sequence_fraction: layer.novel_atom_type_sequence_fraction,
        novel_bond_topology_fraction: layer.novel_bond_topology_fraction,
        novel_coordinate_shape_fraction: layer.novel_coordinate_shape_fraction,
        interpretation: format!(
            "Review chemistry novelty/diversity on `{review_layer}`; novelty is measured against training-reference structural signatures instead of relying only on within-layer uniqueness."
        ),
        benchmark_evidence: build_chemistry_benchmark_evidence(summary),
    }
}

fn build_claim_context(summary: &UnseenPocketExperimentSummary) -> ClaimContext {
    let real_backend_backed = claim_is_real_backend_backed(summary);
    let interaction_mode = interaction_mode_label(summary.config.research.model.interaction_mode);
    let flow_contract = crate::models::flow::current_multimodal_flow_contract(
        &summary.config.research.generation_method.flow_matching,
    );
    let direct_fusion_negative_control =
        summary.config.research.model.interaction_mode == CrossAttentionMode::DirectFusionNegativeControl;
    let evidence_mode = if real_backend_backed {
        "real-backend-backed held-out pocket evidence".to_string()
    } else {
        "heuristic-only held-out pocket evidence".to_string()
    };
    ClaimContext {
        surface_label: summary.config.surface_label.clone(),
        real_backend_backed,
        evidence_mode,
        generation_mode: generation_mode_label(&summary.config.research),
        de_novo_claim_allowed: summary
            .config
            .research
            .data
            .generation_target
            .generation_mode
            .permits_de_novo_claims(),
        target_alignment_policy: flow_contract.target_alignment_policy.clone(),
        target_matching_claim_safe: flow_contract.target_matching_claim_safe,
        full_molecular_flow_claim_allowed: flow_contract.full_molecular_flow_claim_allowed,
        full_molecular_flow_claim_boundary: if flow_contract.full_molecular_flow_claim_allowed {
            "full molecular flow claim allowed: branch contract is complete and target matching is non-index"
                .to_string()
        } else {
            format!(
                "full molecular flow claim rejected: {}",
                flow_contract.claim_gate_reason
            )
        },
        interaction_mode,
        direct_fusion_negative_control,
        preferred_architecture_claim_allowed: !direct_fusion_negative_control,
        interaction_claim_boundary: if direct_fusion_negative_control {
            "direct_fusion_negative_control is an ablation-only surface and cannot support preferred controlled-interaction architecture claims".to_string()
        } else {
            "controlled gated directed interaction is the preferred architecture surface".to_string()
        },
    }
}

fn claim_is_real_backend_backed(summary: &UnseenPocketExperimentSummary) -> bool {
    backend_config_enabled(&summary.config.external_evaluation.chemistry_backend)
        && backend_config_enabled(&summary.config.external_evaluation.pocket_backend)
}

fn backend_config_enabled(config: &ExternalBackendCommandConfig) -> bool {
    config.enabled
        && config
            .executable
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
}
