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
    ClaimReport {
        artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
        run_label: summary
            .config
            .ablation
            .variant_label
            .clone()
            .unwrap_or_else(|| "base_run".to_string()),
        validation: summary.validation.comparison_summary.clone(),
        test: summary.test.comparison_summary.clone(),
        backend_metrics: summary.test.real_generation_metrics.clone(),
        backend_thresholds: build_backend_thresholds(summary),
        backend_review: build_backend_review(summary),
        layered_generation_metrics: summary.test.layered_generation_metrics.clone(),
        chemistry_novelty_diversity: build_chemistry_novelty_diversity(summary),
        claim_context: build_claim_context(summary),
        backend_environment: Some(build_backend_environment_report(summary)),
        reranker_report: build_reranker_report(summary),
        slot_stability: summary.test.slot_stability.clone(),
        leakage_calibration: build_leakage_calibration_report(summary, &ablation_deltas),
        performance_gates: summary.performance_gates.clone(),
        baseline_comparisons: build_baseline_comparisons(summary),
        method_comparison: summary.test.method_comparison.clone(),
        ablation_deltas,
    }
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
    let evidence_mode = if real_backend_backed {
        "real-backend-backed held-out pocket evidence".to_string()
    } else {
        "heuristic-only held-out pocket evidence".to_string()
    };
    ClaimContext {
        surface_label: summary.config.surface_label.clone(),
        real_backend_backed,
        evidence_mode,
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
