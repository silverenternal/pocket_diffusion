fn build_backend_environment_report(
    summary: &UnseenPocketExperimentSummary,
) -> BackendEnvironmentReport {
    let config = &summary.config.external_evaluation;
    let real_backend_backed = claim_is_real_backend_backed(summary);
    let fingerprint_source = (
        &config.chemistry_backend,
        &config.docking_backend,
        &config.pocket_backend,
    );
    BackendEnvironmentReport {
        config_fingerprint: stable_json_hash(&fingerprint_source),
        real_backend_backed,
        prerequisites: vec![
            "Use the canonical experiment config that enables the external chemistry, docking, and pocket backends.".to_string(),
            "Run `python3 tools/reviewer_env_check.py --config <experiment-config>` before reviewer revalidation on a fresh machine.".to_string(),
            "Ensure `python3` can execute `tools/rdkit_validity_backend.py` and `tools/pocket_contact_backend.py` from the repository root.".to_string(),
            "Keep source protein and ligand structure provenance available in the configured dataset so backend scoring does not degrade into missing-structure examples.".to_string(),
        ],
        chemistry_backend: build_backend_command_report(
            "chemistry_validity",
            &config.chemistry_backend,
            &summary.test.real_generation_metrics.chemistry_validity,
        ),
        docking_backend: build_backend_command_report(
            "docking_affinity",
            &config.docking_backend,
            &summary.test.real_generation_metrics.docking_affinity,
        ),
        pocket_backend: build_backend_command_report(
            "pocket_compatibility",
            &config.pocket_backend,
            &summary.test.real_generation_metrics.pocket_compatibility,
        ),
    }
}

fn build_backend_command_report(
    logical_name: &str,
    config: &ExternalBackendCommandConfig,
    metrics: &ReservedBackendMetrics,
) -> BackendCommandReport {
    let command_identity = (&config.executable, &config.args);
    let backend_examples_scored = metrics.metrics.get("backend_examples_scored").copied();
    let schema_version = metrics.metrics.get("schema_version").copied();
    let spawn_failed = metrics
        .metrics
        .get("backend_command_spawn_error")
        .copied()
        .unwrap_or(0.0)
        > 0.0;
    let command_failed = metrics
        .metrics
        .get("backend_command_failed")
        .copied()
        .unwrap_or(0.0)
        > 0.0;
    let runtime_available = backend_config_enabled(config)
        && !spawn_failed
        && !command_failed
        && backend_examples_scored.unwrap_or(0.0) > 0.0;
    BackendCommandReport {
        logical_name: logical_name.to_string(),
        enabled: config.enabled,
        executable: config.executable.clone(),
        args: config.args.clone(),
        command_fingerprint: stable_json_hash(&command_identity),
        runtime_available,
        backend_name: metrics.backend_name.clone(),
        status: metrics.status.clone(),
        schema_version,
        backend_examples_scored,
    }
}

fn build_baseline_comparisons(
    summary: &UnseenPocketExperimentSummary,
) -> Vec<BaselineComparisonRow> {
    let layers = &summary.test.layered_generation_metrics;
    let mut rows = vec![
        BaselineComparisonRow {
            label: "heuristic_raw_rollout_no_repair".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.raw_rollout.clone()),
            test_summary: None,
            interpretation:
                "Raw decoder rollout before geometry repair, bond inference, or reranking."
                    .to_string(),
        },
        BaselineComparisonRow {
            label: "pocket_centroid_repair_proxy".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.repaired_candidates.clone()),
            test_summary: None,
            interpretation:
                "Geometry-repaired candidates expose how much pocket-centroid postprocessing helps."
                    .to_string(),
        },
        BaselineComparisonRow {
            label: "deterministic_proxy_reranker".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.deterministic_proxy_candidates.clone()),
            test_summary: None,
            interpretation:
                "Current deterministic selection proxy before learned reranker calibration."
                    .to_string(),
        },
        BaselineComparisonRow {
            label: "calibrated_reranker".to_string(),
            source: "generation_layer".to_string(),
            candidate_layer: Some(layers.reranked_candidates.clone()),
            test_summary: None,
            interpretation:
                "Active bounded calibrated reranker used on the claim-bearing selection path."
                    .to_string(),
        },
    ];

    if let Some(matrix) = &summary.ablation_matrix {
        for wanted in [
            "objective_surrogate",
            "disable_slots",
            "disable_rollout_pocket_guidance",
            "disable_cross_attention",
        ] {
            if let Some(variant) = matrix
                .variants
                .iter()
                .find(|variant| variant.variant_label == wanted)
            {
                rows.push(BaselineComparisonRow {
                    label: wanted.to_string(),
                    source: "ablation_matrix".to_string(),
                    candidate_layer: None,
                    test_summary: Some(variant.test.clone()),
                    interpretation: match wanted {
                        "objective_surrogate" => {
                            "Surrogate reconstruction objective control.".to_string()
                        }
                        "disable_slots" => "No-slot controlled-interaction control.".to_string(),
                        "disable_rollout_pocket_guidance" => {
                            "No rollout-time pocket-guidance control.".to_string()
                        }
                        "disable_cross_attention" => {
                            "No cross-modal interaction control, not a replacement architecture."
                                .to_string()
                        }
                        _ => "Ablation control.".to_string(),
                    },
                });
            }
        }
    }
    rows
}

fn subtract_optional(candidate: Option<f64>, baseline: Option<f64>) -> Option<f64> {
    candidate
        .zip(baseline)
        .map(|(candidate, baseline)| candidate - baseline)
}

fn build_reranker_report(metrics: &LayeredGenerationMetrics) -> RerankerReport {
    let baseline = metrics.inferred_bond_candidates.clone();
    let reranked = metrics.reranked_candidates.clone();
    let deterministic_proxy = metrics.deterministic_proxy_candidates.clone();
    let calibration = metrics.reranker_calibration.clone();
    let validity_delta = reranked.valid_fraction - baseline.valid_fraction;
    let pocket_delta = reranked.pocket_contact_fraction - baseline.pocket_contact_fraction;
    let clash_delta = baseline.clash_fraction - reranked.clash_fraction;
    let decision = if validity_delta >= -1e-6 && pocket_delta >= -1e-6 && clash_delta >= -1e-6 {
        "bounded calibrated reranking is sufficient to keep adversarial training out of the mainline for this surface; confirm coefficients on larger backend-scored held-out pockets".to_string()
    } else {
        "calibrated reranking did not dominate deterministic selection; expand backend-scored calibration evidence before considering adversarial training".to_string()
    };
    RerankerReport {
        baseline,
        deterministic_proxy,
        reranked,
        calibration,
        decision,
    }
}

