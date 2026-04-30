fn synchronize_method_comparison_evidence(
    layered: &mut LayeredGenerationMetrics,
    method_comparison: &mut MethodComparisonSummary,
    model: &ModelDesignEvaluationMetrics,
) {
    let raw_native_evidence = build_evaluation_raw_native_evidence(layered, model);
    let processed_generation_evidence =
        build_evaluation_processed_generation_evidence(layered, model);

    method_comparison.raw_native_evidence = raw_native_evidence;
    method_comparison.processed_generation_evidence = processed_generation_evidence;
    layered.method_comparison = method_comparison.clone();
}

fn build_evaluation_raw_native_evidence(
    layered: &LayeredGenerationMetrics,
    model: &ModelDesignEvaluationMetrics,
) -> ClaimRawNativeEvidenceSummary {
    let raw = evaluation_layer_metrics_by_name(layered, &model.raw_model_layer)
        .unwrap_or(&layered.raw_rollout);
    let raw_contract = generation_path_contract_for_layer(layered, &model.raw_model_layer);
    let model_native_raw = raw_contract
        .as_ref()
        .map(|row| row.model_native_raw && row.generation_path_class == "model_native_raw")
        .unwrap_or_else(|| layer_name_is_model_native_raw(&model.raw_model_layer));
    let claim_boundary = raw_contract
        .as_ref()
        .map(|row| row.claim_boundary.clone())
        .unwrap_or_else(|| model.raw_vs_processed_note.clone());
    let branch = &layered.flow_head_ablation;

    ClaimRawNativeEvidenceSummary {
        schema_version: default_claim_evidence_schema_version(),
        evidence_role: "model_native_raw_first".to_string(),
        raw_model_layer: model.raw_model_layer.clone(),
        model_native_raw,
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
        strict_pocket_fit_score: Some(evaluation_layer_strict_pocket_fit_score(raw)),
        enabled_flow_branches: branch.enabled_flow_branches.clone(),
        disabled_flow_branches: branch.disabled_flow_branches.clone(),
        full_molecular_flow_claim_allowed: branch.full_molecular_flow_claim_allowed,
        branch_claim_gate_reason: branch.claim_gate_reason.clone(),
        target_matching_claim_safe: branch.target_matching_claim_safe,
        slot_activation_mean: Some(model.slot_activation_mean),
        gate_activation_mean: Some(model.gate_activation_mean),
        leakage_proxy_mean: Some(model.leakage_proxy_mean),
        claim_boundary,
        raw_native_gate: build_evaluation_raw_native_gate(raw, model_native_raw),
    }
}

fn build_evaluation_processed_generation_evidence(
    layered: &LayeredGenerationMetrics,
    model: &ModelDesignEvaluationMetrics,
) -> ClaimProcessedGenerationEvidenceSummary {
    let processed = evaluation_layer_metrics_by_name(layered, &model.processed_layer)
        .unwrap_or(&layered.raw_rollout);

    ClaimProcessedGenerationEvidenceSummary {
        schema_version: default_claim_evidence_schema_version(),
        evidence_role: "additive_processed_or_reranked_evidence".to_string(),
        processed_layer: model.processed_layer.clone(),
        model_native_raw: layer_name_is_model_native_raw(&model.processed_layer),
        candidate_count: processed.candidate_count,
        valid_fraction: Some(model.processed_valid_fraction),
        pocket_contact_fraction: Some(model.processed_pocket_contact_fraction),
        strict_pocket_fit_score: Some(evaluation_layer_strict_pocket_fit_score(processed)),
        mean_centroid_offset: Some(processed.mean_centroid_offset),
        clash_fraction: Some(model.processed_clash_fraction),
        processing_valid_fraction_delta: Some(model.processing_valid_fraction_delta),
        processing_pocket_contact_delta: Some(model.processing_pocket_contact_delta),
        processing_clash_delta: Some(model.processing_clash_delta),
        postprocessor_chain: model.processed_postprocessor_chain.clone(),
        claim_boundary: model.processed_claim_boundary.clone(),
        additive_interpretation:
            "processed/repaired/reranked metrics are additive evidence and do not override raw-native gate outcomes"
                .to_string(),
    }
}

fn build_evaluation_raw_native_gate(
    raw: &CandidateLayerMetrics,
    model_native_raw: bool,
) -> RawNativeClaimGateReport {
    let mut checks = BTreeMap::new();
    checks.insert(
        "raw_candidate_count".to_string(),
        BackendThresholdCheck {
            value: Some(raw.candidate_count as f64),
            threshold: 1.0,
            passed: raw.candidate_count >= 1,
            direction: "min".to_string(),
        },
    );
    checks.insert(
        "raw_layer_model_native".to_string(),
        BackendThresholdCheck {
            value: Some(if model_native_raw { 1.0 } else { 0.0 }),
            threshold: 1.0,
            passed: model_native_raw,
            direction: "min".to_string(),
        },
    );

    let failed_reasons = checks
        .iter()
        .filter_map(|(metric, check)| {
            (!check.passed).then(|| {
                format!(
                    "raw-native evidence audit failed for {metric}: observed {:?} required {} {:.4}",
                    check.value, check.direction, check.threshold
                )
            })
        })
        .collect::<Vec<_>>();
    let passed = failed_reasons.is_empty();
    let decision = if passed {
        "pass: raw-native evidence block is populated from a model-native layer; configured metric thresholds remain in the top-level performance gates"
            .to_string()
    } else {
        "fail: raw-native evidence block is unavailable or not model-native before processed metrics are considered"
            .to_string()
    };

    RawNativeClaimGateReport {
        gate_name: "raw_native_evidence_audit_gate".to_string(),
        passed,
        checks,
        failed_reasons,
        processed_metrics_excluded: true,
        decision,
    }
}

fn evaluation_layer_metrics_by_name<'a>(
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

fn evaluation_layer_strict_pocket_fit_score(layer: &CandidateLayerMetrics) -> f64 {
    if layer.candidate_count == 0 {
        return 0.0;
    }
    let centroid_fit = 1.0 / (1.0 + layer.mean_centroid_offset.max(0.0));
    (layer.valid_fraction
        * layer.pocket_contact_fraction
        * centroid_fit
        * (1.0 - layer.clash_fraction.clamp(0.0, 1.0)))
    .clamp(0.0, 1.0)
}
