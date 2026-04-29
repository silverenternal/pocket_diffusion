fn build_leakage_calibration_report(
    summary: &UnseenPocketExperimentSummary,
    deltas: &[ClaimDeltaSummary],
) -> LeakageCalibrationReport {
    let preferred = default_preferred_leakage_proxy_threshold();
    let hard = default_hard_leakage_proxy_threshold();
    let regression_limit = default_max_leakage_regression_threshold();
    let test_leakage = summary.test.comparison_summary.leakage_proxy_mean;
    let split_checks = &summary.split_report.leakage_checks;
    let max_regression = deltas
        .iter()
        .map(|delta| delta.leakage_proxy_mean_delta)
        .fold(0.0_f64, f64::max);
    let blockers = deltas
        .iter()
        .filter(|delta| {
            delta
                .strict_pocket_fit_score_delta
                .is_some_and(|value| value < -0.05)
                || delta
                    .candidate_valid_fraction_delta
                    .is_some_and(|value| value < -0.05)
        })
        .count();
    let mut reasons = Vec::new();
    if test_leakage > hard {
        reasons.push(format!(
            "test leakage proxy {:.4} exceeds the hard reviewer bound {:.4}",
            test_leakage, hard
        ));
    } else if test_leakage > preferred {
        reasons.push(format!(
            "test leakage proxy {:.4} exceeds the preferred reviewer bound {:.4}",
            test_leakage, preferred
        ));
    }
    if max_regression > regression_limit {
        reasons.push(format!(
            "worst reviewed leakage regression {:.4} exceeds the allowed delta {:.4}",
            max_regression, regression_limit
        ));
    }
    if split_checks.protein_overlap_detected || split_checks.duplicate_example_ids_detected {
        reasons.push(
            "split leakage audit detected cross-split overlap or duplicated example identifiers"
                .to_string(),
        );
    }
    let probe_capacity = build_probe_capacity_report(&summary.config.research.model.semantic_probes);
    let probe_baseline_comparisons = summary.test.proxy_task_metrics.probe_baselines.clone();
    let baselines_support_probe_evidence = probe_baseline_comparisons
        .iter()
        .filter(|row| row.target != "affinity_scalar")
        .all(|row| row.improves_over_trivial == Some(true));
    let no_leakage_claim_supported =
        !probe_capacity.linear_baseline_only && baselines_support_probe_evidence;
    if probe_capacity.linear_baseline_only {
        reasons.push(
            "semantic/leakage probes use only the linear baseline capacity; do not claim absence of leakage from this artifact"
                .to_string(),
        );
    }
    if !baselines_support_probe_evidence {
        reasons.push(
            "one or more probe targets do not beat or lack a trivial-target baseline comparison"
                .to_string(),
        );
    }

    let (reviewer_status, reviewer_passed) = if test_leakage > hard
        || split_checks.protein_overlap_detected
        || split_checks.duplicate_example_ids_detected
    {
        ("fail".to_string(), false)
    } else if test_leakage > preferred
        || max_regression > regression_limit
        || !no_leakage_claim_supported
    {
        ("caution".to_string(), true)
    } else {
        ("pass".to_string(), true)
    };
    let decision = if blockers == 0 && reviewer_status == "pass" {
        "current leakage weight passes reviewer bounds and preserves physically necessary cross-modality dependence on reviewed variants".to_string()
    } else if blockers == 0 && reviewer_status == "caution" {
        "current leakage weight remains usable, but reviewed ablations or the base run sit above the preferred reviewer band; keep leakage interpretation explicit in claim-facing notes".to_string()
    } else if reviewer_status == "caution" {
        format!(
            "{blockers} reviewed variant(s) regress pocket fit or chemistry; base leakage remains bounded, so keep claim wording cautious and cite the ablation deltas explicitly"
        )
    } else {
        format!(
            "{blockers} reviewed variant(s) regress pocket fit or chemistry; keep leakage at or below the current default until a sweep clears these blockers"
        )
    };
    let frozen_probe_calibration = summary.test.frozen_leakage_probe_calibration.clone();
    let capacity_sweep_artifact =
        frozen_probe_artifact_path(summary, &frozen_probe_calibration);
    let leakage_roles = leakage_calibration_role_report(
        summary.config.research.training.loss_weights.delta_leak,
        test_leakage,
        &frozen_probe_calibration,
        capacity_sweep_artifact.clone(),
    );

    LeakageCalibrationReport {
        recommended_delta_leak: summary.config.research.training.loss_weights.delta_leak,
        evaluated_variants: deltas.len(),
        preferred_max_leakage_proxy_mean: preferred,
        hard_max_leakage_proxy_mean: hard,
        max_leakage_proxy_regression: regression_limit,
        probe_capacity,
        probe_baseline_comparisons,
        frozen_probe_calibration,
        leakage_roles,
        capacity_sweep_artifact,
        no_leakage_claim_supported,
        reviewer_status,
        reviewer_passed,
        reviewer_reasons: reasons,
        decision,
    }
}

fn frozen_probe_artifact_path(
    summary: &UnseenPocketExperimentSummary,
    report: &FrozenLeakageProbeCalibrationReport,
) -> Option<String> {
    if report.calibration_status == "not_run" {
        None
    } else {
        Some(
            summary
                .config
                .research
                .training
                .checkpoint_dir
                .join("frozen_leakage_probe_audit.json")
                .display()
                .to_string(),
        )
    }
}

fn leakage_calibration_role_report(
    delta_leak: f64,
    test_leakage_proxy: f64,
    frozen_probe_calibration: &FrozenLeakageProbeCalibrationReport,
    capacity_sweep_artifact: Option<String>,
) -> crate::losses::LeakageEvidenceRoleReport {
    crate::losses::LeakageEvidenceRoleReport {
        optimizer_penalty: crate::losses::LeakageOptimizerPenaltySection {
            active: delta_leak > 0.0,
            effective_delta_leak: delta_leak,
            leak_execution_mode: "configured_training_weight".to_string(),
            interpretation:
                "evaluation artifact records the configured training leakage weight, not a new optimizer step"
                    .to_string(),
            ..crate::losses::LeakageOptimizerPenaltySection::default()
        },
        detached_training_diagnostic: crate::losses::LeakageDetachedTrainingDiagnosticSection {
            similarity_proxy_diagnostic: test_leakage_proxy,
            interpretation:
                "test leakage proxy is a detached evaluation diagnostic and not a held-out frozen-probe estimate"
                    .to_string(),
            ..crate::losses::LeakageDetachedTrainingDiagnosticSection::default()
        },
        frozen_probe_audit: frozen_probe_audit_section(
            frozen_probe_calibration,
            capacity_sweep_artifact,
        ),
        claim_boundary:
            "no-leakage wording requires frozen held-out probe evidence in addition to training-time penalties and proxy diagnostics"
                .to_string(),
        ..crate::losses::LeakageEvidenceRoleReport::default()
    }
}

fn frozen_probe_audit_section(
    report: &FrozenLeakageProbeCalibrationReport,
    artifact: Option<String>,
) -> crate::losses::LeakageFrozenProbeAuditSection {
    let best_improvement_over_baseline = report
        .capacity_sweep
        .iter()
        .map(|row| row.improvement_over_baseline)
        .filter(|value| value.is_finite())
        .reduce(f64::max);
    crate::losses::LeakageFrozenProbeAuditSection {
        status: report.calibration_status.clone(),
        route_count: report.routes.len(),
        capacity_sweep_rows: report.capacity_sweep.len(),
        best_improvement_over_baseline,
        artifact,
        interpretation: report.claim_boundary.clone(),
    }
}

fn build_probe_capacity_report(config: &crate::config::SemanticProbeConfig) -> ProbeCapacityReport {
    let linear_baseline_only = config.hidden_layers == 0;
    ProbeCapacityReport {
        hidden_dim: config.hidden_dim,
        hidden_layers: config.hidden_layers,
        architecture: if linear_baseline_only {
            "linear".to_string()
        } else {
            "mlp_relu".to_string()
        },
        linear_baseline_only,
        interpretation: if linear_baseline_only {
            "linear probe baseline only; low leakage proxy should be treated as diagnostic, not proof that off-modality targets are unrecoverable".to_string()
        } else {
            "configured MLP probe capacity supports a stronger predictive leakage audit when probe losses also beat trivial baselines".to_string()
        },
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

fn compute_slot_stability(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> SlotStabilityMetrics {
    if forwards.is_empty() {
        return SlotStabilityMetrics::default();
    }
    let topology_slot_alignment = per_slot_alignment(
        examples,
        forwards,
        |forward| &forward.slots.topology.slot_activations,
        topology_alignment_target,
    );
    let geometry_slot_alignment = per_slot_alignment(
        examples,
        forwards,
        |forward| &forward.slots.geometry.slot_activations,
        geometry_alignment_target,
    );
    let pocket_slot_alignment = per_slot_alignment(
        examples,
        forwards,
        |forward| &forward.slots.pocket.slot_activations,
        pocket_alignment_target,
    );
    let collapse_warnings = vec![
        slot_collapse_warning("topology", forwards, |forward| &forward.slots.topology),
        slot_collapse_warning("geometry", forwards, |forward| &forward.slots.geometry),
        slot_collapse_warning("pocket", forwards, |forward| &forward.slots.pocket),
    ];
    let modality_usage = vec![
        slot_modality_usage_report(
            "topology",
            "adjacency_bond_structure",
            forwards,
            |forward| &forward.slots.topology,
            &topology_slot_alignment,
        ),
        slot_modality_usage_report(
            "geometry",
            "distance_geometry",
            forwards,
            |forward| &forward.slots.geometry,
            &geometry_slot_alignment,
        ),
        slot_modality_usage_report(
            "pocket",
            "pocket_features",
            forwards,
            |forward| &forward.slots.pocket,
            &pocket_slot_alignment,
        ),
    ];
    let stage_guard_warning_count = modality_usage
        .iter()
        .filter(|report| report.stage_guard_collapse_warning)
        .count();
    SlotStabilityMetrics {
        topology_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.topology.slot_activations)
        }),
        geometry_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.geometry.slot_activations)
        }),
        pocket_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.pocket.slot_activations)
        }),
        topology_signature_similarity: slot_signature_similarity(forwards, |forward| {
            &forward.slots.topology.slots
        }),
        geometry_signature_similarity: slot_signature_similarity(forwards, |forward| {
            &forward.slots.geometry.slots
        }),
        pocket_signature_similarity: slot_signature_similarity(forwards, |forward| {
            &forward.slots.pocket.slots
        }),
        topology_probe_alignment: mean_by(forwards, |forward| {
            if forward.probes.topology_adjacency_logits.numel() == 0 {
                0.0
            } else {
                let probe = forward
                    .probes
                    .topology_adjacency_logits
                    .sigmoid()
                    .mean(tch::Kind::Float);
                let activity = forward
                    .slots
                    .topology
                    .slot_activations
                    .mean(tch::Kind::Float);
                (&probe * &activity).double_value(&[])
            }
        }),
        geometry_probe_alignment: mean_by(forwards, |forward| {
            if forward.probes.geometry_distance_predictions.numel() == 0 {
                0.0
            } else {
                let probe = 1.0
                    / (1.0
                        + forward
                            .probes
                            .geometry_distance_predictions
                            .abs()
                            .mean(tch::Kind::Float)
                            .double_value(&[]));
                probe * active_slot_fraction(&forward.slots.geometry.slot_activations)
            }
        }),
        pocket_probe_alignment: mean_by(forwards, |forward| {
            if forward.probes.pocket_feature_predictions.numel() == 0 {
                0.0
            } else {
                let probe = 1.0
                    / (1.0
                        + forward
                            .probes
                            .pocket_feature_predictions
                            .abs()
                            .mean(tch::Kind::Float)
                            .double_value(&[]));
                probe * active_slot_fraction(&forward.slots.pocket.slot_activations)
            }
        }),
        topology_slot_alignment,
        geometry_slot_alignment,
        pocket_slot_alignment,
        signature_matching: vec![
            slot_signature_match_for_forwards("topology", forwards, |forward| {
                &forward.slots.topology.slots
            }),
            slot_signature_match_for_forwards("geometry", forwards, |forward| {
                &forward.slots.geometry.slots
            }),
            slot_signature_match_for_forwards("pocket", forwards, |forward| {
                &forward.slots.pocket.slots
            }),
        ],
        collapse_warnings,
        modality_usage,
        stage_guard_warning_count,
    }
}

fn mean_by(forwards: &[ResearchForward], f: impl Fn(&ResearchForward) -> f64) -> f64 {
    forwards.iter().map(f).sum::<f64>() / forwards.len().max(1) as f64
}

fn per_slot_alignment(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    activations: impl Fn(&ResearchForward) -> &tch::Tensor,
    target: impl Fn(&crate::data::MolecularExample) -> f64,
) -> Vec<f64> {
    let slot_count = forwards
        .iter()
        .map(|forward| activations(forward).size().first().copied().unwrap_or(0).max(0))
        .max()
        .unwrap_or(0) as usize;
    if slot_count == 0 || examples.is_empty() || forwards.is_empty() {
        return Vec::new();
    }
    let mut totals = vec![0.0; slot_count];
    let mut counts = vec![0usize; slot_count];
    for (example, forward) in examples.iter().zip(forwards.iter()) {
        let slot_activations = activations(forward);
        let target = target(example).max(0.0);
        for slot in 0..slot_count.min(slot_activations.numel()) {
            let activation = slot_activations.double_value(&[slot as i64]).max(0.0);
            totals[slot] += activation * target;
            counts[slot] += 1;
        }
    }
    totals
        .into_iter()
        .zip(counts)
        .map(|(total, count)| total / count.max(1) as f64)
        .collect()
}

fn topology_alignment_target(example: &crate::data::MolecularExample) -> f64 {
    if example.topology.adjacency.numel() == 0 {
        0.0
    } else {
        example
            .topology
            .adjacency
            .mean(tch::Kind::Float)
            .double_value(&[])
    }
}

fn geometry_alignment_target(example: &crate::data::MolecularExample) -> f64 {
    if example.geometry.pairwise_distances.numel() == 0 {
        0.0
    } else {
        1.0 / (1.0
            + example
                .geometry
                .pairwise_distances
                .mean(tch::Kind::Float)
                .double_value(&[])
                .max(0.0))
    }
}

fn pocket_alignment_target(example: &crate::data::MolecularExample) -> f64 {
    if example.pocket.atom_features.numel() == 0 {
        0.0
    } else {
        example
            .pocket
            .atom_features
            .abs()
            .mean(tch::Kind::Float)
            .double_value(&[])
    }
}

fn slot_signature_similarity<'a>(
    forwards: &'a [ResearchForward],
    slots: impl Fn(&'a ResearchForward) -> &'a tch::Tensor,
) -> f64 {
    let signatures = forwards
        .iter()
        .map(|forward| mean_slot(slots(forward)))
        .collect::<Vec<_>>();
    if signatures.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for left in 0..signatures.len() {
        for right in (left + 1)..signatures.len() {
            total += cosine_similarity(&signatures[left], &signatures[right]).abs();
            count += 1;
        }
    }
    total / count.max(1) as f64
}

fn slot_signature_match_for_forwards<'a>(
    modality: &str,
    forwards: &'a [ResearchForward],
    slots: impl Fn(&'a ResearchForward) -> &'a tch::Tensor + Copy,
) -> SlotSignatureMatchReport {
    if forwards.len() < 2 {
        return SlotSignatureMatchReport {
            modality: modality.to_string(),
            comparison_scope: "within_split_repeated_signature_proxy".to_string(),
            collapse_warning: true,
            ..SlotSignatureMatchReport::default()
        };
    }
    let midpoint = (forwards.len() / 2).max(1);
    let left = average_slot_signatures(&forwards[..midpoint], slots);
    let right = average_slot_signatures(&forwards[midpoint..], slots);
    match_slot_signatures(
        modality,
        "within_split_repeated_signature_proxy",
        &left,
        &right,
        0.5,
    )
}

fn average_slot_signatures<'a>(
    forwards: &'a [ResearchForward],
    slots: impl Fn(&'a ResearchForward) -> &'a tch::Tensor,
) -> Vec<Vec<f64>> {
    let Some(first) = forwards.first() else {
        return Vec::new();
    };
    let first_slots = slots(first);
    let slot_count = first_slots.size().first().copied().unwrap_or(0).max(0) as usize;
    let hidden_dim = first_slots
        .size()
        .get(1)
        .copied()
        .unwrap_or(0)
        .max(0)
        .min(32) as usize;
    if slot_count == 0 || hidden_dim == 0 {
        return Vec::new();
    }
    let mut totals = vec![vec![0.0; hidden_dim]; slot_count];
    let mut counts = vec![0usize; slot_count];
    for forward in forwards {
        let tensor = slots(forward);
        for slot in 0..slot_count.min(tensor.size().first().copied().unwrap_or(0).max(0) as usize)
        {
            for dim in 0..hidden_dim {
                totals[slot][dim] += tensor.double_value(&[slot as i64, dim as i64]);
            }
            counts[slot] += 1;
        }
    }
    for (signature, count) in totals.iter_mut().zip(counts) {
        let denom = count.max(1) as f64;
        for value in signature {
            *value /= denom;
        }
    }
    totals
}

fn match_slot_signatures(
    modality: &str,
    comparison_scope: &str,
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    threshold: f64,
) -> SlotSignatureMatchReport {
    let mut pairs = Vec::new();
    for (left_index, left_signature) in left.iter().enumerate() {
        for (right_index, right_signature) in right.iter().enumerate() {
            pairs.push((
                vector_cosine_similarity(left_signature, right_signature).abs(),
                left_index,
                right_index,
            ));
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut matched_left = vec![false; left.len()];
    let mut matched_right = vec![false; right.len()];
    let mut total_similarity = 0.0;
    let mut matched_slot_count = 0usize;
    for (similarity, left_index, right_index) in pairs {
        if similarity < threshold || matched_left[left_index] || matched_right[right_index] {
            continue;
        }
        matched_left[left_index] = true;
        matched_right[right_index] = true;
        total_similarity += similarity;
        matched_slot_count += 1;
    }
    let unmatched_left_slots = matched_left.iter().filter(|matched| !**matched).count();
    let unmatched_right_slots = matched_right.iter().filter(|matched| !**matched).count();
    let mean_matched_similarity = if matched_slot_count == 0 {
        0.0
    } else {
        total_similarity / matched_slot_count as f64
    };
    SlotSignatureMatchReport {
        modality: modality.to_string(),
        comparison_scope: comparison_scope.to_string(),
        matched_slot_count,
        unmatched_left_slots,
        unmatched_right_slots,
        mean_matched_similarity,
        cross_seed_matching_score: if comparison_scope.contains("cross_seed") {
            mean_matched_similarity
        } else {
            0.0
        },
        collapse_warning: matched_slot_count < left.len().min(right.len())
            || mean_matched_similarity < threshold,
    }
}

fn vector_cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    let len = left.len().min(right.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for index in 0..len {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
    }
    dot / (left_norm.sqrt() * right_norm.sqrt()).max(1e-6)
}

fn slot_collapse_warning(
    modality: &str,
    forwards: &[ResearchForward],
    slot_encoding: impl for<'a> Fn(&'a ResearchForward) -> &'a crate::models::SlotEncoding,
) -> SlotCollapseWarning {
    let active_fraction = mean_by(forwards, |forward| {
        active_slot_fraction(&slot_encoding(forward).slot_activations)
    });
    let attention_visible_fraction = mean_by(forwards, |forward| {
        active_slot_fraction(&slot_encoding(forward).active_slot_mask)
    });
    let assignment_entropy = mean_by(forwards, |forward| {
        slot_weight_entropy_for_collapse(&slot_encoding(forward).slot_weights)
    });
    let dominant_slot_fraction = mean_by(forwards, |forward| {
        slot_encoding(forward)
            .slot_weights
            .max()
            .double_value(&[])
            .max(0.0)
    });
    slot_collapse_warning_from_stats(
        modality,
        active_fraction,
        attention_visible_fraction,
        assignment_entropy,
        dominant_slot_fraction,
    )
}

fn slot_modality_usage_report(
    modality: &str,
    target_family: &str,
    forwards: &[ResearchForward],
    slot_encoding: impl for<'a> Fn(&'a ResearchForward) -> &'a crate::models::SlotEncoding,
    alignment: &[f64],
) -> SlotModalityUsageReport {
    let slot_count = forwards
        .iter()
        .map(|forward| {
            slot_encoding(forward)
                .slot_activations
                .size()
                .first()
                .copied()
                .unwrap_or(0)
                .max(0)
        })
        .max()
        .unwrap_or(0) as usize;
    if forwards.is_empty() || slot_count == 0 {
        return SlotModalityUsageReport {
            modality: modality.to_string(),
            semantic_enrichment: semantic_enrichment_summary(target_family, alignment),
            ..SlotModalityUsageReport::default()
        };
    }
    let mut activation_totals = vec![0.0; slot_count];
    let mut activation_counts = vec![0usize; slot_count];
    for forward in forwards {
        let activations = &slot_encoding(forward).slot_activations;
        let observed = slot_count.min(activations.numel());
        for slot in 0..observed {
            activation_totals[slot] += activations.double_value(&[slot as i64]).clamp(0.0, 1.0);
            activation_counts[slot] += 1;
        }
    }
    let mean_activations = activation_totals
        .into_iter()
        .zip(activation_counts)
        .map(|(total, count)| total / count.max(1) as f64)
        .collect::<Vec<_>>();
    let dead_slot_count = mean_activations
        .iter()
        .filter(|activation| **activation <= 0.05)
        .count();
    let diffuse_slot_count = mean_activations
        .iter()
        .filter(|activation| **activation > 0.05 && **activation < 0.50)
        .count();
    let saturated_slot_count = mean_activations
        .iter()
        .filter(|activation| **activation >= 0.95)
        .count();
    let mean_active_slot_fraction = mean_by(forwards, |forward| {
        active_slot_fraction(&slot_encoding(forward).slot_activations)
    });
    let attention_visible_fraction = mean_by(forwards, |forward| {
        active_slot_fraction(&slot_encoding(forward).active_slot_mask)
    });
    let assignment_entropy = mean_by(forwards, |forward| {
        slot_weight_entropy_for_collapse(&slot_encoding(forward).slot_weights)
    });
    let dominant_slot_fraction = mean_by(forwards, |forward| {
        slot_encoding(forward)
            .slot_weights
            .max()
            .double_value(&[])
            .max(0.0)
    });
    let stage_guard_collapse_warning = dead_slot_count == slot_count
        || saturated_slot_count == slot_count
        || dominant_slot_fraction >= 0.85
        || assignment_entropy <= 1.0e-6;
    let denom = slot_count.max(1) as f64;
    SlotModalityUsageReport {
        modality: modality.to_string(),
        sample_count: forwards.len(),
        slot_count,
        active_slot_fraction: mean_active_slot_fraction,
        attention_visible_fraction,
        assignment_entropy,
        dominant_slot_fraction,
        dead_slot_count,
        diffuse_slot_count,
        saturated_slot_count,
        dead_slot_fraction: dead_slot_count as f64 / denom,
        diffuse_slot_fraction: diffuse_slot_count as f64 / denom,
        saturated_slot_fraction: saturated_slot_count as f64 / denom,
        stage_guard_collapse_warning,
        semantic_enrichment: semantic_enrichment_summary(target_family, alignment),
    }
}

fn semantic_enrichment_summary(
    target_family: &str,
    alignment: &[f64],
) -> SlotSemanticEnrichmentSummary {
    let values = alignment
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return SlotSemanticEnrichmentSummary {
            target_family: target_family.to_string(),
            ..SlotSemanticEnrichmentSummary::default()
        };
    }
    let mean_alignment = values.iter().sum::<f64>() / values.len() as f64;
    let max_alignment = values.iter().copied().fold(0.0_f64, f64::max);
    let enrichment_threshold = (mean_alignment * 1.25).max(1.0e-9);
    let enriched_slot_count = values
        .iter()
        .filter(|value| **value >= enrichment_threshold)
        .count();
    let total = values.iter().sum::<f64>().max(1.0e-12);
    let enrichment_entropy = values
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| {
            let probability = value / total;
            -probability * probability.ln()
        })
        .sum::<f64>();
    SlotSemanticEnrichmentSummary {
        target_family: target_family.to_string(),
        mean_alignment,
        max_alignment,
        enriched_slot_count,
        enrichment_entropy,
        role_enrichment_score: max_alignment / mean_alignment.max(1.0e-12),
    }
}

fn slot_weight_entropy_for_collapse(weights: &tch::Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    let normalized = (weights / weights.sum(tch::Kind::Float).clamp_min(1e-12)).clamp_min(1e-12);
    (-(&normalized * normalized.log()).sum(tch::Kind::Float)).double_value(&[])
}

fn slot_collapse_warning_from_stats(
    modality: &str,
    active_slot_fraction: f64,
    attention_visible_fraction: f64,
    assignment_entropy: f64,
    dominant_slot_fraction: f64,
) -> SlotCollapseWarning {
    let (status, warning) = if active_slot_fraction <= 0.0 && attention_visible_fraction <= 0.0 {
        (
            "dead",
            "no active or attention-visible slots were observed for this modality",
        )
    } else if active_slot_fraction >= 0.95 {
        (
            "saturated",
            "nearly every slot is active; specialization pressure may be weak",
        )
    } else if dominant_slot_fraction >= 0.85 {
        (
            "single_slot_dominated",
            "one slot carries most assignment mass; monitor collapse and balance losses",
        )
    } else {
        ("balanced", "slot utilization is within conservative diagnostic bounds")
    };
    SlotCollapseWarning {
        modality: modality.to_string(),
        status: status.to_string(),
        active_slot_fraction,
        attention_visible_fraction,
        assignment_entropy,
        dominant_slot_fraction,
        warning: warning.to_string(),
    }
}
