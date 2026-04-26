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

    let (reviewer_status, reviewer_passed) = if test_leakage > hard
        || split_checks.protein_overlap_detected
        || split_checks.duplicate_example_ids_detected
    {
        ("fail".to_string(), false)
    } else if test_leakage > preferred || max_regression > regression_limit {
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
    LeakageCalibrationReport {
        recommended_delta_leak: summary.config.research.training.loss_weights.delta_leak,
        evaluated_variants: deltas.len(),
        preferred_max_leakage_proxy_mean: preferred,
        hard_max_leakage_proxy_mean: hard,
        max_leakage_proxy_regression: regression_limit,
        reviewer_status,
        reviewer_passed,
        reviewer_reasons: reasons,
        decision,
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

fn compute_slot_stability(forwards: &[ResearchForward]) -> SlotStabilityMetrics {
    if forwards.is_empty() {
        return SlotStabilityMetrics::default();
    }
    SlotStabilityMetrics {
        topology_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.topology.slot_weights)
        }),
        geometry_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.geometry.slot_weights)
        }),
        pocket_activation_mean: mean_by(forwards, |forward| {
            active_slot_fraction(&forward.slots.pocket.slot_weights)
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
                let activity = forward.slots.topology.slot_weights.mean(tch::Kind::Float);
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
                probe * active_slot_fraction(&forward.slots.geometry.slot_weights)
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
                probe * active_slot_fraction(&forward.slots.pocket.slot_weights)
            }
        }),
    }
}

fn mean_by(forwards: &[ResearchForward], f: impl Fn(&ResearchForward) -> f64) -> f64 {
    forwards.iter().map(f).sum::<f64>() / forwards.len().max(1) as f64
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

