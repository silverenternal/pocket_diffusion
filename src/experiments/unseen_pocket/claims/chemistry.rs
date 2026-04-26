fn build_chemistry_benchmark_evidence(
    summary: &UnseenPocketExperimentSummary,
) -> ChemistryBenchmarkEvidence {
    let chemistry = &summary.test.real_generation_metrics.chemistry_validity;
    let layer = if summary
        .test
        .layered_generation_metrics
        .reranked_candidates
        .candidate_count
        > 0
    {
        &summary.test.layered_generation_metrics.reranked_candidates
    } else {
        &summary
            .test
            .layered_generation_metrics
            .inferred_bond_candidates
    };
    let sanitized_fraction = chemistry.metrics.get("rdkit_sanitized_fraction").copied();
    let parseable_fraction = chemistry.metrics.get("rdkit_parseable_fraction").copied();
    let finite_conformer_fraction = chemistry
        .metrics
        .get("rdkit_finite_conformer_fraction")
        .copied();
    let unique_smiles_fraction = chemistry
        .metrics
        .get("rdkit_unique_smiles_fraction")
        .copied()
        .or(summary.test.comparison_summary.unique_smiles_fraction);
    let validity_quality_score = match (sanitized_fraction, unique_smiles_fraction) {
        (Some(sanitized), Some(unique)) => Some((sanitized + unique) / 2.0),
        (Some(sanitized), None) => Some(sanitized),
        (None, Some(unique)) => Some(unique),
        (None, None) => None,
    };
    let novelty_diversity_score = [
        layer.atom_type_sequence_diversity,
        layer.bond_topology_diversity,
        layer.coordinate_shape_diversity,
        layer.novel_atom_type_sequence_fraction,
        layer.novel_bond_topology_fraction,
        layer.novel_coordinate_shape_fraction,
    ]
    .into_iter()
    .sum::<f64>()
        / 6.0;
    let backend_backed = claim_is_real_backend_backed(summary) && sanitized_fraction.is_some();
    let stronger_candidate_threshold = 8;
    let stronger_required_backend_metrics = vec![
        "rdkit_parseable_fraction".to_string(),
        "rdkit_finite_conformer_fraction".to_string(),
        "rdkit_sanitized_fraction".to_string(),
        "rdkit_unique_smiles_fraction".to_string(),
    ];
    let stronger_checks = [
        parseable_fraction.map(|value| value >= 0.95),
        finite_conformer_fraction.map(|value| value >= 0.95),
        sanitized_fraction.map(|value| value >= 0.95),
        unique_smiles_fraction.map(|value| value >= 0.5),
        Some(layer.candidate_count >= stronger_candidate_threshold),
        Some(novelty_diversity_score >= 0.75),
    ];
    let stronger_check_count = stronger_checks.len();
    let stronger_passed = stronger_checks
        .iter()
        .all(|result| matches!(result, Some(true)));
    let stronger_support_score = Some(
        stronger_checks
            .iter()
            .filter(|result| matches!(result, Some(true)))
            .count() as f64
            / stronger_check_count as f64,
    );
    let val_family_count = summary
        .split_report
        .val
        .protein_family_proxy_histogram
        .len();
    let test_family_count = summary
        .split_report
        .test
        .protein_family_proxy_histogram
        .len();
    let parsed_examples = summary.dataset_validation.parsed_examples;
    let retained_label_coverage = summary.dataset_validation.retained_label_coverage;
    let surface_label = summary
        .config
        .surface_label
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let configured_external_benchmark = summary
        .config
        .reviewer_benchmark
        .dataset
        .clone()
        .or_else(|| {
            if surface_label.contains("pdbbindpp")
                || summary
                    .config
                    .research
                    .training
                    .checkpoint_dir
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .contains("pdbbindpp")
            {
                Some("pdbbindpp-2020".to_string())
            } else {
                None
            }
        });
    let external_benchmark_label = configured_external_benchmark
        .clone()
        .unwrap_or_else(|| "configured_external_benchmark".to_string());
    let external_required_checks = vec![
        format!(
            "reviewer_benchmark.dataset is configured for {}",
            external_benchmark_label
        ),
        "dataset_validation.parsed_examples >= 100".to_string(),
        "dataset_validation.retained_label_coverage >= 0.8".to_string(),
        "val protein-family count >= 10".to_string(),
        "test protein-family count >= 10".to_string(),
        "reviewer benchmark-plus chemistry gate already passed".to_string(),
    ];
    let external_checks = [
        configured_external_benchmark.is_some(),
        parsed_examples >= 100,
        retained_label_coverage >= 0.8,
        val_family_count >= 10,
        test_family_count >= 10,
        stronger_passed,
    ];
    let external_benchmark_backed = external_checks.iter().all(|passed| *passed);
    let external_benchmark_support_score = Some(
        external_checks.iter().filter(|passed| **passed).count() as f64
            / external_checks.len() as f64,
    );
    let benchmark_components = if backend_backed {
        let mut components = vec![
            "rdkit_sanitized_fraction".to_string(),
            "rdkit_unique_smiles_fraction".to_string(),
            "atom_type_sequence_diversity".to_string(),
            "bond_topology_diversity".to_string(),
            "coordinate_shape_diversity".to_string(),
            "novel_atom_type_sequence_fraction".to_string(),
            "novel_bond_topology_fraction".to_string(),
            "novel_coordinate_shape_fraction".to_string(),
        ];
        if stronger_passed {
            components.extend([
                "rdkit_parseable_fraction".to_string(),
                "rdkit_finite_conformer_fraction".to_string(),
                "review_candidate_count".to_string(),
                "stronger_benchmark_support_score".to_string(),
            ]);
        }
        if external_benchmark_backed {
            components.extend([
                "external_benchmark_dataset".to_string(),
                "parsed_examples".to_string(),
                "retained_label_coverage".to_string(),
                "val_protein_family_count".to_string(),
                "test_protein_family_count".to_string(),
                "external_benchmark_support_score".to_string(),
            ]);
        }
        components
    } else {
        vec![
            "atom_type_sequence_diversity".to_string(),
            "bond_topology_diversity".to_string(),
            "coordinate_shape_diversity".to_string(),
            "novel_atom_type_sequence_fraction".to_string(),
            "novel_bond_topology_fraction".to_string(),
            "novel_coordinate_shape_fraction".to_string(),
        ]
    };
    let evidence_tier = if backend_backed && external_benchmark_backed {
        "external_benchmark_backed".to_string()
    } else if backend_backed && stronger_passed {
        "reviewer_benchmark_plus".to_string()
    } else if backend_backed {
        "local_benchmark_style".to_string()
    } else {
        "proxy_only".to_string()
    };
    let stronger_benchmark_note = if !backend_backed {
        "Stronger reviewer chemistry evidence is unavailable without an active backend-backed chemistry surface.".to_string()
    } else if stronger_passed {
        format!(
            "This surface clears the stronger reviewer chemistry gate because parseable, finite-conformer, sanitized, and unique-SMILES fractions all pass with review_candidate_count={} and novelty_diversity_score={:.4}.",
            layer.candidate_count,
            novelty_diversity_score,
        )
    } else {
        format!(
            "This surface stays at local benchmark-style chemistry evidence because the stronger reviewer gate is only {:.3} supported; it requires parseable, finite-conformer, sanitized, and unique-SMILES backend quality plus at least {} review-layer candidates and novelty_diversity_score >= 0.75.",
            stronger_support_score.unwrap_or(0.0),
            stronger_candidate_threshold,
        )
    };
    let external_benchmark_note = if !backend_backed {
        "External benchmark-backed chemistry evidence is unavailable without an active backend-backed chemistry surface.".to_string()
    } else if external_benchmark_backed {
        format!(
            "This surface clears the explicit external benchmark-dataset chemistry tier for {} with parsed_examples={}, retained_label_coverage={:.4}, val_family_count={}, test_family_count={}, and reviewer benchmark-plus chemistry already passing.",
            external_benchmark_label,
            parsed_examples,
            retained_label_coverage,
            val_family_count,
            test_family_count,
        )
    } else {
        format!(
            "This surface does not yet clear the explicit external benchmark-dataset chemistry tier; support is {:.3} and requires a configured benchmark dataset label, parsed_examples>=100, retained_label_coverage>=0.8, held-out family counts>=10 on validation/test, and reviewer benchmark-plus chemistry already passing.",
            external_benchmark_support_score.unwrap_or(0.0),
        )
    };
    let interpretation = if backend_backed && external_benchmark_backed {
        format!(
            "Combines backend-backed chemistry quality, held-out-pocket novelty/diversity aggregates, reviewer benchmark-plus checks, and an explicit external benchmark-dataset layer on {} (validity_quality_score={:.4}, novelty_diversity_score={:.4}, reviewer_support_score={:.4}, external_support_score={:.4}).",
            external_benchmark_label,
            validity_quality_score.unwrap_or(0.0),
            novelty_diversity_score,
            stronger_support_score.unwrap_or(0.0),
            external_benchmark_support_score.unwrap_or(0.0),
        )
    } else if backend_backed && stronger_passed {
        format!(
            "Combines backend-backed chemistry quality, held-out-pocket novelty/diversity aggregates, and explicit reviewer benchmark checks (validity_quality_score={:.4}, novelty_diversity_score={:.4}, support_score={:.4}) for a stronger reviewer benchmark-plus chemistry summary.",
            validity_quality_score.unwrap_or(0.0),
            novelty_diversity_score,
            stronger_support_score.unwrap_or(0.0),
        )
    } else if backend_backed {
        format!(
            "Combines backend-measured sanitization and unique-SMILES chemistry quality with held-out-pocket novelty/diversity aggregates (validity_quality_score={:.4}, novelty_diversity_score={:.4}) for a local benchmark-style chemistry summary.",
            validity_quality_score.unwrap_or(0.0),
            novelty_diversity_score,
        )
    } else {
        format!(
            "No active chemistry backend was attached, so chemistry evidence remains proxy-only; the novelty/diversity aggregate ({:.4}) is structural-signature-based rather than backend benchmark-backed.",
            novelty_diversity_score,
        )
    };
    ChemistryBenchmarkEvidence {
        backend_backed,
        sanitized_fraction,
        unique_smiles_fraction,
        review_candidate_count: layer.candidate_count,
        validity_quality_score,
        novelty_diversity_score,
        evidence_tier,
        stronger_reviewer_benchmark: stronger_passed,
        external_benchmark_backed,
        external_benchmark_dataset: if external_benchmark_backed {
            configured_external_benchmark
        } else {
            None
        },
        stronger_review_candidate_threshold: stronger_candidate_threshold,
        stronger_required_backend_metrics,
        stronger_benchmark_support_score: stronger_support_score,
        external_benchmark_support_score,
        stronger_benchmark_note,
        external_required_checks,
        external_benchmark_note,
        benchmark_components,
        interpretation,
    }
}

