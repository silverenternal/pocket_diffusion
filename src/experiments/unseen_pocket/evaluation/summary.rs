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
) -> GenerationQualitySummary {
    GenerationQualitySummary {
        primary_objective: primary_objective_label(
            ablation
                .primary_objective_override
                .unwrap_or(research.training.primary_objective),
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

fn interaction_mode_label(mode: CrossAttentionMode) -> String {
    match mode {
        CrossAttentionMode::Lightweight => "lightweight".to_string(),
        CrossAttentionMode::Transformer => "transformer".to_string(),
    }
}
