#[derive(Debug, Default)]
struct QualityFilterCounts {
    filtered_examples: usize,
    unlabeled_examples: usize,
    ligand_atom_limit: usize,
    pocket_atom_limit: usize,
    missing_source_provenance: usize,
    missing_affinity_metadata: usize,
}

fn apply_quality_filters(
    examples: &mut Vec<MolecularExample>,
    validation: &mut DatasetValidationReport,
    filters: &DataQualityFilterConfig,
) -> Result<(), DataParseError> {
    validation.observed_fallback_fraction = fraction_usize(
        validation.fallback_pocket_extractions,
        validation.parsed_examples,
    ) as f32;
    if let Some(max_fallback_fraction) = filters.max_fallback_fraction {
        if validation.observed_fallback_fraction > max_fallback_fraction {
            return Err(DataParseError::Discovery {
                root: PathBuf::from("."),
                message: format!(
                    "observed fallback fraction {:.4} exceeds configured maximum {:.4}",
                    validation.observed_fallback_fraction, max_fallback_fraction
                ),
            });
        }
    }

    let counts = filter_examples_by_quality(examples, filters);
    validation.quality_filtered_examples = counts.filtered_examples;
    validation.quality_filtered_unlabeled_examples = counts.unlabeled_examples;
    validation.quality_filtered_ligand_atom_limit = counts.ligand_atom_limit;
    validation.quality_filtered_pocket_atom_limit = counts.pocket_atom_limit;
    validation.quality_filtered_missing_source_provenance = counts.missing_source_provenance;
    validation.quality_filtered_missing_affinity_metadata = counts.missing_affinity_metadata;

    if let Some(min_label_coverage) = filters.min_label_coverage {
        let summary = summarize_retained_examples(examples);
        if summary.label_coverage < min_label_coverage {
            return Err(DataParseError::Discovery {
                root: PathBuf::from("."),
                message: format!(
                    "retained label coverage {:.4} is below configured minimum {:.4}",
                    summary.label_coverage, min_label_coverage
                ),
            });
        }
    }
    let summary = summarize_retained_examples(examples);
    if let Some(max_approximate_label_fraction) = filters.max_approximate_label_fraction {
        if summary.approximate_label_fraction > max_approximate_label_fraction {
            return Err(DataParseError::Discovery {
                root: PathBuf::from("."),
                message: format!(
                    "retained approximate-label fraction {:.4} exceeds configured maximum {:.4}",
                    summary.approximate_label_fraction, max_approximate_label_fraction
                ),
            });
        }
    }
    if let Some(min_normalization_provenance_coverage) = filters.min_normalization_provenance_coverage
    {
        if summary.normalization_provenance_coverage < min_normalization_provenance_coverage {
            return Err(DataParseError::Discovery {
                root: PathBuf::from("."),
                message: format!(
                    "retained normalization provenance coverage {:.4} is below configured minimum {:.4}",
                    summary.normalization_provenance_coverage, min_normalization_provenance_coverage
                ),
            });
        }
    }

    Ok(())
}

fn filter_examples_by_quality(
    examples: &mut Vec<MolecularExample>,
    filters: &DataQualityFilterConfig,
) -> QualityFilterCounts {
    let mut counts = QualityFilterCounts::default();
    examples.retain(|example| {
        let mut keep = true;
        if filters.min_label_coverage.is_some() && example.targets.affinity_kcal_mol.is_none() {
            counts.unlabeled_examples += 1;
            keep = false;
        }
        if let Some(max_ligand_atoms) = filters.max_ligand_atoms {
            if ligand_atom_count(example) > max_ligand_atoms {
                counts.ligand_atom_limit += 1;
                keep = false;
            }
        }
        if let Some(max_pocket_atoms) = filters.max_pocket_atoms {
            if pocket_atom_count(example) > max_pocket_atoms {
                counts.pocket_atom_limit += 1;
                keep = false;
            }
        }
        if filters.require_source_structure_provenance
            && (example.source_pocket_path.is_none() || example.source_ligand_path.is_none())
        {
            counts.missing_source_provenance += 1;
            keep = false;
        }
        if filters.require_affinity_metadata
            && example.targets.affinity_kcal_mol.is_some()
            && (example.targets.affinity_measurement_type.is_none()
                || example.targets.affinity_normalization_provenance.is_none())
        {
            counts.missing_affinity_metadata += 1;
            keep = false;
        }
        if !keep {
            counts.filtered_examples += 1;
        }
        keep
    });
    counts
}

fn ligand_atom_count(example: &MolecularExample) -> usize {
    example
        .topology
        .atom_types
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize
}

fn pocket_atom_count(example: &MolecularExample) -> usize {
    example
        .pocket
        .coords
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize
}

fn fraction_usize(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Debug, Default)]
struct RetainedExampleSummary {
    label_coverage: f32,
    source_provenance_coverage: f32,
    approximate_affinity_labels: usize,
    approximate_label_fraction: f32,
    normalization_provenance_coverage: f32,
    missing_normalization_provenance: usize,
    missing_measurement_type: usize,
    measurement_family_histogram: BTreeMap<String, usize>,
    normalization_provenance_values: BTreeSet<String>,
    ligand_atom_count_histogram: BTreeMap<String, usize>,
    pocket_atom_count_histogram: BTreeMap<String, usize>,
    mean_ligand_atom_count: f64,
    coordinate_frame_origin_valid_examples: usize,
    ligand_centered_coordinate_frame_examples: usize,
}

fn summarize_retained_examples(examples: &[MolecularExample]) -> RetainedExampleSummary {
    let labeled_examples: Vec<&MolecularExample> = examples
        .iter()
        .filter(|example| example.targets.affinity_kcal_mol.is_some())
        .collect();
    let approximate_affinity_labels = labeled_examples
        .iter()
        .filter(|example| example.targets.affinity_is_approximate)
        .count();
    let missing_normalization_provenance = labeled_examples
        .iter()
        .filter(|example| example.targets.affinity_normalization_provenance.is_none())
        .count();
    let missing_measurement_type = labeled_examples
        .iter()
        .filter(|example| example.targets.affinity_measurement_type.is_none())
        .count();
    let mut measurement_family_histogram = BTreeMap::new();
    let mut normalization_provenance_values = BTreeSet::new();
    let mut ligand_atom_count_histogram = BTreeMap::new();
    let mut pocket_atom_count_histogram = BTreeMap::new();
    let mut ligand_atom_count_total = 0_usize;
    for example in examples {
        let ligand_atoms = ligand_atom_count(example);
        let pocket_atoms = pocket_atom_count(example);
        ligand_atom_count_total += ligand_atoms;
        *ligand_atom_count_histogram
            .entry(atom_count_bin(ligand_atoms))
            .or_default() += 1;
        *pocket_atom_count_histogram
            .entry(atom_count_bin(pocket_atoms))
            .or_default() += 1;
    }
    for example in &labeled_examples {
        let measurement = example
            .targets
            .affinity_measurement_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        *measurement_family_histogram.entry(measurement).or_default() += 1;
        if let Some(provenance) = &example.targets.affinity_normalization_provenance {
            normalization_provenance_values.insert(provenance.clone());
        }
    }
    let coordinate_frame_origin_valid_examples = examples
        .iter()
        .filter(|example| coordinate_frame_origin_is_finite(example))
        .count();
    let ligand_centered_coordinate_frame_examples = examples
        .iter()
        .filter(|example| example_uses_ligand_centered_model_frame(example))
        .count();

    RetainedExampleSummary {
        label_coverage: fraction_usize(labeled_examples.len(), examples.len()) as f32,
        source_provenance_coverage: fraction_usize(
            examples
                .iter()
                .filter(|example| {
                    example.source_pocket_path.is_some() && example.source_ligand_path.is_some()
                })
                .count(),
            examples.len(),
        ) as f32,
        approximate_affinity_labels,
        approximate_label_fraction: fraction_usize(
            approximate_affinity_labels,
            labeled_examples.len(),
        ) as f32,
        normalization_provenance_coverage: fraction_usize(
            labeled_examples.len().saturating_sub(missing_normalization_provenance),
            labeled_examples.len(),
        ) as f32,
        missing_normalization_provenance,
        missing_measurement_type,
        measurement_family_histogram,
        normalization_provenance_values,
        ligand_atom_count_histogram,
        pocket_atom_count_histogram,
        mean_ligand_atom_count: fraction_usize(ligand_atom_count_total, examples.len()),
        coordinate_frame_origin_valid_examples,
        ligand_centered_coordinate_frame_examples,
    }
}

fn coordinate_frame_origin_is_finite(example: &MolecularExample) -> bool {
    example
        .coordinate_frame_origin
        .iter()
        .all(|value| value.is_finite())
}

fn example_uses_ligand_centered_model_frame(example: &MolecularExample) -> bool {
    if !coordinate_frame_origin_is_finite(example) {
        return false;
    }
    let ligand_atoms = ligand_atom_count(example) as i64;
    if ligand_atoms <= 0 {
        return false;
    }
    tensor_all_finite(&example.geometry.coords)
        && tensor_all_finite(&example.pocket.coords)
        && geometry_centroid_is_ligand_origin(&example.geometry.coords)
}

fn tensor_all_finite(tensor: &tch::Tensor) -> bool {
    if tensor.numel() == 0 {
        return true;
    }
    tensor
        .isfinite()
        .all()
        .to_kind(tch::Kind::Int64)
        .int64_value(&[])
        != 0
}

fn geometry_centroid_is_ligand_origin(coords: &tch::Tensor) -> bool {
    let atom_count = coords.size().first().copied().unwrap_or(0);
    if atom_count <= 0 {
        return false;
    }
    let centroid = coords.mean_dim([0].as_slice(), false, tch::Kind::Float);
    centroid.abs().max().double_value(&[]) <= 1.0e-4
}

fn finalize_validation_report(
    validation: &mut DatasetValidationReport,
    examples: &[MolecularExample],
    generation_target: &GenerationTargetConfig,
) {
    let summary = summarize_retained_examples(examples);
    validation.attached_labels = examples
        .iter()
        .filter(|example| example.targets.affinity_kcal_mol.is_some())
        .count();
    validation.unlabeled_examples = examples.len().saturating_sub(validation.attached_labels);
    validation.retained_label_coverage = summary.label_coverage;
    validation.retained_source_provenance_coverage = summary.source_provenance_coverage;
    validation.retained_approximate_affinity_labels = summary.approximate_affinity_labels;
    validation.retained_approximate_label_fraction = summary.approximate_label_fraction;
    validation.retained_normalization_provenance_coverage =
        summary.normalization_provenance_coverage;
    validation.retained_missing_normalization_provenance =
        summary.missing_normalization_provenance;
    validation.retained_missing_measurement_type = summary.missing_measurement_type;
    validation.retained_measurement_family_count = summary.measurement_family_histogram.len();
    validation.retained_measurement_family_histogram = summary.measurement_family_histogram;
    validation.retained_normalization_provenance_values = summary.normalization_provenance_values;
    validation.retained_ligand_atom_count_histogram = summary.ligand_atom_count_histogram;
    validation.retained_pocket_atom_count_histogram = summary.pocket_atom_count_histogram;
    validation.retained_mean_ligand_atom_count = summary.mean_ligand_atom_count;
    validation.atom_count_prior_provenance =
        atom_count_prior_provenance_label(generation_target).to_string();
    validation.atom_count_prior_mae = atom_count_prior_mae(examples, generation_target);
    validation.coordinate_frame_contract =
        "ligand_centered_model_coordinates_with_coordinate_frame_origin".to_string();
    validation.coordinate_frame_artifact_contract =
        "candidate.coords are ligand-centered model-frame coordinates; coordinate_frame_origin reconstructs source-frame coordinates".to_string();
    validation.coordinate_frame_origin_valid_examples =
        summary.coordinate_frame_origin_valid_examples;
    validation.ligand_centered_coordinate_frame_examples =
        summary.ligand_centered_coordinate_frame_examples;
    validation.pocket_coordinates_centered_upstream = !examples.is_empty()
        && summary.ligand_centered_coordinate_frame_examples == examples.len();
    validation.source_coordinate_reconstruction_supported = !examples.is_empty()
        && summary.coordinate_frame_origin_valid_examples == examples.len()
        && summary.ligand_centered_coordinate_frame_examples == examples.len();
    record_target_context_leakage_contract(validation, examples, generation_target);
}

fn atom_count_bin(count: usize) -> String {
    match count {
        0..=8 => "000-008".to_string(),
        9..=16 => "009-016".to_string(),
        17..=32 => "017-032".to_string(),
        33..=64 => "033-064".to_string(),
        _ => "065-plus".to_string(),
    }
}

fn atom_count_prior_provenance_label(generation_target: &GenerationTargetConfig) -> &'static str {
    match generation_target.generation_mode {
        GenerationModeConfig::PocketOnlyInitializationBaseline => "fixed",
        GenerationModeConfig::DeNovoInitialization => {
            if generation_target
                .de_novo_initialization
                .dataset_calibrated_atom_count
                .is_some()
            {
                "dataset_calibrated"
            } else {
                "pocket_volume"
            }
        }
        GenerationModeConfig::TargetLigandDenoising
        | GenerationModeConfig::LigandRefinement
        | GenerationModeConfig::FlowRefinement => "target_ligand",
    }
}

fn atom_count_prior_mae(
    examples: &[MolecularExample],
    generation_target: &GenerationTargetConfig,
) -> f64 {
    if examples.is_empty() {
        return 0.0;
    }
    let total = examples
        .iter()
        .map(|example| {
            let target = ligand_atom_count(example) as f64;
            let predicted = match generation_target.generation_mode {
                GenerationModeConfig::PocketOnlyInitializationBaseline => {
                    generation_target.pocket_only_initialization.atom_count as f64
                }
                GenerationModeConfig::DeNovoInitialization => {
                    if let Some(atom_count) = generation_target
                        .de_novo_initialization
                        .dataset_calibrated_atom_count
                    {
                        atom_count as f64
                    } else {
                        let pocket_atoms = pocket_atom_count(example) as f64;
                        let divisor = generation_target
                            .de_novo_initialization
                            .pocket_atom_divisor
                            .max(1.0e-6);
                        let raw = (pocket_atoms / divisor).round() as usize;
                        raw.clamp(
                            generation_target.de_novo_initialization.min_atom_count,
                            generation_target.de_novo_initialization.max_atom_count,
                        ) as f64
                    }
                }
                GenerationModeConfig::TargetLigandDenoising
                | GenerationModeConfig::LigandRefinement
                | GenerationModeConfig::FlowRefinement => target,
            };
            (predicted - target).abs()
        })
        .sum::<f64>();
    total / examples.len() as f64
}

fn record_target_context_leakage_contract(
    validation: &mut DatasetValidationReport,
    examples: &[MolecularExample],
    generation_target: &GenerationTargetConfig,
) {
    let mode = generation_target.generation_mode;
    let target_ligand_context_dependency_detected = !examples.is_empty()
        && validation.pocket_coordinates_centered_upstream
        && validation.coordinate_frame_contract.contains("ligand_centered");
    let target_ligand_context_dependency_allowed =
        mode.uses_target_ligand_initialization() && mode != GenerationModeConfig::DeNovoInitialization;
    validation.target_ligand_context_dependency_detected =
        target_ligand_context_dependency_detected;
    validation.target_ligand_context_dependency_allowed =
        target_ligand_context_dependency_allowed;
    validation.generation_target_leakage_contract = if target_ligand_context_dependency_detected {
        format!(
            "generation_mode={} observes ligand-centered pocket/context tensors; this is allowed only for target-ligand refinement modes",
            mode.as_str()
        )
    } else {
        format!(
            "generation_mode={} has no detected target-ligand context dependency in retained examples",
            mode.as_str()
        )
    };
    validation.target_ligand_context_leakage_warnings.clear();
    if target_ligand_context_dependency_detected && !target_ligand_context_dependency_allowed {
        validation.target_ligand_context_leakage_warnings.push(format!(
            "generation_mode={} should not use target-ligand-centered pocket/context tensors for pocket-only or de novo claims",
            mode.as_str()
        ));
    }
}

fn enforce_target_context_leakage_policy(
    validation: &mut DatasetValidationReport,
    generation_target: &GenerationTargetConfig,
    filters: &DataQualityFilterConfig,
) -> Result<(), DataParseError> {
    if filters.reject_target_ligand_context_leakage
        && validation.target_ligand_context_dependency_detected
        && !validation.target_ligand_context_dependency_allowed
    {
        validation.target_ligand_context_dependency_rejected = true;
        return Err(DataParseError::Discovery {
            root: PathBuf::from("."),
            message: format!(
                "generation_mode={} rejects target-ligand-centered pocket/context tensors; use a pocket-only coordinate frame before claim-bearing inference",
                generation_target.generation_mode.as_str()
            ),
        });
    }
    validation.target_ligand_context_dependency_rejected = false;
    Ok(())
}
