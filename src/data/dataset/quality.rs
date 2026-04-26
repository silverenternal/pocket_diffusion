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
    }
}

fn finalize_validation_report(
    validation: &mut DatasetValidationReport,
    examples: &[MolecularExample],
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
}

