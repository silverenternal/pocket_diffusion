//! Dataset traits and simple in-memory implementations.

use std::{collections::{BTreeMap, BTreeSet}, path::PathBuf};

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use super::{
    apply_affinity_labels, discover_pdbbind_like_entries, load_affinity_labels, load_manifest,
    load_manifest_entry, synthetic_phase1_examples, DataParseError, DatasetValidationReport,
    MolecularExample,
};
use crate::config::{DataConfig, DataQualityFilterConfig, DatasetFormat};

/// Common dataset contract for research experiments.
pub trait Dataset {
    /// Access type returned for each example.
    type Item;

    /// Number of examples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow an example by index.
    fn get(&self, index: usize) -> Option<&Self::Item>;
}

/// Simple owned dataset used for Phase 1 integration and tests.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDataset {
    examples: Vec<MolecularExample>,
}

/// Dataset plus load-time validation metadata.
#[derive(Debug, Clone)]
pub struct LoadedDataset {
    /// Parsed in-memory dataset used by the research stack.
    pub dataset: InMemoryDataset,
    /// Structured validation report for the load process.
    pub validation: DatasetValidationReport,
}

impl InMemoryDataset {
    /// Create a dataset from pre-built examples.
    pub fn new(examples: Vec<MolecularExample>) -> Self {
        Self { examples }
    }

    /// Borrow all examples in insertion order.
    pub fn examples(&self) -> &[MolecularExample] {
        &self.examples
    }

    /// Return a copy of the dataset with pocket features resized to the configured width.
    pub fn with_pocket_feature_dim(&self, pocket_feature_dim: i64) -> Self {
        Self::new(
            self.examples
                .iter()
                .map(|example| example.with_pocket_feature_dim(pocket_feature_dim))
                .collect(),
        )
    }

    /// Load examples according to the runtime data configuration.
    pub fn from_data_config(config: &DataConfig) -> Result<Self, DataParseError> {
        Ok(Self::load_from_config(config)?.dataset)
    }

    /// Load examples plus a structured validation report.
    pub fn load_from_config(config: &DataConfig) -> Result<LoadedDataset, DataParseError> {
        let mut validation = DatasetValidationReport::default();
        validation.parsing_mode = match config.parsing_mode {
            crate::config::ParsingMode::Lightweight => "lightweight".to_string(),
            crate::config::ParsingMode::Strict => "strict".to_string(),
        };

        let mut examples =
            match config.dataset_format {
                DatasetFormat::Synthetic => {
                    let examples = synthetic_phase1_examples();
                    validation.discovered_complexes = examples.len();
                    validation.parsed_examples = examples.len();
                    validation.parsed_ligands = examples.len();
                    validation.parsed_pockets = examples.len();
                    examples
                }
                DatasetFormat::ManifestJson => {
                    let manifest_path = config.manifest_path.as_deref().ok_or_else(|| {
                        DataParseError::Discovery {
                            root: config.root_dir.clone(),
                            message: "dataset_format=manifest_json requires manifest_path"
                                .to_string(),
                        }
                    })?;
                    let mut manifest = load_manifest(manifest_path)?;
                    validation.discovered_complexes = manifest.entries.len();
                    if let Some(label_table_path) = config.label_table_path.as_deref() {
                        let loaded_labels = load_affinity_labels(label_table_path)?;
                        validation.loaded_label_rows = loaded_labels.labels.len();
                        validation.label_table_rows_seen = loaded_labels.report.rows_seen;
                        validation.label_table_blank_rows = loaded_labels.report.blank_rows;
                        validation.label_table_comment_rows = loaded_labels.report.comment_rows;
                        validation.approximate_affinity_labels = loaded_labels.report.approximate_rows;
                        validation.loaded_label_measurement_family_histogram =
                            loaded_labels.report.measurement_family_histogram.clone();
                        validation.loaded_label_normalization_provenance_values =
                            loaded_labels.report.normalization_provenance_values.clone();
                        validation.affinity_normalization_warnings = loaded_labels
                            .labels
                            .iter()
                            .filter(|label| label.normalization_warning.is_some())
                            .count();
                        validation.normalization_warning_messages = loaded_labels
                            .labels
                            .iter()
                            .filter_map(|label| label.normalization_warning.clone())
                            .collect();
                        let label_report =
                            apply_affinity_labels(&mut manifest.entries, &loaded_labels.labels);
                        validation.example_id_label_matches = label_report.example_id_matches;
                        validation.protein_id_label_matches = label_report.protein_id_matches;
                        validation.duplicate_example_id_label_rows =
                            label_report.duplicate_example_id_rows;
                        validation.duplicate_protein_id_label_rows =
                            label_report.duplicate_protein_id_rows;
                        validation.unmatched_example_id_label_rows =
                            label_report.unmatched_example_id_rows;
                        validation.unmatched_protein_id_label_rows =
                            label_report.unmatched_protein_id_rows;
                    }
                    let mut examples = Vec::with_capacity(manifest.entries.len());
                    for entry in &manifest.entries {
                        let (example, parsed) = load_manifest_entry(
                            entry,
                            config.pocket_cutoff_angstrom,
                            config.parsing_mode,
                        )?;
                        validation.parsed_examples += 1;
                        validation.parsed_ligands += usize::from(parsed.parsed_ligand);
                        validation.parsed_pockets += usize::from(parsed.parsed_pocket);
                        validation.fallback_pocket_extractions +=
                            usize::from(parsed.used_pocket_fallback);
                        examples.push(example);
                    }
                    examples
                }
                DatasetFormat::PdbbindLikeDir => {
                    let mut entries =
                        discover_pdbbind_like_entries(&config.root_dir, config.parsing_mode)?;
                    validation.discovered_complexes = entries.len();
                    if let Some(label_table_path) = config.label_table_path.as_deref() {
                        let loaded_labels = load_affinity_labels(label_table_path)?;
                        validation.loaded_label_rows = loaded_labels.labels.len();
                        validation.label_table_rows_seen = loaded_labels.report.rows_seen;
                        validation.label_table_blank_rows = loaded_labels.report.blank_rows;
                        validation.label_table_comment_rows = loaded_labels.report.comment_rows;
                        validation.approximate_affinity_labels = loaded_labels.report.approximate_rows;
                        validation.loaded_label_measurement_family_histogram =
                            loaded_labels.report.measurement_family_histogram.clone();
                        validation.loaded_label_normalization_provenance_values =
                            loaded_labels.report.normalization_provenance_values.clone();
                        validation.affinity_normalization_warnings = loaded_labels
                            .labels
                            .iter()
                            .filter(|label| label.normalization_warning.is_some())
                            .count();
                        validation.normalization_warning_messages = loaded_labels
                            .labels
                            .iter()
                            .filter_map(|label| label.normalization_warning.clone())
                            .collect();
                        let label_report =
                            apply_affinity_labels(&mut entries, &loaded_labels.labels);
                        validation.example_id_label_matches = label_report.example_id_matches;
                        validation.protein_id_label_matches = label_report.protein_id_matches;
                        validation.duplicate_example_id_label_rows =
                            label_report.duplicate_example_id_rows;
                        validation.duplicate_protein_id_label_rows =
                            label_report.duplicate_protein_id_rows;
                        validation.unmatched_example_id_label_rows =
                            label_report.unmatched_example_id_rows;
                        validation.unmatched_protein_id_label_rows =
                            label_report.unmatched_protein_id_rows;
                    }
                    let mut examples = Vec::with_capacity(entries.len());
                    for entry in &entries {
                        let (example, parsed) = load_manifest_entry(
                            entry,
                            config.pocket_cutoff_angstrom,
                            config.parsing_mode,
                        )?;
                        validation.parsed_examples += 1;
                        validation.parsed_ligands += usize::from(parsed.parsed_ligand);
                        validation.parsed_pockets += usize::from(parsed.parsed_pocket);
                        validation.fallback_pocket_extractions +=
                            usize::from(parsed.used_pocket_fallback);
                        examples.push(example);
                    }
                    examples
                }
            };

        examples = examples
            .into_iter()
            .map(|example| example.with_generation_config(&config.generation_target))
            .collect();

        apply_quality_filters(&mut examples, &mut validation, &config.quality_filters)?;

        if let Some(limit) = config.max_examples {
            let original_len = examples.len();
            examples.truncate(limit);
            validation.truncated_examples = original_len.saturating_sub(examples.len());
        }
        finalize_validation_report(&mut validation, &examples);

        Ok(LoadedDataset {
            dataset: Self::new(examples),
            validation,
        })
    }

    /// Split by protein id to simulate unseen-pocket evaluation.
    pub fn split_by_protein(&self, val_every: usize, test_every: usize) -> DatasetSplits {
        let mut grouped: BTreeMap<&str, Vec<MolecularExample>> = BTreeMap::new();
        for example in &self.examples {
            grouped
                .entry(example.protein_id.as_str())
                .or_default()
                .push(example.clone());
        }

        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test = Vec::new();

        for (group_ix, (_, group_examples)) in grouped.into_iter().enumerate() {
            let target = if test_every > 0 && group_ix % test_every == 0 {
                &mut test
            } else if val_every > 0 && group_ix % val_every == 0 {
                &mut val
            } else {
                &mut train
            };
            target.extend(group_examples);
        }

        DatasetSplits {
            train: InMemoryDataset::new(train),
            val: InMemoryDataset::new(val),
            test: InMemoryDataset::new(test),
        }
    }

    /// Split by protein identity using configurable fractions and a deterministic seed.
    pub fn split_by_protein_fraction(
        &self,
        val_fraction: f32,
        test_fraction: f32,
        seed: u64,
    ) -> DatasetSplits {
        self.split_by_protein_fraction_with_options(val_fraction, test_fraction, seed, false)
    }

    /// Split by protein identity with optional stratification by dominant measurement family.
    pub fn split_by_protein_fraction_with_options(
        &self,
        val_fraction: f32,
        test_fraction: f32,
        seed: u64,
        stratify_by_measurement: bool,
    ) -> DatasetSplits {
        let mut grouped: BTreeMap<&str, Vec<MolecularExample>> = BTreeMap::new();
        for example in &self.examples {
            grouped
                .entry(example.protein_id.as_str())
                .or_default()
                .push(example.clone());
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test = Vec::new();

        if stratify_by_measurement {
            let mut buckets: BTreeMap<String, Vec<Vec<MolecularExample>>> = BTreeMap::new();
            for group in grouped.into_values() {
                let key = dominant_measurement_type(&group);
                buckets.entry(key).or_default().push(group);
            }

            let split = allocate_group_splits(
                buckets.values().map(Vec::len).sum(),
                val_fraction,
                test_fraction,
            );
            let mut interleaved_groups = Vec::new();
            for groups in buckets.values_mut() {
                groups.shuffle(&mut rng);
            }
            loop {
                let mut progressed = false;
                for groups in buckets.values_mut() {
                    if let Some(group) = groups.pop() {
                        interleaved_groups.push(group);
                        progressed = true;
                    }
                }
                if !progressed {
                    break;
                }
            }

            for (index, group) in interleaved_groups.into_iter().enumerate() {
                if index < split.test_groups {
                    test.extend(group);
                } else if index < split.test_groups + split.val_groups {
                    val.extend(group);
                } else {
                    train.extend(group);
                }
            }
        } else {
            let mut groups: Vec<Vec<MolecularExample>> = grouped.into_values().collect();
            groups.shuffle(&mut rng);
            let split = allocate_group_splits(groups.len(), val_fraction, test_fraction);
            for (index, group) in groups.into_iter().enumerate() {
                if index < split.test_groups {
                    test.extend(group);
                } else if index < split.test_groups + split.val_groups {
                    val.extend(group);
                } else {
                    train.extend(group);
                }
            }
        }

        DatasetSplits {
            train: InMemoryDataset::new(train),
            val: InMemoryDataset::new(val),
            test: InMemoryDataset::new(test),
        }
    }
}

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

#[derive(Debug, Clone, Copy)]
struct GroupSplitAllocation {
    val_groups: usize,
    test_groups: usize,
}

fn allocate_group_splits(
    total_groups: usize,
    val_fraction: f32,
    test_fraction: f32,
) -> GroupSplitAllocation {
    let val_groups = ((total_groups as f32) * val_fraction).round() as usize;
    let test_groups = ((total_groups as f32) * test_fraction).round() as usize;
    let test_groups = test_groups.min(total_groups);
    let val_groups = val_groups.min(total_groups.saturating_sub(test_groups));
    GroupSplitAllocation {
        val_groups,
        test_groups,
    }
}

fn dominant_measurement_type(group: &[MolecularExample]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for example in group {
        let measurement = example
            .targets
            .affinity_measurement_type
            .as_deref()
            .unwrap_or("unknown");
        *counts.entry(measurement).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(measurement, _)| measurement.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

impl Dataset for InMemoryDataset {
    type Item = MolecularExample;

    fn len(&self) -> usize {
        self.examples.len()
    }

    fn get(&self, index: usize) -> Option<&Self::Item> {
        self.examples.get(index)
    }
}

/// Precomputed train/validation/test partitions.
#[derive(Debug, Clone)]
pub struct DatasetSplits {
    /// Training set.
    pub train: InMemoryDataset,
    /// Validation set.
    pub val: InMemoryDataset,
    /// Test set.
    pub test: InMemoryDataset,
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::config::{DatasetFormat, ParsingMode};
    use crate::data::load_affinity_labels;

    #[test]
    fn strict_mode_rejects_nearest_atom_pocket_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1      50.000  50.000  50.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.parsing_mode = ParsingMode::Strict;
        config.pocket_cutoff_angstrom = 2.0;

        assert!(InMemoryDataset::load_from_config(&config).is_err());
    }

    #[test]
    fn label_loading_tracks_approximate_normalization_warnings() {
        let temp = tempfile::tempdir().unwrap();
        let labels_path = temp.path().join("labels.csv");
        fs::write(
            &labels_path,
            "example_id,measurement_type,raw_value,raw_unit\nex-1,IC50,1.2,uM\n",
        )
        .unwrap();

        let labels = load_affinity_labels(&labels_path).unwrap();
        assert_eq!(labels.labels.len(), 1);
        assert_eq!(labels.report.rows_seen, 1);
        assert!(labels.labels[0].is_approximate);
        assert!(labels.labels[0].normalization_warning.is_some());
        assert_eq!(
            labels.labels[0].normalization_provenance.as_deref(),
            Some("IC50_uM_to_delta_g_via_molar")
        );
    }

    #[test]
    fn optional_quality_filters_report_filtered_examples() {
        let mut config = DataConfig::default();
        config.quality_filters.require_source_structure_provenance = true;

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();

        assert_eq!(loaded.dataset.len(), 0);
        assert_eq!(
            loaded.validation.quality_filtered_missing_source_provenance,
            loaded.validation.discovered_complexes
        );
        assert_eq!(
            loaded.validation.quality_filtered_examples,
            loaded.validation.discovered_complexes
        );
    }

    #[test]
    fn quality_filters_can_reject_low_label_coverage() {
        let mut config = DataConfig::default();
        config.quality_filters.min_label_coverage = Some(0.5);

        let error = InMemoryDataset::load_from_config(&config).unwrap_err();

        assert!(error.to_string().contains("retained label coverage"));
    }

    #[test]
    fn quality_filters_can_reject_high_approximate_label_fraction() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");
        let labels_path = temp.path().join("labels.csv");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            &labels_path,
            "example_id,measurement_type,raw_value,raw_unit\nex-1,IC50,1.2,uM\n",
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.label_table_path = Some(labels_path);
        config.quality_filters.max_approximate_label_fraction = Some(0.0);

        let error = InMemoryDataset::load_from_config(&config).unwrap_err();
        assert!(error.to_string().contains("approximate-label fraction"));
    }

    #[test]
    fn dataset_validation_tracks_retained_metadata_contract() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");
        let labels_path = temp.path().join("labels.csv");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            &labels_path,
            "example_id,measurement_type,raw_value,raw_unit\nex-1,Kd,1.2,uM\n",
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.label_table_path = Some(labels_path);
        config.quality_filters.require_source_structure_provenance = true;
        config.quality_filters.require_affinity_metadata = true;
        config.quality_filters.min_normalization_provenance_coverage = Some(1.0);

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();

        assert_eq!(loaded.validation.attached_labels, 1);
        assert_eq!(loaded.validation.label_table_rows_seen, 1);
        assert_eq!(loaded.validation.retained_measurement_family_count, 1);
        assert_eq!(
            loaded
                .validation
                .retained_measurement_family_histogram
                .get("Kd"),
            Some(&1)
        );
        assert_eq!(loaded.validation.retained_approximate_affinity_labels, 0);
        assert_eq!(loaded.validation.retained_approximate_label_fraction, 0.0);
        assert_eq!(loaded.validation.unmatched_example_id_label_rows, 0);
        assert_eq!(loaded.validation.duplicate_example_id_label_rows, 0);
        assert_eq!(
            loaded.validation.retained_normalization_provenance_coverage,
            1.0
        );
        assert_eq!(loaded.validation.retained_missing_measurement_type, 0);
        assert_eq!(loaded.validation.retained_missing_normalization_provenance, 0);
        assert_eq!(loaded.validation.retained_source_provenance_coverage, 1.0);
    }

    #[test]
    fn dataset_validation_tracks_duplicate_and_unmatched_label_rows() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");
        let labels_path = temp.path().join("labels.csv");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            &labels_path,
            "example_id,protein_id,measurement_type,raw_value,raw_unit\nex-1,p-1,Kd,1.2,uM\nex-1,p-1,Ki,2.0,uM\nex-missing,p-missing,Kd,3.0,uM\n",
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.label_table_path = Some(labels_path);

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();

        assert_eq!(loaded.validation.loaded_label_rows, 3);
        assert_eq!(loaded.validation.duplicate_example_id_label_rows, 1);
        assert_eq!(loaded.validation.duplicate_protein_id_label_rows, 1);
        assert_eq!(loaded.validation.unmatched_example_id_label_rows, 1);
        assert_eq!(loaded.validation.unmatched_protein_id_label_rows, 1);
        assert_eq!(
            loaded
                .validation
                .loaded_label_measurement_family_histogram
                .get("Kd"),
            Some(&2)
        );
        assert_eq!(
            loaded
                .validation
                .loaded_label_measurement_family_histogram
                .get("Ki"),
            Some(&1)
        );
    }
}
