const UNAVAILABLE_METADATA: &str = "unavailable";

impl SplitReport {
    /// Build a split report from three dataset partitions.
    pub fn from_datasets(
        train: &InMemoryDataset,
        val: &InMemoryDataset,
        test: &InMemoryDataset,
    ) -> Self {
        Self::from_datasets_with_quality_filters(
            train,
            val,
            test,
            &crate::config::DataQualityFilterConfig::default(),
        )
    }

    /// Build a split report and apply optional claim-bearing split thresholds from data quality config.
    pub fn from_datasets_with_quality_filters(
        train: &InMemoryDataset,
        val: &InMemoryDataset,
        test: &InMemoryDataset,
        quality_filters: &crate::config::DataQualityFilterConfig,
    ) -> Self {
        let train_stats = SplitStats::from_dataset(train);
        let val_stats = SplitStats::from_dataset(val);
        let test_stats = SplitStats::from_dataset(test);

        let train_proteins = protein_set(train);
        let val_proteins = protein_set(val);
        let test_proteins = protein_set(test);
        let train_pockets = pocket_identity_proxy_set(train);
        let val_pockets = pocket_identity_proxy_set(val);
        let test_pockets = pocket_identity_proxy_set(test);
        let train_family_proxies = protein_family_proxy_set(train);
        let val_family_proxies = protein_family_proxy_set(val);
        let test_family_proxies = protein_family_proxy_set(test);
        let train_pocket_family_proxies = pocket_family_proxy_set(train);
        let val_pocket_family_proxies = pocket_family_proxy_set(val);
        let test_pocket_family_proxies = pocket_family_proxy_set(test);
        let train_ligand_scaffold_proxies = ligand_scaffold_proxy_set(train);
        let val_ligand_scaffold_proxies = ligand_scaffold_proxy_set(val);
        let test_ligand_scaffold_proxies = ligand_scaffold_proxy_set(test);

        let train_ids = example_id_set(train);
        let val_ids = example_id_set(val);
        let test_ids = example_id_set(test);

        let train_val_protein_overlap = train_proteins.intersection(&val_proteins).count();
        let train_test_protein_overlap = train_proteins.intersection(&test_proteins).count();
        let val_test_protein_overlap = val_proteins.intersection(&test_proteins).count();
        let train_val_pocket_overlap = train_pockets.intersection(&val_pockets).count();
        let train_test_pocket_overlap = train_pockets.intersection(&test_pockets).count();
        let val_test_pocket_overlap = val_pockets.intersection(&test_pockets).count();
        let train_val_protein_family_proxy_overlap =
            train_family_proxies.intersection(&val_family_proxies).count();
        let train_test_protein_family_proxy_overlap =
            train_family_proxies.intersection(&test_family_proxies).count();
        let val_test_protein_family_proxy_overlap =
            val_family_proxies.intersection(&test_family_proxies).count();
        let train_val_pocket_family_proxy_overlap = train_pocket_family_proxies
            .intersection(&val_pocket_family_proxies)
            .count();
        let train_test_pocket_family_proxy_overlap = train_pocket_family_proxies
            .intersection(&test_pocket_family_proxies)
            .count();
        let val_test_pocket_family_proxy_overlap = val_pocket_family_proxies
            .intersection(&test_pocket_family_proxies)
            .count();
        let train_val_ligand_scaffold_proxy_overlap = train_ligand_scaffold_proxies
            .intersection(&val_ligand_scaffold_proxies)
            .count();
        let train_test_ligand_scaffold_proxy_overlap = train_ligand_scaffold_proxies
            .intersection(&test_ligand_scaffold_proxies)
            .count();
        let val_test_ligand_scaffold_proxy_overlap = val_ligand_scaffold_proxies
            .intersection(&test_ligand_scaffold_proxies)
            .count();

        let duplicated_example_ids = train_ids.intersection(&val_ids).count()
            + train_ids.intersection(&test_ids).count()
            + val_ids.intersection(&test_ids).count();
        let quality_checks = SplitQualityChecks::from_stats_with_quality_filters(
            &train_stats,
            &val_stats,
            &test_stats,
            quality_filters,
        );

        Self {
            train: train_stats,
            val: val_stats,
            test: test_stats,
            leakage_checks: SplitLeakageChecks {
                protein_overlap_detected: train_val_protein_overlap > 0
                    || train_test_protein_overlap > 0
                    || val_test_protein_overlap > 0,
                pocket_overlap_detected: train_val_pocket_overlap > 0
                    || train_test_pocket_overlap > 0
                    || val_test_pocket_overlap > 0,
                protein_family_proxy_overlap_detected: train_val_protein_family_proxy_overlap > 0
                    || train_test_protein_family_proxy_overlap > 0
                    || val_test_protein_family_proxy_overlap > 0,
                pocket_family_proxy_overlap_detected: train_val_pocket_family_proxy_overlap > 0
                    || train_test_pocket_family_proxy_overlap > 0
                    || val_test_pocket_family_proxy_overlap > 0,
                ligand_scaffold_proxy_overlap_detected: train_val_ligand_scaffold_proxy_overlap > 0
                    || train_test_ligand_scaffold_proxy_overlap > 0
                    || val_test_ligand_scaffold_proxy_overlap > 0,
                duplicate_example_ids_detected: duplicated_example_ids > 0,
                train_val_protein_overlap,
                train_test_protein_overlap,
                val_test_protein_overlap,
                train_val_pocket_overlap,
                train_test_pocket_overlap,
                val_test_pocket_overlap,
                train_val_protein_family_proxy_overlap,
                train_test_protein_family_proxy_overlap,
                val_test_protein_family_proxy_overlap,
                train_val_pocket_family_proxy_overlap,
                train_test_pocket_family_proxy_overlap,
                val_test_pocket_family_proxy_overlap,
                train_val_ligand_scaffold_proxy_overlap,
                train_test_ligand_scaffold_proxy_overlap,
                val_test_ligand_scaffold_proxy_overlap,
                duplicated_example_ids,
            },
            quality_checks,
        }
    }

    /// Return an error if configured claim-bearing split thresholds failed.
    pub fn enforce_configured_quality_thresholds(&self) -> Result<(), String> {
        if self.quality_checks.threshold_failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "split quality thresholds failed: {}",
                self.quality_checks.threshold_failures.join("; ")
            ))
        }
    }
}

impl SplitQualityChecks {
    fn from_stats_with_quality_filters(
        train: &SplitStats,
        val: &SplitStats,
        test: &SplitStats,
        quality_filters: &crate::config::DataQualityFilterConfig,
    ) -> Self {
        const MIN_HELDOUT_FAMILIES: usize = 3;
        let weak_val_family_count = val.protein_family_proxy_histogram.len() < MIN_HELDOUT_FAMILIES;
        let weak_test_family_count =
            test.protein_family_proxy_histogram.len() < MIN_HELDOUT_FAMILIES;
        let severe_atom_count_skew_detected =
            severe_ratio_skew(train.average_ligand_atoms, val.average_ligand_atoms)
                || severe_ratio_skew(train.average_ligand_atoms, test.average_ligand_atoms)
                || severe_ratio_skew(train.average_pocket_atoms, val.average_pocket_atoms)
                || severe_ratio_skew(train.average_pocket_atoms, test.average_pocket_atoms);
        let measurement_family_skew_detected = !histogram_keys_cover(
            &train.dominant_measurement_histogram,
            &val.dominant_measurement_histogram,
        ) || !histogram_keys_cover(
            &train.dominant_measurement_histogram,
            &test.dominant_measurement_histogram,
        );
        let suspicious_distribution_collapse_detected =
            observed_histogram_collapsed(&val.pocket_family_proxy_histogram, val.example_count)
                || observed_histogram_collapsed(
                    &test.pocket_family_proxy_histogram,
                    test.example_count,
                )
                || observed_histogram_collapsed(
                    &val.ligand_scaffold_proxy_histogram,
                    val.example_count,
                )
                || observed_histogram_collapsed(
                    &test.ligand_scaffold_proxy_histogram,
                    test.example_count,
                )
                || histogram_collapsed(&val.pocket_atom_count_bins, val.example_count)
                || histogram_collapsed(&test.pocket_atom_count_bins, test.example_count);

        let mut warnings = Vec::new();
        if weak_val_family_count {
            warnings.push(format!(
                "validation split has {} proxy protein families; claim-bearing runs should have at least {MIN_HELDOUT_FAMILIES}",
                val.protein_family_proxy_histogram.len()
            ));
        }
        if weak_test_family_count {
            warnings.push(format!(
                "test split has {} proxy protein families; claim-bearing runs should have at least {MIN_HELDOUT_FAMILIES}",
                test.protein_family_proxy_histogram.len()
            ));
        }
        if severe_atom_count_skew_detected {
            warnings.push(
                "held-out ligand or pocket atom-count averages differ from train by more than 3x"
                    .to_string(),
            );
        }
        if measurement_family_skew_detected {
            warnings.push(
                "validation or test measurement-family labels are not covered by the train split"
                    .to_string(),
            );
        }
        if suspicious_distribution_collapse_detected {
            warnings.push(
                "held-out pocket-family, ligand-scaffold, or pocket-size distribution collapsed to a single observed bucket"
                    .to_string(),
            );
        }
        let metadata_unavailable_fields =
            metadata_unavailable_fields(train, val, test);
        for field in &metadata_unavailable_fields {
            warnings.push(format!("split metadata unavailable: {field}"));
        }
        let mut threshold_failures = Vec::new();
        push_min_threshold_failure(
            &mut threshold_failures,
            "validation protein-family count",
            val.protein_family_proxy_histogram.len(),
            quality_filters.min_validation_protein_families,
        );
        push_min_threshold_failure(
            &mut threshold_failures,
            "test protein-family count",
            test.protein_family_proxy_histogram.len(),
            quality_filters.min_test_protein_families,
        );
        push_min_threshold_failure(
            &mut threshold_failures,
            "validation pocket-family count",
            observed_histogram_key_count(&val.pocket_family_proxy_histogram),
            quality_filters.min_validation_pocket_families,
        );
        push_min_threshold_failure(
            &mut threshold_failures,
            "test pocket-family count",
            observed_histogram_key_count(&test.pocket_family_proxy_histogram),
            quality_filters.min_test_pocket_families,
        );
        push_min_threshold_failure(
            &mut threshold_failures,
            "validation ligand-scaffold count",
            observed_histogram_key_count(&val.ligand_scaffold_proxy_histogram),
            quality_filters.min_validation_ligand_scaffolds,
        );
        push_min_threshold_failure(
            &mut threshold_failures,
            "test ligand-scaffold count",
            observed_histogram_key_count(&test.ligand_scaffold_proxy_histogram),
            quality_filters.min_test_ligand_scaffolds,
        );
        push_min_threshold_failure(
            &mut threshold_failures,
            "validation measurement-family count",
            observed_histogram_key_count(&val.affinity_measurement_family_histogram),
            quality_filters.min_validation_measurement_families,
        );
        push_min_threshold_failure(
            &mut threshold_failures,
            "test measurement-family count",
            observed_histogram_key_count(&test.affinity_measurement_family_histogram),
            quality_filters.min_test_measurement_families,
        );
        for failure in &threshold_failures {
            warnings.push(format!("configured split threshold failed: {failure}"));
        }
        let configured_thresholds_enforced = quality_filters.min_validation_protein_families.is_some()
            || quality_filters.min_test_protein_families.is_some()
            || quality_filters.min_validation_pocket_families.is_some()
            || quality_filters.min_test_pocket_families.is_some()
            || quality_filters.min_validation_ligand_scaffolds.is_some()
            || quality_filters.min_test_ligand_scaffolds.is_some()
            || quality_filters.min_validation_measurement_families.is_some()
            || quality_filters.min_test_measurement_families.is_some();

        Self {
            weak_val_family_count,
            weak_test_family_count,
            severe_atom_count_skew_detected,
            measurement_family_skew_detected,
            suspicious_distribution_collapse_detected,
            metadata_unavailable_fields,
            configured_thresholds_enforced,
            min_validation_protein_families: quality_filters.min_validation_protein_families,
            min_test_protein_families: quality_filters.min_test_protein_families,
            min_validation_pocket_families: quality_filters.min_validation_pocket_families,
            min_test_pocket_families: quality_filters.min_test_pocket_families,
            min_validation_ligand_scaffolds: quality_filters.min_validation_ligand_scaffolds,
            min_test_ligand_scaffolds: quality_filters.min_test_ligand_scaffolds,
            min_validation_measurement_families: quality_filters
                .min_validation_measurement_families,
            min_test_measurement_families: quality_filters.min_test_measurement_families,
            threshold_failures,
            warnings,
        }
    }
}

fn push_min_threshold_failure(
    failures: &mut Vec<String>,
    label: &str,
    observed: usize,
    required: Option<usize>,
) {
    let Some(required) = required else {
        return;
    };
    if observed < required {
        failures.push(format!("{label} {observed} below required {required}"));
    }
}

impl SplitStats {
    fn from_dataset(dataset: &InMemoryDataset) -> Self {
        let example_count = dataset.examples().len();
        let unique_protein_count = protein_set(dataset).len();
        let labeled_example_count = dataset
            .examples()
            .iter()
            .filter(|example| example.targets.affinity_kcal_mol.is_some())
            .count();
        let labeled_fraction = if example_count == 0 {
            0.0
        } else {
            labeled_example_count as f64 / example_count as f64
        };
        let mut dominant_measurement_histogram = BTreeMap::new();
        let mut affinity_measurement_family_histogram = BTreeMap::new();
        let mut ligand_atom_count_bins = BTreeMap::new();
        let mut pocket_atom_count_bins = BTreeMap::new();
        let mut protein_family_proxy_histogram = BTreeMap::new();
        let mut pocket_family_proxy_histogram = BTreeMap::new();
        let mut ligand_scaffold_proxy_histogram = BTreeMap::new();
        let mut ligand_atoms_total = 0usize;
        let mut pocket_atoms_total = 0usize;
        let mut source_pocket_path_count = 0usize;
        let mut source_ligand_path_count = 0usize;
        let mut affinity_measurement_type_count = 0usize;
        let mut affinity_normalization_provenance_count = 0usize;
        for example in dataset.examples() {
            let measurement = example
                .targets
                .affinity_measurement_type
                .as_deref()
                .unwrap_or("unknown")
                .to_string();
            *dominant_measurement_histogram
                .entry(measurement)
                .or_default() += 1;
            let affinity_measurement = example
                .targets
                .affinity_measurement_type
                .as_deref()
                .unwrap_or(UNAVAILABLE_METADATA)
                .to_string();
            *affinity_measurement_family_histogram
                .entry(affinity_measurement)
                .or_default() += 1;
            let ligand_atoms = example
                .topology
                .atom_types
                .size()
                .first()
                .copied()
                .unwrap_or(0)
                .max(0) as usize;
            let pocket_atoms = example
                .pocket
                .coords
                .size()
                .first()
                .copied()
                .unwrap_or(0)
                .max(0) as usize;
            ligand_atoms_total += ligand_atoms;
            pocket_atoms_total += pocket_atoms;
            *ligand_atom_count_bins
                .entry(atom_count_bin(ligand_atoms))
                .or_default() += 1;
            *pocket_atom_count_bins
                .entry(atom_count_bin(pocket_atoms))
                .or_default() += 1;
            *protein_family_proxy_histogram
                .entry(protein_family_proxy(&example.protein_id))
                .or_default() += 1;
            *pocket_family_proxy_histogram
                .entry(pocket_family_proxy(example))
                .or_default() += 1;
            *ligand_scaffold_proxy_histogram
                .entry(ligand_scaffold_proxy(example))
                .or_default() += 1;
            if example.source_pocket_path.is_some() {
                source_pocket_path_count += 1;
            }
            if example.source_ligand_path.is_some() {
                source_ligand_path_count += 1;
            }
            if example.targets.affinity_measurement_type.is_some() {
                affinity_measurement_type_count += 1;
            }
            if example.targets.affinity_normalization_provenance.is_some() {
                affinity_normalization_provenance_count += 1;
            }
        }
        let average_ligand_atoms = if example_count == 0 {
            0.0
        } else {
            ligand_atoms_total as f64 / example_count as f64
        };
        let average_pocket_atoms = if example_count == 0 {
            0.0
        } else {
            pocket_atoms_total as f64 / example_count as f64
        };
        Self {
            example_count,
            unique_protein_count,
            labeled_example_count,
            labeled_fraction,
            dominant_measurement_histogram,
            affinity_measurement_family_histogram,
            ligand_atom_count_bins,
            pocket_atom_count_bins,
            protein_family_proxy_histogram,
            pocket_family_proxy_histogram,
            ligand_scaffold_proxy_histogram,
            metadata_availability: split_metadata_availability(
                example_count,
                source_pocket_path_count,
                source_ligand_path_count,
                affinity_measurement_type_count,
                affinity_normalization_provenance_count,
            ),
            average_ligand_atoms,
            average_pocket_atoms,
        }
    }
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

fn split_metadata_availability(
    example_count: usize,
    source_pocket_path_count: usize,
    source_ligand_path_count: usize,
    affinity_measurement_type_count: usize,
    affinity_normalization_provenance_count: usize,
) -> SplitMetadataAvailability {
    let mut unavailable_fields = Vec::new();
    if example_count > 0 && source_pocket_path_count < example_count {
        unavailable_fields.push("source_pocket_path".to_string());
    }
    if example_count > 0 && source_ligand_path_count < example_count {
        unavailable_fields.push("source_ligand_path".to_string());
    }
    if example_count > 0 && affinity_measurement_type_count < example_count {
        unavailable_fields.push("affinity_measurement_type".to_string());
    }
    if example_count > 0 && affinity_normalization_provenance_count < example_count {
        unavailable_fields.push("affinity_normalization_provenance".to_string());
    }
    SplitMetadataAvailability {
        example_count,
        source_pocket_path_count,
        source_ligand_path_count,
        affinity_measurement_type_count,
        affinity_normalization_provenance_count,
        source_pocket_path_fraction: fraction_count(source_pocket_path_count, example_count),
        source_ligand_path_fraction: fraction_count(source_ligand_path_count, example_count),
        affinity_measurement_type_fraction: fraction_count(
            affinity_measurement_type_count,
            example_count,
        ),
        affinity_normalization_provenance_fraction: fraction_count(
            affinity_normalization_provenance_count,
            example_count,
        ),
        unavailable_fields,
    }
}

fn fraction_count(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}

fn metadata_unavailable_fields(
    train: &SplitStats,
    val: &SplitStats,
    test: &SplitStats,
) -> Vec<String> {
    let mut fields = std::collections::BTreeSet::new();
    for (split, stats) in [("train", train), ("validation", val), ("test", test)] {
        for field in &stats.metadata_availability.unavailable_fields {
            fields.insert(format!("{split}.{field}"));
        }
    }
    fields.into_iter().collect()
}

fn severe_ratio_skew(reference: f64, candidate: f64) -> bool {
    if reference <= 0.0 || candidate <= 0.0 {
        return false;
    }
    let ratio = if reference > candidate {
        reference / candidate
    } else {
        candidate / reference
    };
    ratio > 3.0
}

fn observed_histogram_key_count(histogram: &BTreeMap<String, usize>) -> usize {
    histogram
        .keys()
        .filter(|key| observed_metadata_key(key))
        .count()
}

fn observed_metadata_key(key: &str) -> bool {
    key != UNAVAILABLE_METADATA && key != "unknown"
}

fn observed_histogram_collapsed(histogram: &BTreeMap<String, usize>, example_count: usize) -> bool {
    example_count >= 3 && observed_histogram_key_count(histogram) == 1
}

fn histogram_collapsed(histogram: &BTreeMap<String, usize>, example_count: usize) -> bool {
    example_count >= 3 && histogram.len() == 1
}

fn histogram_keys_cover(
    train: &BTreeMap<String, usize>,
    heldout: &BTreeMap<String, usize>,
) -> bool {
    heldout
        .keys()
        .filter(|key| key.as_str() != "unknown")
        .all(|key| train.contains_key(key))
}

fn protein_family_proxy(protein_id: &str) -> String {
    protein_id.split(|ch| ['_', '-', ':', '.'].contains(&ch))
        .next()
        .filter(|part| !part.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

fn protein_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<&str> {
    dataset
        .examples()
        .iter()
        .map(|example| example.protein_id.as_str())
        .collect()
}

fn protein_family_proxy_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<String> {
    dataset
        .examples()
        .iter()
        .map(|example| protein_family_proxy(&example.protein_id))
        .collect()
}

fn pocket_family_proxy_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<String> {
    dataset
        .examples()
        .iter()
        .map(pocket_family_proxy)
        .filter(|proxy| observed_metadata_key(proxy))
        .collect()
}

fn ligand_scaffold_proxy_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<String> {
    dataset
        .examples()
        .iter()
        .map(ligand_scaffold_proxy)
        .filter(|proxy| observed_metadata_key(proxy))
        .collect()
}

fn pocket_identity_proxy_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<String> {
    dataset
        .examples()
        .iter()
        .map(pocket_identity_proxy)
        .collect()
}

fn pocket_family_proxy(example: &crate::data::MolecularExample) -> String {
    example
        .source_pocket_path
        .as_deref()
        .map(source_path_family_proxy)
        .unwrap_or_else(|| UNAVAILABLE_METADATA.to_string())
}

fn ligand_scaffold_proxy(example: &crate::data::MolecularExample) -> String {
    example
        .source_ligand_path
        .as_deref()
        .map(source_path_family_proxy)
        .unwrap_or_else(|| UNAVAILABLE_METADATA.to_string())
}

fn source_path_family_proxy(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(family_token)
        .filter(|token| !token.is_empty())
        .unwrap_or_else(|| UNAVAILABLE_METADATA.to_string())
}

fn family_token(label: &str) -> String {
    label
        .split(|ch| ['_', '-', ':', '.'].contains(&ch))
        .find(|part| !part.is_empty())
        .unwrap_or(label)
        .to_ascii_lowercase()
}

fn pocket_identity_proxy(example: &crate::data::MolecularExample) -> String {
    if let Some(path) = &example.source_pocket_path {
        return path.to_string_lossy().replace('\\', "/");
    }
    let pocket_atoms = example
        .pocket
        .coords
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0);
    let centroid = if example.pocket.coords.numel() == 0 {
        [0.0_f64; 3]
    } else {
        let centroid = example
            .pocket
            .coords
            .mean_dim([0].as_slice(), false, tch::Kind::Float);
        [
            centroid.double_value(&[0]),
            centroid.double_value(&[1]),
            centroid.double_value(&[2]),
        ]
    };
    format!(
        "{}|n={}|c={:.1},{:.1},{:.1}",
        example.protein_id, pocket_atoms, centroid[0], centroid[1], centroid[2]
    )
}

fn example_id_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<&str> {
    dataset
        .examples()
        .iter()
        .map(|example| example.example_id.as_str())
        .collect()
}

#[cfg(test)]
mod split_quality_tests {
    use super::*;

    fn annotated_example(
        mut example: crate::data::MolecularExample,
        index: usize,
        protein_family: &str,
        pocket_family: &str,
        ligand_scaffold: &str,
        measurement: &str,
    ) -> crate::data::MolecularExample {
        example.example_id = format!("{protein_family}-{pocket_family}-{ligand_scaffold}-{index}");
        example.protein_id = format!("{protein_family}:protein-{index}");
        example.source_pocket_path = Some(std::path::PathBuf::from(format!(
            "pockets/{pocket_family}-pocket-{index}.pdb"
        )));
        example.source_ligand_path = Some(std::path::PathBuf::from(format!(
            "ligands/{ligand_scaffold}-ligand-{index}.sdf"
        )));
        example.targets.affinity_kcal_mol = Some(-7.0 + index as f32 * 0.1);
        example.targets.affinity_measurement_type = Some(measurement.to_string());
        example.targets.affinity_normalization_provenance = Some("fixture".to_string());
        example
    }

    #[test]
    fn split_quality_thresholds_can_block_claim_bearing_heldout_surfaces() {
        let mut examples = crate::data::synthetic_phase1_examples();
        for (index, example) in examples.iter_mut().enumerate() {
            example.protein_id = format!("fam{}:protein{}", index % 4, index);
            example.source_pocket_path = Some(std::path::PathBuf::from(format!(
                "pockets/pocketfam{index}-pocket.pdb"
            )));
            example.source_ligand_path = Some(std::path::PathBuf::from(format!(
                "ligands/scaffold{index}-ligand.sdf"
            )));
            example.targets.affinity_measurement_type = Some(if index % 2 == 0 {
                "Kd".to_string()
            } else {
                "IC50".to_string()
            });
            example.targets.affinity_normalization_provenance = Some("fixture".to_string());
        }
        let train = InMemoryDataset::new(vec![examples[0].clone(), examples[1].clone()]);
        let val = InMemoryDataset::new(vec![examples[2].clone()]);
        let test = InMemoryDataset::new(vec![examples[3].clone()]);
        let quality_filters = crate::config::DataQualityFilterConfig {
            min_validation_protein_families: Some(2),
            min_test_protein_families: Some(2),
            min_validation_pocket_families: Some(2),
            min_test_pocket_families: Some(2),
            min_validation_ligand_scaffolds: Some(2),
            min_test_ligand_scaffolds: Some(2),
            min_validation_measurement_families: Some(2),
            min_test_measurement_families: Some(2),
            ..crate::config::DataQualityFilterConfig::default()
        };

        let report =
            SplitReport::from_datasets_with_quality_filters(&train, &val, &test, &quality_filters);

        assert!(report.quality_checks.configured_thresholds_enforced);
        assert_eq!(report.quality_checks.threshold_failures.len(), 8);
        assert!(report.enforce_configured_quality_thresholds().is_err());
        assert!(report
            .quality_checks
            .threshold_failures
            .iter()
            .any(|failure| failure.contains("pocket-family")));
        assert!(report
            .quality_checks
            .threshold_failures
            .iter()
            .any(|failure| failure.contains("ligand-scaffold")));
        assert!(report
            .quality_checks
            .warnings
            .iter()
            .any(|warning| warning.contains("configured split threshold failed")));
    }

    #[test]
    fn split_leakage_checks_distinguish_protein_pocket_and_family_proxy_overlap() {
        let mut examples = crate::data::synthetic_phase1_examples();
        for (index, example) in examples.iter_mut().enumerate() {
            example.example_id = format!("example-{index}");
            example.protein_id = format!("familyA:protein-{index}");
            example.source_pocket_path = Some(std::path::PathBuf::from(format!(
                "pockets/shared-pocket-{}",
                index % 2
            )));
            example.source_ligand_path = Some(std::path::PathBuf::from(format!(
                "ligands/shared-scaffold-{}",
                index % 2
            )));
        }
        examples[1].protein_id = examples[0].protein_id.clone();
        examples[2].source_pocket_path = examples[0].source_pocket_path.clone();
        examples[2].source_ligand_path = examples[0].source_ligand_path.clone();
        examples[3].protein_id = "familyB:protein-3".to_string();

        let train = InMemoryDataset::new(vec![examples[0].clone()]);
        let val = InMemoryDataset::new(vec![examples[1].clone(), examples[2].clone()]);
        let test = InMemoryDataset::new(vec![examples[3].clone()]);

        let report = SplitReport::from_datasets(&train, &val, &test);

        assert!(report.leakage_checks.protein_overlap_detected);
        assert!(report.leakage_checks.pocket_overlap_detected);
        assert!(report
            .leakage_checks
            .protein_family_proxy_overlap_detected);
        assert!(report
            .leakage_checks
            .pocket_family_proxy_overlap_detected);
        assert!(report
            .leakage_checks
            .ligand_scaffold_proxy_overlap_detected);
        assert_eq!(report.leakage_checks.train_val_protein_overlap, 1);
        assert_eq!(report.leakage_checks.train_val_pocket_overlap, 1);
        assert_eq!(
            report
                .leakage_checks
                .train_val_protein_family_proxy_overlap,
            1
        );
        assert_eq!(
            report
                .leakage_checks
                .train_val_pocket_family_proxy_overlap,
            1
        );
        assert_eq!(
            report
                .leakage_checks
                .train_val_ligand_scaffold_proxy_overlap,
            1
        );
        assert_eq!(report.val.labeled_fraction, 0.0);
    }

    #[test]
    fn split_leakage_checks_pass_for_disjoint_metadata_proxies() {
        let examples = crate::data::synthetic_phase1_examples()
            .into_iter()
            .enumerate()
            .map(|(index, example)| {
                annotated_example(
                    example,
                    index,
                    &format!("fam{index}"),
                    &format!("pocketfam{index}"),
                    &format!("scaffold{index}"),
                    if index % 2 == 0 { "Kd" } else { "Ki" },
                )
            })
            .collect::<Vec<_>>();

        let train = InMemoryDataset::new(vec![examples[0].clone(), examples[1].clone()]);
        let val = InMemoryDataset::new(vec![examples[2].clone()]);
        let test = InMemoryDataset::new(vec![examples[3].clone()]);

        let report = SplitReport::from_datasets(&train, &val, &test);

        assert!(!report.leakage_checks.protein_overlap_detected);
        assert!(!report.leakage_checks.pocket_overlap_detected);
        assert!(!report
            .leakage_checks
            .protein_family_proxy_overlap_detected);
        assert!(!report
            .leakage_checks
            .pocket_family_proxy_overlap_detected);
        assert!(!report
            .leakage_checks
            .ligand_scaffold_proxy_overlap_detected);
        assert!(report.val.metadata_availability.unavailable_fields.is_empty());
        assert_eq!(report.val.pocket_family_proxy_histogram.len(), 1);
        assert_eq!(report.val.ligand_scaffold_proxy_histogram.len(), 1);
    }

    #[test]
    fn split_report_marks_unavailable_optional_metadata_without_overlap_claims() {
        let mut examples = crate::data::synthetic_phase1_examples();
        for (index, example) in examples.iter_mut().enumerate() {
            example.example_id = format!("missing-metadata-{index}");
            example.protein_id = format!("missingFam{index}:protein");
            example.source_pocket_path = None;
            example.source_ligand_path = None;
            example.targets.affinity_measurement_type = None;
            example.targets.affinity_normalization_provenance = None;
        }
        let train = InMemoryDataset::new(vec![examples[0].clone()]);
        let val = InMemoryDataset::new(vec![examples[1].clone()]);
        let test = InMemoryDataset::new(vec![examples[2].clone()]);

        let report = SplitReport::from_datasets(&train, &val, &test);

        assert_eq!(
            report.val.pocket_family_proxy_histogram.get(UNAVAILABLE_METADATA),
            Some(&1)
        );
        assert_eq!(
            report
                .val
                .ligand_scaffold_proxy_histogram
                .get(UNAVAILABLE_METADATA),
            Some(&1)
        );
        assert!(!report
            .leakage_checks
            .pocket_family_proxy_overlap_detected);
        assert!(!report
            .leakage_checks
            .ligand_scaffold_proxy_overlap_detected);
        assert!(report
            .val
            .metadata_availability
            .unavailable_fields
            .contains(&"source_ligand_path".to_string()));
        assert!(report
            .quality_checks
            .metadata_unavailable_fields
            .iter()
            .any(|field| field == "validation.source_ligand_path"));
    }

    #[test]
    fn split_quality_flags_suspicious_distribution_collapse() {
        let template = crate::data::synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap();
        let train = InMemoryDataset::new(
            (0..3)
                .map(|index| {
                    annotated_example(
                        template.clone(),
                        index,
                        &format!("trainfam{index}"),
                        &format!("trainpocket{index}"),
                        &format!("trainscaffold{index}"),
                        "Kd",
                    )
                })
                .collect(),
        );
        let val = InMemoryDataset::new(
            (3..6)
                .map(|index| {
                    annotated_example(
                        template.clone(),
                        index,
                        &format!("valfam{index}"),
                        "collapsedpocket",
                        "collapsedscaffold",
                        "Kd",
                    )
                })
                .collect(),
        );
        let test = InMemoryDataset::new(
            (6..9)
                .map(|index| {
                    annotated_example(
                        template.clone(),
                        index,
                        &format!("testfam{index}"),
                        "collapsedpocket",
                        "collapsedscaffold",
                        "Ki",
                    )
                })
                .collect(),
        );

        let report = SplitReport::from_datasets(&train, &val, &test);

        assert!(report
            .quality_checks
            .suspicious_distribution_collapse_detected);
        assert!(report
            .quality_checks
            .warnings
            .iter()
            .any(|warning| warning.contains("distribution collapsed")));
    }
}
