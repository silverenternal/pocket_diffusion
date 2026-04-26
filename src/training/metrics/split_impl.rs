impl SplitReport {
    /// Build a split report from three dataset partitions.
    pub fn from_datasets(
        train: &InMemoryDataset,
        val: &InMemoryDataset,
        test: &InMemoryDataset,
    ) -> Self {
        let train_stats = SplitStats::from_dataset(train);
        let val_stats = SplitStats::from_dataset(val);
        let test_stats = SplitStats::from_dataset(test);

        let train_proteins = protein_set(train);
        let val_proteins = protein_set(val);
        let test_proteins = protein_set(test);

        let train_ids = example_id_set(train);
        let val_ids = example_id_set(val);
        let test_ids = example_id_set(test);

        let train_val_protein_overlap = train_proteins.intersection(&val_proteins).count();
        let train_test_protein_overlap = train_proteins.intersection(&test_proteins).count();
        let val_test_protein_overlap = val_proteins.intersection(&test_proteins).count();

        let duplicated_example_ids = train_ids.intersection(&val_ids).count()
            + train_ids.intersection(&test_ids).count()
            + val_ids.intersection(&test_ids).count();
        let quality_checks = SplitQualityChecks::from_stats(&train_stats, &val_stats, &test_stats);

        Self {
            train: train_stats,
            val: val_stats,
            test: test_stats,
            leakage_checks: SplitLeakageChecks {
                protein_overlap_detected: train_val_protein_overlap > 0
                    || train_test_protein_overlap > 0
                    || val_test_protein_overlap > 0,
                duplicate_example_ids_detected: duplicated_example_ids > 0,
                train_val_protein_overlap,
                train_test_protein_overlap,
                val_test_protein_overlap,
                duplicated_example_ids,
            },
            quality_checks,
        }
    }
}

impl SplitQualityChecks {
    fn from_stats(train: &SplitStats, val: &SplitStats, test: &SplitStats) -> Self {
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

        Self {
            weak_val_family_count,
            weak_test_family_count,
            severe_atom_count_skew_detected,
            measurement_family_skew_detected,
            warnings,
        }
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
        let mut ligand_atom_count_bins = BTreeMap::new();
        let mut pocket_atom_count_bins = BTreeMap::new();
        let mut protein_family_proxy_histogram = BTreeMap::new();
        let mut ligand_atoms_total = 0usize;
        let mut pocket_atoms_total = 0usize;
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
            ligand_atom_count_bins,
            pocket_atom_count_bins,
            protein_family_proxy_histogram,
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
    protein_id
        .split(|ch: char| ch == '_' || ch == '-' || ch == ':' || ch == '.')
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

fn example_id_set(dataset: &InMemoryDataset) -> std::collections::BTreeSet<&str> {
    dataset
        .examples()
        .iter()
        .map(|example| example.example_id.as_str())
        .collect()
}
