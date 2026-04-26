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

