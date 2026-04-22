//! Dataset traits and simple in-memory implementations.

use std::collections::BTreeMap;

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use super::{
    apply_affinity_labels, discover_pdbbind_like_entries, load_affinity_labels, load_manifest,
    load_manifest_entry, synthetic_phase1_examples, DataParseError, MolecularExample,
};
use crate::config::{DataConfig, DatasetFormat};

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
        let mut examples =
            match config.dataset_format {
                DatasetFormat::Synthetic => synthetic_phase1_examples(),
                DatasetFormat::ManifestJson => {
                    let manifest_path = config.manifest_path.as_deref().ok_or_else(|| {
                        DataParseError::Discovery {
                            root: config.root_dir.clone(),
                            message: "dataset_format=manifest_json requires manifest_path"
                                .to_string(),
                        }
                    })?;
                    let mut manifest = load_manifest(manifest_path)?;
                    if let Some(label_table_path) = config.label_table_path.as_deref() {
                        let labels = load_affinity_labels(label_table_path)?;
                        apply_affinity_labels(&mut manifest.entries, &labels);
                    }
                    manifest
                        .entries
                        .iter()
                        .map(|entry| load_manifest_entry(entry, config.pocket_cutoff_angstrom))
                        .collect::<Result<Vec<_>, _>>()?
                }
                DatasetFormat::PdbbindLikeDir => {
                    let mut entries = discover_pdbbind_like_entries(&config.root_dir)?;
                    if let Some(label_table_path) = config.label_table_path.as_deref() {
                        let labels = load_affinity_labels(label_table_path)?;
                        apply_affinity_labels(&mut entries, &labels);
                    }
                    entries
                        .iter()
                        .map(|entry| load_manifest_entry(entry, config.pocket_cutoff_angstrom))
                        .collect::<Result<Vec<_>, _>>()?
                }
            };

        if let Some(limit) = config.max_examples {
            examples.truncate(limit);
        }

        Ok(Self::new(examples))
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
