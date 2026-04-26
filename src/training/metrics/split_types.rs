/// Dataset split sizes used by a training or experiment run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplitSizes {
    /// Number of examples before splitting.
    pub total: usize,
    /// Number of training examples.
    pub train: usize,
    /// Number of validation examples.
    pub val: usize,
    /// Number of test examples.
    pub test: usize,
}

/// Split-level audit artifact for unseen-pocket experiments and training runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitReport {
    /// Training split audit.
    pub train: SplitStats,
    /// Validation split audit.
    pub val: SplitStats,
    /// Test split audit.
    pub test: SplitStats,
    /// Cross-split leakage checks.
    pub leakage_checks: SplitLeakageChecks,
    /// Conservative split-quality warnings for claim-bearing unseen-pocket runs.
    #[serde(default)]
    pub quality_checks: SplitQualityChecks,
}

/// Per-split statistics needed to audit unseen-pocket claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitStats {
    /// Number of examples in the split.
    pub example_count: usize,
    /// Number of unique protein ids in the split.
    pub unique_protein_count: usize,
    /// Number of labeled examples in the split.
    pub labeled_example_count: usize,
    /// Fraction of examples in the split with labels.
    pub labeled_fraction: f64,
    /// Histogram of dominant measurement families in the split.
    pub dominant_measurement_histogram: BTreeMap<String, usize>,
    /// Histogram of ligand atom-count bins in the split.
    #[serde(default)]
    pub ligand_atom_count_bins: BTreeMap<String, usize>,
    /// Histogram of pocket atom-count bins in the split.
    #[serde(default)]
    pub pocket_atom_count_bins: BTreeMap<String, usize>,
    /// Lightweight protein-family proxy histogram derived from stable protein id prefixes.
    #[serde(default)]
    pub protein_family_proxy_histogram: BTreeMap<String, usize>,
    /// Average ligand atom count in the split.
    #[serde(default)]
    pub average_ligand_atoms: f64,
    /// Average pocket atom count in the split.
    #[serde(default)]
    pub average_pocket_atoms: f64,
}

/// Explicit leakage checks across train/val/test partitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitLeakageChecks {
    /// Whether protein ids overlap across splits.
    pub protein_overlap_detected: bool,
    /// Whether example ids overlap across splits.
    pub duplicate_example_ids_detected: bool,
    /// Number of train/val protein overlaps.
    pub train_val_protein_overlap: usize,
    /// Number of train/test protein overlaps.
    pub train_test_protein_overlap: usize,
    /// Number of val/test protein overlaps.
    pub val_test_protein_overlap: usize,
    /// Number of duplicated example ids across all splits.
    pub duplicated_example_ids: usize,
}

/// Conservative split-quality checks that make weak held-out surfaces explicit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitQualityChecks {
    /// Whether validation has too few proxy protein families for claim-bearing use.
    pub weak_val_family_count: bool,
    /// Whether test has too few proxy protein families for claim-bearing use.
    pub weak_test_family_count: bool,
    /// Whether atom-count distributions are severely skewed against train.
    pub severe_atom_count_skew_detected: bool,
    /// Whether measurement-family coverage differs substantially across splits.
    pub measurement_family_skew_detected: bool,
    /// Human-readable warnings suitable for persisted audit artifacts.
    pub warnings: Vec<String>,
}
