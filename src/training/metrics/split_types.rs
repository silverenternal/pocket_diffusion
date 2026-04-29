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
    /// Histogram of affinity measurement families, with unavailable metadata explicit.
    #[serde(default)]
    pub affinity_measurement_family_histogram: BTreeMap<String, usize>,
    /// Histogram of ligand atom-count bins in the split.
    #[serde(default)]
    pub ligand_atom_count_bins: BTreeMap<String, usize>,
    /// Histogram of pocket atom-count bins in the split.
    #[serde(default)]
    pub pocket_atom_count_bins: BTreeMap<String, usize>,
    /// Lightweight protein-family proxy histogram derived from stable protein id prefixes.
    #[serde(default)]
    pub protein_family_proxy_histogram: BTreeMap<String, usize>,
    /// Lightweight pocket-family proxy histogram derived from source pocket paths when available.
    #[serde(default)]
    pub pocket_family_proxy_histogram: BTreeMap<String, usize>,
    /// Lightweight ligand-scaffold proxy histogram derived from source ligand paths when available.
    #[serde(default)]
    pub ligand_scaffold_proxy_histogram: BTreeMap<String, usize>,
    /// Metadata availability summary so synthetic or legacy datasets do not masquerade as measured.
    #[serde(default)]
    pub metadata_availability: SplitMetadataAvailability,
    /// Average ligand atom count in the split.
    #[serde(default)]
    pub average_ligand_atoms: f64,
    /// Average pocket atom count in the split.
    #[serde(default)]
    pub average_pocket_atoms: f64,
}

/// Per-split availability of optional metadata used by stronger split-quality diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitMetadataAvailability {
    /// Number of examples in the split.
    pub example_count: usize,
    /// Examples retaining a source pocket path.
    pub source_pocket_path_count: usize,
    /// Examples retaining a source ligand path.
    pub source_ligand_path_count: usize,
    /// Labeled examples with measurement-family metadata.
    pub affinity_measurement_type_count: usize,
    /// Labeled examples with normalization-provenance metadata.
    pub affinity_normalization_provenance_count: usize,
    /// Fraction of examples retaining a source pocket path.
    pub source_pocket_path_fraction: f64,
    /// Fraction of examples retaining a source ligand path.
    pub source_ligand_path_fraction: f64,
    /// Fraction of examples with measurement-family metadata.
    pub affinity_measurement_type_fraction: f64,
    /// Fraction of examples with normalization-provenance metadata.
    pub affinity_normalization_provenance_fraction: f64,
    /// Fields unavailable for at least one example in this split.
    pub unavailable_fields: Vec<String>,
}

/// Explicit leakage checks across train/val/test partitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitLeakageChecks {
    /// Whether protein ids overlap across splits.
    pub protein_overlap_detected: bool,
    /// Whether pocket identity proxies overlap across splits.
    #[serde(default)]
    pub pocket_overlap_detected: bool,
    /// Whether protein-family/sequence proxy ids overlap across splits.
    #[serde(default)]
    pub protein_family_proxy_overlap_detected: bool,
    /// Whether pocket-family proxy ids overlap across splits.
    #[serde(default)]
    pub pocket_family_proxy_overlap_detected: bool,
    /// Whether ligand-scaffold proxy ids overlap across splits.
    #[serde(default)]
    pub ligand_scaffold_proxy_overlap_detected: bool,
    /// Whether example ids overlap across splits.
    pub duplicate_example_ids_detected: bool,
    /// Number of train/val protein overlaps.
    pub train_val_protein_overlap: usize,
    /// Number of train/test protein overlaps.
    pub train_test_protein_overlap: usize,
    /// Number of val/test protein overlaps.
    pub val_test_protein_overlap: usize,
    /// Number of train/validation pocket proxy overlaps.
    #[serde(default)]
    pub train_val_pocket_overlap: usize,
    /// Number of train/test pocket proxy overlaps.
    #[serde(default)]
    pub train_test_pocket_overlap: usize,
    /// Number of validation/test pocket proxy overlaps.
    #[serde(default)]
    pub val_test_pocket_overlap: usize,
    /// Number of train/validation protein-family proxy overlaps.
    #[serde(default)]
    pub train_val_protein_family_proxy_overlap: usize,
    /// Number of train/test protein-family proxy overlaps.
    #[serde(default)]
    pub train_test_protein_family_proxy_overlap: usize,
    /// Number of validation/test protein-family proxy overlaps.
    #[serde(default)]
    pub val_test_protein_family_proxy_overlap: usize,
    /// Number of train/validation pocket-family proxy overlaps.
    #[serde(default)]
    pub train_val_pocket_family_proxy_overlap: usize,
    /// Number of train/test pocket-family proxy overlaps.
    #[serde(default)]
    pub train_test_pocket_family_proxy_overlap: usize,
    /// Number of validation/test pocket-family proxy overlaps.
    #[serde(default)]
    pub val_test_pocket_family_proxy_overlap: usize,
    /// Number of train/validation ligand-scaffold proxy overlaps.
    #[serde(default)]
    pub train_val_ligand_scaffold_proxy_overlap: usize,
    /// Number of train/test ligand-scaffold proxy overlaps.
    #[serde(default)]
    pub train_test_ligand_scaffold_proxy_overlap: usize,
    /// Number of validation/test ligand-scaffold proxy overlaps.
    #[serde(default)]
    pub val_test_ligand_scaffold_proxy_overlap: usize,
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
    /// Whether held-out proxy distributions collapse to a single observed bucket.
    #[serde(default)]
    pub suspicious_distribution_collapse_detected: bool,
    /// Metadata fields unavailable in at least one split.
    #[serde(default)]
    pub metadata_unavailable_fields: Vec<String>,
    /// Whether configured claim-bearing split thresholds were evaluated.
    #[serde(default)]
    pub configured_thresholds_enforced: bool,
    /// Minimum validation protein-family threshold, when configured.
    #[serde(default)]
    pub min_validation_protein_families: Option<usize>,
    /// Minimum test protein-family threshold, when configured.
    #[serde(default)]
    pub min_test_protein_families: Option<usize>,
    /// Minimum validation pocket-family threshold, when configured.
    #[serde(default)]
    pub min_validation_pocket_families: Option<usize>,
    /// Minimum test pocket-family threshold, when configured.
    #[serde(default)]
    pub min_test_pocket_families: Option<usize>,
    /// Minimum validation ligand-scaffold threshold, when configured.
    #[serde(default)]
    pub min_validation_ligand_scaffolds: Option<usize>,
    /// Minimum test ligand-scaffold threshold, when configured.
    #[serde(default)]
    pub min_test_ligand_scaffolds: Option<usize>,
    /// Minimum validation measurement-family threshold, when configured.
    #[serde(default)]
    pub min_validation_measurement_families: Option<usize>,
    /// Minimum test measurement-family threshold, when configured.
    #[serde(default)]
    pub min_test_measurement_families: Option<usize>,
    /// Concrete threshold failures that should block claim-bearing configured runs.
    #[serde(default)]
    pub threshold_failures: Vec<String>,
    /// Human-readable warnings suitable for persisted audit artifacts.
    pub warnings: Vec<String>,
}
