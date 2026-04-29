/// Errors raised while converting on-disk assets into research examples.
#[derive(Debug, Error)]
pub enum DataParseError {
    #[error("I/O error while reading dataset assets: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid PDB record in {path}: {message}")]
    InvalidPdb { path: PathBuf, message: String },
    #[error("invalid SDF record in {path}: {message}")]
    InvalidSdf { path: PathBuf, message: String },
    #[error("dataset discovery error under {root}: {message}")]
    Discovery { root: PathBuf, message: String },
    #[error("invalid label table at {path}: {message}")]
    InvalidLabelTable { path: PathBuf, message: String },
}

/// Manifest describing a dataset split-agnostic collection of complexes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Entries loaded into the research stack.
    pub entries: Vec<ManifestEntry>,
}

/// One protein-ligand complex entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Stable example identifier.
    pub example_id: String,
    /// Protein identifier used for unseen-pocket splits.
    pub protein_id: String,
    /// Path to the protein pocket source PDB file.
    pub pocket_path: PathBuf,
    /// Path to the ligand structure source SDF file.
    pub ligand_path: PathBuf,
    /// Optional affinity label in kcal/mol.
    #[serde(default)]
    pub affinity_kcal_mol: Option<f32>,
    /// Optional original measurement type before normalization.
    #[serde(default)]
    pub affinity_measurement_type: Option<String>,
    /// Optional original numeric value before normalization.
    #[serde(default)]
    pub affinity_raw_value: Option<f32>,
    /// Optional original unit before normalization.
    #[serde(default)]
    pub affinity_raw_unit: Option<String>,
    /// Optional normalization provenance for the attached affinity target.
    #[serde(default)]
    pub affinity_normalization_provenance: Option<String>,
    /// Whether the normalized target is only approximate.
    #[serde(default)]
    pub affinity_is_approximate: bool,
    /// Optional warning emitted during normalization.
    #[serde(default)]
    pub affinity_normalization_warning: Option<String>,
}

/// One affinity label row loaded from an external index table.
#[derive(Debug, Clone, PartialEq)]
pub struct AffinityLabel {
    /// Optional example identifier key.
    pub example_id: Option<String>,
    /// Optional protein identifier key.
    pub protein_id: Option<String>,
    /// Affinity value in kcal/mol.
    pub affinity_kcal_mol: f32,
    /// Original measurement type before normalization.
    pub measurement_type: Option<String>,
    /// Original numeric value before normalization.
    pub raw_value: Option<f32>,
    /// Original unit before normalization.
    pub raw_unit: Option<String>,
    /// Normalization provenance for the derived internal target.
    pub normalization_provenance: Option<String>,
    /// Whether the normalization path is only approximate.
    pub is_approximate: bool,
    /// Optional warning emitted during normalization.
    pub normalization_warning: Option<String>,
}

/// Parsed labels plus load-time accounting from one external label table.
#[derive(Debug, Clone)]
pub struct LoadedAffinityLabels {
    /// Parsed labels retained from the source table.
    pub labels: Vec<AffinityLabel>,
    /// Structured accounting for the source table.
    pub report: AffinityLabelLoadReport,
}

/// Row-level accounting for one external affinity label table.
#[derive(Debug, Clone, Default)]
pub struct AffinityLabelLoadReport {
    /// Total non-header rows encountered in the source table.
    pub rows_seen: usize,
    /// Blank rows skipped while loading.
    pub blank_rows: usize,
    /// Comment rows skipped while loading.
    pub comment_rows: usize,
    /// Rows retained as parsed affinity labels.
    pub parsed_rows: usize,
    /// Measurement-family histogram for retained labels.
    pub measurement_family_histogram: BTreeMap<String, usize>,
    /// Distinct normalization provenance values observed while loading labels.
    pub normalization_provenance_values: BTreeSet<String>,
    /// Retained labels derived from approximate families such as `IC50` or `EC50`.
    pub approximate_rows: usize,
}

/// Structured dataset validation artifact for one config-driven load.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetValidationReport {
    /// Number of entries discovered before parsing.
    pub discovered_complexes: usize,
    /// Number of examples successfully parsed into the research stack.
    pub parsed_examples: usize,
    /// Number of ligands successfully parsed.
    pub parsed_ligands: usize,
    /// Number of pockets successfully parsed.
    pub parsed_pockets: usize,
    /// Number of examples carrying an attached affinity label after loading.
    pub attached_labels: usize,
    /// Number of labels matched by `example_id`.
    pub example_id_label_matches: usize,
    /// Number of labels matched by `protein_id`.
    pub protein_id_label_matches: usize,
    /// Number of examples left without affinity labels.
    pub unlabeled_examples: usize,
    /// Number of pocket extraction fallback events.
    pub fallback_pocket_extractions: usize,
    /// Number of examples truncated away by `max_examples`.
    #[serde(default)]
    pub truncated_examples: usize,
    /// Number of parsed examples filtered by optional quality criteria.
    #[serde(default)]
    pub quality_filtered_examples: usize,
    /// Number of examples filtered because they lacked affinity labels.
    #[serde(default)]
    pub quality_filtered_unlabeled_examples: usize,
    /// Number of examples filtered by optional ligand atom-count limits.
    #[serde(default)]
    pub quality_filtered_ligand_atom_limit: usize,
    /// Number of examples filtered by optional pocket atom-count limits.
    #[serde(default)]
    pub quality_filtered_pocket_atom_limit: usize,
    /// Number of examples filtered because source structure provenance was missing.
    #[serde(default)]
    pub quality_filtered_missing_source_provenance: usize,
    /// Number of examples filtered because labeled affinity metadata was incomplete.
    #[serde(default)]
    pub quality_filtered_missing_affinity_metadata: usize,
    /// Label coverage after optional quality filtering and truncation.
    #[serde(default)]
    pub retained_label_coverage: f32,
    /// Pocket fallback fraction observed before optional fallback gating.
    #[serde(default)]
    pub observed_fallback_fraction: f32,
    /// Fraction of retained examples carrying source structure provenance.
    #[serde(default)]
    pub retained_source_provenance_coverage: f32,
    /// Number of examples processed by the rotation-augmentation policy.
    #[serde(default)]
    pub rotation_augmentation_attempted_examples: usize,
    /// Number of examples actually rotated by the rotation-augmentation policy.
    #[serde(default)]
    pub rotation_augmentation_applied_examples: usize,
    /// Number of external label rows loaded.
    pub loaded_label_rows: usize,
    /// Total non-header rows seen in the label table.
    #[serde(default)]
    pub label_table_rows_seen: usize,
    /// Blank rows skipped in the label table.
    #[serde(default)]
    pub label_table_blank_rows: usize,
    /// Comment rows skipped in the label table.
    #[serde(default)]
    pub label_table_comment_rows: usize,
    /// Number of labels normalized through approximate families such as `IC50` or `EC50`.
    pub approximate_affinity_labels: usize,
    /// Histogram of measurement families present in the loaded label table.
    #[serde(default)]
    pub loaded_label_measurement_family_histogram: BTreeMap<String, usize>,
    /// Distinct normalization provenance values seen in the loaded label table.
    #[serde(default)]
    pub loaded_label_normalization_provenance_values: BTreeSet<String>,
    /// Number of retained labeled examples normalized through approximate families.
    #[serde(default)]
    pub retained_approximate_affinity_labels: usize,
    /// Fraction of retained labeled examples that use approximate measurement families.
    #[serde(default)]
    pub retained_approximate_label_fraction: f32,
    /// Number of normalization warnings emitted while loading labels.
    pub affinity_normalization_warnings: usize,
    /// Number of later label rows that overwrote an earlier `example_id` label row.
    #[serde(default)]
    pub duplicate_example_id_label_rows: usize,
    /// Number of later label rows that overwrote an earlier `protein_id` label row.
    #[serde(default)]
    pub duplicate_protein_id_label_rows: usize,
    /// Number of loaded `example_id` label rows that did not attach to any manifest entry.
    #[serde(default)]
    pub unmatched_example_id_label_rows: usize,
    /// Number of loaded `protein_id` label rows that did not attach to any manifest entry.
    #[serde(default)]
    pub unmatched_protein_id_label_rows: usize,
    /// Fraction of retained labeled examples carrying normalization provenance.
    #[serde(default)]
    pub retained_normalization_provenance_coverage: f32,
    /// Number of retained labeled examples missing normalization provenance.
    #[serde(default)]
    pub retained_missing_normalization_provenance: usize,
    /// Number of retained labeled examples missing measurement-family metadata.
    #[serde(default)]
    pub retained_missing_measurement_type: usize,
    /// Histogram of retained measurement families.
    #[serde(default)]
    pub retained_measurement_family_histogram: BTreeMap<String, usize>,
    /// Number of distinct retained measurement families.
    #[serde(default)]
    pub retained_measurement_family_count: usize,
    /// Distinct retained normalization provenance values.
    #[serde(default)]
    pub retained_normalization_provenance_values: BTreeSet<String>,
    /// Histogram of retained ligand atom-count bins for atom-count prior calibration.
    #[serde(default)]
    pub retained_ligand_atom_count_histogram: BTreeMap<String, usize>,
    /// Histogram of retained pocket atom-count bins for atom-count prior calibration.
    #[serde(default)]
    pub retained_pocket_atom_count_histogram: BTreeMap<String, usize>,
    /// Mean retained ligand atom count.
    #[serde(default)]
    pub retained_mean_ligand_atom_count: f64,
    /// Atom-count prior provenance active for the configured generation mode.
    #[serde(default)]
    pub atom_count_prior_provenance: String,
    /// Mean absolute atom-count prediction error when target ligand counts are available.
    #[serde(default)]
    pub atom_count_prior_mae: f64,
    /// Coordinate-frame contract used by retained model examples.
    #[serde(default)]
    pub coordinate_frame_contract: String,
    /// Number of retained examples with finite coordinate-frame origins.
    #[serde(default)]
    pub coordinate_frame_origin_valid_examples: usize,
    /// Number of retained examples whose model coordinates satisfy the ligand-centered frame check.
    #[serde(default)]
    pub ligand_centered_coordinate_frame_examples: usize,
    /// Whether every retained example reports pocket coordinates in the ligand-centered model frame.
    #[serde(default)]
    pub pocket_coordinates_centered_upstream: bool,
    /// Candidate-artifact coordinate-frame contract paired with exported coordinates.
    #[serde(default)]
    pub coordinate_frame_artifact_contract: String,
    /// Whether source-frame coordinates can be reconstructed from model coordinates plus origin.
    #[serde(default)]
    pub source_coordinate_reconstruction_supported: bool,
    /// Generation-mode-specific target-context leakage contract.
    #[serde(default)]
    pub generation_target_leakage_contract: String,
    /// Whether pocket/context tensors depend on target-ligand-derived centering.
    #[serde(default)]
    pub target_ligand_context_dependency_detected: bool,
    /// Whether that target-ligand context dependency is allowed by the active generation mode.
    #[serde(default)]
    pub target_ligand_context_dependency_allowed: bool,
    /// Whether configured validation rejected target-ligand context dependency for this mode.
    #[serde(default)]
    pub target_ligand_context_dependency_rejected: bool,
    /// Warnings explaining target-ligand context leakage risk.
    #[serde(default)]
    pub target_ligand_context_leakage_warnings: Vec<String>,
    /// Warnings emitted by the affinity normalization path.
    pub normalization_warning_messages: Vec<String>,
    /// Active parsing mode used for the dataset load.
    pub parsing_mode: String,
}

/// Report for one parsed manifest entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParsedEntryReport {
    /// Whether ligand parsing succeeded for this entry.
    pub parsed_ligand: bool,
    /// Whether pocket parsing succeeded for this entry.
    pub parsed_pocket: bool,
    /// Whether pocket extraction used the nearest-atom fallback.
    pub used_pocket_fallback: bool,
    /// Whether rotation augmentation was attempted for this entry.
    pub rotation_augmentation_attempted: bool,
    /// Whether rotation augmentation was applied for this entry.
    pub rotation_augmentation_applied: bool,
}

/// Metadata from attaching external affinity labels to manifest entries.
#[derive(Debug, Clone, Copy, Default)]
pub struct LabelAttachmentReport {
    /// Number of labels matched by `example_id`.
    pub example_id_matches: usize,
    /// Number of labels matched by `protein_id`.
    pub protein_id_matches: usize,
    /// Number of duplicate `example_id` rows overwritten during attachment-map construction.
    pub duplicate_example_id_rows: usize,
    /// Number of duplicate `protein_id` rows overwritten during attachment-map construction.
    pub duplicate_protein_id_rows: usize,
    /// Number of loaded `example_id` rows that did not match any manifest entry.
    pub unmatched_example_id_rows: usize,
    /// Number of loaded `protein_id` rows that did not match any manifest entry.
    pub unmatched_protein_id_rows: usize,
}

#[derive(Debug, Clone)]
struct PocketLoadResult {
    pocket: Pocket,
    used_fallback: bool,
}

/// Build a small deterministic synthetic dataset.
pub fn synthetic_phase1_examples() -> Vec<MolecularExample> {
    vec![
        MolecularExample::from_legacy(
            "ex-0",
            "protein-a",
            &toy_ligand(0.0),
            &toy_pocket("protein-a", 0.0),
        ),
        MolecularExample::from_legacy(
            "ex-1",
            "protein-b",
            &toy_ligand(0.4),
            &toy_pocket("protein-b", 0.8),
        ),
        MolecularExample::from_legacy(
            "ex-2",
            "protein-c",
            &toy_ligand(-0.5),
            &toy_pocket("protein-c", -0.2),
        ),
        MolecularExample::from_legacy(
            "ex-3",
            "protein-a",
            &toy_ligand(1.1),
            &toy_pocket("protein-a", 0.6),
        ),
    ]
}
