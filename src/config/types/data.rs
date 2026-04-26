/// Dataset and split configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Root directory for data assets.
    pub root_dir: PathBuf,
    /// Dataset source format used by the real-data loader.
    pub dataset_format: DatasetFormat,
    /// Optional manifest path for explicit sample enumeration.
    pub manifest_path: Option<PathBuf>,
    /// Optional CSV/TSV label table used to attach affinity targets.
    pub label_table_path: Option<PathBuf>,
    /// Lightweight or strict parsing behavior for on-disk assets.
    #[serde(default)]
    pub parsing_mode: ParsingMode,
    /// Maximum ligand atoms retained in a batch.
    pub max_ligand_atoms: usize,
    /// Maximum pocket atoms retained in a batch.
    pub max_pocket_atoms: usize,
    /// Pocket extraction cutoff radius in angstroms.
    pub pocket_cutoff_angstrom: f32,
    /// Optional limit for quick debugging runs.
    pub max_examples: Option<usize>,
    /// Batch size used by the training loader.
    pub batch_size: usize,
    /// Unseen-pocket split seed.
    pub split_seed: u64,
    /// Fraction of examples used for validation.
    pub val_fraction: f32,
    /// Fraction of examples used for test.
    pub test_fraction: f32,
    /// Whether to stratify protein-level splits by dominant affinity measurement type.
    pub stratify_by_measurement: bool,
    /// Optional inclusion/exclusion filters for real-data evidence surfaces.
    #[serde(default)]
    pub quality_filters: DataQualityFilterConfig,
    /// Decoder-side corruption and denoising target generation.
    #[serde(default)]
    pub generation_target: GenerationTargetConfig,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("./data"),
            dataset_format: DatasetFormat::Synthetic,
            manifest_path: None,
            label_table_path: None,
            parsing_mode: ParsingMode::Lightweight,
            max_ligand_atoms: 64,
            max_pocket_atoms: 256,
            pocket_cutoff_angstrom: 6.0,
            max_examples: None,
            batch_size: 4,
            split_seed: 42,
            val_fraction: 0.1,
            test_fraction: 0.1,
            stratify_by_measurement: false,
            quality_filters: DataQualityFilterConfig::default(),
            generation_target: GenerationTargetConfig::default(),
        }
    }
}

impl DataConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.max_ligand_atoms == 0 {
            return Err(ConfigValidationError::new(
                "data.max_ligand_atoms must be greater than zero",
            ));
        }
        if self.max_pocket_atoms == 0 {
            return Err(ConfigValidationError::new(
                "data.max_pocket_atoms must be greater than zero",
            ));
        }
        if self.batch_size == 0 {
            return Err(ConfigValidationError::new(
                "data.batch_size must be greater than zero",
            ));
        }
        if self.pocket_cutoff_angstrom <= 0.0 {
            return Err(ConfigValidationError::new(
                "data.pocket_cutoff_angstrom must be positive",
            ));
        }
        if let Some(limit) = self.max_examples {
            if limit == 0 {
                return Err(ConfigValidationError::new(
                    "data.max_examples must be omitted or greater than zero",
                ));
            }
        }
        validate_split_fractions(self.val_fraction, self.test_fraction)?;
        self.quality_filters.validate()?;
        self.generation_target.validate()?;
        match self.dataset_format {
            DatasetFormat::Synthetic => {}
            DatasetFormat::ManifestJson => {
                let manifest_path = self.manifest_path.as_deref().ok_or_else(|| {
                    ConfigValidationError::new(
                        "data.manifest_path is required when data.dataset_format=manifest_json",
                    )
                })?;
                ensure_file_exists(manifest_path, "data.manifest_path")?;
            }
            DatasetFormat::PdbbindLikeDir => {
                ensure_directory_exists(&self.root_dir, "data.root_dir")?;
            }
        }
        if let Some(label_table_path) = self.label_table_path.as_deref() {
            ensure_file_exists(label_table_path, "data.label_table_path")?;
        }
        Ok(())
    }
}

/// Optional dataset quality filters used to make real-data inclusion criteria reproducible.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataQualityFilterConfig {
    /// Minimum retained labeled fraction required after filtering.
    #[serde(default)]
    pub min_label_coverage: Option<f32>,
    /// Maximum allowed pocket-fallback fraction before the load is rejected.
    #[serde(default)]
    pub max_fallback_fraction: Option<f32>,
    /// Optional atom-count exclusion threshold for parsed ligands.
    #[serde(default)]
    pub max_ligand_atoms: Option<usize>,
    /// Optional atom-count exclusion threshold for parsed pockets.
    #[serde(default)]
    pub max_pocket_atoms: Option<usize>,
    /// Require source protein and ligand structure paths to be retained on examples.
    #[serde(default)]
    pub require_source_structure_provenance: bool,
    /// Require labeled examples to retain measurement-family and normalization metadata.
    #[serde(default)]
    pub require_affinity_metadata: bool,
    /// Maximum retained fraction of approximate measurement families such as `IC50` or `EC50`.
    #[serde(default)]
    pub max_approximate_label_fraction: Option<f32>,
    /// Minimum retained coverage of normalization provenance on labeled examples.
    #[serde(default)]
    pub min_normalization_provenance_coverage: Option<f32>,
}

impl DataQualityFilterConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_optional_fraction(
            self.min_label_coverage,
            "data.quality_filters.min_label_coverage",
        )?;
        validate_optional_fraction(
            self.max_fallback_fraction,
            "data.quality_filters.max_fallback_fraction",
        )?;
        validate_optional_fraction(
            self.max_approximate_label_fraction,
            "data.quality_filters.max_approximate_label_fraction",
        )?;
        validate_optional_fraction(
            self.min_normalization_provenance_coverage,
            "data.quality_filters.min_normalization_provenance_coverage",
        )?;
        if self.max_ligand_atoms == Some(0) {
            return Err(ConfigValidationError::new(
                "data.quality_filters.max_ligand_atoms must be omitted or greater than zero",
            ));
        }
        if self.max_pocket_atoms == Some(0) {
            return Err(ConfigValidationError::new(
                "data.quality_filters.max_pocket_atoms must be omitted or greater than zero",
            ));
        }
        Ok(())
    }
}

