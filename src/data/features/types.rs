/// Per-atom categorical and scalar features for ligand topology.
#[derive(Debug)]
pub struct TopologyFeatures {
    /// Encoded atom types with shape `[num_atoms]`.
    pub atom_types: Tensor,
    /// Encoded bond indices with shape `[2, num_bonds]`.
    pub edge_index: Tensor,
    /// Encoded bond types with shape `[num_bonds]`.
    pub bond_types: Tensor,
    /// Dense adjacency with shape `[num_atoms, num_atoms]`.
    pub adjacency: Tensor,
}

impl Clone for TopologyFeatures {
    fn clone(&self) -> Self {
        Self {
            atom_types: self.atom_types.shallow_clone(),
            edge_index: self.edge_index.shallow_clone(),
            bond_types: self.bond_types.shallow_clone(),
            adjacency: self.adjacency.shallow_clone(),
        }
    }
}

/// Coordinate-driven ligand geometry features.
#[derive(Debug)]
pub struct GeometryFeatures {
    /// Cartesian coordinates with shape `[num_atoms, 3]`.
    pub coords: Tensor,
    /// Pairwise distance matrix with shape `[num_atoms, num_atoms]`.
    pub pairwise_distances: Tensor,
}

impl Clone for GeometryFeatures {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.shallow_clone(),
            pairwise_distances: self.pairwise_distances.shallow_clone(),
        }
    }
}

/// Pocket atom coordinates and local feature vectors.
#[derive(Debug)]
pub struct PocketFeatures {
    /// Pocket coordinates with shape `[num_atoms, 3]`.
    pub coords: Tensor,
    /// Pocket feature matrix with shape `[num_atoms, feature_dim]`.
    pub atom_features: Tensor,
    /// Global pooled pocket summary with shape `[feature_dim]`.
    pub pooled_features: Tensor,
}

/// Default legacy pocket atom feature width before config-driven resizing.
pub const LEGACY_POCKET_FEATURE_DIM: i64 = 6;

impl Clone for PocketFeatures {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.shallow_clone(),
            atom_features: self.atom_features.shallow_clone(),
            pooled_features: self.pooled_features.shallow_clone(),
        }
    }
}

/// Single protein-ligand example consumed by the new research stack.
#[derive(Debug, Clone)]
pub struct MolecularExample {
    /// Stable identifier for logging and split bookkeeping.
    pub example_id: String,
    /// Protein identifier used for unseen-pocket split logic.
    pub protein_id: String,
    /// Topology modality input.
    pub topology: TopologyFeatures,
    /// Geometry modality input.
    pub geometry: GeometryFeatures,
    /// Pocket/context modality input.
    pub pocket: PocketFeatures,
    /// Translation from ligand-centered model coordinates back to source structure coordinates.
    pub coordinate_frame_origin: [f32; 3],
    /// Optional source protein structure path for downstream backend workflows.
    pub source_pocket_path: Option<PathBuf>,
    /// Optional source ligand path for downstream backend workflows.
    pub source_ligand_path: Option<PathBuf>,
    /// Explicit decoder-side supervision for corruption recovery and denoising.
    pub decoder_supervision: DecoderSupervision,
    /// Optional supervised targets attached to the complex.
    pub targets: ExampleTargets,
}

/// Decoder-side supervision separated from encoder conditioning inputs.
#[derive(Debug)]
pub struct DecoderSupervision {
    /// Clean target atom types for corruption recovery.
    pub target_atom_types: Tensor,
    /// Corrupted atom types provided to the decoder.
    pub corrupted_atom_types: Tensor,
    /// Binary mask indicating which atom identities were corrupted.
    pub atom_corruption_mask: Tensor,
    /// Clean Cartesian coordinates used as geometry targets.
    pub target_coords: Tensor,
    /// Noisy decoder input coordinates.
    pub noisy_coords: Tensor,
    /// Deterministic coordinate perturbation added to the clean coordinates.
    pub coordinate_noise: Tensor,
    /// Pairwise target distances derived from clean coordinates.
    pub target_pairwise_distances: Tensor,
    /// Configured number of iterative rollout steps used by training and generation.
    pub rollout_steps: usize,
    /// Geometric decay applied to later rollout losses.
    pub training_step_weight_decay: f64,
    /// Reproducibility metadata for the corruption transform.
    pub corruption_metadata: GenerationCorruptionMetadata,
}

impl Clone for DecoderSupervision {
    fn clone(&self) -> Self {
        Self {
            target_atom_types: self.target_atom_types.shallow_clone(),
            corrupted_atom_types: self.corrupted_atom_types.shallow_clone(),
            atom_corruption_mask: self.atom_corruption_mask.shallow_clone(),
            target_coords: self.target_coords.shallow_clone(),
            noisy_coords: self.noisy_coords.shallow_clone(),
            coordinate_noise: self.coordinate_noise.shallow_clone(),
            target_pairwise_distances: self.target_pairwise_distances.shallow_clone(),
            rollout_steps: self.rollout_steps,
            training_step_weight_decay: self.training_step_weight_decay,
            corruption_metadata: self.corruption_metadata.clone(),
        }
    }
}

/// Optional labels attached to one protein-ligand complex.
#[derive(Debug, Clone, Default)]
pub struct ExampleTargets {
    /// Experimental or curated binding affinity in kcal/mol.
    pub affinity_kcal_mol: Option<f32>,
    /// Original measurement type before normalization, such as `Kd`, `Ki`, `IC50`, or `dG`.
    pub affinity_measurement_type: Option<String>,
    /// Original numeric value before normalization.
    pub affinity_raw_value: Option<f32>,
    /// Original unit before normalization, such as `nM` or `uM`.
    pub affinity_raw_unit: Option<String>,
    /// Normalization path used to derive `affinity_kcal_mol`.
    pub affinity_normalization_provenance: Option<String>,
    /// Whether the normalization path is only an approximation.
    pub affinity_is_approximate: bool,
    /// Optional warning describing approximation or suspicious normalization assumptions.
    pub affinity_normalization_warning: Option<String>,
}
