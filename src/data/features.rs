//! Feature structures used by the modular research pipeline.

use tch::{Device, Kind, Tensor};

use crate::types::{tensor_from_slice, AtomType, Ligand, Pocket};

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
    /// Optional supervised targets attached to the complex.
    pub targets: ExampleTargets,
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

impl MolecularExample {
    /// Build a Phase 1 example from the legacy pocket/ligand structs.
    pub fn from_legacy(
        example_id: impl Into<String>,
        protein_id: impl Into<String>,
        ligand: &Ligand,
        pocket: &Pocket,
    ) -> Self {
        Self::from_legacy_with_targets(
            example_id,
            protein_id,
            ligand,
            pocket,
            ExampleTargets::default(),
        )
    }

    /// Build an example from legacy structs plus optional supervised targets.
    pub fn from_legacy_with_targets(
        example_id: impl Into<String>,
        protein_id: impl Into<String>,
        ligand: &Ligand,
        pocket: &Pocket,
        targets: ExampleTargets,
    ) -> Self {
        let topology = topology_from_ligand(ligand);
        let geometry = geometry_from_ligand(ligand);
        let pocket_features = pocket_features_from_pocket(pocket);
        Self {
            example_id: example_id.into(),
            protein_id: protein_id.into(),
            topology,
            geometry,
            pocket: pocket_features,
            targets,
        }
    }

    /// Move all modality tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            example_id: self.example_id.clone(),
            protein_id: self.protein_id.clone(),
            topology: self.topology.to_device(device),
            geometry: self.geometry.to_device(device),
            pocket: self.pocket.to_device(device),
            targets: self.targets.clone(),
        }
    }

    /// Resize pocket features to the configured model width by zero-padding or truncation.
    pub fn with_pocket_feature_dim(&self, pocket_feature_dim: i64) -> Self {
        Self {
            example_id: self.example_id.clone(),
            protein_id: self.protein_id.clone(),
            topology: self.topology.clone(),
            geometry: self.geometry.clone(),
            pocket: self.pocket.with_feature_dim(pocket_feature_dim),
            targets: self.targets.clone(),
        }
    }
}

impl TopologyFeatures {
    /// Move topology tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            atom_types: self.atom_types.to_device(device),
            edge_index: self.edge_index.to_device(device),
            bond_types: self.bond_types.to_device(device),
            adjacency: self.adjacency.to_device(device),
        }
    }
}

impl GeometryFeatures {
    /// Move geometry tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            coords: self.coords.to_device(device),
            pairwise_distances: self.pairwise_distances.to_device(device),
        }
    }
}

impl PocketFeatures {
    /// Move pocket tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            coords: self.coords.to_device(device),
            atom_features: self.atom_features.to_device(device),
            pooled_features: self.pooled_features.to_device(device),
        }
    }

    /// Resize the per-atom and pooled pocket features to match the configured model width.
    pub fn with_feature_dim(&self, target_dim: i64) -> Self {
        Self {
            coords: self.coords.shallow_clone(),
            atom_features: resize_feature_matrix(&self.atom_features, target_dim),
            pooled_features: resize_feature_vector(&self.pooled_features, target_dim),
        }
    }
}

/// Convert a ligand into topology features.
pub fn topology_from_ligand(ligand: &Ligand) -> TopologyFeatures {
    let num_atoms = ligand.atoms.len() as i64;
    let atom_types: Vec<i64> = ligand
        .atoms
        .iter()
        .map(|atom| atom.atom_type.to_index())
        .collect();
    let atom_types = tensor_from_slice(&atom_types).to_kind(Kind::Int64);

    let mut edge_rows = Vec::with_capacity(ligand.bonds.len() * 2);
    let mut edge_cols = Vec::with_capacity(ligand.bonds.len() * 2);
    for &(src, dst) in &ligand.bonds {
        edge_rows.push(src as i64);
        edge_cols.push(dst as i64);
    }

    let edge_index = if ligand.bonds.is_empty() {
        Tensor::zeros([2, 0], (Kind::Int64, tch::Device::Cpu))
    } else {
        Tensor::stack(
            &[
                tensor_from_slice(&edge_rows).to_kind(Kind::Int64),
                tensor_from_slice(&edge_cols).to_kind(Kind::Int64),
            ],
            0,
        )
    };

    let bond_types = Tensor::zeros([ligand.bonds.len() as i64], (Kind::Int64, tch::Device::Cpu));
    let adjacency = Tensor::zeros([num_atoms, num_atoms], (Kind::Float, tch::Device::Cpu));
    for &(src, dst) in &ligand.bonds {
        let _ = adjacency.get(src as i64).get(dst as i64).fill_(1.0);
        let _ = adjacency.get(dst as i64).get(src as i64).fill_(1.0);
    }

    TopologyFeatures {
        atom_types,
        edge_index,
        bond_types,
        adjacency,
    }
}

/// Convert a ligand into geometry features.
pub fn geometry_from_ligand(ligand: &Ligand) -> GeometryFeatures {
    let coords_flat: Vec<f32> = ligand
        .atoms
        .iter()
        .flat_map(|atom| atom.coords.map(|value| value as f32))
        .collect();
    let num_atoms = ligand.atoms.len() as i64;
    let coords = if coords_flat.is_empty() {
        Tensor::zeros([0, 3], (Kind::Float, tch::Device::Cpu))
    } else {
        tensor_from_slice(&coords_flat).reshape([num_atoms, 3])
    };

    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    let pairwise_distances = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .sqrt();

    GeometryFeatures {
        coords,
        pairwise_distances,
    }
}

/// Convert a pocket into context features.
pub fn pocket_features_from_pocket(pocket: &Pocket) -> PocketFeatures {
    let coords_flat: Vec<f32> = pocket
        .atoms
        .iter()
        .flat_map(|atom| atom.coords.map(|value| value as f32))
        .collect();
    let feature_flat: Vec<f32> = pocket
        .atoms
        .iter()
        .flat_map(|atom| atom_feature_vector(atom.atom_type))
        .collect();
    let num_atoms = pocket.atoms.len() as i64;

    let coords = if coords_flat.is_empty() {
        Tensor::zeros([0, 3], (Kind::Float, tch::Device::Cpu))
    } else {
        tensor_from_slice(&coords_flat).reshape([num_atoms, 3])
    };

    let atom_features = if feature_flat.is_empty() {
        Tensor::zeros(
            [0, LEGACY_POCKET_FEATURE_DIM],
            (Kind::Float, tch::Device::Cpu),
        )
    } else {
        tensor_from_slice(&feature_flat).reshape([num_atoms, LEGACY_POCKET_FEATURE_DIM])
    };

    let pooled_features = if num_atoms == 0 {
        Tensor::zeros([LEGACY_POCKET_FEATURE_DIM], (Kind::Float, tch::Device::Cpu))
    } else {
        atom_features.mean_dim([0].as_slice(), false, Kind::Float)
    };

    PocketFeatures {
        coords,
        atom_features,
        pooled_features,
    }
}

fn atom_feature_vector(atom_type: AtomType) -> [f32; 6] {
    let index = atom_type.to_index() as usize;
    let mut features = [0.0_f32; 6];
    if index < features.len() {
        features[index] = 1.0;
    }
    features
}

fn resize_feature_matrix(tensor: &Tensor, target_dim: i64) -> Tensor {
    let current_dim = tensor.size().get(1).copied().unwrap_or(0);
    if current_dim == target_dim {
        return tensor.shallow_clone();
    }

    let rows = tensor.size().first().copied().unwrap_or(0);
    if current_dim > target_dim {
        return tensor.narrow(1, 0, target_dim);
    }

    let padding = Tensor::zeros(
        [rows, target_dim - current_dim],
        (tensor.kind(), tensor.device()),
    );
    Tensor::cat(&[tensor.shallow_clone(), padding], 1)
}

fn resize_feature_vector(tensor: &Tensor, target_dim: i64) -> Tensor {
    let current_dim = tensor.size().first().copied().unwrap_or(0);
    if current_dim == target_dim {
        return tensor.shallow_clone();
    }

    if current_dim > target_dim {
        return tensor.narrow(0, 0, target_dim);
    }

    let padding = Tensor::zeros([target_dim - current_dim], (tensor.kind(), tensor.device()));
    Tensor::cat(&[tensor.shallow_clone(), padding], 0)
}
