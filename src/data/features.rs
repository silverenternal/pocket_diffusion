//! Feature structures used by the modular research pipeline.

use std::path::PathBuf;

use tch::{Device, Kind, Tensor};

use crate::{
    config::GenerationTargetConfig,
    types::{tensor_from_slice, AtomType, GenerationCorruptionMetadata, Ligand, Pocket},
};

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
        Self::from_legacy_with_targets_and_generation(
            example_id,
            protein_id,
            ligand,
            pocket,
            targets,
            &GenerationTargetConfig::default(),
        )
    }

    /// Build an example from legacy structs plus explicit decoder-supervision config.
    pub fn from_legacy_with_targets_and_generation(
        example_id: impl Into<String>,
        protein_id: impl Into<String>,
        ligand: &Ligand,
        pocket: &Pocket,
        targets: ExampleTargets,
        generation_target: &GenerationTargetConfig,
    ) -> Self {
        let topology = topology_from_ligand(ligand);
        let ligand_center = ligand_centroid(ligand);
        let geometry = geometry_from_ligand_centered(ligand, ligand_center);
        let pocket_features = pocket_features_from_pocket_centered(pocket, ligand_center);
        let example_id = example_id.into();
        let protein_id = protein_id.into();
        let decoder_supervision =
            build_decoder_supervision(&example_id, &topology, &geometry, generation_target);
        Self {
            example_id,
            protein_id,
            topology,
            geometry,
            pocket: pocket_features,
            coordinate_frame_origin: [
                ligand_center[0] as f32,
                ligand_center[1] as f32,
                ligand_center[2] as f32,
            ],
            source_pocket_path: None,
            source_ligand_path: None,
            decoder_supervision,
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
            coordinate_frame_origin: self.coordinate_frame_origin,
            source_pocket_path: self.source_pocket_path.clone(),
            source_ligand_path: self.source_ligand_path.clone(),
            decoder_supervision: self.decoder_supervision.to_device(device),
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
            coordinate_frame_origin: self.coordinate_frame_origin,
            source_pocket_path: self.source_pocket_path.clone(),
            source_ligand_path: self.source_ligand_path.clone(),
            decoder_supervision: self.decoder_supervision.clone(),
            targets: self.targets.clone(),
        }
    }

    /// Rebuild decoder-side supervision using the configured corruption process.
    pub fn with_generation_config(&self, generation_target: &GenerationTargetConfig) -> Self {
        Self {
            example_id: self.example_id.clone(),
            protein_id: self.protein_id.clone(),
            topology: self.topology.clone(),
            geometry: self.geometry.clone(),
            pocket: self.pocket.clone(),
            coordinate_frame_origin: self.coordinate_frame_origin,
            source_pocket_path: self.source_pocket_path.clone(),
            source_ligand_path: self.source_ligand_path.clone(),
            decoder_supervision: build_decoder_supervision(
                &self.example_id,
                &self.topology,
                &self.geometry,
                generation_target,
            ),
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

impl DecoderSupervision {
    /// Move decoder supervision tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            target_atom_types: self.target_atom_types.to_device(device),
            corrupted_atom_types: self.corrupted_atom_types.to_device(device),
            atom_corruption_mask: self.atom_corruption_mask.to_device(device),
            target_coords: self.target_coords.to_device(device),
            noisy_coords: self.noisy_coords.to_device(device),
            coordinate_noise: self.coordinate_noise.to_device(device),
            target_pairwise_distances: self.target_pairwise_distances.to_device(device),
            rollout_steps: self.rollout_steps,
            training_step_weight_decay: self.training_step_weight_decay,
            corruption_metadata: self.corruption_metadata.clone(),
        }
    }

    /// Weight assigned to the provided rollout step under the configured decay schedule.
    pub fn rollout_step_weight(&self, step_index: usize) -> f64 {
        self.training_step_weight_decay.powi(step_index as i32)
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
    geometry_from_ligand_centered(ligand, ligand_centroid(ligand))
}

fn geometry_from_ligand_centered(ligand: &Ligand, center: [f64; 3]) -> GeometryFeatures {
    let coords_flat: Vec<f32> = ligand
        .atoms
        .iter()
        .flat_map(|atom| {
            [
                (atom.coords[0] - center[0]) as f32,
                (atom.coords[1] - center[1]) as f32,
                (atom.coords[2] - center[2]) as f32,
            ]
        })
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
    pocket_features_from_pocket_centered(pocket, [0.0, 0.0, 0.0])
}

fn pocket_features_from_pocket_centered(
    pocket: &Pocket,
    ligand_center: [f64; 3],
) -> PocketFeatures {
    let coords_flat: Vec<f32> = pocket
        .atoms
        .iter()
        .flat_map(|atom| {
            [
                (atom.coords[0] - ligand_center[0]) as f32,
                (atom.coords[1] - ligand_center[1]) as f32,
                (atom.coords[2] - ligand_center[2]) as f32,
            ]
        })
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

fn ligand_centroid(ligand: &Ligand) -> [f64; 3] {
    if ligand.atoms.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let (x, y, z) = ligand.atoms.iter().fold((0.0, 0.0, 0.0), |acc, atom| {
        (
            acc.0 + atom.coords[0],
            acc.1 + atom.coords[1],
            acc.2 + atom.coords[2],
        )
    });
    let denom = ligand.atoms.len() as f64;
    [x / denom, y / denom, z / denom]
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

fn build_decoder_supervision(
    example_id: &str,
    topology: &TopologyFeatures,
    geometry: &GeometryFeatures,
    generation_target: &GenerationTargetConfig,
) -> DecoderSupervision {
    let device = topology.atom_types.device();
    let num_atoms = topology.atom_types.size()[0];
    let target_atom_types = topology.atom_types.shallow_clone();
    let target_coords = geometry.coords.shallow_clone();
    let target_pairwise_distances = geometry.pairwise_distances.shallow_clone();
    let metadata = GenerationCorruptionMetadata {
        atom_mask_ratio: generation_target.atom_mask_ratio,
        coordinate_noise_std: generation_target.coordinate_noise_std,
        corruption_seed: generation_target.corruption_seed,
    };

    if num_atoms == 0 {
        let empty_long = Tensor::zeros([0], (Kind::Int64, device));
        let empty_float = Tensor::zeros([0], (Kind::Float, device));
        return DecoderSupervision {
            target_atom_types,
            corrupted_atom_types: empty_long,
            atom_corruption_mask: empty_float.shallow_clone(),
            target_coords,
            noisy_coords: Tensor::zeros([0, 3], (Kind::Float, device)),
            coordinate_noise: Tensor::zeros([0, 3], (Kind::Float, device)),
            target_pairwise_distances,
            rollout_steps: generation_target.rollout_steps,
            training_step_weight_decay: generation_target.training_step_weight_decay,
            corruption_metadata: metadata,
        };
    }

    let mask_values: Vec<f32> = (0..num_atoms)
        .map(|atom_ix| {
            if should_mask_atom(example_id, atom_ix as usize, generation_target) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let atom_corruption_mask = tensor_from_slice(&mask_values)
        .to_kind(Kind::Float)
        .to_device(device);
    let corrupted_atom_types =
        apply_atom_corruption(&target_atom_types, &atom_corruption_mask, generation_target);

    let noise_flat: Vec<f32> = (0..num_atoms)
        .flat_map(|atom_ix| {
            (0..3).map(move |coord_ix| {
                deterministic_noise(
                    example_id,
                    atom_ix as usize,
                    coord_ix as usize,
                    generation_target,
                )
            })
        })
        .collect();
    let coordinate_noise = tensor_from_slice(&noise_flat)
        .reshape([num_atoms, 3])
        .to_device(device);
    let noisy_coords = &target_coords + &coordinate_noise;

    DecoderSupervision {
        target_atom_types,
        corrupted_atom_types,
        atom_corruption_mask,
        target_coords,
        noisy_coords,
        coordinate_noise,
        target_pairwise_distances,
        rollout_steps: generation_target.rollout_steps,
        training_step_weight_decay: generation_target.training_step_weight_decay,
        corruption_metadata: metadata,
    }
}

fn should_mask_atom(
    example_id: &str,
    atom_ix: usize,
    generation_target: &GenerationTargetConfig,
) -> bool {
    if generation_target.atom_mask_ratio <= 0.0 {
        return false;
    }
    let hash = stable_atom_hash(example_id, atom_ix, generation_target.corruption_seed);
    let normalized = (hash % 10_000) as f32 / 10_000.0;
    normalized < generation_target.atom_mask_ratio
}

fn apply_atom_corruption(
    atom_types: &Tensor,
    mask: &Tensor,
    generation_target: &GenerationTargetConfig,
) -> Tensor {
    let mask_long = mask.to_kind(Kind::Int64);
    let replacement = Tensor::full_like(
        atom_types,
        ((generation_target.corruption_seed % 5) + 1) as i64,
    );
    atom_types * (1 - &mask_long) + replacement * mask_long
}

fn deterministic_noise(
    example_id: &str,
    atom_ix: usize,
    coord_ix: usize,
    generation_target: &GenerationTargetConfig,
) -> f32 {
    if generation_target.coordinate_noise_std == 0.0 {
        return 0.0;
    }
    let hash = stable_atom_hash(
        example_id,
        atom_ix * 17 + coord_ix,
        generation_target.corruption_seed,
    );
    let phase = (hash % 65_521) as f32 / 65_521.0;
    ((phase * std::f32::consts::TAU).sin() * generation_target.coordinate_noise_std) as f32
}

fn stable_atom_hash(example_id: &str, atom_ix: usize, seed: u64) -> u64 {
    let mut hash = seed ^ 0x9e37_79b9_7f4a_7c15_u64;
    for byte in example_id.as_bytes() {
        hash = hash.rotate_left(7) ^ u64::from(*byte);
        hash = hash.wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    hash ^ (atom_ix as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Atom;

    fn ligand_with_offset(offset: f64) -> Ligand {
        Ligand {
            atoms: vec![
                Atom {
                    coords: [offset, offset + 1.0, offset + 2.0],
                    atom_type: AtomType::Carbon,
                    index: 0,
                },
                Atom {
                    coords: [offset + 2.0, offset + 3.0, offset + 4.0],
                    atom_type: AtomType::Oxygen,
                    index: 1,
                },
            ],
            bonds: vec![(0, 1)],
            fingerprint: None,
        }
    }

    #[test]
    fn geometry_features_are_ligand_centered() {
        let ligand = ligand_with_offset(100.0);
        let geometry = geometry_from_ligand(&ligand);

        let centroid = geometry.coords.mean_dim([0].as_slice(), false, Kind::Float);
        for dim in 0..3 {
            assert!(centroid.double_value(&[dim]).abs() < 1e-6);
        }
        assert!((geometry.pairwise_distances.double_value(&[0, 1]) - 12.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn legacy_examples_keep_pocket_coords_in_ligand_centered_frame() {
        let ligand = ligand_with_offset(100.0);
        let pocket = Pocket {
            name: "pocket".to_string(),
            atoms: vec![Atom {
                coords: [102.0, 104.0, 106.0],
                atom_type: AtomType::Nitrogen,
                index: 0,
            }],
        };

        let example = MolecularExample::from_legacy("example", "protein", &ligand, &pocket);
        assert_eq!(example.coordinate_frame_origin, [101.0, 102.0, 103.0]);
        assert_eq!(example.pocket.coords.double_value(&[0, 0]), 1.0);
        assert_eq!(example.pocket.coords.double_value(&[0, 1]), 2.0);
        assert_eq!(example.pocket.coords.double_value(&[0, 2]), 3.0);
    }
}
