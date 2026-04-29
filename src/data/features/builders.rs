/// Convert a ligand into topology features.
pub fn topology_from_ligand(ligand: &Ligand) -> TopologyFeatures {
    let num_atoms = ligand.atoms.len() as i64;
    let atom_types: Vec<i64> = ligand
        .atoms
        .iter()
        .map(|atom| atom.atom_type.to_index())
        .collect();
    let atom_types = tensor_from_slice(&atom_types).to_kind(Kind::Int64);
    let atom_type_roles = ligand
        .atoms
        .iter()
        .map(|atom| atom.atom_type)
        .collect::<Vec<_>>();
    let chemistry_roles = chemistry_role_features_from_atom_types(&atom_type_roles);

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

    let bond_type_values = normalized_ligand_bond_types(ligand);
    let bond_types = if bond_type_values.is_empty() {
        Tensor::zeros([0], (Kind::Int64, tch::Device::Cpu))
    } else {
        tensor_from_slice(&bond_type_values).to_kind(Kind::Int64)
    };
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
        chemistry_roles,
    }
}

fn normalized_ligand_bond_types(ligand: &Ligand) -> Vec<i64> {
    if ligand.bond_types.len() != ligand.bonds.len() {
        return vec![0; ligand.bonds.len()];
    }
    ligand
        .bond_types
        .iter()
        .map(|bond_type| (*bond_type).clamp(0, 7))
        .collect()
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
    let atom_type_roles = pocket
        .atoms
        .iter()
        .map(|atom| atom.atom_type)
        .collect::<Vec<_>>();
    let chemistry_roles = chemistry_role_features_from_atom_types(&atom_type_roles);
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
        chemistry_roles,
    }
}

/// Build deterministic chemistry role rows from available atom types.
pub fn chemistry_role_features_from_atom_types(
    atom_types: &[AtomType],
) -> ChemistryRoleFeatureMatrix {
    if atom_types.is_empty() {
        return ChemistryRoleFeatureMatrix {
            role_vectors: Tensor::zeros(
                [0, CHEMISTRY_ROLE_FEATURE_DIM],
                (Kind::Float, tch::Device::Cpu),
            ),
            availability: Tensor::zeros([0], (Kind::Float, tch::Device::Cpu)),
            provenance: ChemistryRoleFeatureProvenance::Unavailable,
        };
    }

    let roles = atom_types
        .iter()
        .copied()
        .map(heuristic_chemistry_role_feature)
        .collect::<Vec<_>>();
    let role_flat = roles
        .iter()
        .flat_map(|role| role.to_vector())
        .collect::<Vec<_>>();
    let availability = roles
        .iter()
        .map(|role| role.available)
        .collect::<Vec<_>>();

    ChemistryRoleFeatureMatrix {
        role_vectors: tensor_from_slice(&role_flat)
            .reshape([atom_types.len() as i64, CHEMISTRY_ROLE_FEATURE_DIM]),
        availability: tensor_from_slice(&availability),
        provenance: ChemistryRoleFeatureProvenance::Heuristic,
    }
}

/// Deterministic no-backend atom-type chemistry role heuristic.
pub fn heuristic_chemistry_role_feature(atom_type: AtomType) -> ChemistryRoleFeature {
    match atom_type {
        AtomType::Carbon => ChemistryRoleFeature {
            hydrophobic: 1.0,
            available: 1.0,
            unknown: 0.0,
            provenance: ChemistryRoleFeatureProvenance::Heuristic,
            ..ChemistryRoleFeature::unavailable()
        },
        AtomType::Nitrogen => ChemistryRoleFeature {
            donor: 1.0,
            acceptor: 1.0,
            metal_binding: 0.25,
            available: 1.0,
            unknown: 0.0,
            provenance: ChemistryRoleFeatureProvenance::Heuristic,
            ..ChemistryRoleFeature::unavailable()
        },
        AtomType::Oxygen => ChemistryRoleFeature {
            acceptor: 1.0,
            negative: 0.25,
            metal_binding: 0.25,
            available: 1.0,
            unknown: 0.0,
            provenance: ChemistryRoleFeatureProvenance::Heuristic,
            ..ChemistryRoleFeature::unavailable()
        },
        AtomType::Sulfur => ChemistryRoleFeature {
            acceptor: 0.5,
            hydrophobic: 0.5,
            metal_binding: 0.25,
            available: 1.0,
            unknown: 0.0,
            provenance: ChemistryRoleFeatureProvenance::Heuristic,
            ..ChemistryRoleFeature::unavailable()
        },
        AtomType::Hydrogen | AtomType::Other => ChemistryRoleFeature::unavailable(),
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
