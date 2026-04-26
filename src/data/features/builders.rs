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
