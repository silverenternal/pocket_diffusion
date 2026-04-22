//! 统一分子表示接口
//! 实现两种分子表示方案的解耦设计

use ndarray::{Array1, Array2};
use std::fmt::Debug;

/// 统一分子表示 trait（解耦核心）
pub trait MoleculeRepresentation: Clone + Debug + Send + Sync {
    /// 获取原子数量
    fn atom_count(&self) -> usize;

    /// 转换为3D坐标矩阵 [N, 3]
    fn to_3d_coords(&self) -> Array2<f32>;

    /// 获取表示名称（用于实验输出）
    fn name(&self) -> &'static str;

    /// 计算表示的内存占用（字节）
    fn memory_usage(&self) -> usize;
}

// ==================== 方案A：纯3D表示（基线）====================

/// 方案A：纯3D表示（基线方法）
/// 仅存储原子类型和3D坐标，丢失拓扑连接信息
#[derive(Clone, Debug)]
pub struct Molecule3D {
    /// 原子类型索引
    pub atom_types: Vec<usize>,
    /// 3D坐标矩阵 [N, 3]
    pub coords_3d: Array2<f32>,
}

impl MoleculeRepresentation for Molecule3D {
    fn atom_count(&self) -> usize {
        self.atom_types.len()
    }

    fn to_3d_coords(&self) -> Array2<f32> {
        self.coords_3d.clone()
    }

    fn name(&self) -> &'static str {
        "Molecule3D (Baseline)"
    }

    fn memory_usage(&self) -> usize {
        self.atom_types.len() * std::mem::size_of::<usize>()
            + self.coords_3d.len() * std::mem::size_of::<f32>()
    }
}

impl Molecule3D {
    /// 创建新的纯3D分子表示
    pub fn new(atom_types: Vec<usize>, coords_3d: Array2<f32>) -> Self {
        assert_eq!(
            atom_types.len(),
            coords_3d.nrows(),
            "原子类型数量与坐标行数不匹配"
        );
        assert_eq!(coords_3d.ncols(), 3, "坐标必须是[N, 3]形状");
        Self {
            atom_types,
            coords_3d,
        }
    }

    /// 从坐标和原子类型构建
    pub fn from_atoms(atoms: &[(usize, [f32; 3])]) -> Self {
        let atom_types: Vec<usize> = atoms.iter().map(|(t, _)| *t).collect();
        let coords: Vec<f32> = atoms.iter().flat_map(|(_, c)| c.iter().copied()).collect();
        let coords_3d = Array2::from_shape_vec((atoms.len(), 3), coords).unwrap();
        Self::new(atom_types, coords_3d)
    }
}

// ==================== 方案B：2D拓扑 + 3D几何（我们的方法）====================

/// 分子拓扑结构（2D连接信息）
#[derive(Clone, Debug)]
pub struct MolecularTopology {
    /// 原子类型索引
    pub atom_types: Vec<usize>,
    /// 化学键连接（原子索引对，无向）
    pub bonds: Vec<(usize, usize)>,
}

/// 分子几何结构（3D空间信息）
#[derive(Clone, Debug)]
pub struct MolecularGeometry {
    /// 3D坐标矩阵 [N, 3]
    pub coords_3d: Array2<f32>,
}

/// 方案B：2D拓扑 + 3D几何混合表示
/// 显式维护拓扑连接信息，保留更多结构先验
#[derive(Clone, Debug)]
pub struct Molecule2D3D {
    /// 2D拓扑结构
    pub topology: MolecularTopology,
    /// 3D几何结构
    pub geometry: MolecularGeometry,
}

impl MoleculeRepresentation for Molecule2D3D {
    fn atom_count(&self) -> usize {
        self.topology.atom_types.len()
    }

    fn to_3d_coords(&self) -> Array2<f32> {
        self.geometry.coords_3d.clone()
    }

    fn name(&self) -> &'static str {
        "Molecule2D3D (Proposed)"
    }

    fn memory_usage(&self) -> usize {
        self.topology.atom_types.len() * std::mem::size_of::<usize>()
            + self.topology.bonds.len() * std::mem::size_of::<(usize, usize)>()
            + self.geometry.coords_3d.len() * std::mem::size_of::<f32>()
    }
}

impl Molecule2D3D {
    /// 创建新的混合表示
    pub fn new(atom_types: Vec<usize>, bonds: Vec<(usize, usize)>, coords_3d: Array2<f32>) -> Self {
        assert_eq!(
            atom_types.len(),
            coords_3d.nrows(),
            "原子类型数量与坐标行数不匹配"
        );
        assert_eq!(coords_3d.ncols(), 3, "坐标必须是[N, 3]形状");

        // 验证键索引有效
        for &(i, j) in &bonds {
            assert!(i < atom_types.len(), "键索引 {} 超出范围", i);
            assert!(j < atom_types.len(), "键索引 {} 超出范围", j);
        }

        Self {
            topology: MolecularTopology { atom_types, bonds },
            geometry: MolecularGeometry { coords_3d },
        }
    }

    /// 从SDF数据构建混合表示
    pub fn from_sdf_data(
        atom_types: Vec<usize>,
        bonds: Vec<(usize, usize)>,
        coords: &[[f32; 3]],
    ) -> Self {
        let coords_flat: Vec<f32> = coords.iter().flat_map(|c| c.iter().copied()).collect();
        let coords_3d = Array2::from_shape_vec((coords.len(), 3), coords_flat).unwrap();
        Self::new(atom_types, bonds, coords_3d)
    }

    /// 获取相邻原子索引列表
    pub fn get_neighbors(&self, atom_idx: usize) -> Vec<usize> {
        self.topology
            .bonds
            .iter()
            .filter_map(|&(i, j)| {
                if i == atom_idx {
                    Some(j)
                } else if j == atom_idx {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// 获取化学键数量
    pub fn bond_count(&self) -> usize {
        self.topology.bonds.len()
    }
}

// ==================== 统一口袋结构 ====================

/// 统一蛋白口袋结构
#[derive(Clone, Debug)]
pub struct ProteinPocket {
    /// 口袋原子坐标 [N, 3]
    pub coords: Array2<f32>,
    /// 12维口袋特征嵌入
    pub pocket_embedding: Array1<f32>,
}

impl ProteinPocket {
    /// 创建新的蛋白口袋
    pub fn new(coords: Array2<f32>, pocket_embedding: Array1<f32>) -> Self {
        assert_eq!(pocket_embedding.len(), 12, "口袋嵌入必须是12维");
        Self {
            coords,
            pocket_embedding,
        }
    }

    /// 获取口袋原子数量
    pub fn atom_count(&self) -> usize {
        self.coords.nrows()
    }

    /// 计算口袋中心
    pub fn center(&self) -> Array1<f32> {
        self.coords.mean_axis(ndarray::Axis(0)).unwrap()
    }
}

// ==================== 表示转换器 ====================

/// 表示转换器：在不同表示方法之间转换
pub struct RepresentationConverter;

impl RepresentationConverter {
    /// 将3D表示转换为2D+3D表示（需要通过距离推断键）
    pub fn from_3d_to_2d3d(mol_3d: &Molecule3D, bond_cutoff: f32) -> Molecule2D3D {
        let n = mol_3d.atom_count();
        let mut bonds = Vec::new();

        // 通过距离阈值推断化学键
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = (0..3)
                    .map(|k| {
                        let d = mol_3d.coords_3d[[i, k]] - mol_3d.coords_3d[[j, k]];
                        d * d
                    })
                    .sum::<f32>();

                if dist_sq.sqrt() < bond_cutoff {
                    bonds.push((i, j));
                }
            }
        }

        Molecule2D3D::new(mol_3d.atom_types.clone(), bonds, mol_3d.coords_3d.clone())
    }

    /// 将2D+3D表示转换为纯3D表示（丢弃拓扑信息）
    pub fn from_2d3d_to_3d(mol_2d3d: &Molecule2D3D) -> Molecule3D {
        Molecule3D::new(
            mol_2d3d.topology.atom_types.clone(),
            mol_2d3d.geometry.coords_3d.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_molecule_3d_creation() {
        let atom_types = vec![0, 1, 2];
        let coords = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let mol = Molecule3D::new(atom_types, coords);

        assert_eq!(mol.atom_count(), 3);
        assert_eq!(mol.name(), "Molecule3D (Baseline)");
    }

    #[test]
    fn test_molecule_2d3d_creation() {
        let atom_types = vec![0, 1, 2];
        let bonds = vec![(0, 1), (1, 2)];
        let coords = arr2(&[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let mol = Molecule2D3D::new(atom_types, bonds, coords);

        assert_eq!(mol.atom_count(), 3);
        assert_eq!(mol.bond_count(), 2);
        assert_eq!(mol.name(), "Molecule2D3D (Proposed)");
    }

    #[test]
    fn test_neighbor_lookup() {
        let atom_types = vec![0, 1, 2, 3];
        let bonds = vec![(0, 1), (1, 2), (2, 3)];
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ]);
        let mol = Molecule2D3D::new(atom_types, bonds, coords);

        let neighbors = mol.get_neighbors(1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_representation_conversion() {
        let atom_types = vec![0, 1, 2];
        let coords = arr2(&[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let mol_3d = Molecule3D::new(atom_types.clone(), coords.clone());

        // 3D -> 2D3D
        let mol_2d3d = RepresentationConverter::from_3d_to_2d3d(&mol_3d, 2.0);
        assert_eq!(mol_2d3d.atom_count(), 3);
        assert_eq!(mol_2d3d.bond_count(), 2);

        // 2D3D -> 3D
        let mol_3d_back = RepresentationConverter::from_2d3d_to_3d(&mol_2d3d);
        assert_eq!(mol_3d_back.atom_count(), 3);
    }

    #[test]
    fn test_protein_pocket() {
        let coords = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let embedding = Array1::from_vec(vec![0.0; 12]);
        let pocket = ProteinPocket::new(coords, embedding);

        assert_eq!(pocket.atom_count(), 3);
        assert_eq!(pocket.pocket_embedding.len(), 12);
    }
}
