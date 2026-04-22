//! 蛋白口袋特征提取模块
//! 实现12维口袋特征向量提取

use crate::types::{Atom, AtomType, Pocket, PocketEmbedding};

/// 口袋特征提取器
pub struct PocketFeatureExtractor;

impl PocketFeatureExtractor {
    /// 从口袋原子提取12维特征向量
    pub fn extract(pocket: &Pocket) -> PocketEmbedding {
        let atoms = &pocket.atoms;
        let total_atoms = atoms.len() as f64;

        // 各类原子计数
        let mut heavy_atoms = 0;
        let mut carbon_count = 0;
        let mut nitrogen_count = 0;
        let mut oxygen_count = 0;
        let mut sulfur_count = 0;
        let mut other_count = 0;

        for atom in atoms {
            if atom.atom_type.is_heavy() {
                heavy_atoms += 1;
            }
            match atom.atom_type {
                AtomType::Carbon => carbon_count += 1,
                AtomType::Nitrogen => nitrogen_count += 1,
                AtomType::Oxygen => oxygen_count += 1,
                AtomType::Sulfur => sulfur_count += 1,
                AtomType::Hydrogen => {}
                AtomType::Other => other_count += 1,
            }
        }

        // 其他元素占比
        let other_ratio = if total_atoms > 0.0 {
            other_count as f64 / total_atoms
        } else {
            0.0
        };

        // 坐标统计
        let (x_mean, y_mean, z_mean) = Self::compute_coord_means(atoms);
        let coord_std = Self::compute_coord_std(atoms, x_mean, y_mean, z_mean);

        // 口袋半径（质心到最远原子的距离）
        let pocket_radius = Self::compute_pocket_radius(atoms, x_mean, y_mean, z_mean);

        PocketEmbedding {
            total_atoms,
            heavy_atoms: heavy_atoms as f64,
            carbon_count: carbon_count as f64,
            nitrogen_count: nitrogen_count as f64,
            oxygen_count: oxygen_count as f64,
            sulfur_count: sulfur_count as f64,
            other_ratio,
            x_mean,
            y_mean,
            z_mean,
            coord_std,
            pocket_radius,
        }
    }

    /// 计算坐标均值
    fn compute_coord_means(atoms: &[Atom]) -> (f64, f64, f64) {
        if atoms.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let n = atoms.len() as f64;
        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut z_sum = 0.0;

        for atom in atoms {
            x_sum += atom.coords[0];
            y_sum += atom.coords[1];
            z_sum += atom.coords[2];
        }

        (x_sum / n, y_sum / n, z_sum / n)
    }

    /// 计算坐标标准差
    fn compute_coord_std(atoms: &[Atom], x_mean: f64, y_mean: f64, z_mean: f64) -> f64 {
        if atoms.is_empty() {
            return 0.0;
        }

        let n = atoms.len() as f64;
        let mut sum = 0.0;

        for atom in atoms {
            let dx = atom.coords[0] - x_mean;
            let dy = atom.coords[1] - y_mean;
            let dz = atom.coords[2] - z_mean;
            sum += dx * dx + dy * dy + dz * dz;
        }

        (sum / n).sqrt()
    }

    /// 计算口袋半径
    fn compute_pocket_radius(atoms: &[Atom], x_mean: f64, y_mean: f64, z_mean: f64) -> f64 {
        if atoms.is_empty() {
            return 0.0;
        }

        let mut max_dist = 0.0;

        for atom in atoms {
            let dx = atom.coords[0] - x_mean;
            let dy = atom.coords[1] - y_mean;
            let dz = atom.coords[2] - z_mean;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist > max_dist {
                max_dist = dist;
            }
        }

        max_dist
    }

    /// 生成默认的标准化参数（基于PDBbind统计值）
    pub fn default_normalization_params() -> ([f64; 12], [f64; 12]) {
        // PDBbind数据集的统计均值
        let means = [
            150.0, // total_atoms
            120.0, // heavy_atoms
            80.0,  // carbon_count
            25.0,  // nitrogen_count
            15.0,  // oxygen_count
            3.0,   // sulfur_count
            0.02,  // other_ratio
            0.0,   // x_mean
            0.0,   // y_mean
            0.0,   // z_mean
            8.0,   // coord_std
            12.0,  // pocket_radius
        ];

        // PDBbind数据集的统计标准差
        let stds = [
            50.0, // total_atoms
            40.0, // heavy_atoms
            30.0, // carbon_count
            10.0, // nitrogen_count
            8.0,  // oxygen_count
            2.0,  // sulfur_count
            0.03, // other_ratio
            10.0, // x_mean
            10.0, // y_mean
            10.0, // z_mean
            3.0,  // coord_std
            5.0,  // pocket_radius
        ];

        (means, stds)
    }
}

/// 创建示例口袋（用于测试）
pub fn create_example_prrsv_pocket() -> Pocket {
    // PRRSV N蛋白口袋的简化示例原子
    let atoms = vec![
        Atom {
            coords: [0.0, 0.0, 0.0],
            atom_type: AtomType::Carbon,
            index: 0,
        },
        Atom {
            coords: [1.5, 0.0, 0.0],
            atom_type: AtomType::Carbon,
            index: 1,
        },
        Atom {
            coords: [2.0, 1.2, 0.0],
            atom_type: AtomType::Nitrogen,
            index: 2,
        },
        Atom {
            coords: [0.7, 2.0, 0.5],
            atom_type: AtomType::Carbon,
            index: 3,
        },
        Atom {
            coords: [-0.5, 1.2, 0.3],
            atom_type: AtomType::Oxygen,
            index: 4,
        },
        Atom {
            coords: [3.0, 0.5, 1.0],
            atom_type: AtomType::Carbon,
            index: 5,
        },
        Atom {
            coords: [3.8, -0.3, 1.5],
            atom_type: AtomType::Sulfur,
            index: 6,
        },
        Atom {
            coords: [2.5, -1.0, -0.5],
            atom_type: AtomType::Carbon,
            index: 7,
        },
        Atom {
            coords: [1.0, -1.5, -0.8],
            atom_type: AtomType::Nitrogen,
            index: 8,
        },
        Atom {
            coords: [-0.5, -0.8, -1.0],
            atom_type: AtomType::Carbon,
            index: 9,
        },
        Atom {
            coords: [-1.5, 0.2, -0.5],
            atom_type: AtomType::Oxygen,
            index: 10,
        },
        Atom {
            coords: [0.0, 0.5, 1.0],
            atom_type: AtomType::Hydrogen,
            index: 11,
        },
        Atom {
            coords: [1.5, -0.5, 1.2],
            atom_type: AtomType::Hydrogen,
            index: 12,
        },
    ];

    Pocket {
        atoms,
        name: "PRRSV_N_protein_binding_pocket".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pocket_feature_extraction() {
        let pocket = create_example_prrsv_pocket();
        let embedding = PocketFeatureExtractor::extract(&pocket);

        assert_eq!(embedding.total_atoms, 13.0);
        assert_eq!(embedding.heavy_atoms, 11.0);
        assert!(embedding.pocket_radius > 0.0);
    }

    #[test]
    fn test_empty_pocket() {
        let pocket = Pocket {
            atoms: vec![],
            name: "empty".to_string(),
        };
        let embedding = PocketFeatureExtractor::extract(&pocket);

        assert_eq!(embedding.total_atoms, 0.0);
        assert_eq!(embedding.pocket_radius, 0.0);
    }

    #[test]
    fn test_normalization() {
        let pocket = create_example_prrsv_pocket();
        let embedding = PocketFeatureExtractor::extract(&pocket);
        let (means, stds) = PocketFeatureExtractor::default_normalization_params();
        let normalized = embedding.zscore_normalize(&means, &stds);

        // 归一化后值应该在合理范围内
        assert!(normalized.total_atoms < 10.0);
    }
}
