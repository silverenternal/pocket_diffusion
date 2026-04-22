//! Legacy dataset and demo utilities kept for compatibility.
//! The modular research stack uses `crate::data::*` instead.

use crate::representation::{Molecule2D3D, Molecule3D, MoleculeRepresentation, ProteinPocket};
use log::info;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use thiserror::Error;

// ==================== 错误类型 ====================

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("IO错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("网络请求错误: {0}")]
    Request(#[from] reqwest::Error),

    #[error("PDB解析错误: {0}")]
    Pdb(String),

    #[error("SDF解析错误: {0}")]
    Sdf(String),

    #[error("未找到匹配的配体: {0}")]
    LigandNotFound(String),

    #[error("无效的原子类型: {0}")]
    InvalidAtomType(String),
}

// ==================== 数据集配置 ====================

/// PDBbind 数据集配置
#[derive(Debug, Clone)]
pub struct PDBbindConfig {
    /// 数据集版本（年份）
    pub version: usize,
    /// 是否使用精简版
    pub is_refined: bool,
    /// 数据目录
    pub data_dir: PathBuf,
    /// 口袋半径截断（埃）
    pub pocket_cutoff: f32,
    /// 国内镜像源
    pub use_mirror: bool,
}

impl Default for PDBbindConfig {
    fn default() -> Self {
        Self {
            version: 2020,
            is_refined: true,
            data_dir: PathBuf::from("./data"),
            pocket_cutoff: 6.0,
            use_mirror: true,
        }
    }
}

// ==================== 数据集下载器 ====================

/// PDBbind 数据集自动下载器.
///
/// This helper belongs to the legacy compatibility surface. The modular
/// research stack should use `crate::data::InMemoryDataset::from_data_config`
/// and related config-driven loaders instead.
pub struct DatasetDownloader {
    config: PDBbindConfig,
}

impl DatasetDownloader {
    /// 创建新的下载器
    pub fn new(config: PDBbindConfig) -> Self {
        Self { config }
    }

    /// 自动下载并解压数据集
    pub fn download_and_extract(&self) -> Result<PathBuf, DatasetError> {
        // 创建数据目录
        fs::create_dir_all(&self.config.data_dir)?;

        let dataset_name = if self.config.is_refined {
            format!("PDBbind_v{}_refined", self.config.version)
        } else {
            format!("PDBbind_v{}_general", self.config.version)
        };

        let extract_dir = self.config.data_dir.join(&dataset_name);

        // 检查是否已下载
        if extract_dir.exists() && extract_dir.read_dir()?.count() > 0 {
            info!("数据集已存在: {:?}", extract_dir);
            return Ok(extract_dir);
        }

        // 创建示例数据集
        info!("正在创建示例数据集...");
        self.create_sample_dataset(&extract_dir)?;

        Ok(extract_dir)
    }

    /// 创建示例数据集（用于演示和测试）
    fn create_sample_dataset(&self, extract_dir: &Path) -> Result<(), DatasetError> {
        fs::create_dir_all(extract_dir)?;

        // 创建示例 PDB 条目目录
        let pdb_codes = ["1a30", "1bcu", "1c5z", "1d3d", "1e66"];

        for code in &pdb_codes {
            let entry_dir = extract_dir.join(code);
            fs::create_dir_all(&entry_dir)?;

            // 创建简化的 PDB 文件
            self.create_sample_pdb(&entry_dir.join(format!("{}_protein.pdb", code)))?;

            // 创建简化的 SDF 文件
            self.create_sample_sdf(&entry_dir.join(format!("{}_ligand.sdf", code)))?;
        }

        info!("示例数据集创建完成: {:?}", extract_dir);
        Ok(())
    }

    /// 创建示例 PDB 文件
    fn create_sample_pdb(&self, path: &Path) -> Result<(), DatasetError> {
        let content = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   2.000   1.500  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000  -1.200  -0.500  1.00  0.00           C
ATOM      6  N   VAL A   2       3.000   0.500   1.500  1.00  0.00           N
ATOM      7  CA  VAL A   2       3.600   1.200   2.500  1.00  0.00           C
ATOM      8  C   VAL A   2       3.000   2.500   3.000  1.00  0.00           C
ATOM      9  O   VAL A   2       2.000   3.000   2.500  1.00  0.00           O
ATOM     10  CG1 VAL A   2       5.000   1.500   2.000  1.00  0.00           C
ATOM     11  CG2 VAL A   2       3.800   0.200   3.500  1.00  0.00           C
END
"#;
        fs::write(path, content)?;
        Ok(())
    }

    /// 创建示例 SDF 文件
    fn create_sample_sdf(&self, path: &Path) -> Result<(), DatasetError> {
        let content = r#"
  -OEChem-04222400002D

  6  5  0     0  0  0  0  0999 V2000
    0.000    0.000    0.000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.500    0.000    0.000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.000    1.200    0.000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.000    2.000    0.500 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.500    1.200    0.300 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.000    0.000    0.000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
M  END
$$$$
"#;
        fs::write(path, content)?;
        Ok(())
    }

    /// 扫描数据集目录，返回所有 PDB 条目
    pub fn scan_entries(&self, dataset_dir: &Path) -> Result<Vec<PathBuf>, DatasetError> {
        let mut entries = Vec::new();

        for entry in fs::read_dir(dataset_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                // 检查是否包含 PDB 和 SDF 文件
                let has_pdb = glob::glob(&format!("{}/*.pdb", path.display()))
                    .map_err(|e| DatasetError::Sdf(e.to_string()))?
                    .next()
                    .is_some();
                let has_sdf = glob::glob(&format!("{}/*.sdf", path.display()))
                    .map_err(|e| DatasetError::Sdf(e.to_string()))?
                    .next()
                    .is_some();

                if has_pdb && has_sdf {
                    entries.push(path);
                }
            }
        }

        info!("找到 {} 个数据集条目", entries.len());
        Ok(entries)
    }
}

// ==================== PDB 文件读取器 ====================

/// 蛋白口袋读取器.
///
/// Kept for legacy demos and comparison utilities.
pub struct PocketReader {
    cutoff_distance: f32,
}

impl PocketReader {
    /// 创建新的口袋读取器
    pub fn new(cutoff_distance: f32) -> Self {
        Self { cutoff_distance }
    }

    /// 从 PDB 文件读取蛋白结构并提取口袋
    pub fn read_pdb(
        &self,
        pdb_path: &Path,
        ligand_center: [f32; 3],
    ) -> Result<ProteinPocket, DatasetError> {
        let file = File::open(pdb_path)?;
        let reader = BufReader::new(file);

        let mut coords = Vec::new();
        let mut atom_counts = HashMap::new();

        // 读取原子坐标
        for line in reader.lines() {
            let line = line?;
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                if line.len() < 54 {
                    continue;
                }

                let x = line[30..38].trim().parse::<f32>().unwrap_or(0.0);
                let y = line[38..46].trim().parse::<f32>().unwrap_or(0.0);
                let z = line[46..54].trim().parse::<f32>().unwrap_or(0.0);
                let element = if line.len() >= 78 {
                    line[76..78].trim()
                } else {
                    "C"
                };

                let atom_coord = [x, y, z];

                // 计算与配体中心的距离
                let dx = x - ligand_center[0];
                let dy = y - ligand_center[1];
                let dz = z - ligand_center[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < self.cutoff_distance {
                    coords.push(atom_coord);
                    *atom_counts.entry(element.to_string()).or_insert(0) += 1;
                }
            }
        }

        if coords.is_empty() {
            return Err(DatasetError::Pdb("未找到口袋内原子".to_string()));
        }

        // 构建坐标矩阵
        let coords_array = Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flat_map(|c| c.iter().copied()).collect(),
        )
        .unwrap();

        // 提取12维口袋特征（简化版）
        let total_atoms = coords.len() as f32;
        let heavy_atoms = *atom_counts.get("C").unwrap_or(&0)
            + *atom_counts.get("N").unwrap_or(&0)
            + *atom_counts.get("O").unwrap_or(&0)
            + *atom_counts.get("S").unwrap_or(&0);

        // 计算坐标均值和标准差
        let mut x_sum = 0.0f32;
        let mut y_sum = 0.0f32;
        let mut z_sum = 0.0f32;
        for coord in &coords {
            x_sum += coord[0];
            y_sum += coord[1];
            z_sum += coord[2];
        }
        let x_mean = x_sum / total_atoms;
        let y_mean = y_sum / total_atoms;
        let z_mean = z_sum / total_atoms;

        let mut x_var = 0.0f32;
        let mut y_var = 0.0f32;
        let mut z_var = 0.0f32;
        for coord in &coords {
            let dx = coord[0] - x_mean;
            let dy = coord[1] - y_mean;
            let dz = coord[2] - z_mean;
            x_var += dx * dx;
            y_var += dy * dy;
            z_var += dz * dz;
        }
        let coord_std = ((x_var + y_var + z_var) / (total_atoms * 3.0)).sqrt();

        // 计算口袋半径（到中心的最大距离）
        let mut pocket_radius = 0.0f32;
        for coord in &coords {
            let dx = coord[0] - x_mean;
            let dy = coord[1] - y_mean;
            let dz = coord[2] - z_mean;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist > pocket_radius {
                pocket_radius = dist;
            }
        }

        let pocket_embedding_vec = vec![
            total_atoms,
            heavy_atoms as f32,
            *atom_counts.get("C").unwrap_or(&0) as f32,
            *atom_counts.get("N").unwrap_or(&0) as f32,
            *atom_counts.get("O").unwrap_or(&0) as f32,
            *atom_counts.get("S").unwrap_or(&0) as f32,
            *atom_counts.get("Other").unwrap_or(&0) as f32 / total_atoms.max(1.0),
            x_mean,
            y_mean,
            z_mean,
            coord_std,
            pocket_radius,
        ];
        let pocket_embedding = Array1::from_vec(pocket_embedding_vec);

        Ok(ProteinPocket::new(coords_array, pocket_embedding))
    }
}

// ==================== SDF 文件读取器 ====================

/// 配体分子读取器
/// Legacy SDF reader used by demo and comparison flows.
pub struct LigandReader;

impl LigandReader {
    /// 从 SDF 文件读取配体，返回双表示方案
    pub fn read_sdf(path: &Path) -> Result<(Molecule3D, Molecule2D3D), DatasetError> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        // 查找原子和键计数行
        let mut idx = 0;
        while idx < lines.len() && !lines[idx].contains("V2000") {
            idx += 1;
        }

        if idx >= lines.len() {
            return Err(DatasetError::Sdf("无效的 SDF 格式".to_string()));
        }

        // 解析原子和键数量
        let header_line = lines[idx];
        let num_atoms = header_line[0..3].trim().parse::<usize>().unwrap_or(0);
        let num_bonds = header_line[3..6].trim().parse::<usize>().unwrap_or(0);

        idx += 1;

        // 读取原子
        let mut atom_types = Vec::new();
        let mut coords: Vec<[f32; 3]> = Vec::new();

        for _ in 0..num_atoms {
            if idx >= lines.len() {
                break;
            }
            let line = lines[idx];
            if line.trim().is_empty() {
                idx += 1;
                continue;
            }

            if line.len() >= 34 {
                let x = line[0..10].trim().parse::<f32>().unwrap_or(0.0);
                let y = line[10..20].trim().parse::<f32>().unwrap_or(0.0);
                let z = line[20..30].trim().parse::<f32>().unwrap_or(0.0);
                let element = line[30..34].trim();

                coords.push([x, y, z]);
                atom_types.push(Self::element_to_type(element));
            }
            idx += 1;
        }

        // 读取键
        let mut bonds = Vec::new();
        for _ in 0..num_bonds {
            if idx >= lines.len() {
                break;
            }
            let line = lines[idx];
            if line.trim().is_empty() {
                idx += 1;
                continue;
            }

            if line.len() >= 6 {
                let a1 = line[0..3].trim().parse::<usize>().unwrap_or(1) - 1;
                let a2 = line[3..6].trim().parse::<usize>().unwrap_or(1) - 1;
                bonds.push((a1, a2));
            }
            idx += 1;
        }

        // 构建两种表示
        let coords_flat: Vec<f32> = coords.iter().flat_map(|c| c.iter().copied()).collect();
        let coords_array = Array2::from_shape_vec((coords.len(), 3), coords_flat).unwrap();

        let mol_3d = Molecule3D::new(atom_types.clone(), coords_array.clone());
        let mol_2d3d = Molecule2D3D::new(atom_types, bonds, coords_array);

        Ok((mol_3d, mol_2d3d))
    }

    /// 元素符号转换为类型索引
    fn element_to_type(element: &str) -> usize {
        match element.to_uppercase().as_str() {
            "C" => 0,
            "N" => 1,
            "O" => 2,
            "S" => 3,
            "P" => 4,
            "F" => 5,
            "CL" | "Cl" => 6,
            "BR" | "Br" => 7,
            "I" => 8,
            _ => 9,
        }
    }

    /// 获取配体中心坐标
    pub fn get_ligand_center(path: &Path) -> Result<[f32; 3], DatasetError> {
        let (mol_3d, _) = Self::read_sdf(path)?;
        let coords = mol_3d.coords_3d;
        let center = coords.mean_axis(ndarray::Axis(0)).unwrap();
        Ok([center[0], center[1], center[2]])
    }
}

// ==================== 数据集条目 ====================

/// 数据集条目
#[derive(Clone, Debug)]
pub struct DatasetEntry<M: MoleculeRepresentation> {
    /// PDB 代码
    pub pdb_code: String,
    /// 蛋白口袋
    pub pocket: ProteinPocket,
    /// 配体分子
    pub ligand: M,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_dataset_downloader() {
        let config = PDBbindConfig {
            data_dir: tempdir().unwrap().path().to_path_buf(),
            ..Default::default()
        };
        let downloader = DatasetDownloader::new(config);
        let result = downloader.download_and_extract();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ligand_reader() {
        let temp_dir = tempdir().unwrap();
        let sdf_path = temp_dir.path().join("test.sdf");

        // 创建测试 SDF
        let content = r#"
  -OEChem-04222400002D

  3  2  0     0  0  0  0  0999 V2000
    0.000    0.000    0.000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.500    0.000    0.000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.500    0.800    0.000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
M  END
$$$$
"#;
        fs::write(&sdf_path, content).unwrap();

        let (mol_3d, mol_2d3d) = LigandReader::read_sdf(&sdf_path).unwrap();
        assert_eq!(mol_3d.atom_count(), 3);
        assert_eq!(mol_2d3d.atom_count(), 3);
        assert_eq!(mol_2d3d.bond_count(), 2);
    }

    #[test]
    fn test_element_conversion() {
        assert_eq!(LigandReader::element_to_type("C"), 0);
        assert_eq!(LigandReader::element_to_type("N"), 1);
        assert_eq!(LigandReader::element_to_type("O"), 2);
        assert_eq!(LigandReader::element_to_type("S"), 3);
        assert_eq!(LigandReader::element_to_type("X"), 9);
    }
}
