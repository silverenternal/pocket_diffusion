//! 核心数据结构定义

use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

/// 从 slice 创建 Tensor (tch 0.23 兼容) - 浮点类型转换为 Float (f32)
pub fn tensor_from_slice<T: tch::kind::Element>(data: &[T]) -> Tensor {
    let tensor = Tensor::f_from_slice(data).unwrap_or_else(|_| Tensor::new());
    // 只对浮点类型转换为 Float，保持整数类型不变
    if tensor.kind() == Kind::Double {
        tensor.totype(Kind::Float)
    } else {
        tensor
    }
}

/// 原子类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AtomType {
    #[default]
    Carbon, // C
    Nitrogen, // N
    Oxygen,   // O
    Sulfur,   // S
    Hydrogen, // H
    Other,
}

/// 原子结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    /// 3D坐标 (x, y, z)
    pub coords: [f64; 3],
    /// 原子类型
    pub atom_type: AtomType,
    /// 原子序号
    pub index: usize,
}

/// 蛋白口袋结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pocket {
    /// 口袋中的所有原子
    pub atoms: Vec<Atom>,
    /// 口袋名称
    pub name: String,
}

/// 配体/小分子结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ligand {
    /// 配体中的所有原子
    pub atoms: Vec<Atom>,
    /// 化学键连接 (原子索引对)
    pub bonds: Vec<(usize, usize)>,
    /// Morgan指纹ECFP4 (2048位)
    pub fingerprint: Option<Vec<f32>>,
}

/// 12维蛋白口袋嵌入向量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketEmbedding {
    /// 总原子数
    pub total_atoms: f64,
    /// 重原子数（非氢原子数）
    pub heavy_atoms: f64,
    /// 碳原子数
    pub carbon_count: f64,
    /// 氮原子数
    pub nitrogen_count: f64,
    /// 氧原子数
    pub oxygen_count: f64,
    /// 硫原子数
    pub sulfur_count: f64,
    /// 其他元素占比
    pub other_ratio: f64,
    /// X轴坐标均值
    pub x_mean: f64,
    /// Y轴坐标均值
    pub y_mean: f64,
    /// Z轴坐标均值
    pub z_mean: f64,
    /// 坐标标准差
    pub coord_std: f64,
    /// 口袋半径
    pub pocket_radius: f64,
}

/// 蛋白-配体联合图
#[derive(Debug)]
pub struct ProteinLigandGraph {
    /// 节点坐标矩阵 (N x 3)
    pub node_coords: Tensor,
    /// 节点类型 (one-hot编码)
    pub node_types: Tensor,
    /// 边索引矩阵 (2 x E)
    pub edge_index: Tensor,
    /// 边特征矩阵 (E x num_rbf)
    pub edge_features: Tensor,
    /// 蛋白原子数量
    pub num_protein_atoms: usize,
    /// 配体原子数量
    pub num_ligand_atoms: usize,
}

impl Clone for ProteinLigandGraph {
    fn clone(&self) -> Self {
        Self {
            node_coords: self.node_coords.shallow_clone(),
            node_types: self.node_types.shallow_clone(),
            edge_index: self.edge_index.shallow_clone(),
            edge_features: self.edge_features.shallow_clone(),
            num_protein_atoms: self.num_protein_atoms,
            num_ligand_atoms: self.num_ligand_atoms,
        }
    }
}

/// 候选分子
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMolecule {
    /// 配体结构
    pub ligand: Ligand,
    /// 亲和力评分 (kcal/mol)
    pub affinity_score: Option<f64>,
    /// QED药物样性评分
    pub qed_score: Option<f64>,
    /// 可合成性评分SA
    pub sa_score: Option<f64>,
}

/// 生成结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// 所有候选分子
    pub candidates: Vec<CandidateMolecule>,
    /// 筛选后的高分分子
    pub top_candidates: Vec<CandidateMolecule>,
}

impl PocketEmbedding {
    /// 转换为张量
    pub fn to_tensor(&self) -> Tensor {
        let data = [
            self.total_atoms,
            self.heavy_atoms,
            self.carbon_count,
            self.nitrogen_count,
            self.oxygen_count,
            self.sulfur_count,
            self.other_ratio,
            self.x_mean,
            self.y_mean,
            self.z_mean,
            self.coord_std,
            self.pocket_radius,
        ];
        tensor_from_slice(&data)
    }

    /// 转换为Vec<f64>
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.total_atoms,
            self.heavy_atoms,
            self.carbon_count,
            self.nitrogen_count,
            self.oxygen_count,
            self.sulfur_count,
            self.other_ratio,
            self.x_mean,
            self.y_mean,
            self.z_mean,
            self.coord_std,
            self.pocket_radius,
        ]
    }

    /// Z-score标准化
    pub fn zscore_normalize(&self, means: &[f64; 12], stds: &[f64; 12]) -> Self {
        let original = self.to_vec();
        let normalized: Vec<f64> = original
            .iter()
            .zip(means.iter())
            .zip(stds.iter())
            .map(|((&val, &mean), &std)| if std > 1e-8 { (val - mean) / std } else { 0.0 })
            .collect();

        Self {
            total_atoms: normalized[0],
            heavy_atoms: normalized[1],
            carbon_count: normalized[2],
            nitrogen_count: normalized[3],
            oxygen_count: normalized[4],
            sulfur_count: normalized[5],
            other_ratio: normalized[6],
            x_mean: normalized[7],
            y_mean: normalized[8],
            z_mean: normalized[9],
            coord_std: normalized[10],
            pocket_radius: normalized[11],
        }
    }
}

impl Default for PocketEmbedding {
    fn default() -> Self {
        Self {
            total_atoms: 0.0,
            heavy_atoms: 0.0,
            carbon_count: 0.0,
            nitrogen_count: 0.0,
            oxygen_count: 0.0,
            sulfur_count: 0.0,
            other_ratio: 0.0,
            x_mean: 0.0,
            y_mean: 0.0,
            z_mean: 0.0,
            coord_std: 0.0,
            pocket_radius: 0.0,
        }
    }
}

impl AtomType {
    /// 判断是否为重原子（非氢）
    pub fn is_heavy(&self) -> bool {
        !matches!(self, AtomType::Hydrogen)
    }

    /// 转换为one-hot索引
    pub fn to_index(&self) -> i64 {
        match self {
            AtomType::Carbon => 0,
            AtomType::Nitrogen => 1,
            AtomType::Oxygen => 2,
            AtomType::Sulfur => 3,
            AtomType::Hydrogen => 4,
            AtomType::Other => 5,
        }
    }
}
