//! 信息解耦Transformer核心类型定义
//! Information-theoretic disentanglement core types

use serde::{Deserialize, Serialize};
use tch::Tensor;

// ==================== 分支输出 ====================

/// 拓扑分支输出 (2D拓扑信息)
#[derive(Debug)]
pub struct TopoBranchOutput {
    /// 节点特征嵌入 [num_atoms, hidden_dim]
    pub node_features: Tensor,
    /// 邻接矩阵对数几率 [num_atoms, num_atoms]
    pub adjacency_logits: Tensor,
    /// 分支全局嵌入 [hidden_dim]
    pub global_embedding: Tensor,
}

impl Clone for TopoBranchOutput {
    fn clone(&self) -> Self {
        Self {
            node_features: self.node_features.shallow_clone(),
            adjacency_logits: self.adjacency_logits.shallow_clone(),
            global_embedding: self.global_embedding.shallow_clone(),
        }
    }
}

/// 几何分支输出 (3D空间信息)
#[derive(Debug)]
pub struct GeoBranchOutput {
    /// 3D坐标更新 [num_atoms, 3]
    pub coords: Tensor,
    /// 几何特征嵌入 [num_atoms, hidden_dim]
    pub node_features: Tensor,
    /// 分支全局嵌入 [hidden_dim]
    pub global_embedding: Tensor,
}

impl Clone for GeoBranchOutput {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.shallow_clone(),
            node_features: self.node_features.shallow_clone(),
            global_embedding: self.global_embedding.shallow_clone(),
        }
    }
}

/// 口袋分支输出 (口袋条件信息)
#[derive(Debug)]
pub struct PocketBranchOutput {
    /// 口袋特征嵌入 [pocket_dim]
    pub pocket_embedding: Tensor,
    /// 交叉注意力权重 [num_atoms, pocket_size]
    pub cross_attention_weights: Tensor,
    /// 分支全局嵌入 [hidden_dim]
    pub global_embedding: Tensor,
}

impl Clone for PocketBranchOutput {
    fn clone(&self) -> Self {
        Self {
            pocket_embedding: self.pocket_embedding.shallow_clone(),
            cross_attention_weights: self.cross_attention_weights.shallow_clone(),
            global_embedding: self.global_embedding.shallow_clone(),
        }
    }
}

/// 所有分支的联合输出
#[derive(Debug, Clone)]
pub struct DisentangledBranchesOutput {
    /// 拓扑分支输出
    pub topo: TopoBranchOutput,
    /// 几何分支输出
    pub geo: GeoBranchOutput,
    /// 口袋分支输出
    pub pocket: PocketBranchOutput,
    /// 路由概率 [p_topo, p_geo, p_pocket]
    pub routing_probs: [f32; 3],
}

// ==================== 互信息估计结果 ====================

/// 分支间互信息估计结果
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MutualInformationEstimate {
    /// I(Topo; Geo) 拓扑与几何的互信息
    pub mi_topo_geo: f64,
    /// I(Topo; Pocket) 拓扑与口袋的互信息
    pub mi_topo_pocket: f64,
    /// I(Geo; Pocket) 几何与口袋的互信息
    pub mi_geo_pocket: f64,
}

impl MutualInformationEstimate {
    /// 计算平均互信息
    pub fn average(&self) -> f64 {
        (self.mi_topo_geo + self.mi_topo_pocket + self.mi_geo_pocket) / 3.0
    }

    /// 计算解耦度: 1 - 平均互信息 (归一化到[0,1])
    pub fn disentanglement_degree(&self) -> f64 {
        // 使用sigmoid归一化互信息到[0,1]区间
        // 互信息越大表示信息共享越多，解耦度越低
        let avg_mi = self.average();
        let normalized = 1.0 / (1.0 + (-avg_mi).exp());
        1.0 - normalized
    }
}

// ==================== 冗余度估计结果 ====================

/// 分支信息冗余度
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BranchRedundancy {
    /// 拓扑分支冗余度
    pub topo_redundancy: f64,
    /// 几何分支冗余度
    pub geo_redundancy: f64,
    /// 口袋分支冗余度
    pub pocket_redundancy: f64,
}

impl BranchRedundancy {
    /// 计算平均冗余度
    pub fn average(&self) -> f64 {
        (self.topo_redundancy + self.geo_redundancy + self.pocket_redundancy) / 3.0
    }
}

// ==================== 损失函数组件 ====================

/// 所有损失组件
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LossComponents {
    /// 任务损失 L_task
    pub task_loss: f64,
    /// 解耦损失 L_disentangle (MI最小化)
    pub disentangle_loss: f64,
    /// 冗余损失 L_redundancy
    pub redundancy_loss: f64,
    /// 一致性损失 L_consistency
    pub consistency_loss: f64,
}

impl LossComponents {
    /// 计算总损失
    /// L_total = α*L_task + β*L_disentangle + γ*L_redundancy + δ*L_consistency
    pub fn total(&self, alpha: f64, beta: f64, gamma: f64, delta: f64) -> f64 {
        alpha * self.task_loss
            + beta * self.disentangle_loss
            + gamma * self.redundancy_loss
            + delta * self.consistency_loss
    }
}

// ==================== 训练阶段 ====================

/// 训练阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingPhase {
    /// 预热阶段: 只训练任务损失
    Warmup,
    /// 解耦阶段: 加入互信息最小化损失
    Disentanglement,
    /// 精炼阶段: 加入冗余损失和一致性损失
    Refinement,
}

impl TrainingPhase {
    /// 根据当前步数获取训练阶段
    pub fn from_step(step: usize) -> Self {
        if step < 1000 {
            TrainingPhase::Warmup
        } else if step < 6000 {
            TrainingPhase::Disentanglement
        } else {
            TrainingPhase::Refinement
        }
    }

    /// 获取当前阶段的损失权重
    pub fn loss_weights(&self) -> (f64, f64, f64, f64) {
        match self {
            // 预热阶段: 只训练任务损失
            TrainingPhase::Warmup => (1.0, 0.0, 0.0, 0.0),
            // 解耦阶段: 任务损失 + 解耦损失
            TrainingPhase::Disentanglement => (1.0, 0.1, 0.0, 0.0),
            // 精炼阶段: 全部损失
            TrainingPhase::Refinement => (1.0, 0.05, 0.01, 0.01),
        }
    }
}

// ==================== 训练指标 ====================

/// 单步训练指标
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// 当前训练步数
    pub step: usize,
    /// 当前训练阶段
    pub phase: TrainingPhase,
    /// 损失组件
    pub losses: LossComponents,
    /// 互信息估计
    pub mutual_information: MutualInformationEstimate,
    /// 冗余度估计
    pub redundancy: BranchRedundancy,
    /// 解耦度 [0, 1]
    pub disentanglement_degree: f64,
    /// 该步耗时（毫秒）
    pub step_time_ms: f64,
    /// 每秒处理样本数
    pub samples_per_second: f64,
    /// 峰值内存使用（字节）
    pub peak_memory_bytes: u64,
}

// ==================== 实验指标 ====================

/// 完整实验指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    /// 方法名称
    pub method_name: String,
    /// 最终解耦度
    pub final_disentanglement_degree: f64,
    /// 最终平均冗余度
    pub final_redundancy: f64,
    /// 平均训练速度 (samples/sec)
    pub avg_training_speed: f64,
    /// 峰值显存使用 (MB)
    pub peak_memory_mb: f64,
    /// 平均QED分数
    pub avg_qed: Option<f64>,
    /// 平均SA分数
    pub avg_sa: Option<f64>,
    /// 平均结合亲和力 (kcal/mol)
    pub avg_binding_affinity: Option<f64>,
}

// ==================== 配置 ====================

/// 信息解耦Transformer配置
#[derive(Debug, Clone)]
pub struct DisentangleConfig {
    /// 隐藏层维度
    pub hidden_dim: usize,
    /// 输入特征维度
    pub in_dim: usize,
    /// 口袋特征维度
    pub pocket_dim: usize,
    /// MINE网络隐藏层维度
    pub mine_hidden_dim: usize,
    /// 是否使用纯3D基线模式（关闭自动解耦）
    pub use_baseline_3d_only: bool,
    /// 注意力头数
    pub num_heads: usize,
    /// Transformer层数
    pub num_layers: usize,
    /// 预热步数
    pub warmup_steps: usize,
    /// 解耦步数
    pub disentangle_steps: usize,
    /// 精炼步数
    pub refinement_steps: usize,
    /// 日志输出间隔
    pub log_interval: usize,
}

impl Default for DisentangleConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            in_dim: 10,
            pocket_dim: 12,
            mine_hidden_dim: 64,
            use_baseline_3d_only: false,
            num_heads: 8,
            num_layers: 4,
            warmup_steps: 1000,
            disentangle_steps: 5000,
            refinement_steps: 10000,
            log_interval: 100,
        }
    }
}
