//! 损失函数模块
//! Loss Functions Module
//!
//! 实现完整的信息解耦损失函数：
//! L_total = α*L_task + β*L_disentangle + γ*L_redundancy + δ*L_consistency

use tch::{nn, Device, Kind, Tensor};

use crate::disentangle::mine::BranchMutualInformation;
use crate::disentangle::redundancy::RedundancyCalculator;
use crate::disentangle::types::{DisentangledBranchesOutput, LossComponents, TrainingPhase};

/// 任务损失计算器
///
/// 支持交叉熵、均方误差、L1损失等多种任务损失
pub struct TaskLoss {
    device: Device,
}

impl TaskLoss {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// 计算原子类型预测的交叉熵损失
    pub fn atom_type_loss(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let target_idx = targets.unsqueeze(-1);
        let nll = log_probs.gather(-1, &target_idx, false).squeeze_dim(-1);
        -nll.mean(Kind::Float)
    }

    /// 计算坐标回归的MSE损失
    pub fn coordinate_mse_loss(&self, pred_coords: &Tensor, true_coords: &Tensor) -> Tensor {
        (pred_coords - true_coords)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
    }

    /// 计算键类型预测损失
    pub fn bond_type_loss(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let target_idx = targets.unsqueeze(-1);
        let nll = log_probs.gather(-1, &target_idx, false).squeeze_dim(-1);
        -nll.mean(Kind::Float)
    }

    /// 计算口袋条件生成任务损失
    ///
    /// 综合原子类型、坐标、键类型预测
    pub fn compute_task_loss(
        &self,
        pred_atom_types: &Tensor,
        true_atom_types: &Tensor,
        pred_coords: &Tensor,
        true_coords: &Tensor,
        pred_bond_types: Option<&Tensor>,
        true_bond_types: Option<&Tensor>,
    ) -> Tensor {
        let atom_loss = self.atom_type_loss(pred_atom_types, true_atom_types);
        let coord_loss = self.coordinate_mse_loss(pred_coords, true_coords);

        let bond_loss = if let (Some(pl), Some(tl)) = (pred_bond_types, true_bond_types) {
            self.bond_type_loss(pl, tl)
        } else {
            Tensor::from(0.0)
                .to_device(self.device)
                .to_kind(Kind::Float)
        };

        atom_loss + coord_loss + bond_loss
    }
}

/// 解耦损失计算器
///
/// 使用MINE估计分支间互信息并最小化
/// L_disentangle = MI(Topo; Geo) + MI(Topo; Pocket) + MI(Geo; Pocket)
pub struct DisentanglementLoss {
    branch_mi: BranchMutualInformation,
    device: Device,
}

impl DisentanglementLoss {
    pub fn new(vs: &nn::Path, hidden_dim: usize, mine_hidden_dim: usize) -> Self {
        let device = vs.device();
        let branch_mi = BranchMutualInformation::new(vs, hidden_dim, mine_hidden_dim);
        Self { branch_mi, device }
    }

    /// 计算解耦损失
    ///
    /// 最小化所有分支对之间的互信息之和
    pub fn compute_disentangle_loss(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> Tensor {
        let (mi_topo_geo, mi_topo_pocket, mi_geo_pocket) =
            self.branch_mi.estimate_all(topo_emb, geo_emb, pocket_emb);

        // 确保互信息为非负（MINE下界可能为负）
        let mi1 = mi_topo_geo.relu();
        let mi2 = mi_topo_pocket.relu();
        let mi3 = mi_geo_pocket.relu();

        mi1 + mi2 + mi3
    }

    /// 获取各分支对的互信息值（用于监控）
    pub fn get_mi_values(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> (f64, f64, f64) {
        let (mi1, mi2, mi3) = self.branch_mi.estimate_all(topo_emb, geo_emb, pocket_emb);
        (
            mi1.double_value(&[]),
            mi2.double_value(&[]),
            mi3.double_value(&[]),
        )
    }
}

/// 冗余损失计算器
///
/// 使用信息熵量化表示中的冗余信息并最小化
/// L_redundancy = mean(R_topo, R_geo, R_pocket)
pub struct RedundancyLoss {
    calculator: RedundancyCalculator,
    device: Device,
}

impl RedundancyLoss {
    pub fn new(device: Device, use_entropy_based: bool) -> Self {
        let calculator = RedundancyCalculator::new(device, use_entropy_based, 50);
        Self { calculator, device }
    }

    /// 计算冗余损失
    pub fn compute_redundancy_loss(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> Tensor {
        let (r_topo, r_geo, r_pocket) = self.calculator.compute_all(topo_emb, geo_emb, pocket_emb);

        let avg_redundancy = (r_topo + r_geo + r_pocket) / 3.0;

        Tensor::from(avg_redundancy)
            .to_device(self.device)
            .to_kind(Kind::Float)
    }

    /// 获取各分支的冗余度值
    pub fn get_redundancy_values(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> (f64, f64, f64) {
        self.calculator.compute_all(topo_emb, geo_emb, pocket_emb)
    }
}

/// 一致性损失计算器
///
/// 使用对比学习确保不同分支对同一分子的表示在语义上一致
/// L_consistency = InfoNCE(branch1, branch2) + InfoNCE(branch1, branch3) + InfoNCE(branch2, branch3)
pub struct ConsistencyLoss {
    temperature: f64,
    device: Device,
}

impl ConsistencyLoss {
    pub fn new(device: Device, temperature: f64) -> Self {
        Self {
            temperature,
            device,
        }
    }

    /// 计算两个表示之间的InfoNCE损失
    ///
    /// 正样本对：同一分子的不同分支视图
    /// 负样本对：同一批次中不同分子的分支视图
    fn info_nce_loss(&self, z1: &Tensor, z2: &Tensor) -> Tensor {
        let batch_size = z1.size()[0];

        // L2归一化
        let z1_sq_sum = z1
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[-1][..], true, Kind::Float);
        let z1_norm = z1 / z1_sq_sum.sqrt().clamp_min(1e-8);
        let z2_sq_sum = z2
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[-1][..], true, Kind::Float);
        let z2_norm = z2 / z2_sq_sum.sqrt().clamp_min(1e-8);

        // 计算相似度矩阵 [batch_size, batch_size]
        let similarity_matrix = z1_norm.matmul(&z2_norm.transpose(0, 1)) / self.temperature;

        // 对角线是正样本，其他是负样本
        let logits = similarity_matrix.log_softmax(1, Kind::Float);

        // 只取对角线上的正样本损失
        let positive_loss = -logits.diag(0).mean(Kind::Float);

        positive_loss
    }

    /// 计算所有分支对的一致性损失
    pub fn compute_consistency_loss(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> Tensor {
        let loss_topo_geo = self.info_nce_loss(topo_emb, geo_emb);
        let loss_topo_pocket = self.info_nce_loss(topo_emb, pocket_emb);
        let loss_geo_pocket = self.info_nce_loss(geo_emb, pocket_emb);

        loss_topo_geo + loss_topo_pocket + loss_geo_pocket
    }
}

/// 完整损失计算器
///
/// 组合所有损失组件，根据训练阶段动态调整权重
/// L_total = α*L_task + β*L_disentangle + γ*L_redundancy + δ*L_consistency
pub struct TotalLoss {
    task_loss: TaskLoss,
    disentangle_loss: DisentanglementLoss,
    redundancy_loss: RedundancyLoss,
    consistency_loss: ConsistencyLoss,
    device: Device,
}

impl TotalLoss {
    pub fn new(
        vs: &nn::Path,
        hidden_dim: usize,
        mine_hidden_dim: usize,
        device: Device,
        consistency_temperature: f64,
        use_entropy_based: bool,
    ) -> Self {
        let task_loss = TaskLoss::new(device);
        let disentangle_loss = DisentanglementLoss::new(vs, hidden_dim, mine_hidden_dim);
        let redundancy_loss = RedundancyLoss::new(device, use_entropy_based);
        let consistency_loss = ConsistencyLoss::new(device, consistency_temperature);

        Self {
            task_loss,
            disentangle_loss,
            redundancy_loss,
            consistency_loss,
            device,
        }
    }

    /// 获取训练阶段对应的损失权重
    ///
    /// - Warmup: 只训练任务损失
    /// - Disentanglement: 引入解耦和冗余损失
    /// - Refinement: 所有损失同等重要
    pub fn get_loss_weights(&self, phase: TrainingPhase) -> (f64, f64, f64, f64) {
        match phase {
            TrainingPhase::Warmup => (1.0, 0.0, 0.0, 0.0),
            TrainingPhase::Disentanglement => (1.0, 0.5, 0.3, 0.2),
            TrainingPhase::Refinement => (1.0, 0.3, 0.2, 0.5),
        }
    }

    /// 计算完整的损失
    ///
    /// 返回各损失组件和总损失
    pub fn compute_total_loss(
        &self,
        outputs: &DisentangledBranchesOutput,
        true_atom_types: &Tensor,
        true_coords: &Tensor,
        true_bond_types: Option<&Tensor>,
        pred_bond_types: Option<&Tensor>,
        phase: TrainingPhase,
    ) -> LossComponents {
        let (alpha, beta, gamma, delta) = self.get_loss_weights(phase);

        // 任务损失
        let task_loss = self.task_loss.compute_task_loss(
            &outputs.topo.node_features,
            true_atom_types,
            &outputs.geo.coords,
            true_coords,
            pred_bond_types,
            true_bond_types,
        );

        // 获取分支嵌入（使用全局表示）
        let topo_emb = &outputs.topo.global_embedding;
        let geo_emb = &outputs.geo.global_embedding;
        let pocket_emb = &outputs.pocket.global_embedding;

        // 解耦损失
        let disentangle_loss = self
            .disentangle_loss
            .compute_disentangle_loss(topo_emb, geo_emb, pocket_emb);

        // 冗余损失
        let redundancy_loss = self
            .redundancy_loss
            .compute_redundancy_loss(topo_emb, geo_emb, pocket_emb);

        // 一致性损失
        let consistency_loss = self
            .consistency_loss
            .compute_consistency_loss(topo_emb, geo_emb, pocket_emb);

        // 总损失
        let total_loss = task_loss.shallow_clone() * alpha
            + disentangle_loss.shallow_clone() * beta
            + redundancy_loss.shallow_clone() * gamma
            + consistency_loss.shallow_clone() * delta;

        LossComponents {
            task_loss: task_loss.double_value(&[]),
            disentangle_loss: disentangle_loss.double_value(&[]),
            redundancy_loss: redundancy_loss.double_value(&[]),
            consistency_loss: consistency_loss.double_value(&[]),
        }
    }

    /// 获取互信息监控值
    pub fn get_mi_monitor(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> (f64, f64, f64) {
        self.disentangle_loss
            .get_mi_values(topo_emb, geo_emb, pocket_emb)
    }

    /// 获取冗余度监控值
    pub fn get_redundancy_monitor(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> (f64, f64, f64) {
        self.redundancy_loss
            .get_redundancy_values(topo_emb, geo_emb, pocket_emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_loss() {
        let device = Device::Cpu;
        let loss = TaskLoss::new(device);

        let logits = Tensor::randn(&[8, 10], (Kind::Float, device));
        let targets = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]).to_kind(Kind::Int64);

        let atom_loss = loss.atom_type_loss(&logits, &targets);
        assert!(atom_loss.double_value(&[]) > 0.0);

        let pred_coords = Tensor::randn(&[8, 3], (Kind::Float, device));
        let true_coords = Tensor::randn(&[8, 3], (Kind::Float, device));
        let coord_loss = loss.coordinate_mse_loss(&pred_coords, &true_coords);
        assert!(coord_loss.double_value(&[]) > 0.0);
    }

    #[test]
    fn test_consistency_loss() {
        let device = Device::Cpu;
        let loss = ConsistencyLoss::new(device, 0.1);

        // 相似的表示应该有较低的损失
        let z1 = Tensor::randn(&[8, 64], (Kind::Float, device));
        let z2 = &z1 + 0.01 * Tensor::randn(&[8, 64], (Kind::Float, device));
        let z3 = Tensor::randn(&[8, 64], (Kind::Float, device));

        let loss_similar = loss.info_nce_loss(&z1, &z2).double_value(&[]);
        let loss_different = loss.info_nce_loss(&z1, &z3).double_value(&[]);

        println!(
            "Similar loss: {}, Different loss: {}",
            loss_similar, loss_different
        );
    }

    #[test]
    fn test_total_loss_weights() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let total_loss = TotalLoss::new(&vs_root, 64, 32, device, 0.1, true);

        // 验证不同阶段的权重
        let (a1, b1, g1, d1) = total_loss.get_loss_weights(TrainingPhase::Warmup);
        assert_eq!(b1, 0.0);
        assert_eq!(g1, 0.0);
        assert_eq!(d1, 0.0);

        let (a2, b2, g2, d2) = total_loss.get_loss_weights(TrainingPhase::Disentanglement);
        assert!(b2 > 0.0);
        assert!(g2 > 0.0);
        assert!(d2 > 0.0);

        let (a3, b3, g3, d3) = total_loss.get_loss_weights(TrainingPhase::Refinement);
        assert!(a3 > 0.0);
    }
}
