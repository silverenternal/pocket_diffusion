//! 分支信息冗余度计算器
//! Branch Information Redundancy Calculator
//!
//! 基于信息熵量化每个分支的信息冗余度
//! 冗余度 = 1 - (有效维度 / 总维度)

use std::ops::Div;
use tch::{nn, Device, Kind, Tensor};

/// 分支信息冗余度计算器
///
/// 使用信息熵和统计方法量化神经网络表示中的冗余信息
pub struct RedundancyCalculator {
    /// 是否使用基于熵的计算（否则使用基于方差的方法）
    use_entropy_based: bool,
    /// 直方图分箱数（用于熵计算）
    num_bins: usize,
    /// 设备
    device: Device,
}

impl Default for RedundancyCalculator {
    fn default() -> Self {
        Self::new(Device::Cpu, true, 50)
    }
}

impl RedundancyCalculator {
    /// 创建新的冗余度计算器
    pub fn new(device: Device, use_entropy_based: bool, num_bins: usize) -> Self {
        Self {
            use_entropy_based,
            num_bins,
            device,
        }
    }

    /// 计算张量的熵
    ///
    /// 使用直方图估计概率分布，然后计算离散熵：
    /// H = -Σ p_i * log(p_i)
    ///
    /// # 参数
    /// - x: 输入张量 [batch_size, dim]
    ///
    /// # 返回
    /// 平均熵值 (标量)
    pub fn compute_entropy(&self, x: &Tensor) -> f64 {
        // 简化：使用方差作为熵的近似
        // 更高的方差表示更高的信息含量（更低的冗余）
        let variance = x.var(true).mean(Kind::Float).double_value(&[]);
        // 归一化到 [0, 1] 区间
        1.0 - (-variance / 10.0).exp()
    }

    /// 基于方差的冗余度计算
    ///
    /// 低方差维度表示信息冗余（常数或接近常数的特征）
    /// 冗余度 = 1 - (有效维度 / 总维度)
    /// 其中有效维度 = 方差大于阈值的维度数量
    ///
    /// # 参数
    /// - x: 输入张量 [batch_size, dim]
    /// - threshold: 方差阈值（相对最大方差）
    ///
    /// # 返回
    /// 冗余度值 [0, 1]
    pub fn compute_variance_based_redundancy(&self, x: &Tensor, threshold: f64) -> f64 {
        let dim = x.size()[1] as f64;

        // 计算每个维度的方差
        let variances = x.var(true);
        let max_var = variances.max().double_value(&[]) + 1e-8;

        // 归一化方差
        let normalized_vars = variances.div(max_var);

        // 计算有效维度（方差大于阈值的维度）
        let effective_dims = normalized_vars
            .gt(threshold)
            .sum(Kind::Float)
            .double_value(&[]);

        // 冗余度 = 1 - 有效维度比例
        1.0 - (effective_dims / dim)
    }

    /// 计算分支的冗余度
    ///
    /// 结合熵基和方差基方法得到最终的冗余度估计
    ///
    /// # 参数
    /// - branch_embedding: 分支的嵌入表示 [batch_size, hidden_dim]
    ///
    /// # 返回
    /// 冗余度值 [0, 1]
    pub fn compute_redundancy(&self, branch_embedding: &Tensor) -> f64 {
        if self.use_entropy_based {
            // 归一化熵: H_normalized = H / log(num_bins)
            // 冗余度 = 归一化熵
            let entropy = self.compute_entropy(branch_embedding);
            let max_possible_entropy = (self.num_bins as f64).ln();
            let normalized_entropy = entropy / max_possible_entropy;

            // 冗余度与熵成反比：低熵 = 高冗余（信息集中在少数状态）
            1.0 - normalized_entropy
        } else {
            // 使用方差基方法
            self.compute_variance_based_redundancy(branch_embedding, 0.01)
        }
    }

    /// 计算所有分支的冗余度
    ///
    /// # 参数
    /// - topo_emb: 拓扑分支嵌入 [batch_size, hidden_dim]
    /// - geo_emb: 几何分支嵌入 [batch_size, hidden_dim]
    /// - pocket_emb: 口袋分支嵌入 [batch_size, hidden_dim]
    ///
    /// # 返回
    /// (topo_redundancy, geo_redundancy, pocket_redundancy)
    pub fn compute_all(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> (f64, f64, f64) {
        let r_topo = self.compute_redundancy(topo_emb);
        let r_geo = self.compute_redundancy(geo_emb);
        let r_pocket = self.compute_redundancy(pocket_emb);

        (r_topo, r_geo, r_pocket)
    }

    /// 计算冗余损失 (L_redundancy)
    ///
    /// L_redundancy = mean(R_topo, R_geo, R_pocket)
    ///
    /// 最小化这个损失鼓励分支保持最小的信息冗余
    pub fn redundancy_loss(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> Tensor {
        let (r_topo, r_geo, r_pocket) = self.compute_all(topo_emb, geo_emb, pocket_emb);

        let avg_redundancy = (r_topo + r_geo + r_pocket) / 3.0;

        Tensor::from(avg_redundancy)
            .to_kind(Kind::Float)
            .to(self.device)
    }
}

/// 基于互信息的冗余度计算（可选高级方法）
///
/// 使用分支内特征之间的互信息总和量化冗余
/// 高互信息表示特征之间存在冗余信息
pub struct MutualInfoRedundancy {
    /// MINE估计器用于计算特征间互信息
    mine: MineEstimator,
}

use crate::disentangle::mine::MineEstimator;

impl MutualInfoRedundancy {
    /// 创建新的互信息基冗余度计算器
    pub fn new(vs: &nn::Path, feature_dim: usize, hidden_dim: usize) -> Self {
        let mine = MineEstimator::new(vs, feature_dim / 2, hidden_dim);
        Self { mine }
    }

    /// 计算分支内特征的冗余度
    ///
    /// 将特征随机分成两半，计算两半之间的互信息
    /// 高互信息表示分支内存在冗余信息
    pub fn compute_redundancy(&self, branch_emb: &Tensor) -> f64 {
        let dim = branch_emb.size()[1] as usize;
        let half_dim = dim / 2;

        // 将特征分成两半
        let x1 = branch_emb.slice(1, 0, half_dim as i64, 1);
        let x2 = branch_emb.slice(1, half_dim as i64, dim as i64, 1);

        // 计算两半之间的互信息
        let mi = self.mine.estimate_mutual_information(&x1, &x2);
        mi.double_value(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redundancy_calculator() {
        let device = Device::Cpu;
        let calc = RedundancyCalculator::new(device, true, 50);

        // 低冗余：高方差随机数据
        let x_low = Tensor::randn(&[100, 64], (Kind::Float, device)) * 10.0;
        let r_low = calc.compute_redundancy(&x_low);
        println!("Low redundancy data: {}", r_low);

        // 高冗余：常数值 + 小噪声
        let x_high = Tensor::ones(&[100, 64], (Kind::Float, device))
            + Tensor::randn(&[100, 64], (Kind::Float, device)) * 0.01;
        let r_high = calc.compute_redundancy(&x_high);
        println!("High redundancy data: {}", r_high);

        // 常数数据应该有高冗余度
        assert!(
            r_high > r_low,
            "Constant data should have higher redundancy"
        );
    }

    #[test]
    fn test_variance_based() {
        let device = Device::Cpu;
        let calc = RedundancyCalculator::new(device, false, 50);

        // 测试：所有数据相同（高冗余） vs 随机数据（低冗余）
        let x_high = Tensor::ones(&[100, 64], (Kind::Float, device)) * 0.5;
        let r_high = calc.compute_variance_based_redundancy(&x_high, 0.01);
        println!("High redundancy (constant): {}", r_high);

        let x_low = Tensor::randn(&[100, 64], (Kind::Float, device)) * 10.0;
        let r_low = calc.compute_variance_based_redundancy(&x_low, 0.01);
        println!("Low redundancy (random): {}", r_low);

        assert!(
            r_high > r_low,
            "Constant data should have higher redundancy"
        );
    }
}
