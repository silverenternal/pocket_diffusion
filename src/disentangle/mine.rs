//! 互信息神经网络估计器 (MINE)
//! Mutual Information Neural Estimation
//!
//! 参考文献: Belghazi et al., "Mutual Information Neural Estimation", ICML 2018
//!
//! 基于Donsker-Varadhan变分表示:
//! I(X; Z) = sup_{T ∈ F} E_{P_{XZ}}[T(X, Z)] - log(E_{P_X ⊗ P_Z}[e^{T(X', Z')}])

use tch::nn::Module;
use tch::{nn, Device, Kind, Tensor};

/// MINE (互信息神经网络估计器)
///
/// 使用Donsker-Varadhan变分表示估计两个随机变量之间的互信息
pub struct MineEstimator {
    /// 统计网络 (Critic Network) - 计算T(x,z)
    network: nn::Sequential,
    /// 输入维度
    _input_dim: usize,
    /// 隐藏层维度
    _hidden_dim: usize,
    /// 设备
    device: Device,
}

impl MineEstimator {
    /// 创建新的MINE估计器
    pub fn new(vs: &nn::Path, input_dim: usize, hidden_dim: usize) -> Self {
        let device = vs.device();

        // 构建统计网络 T: R^d × R^d → R
        // 输入是两个向量的拼接
        let network = nn::seq()
            .add(nn::linear(
                vs / "linear1",
                input_dim as i64 * 2,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "linear2",
                hidden_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "linear3",
                hidden_dim as i64,
                1,
                Default::default(),
            ));

        Self {
            network,
            _input_dim: input_dim,
            _hidden_dim: hidden_dim,
            device,
        }
    }

    /// 计算统计量 T(x, z)
    ///
    /// # 参数
    /// - x: 第一个变量 [batch_size, dim]
    /// - z: 第二个变量 [batch_size, dim]
    ///
    /// # 返回
    /// T(x, z) 统计量值 [batch_size]
    pub fn forward(&self, x: &Tensor, z: &Tensor) -> Tensor {
        // 拼接 x 和 z
        let input = Tensor::cat(&[x, z], 1);
        self.network.forward(&input).squeeze_dim(1)
    }

    /// 估计两个随机变量之间的互信息
    ///
    /// 使用Donsker-Varadhan公式:
    /// I(X; Z) = sup_T E[Joint] - log(E[Marginal])
    ///
    /// # 参数
    /// - x: 第一个变量的样本 [batch_size, dim]
    /// - z: 第二个变量的样本 [batch_size, dim]
    ///
    /// # 返回
    /// 互信息估计值 (标量)
    pub fn estimate_mutual_information(&self, x: &Tensor, z: &Tensor) -> Tensor {
        let batch_size = x.size()[0];

        // 联合样本: (x_i, z_i) - 来自真实联合分布
        let t_joint = self.forward(x, z);
        let e_joint = t_joint.mean(Kind::Float);

        // 边缘乘积样本: (x_i, z_j) 其中 j ≠ i (洗牌)
        // 通过打乱z的顺序得到边缘分布的乘积
        let perm_indices = Tensor::randperm(batch_size, (Kind::Int64, self.device));
        let z_perm = z.index_select(0, &perm_indices);
        let t_marginal = self.forward(x, &z_perm);

        // E_{P_X ⊗ P_Z}[e^{T(X', Z')}]
        let e_marginal = t_marginal.exp().mean(Kind::Float).log();

        // I(X; Z) ≈ E_joint[T] - log(E_marginal[e^T])
        e_joint - e_marginal
    }

    /// 获取互信息估计损失 (用于最大化互信息)
    ///
    /// 注意: 如果我们要最小化互信息（解耦），需要取负值
    pub fn mi_loss(&self, x: &Tensor, z: &Tensor) -> Tensor {
        // 返回负值表示我们要最大化MI（最小化此损失等价于最大化MI）
        -self.estimate_mutual_information(x, z)
    }
}

/// 分支间互信息估计器
///
/// 估计拓扑、几何、口袋三个分支之间的两两互信息
pub struct BranchMutualInformation {
    /// MINE估计器: Topo ↔ Geo
    mine_topo_geo: MineEstimator,
    /// MINE估计器: Topo ↔ Pocket
    mine_topo_pocket: MineEstimator,
    /// MINE估计器: Geo ↔ Pocket
    mine_geo_pocket: MineEstimator,
}

impl BranchMutualInformation {
    /// 创建新的分支互信息估计器
    pub fn new(vs: &nn::Path, hidden_dim: usize, mine_hidden_dim: usize) -> Self {
        let mine_topo_geo =
            MineEstimator::new(&(vs / "mine_topo_geo"), hidden_dim, mine_hidden_dim);
        let mine_topo_pocket =
            MineEstimator::new(&(vs / "mine_topo_pocket"), hidden_dim, mine_hidden_dim);
        let mine_geo_pocket =
            MineEstimator::new(&(vs / "mine_geo_pocket"), hidden_dim, mine_hidden_dim);

        Self {
            mine_topo_geo,
            mine_topo_pocket,
            mine_geo_pocket,
        }
    }

    /// 估计所有分支对之间的互信息
    ///
    /// # 参数
    /// - topo_emb: 拓扑分支全局嵌入 [batch_size, hidden_dim]
    /// - geo_emb: 几何分支全局嵌入 [batch_size, hidden_dim]
    /// - pocket_emb: 口袋分支全局嵌入 [batch_size, hidden_dim]
    ///
    /// # 返回
    /// (MI_topo_geo, MI_topo_pocket, MI_geo_pocket)
    pub fn estimate_all(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let mi_topo_geo = self
            .mine_topo_geo
            .estimate_mutual_information(topo_emb, geo_emb);
        let mi_topo_pocket = self
            .mine_topo_pocket
            .estimate_mutual_information(topo_emb, pocket_emb);
        let mi_geo_pocket = self
            .mine_geo_pocket
            .estimate_mutual_information(geo_emb, pocket_emb);

        (mi_topo_geo, mi_topo_pocket, mi_geo_pocket)
    }

    /// 计算解耦损失 (L_disentangle)
    ///
    /// L_disentangle = MI_topo_geo + MI_topo_pocket + MI_geo_pocket
    ///
    /// 最小化这个损失等价于最小化分支间的互信息，实现解耦
    pub fn disentangle_loss(
        &self,
        topo_emb: &Tensor,
        geo_emb: &Tensor,
        pocket_emb: &Tensor,
    ) -> Tensor {
        let (mi_topo_geo, mi_topo_pocket, mi_geo_pocket) =
            self.estimate_all(topo_emb, geo_emb, pocket_emb);

        // 最小化所有分支对之间的互信息之和
        mi_topo_geo + mi_topo_pocket + mi_geo_pocket
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mine_estimator() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let mine = MineEstimator::new(&vs_root, 32, 64);

        let x = Tensor::randn(&[16, 32], (Kind::Float, device));
        let z = Tensor::randn(&[16, 32], (Kind::Float, device));

        let mi = mine.estimate_mutual_information(&x, &z);
        let mi_value: f64 = mi.double_value(&[]);

        // 独立变量的互信息应该接近0（正值是由于MINE的上界性质）
        println!("MI estimate for independent vars: {}", mi_value);
    }

    #[test]
    fn test_branch_mi() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let branch_mi = BranchMutualInformation::new(&vs_root, 128, 64);

        let topo = Tensor::randn(&[8, 128], (Kind::Float, device));
        let geo = Tensor::randn(&[8, 128], (Kind::Float, device));
        let pocket = Tensor::randn(&[8, 128], (Kind::Float, device));

        let (mi1, mi2, mi3) = branch_mi.estimate_all(&topo, &geo, &pocket);

        println!("MI(topo;geo) = {}", mi1.double_value(&[]));
        println!("MI(topo;pocket) = {}", mi2.double_value(&[]));
        println!("MI(geo;pocket) = {}", mi3.double_value(&[]));
    }
}
