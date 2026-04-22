//! 自动信息路由模块
//! Automatic Information Routing Module
//!
//! 学习将输入特征动态分配到三个专门分支：拓扑分支、几何分支、口袋分支

use tch::nn::Module;
use tch::{nn, Device, Kind, Tensor};

/// 信息路由网络
///
/// 学习将输入特征分配到三个专门分支
/// 输出路由概率 [p_topo, p_geo, p_pocket]，总和为1
pub struct InformationRouter {
    /// 特征编码网络
    encoder: nn::Sequential,
    /// 路由头，输出三个分支的权重
    router_head: nn::Sequential,
    /// 设备
    device: Device,
    /// 隐藏层维度
    hidden_dim: usize,
}

impl InformationRouter {
    /// 创建新的信息路由器
    ///
    /// # 参数
    /// - vs: 变量存储
    /// - input_dim: 输入特征维度
    /// - hidden_dim: 隐藏层维度
    pub fn new(vs: &nn::Path, input_dim: usize, hidden_dim: usize) -> Self {
        let device = vs.device();

        // 特征编码器
        let encoder = nn::seq()
            .add(nn::linear(
                vs / "enc_linear1",
                input_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "enc_linear2",
                hidden_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());

        // 路由头：输出3个分支的权重
        let router_head = nn::seq()
            .add(nn::linear(
                vs / "router_linear1",
                hidden_dim as i64,
                32,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "router_linear2",
                32,
                3, // 三个分支
                Default::default(),
            ));

        Self {
            encoder,
            router_head,
            device,
            hidden_dim,
        }
    }

    /// 前向传播，计算路由概率
    ///
    /// # 参数
    /// - node_features: 节点特征 [batch_size, num_atoms, feature_dim]
    /// - coords: 坐标信息 [batch_size, num_atoms, 3]
    /// - pocket_features: 口袋特征 [batch_size, pocket_dim]
    ///
    /// # 返回
    /// - routing_probs: 路由概率 [p_topo, p_geo, p_pocket]，总和为1
    /// - gate_activations: 每个特征维度的门控激活 [batch_size, hidden_dim, 3]
    pub fn forward(
        &self,
        node_features: &Tensor,
        coords: &Tensor,
        pocket_features: &Tensor,
    ) -> ([f32; 3], Tensor) {
        // 聚合节点级信息到图级表示
        // 平均节点维度以获得图级表示 [batch_size, feature_dim] 或 [feature_dim]
        let graph_embedding = node_features.mean_dim(0, false, Kind::Float);

        // 添加批次维度（如果需要）- 变成 [1, feature_dim]
        let graph_embedding = if graph_embedding.dim() == 1 {
            graph_embedding.unsqueeze(0)
        } else {
            graph_embedding
        };

        // 编码图表示
        let encoded = self.encoder.forward(&graph_embedding);

        // 计算路由logits并归一化
        let logits = self.router_head.forward(&encoded);
        let routing_probs = logits.softmax(-1, Kind::Float); // [batch_size, 3]

        // 计算批次平均概率
        let avg_probs = routing_probs.mean_dim(0, false, Kind::Float);
        let p_topo = avg_probs.double_value(&[0]) as f32;
        let p_geo = avg_probs.double_value(&[1]) as f32;
        let p_pocket = avg_probs.double_value(&[2]) as f32;

        // 计算每个特征维度的门控激活
        // 使用广播机制将概率应用到每个隐藏维度
        let gate_activations =
            Tensor::zeros(&[1, self.hidden_dim as i64, 3], (Kind::Float, self.device));
        // 填充门控值
        for i in 0..3 {
            let prob = if i == 0 {
                p_topo
            } else if i == 1 {
                p_geo
            } else {
                p_pocket
            };
            let mut slice = gate_activations.slice(2, i as i64, (i + 1) as i64, 1);
            let _ = slice.fill_(prob as f64);
        }

        ([p_topo, p_geo, p_pocket], gate_activations)
    }

    /// 获取路由权重（推理时使用）
    pub fn get_routing_weights(&self, node_features: &Tensor, coords: &Tensor) -> [f32; 3] {
        // 创建空的口袋特征用于推理
        let dummy_pocket = Tensor::zeros(&[1, 12], (Kind::Float, self.device));
        let (probs, _) = self.forward(node_features, coords, &dummy_pocket);
        probs
    }
}

/// 门控特征融合
///
/// 根据路由概率对不同分支的特征进行加权融合
pub struct GatedFusion {
    /// 融合权重投影层
    fusion_projection: nn::Sequential,
}

impl GatedFusion {
    /// 创建新的门控融合模块
    pub fn new(vs: &nn::Path, input_dim: usize, output_dim: usize) -> Self {
        let fusion_projection = nn::seq()
            .add(nn::linear(
                vs / "fusion_linear",
                input_dim as i64 * 3,
                output_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());

        Self { fusion_projection }
    }

    /// 融合三个分支的特征
    ///
    /// # 参数
    /// - topo_features: 拓扑分支特征 [batch_size, dim]
    /// - geo_features: 几何分支特征 [batch_size, dim]
    /// - pocket_features: 口袋分支特征 [batch_size, dim]
    /// - routing_probs: 路由概率 [p_topo, p_geo, p_pocket]
    ///
    /// # 返回
    /// 融合后的特征 [batch_size, output_dim]
    pub fn fuse(
        &self,
        topo_features: &Tensor,
        geo_features: &Tensor,
        pocket_features: &Tensor,
        routing_probs: &[f32; 3],
    ) -> Tensor {
        // 加权拼接
        let weighted_topo = topo_features * routing_probs[0] as f64;
        let weighted_geo = geo_features * routing_probs[1] as f64;
        let weighted_pocket = pocket_features * routing_probs[2] as f64;

        // 拼接所有特征
        let concatenated = Tensor::cat(&[&weighted_topo, &weighted_geo, &weighted_pocket], 1);

        // 投影到输出维度
        self.fusion_projection.forward(&concatenated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_router() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let router = InformationRouter::new(&vs_root, 64, 128);

        // 创建测试输入
        let node_features = Tensor::randn(&[8, 10, 64], (Kind::Float, device));
        let coords = Tensor::randn(&[8, 10, 3], (Kind::Float, device));
        let pocket_features = Tensor::randn(&[8, 12], (Kind::Float, device));

        let (probs, gates) = router.forward(&node_features, &coords, &pocket_features);

        // 验证概率总和约为1
        let sum = probs[0] + probs[1] + probs[2];
        println!("Routing probs: {:?}, sum: {}", probs, sum);
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Routing probabilities should sum to 1"
        );
    }

    #[test]
    fn test_gated_fusion() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let fusion = GatedFusion::new(&vs_root, 64, 128);

        let topo = Tensor::randn(&[8, 64], (Kind::Float, device));
        let geo = Tensor::randn(&[8, 64], (Kind::Float, device));
        let pocket = Tensor::randn(&[8, 64], (Kind::Float, device));
        let probs = [0.33, 0.33, 0.34];

        let fused = fusion.fuse(&topo, &geo, &pocket, &probs);

        assert_eq!(fused.size()[0], 8);
        assert_eq!(fused.size()[1], 128);
    }
}
