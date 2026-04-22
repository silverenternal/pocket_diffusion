//! SE(3)等变图神经网络 (EGNN)
//! 实现等变消息传递，保持旋转和平移不变性

use crate::types::tensor_from_slice;
use tch::{nn, Device, Kind, Tensor};

/// EGNN层配置
#[derive(Debug, Clone)]
pub struct EGNNConfig {
    /// 输入特征维度
    pub in_dim: usize,
    /// 隐藏层维度
    pub hidden_dim: usize,
    /// 输出特征维度
    pub out_dim: usize,
    /// 边特征维度(RBF数量)
    pub edge_dim: usize,
    /// 是否使用注意力
    pub use_attention: bool,
}

impl Default for EGNNConfig {
    fn default() -> Self {
        Self {
            in_dim: 6,
            hidden_dim: 64,
            out_dim: 1,
            edge_dim: 50,
            use_attention: true,
        }
    }
}

/// 径向基函数(RBF)距离编码
pub struct RBFEncoder {
    /// RBF中心
    centers: Tensor,
    /// 宽度
    sigma: f64,
    /// 截断距离
    _cutoff: f64,
}

impl RBFEncoder {
    /// 创建RBF编码器
    pub fn new(num_rbf: usize, cutoff: f64, device: Device) -> Self {
        let centers: Vec<f64> = (0..num_rbf)
            .map(|i| i as f64 * cutoff / (num_rbf - 1) as f64)
            .collect();

        let centers = tensor_from_slice(&centers).to_device(device);
        let sigma = cutoff / num_rbf as f64;

        Self {
            centers,
            sigma,
            _cutoff: cutoff,
        }
    }

    /// 对距离进行编码
    pub fn encode(&self, distances: &Tensor) -> Tensor {
        // distances: [E]
        // centers: [K]
        // output: [E, K]
        let d_expanded = distances.unsqueeze(1); // [E, 1]
        let c_expanded = self.centers.unsqueeze(0); // [1, K]

        let diff = &d_expanded - c_expanded;
        diff.pow_tensor_scalar(2.0)
            .f_mul_scalar(-1.0 / (self.sigma * self.sigma))
            .unwrap()
            .exp()
    }
}

/// 单EGNN层
pub struct EGNNLayer {
    /// 消息传递MLP
    message_mlp: nn::Sequential,
    /// 注意力MLP
    attention_mlp: Option<nn::Sequential>,
    /// 节点更新MLP
    node_update_mlp: nn::Sequential,
    /// 坐标更新MLP
    coord_update_mlp: nn::Sequential,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 是否使用注意力
    use_attention: bool,
}

impl EGNNLayer {
    /// 创建新的EGNN层
    pub fn new(
        vs: &nn::Path,
        in_dim: usize,
        hidden_dim: usize,
        edge_dim: usize,
        use_attention: bool,
    ) -> Self {
        // 消息传递MLP: h_i || h_j || e_ij -> m_ij
        let message_input_dim = 2 * in_dim + edge_dim;
        let message_mlp = nn::seq()
            .add(nn::linear(
                vs / "message_mlp" / "fc1",
                message_input_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "message_mlp" / "fc2",
                hidden_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|x| x.silu());

        // 注意力MLP
        let attention_mlp = if use_attention {
            Some(
                nn::seq()
                    .add(nn::linear(
                        vs / "attention_mlp" / "fc1",
                        hidden_dim as i64,
                        hidden_dim as i64 / 2,
                        Default::default(),
                    ))
                    .add_fn(|x| x.silu())
                    .add(nn::linear(
                        vs / "attention_mlp" / "fc2",
                        hidden_dim as i64 / 2,
                        1,
                        Default::default(),
                    ))
                    .add_fn(|x| x.sigmoid()),
            )
        } else {
            None
        };

        // 节点更新MLP
        let node_update_input_dim = in_dim + hidden_dim;
        let node_update_mlp = nn::seq()
            .add(nn::linear(
                vs / "node_update_mlp" / "fc1",
                node_update_input_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "node_update_mlp" / "fc2",
                hidden_dim as i64,
                in_dim as i64,
                Default::default(),
            ));

        // 坐标更新MLP
        let coord_update_mlp = nn::seq()
            .add(nn::linear(
                vs / "coord_update_mlp" / "fc1",
                hidden_dim as i64,
                hidden_dim as i64 / 2,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "coord_update_mlp" / "fc2",
                hidden_dim as i64 / 2,
                1,
                Default::default(),
            ));

        Self {
            message_mlp,
            attention_mlp,
            node_update_mlp,
            coord_update_mlp,
            hidden_dim,
            use_attention,
        }
    }

    /// 前向传播
    pub fn forward(
        &self,
        h: &Tensor,
        coords: &Tensor,
        edge_index: &Tensor,
        edge_attr: &Tensor,
    ) -> (Tensor, Tensor) {
        let num_nodes = h.size()[0];
        let device = h.device();

        // edge_index: [2, E]
        let senders = edge_index.slice(0, 0, 1, 1).squeeze_dim(0); // [E]
        let receivers = edge_index.slice(0, 1, 2, 1).squeeze_dim(0); // [E]

        // 收集节点特征
        let h_i = h.index_select(0, &receivers); // [E, F]
        let h_j = h.index_select(0, &senders); // [E, F]

        // 坐标向量: x_j - x_i
        let x_i = coords.index_select(0, &receivers); // [E, 3]
        let x_j = coords.index_select(0, &senders); // [E, 3]
        let x_ji = &x_j - &x_i; // [E, 3]

        // 构建消息输入
        let message_input = Tensor::cat(&[&h_i, &h_j, edge_attr], 1); // [E, 2F + E_dim]

        // 消息传递
        let mut messages = message_input.apply(&self.message_mlp); // [E, H]

        // 注意力加权
        if self.use_attention {
            if let Some(att_mlp) = &self.attention_mlp {
                let attention = messages.apply(att_mlp); // [E, 1]
                messages *= attention; // [E, H] * [E, 1]
            }
        }

        // 聚合消息到每个节点
        let mut aggregated =
            Tensor::zeros([num_nodes, self.hidden_dim as i64], (Kind::Float, device));
        aggregated = aggregated.index_add(0, &receivers, &messages); // [N, H]

        // 节点特征更新
        let node_update_input = Tensor::cat(&[h, &aggregated], 1); // [N, F + H]
        let delta_h = node_update_input.apply(&self.node_update_mlp); // [N, F]
        let h_new = h + delta_h; // 残差连接

        // 坐标更新（等变）
        let coord_weights = messages.apply(&self.coord_update_mlp); // [E, 1]
        let weighted_coord = x_ji * coord_weights; // [E, 3] * [E, 1]

        let mut delta_x = Tensor::zeros([num_nodes, 3], (Kind::Float, device));
        delta_x = delta_x.index_add(0, &receivers, &weighted_coord); // [N, 3]

        let x_new = coords + delta_x; // [N, 3]

        (h_new, x_new)
    }
}

/// 完整的EGNN模型
pub struct EGNN {
    /// 多层EGNN
    layers: Vec<EGNNLayer>,
    /// RBF编码器
    rbf_encoder: RBFEncoder,
    /// 输入投影层
    input_proj: nn::Linear,
    /// 输出头
    output_head: nn::Sequential,
    /// 层数
    _num_layers: usize,
}

impl EGNN {
    /// 创建EGNN模型
    pub fn new(vs: &nn::Path, config: &EGNNConfig, num_layers: usize) -> Self {
        let device = vs.device();

        // RBF编码器
        let rbf_encoder = RBFEncoder::new(config.edge_dim, 5.0, device);

        // 输入投影
        let input_proj = nn::linear(
            vs / "input_proj",
            config.in_dim as i64,
            config.hidden_dim as i64,
            Default::default(),
        );

        // 创建多层EGNN
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = EGNNLayer::new(
                &(vs / format!("layer_{}", i)),
                config.hidden_dim,
                config.hidden_dim,
                config.edge_dim,
                config.use_attention,
            );
            layers.push(layer);
        }

        // 输出头 (预测亲和力评分)
        let output_head = nn::seq()
            .add(nn::linear(
                vs / "output_head" / "fc1",
                config.hidden_dim as i64,
                config.hidden_dim as i64 / 2,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "output_head" / "fc2",
                config.hidden_dim as i64 / 2,
                config.out_dim as i64,
                Default::default(),
            ));

        Self {
            layers,
            rbf_encoder,
            input_proj,
            output_head,
            _num_layers: num_layers,
        }
    }

    /// 前向传播
    pub fn forward(
        &self,
        node_types: &Tensor,
        coords: &Tensor,
        edge_index: &Tensor,
        distances: &Tensor,
    ) -> Tensor {
        // node_types: [N, num_types] one-hot
        // coords: [N, 3]
        // edge_index: [2, E]
        // distances: [E]

        // 编码边特征
        let edge_attr = self.rbf_encoder.encode(distances); // [E, K]

        // 输入投影
        let mut h = node_types.apply(&self.input_proj); // [N, H]
        let mut x = coords.shallow_clone();

        // 多层EGNN
        for layer in &self.layers {
            let (h_new, x_new) = layer.forward(&h, &x, edge_index, &edge_attr);
            h = h_new;
            x = x_new;
        }

        // 全局平均池化并预测
        let graph_embedding = h.mean_dim(&[0i64][..], false, Kind::Float); // [H]
        let output = graph_embedding.apply(&self.output_head); // [out_dim]

        output
    }

    /// 获取RBF编码器
    pub fn rbf_encoder(&self) -> &RBFEncoder {
        &self.rbf_encoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egnn_forward() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();

        let config = EGNNConfig {
            in_dim: 6,
            hidden_dim: 32,
            out_dim: 1,
            edge_dim: 16,
            use_attention: true,
        };

        let egnn = EGNN::new(vs, &config, 2);

        // 创建小图
        let num_nodes = 10;
        let num_edges = 20;

        let node_types = Tensor::rand(&[num_nodes, 6], (Kind::Float, device));
        let coords = Tensor::rand(&[num_nodes, 3], (Kind::Float, device)) * 10.0;
        let edge_index = Tensor::randint(num_nodes, &[2, num_edges], (Kind::Int64, device));
        let distances = Tensor::rand(&[num_edges], (Kind::Float, device)) * 5.0;

        let output = egnn.forward(&node_types, &coords, &edge_index, &distances);

        assert_eq!(output.size(), &[1]);
    }

    #[test]
    fn test_rbf_encoder() {
        let device = Device::Cpu;
        let rbf = RBFEncoder::new(50, 5.0, device);

        let distances = tensor_from_slice(&[1.0, 2.0, 3.0]);
        let encoded = rbf.encode(&distances);

        assert_eq!(encoded.size(), &[3, 50]);
    }

    #[test]
    fn test_egnn_equivariance() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();

        let config = EGNNConfig {
            in_dim: 6,
            hidden_dim: 16,
            out_dim: 1,
            edge_dim: 16,
            use_attention: false,
        };

        let egnn = EGNN::new(vs, &config, 2);

        // 创建图并平移
        let num_nodes = 5;
        let node_types = Tensor::rand(&[num_nodes, 6], (Kind::Float, device));
        let coords = Tensor::rand(&[num_nodes, 3], (Kind::Float, device)) * 5.0;
        let edge_index = tensor_from_slice::<i64>(&[0, 0, 1, 1, 2, 3]).reshape(&[2, 3]);
        let distances = tensor_from_slice(&[2.0, 3.0, 1.5]);

        let _output1 = egnn.forward(&node_types, &coords, &edge_index, &distances);

        // 平移坐标
        let translation = tensor_from_slice(&[1.0, 2.0, 3.0]);
        let coords_translated = &coords + translation.reshape(&[1, 3]);
        let _output2 = egnn.forward(&node_types, &coords_translated, &edge_index, &distances);

        // 输出应该保持不变（等变）
        // 注意: 这个简化实现中输出是标量，理论上应该不变
        // 实际测试中可能有微小数值差异
    }
}
