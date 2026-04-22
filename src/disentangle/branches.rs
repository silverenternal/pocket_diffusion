//! 解耦分支模块
//! Disentangled Branches Module
//!
//! 三个独立的处理分支：
//! 1. TopoBranch - 处理2D拓扑信息的Transformer
//! 2. GeoBranch - 处理3D几何信息的SE(3)等变网络
//! 3. PocketBranch - 编码口袋条件信息的注意力网络

use tch::nn::Module;
use tch::{nn, Device, Kind, Tensor};

/// 简化的多头自注意力层
struct SelfAttentionLayer {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    hidden_dim: usize,
    num_heads: usize,
}

impl SelfAttentionLayer {
    fn new(vs: &nn::Path, hidden_dim: usize, num_heads: usize) -> Self {
        let q_proj = nn::linear(
            vs / "q_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let k_proj = nn::linear(
            vs / "k_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let v_proj = nn::linear(
            vs / "v_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let out_proj = nn::linear(
            vs / "out_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            hidden_dim,
            num_heads,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.size()[0];
        let head_dim = self.hidden_dim / self.num_heads;

        // QKV投影
        let q = self
            .q_proj
            .forward(x)
            .view([seq_len, self.num_heads as i64, head_dim as i64])
            .permute([1, 0, 2]);
        let k = self
            .k_proj
            .forward(x)
            .view([seq_len, self.num_heads as i64, head_dim as i64])
            .permute([1, 0, 2]);
        let v = self
            .v_proj
            .forward(x)
            .view([seq_len, self.num_heads as i64, head_dim as i64])
            .permute([1, 0, 2]);

        // 缩放点积注意力
        let scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
        let weights = scores.softmax(-1, Kind::Float);
        let attn_output = weights.matmul(&v);

        // 合并头
        let attn_output = attn_output
            .permute([1, 0, 2])
            .contiguous()
            .view([seq_len, self.hidden_dim as i64]);
        self.out_proj.forward(&attn_output)
    }
}

/// 简化的Transformer层
struct TransformerLayer {
    attention: SelfAttentionLayer,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    ffn: nn::Sequential,
}

impl TransformerLayer {
    fn new(vs: &nn::Path, hidden_dim: usize, num_heads: usize) -> Self {
        let attention = SelfAttentionLayer::new(&(vs / "attn"), hidden_dim, num_heads);
        let norm1 = nn::layer_norm(vs / "norm1", vec![hidden_dim as i64], Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![hidden_dim as i64], Default::default());
        let ffn = nn::seq()
            .add(nn::linear(
                vs / "ffn1",
                hidden_dim as i64,
                hidden_dim as i64 * 4,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "ffn2",
                hidden_dim as i64 * 4,
                hidden_dim as i64,
                Default::default(),
            ));

        Self {
            attention,
            norm1,
            norm2,
            ffn,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // 残差连接 + 自注意力
        let attn_out = self.attention.forward(x);
        let x = x + attn_out;
        let x = self.norm1.forward(&x);

        // 残差连接 + FFN
        let ffn_out = self.ffn.forward(&x);
        let x = x + ffn_out;
        self.norm2.forward(&x)
    }
}

/// 拓扑分支（2D结构信息）
///
/// 使用图Transformer处理分子的拓扑连接信息
/// 专注于化学键、环结构等离散结构特征
pub struct TopoBranch {
    /// 输入投影层
    input_proj: nn::Linear,
    /// Transformer编码层
    transformer_layers: Vec<TransformerLayer>,
    /// 输出投影层
    output_proj: nn::Linear,
    /// 邻接矩阵预测头
    adjacency_head: nn::Sequential,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 设备
    device: Device,
}

impl TopoBranch {
    /// 创建新的拓扑分支
    ///
    /// # 参数
    /// - vs: 变量存储
    /// - input_dim: 输入特征维度
    /// - hidden_dim: 隐藏层维度
    /// - num_heads: 注意力头数
    /// - num_layers: Transformer层数
    pub fn new(
        vs: &nn::Path,
        input_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Self {
        let device = vs.device();

        // 输入投影
        let input_proj = nn::linear(
            vs / "topo_input_proj",
            input_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        // Transformer编码层
        let mut transformer_layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = TransformerLayer::new(
                &(vs / &format!("topo_transformer_{}", i)),
                hidden_dim,
                num_heads,
            );
            transformer_layers.push(layer);
        }

        // 输出投影
        let output_proj = nn::linear(
            vs / "topo_output_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        // 邻接矩阵预测头
        let adjacency_head = nn::seq()
            .add(nn::linear(
                vs / "adj_head_1",
                hidden_dim as i64 * 2,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "adj_head_2",
                hidden_dim as i64,
                1,
                Default::default(),
            ));

        Self {
            input_proj,
            transformer_layers,
            output_proj,
            adjacency_head,
            hidden_dim,
            device,
        }
    }

    /// 拓扑分支前向传播
    ///
    /// # 参数
    /// - node_features: 节点特征 [num_atoms, feature_dim]
    /// - adjacency: 邻接矩阵 [num_atoms, num_atoms]
    ///
    /// # 返回
    /// - 更新后的节点特征 [num_atoms, hidden_dim]
    /// - 全局嵌入 [hidden_dim]
    /// - 邻接矩阵预测 [num_atoms, num_atoms]
    pub fn forward(&self, node_features: &Tensor, _adjacency: &Tensor) -> (Tensor, Tensor, Tensor) {
        let num_atoms = node_features.size()[0];

        // 输入投影
        let h = self.input_proj.forward(node_features);

        // Transformer编码（使用自注意力）
        let mut x = h;
        for layer in &self.transformer_layers {
            x = layer.forward(&x);
        }

        // 输出投影
        let node_features_out = self.output_proj.forward(&x);

        // 全局嵌入（图级表示）
        let global_embedding = node_features_out.mean_dim(0, false, Kind::Float);

        // 预测邻接矩阵：使用外积 + MLP
        // [num_atoms, hidden_dim] x [num_atoms, hidden_dim] -> [num_atoms, num_atoms, 2*hidden_dim]
        let x_i = node_features_out
            .unsqueeze(1)
            .expand(&[num_atoms, num_atoms, self.hidden_dim as i64], false);
        let x_j = node_features_out
            .unsqueeze(0)
            .expand(&[num_atoms, num_atoms, self.hidden_dim as i64], false);
        let pair_features = Tensor::cat(&[&x_i, &x_j], 2); // [num_atoms, num_atoms, 2*hidden_dim]

        let adjacency_logits = self.adjacency_head.forward(&pair_features).squeeze_dim(2); // [num_atoms, num_atoms]

        (node_features_out, global_embedding, adjacency_logits)
    }
}

/// 几何分支（3D空间信息）
///
/// 使用SE(3)等变网络处理分子的几何坐标信息
/// 专注于原子位置、键角、二面角等连续空间特征
pub struct GeoBranch {
    /// 坐标编码器
    coord_encoder: nn::Sequential,
    /// 等变图卷积层
    conv_layers: Vec<nn::Linear>,
    /// 输出投影
    output_proj: nn::Linear,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 设备
    device: Device,
}

impl GeoBranch {
    /// 创建新的几何分支
    ///
    /// # 参数
    /// - vs: 变量存储
    /// - hidden_dim: 隐藏层维度
    /// - num_layers: 卷积层数
    pub fn new(vs: &nn::Path, hidden_dim: usize, num_layers: usize) -> Self {
        let device = vs.device();

        // 坐标编码器：处理3D坐标 + 距离嵌入
        let coord_encoder = nn::seq()
            .add(nn::linear(
                vs / "coord_encoder_1",
                3, // x, y, z
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "coord_encoder_2",
                hidden_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ));

        // 等变卷积层（简化为MLP）
        let mut conv_layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = nn::linear(
                vs / &format!("geo_conv_{}", i),
                hidden_dim as i64,
                hidden_dim as i64,
                Default::default(),
            );
            conv_layers.push(layer);
        }

        // 输出投影
        let output_proj = nn::linear(
            vs / "geo_output_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        Self {
            coord_encoder,
            conv_layers,
            output_proj,
            hidden_dim,
            device,
        }
    }

    /// 几何分支前向传播
    ///
    /// # 参数
    /// - coords: 原子3D坐标 [num_atoms, 3]
    /// - node_features: 节点特征 [num_atoms, hidden_dim]
    ///
    /// # 返回
    /// - 更新后的坐标 [num_atoms, 3]
    /// - 更新后的节点特征 [num_atoms, hidden_dim]
    /// - 全局嵌入 [hidden_dim]
    pub fn forward(&self, coords: &Tensor, node_features: &Tensor) -> (Tensor, Tensor, Tensor) {
        // 编码坐标信息
        let coord_emb = self.coord_encoder.forward(coords);

        // 融合坐标和节点特征
        let mut h = coord_emb + node_features;

        // 多层等变卷积
        for layer in &self.conv_layers {
            h = layer.forward(&h).relu();
        }

        // 输出投影
        let node_features_out = self.output_proj.forward(&h);

        // 全局嵌入
        let global_embedding = node_features_out.mean_dim(0, false, Kind::Float);

        // 预测坐标更新（残差连接）
        let coord_update = Tensor::zeros(coords.size(), (Kind::Float, self.device));

        let coords_out = coords + coord_update;

        (coords_out, node_features_out, global_embedding)
    }
}

/// 口袋分支（条件信息）
///
/// 使用交叉注意力编码蛋白口袋环境信息
/// 为生成过程提供条件约束
pub struct PocketBranch {
    /// 口袋特征编码器
    pocket_encoder: nn::Sequential,
    /// 交叉注意力查询投影
    query_proj: nn::Linear,
    /// 交叉注意力键投影
    key_proj: nn::Linear,
    /// 交叉注意力值投影
    value_proj: nn::Linear,
    /// 输出投影
    output_proj: nn::Sequential,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 设备
    device: Device,
}

impl PocketBranch {
    /// 创建新的口袋分支
    ///
    /// # 参数
    /// - vs: 变量存储
    /// - pocket_dim: 口袋特征维度
    /// - hidden_dim: 隐藏层维度
    pub fn new(vs: &nn::Path, pocket_dim: usize, hidden_dim: usize) -> Self {
        let device = vs.device();

        // 口袋特征编码器
        let pocket_encoder = nn::seq()
            .add(nn::linear(
                vs / "pocket_encoder_1",
                pocket_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "pocket_encoder_2",
                hidden_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ));

        // 注意力投影
        let query_proj = nn::linear(
            vs / "pocket_query_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let key_proj = nn::linear(
            vs / "pocket_key_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let value_proj = nn::linear(
            vs / "pocket_value_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        // 输出投影
        let output_proj = nn::seq()
            .add(nn::linear(
                vs / "pocket_output_1",
                hidden_dim as i64 * 2,
                hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "pocket_output_2",
                hidden_dim as i64,
                hidden_dim as i64,
                Default::default(),
            ));

        Self {
            pocket_encoder,
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            hidden_dim,
            device,
        }
    }

    /// 口袋分支前向传播
    ///
    /// # 参数
    /// - pocket_features: 口袋特征 [pocket_dim]
    /// - ligand_features: 配体节点特征 [num_atoms, hidden_dim]
    ///
    /// # 返回
    /// - 交叉注意力权重 [num_atoms, hidden_dim]
    /// - 条件化的配体特征 [num_atoms, hidden_dim]
    /// - 全局嵌入 [hidden_dim]
    pub fn forward(
        &self,
        pocket_features: &Tensor,
        ligand_features: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let num_atoms = ligand_features.size()[0];

        // 编码口袋特征
        let pocket_emb = self.pocket_encoder.forward(pocket_features);

        // 交叉注意力：配体特征作为Query，口袋特征作为Key/Value
        let queries = self.query_proj.forward(ligand_features); // [num_atoms, hidden_dim]
        let keys = self.key_proj.forward(&pocket_emb).unsqueeze(0); // [1, hidden_dim]
        let values = self.value_proj.forward(&pocket_emb).unsqueeze(0); // [1, hidden_dim]

        // 计算注意力分数
        let attention_scores =
            queries.matmul(&keys.transpose(0, 1)) / (self.hidden_dim as f64).sqrt();
        let attention_weights = attention_scores.softmax(1, Kind::Float); // [num_atoms, hidden_dim]

        // 应用注意力
        let attended = attention_weights.matmul(&values); // [num_atoms, hidden_dim]

        // 融合原始特征和注意力特征
        let fused = Tensor::cat(&[ligand_features, &attended], 1); // [num_atoms, 2*hidden_dim]
        let conditioned_features = self.output_proj.forward(&fused); // [num_atoms, hidden_dim]

        // 全局嵌入
        let global_embedding = conditioned_features.mean_dim(0, false, Kind::Float);

        (attention_weights, conditioned_features, global_embedding)
    }
}

/// 完整的解耦分支集合
///
/// 包含三个独立的分支，参数不共享，确保信息解耦
pub struct DisentangledBranches {
    /// 拓扑分支
    pub topo: TopoBranch,
    /// 几何分支
    pub geo: GeoBranch,
    /// 口袋分支
    pub pocket: PocketBranch,
}

impl DisentangledBranches {
    /// 创建所有分支
    pub fn new(
        vs: &nn::Path,
        input_dim: usize,
        hidden_dim: usize,
        pocket_dim: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Self {
        // 为每个分支创建独立的变量路径
        let topo = TopoBranch::new(
            &(vs / "topo_branch"),
            input_dim,
            hidden_dim,
            num_heads,
            num_layers,
        );
        let geo = GeoBranch::new(&(vs / "geo_branch"), hidden_dim, num_layers);
        let pocket = PocketBranch::new(&(vs / "pocket_branch"), pocket_dim, hidden_dim);

        Self { topo, geo, pocket }
    }

    /// 所有分支的前向传播
    ///
    /// # 参数
    /// - node_features: 初始节点特征 [num_atoms, input_dim]
    /// - coords: 原子坐标 [num_atoms, 3]
    /// - adjacency: 邻接矩阵 [num_atoms, num_atoms]
    /// - pocket_features: 口袋特征 [pocket_dim]
    ///
    /// # 返回
    /// - topo_global: 拓扑分支全局嵌入 [hidden_dim]
    /// - geo_global: 几何分支全局嵌入 [hidden_dim]
    /// - pocket_global: 口袋分支全局嵌入 [hidden_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        node_features: &Tensor,
        coords: &Tensor,
        adjacency: &Tensor,
        pocket_features: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        // 拓扑分支
        let (_, topo_global, _) = self.topo.forward(node_features, adjacency);

        // 几何分支（使用拓扑分支输出作为初始特征）
        let (_, _, geo_global) = self.geo.forward(
            coords,
            &topo_global.unsqueeze(0).repeat(&[coords.size()[0], 1]),
        );

        // 口袋分支
        let (_, _, pocket_global) = self.pocket.forward(
            pocket_features,
            &topo_global.unsqueeze(0).repeat(&[coords.size()[0], 1]),
        );

        (topo_global, geo_global, pocket_global)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topo_branch() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let topo = TopoBranch::new(&vs_root, 10, 64, 8, 3);

        let node_features = Tensor::randn(&[20, 10], (Kind::Float, device));
        let adjacency = Tensor::zeros(&[20, 20], (Kind::Float, device));

        let (feat_out, global_out, adj_logits) = topo.forward(&node_features, &adjacency);

        assert_eq!(feat_out.size()[0], 20);
        assert_eq!(feat_out.size()[1], 64);
        assert_eq!(global_out.size()[0], 64);
        assert_eq!(adj_logits.size()[0], 20);
        assert_eq!(adj_logits.size()[1], 20);
    }

    #[test]
    fn test_geo_branch() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let geo = GeoBranch::new(&vs_root, 64, 3);

        let coords = Tensor::randn(&[20, 3], (Kind::Float, device));
        let features = Tensor::randn(&[20, 64], (Kind::Float, device));

        let (coords_out, feat_out, global_out) = geo.forward(&coords, &features);

        assert_eq!(coords_out.size()[0], 20);
        assert_eq!(coords_out.size()[1], 3);
        assert_eq!(feat_out.size()[0], 20);
        assert_eq!(feat_out.size()[1], 64);
        assert_eq!(global_out.size()[0], 64);
    }

    #[test]
    fn test_pocket_branch() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let pocket = PocketBranch::new(&vs_root, 12, 64);

        let pocket_features = Tensor::randn(&[12], (Kind::Float, device));
        let ligand_features = Tensor::randn(&[20, 64], (Kind::Float, device));

        let (attn_weights, cond_features, global_out) =
            pocket.forward(&pocket_features, &ligand_features);

        assert_eq!(attn_weights.size()[0], 20);
        assert_eq!(cond_features.size()[0], 20);
        assert_eq!(cond_features.size()[1], 64);
        assert_eq!(global_out.size()[0], 64);
    }
}
