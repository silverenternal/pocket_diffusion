//! 解耦跨注意力模块
//! Disentangled Cross-Attention Module
//!
//! 实现三路独立QKV的跨注意力机制，不混合梯度，不混合特征
//! 保持拓扑、几何、口袋三个分支的信息独立性

use tch::nn::Module;
use tch::{nn, Device, Kind, Tensor};

/// 解耦的多头注意力
///
/// 为每个分支维护完全独立的QKV投影
/// 梯度不混合，特征不交叉
pub struct DisentangledMultiHeadAttention {
    /// 拓扑分支Q投影
    topo_q_proj: nn::Linear,
    /// 拓扑分支K投影
    topo_k_proj: nn::Linear,
    /// 拓扑分支V投影
    topo_v_proj: nn::Linear,
    /// 几何分支Q投影
    geo_q_proj: nn::Linear,
    /// 几何分支K投影
    geo_k_proj: nn::Linear,
    /// 几何分支V投影
    geo_v_proj: nn::Linear,
    /// 口袋分支Q投影
    pocket_q_proj: nn::Linear,
    /// 口袋分支K投影
    pocket_k_proj: nn::Linear,
    /// 口袋分支V投影
    pocket_v_proj: nn::Linear,
    /// 输出投影（每个分支独立）
    topo_out_proj: nn::Linear,
    geo_out_proj: nn::Linear,
    pocket_out_proj: nn::Linear,
    /// 注意力头数
    num_heads: usize,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 设备
    _device: Device,
}

impl DisentangledMultiHeadAttention {
    /// 创建新的解耦多头注意力
    ///
    /// # 参数
    /// - vs: 变量存储
    /// - hidden_dim: 隐藏层维度
    /// - num_heads: 注意力头数
    pub fn new(vs: &nn::Path, hidden_dim: usize, num_heads: usize) -> Self {
        let device = vs.device();

        // 拓扑分支的独立QKV投影
        let topo_q_proj = nn::linear(
            vs / "topo_q_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let topo_k_proj = nn::linear(
            vs / "topo_k_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let topo_v_proj = nn::linear(
            vs / "topo_v_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        // 几何分支的独立QKV投影
        let geo_q_proj = nn::linear(
            vs / "geo_q_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let geo_k_proj = nn::linear(
            vs / "geo_k_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let geo_v_proj = nn::linear(
            vs / "geo_v_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        // 口袋分支的独立QKV投影
        let pocket_q_proj = nn::linear(
            vs / "pocket_q_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let pocket_k_proj = nn::linear(
            vs / "pocket_k_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let pocket_v_proj = nn::linear(
            vs / "pocket_v_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        // 每个分支独立的输出投影
        let topo_out_proj = nn::linear(
            vs / "topo_out_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let geo_out_proj = nn::linear(
            vs / "geo_out_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );
        let pocket_out_proj = nn::linear(
            vs / "pocket_out_proj",
            hidden_dim as i64,
            hidden_dim as i64,
            Default::default(),
        );

        Self {
            topo_q_proj,
            topo_k_proj,
            topo_v_proj,
            geo_q_proj,
            geo_k_proj,
            geo_v_proj,
            pocket_q_proj,
            pocket_k_proj,
            pocket_v_proj,
            topo_out_proj,
            geo_out_proj,
            pocket_out_proj,
            num_heads,
            hidden_dim,
            _device: device,
        }
    }

    /// 拓扑分支自注意力（仅使用拓扑分支的QKV）
    ///
    /// 不与其他分支共享任何参数
    pub fn topo_self_attention(&self, x: &Tensor) -> Tensor {
        let q = self.topo_q_proj.forward(x);
        let k = self.topo_k_proj.forward(x);
        let v = self.topo_v_proj.forward(x);

        // 拆分多头
        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);

        // 缩放点积注意力
        let attn_output = self.scaled_dot_product_attention(&q, &k, &v);

        // 合并头并投影
        let attn_output = self.combine_heads(&attn_output);
        self.topo_out_proj.forward(&attn_output)
    }

    /// 几何分支自注意力（仅使用几何分支的QKV）
    pub fn geo_self_attention(&self, x: &Tensor) -> Tensor {
        let q = self.geo_q_proj.forward(x);
        let k = self.geo_k_proj.forward(x);
        let v = self.geo_v_proj.forward(x);

        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);

        let attn_output = self.scaled_dot_product_attention(&q, &k, &v);

        let attn_output = self.combine_heads(&attn_output);
        self.geo_out_proj.forward(&attn_output)
    }

    /// 口袋分支条件交叉注意力
    ///
    /// 使用口袋特征作为Key/Value，配体特征作为Query
    /// 仅使用口袋分支的投影参数
    pub fn pocket_cross_attention(&self, query: &Tensor, key_value: &Tensor) -> Tensor {
        let q = self.pocket_q_proj.forward(query);
        let k = self.pocket_k_proj.forward(key_value);
        let v = self.pocket_v_proj.forward(key_value);

        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);

        let attn_output = self.scaled_dot_product_attention(&q, &k, &v);

        let attn_output = self.combine_heads(&attn_output);
        self.pocket_out_proj.forward(&attn_output)
    }

    /// 拆分多头
    fn split_heads(&self, x: &Tensor) -> Tensor {
        let batch_size = if x.dim() == 2 { 1 } else { x.size()[0] };
        let seq_len = if x.dim() == 2 {
            x.size()[0]
        } else {
            x.size()[1]
        };

        let x_reshaped = x.view([
            batch_size,
            seq_len,
            self.num_heads as i64,
            self.hidden_dim as i64 / self.num_heads as i64,
        ]);

        x_reshaped.permute([0, 2, 1, 3]) // [batch, heads, seq, dim_per_head]
    }

    /// 合并头
    fn combine_heads(&self, x: &Tensor) -> Tensor {
        let batch_size = x.size()[0];
        let seq_len = x.size()[2];

        let x_permuted = x.permute([0, 2, 1, 3]); // [batch, seq, heads, dim_per_head]
        x_permuted
            .contiguous()
            .view([batch_size, seq_len, self.hidden_dim as i64])
    }

    /// 缩放点积注意力
    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        let d_k = self.hidden_dim as f64 / self.num_heads as f64;

        // Q·K^T / sqrt(d_k)
        let scores = q.matmul(&k.transpose(-2, -1)) / d_k.sqrt();

        // softmax
        let weights = scores.softmax(-1, Kind::Float);

        // weights·V
        weights.matmul(v)
    }
}

/// 解耦跨注意力Transformer层
///
/// 包含三路完全独立的自注意力机制
/// 分支间没有参数共享，没有特征混合
pub struct DisentangledAttentionLayer {
    /// 拓扑分支注意力
    topo_attention: DisentangledMultiHeadAttention,
    /// 几何分支注意力
    geo_attention: DisentangledMultiHeadAttention,
    /// 层归一化（每个分支独立）
    topo_norm1: nn::LayerNorm,
    topo_norm2: nn::LayerNorm,
    geo_norm1: nn::LayerNorm,
    geo_norm2: nn::LayerNorm,
    /// 前馈网络（每个分支独立）
    topo_ffn: nn::Sequential,
    geo_ffn: nn::Sequential,
}

impl DisentangledAttentionLayer {
    /// 创建新的解耦注意力层
    pub fn new(vs: &nn::Path, hidden_dim: usize, num_heads: usize) -> Self {
        // 每个分支有独立的注意力模块
        let topo_attention =
            DisentangledMultiHeadAttention::new(&(vs / "topo_attn"), hidden_dim, num_heads);
        let geo_attention =
            DisentangledMultiHeadAttention::new(&(vs / "geo_attn"), hidden_dim, num_heads);

        // 独立的层归一化
        let topo_norm1 = nn::layer_norm(
            vs / "topo_norm1",
            vec![hidden_dim as i64],
            Default::default(),
        );
        let topo_norm2 = nn::layer_norm(
            vs / "topo_norm2",
            vec![hidden_dim as i64],
            Default::default(),
        );
        let geo_norm1 = nn::layer_norm(
            vs / "geo_norm1",
            vec![hidden_dim as i64],
            Default::default(),
        );
        let geo_norm2 = nn::layer_norm(
            vs / "geo_norm2",
            vec![hidden_dim as i64],
            Default::default(),
        );

        // 独立的前馈网络
        let topo_ffn = nn::seq()
            .add(nn::linear(
                vs / "topo_ffn_1",
                hidden_dim as i64,
                hidden_dim as i64 * 4,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "topo_ffn_2",
                hidden_dim as i64 * 4,
                hidden_dim as i64,
                Default::default(),
            ));

        let geo_ffn = nn::seq()
            .add(nn::linear(
                vs / "geo_ffn_1",
                hidden_dim as i64,
                hidden_dim as i64 * 4,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "geo_ffn_2",
                hidden_dim as i64 * 4,
                hidden_dim as i64,
                Default::default(),
            ));

        Self {
            topo_attention,
            geo_attention,
            topo_norm1,
            topo_norm2,
            geo_norm1,
            geo_norm2,
            topo_ffn,
            geo_ffn,
        }
    }

    /// 前向传播 - 拓扑分支（独立处理）
    pub fn forward_topo(&self, x: &Tensor) -> Tensor {
        // 残差连接 + 自注意力
        let attn_output = self.topo_attention.topo_self_attention(x);
        let x = x + attn_output;
        let x = self.topo_norm1.forward(&x);

        // 残差连接 + FFN
        let ffn_output = self.topo_ffn.forward(&x);
        let x = x + ffn_output;
        self.topo_norm2.forward(&x)
    }

    /// 前向传播 - 几何分支（独立处理）
    pub fn forward_geo(&self, x: &Tensor) -> Tensor {
        // 残差连接 + 自注意力
        let attn_output = self.geo_attention.geo_self_attention(x);
        let x = x + attn_output;
        let x = self.geo_norm1.forward(&x);

        // 残差连接 + FFN
        let ffn_output = self.geo_ffn.forward(&x);
        let x = x + ffn_output;
        self.geo_norm2.forward(&x)
    }
}

/// 分支间对比损失
///
/// 通过最小化分支表示之间的互信息来促进解耦
pub struct BranchContrastiveLoss {
    /// 温度参数
    temperature: f64,
}

impl BranchContrastiveLoss {
    /// 创建新的对比损失
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// 计算两个分支表示之间的对比损失
    ///
    /// 正样本对：同一分子的不同分支视图
    /// 负样本对：不同分子的分支视图
    pub fn compute_loss(&self, z1: &Tensor, z2: &Tensor) -> Tensor {
        // 归一化表示（沿最后一个维度计算L2范数）
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disentangled_attention() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();

        let attn = DisentangledMultiHeadAttention::new(&vs_root, 64, 8);

        let x = Tensor::randn(&[8, 10, 64], (Kind::Float, device));

        // 测试每个分支的独立注意力
        let topo_out = attn.topo_self_attention(&x);
        let geo_out = attn.geo_self_attention(&x);

        assert_eq!(topo_out.size()[0], 8);
        assert_eq!(topo_out.size()[1], 10);
        assert_eq!(topo_out.size()[2], 64);
        assert_eq!(geo_out.size()[0], 8);
        assert_eq!(geo_out.size()[1], 10);
        assert_eq!(geo_out.size()[2], 64);

        // 验证两个分支的输出不同（因为参数独立）
        let diff = (&topo_out - &geo_out)
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        assert!(diff > 1e-6, "Branches should produce different outputs");
    }

    #[test]
    fn test_contrastive_loss() {
        let device = Device::Cpu;
        let loss_fn = BranchContrastiveLoss::new(0.1);

        // 相似的表示（应该有较低的损失）
        let z1 = Tensor::randn(&[8, 64], (Kind::Float, device));
        let z2 = &z1 + 0.01 * Tensor::randn(&[8, 64], (Kind::Float, device));
        let loss_similar = loss_fn.compute_loss(&z1, &z2).double_value(&[]);

        // 不相似的表示（应该有较高的损失）
        let z3 = Tensor::randn(&[8, 64], (Kind::Float, device));
        let loss_different = loss_fn.compute_loss(&z1, &z3).double_value(&[]);

        println!(
            "Similar loss: {}, Different loss: {}",
            loss_similar, loss_different
        );
        assert!(
            loss_similar < loss_different,
            "Similar representations should have lower contrastive loss"
        );
    }
}
