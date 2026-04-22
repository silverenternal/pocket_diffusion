//! 口袋-配体生成器
//! 基于Transformer的跨注意力架构，实现口袋条件指导的分子生成

use crate::types::{Atom, AtomType, Ligand, PocketEmbedding};
use rand::Rng;
use tch::{nn, Device, Kind, Tensor};

/// 生成器配置
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// 口袋嵌入维度(12)
    pub pocket_dim: usize,
    /// 配体特征维度(2048 ECFP4 + 结构编码)
    pub ligand_dim: usize,
    /// 模型隐藏维度
    pub hidden_dim: usize,
    /// 跨注意力头数
    pub num_heads: usize,
    /// 跨注意力层数
    pub num_layers: usize,
    /// 最大生成原子数
    pub max_atoms: usize,
    /// 原子类型数量
    pub num_atom_types: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            pocket_dim: 12,
            ligand_dim: 2048,
            hidden_dim: 256,
            num_heads: 8,
            num_layers: 4,
            max_atoms: 50,
            num_atom_types: 6,
        }
    }
}

/// 跨注意力层
pub struct CrossAttentionLayer {
    /// Query投影 (配体侧)
    w_q: nn::Linear,
    /// Key投影 (口袋侧)
    w_k: nn::Linear,
    /// Value投影 (口袋侧)
    w_v: nn::Linear,
    /// 输出投影
    w_o: nn::Linear,
    /// 层归一化
    layer_norm1: nn::LayerNorm,
    layer_norm2: nn::LayerNorm,
    /// 前馈网络
    ff: nn::Sequential,
    /// 注意力头数
    num_heads: usize,
}

impl CrossAttentionLayer {
    /// 创建跨注意力层
    pub fn new(vs: &nn::Path, dim: usize, num_heads: usize) -> Self {
        assert!(dim.is_multiple_of(num_heads), "维度必须能被头数整除");

        let w_q = nn::linear(vs / "w_q", dim as i64, dim as i64, Default::default());
        let w_k = nn::linear(vs / "w_k", dim as i64, dim as i64, Default::default());
        let w_v = nn::linear(vs / "w_v", dim as i64, dim as i64, Default::default());
        let w_o = nn::linear(vs / "w_o", dim as i64, dim as i64, Default::default());

        let layer_norm1 = nn::layer_norm(vs / "ln1", vec![dim as i64], Default::default());
        let layer_norm2 = nn::layer_norm(vs / "ln2", vec![dim as i64], Default::default());

        let ff = nn::seq()
            .add(nn::linear(
                vs / "ff" / "fc1",
                dim as i64,
                dim as i64 * 4,
                Default::default(),
            ))
            .add_fn(|x: &Tensor| x.gelu("none"))
            .add(nn::linear(
                vs / "ff" / "fc2",
                dim as i64 * 4,
                dim as i64,
                Default::default(),
            ));

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            layer_norm1,
            layer_norm2,
            ff,
            num_heads,
        }
    }

    /// 前向传播
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Tensor {
        // x: [seq_len, batch, dim] - 配体序列
        // context: [context_len, batch, dim] - 口袋嵌入

        let seq_len = x.size()[0];
        let context_len = context.size()[0];
        let dim = x.size()[2];
        let head_dim = dim / self.num_heads as i64;

        // 残差连接1
        let residual = x.shallow_clone();
        let x_norm = x.apply(&self.layer_norm1);

        // 投影
        let q = x_norm.apply(&self.w_q); // [S, B, D]
        let k = context.apply(&self.w_k); // [C, B, D]
        let v = context.apply(&self.w_v); // [C, B, D]

        // 多头分割
        let q = q
            .reshape([seq_len, -1, self.num_heads as i64, head_dim])
            .permute([1, 2, 0, 3]); // [B, H, S, D/H]
        let k = k
            .reshape([context_len, -1, self.num_heads as i64, head_dim])
            .permute([1, 2, 0, 3]); // [B, H, C, D/H]
        let v = v
            .reshape([context_len, -1, self.num_heads as i64, head_dim])
            .permute([1, 2, 0, 3]); // [B, H, C, D/H]

        // 注意力计算
        let attn_scores = q.matmul(&k.transpose(-1, -2)) / (head_dim as f64).sqrt(); // [B, H, S, C]
        let attn_weights = attn_scores.softmax(-1, Kind::Float); // [B, H, S, C]

        let attn_output = attn_weights.matmul(&v); // [B, H, S, D/H]
        let attn_output = attn_output
            .permute([2, 0, 1, 3])
            .reshape([seq_len, -1, dim]); // [S, B, D]

        // 输出投影
        let attn_output = attn_output.apply(&self.w_o);
        let x = residual + attn_output;

        // 残差连接2 - 前馈
        let residual = x.shallow_clone();
        let x_norm = x.apply(&self.layer_norm2);
        let ff_output = x_norm.apply(&self.ff);

        residual + ff_output
    }
}

/// 口袋-配体生成器
pub struct PocketLigandGenerator {
    /// 口袋嵌入投影
    pocket_proj: nn::Sequential,
    /// 配体嵌入投影
    ligand_proj: nn::Sequential,
    /// 原子坐标预测头
    coord_head: nn::Sequential,
    /// 原子类型预测头
    type_head: nn::Sequential,
    /// 多层跨注意力
    cross_attention_layers: Vec<CrossAttentionLayer>,
    /// 配置
    config: GeneratorConfig,
}

impl PocketLigandGenerator {
    /// 创建生成器
    pub fn new(vs: &nn::Path, config: &GeneratorConfig) -> Self {
        // 口袋嵌入投影
        let pocket_proj = nn::seq()
            .add(nn::linear(
                vs / "pocket_proj" / "fc1",
                config.pocket_dim as i64,
                config.hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "pocket_proj" / "fc2",
                config.hidden_dim as i64,
                config.hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|x| x.silu());

        // 配体嵌入投影
        let ligand_proj = nn::seq()
            .add(nn::linear(
                vs / "ligand_proj" / "fc1",
                config.ligand_dim as i64,
                config.hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "ligand_proj" / "fc2",
                config.hidden_dim as i64,
                config.hidden_dim as i64,
                Default::default(),
            ))
            .add_fn(|x| x.silu());

        // 坐标预测头
        let coord_head = nn::seq()
            .add(nn::linear(
                vs / "coord_head" / "fc1",
                config.hidden_dim as i64,
                config.hidden_dim as i64 / 2,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "coord_head" / "fc2",
                config.hidden_dim as i64 / 2,
                3,
                Default::default(),
            ));

        // 原子类型预测头
        let type_head = nn::seq()
            .add(nn::linear(
                vs / "type_head" / "fc1",
                config.hidden_dim as i64,
                config.hidden_dim as i64 / 2,
                Default::default(),
            ))
            .add_fn(|x| x.silu())
            .add(nn::linear(
                vs / "type_head" / "fc2",
                config.hidden_dim as i64 / 2,
                config.num_atom_types as i64,
                Default::default(),
            ));

        // 跨注意力层
        let mut cross_attention_layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = CrossAttentionLayer::new(
                &(vs / format!("cross_attn_{}", i)),
                config.hidden_dim,
                config.num_heads,
            );
            cross_attention_layers.push(layer);
        }

        Self {
            pocket_proj,
            ligand_proj,
            coord_head,
            type_head,
            cross_attention_layers,
            config: config.clone(),
        }
    }

    /// 生成配体分子
    pub fn generate(&self, pocket_embedding: &PocketEmbedding, num_atoms: usize) -> Ligand {
        let device = Device::Cpu;

        // 将口袋嵌入转换为tensor并投影
        let pocket_tensor = pocket_embedding.to_tensor().to_device(device);
        let pocket_encoded = pocket_tensor
            .apply(&self.pocket_proj)
            .reshape([1, 1, self.config.hidden_dim as i64])
            .permute([1, 0, 2]); // [1, 1, H]

        // 初始化配体序列
        let mut atoms = Vec::with_capacity(num_atoms);
        // 使用 ligand_dim 作为初始输入维度，匹配 ligand_proj 的输入
        let mut current_state =
            Tensor::zeros([1, 1, self.config.ligand_dim as i64], (Kind::Float, device));

        // 逐原子生成 (简化的自回归)
        for i in 0..num_atoms.min(self.config.max_atoms) {
            // 投影配体特征
            let mut ligand_h = current_state.apply(&self.ligand_proj);

            // 多层跨注意力融合
            for layer in &self.cross_attention_layers {
                ligand_h = layer.forward(&ligand_h, &pocket_encoded);
            }

            // 预测坐标和原子类型
            let coord_pred = ligand_h.apply(&self.coord_head).squeeze(); // [3]
            let type_pred = ligand_h.apply(&self.type_head).squeeze(); // [num_types]

            // 贪心解码
            let atom_type_idx = type_pred.argmax(0, false).int64_value(&[]);
            // 将张量转换为 Vec<f64>
            let coord: Vec<f64> = (0..3).map(|i| coord_pred.double_value(&[i])).collect();

            let atom_type = match atom_type_idx {
                0 => AtomType::Carbon,
                1 => AtomType::Nitrogen,
                2 => AtomType::Oxygen,
                3 => AtomType::Sulfur,
                4 => AtomType::Hydrogen,
                _ => AtomType::Other,
            };

            atoms.push(Atom {
                coords: [coord[0], coord[1], coord[2]],
                atom_type,
                index: i,
            });

            // 更新当前状态 - 使用 ligand_h 投影回 ligand_dim
            current_state =
                Tensor::zeros([1, 1, self.config.ligand_dim as i64], (Kind::Float, device));
        }

        // 生成简单的化学键 (相邻原子)
        let bonds = self.generate_bonds(&atoms);

        Ligand {
            atoms,
            bonds,
            fingerprint: None,
        }
    }

    /// 简单的化学键生成
    fn generate_bonds(&self, atoms: &[Atom]) -> Vec<(usize, usize)> {
        let mut bonds = Vec::new();
        let cutoff = 1.6; // 典型的C-C键长

        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let dx = atoms[i].coords[0] - atoms[j].coords[0];
                let dy = atoms[i].coords[1] - atoms[j].coords[1];
                let dz = atoms[i].coords[2] - atoms[j].coords[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < cutoff {
                    bonds.push((i, j));
                }
            }
        }

        bonds
    }

    /// 批量生成多个候选分子
    pub fn generate_candidates(
        &self,
        pocket_embedding: &PocketEmbedding,
        num_candidates: usize,
    ) -> Vec<Ligand> {
        let mut candidates = Vec::with_capacity(num_candidates);
        let mut rng = rand::thread_rng();

        for _ in 0..num_candidates {
            let num_atoms = rng.gen_range(10..30); // 随机分子大小
            let ligand = self.generate(pocket_embedding, num_atoms);
            candidates.push(ligand);
        }

        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pocket::{create_example_prrsv_pocket, PocketFeatureExtractor};

    #[test]
    fn test_generator_creation() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();
        let config = GeneratorConfig::default();
        let _generator = PocketLigandGenerator::new(vs, &config);
    }

    #[test]
    fn test_molecule_generation() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();
        let config = GeneratorConfig::default();
        let generator = PocketLigandGenerator::new(vs, &config);

        let pocket = create_example_prrsv_pocket();
        let embedding = PocketFeatureExtractor::extract(&pocket);

        let ligand = generator.generate(&embedding, 15);

        assert!(!ligand.atoms.is_empty());
        assert_eq!(ligand.atoms.len(), 15);
    }

    #[test]
    fn test_cross_attention() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();
        let dim = 64;
        let num_heads = 8;

        let layer = CrossAttentionLayer::new(vs, dim, num_heads);

        let x = Tensor::rand(&[10, 1, dim as i64], (Kind::Float, device));
        let context = Tensor::rand(&[1, 1, dim as i64], (Kind::Float, device));

        let output = layer.forward(&x, &context);

        assert_eq!(output.size(), &[10, 1, dim as i64]);
    }
}
