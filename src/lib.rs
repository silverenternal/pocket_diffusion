#![allow(clippy::derivable_impls)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unused_enumerate_index)]

//! Pocket-conditioned modular research crate.
//!
//! The primary actively extended surface is the modular research stack under
//! `config`, `data`, `models`, `training`, and `experiments`.
//! Legacy dataset/demo/comparison APIs remain available for compatibility, and
//! are grouped more explicitly under `legacy`.

pub mod config;
pub mod data;
pub mod dataset;
pub mod disentangle;
pub mod egnn;
pub mod experiment;
pub mod experiments;
pub mod generator;
pub mod legacy;
pub mod losses;
pub mod models;
pub mod pocket;
pub mod representation;
pub mod runtime;
pub mod scorer;
pub mod se3_layers;
pub mod training;
pub mod types;

// se3_layers contains ndarray-based EGNN for comparison experiments
// It intentionally has a different RBFEncoder (ndarray vs tch-based)
pub use se3_layers::{CoordOnlyEGNN, SE3EquivariantLayer, TopologyAwareEGNN};
pub use types::{CandidateMolecule, GenerationResult};

use crate::egnn::EGNNConfig;
use crate::generator::{GeneratorConfig, PocketLigandGenerator};
use crate::pocket::PocketFeatureExtractor;
use crate::scorer::AffinityScorer;
use crate::types::Pocket;
use tch::nn;

/// 完整的生成-评分流水线
pub struct PocketDiffusionPipeline {
    /// 生成器
    generator: PocketLigandGenerator,
    /// 评分器
    scorer: AffinityScorer,
}

impl PocketDiffusionPipeline {
    /// 创建流水线
    pub fn new(vs: &nn::Path) -> Self {
        // 生成器配置
        let gen_config = GeneratorConfig::default();
        let generator = PocketLigandGenerator::new(vs, &gen_config);

        // 评分器配置
        let scorer_config = EGNNConfig {
            in_dim: 6,
            hidden_dim: 64,
            out_dim: 1,
            edge_dim: 50,
            use_attention: true,
        };
        let scorer = AffinityScorer::new(vs, &scorer_config, 3);

        Self { generator, scorer }
    }

    /// 执行完整的生成-筛选流程
    pub fn generate_and_rank(
        &self,
        pocket: &Pocket,
        num_candidates: usize,
        top_k: usize,
    ) -> GenerationResult {
        // 提取口袋特征
        let pocket_embedding = PocketFeatureExtractor::extract(pocket);

        // 生成候选分子
        let ligands = self
            .generator
            .generate_candidates(&pocket_embedding, num_candidates);

        // 评分并筛选
        let ranked = self.scorer.rank_candidates(pocket, &ligands, top_k);

        // 构建候选分子列表
        let candidates: Vec<CandidateMolecule> = ligands
            .into_iter()
            .map(|ligand| CandidateMolecule {
                ligand,
                affinity_score: None,
                qed_score: None,
                sa_score: None,
            })
            .collect();

        // 构建Top-K列表
        let top_candidates: Vec<CandidateMolecule> = ranked
            .iter()
            .map(|&(idx, score)| {
                let mut candidate = candidates[idx].clone();
                candidate.affinity_score = Some(score);
                candidate
            })
            .collect();

        GenerationResult {
            candidates,
            top_candidates,
        }
    }

    /// 获取生成器引用
    pub fn generator(&self) -> &PocketLigandGenerator {
        &self.generator
    }

    /// 获取评分器引用
    pub fn scorer(&self) -> &AffinityScorer {
        &self.scorer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pocket::create_example_prrsv_pocket;
    use tch::Device;

    #[test]
    fn modular_namespace_entrypoints_are_available() {
        let _inspect = |path: &str| training::inspect_dataset_from_config(path);
        let _train = |path: &str, resume: bool| training::run_training_from_config(path, resume);
        let _experiment =
            |path: &str, resume: bool| experiments::run_experiment_from_config(path, resume);
        let _device_parser = |device: &str| runtime::parse_runtime_device(device);
    }

    #[test]
    fn legacy_namespace_entrypoints_are_available() {
        let _demo = legacy::run_legacy_demo;
        let _comparison = legacy::run_comparison_experiment;
        let _pipeline_ctor = legacy::PocketDiffusionPipeline::new;
    }

    #[test]
    fn test_pipeline_creation() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();
        let _pipeline = PocketDiffusionPipeline::new(vs);
    }

    #[test]
    fn test_full_generation() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();
        let pipeline = PocketDiffusionPipeline::new(vs);

        let pocket = create_example_prrsv_pocket();
        let result = pipeline.generate_and_rank(&pocket, 5, 2);

        assert_eq!(result.candidates.len(), 5);
        assert_eq!(result.top_candidates.len(), 2);
    }
}
