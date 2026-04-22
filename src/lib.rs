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

use crate::config::ResearchConfig;
use crate::data::MolecularExample;
use crate::egnn::EGNNConfig;
use crate::generator::{GeneratorConfig, PocketLigandGenerator};
use crate::models::{
    candidate_records_to_legacy, generate_candidates_from_forward, Phase1ResearchSystem,
};
use crate::pocket::PocketFeatureExtractor;
use crate::scorer::AffinityScorer;
use crate::types::{Atom, AtomType, Ligand, Pocket};
use tch::{nn, Device};

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

    /// Route generation through the modular decoder while preserving legacy ranking.
    pub fn generate_and_rank_with_modular_bridge(
        &self,
        pocket: &Pocket,
        num_candidates: usize,
        top_k: usize,
    ) -> GenerationResult {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = bridge_example_from_pocket(pocket, &config);
        let forward = system.forward_example(&example);
        let records = generate_candidates_from_forward(&example, &forward, num_candidates.max(1));
        let candidates = candidate_records_to_legacy(&records);
        let ligands: Vec<Ligand> = candidates
            .iter()
            .map(|candidate| candidate.ligand.clone())
            .collect();
        let ranked = self.scorer.rank_candidates(pocket, &ligands, top_k);
        let top_candidates = ranked
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

fn bridge_example_from_pocket(pocket: &Pocket, config: &ResearchConfig) -> MolecularExample {
    let ligand = seed_ligand_from_pocket(pocket);
    MolecularExample::from_legacy_with_targets_and_generation(
        "legacy-bridge-example",
        pocket.name.clone(),
        &ligand,
        pocket,
        Default::default(),
        &config.data.generation_target,
    )
    .with_pocket_feature_dim(config.model.pocket_feature_dim)
}

fn seed_ligand_from_pocket(pocket: &Pocket) -> Ligand {
    let centroid = if pocket.atoms.is_empty() {
        [0.0_f64, 0.0, 0.0]
    } else {
        let mut center = [0.0_f64; 3];
        for atom in &pocket.atoms {
            center[0] += atom.coords[0];
            center[1] += atom.coords[1];
            center[2] += atom.coords[2];
        }
        let denom = pocket.atoms.len() as f64;
        [center[0] / denom, center[1] / denom, center[2] / denom]
    };
    let atom_types = [
        AtomType::Carbon,
        AtomType::Carbon,
        AtomType::Nitrogen,
        AtomType::Oxygen,
        AtomType::Carbon,
    ];
    let offsets = [
        [-0.8, 0.0, 0.0],
        [0.0, 0.9, 0.0],
        [0.8, 0.0, 0.2],
        [0.0, -0.9, -0.2],
        [0.0, 0.0, 1.0],
    ];
    let atoms = atom_types
        .iter()
        .zip(offsets.iter())
        .enumerate()
        .map(|(index, (atom_type, offset))| Atom {
            coords: [
                centroid[0] + offset[0],
                centroid[1] + offset[1],
                centroid[2] + offset[2],
            ],
            atom_type: *atom_type,
            index,
        })
        .collect();

    Ligand {
        atoms,
        bonds: vec![(0, 1), (1, 2), (1, 3), (1, 4)],
        fingerprint: None,
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
