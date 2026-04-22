//! 基于口袋条件扩散的结构感知小分子生成 - 主程序示例

use pocket_diffusion::*;
use std::env;
use std::path::Path;
use tch::{nn, Device};

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if let Some(path) = value_after_flag(&args, "--experiment-config") {
        run_experiment_from_config(path);
        return;
    }
    if let Some(path) = value_after_flag(&args, "--train-config") {
        run_training_from_config(path);
        return;
    }
    if let Some(path) = value_after_flag(&args, "--inspect-config") {
        inspect_dataset_from_config(path);
        return;
    }
    if args.iter().any(|arg| arg == "--phase4") {
        run_phase4_experiment_demo();
        return;
    }
    if args.iter().any(|arg| arg == "--train-phase3") {
        run_phase3_training_demo();
        return;
    }
    if args.iter().any(|arg| arg == "--phase1") {
        run_phase1_demo();
        return;
    }

    println!("================================================");
    println!("  基于口袋条件扩散的结构感知小分子生成系统");
    println!("  Pocket-Conditioned Diffusion Molecule Gen");
    println!("================================================");

    // 解析命令行参数
    let num_candidates = if args.len() > 1 {
        args[1].parse().unwrap_or(10)
    } else {
        10
    };
    let top_k = if args.len() > 2 {
        args[2].parse().unwrap_or(3)
    } else {
        3
    };

    println!("\n配置:");
    println!("  - 候选分子数量: {}", num_candidates);
    println!("  - Top-K筛选: {}", top_k);

    // 初始化模型
    println!("\n[1/4] 初始化神经网络模型...");
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let pipeline = PocketDiffusionPipeline::new(&vs.root());

    // 创建示例蛋白口袋 (PRRSV核衣壳蛋白)
    println!("\n[2/4] 加载目标蛋白口袋 (PRRSV核衣壳蛋白)...");
    let pocket = create_example_prrsv_pocket();
    let embedding = PocketFeatureExtractor::extract(&pocket);

    println!("  口袋原子数: {}", embedding.total_atoms);
    println!("  重原子数: {}", embedding.heavy_atoms);
    println!("  碳原子数: {}", embedding.carbon_count);
    println!("  氮原子数: {}", embedding.nitrogen_count);
    println!("  氧原子数: {}", embedding.oxygen_count);
    println!("  硫原子数: {}", embedding.sulfur_count);
    println!("  口袋半径: {:.2} Å", embedding.pocket_radius);
    println!("  坐标标准差: {:.2} Å", embedding.coord_std);

    // 生成候选分子
    println!("\n[3/4] 生成候选分子集...");
    let result = pipeline.generate_and_rank(&pocket, num_candidates, top_k);

    println!("  生成候选分子: {} 个", result.candidates.len());
    println!(
        "  平均原子数: {:.1}",
        result
            .candidates
            .iter()
            .map(|c| c.ligand.atoms.len() as f64)
            .sum::<f64>()
            / result.candidates.len() as f64
    );

    // 显示Top-K结果
    println!("\n[4/4] Top-{} 亲和力最高的分子:", top_k);
    println!("--------------------------------");

    for (i, candidate) in result.top_candidates.iter().enumerate() {
        let affinity = candidate.affinity_score.unwrap_or(0.0);
        let num_atoms = candidate.ligand.atoms.len();
        let num_bonds = candidate.ligand.bonds.len();

        println!("\n排名 #{}:", i + 1);
        println!("  结合亲和力: {:.2} kcal/mol", affinity);
        println!("  原子数量: {}", num_atoms);
        println!("  化学键数量: {}", num_bonds);

        // 统计原子类型
        let mut c_count = 0;
        let mut n_count = 0;
        let mut o_count = 0;
        let mut s_count = 0;
        let mut h_count = 0;

        for atom in &candidate.ligand.atoms {
            match atom.atom_type {
                AtomType::Carbon => c_count += 1,
                AtomType::Nitrogen => n_count += 1,
                AtomType::Oxygen => o_count += 1,
                AtomType::Sulfur => s_count += 1,
                AtomType::Hydrogen => h_count += 1,
                _ => {}
            }
        }

        println!(
            "  原子组成: C({}) N({}) O({}) S({}) H({})",
            c_count, n_count, o_count, s_count, h_count
        );

        // 显示前5个原子坐标
        println!("  前5个原子坐标:");
        for (j, atom) in candidate.ligand.atoms.iter().take(5).enumerate() {
            println!(
                "    [{}] ({:?}) [{:7.3}, {:7.3}, {:7.3}]",
                j, atom.atom_type, atom.coords[0], atom.coords[1], atom.coords[2]
            );
        }
    }

    println!("\n================================================");
    println!("  生成完成!");
    println!("================================================");
}

fn run_phase1_demo() {
    println!("================================================");
    println!("  Phase 1 Research Architecture Demo");
    println!("================================================");

    let config = ResearchConfig::default();
    let dataset = InMemoryDataset::new(synthetic_phase1_examples());
    let splits = dataset.split_by_protein(3, 5);
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&vs.root(), &config);

    let train_examples = splits.train.examples();
    let (batch, outputs) = system.forward_batch(train_examples);

    println!("dataset:");
    println!("  train examples: {}", splits.train.len());
    println!("  val examples: {}", splits.val.len());
    println!("  test examples: {}", splits.test.len());

    println!("batch:");
    println!("  ligand atom tensor: {:?}", batch.atom_types.size());
    println!("  ligand coord tensor: {:?}", batch.ligand_coords.size());
    println!(
        "  pocket feature tensor: {:?}",
        batch.pocket_atom_features.size()
    );

    if let Some(first) = outputs.first() {
        println!("encodings:");
        println!(
            "  topology pooled: {:?}, tokens: {:?}",
            first.encodings.topology.pooled_embedding.size(),
            first.encodings.topology.token_embeddings.size()
        );
        println!(
            "  geometry pooled: {:?}, tokens: {:?}",
            first.encodings.geometry.pooled_embedding.size(),
            first.encodings.geometry.token_embeddings.size()
        );
        println!(
            "  pocket pooled: {:?}, tokens: {:?}",
            first.encodings.pocket.pooled_embedding.size(),
            first.encodings.pocket.token_embeddings.size()
        );

        println!("slots:");
        println!(
            "  topology slots: {:?}, weights: {:?}",
            first.slots.topology.slots.size(),
            first.slots.topology.slot_weights.size()
        );
        println!(
            "  geometry slots: {:?}, weights: {:?}",
            first.slots.geometry.slots.size(),
            first.slots.geometry.slot_weights.size()
        );
        println!(
            "  pocket slots: {:?}, weights: {:?}",
            first.slots.pocket.slots.size(),
            first.slots.pocket.slot_weights.size()
        );

        println!("gates:");
        println!(
            "  topo<-geo: {:.4}, topo<-pocket: {:.4}",
            first.interactions.topo_from_geo.gate.double_value(&[0]),
            first.interactions.topo_from_pocket.gate.double_value(&[0])
        );
        println!(
            "  geo<-topo: {:.4}, geo<-pocket: {:.4}",
            first.interactions.geo_from_topo.gate.double_value(&[0]),
            first.interactions.geo_from_pocket.gate.double_value(&[0])
        );
        println!(
            "  pocket<-topo: {:.4}, pocket<-geo: {:.4}",
            first.interactions.pocket_from_topo.gate.double_value(&[0]),
            first.interactions.pocket_from_geo.gate.double_value(&[0])
        );

        println!("probes:");
        println!(
            "  topology adjacency logits: {:?}",
            first.probes.topology_adjacency_logits.size()
        );
        println!(
            "  geometry distance predictions: {:?}",
            first.probes.geometry_distance_predictions.size()
        );
        println!(
            "  pocket feature predictions: {:?}",
            first.probes.pocket_feature_predictions.size()
        );
    }
}

fn run_phase3_training_demo() {
    println!("================================================");
    println!("  Phase 3 Staged Training Demo");
    println!("================================================");

    let mut config = ResearchConfig::default();
    config.training.max_steps = 4;
    config.training.checkpoint_every = 100;
    config.training.log_every = 1;

    let dataset = InMemoryDataset::new(synthetic_phase1_examples());
    let train_examples = dataset.examples().to_vec();

    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&vs.root(), &config);
    let mut trainer =
        ResearchTrainer::new(&vs, config.clone()).expect("trainer init should succeed");

    for _ in 0..config.training.max_steps {
        let metrics = trainer
            .train_step(&vs, &system, &train_examples)
            .expect("train step should succeed");
        println!(
            "step {} [{:?}] total={:.4} task={:.4} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4}",
            metrics.step,
            metrics.stage,
            metrics.losses.total,
            metrics.losses.task,
            metrics.losses.intra_red,
            metrics.losses.probe,
            metrics.losses.leak,
            metrics.losses.gate,
            metrics.losses.slot,
            metrics.losses.consistency,
        );
    }
}

fn run_phase4_experiment_demo() {
    println!("================================================");
    println!("  Phase 4 Unseen-Pocket Experiment Demo");
    println!("================================================");

    let mut config = UnseenPocketExperimentConfig::default();
    config.research.training.max_steps = 4;
    config.research.training.checkpoint_every = 100;

    let summary = UnseenPocketExperiment::run(config).expect("phase4 experiment should succeed");

    println!("training:");
    println!("  steps: {}", summary.training_history.len());
    if let Some(last) = summary.training_history.last() {
        println!(
            "  last stage: {:?}, last total loss: {:.4}",
            last.stage, last.losses.total
        );
    }

    println!("validation:");
    print_eval_metrics(&summary.validation);

    println!("test:");
    print_eval_metrics(&summary.test);
}

fn print_eval_metrics(metrics: &EvaluationMetrics) {
    println!("  validity: {:.4}", metrics.validity);
    println!("  uniqueness: {:.4}", metrics.uniqueness);
    println!("  novelty: {:.4}", metrics.novelty);
    println!("  distance rmse: {:.4}", metrics.distance_rmse);
    println!("  affinity alignment: {:.4}", metrics.affinity_alignment);
    println!("  affinity mae: {:.4}", metrics.affinity_mae);
    println!("  affinity rmse: {:.4}", metrics.affinity_rmse);
    println!("  labeled fraction: {:.4}", metrics.labeled_fraction);
    for group in &metrics.affinity_by_measurement {
        println!(
            "  affinity [{}]: count={} mae={:.4} rmse={:.4}",
            group.measurement_type, group.count, group.mae, group.rmse
        );
    }
    println!("  memory usage mb: {:.4}", metrics.memory_usage_mb);
    println!("  eval time ms: {:.4}", metrics.evaluation_time_ms);
    println!("  reconstruction mse: {:.4}", metrics.reconstruction_mse);
    println!("  slot usage mean: {:.4}", metrics.slot_usage_mean);
    println!("  gate usage mean: {:.4}", metrics.gate_usage_mean);
    println!("  leakage mean: {:.4}", metrics.leakage_mean);
}

fn run_training_from_config(path: impl AsRef<Path>) {
    let config = load_research_config(path.as_ref()).expect("config should load");
    let dataset = InMemoryDataset::from_data_config(&config.data).expect("dataset should load");
    let splits = dataset.split_by_protein_fraction_with_options(
        config.data.val_fraction,
        config.data.test_fraction,
        config.data.split_seed,
        config.data.stratify_by_measurement,
    );

    println!("================================================");
    println!("  Config-Driven Training Run");
    println!("================================================");
    println!("dataset:");
    println!("  total examples: {}", dataset.len());
    println!("  train: {}", splits.train.len());
    println!("  val: {}", splits.val.len());
    println!("  test: {}", splits.test.len());

    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&vs.root(), &config);
    let mut trainer = ResearchTrainer::new(&vs, config.clone()).expect("trainer init should work");
    let train_examples = splits.train.examples().to_vec();

    for _ in 0..config.training.max_steps {
        let metrics = trainer
            .train_step(&vs, &system, &train_examples)
            .expect("train step should succeed");
        println!(
            "step {} [{:?}] total={:.4} task={:.4} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4}",
            metrics.step,
            metrics.stage,
            metrics.losses.total,
            metrics.losses.task,
            metrics.losses.intra_red,
            metrics.losses.probe,
            metrics.losses.leak,
            metrics.losses.gate,
            metrics.losses.slot,
            metrics.losses.consistency,
        );
    }
}

fn run_experiment_from_config(path: impl AsRef<Path>) {
    let summary =
        UnseenPocketExperiment::run(load_experiment_config(path).expect("config should load"))
            .expect("experiment should succeed");

    println!("================================================");
    println!("  Config-Driven Unseen-Pocket Experiment");
    println!("================================================");
    println!("training:");
    println!("  steps: {}", summary.training_history.len());
    if let Some(last) = summary.training_history.last() {
        println!(
            "  last stage: {:?}, last total loss: {:.4}",
            last.stage, last.losses.total
        );
    }
    println!("validation:");
    print_eval_metrics(&summary.validation);
    println!("test:");
    print_eval_metrics(&summary.test);
}

fn inspect_dataset_from_config(path: impl AsRef<Path>) {
    let config = load_research_config(path.as_ref().to_path_buf()).expect("config should load");
    let dataset = InMemoryDataset::from_data_config(&config.data).expect("dataset should load");
    let splits = dataset.split_by_protein_fraction_with_options(
        config.data.val_fraction,
        config.data.test_fraction,
        config.data.split_seed,
        config.data.stratify_by_measurement,
    );

    println!("================================================");
    println!("  Dataset Inspection");
    println!("================================================");
    println!("total examples: {}", dataset.len());
    println!("train examples: {}", splits.train.len());
    println!("val examples: {}", splits.val.len());
    println!("test examples: {}", splits.test.len());

    for example in dataset.examples().iter().take(3) {
        println!(
            "  {} | protein={} | ligand_atoms={} | pocket_atoms={} | affinity={:?} | measurement={:?} {:?} {:?}",
            example.example_id,
            example.protein_id,
            example.geometry.coords.size()[0],
            example.pocket.coords.size()[0],
            example.targets.affinity_kcal_mol,
            example.targets.affinity_measurement_type,
            example.targets.affinity_raw_value,
            example.targets.affinity_raw_unit,
        );
    }
}

fn value_after_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}
