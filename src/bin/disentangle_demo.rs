#![allow(clippy::needless_borrows_for_generic_args)]

//! 信息解耦Transformer训练演示
//! Disentangled Transformer Training Demo

use pocket_diffusion::disentangle::*;
use tch::{nn, Device, Kind, Tensor};

fn main() {
    env_logger::init();

    println!("================================================");
    println!("  自动信息解耦 Transformer 训练演示");
    println!("  Auto-Information Disentanglement Transformer");
    println!("================================================");

    let device = Device::Cpu;
    println!("\n设备: {:?}", device);

    // 配置
    let config = DisentangleConfig {
        hidden_dim: 64,
        in_dim: 10,
        pocket_dim: 12,
        mine_hidden_dim: 32,
        use_baseline_3d_only: false,
        num_heads: 8,
        num_layers: 2,
        warmup_steps: 50,
        disentangle_steps: 100,
        refinement_steps: 150,
        log_interval: 25,
    };

    println!("\n配置:");
    println!("  隐藏维度: {}", config.hidden_dim);
    println!("  注意力头数: {}", config.num_heads);
    println!("  Transformer层数: {}", config.num_layers);
    println!("  预热步数: {}", config.warmup_steps);
    println!("  解耦步数: {}", config.disentangle_steps);
    println!("  精炼步数: {}", config.refinement_steps);
    println!(
        "  总训练步数: {}",
        config.warmup_steps + config.disentangle_steps + config.refinement_steps
    );

    // 初始化变量存储和训练器
    println!("\n[初始化] 创建训练器...");
    let vs = nn::VarStore::new(device);
    let mut trainer = DisentangleTrainer::new(&vs.root(), config.clone());

    println!("\n================================================");
    println!("  开始三阶段训练");
    println!("================================================");

    // 模拟训练循环
    let num_steps = config.warmup_steps + config.disentangle_steps + config.refinement_steps;
    let num_atoms = 20i64;
    let num_classes = 10i64;

    for step in 0..num_steps {
        // 生成模拟数据
        let node_features = Tensor::randn(
            &[num_atoms, config.hidden_dim as i64],
            (Kind::Float, device),
        );
        let coords = Tensor::randn(&[num_atoms, 3i64], (Kind::Float, device)) * 10.0;
        // 真实原子类型: [num_atoms]
        let true_atom_types = Tensor::randint(num_classes, &[num_atoms], (Kind::Int64, device));
        let true_coords = Tensor::randn(&[num_atoms, 3i64], (Kind::Float, device));
        // 原子类型预测logits: [num_atoms, num_classes]
        let pred_atom_logits = Tensor::randn(&[num_atoms, num_classes], (Kind::Float, device));

        // 创建模拟分支输出 - MINE估计器需要2D输入[1, hidden_dim]
        let topo_global = node_features
            .mean_dim(0i64, false, Kind::Float)
            .unsqueeze(0);
        let geo_global = (&node_features * 0.8 + 0.1)
            .mean_dim(0i64, false, Kind::Float)
            .unsqueeze(0);
        let pocket_global = Tensor::randn(&[1i64, config.hidden_dim as i64], (Kind::Float, device));

        let topo_output = TopoBranchOutput {
            node_features: pred_atom_logits.shallow_clone(), // 预测logits用于损失计算
            global_embedding: topo_global,
            adjacency_logits: Tensor::randn(&[num_atoms, num_atoms], (Kind::Float, device)),
        };

        let geo_output = GeoBranchOutput {
            coords: coords.shallow_clone(),
            node_features: &node_features * 0.8 + 0.1,
            global_embedding: geo_global,
        };

        let pocket_output = PocketBranchOutput {
            pocket_embedding: Tensor::randn(&[config.pocket_dim as i64], (Kind::Float, device)),
            cross_attention_weights: Tensor::randn(&[num_atoms, 12i64], (Kind::Float, device))
                .softmax(-1i64, Kind::Float),
            global_embedding: pocket_global,
        };

        let outputs = DisentangledBranchesOutput {
            topo: topo_output,
            geo: geo_output,
            pocket: pocket_output,
            routing_probs: [0.33, 0.33, 0.34],
        };

        // 执行训练步
        let (_total_loss, _loss_components, metrics) =
            trainer.train_step(&outputs, &true_atom_types, &true_coords, None, None);

        // 打印进度
        trainer.print_progress(&metrics);

        // 阶段转换时打印额外信息
        if step == config.warmup_steps - 1 {
            println!("\n  >>> 阶段转换: 预热 → 解耦阶段 <<<\n");
        } else if step == config.warmup_steps + config.disentangle_steps - 1 {
            println!("\n  >>> 阶段转换: 解耦 → 精炼阶段 <<<\n");
        }
    }

    // 训练完成总结
    println!("\n================================================");
    println!("  训练完成!");
    println!("================================================");

    let stats = trainer.statistics();

    println!("\n最终训练统计:");
    println!("  总训练步数: {}", stats.total_steps);

    // 计算各阶段平均
    let avg_losses = stats.phase_average_losses(50);
    let (avg_mi_tg, avg_mi_tp, avg_mi_gp) = stats.phase_average_mi(50);

    println!("\n最终平均损失 (最后50步):");
    println!("  任务损失: {:.4}", avg_losses.task_loss);
    println!("  解耦损失: {:.4}", avg_losses.disentangle_loss);
    println!("  冗余损失: {:.4}", avg_losses.redundancy_loss);
    println!("  一致性损失: {:.4}", avg_losses.consistency_loss);

    println!("\n分支间平均互信息:");
    println!("  MI(拓扑; 几何): {:.4}", avg_mi_tg);
    println!("  MI(拓扑; 口袋): {:.4}", avg_mi_tp);
    println!("  MI(几何; 口袋): {:.4}", avg_mi_gp);

    let final_disentanglement_degree = stats.disentanglement_score();
    println!(
        "\n最终解耦度: {:.4} (越高表示分支信息独立性越好)",
        final_disentanglement_degree
    );

    // 基线比较演示
    println!("\n================================================");
    println!("  基线比较演示");
    println!("================================================");

    let mut comparison = BaselineComparison::new();

    // 模拟基线3D方法的指标（互信息更高表示信息混合更严重）
    let mut baseline_metrics = Vec::new();
    for i in 0..300 {
        baseline_metrics.push(TrainingMetrics {
            step: i,
            phase: if i < 50 {
                TrainingPhase::Warmup
            } else if i < 150 {
                TrainingPhase::Disentanglement
            } else {
                TrainingPhase::Refinement
            },
            losses: LossComponents {
                task_loss: 1.5 - (i as f64) * 0.003,
                disentangle_loss: 0.0,
                redundancy_loss: 0.0,
                consistency_loss: 0.0,
            },
            mutual_information: MutualInformationEstimate {
                mi_topo_geo: 0.8 - (i as f64) * 0.001, // 基线3D方法互信息更高
                mi_topo_pocket: 0.7 - (i as f64) * 0.001,
                mi_geo_pocket: 0.75 - (i as f64) * 0.001,
            },
            redundancy: BranchRedundancy {
                topo_redundancy: 0.4,
                geo_redundancy: 0.35,
                pocket_redundancy: 0.3,
            },
            disentanglement_degree: 0.3 + (i as f64) * 0.001, // 基线解耦度更低
            step_time_ms: 12.5,
            samples_per_second: 80.0,
            peak_memory_bytes: 0,
        });
    }

    comparison.record_baseline_3d(baseline_metrics);

    // 模拟解耦方法的指标
    let mut disentangled_metrics = Vec::new();
    for i in 0..300 {
        disentangled_metrics.push(TrainingMetrics {
            step: i,
            phase: if i < 50 {
                TrainingPhase::Warmup
            } else if i < 150 {
                TrainingPhase::Disentanglement
            } else {
                TrainingPhase::Refinement
            },
            losses: LossComponents {
                task_loss: 1.4 - (i as f64) * 0.003,
                disentangle_loss: 0.1 - (i as f64) * 0.0003,
                redundancy_loss: 0.05 - (i as f64) * 0.0001,
                consistency_loss: 0.02 - (i as f64) * 0.00005,
            },
            mutual_information: MutualInformationEstimate {
                mi_topo_geo: 0.4 - (i as f64) * 0.001, // 解耦方法互信息显著降低
                mi_topo_pocket: 0.35 - (i as f64) * 0.001,
                mi_geo_pocket: 0.38 - (i as f64) * 0.001,
            },
            redundancy: BranchRedundancy {
                topo_redundancy: 0.25,
                geo_redundancy: 0.22,
                pocket_redundancy: 0.2,
            },
            disentanglement_degree: 0.65 + (i as f64) * 0.001, // 解耦方法解耦度更高
            step_time_ms: 14.2,
            samples_per_second: 70.4,
            peak_memory_bytes: 0,
        });
    }

    comparison.record_disentangled(disentangled_metrics);

    // 打印比较报告
    println!("\n{}", comparison.generate_report());

    println!("\n================================================");
    println!("  演示完成!");
    println!("================================================");
}
