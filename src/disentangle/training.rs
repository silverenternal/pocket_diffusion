//! 三阶段训练管道
//! Three-Phase Training Pipeline
//!
//! 阶段1: 预热 (Warmup) - 仅任务损失
//! 阶段2: 解耦 (Disentanglement) - 添加互信息最小化损失
//! 阶段3: 精炼 (Refinement) - 所有损失 + 一致性损失

use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

use crate::disentangle::loss::TotalLoss;
use crate::disentangle::types::{
    DisentangleConfig, DisentangledBranchesOutput, LossComponents, TrainingMetrics, TrainingPhase,
};

/// 训练统计信息收集器
#[derive(Debug, Clone)]
pub struct TrainingStatistics {
    /// 总步数
    pub total_steps: usize,
    /// 当前阶段步数
    pub phase_steps: usize,
    /// 当前阶段
    pub current_phase: TrainingPhase,
    /// 历史损失值
    pub loss_history: Vec<(usize, LossComponents)>,
    /// 历史互信息值
    pub mi_history: Vec<(usize, f64, f64, f64)>,
    /// 历史冗余度值
    pub redundancy_history: Vec<(usize, f64, f64, f64)>,
}

impl Default for TrainingStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingStatistics {
    /// 创建新的训练统计信息收集器
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            phase_steps: 0,
            current_phase: TrainingPhase::Warmup,
            loss_history: Vec::new(),
            mi_history: Vec::new(),
            redundancy_history: Vec::new(),
        }
    }

    /// 记录一步训练的统计信息
    pub fn record_step(
        &mut self,
        losses: LossComponents,
        mi_topo_geo: f64,
        mi_topo_pocket: f64,
        mi_geo_pocket: f64,
        redundancy_topo: f64,
        redundancy_geo: f64,
        redundancy_pocket: f64,
    ) {
        self.loss_history.push((self.total_steps, losses));
        self.mi_history
            .push((self.total_steps, mi_topo_geo, mi_topo_pocket, mi_geo_pocket));
        self.redundancy_history.push((
            self.total_steps,
            redundancy_topo,
            redundancy_geo,
            redundancy_pocket,
        ));
        self.total_steps += 1;
        self.phase_steps += 1;
    }

    /// 转换到新阶段
    pub fn transition_to_phase(&mut self, phase: TrainingPhase) {
        self.current_phase = phase;
        self.phase_steps = 0;
    }

    /// 获取当前阶段的平均损失
    pub fn phase_average_losses(&self, last_n: usize) -> LossComponents {
        let recent = &self.loss_history[self.loss_history.len().saturating_sub(last_n)..];
        let n = recent.len() as f64;
        if n == 0.0 {
            return LossComponents {
                task_loss: 0.0,
                disentangle_loss: 0.0,
                redundancy_loss: 0.0,
                consistency_loss: 0.0,
            };
        }

        let mut task_loss = 0.0;
        let mut disentangle_loss = 0.0;
        let mut redundancy_loss = 0.0;
        let mut consistency_loss = 0.0;

        for (_, l) in recent {
            task_loss += l.task_loss;
            disentangle_loss += l.disentangle_loss;
            redundancy_loss += l.redundancy_loss;
            consistency_loss += l.consistency_loss;
        }

        LossComponents {
            task_loss: task_loss / n,
            disentangle_loss: disentangle_loss / n,
            redundancy_loss: redundancy_loss / n,
            consistency_loss: consistency_loss / n,
        }
    }

    /// 获取当前阶段的平均互信息
    pub fn phase_average_mi(&self, last_n: usize) -> (f64, f64, f64) {
        let recent = &self.mi_history[self.mi_history.len().saturating_sub(last_n)..];
        let n = recent.len() as f64;
        if n == 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let mut mi_topo_geo = 0.0;
        let mut mi_topo_pocket = 0.0;
        let mut mi_geo_pocket = 0.0;

        for (_, m1, m2, m3) in recent {
            mi_topo_geo += m1;
            mi_topo_pocket += m2;
            mi_geo_pocket += m3;
        }

        (mi_topo_geo / n, mi_topo_pocket / n, mi_geo_pocket / n)
    }

    /// 获取解耦度（基于平均互信息）
    pub fn disentanglement_score(&self) -> f64 {
        let (mi1, mi2, mi3) = self.phase_average_mi(50);
        let avg_mi = (mi1 + mi2 + mi3) / 3.0;
        // 使用sigmoid归一化，互信息越小解耦度越高
        1.0 - (1.0 / (1.0 + (-avg_mi).exp()))
    }
}

/// 解耦Transformer训练器
pub struct DisentangleTrainer {
    /// 配置
    config: DisentangleConfig,
    /// 总损失计算器
    loss_calculator: TotalLoss,
    /// 训练统计信息
    statistics: TrainingStatistics,
    /// 设备
    device: Device,
}

impl DisentangleTrainer {
    /// 创建新的训练器
    pub fn new(vs: &nn::Path, config: DisentangleConfig) -> Self {
        let device = vs.device();
        let loss_calculator =
            TotalLoss::new(vs, config.hidden_dim, config.num_heads, device, 0.1, true);
        let statistics = TrainingStatistics::new();

        Self {
            config,
            loss_calculator,
            statistics,
            device,
        }
    }

    /// 获取当前训练阶段
    pub fn current_phase(&self) -> TrainingPhase {
        self.statistics.current_phase
    }

    /// 根据步数确定训练阶段
    pub fn get_phase_for_step(&self, step: usize) -> TrainingPhase {
        if step < self.config.warmup_steps {
            TrainingPhase::Warmup
        } else if step < self.config.warmup_steps + self.config.disentangle_steps {
            TrainingPhase::Disentanglement
        } else {
            TrainingPhase::Refinement
        }
    }

    /// 执行单步训练
    ///
    /// 返回（总损失张量，损失组件，训练指标）
    pub fn train_step(
        &mut self,
        outputs: &DisentangledBranchesOutput,
        true_atom_types: &Tensor,
        true_coords: &Tensor,
        true_bond_types: Option<&Tensor>,
        pred_bond_types: Option<&Tensor>,
    ) -> (Tensor, LossComponents, TrainingMetrics) {
        let start_time = Instant::now();
        let step = self.statistics.total_steps;

        // 确定当前阶段并在需要时转换阶段
        let phase = self.get_phase_for_step(step);
        if phase != self.statistics.current_phase {
            self.statistics.transition_to_phase(phase);
        }

        // 计算损失
        let loss_components = self.loss_calculator.compute_total_loss(
            outputs,
            true_atom_types,
            true_coords,
            true_bond_types,
            pred_bond_types,
            phase,
        );

        // 获取互信息监控值
        let (mi_topo_geo, mi_topo_pocket, mi_geo_pocket) = self.loss_calculator.get_mi_monitor(
            &outputs.topo.global_embedding,
            &outputs.geo.global_embedding,
            &outputs.pocket.global_embedding,
        );

        // 获取冗余度监控值
        let (redundancy_topo, redundancy_geo, redundancy_pocket) =
            self.loss_calculator.get_redundancy_monitor(
                &outputs.topo.global_embedding,
                &outputs.geo.global_embedding,
                &outputs.pocket.global_embedding,
            );

        // 记录统计信息
        self.statistics.record_step(
            loss_components,
            mi_topo_geo,
            mi_topo_pocket,
            mi_geo_pocket,
            redundancy_topo,
            redundancy_geo,
            redundancy_pocket,
        );

        // 计算总损失（带权重）
        let (alpha, beta, gamma, delta) = phase.loss_weights();
        let total_loss = Tensor::from(
            alpha * loss_components.task_loss
                + beta * loss_components.disentangle_loss
                + gamma * loss_components.redundancy_loss
                + delta * loss_components.consistency_loss,
        )
        .to_device(self.device)
        .to_kind(Kind::Float);

        // 计算指标
        let step_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let disentanglement_degree = self.statistics.disentanglement_score();
        let avg_redundancy = (redundancy_topo + redundancy_geo + redundancy_pocket) / 3.0;

        let metrics = TrainingMetrics {
            step,
            phase,
            losses: loss_components,
            mutual_information: crate::disentangle::types::MutualInformationEstimate {
                mi_topo_geo,
                mi_topo_pocket,
                mi_geo_pocket,
            },
            redundancy: crate::disentangle::types::BranchRedundancy {
                topo_redundancy: redundancy_topo,
                geo_redundancy: redundancy_geo,
                pocket_redundancy: redundancy_pocket,
            },
            disentanglement_degree,
            step_time_ms,
            samples_per_second: 1.0 / (step_time_ms / 1000.0),
            peak_memory_bytes: 0, // TODO: 实现显存监控
        };

        (total_loss, loss_components, metrics)
    }

    /// 打印训练进度摘要
    pub fn print_progress(&self, metrics: &TrainingMetrics) {
        if metrics.step % self.config.log_interval == 0 {
            let phase_str = match metrics.phase {
                TrainingPhase::Warmup => "预热",
                TrainingPhase::Disentanglement => "解耦",
                TrainingPhase::Refinement => "精炼",
            };

            println!(
                "[步骤 {}] [阶段: {}] 损失: task={:.4}, disentangle={:.4}, redundancy={:.4}, consistency={:.4}",
                metrics.step,
                phase_str,
                metrics.losses.task_loss,
                metrics.losses.disentangle_loss,
                metrics.losses.redundancy_loss,
                metrics.losses.consistency_loss,
            );
            println!(
                "  MI: topo-geo={:.4}, topo-pocket={:.4}, geo-pocket={:.4} | 解耦度={:.4}",
                metrics.mutual_information.mi_topo_geo,
                metrics.mutual_information.mi_topo_pocket,
                metrics.mutual_information.mi_geo_pocket,
                metrics.disentanglement_degree,
            );
            println!(
                "  冗余度: topo={:.4}, geo={:.4}, pocket={:.4} | 时间={:.2}ms",
                metrics.redundancy.topo_redundancy,
                metrics.redundancy.geo_redundancy,
                metrics.redundancy.pocket_redundancy,
                metrics.step_time_ms,
            );
        }
    }

    /// 获取训练统计信息
    pub fn statistics(&self) -> &TrainingStatistics {
        &self.statistics
    }

    /// 检查训练是否完成
    pub fn is_complete(&self) -> bool {
        let total_needed =
            self.config.warmup_steps + self.config.disentangle_steps + self.config.refinement_steps;
        self.statistics.total_steps >= total_needed
    }
}

/// 基线比较工具
///
/// 比较纯3D方法与解耦方法的性能
pub struct BaselineComparison {
    /// 纯3D方法指标
    pub baseline_3d_metrics: Option<Vec<TrainingMetrics>>,
    /// 解耦方法指标
    pub disentangled_metrics: Option<Vec<TrainingMetrics>>,
}

impl Default for BaselineComparison {
    fn default() -> Self {
        Self::new()
    }
}

impl BaselineComparison {
    /// 创建新的基线比较工具
    pub fn new() -> Self {
        Self {
            baseline_3d_metrics: None,
            disentangled_metrics: None,
        }
    }

    /// 记录3D基线方法的训练历史
    pub fn record_baseline_3d(&mut self, metrics: Vec<TrainingMetrics>) {
        self.baseline_3d_metrics = Some(metrics);
    }

    /// 记录解耦方法的训练历史
    pub fn record_disentangled(&mut self, metrics: Vec<TrainingMetrics>) {
        self.disentangled_metrics = Some(metrics);
    }

    /// 生成比较报告
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== 解耦Transformer vs 纯3D基线 比较报告 ===\n\n");

        if let (Some(baseline), Some(disentangled)) =
            (&self.baseline_3d_metrics, &self.disentangled_metrics)
        {
            // 解耦度比较
            let baseline_final = baseline.last().unwrap();
            let disentangled_final = disentangled.last().unwrap();

            report.push_str(&format!(
                "最终解耦度:\n  基线3D: {:.4}\n  解耦方法: {:.4}\n\n",
                baseline_final.disentanglement_degree, disentangled_final.disentanglement_degree,
            ));

            // 最终任务损失比较
            report.push_str(&format!(
                "最终任务损失:\n  基线3D: {:.4}\n  解耦方法: {:.4}\n\n",
                baseline_final.losses.task_loss, disentangled_final.losses.task_loss,
            ));

            // 训练速度比较
            let baseline_speed =
                baseline.iter().map(|m| m.step_time_ms).sum::<f64>() / baseline.len() as f64;
            let disentangled_speed = disentangled.iter().map(|m| m.step_time_ms).sum::<f64>()
                / disentangled.len() as f64;

            report.push_str(&format!(
                "平均每步训练时间:\n  基线3D: {:.2}ms\n  解耦方法: {:.2}ms\n\n",
                baseline_speed, disentangled_speed,
            ));

            // 互信息比较
            report.push_str(&format!(
                "分支间平均互信息:\n  基线3D: MI(topo;geo)={:.4}, MI(topo;pocket)={:.4}, MI(geo;pocket)={:.4}\n",
                baseline_final.mutual_information.mi_topo_geo,
                baseline_final.mutual_information.mi_topo_pocket,
                baseline_final.mutual_information.mi_geo_pocket,
            ));
            report.push_str(&format!(
                "  解耦方法: MI(topo;geo)={:.4}, MI(topo;pocket)={:.4}, MI(geo;pocket)={:.4}\n\n",
                disentangled_final.mutual_information.mi_topo_geo,
                disentangled_final.mutual_information.mi_topo_pocket,
                disentangled_final.mutual_information.mi_geo_pocket,
            ));

            // 改进百分比
            let mi_improvement = (baseline_final.mutual_information.average()
                - disentangled_final.mutual_information.average())
                / baseline_final.mutual_information.average()
                * 100.0;

            report.push_str(&format!(
                "互信息降低: {:.2}%（互信息越低表示分支解耦越好）\n",
                mi_improvement
            ));
        } else {
            report.push_str("警告: 需要两种方法的指标数据才能生成完整报告\n");
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_statistics() {
        let mut stats = TrainingStatistics::new();
        assert_eq!(stats.current_phase, TrainingPhase::Warmup);

        let losses = LossComponents {
            task_loss: 1.0,
            disentangle_loss: 0.1,
            redundancy_loss: 0.05,
            consistency_loss: 0.02,
        };

        stats.record_step(losses, 0.5, 0.3, 0.2, 0.1, 0.15, 0.12);
        assert_eq!(stats.total_steps, 1);
        assert_eq!(stats.phase_steps, 1);

        // 转换阶段
        stats.transition_to_phase(TrainingPhase::Disentanglement);
        assert_eq!(stats.current_phase, TrainingPhase::Disentanglement);
        assert_eq!(stats.phase_steps, 0);
    }

    #[test]
    fn test_phase_determination() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let config = DisentangleConfig {
            warmup_steps: 100,
            disentangle_steps: 500,
            refinement_steps: 1000,
            ..Default::default()
        };

        let trainer = DisentangleTrainer::new(&vs.root(), config);

        assert_eq!(trainer.get_phase_for_step(50), TrainingPhase::Warmup);
        assert_eq!(
            trainer.get_phase_for_step(150),
            TrainingPhase::Disentanglement
        );
        assert_eq!(trainer.get_phase_for_step(1000), TrainingPhase::Refinement);
    }

    #[test]
    fn test_baseline_comparison() {
        let comparison = BaselineComparison::new();
        let report = comparison.generate_report();
        assert!(report.contains("警告") || report.contains("警告"));
    }
}
