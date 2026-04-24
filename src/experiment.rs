//! Legacy comparison experiment framework.
//!
//! This module contains the older ndarray-based representation comparison path.
//! It is kept for compatibility and exploratory benchmarks. New config-driven
//! research workflows should prefer `crate::experiments` and `crate::training`.
#![allow(deprecated)]

use crate::dataset::{DatasetDownloader, PDBbindConfig};
use crate::representation::{Molecule2D3D, Molecule3D, MoleculeRepresentation};
use crate::se3_layers::{CoordOnlyEGNN, ForwardBenchmark, SE3EquivariantLayer, TopologyAwareEGNN};
use ndarray::Array2;
use std::path::PathBuf;
use std::time::Instant;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

// ==================== 实验配置 ====================

/// 对比实验配置
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// 数据集路径
    pub dataset_path: Option<PathBuf>,
    /// 口袋截断距离
    pub pocket_cutoff: f32,
    /// EGNN 边截断距离
    pub edge_cutoff: f32,
    /// 输入特征维度
    pub in_dim: usize,
    /// 隐藏层维度
    pub hidden_dim: usize,
    /// 边特征维度
    pub edge_dim: usize,
    /// 前向传播测试次数
    pub num_forward_runs: usize,
    /// 是否使用示例数据集
    pub use_sample_data: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            dataset_path: None,
            pocket_cutoff: 6.0,
            edge_cutoff: 5.0,
            in_dim: 10,
            hidden_dim: 64,
            edge_dim: 32,
            num_forward_runs: 100,
            use_sample_data: true,
        }
    }
}

// ==================== 实验结果 ====================

/// 单方法实验结果
#[derive(Debug, Clone)]
pub struct MethodResult {
    /// 方法名称
    pub method_name: String,
    /// 前向传播性能基准
    pub forward_benchmark: ForwardBenchmark,
    /// 平均每分子内存占用（字节）
    pub avg_memory_per_molecule: f64,
    /// 边构建时间（毫秒）
    pub edge_build_time_ms: f64,
    /// 平均每分子原子数
    pub avg_num_atoms: f64,
    /// 平均每分子边数
    pub avg_num_edges: f64,
    /// 处理的分子数
    pub num_molecules: usize,
}

/// 完整对比实验结果
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// 配置
    pub config: ExperimentConfig,
    /// 方案 A（纯 3D）结果
    pub method_3d: MethodResult,
    /// 方案 B（2D+3D）结果
    pub method_2d3d: MethodResult,
    /// 相对性能提升（%）
    pub relative_speedup_percent: f64,
    /// 相对内存节省（%）
    pub relative_memory_saving_percent: f64,
}

impl ComparisonResult {
    /// 打印对比结果表格
    pub fn print_table(&self) {
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║           分子表示方法对比实验结果                                ║");
        println!("╠═══════════════════════════╤═════════════════╤══════════════════╣");
        println!("║           指标            │  纯3D (基线)   │  2D+3D (我们)   ║");
        println!("╠═══════════════════════════╪═════════════════╪══════════════════╣");
        println!(
            "║  平均前向时间 (ms)       │ {:>15.4} │ {:>16.4} ║",
            self.method_3d.forward_benchmark.avg_forward_time_ms,
            self.method_2d3d.forward_benchmark.avg_forward_time_ms
        );
        println!(
            "║  参数量                  │ {:>15} │ {:>16} ║",
            self.method_3d.forward_benchmark.parameter_count,
            self.method_2d3d.forward_benchmark.parameter_count
        );
        println!(
            "║  每分子内存占用 (bytes)  │ {:>15.1} │ {:>16.1} ║",
            self.method_3d.avg_memory_per_molecule, self.method_2d3d.avg_memory_per_molecule
        );
        println!(
            "║  边构建时间 (ms)         │ {:>15.4} │ {:>16.4} ║",
            self.method_3d.edge_build_time_ms, self.method_2d3d.edge_build_time_ms
        );
        println!(
            "║  平均原子数              │ {:>15.1} │ {:>16.1} ║",
            self.method_3d.avg_num_atoms, self.method_2d3d.avg_num_atoms
        );
        println!(
            "║  平均边数                │ {:>15.1} │ {:>16.1} ║",
            self.method_3d.avg_num_edges, self.method_2d3d.avg_num_edges
        );
        println!("╠═══════════════════════════╪═════════════════╪══════════════════╣");
        println!(
            "║  相对性能提升            │                 │ {:>15.1}% ║",
            self.relative_speedup_percent
        );
        println!(
            "║  相对内存变化            │                 │ {:>15.1}% ║",
            self.relative_memory_saving_percent
        );
        println!("╚═══════════════════════════╧═════════════════╧══════════════════╝");
        println!("\n");
    }
}

// ==================== 实验运行器 ====================

/// 对比实验运行器.
///
/// This is a legacy compatibility surface for representation benchmarks.
pub struct ComparisonExperiment {
    config: ExperimentConfig,
}

impl ComparisonExperiment {
    /// 创建新的实验运行器
    pub fn new(config: ExperimentConfig) -> Self {
        Self { config }
    }

    /// 运行完整对比实验
    pub fn run(&self) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
        println!("🚀 开始分子表示方法对比实验...");

        // 准备数据集
        let dataset_dir = self.prepare_dataset()?;
        println!("✓ 数据集准备完成: {:?}", dataset_dir);

        // 下载数据集条目
        let downloader = DatasetDownloader::new(PDBbindConfig {
            data_dir: dataset_dir.parent().unwrap().to_path_buf(),
            ..Default::default()
        });
        let entries = downloader.scan_entries(&dataset_dir)?;
        println!("✓ 找到 {} 个数据集条目", entries.len());

        // 运行方案 A（纯 3D）
        println!("\n📊 运行方案 A: 纯 3D 表示 (基线)...");
        let result_3d = self.run_method::<Molecule3D>(&entries, "Molecule3D (Baseline)")?;

        // 运行方案 B（2D+3D）
        println!("\n📊 运行方案 B: 2D+3D 混合表示 (我们的方法)...");
        let result_2d3d = self.run_method::<Molecule2D3D>(&entries, "Molecule2D3D (Proposed)")?;

        // 计算相对指标
        let relative_speedup = ((result_3d.forward_benchmark.avg_forward_time_ms
            - result_2d3d.forward_benchmark.avg_forward_time_ms)
            / result_3d.forward_benchmark.avg_forward_time_ms)
            * 100.0;

        let relative_memory = ((result_3d.avg_memory_per_molecule
            - result_2d3d.avg_memory_per_molecule)
            / result_3d.avg_memory_per_molecule)
            * 100.0;

        let result = ComparisonResult {
            config: self.config.clone(),
            method_3d: result_3d,
            method_2d3d: result_2d3d,
            relative_speedup_percent: relative_speedup,
            relative_memory_saving_percent: relative_memory,
        };

        // 打印结果
        result.print_table();

        Ok(result)
    }

    /// 准备数据集
    fn prepare_dataset(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(ref path) = self.config.dataset_path {
            Ok(path.clone())
        } else {
            let config = PDBbindConfig {
                data_dir: PathBuf::from("./data"),
                use_mirror: true,
                ..Default::default()
            };
            let downloader = DatasetDownloader::new(config);
            let dataset_dir = downloader.download_and_extract()?;
            Ok(dataset_dir)
        }
    }

    /// 运行单个方法的实验
    fn run_method<M: MoleculeRepresentation + 'static>(
        &self,
        entries: &[PathBuf],
        method_name: &str,
    ) -> Result<MethodResult, Box<dyn std::error::Error>> {
        let mut total_atoms = 0usize;
        let mut total_edges = 0usize;
        let mut total_memory = 0usize;
        let mut edge_build_time = 0u128;

        // 创建 EGNN 层
        let layer = self.create_layer::<M>();
        let param_count = layer.parameter_count();

        // 遍历数据集（只处理前5个）
        let mut processed = 0usize;
        let mut forward_times = Vec::new();

        for entry_path in entries.iter().take(5) {
            let _pdb_code = entry_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();

            // 查找 PDB 和 SDF 文件
            let pdb_files = glob::glob(&format!("{}/*.pdb", entry_path.display()))
                .map_err(|e| e.to_string())?
                .collect::<Vec<_>>();
            let sdf_files = glob::glob(&format!("{}/*.sdf", entry_path.display()))
                .map_err(|e| e.to_string())?
                .collect::<Vec<_>>();

            if pdb_files.is_empty() || sdf_files.is_empty() {
                continue;
            }

            let _pdb_path = pdb_files[0].as_ref().unwrap();
            let sdf_path = sdf_files[0].as_ref().unwrap();

            // 读取配体获取两种表示
            let (mol_3d, mol_2d3d) = crate::dataset::LigandReader::read_sdf(sdf_path)?;

            // 根据类型选择分子
            // SAFETY: TypeId check ensures M matches the source type,
            // and both Molecule3D and Molecule2D3D implement MoleculeRepresentation
            let ligand: M = if std::any::TypeId::of::<M>() == std::any::TypeId::of::<Molecule3D>() {
                unsafe { std::mem::transmute_copy(&mol_3d) }
            } else {
                unsafe { std::mem::transmute_copy(&mol_2d3d) }
            };

            processed += 1;

            // 统计内存和原子数
            let num_atoms = ligand.atom_count();
            total_atoms += num_atoms;
            total_memory += ligand.memory_usage();

            // 构建边并计时
            let coords = ligand.to_3d_coords();
            let edge_start = Instant::now();
            let edges = CoordOnlyEGNN::build_edges_from_coords(&coords, self.config.edge_cutoff);
            edge_build_time += edge_start.elapsed().as_nanos();
            total_edges += edges.len();

            // 前向传播性能测试
            let node_features = Array2::zeros((num_atoms, self.config.in_dim));
            for _ in 0..self.config.num_forward_runs {
                let forward_start = Instant::now();
                let _ = layer.forward(&node_features, &coords, &edges);
                forward_times.push(forward_start.elapsed().as_nanos());
            }
        }

        // 计算平均值
        let avg_forward_time_ms = if forward_times.is_empty() {
            0.0
        } else {
            forward_times.iter().sum::<u128>() as f64 / forward_times.len() as f64 / 1_000_000.0
        };

        let avg_memory_per_molecule = if processed == 0 {
            0.0
        } else {
            total_memory as f64 / processed as f64
        };

        let avg_edge_build_time_ms = if processed == 0 {
            0.0
        } else {
            edge_build_time as f64 / processed as f64 / 1_000_000.0
        };

        Ok(MethodResult {
            method_name: method_name.to_string(),
            forward_benchmark: ForwardBenchmark {
                name: method_name.to_string(),
                avg_forward_time_ms,
                memory_usage_bytes: 0,
                parameter_count: param_count,
                num_runs: forward_times.len(),
            },
            avg_memory_per_molecule,
            edge_build_time_ms: avg_edge_build_time_ms,
            avg_num_atoms: if processed == 0 {
                0.0
            } else {
                total_atoms as f64 / processed as f64
            },
            avg_num_edges: if processed == 0 {
                0.0
            } else {
                total_edges as f64 / processed as f64
            },
            num_molecules: processed,
        })
    }

    /// 根据表示方法创建 EGNN 层
    fn create_layer<M: MoleculeRepresentation + 'static>(&self) -> Box<dyn SE3EquivariantLayer> {
        let type_id = std::any::TypeId::of::<M>();

        if type_id == std::any::TypeId::of::<Molecule3D>() {
            Box::new(CoordOnlyEGNN::new(
                self.config.in_dim,
                self.config.hidden_dim,
                self.config.edge_dim,
                self.config.edge_cutoff,
            ))
        } else if type_id == std::any::TypeId::of::<Molecule2D3D>() {
            Box::new(TopologyAwareEGNN::new(
                self.config.in_dim,
                self.config.hidden_dim,
                self.config.edge_dim,
                self.config.edge_cutoff,
            ))
        } else {
            panic!("未知的分子表示类型")
        }
    }
}

// ==================== 系统资源监控 ====================

/// 系统资源监控器
pub struct ResourceMonitor {
    sys: System,
}

impl ResourceMonitor {
    /// 创建新的监控器
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::new().with_memory(MemoryRefreshKind::everything()),
        );
        Self { sys }
    }

    /// 获取当前内存使用（字节）
    pub fn current_memory_usage(&mut self) -> u64 {
        self.sys.refresh_memory();
        self.sys.used_memory()
    }

    /// 测量闭包的内存增量
    pub fn measure_memory<F, R>(&mut self, f: F) -> (R, u64)
    where
        F: FnOnce() -> R,
    {
        let before = self.current_memory_usage();
        let result = f();
        let after = self.current_memory_usage();
        (result, after.saturating_sub(before))
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]

    use super::*;

    #[test]
    fn test_experiment_config() {
        let config = ExperimentConfig::default();
        assert_eq!(config.in_dim, 10);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.edge_dim, 32);
    }

    #[test]
    fn test_resource_monitor() {
        let mut monitor = ResourceMonitor::new();
        let memory = monitor.current_memory_usage();
        assert!(memory > 0);
    }
}
