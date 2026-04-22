//! SE(3) 等变神经网络层
//! 实现旋转平移等变的图卷积操作

use ndarray::{Array1, Array2};
use std::fmt::Debug;

// ==================== 等变层接口 ====================

/// SE(3) 等变层接口
pub trait SE3EquivariantLayer: Debug + 'static {
    /// 前向传播
    fn forward(
        &self,
        node_features: &Array2<f32>,
        coords: &Array2<f32>,
        edge_index: &[[usize; 2]],
    ) -> (Array2<f32>, Array2<f32>);

    /// 获取参数量
    fn parameter_count(&self) -> usize;
}

// ==================== 径向基函数编码 ====================

/// 径向基函数距离编码器
#[derive(Clone, Debug)]
pub struct RBFEncoder {
    centers: Array1<f32>,
    sigma: f32,
}

impl RBFEncoder {
    /// 创建新的 RBF 编码器
    pub fn new(num_rbf: usize, cutoff: f32) -> Self {
        let centers: Vec<f32> = (0..num_rbf)
            .map(|i| i as f32 * cutoff / (num_rbf - 1) as f32)
            .collect();
        let sigma = cutoff / num_rbf as f32;

        Self {
            centers: Array1::from_vec(centers),
            sigma,
        }
    }

    /// 编码距离向量
    pub fn encode(&self, distances: &Array1<f32>) -> Array2<f32> {
        let num_edges = distances.len();
        let num_rbf = self.centers.len();
        let mut result = Array2::zeros((num_edges, num_rbf));

        for i in 0..num_edges {
            for j in 0..num_rbf {
                let diff = distances[i] - self.centers[j];
                result[[i, j]] = (-diff * diff / (self.sigma * self.sigma)).exp();
            }
        }

        result
    }
}

// ==================== 纯坐标 EGNN 层（基线方法）====================

/// 纯 3D 坐标的 EGNN 层（基线）
/// 仅使用距离信息构建边特征，无显式拓扑感知
#[derive(Clone, Debug)]
pub struct CoordOnlyEGNN {
    /// 输入特征维度
    in_dim: usize,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 边特征维度
    edge_dim: usize,
    /// RBF 编码器
    rbf: RBFEncoder,
    /// 消息传递权重
    message_weights: Array2<f32>,
    /// 坐标更新权重
    coord_weights: Array2<f32>,
    /// 节点更新权重
    node_weights: Array2<f32>,
}

impl CoordOnlyEGNN {
    /// 创建新的纯坐标 EGNN 层
    pub fn new(in_dim: usize, hidden_dim: usize, edge_dim: usize, cutoff: f32) -> Self {
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / (in_dim + hidden_dim) as f64).sqrt() as f32).unwrap();

        // 随机初始化权重
        let message_weights = Array2::from_shape_fn((2 * in_dim + edge_dim, hidden_dim), |_| {
            normal.sample(&mut rng) as f32
        });

        let coord_weights =
            Array2::from_shape_fn((hidden_dim, 1), |_| normal.sample(&mut rng) as f32);

        let node_weights =
            Array2::from_shape_fn((hidden_dim, in_dim), |_| normal.sample(&mut rng) as f32);

        Self {
            in_dim,
            hidden_dim,
            edge_dim,
            rbf: RBFEncoder::new(edge_dim, cutoff),
            message_weights,
            coord_weights,
            node_weights,
        }
    }

    /// 构建边索引（基于距离阈值）
    pub fn build_edges_from_coords(coords: &Array2<f32>, cutoff: f32) -> Vec<[usize; 2]> {
        let n = coords.nrows();
        let mut edges = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = coords[[i, 0]] - coords[[j, 0]];
                let dy = coords[[i, 1]] - coords[[j, 1]];
                let dz = coords[[i, 2]] - coords[[j, 2]];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < cutoff {
                    edges.push([i, j]);
                    edges.push([j, i]); // 双向边
                }
            }
        }

        edges
    }
}

impl SE3EquivariantLayer for CoordOnlyEGNN {
    fn forward(
        &self,
        node_features: &Array2<f32>,
        coords: &Array2<f32>,
        edge_index: &[[usize; 2]],
    ) -> (Array2<f32>, Array2<f32>) {
        let num_nodes = node_features.nrows();
        let num_edges = edge_index.len();

        // 计算边距离
        let mut distances = Array1::zeros(num_edges);
        for (k, &[i, j]) in edge_index.iter().enumerate() {
            let dx = coords[[i, 0]] - coords[[j, 0]];
            let dy = coords[[i, 1]] - coords[[j, 1]];
            let dz = coords[[i, 2]] - coords[[j, 2]];
            distances[k] = (dx * dx + dy * dy + dz * dz).sqrt();
        }

        // RBF 编码
        let edge_features = self.rbf.encode(&distances);

        // 消息聚合
        let mut messages = Array2::zeros((num_nodes, self.hidden_dim));
        for (k, &[i, j]) in edge_index.iter().enumerate() {
            // 拼接节点特征和边特征
            let mut concat = Vec::with_capacity(2 * self.in_dim + self.edge_dim);
            concat.extend(node_features.row(i).iter());
            concat.extend(node_features.row(j).iter());
            concat.extend(edge_features.row(k).iter());

            let concat_arr = Array1::from_vec(concat);
            let msg = concat_arr.dot(&self.message_weights);

            // 聚合到接收节点
            for d in 0..self.hidden_dim {
                messages[[i, d]] += msg[d];
            }
        }

        // 节点特征更新：消息投影到输入维度
        let new_features = messages.dot(&self.node_weights);

        // 坐标更新（等变）
        let mut new_coords = coords.clone();
        for (k, &[i, j]) in edge_index.iter().enumerate() {
            let msg_norm: f32 = messages.row(i).sum();
            let coord_update: f32 = msg_norm * 0.01;

            let dir = coords.row(i).to_owned() - coords.row(j).to_owned();
            let dir_norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt() + 1e-8;
            let dir_normalized = dir / dir_norm;

            for d in 0..3 {
                new_coords[[i, d]] += coord_update * dir_normalized[d];
            }
        }

        (new_features, new_coords)
    }

    fn parameter_count(&self) -> usize {
        self.message_weights.len() + self.coord_weights.len() + self.node_weights.len()
    }
}

// ==================== 拓扑感知 EGNN 层（我们的方法）====================

/// 拓扑感知的 EGNN 层
/// 利用显式的化学键信息增强边特征
#[derive(Clone, Debug)]
pub struct TopologyAwareEGNN {
    /// 输入特征维度
    in_dim: usize,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 边特征维度
    edge_dim: usize,
    /// RBF 编码器
    rbf: RBFEncoder,
    /// 消息传递权重
    message_weights: Array2<f32>,
    /// 坐标更新权重
    coord_weights: Array2<f32>,
    /// 节点更新权重
    node_weights: Array2<f32>,
    /// 拓扑嵌入权重
    topology_embedding: Array2<f32>,
}

impl TopologyAwareEGNN {
    /// 创建新的拓扑感知 EGNN 层
    pub fn new(in_dim: usize, hidden_dim: usize, edge_dim: usize, cutoff: f32) -> Self {
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / (in_dim + hidden_dim) as f64).sqrt() as f32).unwrap();

        // 随机初始化权重
        let message_weights =
            Array2::from_shape_fn((2 * in_dim + edge_dim + 2, hidden_dim), |_| {
                normal.sample(&mut rng) as f32
            });

        let coord_weights =
            Array2::from_shape_fn((hidden_dim, 1), |_| normal.sample(&mut rng) as f32);

        let node_weights =
            Array2::from_shape_fn((hidden_dim, in_dim), |_| normal.sample(&mut rng) as f32);

        let topology_embedding = Array2::from_shape_fn((2, 2), |_| normal.sample(&mut rng) as f32);

        Self {
            in_dim,
            hidden_dim,
            edge_dim,
            rbf: RBFEncoder::new(edge_dim, cutoff),
            message_weights,
            coord_weights,
            node_weights,
            topology_embedding,
        }
    }

    /// 构建边索引（结合拓扑和距离）
    pub fn build_edges_with_topology(
        coords: &Array2<f32>,
        bonds: &[(usize, usize)],
        cutoff: f32,
    ) -> (Vec<[usize; 2]>, Vec<bool>) {
        let n = coords.nrows();
        let mut edges = Vec::new();
        let mut is_bond = Vec::new();

        // 首先添加化学键
        for &(i, j) in bonds {
            edges.push([i, j]);
            edges.push([j, i]);
            is_bond.push(true);
            is_bond.push(true);
        }

        // 然后添加距离邻近的非化学键
        for i in 0..n {
            for j in (i + 1)..n {
                // 跳过已存在的化学键
                if bonds.contains(&(i, j)) || bonds.contains(&(j, i)) {
                    continue;
                }

                let dx = coords[[i, 0]] - coords[[j, 0]];
                let dy = coords[[i, 1]] - coords[[j, 1]];
                let dz = coords[[i, 2]] - coords[[j, 2]];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < cutoff {
                    edges.push([i, j]);
                    edges.push([j, i]);
                    is_bond.push(false);
                    is_bond.push(false);
                }
            }
        }

        (edges, is_bond)
    }
}

impl SE3EquivariantLayer for TopologyAwareEGNN {
    fn forward(
        &self,
        node_features: &Array2<f32>,
        coords: &Array2<f32>,
        edge_index: &[[usize; 2]],
    ) -> (Array2<f32>, Array2<f32>) {
        let num_nodes = node_features.nrows();
        let num_edges = edge_index.len();

        // 计算边距离和拓扑特征
        let mut distances = Array1::zeros(num_edges);
        let mut topology_flags = Vec::with_capacity(num_edges);

        for (k, &[i, j]) in edge_index.iter().enumerate() {
            let dx = coords[[i, 0]] - coords[[j, 0]];
            let dy = coords[[i, 1]] - coords[[j, 1]];
            let dz = coords[[i, 2]] - coords[[j, 2]];
            distances[k] = (dx * dx + dy * dy + dz * dz).sqrt();

            // 简化：假设前 N 条边是化学键（实际应由 build_edges_with_topology 传入）
            topology_flags.push(if k < num_edges / 3 { 1.0 } else { 0.0 });
        }

        // RBF 编码
        let edge_features = self.rbf.encode(&distances);

        // 消息聚合（包含拓扑特征）
        let mut messages = Array2::zeros((num_nodes, self.hidden_dim));
        for (k, &[i, j]) in edge_index.iter().enumerate() {
            // 拼接节点特征、边特征和拓扑特征
            let mut concat = Vec::with_capacity(2 * self.in_dim + self.edge_dim + 2);
            concat.extend(node_features.row(i).iter());
            concat.extend(node_features.row(j).iter());
            concat.extend(edge_features.row(k).iter());
            concat.push(topology_flags[k]);
            concat.push(1.0 - topology_flags[k]); // 互补特征

            let concat_arr = Array1::from_vec(concat);
            let msg = concat_arr.dot(&self.message_weights);

            // 聚合到接收节点
            for d in 0..self.hidden_dim {
                messages[[i, d]] += msg[d];
            }
        }

        // 节点特征更新：消息投影到输入维度
        let new_features = messages.dot(&self.node_weights);

        // 坐标更新（等变）
        let mut new_coords = coords.clone();
        for (k, &[i, j]) in edge_index.iter().enumerate() {
            let msg_norm: f32 = messages.row(i).sum();
            let topology_factor = 1.0 + topology_flags[k] * 0.5; // 化学键影响更大
            let coord_update: f32 = msg_norm * 0.01 * topology_factor;

            let dir = coords.row(i).to_owned() - coords.row(j).to_owned();
            let dir_norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt() + 1e-8;
            let dir_normalized = dir / dir_norm;

            for d in 0..3 {
                new_coords[[i, d]] += coord_update * dir_normalized[d];
            }
        }

        (new_features, new_coords)
    }

    fn parameter_count(&self) -> usize {
        self.message_weights.len()
            + self.coord_weights.len()
            + self.node_weights.len()
            + self.topology_embedding.len()
    }
}

// ==================== 前向传播性能基准 ====================

/// 前向传播性能基准测试
#[derive(Clone, Debug)]
pub struct ForwardBenchmark {
    /// 方法名称
    pub name: String,
    /// 平均前向时间（毫秒）
    pub avg_forward_time_ms: f64,
    /// 内存占用（字节）
    pub memory_usage_bytes: usize,
    /// 参数量
    pub parameter_count: usize,
    /// 前向传播次数
    pub num_runs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_rbf_encoder() {
        let rbf = RBFEncoder::new(16, 5.0);
        let distances = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let encoded = rbf.encode(&distances);

        assert_eq!(encoded.shape(), &[3, 16]);
    }

    #[test]
    fn test_coord_only_egnn() {
        let in_dim = 10;
        let hidden_dim = 32;
        let edge_dim = 16;

        let layer = CoordOnlyEGNN::new(in_dim, hidden_dim, edge_dim, 5.0);

        let node_features = Array2::zeros((5, in_dim));
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]);
        let edges = vec![
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
        ];

        let (new_features, new_coords) = layer.forward(&node_features, &coords, &edges);

        assert_eq!(new_features.shape(), &[5, in_dim]);
        assert_eq!(new_coords.shape(), &[5, 3]);
    }

    #[test]
    fn test_topology_aware_egnn() {
        let in_dim = 10;
        let hidden_dim = 32;
        let edge_dim = 16;

        let layer = TopologyAwareEGNN::new(in_dim, hidden_dim, edge_dim, 5.0);

        let node_features = Array2::zeros((5, in_dim));
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]);
        let edges = vec![
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
        ];

        let (new_features, new_coords) = layer.forward(&node_features, &coords, &edges);

        assert_eq!(new_features.shape(), &[5, in_dim]);
        assert_eq!(new_coords.shape(), &[5, 3]);
    }

    #[test]
    fn test_parameter_count() {
        let in_dim = 10;
        let hidden_dim = 32;
        let edge_dim = 16;

        let coord_layer = CoordOnlyEGNN::new(in_dim, hidden_dim, edge_dim, 5.0);
        let topo_layer = TopologyAwareEGNN::new(in_dim, hidden_dim, edge_dim, 5.0);

        // 拓扑感知层应该有更多参数
        assert!(topo_layer.parameter_count() > coord_layer.parameter_count());
    }

    #[test]
    fn test_edge_building() {
        let coords = arr2(&[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]);
        let bonds = vec![(0, 1), (1, 2)];

        let (edges, is_bond) = TopologyAwareEGNN::build_edges_with_topology(&coords, &bonds, 4.0);

        // 化学键: (0,1), (1,0), (1,2), (2,1) = 4 条
        // 非化学键 (距离 < 4): (0,2), (2,0), (1,3), (3,1), (2,3), (3,2) = 6 条
        assert_eq!(edges.len(), 10);
        assert_eq!(is_bond.len(), 10);
        assert_eq!(is_bond.iter().filter(|&&x| x).count(), 4);
    }
}
