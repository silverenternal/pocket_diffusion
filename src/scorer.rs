//! 亲和力评分器
//! 构建蛋白-配体联合图，使用EGNN预测结合亲和力

use crate::egnn::{EGNNConfig, EGNN};
use crate::types::{tensor_from_slice, Atom, AtomType, Ligand, Pocket, ProteinLigandGraph};
use tch::{nn, Device, Tensor};

/// 亲和力评分器
pub struct AffinityScorer {
    /// EGNN模型
    egnn: EGNN,
    /// 距离阈值
    distance_cutoff: f64,
}

impl AffinityScorer {
    /// 创建评分器
    pub fn new(vs: &nn::Path, config: &EGNNConfig, num_layers: usize) -> Self {
        let egnn = EGNN::new(vs, config, num_layers);

        Self {
            egnn,
            distance_cutoff: 5.0, // 默认5埃截断
        }
    }

    /// 构建蛋白-配体联合图
    pub fn build_graph(&self, pocket: &Pocket, ligand: &Ligand) -> ProteinLigandGraph {
        let device = Device::Cpu;

        // 合并原子坐标
        let num_protein_atoms = pocket.atoms.len();
        let num_ligand_atoms = ligand.atoms.len();
        let total_atoms = num_protein_atoms + num_ligand_atoms;

        // 构建坐标矩阵 [N, 3]
        let mut coords = Vec::with_capacity(total_atoms * 3);
        for atom in &pocket.atoms {
            coords.push(atom.coords[0]);
            coords.push(atom.coords[1]);
            coords.push(atom.coords[2]);
        }
        for atom in &ligand.atoms {
            coords.push(atom.coords[0]);
            coords.push(atom.coords[1]);
            coords.push(atom.coords[2]);
        }
        let coords = tensor_from_slice(&coords).reshape([total_atoms as i64, 3]);

        // 构建节点类型one-hot [N, 6]
        let mut node_types = Vec::with_capacity(total_atoms * 6);
        for atom in &pocket.atoms {
            self.append_one_hot(&mut node_types, &atom.atom_type);
        }
        for atom in &ligand.atoms {
            self.append_one_hot(&mut node_types, &atom.atom_type);
        }
        let node_types = tensor_from_slice(&node_types).reshape([total_atoms as i64, 6]);

        // 构建边索引和距离
        let (edge_index, _distances) =
            self.build_edges(&pocket.atoms, &ligand.atoms, &ligand.bonds);

        ProteinLigandGraph {
            node_coords: coords.to_device(device),
            node_types: node_types.to_device(device),
            edge_index: edge_index.to_device(device),
            edge_features: Tensor::new(), // 将在EGNN中通过RBF计算
            num_protein_atoms,
            num_ligand_atoms,
        }
    }

    /// 构建三类边连接
    fn build_edges(
        &self,
        protein_atoms: &[Atom],
        ligand_atoms: &[Atom],
        ligand_bonds: &[(usize, usize)],
    ) -> (Tensor, Tensor) {
        let num_protein = protein_atoms.len();
        let _num_ligand = ligand_atoms.len();

        let mut edges = Vec::new();
        let mut distances = Vec::new();

        // 1. 配体内部边 (化学键)
        for &(i, j) in ligand_bonds {
            let u = num_protein + i;
            let v = num_protein + j;
            edges.push(u as i64);
            edges.push(v as i64);
            edges.push(v as i64);
            edges.push(u as i64); // 双向

            let dist = self.compute_distance(&ligand_atoms[i], &ligand_atoms[j]);
            distances.push(dist);
            distances.push(dist);
        }

        // 2. 蛋白口袋内部边 (距离阈值)
        for i in 0..num_protein {
            for j in (i + 1)..num_protein {
                let dist = self.compute_distance(&protein_atoms[i], &protein_atoms[j]);
                if dist < self.distance_cutoff && dist > 0.0 {
                    edges.push(i as i64);
                    edges.push(j as i64);
                    edges.push(j as i64);
                    edges.push(i as i64);
                    distances.push(dist);
                    distances.push(dist);
                }
            }
        }

        // 3. 蛋白-配体跨边
        for (i, protein_atom) in protein_atoms.iter().enumerate() {
            for (j, ligand_atom) in ligand_atoms.iter().enumerate() {
                let dist = self.compute_distance(protein_atom, ligand_atom);
                if dist < self.distance_cutoff {
                    let v = num_protein + j;
                    edges.push(i as i64);
                    edges.push(v as i64);
                    edges.push(v as i64);
                    edges.push(i as i64);
                    distances.push(dist);
                    distances.push(dist);
                }
            }
        }

        let num_edges = edges.len() / 2;
        let edge_index = tensor_from_slice::<i64>(&edges).reshape([2, num_edges as i64]);
        let distances = tensor_from_slice(&distances);

        (edge_index, distances)
    }

    /// 计算两个原子间的欧氏距离
    fn compute_distance(&self, a: &Atom, b: &Atom) -> f64 {
        let dx = a.coords[0] - b.coords[0];
        let dy = a.coords[1] - b.coords[1];
        let dz = a.coords[2] - b.coords[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 添加原子类型one-hot编码
    fn append_one_hot(&self, vec: &mut Vec<f32>, atom_type: &AtomType) {
        let idx = match atom_type {
            AtomType::Carbon => 0,
            AtomType::Nitrogen => 1,
            AtomType::Oxygen => 2,
            AtomType::Sulfur => 3,
            AtomType::Hydrogen => 4,
            AtomType::Other => 5,
        };
        for i in 0..6 {
            vec.push(if i == idx { 1.0 } else { 0.0 });
        }
    }

    /// 预测亲和力评分
    pub fn predict(&self, pocket: &Pocket, ligand: &Ligand) -> f64 {
        let graph = self.build_graph(pocket, ligand);

        // 重新计算距离 - 使用张量操作直接计算
        let num_edges = graph.edge_index.size()[1];
        let mut distances = Vec::with_capacity(num_edges as usize);

        // 从张量中提取数据 (使用 double_value 逐个读取)
        let edge_index = &graph.edge_index;
        let coords = &graph.node_coords;

        for e in 0..num_edges as usize {
            let u = edge_index.int64_value(&[0, e as i64]) as usize;
            let v = edge_index.int64_value(&[1, e as i64]) as usize;

            let u0 = coords.double_value(&[u as i64, 0]);
            let u1 = coords.double_value(&[u as i64, 1]);
            let u2 = coords.double_value(&[u as i64, 2]);
            let v0 = coords.double_value(&[v as i64, 0]);
            let v1 = coords.double_value(&[v as i64, 1]);
            let v2 = coords.double_value(&[v as i64, 2]);

            let dx = u0 - v0;
            let dy = u1 - v1;
            let dz = u2 - v2;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            distances.push(dist);
        }

        let distances = tensor_from_slice(&distances);

        // 前向传播
        let output = self.egnn.forward(
            &graph.node_types,
            &graph.node_coords,
            &graph.edge_index,
            &distances,
        );

        output.double_value(&[])
    }

    /// 批量预测并筛选高分分子
    pub fn rank_candidates(
        &self,
        pocket: &Pocket,
        candidates: &[Ligand],
        top_k: usize,
    ) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, ligand)| (i, self.predict(pocket, ligand)))
            .collect();

        // 按亲和力排序 (越低越好 - kcal/mol)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter().take(top_k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pocket::create_example_prrsv_pocket;
    use crate::types::{Atom, AtomType, Ligand};

    #[test]
    fn test_graph_building() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();
        let config = EGNNConfig::default();
        let scorer = AffinityScorer::new(vs, &config, 2);

        let pocket = create_example_prrsv_pocket();
        let ligand = create_example_ligand();

        let graph = scorer.build_graph(&pocket, &ligand);

        assert_eq!(graph.num_protein_atoms, 13);
        assert_eq!(graph.num_ligand_atoms, 5);
        assert_eq!(graph.node_coords.size(), &[18, 3]);
        assert_eq!(graph.node_types.size(), &[18, 6]);
    }

    #[test]
    fn test_affinity_prediction() {
        let device = Device::Cpu;
        let var_store = nn::VarStore::new(device);
        let vs = &var_store.root();
        let config = EGNNConfig {
            in_dim: 6,
            hidden_dim: 32,
            out_dim: 1,
            edge_dim: 16,
            use_attention: true,
        };
        let scorer = AffinityScorer::new(vs, &config, 2);

        let pocket = create_example_prrsv_pocket();
        let ligand = create_example_ligand();

        let affinity = scorer.predict(&pocket, &ligand);

        // 输出应该是一个合理的亲和力值 (-12 ~ 0 kcal/mol)
        println!("Predicted affinity: {} kcal/mol", affinity);
    }

    fn create_example_ligand() -> Ligand {
        let atoms = vec![
            Atom {
                coords: [0.0, 0.0, 0.0],
                atom_type: AtomType::Carbon,
                index: 0,
            },
            Atom {
                coords: [1.5, 0.0, 0.0],
                atom_type: AtomType::Carbon,
                index: 1,
            },
            Atom {
                coords: [2.0, 1.2, 0.0],
                atom_type: AtomType::Nitrogen,
                index: 2,
            },
            Atom {
                coords: [0.7, 2.0, 0.5],
                atom_type: AtomType::Oxygen,
                index: 3,
            },
            Atom {
                coords: [-0.5, 1.2, 0.3],
                atom_type: AtomType::Carbon,
                index: 4,
            },
        ];
        let bonds = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];

        Ligand {
            atoms,
            bonds,
            fingerprint: None,
        }
    }
}
