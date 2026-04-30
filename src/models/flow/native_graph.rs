//! Native molecular graph extraction from full-flow branch logits.

use std::collections::{BTreeMap, BTreeSet};

use tch::Tensor;

use crate::chemistry::native_score::{
    combined_native_score, NATIVE_SCORE_BOND_WEIGHT, NATIVE_SCORE_TOPOLOGY_WEIGHT,
};

/// Native graph extraction algorithm version exposed in rollout diagnostics.
pub const NATIVE_GRAPH_EXTRACTOR_VERSION: &str = "native_graph_extractor_v1";

/// Threshold and valence provenance for native graph extraction.
#[derive(Debug, Clone, Copy)]
pub struct NativeGraphExtractionConfig {
    /// Minimum combined score for non-connectivity edges.
    pub score_threshold: f64,
    /// Maximum density multiplier after connectivity edges have been selected.
    pub max_bond_density_numerator: usize,
    /// Maximum density divisor after connectivity edges have been selected.
    pub max_bond_density_denominator: usize,
}

impl Default for NativeGraphExtractionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.60,
            max_bond_density_numerator: 3,
            max_bond_density_denominator: 2,
        }
    }
}

/// Extracted native graph plus scalar diagnostics.
#[derive(Debug, Clone)]
pub struct NativeGraphExtractionResult {
    /// Thresholded model-native bonds before graph constraints.
    pub raw_bonds: Vec<(usize, usize)>,
    /// Raw native bond type token per raw bond.
    pub raw_bond_types: Vec<i64>,
    /// Undirected native bonds as ordered pairs.
    pub bonds: Vec<(usize, usize)>,
    /// Native bond type token per bond.
    pub bond_types: Vec<i64>,
    /// Scalar extraction diagnostics.
    pub diagnostics: BTreeMap<String, f64>,
}

/// Extract a valence-bounded native graph from molecular flow branch logits.
pub fn predicted_native_graph_from_flow(
    atom_types: &Tensor,
    coords: &Tensor,
    bond_exists_logits: &Tensor,
    bond_type_logits: &Tensor,
    topology_logits: &Tensor,
) -> NativeGraphExtractionResult {
    predicted_native_graph_from_flow_with_config(
        atom_types,
        coords,
        bond_exists_logits,
        bond_type_logits,
        topology_logits,
        NativeGraphExtractionConfig::default(),
    )
}

/// Extract a valence-bounded native graph with explicit thresholds.
pub fn predicted_native_graph_from_flow_with_config(
    atom_types: &Tensor,
    coords: &Tensor,
    bond_exists_logits: &Tensor,
    bond_type_logits: &Tensor,
    topology_logits: &Tensor,
    config: NativeGraphExtractionConfig,
) -> NativeGraphExtractionResult {
    let atom_count = bond_exists_logits
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0) as usize;
    let mut diagnostics = BTreeMap::new();
    diagnostics.insert(
        "molecular_flow_native_graph_extractor_version".to_string(),
        1.0,
    );
    diagnostics.insert(
        "molecular_flow_native_score_threshold".to_string(),
        config.score_threshold,
    );
    diagnostics.insert(
        "molecular_flow_native_score_bond_weight".to_string(),
        NATIVE_SCORE_BOND_WEIGHT,
    );
    diagnostics.insert(
        "molecular_flow_native_score_topology_weight".to_string(),
        NATIVE_SCORE_TOPOLOGY_WEIGHT,
    );
    diagnostics.insert(
        "molecular_flow_native_max_bond_density_numerator".to_string(),
        config.max_bond_density_numerator as f64,
    );
    diagnostics.insert(
        "molecular_flow_native_max_bond_density_denominator".to_string(),
        config.max_bond_density_denominator as f64,
    );
    diagnostics.insert(
        "molecular_flow_native_max_bond_density_ratio".to_string(),
        config.max_bond_density_numerator as f64
            / config.max_bond_density_denominator.max(1) as f64,
    );
    if atom_count <= 1 {
        diagnostics.insert("molecular_flow_native_bond_count".to_string(), 0.0);
        diagnostics.insert(
            "molecular_flow_native_component_count".to_string(),
            atom_count as f64,
        );
        diagnostics.insert("molecular_flow_native_mean_bond_score".to_string(), 0.0);
        diagnostics.insert(
            "molecular_flow_native_valence_violation_fraction".to_string(),
            0.0,
        );
        diagnostics.insert("molecular_flow_raw_bond_logit_pair_count".to_string(), 0.0);
        diagnostics.insert("molecular_flow_raw_native_bond_count".to_string(), 0.0);
        diagnostics.insert("molecular_flow_raw_native_mean_bond_score".to_string(), 0.0);
        diagnostics.insert(
            "molecular_flow_raw_native_density_fraction".to_string(),
            0.0,
        );
        diagnostics.insert(
            "molecular_flow_constrained_native_bond_count".to_string(),
            0.0,
        );
        diagnostics.insert(
            "molecular_flow_raw_to_constrained_removed_bond_count".to_string(),
            0.0,
        );
        diagnostics.insert(
            "molecular_flow_connectivity_guardrail_added_bond_count".to_string(),
            0.0,
        );
        diagnostics.insert(
            "molecular_flow_valence_guardrail_downgrade_count".to_string(),
            0.0,
        );
        diagnostics.insert(
            "molecular_flow_native_graph_guardrail_trigger_count".to_string(),
            0.0,
        );
        diagnostics.insert(
            "molecular_flow_native_chemically_rejected_pair_count".to_string(),
            0.0,
        );
        return NativeGraphExtractionResult {
            raw_bonds: Vec::new(),
            raw_bond_types: Vec::new(),
            bonds: Vec::new(),
            bond_types: Vec::new(),
            diagnostics,
        };
    }

    let bond_probabilities = bond_exists_logits.sigmoid();
    let topology_probabilities = topology_logits.sigmoid();
    let mut candidates = Vec::new();
    let mut chemically_rejected_pair_count = 0usize;
    for left in 0..atom_count {
        for right in (left + 1)..atom_count {
            let left_atom_type = native_atom_type_at(atom_types, left);
            let right_atom_type = native_atom_type_at(atom_types, right);
            if !native_bond_candidate_allowed(left_atom_type, right_atom_type, atom_count) {
                chemically_rejected_pair_count += 1;
                continue;
            }
            let bond_probability = bond_probabilities.double_value(&[left as i64, right as i64]);
            let topology_probability =
                topology_probabilities.double_value(&[left as i64, right as i64]);
            let distance_score =
                native_distance_score(coords, left, right, left_atom_type, right_atom_type);
            let score = combined_native_score(bond_probability, topology_probability);
            let raw_bond_type = bond_type_logits
                .get(left as i64)
                .get(right as i64)
                .argmax(-1, false)
                .int64_value(&[])
                .max(1);
            let raw_order = bond_order_for_native_type(raw_bond_type);
            let mut bond_type = raw_bond_type;
            let mut order = bond_order_for_native_type(bond_type);
            let left_budget = native_valence_budget(left_atom_type);
            let right_budget = native_valence_budget(right_atom_type);
            if order > left_budget.min(right_budget) {
                bond_type = 1;
                order = 1;
            }
            candidates.push(NativeBondCandidate {
                left,
                right,
                score,
                distance_score,
                raw_bond_type,
                raw_order,
                bond_type,
                order,
            });
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.distance_score
                    .partial_cmp(&a.distance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let raw_bonds = candidates
        .iter()
        .filter(|candidate| candidate.score >= config.score_threshold)
        .map(|candidate| ordered_pair(candidate.left, candidate.right))
        .collect::<Vec<_>>();
    let raw_bond_types = candidates
        .iter()
        .filter(|candidate| candidate.score >= config.score_threshold)
        .map(|candidate| candidate.raw_bond_type)
        .collect::<Vec<_>>();
    let raw_native_mean_score = if raw_bonds.is_empty() {
        0.0
    } else {
        candidates
            .iter()
            .filter(|candidate| candidate.score >= config.score_threshold)
            .map(|candidate| candidate.score)
            .sum::<f64>()
            / raw_bonds.len() as f64
    };

    let mut graph = NativeGraphBuilder::new(atom_types);
    let mut dsu = DisjointSet::new(atom_count);
    for candidate in &candidates {
        if graph.component_count(&dsu) <= 1 {
            break;
        }
        if dsu.find_const(candidate.left) == dsu.find_const(candidate.right) {
            continue;
        }
        if graph.try_add(candidate) {
            dsu.union(candidate.left, candidate.right);
        }
    }

    let max_native_bonds = atom_count.saturating_sub(1).max(
        (atom_count * config.max_bond_density_numerator) / config.max_bond_density_denominator,
    );
    for candidate in &candidates {
        if graph.bonds.len() >= max_native_bonds {
            break;
        }
        if candidate.score < config.score_threshold {
            continue;
        }
        let inserted = graph.try_add(candidate);
        if inserted {
            dsu.union(candidate.left, candidate.right);
        }
    }

    let mean_score = if graph.scores.is_empty() {
        0.0
    } else {
        graph.scores.iter().sum::<f64>() / graph.scores.len() as f64
    };
    let constrained_pairs = graph.bonds.iter().copied().collect::<BTreeSet<_>>();
    let raw_pairs = raw_bonds.iter().copied().collect::<BTreeSet<_>>();
    let raw_to_constrained_removed_bond_count = raw_pairs.difference(&constrained_pairs).count();
    let connectivity_guardrail_added_bond_count = graph
        .scores
        .iter()
        .filter(|score| **score < config.score_threshold)
        .count();
    let valence_guardrail_downgrade_count = graph.valence_downgrade_count;
    let raw_bond_logit_pair_count = atom_count * atom_count.saturating_sub(1) / 2;
    let native_graph_guardrail_trigger_count = raw_to_constrained_removed_bond_count
        + connectivity_guardrail_added_bond_count
        + valence_guardrail_downgrade_count;
    diagnostics.insert(
        "molecular_flow_raw_bond_logit_pair_count".to_string(),
        raw_bond_logit_pair_count as f64,
    );
    diagnostics.insert(
        "molecular_flow_raw_native_bond_count".to_string(),
        raw_bonds.len() as f64,
    );
    diagnostics.insert(
        "molecular_flow_raw_native_mean_bond_score".to_string(),
        raw_native_mean_score,
    );
    diagnostics.insert(
        "molecular_flow_raw_native_density_fraction".to_string(),
        if raw_bond_logit_pair_count == 0 {
            0.0
        } else {
            raw_bonds.len() as f64 / raw_bond_logit_pair_count as f64
        },
    );
    diagnostics.insert(
        "molecular_flow_constrained_native_bond_count".to_string(),
        graph.bonds.len() as f64,
    );
    diagnostics.insert(
        "molecular_flow_raw_to_constrained_removed_bond_count".to_string(),
        raw_to_constrained_removed_bond_count as f64,
    );
    diagnostics.insert(
        "molecular_flow_connectivity_guardrail_added_bond_count".to_string(),
        connectivity_guardrail_added_bond_count as f64,
    );
    diagnostics.insert(
        "molecular_flow_valence_guardrail_downgrade_count".to_string(),
        valence_guardrail_downgrade_count as f64,
    );
    diagnostics.insert(
        "molecular_flow_native_graph_guardrail_trigger_count".to_string(),
        native_graph_guardrail_trigger_count as f64,
    );
    diagnostics.insert(
        "molecular_flow_native_chemically_rejected_pair_count".to_string(),
        chemically_rejected_pair_count as f64,
    );
    diagnostics.insert(
        "molecular_flow_native_bond_count".to_string(),
        graph.bonds.len() as f64,
    );
    diagnostics.insert(
        "molecular_flow_native_component_count".to_string(),
        graph.component_count(&dsu) as f64,
    );
    diagnostics.insert(
        "molecular_flow_native_mean_bond_score".to_string(),
        mean_score,
    );
    diagnostics.insert(
        "molecular_flow_native_valence_violation_fraction".to_string(),
        graph.valence_violation_fraction(),
    );
    diagnostics.insert(
        "molecular_flow_native_duplicate_bond_count".to_string(),
        graph.duplicate_bond_count as f64,
    );
    NativeGraphExtractionResult {
        raw_bonds,
        raw_bond_types,
        bonds: graph.bonds,
        bond_types: graph.bond_types,
        diagnostics,
    }
}

#[derive(Debug, Clone)]
struct NativeBondCandidate {
    left: usize,
    right: usize,
    score: f64,
    distance_score: f64,
    raw_bond_type: i64,
    raw_order: i64,
    bond_type: i64,
    order: i64,
}

#[derive(Debug)]
struct NativeGraphBuilder {
    bonds: Vec<(usize, usize)>,
    bond_types: Vec<i64>,
    scores: Vec<f64>,
    used_pairs: BTreeSet<(usize, usize)>,
    valence_used: Vec<i64>,
    valence_budget: Vec<i64>,
    duplicate_bond_count: usize,
    valence_downgrade_count: usize,
}

impl NativeGraphBuilder {
    fn new(atom_types: &Tensor) -> Self {
        let atom_count = atom_types.size().first().copied().unwrap_or(0).max(0) as usize;
        let valence_budget = (0..atom_count)
            .map(|index| native_valence_budget(native_atom_type_at(atom_types, index)))
            .collect::<Vec<_>>();
        Self {
            bonds: Vec::new(),
            bond_types: Vec::new(),
            scores: Vec::new(),
            used_pairs: BTreeSet::new(),
            valence_used: vec![0; atom_count],
            valence_budget,
            duplicate_bond_count: 0,
            valence_downgrade_count: 0,
        }
    }

    fn try_add(&mut self, candidate: &NativeBondCandidate) -> bool {
        let pair = ordered_pair(candidate.left, candidate.right);
        if self.used_pairs.contains(&pair) {
            return false;
        }
        if candidate.left >= self.valence_used.len() || candidate.right >= self.valence_used.len() {
            return false;
        }
        if self.valence_used[candidate.left] + candidate.order > self.valence_budget[candidate.left]
            || self.valence_used[candidate.right] + candidate.order
                > self.valence_budget[candidate.right]
        {
            return false;
        }
        self.used_pairs.insert(pair);
        self.valence_used[candidate.left] += candidate.order;
        self.valence_used[candidate.right] += candidate.order;
        self.bonds.push(pair);
        self.bond_types.push(candidate.bond_type);
        self.scores.push(candidate.score);
        if candidate.raw_bond_type != candidate.bond_type || candidate.raw_order != candidate.order
        {
            self.valence_downgrade_count += 1;
        }
        true
    }

    fn component_count(&self, dsu: &DisjointSet) -> usize {
        dsu.component_count()
    }

    fn valence_violation_fraction(&self) -> f64 {
        if self.valence_used.is_empty() {
            return 0.0;
        }
        let violations = self
            .valence_used
            .iter()
            .zip(self.valence_budget.iter())
            .filter(|(used, budget)| used > budget)
            .count();
        violations as f64 / self.valence_used.len() as f64
    }
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parents: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parents: (0..size).collect(),
        }
    }

    fn find_const(&self, index: usize) -> usize {
        let mut current = index;
        while self.parents[current] != current {
            current = self.parents[current];
        }
        current
    }

    fn find(&mut self, index: usize) -> usize {
        let parent = self.parents[index];
        if parent == index {
            index
        } else {
            let root = self.find(parent);
            self.parents[index] = root;
            root
        }
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root != right_root {
            self.parents[right_root] = left_root;
        }
    }

    fn component_count(&self) -> usize {
        (0..self.parents.len())
            .map(|index| self.find_const(index))
            .collect::<BTreeSet<_>>()
            .len()
    }
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn native_distance_score(
    coords: &Tensor,
    left: usize,
    right: usize,
    left_atom_type: i64,
    right_atom_type: i64,
) -> f64 {
    if coords.size().len() != 2 || coords.size().get(1).copied().unwrap_or(0) != 3 {
        return 0.5;
    }
    let dx = coords.double_value(&[left as i64, 0]) - coords.double_value(&[right as i64, 0]);
    let dy = coords.double_value(&[left as i64, 1]) - coords.double_value(&[right as i64, 1]);
    let dz = coords.double_value(&[left as i64, 2]) - coords.double_value(&[right as i64, 2]);
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    let ideal = native_covalent_radius(left_atom_type) + native_covalent_radius(right_atom_type);
    let width = (ideal * 0.25).clamp(0.18, 0.45);
    (-(distance - ideal).abs() / width).exp().clamp(0.0, 1.0)
}

fn native_bond_candidate_allowed(
    left_atom_type: i64,
    right_atom_type: i64,
    atom_count: usize,
) -> bool {
    if atom_count <= 2 {
        return true;
    }
    !(left_atom_type == 4 && right_atom_type == 4)
}

fn native_valence_budget(atom_type_token: i64) -> i64 {
    match atom_type_token {
        4 => 1,
        3 => 6,
        2 => 2,
        // Nitrogen may be protonated or quaternary in PDBBind ligands.
        1 => 4,
        _ => 4,
    }
}

fn native_atom_type_at(atom_types: &Tensor, index: usize) -> i64 {
    let atom_count = atom_types.size().first().copied().unwrap_or(0).max(0) as usize;
    if atom_count == 0 {
        0
    } else {
        atom_types.int64_value(&[index.min(atom_count - 1) as i64])
    }
}

fn native_covalent_radius(atom_type: i64) -> f64 {
    match atom_type {
        0 => 0.77,
        1 => 0.75,
        2 => 0.73,
        3 => 1.02,
        4 => 0.37,
        _ => 0.77,
    }
}

fn bond_order_for_native_type(bond_type: i64) -> i64 {
    match bond_type {
        3 => 3,
        2 => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;

    fn dense_logits(atom_count: i64, value: f64) -> Tensor {
        Tensor::full(
            [atom_count, atom_count],
            value,
            (Kind::Float, tch::Device::Cpu),
        )
    }

    fn bond_type_logits(atom_count: i64, preferred_type: i64) -> Tensor {
        let logits = Tensor::zeros([atom_count, atom_count, 4], (Kind::Float, tch::Device::Cpu));
        let _ = logits.narrow(2, preferred_type.clamp(0, 3), 1).fill_(5.0);
        logits
    }

    #[test]
    fn native_graph_empty_inputs_return_empty_graph() {
        let atom_types = Tensor::zeros([0], (Kind::Int64, tch::Device::Cpu));
        let coords = Tensor::zeros([0, 3], (Kind::Float, tch::Device::Cpu));
        let logits = Tensor::zeros([0, 0], (Kind::Float, tch::Device::Cpu));
        let bond_types = Tensor::zeros([0, 0, 4], (Kind::Float, tch::Device::Cpu));

        let result =
            predicted_native_graph_from_flow(&atom_types, &coords, &logits, &bond_types, &logits);

        assert!(result.bonds.is_empty());
        assert!(result.raw_bonds.is_empty());
        assert_eq!(
            result.diagnostics["molecular_flow_native_component_count"],
            0.0
        );
    }

    #[test]
    fn native_graph_connects_high_confidence_components_without_duplicates() {
        let atom_types = Tensor::from_slice(&[0_i64, 0, 1, 2]);
        let coords = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, 1.4, 0.0, 0.0, 2.8, 0.0, 0.0, 4.2, 0.0, 0.0,
        ])
        .reshape([4, 3]);
        let logits = dense_logits(4, 4.0);
        let bond_types = bond_type_logits(4, 1);

        let result =
            predicted_native_graph_from_flow(&atom_types, &coords, &logits, &bond_types, &logits);
        let unique = result.bonds.iter().copied().collect::<BTreeSet<_>>();

        assert!(!result.raw_bonds.is_empty());
        assert_eq!(unique.len(), result.bonds.len());
        assert!(result.diagnostics["molecular_flow_native_component_count"] <= 1.0);
        assert_eq!(
            result.diagnostics["molecular_flow_native_duplicate_bond_count"],
            0.0
        );
    }

    #[test]
    fn native_graph_respects_hydrogen_valence_under_dense_logits() {
        let atom_types = Tensor::from_slice(&[4_i64, 4, 4, 4]);
        let coords = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ])
        .reshape([4, 3]);
        let logits = dense_logits(4, 8.0);
        let bond_types = bond_type_logits(4, 3);

        let result =
            predicted_native_graph_from_flow(&atom_types, &coords, &logits, &bond_types, &logits);
        let mut degree = vec![0_usize; 4];
        for (left, right) in &result.bonds {
            degree[*left] += 1;
            degree[*right] += 1;
        }

        assert!(result.raw_bonds.is_empty());
        assert!(result.bonds.is_empty());
        assert!(degree.into_iter().all(|value| value <= 1));
        assert_eq!(
            result.diagnostics["molecular_flow_native_valence_violation_fraction"],
            0.0
        );
        assert!(result.diagnostics["molecular_flow_native_chemically_rejected_pair_count"] > 0.0);
    }

    #[test]
    fn native_graph_threshold_uses_configured_learned_score_without_distance_boost() {
        let atom_types = Tensor::from_slice(&[0_i64, 0, 0]);
        let coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.54, 0.0, 0.0, 3.08, 0.0, 0.0])
            .reshape([3, 3]);
        let logits = dense_logits(3, 0.1);
        let bond_types = bond_type_logits(3, 1);

        let result = predicted_native_graph_from_flow_with_config(
            &atom_types,
            &coords,
            &logits,
            &bond_types,
            &logits,
            NativeGraphExtractionConfig {
                score_threshold: 0.55,
                ..NativeGraphExtractionConfig::default()
            },
        );

        assert!(result.raw_bonds.is_empty());
        assert!(result.diagnostics["molecular_flow_connectivity_guardrail_added_bond_count"] > 0.0);
        assert_eq!(
            result.diagnostics["molecular_flow_raw_native_density_fraction"],
            0.0
        );
    }

    #[test]
    fn native_graph_honors_configured_score_threshold() {
        let atom_types = Tensor::from_slice(&[0_i64, 0, 0]);
        let coords = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.54, 0.0, 0.0, 3.08, 0.0, 0.0])
            .reshape([3, 3]);
        let logits = dense_logits(3, 1.0);
        let bond_types = bond_type_logits(3, 1);

        let default_threshold =
            predicted_native_graph_from_flow(&atom_types, &coords, &logits, &bond_types, &logits);
        let raised_threshold = predicted_native_graph_from_flow_with_config(
            &atom_types,
            &coords,
            &logits,
            &bond_types,
            &logits,
            NativeGraphExtractionConfig {
                score_threshold: 0.75,
                ..NativeGraphExtractionConfig::default()
            },
        );

        assert!(!default_threshold.raw_bonds.is_empty());
        assert!(raised_threshold.raw_bonds.is_empty());
        assert_eq!(
            raised_threshold.diagnostics["molecular_flow_native_score_threshold"],
            0.75
        );
    }

    #[test]
    fn native_graph_rejects_hydrogen_hydrogen_islands() {
        let atom_types = Tensor::from_slice(&[0_i64, 4, 4, 0]);
        let coords = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, 1.1, 0.0, 0.0, 1.2, 0.0, 0.0, 2.3, 0.0, 0.0,
        ])
        .reshape([4, 3]);
        let logits = dense_logits(4, 8.0);
        let bond_types = bond_type_logits(4, 1);

        let result =
            predicted_native_graph_from_flow(&atom_types, &coords, &logits, &bond_types, &logits);

        assert!(!result.bonds.contains(&(1, 2)));
        assert!(result.diagnostics["molecular_flow_native_component_count"] <= 1.0);
        assert_eq!(
            result.diagnostics["molecular_flow_native_chemically_rejected_pair_count"],
            1.0
        );
    }

    #[test]
    fn native_graph_connects_protonated_nitrogen_without_valence_violation() {
        let atom_types = Tensor::from_slice(&[1_i64, 0, 4, 4, 4]);
        let coords = Tensor::from_slice(&[
            0.0_f32, 0.0, 0.0, 1.4, 0.0, 0.0, -0.8, 0.8, 0.0, -0.8, -0.8, 0.0, 0.0, 0.0, 1.0,
        ])
        .reshape([5, 3]);
        let logits = dense_logits(5, 8.0);
        let bond_types = bond_type_logits(5, 1);

        let result =
            predicted_native_graph_from_flow(&atom_types, &coords, &logits, &bond_types, &logits);
        let nitrogen_degree = result
            .bonds
            .iter()
            .filter(|(left, right)| *left == 0 || *right == 0)
            .count();

        assert!(result.diagnostics["molecular_flow_native_component_count"] <= 1.0);
        assert_eq!(nitrogen_degree, 4);
        assert_eq!(
            result.diagnostics["molecular_flow_native_valence_violation_fraction"],
            0.0
        );
    }
}
