//! Molecular target-row matching utilities for de novo flow supervision.

use tch::{Kind, Tensor};

use crate::config::FlowTargetAlignmentPolicy;

const NON_FINITE_ASSIGNMENT_COST: f64 = 1.0e18;
const ATOM_TYPE_MISMATCH_ASSIGNMENT_COST: f64 = 4.0;
const TOPOLOGY_DEGREE_ASSIGNMENT_COST: f64 = 0.25;

/// Optional chemistry evidence used to break coordinate matching ties.
#[derive(Debug, Clone, Copy, Default)]
pub struct TargetMatchingFeatures<'a> {
    /// Atom types for generated rows.
    pub generated_atom_types: Option<&'a Tensor>,
    /// Atom types for target rows.
    pub target_atom_types: Option<&'a Tensor>,
    /// Generated adjacency matrix when a scaffold graph is available.
    pub generated_adjacency: Option<&'a Tensor>,
    /// Target adjacency matrix used to compare lightweight degree signatures.
    pub target_adjacency: Option<&'a Tensor>,
}

/// Cost summary for a generated-to-target row matching result.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetMatchingCostSummary {
    /// Number of generated rows assigned to a target row.
    pub matched_count: usize,
    /// Number of generated rows left unassigned and masked out.
    pub unmatched_generated_count: usize,
    /// Number of target rows not selected by the matching policy.
    pub unmatched_target_count: usize,
    /// Total assignment cost over matched rows.
    pub total_cost: f64,
    /// Mean assignment cost over matched rows.
    pub mean_cost: f64,
    /// Maximum single-row assignment cost.
    pub max_cost: f64,
    /// Whether the assignment was solved exactly for the selected policy.
    pub exact_assignment: bool,
}

/// Generated-row to target-row matching with tensor mask and provenance label.
#[derive(Debug)]
pub struct TargetMatchingResult {
    /// Generated row indices covered by this result.
    pub generated_indices: Vec<i64>,
    /// Target row index for each generated row, or `None` when masked out.
    pub target_indices: Vec<Option<i64>>,
    /// Float mask with one entry per generated row.
    pub mask: Tensor,
    /// Assignment cost summary.
    pub cost_summary: TargetMatchingCostSummary,
    /// Stable policy/provenance label for metrics and artifacts.
    pub provenance: String,
}

/// Match generated atom rows to target atom rows according to a configured policy.
pub fn match_molecular_targets(
    generated_coords: &Tensor,
    target_coords: &Tensor,
    policy: FlowTargetAlignmentPolicy,
) -> Option<TargetMatchingResult> {
    match_molecular_targets_with_features(
        generated_coords,
        target_coords,
        policy,
        TargetMatchingFeatures::default(),
    )
}

/// Match generated atom rows to target atom rows using coordinates plus optional chemistry evidence.
pub fn match_molecular_targets_with_features(
    generated_coords: &Tensor,
    target_coords: &Tensor,
    policy: FlowTargetAlignmentPolicy,
    features: TargetMatchingFeatures<'_>,
) -> Option<TargetMatchingResult> {
    let generated_count = generated_coords.size().first().copied().unwrap_or(0).max(0) as usize;
    let target_count = target_coords.size().first().copied().unwrap_or(0).max(0) as usize;
    let device = generated_coords.device();
    let generated_indices = (0..generated_count as i64).collect::<Vec<_>>();

    match policy {
        FlowTargetAlignmentPolicy::IndexExact | FlowTargetAlignmentPolicy::RejectMismatch => {
            if generated_count != target_count {
                return None;
            }
            let target_indices = (0..generated_count as i64).map(Some).collect::<Vec<_>>();
            Some(build_result(
                generated_indices,
                target_indices,
                target_count,
                device,
                policy.as_str(),
                true,
                0.0,
                0.0,
            ))
        }
        FlowTargetAlignmentPolicy::Truncate
        | FlowTargetAlignmentPolicy::PadWithMask
        | FlowTargetAlignmentPolicy::MaskedTruncate
        | FlowTargetAlignmentPolicy::SampledSubgraph => {
            let target_indices = (0..generated_count)
                .map(|index| (index < target_count).then_some(index as i64))
                .collect::<Vec<_>>();
            Some(build_result(
                generated_indices,
                target_indices,
                target_count,
                device,
                policy.as_str(),
                true,
                0.0,
                0.0,
            ))
        }
        FlowTargetAlignmentPolicy::SmokeOnlyModuloRepeat => {
            let target_indices = (0..generated_count)
                .map(|index| {
                    if target_count == 0 {
                        None
                    } else {
                        Some((index % target_count) as i64)
                    }
                })
                .collect::<Vec<_>>();
            Some(build_result(
                generated_indices,
                target_indices,
                target_count,
                device,
                policy.as_str(),
                false,
                0.0,
                0.0,
            ))
        }
        FlowTargetAlignmentPolicy::HungarianDistance
        | FlowTargetAlignmentPolicy::LightweightOptimalTransport => {
            distance_matching(generated_coords, target_coords, policy, features)
        }
    }
}

fn distance_matching(
    generated_coords: &Tensor,
    target_coords: &Tensor,
    policy: FlowTargetAlignmentPolicy,
    features: TargetMatchingFeatures<'_>,
) -> Option<TargetMatchingResult> {
    let generated = coordinate_rows(generated_coords)?;
    let target = coordinate_rows(target_coords)?;
    let generated_count = generated.len();
    let target_count = target.len();
    let generated_indices = (0..generated_count as i64).collect::<Vec<_>>();
    if generated_count == 0 {
        return Some(build_result(
            generated_indices,
            Vec::new(),
            target_count,
            generated_coords.device(),
            policy.as_str(),
            true,
            0.0,
            0.0,
        ));
    }
    if target_count == 0 {
        return Some(build_result(
            generated_indices,
            vec![None; generated_count],
            target_count,
            generated_coords.device(),
            policy.as_str(),
            true,
            0.0,
            0.0,
        ));
    }

    let mut costs = pairwise_squared_distances(&generated, &target);
    add_chemistry_costs(&mut costs, &features, generated_count, target_count);
    let (pairs, exact_assignment) = if generated_count <= target_count {
        assignment_for_rows(&costs)
    } else {
        let transposed = transpose_costs(&costs);
        let (target_to_generated, exact) = assignment_for_rows(&transposed);
        (
            target_to_generated
                .into_iter()
                .map(|(target_index, generated_index)| (generated_index, target_index))
                .collect::<Vec<_>>(),
            exact,
        )
    };
    let mut target_indices = vec![None; generated_count];
    let mut matched_costs = Vec::with_capacity(pairs.len());
    for (generated_index, target_index) in pairs {
        if generated_index < generated_count && target_index < target_count {
            target_indices[generated_index] = Some(target_index as i64);
            matched_costs.push(costs[generated_index][target_index]);
        }
    }
    let total_cost = matched_costs.iter().sum::<f64>();
    let max_cost = matched_costs.iter().copied().fold(0.0_f64, f64::max);
    Some(build_result(
        generated_indices,
        target_indices,
        target_count,
        generated_coords.device(),
        policy.as_str(),
        exact_assignment,
        total_cost,
        max_cost,
    ))
}

fn build_result(
    generated_indices: Vec<i64>,
    target_indices: Vec<Option<i64>>,
    total_target_count: usize,
    device: tch::Device,
    provenance: &str,
    exact_assignment: bool,
    total_cost: f64,
    max_cost: f64,
) -> TargetMatchingResult {
    let matched_count = target_indices
        .iter()
        .filter(|index| index.is_some())
        .count();
    let unmatched_generated_count = generated_indices.len().saturating_sub(matched_count);
    let selected_targets = target_indices
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let unmatched_target_count = total_target_count.saturating_sub(selected_targets.len());
    let mask_values = target_indices
        .iter()
        .map(|index| if index.is_some() { 1.0_f32 } else { 0.0_f32 })
        .collect::<Vec<_>>();
    TargetMatchingResult {
        generated_indices,
        target_indices,
        mask: Tensor::from_slice(&mask_values)
            .to_device(device)
            .to_kind(Kind::Float),
        cost_summary: TargetMatchingCostSummary {
            matched_count,
            unmatched_generated_count,
            unmatched_target_count,
            total_cost,
            mean_cost: if matched_count == 0 {
                0.0
            } else {
                total_cost / matched_count as f64
            },
            max_cost,
            exact_assignment,
        },
        provenance: provenance.to_string(),
    }
}

fn coordinate_rows(tensor: &Tensor) -> Option<Vec<[f64; 3]>> {
    if tensor.size().len() != 2 || tensor.size()[1] != 3 {
        return None;
    }
    let cpu = tensor.to_device(tch::Device::Cpu).to_kind(Kind::Double);
    let rows = cpu.size().first().copied().unwrap_or(0).max(0);
    Some(
        (0..rows)
            .map(|row| {
                [
                    cpu.double_value(&[row, 0]),
                    cpu.double_value(&[row, 1]),
                    cpu.double_value(&[row, 2]),
                ]
            })
            .collect(),
    )
}

fn pairwise_squared_distances(generated: &[[f64; 3]], target: &[[f64; 3]]) -> Vec<Vec<f64>> {
    generated
        .iter()
        .map(|left| {
            target
                .iter()
                .map(|right| {
                    (left[0] - right[0]).powi(2)
                        + (left[1] - right[1]).powi(2)
                        + (left[2] - right[2]).powi(2)
                })
                .collect()
        })
        .collect()
}

fn add_chemistry_costs(
    costs: &mut [Vec<f64>],
    features: &TargetMatchingFeatures<'_>,
    generated_count: usize,
    target_count: usize,
) {
    if let (Some(generated_types), Some(target_types)) = (
        tensor_i64_rows(features.generated_atom_types, generated_count),
        tensor_i64_rows(features.target_atom_types, target_count),
    ) {
        for (generated_index, row) in costs.iter_mut().enumerate() {
            for (target_index, cost) in row.iter_mut().enumerate() {
                if generated_types[generated_index] != target_types[target_index] {
                    *cost += ATOM_TYPE_MISMATCH_ASSIGNMENT_COST;
                }
            }
        }
    }

    if let (Some(generated_degrees), Some(target_degrees)) = (
        adjacency_degrees(features.generated_adjacency, generated_count),
        adjacency_degrees(features.target_adjacency, target_count),
    ) {
        for (generated_index, row) in costs.iter_mut().enumerate() {
            for (target_index, cost) in row.iter_mut().enumerate() {
                *cost += (generated_degrees[generated_index] - target_degrees[target_index]).abs()
                    * TOPOLOGY_DEGREE_ASSIGNMENT_COST;
            }
        }
    }
}

fn tensor_i64_rows(tensor: Option<&Tensor>, expected_rows: usize) -> Option<Vec<i64>> {
    let tensor = tensor?;
    if tensor.size().len() != 1 || tensor.size()[0] != expected_rows as i64 {
        return None;
    }
    let cpu = tensor.to_device(tch::Device::Cpu).to_kind(Kind::Int64);
    Some(
        (0..expected_rows as i64)
            .map(|row| cpu.int64_value(&[row]))
            .collect(),
    )
}

fn adjacency_degrees(tensor: Option<&Tensor>, expected_rows: usize) -> Option<Vec<f64>> {
    let tensor = tensor?;
    if tensor.size().as_slice() != [expected_rows as i64, expected_rows as i64] {
        return None;
    }
    let cpu = tensor.to_device(tch::Device::Cpu).to_kind(Kind::Double);
    Some(
        (0..expected_rows as i64)
            .map(|row| {
                (0..expected_rows as i64)
                    .map(|col| cpu.double_value(&[row, col]).clamp(0.0, 1.0))
                    .sum::<f64>()
            })
            .collect(),
    )
}

fn transpose_costs(costs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = costs.len();
    let cols = costs.first().map(Vec::len).unwrap_or(0);
    (0..cols)
        .map(|col| (0..rows).map(|row| costs[row][col]).collect())
        .collect()
}

fn assignment_for_rows(costs: &[Vec<f64>]) -> (Vec<(usize, usize)>, bool) {
    let rows = costs.len();
    let cols = costs.first().map(Vec::len).unwrap_or(0);
    if rows == 0 || cols == 0 {
        return (Vec::new(), true);
    }
    hungarian_assignment(costs).map_or_else(|| greedy_assignment(costs), |pairs| (pairs, true))
}

fn hungarian_assignment(costs: &[Vec<f64>]) -> Option<Vec<(usize, usize)>> {
    let row_count = costs.len();
    let col_count = costs.first()?.len();
    if row_count == 0 || col_count == 0 || row_count > col_count {
        return None;
    }
    if costs.iter().any(|row| row.len() != col_count) {
        return None;
    }

    let mut row_potential = vec![0.0; row_count + 1];
    let mut col_potential = vec![0.0; col_count + 1];
    let mut matched_row_for_col = vec![0usize; col_count + 1];
    let mut previous_col = vec![0usize; col_count + 1];

    for row in 1..=row_count {
        matched_row_for_col[0] = row;
        let mut current_col = 0usize;
        let mut min_slack = vec![f64::INFINITY; col_count + 1];
        let mut used_col = vec![false; col_count + 1];
        loop {
            used_col[current_col] = true;
            let current_row = matched_row_for_col[current_col];
            let mut delta = f64::INFINITY;
            let mut next_col = 0usize;

            for col in 1..=col_count {
                if used_col[col] {
                    continue;
                }
                let reduced_cost = sanitized_cost(costs[current_row - 1][col - 1])
                    - row_potential[current_row]
                    - col_potential[col];
                if reduced_cost < min_slack[col] {
                    min_slack[col] = reduced_cost;
                    previous_col[col] = current_col;
                }
                if min_slack[col] < delta {
                    delta = min_slack[col];
                    next_col = col;
                }
            }

            if next_col == 0 || !delta.is_finite() {
                return None;
            }

            for col in 0..=col_count {
                if used_col[col] {
                    row_potential[matched_row_for_col[col]] += delta;
                    col_potential[col] -= delta;
                } else {
                    min_slack[col] -= delta;
                }
            }

            current_col = next_col;
            if matched_row_for_col[current_col] == 0 {
                break;
            }
        }

        loop {
            let col = previous_col[current_col];
            matched_row_for_col[current_col] = matched_row_for_col[col];
            current_col = col;
            if current_col == 0 {
                break;
            }
        }
    }

    let mut pairs = Vec::with_capacity(row_count);
    for (col, row) in matched_row_for_col.into_iter().enumerate().skip(1) {
        if row > 0 {
            pairs.push((row - 1, col - 1));
        }
    }
    pairs.sort_by_key(|(row, _)| *row);
    Some(pairs)
}

fn sanitized_cost(cost: f64) -> f64 {
    if cost.is_finite() {
        cost
    } else {
        NON_FINITE_ASSIGNMENT_COST
    }
}

fn greedy_assignment(costs: &[Vec<f64>]) -> (Vec<(usize, usize)>, bool) {
    let cols = costs.first().map(Vec::len).unwrap_or(0);
    let mut used = vec![false; cols];
    let mut pairs = Vec::new();
    for (row, row_costs) in costs.iter().enumerate() {
        let best = row_costs
            .iter()
            .enumerate()
            .filter(|(col, _)| !used[*col])
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(col, _)| col);
        if let Some(col) = best {
            used[col] = true;
            pairs.push((row, col));
        }
    }
    (pairs, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_distance_matching_is_permutation_invariant() {
        let generated = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0])
            .reshape([3, 3]);
        let target = Tensor::from_slice(&[20.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
            .reshape([3, 3]);

        let matching = match_molecular_targets(
            &generated,
            &target,
            FlowTargetAlignmentPolicy::HungarianDistance,
        )
        .expect("matching should succeed");

        assert_eq!(matching.generated_indices, vec![0, 1, 2]);
        assert_eq!(matching.target_indices, vec![Some(1), Some(2), Some(0)]);
        assert_eq!(matching.provenance, "hungarian_distance");
        assert_eq!(matching.cost_summary.matched_count, 3);
        assert!(matching.cost_summary.exact_assignment);
        assert!(matching.cost_summary.total_cost <= 1.0e-12);
        assert_eq!(matching.mask.sum(Kind::Float).double_value(&[]), 3.0);
    }

    #[test]
    fn chemistry_features_break_coordinate_matching_ties() {
        let generated = Tensor::zeros([2, 3], (Kind::Float, tch::Device::Cpu));
        let target = Tensor::zeros([2, 3], (Kind::Float, tch::Device::Cpu));
        let generated_atom_types = Tensor::from_slice(&[6_i64, 7]);
        let target_atom_types = Tensor::from_slice(&[7_i64, 6]);

        let matching = match_molecular_targets_with_features(
            &generated,
            &target,
            FlowTargetAlignmentPolicy::HungarianDistance,
            TargetMatchingFeatures {
                generated_atom_types: Some(&generated_atom_types),
                target_atom_types: Some(&target_atom_types),
                generated_adjacency: None,
                target_adjacency: None,
            },
        )
        .expect("matching should succeed");

        assert_eq!(matching.target_indices, vec![Some(1), Some(0)]);
        assert!(matching.cost_summary.total_cost <= 1.0e-12);
    }

    #[test]
    fn rectangular_assignment_solves_greedy_counterexample_exactly() {
        let costs = vec![vec![1.0, 2.0], vec![1.0, 100.0]];

        let (pairs, exact) = assignment_for_rows(&costs);

        assert!(exact);
        assert_eq!(pairs, vec![(0, 1), (1, 0)]);
        let total_cost = pairs
            .iter()
            .map(|(row, col)| costs[*row][*col])
            .sum::<f64>();
        assert_eq!(total_cost, 3.0);
    }

    #[test]
    fn rectangular_assignment_remains_exact_above_previous_bruteforce_cutoff() {
        let row_count = 12usize;
        let col_count = 14usize;
        let costs = (0..row_count)
            .map(|row| {
                (0..col_count)
                    .map(|col| {
                        let preferred = row + 2;
                        if col == preferred {
                            0.0
                        } else {
                            (100 + row + col) as f64
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let (pairs, exact) = assignment_for_rows(&costs);

        assert!(exact);
        assert_eq!(pairs.len(), row_count);
        for (row, col) in pairs {
            assert_eq!(col, row + 2);
        }
    }

    #[test]
    fn masked_truncate_reports_unmatched_generated_rows() {
        let generated = Tensor::zeros([4, 3], (Kind::Float, tch::Device::Cpu));
        let target = Tensor::zeros([2, 3], (Kind::Float, tch::Device::Cpu));

        let matching = match_molecular_targets(
            &generated,
            &target,
            FlowTargetAlignmentPolicy::MaskedTruncate,
        )
        .expect("masked truncate should succeed");

        assert_eq!(matching.target_indices, vec![Some(0), Some(1), None, None]);
        assert_eq!(matching.mask.sum(Kind::Float).double_value(&[]), 2.0);
        assert_eq!(matching.cost_summary.unmatched_generated_count, 2);
        assert_eq!(matching.provenance, "masked_truncate");
    }

    #[test]
    fn index_exact_rejects_count_mismatch() {
        let generated = Tensor::zeros([3, 3], (Kind::Float, tch::Device::Cpu));
        let target = Tensor::zeros([2, 3], (Kind::Float, tch::Device::Cpu));

        assert!(match_molecular_targets(
            &generated,
            &target,
            FlowTargetAlignmentPolicy::IndexExact
        )
        .is_none());
    }
}
