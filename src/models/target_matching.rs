//! Molecular target-row matching utilities for de novo flow supervision.

use tch::{Kind, Tensor};

use crate::config::FlowTargetAlignmentPolicy;

const MAX_EXACT_ASSIGNMENT_ROWS: usize = 10;

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
            distance_matching(generated_coords, target_coords, policy)
        }
    }
}

fn distance_matching(
    generated_coords: &Tensor,
    target_coords: &Tensor,
    policy: FlowTargetAlignmentPolicy,
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

    let costs = pairwise_squared_distances(&generated, &target);
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
    if rows <= MAX_EXACT_ASSIGNMENT_ROWS && cols <= 16 {
        exact_assignment(costs).map_or_else(|| greedy_assignment(costs), |pairs| (pairs, true))
    } else {
        greedy_assignment(costs)
    }
}

fn exact_assignment(costs: &[Vec<f64>]) -> Option<Vec<(usize, usize)>> {
    let rows = costs.len();
    let cols = costs.first()?.len();
    if rows > cols {
        return None;
    }
    let mut used = vec![false; cols];
    let mut current = Vec::with_capacity(rows);
    let mut best_pairs = Vec::new();
    let mut best_cost = f64::INFINITY;
    search_assignment(
        0,
        costs,
        &mut used,
        &mut current,
        0.0,
        &mut best_cost,
        &mut best_pairs,
    );
    best_cost.is_finite().then_some(best_pairs)
}

fn search_assignment(
    row: usize,
    costs: &[Vec<f64>],
    used: &mut [bool],
    current: &mut Vec<(usize, usize)>,
    current_cost: f64,
    best_cost: &mut f64,
    best_pairs: &mut Vec<(usize, usize)>,
) {
    if row == costs.len() {
        if current_cost < *best_cost {
            *best_cost = current_cost;
            *best_pairs = current.clone();
        }
        return;
    }
    if current_cost >= *best_cost {
        return;
    }
    for col in 0..costs[row].len() {
        if used[col] {
            continue;
        }
        used[col] = true;
        current.push((row, col));
        search_assignment(
            row + 1,
            costs,
            used,
            current,
            current_cost + costs[row][col],
            best_cost,
            best_pairs,
        );
        current.pop();
        used[col] = false;
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
