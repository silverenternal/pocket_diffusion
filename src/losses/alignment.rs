//! Shared target-alignment helpers for shape-safe training objectives.

use tch::{Device, Kind, Tensor};

use crate::config::FlowTargetAlignmentPolicy;

/// Loss-side alignment policy for generated-vs-target tensor rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LossTargetAlignmentPolicy {
    /// Require exactly matching source and requested shape.
    ExactMatch,
    /// Reject mismatched row counts.
    RejectMismatch,
    /// Copy the leading target rows and mask missing requested rows.
    Truncate,
    /// Pad missing requested rows with zeros and mask them out.
    PadWithMask,
    /// Placeholder for future permutation-invariant matching.
    Matched,
    /// Legacy smoke/debug repetition policy.
    SmokeOnlyModuloRepeat,
}

impl LossTargetAlignmentPolicy {
    /// Stable label for diagnostics and artifacts.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ExactMatch => "exact_match",
            Self::RejectMismatch => "reject_mismatch",
            Self::Truncate => "truncate",
            Self::PadWithMask => "pad_with_mask",
            Self::Matched => "matched",
            Self::SmokeOnlyModuloRepeat => "smoke_only_modulo_repeat",
        }
    }
}

impl From<FlowTargetAlignmentPolicy> for LossTargetAlignmentPolicy {
    fn from(policy: FlowTargetAlignmentPolicy) -> Self {
        match policy {
            FlowTargetAlignmentPolicy::IndexExact => Self::ExactMatch,
            FlowTargetAlignmentPolicy::Truncate => Self::Truncate,
            FlowTargetAlignmentPolicy::MaskedTruncate => Self::PadWithMask,
            FlowTargetAlignmentPolicy::PadWithMask => Self::PadWithMask,
            FlowTargetAlignmentPolicy::HungarianDistance
            | FlowTargetAlignmentPolicy::LightweightOptimalTransport => Self::Matched,
            FlowTargetAlignmentPolicy::SampledSubgraph => Self::Truncate,
            FlowTargetAlignmentPolicy::RejectMismatch => Self::RejectMismatch,
            FlowTargetAlignmentPolicy::SmokeOnlyModuloRepeat => Self::SmokeOnlyModuloRepeat,
        }
    }
}

/// Provenance for an aligned tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AlignmentProvenance {
    /// Human-readable target label.
    pub label: String,
    /// Policy used for this alignment.
    pub policy: &'static str,
    /// Source rows observed in the target tensor.
    pub source_rows: i64,
    /// Requested rows required by the prediction tensor.
    pub requested_rows: i64,
    /// Compact status label.
    pub status: &'static str,
    /// Optional skip/reject reason.
    pub reason: Option<String>,
}

impl AlignmentProvenance {
    fn ok(
        label: impl Into<String>,
        policy: LossTargetAlignmentPolicy,
        source_rows: i64,
        requested_rows: i64,
    ) -> Self {
        Self {
            label: label.into(),
            policy: policy.as_str(),
            source_rows,
            requested_rows,
            status: if source_rows == requested_rows {
                "exact"
            } else {
                "aligned_with_mask"
            },
            reason: None,
        }
    }

    fn rejected(
        label: impl Into<String>,
        policy: LossTargetAlignmentPolicy,
        source_rows: i64,
        requested_rows: i64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            policy: policy.as_str(),
            source_rows,
            requested_rows,
            status: "rejected",
            reason: Some(reason.into()),
        }
    }
}

/// Aligned tensor values plus a row or pair mask.
pub(crate) struct AlignedTensor {
    /// Tensor values resized to the requested shape.
    pub values: Tensor,
    /// Row mask for row alignment or pair mask for square-matrix alignment.
    pub mask: Tensor,
    /// Alignment provenance.
    #[allow(dead_code)] // Returned for diagnostics; not every caller serializes it yet.
    pub provenance: AlignmentProvenance,
}

/// Align a one-dimensional target vector to the requested row count.
pub(crate) fn align_vector(
    target: &Tensor,
    requested_rows: i64,
    kind: Kind,
    device: Device,
    policy: LossTargetAlignmentPolicy,
    label: impl Into<String>,
) -> Option<AlignedTensor> {
    align_rows(target, requested_rows, &[], kind, device, policy, label)
}

/// Align a row-major target tensor to the requested row count and trailing shape.
pub(crate) fn align_rows(
    target: &Tensor,
    requested_rows: i64,
    trailing_shape: &[i64],
    kind: Kind,
    device: Device,
    policy: LossTargetAlignmentPolicy,
    label: impl Into<String>,
) -> Option<AlignedTensor> {
    let label = label.into();
    let requested_rows = requested_rows.max(0);
    let source_rows = target.size().first().copied().unwrap_or(0).max(0);
    let requested_shape = requested_tensor_shape(requested_rows, trailing_shape);
    let source_shape_matches = target.size().as_slice() == requested_shape.as_slice();
    if should_reject(policy, source_rows, requested_rows, source_shape_matches) {
        return rejected_alignment(label, policy, source_rows, requested_rows);
    }
    if policy == LossTargetAlignmentPolicy::Matched && source_rows != requested_rows {
        return rejected_alignment(label, policy, source_rows, requested_rows);
    }
    if requested_rows == 0 {
        return Some(AlignedTensor {
            values: Tensor::zeros(requested_shape.as_slice(), (kind, device)),
            mask: Tensor::zeros([0], (Kind::Float, device)),
            provenance: AlignmentProvenance::ok(label, policy, source_rows, requested_rows),
        });
    }
    if source_shape_matches {
        return Some(AlignedTensor {
            values: target.to_device(device).to_kind(kind),
            mask: Tensor::ones([requested_rows], (Kind::Float, device)),
            provenance: AlignmentProvenance::ok(label, policy, source_rows, requested_rows),
        });
    }
    if target.dim() != requested_shape.len()
        || target.size().get(1..).unwrap_or(&[]) != trailing_shape
    {
        return rejected_alignment(label, policy, source_rows, requested_rows);
    }

    let copy_rows = row_indices(requested_rows, source_rows, policy);
    let values = materialize_rows(target, &copy_rows, trailing_shape, kind, device);
    let mask = row_mask_from_indices(&copy_rows, device);
    Some(AlignedTensor {
        values,
        mask,
        provenance: AlignmentProvenance::ok(label, policy, source_rows, requested_rows),
    })
}

/// Align a square target matrix and return a square pair mask.
pub(crate) fn align_square_matrix(
    target: &Tensor,
    requested_rows: i64,
    kind: Kind,
    device: Device,
    policy: LossTargetAlignmentPolicy,
    label: impl Into<String>,
) -> Option<AlignedTensor> {
    let label = label.into();
    let requested_rows = requested_rows.max(0);
    let source_rows = target.size().first().copied().unwrap_or(0).max(0);
    let source_cols = target.size().get(1).copied().unwrap_or(0).max(0);
    let exact_shape = target.size().as_slice() == [requested_rows, requested_rows];
    if policy == LossTargetAlignmentPolicy::ExactMatch && !exact_shape {
        return rejected_alignment(label, policy, source_rows, requested_rows);
    }
    if policy == LossTargetAlignmentPolicy::RejectMismatch
        && (source_rows != requested_rows || source_cols != requested_rows)
    {
        return rejected_alignment(label, policy, source_rows, requested_rows);
    }
    if policy == LossTargetAlignmentPolicy::Matched
        && (source_rows != requested_rows || source_cols != requested_rows)
    {
        return rejected_alignment(label, policy, source_rows, requested_rows);
    }
    if requested_rows == 0 {
        return Some(AlignedTensor {
            values: Tensor::zeros([0, 0], (kind, device)),
            mask: Tensor::zeros([0, 0], (Kind::Float, device)),
            provenance: AlignmentProvenance::ok(label, policy, source_rows, requested_rows),
        });
    }
    if exact_shape {
        let row_mask = Tensor::ones([requested_rows], (Kind::Float, device));
        return Some(AlignedTensor {
            values: target.to_device(device).to_kind(kind),
            mask: pair_mask_from_row_mask(&row_mask),
            provenance: AlignmentProvenance::ok(label, policy, source_rows, requested_rows),
        });
    }
    if target.dim() != 2 {
        return rejected_alignment(label, policy, source_rows, requested_rows);
    }

    let row_index_map = row_indices(requested_rows, source_rows, policy);
    let col_index_map = row_indices(requested_rows, source_cols, policy);
    let mut values = Vec::with_capacity((requested_rows * requested_rows).max(0) as usize);
    for row in &row_index_map {
        for col in &col_index_map {
            match (row, col) {
                (Some(row), Some(col)) => values.push(target.double_value(&[*row, *col]) as f32),
                _ => values.push(0.0),
            }
        }
    }
    let row_mask = row_mask_from_indices(&row_index_map, device);
    Some(AlignedTensor {
        values: Tensor::from_slice(&values)
            .reshape([requested_rows, requested_rows])
            .to_device(device)
            .to_kind(kind),
        mask: pair_mask_from_row_mask(&row_mask),
        provenance: AlignmentProvenance::ok(label, policy, source_rows, requested_rows),
    })
}

/// Build a pair mask from an atom/row mask and remove diagonal self-pairs.
pub(crate) fn pair_mask_from_row_mask(row_mask: &Tensor) -> Tensor {
    let rows = row_mask.size().first().copied().unwrap_or(0).max(0);
    if rows == 0 {
        return Tensor::zeros([0, 0], (Kind::Float, row_mask.device()));
    }
    let row_mask = row_mask.to_kind(Kind::Float);
    let pair_mask = row_mask.unsqueeze(1) * row_mask.unsqueeze(0);
    let eye = Tensor::eye(rows, (Kind::Float, row_mask.device()));
    pair_mask * (Tensor::ones_like(&eye) - eye)
}

fn should_reject(
    policy: LossTargetAlignmentPolicy,
    source_rows: i64,
    requested_rows: i64,
    exact_shape: bool,
) -> bool {
    match policy {
        LossTargetAlignmentPolicy::ExactMatch => !exact_shape,
        LossTargetAlignmentPolicy::RejectMismatch => source_rows != requested_rows,
        LossTargetAlignmentPolicy::Matched => source_rows != requested_rows,
        LossTargetAlignmentPolicy::Truncate
        | LossTargetAlignmentPolicy::PadWithMask
        | LossTargetAlignmentPolicy::SmokeOnlyModuloRepeat => false,
    }
}

fn rejected_alignment(
    label: String,
    policy: LossTargetAlignmentPolicy,
    source_rows: i64,
    requested_rows: i64,
) -> Option<AlignedTensor> {
    let reason = if policy == LossTargetAlignmentPolicy::Matched {
        "matched alignment policy is reserved for future permutation-invariant matching"
    } else {
        "target shape is incompatible with requested prediction shape"
    };
    let provenance =
        AlignmentProvenance::rejected(label, policy, source_rows, requested_rows, reason);
    log::debug!(
        "skipping aligned target {} with policy {}: {}",
        provenance.label,
        provenance.policy,
        provenance.reason.as_deref().unwrap_or("unknown")
    );
    None
}

fn requested_tensor_shape(rows: i64, trailing_shape: &[i64]) -> Vec<i64> {
    let mut shape = Vec::with_capacity(trailing_shape.len() + 1);
    shape.push(rows);
    shape.extend_from_slice(trailing_shape);
    shape
}

fn row_indices(
    requested_rows: i64,
    source_rows: i64,
    policy: LossTargetAlignmentPolicy,
) -> Vec<Option<i64>> {
    (0..requested_rows)
        .map(|row| aligned_source_index(row, source_rows, policy))
        .collect()
}

fn aligned_source_index(
    requested_index: i64,
    source_count: i64,
    policy: LossTargetAlignmentPolicy,
) -> Option<i64> {
    if source_count <= 0 {
        return None;
    }
    match policy {
        LossTargetAlignmentPolicy::SmokeOnlyModuloRepeat => Some(requested_index % source_count),
        LossTargetAlignmentPolicy::ExactMatch
        | LossTargetAlignmentPolicy::RejectMismatch
        | LossTargetAlignmentPolicy::Truncate
        | LossTargetAlignmentPolicy::PadWithMask
        | LossTargetAlignmentPolicy::Matched => {
            (requested_index < source_count).then_some(requested_index)
        }
    }
}

fn materialize_rows(
    target: &Tensor,
    indices: &[Option<i64>],
    trailing_shape: &[i64],
    kind: Kind,
    device: Device,
) -> Tensor {
    let trailing_count = trailing_shape.iter().product::<i64>().max(1);
    let mut values = Vec::with_capacity(indices.len() * trailing_count as usize);
    for index in indices {
        match index {
            Some(row) => {
                let row_tensor = target.get(*row).reshape([trailing_count]);
                for offset in 0..trailing_count {
                    values.push(row_tensor.double_value(&[offset]) as f32);
                }
            }
            None => values.extend(std::iter::repeat(0.0).take(trailing_count as usize)),
        }
    }
    let mut shape = Vec::with_capacity(trailing_shape.len() + 1);
    shape.push(indices.len() as i64);
    shape.extend_from_slice(trailing_shape);
    Tensor::from_slice(&values)
        .reshape(shape.as_slice())
        .to_device(device)
        .to_kind(kind)
}

fn row_mask_from_indices(indices: &[Option<i64>], device: Device) -> Tensor {
    let values = indices
        .iter()
        .map(|index| if index.is_some() { 1.0_f32 } else { 0.0 })
        .collect::<Vec<_>>();
    Tensor::from_slice(&values).to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_values(mask: &Tensor) -> Vec<f64> {
        (0..mask.size()[0])
            .map(|index| mask.double_value(&[index]))
            .collect()
    }

    #[test]
    fn align_rows_handles_empty_targets() {
        let target = Tensor::zeros([0, 2], (Kind::Float, Device::Cpu));
        let aligned = align_rows(
            &target,
            3,
            &[2],
            Kind::Float,
            Device::Cpu,
            LossTargetAlignmentPolicy::PadWithMask,
            "empty",
        )
        .unwrap();

        assert_eq!(aligned.values.size(), vec![3, 2]);
        assert_eq!(mask_values(&aligned.mask), vec![0.0, 0.0, 0.0]);
        assert_eq!(aligned.provenance.status, "aligned_with_mask");
    }

    #[test]
    fn align_rows_preserves_exact_matches() {
        let target = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([2, 2]);
        let aligned = align_rows(
            &target,
            2,
            &[2],
            Kind::Float,
            Device::Cpu,
            LossTargetAlignmentPolicy::ExactMatch,
            "exact",
        )
        .unwrap();

        assert_eq!(aligned.values.size(), vec![2, 2]);
        assert_eq!(mask_values(&aligned.mask), vec![1.0, 1.0]);
        assert_eq!(aligned.provenance.status, "exact");
    }

    #[test]
    fn align_rows_masks_shorter_targets() {
        let target = Tensor::from_slice(&[1_i64, 2]);
        let aligned = align_vector(
            &target,
            4,
            Kind::Int64,
            Device::Cpu,
            LossTargetAlignmentPolicy::PadWithMask,
            "short",
        )
        .unwrap();

        assert_eq!(aligned.values.size(), vec![4]);
        assert_eq!(aligned.values.int64_value(&[0]), 1);
        assert_eq!(aligned.values.int64_value(&[1]), 2);
        assert_eq!(aligned.values.int64_value(&[2]), 0);
        assert_eq!(mask_values(&aligned.mask), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn align_square_matrix_truncates_longer_targets_and_builds_pair_mask() {
        let target = Tensor::from_slice(&[
            0.0_f32, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0,
        ])
        .reshape([3, 3]);
        let aligned = align_square_matrix(
            &target,
            2,
            Kind::Float,
            Device::Cpu,
            LossTargetAlignmentPolicy::Truncate,
            "long",
        )
        .unwrap();

        assert_eq!(aligned.values.size(), vec![2, 2]);
        assert_eq!(aligned.values.double_value(&[1, 1]), 4.0);
        assert_eq!(aligned.mask.double_value(&[0, 0]), 0.0);
        assert_eq!(aligned.mask.double_value(&[0, 1]), 1.0);
        assert_eq!(aligned.mask.double_value(&[1, 0]), 1.0);
    }

    #[test]
    fn reject_mismatch_returns_none() {
        let target = Tensor::from_slice(&[1.0_f32, 2.0]).reshape([1, 2]);
        let aligned = align_rows(
            &target,
            2,
            &[2],
            Kind::Float,
            Device::Cpu,
            LossTargetAlignmentPolicy::RejectMismatch,
            "reject",
        );

        assert!(aligned.is_none());
    }

    #[test]
    fn matched_policy_is_explicit_future_none_on_mismatch() {
        let target = Tensor::from_slice(&[1.0_f32, 2.0]).reshape([1, 2]);
        let aligned = align_rows(
            &target,
            2,
            &[2],
            Kind::Float,
            Device::Cpu,
            LossTargetAlignmentPolicy::Matched,
            "matched",
        );

        assert!(aligned.is_none());
    }
}
