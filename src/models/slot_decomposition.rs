//! Learned slot decomposition for each modality representation.

use tch::{nn, Kind, Tensor};

use crate::config::SlotDecompositionConfig;

use super::{
    BatchedModalityEncoding, BatchedSlotEncoding, ModalityEncoding, SlotDecomposer, SlotEncoding,
};

/// Soft slot decomposition with sparse activations and shared slot upper bound.
#[derive(Debug)]
pub struct SoftSlotDecomposer {
    slot_logits: nn::Linear,
    slot_activation_logits: nn::Linear,
    slot_projector: nn::Linear,
    reconstruction_projector: nn::Linear,
    num_slots: i64,
    hidden_dim: i64,
    activation_temperature: f64,
    activation_threshold: f64,
    attention_masking: bool,
    minimum_visible_slots: i64,
    activation_mass_evidence_weight: f64,
}

impl SoftSlotDecomposer {
    /// Create a slot decomposer with a fixed slot budget.
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_slots: i64) -> Self {
        Self::new_with_config(
            vs,
            hidden_dim,
            num_slots,
            &SlotDecompositionConfig::default(),
        )
    }

    /// Create a slot decomposer with explicit activation controls.
    pub fn new_with_config(
        vs: &nn::Path,
        hidden_dim: i64,
        num_slots: i64,
        config: &SlotDecompositionConfig,
    ) -> Self {
        let slot_logits = nn::linear(
            vs / "slot_logits",
            hidden_dim,
            num_slots,
            Default::default(),
        );
        let slot_activation_logits = nn::linear(
            vs / "slot_activation_logits",
            hidden_dim,
            num_slots,
            Default::default(),
        );
        let slot_projector = nn::linear(
            vs / "slot_projector",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let reconstruction_projector = nn::linear(
            vs / "reconstruction_projector",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        Self {
            slot_logits,
            slot_activation_logits,
            slot_projector,
            reconstruction_projector,
            num_slots,
            hidden_dim,
            activation_temperature: config.activation_temperature,
            activation_threshold: config.activation_threshold,
            attention_masking: config.attention_masking,
            minimum_visible_slots: config.minimum_visible_slots,
            activation_mass_evidence_weight: config.activation_mass_evidence_weight,
        }
    }

    /// Decompose padded modality tokens into fixed-count slots in one tensor pass.
    pub(crate) fn decompose_batch(&self, input: &BatchedModalityEncoding) -> BatchedSlotEncoding {
        let tokens = &input.token_embeddings;
        let mask = input.token_mask.to_kind(Kind::Float);
        let token_logits = tokens.apply(&self.slot_logits);
        let token_assignments = token_logits.softmax(-1, Kind::Float) * mask.unsqueeze(-1);
        let slot_mass = token_assignments
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            .clamp_min(1e-6);
        let normalized_assignments = token_assignments.shallow_clone() / slot_mass.unsqueeze(1);
        let projected_tokens = tokens.apply(&self.slot_projector);
        let slots = normalized_assignments
            .transpose(1, 2)
            .bmm(&projected_tokens);
        let slot_weights = slot_mass.shallow_clone()
            / slot_mass
                .sum_dim_intlist([1].as_slice(), true, Kind::Float)
                .clamp_min(1e-6);
        let pooled = masked_token_mean(tokens, &mask);
        let slot_activation_logits = pooled.apply(&self.slot_activation_logits)
            + slot_mass_evidence(&slot_weights, self.num_slots)
                * self.activation_mass_evidence_weight;
        let has_tokens = mask
            .sum_dim_intlist([1].as_slice(), true, Kind::Float)
            .gt(0.0)
            .to_kind(Kind::Float);
        let slot_activations =
            (&slot_activation_logits / self.activation_temperature).sigmoid() * &has_tokens;
        let active_slot_mask = active_slot_mask_batch(
            &slot_activations,
            self.activation_threshold,
            self.attention_masking,
            self.minimum_visible_slots,
        ) * has_tokens;
        let active_slot_count =
            active_slot_count_batch(&slot_activations, self.activation_threshold);
        let slots = slots * slot_activations.unsqueeze(-1);
        let reconstructed_tokens = token_assignments
            .bmm(&slots)
            .apply(&self.reconstruction_projector)
            * mask.unsqueeze(-1);

        BatchedSlotEncoding {
            slots,
            slot_weights,
            token_assignments,
            slot_activation_logits,
            slot_activations,
            active_slot_mask,
            active_slot_count,
            reconstructed_tokens,
            token_mask: mask,
        }
    }
}

impl SlotDecomposer<ModalityEncoding, SlotEncoding> for SoftSlotDecomposer {
    fn decompose(&self, input: &ModalityEncoding) -> SlotEncoding {
        let tokens = &input.token_embeddings;
        if tokens.size()[0] == 0 {
            return SlotEncoding {
                slots: Tensor::zeros(
                    [self.num_slots, self.hidden_dim],
                    (Kind::Float, tokens.device()),
                ),
                slot_weights: Tensor::zeros([self.num_slots], (Kind::Float, tokens.device())),
                token_assignments: Tensor::zeros(
                    [0, self.num_slots],
                    (Kind::Float, tokens.device()),
                ),
                slot_activation_logits: Tensor::zeros(
                    [self.num_slots],
                    (Kind::Float, tokens.device()),
                ),
                slot_activations: Tensor::zeros([self.num_slots], (Kind::Float, tokens.device())),
                active_slot_mask: Tensor::zeros([self.num_slots], (Kind::Float, tokens.device())),
                active_slot_count: 0.0,
                reconstructed_tokens: Tensor::zeros(
                    [0, self.hidden_dim],
                    (Kind::Float, tokens.device()),
                ),
            };
        }

        let token_logits = tokens.apply(&self.slot_logits);
        let token_assignments = token_logits.softmax(-1, Kind::Float);
        let slot_mass = token_assignments
            .sum_dim_intlist([0].as_slice(), false, Kind::Float)
            .clamp_min(1e-6);
        let normalized_assignments = token_assignments.shallow_clone() / slot_mass.unsqueeze(0);

        let projected_tokens = tokens.apply(&self.slot_projector);
        let slots = normalized_assignments
            .transpose(0, 1)
            .matmul(&projected_tokens);
        let slot_weights = slot_mass.shallow_clone() / slot_mass.sum(Kind::Float).clamp_min(1e-6);
        let pooled = tokens.mean_dim([0].as_slice(), false, Kind::Float);
        let slot_activation_logits = pooled
            .unsqueeze(0)
            .apply(&self.slot_activation_logits)
            .squeeze_dim(0)
            + slot_mass_evidence(&slot_weights, self.num_slots)
                * self.activation_mass_evidence_weight;
        let slot_activations = (&slot_activation_logits / self.activation_temperature).sigmoid();
        let active_slot_mask = active_slot_mask(
            &slot_activations,
            self.activation_threshold,
            self.attention_masking,
            self.minimum_visible_slots,
        );
        let active_slot_count = active_slot_count(&slot_activations, self.activation_threshold);
        let slots = slots * slot_activations.unsqueeze(-1);
        let reconstructed_tokens = token_assignments
            .matmul(&slots)
            .apply(&self.reconstruction_projector);

        SlotEncoding {
            slots,
            slot_weights,
            token_assignments,
            slot_activation_logits,
            slot_activations,
            active_slot_mask,
            active_slot_count,
            reconstructed_tokens,
        }
    }
}

fn masked_token_mean(tokens: &Tensor, mask: &Tensor) -> Tensor {
    let denom = mask
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .clamp_min(1.0);
    (tokens * mask.unsqueeze(-1)).sum_dim_intlist([1].as_slice(), false, Kind::Float) / denom
}

fn slot_mass_evidence(slot_weights: &Tensor, num_slots: i64) -> Tensor {
    let slot_count = num_slots.max(1) as f64;
    (slot_weights.clamp_min(1e-6) * slot_count).log()
}

fn active_slot_count(slot_activations: &Tensor, activation_threshold: f64) -> f64 {
    slot_activations
        .gt(activation_threshold)
        .to_kind(Kind::Float)
        .sum(Kind::Float)
        .double_value(&[])
}

fn active_slot_mask(
    slot_activations: &Tensor,
    activation_threshold: f64,
    attention_masking: bool,
    minimum_visible_slots: i64,
) -> Tensor {
    if !attention_masking {
        return Tensor::ones(
            slot_activations.size(),
            (Kind::Float, slot_activations.device()),
        );
    }
    let hard_mask = slot_activations
        .gt(activation_threshold)
        .to_kind(Kind::Float);
    let slot_count = slot_activations.size().first().copied().unwrap_or(0).max(0);
    let visible_count = minimum_visible_slots.clamp(0, slot_count);
    if visible_count == 0 || slot_count == 0 {
        return hard_mask;
    }
    let (_, indices) = slot_activations.topk(visible_count, -1, true, true);
    let minimum_mask = Tensor::zeros_like(slot_activations).scatter_value(0, &indices, 1.0);
    hard_mask.maximum(&minimum_mask)
}

fn active_slot_count_batch(slot_activations: &Tensor, activation_threshold: f64) -> Tensor {
    slot_activations
        .gt(activation_threshold)
        .to_kind(Kind::Float)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
}

fn active_slot_mask_batch(
    slot_activations: &Tensor,
    activation_threshold: f64,
    attention_masking: bool,
    minimum_visible_slots: i64,
) -> Tensor {
    if !attention_masking {
        return Tensor::ones(
            slot_activations.size(),
            (Kind::Float, slot_activations.device()),
        );
    }
    let hard_mask = slot_activations
        .gt(activation_threshold)
        .to_kind(Kind::Float);
    let slot_count = slot_activations.size().get(1).copied().unwrap_or(0).max(0);
    let visible_count = minimum_visible_slots.clamp(0, slot_count);
    if visible_count == 0 || slot_count == 0 {
        return hard_mask;
    }
    let (_, indices) = slot_activations.topk(visible_count, -1, true, true);
    let minimum_mask = Tensor::zeros_like(slot_activations).scatter_value(1, &indices, 1.0);
    hard_mask.maximum(&minimum_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn slot_encoding_exposes_assignments_and_independent_activations() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let decomposer = SoftSlotDecomposer::new(&var_store.root(), 6, 4);
        let input = ModalityEncoding {
            token_embeddings: Tensor::ones([3, 6], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::zeros([6], (Kind::Float, Device::Cpu)),
        };

        let output = decomposer.decompose(&input);

        assert_eq!(output.slots.size(), vec![4, 6]);
        assert_eq!(output.token_assignments.size(), vec![3, 4]);
        assert_eq!(output.slot_activation_logits.size(), vec![4]);
        assert_eq!(output.slot_activations.size(), vec![4]);
        assert_eq!(output.active_slot_mask.size(), vec![4]);
        assert!(output.active_slot_count >= 0.0);
        assert!(output.active_slot_count <= 4.0);
        let row_sums = output
            .token_assignments
            .sum_dim_intlist([1].as_slice(), false, Kind::Float);
        let max_assignment_error = (row_sums - 1.0).abs().max().double_value(&[]);
        assert!(max_assignment_error < 1e-5);
        let min_activation = output.slot_activations.min().double_value(&[]);
        let max_activation = output.slot_activations.max().double_value(&[]);
        assert!((0.0..=1.0).contains(&min_activation));
        assert!((0.0..=1.0).contains(&max_activation));
        let min_mask = output.active_slot_mask.min().double_value(&[]);
        let max_mask = output.active_slot_mask.max().double_value(&[]);
        assert!((0.0..=1.0).contains(&min_mask));
        assert!((0.0..=1.0).contains(&max_mask));
    }

    #[test]
    fn batched_slot_encoding_masks_empty_rows_and_preserves_assignments() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let decomposer = SoftSlotDecomposer::new(&var_store.root(), 5, 3);
        let input = BatchedModalityEncoding {
            token_embeddings: Tensor::ones([2, 4, 5], (Kind::Float, Device::Cpu)),
            token_mask: Tensor::from_slice(&[1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .reshape([2, 4]),
            pooled_embedding: Tensor::zeros([2, 5], (Kind::Float, Device::Cpu)),
        };

        let output = decomposer.decompose_batch(&input);

        assert_eq!(output.slots.size(), vec![2, 3, 5]);
        assert_eq!(output.token_assignments.size(), vec![2, 4, 3]);
        assert_eq!(output.slot_activations.size(), vec![2, 3]);
        assert_eq!(output.active_slot_mask.size(), vec![2, 3]);
        assert_eq!(output.active_slot_count.size(), vec![2]);
        assert_eq!(
            output
                .slot_activations
                .get(1)
                .abs()
                .sum(Kind::Float)
                .double_value(&[]),
            0.0
        );
        assert_eq!(
            output
                .active_slot_mask
                .get(1)
                .abs()
                .sum(Kind::Float)
                .double_value(&[]),
            0.0
        );
        assert_eq!(output.active_slot_count.get(1).double_value(&[]), 0.0);
    }

    #[test]
    fn slot_activation_temperature_and_threshold_are_configurable() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = SlotDecompositionConfig {
            activation_temperature: 2.0,
            activation_threshold: 0.9,
            attention_masking: true,
            minimum_visible_slots: 1,
            activation_mass_evidence_weight: 0.5,
            balance_window: 8,
        };
        let decomposer = SoftSlotDecomposer::new_with_config(&var_store.root(), 4, 3, &config);
        let input = ModalityEncoding {
            token_embeddings: Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        };

        let output = decomposer.decompose(&input);

        assert_eq!(output.slot_activations.size(), vec![3]);
        assert!(output.active_slot_count <= 3.0);
    }

    #[test]
    fn slot_activation_logits_include_assignment_mass_evidence() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = SlotDecompositionConfig {
            activation_temperature: 1.0,
            activation_threshold: 0.5,
            attention_masking: true,
            minimum_visible_slots: 1,
            activation_mass_evidence_weight: 1.0,
            balance_window: 8,
        };
        let decomposer = SoftSlotDecomposer::new_with_config(&var_store.root(), 4, 3, &config);
        let input = ModalityEncoding {
            token_embeddings: Tensor::from_slice(&[
                1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .reshape([3, 4]),
            pooled_embedding: Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        };

        let output = decomposer.decompose(&input);
        let pooled = input
            .token_embeddings
            .mean_dim([0].as_slice(), false, Kind::Float);
        let pooled_only = pooled
            .unsqueeze(0)
            .apply(&decomposer.slot_activation_logits)
            .squeeze_dim(0);
        let expected = pooled_only + slot_mass_evidence(&output.slot_weights, decomposer.num_slots);
        let max_error = (output.slot_activation_logits - expected)
            .abs()
            .max()
            .double_value(&[]);

        assert!(max_error < 1e-5);
    }

    #[test]
    fn slot_mass_evidence_is_centered_on_uniform_usage() {
        let uniform = Tensor::from_slice(&[0.25_f32, 0.25, 0.25, 0.25]);
        let skewed = Tensor::from_slice(&[0.7_f32, 0.1, 0.1, 0.1]);

        let uniform_evidence = slot_mass_evidence(&uniform, 4);
        let skewed_evidence = slot_mass_evidence(&skewed, 4);

        assert!(
            uniform_evidence.abs().max().double_value(&[]) < 1e-6,
            "uniform slot usage should not bias activation logits"
        );
        assert!(skewed_evidence.double_value(&[0]) > 0.0);
        assert!(skewed_evidence.double_value(&[1]) < 0.0);
    }

    #[test]
    fn slot_attention_masking_can_be_disabled_for_ablation() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let config = SlotDecompositionConfig {
            activation_temperature: 1.0,
            activation_threshold: 0.99,
            attention_masking: false,
            minimum_visible_slots: 0,
            activation_mass_evidence_weight: 0.5,
            balance_window: 8,
        };
        let decomposer = SoftSlotDecomposer::new_with_config(&var_store.root(), 4, 3, &config);
        let input = ModalityEncoding {
            token_embeddings: Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            pooled_embedding: Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        };

        let output = decomposer.decompose(&input);

        assert_eq!(
            output.active_slot_mask.sum(Kind::Float).double_value(&[]),
            3.0
        );
    }

    #[test]
    fn slot_attention_mask_keeps_minimum_visible_slots_without_changing_hard_count() {
        let activations = Tensor::from_slice(&[0.1_f32, 0.2, 0.3, 0.4]);

        let warm_start_mask = active_slot_mask(&activations, 0.9, true, 2);
        let hard_mask = active_slot_mask(&activations, 0.9, true, 0);
        let hard_count = active_slot_count(&activations, 0.9);

        assert_eq!(hard_count, 0.0);
        assert_eq!(warm_start_mask.sum(Kind::Float).double_value(&[]), 2.0);
        assert_eq!(hard_mask.sum(Kind::Float).double_value(&[]), 0.0);
    }

    #[test]
    fn batched_slot_attention_mask_preserves_empty_rows_during_warm_start() {
        let activations = Tensor::from_slice(&[0.1_f32, 0.2, 0.3, 0.0, 0.0, 0.0]).reshape([2, 3]);
        let has_tokens = Tensor::from_slice(&[1.0_f32, 0.0]).reshape([2, 1]);

        let mask = active_slot_mask_batch(&activations, 0.9, true, 1) * has_tokens;

        assert_eq!(mask.get(0).sum(Kind::Float).double_value(&[]), 1.0);
        assert_eq!(mask.get(1).sum(Kind::Float).double_value(&[]), 0.0);
    }
}
