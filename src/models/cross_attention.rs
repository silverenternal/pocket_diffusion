//! Gated cross-modality attention with lightweight and Transformer-style paths.

use tch::{nn, Kind, Tensor};

use crate::config::{CrossAttentionMode, InteractionGateMode, InteractionTuningConfig};

use super::{
    BatchedCrossAttentionOutput, BatchedSlotEncoding, CrossAttentionOutput, CrossModalInteractor,
    SlotEncoding,
};

/// One directed gated cross-attention path `target <- source`.
#[derive(Debug)]
pub struct GatedCrossAttention {
    query_proj: nn::Linear,
    key_proj: nn::Linear,
    value_proj: nn::Linear,
    gate_proj: nn::Linear,
    output_proj: nn::Linear,
    target_norm: nn::LayerNorm,
    source_norm: nn::LayerNorm,
    refinement_norm: nn::LayerNorm,
    ffn_in: nn::Linear,
    ffn_out: nn::Linear,
    hidden_dim: i64,
    interaction_mode: CrossAttentionMode,
    interaction_tuning: InteractionTuningConfig,
}

impl GatedCrossAttention {
    /// Create a directed cross-attention module with an explicit scalar gate.
    pub fn new(
        vs: &nn::Path,
        hidden_dim: i64,
        interaction_mode: CrossAttentionMode,
        ff_multiplier: i64,
        interaction_tuning: InteractionTuningConfig,
    ) -> Self {
        let query_proj = nn::linear(
            vs / "query_proj",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let key_proj = nn::linear(vs / "key_proj", hidden_dim, hidden_dim, Default::default());
        let value_proj = nn::linear(
            vs / "value_proj",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let gate_proj = nn::linear(vs / "gate_proj", hidden_dim * 2, 1, Default::default());
        let output_proj = nn::linear(
            vs / "output_proj",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let target_norm = nn::layer_norm(vs / "target_norm", vec![hidden_dim], Default::default());
        let source_norm = nn::layer_norm(vs / "source_norm", vec![hidden_dim], Default::default());
        let refinement_norm =
            nn::layer_norm(vs / "refinement_norm", vec![hidden_dim], Default::default());
        let ffn_hidden_dim = hidden_dim * ff_multiplier.max(1);
        let ffn_in = nn::linear(
            vs / "ffn_in",
            hidden_dim,
            ffn_hidden_dim,
            Default::default(),
        );
        let ffn_out = nn::linear(
            vs / "ffn_out",
            ffn_hidden_dim,
            hidden_dim,
            Default::default(),
        );
        Self {
            query_proj,
            key_proj,
            value_proj,
            gate_proj,
            output_proj,
            target_norm,
            source_norm,
            refinement_norm,
            ffn_in,
            ffn_out,
            hidden_dim,
            interaction_mode,
            interaction_tuning,
        }
    }

    /// Apply directed cross-attention from `source` into `target`.
    pub fn forward(&self, target: &SlotEncoding, source: &SlotEncoding) -> CrossAttentionOutput {
        self.forward_with_attention_bias(target, source, None)
    }

    /// Apply directed cross-attention from `source` into `target` with an optional score bias.
    pub(crate) fn forward_with_attention_bias(
        &self,
        target: &SlotEncoding,
        source: &SlotEncoding,
        attention_bias: Option<&Tensor>,
    ) -> CrossAttentionOutput {
        let query_slots = &target.slots;
        let key_slots = &source.slots;
        if query_slots.size()[0] == 0 || key_slots.size()[0] == 0 {
            return CrossAttentionOutput {
                gate: Tensor::zeros([1], (Kind::Float, query_slots.device())),
                forced_open: matches!(
                    self.interaction_mode,
                    CrossAttentionMode::DirectFusionNegativeControl
                ),
                attended_tokens: Tensor::zeros(
                    [query_slots.size()[0], self.hidden_dim],
                    (Kind::Float, query_slots.device()),
                ),
                attention_weights: Tensor::zeros(
                    [query_slots.size()[0], key_slots.size()[0]],
                    (Kind::Float, query_slots.device()),
                ),
            };
        }

        let normalized_queries = query_slots.apply(&self.target_norm);
        let normalized_keys = key_slots.apply(&self.source_norm);
        let queries = normalized_queries.apply(&self.query_proj);
        let keys = normalized_keys.apply(&self.key_proj);
        let values = normalized_keys.apply(&self.value_proj);
        let scale = (self.hidden_dim as f64).sqrt();
        let mut attention_scores = queries.matmul(&keys.transpose(0, 1)) / scale;
        if let Some(bias) = attention_bias {
            attention_scores = attention_scores + bias;
        }
        let source_mask = expand_source_slot_mask(&source.active_slot_mask, &attention_scores);
        let target_mask = expand_target_slot_mask(&target.active_slot_mask, &attention_scores);
        let source_available = source_mask
            .sum_dim_intlist([1].as_slice(), true, Kind::Float)
            .gt(0.0)
            .to_kind(Kind::Float);
        let attention_weights =
            masked_attention_softmax(&attention_scores, &source_mask) * &target_mask;
        let attended = attention_weights.matmul(&values);

        let learned_gate = self.learned_gate(
            query_slots,
            key_slots,
            &attended,
            &target.active_slot_mask,
            &source.active_slot_mask,
        );
        let gate = match self.interaction_mode {
            CrossAttentionMode::DirectFusionNegativeControl => Tensor::ones_like(&learned_gate),
            CrossAttentionMode::Lightweight | CrossAttentionMode::Transformer => learned_gate,
        };
        let forced_open = matches!(
            self.interaction_mode,
            CrossAttentionMode::DirectFusionNegativeControl
        );
        let gate = if forced_open {
            gate
        } else {
            mask_target_slot_gate(gate, &target_mask, &source_available)
        };
        let gated_update = attended * &gate;
        let projected_update = gated_update.apply(&self.output_proj);
        let attended_tokens = match self.interaction_mode {
            CrossAttentionMode::Lightweight | CrossAttentionMode::DirectFusionNegativeControl => {
                projected_update
            }
            CrossAttentionMode::Transformer => {
                let residual_tokens = query_slots
                    + projected_update * self.interaction_tuning.attention_residual_scale;
                let refinement_input = if self.interaction_tuning.transformer_pre_norm {
                    residual_tokens.apply(&self.refinement_norm)
                } else {
                    residual_tokens.shallow_clone()
                };
                let refined_tokens = residual_tokens.shallow_clone()
                    + refinement_input
                        .apply(&self.ffn_in)
                        .relu()
                        .apply(&self.ffn_out)
                        * self.interaction_tuning.ffn_residual_scale;
                let refined_tokens = if self.interaction_tuning.transformer_pre_norm {
                    refined_tokens
                } else {
                    refined_tokens.apply(&self.refinement_norm)
                };
                refined_tokens - query_slots
            }
        };
        let attended_tokens = attended_tokens * target_mask * source_available;

        CrossAttentionOutput {
            gate,
            forced_open,
            attended_tokens,
            attention_weights,
        }
    }

    /// Apply directed cross-attention to padded slot batches.
    pub(crate) fn forward_batch(
        &self,
        target: &BatchedSlotEncoding,
        source: &BatchedSlotEncoding,
        attention_bias: Option<&Tensor>,
    ) -> BatchedCrossAttentionOutput {
        let query_slots = &target.slots;
        let key_slots = &source.slots;
        let batch = query_slots.size()[0];
        let query_len = query_slots.size()[1];
        let key_len = key_slots.size()[1];
        if query_len == 0 || key_len == 0 {
            return BatchedCrossAttentionOutput {
                gate: Tensor::zeros([batch, 1], (Kind::Float, query_slots.device())),
                forced_open: matches!(
                    self.interaction_mode,
                    CrossAttentionMode::DirectFusionNegativeControl
                ),
                attended_tokens: Tensor::zeros(
                    [batch, query_len, self.hidden_dim],
                    (Kind::Float, query_slots.device()),
                ),
                attention_weights: Tensor::zeros(
                    [batch, query_len, key_len],
                    (Kind::Float, query_slots.device()),
                ),
            };
        }

        let normalized_queries = query_slots.apply(&self.target_norm);
        let normalized_keys = key_slots.apply(&self.source_norm);
        let queries = normalized_queries.apply(&self.query_proj);
        let keys = normalized_keys.apply(&self.key_proj);
        let values = normalized_keys.apply(&self.value_proj);
        let scale = (self.hidden_dim as f64).sqrt();
        let mut attention_scores = queries.bmm(&keys.transpose(1, 2)) / scale;
        if let Some(bias) = attention_bias {
            attention_scores = attention_scores + bias;
        }
        let source_mask = expand_source_slot_mask(&source.active_slot_mask, &attention_scores);
        let target_mask = expand_target_slot_mask(&target.active_slot_mask, &attention_scores);
        let source_available = source_mask
            .sum_dim_intlist([2].as_slice(), true, Kind::Float)
            .gt(0.0)
            .to_kind(Kind::Float);
        let attention_weights =
            masked_attention_softmax(&attention_scores, &source_mask) * &target_mask;
        let attended = attention_weights.bmm(&values);

        let learned_gate = self.learned_gate_batch(
            query_slots,
            key_slots,
            &attended,
            &target.active_slot_mask,
            &source.active_slot_mask,
        );
        let gate = match self.interaction_mode {
            CrossAttentionMode::DirectFusionNegativeControl => Tensor::ones_like(&learned_gate),
            CrossAttentionMode::Lightweight | CrossAttentionMode::Transformer => learned_gate,
        };
        let forced_open = matches!(
            self.interaction_mode,
            CrossAttentionMode::DirectFusionNegativeControl
        );
        let gate = if forced_open {
            gate
        } else {
            mask_target_slot_gate(gate, &target_mask, &source_available)
        };
        let gated_update = match gate.size().len() {
            2 => attended * gate.unsqueeze(-1),
            _ => attended * &gate,
        };
        let projected_update = gated_update.apply(&self.output_proj);
        let attended_tokens = match self.interaction_mode {
            CrossAttentionMode::Lightweight | CrossAttentionMode::DirectFusionNegativeControl => {
                projected_update
            }
            CrossAttentionMode::Transformer => {
                let residual_tokens = query_slots
                    + projected_update * self.interaction_tuning.attention_residual_scale;
                let refinement_input = if self.interaction_tuning.transformer_pre_norm {
                    residual_tokens.apply(&self.refinement_norm)
                } else {
                    residual_tokens.shallow_clone()
                };
                let refined_tokens = residual_tokens.shallow_clone()
                    + refinement_input
                        .apply(&self.ffn_in)
                        .relu()
                        .apply(&self.ffn_out)
                        * self.interaction_tuning.ffn_residual_scale;
                let refined_tokens = if self.interaction_tuning.transformer_pre_norm {
                    refined_tokens
                } else {
                    refined_tokens.apply(&self.refinement_norm)
                };
                refined_tokens - query_slots
            }
        };
        let attended_tokens = attended_tokens * target_mask * source_available;

        BatchedCrossAttentionOutput {
            gate,
            forced_open,
            attended_tokens,
            attention_weights,
        }
    }

    fn learned_gate(
        &self,
        query_slots: &Tensor,
        key_slots: &Tensor,
        attended: &Tensor,
        target_slot_mask: &Tensor,
        source_slot_mask: &Tensor,
    ) -> Tensor {
        let logits = match self.interaction_tuning.gate_mode {
            InteractionGateMode::PathScalar => {
                let query_summary = masked_slot_mean(query_slots, target_slot_mask);
                let source_summary = masked_slot_mean(key_slots, source_slot_mask);
                Tensor::cat(&[query_summary, source_summary], 0)
                    .unsqueeze(0)
                    .apply(&self.gate_proj)
                    .squeeze_dim(0)
            }
            InteractionGateMode::TargetSlot => {
                Tensor::cat(&[query_slots.shallow_clone(), attended.shallow_clone()], 1)
                    .apply(&self.gate_proj)
            }
        };
        self.activate_gate(logits)
    }

    fn learned_gate_batch(
        &self,
        query_slots: &Tensor,
        key_slots: &Tensor,
        attended: &Tensor,
        target_slot_mask: &Tensor,
        source_slot_mask: &Tensor,
    ) -> Tensor {
        let logits = match self.interaction_tuning.gate_mode {
            InteractionGateMode::PathScalar => {
                let query_summary = masked_slot_mean_batch(query_slots, target_slot_mask);
                let source_summary = masked_slot_mean_batch(key_slots, source_slot_mask);
                Tensor::cat(&[query_summary, source_summary], -1).apply(&self.gate_proj)
            }
            InteractionGateMode::TargetSlot => {
                Tensor::cat(&[query_slots.shallow_clone(), attended.shallow_clone()], -1)
                    .apply(&self.gate_proj)
            }
        };
        self.activate_gate(logits)
    }

    fn activate_gate(&self, logits: Tensor) -> Tensor {
        ((logits + self.interaction_tuning.gate_bias) / self.interaction_tuning.gate_temperature)
            .sigmoid()
    }
}

fn masked_slot_mean(slots: &Tensor, slot_mask: &Tensor) -> Tensor {
    let mask = slot_mask
        .to_device(slots.device())
        .to_kind(Kind::Float)
        .reshape([slot_mask.size().first().copied().unwrap_or(0).max(0), 1]);
    let hidden_dim = slots.size().get(1).copied().unwrap_or(0).max(0);
    if slots.numel() == 0 || mask.numel() == 0 || mask.size()[0] != slots.size()[0] {
        return Tensor::zeros([hidden_dim], (Kind::Float, slots.device()));
    }
    (slots * &mask).sum_dim_intlist([0].as_slice(), false, Kind::Float)
        / mask.sum(Kind::Float).clamp_min(1.0e-6)
}

fn masked_slot_mean_batch(slots: &Tensor, slot_mask: &Tensor) -> Tensor {
    let hidden_dim = slots.size().get(2).copied().unwrap_or(0).max(0);
    let batch = slots.size().first().copied().unwrap_or(0).max(0);
    let mask = slot_mask.to_device(slots.device()).to_kind(Kind::Float);
    if slots.numel() == 0
        || mask.numel() == 0
        || mask.size().as_slice() != [batch, slots.size().get(1).copied().unwrap_or(0).max(0)]
    {
        return Tensor::zeros([batch, hidden_dim], (Kind::Float, slots.device()));
    }
    let mask = mask.unsqueeze(-1);
    (slots * &mask).sum_dim_intlist([1].as_slice(), false, Kind::Float)
        / mask
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            .clamp_min(1.0e-6)
}

fn mask_target_slot_gate(gate: Tensor, target_mask: &Tensor, source_available: &Tensor) -> Tensor {
    if gate.size() == target_mask.size() {
        gate * target_mask * source_available
    } else {
        gate
    }
}

fn masked_attention_softmax(attention_scores: &Tensor, source_mask: &Tensor) -> Tensor {
    let mask = source_mask.clamp(0.0, 1.0);
    let invalid = Tensor::ones_like(&mask) - &mask;
    let masked_scores = attention_scores + invalid * -1.0e9;
    let weights = masked_scores.softmax(-1, Kind::Float) * &mask;
    let last_dim = (weights.size().len() - 1) as i64;
    let denom = weights
        .sum_dim_intlist([last_dim].as_slice(), true, Kind::Float)
        .clamp_min(1e-6);
    weights / denom
}

fn expand_source_slot_mask(source_mask: &Tensor, attention_scores: &Tensor) -> Tensor {
    let scores_size = attention_scores.size();
    let device = attention_scores.device();
    match scores_size.as_slice() {
        [_, key_len] => {
            let mask = source_mask.to_device(device).to_kind(Kind::Float);
            if mask.size().as_slice() == [*key_len] {
                mask.reshape([1, *key_len])
            } else {
                Tensor::ones([1, *key_len], (Kind::Float, device))
            }
        }
        [batch, _, key_len] => {
            let mask = source_mask.to_device(device).to_kind(Kind::Float);
            match mask.size().as_slice() {
                [mask_batch, mask_keys] if mask_batch == batch && mask_keys == key_len => {
                    mask.unsqueeze(1)
                }
                [mask_keys] if mask_keys == key_len => mask.reshape([1, 1, *key_len]),
                _ => Tensor::ones([*batch, 1, *key_len], (Kind::Float, device)),
            }
        }
        _ => Tensor::ones(attention_scores.size(), (Kind::Float, device)),
    }
}

fn expand_target_slot_mask(target_mask: &Tensor, attention_scores: &Tensor) -> Tensor {
    let scores_size = attention_scores.size();
    let device = attention_scores.device();
    match scores_size.as_slice() {
        [query_len, _] => {
            let mask = target_mask.to_device(device).to_kind(Kind::Float);
            if mask.size().as_slice() == [*query_len] {
                mask.reshape([*query_len, 1])
            } else {
                Tensor::ones([*query_len, 1], (Kind::Float, device))
            }
        }
        [batch, query_len, _] => {
            let mask = target_mask.to_device(device).to_kind(Kind::Float);
            match mask.size().as_slice() {
                [mask_batch, mask_queries] if mask_batch == batch && mask_queries == query_len => {
                    mask.unsqueeze(-1)
                }
                [mask_queries] if mask_queries == query_len => mask.reshape([1, *query_len, 1]),
                _ => Tensor::ones([*batch, *query_len, 1], (Kind::Float, device)),
            }
        }
        _ => Tensor::ones(attention_scores.size(), (Kind::Float, device)),
    }
}

impl CrossModalInteractor<(&SlotEncoding, &SlotEncoding), CrossAttentionOutput>
    for GatedCrossAttention
{
    fn interact(&self, input: &(&SlotEncoding, &SlotEncoding)) -> CrossAttentionOutput {
        self.forward(input.0, input.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn cross_attention_masks_inactive_source_and_target_slots() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::Lightweight,
            2,
            InteractionTuningConfig::default(),
        );
        let target = slot_encoding(
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 0.0],
        );
        let source = slot_encoding(
            Tensor::ones([3, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 0.0, 1.0],
        );

        let output = attention.forward(&target, &source);

        assert!(
            output
                .attention_weights
                .select(1, 1)
                .abs()
                .max()
                .double_value(&[])
                < 1e-6
        );
        assert!(
            output
                .attention_weights
                .get(1)
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                < 1e-6
        );
        assert!(
            output
                .attended_tokens
                .get(1)
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                < 1e-6
        );
    }

    #[test]
    fn cross_attention_all_masked_source_has_zero_attention_and_update() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::Lightweight,
            2,
            InteractionTuningConfig::default(),
        );
        let target = slot_encoding(
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 1.0],
        );
        let source = slot_encoding(
            Tensor::ones([3, 4], (Kind::Float, Device::Cpu)),
            &[0.0, 0.0, 0.0],
        );

        let output = attention.forward(&target, &source);

        assert!(
            output
                .attention_weights
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                < 1e-6
        );
        assert!(
            output
                .attended_tokens
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                < 1e-6
        );
    }

    #[test]
    fn direct_fusion_negative_control_forces_gate_open() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::DirectFusionNegativeControl,
            2,
            InteractionTuningConfig {
                gate_mode: InteractionGateMode::PathScalar,
                ..InteractionTuningConfig::default()
            },
        );
        let target = slot_encoding(
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 1.0],
        );
        let source = slot_encoding(
            Tensor::ones([3, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 1.0, 1.0],
        );

        let output = attention.forward(&target, &source);

        assert!((output.gate.double_value(&[0]) - 1.0).abs() < 1e-6);
        assert!(output.forced_open);
        assert!(
            output
                .attended_tokens
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                > 0.0
        );
    }

    #[test]
    fn target_slot_gate_mode_emits_one_gate_per_target_slot() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::Lightweight,
            2,
            InteractionTuningConfig {
                gate_mode: InteractionGateMode::TargetSlot,
                ..InteractionTuningConfig::default()
            },
        );
        let target = slot_encoding(
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 1.0],
        );
        let source = slot_encoding(
            Tensor::ones([3, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 1.0, 1.0],
        );

        let output = attention.forward(&target, &source);

        assert_eq!(output.gate.size(), vec![2, 1]);
        assert!(output.gate.min().double_value(&[]) >= 0.0);
        assert!(output.gate.max().double_value(&[]) <= 1.0);
    }

    #[test]
    fn target_slot_gate_mode_masks_inactive_target_slots() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::Lightweight,
            2,
            InteractionTuningConfig {
                gate_mode: InteractionGateMode::TargetSlot,
                ..InteractionTuningConfig::default()
            },
        );
        let target = slot_encoding(
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 0.0],
        );
        let source = slot_encoding(
            Tensor::ones([3, 4], (Kind::Float, Device::Cpu)),
            &[1.0, 1.0, 1.0],
        );

        let output = attention.forward(&target, &source);

        assert_eq!(output.gate.size(), vec![2, 1]);
        assert!(output.gate.double_value(&[0, 0]) > 0.0);
        assert_eq!(output.gate.double_value(&[1, 0]), 0.0);
    }

    #[test]
    fn path_scalar_gate_summary_ignores_inactive_slots() {
        let slots =
            Tensor::from_slice(&[2.0_f32, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0]).reshape([2, 4]);
        let mask = Tensor::from_slice(&[1.0_f32, 0.0]);

        let summary = masked_slot_mean(&slots, &mask);

        assert_eq!(summary.double_value(&[0]), 2.0);
    }

    #[test]
    fn batched_target_slot_gate_mode_emits_gate_tensor_per_target_slot() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::Transformer,
            2,
            InteractionTuningConfig {
                gate_mode: InteractionGateMode::TargetSlot,
                ..InteractionTuningConfig::default()
            },
        );
        let target = batched_slot_encoding(
            Tensor::ones([2, 2, 4], (Kind::Float, Device::Cpu)),
            Tensor::ones([2, 2], (Kind::Float, Device::Cpu)),
        );
        let source = batched_slot_encoding(
            Tensor::ones([2, 3, 4], (Kind::Float, Device::Cpu)),
            Tensor::ones([2, 3], (Kind::Float, Device::Cpu)),
        );

        let output = attention.forward_batch(&target, &source, None);

        assert_eq!(output.gate.size(), vec![2, 2, 1]);
        assert_eq!(output.attended_tokens.size(), vec![2, 2, 4]);
    }

    #[test]
    fn batched_target_slot_gate_mode_masks_inactive_target_slots() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::Transformer,
            2,
            InteractionTuningConfig {
                gate_mode: InteractionGateMode::TargetSlot,
                ..InteractionTuningConfig::default()
            },
        );
        let target = batched_slot_encoding(
            Tensor::ones([2, 2, 4], (Kind::Float, Device::Cpu)),
            Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 1.0]).reshape([2, 2]),
        );
        let source = batched_slot_encoding(
            Tensor::ones([2, 3, 4], (Kind::Float, Device::Cpu)),
            Tensor::ones([2, 3], (Kind::Float, Device::Cpu)),
        );

        let output = attention.forward_batch(&target, &source, None);

        assert_eq!(output.gate.size(), vec![2, 2, 1]);
        assert_eq!(output.gate.double_value(&[0, 1, 0]), 0.0);
        assert_eq!(output.gate.double_value(&[1, 0, 0]), 0.0);
    }

    #[test]
    fn batched_cross_attention_masks_source_slots_per_example() {
        let vs = nn::VarStore::new(Device::Cpu);
        let attention = GatedCrossAttention::new(
            &vs.root(),
            4,
            CrossAttentionMode::Lightweight,
            2,
            InteractionTuningConfig::default(),
        );
        let target = batched_slot_encoding(
            Tensor::ones([2, 2, 4], (Kind::Float, Device::Cpu)),
            Tensor::ones([2, 2], (Kind::Float, Device::Cpu)),
        );
        let source = batched_slot_encoding(
            Tensor::ones([2, 3, 4], (Kind::Float, Device::Cpu)),
            Tensor::from_slice(&[1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0]).reshape([2, 3]),
        );

        let output = attention.forward_batch(&target, &source, None);

        assert!(
            output
                .attention_weights
                .get(0)
                .select(1, 1)
                .abs()
                .max()
                .double_value(&[])
                < 1e-6
        );
        assert!(
            output
                .attention_weights
                .get(1)
                .select(1, 0)
                .abs()
                .max()
                .double_value(&[])
                < 1e-6
        );
        assert!(
            output
                .attention_weights
                .get(1)
                .select(1, 2)
                .abs()
                .max()
                .double_value(&[])
                < 1e-6
        );
    }

    fn slot_encoding(slots: Tensor, mask_values: &[f32]) -> SlotEncoding {
        let device = slots.device();
        let slot_count = slots.size()[0];
        let hidden_dim = slots.size()[1];
        let active_slot_mask = Tensor::from_slice(mask_values).to_device(device);
        SlotEncoding {
            slots,
            slot_weights: Tensor::ones([slot_count], (Kind::Float, device)) / slot_count as f64,
            token_assignments: Tensor::zeros([0, slot_count], (Kind::Float, device)),
            slot_activation_logits: active_slot_mask.shallow_clone(),
            slot_activations: active_slot_mask.shallow_clone(),
            active_slot_mask: active_slot_mask.shallow_clone(),
            active_slot_count: active_slot_mask.sum(Kind::Float).double_value(&[]),
            reconstructed_tokens: Tensor::zeros([0, hidden_dim], (Kind::Float, device)),
        }
    }

    fn batched_slot_encoding(slots: Tensor, active_slot_mask: Tensor) -> BatchedSlotEncoding {
        let device = slots.device();
        let batch = slots.size()[0];
        let slot_count = slots.size()[1];
        let hidden_dim = slots.size()[2];
        BatchedSlotEncoding {
            slots,
            slot_weights: Tensor::ones([batch, slot_count], (Kind::Float, device))
                / slot_count as f64,
            token_assignments: Tensor::zeros([batch, 0, slot_count], (Kind::Float, device)),
            slot_activation_logits: active_slot_mask.shallow_clone(),
            slot_activations: active_slot_mask.shallow_clone(),
            active_slot_mask: active_slot_mask.shallow_clone(),
            active_slot_count: active_slot_mask.sum_dim_intlist([1].as_slice(), false, Kind::Float),
            reconstructed_tokens: Tensor::zeros([batch, 0, hidden_dim], (Kind::Float, device)),
            token_mask: Tensor::zeros([batch, 0], (Kind::Float, device)),
        }
    }
}
