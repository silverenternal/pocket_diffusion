//! Gated cross-modality attention with lightweight and Transformer-style paths.

use tch::{nn, Kind, Tensor};

use crate::config::{CrossAttentionMode, InteractionTuningConfig};

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
        let query_slots = &target.slots;
        let key_slots = &source.slots;
        if query_slots.size()[0] == 0 || key_slots.size()[0] == 0 {
            return CrossAttentionOutput {
                gate: Tensor::zeros([1], (Kind::Float, query_slots.device())),
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
        let attention_scores = queries.matmul(&keys.transpose(0, 1)) / scale;
        let attention_weights = attention_scores.softmax(-1, Kind::Float);
        let attended = attention_weights.matmul(&values);

        let query_summary = query_slots.mean_dim([0].as_slice(), false, Kind::Float);
        let source_summary = key_slots.mean_dim([0].as_slice(), false, Kind::Float);
        let gate = ((Tensor::cat(&[query_summary, source_summary], 0)
            .unsqueeze(0)
            .apply(&self.gate_proj)
            + self.interaction_tuning.gate_bias)
            / self.interaction_tuning.gate_temperature)
            .sigmoid()
            .squeeze_dim(0);
        let gated_update = attended * &gate;
        let projected_update = gated_update.apply(&self.output_proj);
        let attended_tokens = match self.interaction_mode {
            CrossAttentionMode::Lightweight => projected_update,
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

        CrossAttentionOutput {
            gate,
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
            attention_scores += bias;
        }
        let attention_weights = attention_scores.softmax(-1, Kind::Float);
        let attended = attention_weights.bmm(&values);

        let query_summary = query_slots.mean_dim([1].as_slice(), false, Kind::Float);
        let source_summary = key_slots.mean_dim([1].as_slice(), false, Kind::Float);
        let gate = ((Tensor::cat(&[query_summary, source_summary], -1).apply(&self.gate_proj)
            + self.interaction_tuning.gate_bias)
            / self.interaction_tuning.gate_temperature)
            .sigmoid();
        let gated_update = attended * gate.unsqueeze(-1);
        let projected_update = gated_update.apply(&self.output_proj);
        let attended_tokens = match self.interaction_mode {
            CrossAttentionMode::Lightweight => projected_update,
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

        BatchedCrossAttentionOutput {
            gate,
            attended_tokens,
            attention_weights,
        }
    }
}

impl CrossModalInteractor<(&SlotEncoding, &SlotEncoding), CrossAttentionOutput>
    for GatedCrossAttention
{
    fn interact(&self, input: &(&SlotEncoding, &SlotEncoding)) -> CrossAttentionOutput {
        self.forward(input.0, input.1)
    }
}
