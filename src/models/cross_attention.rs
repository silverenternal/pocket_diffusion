//! Gated cross-modality attention with modality-specific projections.

use tch::{nn, Kind, Tensor};

use super::{CrossAttentionOutput, CrossModalInteractor, SlotEncoding};

/// One directed gated cross-attention path `target <- source`.
#[derive(Debug)]
pub struct GatedCrossAttention {
    query_proj: nn::Linear,
    key_proj: nn::Linear,
    value_proj: nn::Linear,
    gate_proj: nn::Linear,
    output_proj: nn::Linear,
    hidden_dim: i64,
}

impl GatedCrossAttention {
    /// Create a directed cross-attention module with an explicit scalar gate.
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
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
        Self {
            query_proj,
            key_proj,
            value_proj,
            gate_proj,
            output_proj,
            hidden_dim,
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

        let queries = query_slots.apply(&self.query_proj);
        let keys = key_slots.apply(&self.key_proj);
        let values = key_slots.apply(&self.value_proj);
        let scale = (self.hidden_dim as f64).sqrt();
        let attention_scores = queries.matmul(&keys.transpose(0, 1)) / scale;
        let attention_weights = attention_scores.softmax(-1, Kind::Float);
        let attended = attention_weights.matmul(&values);

        let query_summary = query_slots.mean_dim([0].as_slice(), false, Kind::Float);
        let source_summary = key_slots.mean_dim([0].as_slice(), false, Kind::Float);
        let gate = Tensor::cat(&[query_summary, source_summary], 0)
            .unsqueeze(0)
            .apply(&self.gate_proj)
            .sigmoid()
            .squeeze_dim(0);
        let attended_tokens = (attended * &gate).apply(&self.output_proj);

        CrossAttentionOutput {
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
