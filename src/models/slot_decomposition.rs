//! Learned slot decomposition for each modality representation.

use tch::{nn, Kind, Tensor};

use super::{
    BatchedModalityEncoding, BatchedSlotEncoding, ModalityEncoding, SlotDecomposer, SlotEncoding,
};

/// Soft slot decomposition with sparse activations and shared slot upper bound.
#[derive(Debug)]
pub struct SoftSlotDecomposer {
    slot_logits: nn::Linear,
    slot_projector: nn::Linear,
    reconstruction_projector: nn::Linear,
    num_slots: i64,
    hidden_dim: i64,
}

impl SoftSlotDecomposer {
    /// Create a slot decomposer with a fixed slot budget.
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_slots: i64) -> Self {
        let slot_logits = nn::linear(
            vs / "slot_logits",
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
            slot_projector,
            reconstruction_projector,
            num_slots,
            hidden_dim,
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
        let reconstructed_tokens = token_assignments
            .bmm(&slots)
            .apply(&self.reconstruction_projector)
            * mask.unsqueeze(-1);

        BatchedSlotEncoding {
            slots,
            slot_weights,
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
        let reconstructed_tokens = token_assignments
            .matmul(&slots)
            .apply(&self.reconstruction_projector);

        SlotEncoding {
            slots,
            slot_weights,
            reconstructed_tokens,
        }
    }
}
