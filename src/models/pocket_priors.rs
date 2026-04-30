//! Explicit pocket-conditioned priors for de novo size and composition.

use tch::{nn, Tensor};

use super::ModalityEncoding;

/// Output of the pocket-conditioned prior heads.
#[derive(Debug)]
pub struct PocketConditionedPriorOutput {
    /// Logits over ligand atom-count classes `0..=max_atom_count`.
    pub atom_count_logits: Tensor,
    /// Logits over coarse element/composition tokens.
    pub composition_logits: Tensor,
    /// Maximum atom-count class represented by `atom_count_logits`.
    pub max_atom_count: usize,
    /// Atom vocabulary size represented by `composition_logits`.
    pub atom_vocab_size: i64,
    /// Feature source used by the explicit head.
    pub conditioning_source: String,
}

impl Clone for PocketConditionedPriorOutput {
    fn clone(&self) -> Self {
        Self {
            atom_count_logits: self.atom_count_logits.shallow_clone(),
            composition_logits: self.composition_logits.shallow_clone(),
            max_atom_count: self.max_atom_count,
            atom_vocab_size: self.atom_vocab_size,
            conditioning_source: self.conditioning_source.clone(),
        }
    }
}

/// Lightweight explicit head that maps pocket/context encoding to de novo priors.
#[derive(Debug)]
pub struct PocketConditionedPriorHead {
    atom_count_head: nn::Linear,
    composition_head: nn::Linear,
    max_atom_count: usize,
    atom_vocab_size: i64,
}

impl PocketConditionedPriorHead {
    /// Construct pocket-conditioned size and composition heads.
    pub fn new(
        vs: &nn::Path,
        hidden_dim: i64,
        max_atom_count: usize,
        atom_vocab_size: i64,
    ) -> Self {
        let atom_count_classes = max_atom_count.max(1) as i64 + 1;
        let usable_vocab = atom_vocab_size.max(1);
        Self {
            atom_count_head: nn::linear(
                vs / "atom_count_head",
                hidden_dim,
                atom_count_classes,
                Default::default(),
            ),
            composition_head: nn::linear(
                vs / "composition_head",
                hidden_dim,
                usable_vocab,
                Default::default(),
            ),
            max_atom_count: max_atom_count.max(1),
            atom_vocab_size: usable_vocab,
        }
    }

    /// Predict de novo prior logits from the pocket/context branch only.
    pub fn forward(&self, pocket: &ModalityEncoding) -> PocketConditionedPriorOutput {
        PocketConditionedPriorOutput {
            atom_count_logits: pocket
                .pooled_embedding
                .unsqueeze(0)
                .apply(&self.atom_count_head)
                .squeeze_dim(0),
            composition_logits: pocket
                .pooled_embedding
                .unsqueeze(0)
                .apply(&self.composition_head)
                .squeeze_dim(0),
            max_atom_count: self.max_atom_count,
            atom_vocab_size: self.atom_vocab_size,
            conditioning_source: "pocket_encoder_pooled_embedding_explicit_prior_head".to_string(),
        }
    }
}
