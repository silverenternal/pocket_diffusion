use crate::data::MolecularExample;
use tch::Tensor;

use super::{
    BatchedCrossAttentionOutput, BatchedCrossModalInteractions, BatchedDecomposedModalities,
    BatchedEncodedModalities, BatchedModalityEncoding, BatchedSlotEncoding, CrossAttentionOutput,
    CrossModalInteractions, DecomposedModalities, EncodedModalities, ModalityEncoding,
    SlotEncoding,
};

pub(crate) fn merge_slot_contexts(
    base_slots: &Tensor,
    updates: &[&CrossAttentionOutput],
) -> Tensor {
    let mut merged = base_slots.shallow_clone();
    for update in updates {
        merged = &merged + &update.attended_tokens;
    }
    merged
}

pub(crate) fn slice_encoded_modalities(
    batched: &BatchedEncodedModalities,
    batch_index: i64,
    example: &MolecularExample,
) -> EncodedModalities {
    let ligand_atoms = example.topology.atom_types.size()[0];
    let pocket_atoms = example.pocket.coords.size()[0];
    EncodedModalities {
        topology: slice_modality_encoding(&batched.topology, batch_index, ligand_atoms),
        geometry: slice_modality_encoding(&batched.geometry, batch_index, ligand_atoms),
        pocket: slice_modality_encoding(&batched.pocket, batch_index, pocket_atoms),
    }
}

fn slice_modality_encoding(
    batched: &BatchedModalityEncoding,
    batch_index: i64,
    token_count: i64,
) -> ModalityEncoding {
    ModalityEncoding {
        token_embeddings: batched
            .token_embeddings
            .get(batch_index)
            .narrow(0, 0, token_count)
            .shallow_clone(),
        pooled_embedding: batched.pooled_embedding.get(batch_index),
    }
}

pub(crate) fn slice_decomposed_modalities(
    batched: &BatchedDecomposedModalities,
    batch_index: i64,
    example: &MolecularExample,
) -> DecomposedModalities {
    let ligand_atoms = example.topology.atom_types.size()[0];
    let pocket_atoms = example.pocket.coords.size()[0];
    DecomposedModalities {
        topology: slice_slot_encoding(&batched.topology, batch_index, ligand_atoms),
        geometry: slice_slot_encoding(&batched.geometry, batch_index, ligand_atoms),
        pocket: slice_slot_encoding(&batched.pocket, batch_index, pocket_atoms),
    }
}

fn slice_slot_encoding(
    batched: &BatchedSlotEncoding,
    batch_index: i64,
    token_count: i64,
) -> SlotEncoding {
    SlotEncoding {
        slots: batched.slots.get(batch_index),
        slot_weights: batched.slot_weights.get(batch_index),
        token_assignments: batched
            .token_assignments
            .get(batch_index)
            .narrow(0, 0, token_count)
            .shallow_clone(),
        slot_activation_logits: batched.slot_activation_logits.get(batch_index),
        slot_activations: batched.slot_activations.get(batch_index),
        active_slot_mask: batched.active_slot_mask.get(batch_index),
        active_slot_count: batched.active_slot_count.get(batch_index).double_value(&[]),
        reconstructed_tokens: batched
            .reconstructed_tokens
            .get(batch_index)
            .narrow(0, 0, token_count)
            .shallow_clone(),
    }
}

pub(crate) fn slice_cross_modal_interactions(
    batched: &BatchedCrossModalInteractions,
    batch_index: i64,
) -> CrossModalInteractions {
    CrossModalInteractions {
        topo_from_geo: slice_cross_attention(&batched.topo_from_geo, batch_index),
        topo_from_pocket: slice_cross_attention(&batched.topo_from_pocket, batch_index),
        geo_from_topo: slice_cross_attention(&batched.geo_from_topo, batch_index),
        geo_from_pocket: slice_cross_attention(&batched.geo_from_pocket, batch_index),
        pocket_from_topo: slice_cross_attention(&batched.pocket_from_topo, batch_index),
        pocket_from_geo: slice_cross_attention(&batched.pocket_from_geo, batch_index),
    }
}

fn slice_cross_attention(
    batched: &BatchedCrossAttentionOutput,
    batch_index: i64,
) -> CrossAttentionOutput {
    CrossAttentionOutput {
        gate: batched.gate.get(batch_index),
        forced_open: batched.forced_open,
        attended_tokens: batched.attended_tokens.get(batch_index),
        attention_weights: batched.attention_weights.get(batch_index),
    }
}
