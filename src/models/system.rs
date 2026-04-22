//! Phase 2 system wiring for separate encoders, slots, controlled interaction, and probes.

use tch::{nn, Kind, Tensor};

use super::{
    ConditionedGenerationState, ConditionedLigandDecoder, CrossAttentionOutput, DecoderOutput,
    Encoder, GatedCrossAttention, GeometryEncoderImpl, ModalityEncoding, ModularLigandDecoder,
    PartialLigandState, PocketEncoderImpl, ProbeOutputs, SemanticProbeHeads, SlotDecomposer,
    SlotEncoding, SoftSlotDecomposer, TopologyEncoderImpl,
};
use crate::{
    config::ResearchConfig,
    data::{MolecularBatch, MolecularExample},
};

/// Output bundle for the separate modality encoders.
#[derive(Debug, Clone)]
pub(crate) struct EncodedModalities {
    /// Topology encoding.
    pub topology: ModalityEncoding,
    /// Geometry encoding.
    pub geometry: ModalityEncoding,
    /// Pocket/context encoding.
    pub pocket: ModalityEncoding,
}

/// Slot-decomposed outputs for each modality.
#[derive(Debug, Clone)]
pub(crate) struct DecomposedModalities {
    /// Topology slots.
    pub topology: SlotEncoding,
    /// Geometry slots.
    pub geometry: SlotEncoding,
    /// Pocket/context slots.
    pub pocket: SlotEncoding,
}

/// Directed gated cross-modality interactions.
#[derive(Debug, Clone)]
pub(crate) struct CrossModalInteractions {
    /// Topology receiving information from geometry.
    pub topo_from_geo: CrossAttentionOutput,
    /// Topology receiving information from pocket context.
    pub topo_from_pocket: CrossAttentionOutput,
    /// Geometry receiving information from topology.
    pub geo_from_topo: CrossAttentionOutput,
    /// Geometry receiving information from pocket context.
    pub geo_from_pocket: CrossAttentionOutput,
    /// Pocket receiving information from topology.
    pub pocket_from_topo: CrossAttentionOutput,
    /// Pocket receiving information from geometry.
    pub pocket_from_geo: CrossAttentionOutput,
}

/// Decoder-facing generation bundle produced by the modular backbone.
#[derive(Debug, Clone)]
pub(crate) struct GenerationForward {
    /// Explicit decoder input state with separated modality conditioning.
    pub state: ConditionedGenerationState,
    /// Decoder output for the current ligand draft.
    pub decoded: DecoderOutput,
}

/// Full Phase 2 forward-pass bundle.
#[derive(Debug, Clone)]
pub(crate) struct ResearchForward {
    /// Pre-decomposition modality encodings.
    pub encodings: EncodedModalities,
    /// Slot-decomposed modality encodings.
    pub slots: DecomposedModalities,
    /// Directed gated interactions.
    pub interactions: CrossModalInteractions,
    /// Semantic probe predictions.
    pub probes: ProbeOutputs,
    /// Decoder-facing conditioned generation path.
    pub generation: GenerationForward,
}

/// Research system that keeps encoders separate and adds structured interactions on top.
#[derive(Debug)]
pub struct Phase1ResearchSystem {
    /// Topology encoder.
    pub topo_encoder: TopologyEncoderImpl,
    /// Geometry encoder.
    pub geo_encoder: GeometryEncoderImpl,
    /// Pocket encoder.
    pub pocket_encoder: PocketEncoderImpl,
    /// Topology slot decomposer.
    pub topo_slots: SoftSlotDecomposer,
    /// Geometry slot decomposer.
    pub geo_slots: SoftSlotDecomposer,
    /// Pocket slot decomposer.
    pub pocket_slots: SoftSlotDecomposer,
    /// Directed interactions into topology.
    pub topo_from_geo: GatedCrossAttention,
    pub topo_from_pocket: GatedCrossAttention,
    /// Directed interactions into geometry.
    pub geo_from_topo: GatedCrossAttention,
    pub geo_from_pocket: GatedCrossAttention,
    /// Directed interactions into pocket.
    pub pocket_from_topo: GatedCrossAttention,
    pub pocket_from_geo: GatedCrossAttention,
    /// Minimal modular ligand decoder.
    pub ligand_decoder: ModularLigandDecoder,
    /// Semantic probe heads.
    pub probes: SemanticProbeHeads,
}

impl Phase1ResearchSystem {
    /// Construct the modular Phase 2 system from configuration.
    pub fn new(vs: &nn::Path, config: &ResearchConfig) -> Self {
        let topo_encoder = TopologyEncoderImpl::new(
            &(vs / "topology"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
        );
        let geo_encoder = GeometryEncoderImpl::new(&(vs / "geometry"), config.model.hidden_dim);
        let pocket_encoder = PocketEncoderImpl::new(
            &(vs / "pocket"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
        );
        let topo_slots = SoftSlotDecomposer::new(
            &(vs / "slot_topology"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geo_slots = SoftSlotDecomposer::new(
            &(vs / "slot_geometry"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket_slots = SoftSlotDecomposer::new(
            &(vs / "slot_pocket"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let topo_from_geo =
            GatedCrossAttention::new(&(vs / "topo_from_geo"), config.model.hidden_dim);
        let topo_from_pocket =
            GatedCrossAttention::new(&(vs / "topo_from_pocket"), config.model.hidden_dim);
        let geo_from_topo =
            GatedCrossAttention::new(&(vs / "geo_from_topo"), config.model.hidden_dim);
        let geo_from_pocket =
            GatedCrossAttention::new(&(vs / "geo_from_pocket"), config.model.hidden_dim);
        let pocket_from_topo =
            GatedCrossAttention::new(&(vs / "pocket_from_topo"), config.model.hidden_dim);
        let pocket_from_geo =
            GatedCrossAttention::new(&(vs / "pocket_from_geo"), config.model.hidden_dim);
        let ligand_decoder = ModularLigandDecoder::new(
            &(vs / "ligand_decoder"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
        );
        let probes = SemanticProbeHeads::new(
            &(vs / "probes"),
            config.model.hidden_dim,
            config.model.pocket_feature_dim,
        );

        Self {
            topo_encoder,
            geo_encoder,
            pocket_encoder,
            topo_slots,
            geo_slots,
            pocket_slots,
            topo_from_geo,
            topo_from_pocket,
            geo_from_topo,
            geo_from_pocket,
            pocket_from_topo,
            pocket_from_geo,
            ligand_decoder,
            probes,
        }
    }

    /// Run the three modality encoders for one example.
    pub(crate) fn encode_example(&self, example: &MolecularExample) -> EncodedModalities {
        EncodedModalities {
            topology: self.topo_encoder.encode(&example.topology),
            geometry: self.geo_encoder.encode(&example.geometry),
            pocket: self.pocket_encoder.encode(&example.pocket),
        }
    }

    /// Decompose the three modality encodings into learned slots.
    pub(crate) fn decompose_modalities(
        &self,
        encodings: &EncodedModalities,
    ) -> DecomposedModalities {
        DecomposedModalities {
            topology: self.topo_slots.decompose(&encodings.topology),
            geometry: self.geo_slots.decompose(&encodings.geometry),
            pocket: self.pocket_slots.decompose(&encodings.pocket),
        }
    }

    /// Apply all directed cross-modality interactions.
    pub(crate) fn interact_modalities(
        &self,
        slots: &DecomposedModalities,
    ) -> CrossModalInteractions {
        CrossModalInteractions {
            topo_from_geo: self.topo_from_geo.forward(&slots.topology, &slots.geometry),
            topo_from_pocket: self
                .topo_from_pocket
                .forward(&slots.topology, &slots.pocket),
            geo_from_topo: self.geo_from_topo.forward(&slots.geometry, &slots.topology),
            geo_from_pocket: self.geo_from_pocket.forward(&slots.geometry, &slots.pocket),
            pocket_from_topo: self
                .pocket_from_topo
                .forward(&slots.pocket, &slots.topology),
            pocket_from_geo: self.pocket_from_geo.forward(&slots.pocket, &slots.geometry),
        }
    }

    /// Run the full Phase 2 forward pass for one example.
    pub(crate) fn forward_example(&self, example: &MolecularExample) -> ResearchForward {
        let encodings = self.encode_example(example);
        let slots = self.decompose_modalities(&encodings);
        let interactions = self.interact_modalities(&slots);
        let probes = self.probes.forward(
            &encodings.topology,
            &encodings.geometry,
            &encodings.pocket,
            &slots.topology,
            &slots.geometry,
            &slots.pocket,
        );
        let generation_state = self.build_generation_state(example, &slots, &interactions);
        let decoded = self.ligand_decoder.decode(&generation_state);

        ResearchForward {
            encodings,
            slots,
            interactions,
            probes,
            generation: GenerationForward {
                state: generation_state,
                decoded,
            },
        }
    }

    /// Collate and encode a small batch.
    #[allow(dead_code)]
    pub(crate) fn encode_batch(
        &self,
        examples: &[MolecularExample],
    ) -> (MolecularBatch, Vec<EncodedModalities>) {
        let batch = MolecularBatch::collate(examples);
        let outputs = examples
            .iter()
            .map(|example| self.encode_example(example))
            .collect();
        (batch, outputs)
    }

    /// Collate and run the full Phase 2 forward pass on a small batch.
    pub(crate) fn forward_batch(
        &self,
        examples: &[MolecularExample],
    ) -> (MolecularBatch, Vec<ResearchForward>) {
        let batch = MolecularBatch::collate(examples);
        let outputs = examples
            .iter()
            .map(|example| self.forward_example(example))
            .collect();
        (batch, outputs)
    }

    fn build_generation_state(
        &self,
        example: &MolecularExample,
        slots: &DecomposedModalities,
        interactions: &CrossModalInteractions,
    ) -> ConditionedGenerationState {
        let num_atoms = example.decoder_supervision.corrupted_atom_types.size()[0];
        let device = example.decoder_supervision.corrupted_atom_types.device();

        ConditionedGenerationState {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            partial_ligand: PartialLigandState {
                atom_types: example
                    .decoder_supervision
                    .corrupted_atom_types
                    .shallow_clone(),
                coords: example.decoder_supervision.noisy_coords.shallow_clone(),
                atom_mask: Tensor::ones([num_atoms], (Kind::Float, device)),
                step_index: 0,
            },
            topology_context: merge_slot_contexts(
                &slots.topology.slots,
                &[&interactions.topo_from_geo, &interactions.topo_from_pocket],
            ),
            geometry_context: merge_slot_contexts(
                &slots.geometry.slots,
                &[&interactions.geo_from_topo, &interactions.geo_from_pocket],
            ),
            pocket_context: merge_slot_contexts(
                &slots.pocket.slots,
                &[
                    &interactions.pocket_from_topo,
                    &interactions.pocket_from_geo,
                ],
            ),
        }
    }
}

fn merge_slot_contexts(base_slots: &Tensor, updates: &[&CrossAttentionOutput]) -> Tensor {
    let mut merged = base_slots.shallow_clone();
    for update in updates {
        merged = &merged + &update.attended_tokens;
    }
    merged
}
