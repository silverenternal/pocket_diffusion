//! Phase 2 system wiring for separate encoders, slots, controlled interaction, and probes.

use tch::{nn, Kind, Tensor};

use super::{
    BatchedCrossAttentionOutput, BatchedModalityEncoding, BatchedSlotEncoding,
    ConditionedGenerationState, ConditionedLigandDecoder, CrossAttentionOutput, DecoderOutput,
    Encoder, GatedCrossAttention, GenerationRolloutRecord, GenerationStepRecord,
    GeometryEncoderImpl, ModalityEncoding, ModularLigandDecoder, PartialLigandState,
    PocketEncoderImpl, ProbeOutputs, SemanticProbeHeads, SlotDecomposer, SlotEncoding,
    SoftSlotDecomposer, TopologyEncoderImpl,
};
use crate::{
    config::{GenerationRolloutMode, GenerationTargetConfig, ResearchConfig},
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

/// Batched output bundle for the separate modality encoders.
#[derive(Debug, Clone)]
pub(crate) struct BatchedEncodedModalities {
    /// Topology encoding.
    pub topology: BatchedModalityEncoding,
    /// Geometry encoding.
    pub geometry: BatchedModalityEncoding,
    /// Pocket/context encoding.
    pub pocket: BatchedModalityEncoding,
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

/// Batched slot-decomposed outputs for each modality.
#[derive(Debug, Clone)]
pub(crate) struct BatchedDecomposedModalities {
    /// Topology slots.
    pub topology: BatchedSlotEncoding,
    /// Geometry slots.
    pub geometry: BatchedSlotEncoding,
    /// Pocket/context slots.
    pub pocket: BatchedSlotEncoding,
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

/// Batched directed gated cross-modality interactions.
#[derive(Debug, Clone)]
pub(crate) struct BatchedCrossModalInteractions {
    /// Topology receiving information from geometry.
    pub topo_from_geo: BatchedCrossAttentionOutput,
    /// Topology receiving information from pocket context.
    pub topo_from_pocket: BatchedCrossAttentionOutput,
    /// Geometry receiving information from topology.
    pub geo_from_topo: BatchedCrossAttentionOutput,
    /// Geometry receiving information from pocket context.
    pub geo_from_pocket: BatchedCrossAttentionOutput,
    /// Pocket receiving information from topology.
    pub pocket_from_topo: BatchedCrossAttentionOutput,
    /// Pocket receiving information from geometry.
    pub pocket_from_geo: BatchedCrossAttentionOutput,
}

/// Decoder-facing generation bundle produced by the modular backbone.
#[derive(Debug, Clone)]
pub(crate) struct GenerationForward {
    /// Explicit decoder input state with separated modality conditioning.
    pub state: ConditionedGenerationState,
    /// Decoder output for the current ligand draft.
    pub decoded: DecoderOutput,
    /// Iterative rollout trace aligned with the active generation semantics.
    pub rollout: GenerationRolloutRecord,
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
    generation_target: GenerationTargetConfig,
    geometry_attention_bias_scale: f64,
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
        let topo_from_geo = GatedCrossAttention::new(
            &(vs / "topo_from_geo"),
            config.model.hidden_dim,
            config.model.interaction_mode,
            config.model.interaction_ff_multiplier,
            config.model.interaction_tuning.clone(),
        );
        let topo_from_pocket = GatedCrossAttention::new(
            &(vs / "topo_from_pocket"),
            config.model.hidden_dim,
            config.model.interaction_mode,
            config.model.interaction_ff_multiplier,
            config.model.interaction_tuning.clone(),
        );
        let geo_from_topo = GatedCrossAttention::new(
            &(vs / "geo_from_topo"),
            config.model.hidden_dim,
            config.model.interaction_mode,
            config.model.interaction_ff_multiplier,
            config.model.interaction_tuning.clone(),
        );
        let geo_from_pocket = GatedCrossAttention::new(
            &(vs / "geo_from_pocket"),
            config.model.hidden_dim,
            config.model.interaction_mode,
            config.model.interaction_ff_multiplier,
            config.model.interaction_tuning.clone(),
        );
        let pocket_from_topo = GatedCrossAttention::new(
            &(vs / "pocket_from_topo"),
            config.model.hidden_dim,
            config.model.interaction_mode,
            config.model.interaction_ff_multiplier,
            config.model.interaction_tuning.clone(),
        );
        let pocket_from_geo = GatedCrossAttention::new(
            &(vs / "pocket_from_geo"),
            config.model.hidden_dim,
            config.model.interaction_mode,
            config.model.interaction_ff_multiplier,
            config.model.interaction_tuning.clone(),
        );
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
            generation_target: config.data.generation_target.clone(),
            geometry_attention_bias_scale: config
                .model
                .interaction_tuning
                .geometry_attention_bias_scale,
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

    /// Run the three modality encoders over already-collated padded tensors.
    pub(crate) fn encode_batch_inputs(&self, batch: &MolecularBatch) -> BatchedEncodedModalities {
        let inputs = &batch.encoder_inputs;
        BatchedEncodedModalities {
            topology: self.topo_encoder.encode_batch(
                &inputs.atom_types,
                &inputs.adjacency,
                &inputs.ligand_mask,
            ),
            geometry: self.geo_encoder.encode_batch(
                &inputs.ligand_coords,
                &inputs.pairwise_distances,
                &inputs.ligand_mask,
            ),
            pocket: self.pocket_encoder.encode_batch(
                &inputs.pocket_atom_features,
                &inputs.pocket_coords,
                &inputs.pocket_pooled_features,
                &inputs.pocket_mask,
            ),
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

    /// Decompose batched modality encodings into learned slots.
    pub(crate) fn decompose_batched_modalities(
        &self,
        encodings: &BatchedEncodedModalities,
    ) -> BatchedDecomposedModalities {
        BatchedDecomposedModalities {
            topology: self.topo_slots.decompose_batch(&encodings.topology),
            geometry: self.geo_slots.decompose_batch(&encodings.geometry),
            pocket: self.pocket_slots.decompose_batch(&encodings.pocket),
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

    /// Apply all directed cross-modality interactions over slot batches.
    pub(crate) fn interact_batched_modalities(
        &self,
        batch: &MolecularBatch,
        slots: &BatchedDecomposedModalities,
    ) -> BatchedCrossModalInteractions {
        let geo_pocket_bias = ligand_pocket_slot_attention_bias(
            &batch.encoder_inputs.ligand_coords,
            &batch.encoder_inputs.ligand_mask,
            &batch.encoder_inputs.pocket_coords,
            &batch.encoder_inputs.pocket_mask,
            slots.geometry.slots.size()[1],
            slots.pocket.slots.size()[1],
        ) * self.geometry_attention_bias_scale;
        let pocket_geo_bias = geo_pocket_bias.transpose(1, 2);
        BatchedCrossModalInteractions {
            topo_from_geo: self
                .topo_from_geo
                .forward_batch(&slots.topology, &slots.geometry, None),
            topo_from_pocket: self.topo_from_pocket.forward_batch(
                &slots.topology,
                &slots.pocket,
                None,
            ),
            geo_from_topo: self
                .geo_from_topo
                .forward_batch(&slots.geometry, &slots.topology, None),
            geo_from_pocket: self.geo_from_pocket.forward_batch(
                &slots.geometry,
                &slots.pocket,
                Some(&geo_pocket_bias),
            ),
            pocket_from_topo: self.pocket_from_topo.forward_batch(
                &slots.pocket,
                &slots.topology,
                None,
            ),
            pocket_from_geo: self.pocket_from_geo.forward_batch(
                &slots.pocket,
                &slots.geometry,
                Some(&pocket_geo_bias),
            ),
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
        let rollout = self.rollout_generation(example, &generation_state);

        ResearchForward {
            encodings,
            slots,
            interactions,
            probes,
            generation: GenerationForward {
                state: generation_state,
                decoded,
                rollout,
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
        let batched = self.encode_batch_inputs(&batch);
        let outputs = examples
            .iter()
            .enumerate()
            .map(|(index, example)| slice_encoded_modalities(&batched, index as i64, example))
            .collect();
        (batch, outputs)
    }

    /// Collate and run the full Phase 2 forward pass on a small batch.
    pub(crate) fn forward_batch(
        &self,
        examples: &[MolecularExample],
    ) -> (MolecularBatch, Vec<ResearchForward>) {
        let batch = MolecularBatch::collate(examples);
        let batched_encodings = self.encode_batch_inputs(&batch);
        let batched_slots = self.decompose_batched_modalities(&batched_encodings);
        let batched_interactions = self.interact_batched_modalities(&batch, &batched_slots);
        let outputs = examples
            .iter()
            .enumerate()
            .map(|(index, example)| {
                self.forward_from_batched_parts(
                    example,
                    index as i64,
                    &batched_encodings,
                    &batched_slots,
                    &batched_interactions,
                )
            })
            .collect();
        (batch, outputs)
    }

    fn forward_from_batched_parts(
        &self,
        example: &MolecularExample,
        batch_index: i64,
        encodings: &BatchedEncodedModalities,
        slots: &BatchedDecomposedModalities,
        interactions: &BatchedCrossModalInteractions,
    ) -> ResearchForward {
        let encodings = slice_encoded_modalities(encodings, batch_index, example);
        let slots = slice_decomposed_modalities(slots, batch_index, example);
        let interactions = slice_cross_modal_interactions(interactions, batch_index);
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
        let rollout = self.rollout_generation(example, &generation_state);

        ResearchForward {
            encodings,
            slots,
            interactions,
            probes,
            generation: GenerationForward {
                state: generation_state,
                decoded,
                rollout,
            },
        }
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

    fn rollout_generation(
        &self,
        example: &MolecularExample,
        initial_state: &ConditionedGenerationState,
    ) -> GenerationRolloutRecord {
        let mut state = initial_state.clone();
        let mut steps = Vec::with_capacity(self.generation_target.rollout_steps);
        let mut stopped_early = false;
        let mut stable_steps = 0_usize;
        let mut previous_coord_delta: Option<Tensor> = None;
        let mut previous_atom_logits: Option<Tensor> = None;

        for step_index in 0..self.generation_target.rollout_steps {
            state.partial_ligand.step_index = step_index as i64;
            let decoded = self.ligand_decoder.decode(&state);
            let stop_probability = decoded
                .stop_logit
                .sigmoid()
                .mean(Kind::Float)
                .double_value(&[]);
            let (next_atom_types, atom_change_fraction, updated_atom_logits) = self
                .next_atom_state(
                    &state.partial_ligand.atom_types,
                    &decoded.atom_type_logits,
                    previous_atom_logits.as_ref(),
                    step_index,
                );
            let (next_coords, mean_displacement, coordinate_step_scale, updated_coord_delta) = self
                .next_coordinate_state(
                    example,
                    &state.partial_ligand.coords,
                    &decoded.coordinate_deltas,
                    step_index,
                    previous_coord_delta.as_ref(),
                );
            previous_atom_logits = updated_atom_logits;
            previous_coord_delta = updated_coord_delta;
            let stable_now = mean_displacement <= self.generation_target.stop_delta_threshold
                && atom_change_fraction <= self.generation_target.stop_delta_threshold;
            stable_steps = if stable_now { stable_steps + 1 } else { 0 };
            let stop_ready = step_index + 1 >= self.generation_target.min_rollout_steps;
            let should_stop = stop_ready
                && match self.generation_target.rollout_mode {
                    GenerationRolloutMode::Lightweight => {
                        stop_probability >= self.generation_target.stop_probability_threshold
                    }
                    GenerationRolloutMode::MomentumRefine => {
                        stop_probability >= self.generation_target.stop_probability_threshold
                            || stable_steps >= self.generation_target.stop_patience
                    }
                };

            steps.push(GenerationStepRecord {
                step_index,
                stop_probability,
                stopped: should_stop,
                atom_types: tensor_to_i64_vec(&next_atom_types),
                coords: tensor_to_coords(&next_coords),
                mean_displacement,
                atom_change_fraction,
                coordinate_step_scale,
            });

            state.partial_ligand.atom_types = next_atom_types;
            state.partial_ligand.coords = next_coords;

            if should_stop {
                stopped_early = true;
                break;
            }
        }

        if steps.is_empty() {
            steps.push(GenerationStepRecord {
                step_index: 0,
                stop_probability: 0.0,
                stopped: false,
                atom_types: tensor_to_i64_vec(&example.decoder_supervision.corrupted_atom_types),
                coords: tensor_to_coords(&example.decoder_supervision.noisy_coords),
                mean_displacement: 0.0,
                atom_change_fraction: 0.0,
                coordinate_step_scale: self.generation_target.coordinate_step_scale,
            });
        }

        GenerationRolloutRecord {
            example_id: initial_state.example_id.clone(),
            protein_id: initial_state.protein_id.clone(),
            configured_steps: self.generation_target.rollout_steps,
            executed_steps: steps.len(),
            stopped_early,
            steps,
        }
    }

    fn next_atom_state(
        &self,
        current_atom_types: &Tensor,
        atom_type_logits: &Tensor,
        previous_atom_logits: Option<&Tensor>,
        step_index: usize,
    ) -> (Tensor, f64, Option<Tensor>) {
        if atom_type_logits.numel() == 0 {
            return (
                current_atom_types.shallow_clone(),
                0.0,
                previous_atom_logits.map(Tensor::shallow_clone),
            );
        }

        let committed_logits = match self.generation_target.rollout_mode {
            GenerationRolloutMode::Lightweight => atom_type_logits.shallow_clone(),
            GenerationRolloutMode::MomentumRefine => {
                let blended = if let Some(previous) = previous_atom_logits {
                    previous * self.generation_target.atom_momentum
                        + atom_type_logits * (1.0 - self.generation_target.atom_momentum)
                } else {
                    atom_type_logits.shallow_clone()
                };
                blended / self.generation_target.atom_commit_temperature
            }
        };
        let next_atom_types = if self.generation_target.sampling_temperature > 0.0 {
            sample_atom_types(
                &committed_logits,
                self.generation_target.sampling_temperature,
                self.generation_target.sampling_top_k,
                self.generation_target.sampling_top_p,
                self.generation_target.sampling_seed,
                step_index,
            )
        } else {
            committed_logits.argmax(-1, false)
        };
        let atom_change_fraction = atom_change_fraction(current_atom_types, &next_atom_types);
        (
            next_atom_types,
            atom_change_fraction,
            Some(committed_logits),
        )
    }

    fn next_coordinate_state(
        &self,
        example: &MolecularExample,
        current_coords: &Tensor,
        coordinate_deltas: &Tensor,
        step_index: usize,
        previous_coord_delta: Option<&Tensor>,
    ) -> (Tensor, f64, f64, Option<Tensor>) {
        if coordinate_deltas.numel() == 0 {
            return (
                current_coords.shallow_clone(),
                0.0,
                self.generation_target.coordinate_step_scale,
                previous_coord_delta.map(Tensor::shallow_clone),
            );
        }

        let normalized_delta = clip_coordinate_delta_norm(
            coordinate_deltas,
            self.generation_target.max_coordinate_delta_norm,
        );
        let effective_delta = match self.generation_target.rollout_mode {
            GenerationRolloutMode::Lightweight => normalized_delta,
            GenerationRolloutMode::MomentumRefine => {
                if let Some(previous) = previous_coord_delta {
                    previous * self.generation_target.coordinate_momentum
                        + normalized_delta * (1.0 - self.generation_target.coordinate_momentum)
                } else {
                    normalized_delta
                }
            }
        };
        let anneal = match self.generation_target.rollout_mode {
            GenerationRolloutMode::Lightweight => 1.0,
            GenerationRolloutMode::MomentumRefine => {
                let fraction =
                    step_index as f64 / self.generation_target.rollout_steps.max(1) as f64;
                (1.0 - 0.35 * fraction).max(0.5)
            }
        };
        let coordinate_step_scale = self.generation_target.coordinate_step_scale * anneal;
        let scaled_delta = &effective_delta * coordinate_step_scale;
        let pocket_guidance = pocket_guidance_delta(
            current_coords,
            &example.pocket.coords,
            coordinate_step_scale,
            step_index,
            self.generation_target.rollout_steps,
        ) * self.generation_target.pocket_guidance_scale;
        let sampling_noise = deterministic_coordinate_noise(
            current_coords,
            self.generation_target.coordinate_sampling_noise_std,
            self.generation_target.sampling_seed,
            step_index,
        );
        let effective_update = &scaled_delta + &pocket_guidance + &sampling_noise;
        let unconstrained_next = current_coords + &effective_update;
        let next_coords = constrain_to_pocket_envelope(
            &unconstrained_next,
            &example.pocket.coords,
            self.generation_target.pocket_guidance_scale,
        );
        let realized_update = &next_coords - current_coords;
        let mean_displacement = per_atom_displacement_mean(&realized_update);
        (
            next_coords,
            mean_displacement,
            coordinate_step_scale,
            Some((&realized_update / coordinate_step_scale.max(1e-6)).detach()),
        )
    }
}

fn sample_atom_types(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
    step_index: usize,
) -> Tensor {
    let shape = logits.size();
    let atom_count = shape.first().copied().unwrap_or(0).max(0) as usize;
    let vocab = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    if atom_count == 0 || vocab == 0 {
        return logits.argmax(-1, false);
    }

    let mut sampled = Vec::with_capacity(atom_count);
    for atom_ix in 0..atom_count {
        let mut probabilities = (0..vocab)
            .map(|token_ix| {
                let value =
                    logits.double_value(&[atom_ix as i64, token_ix as i64]) / temperature.max(1e-6);
                (token_ix as i64, value)
            })
            .collect::<Vec<_>>();
        probabilities.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = if top_k == 0 {
            probabilities.len()
        } else {
            top_k.min(probabilities.len())
        };
        probabilities.truncate(keep.max(1));

        let max_logit = probabilities
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weighted = probabilities
            .into_iter()
            .map(|(token_ix, value)| (token_ix, (value - max_logit).exp()))
            .collect::<Vec<_>>();
        let total = weighted
            .iter()
            .map(|(_, weight)| *weight)
            .sum::<f64>()
            .max(1e-12);
        for (_, weight) in &mut weighted {
            *weight /= total;
        }
        weighted.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cumulative = 0.0;
        let mut filtered = Vec::new();
        for (token_ix, probability) in weighted {
            cumulative += probability;
            filtered.push((token_ix, probability));
            if cumulative >= top_p {
                break;
            }
        }
        let filtered_total = filtered
            .iter()
            .map(|(_, probability)| *probability)
            .sum::<f64>()
            .max(1e-12);
        let draw = deterministic_unit(seed, step_index, atom_ix, 0);
        let mut running = 0.0;
        let mut picked = filtered.last().map(|(token_ix, _)| *token_ix).unwrap_or(0);
        for (token_ix, probability) in filtered {
            running += probability / filtered_total;
            if draw <= running {
                picked = token_ix;
                break;
            }
        }
        sampled.push(picked);
    }
    Tensor::from_slice(&sampled)
        .to_kind(Kind::Int64)
        .to_device(logits.device())
}

fn deterministic_coordinate_noise(
    coords: &Tensor,
    std: f64,
    seed: u64,
    step_index: usize,
) -> Tensor {
    if std <= 0.0 || coords.numel() == 0 {
        return Tensor::zeros_like(coords);
    }
    let shape = coords.size();
    let atom_count = shape.first().copied().unwrap_or(0).max(0) as usize;
    let mut values = Vec::with_capacity(atom_count * 3);
    for atom_ix in 0..atom_count {
        for axis in 0..3 {
            let centered = deterministic_unit(seed, step_index, atom_ix, axis + 1) * 2.0 - 1.0;
            values.push((centered * std) as f32);
        }
    }
    Tensor::from_slice(&values)
        .reshape([atom_count as i64, 3])
        .to_device(coords.device())
}

fn deterministic_unit(seed: u64, step_index: usize, atom_ix: usize, stream: usize) -> f64 {
    let mut value = seed
        ^ ((step_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ ((atom_ix as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        ^ ((stream as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value as f64) / (u64::MAX as f64)
}

fn merge_slot_contexts(base_slots: &Tensor, updates: &[&CrossAttentionOutput]) -> Tensor {
    let mut merged = base_slots.shallow_clone();
    for update in updates {
        merged = &merged + &update.attended_tokens;
    }
    merged
}

fn slice_encoded_modalities(
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

fn slice_decomposed_modalities(
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
        reconstructed_tokens: batched
            .reconstructed_tokens
            .get(batch_index)
            .narrow(0, 0, token_count)
            .shallow_clone(),
    }
}

fn slice_cross_modal_interactions(
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
        attended_tokens: batched.attended_tokens.get(batch_index),
        attention_weights: batched.attention_weights.get(batch_index),
    }
}

fn ligand_pocket_slot_attention_bias(
    ligand_coords: &Tensor,
    ligand_mask: &Tensor,
    pocket_coords: &Tensor,
    pocket_mask: &Tensor,
    ligand_slots: i64,
    pocket_slots: i64,
) -> Tensor {
    let batch = ligand_coords.size()[0];
    if ligand_slots == 0 || pocket_slots == 0 {
        return Tensor::zeros(
            [batch, ligand_slots, pocket_slots],
            (Kind::Float, ligand_coords.device()),
        );
    }
    let pair_mask = ligand_mask.unsqueeze(2) * pocket_mask.unsqueeze(1);
    let diffs = ligand_coords.unsqueeze(2) - pocket_coords.unsqueeze(1);
    let distances = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([3].as_slice(), false, Kind::Float)
        .sqrt();
    let valid_distances = distances.shallow_clone() * &pair_mask;
    let denom = pair_mask
        .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        .clamp_min(1.0);
    let mean_distance =
        valid_distances.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float) / denom;
    let contact_fraction = distances.lt(4.5).to_kind(Kind::Float) * &pair_mask;
    let contact_fraction = contact_fraction.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        / pair_mask
            .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
            .clamp_min(1.0);
    let scalar_bias = contact_fraction - mean_distance.clamp_max(12.0) / 12.0;
    scalar_bias
        .reshape([batch, 1, 1])
        .expand([batch, ligand_slots, pocket_slots], true)
}

fn tensor_to_i64_vec(tensor: &Tensor) -> Vec<i64> {
    let len = tensor.size().first().copied().unwrap_or(0).max(0) as usize;
    (0..len)
        .map(|index| tensor.int64_value(&[index as i64]))
        .collect()
}

fn tensor_to_coords(tensor: &Tensor) -> Vec<[f32; 3]> {
    let rows = tensor.size().first().copied().unwrap_or(0).max(0) as usize;
    (0..rows)
        .map(|row| {
            [
                tensor.double_value(&[row as i64, 0]) as f32,
                tensor.double_value(&[row as i64, 1]) as f32,
                tensor.double_value(&[row as i64, 2]) as f32,
            ]
        })
        .collect()
}

fn clip_coordinate_delta_norm(delta: &Tensor, max_norm: f64) -> Tensor {
    let norms = delta
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .sqrt();
    let scale = norms.clamp_min(max_norm).reciprocal() * max_norm;
    let mask = norms.gt(max_norm).to_kind(Kind::Float);
    let ones = Tensor::ones_like(&mask);
    let safe_scale = &mask * scale + (&ones - &mask);
    delta * safe_scale
}

fn per_atom_displacement_mean(delta: &Tensor) -> f64 {
    if delta.numel() == 0 {
        return 0.0;
    }
    delta
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
}

fn atom_change_fraction(current: &Tensor, next: &Tensor) -> f64 {
    if current.numel() == 0 || next.numel() == 0 {
        return 0.0;
    }
    current
        .ne_tensor(next)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn pocket_guidance_delta(
    current_coords: &Tensor,
    pocket_coords: &Tensor,
    coordinate_step_scale: f64,
    step_index: usize,
    rollout_steps: usize,
) -> Tensor {
    if current_coords.numel() == 0 || pocket_coords.numel() == 0 {
        return Tensor::zeros_like(current_coords);
    }

    let ligand_centroid = current_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let centroid_offset = &pocket_centroid - &ligand_centroid;
    let pocket_radius = pocket_radius_from_coords(pocket_coords, &pocket_centroid);
    let centroid_distance = centroid_offset
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt()
        .double_value(&[]);
    let overflow = (centroid_distance - pocket_radius * 0.85).max(0.0);
    if overflow <= 0.0 {
        return Tensor::zeros_like(current_coords);
    }

    let progress = (step_index + 1) as f64 / rollout_steps.max(1) as f64;
    let guidance_scale = (0.08 + 0.22 * progress + overflow.min(pocket_radius.max(1.0))).min(0.45)
        * coordinate_step_scale;
    centroid_offset.unsqueeze(0).expand_as(current_coords) * guidance_scale
}

fn constrain_to_pocket_envelope(
    coords: &Tensor,
    pocket_coords: &Tensor,
    pocket_guidance_scale: f64,
) -> Tensor {
    if coords.numel() == 0 || pocket_coords.numel() == 0 || pocket_guidance_scale <= 0.0 {
        return coords.shallow_clone();
    }

    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_radius = pocket_radius_from_coords(pocket_coords, &pocket_centroid).max(1.0);
    let max_radius = pocket_radius + 1.5;
    let offsets = coords - pocket_centroid.unsqueeze(0);
    let radii = offsets
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .sqrt()
        .clamp_min(1e-6);
    let scale = radii.clamp_max(max_radius) / &radii;
    let projected = pocket_centroid.unsqueeze(0) + offsets * scale;
    let outside = radii.gt(max_radius).to_kind(Kind::Float);
    let ones = Tensor::ones_like(&outside);
    &outside * projected + (&ones - &outside) * coords
}

fn pocket_radius_from_coords(pocket_coords: &Tensor, pocket_centroid: &Tensor) -> f64 {
    if pocket_coords.numel() == 0 {
        return 0.0;
    }
    (pocket_coords - pocket_centroid.unsqueeze(0))
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::ResearchConfig, data::synthetic_phase1_examples};
    use tch::{nn, Device};

    #[test]
    fn batched_backbone_matches_single_example_backbone_shapes() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = synthetic_phase1_examples()
            .into_iter()
            .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
            .collect::<Vec<_>>();

        let single = system.forward_example(&examples[0]);
        let (_, batched) = system.forward_batch(&examples[..2]);
        let from_batch = &batched[0];

        assert_eq!(
            single.encodings.topology.token_embeddings.size(),
            from_batch.encodings.topology.token_embeddings.size()
        );
        assert_eq!(
            single.slots.geometry.slots.size(),
            from_batch.slots.geometry.slots.size()
        );
        assert_eq!(
            single.interactions.geo_from_pocket.attention_weights.size(),
            from_batch
                .interactions
                .geo_from_pocket
                .attention_weights
                .size()
        );
        assert_eq!(
            from_batch.generation.decoded.atom_type_logits.size()[0],
            examples[0].topology.atom_types.size()[0]
        );
    }

    #[test]
    fn stochastic_rollout_sampling_is_reproducible_for_fixed_seed() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.sampling_temperature = 1.0;
        config.data.generation_target.sampling_top_k = 4;
        config.data.generation_target.sampling_top_p = 0.9;
        config.data.generation_target.coordinate_sampling_noise_std = 0.02;
        config.data.generation_target.sampling_seed = 12345;

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let left = system.forward_example(&example).generation.rollout;
        let right = system.forward_example(&example).generation.rollout;

        assert_eq!(left.steps.len(), right.steps.len());
        assert_eq!(
            left.steps.last().unwrap().atom_types,
            right.steps.last().unwrap().atom_types
        );
        assert_eq!(
            left.steps.last().unwrap().coords,
            right.steps.last().unwrap().coords
        );
    }
}
