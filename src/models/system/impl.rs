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
    /// Geometry-only flow-matching velocity head.
    pub flow_matching_head: GeometryFlowMatchingHead,
    generation_target: GenerationTargetConfig,
    generation_backend_family: GenerationBackendFamilyConfig,
    flow_matching_config: crate::config::FlowMatchingConfig,
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
        let flow_matching_head =
            GeometryFlowMatchingHead::new(&(vs / "flow_matching_head"), config.model.hidden_dim);

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
            flow_matching_head,
            generation_target: config.data.generation_target.clone(),
            generation_backend_family: resolved_generation_backend_family(config),
            flow_matching_config: config.generation_method.flow_matching.clone(),
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
        let flow_matching =
            self.flow_matching_training_record(example, &generation_state, &interactions);
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
                flow_matching,
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
        let flow_matching =
            self.flow_matching_training_record(example, &generation_state, &interactions);
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
                flow_matching,
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

    fn flow_matching_training_record(
        &self,
        example: &MolecularExample,
        generation_state: &ConditionedGenerationState,
        interactions: &CrossModalInteractions,
    ) -> Option<FlowMatchingTrainingRecord> {
        if self.generation_backend_family != GenerationBackendFamilyConfig::FlowMatching {
            return None;
        }
        let x1 = example.decoder_supervision.target_coords.shallow_clone();
        let x0 = flow_matching_x0(
            example,
            self.flow_matching_config.noise_scale,
            self.flow_matching_config.use_corrupted_x0,
        );
        if x1.size() != x0.size() {
            return None;
        }
        let t = flow_matching_t_from_example(example);
        let xt = &x0 * (1.0 - t) + &x1 * t;
        let flow_state = FlowState {
            coords: xt.shallow_clone(),
            x0_coords: x0.shallow_clone(),
            target_coords: Some(x1.shallow_clone()),
            t,
        };
        let conditioning =
            flow_conditioning_state(generation_state, gate_summary_from_interactions(interactions));
        let predicted_velocity = self
            .flow_matching_head
            .predict_velocity(&flow_state, &conditioning)
            .ok()?
            .velocity;
        Some(FlowMatchingTrainingRecord {
            predicted_velocity,
            target_velocity: x1 - x0,
            sampled_coords: xt,
            t,
            atom_mask: generation_state.partial_ligand.atom_mask.shallow_clone(),
        })
    }

    fn rollout_generation(
        &self,
        example: &MolecularExample,
        initial_state: &ConditionedGenerationState,
    ) -> GenerationRolloutRecord {
        if self.generation_backend_family == GenerationBackendFamilyConfig::FlowMatching {
            return self.rollout_flow_matching(example, initial_state);
        }
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

    fn rollout_flow_matching(
        &self,
        example: &MolecularExample,
        initial_state: &ConditionedGenerationState,
    ) -> GenerationRolloutRecord {
        let configured_steps = self.flow_matching_config.steps.max(1);
        let mut coords = flow_matching_x0(
            example,
            self.flow_matching_config.noise_scale,
            self.flow_matching_config.use_corrupted_x0,
        );
        let atom_types = tensor_to_i64_vec(&initial_state.partial_ligand.atom_types);
        let conditioning =
            flow_conditioning_state(initial_state, GenerationGateSummary::default());
        let x0_coords = coords.shallow_clone();
        let dt = 1.0 / configured_steps as f64;
        let mut steps = Vec::with_capacity(configured_steps);
        let mut previous = coords.shallow_clone();

        for step_index in 0..configured_steps {
            let t = step_index as f64 / configured_steps as f64;
            let flow_state = FlowState {
                coords: coords.shallow_clone(),
                x0_coords: x0_coords.shallow_clone(),
                target_coords: None,
                t,
            };
            let velocity_1 = self
                .flow_matching_head
                .predict_velocity(&flow_state, &conditioning)
                .map(|field| field.velocity)
                .unwrap_or_else(|_| Tensor::zeros_like(&coords));
            let update = match self.flow_matching_config.integration_method {
                FlowMatchingIntegrationMethod::Euler => velocity_1 * dt,
                FlowMatchingIntegrationMethod::Heun => {
                    let predictor = &coords + &(velocity_1.shallow_clone() * dt);
                    let predictor_state = FlowState {
                        coords: predictor,
                        x0_coords: x0_coords.shallow_clone(),
                        target_coords: None,
                        t: (t + dt).min(1.0),
                    };
                    let velocity_2 = self
                        .flow_matching_head
                        .predict_velocity(&predictor_state, &conditioning)
                        .map(|field| field.velocity)
                        .unwrap_or_else(|_| Tensor::zeros_like(&coords));
                    (velocity_1 + velocity_2) * (0.5 * dt)
                }
            };
            coords += update;
            coords = constrain_to_pocket_envelope(
                &coords,
                &example.pocket.coords,
                self.generation_target.pocket_guidance_scale,
            );
            let delta = &coords - &previous;
            let mean_displacement = per_atom_displacement_mean(&delta);
            previous = coords.shallow_clone();
            steps.push(GenerationStepRecord {
                step_index,
                stop_probability: 0.0,
                stopped: false,
                atom_types: atom_types.clone(),
                coords: tensor_to_coords(&coords),
                mean_displacement,
                atom_change_fraction: 0.0,
                coordinate_step_scale: dt,
            });
        }

        GenerationRolloutRecord {
            example_id: initial_state.example_id.clone(),
            protein_id: initial_state.protein_id.clone(),
            configured_steps,
            executed_steps: steps.len(),
            stopped_early: false,
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
