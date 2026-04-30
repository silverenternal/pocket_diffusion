impl Phase1ResearchSystem {
    /// Run the full Phase 2 forward pass for one example.
    pub(crate) fn forward_example(&self, example: &MolecularExample) -> ResearchForward {
        self.forward_example_with_interaction_context(
            example,
            InteractionExecutionContext::default(),
        )
    }

    /// Run only optimizer-facing forward phases and skip sampled rollout construction.
    #[allow(dead_code)] // Used by boundary tests and future trainer runner split.
    pub(crate) fn forward_example_optimizer_record(
        &self,
        example: &MolecularExample,
    ) -> OptimizerForwardRecord {
        self.forward_example_optimizer_record_with_interaction_context(
            example,
            InteractionExecutionContext::default(),
        )
    }

    pub(crate) fn forward_example_with_interaction_context(
        &self,
        example: &MolecularExample,
        interaction_execution_context: InteractionExecutionContext,
    ) -> ResearchForward {
        self.forward_example_with_interaction_context_and_rollout_diagnostics(
            example,
            interaction_execution_context,
            true,
        )
    }

    /// Run one or more bounded de novo initialization samples for a single pocket.
    pub(crate) fn forward_example_generation_samples(
        &self,
        example: &MolecularExample,
    ) -> Vec<ResearchForward> {
        self.forward_example_generation_samples_with_interaction_context_and_rollout_diagnostics(
            example,
            InteractionExecutionContext::default(),
            true,
        )
    }

    /// Run config-gated multi-sample generation forwards with explicit interaction context.
    pub(crate) fn forward_example_generation_samples_with_interaction_context_and_rollout_diagnostics(
        &self,
        example: &MolecularExample,
        interaction_execution_context: InteractionExecutionContext,
        build_rollout_diagnostics: bool,
    ) -> Vec<ResearchForward> {
        let sample_count = self.generation_initialization_sample_count();
        if sample_count == 1
            && !self.generation_target.multi_sample_initialization.enabled
        {
            return vec![self.forward_example_with_interaction_context_and_rollout_diagnostics(
                example,
                interaction_execution_context,
                build_rollout_diagnostics,
            )];
        }
        (0..sample_count)
            .map(|sample_index| {
                let sample_context = self.generation_sample_context(
                    interaction_execution_context.clone(),
                    sample_index,
                    sample_count,
                );
                self.forward_example_with_interaction_context_and_rollout_diagnostics(
                    example,
                    sample_context,
                    build_rollout_diagnostics,
                )
            })
            .collect()
    }

    pub(crate) fn generation_initialization_sample_count(&self) -> usize {
        if self.generation_mode == GenerationModeConfig::DeNovoInitialization {
            self.generation_target
                .multi_sample_initialization
                .effective_sample_count()
        } else {
            1
        }
    }

    fn generation_sample_context(
        &self,
        mut context: InteractionExecutionContext,
        sample_index: usize,
        sample_count: usize,
    ) -> InteractionExecutionContext {
        if self.generation_mode != GenerationModeConfig::DeNovoInitialization
            || !self.generation_target.multi_sample_initialization.enabled
        {
            return context;
        }
        let policy = &self.generation_target.multi_sample_initialization;
        let initializer_seed = self.generation_target.de_novo_initialization.seed;
        context.generation_sample_index = Some(sample_index);
        context.generation_sample_count = Some(sample_count);
        context.generation_sample_seed = Some(policy.derived_seed(initializer_seed, sample_index));
        context.generation_sample_seed_provenance =
            Some(policy.seed_provenance(initializer_seed, sample_index));
        context
    }

    fn forward_example_with_interaction_context_and_rollout_diagnostics(
        &self,
        example: &MolecularExample,
        interaction_execution_context: InteractionExecutionContext,
        build_rollout_diagnostics: bool,
    ) -> ResearchForward {
        let optimizer_record =
            self.forward_example_optimizer_record_with_interaction_context(
                example,
                interaction_execution_context,
            );
        self.forward_from_optimizer_record(example, optimizer_record, build_rollout_diagnostics)
    }

    fn forward_from_optimizer_record(
        &self,
        example: &MolecularExample,
        optimizer_record: OptimizerForwardRecord,
        build_rollout_diagnostics: bool,
    ) -> ResearchForward {
        let rollout_path_means =
            gate_summary_from_interaction_diagnostics(&optimizer_record.interaction_diagnostics);
        let rollout_training =
            self.rollout_training_record(example, &optimizer_record.state, &optimizer_record.decoded);
        let rollout = if build_rollout_diagnostics {
            no_grad(|| {
                self.rollout_generation(
                    example,
                    &optimizer_record.state,
                    rollout_path_means,
                    &optimizer_record.interaction_context,
                )
            })
        } else {
            self.skipped_rollout_diagnostics_record(
                &optimizer_record.state,
                rollout_path_means,
                &optimizer_record.interaction_context,
            )
        };
        ResearchForward {
            encodings: optimizer_record.encodings,
            slots: optimizer_record.slots,
            diagnostics: optimizer_record.diagnostics,
            interactions: optimizer_record.interactions,
            interaction_diagnostics: optimizer_record.interaction_diagnostics,
            sync_context: optimizer_record.sync_context,
            probes: optimizer_record.probes,
            generation: GenerationForward {
                generation_mode: optimizer_record.generation_mode,
                state: optimizer_record.state,
                decoded: optimizer_record.decoded,
                rollout,
                rollout_training,
                flow_matching: optimizer_record.flow_matching,
                pocket_priors: optimizer_record.pocket_priors,
            },
        }
    }

    fn rollout_training_record(
        &self,
        example: &MolecularExample,
        initial_state: &ConditionedGenerationState,
        initial_decoded: &DecoderOutput,
    ) -> RolloutTrainingRecord {
        let configured_steps = self.rollout_training_config.rollout_steps.clamp(1, 3);
        let detach_policy = self.rollout_training_config.detach_policy.as_str().to_string();
        let mode_allowed = self.rollout_training_config.allows_mode(self.generation_mode);
        if !self.rollout_training_config.enabled || !mode_allowed {
            let mut record = RolloutTrainingRecord::disabled(configured_steps, detach_policy);
            record.enabled = self.rollout_training_config.enabled;
            record.mode_allowed = mode_allowed;
            record.memory_control = if self.rollout_training_config.enabled {
                "skipped_by_generation_mode_gate".to_string()
            } else {
                "disabled".to_string()
            };
            return record;
        }

        let mut state = initial_state.clone();
        let mut steps = Vec::with_capacity(configured_steps);
        let mut previous_coord_delta: Option<Tensor> = None;
        for step_index in 0..configured_steps {
            state.partial_ligand.step_index = step_index as i64;
            let decoded = if step_index == 0 {
                initial_decoded.clone()
            } else {
                self.generator_stack.ligand_decoder.decode(&state)
            };
            let (next_coords, _mean_displacement, _coordinate_step_scale, updated_coord_delta) =
                self.next_coordinate_state(
                    example,
                    &state.partial_ligand.coords,
                    &decoded.coordinate_deltas,
                    step_index,
                    previous_coord_delta.as_ref(),
                );
            let detached_before_next_step = matches!(
                self.rollout_training_config.detach_policy,
                RolloutTrainingDetachPolicy::DetachBetweenSteps
            );
            steps.push(RolloutTrainingStepRecord {
                step_index,
                atom_type_logits: decoded.atom_type_logits.shallow_clone(),
                coords: next_coords.shallow_clone(),
                coordinate_deltas: &next_coords - &state.partial_ligand.coords,
                stop_logit: decoded.stop_logit.shallow_clone(),
                atom_mask: state.partial_ligand.atom_mask.shallow_clone(),
                detached_before_next_step,
            });
            state.partial_ligand.atom_types = confidence_gated_atom_type_commit(
                &decoded.atom_type_logits,
                &state.partial_ligand.atom_types,
            );
            state.partial_ligand.coords = if detached_before_next_step {
                next_coords.detach()
            } else {
                next_coords
            };
            previous_coord_delta = updated_coord_delta;
        }

        RolloutTrainingRecord {
            enabled: true,
            mode_allowed: true,
            configured_steps,
            executed_steps: steps.len(),
            detach_policy,
            memory_control: format!(
                "bounded_steps={configured_steps};max_batch_examples={};detach_policy={}",
                self.rollout_training_config.max_batch_examples,
                self.rollout_training_config.detach_policy.as_str()
            ),
            target_source: "generated_rollout_state".to_string(),
            steps,
        }
    }

    fn forward_example_optimizer_record_with_interaction_context(
        &self,
        example: &MolecularExample,
        interaction_execution_context: InteractionExecutionContext,
    ) -> OptimizerForwardRecord {
        let mut interaction_execution_context = interaction_execution_context;
        if self.generation_backend_family == GenerationBackendFamilyConfig::FlowMatching
            && interaction_execution_context.flow_t.is_none()
        {
            interaction_execution_context.flow_t =
                Some(flow_matching_t_from_example(example, Some(&interaction_execution_context)));
        }
        let mut conditioning_example_storage = None;
        let raw_encodings = if self.generation_mode == GenerationModeConfig::DeNovoInitialization {
            let (topology, geometry, pocket) =
                self.de_novo_conditioning_modalities(example, &interaction_execution_context);
            let mut conditioning_example = example.clone();
            conditioning_example.topology = topology.clone();
            conditioning_example.geometry = geometry.clone();
            conditioning_example.pocket = pocket.clone();
            conditioning_example_storage = Some(conditioning_example);
            EncodedModalities {
                topology: self.encoder_stack.topology_branch.encode(&topology),
                geometry: self.encoder_stack.geometry_branch.encode(&geometry),
                pocket: self.encoder_stack.pocket_branch.encode(&pocket),
            }
        } else {
            self.encode_example(example)
        };
        let conditioning_example = conditioning_example_storage.as_ref().unwrap_or(example);
        let encodings = self.apply_modality_focus_to_encodings(raw_encodings);
        let slots = self.decompose_modalities(&encodings);
        let mut sync_context = self.sync_context_from_example(conditioning_example, &slots);
        let diagnostics = SemanticDiagnosticsBundle::from_modalities(
            (&encodings.topology, &slots.topology),
            (&encodings.geometry, &slots.geometry),
            (&encodings.pocket, &slots.pocket),
        );
        let (interactions, mut interaction_diagnostics) = self
            .interaction_stack
            .block
            .forward_example_with_diagnostics_with_context(
                conditioning_example,
                &slots,
                interaction_execution_context.clone(),
            );
        attach_topology_pocket_pharmacophore_path_diagnostics(
            &mut interaction_diagnostics.topo_from_pocket,
            &mut interaction_diagnostics.pocket_from_topo,
            &conditioning_example.topology.chemistry_roles.role_vectors,
            &slots.topology.slot_weights,
            &conditioning_example.pocket.chemistry_roles.role_vectors,
            &slots.pocket.slot_weights,
            &interactions.topo_from_pocket.attention_weights,
            &interactions.pocket_from_topo.attention_weights,
        );
        let probes = self.apply_modality_focus_to_probes(self.probe_stack.probes.forward(
            &encodings.topology,
            &encodings.geometry,
            &encodings.pocket,
            &slots.topology,
            &slots.geometry,
            &slots.pocket,
        ));
        let pocket_priors = self.generator_stack.pocket_prior_head.forward(&encodings.pocket);
        let generation_state = self.build_generation_state(
            example,
            &slots,
            &interactions,
            &interaction_execution_context,
        );
        let decoded = self.generator_stack.ligand_decoder.decode(&generation_state);
        let flow_matching = self.flow_matching_training_record(
            example,
            &generation_state,
            &interactions,
            &interaction_execution_context,
        );
        sync_context.flow_t = flow_matching.as_ref().map(|record| record.t);

        OptimizerForwardRecord {
            encodings,
            slots,
            diagnostics,
            interactions,
            interaction_diagnostics,
            sync_context,
            probes,
            generation_mode: self.generation_mode,
            state: generation_state,
            decoded,
            flow_matching,
            pocket_priors,
            interaction_context: interaction_execution_context,
        }
    }

    /// Collate and encode a small batch.
    #[allow(dead_code)]
    pub(crate) fn encode_batch(
        &self,
        examples: &[MolecularExample],
    ) -> (MolecularBatch, Vec<EncodedModalities>) {
        let batch = MolecularBatch::collate(examples);
        if self.generation_mode == GenerationModeConfig::DeNovoInitialization {
            let outputs = examples
                .iter()
                .map(|example| self.encode_example(example))
                .collect();
            return (batch, outputs);
        }
        let batched = self.apply_batched_modality_focus_to_encodings(self.encode_batch_inputs(&batch));
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
        self.forward_batch_with_interaction_context(
            examples,
            InteractionExecutionContext::default(),
        )
    }

    pub(crate) fn forward_batch_with_interaction_context(
        &self,
        examples: &[MolecularExample],
        interaction_execution_context: InteractionExecutionContext,
    ) -> (MolecularBatch, Vec<ResearchForward>) {
        self.forward_batch_with_interaction_context_and_rollout_diagnostics(
            examples,
            interaction_execution_context,
            true,
        )
    }

    pub(crate) fn forward_batch_with_interaction_context_and_rollout_diagnostics(
        &self,
        examples: &[MolecularExample],
        interaction_execution_context: InteractionExecutionContext,
        build_rollout_diagnostics: bool,
    ) -> (MolecularBatch, Vec<ResearchForward>) {
        let batch = MolecularBatch::collate(examples);
        if self.generation_mode == GenerationModeConfig::DeNovoInitialization
            || (self.generation_backend_family == GenerationBackendFamilyConfig::FlowMatching
                && self.interaction_stack.block.uses_flow_time_conditioning())
        {
            let outputs = examples
                .iter()
                .map(|example| {
                    self.forward_example_with_interaction_context_and_rollout_diagnostics(
                        example,
                        interaction_execution_context.clone(),
                        build_rollout_diagnostics,
                    )
                })
                .collect();
            return (batch, outputs);
        }
        let batched_encodings =
            self.apply_batched_modality_focus_to_encodings(self.encode_batch_inputs(&batch));
        let batched_slots = self.decompose_batched_modalities(&batched_encodings);
        let geo_from_pocket_bias =
            self.batched_geo_from_pocket_attention_bias(&batch, &batched_slots);
        let pocket_from_geo_bias = geo_from_pocket_bias.values.transpose(1, 2).shallow_clone();
        let (batched_interactions, _batched_interaction_diagnostics) = self
            .interact_batched_modalities_with_diagnostics_with_context(
                &batch,
                &batched_slots,
                interaction_execution_context.clone(),
            );
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
                    &geo_from_pocket_bias.values,
                    &pocket_from_geo_bias,
                    geo_from_pocket_bias.chemistry_role_coverage,
                    &interaction_execution_context,
                    build_rollout_diagnostics,
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
        geo_from_pocket_bias: &Tensor,
        pocket_from_geo_bias: &Tensor,
        chemistry_role_coverage: f64,
        interaction_execution_context: &InteractionExecutionContext,
        build_rollout_diagnostics: bool,
    ) -> ResearchForward {
        let mut diagnostic_context = interaction_execution_context.clone();
        if self.generation_backend_family == GenerationBackendFamilyConfig::FlowMatching
            && diagnostic_context.flow_t.is_none()
        {
            diagnostic_context.flow_t =
                Some(flow_matching_t_from_example(example, Some(&diagnostic_context)));
        }
        let encodings = slice_encoded_modalities(encodings, batch_index, example);
        let slots = slice_decomposed_modalities(slots, batch_index, example);
        let interactions = slice_cross_modal_interactions(interactions, batch_index);
        let mut interaction_diagnostics =
            crate::models::interaction::CrossModalInteractionDiagnostics {
                topo_from_geo: interaction_path_diagnostics(
                    InteractionPath::TopologyFromGeometry,
                    &interactions.topo_from_geo,
                    self.interaction_stack.block.path_scale_for(
                        InteractionPath::TopologyFromGeometry,
                        &diagnostic_context,
                    ),
                    None,
                    &diagnostic_context,
                    InteractionDiagnosticProvenance::PerExample,
                ),
                topo_from_pocket: interaction_path_diagnostics(
                    InteractionPath::TopologyFromPocket,
                    &interactions.topo_from_pocket,
                    self.interaction_stack.block.path_scale_for(
                        InteractionPath::TopologyFromPocket,
                        &diagnostic_context,
                    ),
                    None,
                    &diagnostic_context,
                    InteractionDiagnosticProvenance::PerExample,
                ),
                geo_from_topo: interaction_path_diagnostics(
                    InteractionPath::GeometryFromTopology,
                    &interactions.geo_from_topo,
                    self.interaction_stack.block.path_scale_for(
                        InteractionPath::GeometryFromTopology,
                        &diagnostic_context,
                    ),
                    None,
                    &diagnostic_context,
                    InteractionDiagnosticProvenance::PerExample,
                ),
                geo_from_pocket: interaction_path_diagnostics(
                    InteractionPath::GeometryFromPocket,
                    &interactions.geo_from_pocket,
                    self.interaction_stack.block.path_scale_for(
                        InteractionPath::GeometryFromPocket,
                        &diagnostic_context,
                    ),
                    Some((
                        geo_from_pocket_bias.get(batch_index),
                        self.interaction_stack.block.geometry_attention_bias_scale(),
                        Some(chemistry_role_coverage),
                    )),
                    &diagnostic_context,
                    InteractionDiagnosticProvenance::PerExample,
                ),
                pocket_from_topo: interaction_path_diagnostics(
                    InteractionPath::PocketFromTopology,
                    &interactions.pocket_from_topo,
                    self.interaction_stack.block.path_scale_for(
                        InteractionPath::PocketFromTopology,
                        &diagnostic_context,
                    ),
                    None,
                    &diagnostic_context,
                    InteractionDiagnosticProvenance::PerExample,
                ),
                pocket_from_geo: interaction_path_diagnostics(
                    InteractionPath::PocketFromGeometry,
                    &interactions.pocket_from_geo,
                    self.interaction_stack.block.path_scale_for(
                        InteractionPath::PocketFromGeometry,
                        &diagnostic_context,
                    ),
                    Some((
                        pocket_from_geo_bias.get(batch_index),
                        self.interaction_stack.block.geometry_attention_bias_scale(),
                        Some(chemistry_role_coverage),
                    )),
                    &diagnostic_context,
                    InteractionDiagnosticProvenance::PerExample,
                ),
            };
        attach_topology_pocket_pharmacophore_path_diagnostics(
            &mut interaction_diagnostics.topo_from_pocket,
            &mut interaction_diagnostics.pocket_from_topo,
            &example.topology.chemistry_roles.role_vectors,
            &slots.topology.slot_weights,
            &example.pocket.chemistry_roles.role_vectors,
            &slots.pocket.slot_weights,
            &interactions.topo_from_pocket.attention_weights,
            &interactions.pocket_from_topo.attention_weights,
        );
        let diagnostics = SemanticDiagnosticsBundle::from_modalities(
            (&encodings.topology, &slots.topology),
            (&encodings.geometry, &slots.geometry),
            (&encodings.pocket, &slots.pocket),
        );
        let probes = self.apply_modality_focus_to_probes(self.probe_stack.probes.forward(
            &encodings.topology,
            &encodings.geometry,
            &encodings.pocket,
            &slots.topology,
            &slots.geometry,
            &slots.pocket,
        ));
        let pocket_priors = self.generator_stack.pocket_prior_head.forward(&encodings.pocket);
        let generation_state =
            self.build_generation_state(example, &slots, &interactions, &diagnostic_context);
        let decoded = self.generator_stack.ligand_decoder.decode(&generation_state);
        let flow_matching = self.flow_matching_training_record(
            example,
            &generation_state,
            &interactions,
            &diagnostic_context,
        );
        let mut sync_context = self.sync_context_from_example(example, &slots);
        sync_context.flow_t = flow_matching.as_ref().map(|record| record.t);
        let rollout_path_means =
            gate_summary_from_interaction_diagnostics(&interaction_diagnostics);
        let rollout_training =
            self.rollout_training_record(example, &generation_state, &decoded);
        let rollout = if build_rollout_diagnostics {
            no_grad(|| {
                self.rollout_generation(
                    example,
                    &generation_state,
                    rollout_path_means,
                    &diagnostic_context,
                )
            })
        } else {
            self.skipped_rollout_diagnostics_record(
                &generation_state,
                rollout_path_means,
                &diagnostic_context,
            )
        };

        ResearchForward {
            encodings,
            slots,
            diagnostics,
            interactions,
            interaction_diagnostics,
            sync_context,
            probes,
            generation: GenerationForward {
                generation_mode: self.generation_mode,
                state: generation_state,
                decoded,
                rollout,
                rollout_training,
                flow_matching,
                pocket_priors,
            },
        }
    }

    fn build_generation_state(
        &self,
        example: &MolecularExample,
        slots: &DecomposedModalities,
        interactions: &CrossModalInteractions,
        interaction_execution_context: &InteractionExecutionContext,
    ) -> ConditionedGenerationState {
        let partial_ligand =
            self.initial_partial_ligand_state(example, Some(interaction_execution_context));
        let device = partial_ligand.atom_types.device();
        let num_atoms = partial_ligand.atom_types.size()[0];

        ConditionedGenerationState {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            partial_ligand: PartialLigandState {
                atom_mask: Tensor::ones([num_atoms], (Kind::Float, device)),
                ..partial_ligand
            },
            topology_context: masked_slot_context(
                merge_slot_contexts(
                    &slots.topology.slots,
                    &[&interactions.topo_from_geo, &interactions.topo_from_pocket],
                ),
                &slots.topology.active_slot_mask,
            ),
            geometry_context: masked_slot_context(
                merge_slot_contexts(
                    &slots.geometry.slots,
                    &[&interactions.geo_from_topo, &interactions.geo_from_pocket],
                ),
                &slots.geometry.active_slot_mask,
            ),
            pocket_context: masked_slot_context(
                merge_slot_contexts(
                    &slots.pocket.slots,
                    &[
                        &interactions.pocket_from_topo,
                        &interactions.pocket_from_geo,
                    ],
                ),
                &slots.pocket.active_slot_mask,
            ),
            topology_slot_mask: slots.topology.active_slot_mask.shallow_clone(),
            geometry_slot_mask: slots.geometry.active_slot_mask.shallow_clone(),
            pocket_slot_mask: slots.pocket.active_slot_mask.shallow_clone(),
        }
    }

    fn initial_partial_ligand_state(
        &self,
        example: &MolecularExample,
        interaction_execution_context: Option<&InteractionExecutionContext>,
    ) -> PartialLigandState {
        if self.generation_mode == GenerationModeConfig::DeNovoInitialization {
            let config = &self.generation_target.de_novo_initialization;
            let (min_atom_count, max_atom_count) =
                if let Some(atom_count) = config.dataset_calibrated_atom_count {
                    (atom_count, atom_count)
                } else {
                    (config.min_atom_count, config.max_atom_count)
                };
            let initializer = DeNovoScaffoldInitializer {
                atom_count_prior: PocketVolumeAtomCountPrior {
                    min_atom_count,
                    max_atom_count,
                    pocket_atom_divisor: config.pocket_atom_divisor,
                },
                atom_vocab_size: self.atom_vocab_size,
                radius_fraction: config.radius_fraction,
                seed: interaction_execution_context
                    .and_then(|context| context.generation_sample_seed)
                    .unwrap_or(config.seed),
            };
            let centered_pocket = pocket_centered_features(&example.pocket);
            return initializer.initialize(&PocketInitializationContext {
                example_id: &example.example_id,
                protein_id: &example.protein_id,
                pocket_coords: &centered_pocket.coords,
            });
        }
        if self.generation_mode == GenerationModeConfig::PocketOnlyInitializationBaseline {
            let config = &self.generation_target.pocket_only_initialization;
            let initializer = PocketCentroidScaffoldInitializer {
                atom_count_prior: FixedAtomCountPrior {
                    atom_count: config.atom_count,
                },
                atom_type_token: config.atom_type_token,
                radius_fraction: config.radius_fraction,
                coordinate_seed: interaction_execution_context
                    .and_then(|context| context.generation_sample_seed)
                    .unwrap_or(config.coordinate_seed),
            };
            return initializer.initialize(&PocketInitializationContext {
                example_id: &example.example_id,
                protein_id: &example.protein_id,
                pocket_coords: &example.pocket.coords,
            });
        }

        let num_atoms = example.decoder_supervision.corrupted_atom_types.size()[0];
        let device = example.decoder_supervision.corrupted_atom_types.device();
        PartialLigandState {
            atom_types: example
                .decoder_supervision
                .corrupted_atom_types
                .shallow_clone(),
            coords: example.decoder_supervision.noisy_coords.shallow_clone(),
            atom_mask: Tensor::ones([num_atoms], (Kind::Float, device)),
            step_index: 0,
        }
    }
}

fn masked_slot_context(context: Tensor, active_slot_mask: &Tensor) -> Tensor {
    let slot_count = context.size()[0];
    let device = context.device();
    if active_slot_mask.size().as_slice() == [slot_count] {
        context * active_slot_mask.to_device(device).unsqueeze(-1)
    } else {
        context
    }
}
