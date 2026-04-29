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
            self.skipped_rollout_diagnostics_record(&optimizer_record.state, rollout_path_means)
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
                flow_matching: optimizer_record.flow_matching,
            },
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
            let (topology, geometry, pocket) = self.de_novo_conditioning_modalities(example);
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
        let generation_state = self.build_generation_state(example, &slots, &interactions);
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
        let generation_state = self.build_generation_state(example, &slots, &interactions);
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
            self.skipped_rollout_diagnostics_record(&generation_state, rollout_path_means)
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
        let partial_ligand = self.initial_partial_ligand_state(example);
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

    fn initial_partial_ligand_state(&self, example: &MolecularExample) -> PartialLigandState {
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
                seed: config.seed,
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
                coordinate_seed: config.coordinate_seed,
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
