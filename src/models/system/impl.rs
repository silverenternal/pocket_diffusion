/// Grouped topology, geometry, and pocket encoder/slot branches.
#[derive(Debug)]
pub struct EncoderStack {
    /// Topology semantic branch: graph encoder plus topology slots.
    pub topology_branch: TopologySemanticBranch,
    /// Geometry semantic branch: coordinate encoder plus geometry slots.
    pub geometry_branch: GeometrySemanticBranch,
    /// Pocket semantic branch: context encoder plus pocket slots.
    pub pocket_branch: PocketSemanticBranch,
}

/// Grouped controlled cross-modality interaction components.
#[derive(Debug)]
pub struct InteractionStack {
    /// Explicit gated cross-modal interaction block.
    pub block: CrossModalInteractionBlock,
}

/// Grouped generation and flow heads.
#[derive(Debug)]
pub struct GeneratorStack {
    /// Minimal modular ligand decoder.
    pub ligand_decoder: ModularLigandDecoder,
    /// Config-selected flow-matching velocity head.
    pub flow_matching_head: FlowVelocityHead,
    /// Full molecular flow branches for atom type, bond, topology, and context transport.
    pub molecular_flow_head: FullMolecularFlowHead,
}

/// Grouped semantic and leakage probe heads.
#[derive(Debug)]
pub struct ProbeStack {
    /// Semantic probe heads.
    pub probes: SemanticProbeHeads,
}

/// Research system that keeps encoders separate and adds structured interactions on top.
#[derive(Debug)]
pub struct Phase1ResearchSystem {
    /// Structurally separate topology, geometry, and pocket encoder branches.
    pub encoder_stack: EncoderStack,
    /// Explicit directed gated interaction stack.
    pub interaction_stack: InteractionStack,
    /// Decoder and flow heads used by generation paths.
    pub generator_stack: GeneratorStack,
    /// Semantic probe stack.
    pub probe_stack: ProbeStack,
    generation_target: GenerationTargetConfig,
    generation_backend_family: GenerationBackendFamilyConfig,
    generation_mode: GenerationModeConfig,
    flow_matching_config: crate::config::FlowMatchingConfig,
    modality_focus: crate::config::ModalityFocusConfig,
    atom_vocab_size: i64,
}

impl Phase1ResearchSystem {
    /// Construct the modular Phase 2 system from configuration.
    pub fn new(vs: &nn::Path, config: &ResearchConfig) -> Self {
        let topology_branch = TopologySemanticBranch::from_parts(
            TopologyEncoderImpl::new_with_config(
                &(vs / "topology"),
                config.model.atom_vocab_size,
                config.model.hidden_dim,
                &config.model.topology_encoder,
            ),
            SoftSlotDecomposer::new_with_config(
                &(vs / "slot_topology"),
                config.model.hidden_dim,
                config.model.num_slots,
                &config.model.slot_decomposition,
            ),
        );
        let geometry_branch = GeometrySemanticBranch::from_parts(
            GeometryEncoderImpl::new_with_config(
                &(vs / "geometry"),
                config.model.hidden_dim,
                &config.model.geometry_encoder,
            ),
            SoftSlotDecomposer::new_with_config(
                &(vs / "slot_geometry"),
                config.model.hidden_dim,
                config.model.num_slots,
                &config.model.slot_decomposition,
            ),
        );
        let pocket_branch = PocketSemanticBranch::from_parts(
            PocketEncoderImpl::new_with_config(
                &(vs / "pocket"),
                config.model.pocket_feature_dim,
                config.model.hidden_dim,
                &config.model.pocket_encoder,
            ),
            SoftSlotDecomposer::new_with_config(
                &(vs / "slot_pocket"),
                config.model.hidden_dim,
                config.model.num_slots,
                &config.model.slot_decomposition,
            ),
        );
        let interaction_block = CrossModalInteractionBlock::new(vs, config);
        let ligand_decoder = ModularLigandDecoder::new_with_config(
            &(vs / "ligand_decoder"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            &config.model.decoder_conditioning,
        );
        let probes = SemanticProbeHeads::new_with_config(
            &(vs / "probes"),
            config.model.hidden_dim,
            config.model.pocket_feature_dim,
            &config.model.semantic_probes,
        );
        let pairwise_geometry = config.model.pairwise_geometry.enabled.then_some(
            crate::models::PairwiseGeometryConfig {
                radius: config.model.pairwise_geometry.radius as f32,
                max_neighbors: config.model.pairwise_geometry.max_neighbors,
                residual_scale: config.model.pairwise_geometry.residual_scale,
            },
        );
        let flow_matching_head = match config.model.flow_velocity_head.kind {
            FlowVelocityHeadKind::Geometry => {
                FlowVelocityHead::Geometry(Box::new(match pairwise_geometry.clone() {
                    Some(pairwise) => GeometryFlowMatchingHead::new_with_pairwise(
                        &(vs / "flow_matching_head"),
                        config.model.hidden_dim,
                        pairwise,
                    ),
                    None => GeometryFlowMatchingHead::new(
                        &(vs / "flow_matching_head"),
                        config.model.hidden_dim,
                    ),
                }))
            }
            FlowVelocityHeadKind::AtomPocketCrossAttention => {
                FlowVelocityHead::AtomPocketCrossAttention(Box::new(
                    crate::models::AtomPocketCrossAttentionVelocityHead::new(
                        &(vs / "flow_matching_head"),
                        crate::models::AtomPocketCrossAttentionVelocityConfig {
                            enabled: true,
                            hidden_dim: config.model.hidden_dim,
                            gate_initial_bias: config.model.flow_velocity_head.gate_initial_bias,
                            pairwise_geometry,
                        },
                    ),
                ))
            }
        };
        let molecular_flow_head = FullMolecularFlowHead::new_with_conditioning_kind(
            &(vs / "molecular_flow_head"),
            config.model.atom_vocab_size,
            config.model.bond_vocab_size,
            config.model.hidden_dim,
            config.model.decoder_conditioning.kind,
        );

        let generation_backend_family = resolved_generation_backend_family(config);
        let generation_mode = config
            .data
            .generation_target
            .generation_mode
            .resolved_for_backend(generation_backend_family);

        Self {
            encoder_stack: EncoderStack {
                topology_branch,
                geometry_branch,
                pocket_branch,
            },
            interaction_stack: InteractionStack {
                block: interaction_block,
            },
            generator_stack: GeneratorStack {
                ligand_decoder,
                flow_matching_head,
                molecular_flow_head,
            },
            probe_stack: ProbeStack { probes },
            generation_target: config.data.generation_target.clone(),
            generation_backend_family,
            generation_mode,
            flow_matching_config: config.generation_method.flow_matching.clone(),
            modality_focus: config.model.modality_focus,
            atom_vocab_size: config.model.atom_vocab_size,
        }
    }

    /// Run the three modality encoders for one example.
    pub(crate) fn encode_example(&self, example: &MolecularExample) -> EncodedModalities {
        if self.generation_mode == GenerationModeConfig::DeNovoInitialization {
            let (topology, geometry, pocket) = self.de_novo_conditioning_modalities(example);
            return EncodedModalities {
                topology: self.encoder_stack.topology_branch.encode(&topology),
                geometry: self.encoder_stack.geometry_branch.encode(&geometry),
                pocket: self.encoder_stack.pocket_branch.encode(&pocket),
            };
        }
        EncodedModalities {
            topology: self.encoder_stack.topology_branch.encode(&example.topology),
            geometry: self.encoder_stack.geometry_branch.encode(&example.geometry),
            pocket: self.encoder_stack.pocket_branch.encode(&example.pocket),
        }
    }

    /// Run the three modality encoders over already-collated padded tensors.
    pub(crate) fn encode_batch_inputs(&self, batch: &MolecularBatch) -> BatchedEncodedModalities {
        let inputs = &batch.encoder_inputs;
        BatchedEncodedModalities {
            topology: self.encoder_stack.topology_branch.encode_batch(
                &inputs.atom_types,
                &inputs.adjacency,
                &inputs.bond_type_adjacency,
                &inputs.ligand_mask,
            ),
            geometry: self.encoder_stack.geometry_branch.encode_batch(
                &inputs.ligand_coords,
                &inputs.pairwise_distances,
                &inputs.ligand_mask,
            ),
            pocket: self.encoder_stack.pocket_branch.encode_batch(
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
        self.apply_modality_focus(DecomposedModalities {
            topology: self.encoder_stack.topology_branch.decompose(&encodings.topology),
            geometry: self.encoder_stack.geometry_branch.decompose(&encodings.geometry),
            pocket: self.encoder_stack.pocket_branch.decompose(&encodings.pocket),
        })
    }

    /// Decompose batched modality encodings into learned slots.
    pub(crate) fn decompose_batched_modalities(
        &self,
        encodings: &BatchedEncodedModalities,
    ) -> BatchedDecomposedModalities {
        self.apply_batched_modality_focus(BatchedDecomposedModalities {
            topology: self
                .encoder_stack
                .topology_branch
                .decompose_batch(&encodings.topology),
            geometry: self
                .encoder_stack
                .geometry_branch
                .decompose_batch(&encodings.geometry),
            pocket: self
                .encoder_stack
                .pocket_branch
                .decompose_batch(&encodings.pocket),
        })
    }

    pub(crate) fn apply_modality_focus_to_encodings(
        &self,
        mut encodings: EncodedModalities,
    ) -> EncodedModalities {
        if !self.modality_focus.keep_topology() {
            zero_modality_encoding(&mut encodings.topology);
        }
        if !self.modality_focus.keep_geometry() {
            zero_modality_encoding(&mut encodings.geometry);
        }
        if !self.modality_focus.keep_pocket() {
            zero_modality_encoding(&mut encodings.pocket);
        }
        encodings
    }

    pub(crate) fn apply_batched_modality_focus_to_encodings(
        &self,
        mut encodings: BatchedEncodedModalities,
    ) -> BatchedEncodedModalities {
        if !self.modality_focus.keep_topology() {
            zero_batched_modality_encoding(&mut encodings.topology);
        }
        if !self.modality_focus.keep_geometry() {
            zero_batched_modality_encoding(&mut encodings.geometry);
        }
        if !self.modality_focus.keep_pocket() {
            zero_batched_modality_encoding(&mut encodings.pocket);
        }
        encodings
    }

    pub(crate) fn apply_modality_focus_to_probes(&self, mut probes: ProbeOutputs) -> ProbeOutputs {
        let keep_topology = self.modality_focus.keep_topology();
        let keep_geometry = self.modality_focus.keep_geometry();
        let keep_pocket = self.modality_focus.keep_pocket();

        if !keep_topology {
            zero_probe_tensor(&mut probes.topology_adjacency_logits);
            zero_probe_tensor(&mut probes.ligand_pharmacophore_role_logits);
        }
        if !keep_geometry {
            zero_probe_tensor(&mut probes.geometry_distance_predictions);
        }
        if !keep_pocket {
            zero_probe_tensor(&mut probes.pocket_feature_predictions);
            zero_probe_tensor(&mut probes.pocket_pharmacophore_role_logits);
        }
        if !(keep_topology && keep_geometry) {
            zero_probe_tensor(&mut probes.topology_to_geometry_scalar_logits);
            zero_probe_tensor(&mut probes.geometry_to_topology_scalar_logits);
            zero_probe_tensor(
                &mut probes
                    .leakage_probe_fit
                    .topology_to_geometry_scalar_logits,
            );
            zero_probe_tensor(
                &mut probes
                    .leakage_probe_fit
                    .geometry_to_topology_scalar_logits,
            );
            zero_probe_tensor(
                &mut probes
                    .leakage_encoder_penalty
                    .topology_to_geometry_scalar_logits,
            );
            zero_probe_tensor(
                &mut probes
                    .leakage_encoder_penalty
                    .geometry_to_topology_scalar_logits,
            );
        }
        if !(keep_geometry && keep_pocket) {
            zero_probe_tensor(&mut probes.pocket_to_geometry_scalar_logits);
            zero_probe_tensor(&mut probes.geometry_to_pocket_role_logits);
            zero_probe_tensor(
                &mut probes.leakage_probe_fit.pocket_to_geometry_scalar_logits,
            );
            zero_probe_tensor(&mut probes.leakage_probe_fit.geometry_to_pocket_role_logits);
            zero_probe_tensor(
                &mut probes
                    .leakage_encoder_penalty
                    .pocket_to_geometry_scalar_logits,
            );
            zero_probe_tensor(
                &mut probes
                    .leakage_encoder_penalty
                    .geometry_to_pocket_role_logits,
            );
        }
        if !(keep_topology && keep_pocket) {
            zero_probe_tensor(&mut probes.topology_to_pocket_role_logits);
            zero_probe_tensor(&mut probes.pocket_to_ligand_role_logits);
            zero_probe_tensor(&mut probes.pocket_to_topology_role_logits);
            zero_probe_tensor(&mut probes.leakage_probe_fit.topology_to_pocket_role_logits);
            zero_probe_tensor(&mut probes.leakage_probe_fit.pocket_to_ligand_role_logits);
            zero_probe_tensor(&mut probes.leakage_probe_fit.pocket_to_topology_role_logits);
            zero_probe_tensor(
                &mut probes
                    .leakage_encoder_penalty
                    .topology_to_pocket_role_logits,
            );
            zero_probe_tensor(
                &mut probes
                    .leakage_encoder_penalty
                    .pocket_to_ligand_role_logits,
            );
            zero_probe_tensor(
                &mut probes
                    .leakage_encoder_penalty
                    .pocket_to_topology_role_logits,
            );
        }
        if !(keep_topology && keep_geometry && keep_pocket) {
            zero_probe_tensor(&mut probes.affinity_prediction);
        }
        probes
    }

    fn apply_modality_focus(&self, mut slots: DecomposedModalities) -> DecomposedModalities {
        if !self.modality_focus.keep_topology() {
            zero_slot_encoding(&mut slots.topology);
        }
        if !self.modality_focus.keep_geometry() {
            zero_slot_encoding(&mut slots.geometry);
        }
        if !self.modality_focus.keep_pocket() {
            zero_slot_encoding(&mut slots.pocket);
        }
        slots
    }

    fn apply_batched_modality_focus(
        &self,
        mut slots: BatchedDecomposedModalities,
    ) -> BatchedDecomposedModalities {
        if !self.modality_focus.keep_topology() {
            zero_batched_slot_encoding(&mut slots.topology);
        }
        if !self.modality_focus.keep_geometry() {
            zero_batched_slot_encoding(&mut slots.geometry);
        }
        if !self.modality_focus.keep_pocket() {
            zero_batched_slot_encoding(&mut slots.pocket);
        }
        slots
    }

    /// Apply all directed cross-modality interactions.
    #[allow(dead_code)] // Compatibility wrapper for ablations that bypass diagnostics.
    pub(crate) fn interact_modalities(
        &self,
        slots: &DecomposedModalities,
    ) -> CrossModalInteractions {
        self.interaction_stack.block.forward(slots)
    }

    #[allow(dead_code)] // Compatibility wrapper for default-context diagnostic callers.
    pub(crate) fn interact_modalities_with_diagnostics(
        &self,
        slots: &DecomposedModalities,
    ) -> (
        CrossModalInteractions,
        crate::models::interaction::CrossModalInteractionDiagnostics,
    ) {
        self.interaction_stack.block.forward_with_diagnostics(slots)
    }

    /// Apply all directed cross-modality interactions over slot batches.
    #[allow(dead_code)] // Compatibility wrapper for batched ablations that bypass diagnostics.
    pub(crate) fn interact_batched_modalities(
        &self,
        batch: &MolecularBatch,
        slots: &BatchedDecomposedModalities,
    ) -> BatchedCrossModalInteractions {
        self.interaction_stack.block.forward_batch(batch, slots)
    }

    #[allow(dead_code)] // Compatibility wrapper for batched default-context diagnostic callers.
    pub(crate) fn interact_batched_modalities_with_diagnostics(
        &self,
        batch: &MolecularBatch,
        slots: &BatchedDecomposedModalities,
    ) -> (
        BatchedCrossModalInteractions,
        crate::models::interaction::BatchedCrossModalInteractionDiagnostics,
    ) {
        self.interaction_stack
            .block
            .forward_batch_with_diagnostics(batch, slots)
    }

    pub(crate) fn interact_batched_modalities_with_diagnostics_with_context(
        &self,
        batch: &MolecularBatch,
        slots: &BatchedDecomposedModalities,
        context: InteractionExecutionContext,
    ) -> (
        BatchedCrossModalInteractions,
        crate::models::interaction::BatchedCrossModalInteractionDiagnostics,
    ) {
        self.interaction_stack
            .block
            .forward_batch_with_diagnostics_with_context(batch, slots, context)
    }

    fn batched_geo_from_pocket_attention_bias(
        &self,
        batch: &MolecularBatch,
        slots: &BatchedDecomposedModalities,
    ) -> crate::models::interaction::LigandPocketSlotAttentionBias {
        self.interaction_stack
            .block
            .ligand_pocket_bias_for_batch(batch, slots)
    }

    fn decoder_capability_label(&self) -> &'static str {
        self.generation_mode
            .compatibility_contract()
            .decoder_capability_label
    }

    fn atom_count_prior_provenance_label(&self) -> &'static str {
        match self.generation_mode {
            GenerationModeConfig::PocketOnlyInitializationBaseline => "fixed",
            GenerationModeConfig::DeNovoInitialization => {
                if self
                    .generation_target
                    .de_novo_initialization
                    .dataset_calibrated_atom_count
                    .is_some()
                {
                    "dataset_calibrated"
                } else {
                    "pocket_volume"
                }
            }
            GenerationModeConfig::TargetLigandDenoising
            | GenerationModeConfig::LigandRefinement
            | GenerationModeConfig::FlowRefinement => "target_ligand",
        }
    }

    fn conditioning_coordinate_frame_label(&self) -> &'static str {
        match self.generation_mode {
            GenerationModeConfig::DeNovoInitialization => {
                "pocket_centroid_centered_conditioning_no_target_ligand_frame"
            }
            GenerationModeConfig::PocketOnlyInitializationBaseline => {
                "pocket_model_frame_fixed_atom_baseline"
            }
            GenerationModeConfig::TargetLigandDenoising
            | GenerationModeConfig::LigandRefinement
            | GenerationModeConfig::FlowRefinement => {
                "ligand_centered_training_supervision_only"
            }
        }
    }

    fn de_novo_conditioning_modalities(
        &self,
        example: &MolecularExample,
    ) -> (TopologyFeatures, GeometryFeatures, PocketFeatures) {
        let scaffold = self.initial_partial_ligand_state(example);
        let device = scaffold.coords.device();
        let topology = topology_from_partial_ligand(&scaffold, self.flow_matching_config.noise_scale)
            .to_device(device);
        let geometry = geometry_from_coords(&scaffold.coords);
        let pocket = pocket_centered_features(&example.pocket);
        (topology, geometry, pocket)
    }
}

fn topology_from_partial_ligand(
    ligand: &PartialLigandState,
    distance_bond_threshold: f64,
) -> TopologyFeatures {
    let atom_count = ligand.atom_types.size().first().copied().unwrap_or(0).max(0);
    let device = ligand.atom_types.device();
    let adjacency = scaffold_adjacency_from_coords(&ligand.coords, distance_bond_threshold);
    let mut edge_rows = Vec::new();
    let mut edge_cols = Vec::new();
    let mut bond_types = Vec::new();
    for left in 0..atom_count {
        for right in (left + 1)..atom_count {
            if adjacency.double_value(&[left, right]) > 0.5 {
                edge_rows.push(left);
                edge_cols.push(right);
                bond_types.push(1_i64);
            }
        }
    }
    let edge_index = if edge_rows.is_empty() {
        Tensor::zeros([2, 0], (Kind::Int64, device))
    } else {
        Tensor::stack(
            &[
                Tensor::from_slice(&edge_rows).to_device(device),
                Tensor::from_slice(&edge_cols).to_device(device),
            ],
            0,
        )
    };
    let bond_types = if bond_types.is_empty() {
        Tensor::zeros([0], (Kind::Int64, device))
    } else {
        Tensor::from_slice(&bond_types).to_device(device)
    };
    let role_atom_types = (0..atom_count)
        .map(|index| atom_type_from_token(ligand.atom_types.int64_value(&[index])))
        .collect::<Vec<_>>();
    let chemistry_roles =
        chemistry_role_features_from_atom_types(&role_atom_types).to_device(device);
    TopologyFeatures {
        atom_types: ligand.atom_types.shallow_clone(),
        edge_index,
        bond_types,
        adjacency,
        chemistry_roles,
    }
}

pub(super) fn scaffold_adjacency_from_coords(coords: &Tensor, noise_scale: f64) -> Tensor {
    let atom_count = coords.size().first().copied().unwrap_or(0).max(0);
    if atom_count <= 1 {
        return Tensor::zeros([atom_count, atom_count], (Kind::Float, coords.device()));
    }
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    let distances = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .sqrt();
    let threshold = (1.75 + noise_scale.max(0.0)).min(3.0);
    let adjacency = distances.lt(threshold).to_kind(Kind::Float);
    let eye = Tensor::eye(atom_count, (Kind::Float, coords.device()));
    adjacency * (Tensor::ones_like(&eye) - eye)
}

fn geometry_from_coords(coords: &Tensor) -> GeometryFeatures {
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    let pairwise_distances = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .sqrt();
    GeometryFeatures {
        coords: coords.shallow_clone(),
        pairwise_distances,
    }
}

fn pocket_centered_features(pocket: &PocketFeatures) -> PocketFeatures {
    if pocket.coords.numel() == 0 {
        return pocket.clone();
    }
    let centroid = pocket.coords.mean_dim([0].as_slice(), false, Kind::Float);
    PocketFeatures {
        coords: &pocket.coords - centroid.unsqueeze(0),
        atom_features: pocket.atom_features.shallow_clone(),
        pooled_features: pocket.pooled_features.shallow_clone(),
        chemistry_roles: pocket.chemistry_roles.clone(),
    }
}

fn atom_type_from_token(token: i64) -> AtomType {
    match token {
        0 => AtomType::Carbon,
        1 => AtomType::Nitrogen,
        2 => AtomType::Oxygen,
        3 => AtomType::Sulfur,
        4 => AtomType::Hydrogen,
        _ => AtomType::Other,
    }
}

fn zero_modality_encoding(encoding: &mut ModalityEncoding) {
    encoding.token_embeddings = Tensor::zeros_like(&encoding.token_embeddings);
    encoding.pooled_embedding = Tensor::zeros_like(&encoding.pooled_embedding);
}

fn zero_batched_modality_encoding(encoding: &mut BatchedModalityEncoding) {
    encoding.token_embeddings = Tensor::zeros_like(&encoding.token_embeddings);
    encoding.token_mask = Tensor::zeros_like(&encoding.token_mask);
    encoding.pooled_embedding = Tensor::zeros_like(&encoding.pooled_embedding);
}

fn zero_probe_tensor(tensor: &mut Tensor) {
    *tensor = Tensor::zeros([0], (Kind::Float, tensor.device()));
}

fn zero_slot_encoding(encoding: &mut SlotEncoding) {
    encoding.slots = Tensor::zeros_like(&encoding.slots);
    encoding.slot_weights = Tensor::zeros_like(&encoding.slot_weights);
    encoding.token_assignments = Tensor::zeros_like(&encoding.token_assignments);
    encoding.slot_activation_logits = Tensor::zeros_like(&encoding.slot_activation_logits);
    encoding.slot_activations = Tensor::zeros_like(&encoding.slot_activations);
    encoding.active_slot_mask = Tensor::zeros_like(&encoding.active_slot_mask);
    encoding.active_slot_count = 0.0;
    encoding.reconstructed_tokens = Tensor::zeros_like(&encoding.reconstructed_tokens);
}

fn zero_batched_slot_encoding(encoding: &mut BatchedSlotEncoding) {
    encoding.slots = Tensor::zeros_like(&encoding.slots);
    encoding.slot_weights = Tensor::zeros_like(&encoding.slot_weights);
    encoding.token_assignments = Tensor::zeros_like(&encoding.token_assignments);
    encoding.slot_activation_logits = Tensor::zeros_like(&encoding.slot_activation_logits);
    encoding.slot_activations = Tensor::zeros_like(&encoding.slot_activations);
    encoding.active_slot_mask = Tensor::zeros_like(&encoding.active_slot_mask);
    encoding.active_slot_count = Tensor::zeros_like(&encoding.active_slot_count);
    encoding.reconstructed_tokens = Tensor::zeros_like(&encoding.reconstructed_tokens);
    encoding.token_mask = Tensor::zeros_like(&encoding.token_mask);
}
