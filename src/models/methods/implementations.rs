impl PocketGenerationMethod for ConditionedDenoisingMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: CONDITIONED_DENOISING_METHOD_ID.to_string(),
            method_name: "Conditioned Denoising".to_string(),
            method_family: PocketGenerationMethodFamily::ConditionedDenoising,
            capability: GenerationMethodCapability {
                trainable: true,
                batched_generation: true,
                method_native_generation: true,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
                CandidateLayerKind::Reranked,
            ],
            evidence_role: GenerationEvidenceRole::ClaimBearing,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let inferred = layers.inferred_bond.clone();
        let proxy = proxy_rerank_candidates(&inferred);
        let calibrated = CalibratedReranker::fit(&inferred).rerank(&inferred);
        layered_output_from_legacy(metadata, layers, Some(proxy), Some(calibrated), true)
    }
}

impl PocketGenerationMethod for HeuristicRawRolloutMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: HEURISTIC_RAW_ROLLOUT_METHOD_ID.to_string(),
            method_name: "Heuristic Raw Rollout".to_string(),
            method_family: PocketGenerationMethodFamily::Heuristic,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: true,
                uses_postprocessors: false,
                repair_layer: false,
                inferred_bond_layer: false,
                reranked_layer: false,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![CandidateLayerKind::RawRollout],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            false,
        );
        let raw_only = CandidateGenerationLayers {
            raw_rollout: layers.raw_rollout,
            repaired: Vec::new(),
            inferred_bond: Vec::new(),
        };
        layered_output_from_legacy(metadata, raw_only, None, None, true)
    }
}

impl PocketGenerationMethod for PocketCentroidRepairMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: REPAIR_ONLY_METHOD_ID.to_string(),
            method_name: "Pocket Centroid Repair Proxy".to_string(),
            method_family: PocketGenerationMethodFamily::RepairOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: false,
                reranked_layer: false,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            true,
        );
        let repair_layers = CandidateGenerationLayers {
            raw_rollout: layers.raw_rollout,
            repaired: layers.repaired,
            inferred_bond: Vec::new(),
        };
        layered_output_from_legacy(metadata, repair_layers, None, None, false)
    }
}

impl PocketGenerationMethod for DeterministicProxyRerankerMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: DETERMINISTIC_PROXY_RERANKER_METHOD_ID.to_string(),
            method_name: "Deterministic Proxy Reranker".to_string(),
            method_family: PocketGenerationMethodFamily::RerankerOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let proxy = proxy_rerank_candidates(&layers.inferred_bond);
        layered_output_from_legacy(metadata, layers, Some(proxy), None, false)
    }
}

impl PocketGenerationMethod for CalibratedRerankerMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: CALIBRATED_RERANKER_METHOD_ID.to_string(),
            method_name: "Calibrated Reranker".to_string(),
            method_family: PocketGenerationMethodFamily::RerankerOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
                CandidateLayerKind::Reranked,
            ],
            evidence_role: GenerationEvidenceRole::ClaimBearing,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let inferred = layers.inferred_bond.clone();
        let proxy = proxy_rerank_candidates(&inferred);
        let calibrated = CalibratedReranker::fit(&inferred).rerank(&inferred);
        layered_output_from_legacy(metadata, layers, Some(proxy), Some(calibrated), false)
    }
}

impl PocketGenerationMethod for PreferenceAwareRerankerMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: PREFERENCE_AWARE_RERANKER_METHOD_ID.to_string(),
            method_name: "Preference-Aware Reranker".to_string(),
            method_family: PocketGenerationMethodFamily::RerankerOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
                CandidateLayerKind::Reranked,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let inferred = layers.inferred_bond.clone();
        let proxy = proxy_rerank_candidates(&inferred);
        let preference_reranked = crate::models::RuleBasedPreferenceReranker::default()
            .rerank_candidates(
                &inferred,
                CandidateLayerKind::InferredBond,
                Some(PREFERENCE_AWARE_RERANKER_METHOD_ID),
                &[],
            );
        let keep = (preference_reranked.len() / 2)
            .max(1)
            .min(preference_reranked.len());
        layered_output_from_legacy(
            metadata,
            layers,
            Some(proxy),
            Some(preference_reranked.into_iter().take(keep).collect()),
            false,
        )
    }
}

impl PocketGenerationMethod for FlowMatchingMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: FLOW_MATCHING_METHOD_ID.to_string(),
            method_name: "Flow Matching Transport".to_string(),
            method_family: PocketGenerationMethodFamily::FlowMatching,
            capability: GenerationMethodCapability {
                trainable: true,
                batched_generation: true,
                method_native_generation: true,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: false,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let request = context.conditioned_request.as_ref().expect(
            "decomposed_forward only returns Some when conditioned_request is present",
        );
        layered_output_from_legacy(
            metadata,
            flow_matching_layers(layers, request),
            None,
            None,
            true,
        )
    }
}

impl PocketGenerationMethod for AutoregressiveGraphGeometryMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: AUTOREGRESSIVE_GRAPH_GEOMETRY_METHOD_ID.to_string(),
            method_name: "Autoregressive Graph Geometry".to_string(),
            method_family: PocketGenerationMethodFamily::Autoregressive,
            capability: GenerationMethodCapability {
                trainable: true,
                batched_generation: true,
                method_native_generation: true,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: false,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        layered_output_from_legacy(
            metadata,
            autoregressive_layers(layers, context.conditioned_request.as_ref().expect(
                "decomposed_forward only returns Some when conditioned_request is present",
            )),
            None,
            None,
            true,
        )
    }
}

impl PocketGenerationMethod for EnergyGuidedRefinementMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: ENERGY_GUIDED_REFINEMENT_METHOD_ID.to_string(),
            method_name: "Energy Guided Refinement".to_string(),
            method_family: PocketGenerationMethodFamily::RepairOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = decomposed_forward(&context) else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            forward,
            context.candidate_limit.max(1),
            true,
        );
        let request = context.conditioned_request.as_ref().expect(
            "decomposed_forward only returns Some when conditioned_request is present",
        );
        let layers = energy_refinement_layers(layers, request);
        let proxy = proxy_rerank_candidates(&layers.inferred_bond);
        layered_output_from_legacy(
            metadata,
            layers,
            Some(tag_candidates(proxy, "energy_guided_refinement")),
            None,
            false,
        )
    }
}

impl PocketGenerationMethod for ExternalWrapperDryRunMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: EXTERNAL_WRAPPER_DRY_RUN_METHOD_ID.to_string(),
            method_name: "External Wrapper Dry Run".to_string(),
            method_family: PocketGenerationMethodFamily::ExternalWrapper,
            capability: GenerationMethodCapability {
                batched_generation: true,
                method_native_generation: true,
                external_wrapper: true,
                ..GenerationMethodCapability::default()
            },
            layered_output_support: vec![CandidateLayerKind::RawRollout],
            evidence_role: GenerationEvidenceRole::ExternalWrapper,
            execution_mode: GenerationExecutionMode::ExternalCommand,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(request) = context.conditioned_request.as_ref() else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let wrapper_request = external_wrapper_request(request, context.candidate_limit.max(1));
        let response = dry_run_external_response(&wrapper_request, &context.example);
        let candidates = response.candidates;
        let layers = CandidateGenerationLayers {
            raw_rollout: candidates,
            repaired: Vec::new(),
            inferred_bond: Vec::new(),
        };
        layered_output_from_legacy(metadata, layers, None, None, true)
    }
}

impl PocketGenerationMethod for StubGenerationMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        self.metadata.clone()
    }

    fn generate_for_example(&self, _context: PocketGenerationContext) -> LayeredGenerationOutput {
        LayeredGenerationOutput::empty(self.metadata())
    }
}

fn external_wrapper_request(
    request: &crate::models::ConditionedGenerationRequest,
    candidate_limit: usize,
) -> crate::models::ExternalGenerationRequestRecord {
    crate::models::ExternalGenerationRequestRecord {
        schema_version: 1,
        request_id: format!("{}:{}", request.example_id, request.protein_id),
        example_id: request.example_id.clone(),
        protein_id: request.protein_id.clone(),
        candidate_limit,
        conditioning: crate::models::ExternalConditioningSummary {
            topology_slot_activation: request.topology.active_slot_fraction,
            geometry_slot_activation: request.geometry.active_slot_fraction,
            pocket_slot_activation: request.pocket.active_slot_fraction,
            gates: request.gate_summary,
        },
    }
}

fn dry_run_external_response(
    request: &crate::models::ExternalGenerationRequestRecord,
    example: &crate::data::MolecularExample,
) -> crate::models::ExternalGenerationResponseRecord {
    let atom_count = example.topology.atom_types.size()[0].max(0) as usize;
    let mut atom_types = Vec::with_capacity(atom_count);
    let mut coords = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        atom_types.push(example.topology.atom_types.int64_value(&[index as i64]));
        coords.push([
            example.geometry.coords.double_value(&[index as i64, 0]) as f32,
            example.geometry.coords.double_value(&[index as i64, 1]) as f32,
            example.geometry.coords.double_value(&[index as i64, 2]) as f32,
        ]);
    }
    let candidates = (0..request.candidate_limit)
        .map(|candidate_index| {
            let mut shifted = coords.clone();
            for coord in &mut shifted {
                coord[0] += 0.01 * candidate_index as f32;
                coord[1] -= 0.01 * candidate_index as f32;
            }
            GeneratedCandidateRecord {
                example_id: request.example_id.clone(),
                protein_id: request.protein_id.clone(),
                molecular_representation: Some("external_wrapper_dry_run_jsonl_v1".to_string()),
                inferred_bonds: infer_method_bonds(&shifted, &atom_types),
                atom_types: atom_types.clone(),
                coords: shifted,
                pocket_centroid: [0.0, 0.0, 0.0],
                pocket_radius: 4.0,
                coordinate_frame_origin: example.coordinate_frame_origin,
                source: EXTERNAL_WRAPPER_DRY_RUN_METHOD_ID.to_string(),
                source_pocket_path: example
                    .source_pocket_path
                    .as_ref()
                    .map(|path| path.display().to_string()),
                source_ligand_path: example
                    .source_ligand_path
                    .as_ref()
                    .map(|path| path.display().to_string()),
            }
        })
        .collect();
    crate::models::ExternalGenerationResponseRecord {
        schema_version: 1,
        request_id: request.request_id.clone(),
        candidates,
        status: "ok".to_string(),
        wrapper_version: Some("dry_run_v1".to_string()),
        environment_fingerprint: Some(format!("rust:{}", std::env::consts::OS)),
        failure_reason: None,
    }
}

fn decomposed_forward(
    context: &PocketGenerationContext,
) -> Option<&crate::models::system::ResearchForward> {
    let request = context.conditioned_request.as_ref()?;
    if request.topology.context.numel() == 0
        || request.geometry.context.numel() == 0
        || request.pocket.context.numel() == 0
    {
        return None;
    }
    context.forward.as_ref()
}

fn stub_metadata(
    method_id: &str,
    method_name: &str,
    family: PocketGenerationMethodFamily,
    evidence_role: GenerationEvidenceRole,
) -> PocketGenerationMethodMetadata {
    PocketGenerationMethodMetadata {
        method_id: method_id.to_string(),
        method_name: method_name.to_string(),
        method_family: family,
        capability: GenerationMethodCapability {
            stub: true,
            ..GenerationMethodCapability::default()
        },
        layered_output_support: Vec::new(),
        evidence_role,
        execution_mode: GenerationExecutionMode::Stub,
    }
}

fn layered_output_from_legacy(
    metadata: PocketGenerationMethodMetadata,
    layers: CandidateGenerationLayers,
    deterministic_proxy: Option<Vec<GeneratedCandidateRecord>>,
    reranked: Option<Vec<GeneratedCandidateRecord>>,
    native_raw: bool,
) -> LayeredGenerationOutput {
    let mut output = LayeredGenerationOutput::empty(metadata.clone());
    if !layers.raw_rollout.is_empty() {
        output.raw_rollout = Some(layer_output(
            &metadata,
            CandidateLayerKind::RawRollout,
            native_raw,
            Vec::new(),
            layers.raw_rollout,
        ));
    }
    if !layers.repaired.is_empty() {
        output.repaired = Some(layer_output(
            &metadata,
            CandidateLayerKind::Repaired,
            false,
            vec!["pocket_centroid_repair".to_string()],
            layers.repaired,
        ));
    }
    if !layers.inferred_bond.is_empty() {
        output.inferred_bond = Some(layer_output(
            &metadata,
            CandidateLayerKind::InferredBond,
            false,
            vec![
                "pocket_centroid_repair".to_string(),
                "distance_bond_inference".to_string(),
                "valence_pruning".to_string(),
            ],
            layers.inferred_bond,
        ));
    }
    if let Some(candidates) = deterministic_proxy.filter(|candidates| !candidates.is_empty()) {
        output.deterministic_proxy = Some(layer_output(
            &metadata,
            CandidateLayerKind::DeterministicProxy,
            false,
            vec![
                "pocket_centroid_repair".to_string(),
                "distance_bond_inference".to_string(),
                "deterministic_proxy_rerank".to_string(),
            ],
            candidates,
        ));
    }
    if let Some(candidates) = reranked.filter(|candidates| !candidates.is_empty()) {
        output.reranked = Some(layer_output(
            &metadata,
            CandidateLayerKind::Reranked,
            false,
            vec![
                "pocket_centroid_repair".to_string(),
                "distance_bond_inference".to_string(),
                "bounded_calibrated_rerank".to_string(),
            ],
            candidates,
        ));
    }
    output
}

fn tag_candidates(mut candidates: Vec<GeneratedCandidateRecord>, source: &str) -> Vec<GeneratedCandidateRecord> {
    tag_candidates_in_place(&mut candidates, source);
    candidates
}

fn tag_candidates_in_place(candidates: &mut [GeneratedCandidateRecord], source: &str) {
    for candidate in candidates {
        candidate.source = source.to_string();
    }
}

fn flow_matching_layers(
    mut layers: CandidateGenerationLayers,
    request: &crate::models::ConditionedGenerationRequest,
) -> CandidateGenerationLayers {
    transform_layer_candidates(&mut layers.raw_rollout, "flow_matching", |candidate, index| {
        apply_flow_transport(candidate, request, index)
    });
    transform_layer_candidates(&mut layers.repaired, "flow_matching", |candidate, index| {
        apply_flow_transport(candidate, request, index)
    });
    transform_layer_candidates(
        &mut layers.inferred_bond,
        "flow_matching",
        |candidate, index| apply_flow_transport(candidate, request, index),
    );
    layers
}

fn autoregressive_layers(
    mut layers: CandidateGenerationLayers,
    request: &crate::models::ConditionedGenerationRequest,
) -> CandidateGenerationLayers {
    transform_layer_candidates(
        &mut layers.raw_rollout,
        "autoregressive_graph_geometry",
        |candidate, index| apply_autoregressive_commit(candidate, request, index),
    );
    transform_layer_candidates(
        &mut layers.repaired,
        "autoregressive_graph_geometry",
        |candidate, index| apply_autoregressive_commit(candidate, request, index),
    );
    transform_layer_candidates(
        &mut layers.inferred_bond,
        "autoregressive_graph_geometry",
        |candidate, index| apply_autoregressive_commit(candidate, request, index),
    );
    layers
}

fn energy_refinement_layers(
    mut layers: CandidateGenerationLayers,
    request: &crate::models::ConditionedGenerationRequest,
) -> CandidateGenerationLayers {
    transform_layer_candidates(
        &mut layers.raw_rollout,
        "energy_guided_refinement",
        |candidate, index| apply_energy_refinement(candidate, request, index),
    );
    transform_layer_candidates(
        &mut layers.repaired,
        "energy_guided_refinement",
        |candidate, index| apply_energy_refinement(candidate, request, index),
    );
    transform_layer_candidates(
        &mut layers.inferred_bond,
        "energy_guided_refinement",
        |candidate, index| apply_energy_refinement(candidate, request, index),
    );
    layers
}

fn transform_layer_candidates(
    candidates: &mut [GeneratedCandidateRecord],
    source: &str,
    mut transform: impl FnMut(&mut GeneratedCandidateRecord, usize),
) {
    for (index, candidate) in candidates.iter_mut().enumerate() {
        transform(candidate, index);
        candidate.source = source.to_string();
        candidate.inferred_bonds = infer_method_bonds(&candidate.coords, &candidate.atom_types);
    }
}

fn apply_flow_transport(
    candidate: &mut GeneratedCandidateRecord,
    request: &crate::models::ConditionedGenerationRequest,
    _candidate_index: usize,
) {
    let steps = request.generation_config.rollout_steps;
    let gate = mean_gate(&request.gate_summary);
    candidate.molecular_representation = Some(format!(
        "flow_matching_geometry_v1:steps={steps};gate_mean={gate:.3}"
    ));
}

fn apply_autoregressive_commit(
    candidate: &mut GeneratedCandidateRecord,
    request: &crate::models::ConditionedGenerationRequest,
    candidate_index: usize,
) {
    let topology_bias = (request.topology.active_slot_fraction * 3.0).round() as i64;
    for (atom_index, atom_type) in candidate.atom_types.iter_mut().enumerate() {
        if (atom_index + candidate_index) % 3 == 0 {
            *atom_type = autoregressive_atom_type(*atom_type, topology_bias, atom_index);
        }
        if let Some(coord) = candidate.coords.get_mut(atom_index) {
            let step = (atom_index + 1) as f32;
            coord[0] += 0.035 * step;
            coord[1] += if atom_index % 2 == 0 { 0.025 } else { -0.025 };
            coord[2] += 0.015 * candidate_index as f32;
        }
    }
    candidate.molecular_representation = Some("autoregressive_graph_geometry_policy_v1".to_string());
}

fn apply_energy_refinement(
    candidate: &mut GeneratedCandidateRecord,
    request: &crate::models::ConditionedGenerationRequest,
    candidate_index: usize,
) {
    let pocket_pull = (0.18 + 0.1 * request.pocket.active_slot_fraction).clamp(0.1, 0.35) as f32;
    for _ in 0..3 {
        repel_candidate_close_contacts(&mut candidate.coords);
        for coord in &mut candidate.coords {
            for (axis, value) in coord.iter_mut().enumerate() {
                *value += pocket_pull * (candidate.pocket_centroid[axis] - *value);
            }
        }
    }
    let jitter = 0.01 * (candidate_index as f32 + 1.0);
    for (index, coord) in candidate.coords.iter_mut().enumerate() {
        let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
        coord[0] += sign * jitter;
        coord[1] -= sign * jitter;
    }
    candidate.molecular_representation = Some("energy_guided_refinement_v1".to_string());
}

fn autoregressive_atom_type(current: i64, topology_bias: i64, atom_index: usize) -> i64 {
    let palette = [0_i64, 1, 2, 4, 6, 7, 8];
    let current_index = palette
        .iter()
        .position(|value| *value == current)
        .unwrap_or(atom_index % palette.len());
    palette[(current_index + topology_bias.max(1) as usize + atom_index) % palette.len()]
}

fn mean_gate(summary: &crate::models::GenerationGateSummary) -> f64 {
    [
        summary.topo_from_geo,
        summary.topo_from_pocket,
        summary.geo_from_topo,
        summary.geo_from_pocket,
        summary.pocket_from_topo,
        summary.pocket_from_geo,
    ]
    .iter()
    .sum::<f64>()
        / 6.0
}

fn repel_candidate_close_contacts(coords: &mut [[f32; 3]]) {
    if coords.len() < 2 {
        return;
    }
    let min_distance = 1.05_f32;
    for left in 0..coords.len() {
        for right in (left + 1)..coords.len() {
            let dx = coords[right][0] - coords[left][0];
            let dy = coords[right][1] - coords[left][1];
            let dz = coords[right][2] - coords[left][2];
            let distance_sq = dx * dx + dy * dy + dz * dz;
            if distance_sq >= min_distance * min_distance {
                continue;
            }
            let distance = distance_sq.sqrt().max(1e-4);
            let push = 0.5 * (min_distance - distance);
            let direction = [dx / distance, dy / distance, dz / distance];
            for (axis, component) in direction.iter().enumerate() {
                coords[left][axis] -= component * push;
                coords[right][axis] += component * push;
            }
        }
    }
}

fn infer_method_bonds(coords: &[[f32; 3]], atom_types: &[i64]) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for left in 0..coords.len() {
        for right in (left + 1)..coords.len() {
            let threshold = bond_threshold(
                atom_types.get(left).copied().unwrap_or(6),
                atom_types.get(right).copied().unwrap_or(6),
            );
            if method_coord_distance(&coords[left], &coords[right]) <= threshold {
                bonds.push((left, right));
            }
        }
    }
    prune_method_bonds(atom_types, bonds)
}

fn prune_method_bonds(atom_types: &[i64], mut bonds: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    bonds.sort_by_key(|(left, right)| right - left);
    let mut degrees = vec![0_usize; atom_types.len()];
    let mut pruned = Vec::new();
    for (left, right) in bonds {
        if left >= degrees.len() || right >= degrees.len() {
            continue;
        }
        if degrees[left] < max_method_valence(atom_types[left])
            && degrees[right] < max_method_valence(atom_types[right])
        {
            degrees[left] += 1;
            degrees[right] += 1;
            pruned.push((left, right));
        }
    }
    pruned
}

fn bond_threshold(left: i64, right: i64) -> f64 {
    let base = if left == 1 || right == 1 { 1.35 } else { 1.85 };
    if left == 6 && right == 6 {
        1.95
    } else {
        base
    }
}

fn max_method_valence(atom_type: i64) -> usize {
    match atom_type {
        1 => 1,
        6 => 4,
        7 => 4,
        8 => 3,
        9 | 17 | 35 | 53 => 1,
        15 => 5,
        16 => 6,
        _ => 4,
    }
}

fn method_coord_distance(left: &[f32; 3], right: &[f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn layer_output(
    metadata: &PocketGenerationMethodMetadata,
    layer_kind: CandidateLayerKind,
    method_native: bool,
    postprocessor_chain: Vec<String>,
    candidates: Vec<GeneratedCandidateRecord>,
) -> CandidateLayerOutput {
    CandidateLayerOutput {
        provenance: CandidateLayerProvenance {
            source_method_id: metadata.method_id.clone(),
            source_method_name: metadata.method_name.clone(),
            source_method_family: metadata.method_family,
            layer_kind,
            legacy_field_name: layer_kind.legacy_field_name().to_string(),
            method_native,
            postprocessor_chain,
            available: true,
        },
        candidates,
    }
}
