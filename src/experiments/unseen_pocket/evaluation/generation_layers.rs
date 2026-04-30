fn disabled_real_generation_metrics() -> RealGenerationMetrics {
    RealGenerationMetrics {
        chemistry_validity: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "chemistry-validity backend unavailable for this evaluation".to_string(),
        },
        docking_affinity: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "docking or affinity backend unavailable for this evaluation".to_string(),
        },
        pocket_compatibility: ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: "pocket-compatibility backend unavailable for this evaluation".to_string(),
        },
    }
}

fn empty_layered_generation_metrics() -> LayeredGenerationMetrics {
    let empty = summarize_candidate_layer(&[], &NoveltyReferenceSignatures::default());
    LayeredGenerationMetrics {
        raw_flow: empty.clone(),
        constrained_flow: empty.clone(),
        repaired: empty.clone(),
        raw_rollout: empty.clone(),
        repaired_candidates: empty.clone(),
        inferred_bond_candidates: empty.clone(),
        reranked_candidates: empty.clone(),
        deterministic_proxy_candidates: empty,
        reranker_calibration: RerankerCalibrationReport::default(),
        backend_scored_candidates: BTreeMap::new(),
        method_comparison: MethodComparisonSummary::default(),
        flow_head_ablation: FlowHeadAblationDiagnostics::default(),
        flow_head_diagnostics: BTreeMap::new(),
        generation_path_contract: canonical_generation_path_contract(),
        repair_case_audit: RepairCaseAuditReport::default(),
    }
}

fn planned_metric_interfaces() -> Vec<PlannedMetricInterface> {
    vec![
        PlannedMetricInterface {
            interface_id: "chemistry_property_bundle".to_string(),
            description:
                "Backend-agnostic chemistry property interface reserved for future method comparison."
                    .to_string(),
            backend_agnostic: true,
        },
        PlannedMetricInterface {
            interface_id: "scaffold_novelty_bundle".to_string(),
            description:
                "Backend-agnostic scaffold novelty interface reserved for future method comparison."
                    .to_string(),
            backend_agnostic: true,
        },
    ]
}

fn build_method_comparison_summary(
    research: &ResearchConfig,
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    ablation: &AblationConfig,
    example_limit: usize,
) -> MethodComparisonSummary {
    let mut configured_method_ids = Vec::new();
    configured_method_ids.push(research.generation_method.primary_backend_id().to_string());
    if research.generation_method.enable_comparison_runner {
        configured_method_ids.extend(research.generation_method.comparison_backend_ids());
    }
    if research.preference_alignment.enable_preference_reranking {
        configured_method_ids.push("preference_aware_reranker".to_string());
    }
    let mut seen = std::collections::BTreeSet::new();
    let method_ids = configured_method_ids
        .into_iter()
        .filter(|method_id| seen.insert(method_id.clone()))
        .collect::<Vec<_>>();
    let contexts = examples
        .iter()
        .zip(forwards.iter())
        .take(example_limit)
        .map(|(example, forward)| PocketGenerationContext {
            example: example.clone(),
            conditioned_request: None,
            forward: Some(forward.clone()),
            candidate_limit: research.generation_method.candidate_count.max(1),
            enable_repair: !ablation.disable_candidate_repair,
        })
        .map(|context| context.with_conditioned_request(&research.data.generation_target))
        .collect::<Vec<_>>();

    let mut flow_raw: Option<CandidateLayerMetrics> = None;
    let mut flow_repaired: Option<CandidateLayerMetrics> = None;
    let methods = method_ids
        .iter()
        .filter_map(|method_id| PocketGenerationMethodRegistry::build(method_id).ok())
        .map(|method| {
            let metadata = method.metadata();
            let started = std::time::Instant::now();
            let outputs = method.generate_batch(contexts.clone());
            let merged = merge_method_outputs(metadata, outputs);
            if merged.metadata.method_id == "flow_matching" {
                if let Some(raw) = merged.raw_rollout.as_ref() {
                    flow_raw = Some(summarize_candidate_layer(
                        &raw.candidates,
                        &NoveltyReferenceSignatures::default(),
                    ));
                }
                if let Some(repaired) = merged.repaired.as_ref() {
                    flow_repaired = Some(summarize_candidate_layer(
                        &repaired.candidates,
                        &NoveltyReferenceSignatures::default(),
                    ));
                }
            }
            let mut row = summarize_method_output(&merged);
            row.wall_time_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            row.sampling_steps = sampling_steps_for_method(research, &row.method_id);
            row.slot_activation_mean = shared_slot_activation_mean(&contexts);
            row.gate_activation_mean = shared_gate_activation_mean(&contexts);
            row
        })
        .collect::<Vec<_>>();
    let flow_vs_denoising = flow_vs_denoising_delta(&methods);
    let active_method = PocketGenerationMethodRegistry::metadata(
        research.generation_method.primary_backend_id(),
    )
    .ok();
    let active_row = active_method
        .as_ref()
        .and_then(|metadata| methods.iter().find(|row| row.method_id == metadata.method_id));

    MethodComparisonSummary {
        active_method: active_method.clone(),
        active_method_family: active_row
            .map(|row| row.method_family.clone())
            .or_else(|| {
                active_method
                    .as_ref()
                    .map(|metadata| format!("{:?}", metadata.method_family).to_ascii_lowercase())
            }),
        raw_native_evidence: ClaimRawNativeEvidenceSummary::default(),
        processed_generation_evidence: ClaimProcessedGenerationEvidenceSummary::default(),
        active_selected_metric_layer: active_row.and_then(|row| row.selected_metric_layer.clone()),
        methods,
        planned_metric_interfaces: planned_metric_interfaces(),
        preference_alignment: PreferenceAlignmentSummary::default(),
        flow_metrics: FlowMethodMetrics {
            raw_output: flow_raw,
            repaired_output: flow_repaired,
            versus_conditioned_denoising: flow_vs_denoising,
        },
    }
}

fn shared_slot_activation_mean(contexts: &[PocketGenerationContext]) -> Option<f64> {
    let values = contexts
        .iter()
        .filter_map(|context| context.conditioned_request.as_ref())
        .map(|request| {
            (request.topology.active_slot_fraction
                + request.geometry.active_slot_fraction
                + request.pocket.active_slot_fraction)
                / 3.0
        })
        .collect::<Vec<_>>();
    mean_finite(&values)
}

fn shared_gate_activation_mean(contexts: &[PocketGenerationContext]) -> Option<f64> {
    let values = contexts
        .iter()
        .filter_map(|context| context.conditioned_request.as_ref())
        .map(|request| {
            let gates = &request.gate_summary;
            (gates.topo_from_geo
                + gates.topo_from_pocket
                + gates.geo_from_topo
                + gates.geo_from_pocket
                + gates.pocket_from_topo
                + gates.pocket_from_geo)
                / 6.0
        })
        .collect::<Vec<_>>();
    mean_finite(&values)
}

fn mean_finite(values: &[f64]) -> Option<f64> {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    (!finite.is_empty()).then(|| finite.iter().sum::<f64>() / finite.len() as f64)
}

fn flow_head_ablation_diagnostics(
    research: &ResearchConfig,
    diagnostics_available: bool,
) -> FlowHeadAblationDiagnostics {
    let flow_contract =
        crate::models::flow::current_multimodal_flow_contract(&research.generation_method.flow_matching);
    let head_kind = flow_head_kind_label(research.model.flow_velocity_head.kind);
    let pairwise_geometry_enabled = research.model.pairwise_geometry.enabled;
    let equivariant_geometry_head = research.model.flow_velocity_head.kind
        == crate::config::FlowVelocityHeadKind::EquivariantGeometry;
    let local_atom_pocket_attention = research.model.flow_velocity_head.kind
        == crate::config::FlowVelocityHeadKind::AtomPocketCrossAttention;
    let decoder_conditioning_kind =
        decoder_conditioning_kind_label(research.model.decoder_conditioning.kind);
    let slot_local_conditioning_enabled = research.model.decoder_conditioning.kind
        == crate::config::DecoderConditioningKind::LocalAtomSlotAttention;
    let ablation_label = match (
        equivariant_geometry_head,
        local_atom_pocket_attention,
        pairwise_geometry_enabled,
    ) {
        (true, _, false) => "equivariant_geometry",
        (true, _, true) => "equivariant_pairwise_geometry",
        (false, false, false) => "geometry_mean_pooling",
        (false, false, true) => "pairwise_geometry",
        (false, true, false) => "local_pocket_attention",
        (false, true, true) => "pairwise_plus_local_pocket",
    };
    let claim_boundary = flow_head_claim_boundary(
        &flow_contract,
        research.generation_method.flow_matching.geometry_only,
    );
    FlowHeadAblationDiagnostics {
        schema_version: 1,
        head_kind: head_kind.to_string(),
        local_atom_pocket_attention,
        equivariant_geometry_head,
        pairwise_geometry_enabled,
        ablation_label: ablation_label.to_string(),
        decoder_conditioning_kind: decoder_conditioning_kind.to_string(),
        molecular_flow_conditioning_kind: decoder_conditioning_kind.to_string(),
        slot_local_conditioning_enabled,
        mean_pooled_conditioning_ablation: !slot_local_conditioning_enabled,
        diagnostics_available,
        claim_boundary,
        enabled_flow_branches: flow_contract.enabled_branches,
        disabled_flow_branches: flow_contract.disabled_branches,
        full_molecular_flow_claim_allowed: flow_contract.full_molecular_flow_claim_allowed,
        claim_gate_reason: flow_contract.claim_gate_reason,
        target_alignment_policy: flow_contract.target_alignment_policy,
        target_matching_claim_safe: flow_contract.target_matching_claim_safe,
        target_matching_artifact_fields: vec![
            "training_history[].losses.primary.branch_schedule.entries[].target_matching_policy"
                .to_string(),
            "training_history[].losses.primary.branch_schedule.entries[].target_matching_coverage"
                .to_string(),
            "training_history[].losses.primary.branch_schedule.entries[].target_matching_mean_cost"
                .to_string(),
            "training_history[].losses.primary.branch_schedule.entries[].unweighted_value"
                .to_string(),
            "training_history[].losses.primary.branch_schedule.entries[].weighted_value"
                .to_string(),
            "training_history[].losses.primary.branch_schedule.entries[].schedule_multiplier"
                .to_string(),
        ],
    }
}

fn flow_head_claim_boundary(
    flow_contract: &crate::models::flow::MultiModalFlowContract,
    geometry_only: bool,
) -> String {
    if flow_contract.full_molecular_flow_claim_allowed {
        return "full multi-modal molecular flow active: geometry, atom_type, bond, topology, and pocket_context branches are optimizer-facing; the velocity-head label identifies only the coordinate subhead"
            .to_string();
    }
    if geometry_only {
        return "geometry-only coordinate velocity baseline; topology, bond, atom-type, and pocket-context flow branches are not active for model-quality claims"
            .to_string();
    }
    if flow_contract.disabled_branches.is_empty() {
        return format!(
            "multi-modal branch set active but full-flow claim gate is blocked by {}; inspect target matching, branch weights, and schedule before citing full molecular-flow metrics",
            flow_contract.claim_gate_reason
        );
    }
    format!(
        "partial multi-modal flow active with missing branch(es) [{}]; metrics are not full molecular-flow evidence",
        flow_contract.disabled_branches.join(", ")
    )
}

fn decoder_conditioning_kind_label(kind: crate::config::DecoderConditioningKind) -> &'static str {
    match kind {
        crate::config::DecoderConditioningKind::MeanPooled => "mean_pooled",
        crate::config::DecoderConditioningKind::LocalAtomSlotAttention => {
            "local_atom_slot_attention"
        }
    }
}

fn flow_head_kind_label(kind: crate::config::FlowVelocityHeadKind) -> &'static str {
    match kind {
        crate::config::FlowVelocityHeadKind::Geometry => "geometry",
        crate::config::FlowVelocityHeadKind::EquivariantGeometry => "equivariant_geometry",
        crate::config::FlowVelocityHeadKind::AtomPocketCrossAttention => {
            "atom_pocket_cross_attention"
        }
    }
}

fn flow_vs_denoising_delta(methods: &[MethodComparisonRow]) -> Option<FlowVsDenoisingDelta> {
    let flow = methods
        .iter()
        .find(|row| row.method_id == "flow_matching")?;
    let denoising = methods
        .iter()
        .find(|row| row.method_id == "conditioned_denoising")?;
    Some(FlowVsDenoisingDelta {
        native_valid_fraction_delta: flow.native_valid_fraction.unwrap_or(0.0)
            - denoising.native_valid_fraction.unwrap_or(0.0),
        native_pocket_contact_fraction_delta: flow.native_pocket_contact_fraction.unwrap_or(0.0)
            - denoising.native_pocket_contact_fraction.unwrap_or(0.0),
        native_clash_fraction_delta: flow.native_clash_fraction.unwrap_or(0.0)
            - denoising.native_clash_fraction.unwrap_or(0.0),
    })
}

fn sampling_steps_for_method(research: &ResearchConfig, method_id: &str) -> Option<usize> {
    if method_id == "flow_matching" {
        return Some(research.generation_method.flow_matching.steps.max(1));
    }
    if research.generation_method.primary_backend_id() == method_id {
        return research
            .generation_method
            .primary_backend
            .sampling_steps
            .or(Some(research.data.generation_target.rollout_steps));
    }
    research
        .generation_method
        .comparison_backends
        .iter()
        .find(|backend| backend.backend_id == method_id)
        .and_then(|backend| backend.sampling_steps)
        .or(Some(research.data.generation_target.rollout_steps))
}

fn merge_method_outputs(
    metadata: PocketGenerationMethodMetadata,
    outputs: Vec<LayeredGenerationOutput>,
) -> LayeredGenerationOutput {
    let mut merged = LayeredGenerationOutput::empty(metadata);
    for mut output in outputs {
        merge_layer(&mut merged.raw_rollout, output.raw_rollout.take());
        merge_layer(&mut merged.repaired, output.repaired.take());
        merge_layer(&mut merged.inferred_bond, output.inferred_bond.take());
        merge_layer(
            &mut merged.deterministic_proxy,
            output.deterministic_proxy.take(),
        );
        merge_layer(&mut merged.reranked, output.reranked.take());
    }
    merged
}

fn merge_layer(
    target: &mut Option<crate::models::CandidateLayerOutput>,
    next: Option<crate::models::CandidateLayerOutput>,
) {
    match (target, next) {
        (Some(existing), Some(mut next_layer)) => {
            existing.candidates.append(&mut next_layer.candidates);
        }
        (slot @ None, Some(next_layer)) => {
            *slot = Some(next_layer);
        }
        _ => {}
    }
}

fn evaluate_real_generation_metrics(
    examples: &[crate::data::MolecularExample],
    train_examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    research: &ResearchConfig,
    ablation: &AblationConfig,
    external_evaluation: &ExternalEvaluationConfig,
    split_label: &str,
) -> (
    RealGenerationMetrics,
    LayeredGenerationMetrics,
    MethodComparisonSummary,
) {
    let candidate_limit = research.generation_method.candidate_count.max(1);
    let active_method =
        PocketGenerationMethodRegistry::build(research.generation_method.primary_backend_id())
            .unwrap_or_else(|message| {
                panic!(
                    "invalid generation_method.primary_backend_id `{}` reached evaluation after validation: {message}",
                    research.generation_method.primary_backend_id()
                )
            });
    let method_outputs = active_method.generate_batch(
        examples
            .iter()
            .zip(forwards.iter())
            .take(external_evaluation.generation_artifact_example_limit)
            .map(|(example, forward)| PocketGenerationContext {
                example: example.clone(),
                conditioned_request: None,
                forward: Some(forward.clone()),
                candidate_limit,
                enable_repair: !ablation.disable_candidate_repair,
            })
            .map(|context| context.with_conditioned_request(&research.data.generation_target))
            .collect(),
    );
    let active_output = merge_method_outputs(active_method.metadata(), method_outputs);
    let (raw_rollout, repaired, candidates, proxy_reranked, reranked) =
        flatten_layered_output(&active_output);
    let raw_rollout = raw_rollout
        .into_iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .collect::<Vec<_>>();
    let repaired = repaired
        .into_iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .collect::<Vec<_>>();
    let candidates = candidates
        .into_iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .collect::<Vec<_>>();
    let proxy_reranked = proxy_reranked
        .into_iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .collect::<Vec<_>>();
    let reranked = reranked
        .into_iter()
        .take(external_evaluation.generation_artifact_candidate_limit)
        .collect::<Vec<_>>();
    let backend_selection = select_backend_candidate_layer(
        research,
        &raw_rollout,
        &candidates,
        &reranked,
        &proxy_reranked,
    );
    let backend_candidates = backend_selection.candidates.clone();
    let novelty_reference = novelty_reference_signatures(train_examples);
    let calibrated_reranker = CalibratedReranker::fit(&candidates);
    let mut method_comparison = build_method_comparison_summary(
        research,
        examples,
        forwards,
        ablation,
        external_evaluation.generation_artifact_example_limit,
    );

    let raw_flow_metrics = summarize_candidate_layer(&raw_rollout, &novelty_reference);
    let repaired_metrics = summarize_candidate_layer(&repaired, &novelty_reference);
    let constrained_flow_metrics = summarize_candidate_layer(&candidates, &novelty_reference);
    let reranked_metrics = summarize_candidate_layer(&reranked, &novelty_reference);
    let deterministic_proxy_metrics =
        summarize_candidate_layer(&proxy_reranked, &novelty_reference);
    let mut layered = LayeredGenerationMetrics {
        raw_flow: raw_flow_metrics.clone(),
        constrained_flow: constrained_flow_metrics.clone(),
        repaired: repaired_metrics.clone(),
        raw_rollout: raw_flow_metrics,
        repaired_candidates: repaired_metrics,
        inferred_bond_candidates: constrained_flow_metrics,
        reranked_candidates: reranked_metrics,
        deterministic_proxy_candidates: deterministic_proxy_metrics,
        reranker_calibration: calibrated_reranker.report(),
        backend_scored_candidates: BTreeMap::new(),
        method_comparison: method_comparison.clone(),
        flow_head_ablation: flow_head_ablation_diagnostics(research, false),
        flow_head_diagnostics: BTreeMap::new(),
        generation_path_contract: canonical_generation_path_contract(),
        repair_case_audit: RepairCaseAuditReport::default(),
    };
    apply_raw_rollout_stability(&mut layered.raw_rollout, forwards);
    layered.flow_head_diagnostics = aggregate_final_flow_diagnostics(forwards);
    layered.flow_head_ablation =
        flow_head_ablation_diagnostics(research, !layered.flow_head_diagnostics.is_empty());
    layered.raw_flow = layered.raw_rollout.clone();
    layered.repair_case_audit = build_repair_case_audit(
        split_label,
        &raw_rollout,
        &repaired,
        &layered.raw_rollout,
        &layered.repaired_candidates,
    );

    if candidates.is_empty() {
        let disabled = disabled_real_generation_metrics();
        let preference_summary = build_and_persist_preference_artifacts(
            research,
            split_label,
            &raw_rollout,
            &repaired,
            &candidates,
            &proxy_reranked,
            &reranked,
            &disabled,
        );
        method_comparison.preference_alignment = preference_summary.clone();
        layered.method_comparison.preference_alignment = preference_summary;
        maybe_persist_generation_artifacts(
            research,
            external_evaluation,
            split_label,
            &raw_rollout,
            &repaired,
            &candidates,
            &reranked,
            &proxy_reranked,
            &disabled,
            &layered,
            &active_output,
        );
        return (disabled, layered, method_comparison);
    }

    let heuristic_chemistry =
        HeuristicChemistryValidityEvaluator.evaluate_chemistry(&backend_candidates);
    let chemistry = if external_evaluation.chemistry_backend.enabled {
        CommandChemistryValidityEvaluator {
            config: external_evaluation.chemistry_backend.clone(),
        }
        .evaluate_chemistry(&backend_candidates)
    } else {
        heuristic_chemistry.clone()
    };
    let heuristic_docking = HeuristicDockingEvaluator.evaluate_docking(&backend_candidates);
    let docking = if external_evaluation.docking_backend.enabled {
        CommandDockingEvaluator {
            config: external_evaluation.docking_backend.clone(),
        }
        .evaluate_docking(&backend_candidates)
    } else {
        heuristic_docking.clone()
    };
    let heuristic_pocket =
        HeuristicPocketCompatibilityEvaluator.evaluate_pocket_compatibility(&backend_candidates);
    let pocket = if external_evaluation.pocket_backend.enabled {
        CommandPocketCompatibilityEvaluator {
            config: external_evaluation.pocket_backend.clone(),
        }
        .evaluate_pocket_compatibility(&backend_candidates)
    } else {
        heuristic_pocket.clone()
    };

    let real_generation = RealGenerationMetrics {
        chemistry_validity: merge_backend_reports(
            chemistry,
            heuristic_chemistry,
            external_evaluation.chemistry_backend.enabled,
            backend_status(
                external_evaluation.chemistry_backend.enabled,
                "external chemistry-validity backend on modular rollout candidates",
                "active heuristic chemistry-validity backend on modular rollout candidates",
                &backend_selection,
            ),
        ),
        docking_affinity: merge_backend_reports(
            docking,
            heuristic_docking,
            external_evaluation.docking_backend.enabled,
            backend_status(
                external_evaluation.docking_backend.enabled,
                "external docking backend on modular rollout candidates",
                "active heuristic docking-oriented hook on modular rollout candidates",
                &backend_selection,
            ),
        ),
        pocket_compatibility: merge_backend_reports(
            pocket,
            heuristic_pocket,
            external_evaluation.pocket_backend.enabled,
            backend_status(
                external_evaluation.pocket_backend.enabled,
                "external pocket-compatibility backend on modular rollout candidates",
                "active heuristic pocket-compatibility hook on modular rollout candidates",
                &backend_selection,
            ),
        ),
    };
    layered.backend_scored_candidates = backend_metric_layers(&real_generation);
    let preference_summary = build_and_persist_preference_artifacts(
        research,
        split_label,
        &raw_rollout,
        &repaired,
        &candidates,
        &proxy_reranked,
        &reranked,
        &real_generation,
    );
    method_comparison.preference_alignment = preference_summary.clone();
    layered.method_comparison.preference_alignment = preference_summary;
    maybe_persist_generation_artifacts(
        research,
        external_evaluation,
        split_label,
        &raw_rollout,
        &repaired,
        &candidates,
        &reranked,
        &proxy_reranked,
        &real_generation,
        &layered,
        &active_output,
    );
    (real_generation, layered, method_comparison)
}

#[derive(Debug, Clone)]
struct BackendCandidateLayerSelection {
    candidates: Vec<GeneratedCandidateRecord>,
    layer_name: &'static str,
    model_native_raw: bool,
    reason: &'static str,
}

fn select_backend_candidate_layer(
    research: &ResearchConfig,
    raw_rollout: &[GeneratedCandidateRecord],
    inferred_bond: &[GeneratedCandidateRecord],
    reranked: &[GeneratedCandidateRecord],
    deterministic_proxy: &[GeneratedCandidateRecord],
) -> BackendCandidateLayerSelection {
    let flow_contract =
        crate::models::flow::current_multimodal_flow_contract(&research.generation_method.flow_matching);
    let full_branch_runtime = !research.generation_method.flow_matching.geometry_only
        && flow_contract.disabled_branches.is_empty();
    if full_branch_runtime && !raw_rollout.is_empty() {
        return BackendCandidateLayerSelection {
            candidates: raw_rollout.to_vec(),
            layer_name: "raw_rollout",
            model_native_raw: true,
            reason: if flow_contract.full_molecular_flow_claim_allowed {
                "full_molecular_flow_raw_rollout_preferred"
            } else {
                "full_branch_raw_rollout_preferred_claim_gate_blocked"
            },
        };
    }
    final_backend_candidate_layer_selection(inferred_bond, reranked, deterministic_proxy)
}

fn final_backend_candidate_layer_selection(
    inferred_bond: &[GeneratedCandidateRecord],
    reranked: &[GeneratedCandidateRecord],
    deterministic_proxy: &[GeneratedCandidateRecord],
) -> BackendCandidateLayerSelection {
    best_backend_compatible_candidate(reranked, 0)
        .into_iter()
        .chain(best_backend_compatible_candidate(deterministic_proxy, 1))
        .chain(best_backend_compatible_candidate(inferred_bond, 2))
        .max_by(|left, right| {
            left.selection_score
                .partial_cmp(&right.selection_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.layer_priority.cmp(&left.layer_priority))
        })
        .map(|candidate| BackendCandidateLayerSelection {
            candidates: vec![candidate.record],
            layer_name: candidate.layer_name,
            model_native_raw: candidate.model_native_raw,
            reason: "best_backend_compatible_candidate",
        })
        .or_else(|| {
            if !reranked.is_empty() {
                Some(BackendCandidateLayerSelection {
                    candidates: reranked.to_vec(),
                    layer_name: "reranked_candidates",
                    model_native_raw: false,
                    reason: "fallback_reranked_candidates",
                })
            } else if !deterministic_proxy.is_empty() {
                Some(BackendCandidateLayerSelection {
                    candidates: deterministic_proxy.to_vec(),
                    layer_name: "deterministic_proxy_candidates",
                    model_native_raw: false,
                    reason: "fallback_deterministic_proxy_candidates",
                })
            } else if !inferred_bond.is_empty() {
                Some(BackendCandidateLayerSelection {
                    candidates: inferred_bond.to_vec(),
                    layer_name: "inferred_bond_candidates",
                    model_native_raw: false,
                    reason: "fallback_inferred_bond_candidates",
                })
            } else {
                None
            }
        })
        .unwrap_or_else(|| BackendCandidateLayerSelection {
            candidates: Vec::new(),
            layer_name: "unavailable",
            model_native_raw: false,
            reason: "no_backend_candidates",
        })
}

#[derive(Debug)]
struct BackendSelectionCandidate {
    record: GeneratedCandidateRecord,
    selection_score: f64,
    layer_priority: usize,
    layer_name: &'static str,
    model_native_raw: bool,
}

fn best_backend_compatible_candidate(
    candidates: &[GeneratedCandidateRecord],
    layer_priority: usize,
) -> Option<BackendSelectionCandidate> {
    if candidates.is_empty() {
        return None;
    }
    candidates
        .iter()
        .filter_map(|candidate| {
            let clash_fraction = candidate_backend_pocket_clash_fraction(candidate)?;
            (clash_fraction <= 0.0).then(|| BackendSelectionCandidate {
                record: candidate.clone(),
                selection_score: backend_claim_selection_score(candidate),
                layer_priority,
                layer_name: backend_layer_name_for_priority(layer_priority),
                model_native_raw: false,
            })
        })
        .max_by(|left, right| {
            left.selection_score
                .partial_cmp(&right.selection_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn backend_layer_name_for_priority(layer_priority: usize) -> &'static str {
    match layer_priority {
        0 => "reranked_candidates",
        1 => "deterministic_proxy_candidates",
        2 => "inferred_bond_candidates",
        _ => "backend_selected_candidates",
    }
}

fn backend_status(
    external_enabled: bool,
    external: &str,
    heuristic: &str,
    selection: &BackendCandidateLayerSelection,
) -> String {
    let base = if external_enabled {
        external.to_string()
    } else {
        heuristic.to_string()
    };
    format!(
        "{base};metric_candidate_layer={};model_native_raw={};selection_reason={}",
        selection.layer_name, selection.model_native_raw, selection.reason
    )
}

fn merge_backend_reports(
    primary: crate::models::ExternalEvaluationReport,
    heuristic: crate::models::ExternalEvaluationReport,
    external_enabled: bool,
    enabled_status: impl Into<String>,
) -> ReservedBackendMetrics {
    let mut merged = report_to_metrics(primary, enabled_status);
    if external_enabled {
        for metric in heuristic.metrics {
            merged
                .metrics
                .insert(format!("heuristic_{}", metric.metric_name), metric.value);
        }
    }
    merged
}

fn backend_metric_layers(
    metrics: &RealGenerationMetrics,
) -> BTreeMap<String, BTreeMap<String, f64>> {
    BTreeMap::from([
        (
            "chemistry_validity".to_string(),
            metrics.chemistry_validity.metrics.clone(),
        ),
        (
            "docking_affinity".to_string(),
            metrics.docking_affinity.metrics.clone(),
        ),
        (
            "pocket_compatibility".to_string(),
            metrics.pocket_compatibility.metrics.clone(),
        ),
    ])
}

fn summarize_candidate_layer(
    candidates: &[GeneratedCandidateRecord],
    novelty_reference: &NoveltyReferenceSignatures,
) -> CandidateLayerMetrics {
    if candidates.is_empty() {
        return CandidateLayerMetrics {
            layer_name: "unavailable".to_string(),
            pocket_interaction_provenance: "unavailable".to_string(),
            generation_mode: "unavailable".to_string(),
            candidate_count: 0,
            valid_fraction: 0.0,
            pocket_contact_fraction: 0.0,
            pocket_distance_bin_accuracy: 0.0,
            pocket_contact_precision_proxy: 0.0,
            pocket_contact_recall_proxy: 0.0,
            pocket_role_compatibility_proxy: 0.0,
            mean_centroid_offset: 0.0,
            clash_fraction: 0.0,
            mean_displacement: 0.0,
            atom_change_fraction: 0.0,
            uniqueness_proxy_fraction: 0.0,
            diversity_eligible_fraction: 0.0,
            validity_conditioned_unique_fraction: 0.0,
            equivalence_duplicate_fraction: 0.0,
            invalid_diversity_excluded_fraction: 0.0,
            diversity_metric_source: "validity_conditioned_permutation_invariant_equivalence"
                .to_string(),
            atom_type_sequence_diversity: 0.0,
            bond_topology_diversity: 0.0,
            coordinate_shape_diversity: 0.0,
            novel_atom_type_sequence_fraction: 0.0,
            novel_bond_topology_fraction: 0.0,
            novel_coordinate_shape_fraction: 0.0,
            scaffold_novelty_fraction: 0.0,
            unique_scaffold_fraction: 0.0,
            pairwise_tanimoto_mean: 0.0,
            nearest_train_similarity: 0.0,
            scaffold_metric_coverage_fraction: 0.0,
            hydrogen_bond_proxy: 0.0,
            hydrophobic_contact_proxy: 0.0,
            residue_contact_count: 0.0,
            residue_identity_coverage_fraction: 0.0,
            key_residue_contact_coverage: 0.0,
            clash_burden: 0.0,
            contact_balance: 0.0,
            interaction_profile_coverage_fraction: 0.0,
            native_bond_count_mean: 0.0,
            native_component_count_mean: 0.0,
            native_valence_violation_fraction: 0.0,
            native_disconnected_fragment_fraction: 0.0,
            native_bond_order_conflict_fraction: 0.0,
            native_graph_repair_delta_mean: 0.0,
            native_raw_to_constrained_removed_bond_count_mean: 0.0,
            native_connectivity_guardrail_added_bond_count_mean: 0.0,
            native_valence_guardrail_downgrade_count_mean: 0.0,
            topology_bond_sync_fraction: 0.0,
            atom_type_entropy: 0.0,
            native_graph_valid_fraction: 0.0,
            native_graph_metric_source: "unavailable".to_string(),
        };
    }
    let total = candidates.len() as f64;
    let generation_mode = candidates
        .first()
        .map(|candidate| candidate.generation_mode.clone())
        .unwrap_or_else(|| "unavailable".to_string());
    let layer_name = candidates
        .first()
        .map(|candidate| candidate.generation_layer.clone())
        .unwrap_or_else(|| "unavailable".to_string());
    let pocket_interaction_provenance =
        candidate_layer_pocket_interaction_provenance(candidates.first());
    let valid_fraction = candidates
        .iter()
        .filter(|candidate| candidate_is_valid(candidate))
        .count() as f64
        / total;
    let pocket_contact_fraction = candidates
        .iter()
        .filter(|candidate| candidate_has_pocket_contact(candidate))
        .count() as f64
        / total;
    let pocket_distance_bin_accuracy = candidates
        .iter()
        .map(candidate_pocket_distance_bin_accuracy_proxy)
        .sum::<f64>()
        / total;
    let pocket_contact_recall_proxy = candidates
        .iter()
        .map(candidate_pocket_contact_recall_proxy)
        .sum::<f64>()
        / total;
    let mean_centroid_offset = candidates
        .iter()
        .map(candidate_centroid_offset)
        .filter(|value| value.is_finite())
        .sum::<f64>()
        / total;
    let clash_fraction = candidates.iter().map(candidate_clash_fraction).sum::<f64>() / total;
    let pocket_contact_precision_proxy =
        (pocket_contact_fraction * (1.0 - clash_fraction.clamp(0.0, 1.0))).clamp(0.0, 1.0);
    let pocket_role_compatibility_proxy = candidates
        .iter()
        .map(candidate_pocket_role_compatibility_proxy)
        .sum::<f64>()
        / total;
    let diversity = summarize_equivalence_diversity(candidates);
    let uniqueness_proxy_fraction = diversity.unique_fraction_all_candidates;
    let atom_type_sequence_diversity = diversity_fraction(candidates, candidate_atom_signature);
    let bond_topology_diversity = diversity_fraction(candidates, candidate_bond_signature);
    let coordinate_shape_diversity = diversity_fraction(candidates, candidate_shape_signature);
    let novel_atom_type_sequence_fraction = novelty_fraction(
        candidates,
        candidate_atom_signature,
        &novelty_reference.atom_signatures,
    );
    let novel_bond_topology_fraction = novelty_fraction(
        candidates,
        candidate_bond_signature,
        &novelty_reference.bond_signatures,
    );
    let novel_coordinate_shape_fraction = novelty_fraction(
        candidates,
        candidate_shape_signature,
        &novelty_reference.shape_signatures,
    );
    let scaffold_novelty_fraction = novelty_fraction(
        candidates,
        candidate_scaffold_signature,
        &novelty_reference.scaffold_signatures,
    );
    let unique_scaffold_fraction = diversity_fraction(candidates, candidate_scaffold_signature);
    let candidate_fingerprints = candidates
        .iter()
        .filter_map(candidate_structural_fingerprint)
        .collect::<Vec<_>>();
    let scaffold_metric_coverage_fraction = candidate_fingerprints.len() as f64 / total;
    let pairwise_tanimoto_mean = mean_pairwise_tanimoto(&candidate_fingerprints).unwrap_or(0.0);
    let nearest_train_similarity =
        mean_nearest_train_similarity(&candidate_fingerprints, &novelty_reference.fingerprints)
            .unwrap_or(0.0);
    let hydrogen_bond_proxy = candidates
        .iter()
        .map(candidate_hydrogen_bond_proxy)
        .sum::<f64>()
        / total;
    let hydrophobic_contact_proxy = candidates
        .iter()
        .map(candidate_hydrophobic_contact_proxy)
        .sum::<f64>()
        / total;
    let residue_contact_count = candidates
        .iter()
        .map(candidate_residue_contact_count_proxy)
        .sum::<f64>()
        / total;
    let residue_identity_coverage_fraction = candidates
        .iter()
        .filter(|candidate| {
            candidate
                .source_pocket_path
                .as_deref()
                .is_some_and(|path| !path.is_empty())
        })
        .count() as f64
        / total;
    let key_residue_contact_coverage = candidates
        .iter()
        .map(candidate_key_residue_contact_coverage_proxy)
        .sum::<f64>()
        / total;
    let clash_burden = clash_fraction;
    let contact_balance = candidates
        .iter()
        .map(candidate_contact_balance)
        .sum::<f64>()
        / total;
    let interaction_profile_coverage_fraction = candidates
        .iter()
        .filter(|candidate| candidate_is_valid(candidate))
        .count() as f64
        / total;
    let native_bond_count_mean = candidates
        .iter()
        .map(|candidate| candidate.bond_count as f64)
        .sum::<f64>()
        / total;
    let native_component_count_mean = candidates
        .iter()
        .map(native_component_count)
        .sum::<f64>()
        / total;
    let native_valence_violation_fraction = candidates
        .iter()
        .map(candidate_valence_violation_fraction)
        .sum::<f64>()
        / total;
    let native_disconnected_fragment_fraction = candidates
        .iter()
        .filter(|candidate| native_component_count(candidate) > 1.0)
        .count() as f64
        / total;
    let native_bond_order_conflict_fraction = candidates
        .iter()
        .filter(|candidate| candidate_bond_order_conflict(candidate))
        .count() as f64
        / total;
    let native_graph_repair_delta_mean = candidates
        .iter()
        .map(candidate_native_graph_repair_delta)
        .sum::<f64>()
        / total;
    let native_raw_to_constrained_removed_bond_count_mean = candidates
        .iter()
        .map(candidate_native_raw_to_constrained_removed_bond_count)
        .sum::<f64>()
        / total;
    let native_connectivity_guardrail_added_bond_count_mean = candidates
        .iter()
        .map(candidate_native_connectivity_guardrail_added_bond_count)
        .sum::<f64>()
        / total;
    let native_valence_guardrail_downgrade_count_mean = candidates
        .iter()
        .map(candidate_native_valence_guardrail_downgrade_count)
        .sum::<f64>()
        / total;
    let topology_bond_sync_fraction = candidates
        .iter()
        .filter(|candidate| candidate_topology_bond_payload_synced(candidate))
        .count() as f64
        / total;
    let atom_type_entropy = candidate_atom_type_entropy(candidates);
    let native_graph_valid_fraction = candidates
        .iter()
        .filter(|candidate| candidate_native_graph_valid(candidate))
        .count() as f64
        / total;
    let native_graph_metric_source = candidates
        .first()
        .map(|candidate| {
            if candidate.model_native_raw {
                "raw_model_native_pre_repair".to_string()
            } else {
                format!("postprocessed_or_constrained:{}", candidate.generation_layer)
            }
        })
        .unwrap_or_else(|| "unavailable".to_string());

    CandidateLayerMetrics {
        layer_name,
        pocket_interaction_provenance,
        generation_mode,
        candidate_count: candidates.len(),
        valid_fraction,
        pocket_contact_fraction,
        pocket_distance_bin_accuracy,
        pocket_contact_precision_proxy,
        pocket_contact_recall_proxy,
        pocket_role_compatibility_proxy,
        mean_centroid_offset,
        clash_fraction,
        mean_displacement: 0.0,
        atom_change_fraction: 0.0,
        uniqueness_proxy_fraction,
        diversity_eligible_fraction: diversity.eligible_fraction,
        validity_conditioned_unique_fraction: diversity.validity_conditioned_unique_fraction,
        equivalence_duplicate_fraction: diversity.equivalence_duplicate_fraction,
        invalid_diversity_excluded_fraction: diversity.invalid_excluded_fraction,
        diversity_metric_source: diversity.metric_source,
        atom_type_sequence_diversity,
        bond_topology_diversity,
        coordinate_shape_diversity,
        novel_atom_type_sequence_fraction,
        novel_bond_topology_fraction,
        novel_coordinate_shape_fraction,
        scaffold_novelty_fraction,
        unique_scaffold_fraction,
        pairwise_tanimoto_mean,
        nearest_train_similarity,
        scaffold_metric_coverage_fraction,
        hydrogen_bond_proxy,
        hydrophobic_contact_proxy,
        residue_contact_count,
        residue_identity_coverage_fraction,
        key_residue_contact_coverage,
        clash_burden,
        contact_balance,
        interaction_profile_coverage_fraction,
        native_bond_count_mean,
        native_component_count_mean,
        native_valence_violation_fraction,
        native_disconnected_fragment_fraction,
        native_bond_order_conflict_fraction,
        native_graph_repair_delta_mean,
        native_raw_to_constrained_removed_bond_count_mean,
        native_connectivity_guardrail_added_bond_count_mean,
        native_valence_guardrail_downgrade_count_mean,
        topology_bond_sync_fraction,
        atom_type_entropy,
        native_graph_valid_fraction,
        native_graph_metric_source,
    }
}

fn native_component_count(candidate: &GeneratedCandidateRecord) -> f64 {
    let atom_count = candidate.atom_types.len();
    if atom_count == 0 {
        return 0.0;
    }
    let mut parent = (0..atom_count).collect::<Vec<_>>();
    for &(left, right) in &candidate.inferred_bonds {
        if left < atom_count && right < atom_count {
            union_components(&mut parent, left, right);
        }
    }
    (0..atom_count)
        .map(|index| find_component(&mut parent, index))
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64
}

fn union_components(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find_component(parent, left);
    let right_root = find_component(parent, right);
    if left_root != right_root {
        parent[right_root] = left_root;
    }
}

fn find_component(parent: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parent[root] != root {
        root = parent[root];
    }
    let mut current = index;
    while parent[current] != current {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

fn candidate_valence_violation_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    let atom_count = candidate.atom_types.len();
    if atom_count == 0 {
        0.0
    } else {
        candidate.valence_violation_count as f64 / atom_count as f64
    }
}

fn candidate_topology_bond_payload_synced(candidate: &GeneratedCandidateRecord) -> bool {
    let atom_count = candidate.atom_types.len();
    candidate.bond_count == candidate.inferred_bonds.len()
        && candidate
            .inferred_bonds
            .iter()
            .all(|&(left, right)| left < atom_count && right < atom_count && left != right)
}

fn candidate_bond_order_conflict(candidate: &GeneratedCandidateRecord) -> bool {
    if !candidate_topology_bond_payload_synced(candidate) {
        return true;
    }
    candidate_representation_usize(candidate, "native_bond_types")
        .map(|native_type_count| native_type_count != candidate.bond_count)
        .unwrap_or(false)
}

fn candidate_native_graph_repair_delta(candidate: &GeneratedCandidateRecord) -> f64 {
    candidate_representation_usize(candidate, "guardrail_delta")
        .or_else(|| candidate_representation_usize(candidate, "raw_to_constrained_delta"))
        .unwrap_or(0) as f64
}

fn candidate_native_raw_to_constrained_removed_bond_count(
    candidate: &GeneratedCandidateRecord,
) -> f64 {
    candidate_representation_usize(candidate, "raw_to_constrained_delta").unwrap_or(0) as f64
}

fn candidate_native_valence_guardrail_downgrade_count(candidate: &GeneratedCandidateRecord) -> f64 {
    candidate_representation_usize(candidate, "valence_downgrades").unwrap_or(0) as f64
}

fn candidate_native_connectivity_guardrail_added_bond_count(
    candidate: &GeneratedCandidateRecord,
) -> f64 {
    let total = candidate_representation_usize(candidate, "guardrail_delta").unwrap_or(0);
    let removed =
        candidate_representation_usize(candidate, "raw_to_constrained_delta").unwrap_or(0);
    let downgraded = candidate_representation_usize(candidate, "valence_downgrades").unwrap_or(0);
    total.saturating_sub(removed + downgraded) as f64
}

fn candidate_representation_usize(
    candidate: &GeneratedCandidateRecord,
    key: &str,
) -> Option<usize> {
    let representation = candidate.molecular_representation.as_ref()?;
    representation.split(';').find_map(|field| {
        let (name, value) = field.split_once('=')?;
        (name == key).then(|| value.parse::<usize>().ok()).flatten()
    })
}

fn candidate_native_graph_valid(candidate: &GeneratedCandidateRecord) -> bool {
    candidate_is_valid(candidate)
        && candidate_topology_bond_payload_synced(candidate)
        && candidate.valence_violation_count == 0
        && native_component_count(candidate) <= 1.0
}

fn candidate_atom_type_entropy(candidates: &[GeneratedCandidateRecord]) -> f64 {
    let mut counts = std::collections::BTreeMap::<i64, usize>::new();
    let mut total = 0usize;
    for candidate in candidates {
        for atom_type in &candidate.atom_types {
            *counts.entry(*atom_type).or_default() += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    counts
        .values()
        .map(|count| {
            let probability = *count as f64 / total as f64;
            -probability * probability.max(1.0e-12).ln()
        })
        .sum()
}

fn diversity_fraction(
    candidates: &[GeneratedCandidateRecord],
    signature: fn(&GeneratedCandidateRecord) -> String,
) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates
        .iter()
        .map(signature)
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64
        / candidates.len() as f64
}

#[derive(Debug, Clone)]
struct EquivalenceDiversitySummary {
    unique_fraction_all_candidates: f64,
    eligible_fraction: f64,
    validity_conditioned_unique_fraction: f64,
    equivalence_duplicate_fraction: f64,
    invalid_excluded_fraction: f64,
    metric_source: String,
}

fn summarize_equivalence_diversity(
    candidates: &[GeneratedCandidateRecord],
) -> EquivalenceDiversitySummary {
    let total = candidates.len();
    if total == 0 {
        return EquivalenceDiversitySummary {
            unique_fraction_all_candidates: 0.0,
            eligible_fraction: 0.0,
            validity_conditioned_unique_fraction: 0.0,
            equivalence_duplicate_fraction: 0.0,
            invalid_excluded_fraction: 0.0,
            metric_source: "validity_conditioned_permutation_invariant_equivalence".to_string(),
        };
    }

    let signatures = candidates
        .iter()
        .filter_map(candidate_equivalence_signature)
        .collect::<std::collections::BTreeSet<_>>();
    let eligible = candidates
        .iter()
        .filter(|candidate| candidate_diversity_eligible(candidate))
        .count();
    let unique = signatures.len();
    let total = total as f64;
    let eligible_f = eligible as f64;
    let validity_conditioned_unique_fraction = if eligible == 0 {
        0.0
    } else {
        unique as f64 / eligible_f
    };

    EquivalenceDiversitySummary {
        unique_fraction_all_candidates: unique as f64 / total,
        eligible_fraction: eligible_f / total,
        validity_conditioned_unique_fraction,
        equivalence_duplicate_fraction: if eligible == 0 {
            0.0
        } else {
            1.0 - validity_conditioned_unique_fraction
        },
        invalid_excluded_fraction: (total - eligible_f) / total,
        metric_source: "validity_conditioned_permutation_invariant_equivalence".to_string(),
    }
}

fn candidate_diversity_eligible(candidate: &GeneratedCandidateRecord) -> bool {
    candidate_is_valid(candidate)
        && candidate_topology_bond_payload_synced(candidate)
        && candidate.valence_violation_count == 0
}

fn candidate_equivalence_signature(candidate: &GeneratedCandidateRecord) -> Option<String> {
    if !candidate_diversity_eligible(candidate) {
        return None;
    }
    Some(format!(
        "atoms={};bonds={};pair_dist={};pocket_radial={}",
        atom_composition_signature(candidate),
        typed_bond_multiset_signature(candidate),
        pairwise_distance_signature(candidate),
        pocket_radial_signature(candidate)
    ))
}

fn atom_composition_signature(candidate: &GeneratedCandidateRecord) -> String {
    counted_signature(candidate.atom_types.iter().map(|atom_type| format!("{atom_type}")))
}

fn typed_bond_multiset_signature(candidate: &GeneratedCandidateRecord) -> String {
    counted_signature(candidate.inferred_bonds.iter().filter_map(|(left, right)| {
        let left_type = candidate.atom_types.get(*left)?;
        let right_type = candidate.atom_types.get(*right)?;
        let (low, high) = if left_type <= right_type {
            (*left_type, *right_type)
        } else {
            (*right_type, *left_type)
        };
        Some(format!("{low}-{high}"))
    }))
}

fn pairwise_distance_signature(candidate: &GeneratedCandidateRecord) -> String {
    counted_signature((0..candidate.coords.len()).flat_map(|left| {
        ((left + 1)..candidate.coords.len()).map(move |right| {
            let left_type = candidate.atom_types[left];
            let right_type = candidate.atom_types[right];
            let (low, high) = if left_type <= right_type {
                (left_type, right_type)
            } else {
                (right_type, left_type)
            };
            let bucket = distance_bucket(coord_distance(&candidate.coords[left], &candidate.coords[right]));
            format!("{low}-{high}:{bucket}")
        })
    }))
}

fn pocket_radial_signature(candidate: &GeneratedCandidateRecord) -> String {
    counted_signature(candidate.atom_types.iter().zip(candidate.coords.iter()).map(
        |(atom_type, coord)| {
            let bucket = distance_bucket(coord_distance(coord, &candidate.pocket_centroid));
            format!("{atom_type}:{bucket}")
        },
    ))
}

fn distance_bucket(distance: f64) -> i64 {
    (distance * 10.0).round() as i64
}

fn counted_signature(values: impl IntoIterator<Item = String>) -> String {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for value in values {
        *counts.entry(value).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(value, count)| format!("{value}x{count}"))
        .collect::<Vec<_>>()
        .join("|")
}

#[derive(Debug, Clone, Default)]
struct NoveltyReferenceSignatures {
    atom_signatures: std::collections::BTreeSet<String>,
    bond_signatures: std::collections::BTreeSet<String>,
    shape_signatures: std::collections::BTreeSet<String>,
    scaffold_signatures: std::collections::BTreeSet<String>,
    fingerprints: Vec<std::collections::BTreeSet<String>>,
}

fn novelty_reference_signatures(
    train_examples: &[crate::data::MolecularExample],
) -> NoveltyReferenceSignatures {
    NoveltyReferenceSignatures {
        atom_signatures: train_examples.iter().map(example_atom_signature).collect(),
        bond_signatures: train_examples.iter().map(example_bond_signature).collect(),
        shape_signatures: train_examples.iter().map(example_shape_signature).collect(),
        scaffold_signatures: train_examples
            .iter()
            .map(example_scaffold_signature)
            .collect(),
        fingerprints: train_examples
            .iter()
            .filter_map(example_structural_fingerprint)
            .collect(),
    }
}

fn novelty_fraction(
    candidates: &[GeneratedCandidateRecord],
    signature: fn(&GeneratedCandidateRecord) -> String,
    references: &std::collections::BTreeSet<String>,
) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates
        .iter()
        .map(signature)
        .filter(|value| !references.contains(value))
        .count() as f64
        / candidates.len() as f64
}

fn candidate_scaffold_signature(candidate: &GeneratedCandidateRecord) -> String {
    let atom_count = candidate.atom_types.len();
    if atom_count == 0 {
        return "empty".to_string();
    }
    let core_atoms = scaffold_core_atoms(atom_count, candidate.inferred_bonds.iter().copied());
    let mut atom_labels = core_atoms
        .iter()
        .filter_map(|atom_ix| {
            candidate
                .atom_types
                .get(*atom_ix)
                .map(|atom_type| format!("{atom_ix}:{atom_type}"))
        })
        .collect::<Vec<_>>();
    atom_labels.sort();
    let mut core_bonds = candidate
        .inferred_bonds
        .iter()
        .filter(|(left, right)| core_atoms.contains(left) && core_atoms.contains(right))
        .map(|(left, right)| ordered_bond_label(*left, *right))
        .collect::<Vec<_>>();
    core_bonds.sort();
    format!(
        "atoms={};bonds={}",
        atom_labels.join(","),
        core_bonds.join("|")
    )
}

fn example_scaffold_signature(example: &crate::data::MolecularExample) -> String {
    let atom_count = ligand_atom_count(example);
    if atom_count == 0 {
        return "empty".to_string();
    }
    let bonds = example_bond_pairs(example);
    let core_atoms = scaffold_core_atoms(atom_count, bonds.iter().copied());
    let mut atom_labels = core_atoms
        .iter()
        .map(|atom_ix| {
            let atom_type = example.topology.atom_types.int64_value(&[*atom_ix as i64]);
            format!("{atom_ix}:{atom_type}")
        })
        .collect::<Vec<_>>();
    atom_labels.sort();
    let mut core_bonds = bonds
        .iter()
        .filter(|(left, right)| core_atoms.contains(left) && core_atoms.contains(right))
        .map(|(left, right)| ordered_bond_label(*left, *right))
        .collect::<Vec<_>>();
    core_bonds.sort();
    format!(
        "atoms={};bonds={}",
        atom_labels.join(","),
        core_bonds.join("|")
    )
}

fn scaffold_core_atoms<I>(atom_count: usize, bonds: I) -> std::collections::BTreeSet<usize>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut degrees = vec![0_usize; atom_count];
    for (left, right) in bonds {
        if left < atom_count && right < atom_count && left != right {
            degrees[left] += 1;
            degrees[right] += 1;
        }
    }
    let core = degrees
        .iter()
        .enumerate()
        .filter_map(|(atom_ix, degree)| (*degree > 1).then_some(atom_ix))
        .collect::<std::collections::BTreeSet<_>>();
    if core.is_empty() {
        (0..atom_count).collect()
    } else {
        core
    }
}

fn candidate_structural_fingerprint(
    candidate: &GeneratedCandidateRecord,
) -> Option<std::collections::BTreeSet<String>> {
    if !candidate_is_valid(candidate) {
        return None;
    }
    let mut fp = std::collections::BTreeSet::new();
    for atom_type in &candidate.atom_types {
        fp.insert(format!("atom:{atom_type}"));
    }
    for (left, right) in &candidate.inferred_bonds {
        let left_type = candidate.atom_types.get(*left).copied().unwrap_or_default();
        let right_type = candidate
            .atom_types
            .get(*right)
            .copied()
            .unwrap_or_default();
        let (low, high) = if left_type <= right_type {
            (left_type, right_type)
        } else {
            (right_type, left_type)
        };
        fp.insert(format!("bond_atom_pair:{low}-{high}"));
    }
    fp.insert(format!(
        "scaffold:{}",
        candidate_scaffold_signature(candidate)
    ));
    Some(fp)
}

fn example_structural_fingerprint(
    example: &crate::data::MolecularExample,
) -> Option<std::collections::BTreeSet<String>> {
    let atom_count = ligand_atom_count(example);
    if atom_count == 0 {
        return None;
    }
    let mut fp = std::collections::BTreeSet::new();
    for index in 0..atom_count {
        let atom_type = example.topology.atom_types.int64_value(&[index as i64]);
        fp.insert(format!("atom:{atom_type}"));
    }
    for (left, right) in example_bond_pairs(example) {
        let left_type = example.topology.atom_types.int64_value(&[left as i64]);
        let right_type = example.topology.atom_types.int64_value(&[right as i64]);
        let (low, high) = if left_type <= right_type {
            (left_type, right_type)
        } else {
            (right_type, left_type)
        };
        fp.insert(format!("bond_atom_pair:{low}-{high}"));
    }
    fp.insert(format!("scaffold:{}", example_scaffold_signature(example)));
    Some(fp)
}

fn example_bond_pairs(example: &crate::data::MolecularExample) -> Vec<(usize, usize)> {
    let atom_count = ligand_atom_count(example);
    let mut bonds = Vec::new();
    for left in 0..atom_count {
        for right in (left + 1)..atom_count {
            if example
                .topology
                .adjacency
                .double_value(&[left as i64, right as i64])
                > 0.5
            {
                bonds.push((left, right));
            }
        }
    }
    bonds
}

fn ordered_bond_label(left: usize, right: usize) -> String {
    let (low, high) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    format!("{low}-{high}")
}

fn mean_pairwise_tanimoto(fingerprints: &[std::collections::BTreeSet<String>]) -> Option<f64> {
    if fingerprints.len() < 2 {
        return None;
    }
    let mut total = 0.0;
    let mut count = 0_usize;
    for left in 0..fingerprints.len() {
        for right in (left + 1)..fingerprints.len() {
            total += tanimoto(&fingerprints[left], &fingerprints[right]);
            count += 1;
        }
    }
    (count > 0).then(|| total / count as f64)
}

fn mean_nearest_train_similarity(
    candidates: &[std::collections::BTreeSet<String>],
    references: &[std::collections::BTreeSet<String>],
) -> Option<f64> {
    if candidates.is_empty() || references.is_empty() {
        return None;
    }
    let total = candidates
        .iter()
        .map(|candidate| {
            references
                .iter()
                .map(|reference| tanimoto(candidate, reference))
                .fold(0.0_f64, f64::max)
        })
        .sum::<f64>();
    Some(total / candidates.len() as f64)
}

fn tanimoto(
    left: &std::collections::BTreeSet<String>,
    right: &std::collections::BTreeSet<String>,
) -> f64 {
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn candidate_hydrogen_bond_proxy(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return 0.0;
    }
    let polar_contacts = candidate
        .atom_types
        .iter()
        .zip(candidate.coords.iter())
        .filter(|(atom_type, coord)| {
            ligand_atom_is_polar(**atom_type)
                && coord_distance(coord, &candidate.pocket_centroid)
                    <= (candidate.pocket_radius + 1.8) as f64
        })
        .count();
    (polar_contacts as f64 / candidate.coords.len() as f64).clamp(0.0, 1.0)
}

fn candidate_hydrophobic_contact_proxy(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return 0.0;
    }
    let hydrophobic_contacts = candidate
        .atom_types
        .iter()
        .zip(candidate.coords.iter())
        .filter(|(atom_type, coord)| {
            ligand_atom_is_hydrophobic(**atom_type)
                && coord_distance(coord, &candidate.pocket_centroid)
                    <= (candidate.pocket_radius + 2.2) as f64
        })
        .count();
    (hydrophobic_contacts as f64 / candidate.coords.len() as f64).clamp(0.0, 1.0)
}

fn candidate_residue_contact_count_proxy(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() || !candidate_has_pocket_contact(candidate) {
        0.0
    } else {
        1.0
    }
}

fn candidate_key_residue_contact_coverage_proxy(candidate: &GeneratedCandidateRecord) -> f64 {
    candidate_hydrogen_bond_proxy(candidate)
        .max(candidate_hydrophobic_contact_proxy(candidate))
        .clamp(0.0, 1.0)
}

fn candidate_contact_balance(candidate: &GeneratedCandidateRecord) -> f64 {
    let contact = if candidate_has_pocket_contact(candidate) {
        candidate_hydrogen_bond_proxy(candidate).max(candidate_hydrophobic_contact_proxy(candidate))
    } else {
        0.0
    };
    (contact * (1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0))).clamp(0.0, 1.0)
}

fn candidate_layer_pocket_interaction_provenance(
    candidate: Option<&GeneratedCandidateRecord>,
) -> String {
    let Some(candidate) = candidate else {
        return "unavailable".to_string();
    };
    if candidate.model_native_raw {
        format!(
            "raw_model_native:{}:{}",
            candidate.generation_layer, candidate.generation_path_class
        )
    } else if candidate.postprocessor_chain.is_empty() {
        format!(
            "processed_or_selected:{}:{}",
            candidate.generation_layer, candidate.generation_path_class
        )
    } else {
        format!(
            "postprocessed:{}:{}",
            candidate.generation_layer,
            candidate.postprocessor_chain.join("+")
        )
    }
}

fn candidate_pocket_distance_bin_accuracy_proxy(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return 0.0;
    }
    let target_bin = pocket_distance_bin(candidate.pocket_radius as f64);
    let matching = candidate
        .coords
        .iter()
        .filter(|coord| pocket_distance_bin(coord_distance(coord, &candidate.pocket_centroid)) == target_bin)
        .count();
    matching as f64 / candidate.coords.len() as f64
}

fn candidate_pocket_contact_recall_proxy(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return 0.0;
    }
    let cutoff = (candidate.pocket_radius + 2.0) as f64;
    let contacted = candidate
        .coords
        .iter()
        .filter(|coord| coord_distance(coord, &candidate.pocket_centroid) <= cutoff)
        .count();
    contacted as f64 / candidate.coords.len() as f64
}

fn candidate_pocket_role_compatibility_proxy(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() || candidate.atom_types.len() != candidate.coords.len() {
        return 0.0;
    }
    let contact_cutoff = (candidate.pocket_radius + 2.0) as f64;
    let compatible = candidate
        .atom_types
        .iter()
        .zip(candidate.coords.iter())
        .filter(|(atom_type, coord)| {
            coord_distance(coord, &candidate.pocket_centroid) <= contact_cutoff
                && (ligand_atom_is_polar(**atom_type) || ligand_atom_is_hydrophobic(**atom_type))
        })
        .count();
    compatible as f64 / candidate.coords.len() as f64
}

fn pocket_distance_bin(distance: f64) -> usize {
    if distance < 2.0 {
        0
    } else if distance < 4.0 {
        1
    } else if distance < 6.0 {
        2
    } else {
        3
    }
}

fn ligand_atom_is_polar(atom_type: i64) -> bool {
    (1..=3).contains(&atom_type)
}

fn ligand_atom_is_hydrophobic(atom_type: i64) -> bool {
    atom_type == 0
}

#[cfg(test)]
mod generation_diversity_tests {
    use super::*;

    fn candidate(
        atom_types: Vec<i64>,
        coords: Vec<[f32; 3]>,
        bonds: Vec<(usize, usize)>,
    ) -> GeneratedCandidateRecord {
        GeneratedCandidateRecord {
            example_id: "example".to_string(),
            protein_id: "protein".to_string(),
            molecular_representation: None,
            atom_types,
            coords,
            inferred_bonds: bonds.clone(),
            bond_count: bonds.len(),
            valence_violation_count: 0,
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 2.5,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: "test".to_string(),
            generation_mode: "de_novo_initialization".to_string(),
            generation_layer: "raw_flow".to_string(),
            generation_path_class: "model_native_raw".to_string(),
            model_native_raw: true,
            postprocessor_chain: Vec::new(),
            claim_boundary:
                "raw model-native decoder output before repair, reranking, or backend scoring"
                    .to_string(),
            source_pocket_path: None,
            source_ligand_path: None,
        }
    }

    #[test]
    fn duplicate_samples_do_not_increase_equivalence_diversity() {
        let first = candidate(vec![0, 1], vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], vec![(0, 1)]);
        let metrics = summarize_candidate_layer(
            &[first.clone(), first],
            &NoveltyReferenceSignatures::default(),
        );

        assert_eq!(metrics.diversity_eligible_fraction, 1.0);
        assert_eq!(metrics.uniqueness_proxy_fraction, 0.5);
        assert_eq!(metrics.validity_conditioned_unique_fraction, 0.5);
        assert_eq!(metrics.equivalence_duplicate_fraction, 0.5);
        assert_eq!(metrics.invalid_diversity_excluded_fraction, 0.0);
    }

    #[test]
    fn permuted_equivalent_samples_share_one_equivalence_class() {
        let canonical =
            candidate(vec![0, 1], vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], vec![(0, 1)]);
        let permuted =
            candidate(vec![1, 0], vec![[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]], vec![(1, 0)]);
        let metrics =
            summarize_candidate_layer(&[canonical, permuted], &NoveltyReferenceSignatures::default());

        assert_eq!(metrics.diversity_eligible_fraction, 1.0);
        assert_eq!(metrics.uniqueness_proxy_fraction, 0.5);
        assert_eq!(metrics.validity_conditioned_unique_fraction, 0.5);
    }

    #[test]
    fn genuinely_different_valid_samples_are_counted_as_diverse() {
        let first = candidate(vec![0, 1], vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], vec![(0, 1)]);
        let second = candidate(vec![0, 2], vec![[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]], vec![(0, 1)]);
        let metrics =
            summarize_candidate_layer(&[first, second], &NoveltyReferenceSignatures::default());

        assert_eq!(metrics.diversity_eligible_fraction, 1.0);
        assert_eq!(metrics.uniqueness_proxy_fraction, 1.0);
        assert_eq!(metrics.validity_conditioned_unique_fraction, 1.0);
        assert_eq!(metrics.equivalence_duplicate_fraction, 0.0);
    }

    #[test]
    fn invalid_samples_are_excluded_from_diversity_credit() {
        let valid = candidate(vec![0, 1], vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], vec![(0, 1)]);
        let invalid = candidate(vec![0, 1], vec![[0.0, 0.0, 0.0]], vec![(0, 1)]);
        let metrics =
            summarize_candidate_layer(&[valid, invalid], &NoveltyReferenceSignatures::default());

        assert_eq!(metrics.diversity_eligible_fraction, 0.5);
        assert_eq!(metrics.uniqueness_proxy_fraction, 0.5);
        assert_eq!(metrics.validity_conditioned_unique_fraction, 1.0);
        assert_eq!(metrics.invalid_diversity_excluded_fraction, 0.5);
    }
}
