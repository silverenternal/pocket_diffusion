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
    LayeredGenerationMetrics {
        raw_rollout: summarize_candidate_layer(&[], &NoveltyReferenceSignatures::default()),
        repaired_candidates: summarize_candidate_layer(&[], &NoveltyReferenceSignatures::default()),
        inferred_bond_candidates: summarize_candidate_layer(
            &[],
            &NoveltyReferenceSignatures::default(),
        ),
        reranked_candidates: summarize_candidate_layer(&[], &NoveltyReferenceSignatures::default()),
        deterministic_proxy_candidates: summarize_candidate_layer(
            &[],
            &NoveltyReferenceSignatures::default(),
        ),
        reranker_calibration: RerankerCalibrationReport::default(),
        backend_scored_candidates: BTreeMap::new(),
        method_comparison: MethodComparisonSummary::default(),
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
    configured_method_ids.push(
        research
            .generation_method
            .primary_backend_id()
            .to_string(),
    );
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

    MethodComparisonSummary {
        active_method: PocketGenerationMethodRegistry::metadata(
            research.generation_method.primary_backend_id(),
        )
        .ok(),
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

fn flow_vs_denoising_delta(
    methods: &[MethodComparisonRow],
) -> Option<FlowVsDenoisingDelta> {
    let flow = methods.iter().find(|row| row.method_id == "flow_matching")?;
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
    let active_method = PocketGenerationMethodRegistry::build(
        research.generation_method.primary_backend_id(),
    )
    .unwrap_or_else(|_| {
        PocketGenerationMethodRegistry::build("conditioned_denoising")
            .expect("conditioned_denoising registry entry must exist")
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
    let backend_candidates = final_backend_candidate_layer(&candidates, &reranked, &proxy_reranked);
    let novelty_reference = novelty_reference_signatures(train_examples);
    let calibrated_reranker = CalibratedReranker::fit(&candidates);
    let mut method_comparison = build_method_comparison_summary(
        research,
        examples,
        forwards,
        ablation,
        external_evaluation.generation_artifact_example_limit,
    );

    let mut layered = LayeredGenerationMetrics {
        raw_rollout: summarize_candidate_layer(&raw_rollout, &novelty_reference),
        repaired_candidates: summarize_candidate_layer(&repaired, &novelty_reference),
        inferred_bond_candidates: summarize_candidate_layer(&candidates, &novelty_reference),
        reranked_candidates: summarize_candidate_layer(&reranked, &novelty_reference),
        deterministic_proxy_candidates: summarize_candidate_layer(
            &proxy_reranked,
            &novelty_reference,
        ),
        reranker_calibration: calibrated_reranker.report(),
        backend_scored_candidates: BTreeMap::new(),
        method_comparison: method_comparison.clone(),
    };
    apply_raw_rollout_stability(&mut layered.raw_rollout, forwards);

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

fn final_backend_candidate_layer(
    inferred_bond: &[GeneratedCandidateRecord],
    reranked: &[GeneratedCandidateRecord],
    deterministic_proxy: &[GeneratedCandidateRecord],
) -> Vec<GeneratedCandidateRecord> {
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
        .map(|candidate| vec![candidate.record])
        .or_else(|| {
            if !reranked.is_empty() {
                Some(reranked.to_vec())
            } else if !deterministic_proxy.is_empty() {
                Some(deterministic_proxy.to_vec())
            } else if !inferred_bond.is_empty() {
                Some(inferred_bond.to_vec())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

#[derive(Debug)]
struct BackendSelectionCandidate {
    record: GeneratedCandidateRecord,
    selection_score: f64,
    layer_priority: usize,
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
            })
        })
        .max_by(|left, right| {
            left.selection_score
                .partial_cmp(&right.selection_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn backend_status(external_enabled: bool, external: &str, heuristic: &str) -> String {
    if external_enabled {
        external.to_string()
    } else {
        heuristic.to_string()
    }
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
            candidate_count: 0,
            valid_fraction: 0.0,
            pocket_contact_fraction: 0.0,
            mean_centroid_offset: 0.0,
            clash_fraction: 0.0,
            mean_displacement: 0.0,
            atom_change_fraction: 0.0,
            uniqueness_proxy_fraction: 0.0,
            atom_type_sequence_diversity: 0.0,
            bond_topology_diversity: 0.0,
            coordinate_shape_diversity: 0.0,
            novel_atom_type_sequence_fraction: 0.0,
            novel_bond_topology_fraction: 0.0,
            novel_coordinate_shape_fraction: 0.0,
        };
    }
    let total = candidates.len() as f64;
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
    let mean_centroid_offset = candidates
        .iter()
        .map(candidate_centroid_offset)
        .filter(|value| value.is_finite())
        .sum::<f64>()
        / total;
    let clash_fraction = candidates.iter().map(candidate_clash_fraction).sum::<f64>() / total;
    let uniqueness_proxy_fraction = candidates
        .iter()
        .map(candidate_uniqueness_signature)
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64
        / total;
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

    CandidateLayerMetrics {
        candidate_count: candidates.len(),
        valid_fraction,
        pocket_contact_fraction,
        mean_centroid_offset,
        clash_fraction,
        mean_displacement: 0.0,
        atom_change_fraction: 0.0,
        uniqueness_proxy_fraction,
        atom_type_sequence_diversity,
        bond_topology_diversity,
        coordinate_shape_diversity,
        novel_atom_type_sequence_fraction,
        novel_bond_topology_fraction,
        novel_coordinate_shape_fraction,
    }
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

#[derive(Debug, Clone, Default)]
struct NoveltyReferenceSignatures {
    atom_signatures: std::collections::BTreeSet<String>,
    bond_signatures: std::collections::BTreeSet<String>,
    shape_signatures: std::collections::BTreeSet<String>,
}

fn novelty_reference_signatures(
    train_examples: &[crate::data::MolecularExample],
) -> NoveltyReferenceSignatures {
    NoveltyReferenceSignatures {
        atom_signatures: train_examples.iter().map(example_atom_signature).collect(),
        bond_signatures: train_examples.iter().map(example_bond_signature).collect(),
        shape_signatures: train_examples.iter().map(example_shape_signature).collect(),
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
