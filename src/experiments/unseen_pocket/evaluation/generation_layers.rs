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
    };
    apply_raw_rollout_stability(&mut layered.raw_rollout, forwards);
    layered.raw_flow = layered.raw_rollout.clone();

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
            scaffold_novelty_fraction: 0.0,
            unique_scaffold_fraction: 0.0,
            pairwise_tanimoto_mean: 0.0,
            nearest_train_similarity: 0.0,
            scaffold_metric_coverage_fraction: 0.0,
            hydrogen_bond_proxy: 0.0,
            hydrophobic_contact_proxy: 0.0,
            residue_contact_count: 0.0,
            key_residue_contact_coverage: 0.0,
            clash_burden: 0.0,
            contact_balance: 0.0,
            interaction_profile_coverage_fraction: 0.0,
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
        scaffold_novelty_fraction,
        unique_scaffold_fraction,
        pairwise_tanimoto_mean,
        nearest_train_similarity,
        scaffold_metric_coverage_fraction,
        hydrogen_bond_proxy,
        hydrophobic_contact_proxy,
        residue_contact_count,
        key_residue_contact_coverage,
        clash_burden,
        contact_balance,
        interaction_profile_coverage_fraction,
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

fn ligand_atom_is_polar(atom_type: i64) -> bool {
    matches!(atom_type, 1 | 2 | 3)
}

fn ligand_atom_is_hydrophobic(atom_type: i64) -> bool {
    atom_type == 0
}
