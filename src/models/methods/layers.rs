type LayeredCandidates = (
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
);

/// Flatten layered outputs into legacy layer lists for backward-compatible artifact writers.
pub fn flatten_layered_output(
    output: &LayeredGenerationOutput,
) -> LayeredCandidates {
    (
        output
            .raw_rollout
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .repaired
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .inferred_bond
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .deterministic_proxy
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .reranked
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
    )
}

/// Build a compact method comparison row from one layered output.
pub fn summarize_method_output(output: &LayeredGenerationOutput) -> MethodComparisonRow {
    let supported_layers = output
        .metadata
        .layered_output_support
        .iter()
        .map(|layer| layer.legacy_field_name().to_string())
        .collect::<Vec<_>>();
    let available_layers = [
        output.raw_geometry.as_ref(),
        output.raw_rollout.as_ref(),
        output.bond_logits_refined.as_ref(),
        output.valence_refined.as_ref(),
        output.repaired.as_ref(),
        output.inferred_bond.as_ref(),
        output.deterministic_proxy.as_ref(),
        output.reranked.as_ref(),
    ]
    .into_iter()
    .flatten()
    .filter(|layer| !layer.candidates.is_empty())
    .map(|layer| layer.provenance.legacy_field_name.clone())
    .collect::<Vec<_>>();
    let selected_layer = selected_metric_layer(output);
    let native_layer = selected_layer.map(|layer| layer.provenance.legacy_field_name.clone());
    let selected_metric_layer_path_class =
        selected_layer.map(|layer| layer.provenance.generation_path_class.clone());
    let selected_metric_layer_model_native_raw = selected_layer
        .map(|layer| {
            layer.provenance.method_native && layer.provenance.layer_kind.is_model_native_raw()
        })
        .unwrap_or(false);
    let native_candidate_count = selected_layer
        .map(|layer| layer.candidates.len())
        .unwrap_or(0);
    let native_candidates = selected_layer.map(|layer| layer.candidates.as_slice());
    let repair_gain_valid_fraction = output
        .raw_rollout
        .as_ref()
        .zip(output.repaired.as_ref())
        .map(|(raw, repaired)| valid_fraction(&repaired.candidates) - valid_fraction(&raw.candidates));
    let rerank_gain_valid_fraction = output
        .inferred_bond
        .as_ref()
        .zip(output.reranked.as_ref())
        .map(|(inferred, reranked)| {
            valid_fraction(&reranked.candidates) - valid_fraction(&inferred.candidates)
        });

    MethodComparisonRow {
        method_id: output.metadata.method_id.clone(),
        method_name: output.metadata.method_name.clone(),
        method_family: format!("{:?}", output.metadata.method_family).to_ascii_lowercase(),
        evidence_role: format!("{:?}", output.metadata.evidence_role).to_ascii_lowercase(),
        available: native_candidate_count > 0
            || output.raw_geometry.is_some()
            || output.raw_rollout.is_some()
            || output.bond_logits_refined.is_some()
            || output.valence_refined.is_some()
            || output.repaired.is_some()
            || output.inferred_bond.is_some()
            || output.deterministic_proxy.is_some()
            || output.reranked.is_some(),
        native_layer: native_layer.clone(),
        selected_metric_layer: native_layer,
        selected_metric_layer_path_class,
        selected_metric_layer_model_native_raw,
        supported_layers,
        trainable: output.metadata.capability.trainable,
        parameter_count: None,
        sampling_steps: None,
        wall_time_ms: None,
        memory_estimate_bytes: Some(layered_candidate_memory_estimate(output)),
        available_layers,
        native_candidate_count,
        native_valid_fraction: native_candidates.map(valid_fraction),
        native_pocket_contact_fraction: native_candidates.map(pocket_contact_fraction),
        native_clash_fraction: native_candidates.map(mean_clash_fraction),
        slot_activation_mean: None,
        gate_activation_mean: None,
        leakage_proxy_mean: None,
        repair_gain_valid_fraction,
        rerank_gain_valid_fraction,
    }
}

fn selected_metric_layer(output: &LayeredGenerationOutput) -> Option<&CandidateLayerOutput> {
    preferred_model_native_layer(output).or_else(|| match output.metadata.method_family {
        PocketGenerationMethodFamily::RepairOnly => output.repaired.as_ref(),
        PocketGenerationMethodFamily::RerankerOnly => output
            .reranked
            .as_ref()
            .or(output.deterministic_proxy.as_ref()),
        _ => None,
    })
}

fn preferred_model_native_layer(output: &LayeredGenerationOutput) -> Option<&CandidateLayerOutput> {
    [
        output.raw_rollout.as_ref(),
        output.raw_geometry.as_ref(),
        output.bond_logits_refined.as_ref(),
        output.valence_refined.as_ref(),
        output.inferred_bond.as_ref(),
        output.repaired.as_ref(),
        output.deterministic_proxy.as_ref(),
        output.reranked.as_ref(),
    ]
    .into_iter()
    .flatten()
    .find(|layer| layer.provenance.method_native)
}

fn layered_candidate_memory_estimate(output: &LayeredGenerationOutput) -> usize {
    [
        output.raw_geometry.as_ref(),
        output.raw_rollout.as_ref(),
        output.bond_logits_refined.as_ref(),
        output.valence_refined.as_ref(),
        output.repaired.as_ref(),
        output.inferred_bond.as_ref(),
        output.deterministic_proxy.as_ref(),
        output.reranked.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(|layer| {
        layer
            .candidates
            .iter()
            .map(candidate_memory_estimate)
            .sum::<usize>()
    })
    .sum()
}

fn candidate_memory_estimate(candidate: &GeneratedCandidateRecord) -> usize {
    candidate.atom_types.len() * std::mem::size_of::<i64>()
        + candidate.coords.len() * std::mem::size_of::<[f32; 3]>()
        + candidate.inferred_bonds.len() * std::mem::size_of::<(usize, usize)>()
        + candidate
            .molecular_representation
            .as_ref()
            .map(|value| value.len())
            .unwrap_or(0)
}

fn valid_fraction(candidates: &[GeneratedCandidateRecord]) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates.iter().filter(|candidate| candidate_is_valid(candidate)).count() as f64
        / candidates.len() as f64
}

fn pocket_contact_fraction(candidates: &[GeneratedCandidateRecord]) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates
        .iter()
        .filter(|candidate| candidate_has_pocket_contact(candidate))
        .count() as f64
        / candidates.len() as f64
}

fn mean_clash_fraction(candidates: &[GeneratedCandidateRecord]) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates
        .iter()
        .map(candidate_clash_fraction)
        .sum::<f64>()
        / candidates.len() as f64
}
