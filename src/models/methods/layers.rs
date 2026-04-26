/// Flatten layered outputs into legacy layer lists for backward-compatible artifact writers.
pub fn flatten_layered_output(
    output: &LayeredGenerationOutput,
) -> (
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
) {
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
        output.raw_rollout.as_ref(),
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
    let native_layer = [
        output.raw_rollout.as_ref(),
        output.repaired.as_ref(),
        output.inferred_bond.as_ref(),
        output.deterministic_proxy.as_ref(),
        output.reranked.as_ref(),
    ]
    .into_iter()
    .flatten()
    .find(|layer| layer.provenance.method_native)
    .map(|layer| layer.provenance.legacy_field_name.clone())
    .or_else(|| {
        if output.metadata.method_family == PocketGenerationMethodFamily::RerankerOnly {
            output
                .reranked
                .as_ref()
                .map(|layer| layer.provenance.legacy_field_name.clone())
        } else {
            None
        }
    });
    let native_candidate_count = native_layer
        .as_ref()
        .and_then(|layer_name| match layer_name.as_str() {
            "raw_rollout" => output.raw_rollout.as_ref(),
            "repaired_candidates" => output.repaired.as_ref(),
            "inferred_bond_candidates" => output.inferred_bond.as_ref(),
            "deterministic_proxy_candidates" => output.deterministic_proxy.as_ref(),
            "reranked_candidates" => output.reranked.as_ref(),
            _ => None,
        })
        .map(|layer| layer.candidates.len())
        .unwrap_or(0);
    let native_candidates = native_layer
        .as_ref()
        .and_then(|layer_name| match layer_name.as_str() {
            "raw_rollout" => output.raw_rollout.as_ref(),
            "repaired_candidates" => output.repaired.as_ref(),
            "inferred_bond_candidates" => output.inferred_bond.as_ref(),
            "deterministic_proxy_candidates" => output.deterministic_proxy.as_ref(),
            "reranked_candidates" => output.reranked.as_ref(),
            _ => None,
        })
        .map(|layer| layer.candidates.as_slice());
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
            || output.raw_rollout.is_some()
            || output.repaired.is_some()
            || output.inferred_bond.is_some()
            || output.deterministic_proxy.is_some()
            || output.reranked.is_some(),
        native_layer,
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

fn layered_candidate_memory_estimate(output: &LayeredGenerationOutput) -> usize {
    [
        output.raw_rollout.as_ref(),
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
