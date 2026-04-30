#[allow(dead_code)]
fn proxy_rerank_candidates(
    candidates: &[GeneratedCandidateRecord],
) -> Vec<GeneratedCandidateRecord> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|left, right| {
        proxy_rerank_score(right)
            .partial_cmp(&proxy_rerank_score(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = (ranked.len() / 2).max(1).min(ranked.len());
    ranked.truncate(keep);
    ranked
}

#[allow(dead_code)]
fn proxy_rerank_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let valid = if candidate_is_valid(candidate) {
        1.0
    } else {
        0.0
    };
    let contact = if candidate_has_pocket_contact(candidate) {
        1.0
    } else {
        0.0
    };
    let centroid = 1.0 / (1.0 + candidate_centroid_offset(candidate).max(0.0));
    let clash = 1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0);
    let valence = if valence_sane_proxy(candidate) {
        1.0
    } else {
        0.0
    };
    0.25 * valid + 0.25 * contact + 0.2 * centroid + 0.2 * clash + 0.1 * valence
}

#[derive(Debug, Clone)]
struct CalibratedReranker {
    coefficients: BTreeMap<String, f64>,
    target_mean: f64,
    fitted_candidate_count: usize,
}

impl CalibratedReranker {
    fn fit(candidates: &[GeneratedCandidateRecord]) -> Self {
        let feature_names = reranker_feature_names();
        if candidates.is_empty() {
            return Self {
                coefficients: default_reranker_coefficients(),
                target_mean: 0.0,
                fitted_candidate_count: 0,
            };
        }

        let features = candidates.iter().map(reranker_features).collect::<Vec<_>>();
        let targets = candidates
            .iter()
            .map(backend_compatible_rerank_target)
            .collect::<Vec<_>>();
        let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let mut means = vec![0.0; feature_names.len()];
        for row in &features {
            for (index, value) in row.iter().enumerate() {
                means[index] += value;
            }
        }
        for mean in &mut means {
            *mean /= features.len() as f64;
        }

        let mut weights = vec![0.0; feature_names.len()];
        for (row, target) in features.iter().zip(targets.iter()) {
            for (index, value) in row.iter().enumerate() {
                weights[index] += (value - means[index]) * (target - target_mean);
            }
        }
        for weight in &mut weights {
            *weight = weight.max(0.0);
        }
        let sum = weights.iter().sum::<f64>();
        let coefficients = if sum <= 1e-12 {
            default_reranker_coefficients()
        } else {
            feature_names
                .iter()
                .zip(weights.iter())
                .map(|(name, weight)| ((*name).to_string(), weight / sum))
                .collect()
        };

        Self {
            coefficients,
            target_mean,
            fitted_candidate_count: candidates.len(),
        }
    }

    #[allow(dead_code)]
    fn rerank(&self, candidates: &[GeneratedCandidateRecord]) -> Vec<GeneratedCandidateRecord> {
        let mut ranked = candidates.to_vec();
        ranked.sort_by(|left, right| {
            self.score(right)
                .partial_cmp(&self.score(left))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.source.cmp(&right.source))
        });
        let keep = (ranked.len() / 2).max(1).min(ranked.len());
        ranked.truncate(keep);
        ranked
    }

    #[allow(dead_code)]
    fn score(&self, candidate: &GeneratedCandidateRecord) -> f64 {
        reranker_feature_names()
            .iter()
            .zip(reranker_features(candidate).iter())
            .map(|(name, value)| self.coefficients.get(*name).copied().unwrap_or(0.0) * value)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    fn report(&self) -> RerankerCalibrationReport {
        RerankerCalibrationReport {
            method: "bounded_nonnegative_feature_target_covariance_v1".to_string(),
            coefficients: self.coefficients.clone(),
            target_mean: self.target_mean,
            fitted_candidate_count: self.fitted_candidate_count,
        }
    }
}

fn reranker_feature_names() -> [&'static str; 6] {
    [
        "valid",
        "valence_sane",
        "pocket_contact",
        "centroid_fit",
        "clash_free",
        "bond_density_fit",
    ]
}

fn reranker_features(candidate: &GeneratedCandidateRecord) -> Vec<f64> {
    vec![
        if candidate_is_valid(candidate) {
            1.0
        } else {
            0.0
        },
        if valence_sane_proxy(candidate) {
            1.0
        } else {
            0.0
        },
        if candidate_has_pocket_contact(candidate) {
            1.0
        } else {
            0.0
        },
        centroid_fit_feature(candidate),
        1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0),
        bond_density_fit_feature(candidate),
    ]
}

fn default_reranker_coefficients() -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("valid".to_string(), 0.22),
        ("valence_sane".to_string(), 0.18),
        ("pocket_contact".to_string(), 0.20),
        ("centroid_fit".to_string(), 0.18),
        ("clash_free".to_string(), 0.17),
        ("bond_density_fit".to_string(), 0.05),
    ])
}

fn backend_compatible_rerank_target(candidate: &GeneratedCandidateRecord) -> f64 {
    let features = reranker_features(candidate);
    (0.24 * features[0]
        + 0.18 * features[1]
        + 0.20 * features[2]
        + 0.20 * features[3]
        + 0.15 * features[4]
        + 0.03 * features[5])
        .clamp(0.0, 1.0)
}

fn centroid_fit_feature(candidate: &GeneratedCandidateRecord) -> f64 {
    let radius = (candidate.pocket_radius as f64).max(1.0);
    (1.0 - candidate_centroid_offset(candidate) / (radius + 2.0)).clamp(0.0, 1.0)
}

fn bond_density_fit_feature(candidate: &GeneratedCandidateRecord) -> f64 {
    let atoms = candidate.atom_types.len();
    if atoms < 2 {
        return 0.0;
    }
    let density = candidate.inferred_bonds.len() as f64 / atoms as f64;
    (1.0 - (density - 1.05).abs() / 1.05).clamp(0.0, 1.0)
}

fn valence_sane_proxy(candidate: &GeneratedCandidateRecord) -> bool {
    if candidate.atom_types.is_empty() {
        return false;
    }
    let mut degrees = vec![0usize; candidate.atom_types.len()];
    for &(left, right) in &candidate.inferred_bonds {
        if left < degrees.len() && right < degrees.len() {
            degrees[left] += 1;
            degrees[right] += 1;
        }
    }
    degrees
        .iter()
        .zip(candidate.atom_types.iter())
        .all(|(degree, atom_type)| *degree <= max_reasonable_valence(*atom_type))
}

fn max_reasonable_valence(atom_type: i64) -> usize {
    match atom_type {
        0 => 4,
        1 => 4,
        2 => 2,
        3 => 6,
        4 => 1,
        _ => 4,
    }
}

fn apply_raw_rollout_stability(layer: &mut CandidateLayerMetrics, forwards: &[ResearchForward]) {
    let final_steps = forwards
        .iter()
        .filter_map(|forward| forward.generation.rollout.steps.last())
        .collect::<Vec<_>>();
    if final_steps.is_empty() {
        return;
    }
    let denom = final_steps.len() as f64;
    layer.mean_displacement = final_steps
        .iter()
        .map(|step| step.mean_displacement)
        .sum::<f64>()
        / denom;
    layer.atom_change_fraction = final_steps
        .iter()
        .map(|step| step.atom_change_fraction)
        .sum::<f64>()
        / denom;
}

fn aggregate_final_flow_diagnostics(forwards: &[ResearchForward]) -> BTreeMap<String, f64> {
    let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for diagnostics in forwards
        .iter()
        .filter_map(|forward| forward.generation.rollout.steps.last())
        .map(|step| &step.flow_diagnostics)
    {
        for (key, value) in diagnostics {
            if value.is_finite() {
                let entry = sums.entry(key.clone()).or_insert((0.0, 0));
                entry.0 += *value;
                entry.1 += 1;
            }
        }
    }
    sums.into_iter()
        .filter_map(|(key, (sum, count))| (count > 0).then(|| (key, sum / count as f64)))
        .collect()
}

#[derive(Debug, Serialize)]
struct LayeredGenerationArtifact<'a> {
    schema_version: u32,
    split_label: &'a str,
    active_method: &'a PocketGenerationMethodMetadata,
    candidate_metrics_artifact: String,
    coordinate_frame_contract: &'static str,
    backend_selection: BackendSelectionArtifact,
    layered_metrics: &'a LayeredGenerationMetrics,
    raw_geometry_candidates: Vec<GeneratedCandidateRecord>,
    raw_rollout_candidates: &'a [GeneratedCandidateRecord],
    bond_logits_refined_candidates: Vec<GeneratedCandidateRecord>,
    valence_refined_candidates: Vec<GeneratedCandidateRecord>,
    repaired_candidates: &'a [GeneratedCandidateRecord],
    inferred_bond_candidates: &'a [GeneratedCandidateRecord],
    deterministic_proxy_candidates: &'a [GeneratedCandidateRecord],
    reranked_candidates: &'a [GeneratedCandidateRecord],
    method_layer_outputs: &'a LayeredGenerationOutput,
    backend_metrics: &'a RealGenerationMetrics,
    repair_case_audit: &'a RepairCaseAuditReport,
    backend_failure_examples: Vec<BackendFailureExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendSelectionArtifact {
    primary_backend: BackendConfigArtifact,
    comparison_backends: Vec<BackendConfigArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendConfigArtifact {
    backend_id: String,
    family: String,
    trainable: bool,
    checkpoint_path: Option<String>,
    sampling_steps: Option<usize>,
    sampling_temperature: Option<f64>,
    external_wrapper_enabled: bool,
}

impl BackendSelectionArtifact {
    fn from_research(research: &ResearchConfig) -> Self {
        let primary_backend_id = research.generation_method.primary_backend_id();
        let primary_backend =
            if research.generation_method.primary_backend.backend_id == primary_backend_id {
                BackendConfigArtifact::from_config(&research.generation_method.primary_backend)
            } else {
                BackendConfigArtifact::from_method_id(primary_backend_id)
            };

        Self {
            primary_backend,
            comparison_backends: comparison_backend_artifacts(research),
        }
    }
}

impl BackendConfigArtifact {
    fn from_config(config: &crate::config::GenerationBackendConfig) -> Self {
        Self {
            backend_id: config.backend_id.clone(),
            family: format!("{:?}", config.family).to_ascii_lowercase(),
            trainable: config.trainable,
            checkpoint_path: config
                .checkpoint_path
                .as_ref()
                .map(|path| path.display().to_string()),
            sampling_steps: config.sampling_steps,
            sampling_temperature: config.sampling_temperature,
            external_wrapper_enabled: config.external_wrapper.enabled,
        }
    }

    fn from_method_id(method_id: &str) -> Self {
        let metadata = crate::models::PocketGenerationMethodRegistry::metadata(method_id).ok();
        Self {
            backend_id: method_id.to_string(),
            family: metadata
                .as_ref()
                .map(|metadata| format!("{:?}", metadata.method_family).to_ascii_lowercase())
                .unwrap_or_else(|| "unknown".to_string()),
            trainable: metadata
                .as_ref()
                .map(|metadata| metadata.capability.trainable)
                .unwrap_or(false),
            checkpoint_path: None,
            sampling_steps: None,
            sampling_temperature: None,
            external_wrapper_enabled: metadata
                .as_ref()
                .map(|metadata| metadata.capability.external_wrapper)
                .unwrap_or(false),
        }
    }
}

fn comparison_backend_artifacts(research: &ResearchConfig) -> Vec<BackendConfigArtifact> {
    let mut artifacts = research
        .generation_method
        .comparison_methods
        .iter()
        .map(|method_id| {
            research
                .generation_method
                .comparison_backends
                .iter()
                .find(|backend| backend.backend_id == *method_id)
                .map(BackendConfigArtifact::from_config)
                .unwrap_or_else(|| BackendConfigArtifact::from_method_id(method_id))
        })
        .collect::<Vec<_>>();
    for backend in &research.generation_method.comparison_backends {
        if !artifacts
            .iter()
            .any(|artifact| artifact.backend_id == backend.backend_id)
        {
            artifacts.push(BackendConfigArtifact::from_config(backend));
        }
    }
    artifacts
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendFailureExample {
    backend: String,
    example_id: String,
    protein_id: String,
    candidate_source: String,
    reason: String,
    backend_status: String,
    source_pocket_path: Option<String>,
    source_ligand_path: Option<String>,
}

fn build_repair_case_audit(
    split_label: &str,
    raw_rollout: &[GeneratedCandidateRecord],
    repaired: &[GeneratedCandidateRecord],
    raw_metrics: &CandidateLayerMetrics,
    repaired_metrics: &CandidateLayerMetrics,
) -> RepairCaseAuditReport {
    let paired_count = raw_rollout.len().min(repaired.len());
    let mut help_cases = Vec::new();
    let mut harm_cases = Vec::new();
    let mut neutral_cases = Vec::new();
    let mut raw_failure_cases = Vec::new();

    for index in 0..paired_count {
        let raw = &raw_rollout[index];
        let repaired_candidate = &repaired[index];
        let delta =
            repair_candidate_quality_score(repaired_candidate) - repair_candidate_quality_score(raw);
        if delta > 0.05 {
            help_cases.push(repair_case_record(
                "repair_helps",
                index,
                index,
                raw,
                repaired_candidate,
            ));
        } else if delta < -0.05 {
            harm_cases.push(repair_case_record(
                "repair_harms",
                index,
                index,
                raw,
                repaired_candidate,
            ));
        } else {
            neutral_cases.push(repair_case_record(
                "repair_neutral",
                index,
                index,
                raw,
                repaired_candidate,
            ));
        }
        if repair_raw_failure(raw) {
            raw_failure_cases.push(repair_case_record(
                "raw_failure",
                index,
                index,
                raw,
                repaired_candidate,
            ));
        }
    }

    if paired_count == 0 {
        for (index, raw) in raw_rollout.iter().enumerate() {
            if repair_raw_failure(raw) {
                raw_failure_cases.push(repair_case_record(
                    "raw_failure",
                    index,
                    index,
                    raw,
                    raw,
                ));
            }
        }
    }

    help_cases.sort_by(|left, right| {
        right
            .strict_pocket_fit_delta
            .partial_cmp(&left.strict_pocket_fit_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    harm_cases.sort_by(|left, right| {
        left.strict_pocket_fit_delta
            .partial_cmp(&right.strict_pocket_fit_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    neutral_cases.sort_by(|left, right| {
        left.strict_pocket_fit_delta
            .abs()
            .partial_cmp(&right.strict_pocket_fit_delta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    raw_failure_cases.sort_by(|left, right| {
        left.raw_metrics
            .strict_pocket_fit_score
            .partial_cmp(&right.raw_metrics.strict_pocket_fit_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut case_counts = BTreeMap::new();
    case_counts.insert("repair_helps".to_string(), help_cases.len());
    case_counts.insert("repair_harms".to_string(), harm_cases.len());
    case_counts.insert("repair_neutral".to_string(), neutral_cases.len());
    case_counts.insert("raw_failure".to_string(), raw_failure_cases.len());

    RepairCaseAuditReport {
        schema_version: default_repair_case_audit_schema_version(),
        split_label: split_label.to_string(),
        raw_vs_repaired_delta: RepairLayerDeltaSummary {
            raw_layer: "raw_rollout".to_string(),
            repaired_layer: "repaired_candidates".to_string(),
            raw_candidate_count: raw_metrics.candidate_count,
            repaired_candidate_count: repaired_metrics.candidate_count,
            valid_fraction_delta: repaired_metrics.valid_fraction - raw_metrics.valid_fraction,
            pocket_contact_fraction_delta: repaired_metrics.pocket_contact_fraction
                - raw_metrics.pocket_contact_fraction,
            strict_pocket_fit_score_delta: layer_native_quality(repaired_metrics)
                - layer_native_quality(raw_metrics),
            mean_centroid_offset_delta: repaired_metrics.mean_centroid_offset
                - raw_metrics.mean_centroid_offset,
            clash_fraction_delta: repaired_metrics.clash_fraction - raw_metrics.clash_fraction,
            native_graph_valid_fraction_delta: repaired_metrics.native_graph_valid_fraction
                - raw_metrics.native_graph_valid_fraction,
        },
        no_repair_ablation: NoRepairAblationMetrics {
            repair_enabled: !repaired.is_empty(),
            no_repair_layer: "raw_rollout".to_string(),
            no_repair_metrics: raw_metrics.clone(),
            interpretation:
                "raw_rollout is the no-repair baseline for this run; repaired layers are postprocessing evidence only"
                    .to_string(),
        },
        case_counts,
        repair_helps: help_cases.into_iter().take(8).collect(),
        repair_harms: harm_cases.into_iter().take(8).collect(),
        repair_neutral: neutral_cases.into_iter().take(8).collect(),
        raw_failures: raw_failure_cases.into_iter().take(8).collect(),
        artifact_name: Some(format!("repair_case_audit_{split_label}.json")),
        claim_boundary:
            "Repair-case audit is postprocessing evidence: raw_rollout remains the raw model-native no-repair baseline, and repaired improvements must not be cited as raw generation quality."
                .to_string(),
    }
}

fn repair_case_record(
    role: &str,
    raw_index: usize,
    repaired_index: usize,
    raw: &GeneratedCandidateRecord,
    repaired: &GeneratedCandidateRecord,
) -> RepairCaseRecord {
    let raw_metrics = repair_candidate_snapshot(raw);
    let repaired_metrics = repair_candidate_snapshot(repaired);
    let strict_pocket_fit_delta =
        repaired_metrics.strict_pocket_fit_score - raw_metrics.strict_pocket_fit_score;
    RepairCaseRecord {
        case_role: role.to_string(),
        example_id: raw.example_id.clone(),
        protein_id: raw.protein_id.clone(),
        raw_candidate_index: raw_index,
        repaired_candidate_index: repaired_index,
        strict_pocket_fit_delta,
        raw_metrics,
        repaired_metrics,
        postprocessor_chain: repaired.postprocessor_chain.clone(),
        interpretation: match role {
            "repair_helps" => {
                "repair improves strict pocket-fit proxy for this paired candidate; cite only as postprocessing-dependent evidence"
            }
            "repair_harms" => {
                "repair degrades strict pocket-fit proxy for this paired candidate; preserve as a failure case"
            }
            "raw_failure" => {
                "raw no-repair candidate fails at least one validity, graph, contact, clash, or pocket-fit check before repair"
            }
            _ => "repair is approximately neutral for this paired candidate under the strict pocket-fit proxy",
        }
        .to_string(),
    }
}

fn repair_candidate_snapshot(candidate: &GeneratedCandidateRecord) -> RepairCandidateMetricSnapshot {
    RepairCandidateMetricSnapshot {
        valid: candidate_is_valid(candidate),
        native_graph_valid: candidate_native_graph_valid(candidate),
        pocket_contact: candidate_has_pocket_contact(candidate),
        centroid_offset: candidate_centroid_offset(candidate),
        clash_fraction: candidate_clash_fraction(candidate),
        strict_pocket_fit_score: repair_candidate_quality_score(candidate),
        bond_count: candidate.bond_count,
        component_count: native_component_count(candidate),
        valence_violation_fraction: candidate_valence_violation_fraction(candidate),
    }
}

fn repair_candidate_quality_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let valid = if candidate_is_valid(candidate) { 1.0 } else { 0.0 };
    let contact = if candidate_has_pocket_contact(candidate) {
        1.0
    } else {
        0.0
    };
    let centroid_fit = 1.0 / (1.0 + candidate_centroid_offset(candidate).max(0.0));
    let clash_fit = 1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0);
    (valid * contact * centroid_fit * clash_fit).clamp(0.0, 1.0)
}

fn repair_raw_failure(candidate: &GeneratedCandidateRecord) -> bool {
    !candidate_is_valid(candidate)
        || !candidate_native_graph_valid(candidate)
        || !candidate_has_pocket_contact(candidate)
        || candidate_clash_fraction(candidate) > 0.1
        || repair_candidate_quality_score(candidate) < 0.25
}

fn maybe_persist_generation_artifacts(
    research: &ResearchConfig,
    external_evaluation: &ExternalEvaluationConfig,
    split_label: &str,
    raw_rollout: &[GeneratedCandidateRecord],
    repaired: &[GeneratedCandidateRecord],
    inferred_bond: &[GeneratedCandidateRecord],
    reranked: &[GeneratedCandidateRecord],
    deterministic_proxy: &[GeneratedCandidateRecord],
    backend_metrics: &RealGenerationMetrics,
    layered_metrics: &LayeredGenerationMetrics,
    method_layer_outputs: &LayeredGenerationOutput,
) {
    if !external_evaluation.persist_generation_artifacts {
        return;
    }
    if fs::create_dir_all(&research.training.checkpoint_dir).is_err() {
        return;
    }
    let artifact = LayeredGenerationArtifact {
        schema_version: 3,
        split_label,
        active_method: &method_layer_outputs.metadata,
        candidate_metrics_artifact: format!("candidate_metrics_{split_label}.jsonl"),
        coordinate_frame_contract: candidate_coordinate_frame_contract(),
        backend_selection: BackendSelectionArtifact::from_research(research),
        layered_metrics,
        raw_geometry_candidates: method_layer_outputs
            .raw_geometry
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        raw_rollout_candidates: raw_rollout,
        bond_logits_refined_candidates: method_layer_outputs
            .bond_logits_refined
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        valence_refined_candidates: method_layer_outputs
            .valence_refined
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        repaired_candidates: repaired,
        inferred_bond_candidates: inferred_bond,
        deterministic_proxy_candidates: deterministic_proxy,
        reranked_candidates: reranked,
        method_layer_outputs,
        backend_metrics,
        repair_case_audit: &layered_metrics.repair_case_audit,
        backend_failure_examples: backend_failure_examples(inferred_bond, backend_metrics, 16),
    };
    if let Ok(content) = serde_json::to_string_pretty(&artifact) {
        let path = research
            .training
            .checkpoint_dir
            .join(format!("generation_layers_{split_label}.json"));
        if let Err(error) = fs::write(&path, content) {
            log::warn!(
                "failed to persist generation layer artifact {}: {}",
                path.display(),
                error
            );
        }
    }
    persist_candidate_metrics_jsonl(
        &research
            .training
            .checkpoint_dir
            .join(format!("candidate_metrics_{split_label}.jsonl")),
        split_label,
        method_layer_outputs.metadata.method_id.as_str(),
        [
            ("raw_flow", raw_rollout),
            ("repaired", repaired),
            ("constrained_flow", inferred_bond),
            ("deterministic_proxy", deterministic_proxy),
            ("reranked", reranked),
        ],
        backend_metrics,
    );
    persist_json_artifact(
        &research
            .training
            .checkpoint_dir
            .join(format!("repair_case_audit_{split_label}.json")),
        &layered_metrics.repair_case_audit,
        "repair case audit",
    );
}

#[derive(Debug, Serialize)]
struct CandidateMetricRecord<'a> {
    schema_version: u32,
    candidate_id: String,
    example_id: &'a str,
    protein_id: &'a str,
    split_label: &'a str,
    layer: &'a str,
    method_id: &'a str,
    candidate_source: &'a str,
    coordinate_frame_contract: &'static str,
    coordinate_frame_origin: [f32; 3],
    metrics: BTreeMap<String, f64>,
    backend_statuses: BTreeMap<String, String>,
    source_artifacts: BTreeMap<String, String>,
}

fn persist_candidate_metrics_jsonl<'a>(
    path: &Path,
    split_label: &'a str,
    method_id: &'a str,
    layers: impl IntoIterator<Item = (&'a str, &'a [GeneratedCandidateRecord])>,
    backend_metrics: &RealGenerationMetrics,
) {
    let backend_statuses = BTreeMap::from([
        (
            "chemistry_validity".to_string(),
            backend_metrics.chemistry_validity.status.clone(),
        ),
        (
            "docking_affinity".to_string(),
            backend_metrics.docking_affinity.status.clone(),
        ),
        (
            "pocket_compatibility".to_string(),
            backend_metrics.pocket_compatibility.status.clone(),
        ),
    ]);
    let mut lines = Vec::new();
    for (layer, candidates) in layers {
        for (index, candidate) in candidates.iter().enumerate() {
            let candidate_id = format!("{method_id}:{layer}:{}:{}", candidate.example_id, index);
            let record = CandidateMetricRecord {
                schema_version: 1,
                candidate_id,
                example_id: &candidate.example_id,
                protein_id: &candidate.protein_id,
                split_label,
                layer,
                method_id,
                candidate_source: &candidate.source,
                coordinate_frame_contract: candidate_coordinate_frame_contract(),
                coordinate_frame_origin: candidate.coordinate_frame_origin,
                metrics: candidate_metric_map(candidate),
                backend_statuses: backend_statuses.clone(),
                source_artifacts: candidate_source_artifacts(candidate),
            };
            if let Ok(line) = serde_json::to_string(&record) {
                lines.push(line);
            }
        }
    }
    if let Err(error) = fs::write(path, lines.join("\n") + "\n") {
        log::warn!(
            "failed to persist candidate metric artifact {}: {}",
            path.display(),
            error
        );
    }
}

fn candidate_coordinate_frame_contract() -> &'static str {
    "candidate.coords are ligand-centered model-frame coordinates; coordinate_frame_origin reconstructs source-frame coordinates"
}

fn candidate_metric_map(candidate: &GeneratedCandidateRecord) -> BTreeMap<String, f64> {
    BTreeMap::from([
        (
            "valid_fraction".to_string(),
            if candidate_is_valid(candidate) {
                1.0
            } else {
                0.0
            },
        ),
        (
            "pocket_contact_fraction".to_string(),
            if candidate_has_pocket_contact(candidate) {
                1.0
            } else {
                0.0
            },
        ),
        (
            "mean_centroid_offset".to_string(),
            candidate_centroid_offset(candidate),
        ),
        (
            "clash_fraction".to_string(),
            candidate_clash_fraction(candidate),
        ),
        (
            "hydrogen_bond_proxy".to_string(),
            candidate_hydrogen_bond_proxy(candidate),
        ),
        (
            "hydrophobic_contact_proxy".to_string(),
            candidate_hydrophobic_contact_proxy(candidate),
        ),
        (
            "contact_balance".to_string(),
            candidate_contact_balance(candidate),
        ),
        (
            "scaffold_metric_coverage_fraction".to_string(),
            if candidate_structural_fingerprint(candidate).is_some() {
                1.0
            } else {
                0.0
            },
        ),
    ])
}

fn candidate_source_artifacts(candidate: &GeneratedCandidateRecord) -> BTreeMap<String, String> {
    let mut artifacts = BTreeMap::new();
    if let Some(path) = &candidate.source_pocket_path {
        artifacts.insert("source_pocket_path".to_string(), path.clone());
    }
    if let Some(path) = &candidate.source_ligand_path {
        artifacts.insert("source_ligand_path".to_string(), path.clone());
    }
    artifacts
}

fn build_and_persist_preference_artifacts(
    research: &ResearchConfig,
    split_label: &str,
    raw_rollout: &[GeneratedCandidateRecord],
    repaired: &[GeneratedCandidateRecord],
    inferred_bond: &[GeneratedCandidateRecord],
    deterministic_proxy: &[GeneratedCandidateRecord],
    reranked: &[GeneratedCandidateRecord],
    backend_metrics: &RealGenerationMetrics,
) -> PreferenceAlignmentSummary {
    let config = &research.preference_alignment;
    let mut summary = PreferenceAlignmentSummary {
        profile_extraction_enabled: config.enable_profile_extraction,
        pair_construction_enabled: config.enable_pair_construction,
        missing_artifacts_mean_unavailable: config.missing_artifacts_mean_unavailable,
        ..PreferenceAlignmentSummary::default()
    };
    if !config.enable_profile_extraction {
        return summary;
    }

    if fs::create_dir_all(&research.training.checkpoint_dir).is_err() {
        return summary;
    }

    let backend_reports = preference_backend_reports(backend_metrics);
    let active_method_id = research.generation_method.primary_backend_id();
    let mut profiles = Vec::new();
    profiles.extend(extract_interaction_profiles(
        raw_rollout,
        crate::models::CandidateLayerKind::RawRollout,
        Some(active_method_id),
        &backend_reports,
    ));
    profiles.extend(extract_interaction_profiles(
        repaired,
        crate::models::CandidateLayerKind::Repaired,
        Some(active_method_id),
        &backend_reports,
    ));
    profiles.extend(extract_interaction_profiles(
        inferred_bond,
        crate::models::CandidateLayerKind::InferredBond,
        Some(active_method_id),
        &backend_reports,
    ));
    profiles.extend(extract_interaction_profiles(
        deterministic_proxy,
        crate::models::CandidateLayerKind::DeterministicProxy,
        Some(active_method_id),
        &backend_reports,
    ));
    profiles.extend(extract_interaction_profiles(
        reranked,
        crate::models::CandidateLayerKind::Reranked,
        Some(active_method_id),
        &backend_reports,
    ));

    let profile_artifact_name = format!("preference_profiles_{split_label}.json");
    let profile_artifact = PreferenceProfileArtifact::new(split_label, profiles.clone());
    summary.profile_count = profile_artifact.profile_count;
    summary.profile_artifact = Some(profile_artifact_name.clone());
    persist_json_artifact(
        &research
            .training
            .checkpoint_dir
            .join(&profile_artifact_name),
        &profile_artifact,
        "preference profile",
    );

    let mut pairs = Vec::new();
    if config.enable_pair_construction {
        pairs = build_bounded_preference_pairs(
            &profiles,
            PreferenceConstructionConfig {
                max_clash_fraction: config.max_clash_fraction,
                min_strict_pocket_fit_score: config.min_strict_pocket_fit_score,
                require_valence_sane: true,
                min_soft_margin: config.min_soft_margin,
            },
            config.max_pairs_per_example,
        );
        let pair_artifact_name = format!("preference_pairs_{split_label}.json");
        let pair_artifact = PreferencePairArtifact::new(split_label, &profiles, pairs.clone());
        summary.preference_pair_count = pair_artifact.pair_count;
        summary.preference_pair_artifact = Some(pair_artifact_name.clone());
        persist_json_artifact(
            &research.training.checkpoint_dir.join(&pair_artifact_name),
            &pair_artifact,
            "preference pair",
        );
    }

    if config.enable_preference_reranking {
        let selected_count = RuleBasedPreferenceReranker::default()
            .rerank_profiles(&profiles)
            .len()
            .min((profiles.len() / 2).max(1));
        let reranker_summary_name = "preference_reranker_summary.json".to_string();
        let reranker_summary = preference_reranker_summary(
            config.enable_preference_reranking,
            profiles.len(),
            selected_count,
            &pairs,
        );
        summary.reranker_summary_artifact = Some(reranker_summary_name.clone());
        persist_json_artifact(
            &research
                .training
                .checkpoint_dir
                .join(&reranker_summary_name),
            &reranker_summary,
            "preference reranker summary",
        );
    }

    summary
}

fn preference_backend_reports(metrics: &RealGenerationMetrics) -> Vec<ExternalEvaluationReport> {
    [
        ("chemistry_validity", &metrics.chemistry_validity.metrics),
        ("docking_affinity", &metrics.docking_affinity.metrics),
        (
            "pocket_compatibility",
            &metrics.pocket_compatibility.metrics,
        ),
    ]
    .into_iter()
    .map(|(backend_name, metrics)| ExternalEvaluationReport {
        backend_name: backend_name.to_string(),
        metrics: metrics
            .iter()
            .map(|(metric_name, value)| ExternalMetricRecord {
                metric_name: metric_name.clone(),
                value: *value,
            })
            .collect(),
    })
    .collect()
}

fn preference_reranker_summary(
    enabled: bool,
    candidate_count: usize,
    selected_count: usize,
    pairs: &[crate::models::PreferencePair],
) -> PreferenceRerankerSummaryArtifact {
    let pair_count = pairs.len() as f64;
    let rule_based = pairs
        .iter()
        .filter(|pair| pair.preference_source == crate::models::PreferenceSource::RuleBased)
        .count() as f64;
    let backend_based = pairs
        .iter()
        .filter(|pair| pair.preference_source == crate::models::PreferenceSource::BackendBased)
        .count() as f64;
    let hard_constraint_wins = pairs
        .iter()
        .filter(|pair| pair.hard_constraint_flags.values().any(|value| *value))
        .count();
    let soft_preference_wins = pairs
        .iter()
        .filter(|pair| pair.soft_preference_flags.values().any(|value| *value))
        .count();
    let mean_preference_strength = if pair_count > 0.0 {
        pairs
            .iter()
            .map(|pair| pair.preference_strength)
            .sum::<f64>()
            / pair_count
    } else {
        0.0
    };
    let backend_supported_pair_fraction = if pair_count > 0.0 {
        backend_based / pair_count
    } else {
        0.0
    };
    let rule_only_pair_fraction = if pair_count > 0.0 {
        rule_based / pair_count
    } else {
        0.0
    };
    PreferenceRerankerSummaryArtifact {
        schema_version: crate::models::PREFERENCE_RERANKER_SCHEMA_VERSION,
        enabled,
        candidate_count,
        selected_count,
        feature_weights: RuleBasedPreferenceReranker::feature_weights(),
        rule_based_pair_fraction: rule_only_pair_fraction,
        backend_based_pair_fraction: backend_supported_pair_fraction,
        hard_constraint_wins,
        backend_supported_pair_fraction,
        rule_only_pair_fraction,
        missing_backend_evidence_fraction: 1.0 - backend_supported_pair_fraction,
        mean_preference_strength,
        hard_constraint_win_fraction: if pair_count > 0.0 {
            hard_constraint_wins as f64 / pair_count
        } else {
            0.0
        },
        soft_preference_wins,
    }
}

fn persist_json_artifact<T: Serialize>(path: &Path, value: &T, label: &str) {
    match serde_json::to_string_pretty(value) {
        Ok(content) => {
            if let Err(error) = fs::write(path, content) {
                log::warn!(
                    "failed to persist {label} artifact {}: {}",
                    path.display(),
                    error
                );
            }
        }
        Err(error) => {
            log::warn!(
                "failed to serialize {label} artifact {}: {}",
                path.display(),
                error
            );
        }
    }
}

fn backend_failure_examples(
    candidates: &[GeneratedCandidateRecord],
    metrics: &RealGenerationMetrics,
    limit: usize,
) -> Vec<BackendFailureExample> {
    let mut examples = Vec::new();
    collect_backend_command_failures(
        &mut examples,
        "chemistry_validity",
        &metrics.chemistry_validity,
        candidates,
        limit,
    );
    collect_backend_command_failures(
        &mut examples,
        "docking_affinity",
        &metrics.docking_affinity,
        candidates,
        limit,
    );
    collect_backend_command_failures(
        &mut examples,
        "pocket_compatibility",
        &metrics.pocket_compatibility,
        candidates,
        limit,
    );

    for candidate in candidates {
        if examples.len() >= limit {
            break;
        }
        if !candidate_is_valid(candidate) {
            examples.push(failure_example(
                "chemistry_validity",
                candidate,
                "candidate has inconsistent atom/coordinate lengths or non-finite coordinates",
                &metrics.chemistry_validity.status,
            ));
        } else if !valence_sane_proxy(candidate) {
            examples.push(failure_example(
                "chemistry_validity",
                candidate,
                "candidate exceeds lightweight valence sanity proxy",
                &metrics.chemistry_validity.status,
            ));
        }
        if examples.len() >= limit {
            break;
        }
        if candidate.source_pocket_path.is_none() {
            examples.push(failure_example(
                "pocket_compatibility",
                candidate,
                "candidate is missing source pocket structure provenance",
                &metrics.pocket_compatibility.status,
            ));
        } else if candidate_clash_fraction(candidate) > 0.0 {
            examples.push(failure_example(
                "pocket_compatibility",
                candidate,
                "candidate has nonzero lightweight pocket clash fraction",
                &metrics.pocket_compatibility.status,
            ));
        }
    }
    examples
}

fn collect_backend_command_failures(
    failures: &mut Vec<BackendFailureExample>,
    backend: &str,
    metrics: &ReservedBackendMetrics,
    candidates: &[GeneratedCandidateRecord],
    limit: usize,
) {
    if failures.len() >= limit {
        return;
    }
    let reason = backend_command_failure_reason(metrics).or_else(|| {
        let missing = metrics
            .metrics
            .get("backend_missing_structure_fraction")
            .copied()
            .unwrap_or(0.0);
        (missing > 0.0).then(|| {
            format!("backend_missing_structure_fraction={missing:.4} for scored candidates")
        })
    });
    let Some(reason) = reason else {
        return;
    };
    for candidate in candidates.iter().take(limit.saturating_sub(failures.len())) {
        failures.push(failure_example(
            backend,
            candidate,
            &reason,
            &metrics.status,
        ));
    }
}

fn backend_command_failure_reason(metrics: &ReservedBackendMetrics) -> Option<String> {
    if let Some((name, _)) = metrics
        .metrics
        .iter()
        .find(|(name, value)| name.ends_with("_available") && **value <= 0.0)
    {
        return Some(format!("{name}=0"));
    }
    [
        "backend_command_failed",
        "backend_command_spawn_error",
        "backend_output_parse_error",
        "backend_missing_schema_version",
        "backend_missing_examples_scored",
        "backend_missing_executable",
        "backend_tempfile_error",
        "backend_input_write_error",
    ]
    .iter()
    .find(|name| metrics.metrics.get(**name).copied().unwrap_or(0.0) > 0.0)
    .map(|name| format!("{name}=1"))
}

fn failure_example(
    backend: &str,
    candidate: &GeneratedCandidateRecord,
    reason: &str,
    backend_status: &str,
) -> BackendFailureExample {
    BackendFailureExample {
        backend: backend.to_string(),
        example_id: candidate.example_id.clone(),
        protein_id: candidate.protein_id.clone(),
        candidate_source: candidate.source.clone(),
        reason: reason.to_string(),
        backend_status: backend_status.to_string(),
        source_pocket_path: candidate.source_pocket_path.clone(),
        source_ligand_path: candidate.source_ligand_path.clone(),
    }
}

fn candidate_is_valid(candidate: &GeneratedCandidateRecord) -> bool {
    !candidate.atom_types.is_empty()
        && candidate.atom_types.len() == candidate.coords.len()
        && candidate
            .coords
            .iter()
            .all(|coord| coord.iter().all(|value| value.is_finite()))
}

fn candidate_has_pocket_contact(candidate: &GeneratedCandidateRecord) -> bool {
    candidate.coords.iter().any(|coord| {
        coord_distance(coord, &candidate.pocket_centroid) <= (candidate.pocket_radius + 2.0) as f64
    })
}

fn candidate_centroid_offset(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return f64::INFINITY;
    }
    let mut centroid = [0.0_f64; 3];
    for coord in &candidate.coords {
        centroid[0] += coord[0] as f64;
        centroid[1] += coord[1] as f64;
        centroid[2] += coord[2] as f64;
    }
    let denom = candidate.coords.len() as f64;
    let centroid = [
        (centroid[0] / denom) as f32,
        (centroid[1] / denom) as f32,
        (centroid[2] / denom) as f32,
    ];
    coord_distance(&centroid, &candidate.pocket_centroid)
}

fn candidate_clash_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.len() < 2 {
        return 0.0;
    }
    let bonds = candidate
        .inferred_bonds
        .iter()
        .map(|&(left, right)| {
            if left < right {
                (left, right)
            } else {
                (right, left)
            }
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut total = 0_usize;
    let mut clashing = 0_usize;
    for left in 0..candidate.coords.len() {
        for right in (left + 1)..candidate.coords.len() {
            if bonds.contains(&(left, right)) {
                continue;
            }
            total += 1;
            if coord_distance(&candidate.coords[left], &candidate.coords[right]) < 0.9 {
                clashing += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        clashing as f64 / total as f64
    }
}

fn candidate_backend_pocket_clash_fraction(candidate: &GeneratedCandidateRecord) -> Option<f64> {
    let pocket_path = candidate.source_pocket_path.as_deref()?;
    let pocket_coords = parse_pdb_coords(pocket_path)?;
    let candidate_coords = backend_candidate_coords(candidate);
    if pocket_coords.is_empty() || candidate_coords.is_empty() {
        return None;
    }
    let clashing = candidate_coords
        .iter()
        .filter(|coord| {
            pocket_coords
                .iter()
                .any(|pocket| coord_distance(coord, pocket) < 1.2)
        })
        .count();
    Some(clashing as f64 / candidate_coords.len() as f64)
}

fn candidate_backend_atom_coverage_fraction(candidate: &GeneratedCandidateRecord) -> Option<f64> {
    let pocket_path = candidate.source_pocket_path.as_deref()?;
    let pocket_coords = parse_pdb_coords(pocket_path)?;
    let candidate_coords = backend_candidate_coords(candidate);
    if pocket_coords.is_empty() || candidate_coords.is_empty() {
        return None;
    }
    let covered = candidate_coords
        .iter()
        .filter(|coord| {
            pocket_coords
                .iter()
                .any(|pocket| coord_distance(coord, pocket) <= 3.5)
        })
        .count();
    Some(covered as f64 / candidate_coords.len() as f64)
}

fn backend_claim_selection_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let clash_penalty = candidate_backend_pocket_clash_fraction(candidate).unwrap_or(1.0);
    let coverage = candidate_backend_atom_coverage_fraction(candidate).unwrap_or(0.0);
    let centroid_fit = 1.0 / (1.0 + candidate_centroid_offset(candidate).max(0.0));
    let valence_bonus = if valence_sane_proxy(candidate) {
        1.0
    } else {
        0.98
    };
    ((coverage * centroid_fit * valence_bonus) - clash_penalty).clamp(0.0, 1.0)
}

fn backend_candidate_coords(candidate: &GeneratedCandidateRecord) -> Vec<[f32; 3]> {
    let origin = candidate.coordinate_frame_origin;
    candidate
        .coords
        .iter()
        .map(|coord| {
            [
                coord[0] + origin[0],
                coord[1] + origin[1],
                coord[2] + origin[2],
            ]
        })
        .collect()
}

fn parse_pdb_coords(path: &str) -> Option<Vec<[f32; 3]>> {
    let content = fs::read_to_string(path).ok()?;
    let mut coords = Vec::new();
    for line in content.lines() {
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        let x = line.get(30..38)?.trim().parse::<f32>().ok()?;
        let y = line.get(38..46)?.trim().parse::<f32>().ok()?;
        let z = line.get(46..54)?.trim().parse::<f32>().ok()?;
        coords.push([x, y, z]);
    }
    Some(coords)
}

#[allow(dead_code)] // Kept for audit tooling that needs a single legacy uniqueness key.
fn candidate_uniqueness_signature(candidate: &GeneratedCandidateRecord) -> String {
    format!(
        "{}::{}::{}",
        candidate_atom_signature(candidate),
        candidate_bond_signature(candidate),
        candidate_shape_signature(candidate)
    )
}

fn candidate_atom_signature(candidate: &GeneratedCandidateRecord) -> String {
    format!("{:?}", candidate.atom_types)
}

fn example_atom_signature(example: &crate::data::MolecularExample) -> String {
    let atom_count = ligand_atom_count(example);
    let mut atom_types = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        atom_types.push(example.topology.atom_types.int64_value(&[index as i64]));
    }
    format!("{:?}", atom_types)
}

fn candidate_bond_signature(candidate: &GeneratedCandidateRecord) -> String {
    let mut bonds = candidate
        .inferred_bonds
        .iter()
        .map(|(left, right)| {
            let (low, high) = if left <= right {
                (*left, *right)
            } else {
                (*right, *left)
            };
            format!("{low}-{high}")
        })
        .collect::<Vec<_>>();
    bonds.sort();
    bonds.join("|")
}

fn example_bond_signature(example: &crate::data::MolecularExample) -> String {
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
                bonds.push(format!("{left}-{right}"));
            }
        }
    }
    bonds.sort();
    bonds.join("|")
}

fn candidate_shape_signature(candidate: &GeneratedCandidateRecord) -> String {
    let coord_buckets = candidate
        .coords
        .iter()
        .map(|coord| {
            format!(
                "{:.1}:{:.1}:{:.1}",
                coord[0] as f64, coord[1] as f64, coord[2] as f64
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    coord_buckets
}

fn example_shape_signature(example: &crate::data::MolecularExample) -> String {
    let atom_count = ligand_atom_count(example);
    let mut coord_buckets = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        let x = example.geometry.coords.double_value(&[index as i64, 0]) as f32;
        let y = example.geometry.coords.double_value(&[index as i64, 1]) as f32;
        let z = example.geometry.coords.double_value(&[index as i64, 2]) as f32;
        coord_buckets.push(format!("{:.1}:{:.1}:{:.1}", x as f64, y as f64, z as f64));
    }
    coord_buckets.join("|")
}

fn coord_distance(left: &[f32; 3], right: &[f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}
