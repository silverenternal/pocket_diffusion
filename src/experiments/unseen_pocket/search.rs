pub fn load_experiment_config(
    path: impl AsRef<std::path::Path>,
) -> Result<UnseenPocketExperimentConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&content)?;
    if value.get("research").is_some() {
        return Ok(serde_json::from_value(value)?);
    }
    let research: ResearchConfig = serde_json::from_value(value)?;
    Ok(UnseenPocketExperimentConfig {
        research,
        ..UnseenPocketExperimentConfig::default()
    })
}

/// Execute the bounded multi-surface automated search rooted at one config.
pub fn run_automated_search(
    orchestrator_path: impl AsRef<Path>,
) -> Result<AutomatedSearchSummary, Box<dyn std::error::Error>> {
    let orchestrator_path = orchestrator_path.as_ref();
    let orchestrator_dir = orchestrator_path.parent().unwrap_or_else(|| Path::new("."));
    let config = load_experiment_config(orchestrator_path)?;
    if !config.automated_search.enabled {
        return Err("automated_search.enabled must be true for the search entrypoint".into());
    }
    validate_experiment_config_with_source(&config, orchestrator_path)?;
    let search = &config.automated_search;
    for surface_path in &search.surface_configs {
        let resolved = resolve_relative_path(orchestrator_dir, surface_path);
        if !resolved.is_file() {
            return Err(format!("search surface config not found: {}", resolved.display()).into());
        }
    }
    let artifact_root = resolve_output_artifact_path(orchestrator_dir, &search.artifact_root_dir);
    fs::create_dir_all(&artifact_root)?;

    let candidates = build_search_candidates(&config.research, search)?;
    let mut summaries = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        summaries.push(run_search_candidate(
            orchestrator_dir,
            &artifact_root,
            search,
            &candidate,
        )?);
    }
    summaries.sort_by(search_candidate_rank_cmp);
    let winning_candidate_id = summaries
        .iter()
        .find(|candidate| candidate.gate_result.passed)
        .map(|candidate| candidate.candidate_id.clone());
    let roadmap_decision = summarize_search_roadmap(&summaries);
    let summary = AutomatedSearchSummary {
        artifact_root: artifact_root.clone(),
        strategy: search.strategy,
        hard_gates: search.hard_gates.clone(),
        score_weights: search.score_weights.clone(),
        ranked_candidates: summaries,
        winning_candidate_id,
        roadmap_decision,
    };
    fs::write(
        artifact_root.join("search_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

/// Execute repeated experiment runs and persist aggregate seed-level uncertainty.
pub fn run_multi_seed_experiment(
    path: impl AsRef<Path>,
) -> Result<MultiSeedExperimentSummary, Box<dyn std::error::Error>> {
    let source_path = path.as_ref();
    let source_dir = source_path.parent().unwrap_or_else(|| Path::new("."));
    let config = load_experiment_config(source_path)?;
    if !config.multi_seed.enabled {
        return Err("multi_seed.enabled must be true for the multi-seed entrypoint".into());
    }
    validate_experiment_config_with_source(&config, source_path)?;
    let artifact_root = resolve_output_artifact_path(source_dir, &config.multi_seed.artifact_root_dir);
    fs::create_dir_all(&artifact_root)?;

    let mut seed_runs = Vec::with_capacity(config.multi_seed.seeds.len());
    for seed in &config.multi_seed.seeds {
        let mut seed_config = config.clone();
        seed_config.multi_seed.enabled = false;
        seed_config.research.data.split_seed = *seed;
        seed_config.research.data.generation_target.corruption_seed = seed.saturating_add(10_000);
        seed_config.research.data.generation_target.sampling_seed = seed.saturating_add(20_000);
        seed_config.research.training.checkpoint_dir = artifact_root.join(format!("seed_{seed}"));
        let run_summary = UnseenPocketExperiment::run_with_options(seed_config, false)?;
        seed_runs.push(MultiSeedRunSummary::from_experiment(*seed, &run_summary));
    }
    let summary = MultiSeedExperimentSummary {
        artifact_root: artifact_root.clone(),
        source_config: source_path.to_path_buf(),
        seed_runs,
        aggregates: MultiSeedAggregateReport::default(),
        stability_decision: String::new(),
    }
    .with_aggregates();
    fs::write(
        artifact_root.join("multi_seed_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

fn run_search_candidate(
    orchestrator_dir: &Path,
    artifact_root: &Path,
    search: &AutomatedSearchConfig,
    candidate: &SearchCandidate,
) -> Result<AutomatedSearchCandidateSummary, Box<dyn std::error::Error>> {
    let candidate_dir = artifact_root.join(&candidate.id);
    fs::create_dir_all(&candidate_dir)?;

    let mut surface_summaries = Vec::with_capacity(search.surface_configs.len());
    for surface_path in &search.surface_configs {
        let source_config = resolve_relative_path(orchestrator_dir, surface_path);
        let mut surface_config = load_experiment_config(&source_config)?;
        validate_experiment_config_with_source(&surface_config, &source_config)?;
        apply_search_candidate(
            &mut surface_config,
            candidate,
            &candidate_dir,
            &source_config,
        );
        let summary = UnseenPocketExperiment::run_with_options(surface_config, false)?;
        let claim_path = summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("claim_summary.json");
        let claim_report: ClaimReport = serde_json::from_str(&fs::read_to_string(&claim_path)?)?;
        surface_summaries.push(AutomatedSearchSurfaceSummary {
            surface_label: surface_label_from_config(&summary.config, &source_config),
            source_config: source_config.clone(),
            artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
            claim_report,
        });
    }

    let review_path = candidate_dir.join("interaction_mode_review.json");
    let aggregate_interaction_review = if review_path.exists() {
        Some(serde_json::from_str(&fs::read_to_string(review_path)?)?)
    } else {
        None
    };
    let gate_result = evaluate_search_gates(&surface_summaries, &search.hard_gates);
    let score = if gate_result.passed {
        Some(score_search_candidate(
            &surface_summaries,
            aggregate_interaction_review.as_ref(),
            &search.score_weights,
        ))
    } else {
        None
    };
    let candidate_summary = AutomatedSearchCandidateSummary {
        candidate_id: candidate.id.clone(),
        overrides: candidate
            .overrides
            .iter()
            .map(SearchOverride::label)
            .collect(),
        artifact_dir: candidate_dir.clone(),
        surfaces: surface_summaries,
        aggregate_interaction_review,
        gate_result,
        score,
    };
    fs::write(
        candidate_dir.join("candidate_summary.json"),
        serde_json::to_string_pretty(&candidate_summary)?,
    )?;
    Ok(candidate_summary)
}

fn build_search_candidates(
    base: &ResearchConfig,
    search: &AutomatedSearchConfig,
) -> Result<Vec<SearchCandidate>, Box<dyn std::error::Error>> {
    let axes = search_axes(base, &search.search_space);
    if axes.is_empty() {
        return Err("automated_search.search_space produced no candidate axes".into());
    }

    let mut candidates = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    if search.include_base_candidate {
        let base_candidate = SearchCandidate {
            id: "candidate_000_base".to_string(),
            overrides: Vec::new(),
        };
        seen.insert(candidate_signature(&base_candidate));
        candidates.push(base_candidate);
    }

    let mut enumerated = match search.strategy {
        AutomatedSearchStrategy::Grid => build_grid_candidates(&axes, search.max_candidates),
        AutomatedSearchStrategy::Random => {
            build_random_candidates(&axes, search.max_candidates, search.random_seed)
        }
    };
    for overrides in enumerated.drain(..) {
        let next = SearchCandidate {
            id: format!("candidate_{:03}", candidates.len()),
            overrides,
        };
        if seen.insert(candidate_signature(&next)) {
            candidates.push(next);
        }
        if candidates.len() >= search.max_candidates {
            break;
        }
    }
    if candidates.is_empty() {
        return Err("automated search failed to generate any candidate settings".into());
    }
    Ok(candidates)
}

fn search_axes(
    base: &ResearchConfig,
    search_space: &AutomatedSearchSpaceConfig,
) -> Vec<Vec<SearchOverride>> {
    let mut axes = Vec::new();
    push_f64_axis(&mut axes, &search_space.gate_temperature, |value| {
        SearchOverride::GateTemperature(value)
    });
    push_f64_axis(&mut axes, &search_space.gate_bias, |value| {
        SearchOverride::GateBias(value)
    });
    push_f64_axis(&mut axes, &search_space.attention_residual_scale, |value| {
        SearchOverride::AttentionResidualScale(value)
    });
    push_f64_axis(&mut axes, &search_space.ffn_residual_scale, |value| {
        SearchOverride::FfnResidualScale(value)
    });
    push_usize_axis(&mut axes, &search_space.rollout_steps, |value| {
        SearchOverride::RolloutSteps(value)
    });
    push_usize_axis(&mut axes, &search_space.min_rollout_steps, |value| {
        SearchOverride::MinRolloutSteps(value)
    });
    push_f64_axis(
        &mut axes,
        &search_space.stop_probability_threshold,
        SearchOverride::StopProbabilityThreshold,
    );
    push_f64_axis(&mut axes, &search_space.coordinate_step_scale, |value| {
        SearchOverride::CoordinateStepScale(value)
    });
    push_f64_axis(
        &mut axes,
        &search_space.rollout_eval_step_weight_decay,
        SearchOverride::RolloutEvalStepWeightDecay,
    );
    push_f64_axis(&mut axes, &search_space.coordinate_momentum, |value| {
        SearchOverride::CoordinateMomentum(value)
    });
    push_f64_axis(&mut axes, &search_space.atom_momentum, |value| {
        SearchOverride::AtomMomentum(value)
    });
    push_f64_axis(&mut axes, &search_space.atom_commit_temperature, |value| {
        SearchOverride::AtomCommitTemperature(value)
    });
    push_f64_axis(
        &mut axes,
        &search_space.max_coordinate_delta_norm,
        SearchOverride::MaxCoordinateDeltaNorm,
    );
    push_f64_axis(&mut axes, &search_space.stop_delta_threshold, |value| {
        SearchOverride::StopDeltaThreshold(value)
    });
    push_usize_axis(&mut axes, &search_space.stop_patience, |value| {
        SearchOverride::StopPatience(value)
    });
    push_f64_axis(&mut axes, &search_space.beta_intra_red, |value| {
        SearchOverride::BetaIntraRed(value)
    });
    push_f64_axis(&mut axes, &search_space.gamma_probe, |value| {
        SearchOverride::GammaProbe(value)
    });
    push_f64_axis(&mut axes, &search_space.delta_leak, |value| {
        SearchOverride::DeltaLeak(value)
    });
    push_f64_axis(&mut axes, &search_space.eta_gate, |value| {
        SearchOverride::EtaGate(value)
    });
    push_f64_axis(&mut axes, &search_space.mu_slot, |value| {
        SearchOverride::MuSlot(value)
    });

    let base_overrides = vec![
        SearchOverride::GateTemperature(base.model.interaction_tuning.gate_temperature),
        SearchOverride::GateBias(base.model.interaction_tuning.gate_bias),
        SearchOverride::AttentionResidualScale(
            base.model.interaction_tuning.attention_residual_scale,
        ),
        SearchOverride::FfnResidualScale(base.model.interaction_tuning.ffn_residual_scale),
        SearchOverride::RolloutSteps(base.data.generation_target.rollout_steps),
        SearchOverride::MinRolloutSteps(base.data.generation_target.min_rollout_steps),
        SearchOverride::StopProbabilityThreshold(
            base.data.generation_target.stop_probability_threshold,
        ),
        SearchOverride::CoordinateStepScale(base.data.generation_target.coordinate_step_scale),
        SearchOverride::RolloutEvalStepWeightDecay(
            base.data.generation_target.rollout_eval_step_weight_decay,
        ),
        SearchOverride::CoordinateMomentum(base.data.generation_target.coordinate_momentum),
        SearchOverride::AtomMomentum(base.data.generation_target.atom_momentum),
        SearchOverride::AtomCommitTemperature(base.data.generation_target.atom_commit_temperature),
        SearchOverride::MaxCoordinateDeltaNorm(
            base.data.generation_target.max_coordinate_delta_norm,
        ),
        SearchOverride::StopDeltaThreshold(base.data.generation_target.stop_delta_threshold),
        SearchOverride::StopPatience(base.data.generation_target.stop_patience),
        SearchOverride::BetaIntraRed(base.training.loss_weights.beta_intra_red),
        SearchOverride::GammaProbe(base.training.loss_weights.gamma_probe),
        SearchOverride::DeltaLeak(base.training.loss_weights.delta_leak),
        SearchOverride::EtaGate(base.training.loss_weights.eta_gate),
        SearchOverride::MuSlot(base.training.loss_weights.mu_slot),
    ];
    axes.retain(|axis| {
        let Some(first) = axis.first() else {
            return false;
        };
        !base_overrides.iter().any(|base_override| {
            base_override.same_axis(first) && axis.len() == 1 && base_override.same_value(first)
        })
    });
    axes
}

fn build_grid_candidates(
    axes: &[Vec<SearchOverride>],
    max_candidates: usize,
) -> Vec<Vec<SearchOverride>> {
    let mut candidates = Vec::new();
    let mut current = Vec::new();
    build_grid_candidates_recursive(axes, 0, &mut current, &mut candidates, max_candidates);
    candidates
}

fn build_grid_candidates_recursive(
    axes: &[Vec<SearchOverride>],
    axis_index: usize,
    current: &mut Vec<SearchOverride>,
    out: &mut Vec<Vec<SearchOverride>>,
    max_candidates: usize,
) {
    if out.len() >= max_candidates {
        return;
    }
    if axis_index == axes.len() {
        out.push(current.clone());
        return;
    }
    for choice in &axes[axis_index] {
        current.push(choice.clone());
        build_grid_candidates_recursive(axes, axis_index + 1, current, out, max_candidates);
        current.pop();
        if out.len() >= max_candidates {
            break;
        }
    }
}

fn build_random_candidates(
    axes: &[Vec<SearchOverride>],
    max_candidates: usize,
    seed: u64,
) -> Vec<Vec<SearchOverride>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut results = Vec::new();
    let sample_count = max_candidates.saturating_mul(4).max(max_candidates);
    for _ in 0..sample_count {
        let candidate = axes
            .iter()
            .filter_map(|axis| axis.choose(&mut rng).cloned())
            .collect::<Vec<_>>();
        results.push(candidate);
        if results.len() >= max_candidates {
            break;
        }
    }
    results
}

fn push_f64_axis(
    axes: &mut Vec<Vec<SearchOverride>>,
    values: &[f64],
    build: impl Fn(f64) -> SearchOverride,
) {
    if values.is_empty() {
        return;
    }
    axes.push(values.iter().copied().map(build).collect());
}

fn push_usize_axis(
    axes: &mut Vec<Vec<SearchOverride>>,
    values: &[usize],
    build: impl Fn(usize) -> SearchOverride,
) {
    if values.is_empty() {
        return;
    }
    axes.push(values.iter().copied().map(build).collect());
}

fn candidate_signature(candidate: &SearchCandidate) -> String {
    candidate
        .overrides
        .iter()
        .map(SearchOverride::label)
        .collect::<Vec<_>>()
        .join("|")
}

fn apply_search_candidate(
    config: &mut UnseenPocketExperimentConfig,
    candidate: &SearchCandidate,
    candidate_dir: &Path,
    source_config: &Path,
) {
    for override_value in &candidate.overrides {
        override_value.apply(config);
    }
    let surface_label = surface_label_from_config(config, source_config);
    config.research.training.checkpoint_dir = candidate_dir.join(surface_label);
}

fn resolve_relative_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn resolve_output_artifact_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if starts_with_repo_artifact_root(path) {
        return path.to_path_buf();
    }
    base_dir.join(path)
}

fn starts_with_repo_artifact_root(path: &Path) -> bool {
    path.components()
        .find_map(|component| match component {
            std::path::Component::CurDir => None,
            std::path::Component::Normal(value) => value.to_str(),
            _ => Some(""),
        })
        .is_some_and(|first| matches!(first, "checkpoints" | "artifacts"))
}

fn evaluate_search_gates(
    surfaces: &[AutomatedSearchSurfaceSummary],
    gates: &AutomatedSearchHardGateConfig,
) -> AutomatedSearchGateResult {
    let mut blocked_reasons = Vec::new();
    for surface in surfaces {
        let label = &surface.surface_label;
        check_min_gate(
            &mut blocked_reasons,
            label,
            "candidate_valid_fraction",
            surface.claim_report.test.candidate_valid_fraction,
            gates.minimum_candidate_valid_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "rdkit_sanitized_fraction",
            backend_metric(
                &surface.claim_report.backend_metrics.chemistry_validity,
                "rdkit_sanitized_fraction",
            )
            .or(surface.claim_report.test.candidate_valid_fraction),
            gates.minimum_sanitized_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "unique_smiles_fraction",
            surface
                .claim_report
                .test
                .unique_smiles_fraction
                .or_else(|| {
                    backend_metric(
                        &surface.claim_report.backend_metrics.chemistry_validity,
                        "rdkit_unique_smiles_fraction",
                    )
                }),
            gates.minimum_unique_smiles_fraction,
        );
        check_max_gate(
            &mut blocked_reasons,
            label,
            "clash_fraction",
            backend_metric(
                &surface.claim_report.backend_metrics.pocket_compatibility,
                "clash_fraction",
            )
            .or_else(|| {
                backend_metric(
                    &surface.claim_report.backend_metrics.docking_affinity,
                    "clash_fraction",
                )
            }),
            gates.maximum_clash_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "strict_pocket_fit_score",
            surface.claim_report.test.strict_pocket_fit_score,
            gates.minimum_strict_pocket_fit_score,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "pocket_contact_fraction",
            surface.claim_report.test.pocket_contact_fraction,
            gates.minimum_pocket_contact_fraction,
        );
        check_min_gate(
            &mut blocked_reasons,
            label,
            "pocket_compatibility_fraction",
            surface.claim_report.test.pocket_compatibility_fraction,
            gates.minimum_pocket_compatibility_fraction,
        );
        let raw = &surface.claim_report.layered_generation_metrics.raw_rollout;
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_centroid_offset",
            Some(raw.mean_centroid_offset),
            gates.maximum_raw_centroid_offset,
        );
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_clash_fraction",
            Some(raw.clash_fraction),
            gates.maximum_raw_clash_fraction,
        );
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_mean_displacement",
            Some(raw.mean_displacement),
            gates.maximum_raw_mean_displacement,
        );
        check_optional_max_gate(
            &mut blocked_reasons,
            label,
            "raw_atom_change_fraction",
            Some(raw.atom_change_fraction),
            gates.maximum_raw_atom_change_fraction,
        );
        check_optional_min_gate(
            &mut blocked_reasons,
            label,
            "raw_uniqueness_proxy_fraction",
            Some(raw.uniqueness_proxy_fraction),
            gates.minimum_raw_uniqueness_proxy_fraction,
        );
    }
    AutomatedSearchGateResult {
        passed: blocked_reasons.is_empty(),
        blocked_reasons,
    }
}

fn check_optional_min_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: Option<f64>,
) {
    if let Some(threshold) = threshold {
        check_min_gate(blocked_reasons, surface_label, metric, value, threshold);
    }
}

fn check_optional_max_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: Option<f64>,
) {
    if let Some(threshold) = threshold {
        check_max_gate(blocked_reasons, surface_label, metric, value, threshold);
    }
}

fn check_min_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: f64,
) {
    match value {
        Some(value) if value + 1e-12 >= threshold => {}
        Some(value) => blocked_reasons.push(format!(
            "{surface_label}:{metric}={value:.4} below minimum {threshold:.4}"
        )),
        None => blocked_reasons.push(format!(
            "{surface_label}:{metric} missing; minimum {threshold:.4} required"
        )),
    }
}

fn check_max_gate(
    blocked_reasons: &mut Vec<String>,
    surface_label: &str,
    metric: &str,
    value: Option<f64>,
    threshold: f64,
) {
    match value {
        Some(value) if value <= threshold + 1e-12 => {}
        Some(value) => blocked_reasons.push(format!(
            "{surface_label}:{metric}={value:.4} above maximum {threshold:.4}"
        )),
        None => blocked_reasons.push(format!(
            "{surface_label}:{metric} missing; maximum {threshold:.4} required"
        )),
    }
}

fn score_search_candidate(
    surfaces: &[AutomatedSearchSurfaceSummary],
    review: Option<&InteractionModeReview>,
    weights: &AutomatedSearchScoreWeightConfig,
) -> f64 {
    let surface_count = surfaces.len().max(1) as f64;
    let mut chemistry = 0.0;
    let mut uniqueness = 0.0;
    let mut geometry = 0.0;
    let mut pocket = 0.0;
    let mut specialization = 0.0;
    let mut utilization = 0.0;

    for surface in surfaces {
        let test = &surface.claim_report.test;
        chemistry += mean_present(&[
            test.candidate_valid_fraction,
            backend_metric(
                &surface.claim_report.backend_metrics.chemistry_validity,
                "rdkit_sanitized_fraction",
            ),
        ]);
        uniqueness += mean_present(&[test.unique_smiles_fraction]);
        geometry += mean_present(&[
            test.strict_pocket_fit_score,
            test.mean_centroid_offset.map(|offset| 1.0 / (1.0 + offset)),
        ]);
        pocket += mean_present(&[
            test.pocket_contact_fraction,
            test.pocket_compatibility_fraction,
        ]);
        specialization += mean_present(&[
            Some(test.topology_specialization_score),
            Some(test.geometry_specialization_score),
            Some(test.pocket_specialization_score),
            Some(1.0 / (1.0 + test.leakage_proxy_mean.max(0.0))),
        ]);
        utilization += mean_present(&[
            Some(1.0 / (1.0 + test.slot_activation_mean.max(0.0))),
            Some(1.0 / (1.0 + test.gate_activation_mean.max(0.0))),
        ]);
    }

    chemistry /= surface_count;
    uniqueness /= surface_count;
    geometry /= surface_count;
    pocket /= surface_count;
    specialization /= surface_count;
    utilization /= surface_count;
    let interaction_review = review.map(interaction_review_score).unwrap_or(0.0);
    weights.chemistry * chemistry
        + weights.uniqueness * uniqueness
        + weights.geometry * geometry
        + weights.pocket * pocket
        + weights.specialization * specialization
        + weights.utilization * utilization
        + weights.interaction_review * interaction_review
}

fn interaction_review_score(review: &InteractionModeReview) -> f64 {
    let tally = &review.aggregate_test_tally;
    let total = tally.lightweight_wins + tally.transformer_wins + tally.ties;
    if total == 0 {
        return 0.0;
    }
    (tally.transformer_wins as f64 - tally.lightweight_wins as f64) / total as f64
}

fn mean_present(values: &[Option<f64>]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().flatten() {
        if value.is_finite() {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn backend_metric(metrics: &ReservedBackendMetrics, key: &str) -> Option<f64> {
    metrics
        .metrics
        .get(key)
        .copied()
        .filter(|value| value.is_finite())
}

fn search_candidate_rank_cmp(
    left: &AutomatedSearchCandidateSummary,
    right: &AutomatedSearchCandidateSummary,
) -> std::cmp::Ordering {
    match (left.gate_result.passed, right.gate_result.passed) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id)),
    }
}

fn summarize_search_roadmap(candidates: &[AutomatedSearchCandidateSummary]) -> String {
    let survived = candidates
        .iter()
        .filter(|candidate| candidate.gate_result.passed)
        .count();
    if survived > 0 {
        return format!(
            "{survived} candidate(s) survived chemistry, uniqueness, clash, and strict pocket-fit gates; keep automated search plus lightweight reranking as the mainline before considering adversarial training."
        );
    }
    let blocked = candidates.len();
    format!(
        "all {blocked} candidate(s) were blocked by hard regression gates; inspect blocked_reasons and expand bounded search or add a reranker before attempting higher-risk adversarial training."
    )
}

fn surface_label_from_config(
    config: &UnseenPocketExperimentConfig,
    source_config: &Path,
) -> String {
    config
        .surface_label
        .clone()
        .or_else(|| config.ablation.variant_label.clone())
        .unwrap_or_else(|| {
            source_config
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("surface")
                .trim_start_matches("unseen_pocket_")
                .to_string()
        })
}

impl SearchOverride {
    fn apply(&self, config: &mut UnseenPocketExperimentConfig) {
        match self {
            SearchOverride::GateTemperature(value) => {
                config.research.model.interaction_tuning.gate_temperature = *value;
            }
            SearchOverride::GateBias(value) => {
                config.research.model.interaction_tuning.gate_bias = *value;
            }
            SearchOverride::AttentionResidualScale(value) => {
                config
                    .research
                    .model
                    .interaction_tuning
                    .attention_residual_scale = *value;
            }
            SearchOverride::FfnResidualScale(value) => {
                config.research.model.interaction_tuning.ffn_residual_scale = *value;
            }
            SearchOverride::RolloutSteps(value) => {
                config.research.data.generation_target.rollout_steps = *value;
            }
            SearchOverride::MinRolloutSteps(value) => {
                config.research.data.generation_target.min_rollout_steps = *value;
            }
            SearchOverride::StopProbabilityThreshold(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .stop_probability_threshold = *value;
            }
            SearchOverride::CoordinateStepScale(value) => {
                config.research.data.generation_target.coordinate_step_scale = *value;
            }
            SearchOverride::RolloutEvalStepWeightDecay(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .rollout_eval_step_weight_decay = *value;
            }
            SearchOverride::CoordinateMomentum(value) => {
                config.research.data.generation_target.coordinate_momentum = *value;
            }
            SearchOverride::AtomMomentum(value) => {
                config.research.data.generation_target.atom_momentum = *value;
            }
            SearchOverride::AtomCommitTemperature(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .atom_commit_temperature = *value;
            }
            SearchOverride::MaxCoordinateDeltaNorm(value) => {
                config
                    .research
                    .data
                    .generation_target
                    .max_coordinate_delta_norm = *value;
            }
            SearchOverride::StopDeltaThreshold(value) => {
                config.research.data.generation_target.stop_delta_threshold = *value;
            }
            SearchOverride::StopPatience(value) => {
                config.research.data.generation_target.stop_patience = *value;
            }
            SearchOverride::BetaIntraRed(value) => {
                config.research.training.loss_weights.beta_intra_red = *value;
            }
            SearchOverride::GammaProbe(value) => {
                config.research.training.loss_weights.gamma_probe = *value;
            }
            SearchOverride::DeltaLeak(value) => {
                config.research.training.loss_weights.delta_leak = *value;
            }
            SearchOverride::EtaGate(value) => {
                config.research.training.loss_weights.eta_gate = *value;
            }
            SearchOverride::MuSlot(value) => {
                config.research.training.loss_weights.mu_slot = *value;
            }
        }
    }

    fn label(&self) -> String {
        match self {
            SearchOverride::GateTemperature(value) => format!("gate_temperature={value:.6}"),
            SearchOverride::GateBias(value) => format!("gate_bias={value:.6}"),
            SearchOverride::AttentionResidualScale(value) => {
                format!("attention_residual_scale={value:.6}")
            }
            SearchOverride::FfnResidualScale(value) => format!("ffn_residual_scale={value:.6}"),
            SearchOverride::RolloutSteps(value) => format!("rollout_steps={value}"),
            SearchOverride::MinRolloutSteps(value) => format!("min_rollout_steps={value}"),
            SearchOverride::StopProbabilityThreshold(value) => {
                format!("stop_probability_threshold={value:.6}")
            }
            SearchOverride::CoordinateStepScale(value) => {
                format!("coordinate_step_scale={value:.6}")
            }
            SearchOverride::RolloutEvalStepWeightDecay(value) => {
                format!("rollout_eval_step_weight_decay={value:.6}")
            }
            SearchOverride::CoordinateMomentum(value) => format!("coordinate_momentum={value:.6}"),
            SearchOverride::AtomMomentum(value) => format!("atom_momentum={value:.6}"),
            SearchOverride::AtomCommitTemperature(value) => {
                format!("atom_commit_temperature={value:.6}")
            }
            SearchOverride::MaxCoordinateDeltaNorm(value) => {
                format!("max_coordinate_delta_norm={value:.6}")
            }
            SearchOverride::StopDeltaThreshold(value) => {
                format!("stop_delta_threshold={value:.6}")
            }
            SearchOverride::StopPatience(value) => format!("stop_patience={value}"),
            SearchOverride::BetaIntraRed(value) => format!("beta_intra_red={value:.6}"),
            SearchOverride::GammaProbe(value) => format!("gamma_probe={value:.6}"),
            SearchOverride::DeltaLeak(value) => format!("delta_leak={value:.6}"),
            SearchOverride::EtaGate(value) => format!("eta_gate={value:.6}"),
            SearchOverride::MuSlot(value) => format!("mu_slot={value:.6}"),
        }
    }

    fn same_axis(&self, other: &SearchOverride) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    fn same_value(&self, other: &SearchOverride) -> bool {
        self.label() == other.label()
    }
}
