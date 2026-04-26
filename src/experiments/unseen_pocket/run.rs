pub struct UnseenPocketExperiment;

impl UnseenPocketExperiment {
    /// Execute the experiment using the configured dataset loader.
    pub fn run(
        config: UnseenPocketExperimentConfig,
    ) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
        Self::run_with_options(config, false)
    }

    /// Execute the experiment with optional checkpoint resume.
    pub fn run_with_options(
        mut config: UnseenPocketExperimentConfig,
        resume_from_latest: bool,
    ) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
        config.validate()?;
        if let Some(mode) = config.ablation.interaction_mode_override {
            config.research.model.interaction_mode = mode;
        }
        if config.ablation.disable_geometry_interaction_bias {
            config
                .research
                .model
                .interaction_tuning
                .geometry_attention_bias_scale = 0.0;
        }
        if config.ablation.disable_rollout_pocket_guidance {
            config.research.data.generation_target.pocket_guidance_scale = 0.0;
        }
        let loaded = InMemoryDataset::load_from_config(&config.research.data)?;
        let dataset = loaded
            .dataset
            .with_pocket_feature_dim(config.research.model.pocket_feature_dim);
        let splits = dataset.split_by_protein_fraction_with_options(
            config.research.data.val_fraction,
            config.research.data.test_fraction,
            config.research.data.split_seed,
            config.research.data.stratify_by_measurement,
        );

        if splits.train.examples().is_empty() {
            return Err(
                "training split is empty; adjust val/test fractions or dataset size".into(),
            );
        }

        let device = parse_runtime_device(&config.research.runtime.device)?;
        let mut var_store = nn::VarStore::new(device);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config.research);
        let mut trainer = ResearchTrainer::new(&var_store, config.research.clone())?;
        trainer.set_dataset_validation_fingerprint(stable_json_hash(&loaded.validation));
        let mut resumed_checkpoint_metadata = None;
        if resume_from_latest {
            if let Some(checkpoint) = trainer.resume_from_latest(&mut var_store)? {
                resumed_checkpoint_metadata = Some(checkpoint.metadata.clone());
                if let Some(history) = load_experiment_history(
                    &config.research.training.checkpoint_dir,
                    checkpoint.metadata.step,
                )? {
                    trainer.replace_history(history);
                }
                log::info!(
                    "resumed experiment training from step {} at {}",
                    checkpoint.metadata.step,
                    checkpoint.weights_path.display()
                );
            }
        }
        let train_examples: Vec<_> = splits
            .train
            .examples()
            .iter()
            .map(|example| example.to_device(device))
            .collect();
        trainer.fit(&var_store, &system, &train_examples)?;

        let train_proteins: std::collections::BTreeSet<&str> = splits
            .train
            .examples()
            .iter()
            .map(|example| example.protein_id.as_str())
            .collect();

        let validation = evaluate_split(
            &system,
            splits.val.examples(),
            splits.train.examples(),
            &train_proteins,
            &config.research,
            config.ablation.clone(),
            &config.external_evaluation,
            "validation",
            device,
        );
        let test = evaluate_split(
            &system,
            splits.test.examples(),
            splits.train.examples(),
            &train_proteins,
            &config.research,
            config.ablation.clone(),
            &config.external_evaluation,
            "test",
            device,
        );

        let reproducibility = reproducibility_metadata(
            &config.research,
            &loaded.validation,
            resumed_checkpoint_metadata.as_ref(),
        );
        let dataset_validation = loaded.validation;
        let mut summary = UnseenPocketExperimentSummary {
            config,
            dataset_validation,
            split_report: SplitReport::from_datasets(&splits.train, &splits.val, &splits.test),
            reproducibility,
            training_history: trainer.history().to_vec(),
            validation,
            test,
            ablation_matrix: None,
            performance_gates: PerformanceGateReport::default(),
        };
        summary.performance_gates = build_performance_gate_report(
            &summary.config.performance_gates,
            &summary.validation.resource_usage,
            &summary.test.resource_usage,
        );
        if summary.config.ablation_matrix.enabled {
            let matrix = run_ablation_matrix(&summary.config)?;
            persist_ablation_matrix(&summary.config.research.training.checkpoint_dir, &matrix)?;
            summary.ablation_matrix = Some(matrix);
        }
        persist_experiment_summary(&summary)?;
        Ok(summary)
    }
}

fn persist_experiment_summary(
    summary: &UnseenPocketExperimentSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(&summary.config.research.training.checkpoint_dir)?;
    let artifact_dir = &summary.config.research.training.checkpoint_dir;
    let config_snapshot = artifact_dir.join("config.snapshot.json");
    let validation_path = artifact_dir.join("dataset_validation_report.json");
    let validation_alias_path = artifact_dir.join("dataset_validation.json");
    let split_report_path = artifact_dir.join("split_report.json");
    let summary_path = artifact_dir.join("experiment_summary.json");
    let claim_summary_path = artifact_dir.join("claim_summary.json");
    let bundle_path = artifact_dir.join("run_artifacts.json");
    fs::write(
        &config_snapshot,
        serde_json::to_string_pretty(&summary.config)?,
    )?;
    let validation_json = serde_json::to_string_pretty(&summary.dataset_validation)?;
    fs::write(&validation_path, &validation_json)?;
    fs::write(&validation_alias_path, validation_json)?;
    fs::write(
        &split_report_path,
        serde_json::to_string_pretty(&summary.split_report)?,
    )?;
    fs::write(&summary_path, serde_json::to_string_pretty(summary)?)?;
    fs::write(
        &claim_summary_path,
        serde_json::to_string_pretty(&build_claim_report(summary))?,
    )?;
    let bundle = RunArtifactBundle {
        schema_version: summary.reproducibility.artifact_bundle_schema_version,
        run_kind: RunKind::Experiment,
        artifact_dir: artifact_dir.clone(),
        config_hash: summary.reproducibility.config_hash.clone(),
        dataset_validation_fingerprint: summary
            .reproducibility
            .dataset_validation_fingerprint
            .clone(),
        metric_schema_version: summary.reproducibility.metric_schema_version,
        backend_environment: Some(build_backend_environment_report(summary)),
        paths: RunArtifactPaths {
            config_snapshot,
            dataset_validation_report: validation_path,
            split_report: split_report_path,
            run_summary: summary_path,
            run_bundle: bundle_path.clone(),
            latest_checkpoint: Some(artifact_dir.join("latest.ot")),
        },
    };
    fs::write(bundle_path, serde_json::to_string_pretty(&bundle)?)?;
    persist_interaction_reviews(summary)?;
    Ok(())
}

fn build_performance_gate_report(
    config: &PerformanceGateConfig,
    validation: &ResourceUsageMetrics,
    test: &ResourceUsageMetrics,
) -> PerformanceGateReport {
    let mut failed_reasons = Vec::new();
    if let Some(minimum) = config.min_validation_examples_per_second {
        if validation.examples_per_second < minimum {
            failed_reasons.push(format!(
                "validation examples/sec {:.4} below minimum {:.4}",
                validation.examples_per_second, minimum
            ));
        }
    }
    if let Some(minimum) = config.min_test_examples_per_second {
        if test.examples_per_second < minimum {
            failed_reasons.push(format!(
                "test examples/sec {:.4} below minimum {:.4}",
                test.examples_per_second, minimum
            ));
        }
    }
    if let Some(maximum) = config.max_validation_memory_mb {
        if validation.memory_usage_mb > maximum {
            failed_reasons.push(format!(
                "validation memory delta {:.4} MB above maximum {:.4}",
                validation.memory_usage_mb, maximum
            ));
        }
    }
    if let Some(maximum) = config.max_test_memory_mb {
        if test.memory_usage_mb > maximum {
            failed_reasons.push(format!(
                "test memory delta {:.4} MB above maximum {:.4}",
                test.memory_usage_mb, maximum
            ));
        }
    }
    PerformanceGateReport {
        passed: failed_reasons.is_empty(),
        failed_reasons,
        validation_examples_per_second: validation.examples_per_second,
        test_examples_per_second: test.examples_per_second,
        validation_memory_mb: validation.memory_usage_mb,
        test_memory_mb: test.memory_usage_mb,
    }
}

fn persist_interaction_reviews(
    summary: &UnseenPocketExperimentSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(matrix) = summary.ablation_matrix.as_ref() else {
        return Ok(());
    };
    let Some(surface_review) = build_surface_interaction_review(summary, matrix) else {
        return Ok(());
    };
    let artifact_dir = &summary.config.research.training.checkpoint_dir;
    fs::write(
        artifact_dir.join("interaction_mode_review.json"),
        serde_json::to_string_pretty(&surface_review)?,
    )?;
    if let Some(shared_review) = build_shared_interaction_review(artifact_dir.parent())? {
        fs::write(
            shared_review
                .review_root
                .join("interaction_mode_review.json"),
            serde_json::to_string_pretty(&shared_review)?,
        )?;
    }
    Ok(())
}

fn build_surface_interaction_review(
    summary: &UnseenPocketExperimentSummary,
    matrix: &AblationMatrixSummary,
) -> Option<InteractionSurfaceReview> {
    let lightweight = find_interaction_summary(summary, matrix, CrossAttentionMode::Lightweight)?;
    let transformer = find_interaction_summary(summary, matrix, CrossAttentionMode::Transformer)?;
    Some(InteractionSurfaceReview {
        surface_label: surface_label_from_dir(&summary.config.research.training.checkpoint_dir),
        artifact_dir: summary.config.research.training.checkpoint_dir.clone(),
        validation: build_split_review(&lightweight.validation, &transformer.validation),
        test: build_split_review(&lightweight.test, &transformer.test),
    })
}

fn build_shared_interaction_review(
    root_dir: Option<&std::path::Path>,
) -> Result<Option<InteractionModeReview>, Box<dyn std::error::Error>> {
    let Some(root_dir) = root_dir else {
        return Ok(None);
    };
    let mut surfaces = Vec::new();
    for surface_name in ["claim_matrix", "harder_pressure", "tight_geometry_pressure"] {
        let artifact_dir = root_dir.join(surface_name);
        let claim_path = artifact_dir.join("claim_summary.json");
        let matrix_path = artifact_dir.join("ablation_matrix_summary.json");
        if !claim_path.exists() || !matrix_path.exists() {
            continue;
        }
        let claim: ClaimReport = serde_json::from_str(&fs::read_to_string(claim_path)?)?;
        let matrix: AblationMatrixSummary =
            serde_json::from_str(&fs::read_to_string(matrix_path)?)?;
        if let Some(surface_review) =
            build_surface_interaction_review_from_artifacts(artifact_dir.clone(), claim, matrix)
        {
            surfaces.push(surface_review);
        }
    }
    if surfaces.is_empty() {
        return Ok(None);
    }

    let mut aggregate_test_tally = InteractionWinLossTally::default();
    for surface in &surfaces {
        accumulate_tally(&mut aggregate_test_tally, &surface.test.tally);
    }
    let recommendation = summarize_interaction_recommendation(&surfaces, &aggregate_test_tally);
    Ok(Some(InteractionModeReview {
        review_root: root_dir.to_path_buf(),
        surfaces,
        aggregate_test_tally,
        recommendation,
    }))
}

fn build_surface_interaction_review_from_artifacts(
    artifact_dir: std::path::PathBuf,
    claim: ClaimReport,
    matrix: AblationMatrixSummary,
) -> Option<InteractionSurfaceReview> {
    let lightweight =
        find_interaction_summary_from_artifacts(&claim, &matrix, CrossAttentionMode::Lightweight)?;
    let transformer =
        find_interaction_summary_from_artifacts(&claim, &matrix, CrossAttentionMode::Transformer)?;
    Some(InteractionSurfaceReview {
        surface_label: surface_label_from_dir(&artifact_dir),
        artifact_dir,
        validation: build_split_review(&lightweight.validation, &transformer.validation),
        test: build_split_review(&lightweight.test, &transformer.test),
    })
}

fn build_split_review(
    lightweight: &GenerationQualitySummary,
    transformer: &GenerationQualitySummary,
) -> InteractionSplitReview {
    let geometric_fit = vec![
        metric_verdict(
            "candidate_valid_fraction",
            lightweight.candidate_valid_fraction,
            transformer.candidate_valid_fraction,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "pocket_contact_fraction",
            lightweight.pocket_contact_fraction,
            transformer.pocket_contact_fraction,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "pocket_compatibility_fraction",
            lightweight.pocket_compatibility_fraction,
            transformer.pocket_compatibility_fraction,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "mean_centroid_offset",
            lightweight.mean_centroid_offset,
            transformer.mean_centroid_offset,
            MetricDirection::LowerIsBetter,
        ),
        metric_verdict(
            "strict_pocket_fit_score",
            lightweight.strict_pocket_fit_score,
            transformer.strict_pocket_fit_score,
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "unique_smiles_fraction",
            lightweight.unique_smiles_fraction,
            transformer.unique_smiles_fraction,
            MetricDirection::HigherIsBetter,
        ),
    ];
    let specialization = vec![
        metric_verdict(
            "topology_specialization_score",
            Some(lightweight.topology_specialization_score),
            Some(transformer.topology_specialization_score),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "geometry_specialization_score",
            Some(lightweight.geometry_specialization_score),
            Some(transformer.geometry_specialization_score),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "pocket_specialization_score",
            Some(lightweight.pocket_specialization_score),
            Some(transformer.pocket_specialization_score),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "leakage_proxy_mean",
            Some(lightweight.leakage_proxy_mean),
            Some(transformer.leakage_proxy_mean),
            MetricDirection::LowerIsBetter,
        ),
    ];
    let utilization = vec![
        metric_verdict(
            "slot_activation_mean",
            Some(lightweight.slot_activation_mean),
            Some(transformer.slot_activation_mean),
            MetricDirection::HigherIsBetter,
        ),
        metric_verdict(
            "gate_activation_mean",
            Some(lightweight.gate_activation_mean),
            Some(transformer.gate_activation_mean),
            MetricDirection::HigherIsBetter,
        ),
    ];
    let mut tally = InteractionWinLossTally::default();
    for verdict in geometric_fit
        .iter()
        .chain(specialization.iter())
        .chain(utilization.iter())
    {
        update_tally(&mut tally, verdict.winner);
    }
    InteractionSplitReview {
        lightweight: lightweight.clone(),
        transformer: transformer.clone(),
        tally,
        geometric_fit,
        specialization,
        utilization,
    }
}

fn metric_verdict(
    metric: &str,
    lightweight: Option<f64>,
    transformer: Option<f64>,
    direction: MetricDirection,
) -> InteractionMetricVerdict {
    const EPSILON: f64 = 1e-6;
    let (winner, preferred_delta) = match (lightweight, transformer) {
        (Some(lightweight), Some(transformer)) => {
            let signed_delta = match direction {
                MetricDirection::HigherIsBetter => lightweight - transformer,
                MetricDirection::LowerIsBetter => transformer - lightweight,
            };
            if signed_delta.abs() <= EPSILON {
                (MetricWinner::Tie, 0.0)
            } else if signed_delta > 0.0 {
                (MetricWinner::Lightweight, signed_delta)
            } else {
                (MetricWinner::Transformer, -signed_delta)
            }
        }
        _ => (MetricWinner::Tie, 0.0),
    };
    InteractionMetricVerdict {
        metric: metric.to_string(),
        direction,
        lightweight,
        transformer,
        winner,
        preferred_delta,
    }
}

fn update_tally(tally: &mut InteractionWinLossTally, winner: MetricWinner) {
    match winner {
        MetricWinner::Lightweight => tally.lightweight_wins += 1,
        MetricWinner::Transformer => tally.transformer_wins += 1,
        MetricWinner::Tie => tally.ties += 1,
    }
}

fn accumulate_tally(target: &mut InteractionWinLossTally, source: &InteractionWinLossTally) {
    target.lightweight_wins += source.lightweight_wins;
    target.transformer_wins += source.transformer_wins;
    target.ties += source.ties;
}

fn summarize_interaction_recommendation(
    surfaces: &[InteractionSurfaceReview],
    aggregate_test_tally: &InteractionWinLossTally,
) -> String {
    let lightweight_surface_wins = surfaces
        .iter()
        .filter(|surface| surface.test.tally.lightweight_wins > surface.test.tally.transformer_wins)
        .count();
    let transformer_surface_wins = surfaces
        .iter()
        .filter(|surface| surface.test.tally.transformer_wins > surface.test.tally.lightweight_wins)
        .count();
    if lightweight_surface_wins >= 2 && transformer_surface_wins == 0 {
        "keep interaction tuning active; lightweight still wins most reviewed test surfaces, so stronger claim-oriented interpretation work is premature".to_string()
    } else if transformer_surface_wins >= 2 && lightweight_surface_wins == 0 {
        "Transformer interaction now has a consistent reviewed advantage across test surfaces; claim-oriented interpretation work can start cautiously while keeping regression gates live".to_string()
    } else if aggregate_test_tally.transformer_wins > aggregate_test_tally.lightweight_wins {
        "interaction tuning is still active, but the tuned Transformer path now has the broader reviewed metric lead; require one more clean surface refresh before escalating claim scope".to_string()
    } else {
        "interaction tuning should remain the roadmap focus; reviewed surfaces still show a bounded tradeoff instead of a clean interaction-mode winner".to_string()
    }
}

fn surface_label_from_dir(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown_surface")
        .to_string()
}

fn find_interaction_summary<'a>(
    summary: &UnseenPocketExperimentSummary,
    matrix: &AblationMatrixSummary,
    mode: CrossAttentionMode,
) -> Option<AblationRunSummary> {
    match mode {
        CrossAttentionMode::Lightweight => matrix
            .variants
            .iter()
            .find(|variant| variant.test.interaction_mode == "lightweight")
            .cloned(),
        CrossAttentionMode::Transformer => {
            if summary.test.comparison_summary.interaction_mode == "transformer" {
                Some(AblationRunSummary {
                    variant_label: summary
                        .config
                        .ablation
                        .variant_label
                        .clone()
                        .unwrap_or_else(|| "base_run".to_string()),
                    validation: summary.validation.comparison_summary.clone(),
                    test: summary.test.comparison_summary.clone(),
                })
            } else {
                matrix
                    .variants
                    .iter()
                    .find(|variant| variant.test.interaction_mode == "transformer")
                    .cloned()
            }
        }
    }
}

fn find_interaction_summary_from_artifacts(
    claim: &ClaimReport,
    matrix: &AblationMatrixSummary,
    mode: CrossAttentionMode,
) -> Option<AblationRunSummary> {
    match mode {
        CrossAttentionMode::Lightweight => matrix
            .variants
            .iter()
            .find(|variant| variant.test.interaction_mode == "lightweight")
            .cloned(),
        CrossAttentionMode::Transformer => {
            if claim.test.interaction_mode == "transformer" {
                Some(AblationRunSummary {
                    variant_label: claim.run_label.clone(),
                    validation: claim.validation.clone(),
                    test: claim.test.clone(),
                })
            } else {
                matrix
                    .variants
                    .iter()
                    .find(|variant| variant.test.interaction_mode == "transformer")
                    .cloned()
            }
        }
    }
}

fn load_experiment_history(
    checkpoint_dir: &std::path::Path,
    resumed_step: usize,
) -> Result<Option<Vec<StepMetrics>>, Box<dyn std::error::Error>> {
    let path = checkpoint_dir.join("experiment_summary.json");
    if !path.exists() {
        return Ok(None);
    }

    let summary: UnseenPocketExperimentSummary = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(Some(
        summary
            .training_history
            .into_iter()
            .filter(|metrics| metrics.step <= resumed_step)
            .collect(),
    ))
}

