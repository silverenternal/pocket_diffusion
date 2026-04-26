#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn unseen_pocket_experiment_smoke_test() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = UnseenPocketExperimentConfig::default();
        config.research.training.max_steps = 2;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 1;
        config.research.training.schedule.stage3_steps = 2;
        config.research.training.checkpoint_every = 100;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = temp.path().join("checkpoints");

        let summary = UnseenPocketExperiment::run(config).unwrap();
        assert_eq!(summary.training_history.len(), 2);
        assert!(
            summary
                .validation
                .representation_diagnostics
                .finite_forward_fraction
                >= 0.0
        );
        assert!(
            summary
                .test
                .representation_diagnostics
                .finite_forward_fraction
                >= 0.0
        );
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("experiment_summary.json")
            .exists());
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("dataset_validation_report.json")
            .exists());
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("claim_summary.json")
            .exists());
        assert_eq!(summary.dataset_validation.parsed_examples, 4);
    }

    #[test]
    fn resumed_experiment_preserves_prior_history() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint_dir = temp.path().join("checkpoints");

        let mut config = UnseenPocketExperimentConfig::default();
        config.research.training.max_steps = 2;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 1;
        config.research.training.schedule.stage3_steps = 2;
        config.research.training.checkpoint_every = 1;
        config.research.training.log_every = 100;
        config.research.data.batch_size = 2;
        config.research.runtime.device = "cpu".to_string();
        config.research.training.checkpoint_dir = checkpoint_dir;

        let _ = UnseenPocketExperiment::run(config.clone()).unwrap();

        config.research.training.max_steps = 4;
        config.research.training.schedule.stage1_steps = 1;
        config.research.training.schedule.stage2_steps = 2;
        config.research.training.schedule.stage3_steps = 3;
        let summary = UnseenPocketExperiment::run_with_options(config, true).unwrap();

        assert_eq!(summary.training_history.len(), 4);
        assert_eq!(summary.training_history[0].step, 0);
        assert_eq!(summary.training_history[1].step, 1);
        assert_eq!(summary.training_history[2].step, 2);
        assert_eq!(summary.training_history[3].step, 3);
        assert!(!summary.split_report.leakage_checks.protein_overlap_detected);
    }

    #[test]
    fn comparison_summary_surfaces_strict_generation_metrics() {
        let mut docking = ReservedBackendMetrics {
            available: true,
            backend_name: Some("heuristic_docking_hook_v1".to_string()),
            metrics: BTreeMap::new(),
            status: "test".to_string(),
        };
        docking
            .metrics
            .insert("mean_centroid_offset".to_string(), 1.75);
        docking
            .metrics
            .insert("centroid_fit_score".to_string(), 0.36);
        let mut pocket = ReservedBackendMetrics {
            available: true,
            backend_name: Some("heuristic_pocket_compatibility_v1".to_string()),
            metrics: BTreeMap::new(),
            status: "test".to_string(),
        };
        pocket
            .metrics
            .insert("strict_pocket_fit_score".to_string(), 0.31);
        let metrics = RealGenerationMetrics {
            chemistry_validity: ReservedBackendMetrics {
                available: true,
                backend_name: Some("heuristic_validity_v1".to_string()),
                metrics: BTreeMap::from([("valid_fraction".to_string(), 1.0)]),
                status: "test".to_string(),
            },
            docking_affinity: docking,
            pocket_compatibility: pocket,
        };

        let summary = build_comparison_summary(
            &ResearchConfig::default(),
            &AblationConfig::default(),
            1.0,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            0.05,
            &metrics,
        );

        assert_eq!(summary.mean_centroid_offset, Some(1.75));
        assert_eq!(summary.strict_pocket_fit_score, Some(0.31));
        assert_eq!(summary.unique_smiles_fraction, None);
    }

    #[test]
    fn comparison_summary_surfaces_uniqueness_from_rdkit_metrics() {
        let metrics = RealGenerationMetrics {
            chemistry_validity: ReservedBackendMetrics {
                available: true,
                backend_name: Some("external_command_chemistry".to_string()),
                metrics: BTreeMap::from([
                    ("valid_fraction".to_string(), 1.0),
                    ("rdkit_unique_smiles_fraction".to_string(), 0.67),
                ]),
                status: "test".to_string(),
            },
            docking_affinity: ReservedBackendMetrics {
                available: false,
                backend_name: None,
                metrics: BTreeMap::new(),
                status: "test".to_string(),
            },
            pocket_compatibility: ReservedBackendMetrics {
                available: false,
                backend_name: None,
                metrics: BTreeMap::new(),
                status: "test".to_string(),
            },
        };

        let summary = build_comparison_summary(
            &ResearchConfig::default(),
            &AblationConfig::default(),
            1.0,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            0.05,
            &metrics,
        );

        assert_eq!(summary.unique_smiles_fraction, Some(0.67));
    }

    #[test]
    fn automated_search_generates_bounded_candidates() {
        let mut search = AutomatedSearchConfig {
            enabled: true,
            max_candidates: 3,
            include_base_candidate: true,
            ..AutomatedSearchConfig::default()
        };
        search.search_space.gate_temperature = vec![1.0, 1.2];
        search.search_space.coordinate_step_scale = vec![0.65, 0.8];

        let candidates = build_search_candidates(&ResearchConfig::default(), &search).unwrap();

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].id, "candidate_000_base");
        assert!(candidates
            .iter()
            .any(|candidate| !candidate.overrides.is_empty()));
    }

    #[test]
    fn automated_search_hard_gates_block_cross_surface_regression() {
        let surface = AutomatedSearchSurfaceSummary {
            surface_label: "tight_geometry_pressure".to_string(),
            source_config: "configs/unseen_pocket_tight_geometry_pressure.json".into(),
            artifact_dir: "checkpoints/tight_geometry_pressure".into(),
            claim_report: ClaimReport {
                artifact_dir: "checkpoints/tight_geometry_pressure".into(),
                run_label: "test".to_string(),
                validation: test_generation_quality_summary(),
                test: GenerationQualitySummary {
                    strict_pocket_fit_score: Some(0.35),
                    unique_smiles_fraction: Some(0.33),
                    ..test_generation_quality_summary()
                },
                backend_metrics: RealGenerationMetrics {
                    chemistry_validity: ReservedBackendMetrics {
                        available: true,
                        backend_name: Some("rdkit".to_string()),
                        metrics: BTreeMap::from([
                            ("valid_fraction".to_string(), 1.0),
                            ("rdkit_sanitized_fraction".to_string(), 1.0),
                        ]),
                        status: "test".to_string(),
                    },
                    docking_affinity: ReservedBackendMetrics {
                        available: true,
                        backend_name: Some("pocket".to_string()),
                        metrics: BTreeMap::new(),
                        status: "test".to_string(),
                    },
                    pocket_compatibility: ReservedBackendMetrics {
                        available: true,
                        backend_name: Some("pocket".to_string()),
                        metrics: BTreeMap::from([("clash_fraction".to_string(), 0.0)]),
                        status: "test".to_string(),
                    },
                },
                backend_thresholds: BTreeMap::new(),
                backend_review: BackendReviewReport::default(),
                layered_generation_metrics: empty_layered_generation_metrics(),
                chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary::default(),
                claim_context: ClaimContext::default(),
                backend_environment: None,
                ablation_deltas: Vec::new(),
                reranker_report: RerankerReport::default(),
                slot_stability: SlotStabilityMetrics::default(),
                leakage_calibration: LeakageCalibrationReport::default(),
                performance_gates: PerformanceGateReport::default(),
                baseline_comparisons: Vec::new(),
                method_comparison: MethodComparisonSummary::default(),
            },
        };

        let gate_result =
            evaluate_search_gates(&[surface], &AutomatedSearchHardGateConfig::default());

        assert!(!gate_result.passed);
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("unique_smiles_fraction")));
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("strict_pocket_fit_score")));
    }

    #[test]
    fn automated_search_can_block_raw_model_regressions() {
        let mut surface = AutomatedSearchSurfaceSummary {
            surface_label: "raw_gate_surface".to_string(),
            source_config: "configs/raw_gate.json".into(),
            artifact_dir: "checkpoints/raw_gate".into(),
            claim_report: test_claim_report(),
        };
        surface.claim_report.layered_generation_metrics.raw_rollout = CandidateLayerMetrics {
            candidate_count: 3,
            valid_fraction: 1.0,
            pocket_contact_fraction: 1.0,
            mean_centroid_offset: 9.5,
            clash_fraction: 0.4,
            mean_displacement: 0.8,
            atom_change_fraction: 0.5,
            uniqueness_proxy_fraction: 0.2,
            atom_type_sequence_diversity: 0.2,
            bond_topology_diversity: 0.2,
            coordinate_shape_diversity: 0.2,
            novel_atom_type_sequence_fraction: 0.2,
            novel_bond_topology_fraction: 0.2,
            novel_coordinate_shape_fraction: 0.2,
        };
        let gates = AutomatedSearchHardGateConfig {
            maximum_raw_centroid_offset: Some(3.0),
            maximum_raw_clash_fraction: Some(0.1),
            maximum_raw_mean_displacement: Some(0.5),
            maximum_raw_atom_change_fraction: Some(0.25),
            minimum_raw_uniqueness_proxy_fraction: Some(0.5),
            ..AutomatedSearchHardGateConfig::default()
        };

        let gate_result = evaluate_search_gates(&[surface], &gates);

        assert!(!gate_result.passed);
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("raw_centroid_offset")));
        assert!(gate_result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("raw_uniqueness_proxy_fraction")));
    }

    #[test]
    fn chemistry_benchmark_evidence_uses_configured_external_dataset() {
        let mut summary: UnseenPocketExperimentSummary = serde_json::from_str(
            &std::fs::read_to_string("checkpoints/pdbbindpp_real_backends/experiment_summary.json")
                .expect("existing experiment summary should be readable for benchmark-evidence tests"),
        )
        .expect("existing experiment summary should deserialize");
        summary.config.surface_label = Some("lp_pdbbind_refined_real_backends".to_string());
        summary.config.reviewer_benchmark.dataset = Some("lp_pdbbind_refined".to_string());
        summary.config.research.training.checkpoint_dir =
            PathBuf::from("./checkpoints/lp_pdbbind_refined_real_backends");
        summary.dataset_validation.parsed_examples = 5048;
        summary.dataset_validation.retained_label_coverage = 1.0;
        summary.split_report.val.protein_family_proxy_histogram = BTreeMap::from_iter(
            (0..12).map(|ix| (format!("val_family_{ix}"), 1usize)),
        );
        summary.split_report.test.protein_family_proxy_histogram = BTreeMap::from_iter(
            (0..12).map(|ix| (format!("test_family_{ix}"), 1usize)),
        );
        summary.test.comparison_summary.candidate_valid_fraction = Some(1.0);
        summary.test.comparison_summary.unique_smiles_fraction = Some(1.0);
        summary.test.comparison_summary.strict_pocket_fit_score = Some(0.6);
        summary.test.comparison_summary.leakage_proxy_mean = 0.05;
        summary.test.representation_diagnostics.leakage_proxy_mean = 0.05;
        summary.test.layered_generation_metrics.reranked_candidates = CandidateLayerMetrics {
            candidate_count: 8,
            valid_fraction: 1.0,
            pocket_contact_fraction: 1.0,
            mean_centroid_offset: 0.4,
            clash_fraction: 0.0,
            mean_displacement: 0.1,
            atom_change_fraction: 0.1,
            uniqueness_proxy_fraction: 1.0,
            atom_type_sequence_diversity: 1.0,
            bond_topology_diversity: 1.0,
            coordinate_shape_diversity: 1.0,
            novel_atom_type_sequence_fraction: 1.0,
            novel_bond_topology_fraction: 1.0,
            novel_coordinate_shape_fraction: 1.0,
        };
        summary.test.real_generation_metrics.chemistry_validity.metrics = BTreeMap::from([
            ("rdkit_parseable_fraction".to_string(), 1.0),
            ("rdkit_finite_conformer_fraction".to_string(), 1.0),
            ("rdkit_sanitized_fraction".to_string(), 1.0),
            ("rdkit_unique_smiles_fraction".to_string(), 1.0),
        ]);

        let evidence = build_chemistry_benchmark_evidence(&summary);

        assert!(evidence.external_benchmark_backed);
        assert_eq!(
            evidence.external_benchmark_dataset.as_deref(),
            Some("lp_pdbbind_refined")
        );
        assert!(evidence
            .external_benchmark_note
            .contains("lp_pdbbind_refined"));
    }

    #[test]
    fn split_review_counts_wins_losses_and_ties() {
        let lightweight = GenerationQualitySummary {
            primary_objective: "conditioned_denoising".to_string(),
            variant_label: Some("interaction_lightweight".to_string()),
            interaction_mode: "lightweight".to_string(),
            candidate_valid_fraction: Some(1.0),
            pocket_contact_fraction: Some(1.0),
            pocket_compatibility_fraction: Some(1.0),
            mean_centroid_offset: Some(0.8),
            strict_pocket_fit_score: Some(0.55),
            unique_smiles_fraction: Some(0.8),
            unseen_protein_fraction: 1.0,
            topology_specialization_score: 0.2,
            geometry_specialization_score: 0.4,
            pocket_specialization_score: 0.5,
            slot_activation_mean: 0.7,
            gate_activation_mean: 0.3,
            leakage_proxy_mean: 0.08,
        };
        let transformer = GenerationQualitySummary {
            primary_objective: "conditioned_denoising".to_string(),
            variant_label: Some("interaction_transformer".to_string()),
            interaction_mode: "transformer".to_string(),
            candidate_valid_fraction: Some(1.0),
            pocket_contact_fraction: Some(1.0),
            pocket_compatibility_fraction: Some(1.0),
            mean_centroid_offset: Some(0.9),
            strict_pocket_fit_score: Some(0.5),
            unique_smiles_fraction: Some(0.8),
            unseen_protein_fraction: 1.0,
            topology_specialization_score: 0.6,
            geometry_specialization_score: 0.7,
            pocket_specialization_score: 0.55,
            slot_activation_mean: 0.8,
            gate_activation_mean: 0.4,
            leakage_proxy_mean: 0.04,
        };

        let review = build_split_review(&lightweight, &transformer);

        assert_eq!(review.tally.lightweight_wins, 2);
        assert_eq!(review.tally.transformer_wins, 6);
        assert_eq!(review.tally.ties, 4);
        assert_eq!(review.geometric_fit[3].winner, MetricWinner::Lightweight);
        assert_eq!(review.specialization[0].winner, MetricWinner::Transformer);
        assert_eq!(review.utilization[0].winner, MetricWinner::Transformer);
    }

    #[test]
    fn multi_seed_metric_aggregate_reports_deterministic_confidence_interval() {
        let aggregate = MultiSeedMetricAggregate::from_values(&[1.0, 2.0, 3.0]);

        assert_eq!(aggregate.count, 3);
        assert!((aggregate.mean - 2.0).abs() < 1e-12);
        assert!((aggregate.std - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((aggregate.standard_error - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((aggregate.confidence95_low - (2.0 - 4.303 / 3.0_f64.sqrt())).abs() < 1e-12);
        assert!((aggregate.confidence95_high - (2.0 + 4.303 / 3.0_f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn performance_gate_report_tracks_threshold_failures() {
        let config = PerformanceGateConfig {
            min_validation_examples_per_second: Some(10.0),
            min_test_examples_per_second: Some(20.0),
            max_validation_memory_mb: Some(5.0),
            max_test_memory_mb: Some(5.0),
        };
        let validation = ResourceUsageMetrics {
            memory_usage_mb: 6.0,
            evaluation_time_ms: 100.0,
            examples_per_second: 9.0,
            average_ligand_atoms: 5.0,
            average_pocket_atoms: 10.0,
        };
        let test = ResourceUsageMetrics {
            memory_usage_mb: 1.0,
            evaluation_time_ms: 100.0,
            examples_per_second: 30.0,
            average_ligand_atoms: 5.0,
            average_pocket_atoms: 10.0,
        };

        let report = build_performance_gate_report(&config, &validation, &test);

        assert!(!report.passed);
        assert_eq!(report.failed_reasons.len(), 2);
    }

    #[test]
    fn ablation_variants_cover_backend_and_regularizer_dimensions() {
        let mut config = UnseenPocketExperimentConfig::default();
        config.ablation_matrix.include_backend_family_ablation = true;
        config.ablation_matrix.include_slot_count_ablation = true;
        config.ablation_matrix.include_gate_sparsity_ablation = true;
        config.ablation_matrix.include_leakage_penalty_ablation = true;
        config.research.model.num_slots = 6;
        config.research.training.loss_weights.eta_gate = 0.2;
        config.research.training.loss_weights.delta_leak = 0.1;

        let labels = ablation_variants(&config)
            .into_iter()
            .filter_map(|variant| variant.variant_label)
            .collect::<std::collections::BTreeSet<_>>();

        assert!(labels.contains("backend_flow_matching"));
        assert!(labels.contains("backend_autoregressive_graph_geometry"));
        assert!(labels.contains("backend_energy_guided_refinement"));
        assert!(labels.contains("slot_count_reduced"));
        assert!(labels.contains("gate_sparsity_disabled"));
        assert!(labels.contains("leakage_penalty_disabled"));
    }

    #[test]
    fn candidate_layer_reports_diversity_proxies() {
        let mut first = test_candidate(vec![6, 6], vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
        first.inferred_bonds = vec![(0, 1)];
        let mut second = test_candidate(vec![6, 8], vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        second.inferred_bonds = vec![(0, 1)];

        let metrics =
            summarize_candidate_layer(&[first, second], &NoveltyReferenceSignatures::default());

        assert_eq!(metrics.candidate_count, 2);
        assert_eq!(metrics.atom_type_sequence_diversity, 1.0);
        assert_eq!(metrics.bond_topology_diversity, 0.5);
        assert_eq!(metrics.coordinate_shape_diversity, 1.0);
        assert_eq!(metrics.novel_atom_type_sequence_fraction, 1.0);
        assert_eq!(metrics.novel_bond_topology_fraction, 1.0);
        assert_eq!(metrics.novel_coordinate_shape_fraction, 1.0);
    }

    #[test]
    fn candidate_layer_reports_novelty_against_reference_examples() {
        let reference = crate::data::MolecularExample::from_legacy(
            "ref",
            "protein",
            &crate::types::Ligand {
                atoms: vec![
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [0.0, 0.0, 0.0],
                        index: 0,
                    },
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [1.5, 0.0, 0.0],
                        index: 1,
                    },
                ],
                bonds: vec![(0, 1)],
                fingerprint: None,
            },
            &crate::types::Pocket {
                name: "pocket".to_string(),
                atoms: vec![
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [0.0, 0.0, 0.0],
                        index: 0,
                    },
                    crate::types::Atom {
                        atom_type: crate::types::AtomType::Carbon,
                        coords: [2.0, 0.0, 0.0],
                        index: 1,
                    },
                ],
            },
        );
        let novelty_reference = novelty_reference_signatures(&[reference]);
        let mut seen = test_candidate(vec![0, 0], vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
        seen.inferred_bonds = vec![(0, 1)];
        let mut novel = test_candidate(vec![0, 2], vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        novel.inferred_bonds = vec![(0, 1)];

        let metrics = summarize_candidate_layer(&[seen, novel], &novelty_reference);

        assert_eq!(metrics.novel_atom_type_sequence_fraction, 0.5);
        assert_eq!(metrics.novel_bond_topology_fraction, 0.0);
        assert_eq!(metrics.novel_coordinate_shape_fraction, 1.0);
    }

    #[test]
    fn backend_metrics_choose_best_clean_candidate_across_layers() {
        let temp = tempfile::tempdir().unwrap();
        let pocket_path = temp.path().join("pocket.pdb");
        std::fs::write(
            &pocket_path,
            concat!(
                "ATOM      1  C   LIG A   1       0.000   0.000   0.000\n",
                "ATOM      2  C   LIG A   1       0.200   0.000   0.000\n",
                "ATOM      3  C   LIG A   1       0.400   0.000   0.000\n",
                "ATOM      4  C   LIG A   1       0.600   0.000   0.000\n",
                "ATOM      5  C   LIG A   1       0.800   0.000   0.000\n",
            ),
        )
        .unwrap();

        let mut inferred = test_candidate(vec![6, 6], vec![[2.1, 0.0, 0.0], [2.5, 0.0, 0.0]]);
        inferred.source_pocket_path = Some(pocket_path.display().to_string());
        let mut reranked = test_candidate(vec![6, 6], vec![[4.0, 0.0, 0.0], [4.4, 0.0, 0.0]]);
        reranked.source_pocket_path = Some(pocket_path.display().to_string());
        let reranked = vec![reranked];
        let proxy = vec![test_candidate(
            vec![6, 6],
            vec![[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )];

        let selected = final_backend_candidate_layer(&[inferred], &reranked, &proxy);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [2.1, 0.0, 0.0]);
    }

    #[test]
    fn backend_metrics_fall_back_to_proxy_when_reranked_is_empty() {
        let inferred = vec![test_candidate(
            vec![6, 6],
            vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        )];
        let proxy = vec![test_candidate(
            vec![6, 6],
            vec![[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )];

        let selected = final_backend_candidate_layer(&inferred, &[], &proxy);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn backend_metrics_skip_reranked_candidate_with_pocket_clash() {
        let temp = tempfile::tempdir().unwrap();
        let pocket_path = temp.path().join("pocket.pdb");
        std::fs::write(
            &pocket_path,
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000\n",
        )
        .unwrap();

        let inferred = vec![test_candidate(
            vec![6, 6],
            vec![[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        )];
        let mut clashing = test_candidate(vec![6, 6], vec![[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]);
        clashing.source_pocket_path = Some(pocket_path.display().to_string());
        let mut clean = test_candidate(vec![6, 6], vec![[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]]);
        clean.source_pocket_path = Some(pocket_path.display().to_string());
        let reranked = vec![clashing, clean];

        let selected = final_backend_candidate_layer(&inferred, &reranked, &[]);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [2.0, 0.0, 0.0]);
    }

    #[test]
    fn backend_metrics_fall_back_when_all_reranked_candidates_clash() {
        let temp = tempfile::tempdir().unwrap();
        let pocket_path = temp.path().join("pocket.pdb");
        std::fs::write(
            &pocket_path,
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000\n",
        )
        .unwrap();

        let mut inferred = test_candidate(vec![6, 6], vec![[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]]);
        inferred.source_pocket_path = Some(pocket_path.display().to_string());
        let mut clashing = test_candidate(vec![6, 6], vec![[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]);
        clashing.source_pocket_path = Some(pocket_path.display().to_string());

        let inferred = [inferred];
        let reranked = [clashing];
        let selected = final_backend_candidate_layer(&inferred, &reranked, &[]);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0], [2.0, 0.0, 0.0]);
    }

    fn test_generation_quality_summary() -> GenerationQualitySummary {
        GenerationQualitySummary {
            primary_objective: "conditioned_denoising".to_string(),
            variant_label: Some("test".to_string()),
            interaction_mode: "transformer".to_string(),
            candidate_valid_fraction: Some(1.0),
            pocket_contact_fraction: Some(1.0),
            pocket_compatibility_fraction: Some(1.0),
            mean_centroid_offset: Some(0.5),
            strict_pocket_fit_score: Some(0.6),
            unique_smiles_fraction: Some(0.8),
            unseen_protein_fraction: 1.0,
            topology_specialization_score: 0.5,
            geometry_specialization_score: 0.5,
            pocket_specialization_score: 0.5,
            slot_activation_mean: 0.4,
            gate_activation_mean: 0.3,
            leakage_proxy_mean: 0.05,
        }
    }

    fn test_claim_report() -> ClaimReport {
        ClaimReport {
            artifact_dir: "checkpoints/test".into(),
            run_label: "test".to_string(),
            validation: test_generation_quality_summary(),
            test: test_generation_quality_summary(),
            backend_metrics: RealGenerationMetrics {
                chemistry_validity: ReservedBackendMetrics {
                    available: true,
                    backend_name: Some("chemistry".to_string()),
                    metrics: BTreeMap::from([
                        ("valid_fraction".to_string(), 1.0),
                        ("rdkit_sanitized_fraction".to_string(), 1.0),
                    ]),
                    status: "test".to_string(),
                },
                docking_affinity: ReservedBackendMetrics {
                    available: true,
                    backend_name: Some("docking".to_string()),
                    metrics: BTreeMap::new(),
                    status: "test".to_string(),
                },
                pocket_compatibility: ReservedBackendMetrics {
                    available: true,
                    backend_name: Some("pocket".to_string()),
                    metrics: BTreeMap::from([("clash_fraction".to_string(), 0.0)]),
                    status: "test".to_string(),
                },
            },
            backend_thresholds: BTreeMap::new(),
            backend_review: BackendReviewReport::default(),
            layered_generation_metrics: empty_layered_generation_metrics(),
            chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary::default(),
            claim_context: ClaimContext::default(),
            backend_environment: None,
            ablation_deltas: Vec::new(),
            reranker_report: RerankerReport::default(),
            slot_stability: SlotStabilityMetrics::default(),
            leakage_calibration: LeakageCalibrationReport::default(),
            performance_gates: PerformanceGateReport::default(),
            baseline_comparisons: Vec::new(),
            method_comparison: MethodComparisonSummary::default(),
        }
    }

    fn test_candidate(atom_types: Vec<i64>, coords: Vec<[f32; 3]>) -> GeneratedCandidateRecord {
        GeneratedCandidateRecord {
            example_id: "example".to_string(),
            protein_id: "protein".to_string(),
            molecular_representation: None,
            atom_types,
            coords,
            inferred_bonds: Vec::new(),
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 6.0,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: "test".to_string(),
            source_pocket_path: Some("pocket.pdb".to_string()),
            source_ligand_path: Some("ligand.sdf".to_string()),
        }
    }
}
