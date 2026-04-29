#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn unavailable_backend_metrics(status: &str) -> ReservedBackendMetrics {
        ReservedBackendMetrics {
            available: false,
            backend_name: None,
            metrics: BTreeMap::new(),
            status: status.to_string(),
        }
    }

    #[test]
    fn cross_attention_velocity_head_ablation_labels_are_stable() {
        let mut research = ResearchConfig::default();
        let baseline = flow_head_ablation_diagnostics(&research, false);
        assert_eq!(baseline.ablation_label, "geometry_mean_pooling");
        assert_eq!(baseline.head_kind, "geometry");
        assert!(!baseline.local_atom_pocket_attention);
        assert_eq!(baseline.enabled_flow_branches, vec!["geometry"]);
        assert!(baseline
            .disabled_flow_branches
            .contains(&"atom_type".to_string()));
        assert!(!baseline.full_molecular_flow_claim_allowed);
        assert_eq!(baseline.claim_gate_reason, "claim_not_requested");
        assert_eq!(
            baseline.decoder_conditioning_kind,
            "local_atom_slot_attention"
        );
        assert_eq!(
            baseline.molecular_flow_conditioning_kind,
            "local_atom_slot_attention"
        );
        assert!(baseline.slot_local_conditioning_enabled);
        assert!(!baseline.mean_pooled_conditioning_ablation);

        research.model.decoder_conditioning.kind =
            crate::config::DecoderConditioningKind::MeanPooled;
        let mean_pooled = flow_head_ablation_diagnostics(&research, false);
        assert_eq!(mean_pooled.decoder_conditioning_kind, "mean_pooled");
        assert_eq!(
            mean_pooled.molecular_flow_conditioning_kind,
            "mean_pooled"
        );
        assert!(!mean_pooled.slot_local_conditioning_enabled);
        assert!(mean_pooled.mean_pooled_conditioning_ablation);
        research.model.decoder_conditioning.kind =
            crate::config::DecoderConditioningKind::LocalAtomSlotAttention;

        research.model.flow_velocity_head.kind =
            crate::config::FlowVelocityHeadKind::AtomPocketCrossAttention;
        let local = flow_head_ablation_diagnostics(&research, true);
        assert_eq!(local.ablation_label, "local_pocket_attention");
        assert_eq!(local.head_kind, "atom_pocket_cross_attention");
        assert!(local.local_atom_pocket_attention);
        assert!(local.diagnostics_available);

        research.model.pairwise_geometry.enabled = true;
        let combined = flow_head_ablation_diagnostics(&research, true);
        assert_eq!(combined.ablation_label, "pairwise_plus_local_pocket");
        assert!(combined.pairwise_geometry_enabled);
    }

    #[test]
    fn unseen_pocket_experiment_smoke_writes_leakage_audit_artifact() {
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
        let slot_report_path = summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("slot_semantic_report.json");
        assert!(slot_report_path.exists());
        let slot_report: SlotSemanticReportArtifact =
            serde_json::from_str(&std::fs::read_to_string(slot_report_path).unwrap()).unwrap();
        assert_eq!(slot_report.schema_version, 1);
        let latest_training = slot_report.latest_training.as_ref().unwrap();
        assert_eq!(latest_training.slot_signatures.len(), 3);
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
        let leakage_audit_path = summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("frozen_leakage_probe_audit.json");
        assert!(leakage_audit_path.exists());
        let leakage_audit: FrozenLeakageProbeCalibrationReport =
            serde_json::from_str(&std::fs::read_to_string(&leakage_audit_path).unwrap()).unwrap();
        assert_eq!(leakage_audit.split_name, "test");
        assert!(matches!(
            leakage_audit.calibration_status.as_str(),
            "ok" | "insufficient_data"
        ));
        assert_eq!(summary.dataset_validation.parsed_examples, 4);
        assert_eq!(
            summary.coordinate_frame.coordinate_frame_contract,
            summary.dataset_validation.coordinate_frame_contract
        );
        assert_eq!(
            summary.coordinate_frame.rotation_consistency_role,
            "diagnostic_not_exact_equivariance_claim"
        );

        let claim_report_path = summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("claim_summary.json");
        let claim_report_payload = std::fs::read_to_string(claim_report_path).unwrap();
        let claim_report: ClaimReport = serde_json::from_str(&claim_report_payload).unwrap();
        let raw_index = claim_report_payload.find("\"raw_native_evidence\"").unwrap();
        let processed_index = claim_report_payload
            .find("\"processed_generation_evidence\"")
            .unwrap();
        let validation_index = claim_report_payload.find("\"validation\"").unwrap();
        assert!(raw_index < processed_index);
        assert!(processed_index < validation_index);
        assert!(claim_report.layer_provenance_audit.claim_safe);
        assert!(claim_report.layer_provenance_audit.raw_layer_model_native);
        assert!(
            claim_report
                .layer_provenance_audit
                .processed_layer_has_contract
                || claim_report.layer_provenance_audit.processed_layer == "unavailable"
        );
        assert_eq!(claim_report.model_design.raw_model_layer, "raw_rollout");
        assert_eq!(
            claim_report.raw_native_evidence.evidence_role,
            "model_native_raw_first"
        );
        assert_eq!(
            claim_report.raw_native_evidence.valid_fraction,
            Some(claim_report.model_design.raw_model_valid_fraction)
        );
        assert_eq!(
            claim_report.raw_native_evidence.native_graph_valid_fraction,
            Some(claim_report.model_design.raw_native_graph_valid_fraction)
        );
        assert!(claim_report
            .raw_native_evidence
            .strict_pocket_fit_score
            .is_some());
        assert!(claim_report
            .raw_native_evidence
            .leakage_proxy_mean
            .is_some());
        assert_eq!(
            claim_report
                .method_comparison
                .raw_native_evidence
                .evidence_role,
            "model_native_raw_first"
        );
        assert_eq!(
            claim_report
                .processed_generation_evidence
                .evidence_role,
            "additive_processed_or_reranked_evidence"
        );
        assert_eq!(
            claim_report.postprocessing_repair_audit.split_label,
            "test"
        );
        assert!(claim_report
            .postprocessing_repair_audit
            .claim_boundary
            .contains("postprocessing evidence"));
        assert_eq!(
            claim_report
                .postprocessing_repair_audit
                .no_repair_ablation
                .no_repair_layer,
            "raw_rollout"
        );
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("repair_case_audit_test.json")
            .exists());
        assert_eq!(claim_report.train_eval_alignment.schema_version, 1);
        assert_eq!(
            claim_report
                .leakage_calibration
                .frozen_probe_calibration
                .split_name,
            "test"
        );
        assert_eq!(
            claim_report
                .leakage_calibration
                .leakage_roles
                .frozen_probe_audit
                .status,
            leakage_audit.calibration_status
        );
        assert!(claim_report
            .leakage_calibration
            .capacity_sweep_artifact
            .as_deref()
            .is_some_and(|path| path.ends_with("frozen_leakage_probe_audit.json")));

        let mut missing_processed_provenance = summary.clone();
        missing_processed_provenance.test.model_design.processed_layer =
            "reranked_candidates".to_string();
        missing_processed_provenance
            .test
            .layered_generation_metrics
            .generation_path_contract
            .clear();
        let audit = build_layer_provenance_audit(&missing_processed_provenance);
        assert!(!audit.claim_safe);
        assert!(!audit.processed_layer_has_contract);
    }

    #[test]
    fn evaluation_claim_summary_raw_native_gate_fails_even_when_processed_improves() {
        let mut summary =
            test_experiment_summary_with_interaction_mode(CrossAttentionMode::Transformer);
        summary.config.performance_gates = PerformanceGateConfig {
            min_test_raw_model_valid_fraction: Some(0.8),
            min_test_raw_model_pocket_contact_fraction: Some(0.7),
            max_test_raw_model_clash_fraction: Some(0.1),
            min_test_raw_native_graph_valid_fraction: Some(0.9),
            ..PerformanceGateConfig::default()
        };
        summary.test.model_design = ModelDesignEvaluationMetrics {
            raw_model_layer: "raw_rollout".to_string(),
            processed_layer: "reranked_candidates".to_string(),
            raw_model_valid_fraction: 0.5,
            raw_model_pocket_contact_fraction: 0.6,
            raw_model_clash_fraction: 0.2,
            raw_model_mean_displacement: 0.8,
            raw_native_graph_valid_fraction: 0.4,
            raw_native_bond_count_mean: 2.0,
            raw_native_component_count_mean: 2.0,
            raw_native_valence_violation_fraction: 0.3,
            raw_native_topology_bond_sync_fraction: 0.9,
            processed_valid_fraction: 1.0,
            processed_pocket_contact_fraction: 1.0,
            processed_clash_fraction: 0.0,
            processing_valid_fraction_delta: 0.5,
            processing_pocket_contact_delta: 0.4,
            processing_clash_delta: -0.2,
            slot_activation_mean: 0.4,
            gate_activation_mean: 0.3,
            leakage_proxy_mean: 0.05,
            processed_postprocessor_chain: vec!["calibrated_reranking".to_string()],
            processed_claim_boundary:
                "reranked candidates are processed evidence, not raw-native model capability"
                    .to_string(),
            ..ModelDesignEvaluationMetrics::default()
        };
        summary.test.layered_generation_metrics.raw_rollout = CandidateLayerMetrics {
            candidate_count: 4,
            valid_fraction: 0.5,
            pocket_contact_fraction: 0.6,
            mean_centroid_offset: 4.0,
            clash_fraction: 0.2,
            mean_displacement: 0.8,
            uniqueness_proxy_fraction: 0.5,
            native_graph_valid_fraction: 0.4,
            native_bond_count_mean: 2.0,
            native_component_count_mean: 2.0,
            native_valence_violation_fraction: 0.3,
            topology_bond_sync_fraction: 0.9,
            ..CandidateLayerMetrics::default()
        };
        summary.test.layered_generation_metrics.raw_flow =
            summary.test.layered_generation_metrics.raw_rollout.clone();
        summary.test.layered_generation_metrics.reranked_candidates = CandidateLayerMetrics {
            candidate_count: 4,
            valid_fraction: 1.0,
            pocket_contact_fraction: 1.0,
            mean_centroid_offset: 0.5,
            clash_fraction: 0.0,
            uniqueness_proxy_fraction: 1.0,
            ..CandidateLayerMetrics::default()
        };

        let claim = build_claim_report(&summary);

        assert!(!claim.raw_native_evidence.raw_native_gate.passed);
        assert!(claim
            .raw_native_evidence
            .raw_native_gate
            .processed_metrics_excluded);
        assert!(claim
            .raw_native_evidence
            .raw_native_gate
            .failed_reasons
            .iter()
            .any(|reason| reason.contains("raw_model_valid_fraction")));
        assert_eq!(
            claim.processed_generation_evidence.valid_fraction,
            Some(1.0)
        );
        assert_eq!(
            claim
                .method_comparison
                .raw_native_evidence
                .raw_native_gate
                .passed,
            claim.raw_native_evidence.raw_native_gate.passed
        );
    }

    #[test]
    fn repair_case_audit_reports_help_harm_neutral_and_no_repair_ablation() {
        let raw_help = test_candidate(vec![6], vec![[8.0, 0.0, 0.0]]);
        let repaired_help = repaired_test_candidate(vec![6], vec![[1.0, 0.0, 0.0]]);
        let raw_harm = test_candidate(vec![6], vec![[1.0, 0.0, 0.0]]);
        let repaired_harm = repaired_test_candidate(vec![6], vec![[8.0, 0.0, 0.0]]);
        let raw_neutral = test_candidate(vec![6], vec![[2.0, 0.0, 0.0]]);
        let repaired_neutral = repaired_test_candidate(vec![6], vec![[2.05, 0.0, 0.0]]);
        let raw = vec![raw_help, raw_harm, raw_neutral];
        let repaired = vec![repaired_help, repaired_harm, repaired_neutral];
        let novelty = NoveltyReferenceSignatures::default();
        let raw_metrics = summarize_candidate_layer(&raw, &novelty);
        let repaired_metrics = summarize_candidate_layer(&repaired, &novelty);

        let audit =
            build_repair_case_audit("test", &raw, &repaired, &raw_metrics, &repaired_metrics);

        assert_eq!(audit.no_repair_ablation.no_repair_layer, "raw_rollout");
        assert!(audit.no_repair_ablation.repair_enabled);
        assert_eq!(audit.repair_helps.len(), 1);
        assert_eq!(audit.repair_harms.len(), 1);
        assert_eq!(audit.repair_neutral.len(), 1);
        assert!(!audit.raw_failures.is_empty());
        assert!(audit.raw_vs_repaired_delta.strict_pocket_fit_score_delta.is_finite());
        assert!(audit
            .claim_boundary
            .contains("must not be cited as raw generation quality"));
    }

    #[test]
    fn evaluation_reports_chemistry_collaboration_provenance() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.best_metric = "finite_forward_fraction".to_string();
        let examples = crate::data::synthetic_phase1_examples()
            .into_iter()
            .take(2)
            .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
            .collect::<Vec<_>>();
        let train_proteins = examples
            .iter()
            .map(|example| example.protein_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);

        let metrics = evaluate_split(
            &system,
            &examples,
            &examples,
            &train_proteins,
            &config,
            AblationConfig::default(),
            &ExternalEvaluationConfig::default(),
            "validation",
            tch::Device::Cpu,
        );

        assert!(!metrics
            .chemistry_collaboration
            .gate_usage_by_chemical_role
            .is_empty());
        assert!(metrics
            .chemistry_collaboration
            .bond_length_guardrail_mean
            .value
            .is_some());
        assert_eq!(
            metrics
                .chemistry_collaboration
                .key_residue_contact_coverage
                .value,
            None
        );
        assert_eq!(
            metrics
                .chemistry_collaboration
                .key_residue_contact_coverage
                .provenance,
            ChemistryMetricProvenance::Unavailable
        );
        assert_eq!(
            metrics
                .comparison_summary
                .chemistry_collaboration
                .key_residue_contact_coverage
                .provenance,
            ChemistryMetricProvenance::Unavailable
        );
        assert_eq!(metrics.proxy_task_metrics.probe_baselines.len(), 6);
        assert!(metrics.proxy_task_metrics.probe_baselines.iter().any(|row| {
            row.target == "ligand_pharmacophore_roles"
                && row.supervision_status == "available"
                && row.available_count > 0
        }));
        assert!(metrics
            .proxy_task_metrics
            .probe_baselines
            .iter()
            .all(|row| !row.supervision_status.is_empty()));
        assert_eq!(metrics.model_design.raw_model_layer, "raw_rollout");
        assert!(metrics
            .model_design
            .raw_vs_processed_note
            .contains("model-native rollout quality"));
        assert!(!metrics.model_design.processed_claim_boundary.is_empty());
        assert!(metrics.model_design.raw_native_bond_count_mean.is_finite());
        assert!(metrics
            .model_design
            .raw_native_component_count_mean
            .is_finite());
        assert!(metrics
            .model_design
            .raw_native_valence_violation_fraction
            .is_finite());
        assert!(metrics
            .model_design
            .raw_native_topology_bond_sync_fraction
            .is_finite());
        assert!(metrics.model_design.raw_native_atom_type_entropy.is_finite());
        assert!(metrics
            .model_design
            .raw_native_graph_valid_fraction
            .is_finite());
        assert!(metrics.model_design.geometry_consistency_score.is_finite());
        assert!(metrics.model_design.examples_per_second.is_finite());
        assert!(!metrics
            .slot_stability
            .topology_slot_alignment
            .is_empty());
        assert_eq!(metrics.slot_stability.signature_matching.len(), 3);
        assert_eq!(metrics.slot_stability.collapse_warnings.len(), 3);
        assert_eq!(metrics.slot_stability.modality_usage.len(), 3);
        assert_eq!(
            metrics.slot_stability.stage_guard_warning_count,
            metrics
                .slot_stability
                .modality_usage
                .iter()
                .filter(|usage| usage.stage_guard_collapse_warning)
                .count()
        );
        for usage in &metrics.slot_stability.modality_usage {
            assert!(usage.sample_count > 0);
            assert!(usage.slot_count > 0);
            assert!(
                usage.dead_slot_count + usage.diffuse_slot_count + usage.saturated_slot_count
                    <= usage.slot_count
            );
            assert!(!usage.semantic_enrichment.target_family.is_empty());
            assert!(usage.semantic_enrichment.role_enrichment_score.is_finite());
        }
        assert!(metrics.model_design.gate_saturation_fraction.is_finite());
        assert_eq!(
            metrics.resource_usage.evaluation_batch_size,
            config.data.batch_size
        );
        assert_eq!(metrics.resource_usage.forward_batch_count, 1);
        assert!(metrics.resource_usage.no_grad);
        assert!(metrics.resource_usage.batched_forward);
        assert!(metrics.resource_usage.examples_per_second.is_finite());
        assert_eq!(metrics.train_eval_alignment.schema_version, 1);
        assert!(metrics
            .train_eval_alignment
            .metric_rows
            .iter()
            .any(|row| row.metric_name == "rollout_eval_recovery"
                && !row.optimizer_facing
                && row.detached_diagnostic));
        assert!(metrics
            .train_eval_alignment
            .metric_rows
            .iter()
            .any(|row| row.metric_name == "raw_model_valid_fraction"
                && row.candidate_layer.as_deref() == Some("raw_rollout")
                && row.model_native_raw));
        assert!(metrics
            .train_eval_alignment
            .metric_rows
            .iter()
            .any(|row| row.metric_name == "raw_native_graph_valid_fraction"
                && row.candidate_layer.as_deref() == Some("raw_rollout")
                && row.model_native_raw
                && row.claim_boundary.contains("before repair")));
        assert!(metrics
            .train_eval_alignment
            .metric_rows
            .iter()
            .any(|row| row.metric_name == "processed_valid_fraction"
                && row.claim_boundary.contains("must not overwrite raw fields")));
        assert!(metrics
            .train_eval_alignment
            .backend_coverage
            .iter()
            .any(|row| row.backend_slot == "pocket_compatibility"));
        assert_eq!(
            metrics.train_eval_alignment.best_metric_review.status,
            "smoke_default"
        );
    }

    #[test]
    fn slot_signature_matching_is_permutation_aware() {
        let left = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let permuted = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let report = match_slot_signatures("topology", "cross_seed_smoke", &left, &permuted, 0.9);

        assert_eq!(report.matched_slot_count, 2);
        assert_eq!(report.unmatched_left_slots, 0);
        assert_eq!(report.unmatched_right_slots, 0);
        assert!(!report.collapse_warning);
        assert!(report.mean_matched_similarity > 0.99);
        assert!(report.cross_seed_matching_score > 0.99);

        let collapsed = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let collapsed_report =
            match_slot_signatures("topology", "cross_seed_smoke", &left, &collapsed, 0.9);
        assert_eq!(collapsed_report.matched_slot_count, 1);
        assert!(collapsed_report.collapse_warning);
    }

    #[test]
    fn slot_collapse_warning_statuses_cover_dead_saturated_and_balanced() {
        let dead = slot_collapse_warning_from_stats("geometry", 0.0, 0.0, 0.0, 0.0);
        assert_eq!(dead.status, "dead");

        let saturated = slot_collapse_warning_from_stats("geometry", 1.0, 1.0, 1.3, 0.3);
        assert_eq!(saturated.status, "saturated");

        let dominated = slot_collapse_warning_from_stats("geometry", 0.5, 0.5, 0.2, 0.9);
        assert_eq!(dominated.status, "single_slot_dominated");

        let balanced = slot_collapse_warning_from_stats("geometry", 0.5, 0.5, 1.2, 0.4);
        assert_eq!(balanced.status, "balanced");
    }

    #[test]
    fn evaluation_batched_forward_matches_single_batch_metrics() {
        let mut batched_config = ResearchConfig::default();
        batched_config.data.batch_size = 2;
        let mut single_config = batched_config.clone();
        single_config.data.batch_size = 1;
        let examples = crate::data::synthetic_phase1_examples()
            .into_iter()
            .take(2)
            .map(|example| example.with_pocket_feature_dim(batched_config.model.pocket_feature_dim))
            .collect::<Vec<_>>();
        let train_proteins = examples
            .iter()
            .map(|example| example.protein_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &batched_config);

        let batched = evaluate_split(
            &system,
            &examples,
            &examples,
            &train_proteins,
            &batched_config,
            AblationConfig::default(),
            &ExternalEvaluationConfig::default(),
            "validation",
            tch::Device::Cpu,
        );
        let single = evaluate_split(
            &system,
            &examples,
            &examples,
            &train_proteins,
            &single_config,
            AblationConfig::default(),
            &ExternalEvaluationConfig::default(),
            "validation",
            tch::Device::Cpu,
        );

        assert_eq!(batched.resource_usage.evaluation_batch_size, 2);
        assert_eq!(single.resource_usage.evaluation_batch_size, 1);
        assert_eq!(batched.resource_usage.forward_batch_count, 1);
        assert_eq!(single.resource_usage.forward_batch_count, examples.len());
        assert_eq!(batched.resource_usage.per_example_forward_count, 0);
        assert_eq!(single.resource_usage.per_example_forward_count, examples.len());
        assert!(batched.resource_usage.no_grad);
        assert!(single.resource_usage.no_grad);
        assert!(batched.resource_usage.batched_forward);
        assert!(single.resource_usage.batched_forward);
        assert_f64_close(
            "finite forward fraction",
            batched.representation_diagnostics.finite_forward_fraction,
            single.representation_diagnostics.finite_forward_fraction,
        );
        assert_f64_close(
            "distance probe rmse",
            batched.representation_diagnostics.distance_probe_rmse,
            single.representation_diagnostics.distance_probe_rmse,
        );
        assert_f64_close(
            "topology reconstruction mse",
            batched.representation_diagnostics.topology_reconstruction_mse,
            single.representation_diagnostics.topology_reconstruction_mse,
        );
        assert_f64_close(
            "leakage proxy mean",
            batched.representation_diagnostics.leakage_proxy_mean,
            single.representation_diagnostics.leakage_proxy_mean,
        );
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

    fn assert_f64_close(name: &str, left: f64, right: f64) {
        let tolerance = 1e-5;
        assert!(
            (left - right).abs() <= tolerance,
            "{name} differs: left={left:.8} right={right:.8}"
        );
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
            &ChemistryCollaborationMetrics::default(),
        );

        assert_eq!(summary.mean_centroid_offset, Some(1.75));
        assert_eq!(summary.strict_pocket_fit_score, Some(0.31));
        assert_eq!(summary.unique_smiles_fraction, None);
        assert_eq!(
            summary.primary_objective_provenance,
            "decoder_anchored_conditioned_denoising"
        );
        assert_eq!(
            summary.primary_objective_claim_boundary,
            "target_ligand_denoising_or_refinement_training_signal"
        );
    }

    #[test]
    fn comparison_summary_labels_objective_claim_boundaries() {
        let metrics = RealGenerationMetrics {
            chemistry_validity: unavailable_backend_metrics("test"),
            docking_affinity: unavailable_backend_metrics("test"),
            pocket_compatibility: unavailable_backend_metrics("test"),
        };
        let mut surrogate = AblationConfig::default();
        surrogate.primary_objective_override =
            Some(crate::config::PrimaryObjectiveConfig::SurrogateReconstruction);
        let surrogate_summary = build_comparison_summary(
            &ResearchConfig::default(),
            &surrogate,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            &metrics,
            &ChemistryCollaborationMetrics::default(),
        );
        assert_eq!(
            surrogate_summary.primary_objective_provenance,
            "bootstrap_debug_shape_safe_surrogate"
        );
        assert_eq!(
            surrogate_summary.primary_objective_claim_boundary,
            "bootstrap_debug_or_shape_safe_baseline_not_generation_quality"
        );

        let mut hybrid = AblationConfig::default();
        hybrid.primary_objective_override =
            Some(crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching);
        let hybrid_summary = build_comparison_summary(
            &ResearchConfig::default(),
            &hybrid,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            &metrics,
            &ChemistryCollaborationMetrics::default(),
        );
        assert_eq!(
            hybrid_summary.primary_objective_claim_boundary,
            "hybrid_training_composition_not_a_separate_generation_mode"
        );
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
            &ChemistryCollaborationMetrics::default(),
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
    fn output_artifact_paths_keep_repo_roots() {
        let base_dir = std::path::Path::new("configs");

        assert_eq!(
            resolve_output_artifact_path(
                base_dir,
                std::path::Path::new("./checkpoints/automated_search")
            ),
            PathBuf::from("./checkpoints/automated_search")
        );
        assert_eq!(
            resolve_output_artifact_path(base_dir, std::path::Path::new("artifacts/evidence/q2")),
            PathBuf::from("artifacts/evidence/q2")
        );
        assert_eq!(
            resolve_output_artifact_path(base_dir, std::path::Path::new("../checkpoints/legacy")),
            PathBuf::from("configs/../checkpoints/legacy")
        );
        assert_eq!(
            resolve_relative_path(base_dir, std::path::Path::new("./surface.json")),
            PathBuf::from("configs/./surface.json")
        );
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
                raw_native_evidence: ClaimRawNativeEvidenceSummary::default(),
                processed_generation_evidence: ClaimProcessedGenerationEvidenceSummary::default(),
                postprocessing_repair_audit: RepairCaseAuditReport::default(),
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
                model_design: ModelDesignEvaluationMetrics::default(),
                layer_provenance_audit: ClaimLayerProvenanceAudit::default(),
                chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary::default(),
                chemistry_collaboration: ChemistryCollaborationMetrics::default(),
                claim_context: ClaimContext::default(),
                backend_environment: None,
                ablation_deltas: Vec::new(),
                reranker_report: RerankerReport::default(),
                slot_stability: SlotStabilityMetrics::default(),
                leakage_calibration: LeakageCalibrationReport::default(),
                performance_gates: PerformanceGateReport::default(),
                baseline_comparisons: Vec::new(),
                method_comparison: MethodComparisonSummary::default(),
                train_eval_alignment: TrainEvalAlignmentReport::default(),
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
            ..Default::default()
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
                .expect(
                    "existing experiment summary should be readable for benchmark-evidence tests",
                ),
        )
        .expect("existing experiment summary should deserialize");
        summary.config.surface_label = Some("lp_pdbbind_refined_real_backends".to_string());
        summary.config.reviewer_benchmark.dataset = Some("lp_pdbbind_refined".to_string());
        summary.config.research.training.checkpoint_dir =
            PathBuf::from("./checkpoints/lp_pdbbind_refined_real_backends");
        summary.dataset_validation.parsed_examples = 5048;
        summary.dataset_validation.retained_label_coverage = 1.0;
        summary.split_report.val.protein_family_proxy_histogram =
            BTreeMap::from_iter((0..12).map(|ix| (format!("val_family_{ix}"), 1usize)));
        summary.split_report.test.protein_family_proxy_histogram =
            BTreeMap::from_iter((0..12).map(|ix| (format!("test_family_{ix}"), 1usize)));
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
            ..Default::default()
        };
        summary
            .test
            .real_generation_metrics
            .chemistry_validity
            .metrics = BTreeMap::from([
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
    fn leakage_calibration_reports_probe_capacity_and_trivial_baselines() {
        let mut summary: UnseenPocketExperimentSummary = serde_json::from_str(
            &std::fs::read_to_string("checkpoints/pdbbindpp_real_backends/experiment_summary.json")
                .expect("existing experiment summary should be readable for leakage tests"),
        )
        .expect("existing experiment summary should deserialize");
        summary.config.research.model.semantic_probes.hidden_layers = 1;
        summary.config.research.model.semantic_probes.hidden_dim = 64;
        summary.test.proxy_task_metrics.probe_baselines = vec![ProbeBaselineMetric {
            target: "topology_adjacency".to_string(),
            loss_kind: "binary_cross_entropy".to_string(),
            observed_loss: Some(0.2),
            trivial_baseline_loss: Some(0.5),
            improves_over_trivial: Some(true),
            supervision_status: "available".to_string(),
            available_count: 2,
            interpretation: "test".to_string(),
        }];

        let report = build_leakage_calibration_report(&summary, &[]);

        assert_eq!(report.probe_capacity.hidden_layers, 1);
        assert_eq!(report.probe_capacity.hidden_dim, 64);
        assert_eq!(report.probe_capacity.architecture, "mlp_relu");
        assert_eq!(report.probe_baseline_comparisons.len(), 1);
        assert!(report.probe_baseline_comparisons[0].improves_over_trivial == Some(true));
        assert_eq!(
            report.frozen_probe_calibration.calibration_status,
            "not_run"
        );
        assert!(report.leakage_roles.optimizer_penalty.active);
        assert_eq!(
            report.leakage_roles.frozen_probe_audit.status,
            "not_run"
        );
        assert!(report.capacity_sweep_artifact.is_none());
    }

    #[test]
    fn split_review_counts_wins_losses_and_ties() {
        let lightweight = GenerationQualitySummary {
            generation_mode: "target_ligand_denoising".to_string(),
            primary_objective: "conditioned_denoising".to_string(),
            primary_objective_provenance: "decoder_anchored_conditioned_denoising".to_string(),
            primary_objective_claim_boundary:
                "target_ligand_denoising_or_refinement_training_signal".to_string(),
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
            chemistry_collaboration: ChemistryCollaborationMetrics::default(),
        };
        let transformer = GenerationQualitySummary {
            generation_mode: "target_ligand_denoising".to_string(),
            primary_objective: "conditioned_denoising".to_string(),
            primary_objective_provenance: "decoder_anchored_conditioned_denoising".to_string(),
            primary_objective_claim_boundary:
                "target_ligand_denoising_or_refinement_training_signal".to_string(),
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
            chemistry_collaboration: ChemistryCollaborationMetrics::default(),
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
            min_test_raw_model_valid_fraction: Some(0.8),
            min_test_raw_model_pocket_contact_fraction: Some(0.7),
            max_test_raw_model_clash_fraction: Some(0.1),
            min_test_raw_native_graph_valid_fraction: Some(0.9),
        };
        let validation = ResourceUsageMetrics {
            memory_usage_mb: 6.0,
            evaluation_time_ms: 100.0,
            examples_per_second: 9.0,
            evaluation_batch_size: 2,
            forward_batch_count: 1,
            per_example_forward_count: 0,
            no_grad: true,
            batched_forward: true,
            de_novo_per_example_reason: None,
            average_ligand_atoms: 5.0,
            average_pocket_atoms: 10.0,
        };
        let test = ResourceUsageMetrics {
            memory_usage_mb: 1.0,
            evaluation_time_ms: 100.0,
            examples_per_second: 30.0,
            evaluation_batch_size: 2,
            forward_batch_count: 1,
            per_example_forward_count: 0,
            no_grad: true,
            batched_forward: true,
            de_novo_per_example_reason: None,
            average_ligand_atoms: 5.0,
            average_pocket_atoms: 10.0,
        };

        let report = build_performance_gate_report(
            &config,
            &validation,
            &test,
            &ModelDesignEvaluationMetrics {
                raw_model_valid_fraction: 0.5,
                raw_model_pocket_contact_fraction: 0.6,
                raw_model_clash_fraction: 0.2,
                raw_native_graph_valid_fraction: 0.4,
                processed_valid_fraction: 1.0,
                processed_pocket_contact_fraction: 1.0,
                processed_clash_fraction: 0.0,
                ..ModelDesignEvaluationMetrics::default()
            },
        );

        assert!(!report.passed);
        assert_eq!(report.failed_reasons.len(), 6);
        assert!(report
            .failed_reasons
            .iter()
            .any(|reason| reason.contains("raw_model_valid_fraction")));
        assert!(report
            .failed_reasons
            .iter()
            .any(|reason| reason.contains("raw_native_graph_valid_fraction")));
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
    fn q7_core_ablation_matrix_covers_model_design_axes() {
        let mut config = UnseenPocketExperimentConfig::default();
        config.ablation_matrix.include_disable_slots = true;
        config.ablation_matrix.include_disable_probes = true;
        config.ablation_matrix.include_disable_leakage = true;
        config.ablation_matrix.include_generation_mode_ablation = true;
        config.ablation_matrix.include_topology_encoder_ablation = true;
        config.ablation_matrix.include_geometry_operator_ablation = true;
        config.ablation_matrix.include_pocket_encoder_ablation = true;
        config.ablation_matrix.include_decoder_conditioning_ablation = true;
        config.ablation_matrix.include_slot_count_ablation = true;
        config.ablation_matrix.include_slot_attention_masking_ablation = true;
        config.ablation_matrix.include_gate_sparsity_ablation = true;
        config.ablation_matrix.include_gate_scale_ablation = true;
        config.ablation_matrix.include_disable_candidate_repair = true;
        config
            .ablation_matrix
            .include_direct_fusion_negative_control = true;
        config.ablation_matrix.include_redundancy_ablation = true;
        config.ablation_matrix.include_modality_focus_ablation = true;
        config.ablation_matrix.include_staged_schedule_ablation = true;
        config.research.model.num_slots = 8;
        config.research.training.loss_weights.eta_gate = 0.05;
        config.research.training.loss_weights.delta_leak = 0.05;

        let variants = ablation_variants(&config);
        let labels = variants
            .iter()
            .filter_map(|variant| variant.variant_label.as_deref())
            .collect::<std::collections::BTreeSet<_>>();

        for required in [
            "generation_mode_ligand_refinement",
            "generation_mode_pocket_only_initialization_baseline",
            "de_novo_full_molecular_flow",
            "topology_encoder_lightweight",
            "topology_encoder_typed_message_passing",
            "geometry_operator_raw_coordinate_projection",
            "geometry_operator_local_frame_pair_message",
            "pocket_encoder_feature_projection",
            "pocket_encoder_ligand_relative_local_frame",
            "disable_slots",
            "slot_count_reduced",
            "slot_attention_masking_disabled",
            "disable_leakage",
            "leakage_penalty_disabled",
            "redundancy_disabled",
            "disable_probes",
            "gate_sparsity_disabled",
            "interaction_gate_temperature_high",
            "direct_fusion_negative_control",
            "topology_only",
            "geometry_only",
            "pocket_only",
            "staged_schedule_disabled",
            "decoder_conditioning_mean_pooled",
            "disable_candidate_repair",
        ] {
            assert!(labels.contains(required), "missing {required}");
        }

        let matrix_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("configs/q7_core_ablation_matrix.json");
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(matrix_path).unwrap()).unwrap();
        let axes = manifest["axes"].as_array().unwrap();
        assert!(axes.iter().any(|axis| axis["axis"] == "generation_mode"
            && axis["variant"] == "generation_mode_pocket_only_initialization_baseline"
            && axis["supported"] == true));
        assert!(axes.iter().any(|axis| axis["axis"] == "generation_mode"
            && axis["variant"] == "de_novo_full_molecular_flow"
            && axis["supported"] == true));
        assert!(axes.iter().any(|axis| axis["axis"] == "topology_encoder"
            && axis["variant"] == "topology_encoder_typed_message_passing"));
        assert!(axes.iter().any(|axis| axis["axis"] == "slot_attention_masking"
            && axis["variant"] == "slot_attention_masking_disabled"));
        assert!(axes.iter().any(|axis| axis["axis"] == "gate_scale"
            && axis["variant"] == "interaction_gate_temperature_high"));
        assert!(axes.iter().any(|axis| axis["axis"] == "interaction_negative_control"
            && axis["variant"] == "direct_fusion_negative_control"));
        assert!(axes
            .iter()
            .any(|axis| axis["axis"] == "redundancy"
                && axis["variant"] == "redundancy_disabled"));
        assert!(axes.iter().any(|axis| axis["axis"] == "modality_focus"
            && axis["variant"] == "topology_only"));
        assert!(axes.iter().any(|axis| axis["axis"] == "training_schedule"
            && axis["variant"] == "staged_schedule_disabled"));
        assert!(axes
            .iter()
            .any(|axis| axis["axis"] == "decoder_conditioning"));
        assert!(manifest["minimum_report_fields"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "model_design.raw_model_valid_fraction"));
    }

    #[test]
    fn claim_context_limits_direct_fusion_negative_control_surfaces() {
        assert_eq!(
            ResearchConfig::default().model.interaction_mode,
            CrossAttentionMode::Transformer
        );

        let controlled = build_claim_report(&test_experiment_summary_with_interaction_mode(
            CrossAttentionMode::Transformer,
        ));
        assert_eq!(controlled.claim_context.interaction_mode, "transformer");
        assert!(!controlled.claim_context.direct_fusion_negative_control);
        assert!(controlled
            .claim_context
            .preferred_architecture_claim_allowed);

        let direct_fusion = build_claim_report(&test_experiment_summary_with_interaction_mode(
            CrossAttentionMode::DirectFusionNegativeControl,
        ));
        assert_eq!(
            direct_fusion.claim_context.interaction_mode,
            "direct_fusion_negative_control"
        );
        assert!(direct_fusion
            .claim_context
            .direct_fusion_negative_control);
        assert!(!direct_fusion
            .claim_context
            .preferred_architecture_claim_allowed);
        assert!(direct_fusion
            .claim_context
            .interaction_claim_boundary
            .contains("ablation-only"));
    }

    #[test]
    fn candidate_layer_reports_diversity_proxies() {
        let mut first = test_candidate(vec![6, 6], vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
        first.inferred_bonds = vec![(0, 1)];
        first.bond_count = first.inferred_bonds.len();
        let mut second = test_candidate(vec![6, 8], vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        second.inferred_bonds = vec![(0, 1)];
        second.bond_count = second.inferred_bonds.len();

        let metrics =
            summarize_candidate_layer(&[first, second], &NoveltyReferenceSignatures::default());

        assert_eq!(metrics.candidate_count, 2);
        assert_eq!(metrics.atom_type_sequence_diversity, 1.0);
        assert_eq!(metrics.bond_topology_diversity, 0.5);
        assert_eq!(metrics.coordinate_shape_diversity, 1.0);
        assert_eq!(metrics.novel_atom_type_sequence_fraction, 1.0);
        assert_eq!(metrics.novel_bond_topology_fraction, 1.0);
        assert_eq!(metrics.novel_coordinate_shape_fraction, 1.0);
        assert_eq!(metrics.native_bond_count_mean, 1.0);
        assert_eq!(metrics.native_component_count_mean, 1.0);
        assert_eq!(metrics.native_valence_violation_fraction, 0.0);
        assert_eq!(metrics.topology_bond_sync_fraction, 1.0);
        assert!(metrics.atom_type_entropy.is_finite());
        assert_eq!(metrics.native_graph_valid_fraction, 1.0);
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
                bond_types: vec![1],
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
            generation_mode: "target_ligand_denoising".to_string(),
            primary_objective: "conditioned_denoising".to_string(),
            primary_objective_provenance: "decoder_anchored_conditioned_denoising".to_string(),
            primary_objective_claim_boundary:
                "target_ligand_denoising_or_refinement_training_signal".to_string(),
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
            chemistry_collaboration: ChemistryCollaborationMetrics::default(),
        }
    }

    fn test_experiment_summary_with_interaction_mode(
        mode: CrossAttentionMode,
    ) -> UnseenPocketExperimentSummary {
        let mut config = UnseenPocketExperimentConfig::default();
        config.research.model.interaction_mode = mode;
        let dataset_validation = crate::data::DatasetValidationReport::default();
        let empty_train = InMemoryDataset::new(Vec::new());
        let empty_val = InMemoryDataset::new(Vec::new());
        let empty_test = InMemoryDataset::new(Vec::new());
        let split_report = SplitReport::from_datasets(&empty_train, &empty_val, &empty_test);
        let reproducibility = reproducibility_metadata(&config.research, &dataset_validation, None);
        let evaluation = test_evaluation_metrics_for_research(&config.research);
        let coordinate_frame = CoordinateFrameProvenance::from_dataset_validation(&dataset_validation);
        UnseenPocketExperimentSummary {
            config,
            dataset_validation,
            coordinate_frame,
            split_report,
            reproducibility,
            training_history: Vec::new(),
            validation: evaluation.clone(),
            test: evaluation,
            ablation_matrix: None,
            performance_gates: PerformanceGateReport::default(),
        }
    }

    fn test_evaluation_metrics_for_research(research: &ResearchConfig) -> EvaluationMetrics {
        let mut comparison_summary = test_generation_quality_summary();
        comparison_summary.interaction_mode = interaction_mode_label(research.model.interaction_mode);
        comparison_summary.generation_mode = generation_mode_label(research);
        EvaluationMetrics {
            representation_diagnostics: RepresentationDiagnostics {
                finite_forward_fraction: 0.0,
                unique_complex_fraction: 0.0,
                unseen_protein_fraction: 0.0,
                distance_probe_rmse: 0.0,
                topology_pocket_cosine_alignment: 0.0,
                topology_reconstruction_mse: 0.0,
                slot_activation_mean: 0.0,
                slot_assignment_entropy_mean: 0.0,
                slot_activation_probability_mean: 0.0,
                attention_visible_slot_fraction: 0.0,
                gate_activation_mean: 0.0,
                leakage_proxy_mean: 0.0,
            },
            proxy_task_metrics: ProxyTaskMetrics {
                affinity_probe_mae: 0.0,
                affinity_probe_rmse: 0.0,
                labeled_fraction: 0.0,
                affinity_by_measurement: Vec::new(),
                probe_baselines: Vec::new(),
            },
            split_context: SplitContextMetrics {
                example_count: 0,
                unique_complex_count: 0,
                unique_protein_count: 0,
                train_reference_protein_count: 0,
                ligand_atom_count_bins: BTreeMap::new(),
                pocket_atom_count_bins: BTreeMap::new(),
                measurement_family_histogram: BTreeMap::new(),
            },
            resource_usage: ResourceUsageMetrics {
                memory_usage_mb: 0.0,
                evaluation_time_ms: 0.0,
                examples_per_second: 0.0,
                evaluation_batch_size: research.data.batch_size,
                forward_batch_count: 0,
                per_example_forward_count: 0,
                no_grad: true,
                batched_forward: true,
                de_novo_per_example_reason: None,
                average_ligand_atoms: 0.0,
                average_pocket_atoms: 0.0,
            },
            model_design: ModelDesignEvaluationMetrics::default(),
            real_generation_metrics: disabled_real_generation_metrics(),
            layered_generation_metrics: empty_layered_generation_metrics(),
            method_comparison: MethodComparisonSummary::default(),
            train_eval_alignment: TrainEvalAlignmentReport::default(),
            chemistry_collaboration: ChemistryCollaborationMetrics::default(),
            frozen_leakage_probe_calibration: FrozenLeakageProbeCalibrationReport::default(),
            comparison_summary,
            slot_stability: SlotStabilityMetrics::default(),
            strata: Vec::new(),
        }
    }

    fn test_claim_report() -> ClaimReport {
        ClaimReport {
            artifact_dir: "checkpoints/test".into(),
            run_label: "test".to_string(),
            raw_native_evidence: ClaimRawNativeEvidenceSummary::default(),
            processed_generation_evidence: ClaimProcessedGenerationEvidenceSummary::default(),
            postprocessing_repair_audit: RepairCaseAuditReport::default(),
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
            model_design: ModelDesignEvaluationMetrics::default(),
            layer_provenance_audit: ClaimLayerProvenanceAudit::default(),
            chemistry_novelty_diversity: ChemistryNoveltyDiversitySummary::default(),
            chemistry_collaboration: ChemistryCollaborationMetrics::default(),
            claim_context: ClaimContext::default(),
            backend_environment: None,
            ablation_deltas: Vec::new(),
            reranker_report: RerankerReport::default(),
            slot_stability: SlotStabilityMetrics::default(),
            leakage_calibration: LeakageCalibrationReport::default(),
            performance_gates: PerformanceGateReport::default(),
            baseline_comparisons: Vec::new(),
            method_comparison: MethodComparisonSummary::default(),
            train_eval_alignment: TrainEvalAlignmentReport::default(),
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
            bond_count: 0,
            valence_violation_count: 0,
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 6.0,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: "test".to_string(),
            generation_mode: "target_ligand_denoising".to_string(),
            generation_layer: "raw_flow".to_string(),
            generation_path_class: "model_native_raw".to_string(),
            model_native_raw: true,
            postprocessor_chain: Vec::new(),
            claim_boundary:
                "raw model-native output before repair, constraints, reranking, or backend scoring"
                    .to_string(),
            source_pocket_path: Some("pocket.pdb".to_string()),
            source_ligand_path: Some("ligand.sdf".to_string()),
        }
    }

    fn repaired_test_candidate(
        atom_types: Vec<i64>,
        coords: Vec<[f32; 3]>,
    ) -> GeneratedCandidateRecord {
        let mut candidate = test_candidate(atom_types, coords);
        candidate.source = "repaired".to_string();
        candidate.generation_layer = "repaired_candidates".to_string();
        candidate.generation_path_class = "postprocessed_repaired".to_string();
        candidate.model_native_raw = false;
        candidate.postprocessor_chain = vec!["geometry_repair".to_string()];
        candidate.claim_boundary =
            "geometry-repaired candidate; postprocessing evidence only".to_string();
        candidate
    }
}
