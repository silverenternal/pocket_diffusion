use pocket_diffusion::experiments::{
    CandidateLayerMetrics, ClaimReport, EvaluationMetrics, LayeredGenerationMetrics,
    TrainEvalAlignmentReport,
};
use pocket_diffusion::models::{
    extract_interaction_profiles, CandidateLayerKind, GeneratedCandidateRecord,
    PreferenceDatasetBuilder, PreferencePair, PreferencePairArtifact, PreferenceProfileArtifact,
    PreferenceReasonCode, RuleBasedPreferenceDatasetBuilder, INTERACTION_PROFILE_SCHEMA_VERSION,
    PREFERENCE_PAIR_SCHEMA_VERSION,
};
use pocket_diffusion::training::{
    PrimaryBranchComponentAudit, PrimaryBranchWeightRecord, ResumeContinuityMode, ResumeMode,
    SplitReport, TrainingRunSummary,
};

#[test]
fn artifact_compatibility_older_claim_artifact_defaults_recent_optional_sections() {
    let json = r#"{
      "artifact_dir": "./checkpoints/legacy",
      "run_label": "legacy_base",
      "validation": {
        "primary_objective": "conditioned_denoising",
        "variant_label": null,
        "interaction_mode": "lightweight",
        "candidate_valid_fraction": 0.5,
        "pocket_contact_fraction": 0.5,
        "pocket_compatibility_fraction": 0.5,
        "mean_centroid_offset": 1.0,
        "strict_pocket_fit_score": 0.2,
        "unique_smiles_fraction": 0.5,
        "unseen_protein_fraction": 1.0,
        "topology_specialization_score": 0.1,
        "geometry_specialization_score": 0.1,
        "pocket_specialization_score": 0.1,
        "slot_activation_mean": 0.2,
        "gate_activation_mean": 0.3,
        "leakage_proxy_mean": 0.4
      },
      "test": {
        "primary_objective": "conditioned_denoising",
        "variant_label": null,
        "interaction_mode": "lightweight",
        "candidate_valid_fraction": 0.5,
        "pocket_contact_fraction": 0.5,
        "pocket_compatibility_fraction": 0.5,
        "mean_centroid_offset": 1.0,
        "strict_pocket_fit_score": 0.2,
        "unique_smiles_fraction": 0.5,
        "unseen_protein_fraction": 1.0,
        "topology_specialization_score": 0.1,
        "geometry_specialization_score": 0.1,
        "pocket_specialization_score": 0.1,
        "slot_activation_mean": 0.2,
        "gate_activation_mean": 0.3,
        "leakage_proxy_mean": 0.4
      },
      "backend_metrics": {
        "chemistry_validity": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"},
        "docking_affinity": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"},
        "pocket_compatibility": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"}
      },
      "ablation_deltas": []
    }"#;

    let report: ClaimReport = serde_json::from_str(json).unwrap();
    assert_eq!(report.reranker_report.baseline.candidate_count, 0);
    assert_eq!(report.slot_stability.topology_activation_mean, 0.0);
    assert!(report.backend_thresholds.is_empty());
    assert_eq!(report.backend_review.reviewer_status, "");
    assert!(report.method_comparison.methods.is_empty());
    assert_eq!(report.validation.generation_mode, "target_ligand_denoising");
    assert_eq!(
        report.claim_context.generation_mode,
        "target_ligand_denoising"
    );
    assert!(!report.claim_context.de_novo_claim_allowed);
    assert_eq!(
        report.claim_context.target_alignment_policy,
        "pad_with_mask"
    );
    assert!(!report.claim_context.target_matching_claim_safe);
    assert!(!report.claim_context.full_molecular_flow_claim_allowed);
    assert_eq!(report.model_design.raw_model_layer, "raw_rollout");
    assert!(!report.layer_provenance_audit.claim_safe);
    assert_eq!(
        report.layer_provenance_audit.decision,
        "layer provenance audit unavailable"
    );
    assert_eq!(
        report.raw_native_evidence.evidence_role,
        "model_native_raw_first"
    );
    assert_eq!(report.raw_native_evidence.raw_model_layer, "unavailable");
    assert!(!report.raw_native_evidence.raw_native_gate.passed);
    assert_eq!(
        report.processed_generation_evidence.processed_layer,
        "unavailable"
    );
    assert!(report
        .postprocessing_repair_audit
        .claim_boundary
        .contains("do not cite repaired layers"));
    assert_eq!(
        report.train_eval_alignment.decision,
        TrainEvalAlignmentReport::default().decision
    );
}

#[test]
fn drug_level_claim_contract_declares_layered_backend_provenance() {
    let payload = std::fs::read_to_string("configs/drug_level_claim_contract.json").unwrap();
    let contract: pocket_diffusion::experiments::DrugLevelClaimContract =
        serde_json::from_str(&payload).unwrap();
    assert!(contract.schema_version >= 1);
    for layer in ["raw_flow", "constrained_flow", "repaired", "reranked"] {
        assert!(contract.layers.iter().any(|entry| entry == layer));
    }
    for field in [
        "backend_name",
        "backend_version_or_command",
        "input_count",
        "scored_count",
        "coverage_fraction",
        "failure_count",
        "status",
    ] {
        assert!(contract
            .required_backend_fields
            .iter()
            .any(|entry| entry == field));
    }
    assert!(contract
        .capability_groups
        .get("raw_model_capability")
        .unwrap()
        .contains(&"raw_flow".to_string()));
    assert!(contract
        .capability_groups
        .get("postprocessed_performance")
        .unwrap()
        .contains(&"reranked".to_string()));
    assert!(contract
        .metric_groups
        .get("binding")
        .unwrap()
        .required_metrics
        .contains(&"docking_score_coverage_fraction".to_string()));
}

#[test]
fn layered_generation_metrics_emit_canonical_and_legacy_layers() {
    let mut layers = LayeredGenerationMetrics::default();
    layers.generation_path_contract =
        pocket_diffusion::experiments::canonical_generation_path_contract();
    layers.raw_flow.candidate_count = 2;
    layers.raw_rollout = layers.raw_flow.clone();
    layers.constrained_flow.candidate_count = 3;
    layers.inferred_bond_candidates = layers.constrained_flow.clone();
    layers.repaired.candidate_count = 4;
    layers.repaired_candidates = layers.repaired.clone();
    layers.reranked_candidates.candidate_count = 1;

    let payload = serde_json::to_string(&layers).unwrap();
    assert!(payload.contains("\"raw_flow\""));
    assert!(payload.contains("\"constrained_flow\""));
    assert!(payload.contains("\"repaired\""));
    assert!(payload.contains("\"raw_rollout\""));
    assert!(payload.contains("\"repaired_candidates\""));
    assert!(payload.contains("\"inferred_bond_candidates\""));
    assert!(payload.contains("\"generation_path_contract\""));
    assert!(payload.contains("\"model_native_raw\""));
    assert!(payload.contains("\"claim_boundary\""));
    assert!(payload.contains("\"raw_molecular_flow_logits\""));
    assert!(payload.contains("\"raw_native_graph_extraction\""));
    assert!(payload.contains("\"constrained_native_graph\""));
    assert!(payload.contains("\"target_matching_artifact_fields\""));
    assert!(payload.contains("\"decoder_conditioning_kind\""));
    assert!(payload.contains("\"molecular_flow_conditioning_kind\""));
    assert!(payload.contains("\"slot_local_conditioning_enabled\""));
    assert!(payload.contains("\"mean_pooled_conditioning_ablation\""));
    assert!(payload.contains("target_matching_coverage"));
    assert!(layers
        .generation_path_contract
        .iter()
        .any(|row| row.legacy_field_name == "raw_flow"));
    assert!(layers
        .generation_path_contract
        .iter()
        .any(|row| row.legacy_field_name == "constrained_flow"));
    assert!(layers
        .generation_path_contract
        .iter()
        .any(|row| row.legacy_field_name == "deterministic_proxy_candidates"));
}

#[test]
fn branch_weight_record_serializes_matching_provenance_fields() {
    let record = PrimaryBranchWeightRecord {
        branch_name: "geometry".to_string(),
        unweighted_value: 2.0,
        effective_weight: 1.0,
        schedule_multiplier: 1.0,
        weighted_value: 2.0,
        optimizer_facing: true,
        provenance: "molecular_flow_contract_v1".to_string(),
        component_audit: PrimaryBranchComponentAudit::for_branch("geometry"),
        target_matching_policy: Some("hungarian_distance".to_string()),
        target_matching_mean_cost: Some(0.25),
        target_matching_max_cost: Some(0.5),
        target_matching_total_cost: Some(1.0),
        target_matching_coverage: Some(0.8),
        target_matching_matched_count: Some(4),
        target_matching_unmatched_generated_count: Some(1),
        target_matching_unmatched_target_count: Some(0),
        target_matching_exact_assignment: Some(true),
    };

    let value = serde_json::to_value(&record).unwrap();
    assert_eq!(value["unweighted_value"], 2.0);
    assert_eq!(value["schedule_multiplier"], 1.0);
    assert_eq!(value["weighted_value"], 2.0);
    assert_eq!(value["component_audit"]["branch_name"], "geometry");
    assert_eq!(value["target_matching_policy"], "hungarian_distance");
    assert_eq!(value["target_matching_coverage"], 0.8);
    assert_eq!(value["target_matching_matched_count"], 4);
    assert_eq!(value["target_matching_unmatched_generated_count"], 1);
    assert_eq!(value["target_matching_exact_assignment"], true);

    let legacy: PrimaryBranchWeightRecord = serde_json::from_value(serde_json::json!({
        "branch_name": "geometry",
        "effective_weight": 1.0,
        "optimizer_facing": true,
        "provenance": "molecular_flow_contract_v1"
    }))
    .unwrap();
    assert_eq!(legacy.unweighted_value, 0.0);
    assert_eq!(legacy.schedule_multiplier, 0.0);
    assert_eq!(legacy.weighted_value, 0.0);
    assert_eq!(legacy.component_audit.observed_component_count, 0);
    assert!(legacy.target_matching_policy.is_none());
    assert!(legacy.target_matching_coverage.is_none());
}

#[test]
fn candidate_layer_provenance_declares_generation_path_contract() {
    assert_eq!(
        CandidateLayerKind::RawRollout.canonical_generation_layer(),
        "raw_flow"
    );
    assert_eq!(
        CandidateLayerKind::Reranked.generation_path_class(),
        "reranked"
    );
    assert!(CandidateLayerKind::RawRollout.is_model_native_raw());
    assert!(!CandidateLayerKind::Repaired.is_model_native_raw());
    assert!(CandidateLayerKind::Repaired
        .claim_boundary()
        .contains("geometry-repaired"));
    let candidate = preference_candidate("layered", vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]);
    assert_eq!(candidate.generation_mode, "target_ligand_denoising");
    assert_eq!(candidate.generation_layer, "raw_flow");
    assert_eq!(candidate.generation_path_class, "model_native_raw");
    assert!(candidate.model_native_raw);
}

#[test]
fn reranker_report_exposes_layer_attribution_metrics() {
    let json = r#"{
      "baseline": {"candidate_count": 1, "valid_fraction": 0.8, "pocket_contact_fraction": 0.7, "mean_centroid_offset": 1.0, "clash_fraction": 0.2, "uniqueness_proxy_fraction": 0.5},
      "reranked": {"candidate_count": 1, "valid_fraction": 0.9, "pocket_contact_fraction": 0.8, "mean_centroid_offset": 0.8, "clash_fraction": 0.1, "uniqueness_proxy_fraction": 0.6},
      "deterministic_proxy": {"candidate_count": 1, "valid_fraction": 0.85, "pocket_contact_fraction": 0.75, "mean_centroid_offset": 0.9, "clash_fraction": 0.15, "uniqueness_proxy_fraction": 0.55},
      "calibration": {"method": "test", "coefficients": {}, "target_mean": 0.0, "fitted_candidate_count": 0},
      "raw_strict_pocket_fit": 0.2,
      "raw_docking_score": null,
      "raw_qed": null,
      "raw_sa": null,
      "repair_dependency_score": 0.3,
      "reranker_gain": {"valid_fraction": 0.1, "clash_reduction": 0.1},
      "flow_native_quality": 0.2,
      "layer_attribution_note": "test attribution",
      "decision": "test"
    }"#;
    let report: pocket_diffusion::experiments::RerankerReport = serde_json::from_str(json).unwrap();
    assert_eq!(report.raw_strict_pocket_fit, Some(0.2));
    assert_eq!(report.raw_docking_score, None);
    assert_eq!(report.repair_dependency_score, 0.3);
    assert!(report.reranker_gain.contains_key("valid_fraction"));
    assert!(report.layer_attribution_note.contains("attribution"));
}

#[test]
fn split_report_accepts_pre_stratification_schema() {
    let json = r#"{
      "train": {
        "example_count": 1,
        "unique_protein_count": 1,
        "labeled_example_count": 1,
        "labeled_fraction": 1.0,
        "dominant_measurement_histogram": {"Kd": 1}
      },
      "val": {
        "example_count": 0,
        "unique_protein_count": 0,
        "labeled_example_count": 0,
        "labeled_fraction": 0.0,
        "dominant_measurement_histogram": {}
      },
      "test": {
        "example_count": 0,
        "unique_protein_count": 0,
        "labeled_example_count": 0,
        "labeled_fraction": 0.0,
        "dominant_measurement_histogram": {}
      },
      "leakage_checks": {
        "protein_overlap_detected": false,
        "duplicate_example_ids_detected": false,
        "train_val_protein_overlap": 0,
        "train_test_protein_overlap": 0,
        "val_test_protein_overlap": 0,
        "duplicated_example_ids": 0
      }
    }"#;

    let report: SplitReport = serde_json::from_str(json).unwrap();
    assert!(report.train.ligand_atom_count_bins.is_empty());
}

#[test]
fn evaluation_metrics_accepts_pre_strata_resource_schema() {
    let json = r#"{
      "representation_diagnostics": {
        "finite_forward_fraction": 1.0,
        "unique_complex_fraction": 1.0,
        "unseen_protein_fraction": 1.0,
        "distance_probe_rmse": 0.0,
        "topology_pocket_cosine_alignment": 0.0,
        "topology_reconstruction_mse": 0.0,
        "slot_activation_mean": 0.0,
        "gate_activation_mean": 0.0,
        "leakage_proxy_mean": 0.0
      },
      "proxy_task_metrics": {
        "affinity_probe_mae": 0.0,
        "affinity_probe_rmse": 0.0,
        "labeled_fraction": 0.0,
        "affinity_by_measurement": []
      },
      "split_context": {
        "example_count": 0,
        "unique_complex_count": 0,
        "unique_protein_count": 0,
        "train_reference_protein_count": 0
      },
      "resource_usage": {
        "memory_usage_mb": 0.0,
        "evaluation_time_ms": 0.0
      },
      "real_generation_metrics": {
        "chemistry_validity": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"},
        "docking_affinity": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"},
        "pocket_compatibility": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"}
      },
      "layered_generation_metrics": {
        "raw_rollout": {"candidate_count": 0, "valid_fraction": 0.0, "pocket_contact_fraction": 0.0, "mean_centroid_offset": 0.0, "clash_fraction": 0.0, "uniqueness_proxy_fraction": 0.0},
        "repaired_candidates": {"candidate_count": 0, "valid_fraction": 0.0, "pocket_contact_fraction": 0.0, "mean_centroid_offset": 0.0, "clash_fraction": 0.0, "uniqueness_proxy_fraction": 0.0},
        "inferred_bond_candidates": {"candidate_count": 0, "valid_fraction": 0.0, "pocket_contact_fraction": 0.0, "mean_centroid_offset": 0.0, "clash_fraction": 0.0, "uniqueness_proxy_fraction": 0.0},
        "backend_scored_candidates": {}
      },
      "comparison_summary": {
        "primary_objective": "conditioned_denoising",
        "variant_label": null,
        "interaction_mode": "lightweight",
        "candidate_valid_fraction": null,
        "pocket_contact_fraction": null,
        "pocket_compatibility_fraction": null,
        "mean_centroid_offset": null,
        "strict_pocket_fit_score": null,
        "unique_smiles_fraction": null,
        "unseen_protein_fraction": 0.0,
        "topology_specialization_score": 0.0,
        "geometry_specialization_score": 0.0,
        "pocket_specialization_score": 0.0,
        "slot_activation_mean": 0.0,
        "gate_activation_mean": 0.0,
        "leakage_proxy_mean": 0.0
      }
    }"#;

    let metrics: EvaluationMetrics = serde_json::from_str(json).unwrap();
    assert!(metrics.strata.is_empty());
    assert_eq!(metrics.model_design.raw_model_layer, "raw_rollout");
    assert_eq!(metrics.model_design.processed_layer, "unavailable");
    assert_eq!(CandidateLayerMetrics::default().candidate_count, 0);
    assert!(metrics.method_comparison.methods.is_empty());
    assert!(
        metrics
            .method_comparison
            .preference_alignment
            .missing_artifacts_mean_unavailable
    );
}

#[test]
fn q7_claim_boundary_contract_keeps_evidence_tiers_explicit() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let contract_path = root.join("configs/q7_claim_boundary_contract.json");
    let contract: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(contract_path).unwrap()).unwrap();
    let tiers = contract["evidence_tiers"].as_array().unwrap();
    for required in [
        "synthetic_smoke",
        "real_data_debug",
        "reviewer_scale_unseen_pocket",
        "heuristic_metric",
        "backend_supported",
        "docking_supported",
        "experimental",
    ] {
        assert!(
            tiers.iter().any(|tier| tier["tier"] == required),
            "missing evidence tier {required}"
        );
    }
    assert!(contract["q7_required_model_design_fields"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field == "test.model_design.raw_model_valid_fraction"));
    assert!(contract["q7_required_ablation_variants"]
        .as_array()
        .unwrap()
        .iter()
        .any(|variant| variant == "decoder_conditioning_mean_pooled"));
    assert!(contract["raw_vs_processed_rule"]
        .as_str()
        .unwrap()
        .contains("postprocessing chain"));
    assert!(contract["candidate_record_layer_provenance_fields"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field == "generation_layer"));
    assert!(contract["claim_report_layer_provenance_fields"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field == "claim_report.layer_provenance_audit.claim_safe"));

    let doc = std::fs::read_to_string(root.join("docs/q7_claim_boundaries.md")).unwrap();
    assert!(doc.contains("experimental binding affinity"));
    assert!(doc.contains("model_design.raw_model_*"));
    assert!(doc.contains("layer_provenance_audit"));
}

#[test]
fn preference_profile_and_pair_schema_round_trip() {
    let candidate = preference_candidate("good", vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]);
    let profiles =
        extract_interaction_profiles(&[candidate], CandidateLayerKind::Reranked, None, &[]);
    assert_eq!(
        profiles[0].schema_version,
        INTERACTION_PROFILE_SCHEMA_VERSION
    );
    let json = serde_json::to_string(&profiles[0]).unwrap();
    let restored: pocket_diffusion::models::InteractionProfile =
        serde_json::from_str(&json).unwrap();
    assert_eq!(restored.schema_version, INTERACTION_PROFILE_SCHEMA_VERSION);
}

#[test]
fn preference_pair_artifact_schema_preserves_reasons() {
    let good = extract_interaction_profiles(
        &[preference_candidate(
            "same",
            vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        )],
        CandidateLayerKind::Reranked,
        None,
        &[],
    )
    .remove(0);
    let bad = extract_interaction_profiles(
        &[preference_candidate(
            "same",
            vec![[6.0, 0.0, 0.0], [6.1, 0.0, 0.0]],
        )],
        CandidateLayerKind::Reranked,
        None,
        &[],
    )
    .remove(0);
    let pairs = RuleBasedPreferenceDatasetBuilder::default().build_pairs(&[good, bad]);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].schema_version, PREFERENCE_PAIR_SCHEMA_VERSION);
    assert!(pairs[0]
        .preference_reason
        .contains(&PreferenceReasonCode::BetterStrictPocketFit));
    let json = serde_json::to_string(&pairs[0]).unwrap();
    let restored: PreferencePair = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.schema_version, PREFERENCE_PAIR_SCHEMA_VERSION);
    assert!(restored
        .feature_deltas
        .contains_key("strict_pocket_fit_score"));
}

#[test]
fn preference_artifact_envelopes_round_trip_with_empty_records() {
    let profiles = PreferenceProfileArtifact::new("validation", Vec::new());
    assert_eq!(profiles.schema_version, INTERACTION_PROFILE_SCHEMA_VERSION);
    assert_eq!(profiles.profile_count, 0);
    assert!(profiles.records.is_empty());
    let restored_profiles: PreferenceProfileArtifact =
        serde_json::from_str(&serde_json::to_string(&profiles).unwrap()).unwrap();
    assert_eq!(restored_profiles.split, "validation");

    let pairs = PreferencePairArtifact::new("test", &[], Vec::new());
    assert_eq!(pairs.schema_version, PREFERENCE_PAIR_SCHEMA_VERSION);
    assert_eq!(pairs.pair_count, 0);
    assert_eq!(pairs.backend_supported_pair_fraction, 0.0);
    assert_eq!(pairs.rule_only_pair_fraction, 0.0);
    assert_eq!(pairs.missing_backend_evidence_fraction, 0.0);
    assert_eq!(pairs.mean_preference_strength, 0.0);
    assert_eq!(pairs.hard_constraint_win_fraction, 0.0);
    assert!(pairs.records.is_empty());
    let restored_pairs: PreferencePairArtifact =
        serde_json::from_str(&serde_json::to_string(&pairs).unwrap()).unwrap();
    assert_eq!(restored_pairs.split, "test");
}

#[test]
fn training_summary_defaults_new_resume_continuity_fields() {
    let split_report: SplitReport = serde_json::from_str(
        r#"{
          "train": {
            "example_count": 1,
            "unique_protein_count": 1,
            "labeled_example_count": 1,
            "labeled_fraction": 1.0,
            "dominant_measurement_histogram": {"Kd": 1}
          },
          "val": {
            "example_count": 0,
            "unique_protein_count": 0,
            "labeled_example_count": 0,
            "labeled_fraction": 0.0,
            "dominant_measurement_histogram": {}
          },
          "test": {
            "example_count": 0,
            "unique_protein_count": 0,
            "labeled_example_count": 0,
            "labeled_fraction": 0.0,
            "dominant_measurement_histogram": {}
          },
          "leakage_checks": {
            "protein_overlap_detected": false,
            "duplicate_example_ids_detected": false,
            "train_val_protein_overlap": 0,
            "train_test_protein_overlap": 0,
            "val_test_protein_overlap": 0,
            "duplicated_example_ids": 0
          }
        }"#,
    )
    .unwrap();
    let evaluation: EvaluationMetrics = serde_json::from_str(
        r#"{
          "representation_diagnostics": {
            "finite_forward_fraction": 1.0,
            "unique_complex_fraction": 1.0,
            "unseen_protein_fraction": 1.0,
            "distance_probe_rmse": 0.0,
            "topology_pocket_cosine_alignment": 0.0,
            "topology_reconstruction_mse": 0.0,
            "slot_activation_mean": 0.0,
            "gate_activation_mean": 0.0,
            "leakage_proxy_mean": 0.0
          },
          "proxy_task_metrics": {
            "affinity_probe_mae": 0.0,
            "affinity_probe_rmse": 0.0,
            "labeled_fraction": 0.0,
            "affinity_by_measurement": []
          },
          "split_context": {
            "example_count": 0,
            "unique_complex_count": 0,
            "unique_protein_count": 0,
            "train_reference_protein_count": 0
          },
          "resource_usage": {
            "memory_usage_mb": 0.0,
            "evaluation_time_ms": 0.0
          },
          "real_generation_metrics": {
            "chemistry_validity": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"},
            "docking_affinity": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"},
            "pocket_compatibility": {"available": false, "backend_name": null, "metrics": {}, "status": "legacy"}
          },
          "layered_generation_metrics": {
            "raw_rollout": {"candidate_count": 0, "valid_fraction": 0.0, "pocket_contact_fraction": 0.0, "mean_centroid_offset": 0.0, "clash_fraction": 0.0, "uniqueness_proxy_fraction": 0.0},
            "repaired_candidates": {"candidate_count": 0, "valid_fraction": 0.0, "pocket_contact_fraction": 0.0, "mean_centroid_offset": 0.0, "clash_fraction": 0.0, "uniqueness_proxy_fraction": 0.0},
            "inferred_bond_candidates": {"candidate_count": 0, "valid_fraction": 0.0, "pocket_contact_fraction": 0.0, "mean_centroid_offset": 0.0, "clash_fraction": 0.0, "uniqueness_proxy_fraction": 0.0},
            "backend_scored_candidates": {}
          },
          "comparison_summary": {
            "primary_objective": "conditioned_denoising",
            "variant_label": null,
            "interaction_mode": "lightweight",
            "candidate_valid_fraction": null,
            "pocket_contact_fraction": null,
            "pocket_compatibility_fraction": null,
            "mean_centroid_offset": null,
            "strict_pocket_fit_score": null,
            "unique_smiles_fraction": null,
            "unseen_protein_fraction": 0.0,
            "topology_specialization_score": 0.0,
            "geometry_specialization_score": 0.0,
            "pocket_specialization_score": 0.0,
            "slot_activation_mean": 0.0,
            "gate_activation_mean": 0.0,
            "leakage_proxy_mean": 0.0
          }
        }"#,
    )
    .unwrap();
    let mut json = serde_json::to_value(TrainingRunSummary {
        config: pocket_diffusion::config::ResearchConfig::default(),
        dataset_validation: pocket_diffusion::data::DatasetValidationReport::default(),
        coordinate_frame: pocket_diffusion::training::CoordinateFrameProvenance::default(),
        splits: pocket_diffusion::training::DatasetSplitSizes {
            total: 1,
            train: 1,
            val: 0,
            test: 0,
        },
        split_report,
        resumed_from_step: None,
        reproducibility: pocket_diffusion::training::ReproducibilityMetadata {
            config_hash: "cfg".to_string(),
            dataset_validation_fingerprint: "data".to_string(),
            metric_schema_version: 1,
            artifact_bundle_schema_version: 1,
            determinism_controls: pocket_diffusion::training::DeterminismControls::default(),
            replay_tolerance: pocket_diffusion::training::ReplayTolerance::default(),
            resume_contract: pocket_diffusion::training::ResumeContract {
                version: "weights+history+step".to_string(),
                restores_model_weights: true,
                restores_step: true,
                restores_history: true,
                restores_optimizer_state: false,
                resume_mode: ResumeMode::WeightsOnlyResume,
                continuity_mode: ResumeContinuityMode::MetadataOnlyContinuation,
                supports_strict_replay: false,
                notes: "legacy summary without continuity mode".to_string(),
            },
            resume_provenance: pocket_diffusion::training::ResumeProvenance {
                resumed: false,
                resumed_from_step: None,
                checkpoint_config_hash: None,
                checkpoint_dataset_fingerprint: None,
                restored_optimizer_state_metadata: false,
                restored_scheduler_state_metadata: false,
                resume_mode: ResumeMode::FreshRun,
                continuity_mode: ResumeContinuityMode::FreshRun,
                strict_replay_achieved: false,
            },
        },
        training_history: Vec::new(),
        objective_coverage: pocket_diffusion::training::ObjectiveCoverageReport::default(),
        validation_history: Vec::new(),
        best_checkpoint: None,
        early_stopping: pocket_diffusion::training::EarlyStoppingSummary::default(),
        validation: evaluation.clone(),
        test: evaluation,
    })
    .unwrap();
    json["reproducibility"]["resume_contract"]
        .as_object_mut()
        .unwrap()
        .remove("continuity_mode");
    json.as_object_mut().unwrap().remove("validation_history");
    json.as_object_mut().unwrap().remove("best_checkpoint");
    json.as_object_mut().unwrap().remove("early_stopping");
    json.as_object_mut().unwrap().remove("objective_coverage");
    json.as_object_mut().unwrap().remove("coordinate_frame");
    json["reproducibility"]["resume_contract"]
        .as_object_mut()
        .unwrap()
        .remove("resume_mode");
    json["reproducibility"]
        .as_object_mut()
        .unwrap()
        .remove("determinism_controls");
    json["reproducibility"]
        .as_object_mut()
        .unwrap()
        .remove("replay_tolerance");
    json["reproducibility"]["resume_contract"]
        .as_object_mut()
        .unwrap()
        .remove("supports_strict_replay");
    json["reproducibility"]["resume_provenance"]
        .as_object_mut()
        .unwrap()
        .remove("restored_optimizer_state_metadata");
    json["reproducibility"]["resume_provenance"]
        .as_object_mut()
        .unwrap()
        .remove("restored_scheduler_state_metadata");
    json["reproducibility"]["resume_provenance"]
        .as_object_mut()
        .unwrap()
        .remove("continuity_mode");
    json["reproducibility"]["resume_provenance"]
        .as_object_mut()
        .unwrap()
        .remove("resume_mode");
    json["reproducibility"]["resume_provenance"]
        .as_object_mut()
        .unwrap()
        .remove("strict_replay_achieved");
    let json = serde_json::to_string(&json).unwrap();

    let summary: TrainingRunSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(
        summary.reproducibility.resume_contract.continuity_mode,
        ResumeContinuityMode::FreshRun
    );
    assert!(
        !summary
            .reproducibility
            .resume_contract
            .supports_strict_replay
    );
    assert_eq!(
        summary.reproducibility.resume_contract.resume_mode,
        ResumeMode::FreshRun
    );
    assert_eq!(
        summary.reproducibility.resume_provenance.continuity_mode,
        ResumeContinuityMode::FreshRun
    );
    assert_eq!(
        summary.reproducibility.resume_provenance.resume_mode,
        ResumeMode::FreshRun
    );
    assert!(
        !summary
            .reproducibility
            .resume_provenance
            .strict_replay_achieved
    );
    assert!(summary.validation_history.is_empty());
    assert!(summary.best_checkpoint.is_none());
    assert!(!summary.early_stopping.enabled);
    assert!(summary.objective_coverage.records.is_empty());
    assert_eq!(
        summary.coordinate_frame.rotation_consistency_role,
        "diagnostic_not_exact_equivariance_claim"
    );
}

fn preference_candidate(example_id: &str, coords: Vec<[f32; 3]>) -> GeneratedCandidateRecord {
    GeneratedCandidateRecord {
        example_id: example_id.to_string(),
        protein_id: "protein".to_string(),
        molecular_representation: None,
        atom_types: vec![6; coords.len()],
        coords,
        inferred_bonds: vec![(0, 1)],
        bond_count: 1,
        valence_violation_count: 0,
        pocket_centroid: [0.0, 0.0, 0.0],
        pocket_radius: 3.0,
        coordinate_frame_origin: [0.0, 0.0, 0.0],
        source: "test".to_string(),
        generation_mode: "target_ligand_denoising".to_string(),
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
