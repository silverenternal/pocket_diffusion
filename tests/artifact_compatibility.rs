use pocket_diffusion::experiments::{CandidateLayerMetrics, ClaimReport, EvaluationMetrics};
use pocket_diffusion::training::{ResumeContinuityMode, SplitReport, TrainingRunSummary};

#[test]
fn older_claim_artifact_defaults_recent_optional_sections() {
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
    assert_eq!(CandidateLayerMetrics::default().candidate_count, 0);
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
                restores_optimizer_state: true,
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
                continuity_mode: ResumeContinuityMode::FreshRun,
                strict_replay_achieved: false,
            },
        },
        training_history: Vec::new(),
        validation: evaluation.clone(),
        test: evaluation,
    })
    .unwrap();
    json["reproducibility"]["resume_contract"]
        .as_object_mut()
        .unwrap()
        .remove("continuity_mode");
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
        summary.reproducibility.resume_provenance.continuity_mode,
        ResumeContinuityMode::FreshRun
    );
    assert!(
        !summary
            .reproducibility
            .resume_provenance
            .strict_replay_achieved
    );
}
