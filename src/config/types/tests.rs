#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_research_config_from_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        fs::write(
            &path,
            r#"{
                "data": {
                    "root_dir": "./data",
                    "dataset_format": "synthetic",
                    "manifest_path": null,
                    "label_table_path": null,
                    "max_ligand_atoms": 64,
                    "max_pocket_atoms": 256,
                    "pocket_cutoff_angstrom": 6.0,
                    "max_examples": 2,
                    "batch_size": 3,
                    "split_seed": 7,
                    "val_fraction": 0.2,
                    "test_fraction": 0.2,
                    "stratify_by_measurement": false
                },
                "model": {
                    "hidden_dim": 32,
                    "num_slots": 4,
                    "atom_vocab_size": 16,
                    "bond_vocab_size": 4,
                    "pocket_feature_dim": 12,
                    "pair_feature_dim": 8
                },
                "training": {
                    "learning_rate": 0.001,
                    "max_steps": 5,
                    "schedule": {
                        "stage1_steps": 1,
                        "stage2_steps": 2,
                        "stage3_steps": 3
                    },
                    "loss_weights": {
                        "alpha_task": 1.0,
                        "beta_intra_red": 0.1,
                        "gamma_probe": 0.2,
                        "delta_leak": 0.05,
                        "eta_gate": 0.05,
                        "mu_slot": 0.05,
                        "nu_consistency": 0.1
                    },
                    "checkpoint_dir": "./checkpoints",
                    "checkpoint_every": 2,
                    "log_every": 1,
                    "affinity_weighting": "none"
                },
                "runtime": {
                    "device": "cpu",
                    "data_workers": 0
                }
            }"#,
        )
        .unwrap();

        let config = load_research_config(&path).unwrap();
        assert_eq!(config.data.batch_size, 3);
        assert!(!config.training.data_order.shuffle);
        assert_eq!(config.training.data_order.sampler_seed, 0);
        assert!(!config.training.data_order.drop_last);
        assert_eq!(config.training.data_order.max_epochs, None);
        assert_eq!(config.model.pocket_feature_dim, 12);
        assert_eq!(config.runtime.device, "cpu");
        assert_eq!(
            config.training.primary_objective,
            PrimaryObjectiveConfig::ConditionedDenoising
        );
        assert_eq!(
            config.model.geometry_encoder.operator,
            GeometryOperatorKind::PairDistanceKernel
        );
        assert_eq!(
            config.model.pocket_encoder.kind,
            PocketEncoderKind::LocalMessagePassing
        );
        assert_eq!(config.model.slot_decomposition.activation_threshold, 0.5);
    }

    #[test]
    fn default_training_objective_is_decoder_anchored() {
        assert_eq!(
            TrainingConfig::default().primary_objective,
            PrimaryObjectiveConfig::ConditionedDenoising
        );
        assert_eq!(TrainingConfig::default().gradient_clipping.global_norm, None);
        assert_eq!(TrainingConfig::default().best_metric, "auto");
        assert!(TrainingConfig::default().build_rollout_diagnostics);
    }

    #[test]
    fn gradient_clipping_config_validates_optional_global_norm() {
        let mut config = ResearchConfig::default();
        config.training.gradient_clipping.global_norm = Some(5.0);
        assert!(config.validate().is_ok());

        config.training.gradient_clipping.global_norm = Some(0.0);
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("training.gradient_clipping.global_norm must be omitted or finite and positive"));
    }

    #[test]
    fn validation_checkpoint_config_validates_best_metric_and_patience() {
        let mut config = ResearchConfig::default();
        config.training.validation_every = 2;
        config.training.best_metric = "auto".to_string();
        assert!(config.validate().is_ok());
        assert_eq!(config.resolved_best_metric(), "distance_probe_rmse");

        config.training.best_metric = "validation.finite_forward_fraction".to_string();
        config.training.early_stopping_patience = Some(2);
        assert!(config.validate().is_ok());
        assert_eq!(config.resolved_best_metric(), "finite_forward_fraction");

        config.training.best_metric = "unknown_metric".to_string();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("training.best_metric"));

        config.training.best_metric = "finite_forward_fraction".to_string();
        config.training.validation_every = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("training.validation_every"));
    }

    #[test]
    fn auto_best_metric_resolves_to_quality_metric_for_claim_bearing_flow() {
        let mut config = ResearchConfig::default();
        config.training.best_metric = "auto".to_string();
        config.training.primary_objective = PrimaryObjectiveConfig::FlowMatching;
        config.data.generation_target.generation_mode = GenerationModeConfig::DeNovoInitialization;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.geometry_only = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = FlowBranchKind::ALL.to_vec();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy = FlowTargetAlignmentPolicy::HungarianDistance;

        assert!(config.validate().is_ok());
        assert_eq!(config.resolved_best_metric(), "strict_pocket_fit_score");
        assert_ne!(config.resolved_best_metric(), "finite_forward_fraction");
    }

    #[test]
    fn auto_best_metric_resolves_for_pocket_baseline_profile() {
        let mut config = ResearchConfig::default();
        config.training.best_metric = "auto".to_string();
        config.data.generation_target.generation_mode =
            GenerationModeConfig::PocketOnlyInitializationBaseline;
        config.training.primary_objective = PrimaryObjectiveConfig::SurrogateReconstruction;

        assert!(config.validate().is_ok());
        assert_eq!(config.resolved_best_metric(), "candidate_valid_fraction");
    }

    #[test]
    fn data_quality_filters_validate_claim_split_thresholds() {
        let mut config = ResearchConfig::default();
        config.data.quality_filters.min_test_protein_families = Some(2);
        config.data.quality_filters.min_validation_pocket_families = Some(2);
        config.data.quality_filters.min_test_ligand_scaffolds = Some(2);
        config.data.quality_filters.min_validation_measurement_families = Some(1);
        config.data.quality_filters.reject_target_ligand_context_leakage = true;
        assert!(config.validate().is_ok());

        config.data.quality_filters.min_validation_pocket_families = Some(0);
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("data.quality_filters.min_validation_pocket_families"));
    }

    #[test]
    fn validate_rejects_unimplemented_trainable_rollout_loss_switch() {
        let mut config = ResearchConfig::default();
        config.training.enable_trainable_rollout_loss = true;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("training.enable_trainable_rollout_loss is not implemented"));
        assert!(err.to_string().contains("rollout_eval"));
    }

    #[test]
    fn generation_target_accepts_legacy_rollout_weight_decay_alias() {
        let config: GenerationTargetConfig = serde_json::from_str(
            r#"{
                "atom_mask_ratio": 0.15,
                "coordinate_noise_std": 0.08,
                "corruption_seed": 1337,
                "rollout_steps": 4,
                "min_rollout_steps": 2,
                "stop_probability_threshold": 0.82,
                "coordinate_step_scale": 0.8,
                "training_step_weight_decay": 0.77
            }"#,
        )
        .unwrap();

        assert_eq!(config.rollout_eval_step_weight_decay, 0.77);
    }

    #[test]
    fn validate_rejects_invalid_split_fractions() {
        let mut config = ResearchConfig::default();
        config.data.val_fraction = 0.6;
        config.data.test_fraction = 0.4;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("data.val_fraction + data.test_fraction must be < 1.0"));
    }

    #[test]
    fn validate_rejects_manifest_mode_without_manifest_path() {
        let mut config = ResearchConfig::default();
        config.data.dataset_format = DatasetFormat::ManifestJson;
        config.training.max_steps = 8;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("data.manifest_path is required when data.dataset_format=manifest_json"));
    }

    #[test]
    fn validate_rejects_zero_training_data_order_max_epochs() {
        let mut config = ResearchConfig::default();
        config.training.data_order.max_epochs = Some(0);

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("training.data_order.max_epochs must be omitted or greater than zero"));
    }

    #[test]
    fn generation_backend_config_validates_external_wrapper_scope() {
        let mut config = ResearchConfig::default();
        config.generation_method.primary_backend.external_wrapper.enabled = true;
        config.generation_method.primary_backend.external_wrapper.executable =
            Some("external-generator".to_string());

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("external_wrapper may only be enabled for family=external_wrapper"));
    }

    #[test]
    fn generation_mode_compatibility_contract_covers_every_variant() {
        for mode in GenerationModeConfig::ALL {
            let contract = mode.compatibility_contract();
            assert_eq!(contract.generation_mode, mode);
            assert!(!contract.claim_label.is_empty());
            assert!(!contract.atom_count_source.is_empty());
            assert!(!contract.topology_source.is_empty());
            assert!(!contract.geometry_source.is_empty());
            assert_eq!(
                contract.pocket_context_availability,
                "inference_available"
            );
            assert_eq!(
                contract.target_ligand_atom_type_availability,
                "target_supervision_only"
            );
            assert_eq!(
                contract.target_ligand_topology_availability,
                "target_supervision_only"
            );
            assert_eq!(
                contract.target_ligand_coordinate_availability,
                "target_supervision_only"
            );
            assert_eq!(
                contract.postprocessing_layer_availability,
                "postprocessing_only"
            );
            assert!(!contract.decoder_capability_label.is_empty());
        }

        let target = GenerationModeConfig::TargetLigandDenoising.compatibility_contract();
        assert!(target.target_ligand_topology);
        assert!(target.target_ligand_geometry);
        assert!(target.fixed_atom_count);
        assert!(target.supports_primary_objective(PrimaryObjectiveConfig::ConditionedDenoising));

        let pocket =
            GenerationModeConfig::PocketOnlyInitializationBaseline.compatibility_contract();
        assert!(!pocket.target_ligand_topology);
        assert!(!pocket.target_ligand_geometry);
        assert!(pocket.pocket_only_initialization);
        assert!(pocket.supports_primary_objective(PrimaryObjectiveConfig::SurrogateReconstruction));
        assert!(!pocket.supports_primary_objective(PrimaryObjectiveConfig::ConditionedDenoising));

        let de_novo = GenerationModeConfig::DeNovoInitialization.compatibility_contract();
        assert!(de_novo.supported);
        assert!(de_novo.graph_growth);
        assert!(de_novo.supports_backend_family(GenerationBackendFamilyConfig::FlowMatching));
        assert!(de_novo.supports_primary_objective(PrimaryObjectiveConfig::FlowMatching));
    }

    #[test]
    fn validate_rejects_incompatible_generation_mode_objective_pairs() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.generation_mode =
            GenerationModeConfig::PocketOnlyInitializationBaseline;

        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("training.primary_objective=conditioned_denoising"));
        assert!(err.contains("pocket_only_initialization_baseline"));
        assert!(err.contains("surrogate_reconstruction"));

        config.training.primary_objective = PrimaryObjectiveConfig::SurrogateReconstruction;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn flow_backend_resolves_generation_mode_to_flow_refinement_contract() {
        let mut config = ResearchConfig::default();
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..GenerationBackendConfig::default()
        };

        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("training.primary_objective=conditioned_denoising"));
        assert!(err.contains("generation_mode=flow_refinement"));

        config.training.primary_objective = PrimaryObjectiveConfig::FlowMatching;
        assert!(config.validate().is_ok());
        assert_eq!(
            config
                .data
                .generation_target
                .generation_mode
                .resolved_for_backend(config.generation_method.resolved_primary_backend_family()),
            GenerationModeConfig::FlowRefinement
        );
    }

    #[test]
    fn temporal_interaction_policy_validation_rejects_unknown_path() {
        let mut config = ResearchConfig::default();
        config.model.temporal_interaction_policy.stage_multipliers = vec![
            InteractionPathStageMultiplier {
                training_stage: 0,
                path: "unknown_path".to_string(),
                multiplier: 0.8,
            },
        ];

        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains(
                "model.temporal_interaction_policy.stage_multipliers contains unknown path 'unknown_path'"
            )
        );
    }

    #[test]
    fn interaction_gate_path_weights_validate_path_and_scale() {
        let mut config = ResearchConfig::default();
        assert_eq!(
            config.model.interaction_tuning.gate_mode,
            InteractionGateMode::PathScalar
        );
        config.model.interaction_tuning.gate_mode = InteractionGateMode::TargetSlot;
        assert!(config.validate().is_ok());

        config
            .model
            .interaction_tuning
            .gate_regularization_path_weights = vec![InteractionPathGateRegularizationWeight {
            path: "geo_from_pocket".to_string(),
            weight: 0.0,
        }];
        assert!(config.validate().is_ok());

        config
            .model
            .interaction_tuning
            .gate_regularization_path_weights = vec![InteractionPathGateRegularizationWeight {
            path: "unknown_path".to_string(),
            weight: 1.0,
        }];
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains(
            "model.interaction_tuning.gate_regularization_path_weights contains unknown path"
        ));

        config
            .model
            .interaction_tuning
            .gate_regularization_path_weights = vec![InteractionPathGateRegularizationWeight {
            path: "geo_from_pocket".to_string(),
            weight: -1.0,
        }];
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains(
            "model.interaction_tuning.gate_regularization_path_weights requires non-negative finite weights"
        ));
    }

    #[test]
    fn adaptive_stage_guard_defaults_off_and_validates_thresholds() {
        let mut config = ResearchConfig::default();
        assert!(!config.training.adaptive_stage_guard.enabled);
        assert!(config.validate().is_ok());

        config
            .training
            .adaptive_stage_guard
            .enabled = true;
        config
            .training
            .adaptive_stage_guard
            .readiness_window = 0;
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("training.adaptive_stage_guard.readiness_window"));

        config
            .training
            .adaptive_stage_guard
            .readiness_window = 2;
        config
            .training
            .adaptive_stage_guard
            .max_gate_saturation_fraction = 1.5;
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("training.adaptive_stage_guard.max_gate_saturation_fraction"));

        config
            .training
            .adaptive_stage_guard
            .max_gate_saturation_fraction = 1.0;
        config
            .training
            .adaptive_stage_guard
            .min_slot_signature_matching_score = Some(-0.1);
        let err = config.validate().unwrap_err().to_string();
        assert!(err
            .contains("training.adaptive_stage_guard.min_slot_signature_matching_score"));

        config
            .training
            .adaptive_stage_guard
            .min_slot_signature_matching_score = Some(0.5);
        config
            .training
            .adaptive_stage_guard
            .max_leakage_diagnostic = Some(f64::NAN);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("training.adaptive_stage_guard.max_leakage_diagnostic"));
    }

    #[test]
    fn preference_alignment_defaults_off_and_validates_dependencies() {
        let mut config = ResearchConfig::default();
        assert!(!config.preference_alignment.enable_profile_extraction);
        assert!(config
            .preference_alignment
            .missing_artifacts_mean_unavailable);

        config.preference_alignment.enable_pair_construction = true;
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("enable_pair_construction requires enable_profile_extraction"));
    }

    #[test]
    fn flow_matching_non_geometry_only_requires_full_branch_set() {
        let mut config = ResearchConfig::default();
        config.generation_method.flow_matching.geometry_only = false;
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("geometry_only=false requires geometry, atom_type, bond, topology, and pocket_context"));
    }

    #[test]
    fn flow_matching_multimodal_defaults_to_geometry_branch_only() {
        let config = ResearchConfig::default();

        assert_eq!(
            config
                .generation_method
                .flow_matching
                .multi_modal
                .enabled_branches,
            vec![FlowBranchKind::Geometry]
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn flow_matching_multimodal_accepts_non_geometry_branches() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = vec![FlowBranchKind::Geometry, FlowBranchKind::AtomType];

        assert!(config.validate().is_ok());
    }

    #[test]
    fn flow_matching_multimodal_allows_full_molecular_flow_claims_with_all_branches() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = vec![
            FlowBranchKind::Geometry,
            FlowBranchKind::AtomType,
            FlowBranchKind::Bond,
            FlowBranchKind::Topology,
            FlowBranchKind::PocketContext,
        ];
        config
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy = FlowTargetAlignmentPolicy::HungarianDistance;

        assert!(config.validate().is_ok());
    }

    #[test]
    fn flow_matching_multimodal_rejects_invalid_branch_weights() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_loss_weights
            .bond = -0.1;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("branch_loss_weights.bond must be finite and non-negative"));
    }

    #[test]
    fn flow_matching_branch_schedule_warmup_is_independent_per_branch() {
        let mut schedule = FlowBranchScheduleConfig::default();
        schedule.atom_type.start_step = 2;
        schedule.atom_type.warmup_steps = 4;
        schedule.bond.enabled = false;
        let weights = FlowBranchLossWeights {
            geometry: 2.0,
            atom_type: 3.0,
            bond: 5.0,
            topology: 7.0,
            pocket_context: 11.0,
            synchronization: 13.0,
        };

        let step0 = schedule.effective_weights(&weights, Some(0));
        let step3 = schedule.effective_weights(&weights, Some(3));
        let final_weights = schedule.effective_weights(&weights, None);

        assert_eq!(step0.geometry, 2.0);
        assert_eq!(step0.atom_type, 0.0);
        assert_eq!(step0.bond, 0.0);
        assert!((step3.atom_type - 1.5).abs() < 1e-12);
        assert_eq!(final_weights.atom_type, 3.0);
        assert_eq!(final_weights.bond, 0.0);
    }

    #[test]
    fn flow_matching_branch_schedule_rejects_all_zero_primary_branches() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .geometry
            .enabled = false;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("branch_schedule must leave at least one enabled branch"));
    }

    #[test]
    fn flow_matching_present_zero_weight_branches_require_explicit_ablation_flag() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = vec![FlowBranchKind::Geometry, FlowBranchKind::AtomType];
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .atom_type
            .final_weight_multiplier = 0.0;

        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("present branch(es) [atom_type]"));
        assert!(err.contains("allow_zero_weight_branch_ablation=true"));

        config
            .generation_method
            .flow_matching
            .multi_modal
            .allow_zero_weight_branch_ablation = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn flow_matching_present_zero_static_weight_requires_explicit_ablation_flag() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_loss_weights
            .geometry = 0.0;

        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("branch_schedule must leave at least one enabled branch"));

        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = vec![FlowBranchKind::Geometry, FlowBranchKind::AtomType];
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_loss_weights
            .atom_type = 0.5;
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("present branch(es) [geometry]"));

        config
            .generation_method
            .flow_matching
            .multi_modal
            .allow_zero_weight_branch_ablation = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn full_molecular_flow_claim_rejects_disabled_required_branch_schedule() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = FlowBranchKind::ALL.to_vec();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .bond
            .final_weight_multiplier = 0.0;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("full_molecular_flow claim configs require branch_schedule.bond"));
    }

    #[test]
    fn full_molecular_flow_claim_rejects_zero_required_branch_loss_weight() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = FlowBranchKind::ALL.to_vec();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_loss_weights
            .topology = 0.0;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("full_molecular_flow claim configs require branch_loss_weights.topology"));
    }

    #[test]
    fn full_molecular_flow_claim_rejects_all_zero_initial_branch_schedule() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = FlowBranchKind::ALL.to_vec();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .geometry
            .start_step = 10;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .atom_type
            .start_step = 10;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .bond
            .start_step = 10;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .topology
            .start_step = 10;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .pocket_context
            .start_step = 10;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("require at least one branch_schedule entry to be active at step 0"));
    }

    #[test]
    fn objective_diagnostic_configs_validate_numeric_bounds() {
        let mut config = ResearchConfig::default();
        config
            .training
            .objective_scale_diagnostics
            .running_scale_momentum = Some(0.5);
        config.training.objective_gradient_diagnostics.enabled = true;
        config
            .training
            .objective_gradient_diagnostics
            .sample_every_steps = 1;
        config
            .training
            .objective_gradient_diagnostics
            .sampling_mode = ObjectiveGradientSamplingMode::LossShareProxy;
        config
            .training
            .objective_gradient_diagnostics
            .included_families = vec!["primary".to_string(), "auxiliary:gate".to_string()];
        assert!(config.validate().is_ok());

        config
            .training
            .objective_gradient_diagnostics
            .included_families = vec!["unknown_family".to_string()];
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("objective_gradient_diagnostics.included_families"));

        config
            .training
            .objective_gradient_diagnostics
            .included_families = Vec::new();
        config.training.objective_scale_diagnostics.warning_ratio = 0.0;
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("objective_scale_diagnostics.warning_ratio"));
    }

    #[test]
    fn full_molecular_flow_claim_requires_non_index_target_matching() {
        let mut config = ResearchConfig::default();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = FlowBranchKind::ALL.to_vec();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .claim_full_molecular_flow = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .target_alignment_policy = FlowTargetAlignmentPolicy::SmokeOnlyModuloRepeat;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("require non-index target matching"));
    }

    #[test]
    fn de_novo_dataset_calibrated_atom_count_validates_bounds() {
        let mut config = ResearchConfig::default();
        config
            .data
            .generation_target
            .de_novo_initialization
            .dataset_calibrated_atom_count = Some(12);
        assert!(config.validate().is_ok());

        config
            .data
            .generation_target
            .de_novo_initialization
            .dataset_calibrated_atom_count = Some(10_000);
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("dataset_calibrated_atom_count must be within"));
    }

    #[test]
    fn topology_encoder_config_selects_message_passing_by_default_and_validates_depth() {
        let mut config = ResearchConfig::default();
        assert_eq!(
            config.model.topology_encoder.kind,
            TopologyEncoderKind::MessagePassing
        );
        assert_eq!(config.model.topology_encoder.bond_type_vocab_size, 8);
        config.model.topology_encoder.kind = TopologyEncoderKind::TypedMessagePassing;
        assert!(config.validate().is_ok());
        config.model.topology_encoder.message_passing_layers = 0;

        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains(
            "model.topology_encoder.message_passing_layers must be greater than zero"
        ));
    }

    #[test]
    fn geometry_encoder_config_selects_distance_kernel_by_default_and_validates_width() {
        let mut config = ResearchConfig::default();
        assert_eq!(
            config.model.geometry_encoder.operator,
            GeometryOperatorKind::PairDistanceKernel
        );
        config.model.geometry_encoder.operator = GeometryOperatorKind::LocalFramePairMessage;
        assert!(config.validate().is_ok());
        config.model.geometry_encoder.distance_kernel_count = 0;

        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("model.geometry_encoder.distance_kernel_count must be greater than zero"));
    }

    #[test]
    fn pocket_encoder_config_selects_local_message_passing_by_default_and_validates_depth() {
        let mut config = ResearchConfig::default();
        assert_eq!(
            config.model.pocket_encoder.kind,
            PocketEncoderKind::LocalMessagePassing
        );
        config.model.pocket_encoder.kind = PocketEncoderKind::LigandRelativeLocalFrame;
        assert!(config.validate().is_ok());
        config.model.pocket_encoder.message_passing_layers = 0;

        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains(
            "model.pocket_encoder.message_passing_layers must be greater than zero"
        ));
    }

    #[test]
    fn slot_decomposition_config_exposes_activation_ablation_controls() {
        let mut config = ResearchConfig::default();
        assert_eq!(config.model.num_slots, 8);
        assert_eq!(config.model.slot_decomposition.activation_temperature, 1.0);
        assert_eq!(config.model.slot_decomposition.activation_threshold, 0.5);
        assert!(config.model.slot_decomposition.attention_masking);
        assert_eq!(config.model.slot_decomposition.minimum_visible_slots, 1);
        assert_eq!(config.model.slot_decomposition.balance_window, 32);

        config.model.slot_decomposition.activation_temperature = 0.0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains(
            "model.slot_decomposition.activation_temperature must be finite and positive"
        ));

        let mut config = ResearchConfig::default();
        config.model.slot_decomposition.minimum_visible_slots = -1;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains(
            "model.slot_decomposition.minimum_visible_slots must be non-negative"
        ));
    }

    #[test]
    fn semantic_probe_config_exposes_capacity_ablation_controls() {
        let mut config = ResearchConfig::default();
        assert_eq!(config.model.semantic_probes.hidden_layers, 0);
        assert_eq!(config.model.semantic_probes.hidden_dim, 128);

        config.model.semantic_probes.hidden_layers = 2;
        config.model.semantic_probes.hidden_dim = 64;
        assert!(config.validate().is_ok());

        config.model.semantic_probes.hidden_dim = 0;
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("model.semantic_probes.hidden_dim must be greater than zero"));
    }

    #[test]
    fn q7_training_presets_load_and_separate_smoke_from_research() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let smoke = load_research_config(root.join("configs/q7_smoke_training_preset.json"))
            .expect("smoke preset should parse");
        let debug = load_research_config(
            root.join("configs/q7_small_real_debug_training_preset.json"),
        )
        .expect("small real-data debug preset should parse");
        let reviewer = load_research_config(
            root.join("configs/q7_reviewer_unseen_pocket_training_preset.json"),
        )
        .expect("reviewer unseen-pocket preset should parse");

        for config in [&smoke, &debug, &reviewer] {
            config.validate().expect("q7 preset should validate");
            assert_eq!(
                config.training.primary_objective,
                PrimaryObjectiveConfig::ConditionedDenoising
            );
            assert_eq!(
                config.generation_method.primary_backend.family,
                GenerationBackendFamilyConfig::ConditionedDenoising
            );
            assert!(!is_configs_checkpoint_tree(&config.training.checkpoint_dir));
        }

        assert_eq!(smoke.data.dataset_format, DatasetFormat::Synthetic);
        assert!(smoke.training.max_steps <= 8);
        assert_eq!(smoke.data.batch_size, 2);
        assert_eq!(
            smoke.model.decoder_conditioning.kind,
            DecoderConditioningKind::MeanPooled
        );

        assert_eq!(debug.data.dataset_format, DatasetFormat::ManifestJson);
        assert_eq!(debug.data.max_examples, Some(32));
        assert!(debug.training.max_steps >= 100);
        assert!(debug.training.max_steps > smoke.training.max_steps);
        assert_eq!(debug.training.gradient_clipping.global_norm, Some(5.0));
        assert_eq!(
            debug.model.decoder_conditioning.kind,
            DecoderConditioningKind::LocalAtomSlotAttention
        );
        assert!(debug.training.loss_weights.sigma_pocket_clash > 0.0);

        assert_eq!(reviewer.data.dataset_format, DatasetFormat::ManifestJson);
        assert_eq!(reviewer.data.max_examples, None);
        assert!(reviewer.training.max_steps >= 1000);
        assert!(reviewer.data.batch_size >= debug.data.batch_size);
        assert_eq!(reviewer.training.gradient_clipping.global_norm, Some(5.0));
        assert!(reviewer.generation_method.enable_comparison_runner);
        assert!(reviewer.data.rotation_augmentation.enabled);
        assert!(reviewer.training.loss_weights.sigma_pocket_clash > 0.0);
        assert!(reviewer.training.schedule.stage3_steps < reviewer.training.max_steps);
    }

    #[test]
    fn decoder_conditioning_config_keeps_mean_pooled_baseline_selectable() {
        let mut config = ResearchConfig::default();
        assert_eq!(
            config.model.decoder_conditioning.kind,
            DecoderConditioningKind::LocalAtomSlotAttention
        );

        config.model.decoder_conditioning.kind = DecoderConditioningKind::MeanPooled;
        assert!(config.validate().is_ok());

        config.model.decoder_conditioning.local_gate_initial_bias = f64::NAN;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains(
            "model.decoder_conditioning.local_gate_initial_bias must be finite"
        ));
    }

    #[test]
    fn resume_config_can_require_exact_optimizer_state_explicitly() {
        let mut config = ResearchConfig::default();
        assert!(!config.training.resume.require_optimizer_exact);

        config.training.resume.require_optimizer_exact = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pharmacophore_probe_config_independently_controls_role_probes_and_leakage() {
        let mut config = ResearchConfig::default();
        assert!(!config.training.pharmacophore_probes.role_probe_enabled());
        assert!(!config.training.pharmacophore_probes.role_leakage_enabled());
        assert_eq!(
            config.training.explicit_leakage_probes.training_semantics,
            ExplicitLeakageProbeTrainingSemantics::AdversarialPenalty
        );

        config
            .training
            .pharmacophore_probes
            .enable_ligand_role_probe = true;
        assert!(config.training.pharmacophore_probes.role_probe_enabled());
        assert!(!config.training.pharmacophore_probes.role_leakage_enabled());
        assert!(config.validate().is_ok());

        config
            .training
            .pharmacophore_probes
            .enable_topology_to_pocket_role_leakage = true;
        assert!(config.training.pharmacophore_probes.role_leakage_enabled());
        config.training.pharmacophore_probes.leakage_margin = f64::NAN;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains(
            "training.pharmacophore_probes.leakage_margin must be finite and non-negative"
        ));
    }

    #[test]
    fn chemistry_warmup_config_validates_stage_numbers() {
        let mut config = ResearchConfig::default();
        assert_eq!(config.training.chemistry_warmup.pocket_envelope_start_stage, 2);
        assert_eq!(
            config
                .training
                .chemistry_warmup
                .pharmacophore_probe_start_stage,
            3
        );
        assert!(config.validate().is_ok());

        config
            .training
            .chemistry_warmup
            .bond_length_guardrail_start_stage = 5;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains(
            "training.chemistry_warmup.bond_length_guardrail_start_stage must be a one-based training stage in [1, 4]"
        ));
    }
}
