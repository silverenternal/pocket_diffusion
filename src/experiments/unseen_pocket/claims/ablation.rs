fn run_ablation_matrix(
    config: &UnseenPocketExperimentConfig,
) -> Result<AblationMatrixSummary, Box<dyn std::error::Error>> {
    let mut variants = Vec::new();
    for ablation in ablation_variants(config) {
        let mut variant_config = config.clone();
        variant_config.ablation = ablation.clone();
        variant_config.ablation_matrix.enabled = false;
        if let Some(primary) = ablation.primary_objective_override {
            variant_config.research.training.primary_objective = primary;
        }
        if let Some(backend) = ablation.generation_backend_override.clone() {
            variant_config.research.generation_method.active_method = backend.backend_id.clone();
            variant_config.research.generation_method.primary_backend = backend;
        }
        if let Some(mode) = ablation.generation_mode_override {
            variant_config.research.data.generation_target.generation_mode = mode;
        }
        if let Some(flow_matching) = ablation.flow_matching_override.clone() {
            variant_config.research.generation_method.flow_matching = flow_matching;
        }
        if let Some(num_slots) = ablation.num_slots_override {
            variant_config.research.model.num_slots = num_slots;
        }
        if let Some(attention_masking) = ablation.slot_attention_masking_override {
            variant_config
                .research
                .model
                .slot_decomposition
                .attention_masking = attention_masking;
        }
        if let Some(eta_gate) = ablation.eta_gate_override {
            variant_config.research.training.loss_weights.eta_gate = eta_gate;
        }
        if let Some(gate_temperature) = ablation.interaction_gate_temperature_override {
            variant_config
                .research
                .model
                .interaction_tuning
                .gate_temperature = gate_temperature;
        }
        if let Some(delta_leak) = ablation.delta_leak_override {
            variant_config.research.training.loss_weights.delta_leak = delta_leak;
        }
        if ablation.disable_redundancy {
            variant_config.research.training.loss_weights.beta_intra_red = 0.0;
        }
        if ablation.disable_staged_schedule {
            variant_config.research.training.schedule.stage1_steps = 0;
            variant_config.research.training.schedule.stage2_steps = 0;
            variant_config.research.training.schedule.stage3_steps = 0;
            variant_config
                .research
                .training
                .chemistry_warmup
                .pocket_envelope_start_stage = 1;
            variant_config
                .research
                .training
                .chemistry_warmup
                .valence_guardrail_start_stage = 1;
            variant_config
                .research
                .training
                .chemistry_warmup
                .bond_length_guardrail_start_stage = 1;
            variant_config
                .research
                .training
                .chemistry_warmup
                .pharmacophore_probe_start_stage = 1;
            variant_config
                .research
                .training
                .chemistry_warmup
                .pharmacophore_leakage_start_stage = 1;
        }
        if let Some(focus) = ablation.modality_focus_override {
            variant_config.research.model.modality_focus = focus;
        }
        if let Some(kind) = ablation.topology_encoder_kind_override {
            variant_config.research.model.topology_encoder.kind = kind;
        }
        if let Some(operator) = ablation.geometry_operator_override {
            variant_config.research.model.geometry_encoder.operator = operator;
        }
        if let Some(kind) = ablation.pocket_encoder_kind_override {
            variant_config.research.model.pocket_encoder.kind = kind;
        }
        if let Some(kind) = ablation.decoder_conditioning_override {
            variant_config.research.model.decoder_conditioning.kind = kind;
        }
        if let Some(mode) = ablation.interaction_mode_override {
            variant_config.research.model.interaction_mode = mode;
        }
        if let Some(label) = &ablation.variant_label {
            variant_config.research.training.checkpoint_dir = config
                .research
                .training
                .checkpoint_dir
                .join("ablations")
                .join(label);
        }
        let summary = UnseenPocketExperiment::run_with_options(variant_config, false)?;
        variants.push(AblationRunSummary {
            variant_label: ablation
                .variant_label
                .clone()
                .unwrap_or_else(|| "unnamed_variant".to_string()),
            validation: summary.validation.comparison_summary,
            test: summary.test.comparison_summary,
        });
    }

    Ok(AblationMatrixSummary {
        artifact_dir: config.research.training.checkpoint_dir.join("ablations"),
        variants,
    })
}

fn ablation_variants(config: &UnseenPocketExperimentConfig) -> Vec<AblationConfig> {
    let mut variants = Vec::new();
    if config.ablation_matrix.include_surrogate_objective
        && config.research.training.primary_objective
            != crate::config::PrimaryObjectiveConfig::SurrogateReconstruction
    {
        variants.push(AblationConfig {
            primary_objective_override: Some(
                crate::config::PrimaryObjectiveConfig::SurrogateReconstruction,
            ),
            variant_label: Some("objective_surrogate".to_string()),
            ..AblationConfig::default()
        });
    }
    if config.ablation_matrix.include_conditioned_denoising
        && config.research.training.primary_objective
            != crate::config::PrimaryObjectiveConfig::ConditionedDenoising
    {
        variants.push(AblationConfig {
            primary_objective_override: Some(
                crate::config::PrimaryObjectiveConfig::ConditionedDenoising,
            ),
            variant_label: Some("objective_conditioned_denoising".to_string()),
            ..AblationConfig::default()
        });
    }
    if config.ablation_matrix.include_generation_mode_ablation
        && config.research.data.generation_target.generation_mode
            != crate::config::GenerationModeConfig::LigandRefinement
    {
        variants.push(AblationConfig {
            generation_mode_override: Some(crate::config::GenerationModeConfig::LigandRefinement),
            variant_label: Some("generation_mode_ligand_refinement".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_generation_mode_ablation
        && config.research.data.generation_target.generation_mode
            != crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline
    {
        variants.push(AblationConfig {
            generation_mode_override: Some(
                crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline,
            ),
            variant_label: Some("generation_mode_pocket_only_initialization_baseline".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_generation_mode_ablation
        && config.research.data.generation_target.generation_mode
            != crate::config::GenerationModeConfig::DeNovoInitialization
    {
        let mut flow_matching = config.research.generation_method.flow_matching.clone();
        flow_matching.geometry_only = false;
        flow_matching.use_corrupted_x0 = false;
        flow_matching.multi_modal.enabled_branches = vec![
            crate::config::FlowBranchKind::Geometry,
            crate::config::FlowBranchKind::AtomType,
            crate::config::FlowBranchKind::Bond,
            crate::config::FlowBranchKind::Topology,
            crate::config::FlowBranchKind::PocketContext,
        ];
        flow_matching.multi_modal.claim_full_molecular_flow = true;
        variants.push(AblationConfig {
            primary_objective_override: Some(crate::config::PrimaryObjectiveConfig::FlowMatching),
            generation_backend_override: Some(crate::config::GenerationBackendConfig {
                backend_id: "flow_matching".to_string(),
                family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
                trainable: true,
                ..crate::config::GenerationBackendConfig::default()
            }),
            generation_mode_override: Some(
                crate::config::GenerationModeConfig::DeNovoInitialization,
            ),
            flow_matching_override: Some(flow_matching),
            variant_label: Some("de_novo_full_molecular_flow".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_disable_slots {
        variants.push(AblationConfig {
            disable_slots: true,
            variant_label: Some("disable_slots".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_disable_cross_attention {
        variants.push(AblationConfig {
            disable_cross_attention: true,
            variant_label: Some("disable_cross_attention".to_string()),
            ..config.ablation.clone()
        });
    }
    if config
        .ablation_matrix
        .include_disable_geometry_interaction_bias
    {
        variants.push(AblationConfig {
            disable_geometry_interaction_bias: true,
            variant_label: Some("disable_geometry_interaction_bias".to_string()),
            ..config.ablation.clone()
        });
    }
    if config
        .ablation_matrix
        .include_disable_rollout_pocket_guidance
    {
        variants.push(AblationConfig {
            disable_rollout_pocket_guidance: true,
            variant_label: Some("disable_rollout_pocket_guidance".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_disable_candidate_repair {
        variants.push(AblationConfig {
            disable_candidate_repair: true,
            variant_label: Some("disable_candidate_repair".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_backend_family_ablation {
        push_backend_variant(
            &mut variants,
            config,
            "backend_flow_matching",
            "flow_matching",
            crate::config::GenerationBackendFamilyConfig::FlowMatching,
            true,
        );
        push_backend_variant(
            &mut variants,
            config,
            "backend_autoregressive_graph_geometry",
            "autoregressive_graph_geometry",
            crate::config::GenerationBackendFamilyConfig::Autoregressive,
            true,
        );
        push_backend_variant(
            &mut variants,
            config,
            "backend_energy_guided_refinement",
            "energy_guided_refinement",
            crate::config::GenerationBackendFamilyConfig::EnergyGuidedRefinement,
            false,
        );
    }
    if config.ablation_matrix.include_slot_count_ablation && config.research.model.num_slots > 1 {
        variants.push(AblationConfig {
            num_slots_override: Some((config.research.model.num_slots / 2).max(1)),
            variant_label: Some("slot_count_reduced".to_string()),
            ..config.ablation.clone()
        });
    }
    if config
        .ablation_matrix
        .include_slot_attention_masking_ablation
        && config.research.model.slot_decomposition.attention_masking
    {
        variants.push(AblationConfig {
            slot_attention_masking_override: Some(false),
            variant_label: Some("slot_attention_masking_disabled".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_gate_sparsity_ablation
        && config.research.training.loss_weights.eta_gate > 0.0
    {
        variants.push(AblationConfig {
            eta_gate_override: Some(0.0),
            variant_label: Some("gate_sparsity_disabled".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_gate_scale_ablation
        && (config
            .research
            .model
            .interaction_tuning
            .gate_temperature
            - 2.0)
            .abs()
            > f64::EPSILON
    {
        variants.push(AblationConfig {
            interaction_gate_temperature_override: Some(2.0),
            variant_label: Some("interaction_gate_temperature_high".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_leakage_penalty_ablation
        && config.research.training.loss_weights.delta_leak > 0.0
    {
        variants.push(AblationConfig {
            delta_leak_override: Some(0.0),
            variant_label: Some("leakage_penalty_disabled".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_redundancy_ablation
        && config.research.training.loss_weights.beta_intra_red > 0.0
    {
        variants.push(AblationConfig {
            disable_redundancy: true,
            variant_label: Some("redundancy_disabled".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_staged_schedule_ablation {
        variants.push(AblationConfig {
            disable_staged_schedule: true,
            variant_label: Some("staged_schedule_disabled".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_modality_focus_ablation {
        for (focus, label) in [
            (
                crate::config::ModalityFocusConfig::TopologyOnly,
                "topology_only",
            ),
            (
                crate::config::ModalityFocusConfig::GeometryOnly,
                "geometry_only",
            ),
            (crate::config::ModalityFocusConfig::PocketOnly, "pocket_only"),
        ] {
            if config.research.model.modality_focus != focus {
                variants.push(AblationConfig {
                    modality_focus_override: Some(focus),
                    variant_label: Some(label.to_string()),
                    ..config.ablation.clone()
                });
            }
        }
    }
    if config.ablation_matrix.include_disable_probes {
        variants.push(AblationConfig {
            disable_probes: true,
            variant_label: Some("disable_probes".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_disable_leakage {
        variants.push(AblationConfig {
            disable_leakage: true,
            variant_label: Some("disable_leakage".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_topology_encoder_ablation {
        for (kind, label) in [
            (
                crate::config::TopologyEncoderKind::Lightweight,
                "topology_encoder_lightweight",
            ),
            (
                crate::config::TopologyEncoderKind::TypedMessagePassing,
                "topology_encoder_typed_message_passing",
            ),
        ] {
            if config.research.model.topology_encoder.kind != kind {
                variants.push(AblationConfig {
                    topology_encoder_kind_override: Some(kind),
                    variant_label: Some(label.to_string()),
                    ..config.ablation.clone()
                });
            }
        }
    }
    if config.ablation_matrix.include_geometry_operator_ablation {
        for (operator, label) in [
            (
                crate::config::GeometryOperatorKind::RawCoordinateProjection,
                "geometry_operator_raw_coordinate_projection",
            ),
            (
                crate::config::GeometryOperatorKind::PairDistanceKernel,
                "geometry_operator_pair_distance_kernel",
            ),
            (
                crate::config::GeometryOperatorKind::LocalFramePairMessage,
                "geometry_operator_local_frame_pair_message",
            ),
        ] {
            if config.research.model.geometry_encoder.operator != operator {
                variants.push(AblationConfig {
                    geometry_operator_override: Some(operator),
                    variant_label: Some(label.to_string()),
                    ..config.ablation.clone()
                });
            }
        }
    }
    if config.ablation_matrix.include_pocket_encoder_ablation {
        for (kind, label) in [
            (
                crate::config::PocketEncoderKind::FeatureProjection,
                "pocket_encoder_feature_projection",
            ),
            (
                crate::config::PocketEncoderKind::LocalMessagePassing,
                "pocket_encoder_local_message_passing",
            ),
            (
                crate::config::PocketEncoderKind::LigandRelativeLocalFrame,
                "pocket_encoder_ligand_relative_local_frame",
            ),
        ] {
            if config.research.model.pocket_encoder.kind != kind {
                variants.push(AblationConfig {
                    pocket_encoder_kind_override: Some(kind),
                    variant_label: Some(label.to_string()),
                    ..config.ablation.clone()
                });
            }
        }
    }
    if config.ablation_matrix.include_decoder_conditioning_ablation
        && config.research.model.decoder_conditioning.kind
            != crate::config::DecoderConditioningKind::MeanPooled
    {
        variants.push(AblationConfig {
            decoder_conditioning_override: Some(crate::config::DecoderConditioningKind::MeanPooled),
            variant_label: Some("decoder_conditioning_mean_pooled".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_lightweight_interaction
        && config.research.model.interaction_mode != CrossAttentionMode::Lightweight
    {
        variants.push(AblationConfig {
            interaction_mode_override: Some(CrossAttentionMode::Lightweight),
            variant_label: Some("interaction_lightweight".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_transformer_interaction
        && config.research.model.interaction_mode != CrossAttentionMode::Transformer
    {
        variants.push(AblationConfig {
            interaction_mode_override: Some(CrossAttentionMode::Transformer),
            variant_label: Some("interaction_transformer".to_string()),
            ..config.ablation.clone()
        });
    }
    if config.ablation_matrix.include_direct_fusion_negative_control
        && config.research.model.interaction_mode != CrossAttentionMode::DirectFusionNegativeControl
    {
        variants.push(AblationConfig {
            interaction_mode_override: Some(CrossAttentionMode::DirectFusionNegativeControl),
            variant_label: Some("direct_fusion_negative_control".to_string()),
            ..config.ablation.clone()
        });
    }
    variants
}

fn push_backend_variant(
    variants: &mut Vec<AblationConfig>,
    config: &UnseenPocketExperimentConfig,
    label: &str,
    backend_id: &str,
    family: crate::config::GenerationBackendFamilyConfig,
    trainable: bool,
) {
    if config.research.generation_method.primary_backend_id() == backend_id {
        return;
    }
    variants.push(AblationConfig {
        generation_backend_override: Some(crate::config::GenerationBackendConfig {
            backend_id: backend_id.to_string(),
            family,
            trainable,
            ..crate::config::GenerationBackendConfig::default()
        }),
        variant_label: Some(label.to_string()),
        ..config.ablation.clone()
    });
}

fn persist_ablation_matrix(
    checkpoint_dir: &std::path::Path,
    matrix: &AblationMatrixSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(checkpoint_dir)?;
    fs::write(
        checkpoint_dir.join("ablation_matrix_summary.json"),
        serde_json::to_string_pretty(matrix)?,
    )?;
    Ok(())
}
