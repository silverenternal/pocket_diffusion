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
        if let Some(num_slots) = ablation.num_slots_override {
            variant_config.research.model.num_slots = num_slots;
        }
        if let Some(eta_gate) = ablation.eta_gate_override {
            variant_config.research.training.loss_weights.eta_gate = eta_gate;
        }
        if let Some(delta_leak) = ablation.delta_leak_override {
            variant_config.research.training.loss_weights.delta_leak = delta_leak;
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
    if config.ablation_matrix.include_gate_sparsity_ablation
        && config.research.training.loss_weights.eta_gate > 0.0
    {
        variants.push(AblationConfig {
            eta_gate_override: Some(0.0),
            variant_label: Some("gate_sparsity_disabled".to_string()),
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
    if config.ablation_matrix.include_disable_probes {
        variants.push(AblationConfig {
            disable_probes: true,
            variant_label: Some("disable_probes".to_string()),
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
