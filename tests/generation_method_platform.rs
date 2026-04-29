use pocket_diffusion::config::GenerationModeConfig;
use pocket_diffusion::experiments::{
    validate_experiment_config_with_source, UnseenPocketExperimentConfig,
};
use pocket_diffusion::models::PocketGenerationMethodRegistry;

#[test]
fn method_registry_exposes_active_and_stub_methods() {
    for method_id in [
        "conditioned_denoising",
        "heuristic_raw_rollout_no_repair",
        "pocket_centroid_repair_proxy",
        "deterministic_proxy_reranker",
        "calibrated_reranker",
        "flow_matching_stub",
        "diffusion_stub",
        "autoregressive_stub",
        "external_wrapper_stub",
    ] {
        assert!(PocketGenerationMethodRegistry::contains(method_id));
        let metadata = PocketGenerationMethodRegistry::metadata(method_id).unwrap();
        assert_eq!(metadata.method_id, method_id);
    }
}

#[test]
fn experiment_config_rejects_unknown_generation_method() {
    let mut config = UnseenPocketExperimentConfig::default();
    config.research.generation_method.active_method = "missing_method".to_string();
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("unknown generation_method.active_method"));
    assert!(error.contains("missing_method"));
    assert!(error.contains("family="));
    assert!(error.contains("config_path="));
}

#[test]
fn experiment_config_source_validation_reports_config_path() {
    let mut config = UnseenPocketExperimentConfig::default();
    config.research.generation_method.active_method = "missing_method".to_string();
    let error = validate_experiment_config_with_source(&config, "configs/reviewer_surface.json")
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing_method"));
    assert!(error.contains("family="));
    assert!(error.contains("configs/reviewer_surface.json"));
}

#[test]
fn generation_method_platform_de_novo_generation_mode_requires_flow_contract() {
    let mut config = UnseenPocketExperimentConfig::default();
    config.research.data.generation_target.generation_mode =
        GenerationModeConfig::DeNovoInitialization;
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("de_novo_initialization"));
    assert!(error.contains("flow_matching"));
}

#[test]
fn generation_method_platform_allows_pocket_only_initialization_baseline() {
    let mut config = UnseenPocketExperimentConfig::default();
    config.research.data.generation_target.generation_mode =
        GenerationModeConfig::PocketOnlyInitializationBaseline;
    config.research.training.primary_objective =
        pocket_diffusion::config::PrimaryObjectiveConfig::SurrogateReconstruction;
    config
        .research
        .data
        .generation_target
        .pocket_only_initialization
        .atom_count = 8;

    config.validate().unwrap();
    assert!(!config
        .research
        .data
        .generation_target
        .generation_mode
        .permits_de_novo_claims());
}
