use pocket_diffusion::experiments::UnseenPocketExperimentConfig;
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
}
