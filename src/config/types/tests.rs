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
        assert_eq!(config.model.pocket_feature_dim, 12);
        assert_eq!(config.runtime.device, "cpu");
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
    fn flow_matching_generation_config_validates_geometry_only_guard() {
        let mut config = ResearchConfig::default();
        config.generation_method.flow_matching.geometry_only = false;
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("flow_matching.geometry_only must remain true"));
    }
}
