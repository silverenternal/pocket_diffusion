#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::config::{DatasetFormat, GenerationModeConfig, ParsingMode};
    use crate::data::load_affinity_labels;

    #[test]
    fn in_memory_dataset_exposes_indexed_source_boundary() {
        let dataset = InMemoryDataset::new(synthetic_phase1_examples());

        assert_eq!(dataset.len(), dataset.examples().len());
        assert!(dataset.materialized_examples().is_some());

        let collected = collect_examples_from_source(&dataset).unwrap();
        let collected_ids = collected
            .iter()
            .map(|example| example.example_id.as_str())
            .collect::<Vec<_>>();
        let source_ids = dataset
            .examples()
            .iter()
            .map(|example| example.example_id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(collected_ids, source_ids);
    }

    #[test]
    fn strict_mode_rejects_nearest_atom_pocket_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1      50.000  50.000  50.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.parsing_mode = ParsingMode::Strict;
        config.pocket_cutoff_angstrom = 2.0;

        assert!(InMemoryDataset::load_from_config(&config).is_err());
    }

    #[test]
    fn label_loading_tracks_approximate_normalization_warnings() {
        let temp = tempfile::tempdir().unwrap();
        let labels_path = temp.path().join("labels.csv");
        fs::write(
            &labels_path,
            "example_id,measurement_type,raw_value,raw_unit\nex-1,IC50,1.2,uM\n",
        )
        .unwrap();

        let labels = load_affinity_labels(&labels_path).unwrap();
        assert_eq!(labels.labels.len(), 1);
        assert_eq!(labels.report.rows_seen, 1);
        assert!(labels.labels[0].is_approximate);
        assert!(labels.labels[0].normalization_warning.is_some());
        assert_eq!(
            labels.labels[0].normalization_provenance.as_deref(),
            Some("IC50_uM_to_delta_g_via_molar")
        );
    }

    #[test]
    fn optional_quality_filters_report_filtered_examples() {
        let mut config = DataConfig::default();
        config.quality_filters.require_source_structure_provenance = true;

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();

        assert_eq!(loaded.dataset.len(), 0);
        assert_eq!(
            loaded.validation.quality_filtered_missing_source_provenance,
            loaded.validation.discovered_complexes
        );
        assert_eq!(
            loaded.validation.quality_filtered_examples,
            loaded.validation.discovered_complexes
        );
    }

    #[test]
    fn quality_filters_can_reject_low_label_coverage() {
        let mut config = DataConfig::default();
        config.quality_filters.min_label_coverage = Some(0.5);

        let error = InMemoryDataset::load_from_config(&config).unwrap_err();

        assert!(error.to_string().contains("retained label coverage"));
    }

    #[test]
    fn quality_filters_can_reject_high_approximate_label_fraction() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");
        let labels_path = temp.path().join("labels.csv");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            &labels_path,
            "example_id,measurement_type,raw_value,raw_unit\nex-1,IC50,1.2,uM\n",
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.label_table_path = Some(labels_path);
        config.quality_filters.max_approximate_label_fraction = Some(0.0);

        let error = InMemoryDataset::load_from_config(&config).unwrap_err();
        assert!(error.to_string().contains("approximate-label fraction"));
    }

    #[test]
    fn dataset_validation_tracks_retained_metadata_contract() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");
        let labels_path = temp.path().join("labels.csv");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            &labels_path,
            "example_id,measurement_type,raw_value,raw_unit\nex-1,Kd,1.2,uM\n",
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.label_table_path = Some(labels_path);
        config.quality_filters.require_source_structure_provenance = true;
        config.quality_filters.require_affinity_metadata = true;
        config.quality_filters.min_normalization_provenance_coverage = Some(1.0);

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();

        assert_eq!(loaded.validation.attached_labels, 1);
        assert_eq!(loaded.validation.label_table_rows_seen, 1);
        assert_eq!(loaded.validation.retained_measurement_family_count, 1);
        assert_eq!(
            loaded
                .validation
                .retained_measurement_family_histogram
                .get("Kd"),
            Some(&1)
        );
        assert_eq!(loaded.validation.retained_approximate_affinity_labels, 0);
        assert_eq!(loaded.validation.retained_approximate_label_fraction, 0.0);
        assert_eq!(loaded.validation.unmatched_example_id_label_rows, 0);
        assert_eq!(loaded.validation.duplicate_example_id_label_rows, 0);
        assert_eq!(
            loaded.validation.retained_normalization_provenance_coverage,
            1.0
        );
        assert_eq!(loaded.validation.retained_missing_measurement_type, 0);
        assert_eq!(loaded.validation.retained_missing_normalization_provenance, 0);
        assert_eq!(loaded.validation.retained_source_provenance_coverage, 1.0);
        assert_eq!(
            loaded
                .validation
                .retained_ligand_atom_count_histogram
                .get("000-008"),
            Some(&1)
        );
        assert_eq!(
            loaded
                .validation
                .retained_pocket_atom_count_histogram
                .get("000-008"),
            Some(&1)
        );
        assert_eq!(loaded.validation.retained_mean_ligand_atom_count, 1.0);
        assert_eq!(loaded.validation.atom_count_prior_provenance, "target_ligand");
        assert_eq!(loaded.validation.atom_count_prior_mae, 0.0);
        assert_eq!(
            loaded.validation.coordinate_frame_contract,
            "ligand_centered_model_coordinates_with_coordinate_frame_origin"
        );
        assert_eq!(loaded.validation.coordinate_frame_origin_valid_examples, 1);
        assert_eq!(
            loaded.validation.ligand_centered_coordinate_frame_examples,
            1
        );
        assert!(loaded.validation.pocket_coordinates_centered_upstream);
        assert!(loaded.validation.source_coordinate_reconstruction_supported);
        assert!(loaded
            .validation
            .coordinate_frame_artifact_contract
            .contains("coordinate_frame_origin"));
        assert!(loaded
            .validation
            .target_ligand_context_dependency_detected);
        assert!(loaded.validation.target_ligand_context_dependency_allowed);
        assert!(loaded
            .validation
            .target_ligand_context_leakage_warnings
            .is_empty());
    }

    #[test]
    fn pocket_only_context_leakage_is_reported_and_optionally_rejected() {
        let mut config = DataConfig::default();
        config.generation_target.generation_mode =
            GenerationModeConfig::PocketOnlyInitializationBaseline;

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();
        assert!(loaded
            .validation
            .target_ligand_context_dependency_detected);
        assert!(!loaded
            .validation
            .target_ligand_context_dependency_allowed);
        assert!(loaded
            .validation
            .target_ligand_context_leakage_warnings
            .iter()
            .any(|warning| warning.contains("pocket-only")));

        config.quality_filters.reject_target_ligand_context_leakage = true;
        let error = InMemoryDataset::load_from_config(&config).unwrap_err();
        assert!(error
            .to_string()
            .contains("rejects target-ligand-centered pocket/context tensors"));
    }

    #[test]
    fn de_novo_context_leakage_is_reported_and_optionally_rejected() {
        let mut config = DataConfig::default();
        config.generation_target.generation_mode = GenerationModeConfig::DeNovoInitialization;

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();
        assert!(loaded
            .validation
            .target_ligand_context_dependency_detected);
        assert!(!loaded
            .validation
            .target_ligand_context_dependency_allowed);
        assert!(loaded
            .validation
            .generation_target_leakage_contract
            .contains("de_novo_initialization"));
        assert!(loaded
            .validation
            .target_ligand_context_leakage_warnings
            .iter()
            .any(|warning| warning.contains("de novo")));

        config.quality_filters.reject_target_ligand_context_leakage = true;
        let error = InMemoryDataset::load_from_config(&config).unwrap_err();
        assert!(error
            .to_string()
            .contains("generation_mode=de_novo_initialization rejects"));
    }

    #[test]
    fn dataset_validation_tracks_duplicate_and_unmatched_label_rows() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");
        let labels_path = temp.path().join("labels.csv");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            &labels_path,
            "example_id,protein_id,measurement_type,raw_value,raw_unit\nex-1,p-1,Kd,1.2,uM\nex-1,p-1,Ki,2.0,uM\nex-missing,p-missing,Kd,3.0,uM\n",
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.label_table_path = Some(labels_path);

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();

        assert_eq!(loaded.validation.loaded_label_rows, 3);
        assert_eq!(loaded.validation.duplicate_example_id_label_rows, 1);
        assert_eq!(loaded.validation.duplicate_protein_id_label_rows, 1);
        assert_eq!(loaded.validation.unmatched_example_id_label_rows, 1);
        assert_eq!(loaded.validation.unmatched_protein_id_label_rows, 1);
        assert_eq!(
            loaded
                .validation
                .loaded_label_measurement_family_histogram
                .get("Kd"),
            Some(&2)
        );
        assert_eq!(
            loaded
                .validation
                .loaded_label_measurement_family_histogram
                .get("Ki"),
            Some(&1)
        );
    }

    #[test]
    fn dataset_validation_tracks_rotation_augmentation_attempts_and_applications() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");
        let manifest_path = temp.path().join("manifest.json");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  3  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.2000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n    0.0000    1.2000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(
            &pocket_path,
            "ATOM      1  C   GLY A   1       0.200   0.100   0.300  1.00 20.00           C\nATOM      2  N   GLY A   2       1.000   0.500   0.500  1.00 20.00           N\nATOM      3  O   GLY A   3       0.200   1.200  -0.200  1.00 20.00           O\n",
        )
        .unwrap();
        fs::write(
            &manifest_path,
            serde_json::json!({
                "entries": [{
                    "example_id": "ex-1",
                    "protein_id": "p-1",
                    "pocket_path": "pocket.pdb",
                    "ligand_path": "ligand.sdf"
                }]
            })
            .to_string(),
        )
        .unwrap();

        let mut config = DataConfig::default();
        config.dataset_format = DatasetFormat::ManifestJson;
        config.root_dir = temp.path().to_path_buf();
        config.manifest_path = Some(manifest_path);
        config.rotation_augmentation.enabled = true;
        config.rotation_augmentation.probability = 1.0;
        config.rotation_augmentation.seed = 17;

        let loaded = InMemoryDataset::load_from_config(&config).unwrap();

        assert_eq!(loaded.validation.rotation_augmentation_attempted_examples, 1);
        assert_eq!(loaded.validation.rotation_augmentation_applied_examples, 1);
        assert_eq!(loaded.validation.parsed_examples, 1);
        assert_eq!(loaded.validation.parsed_ligands, 1);
        assert_eq!(loaded.validation.parsed_pockets, 1);
        assert_eq!(loaded.dataset.len(), 1);
    }
}
