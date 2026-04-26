#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::config::{DatasetFormat, ParsingMode};
    use crate::data::load_affinity_labels;

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
}
