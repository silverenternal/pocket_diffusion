fn toy_ligand(offset: f64) -> Ligand {
    Ligand {
        atoms: vec![
            Atom {
                coords: [0.0 + offset, 0.0, 0.0],
                atom_type: AtomType::Carbon,
                index: 0,
            },
            Atom {
                coords: [1.3 + offset, 0.1, 0.0],
                atom_type: AtomType::Nitrogen,
                index: 1,
            },
            Atom {
                coords: [2.1 + offset, 1.0, 0.2],
                atom_type: AtomType::Oxygen,
                index: 2,
            },
        ],
        bonds: vec![(0, 1), (1, 2)],
        bond_types: vec![1, 1],
        fingerprint: None,
    }
}

fn toy_pocket(name: &str, offset: f64) -> Pocket {
    Pocket {
        name: name.to_string(),
        atoms: vec![
            Atom {
                coords: [offset, 0.0, 0.0],
                atom_type: AtomType::Carbon,
                index: 0,
            },
            Atom {
                coords: [0.7 + offset, 1.0, 0.4],
                atom_type: AtomType::Oxygen,
                index: 1,
            },
            Atom {
                coords: [1.5 + offset, -0.8, 0.2],
                atom_type: AtomType::Nitrogen,
                index: 2,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::config::types::RotationAugmentationConfig;

    #[test]
    fn malformed_sdf_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let sdf_path = temp.path().join("bad.sdf");
        fs::write(&sdf_path, "broken sdf\n").unwrap();

        let err = load_ligand_from_sdf(&sdf_path).unwrap_err();
        assert!(matches!(err, DataParseError::InvalidSdf { .. }));
    }

    #[test]
    fn sdf_bond_parser_handles_contiguous_fixed_width_indices() {
        assert_eq!(
            parse_v2000_bond_indices(" 34100  1  0  0  0"),
            Some((34, 100))
        );
        assert_eq!(parse_v2000_bond_indices("  2  5  1  0  0  0"), Some((2, 5)));
        assert_eq!(
            parse_v2000_bond_record("  2  5  2  0  0  0"),
            Some((2, 5, 2))
        );
    }

    #[test]
    fn strict_mode_rejects_short_pdb_atom_records() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("bad.pdb");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(&pocket_path, "ATOM      1  C\n").unwrap();

        let entry = ManifestEntry {
            example_id: "ex-1".to_string(),
            protein_id: "prot-1".to_string(),
            pocket_path,
            ligand_path,
            affinity_kcal_mol: None,
            affinity_measurement_type: None,
            affinity_raw_value: None,
            affinity_raw_unit: None,
            affinity_normalization_provenance: None,
            affinity_is_approximate: false,
            affinity_normalization_warning: None,
        };

        let mut rotation_rng = StdRng::seed_from_u64(17);
        let err = load_manifest_entry(
            &entry,
            6.0,
            ParsingMode::Strict,
            &RotationAugmentationConfig::default(),
            &mut rotation_rng,
        )
        .unwrap_err();
        assert!(matches!(err, DataParseError::InvalidPdb { .. }));
    }

    #[test]
    fn rotation_augmentation_changes_geometry_but_preserves_distances() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("pocket.pdb");

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

        let entry = ManifestEntry {
            example_id: "rot-1".to_string(),
            protein_id: "prot-1".to_string(),
            pocket_path,
            ligand_path,
            affinity_kcal_mol: None,
            affinity_measurement_type: None,
            affinity_raw_value: None,
            affinity_raw_unit: None,
            affinity_normalization_provenance: None,
            affinity_is_approximate: false,
            affinity_normalization_warning: None,
        };

        let mut no_aug_rng = StdRng::seed_from_u64(1);
        let (baseline, baseline_parsed) = load_manifest_entry(
            &entry,
            6.0,
            ParsingMode::Lightweight,
            &RotationAugmentationConfig::default(),
            &mut no_aug_rng,
        )
        .unwrap();
        let mut aug_rng = StdRng::seed_from_u64(1);
        let (rotated, rotated_parsed) = load_manifest_entry(
            &entry,
            6.0,
            ParsingMode::Lightweight,
            &RotationAugmentationConfig {
                enabled: true,
                probability: 1.0,
                ..RotationAugmentationConfig::default()
            },
            &mut aug_rng,
        )
        .unwrap();

        assert!(!baseline_parsed.rotation_augmentation_applied);
        assert!(rotated_parsed.rotation_augmentation_attempted);
        assert!(rotated_parsed.rotation_augmentation_applied);

        let mut max_coord_delta = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let baseline_coord = baseline.geometry.coords.double_value(&[i, j]);
                let rotated_coord = rotated.geometry.coords.double_value(&[i, j]);
                max_coord_delta = max_coord_delta.max((baseline_coord - rotated_coord).abs());
            }
        }
        assert!(
            max_coord_delta > 1e-6,
            "rotation augmentation should change geometry coordinates when applied"
        );

        for i in 0..3 {
            for j in 0..3 {
                let d0 = baseline.geometry.pairwise_distances.double_value(&[i, j]);
                let d1 = rotated.geometry.pairwise_distances.double_value(&[i, j]);
                assert!(
                    (d0 - d1).abs() <= 1e-6,
                    "pairwise distances should remain invariant under rigid rotation"
                );
            }
        }
    }
}
