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

    use super::*;

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

        let err = load_manifest_entry(&entry, 6.0, ParsingMode::Strict).unwrap_err();
        assert!(matches!(err, DataParseError::InvalidPdb { .. }));
    }
}
