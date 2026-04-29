#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Atom;

    fn ligand_with_offset(offset: f64) -> Ligand {
        Ligand {
            atoms: vec![
                Atom {
                    coords: [offset, offset + 1.0, offset + 2.0],
                    atom_type: AtomType::Carbon,
                    index: 0,
                },
                Atom {
                    coords: [offset + 2.0, offset + 3.0, offset + 4.0],
                    atom_type: AtomType::Oxygen,
                    index: 1,
                },
            ],
            bonds: vec![(0, 1)],
            bond_types: vec![1],
            fingerprint: None,
        }
    }

    #[test]
    fn geometry_features_are_ligand_centered() {
        let ligand = ligand_with_offset(100.0);
        let geometry = geometry_from_ligand(&ligand);

        let centroid = geometry.coords.mean_dim([0].as_slice(), false, Kind::Float);
        for dim in 0..3 {
            assert!(centroid.double_value(&[dim]).abs() < 1e-6);
        }
        assert!((geometry.pairwise_distances.double_value(&[0, 1]) - 12.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn topology_features_preserve_ligand_bond_types() {
        let ligand = ligand_with_offset(0.0);
        let topology = topology_from_ligand(&ligand);

        assert_eq!(topology.bond_types.size(), vec![1]);
        assert_eq!(topology.bond_types.int64_value(&[0]), 1);
    }

    #[test]
    fn topology_features_fall_back_to_unknown_bond_types_when_missing() {
        let mut ligand = ligand_with_offset(0.0);
        ligand.bond_types.clear();

        let topology = topology_from_ligand(&ligand);

        assert_eq!(topology.bond_types.size(), vec![1]);
        assert_eq!(topology.bond_types.int64_value(&[0]), 0);
    }

    #[test]
    fn legacy_examples_keep_pocket_coords_in_ligand_centered_frame() {
        let ligand = ligand_with_offset(100.0);
        let pocket = Pocket {
            name: "pocket".to_string(),
            atoms: vec![Atom {
                coords: [102.0, 104.0, 106.0],
                atom_type: AtomType::Nitrogen,
                index: 0,
            }],
        };

        let example = MolecularExample::from_legacy("example", "protein", &ligand, &pocket);
        assert_eq!(example.coordinate_frame_origin, [101.0, 102.0, 103.0]);
        assert_eq!(example.pocket.coords.double_value(&[0, 0]), 1.0);
        assert_eq!(example.pocket.coords.double_value(&[0, 1]), 2.0);
        assert_eq!(example.pocket.coords.double_value(&[0, 2]), 3.0);
    }

    #[test]
    fn source_translation_preserves_ligand_centered_model_frame() {
        let ligand = ligand_with_offset(25.0);
        let pocket = Pocket {
            name: "pocket".to_string(),
            atoms: vec![
                Atom {
                    coords: [27.0, 29.0, 31.0],
                    atom_type: AtomType::Nitrogen,
                    index: 0,
                },
                Atom {
                    coords: [25.5, 27.5, 29.5],
                    atom_type: AtomType::Oxygen,
                    index: 1,
                },
            ],
        };
        let shift = [9.5, -3.25, 4.0];
        let translated_ligand = translate_ligand(&ligand, shift);
        let translated_pocket = translate_pocket(&pocket, shift);

        let base = MolecularExample::from_legacy("base", "protein", &ligand, &pocket);
        let translated = MolecularExample::from_legacy(
            "translated",
            "protein",
            &translated_ligand,
            &translated_pocket,
        );

        let geometry_delta = (&base.geometry.coords - &translated.geometry.coords)
            .abs()
            .max()
            .double_value(&[]);
        let pocket_delta = (&base.pocket.coords - &translated.pocket.coords)
            .abs()
            .max()
            .double_value(&[]);
        assert!(geometry_delta < 1e-6, "geometry delta was {geometry_delta}");
        assert!(pocket_delta < 1e-6, "pocket delta was {pocket_delta}");
        for axis in 0..3 {
            let expected = base.coordinate_frame_origin[axis] + shift[axis] as f32;
            assert!(
                (translated.coordinate_frame_origin[axis] - expected).abs() < 1e-6,
                "axis {axis} origin mismatch"
            );
        }
    }

    fn translate_ligand(ligand: &Ligand, shift: [f64; 3]) -> Ligand {
        let mut translated = ligand.clone();
        for atom in &mut translated.atoms {
            translate_atom(atom, shift);
        }
        translated
    }

    fn translate_pocket(pocket: &Pocket, shift: [f64; 3]) -> Pocket {
        let mut translated = pocket.clone();
        for atom in &mut translated.atoms {
            translate_atom(atom, shift);
        }
        translated
    }

    fn translate_atom(atom: &mut Atom, shift: [f64; 3]) {
        for (coord, delta) in atom.coords.iter_mut().zip(shift) {
            *coord += delta;
        }
    }

    #[test]
    fn heuristic_chemistry_roles_are_finite_and_mark_unknown_atoms() {
        let roles = chemistry_role_features_from_atom_types(&[
            AtomType::Carbon,
            AtomType::Nitrogen,
            AtomType::Oxygen,
            AtomType::Sulfur,
            AtomType::Other,
        ]);

        assert_eq!(
            roles.role_vectors.size(),
            [5, CHEMISTRY_ROLE_FEATURE_DIM]
        );
        assert_eq!(roles.provenance, ChemistryRoleFeatureProvenance::Heuristic);
        assert!(
            roles
                .role_vectors
                .isfinite()
                .all()
                .to_kind(Kind::Int64)
                .int64_value(&[])
                != 0
        );
        assert_eq!(roles.role_vectors.double_value(&[0, 2]), 1.0);
        assert_eq!(roles.role_vectors.double_value(&[1, 0]), 1.0);
        assert_eq!(roles.role_vectors.double_value(&[2, 1]), 1.0);
        assert_eq!(roles.role_vectors.double_value(&[4, 7]), 1.0);
        assert_eq!(roles.role_vectors.double_value(&[4, 8]), 0.0);
        assert_eq!(roles.availability.double_value(&[4]), 0.0);
    }

    #[test]
    fn legacy_examples_attach_ligand_and_pocket_chemistry_roles() {
        let ligand = ligand_with_offset(0.0);
        let pocket = Pocket {
            name: "pocket".to_string(),
            atoms: vec![
                Atom {
                    coords: [0.0, 0.0, 0.0],
                    atom_type: AtomType::Nitrogen,
                    index: 0,
                },
                Atom {
                    coords: [1.0, 0.0, 0.0],
                    atom_type: AtomType::Other,
                    index: 1,
                },
            ],
        };

        let example = MolecularExample::from_legacy("example", "protein", &ligand, &pocket);
        assert_eq!(
            example.topology.chemistry_roles.role_vectors.size(),
            [2, CHEMISTRY_ROLE_FEATURE_DIM]
        );
        assert_eq!(
            example.pocket.chemistry_roles.role_vectors.size(),
            [2, CHEMISTRY_ROLE_FEATURE_DIM]
        );
        assert_eq!(
            example.pocket.chemistry_roles.availability.double_value(&[0]),
            1.0
        );
        assert_eq!(
            example.pocket.chemistry_roles.availability.double_value(&[1]),
            0.0
        );
    }
}
