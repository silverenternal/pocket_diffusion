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
}
