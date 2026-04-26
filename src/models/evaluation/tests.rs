#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_with_coords(coords: Vec<[f32; 3]>) -> GeneratedCandidateRecord {
        GeneratedCandidateRecord {
            example_id: "example".to_string(),
            protein_id: "protein".to_string(),
            molecular_representation: None,
            atom_types: vec![0; coords.len()],
            inferred_bonds: infer_bonds(&coords),
            coords,
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 2.5,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: "test".to_string(),
            source_pocket_path: None,
            source_ligand_path: None,
        }
    }

    #[test]
    fn strict_pocket_fit_prefers_centered_candidates() {
        let centered =
            candidate_with_coords(vec![[0.2, 0.0, 0.0], [1.2, 0.0, 0.0], [0.7, 0.8, 0.0]]);
        let shifted =
            candidate_with_coords(vec![[2.8, 0.0, 0.0], [3.8, 0.0, 0.0], [3.3, 0.8, 0.0]]);

        assert!(strict_pocket_fit_score(&centered) > strict_pocket_fit_score(&shifted));
        assert!(centroid_fit_score(&centered) > centroid_fit_score(&shifted));
    }

    #[test]
    fn clash_fraction_ignores_inferred_bonds() {
        let bonded = candidate_with_coords(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert_eq!(non_bonded_clash_fraction(&bonded), 0.0);
    }

    #[test]
    fn repair_candidate_geometry_pushes_apart_close_contacts() {
        let repaired = repair_candidate_geometry(
            &[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            &[],
            [0.0, 0.0, 0.0],
            2.5,
            1,
            3,
        );
        assert!(euclidean(&repaired[0], &repaired[1]) >= 1.0);
        assert!(euclidean(&repaired[1], &repaired[2]) >= 1.0);
    }

    #[test]
    fn prune_bonds_respects_atom_valence_limits() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let bonds = vec![(0, 1), (0, 2), (0, 3)];
        let pruned = prune_bonds_for_valence(&coords, &[4, 0, 0, 0], &bonds);
        assert_eq!(pruned.len(), 1);
    }

    #[test]
    fn command_backend_reports_timeout() {
        let config = ExternalBackendCommandConfig {
            enabled: true,
            executable: Some("sh".to_string()),
            args: vec!["-c".to_string(), "sleep 1".to_string()],
            timeout_ms: 1,
        };
        let report = evaluate_via_command("timeout_backend", &config, &[]);
        assert!(report
            .metrics
            .iter()
            .any(|metric| metric.metric_name == "backend_command_timeout"));
    }
}
