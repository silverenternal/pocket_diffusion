use super::candidates::repair_candidate_geometry;
use super::scoring::{
    centroid_fit_score, euclidean, evaluate_via_command, infer_bonds, non_bonded_clash_fraction,
    prune_bonds_for_valence, strict_pocket_fit_score,
};
use super::*;
use crate::models::Phase1ResearchSystem;
use crate::{config::ResearchConfig, data::synthetic_phase1_examples};
use tch::{nn, Device};

fn candidate_with_coords(coords: Vec<[f32; 3]>) -> GeneratedCandidateRecord {
    GeneratedCandidateRecord {
        example_id: "example".to_string(),
        protein_id: "protein".to_string(),
        molecular_representation: None,
        atom_types: vec![0; coords.len()],
        inferred_bonds: infer_bonds(&coords),
        bond_count: infer_bonds(&coords).len(),
        valence_violation_count: 0,
        coords,
        pocket_centroid: [0.0, 0.0, 0.0],
        pocket_radius: 2.5,
        coordinate_frame_origin: [0.0, 0.0, 0.0],
        source: "test".to_string(),
        generation_mode: "target_ligand_denoising".to_string(),
        generation_layer: "raw_flow".to_string(),
        generation_path_class: "model_native_raw".to_string(),
        model_native_raw: true,
        postprocessor_chain: Vec::new(),
        claim_boundary:
            "raw model-native output before repair, constraints, reranking, or backend scoring"
                .to_string(),
        source_pocket_path: None,
        source_ligand_path: None,
    }
}

#[test]
fn strict_pocket_fit_prefers_centered_candidates() {
    let centered = candidate_with_coords(vec![[0.2, 0.0, 0.0], [1.2, 0.0, 0.0], [0.7, 0.8, 0.0]]);
    let shifted = candidate_with_coords(vec![[2.8, 0.0, 0.0], [3.8, 0.0, 0.0], [3.3, 0.8, 0.0]]);

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

#[test]
fn layered_candidates_with_repair_disable_disable_repair_preserves_coordinates() {
    let config = ResearchConfig::default();
    let example = synthetic_phase1_examples()
        .into_iter()
        .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
        .next()
        .unwrap();
    let var_store = nn::VarStore::new(Device::Cpu);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let forward = system.forward_example(&example);

    let layers_no_repair = generate_layered_candidates_with_options(&example, &forward, 2, false);
    for (raw, repaired) in layers_no_repair
        .raw_rollout
        .iter()
        .zip(layers_no_repair.repaired.iter())
    {
        assert_eq!(raw.coords, repaired.coords);
        assert_eq!(raw.inferred_bonds, repaired.inferred_bonds);
    }

    let layers_with_repair = generate_layered_candidates_with_options(&example, &forward, 2, true);
    assert_eq!(
        layers_with_repair.raw_rollout[0].generation_layer,
        "raw_flow"
    );
    assert_eq!(
        layers_with_repair.raw_rollout[0].generation_path_class,
        "model_native_raw"
    );
    assert!(layers_with_repair.raw_rollout[0].model_native_raw);
    assert!(layers_with_repair.raw_rollout[0]
        .postprocessor_chain
        .is_empty());
    assert_eq!(
        layers_with_repair.inferred_bond[0].generation_layer,
        "constrained_flow"
    );
    assert_eq!(
        layers_with_repair.inferred_bond[0].generation_path_class,
        "constrained"
    );
    assert!(!layers_with_repair.inferred_bond[0].model_native_raw);
    assert!(layers_with_repair.inferred_bond[0]
        .postprocessor_chain
        .iter()
        .any(|step| step == "distance_bond_inference"));
    let first_repair_delta = layers_with_repair.repaired[0]
        .coords
        .iter()
        .zip(layers_with_repair.raw_rollout[0].coords.iter())
        .any(|(repaired, raw)| {
            (repaired[0] - raw[0]).abs() > 1e-6
                || (repaired[1] - raw[1]).abs() > 1e-6
                || (repaired[2] - raw[2]).abs() > 1e-6
        });
    assert!(
        first_repair_delta,
        "repair-enabled and repair-disabled coordinates must differ"
    );
}

#[test]
fn flow_generation_layers_keep_raw_native_constrained_and_repaired_graphs_separate() {
    let mut config = ResearchConfig::default();
    config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
    config.generation_method.active_method = "flow_matching".to_string();
    config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
        backend_id: "flow_matching".to_string(),
        family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
        trainable: true,
        ..crate::config::GenerationBackendConfig::default()
    };
    config.generation_method.flow_matching.geometry_only = false;
    config
        .generation_method
        .flow_matching
        .multi_modal
        .enabled_branches = vec![
        crate::config::FlowBranchKind::Geometry,
        crate::config::FlowBranchKind::AtomType,
        crate::config::FlowBranchKind::Bond,
        crate::config::FlowBranchKind::Topology,
        crate::config::FlowBranchKind::PocketContext,
    ];
    let example = synthetic_phase1_examples()
        .into_iter()
        .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
        .next()
        .unwrap();
    let var_store = nn::VarStore::new(Device::Cpu);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let forward = system.forward_example(&example);
    let final_step = forward.generation.rollout.steps.last().unwrap();
    let layers = generate_layered_candidates_with_options(&example, &forward, 1, true);

    assert_eq!(
        final_step.native_graph_provenance.raw_logits_layer,
        "raw_molecular_flow_logits"
    );
    assert_eq!(
        final_step
            .native_graph_provenance
            .raw_native_extraction_layer,
        "raw_native_graph_extraction"
    );
    assert_eq!(
        final_step.native_graph_provenance.constrained_graph_layer,
        "constrained_native_graph"
    );
    assert_eq!(
        layers.raw_rollout[0].inferred_bonds,
        final_step.native_bonds
    );
    assert_eq!(
        layers.bond_logits_refined[0].inferred_bonds,
        final_step.constrained_native_bonds
    );
    assert!(layers.raw_rollout[0].model_native_raw);
    assert!(layers.raw_rollout[0].postprocessor_chain.is_empty());
    assert!(!layers.bond_logits_refined[0].model_native_raw);
    assert_eq!(
        layers.bond_logits_refined[0].generation_path_class,
        "constrained"
    );
    assert!(layers.bond_logits_refined[0]
        .postprocessor_chain
        .iter()
        .any(|step| step == "bond_logits_refinement_no_coordinate_move"));
    assert_eq!(layers.repaired[0].generation_path_class, "repaired");
    assert!(layers.repaired[0]
        .postprocessor_chain
        .iter()
        .any(|step| step == "pocket_centroid_repair"));
}

#[test]
fn claim_facing_generator_uses_inferred_bond_layer() {
    let config = ResearchConfig::default();
    let example = synthetic_phase1_examples()
        .into_iter()
        .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
        .next()
        .unwrap();
    let var_store = nn::VarStore::new(Device::Cpu);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let forward = system.forward_example(&example);

    let layered = generate_layered_candidates_with_options(&example, &forward, 3, true);
    let claim_facing = generate_claim_facing_candidates_from_forward(&example, &forward, 3);

    assert_eq!(claim_facing.len(), layered.inferred_bond.len());
    for (claim, inferred) in claim_facing.iter().zip(&layered.inferred_bond) {
        assert_eq!(claim.example_id, inferred.example_id);
        assert_eq!(claim.protein_id, inferred.protein_id);
        assert_eq!(claim.coords, inferred.coords);
        assert_eq!(claim.atom_types, inferred.atom_types);
        assert_eq!(claim.inferred_bonds, inferred.inferred_bonds);
        assert_eq!(
            claim.coordinate_frame_origin,
            inferred.coordinate_frame_origin
        );
    }
}
