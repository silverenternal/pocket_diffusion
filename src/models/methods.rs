//! Method registry and concrete generation-method implementations.
//!
//! Registry, method implementations, layered output summaries, and reranking
//! helpers are split under a compatibility facade.

use std::collections::{BTreeMap, BTreeSet};

use super::evaluation::{generate_layered_candidates_with_options, CandidateGenerationLayers};
use super::{
    CandidateLayerKind, CandidateLayerOutput, CandidateLayerProvenance, GeneratedCandidateRecord,
    GenerationEvidenceRole, GenerationExecutionMode, GenerationMethodCapability,
    LayeredGenerationOutput, PocketGenerationContext, PocketGenerationMethod,
    PocketGenerationMethodFamily, PocketGenerationMethodMetadata,
};

include!("methods/registry.rs");
include!("methods/implementations.rs");
include!("methods/layers.rs");
include!("methods/reranking.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_backends_expose_metadata_without_silent_empty_rows() {
        for method_id in PocketGenerationMethodRegistry::registered_method_ids() {
            let metadata = PocketGenerationMethodRegistry::metadata(method_id).unwrap();
            let output = LayeredGenerationOutput::empty(metadata.clone());
            let row = summarize_method_output(&output);

            if metadata.capability.stub {
                assert_eq!(metadata.execution_mode, GenerationExecutionMode::Stub);
                assert!(!metadata.method_id.is_empty());
            } else {
                assert!(
                    !metadata.layered_output_support.is_empty(),
                    "{method_id} must declare supported layers"
                );
                assert!(
                    row.supported_layers.len() == metadata.layered_output_support.len(),
                    "{method_id} must report declared layer support"
                );
            }
        }
    }

    #[test]
    fn backend_config_family_must_match_registry_metadata() {
        let valid = crate::config::GenerationBackendConfig {
            backend_id: FLOW_MATCHING_METHOD_ID.to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            checkpoint_path: None,
            sampling_steps: Some(4),
            sampling_temperature: Some(0.0),
            external_wrapper: crate::config::ExternalBackendCommandConfig::default(),
        };
        PocketGenerationMethodRegistry::validate_backend_config(&valid).unwrap();

        let invalid = crate::config::GenerationBackendConfig {
            family: crate::config::GenerationBackendFamilyConfig::Diffusion,
            ..valid
        };
        assert!(PocketGenerationMethodRegistry::validate_backend_config(&invalid).is_err());
    }

    #[test]
    fn native_metric_layer_selection_is_method_family_aware() {
        let base =
            comparison_row_with_layers(CONDITIONED_DENOISING_METHOD_ID, true, false, false, false);
        assert_eq!(base.method_family, "conditioneddenoising");
        assert_eq!(base.selected_metric_layer.as_deref(), Some("raw_rollout"));
        assert!(base.selected_metric_layer_model_native_raw);

        let repair = comparison_row_with_layers(REPAIR_ONLY_METHOD_ID, false, true, false, false);
        assert_eq!(repair.method_family, "repaironly");
        assert_eq!(
            repair.selected_metric_layer.as_deref(),
            Some("repaired_candidates")
        );
        assert_eq!(
            repair.selected_metric_layer_path_class.as_deref(),
            Some("repaired")
        );
        assert!(!repair.selected_metric_layer_model_native_raw);

        let deterministic_proxy = comparison_row_with_layers(
            DETERMINISTIC_PROXY_RERANKER_METHOD_ID,
            false,
            true,
            true,
            false,
        );
        assert_eq!(deterministic_proxy.method_family, "rerankeronly");
        assert_eq!(
            deterministic_proxy.selected_metric_layer.as_deref(),
            Some("deterministic_proxy_candidates")
        );
        assert_eq!(
            deterministic_proxy
                .selected_metric_layer_path_class
                .as_deref(),
            Some("reranked")
        );

        let calibrated =
            comparison_row_with_layers(CALIBRATED_RERANKER_METHOD_ID, false, true, true, true);
        assert_eq!(calibrated.method_family, "rerankeronly");
        assert_eq!(
            calibrated.selected_metric_layer.as_deref(),
            Some("reranked_candidates")
        );

        let hybrid =
            comparison_row_with_layers(CONDITIONED_DENOISING_METHOD_ID, true, true, true, true);
        assert_eq!(hybrid.selected_metric_layer.as_deref(), Some("raw_rollout"));
        assert_eq!(
            hybrid.selected_metric_layer_path_class.as_deref(),
            Some("model_native_raw")
        );
        assert!(hybrid.selected_metric_layer_model_native_raw);
    }

    fn comparison_row_with_layers(
        method_id: &str,
        native_raw: bool,
        include_repaired: bool,
        include_deterministic_proxy: bool,
        include_reranked: bool,
    ) -> MethodComparisonRow {
        let metadata = PocketGenerationMethodRegistry::metadata(method_id).unwrap();
        let mut output = LayeredGenerationOutput::empty(metadata.clone());
        output.raw_rollout = Some(layer_output(
            &metadata,
            CandidateLayerKind::RawRollout,
            native_raw,
            Vec::new(),
            vec![method_test_candidate("raw")],
        ));
        if include_repaired {
            output.repaired = Some(layer_output(
                &metadata,
                CandidateLayerKind::Repaired,
                false,
                vec!["pocket_centroid_repair".to_string()],
                vec![method_test_candidate("repaired")],
            ));
        }
        if include_deterministic_proxy {
            output.deterministic_proxy = Some(layer_output(
                &metadata,
                CandidateLayerKind::DeterministicProxy,
                false,
                vec!["deterministic_proxy_rerank".to_string()],
                vec![method_test_candidate("deterministic_proxy")],
            ));
        }
        if include_reranked {
            output.reranked = Some(layer_output(
                &metadata,
                CandidateLayerKind::Reranked,
                false,
                vec!["bounded_calibrated_rerank".to_string()],
                vec![method_test_candidate("reranked")],
            ));
        }

        let row = summarize_method_output(&output);
        let serialized = serde_json::to_value(&row).unwrap();
        assert!(serialized.get("selected_metric_layer").is_some());
        assert!(serialized.get("method_family").is_some());
        row
    }

    fn method_test_candidate(source: &str) -> GeneratedCandidateRecord {
        GeneratedCandidateRecord {
            example_id: "example".to_string(),
            protein_id: "protein".to_string(),
            molecular_representation: None,
            atom_types: vec![6, 8],
            coords: vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            inferred_bonds: vec![(0, 1)],
            bond_count: 1,
            valence_violation_count: 0,
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 4.0,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: source.to_string(),
            generation_mode: crate::config::GenerationModeConfig::TargetLigandDenoising
                .as_str()
                .to_string(),
            generation_layer: "unassigned".to_string(),
            generation_path_class: "unassigned".to_string(),
            model_native_raw: false,
            postprocessor_chain: Vec::new(),
            claim_boundary: String::new(),
            source_pocket_path: None,
            source_ligand_path: None,
        }
    }

    #[test]
    fn promoted_backend_families_emit_distinct_runnable_candidates() {
        let mut config = crate::config::ResearchConfig::default();
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = FLOW_MATCHING_METHOD_ID.to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: FLOW_MATCHING_METHOD_ID.to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        config.generation_method.flow_matching.geometry_only = false;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .enabled_branches = crate::config::FlowBranchKind::ALL.to_vec();
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let system = crate::models::Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = crate::data::synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let forward = system.forward_example(&example);
        let raw_native_decoder_bonds = forward
            .generation
            .rollout
            .steps
            .last()
            .map(|step| step.native_bonds.clone())
            .unwrap_or_default();
        let constrained_native_decoder_bonds = forward
            .generation
            .rollout
            .steps
            .last()
            .map(|step| step.constrained_native_bonds.clone())
            .unwrap_or_default();

        for (method_id, expected_representation) in [
            (FLOW_MATCHING_METHOD_ID, "method=flow_matching"),
            (
                AUTOREGRESSIVE_GRAPH_GEOMETRY_METHOD_ID,
                "autoregressive_graph_geometry_policy_v1",
            ),
            (
                ENERGY_GUIDED_REFINEMENT_METHOD_ID,
                "energy_guided_refinement_v1",
            ),
        ] {
            let method = PocketGenerationMethodRegistry::build(method_id).unwrap();
            let context = PocketGenerationContext {
                example: example.clone(),
                conditioned_request: None,
                forward: Some(forward.clone()),
                candidate_limit: 2,
                enable_repair: true,
            }
            .with_conditioned_request(&config.data.generation_target);

            let output = method.generate_for_example(context);
            let raw = output
                .raw_rollout
                .as_ref()
                .expect("promoted backend should emit raw candidates");
            assert!(!raw.candidates.is_empty());
            assert!(raw.candidates.iter().all(|candidate| {
                candidate.source == method_id
                    && candidate
                        .molecular_representation
                        .as_deref()
                        .map(|repr| repr.contains(expected_representation))
                        .unwrap_or(false)
                    && candidate.atom_types.len() == candidate.coords.len()
            }));
            if method_id == FLOW_MATCHING_METHOD_ID {
                assert!(raw.candidates.iter().all(|candidate| candidate
                    .molecular_representation
                    .as_deref()
                    .is_some_and(|repr| repr.contains("source=raw_rollout"))));
                assert_eq!(
                    raw.candidates[0].inferred_bonds, raw_native_decoder_bonds,
                    "flow matching raw rollout must preserve raw native-decoder bonds instead of distance-reinferring them"
                );
                let constrained = output
                    .bond_logits_refined
                    .as_ref()
                    .expect("flow matching method should expose constrained native graph layer");
                assert_eq!(
                    constrained.candidates[0].inferred_bonds,
                    constrained_native_decoder_bonds,
                    "flow matching constrained layer must preserve constrained native-decoder bonds separately from raw rollout"
                );
            }
        }
    }

    #[test]
    fn external_wrapper_dry_run_uses_jsonl_contract_shape() {
        let config = crate::config::ResearchConfig::default();
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let system = crate::models::Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = crate::data::synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let forward = system.forward_example(&example);
        let method =
            PocketGenerationMethodRegistry::build(EXTERNAL_WRAPPER_DRY_RUN_METHOD_ID).unwrap();
        let context = PocketGenerationContext {
            example,
            conditioned_request: None,
            forward: Some(forward),
            candidate_limit: 2,
            enable_repair: false,
        }
        .with_conditioned_request(&config.data.generation_target);

        let output = method.generate_for_example(context);
        let raw = output.raw_rollout.expect("dry-run wrapper emits raw layer");
        assert_eq!(raw.candidates.len(), 2);
        assert!(raw.candidates.iter().all(|candidate| {
            candidate.source == EXTERNAL_WRAPPER_DRY_RUN_METHOD_ID
                && candidate.molecular_representation.as_deref()
                    == Some("external_wrapper_dry_run_jsonl_v1")
        }));
    }
}
