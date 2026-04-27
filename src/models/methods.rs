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
    fn promoted_backend_families_emit_distinct_runnable_candidates() {
        let config = crate::config::ResearchConfig::default();
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let system = crate::models::Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = crate::data::synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let forward = system.forward_example(&example);

        for (method_id, expected_representation) in [
            (FLOW_MATCHING_METHOD_ID, "flow_matching_geometry_v1"),
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
                        .map(|repr| repr.starts_with(expected_representation))
                        .unwrap_or(false)
                    && candidate.atom_types.len() == candidate.coords.len()
            }));
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
