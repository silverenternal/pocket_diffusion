#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::ResearchConfig, data::synthetic_phase1_examples};
    use tch::{nn, Device};

    #[test]
    fn batched_backbone_matches_single_example_backbone_shapes() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let examples = synthetic_phase1_examples()
            .into_iter()
            .map(|example| example.with_pocket_feature_dim(config.model.pocket_feature_dim))
            .collect::<Vec<_>>();

        let single = system.forward_example(&examples[0]);
        let (_, batched) = system.forward_batch(&examples[..2]);
        let from_batch = &batched[0];

        assert_eq!(
            single.encodings.topology.token_embeddings.size(),
            from_batch.encodings.topology.token_embeddings.size()
        );
        assert_eq!(
            single.slots.geometry.slots.size(),
            from_batch.slots.geometry.slots.size()
        );
        assert_eq!(
            single.interactions.geo_from_pocket.attention_weights.size(),
            from_batch
                .interactions
                .geo_from_pocket
                .attention_weights
                .size()
        );
        assert_eq!(
            from_batch.generation.decoded.atom_type_logits.size()[0],
            examples[0].topology.atom_types.size()[0]
        );
    }

    #[test]
    fn stochastic_rollout_sampling_is_reproducible_for_fixed_seed() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.sampling_temperature = 1.0;
        config.data.generation_target.sampling_top_k = 4;
        config.data.generation_target.sampling_top_p = 0.9;
        config.data.generation_target.coordinate_sampling_noise_std = 0.02;
        config.data.generation_target.sampling_seed = 12345;

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let left = system.forward_example(&example).generation.rollout;
        let right = system.forward_example(&example).generation.rollout;

        assert_eq!(left.steps.len(), right.steps.len());
        assert_eq!(
            left.steps.last().unwrap().atom_types,
            right.steps.last().unwrap().atom_types
        );
        assert_eq!(
            left.steps.last().unwrap().coords,
            right.steps.last().unwrap().coords
        );
    }

    #[test]
    fn conditioned_generation_request_exposes_separate_modalities() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let forward = system.forward_example(&example);

        let request = crate::models::ConditionedGenerationRequest::from_forward(
            &example,
            &forward,
            &config.data.generation_target,
        );

        assert_eq!(request.example_id, example.example_id);
        assert_eq!(request.protein_id, example.protein_id);
        assert_eq!(
            request.topology.context.size(),
            forward.generation.state.topology_context.size()
        );
        assert_eq!(
            request.geometry.context.size(),
            forward.generation.state.geometry_context.size()
        );
        assert_eq!(
            request.pocket.context.size(),
            forward.generation.state.pocket_context.size()
        );
        assert!(request.topology.active_slot_fraction >= 0.0);
        assert!(request.geometry.active_slot_fraction >= 0.0);
        assert!(request.pocket.active_slot_fraction >= 0.0);
        assert!(request.gate_summary.topo_from_geo.is_finite());
        assert_eq!(
            request.generation_config.rollout_steps,
            config.data.generation_target.rollout_steps
        );
    }
}
