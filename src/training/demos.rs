//! Standalone demo runners for the modular research stack.

use tch::nn;

use crate::{
    config::ResearchConfig,
    data::{synthetic_phase1_examples, Dataset, InMemoryDataset},
    experiments::{UnseenPocketExperiment, UnseenPocketExperimentConfig},
    models::Phase1ResearchSystem,
};

use super::{print_eval_metrics, print_step_metrics, ResearchTrainer};

/// Run the Phase 1 architecture smoke demo and print tensor shapes plus gates.
pub fn run_phase1_demo() {
    println!("================================================");
    println!("  Phase 1 Research Architecture Demo");
    println!("================================================");

    let config = ResearchConfig::default();
    let device = config
        .runtime
        .resolve_device()
        .expect("phase1 demo should resolve runtime device");
    let dataset = InMemoryDataset::new(synthetic_phase1_examples())
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    let splits = dataset.split_by_protein(3, 5);
    let vs = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&vs.root(), &config);

    let train_examples = splits.train.examples();
    let (batch, outputs) = system.forward_batch(train_examples);

    println!("dataset:");
    println!("  train examples: {}", splits.train.len());
    println!("  val examples: {}", splits.val.len());
    println!("  test examples: {}", splits.test.len());
    println!("runtime:");
    println!("  device: {}", config.runtime.device);

    println!("batch:");
    println!(
        "  ligand atom tensor: {:?}",
        batch.encoder_inputs.atom_types.size()
    );
    println!(
        "  ligand coord tensor: {:?}",
        batch.encoder_inputs.ligand_coords.size()
    );
    println!(
        "  pocket feature tensor: {:?}",
        batch.encoder_inputs.pocket_atom_features.size()
    );

    if let Some(first) = outputs.first() {
        println!("encodings:");
        println!(
            "  topology pooled: {:?}, tokens: {:?}",
            first.encodings.topology.pooled_embedding.size(),
            first.encodings.topology.token_embeddings.size()
        );
        println!(
            "  geometry pooled: {:?}, tokens: {:?}",
            first.encodings.geometry.pooled_embedding.size(),
            first.encodings.geometry.token_embeddings.size()
        );
        println!(
            "  pocket pooled: {:?}, tokens: {:?}",
            first.encodings.pocket.pooled_embedding.size(),
            first.encodings.pocket.token_embeddings.size()
        );

        println!("slots:");
        println!(
            "  topology slots: {:?}, weights: {:?}",
            first.slots.topology.slots.size(),
            first.slots.topology.slot_weights.size()
        );
        println!(
            "  geometry slots: {:?}, weights: {:?}",
            first.slots.geometry.slots.size(),
            first.slots.geometry.slot_weights.size()
        );
        println!(
            "  pocket slots: {:?}, weights: {:?}",
            first.slots.pocket.slots.size(),
            first.slots.pocket.slot_weights.size()
        );

        println!("gates:");
        println!(
            "  topo<-geo: {:.4}, topo<-pocket: {:.4}",
            first.interactions.topo_from_geo.gate.double_value(&[0]),
            first.interactions.topo_from_pocket.gate.double_value(&[0])
        );
        println!(
            "  geo<-topo: {:.4}, geo<-pocket: {:.4}",
            first.interactions.geo_from_topo.gate.double_value(&[0]),
            first.interactions.geo_from_pocket.gate.double_value(&[0])
        );
        println!(
            "  pocket<-topo: {:.4}, pocket<-geo: {:.4}",
            first.interactions.pocket_from_topo.gate.double_value(&[0]),
            first.interactions.pocket_from_geo.gate.double_value(&[0])
        );

        println!("probes:");
        println!(
            "  topology adjacency logits: {:?}",
            first.probes.topology_adjacency_logits.size()
        );
        println!(
            "  geometry distance predictions: {:?}",
            first.probes.geometry_distance_predictions.size()
        );
        println!(
            "  pocket feature predictions: {:?}",
            first.probes.pocket_feature_predictions.size()
        );
    }
}

/// Run a short staged training demo over synthetic data.
pub fn run_phase3_training_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================");
    println!("  Phase 3 Staged Training Demo");
    println!("================================================");

    let mut config = ResearchConfig::default();
    config.training.max_steps = 4;
    config.training.checkpoint_every = 100;
    config.training.log_every = 1;
    let device = config
        .runtime
        .resolve_device()
        .expect("phase3 demo should resolve runtime device");

    let dataset = InMemoryDataset::new(synthetic_phase1_examples())
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    let train_examples = dataset
        .examples()
        .iter()
        .map(|example| example.to_device(device))
        .collect::<Vec<_>>();

    let vs = nn::VarStore::new(device);
    let system = Phase1ResearchSystem::new(&vs.root(), &config);
    let mut trainer = ResearchTrainer::new(&vs, config).expect("trainer init should succeed");

    println!("runtime:");
    println!("  device: {:?}", device);
    for metrics in trainer.fit(&vs, &system, &train_examples)? {
        print_step_metrics(&metrics);
    }
    Ok(())
}

/// Run the Phase 4 unseen-pocket experiment demo with a compact summary.
pub fn run_phase4_experiment_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================");
    println!("  Phase 4 Unseen-Pocket Experiment Demo");
    println!("================================================");

    let mut config = UnseenPocketExperimentConfig::default();
    config.research.training.max_steps = 4;
    config.research.training.checkpoint_every = 100;
    println!("runtime:");
    println!("  device: {}", config.research.runtime.device);

    let summary = UnseenPocketExperiment::run_with_options(config, false)
        .expect("phase4 experiment should succeed");

    println!("training:");
    println!("  steps: {}", summary.training_history.len());
    if let Some(last) = summary.training_history.last() {
        println!(
            "  last stage: {:?}, last total loss: {:.4}",
            last.stage, last.losses.total
        );
    }

    println!("validation:");
    print_eval_metrics(&summary.validation);

    println!("test:");
    print_eval_metrics(&summary.test);
    Ok(())
}
