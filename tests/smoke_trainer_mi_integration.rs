//! Smoke test: MI calculation in Trainer integration

use pocket_diffusion::{
    config::ResearchConfig,
    data::{InMemoryDataset, synthetic_phase1_examples},
    models::Phase1ResearchSystem,
    training::ResearchTrainer,
};
use tch::{nn, Device};

/// Verify that MI is computed and logged during training steps.
#[test]
fn trainer_computes_mi_during_training() {
    let mut config = ResearchConfig::default();
    config.data.batch_size = 2;
    config.training.max_steps = 2;
    config.training.checkpoint_every = 100;
    config.training.log_every = 100;

    let dataset = InMemoryDataset::new(synthetic_phase1_examples())
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    
    let var_store = nn::VarStore::new(Device::Cpu);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let mut trainer = ResearchTrainer::new(&var_store, config).unwrap();

    let metrics = trainer
        .fit(&var_store, &system, dataset.examples())
        .unwrap();

    // Verify MI metrics are present and finite
    assert!(!metrics.is_empty(), "Should have training metrics");
    
    for step_metrics in &metrics {
        let mi = &step_metrics.losses.auxiliaries;
        
        // MI values should be present (not NaN)
        assert!(
            mi.mi_topo_geo.is_finite(),
            "mi_topo_geo should be finite, got {}",
            mi.mi_topo_geo
        );
        assert!(
            mi.mi_topo_pocket.is_finite(),
            "mi_topo_pocket should be finite, got {}",
            mi.mi_topo_pocket
        );
        assert!(
            mi.mi_geo_pocket.is_finite(),
            "mi_geo_pocket should be finite, got {}",
            mi.mi_geo_pocket
        );
        
        // MI should be non-negative
        assert!(
            mi.mi_topo_geo >= 0.0,
            "mi_topo_geo should be non-negative"
        );
        assert!(
            mi.mi_topo_pocket >= 0.0,
            "mi_topo_pocket should be non-negative"
        );
        assert!(
            mi.mi_geo_pocket >= 0.0,
            "mi_geo_pocket should be non-negative"
        );
    }
}

/// Verify that total loss remains finite with MI monitoring enabled.
#[test]
fn trainer_loss_finite_with_mi() {
    let mut config = ResearchConfig::default();
    config.data.batch_size = 2;
    config.training.max_steps = 5;
    config.training.checkpoint_every = 100;
    config.training.log_every = 100;

    let dataset = InMemoryDataset::new(synthetic_phase1_examples())
        .with_pocket_feature_dim(config.model.pocket_feature_dim);
    
    let var_store = nn::VarStore::new(Device::Cpu);
    let system = Phase1ResearchSystem::new(&var_store.root(), &config);
    let mut trainer = ResearchTrainer::new(&var_store, config).unwrap();

    let metrics = trainer
        .fit(&var_store, &system, dataset.examples())
        .unwrap();

    // All total losses should be finite
    for step_metrics in &metrics {
        assert!(
            step_metrics.losses.total.is_finite(),
            "Step {} total loss should be finite, got {}",
            step_metrics.step,
            step_metrics.losses.total
        );
    }
}
