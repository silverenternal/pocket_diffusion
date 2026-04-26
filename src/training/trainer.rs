//! Runnable staged trainer for the new research stack.

use std::collections::BTreeMap;

use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::{
    config::{AffinityWeighting, ResearchConfig},
    data::{ExampleBatchIter, MolecularExample},
    losses::{
        build_primary_objective, compute_primary_objective_batch, ConsistencyLoss, GateLoss,
        IntraRedundancyLoss, LeakageLoss, PocketGeometryAuxLoss, ProbeLoss,
    },
    models::{Phase1ResearchSystem, ResearchForward, TaskDrivenObjective},
    training::{stable_json_hash, METRIC_SCHEMA_VERSION, RESUME_CONTRACT_VERSION},
};

use super::{
    AuxiliaryLossMetrics, BackendTrainingMetadata, CheckpointManager, LoadedCheckpoint,
    LossBreakdown, OptimizerStateMetadata, PrimaryObjectiveMetrics, SchedulerStateMetadata,
    StageScheduler, StepMetrics, TrainingStage,
};

/// Trainer that applies staged auxiliary losses to the new modular system.
pub struct ResearchTrainer {
    optimizer: nn::Optimizer,
    scheduler: StageScheduler,
    checkpoints: CheckpointManager,
    primary_objective: Box<dyn TaskDrivenObjective<ResearchForward>>,
    redundancy_loss: IntraRedundancyLoss,
    probe_loss: ProbeLoss,
    leakage_loss: LeakageLoss,
    gate_loss: GateLoss,
    consistency_loss: ConsistencyLoss,
    pocket_geometry_loss: PocketGeometryAuxLoss,
    config: ResearchConfig,
    dataset_validation_fingerprint: Option<String>,
    affinity_measurement_weights: BTreeMap<String, f64>,
    step: usize,
    history: Vec<StepMetrics>,
    last_stage: Option<TrainingStage>,
    restored_optimizer_state: Option<OptimizerStateMetadata>,
    restored_scheduler_state: Option<SchedulerStateMetadata>,
}

impl ResearchTrainer {
    /// Create a new trainer from the shared var store and config.
    pub fn new(
        var_store: &nn::VarStore,
        config: ResearchConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let optimizer = nn::Adam::default().build(var_store, config.training.learning_rate)?;
        let scheduler = StageScheduler::new(
            config.training.schedule.clone(),
            config.training.loss_weights.clone(),
        );
        let checkpoints = CheckpointManager::new(config.training.checkpoint_dir.clone());

        Ok(Self {
            optimizer,
            scheduler,
            checkpoints,
            primary_objective: build_primary_objective(config.training.primary_objective),
            redundancy_loss: IntraRedundancyLoss::default(),
            probe_loss: ProbeLoss,
            leakage_loss: LeakageLoss::default(),
            gate_loss: GateLoss,
            consistency_loss: ConsistencyLoss::default(),
            pocket_geometry_loss: PocketGeometryAuxLoss::default(),
            config,
            dataset_validation_fingerprint: None,
            affinity_measurement_weights: BTreeMap::new(),
            step: 0,
            history: Vec::new(),
            last_stage: None,
            restored_optimizer_state: None,
            restored_scheduler_state: None,
        })
    }

    /// Resume from the latest checkpoint in the configured directory.
    pub fn resume_from_latest(
        &mut self,
        var_store: &mut nn::VarStore,
    ) -> Result<Option<LoadedCheckpoint>, Box<dyn std::error::Error>> {
        let checkpoint = self.checkpoints.load_latest(var_store)?;
        if let Some(loaded) = &checkpoint {
            self.validate_checkpoint_compatibility(&loaded.metadata)?;
            if let Some(metrics) = loaded.metadata.metrics.clone() {
                self.history.push(metrics);
            }
            self.restored_optimizer_state = loaded.metadata.optimizer_state.clone();
            self.restored_scheduler_state = loaded.metadata.scheduler_state.clone();
            if let Some(optimizer_state) = &self.restored_optimizer_state {
                self.optimizer.set_lr(optimizer_state.learning_rate);
                self.optimizer
                    .set_weight_decay(optimizer_state.weight_decay);
            }
            self.step = loaded.metadata.step.saturating_add(1);
            self.last_stage = Some(self.scheduler.stage_for_step(loaded.metadata.step));
        }
        Ok(checkpoint)
    }

    /// Run one optimization step over a mini-batch of examples.
    pub fn train_batch_step(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        examples: &[MolecularExample],
    ) -> Result<StepMetrics, Box<dyn std::error::Error>> {
        if examples.is_empty() {
            return Err("cannot train on an empty mini-batch".into());
        }
        let (_, forwards) = system.forward_batch(examples);
        let weights = self.scheduler.weights_for_step(self.step);
        let stage = self.scheduler.stage_for_step(self.step);
        if self.last_stage != Some(stage) {
            log::info!(
                "entering {:?} at step {} with weights primary={:.4} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4} pocket_contact={:.4} pocket_clash={:.4}",
                stage,
                self.step,
                weights.primary,
                weights.intra_red,
                weights.probe,
                weights.leak,
                weights.gate,
                weights.slot,
                weights.consistency,
                weights.pocket_contact,
                weights.pocket_clash,
            );
            self.last_stage = Some(stage);
        }

        self.affinity_measurement_weights =
            measurement_weights(examples, self.config.training.affinity_weighting);
        let primary =
            compute_primary_objective_batch(self.primary_objective.as_ref(), examples, &forwards);
        let intra_red = self.redundancy_loss.compute_batch(&forwards);
        let probe = self
            .probe_loss
            .compute_batch_weighted(examples, &forwards, |example| {
                self.affinity_weight_for(example)
            });
        let leak = self.leakage_loss.compute_batch(examples, &forwards);
        let gate = self.gate_loss.compute_batch(&forwards);
        let slot = slot_loss_from_batch(&forwards, var_store.device());
        let consistency = self.consistency_loss.compute_batch(examples, &forwards);
        let (pocket_contact, pocket_clash) =
            self.pocket_geometry_loss.compute_batch(examples, &forwards);

        let total = &primary * weights.primary
            + &intra_red * weights.intra_red
            + &probe * weights.probe
            + &leak * weights.leak
            + &gate * weights.gate
            + &slot * weights.slot
            + &consistency * weights.consistency
            + &pocket_contact * weights.pocket_contact
            + &pocket_clash * weights.pocket_clash;

        let primary_value = scalar_or_nan(&primary);
        let intra_red_value = scalar_or_nan(&intra_red);
        let probe_value = scalar_or_nan(&probe);
        let leak_value = scalar_or_nan(&leak);
        let gate_value = scalar_or_nan(&gate);
        let slot_value = scalar_or_nan(&slot);
        let consistency_value = scalar_or_nan(&consistency);
        let pocket_contact_value = scalar_or_nan(&pocket_contact);
        let pocket_clash_value = scalar_or_nan(&pocket_clash);
        let total_value = scalar_or_nan(&total);

        self.optimizer.zero_grad();
        if tensor_is_finite(&total) {
            total.backward();
            self.optimizer.step();
        } else {
            log::warn!(
                "skipping optimizer step {} because total loss became non-finite",
                self.step
            );
        }

        let metrics = StepMetrics {
            step: self.step,
            stage,
            losses: LossBreakdown {
                primary: PrimaryObjectiveMetrics {
                    objective_name: self.primary_objective.name().to_string(),
                    primary_value,
                    decoder_anchored: self.primary_objective.name() != "surrogate_reconstruction",
                },
                auxiliaries: AuxiliaryLossMetrics {
                    intra_red: intra_red_value,
                    probe: probe_value,
                    leak: leak_value,
                    gate: gate_value,
                    slot: slot_value,
                    consistency: consistency_value,
                    pocket_contact: pocket_contact_value,
                    pocket_clash: pocket_clash_value,
                },
                total: total_value,
            },
        };

        if self.config.training.checkpoint_every > 0
            && self.step % self.config.training.checkpoint_every == 0
        {
            self.checkpoints.save(
                var_store,
                self.step,
                Some(&metrics),
                Some(stable_json_hash(&self.config)),
                self.dataset_validation_fingerprint.clone(),
                METRIC_SCHEMA_VERSION,
                RESUME_CONTRACT_VERSION,
                Some(self.optimizer_state_metadata()),
                Some(self.scheduler_state_metadata(self.step)),
                Some(self.backend_training_metadata()),
            )?;
        }
        if self.config.training.log_every > 0 && self.step % self.config.training.log_every == 0 {
            log::info!(
                "step {} [{:?}] total={:.4} primary:{}={:.4} decoder_anchor={} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4} pocket_contact={:.4} pocket_clash={:.4}",
                metrics.step,
                metrics.stage,
                metrics.losses.total,
                metrics.losses.primary.objective_name,
                metrics.losses.primary.primary_value,
                metrics.losses.primary.decoder_anchored,
                metrics.losses.auxiliaries.intra_red,
                metrics.losses.auxiliaries.probe,
                metrics.losses.auxiliaries.leak,
                metrics.losses.auxiliaries.gate,
                metrics.losses.auxiliaries.slot,
                metrics.losses.auxiliaries.consistency,
                metrics.losses.auxiliaries.pocket_contact,
                metrics.losses.auxiliaries.pocket_clash,
            );
        }
        self.history.push(metrics.clone());
        self.step += 1;
        Ok(metrics)
    }

    /// Compatibility wrapper for older call sites.
    pub fn train_step(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        examples: &[MolecularExample],
    ) -> Result<StepMetrics, Box<dyn std::error::Error>> {
        self.train_batch_step(var_store, system, examples)
    }

    /// Train over deterministic mini-batches until `max_steps` is reached.
    pub fn fit(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        examples: &[MolecularExample],
    ) -> Result<Vec<StepMetrics>, Box<dyn std::error::Error>> {
        if examples.is_empty() {
            return Err("training examples are empty".into());
        }

        let mut steps = Vec::new();
        while self.step < self.config.training.max_steps {
            let mut progressed = false;
            for batch in ExampleBatchIter::new(examples, self.config.data.batch_size) {
                if self.step >= self.config.training.max_steps {
                    break;
                }
                let metrics = self.train_batch_step(var_store, system, batch)?;
                steps.push(metrics);
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        Ok(steps)
    }

    /// Borrow trainer history.
    pub fn history(&self) -> &[StepMetrics] {
        &self.history
    }

    /// Replace the in-memory history, typically after restoring a prior run summary.
    pub fn replace_history(&mut self, history: Vec<StepMetrics>) {
        self.history = history;
    }

    /// Current training stage.
    pub fn stage(&self) -> TrainingStage {
        self.scheduler.stage_for_step(self.step)
    }

    /// Next global step index.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Checkpoint manager used by this trainer.
    pub fn checkpoints(&self) -> &CheckpointManager {
        &self.checkpoints
    }

    /// Optimizer resume metadata restored from the latest checkpoint, when present.
    pub fn restored_optimizer_state(&self) -> Option<&OptimizerStateMetadata> {
        self.restored_optimizer_state.as_ref()
    }

    /// Scheduler resume metadata restored from the latest checkpoint, when present.
    pub fn restored_scheduler_state(&self) -> Option<&SchedulerStateMetadata> {
        self.restored_scheduler_state.as_ref()
    }

    /// Attach the dataset validation fingerprint used by the current run.
    pub fn set_dataset_validation_fingerprint(&mut self, fingerprint: String) {
        self.dataset_validation_fingerprint = Some(fingerprint);
    }

    fn affinity_weight_for(&self, example: &MolecularExample) -> f64 {
        let measurement = example
            .targets
            .affinity_measurement_type
            .as_deref()
            .unwrap_or("unknown");
        self.affinity_measurement_weights
            .get(measurement)
            .copied()
            .unwrap_or(1.0)
    }

    fn optimizer_state_metadata(&self) -> OptimizerStateMetadata {
        OptimizerStateMetadata {
            optimizer_kind: "adam".to_string(),
            learning_rate: self.config.training.learning_rate,
            weight_decay: 0.0,
            internal_state_persisted: false,
        }
    }

    fn scheduler_state_metadata(&self, step: usize) -> SchedulerStateMetadata {
        let stage = self.scheduler.stage_for_step(step);
        let weights = self.scheduler.weights_for_step(step);
        SchedulerStateMetadata {
            stage: format!("{stage:?}"),
            primary_weight: weights.primary,
            intra_red_weight: weights.intra_red,
            probe_weight: weights.probe,
            leak_weight: weights.leak,
            gate_weight: weights.gate,
            slot_weight: weights.slot,
            consistency_weight: weights.consistency,
            pocket_contact_weight: weights.pocket_contact,
            pocket_clash_weight: weights.pocket_clash,
        }
    }

    fn backend_training_metadata(&self) -> BackendTrainingMetadata {
        let backend_id = self
            .config
            .generation_method
            .primary_backend_id()
            .to_string();
        let metadata = crate::models::PocketGenerationMethodRegistry::metadata(&backend_id).ok();
        BackendTrainingMetadata {
            backend_id,
            backend_family: metadata
                .as_ref()
                .map(|metadata| format!("{:?}", metadata.method_family).to_ascii_lowercase())
                .unwrap_or_else(|| "unknown".to_string()),
            objective_name: self.primary_objective.name().to_string(),
            trainable_backend: metadata
                .as_ref()
                .map(|metadata| metadata.capability.trainable)
                .unwrap_or(false),
            shared_auxiliary_objectives: vec![
                "L_consistency".to_string(),
                "L_intra_red".to_string(),
                "L_probe".to_string(),
                "L_leak".to_string(),
                "L_gate".to_string(),
                "L_slot".to_string(),
            ],
        }
    }

    fn validate_checkpoint_compatibility(
        &self,
        checkpoint: &crate::training::CheckpointMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(saved) = checkpoint.backend_training.as_ref() else {
            return Ok(());
        };
        let current = self.backend_training_metadata();
        if saved.backend_id != current.backend_id
            || saved.backend_family != current.backend_family
            || saved.objective_name != current.objective_name
        {
            return Err(format!(
                "checkpoint backend/objective mismatch: saved {}:{}:{} but current {}:{}:{}",
                saved.backend_id,
                saved.backend_family,
                saved.objective_name,
                current.backend_id,
                current.backend_family,
                current.objective_name
            )
            .into());
        }
        Ok(())
    }
}

fn measurement_weights(
    examples: &[MolecularExample],
    strategy: AffinityWeighting,
) -> BTreeMap<String, f64> {
    if strategy == AffinityWeighting::None {
        return BTreeMap::new();
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for example in examples {
        if example.targets.affinity_kcal_mol.is_some() {
            let measurement = example
                .targets
                .affinity_measurement_type
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            *counts.entry(measurement).or_default() += 1;
        }
    }

    if counts.is_empty() {
        return BTreeMap::new();
    }

    let total = counts.values().sum::<usize>() as f64;
    let families = counts.len() as f64;
    counts
        .into_iter()
        .map(|(measurement, count)| (measurement, total / (families * count as f64)))
        .collect()
}

fn slot_loss_from_forward(forward: &ResearchForward) -> Tensor {
    let topo = slot_penalty(&forward.slots.topology.slot_weights);
    let geo = slot_penalty(&forward.slots.geometry.slot_weights);
    let pocket = slot_penalty(&forward.slots.pocket.slot_weights);
    (topo + geo + pocket) / 3.0
}

fn slot_loss_from_batch(forwards: &[ResearchForward], device: tch::Device) -> Tensor {
    if forwards.is_empty() {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let mut total = Tensor::zeros([1], (Kind::Float, device));
    for forward in forwards {
        total += slot_loss_from_forward(forward);
    }
    total / forwards.len() as f64
}

fn slot_penalty(weights: &Tensor) -> Tensor {
    if weights.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, weights.device()));
    }
    let sparsity = weights.abs().mean(Kind::Float);
    let uniform = Tensor::full(
        weights.size(),
        1.0 / (weights.size()[0].max(1) as f64),
        (Kind::Float, weights.device()),
    );
    let balance = (weights - uniform).pow_tensor_scalar(2.0).mean(Kind::Float);
    sparsity + balance
}

fn tensor_is_finite(tensor: &Tensor) -> bool {
    tensor
        .isfinite()
        .all()
        .to_kind(Kind::Int64)
        .int64_value(&[])
        != 0
}

fn scalar_or_nan(tensor: &Tensor) -> f64 {
    if tensor_is_finite(tensor) {
        tensor.double_value(&[])
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device};

    use crate::{config::ResearchConfig, data::InMemoryDataset, models::Phase1ResearchSystem};

    #[test]
    fn fit_respects_batch_size_and_max_steps() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 3;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = ResearchTrainer::new(&var_store, config).unwrap();

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();
        assert_eq!(metrics.len(), 3);
        assert_eq!(trainer.step(), 3);
    }

    #[test]
    fn resume_restores_step_from_latest_checkpoint() {
        let temp = tempfile::tempdir().unwrap();

        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 2;
        config.training.checkpoint_every = 1;
        config.training.log_every = 100;
        config.training.checkpoint_dir = temp.path().join("checkpoints");

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = ResearchTrainer::new(&var_store, config.clone()).unwrap();
        trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        let mut resumed_store = nn::VarStore::new(Device::Cpu);
        let _resumed_system = Phase1ResearchSystem::new(&resumed_store.root(), &config);
        let mut resumed_trainer = ResearchTrainer::new(&resumed_store, config).unwrap();
        let checkpoint = resumed_trainer
            .resume_from_latest(&mut resumed_store)
            .unwrap()
            .unwrap();

        assert_eq!(checkpoint.metadata.step, 1);
        assert_eq!(resumed_trainer.step(), 2);
        assert_eq!(
            resumed_trainer
                .restored_optimizer_state()
                .map(|state| state.optimizer_kind.as_str()),
            Some("adam")
        );
        assert_eq!(
            resumed_trainer
                .restored_scheduler_state()
                .map(|state| state.stage.as_str()),
            Some("Stage1")
        );
    }

    #[test]
    fn conditioned_denoising_training_remains_finite_on_synthetic_examples() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 3;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = ResearchTrainer::new(&var_store, config).unwrap();

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        assert!(!metrics.is_empty());
        assert!(metrics.iter().all(|step| step.losses.total.is_finite()));
        assert!(metrics
            .iter()
            .all(|step| step.losses.primary.primary_value.is_finite()));
    }

    #[test]
    fn trainable_backend_families_have_finite_synthetic_training_smoke() {
        for (backend_id, family) in [
            (
                "flow_matching",
                crate::config::GenerationBackendFamilyConfig::FlowMatching,
            ),
            (
                "autoregressive_graph_geometry",
                crate::config::GenerationBackendFamilyConfig::Autoregressive,
            ),
        ] {
            let mut config = ResearchConfig::default();
            config.data.batch_size = 2;
            config.training.max_steps = 2;
            config.training.checkpoint_every = 100;
            config.training.log_every = 100;
            config.training.primary_objective =
                crate::config::PrimaryObjectiveConfig::ConditionedDenoising;
            config.generation_method.active_method = backend_id.to_string();
            config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
                backend_id: backend_id.to_string(),
                family,
                trainable: true,
                ..crate::config::GenerationBackendConfig::default()
            };

            let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
                .with_pocket_feature_dim(config.model.pocket_feature_dim);
            let var_store = nn::VarStore::new(Device::Cpu);
            let system = Phase1ResearchSystem::new(&var_store.root(), &config);
            let mut trainer = ResearchTrainer::new(&var_store, config).unwrap();

            let metrics = trainer
                .fit(&var_store, &system, dataset.examples())
                .unwrap();

            assert!(!metrics.is_empty());
            assert!(metrics.iter().all(|step| step.losses.total.is_finite()));
        }
    }

    #[test]
    fn resume_rejects_incompatible_backend_metadata() {
        let temp = tempfile::tempdir().unwrap();

        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 1;
        config.training.log_every = 100;
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising;
        config.training.checkpoint_dir = temp.path().join("checkpoints");

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = ResearchTrainer::new(&var_store, config.clone()).unwrap();
        trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        let mut incompatible = config.clone();
        incompatible.generation_method.active_method = "flow_matching".to_string();
        incompatible.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        let mut resumed_store = nn::VarStore::new(Device::Cpu);
        let _system = Phase1ResearchSystem::new(&resumed_store.root(), &incompatible);
        let mut resumed = ResearchTrainer::new(&resumed_store, incompatible).unwrap();

        let error = resumed
            .resume_from_latest(&mut resumed_store)
            .unwrap_err()
            .to_string();
        assert!(error.contains("checkpoint backend/objective mismatch"));
    }

    #[test]
    fn pocket_geometry_auxiliary_losses_are_reported_when_enabled() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 2;
        config.training.schedule.stage1_steps = 0;
        config.training.schedule.stage2_steps = 1;
        config.training.schedule.stage3_steps = 2;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.loss_weights.rho_pocket_contact = 0.2;
        config.training.loss_weights.sigma_pocket_clash = 0.3;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = ResearchTrainer::new(&var_store, config).unwrap();

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        assert!(metrics
            .iter()
            .all(|step| step.losses.auxiliaries.pocket_contact.is_finite()));
        assert!(metrics
            .iter()
            .all(|step| step.losses.auxiliaries.pocket_clash.is_finite()));
        assert!(metrics
            .iter()
            .any(|step| step.losses.auxiliaries.pocket_contact >= 0.0));
    }

    #[test]
    fn batched_loss_api_matches_per_example_aggregation() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 3;
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising;
        config.training.affinity_weighting = AffinityWeighting::InverseFrequency;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..3];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let mut trainer = ResearchTrainer::new(&var_store, config).unwrap();
        trainer.affinity_measurement_weights =
            measurement_weights(examples, trainer.config.training.affinity_weighting);

        let primary_batch = compute_primary_objective_batch(
            trainer.primary_objective.as_ref(),
            examples,
            &forwards,
        );
        let probe_batch =
            trainer
                .probe_loss
                .compute_batch_weighted(examples, &forwards, |example| {
                    trainer.affinity_weight_for(example)
                });
        let redundancy_batch = trainer.redundancy_loss.compute_batch(&forwards);
        let leakage_batch = trainer.leakage_loss.compute_batch(examples, &forwards);
        let gate_batch = trainer.gate_loss.compute_batch(&forwards);
        let consistency_batch = trainer.consistency_loss.compute_batch(examples, &forwards);
        let (contact_batch, clash_batch) = trainer
            .pocket_geometry_loss
            .compute_batch(examples, &forwards);

        let denom = examples.len() as f64;
        let mut primary_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut probe_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut redundancy_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut leakage_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut gate_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut consistency_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut contact_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let mut clash_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            primary_manual += trainer.primary_objective.compute(example, forward);
            probe_manual += trainer.probe_loss.compute_weighted(
                example,
                forward,
                trainer.affinity_weight_for(example),
            );
            redundancy_manual += trainer.redundancy_loss.compute(&forward.slots);
            leakage_manual += trainer.leakage_loss.compute(example, forward);
            gate_manual += trainer.gate_loss.compute(&forward.interactions);
            consistency_manual += trainer.consistency_loss.compute(example, forward);
            let (contact, clash) = trainer.pocket_geometry_loss.compute(example, forward);
            contact_manual += contact;
            clash_manual += clash;
        }

        assert_close(&primary_batch, &(primary_manual / denom));
        assert_close(&probe_batch, &(probe_manual / denom));
        assert_close(&redundancy_batch, &(redundancy_manual / denom));
        assert_close(&leakage_batch, &(leakage_manual / denom));
        assert_close(&gate_batch, &(gate_manual / denom));
        assert_close(&consistency_batch, &(consistency_manual / denom));
        assert_close(&contact_batch, &(contact_manual / denom));
        assert_close(&clash_batch, &(clash_manual / denom));
    }

    fn assert_close(left: &Tensor, right: &Tensor) {
        let delta = (left - right).abs().double_value(&[]);
        assert!(
            delta <= 1e-5,
            "loss mismatch: left={} right={} delta={delta}",
            left.double_value(&[]),
            right.double_value(&[])
        );
    }
}
