//! Runnable staged trainer for the new research stack.

use std::collections::BTreeMap;

use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::{
    config::{AffinityWeighting, ResearchConfig},
    data::{ExampleBatchIter, MolecularExample},
    losses::{ConsistencyLoss, GateLoss, IntraRedundancyLoss, LeakageLoss, ProbeLoss, TaskLoss},
    models::{Phase1ResearchSystem, ResearchForward},
};

use super::{
    CheckpointManager, LoadedCheckpoint, LossBreakdown, StageScheduler, StepMetrics, TrainingStage,
};

/// Trainer that applies staged auxiliary losses to the new modular system.
pub struct ResearchTrainer {
    optimizer: nn::Optimizer,
    scheduler: StageScheduler,
    checkpoints: CheckpointManager,
    task_loss: TaskLoss,
    redundancy_loss: IntraRedundancyLoss,
    probe_loss: ProbeLoss,
    leakage_loss: LeakageLoss,
    gate_loss: GateLoss,
    consistency_loss: ConsistencyLoss,
    config: ResearchConfig,
    affinity_measurement_weights: BTreeMap<String, f64>,
    step: usize,
    history: Vec<StepMetrics>,
    last_stage: Option<TrainingStage>,
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
            task_loss: TaskLoss,
            redundancy_loss: IntraRedundancyLoss::default(),
            probe_loss: ProbeLoss,
            leakage_loss: LeakageLoss::default(),
            gate_loss: GateLoss,
            consistency_loss: ConsistencyLoss::default(),
            config,
            affinity_measurement_weights: BTreeMap::new(),
            step: 0,
            history: Vec::new(),
            last_stage: None,
        })
    }

    /// Resume from the latest checkpoint in the configured directory.
    pub fn resume_from_latest(
        &mut self,
        var_store: &mut nn::VarStore,
    ) -> Result<Option<LoadedCheckpoint>, Box<dyn std::error::Error>> {
        let checkpoint = self.checkpoints.load_latest(var_store)?;
        if let Some(loaded) = &checkpoint {
            if let Some(metrics) = loaded.metadata.metrics.clone() {
                self.history.push(metrics);
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
                "entering {:?} at step {} with weights task={:.4} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4}",
                stage,
                self.step,
                weights.task,
                weights.intra_red,
                weights.probe,
                weights.leak,
                weights.gate,
                weights.slot,
                weights.consistency,
            );
            self.last_stage = Some(stage);
        }

        let mut task_acc = Tensor::zeros([1], (Kind::Float, var_store.device()));
        let mut intra_red_acc = Tensor::zeros([1], (Kind::Float, var_store.device()));
        let mut probe_acc = Tensor::zeros([1], (Kind::Float, var_store.device()));
        let mut leak_acc = Tensor::zeros([1], (Kind::Float, var_store.device()));
        let mut gate_acc = Tensor::zeros([1], (Kind::Float, var_store.device()));
        let mut slot_acc = Tensor::zeros([1], (Kind::Float, var_store.device()));
        let mut consistency_acc = Tensor::zeros([1], (Kind::Float, var_store.device()));
        self.affinity_measurement_weights =
            measurement_weights(examples, self.config.training.affinity_weighting);

        for (example, forward) in examples.iter().zip(forwards.iter()) {
            task_acc += self.task_loss.compute(forward);
            intra_red_acc += self.redundancy_loss.compute(&forward.slots);
            probe_acc += self.probe_loss.compute_weighted(
                example,
                forward,
                self.affinity_weight_for(example),
            );
            leak_acc += self.leakage_loss.compute(example, forward);
            gate_acc += self.gate_loss.compute(&forward.interactions);
            slot_acc += slot_loss_from_forward(forward);
            consistency_acc += self.consistency_loss.compute(example, forward);
        }

        let denom = (examples.len().max(1)) as f64;
        let task = task_acc / denom;
        let intra_red = intra_red_acc / denom;
        let probe = probe_acc / denom;
        let leak = leak_acc / denom;
        let gate = gate_acc / denom;
        let slot = slot_acc / denom;
        let consistency = consistency_acc / denom;

        let total = &task * weights.task
            + &intra_red * weights.intra_red
            + &probe * weights.probe
            + &leak * weights.leak
            + &gate * weights.gate
            + &slot * weights.slot
            + &consistency * weights.consistency;

        self.optimizer.zero_grad();
        total.backward();
        self.optimizer.step();

        let metrics = StepMetrics {
            step: self.step,
            stage,
            losses: LossBreakdown {
                task: task.double_value(&[]),
                intra_red: intra_red.double_value(&[]),
                probe: probe.double_value(&[]),
                leak: leak.double_value(&[]),
                gate: gate.double_value(&[]),
                slot: slot.double_value(&[]),
                consistency: consistency.double_value(&[]),
                total: total.double_value(&[]),
            },
        };

        if self.config.training.checkpoint_every > 0
            && self.step % self.config.training.checkpoint_every == 0
        {
            self.checkpoints
                .save(var_store, self.step, Some(&metrics))?;
        }
        if self.config.training.log_every > 0 && self.step % self.config.training.log_every == 0 {
            log::info!(
                "step {} [{:?}] total={:.4} task={:.4} intra_red={:.4} probe={:.4} leak={:.4} gate={:.4} slot={:.4} consistency={:.4}",
                metrics.step,
                metrics.stage,
                metrics.losses.total,
                metrics.losses.task,
                metrics.losses.intra_red,
                metrics.losses.probe,
                metrics.losses.leak,
                metrics.losses.gate,
                metrics.losses.slot,
                metrics.losses.consistency,
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
    }
}
