//! Runnable staged trainer for the new research stack.

use std::collections::BTreeMap;

use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::{
    config::{AffinityWeighting, ResearchConfig},
    data::MolecularExample,
    losses::{ConsistencyLoss, GateLoss, IntraRedundancyLoss, LeakageLoss, ProbeLoss, TaskLoss},
    models::{Phase1ResearchSystem, ResearchForward},
};

use super::{CheckpointManager, LossBreakdown, StageScheduler, StepMetrics, TrainingStage};

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
        })
    }

    /// Run one optimization step over a small batch of examples.
    pub fn train_step(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        examples: &[MolecularExample],
    ) -> Result<StepMetrics, Box<dyn std::error::Error>> {
        let (_, forwards) = system.forward_batch(examples);
        let weights = self.scheduler.weights_for_step(self.step);
        let stage = self.scheduler.stage_for_step(self.step);

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

        if self.step % self.config.training.checkpoint_every == 0 {
            self.checkpoints
                .save(var_store, self.step, Some(&metrics))?;
        }
        self.history.push(metrics.clone());
        self.step += 1;
        Ok(metrics)
    }

    /// Borrow trainer history.
    pub fn history(&self) -> &[StepMetrics] {
        &self.history
    }

    /// Current training stage.
    pub fn stage(&self) -> TrainingStage {
        self.scheduler.stage_for_step(self.step)
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
