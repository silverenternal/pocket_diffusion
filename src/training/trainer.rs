//! Runnable staged trainer for the new research stack.

use rand::SeedableRng;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::error::Error;
use std::time::Instant;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::{
    config::{
        AffinityWeighting, CrossAttentionMode, GenerationBackendFamilyConfig, ModalityFocusConfig,
        ObjectiveGradientSamplingMode, PrimaryObjectiveConfig, ResearchConfig,
    },
    data::{
        sample_order_seed_for_epoch, ExampleBatchSampler, MolecularExample, MolecularExampleSource,
    },
    losses::{
        auxiliary::{AuxiliaryObjectiveExecutionPlan, AuxiliaryObjectiveTensors},
        build_primary_objective, compute_primary_objective_batch_with_components,
        AuxiliaryObjectiveBlock, PrimaryObjectiveWithComponents,
    },
    models::{interaction::InteractionExecutionContext, Phase1ResearchSystem, ResearchForward},
    training::{
        determinism_controls_from_config, stable_json_hash, METRIC_SCHEMA_VERSION,
        RESUME_CONTRACT_VERSION,
    },
};

use super::{
    AuxiliaryLossMetrics, AuxiliaryObjectiveFamily, AuxiliaryObjectiveReport,
    BackendTrainingMetadata, CheckpointManager, EffectiveLossWeights, GradientHealthMetrics,
    GradientModuleMetrics, InteractionStepMetrics, LoadedCheckpoint, LossBreakdown,
    ObjectiveExecutionCountMetrics, ObjectiveGradientDiagnostics, ObjectiveGradientFamilyMetrics,
    OptimizerStateMetadata, PrimaryBranchScheduleReport, PrimaryBranchWeightRecord,
    PrimaryObjectiveMetrics, SchedulerStateMetadata, SlotUtilizationStepMetrics,
    StageProgressMetrics, StageScheduler, StepMetrics, SynchronizationHealthMetrics,
    TrainingRuntimeProfileMetrics, TrainingStage,
};

/// Trainer that applies staged auxiliary losses to the new modular system.
pub struct ResearchTrainer {
    optimizer: nn::Optimizer,
    scheduler: StageScheduler,
    checkpoints: CheckpointManager,
    primary_objective: Box<dyn PrimaryObjectiveWithComponents>,
    auxiliary_objectives: AuxiliaryObjectiveBlock,
    config: ResearchConfig,
    dataset_validation_fingerprint: Option<String>,
    affinity_measurement_weights: BTreeMap<String, f64>,
    step: usize,
    history: Vec<StepMetrics>,
    last_stage: Option<TrainingStage>,
    restored_optimizer_state: Option<OptimizerStateMetadata>,
    restored_scheduler_state: Option<SchedulerStateMetadata>,
    primary_component_scale_ema: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
struct TrainingBatchOrderMetadata {
    epoch_index: usize,
    sample_order_seed: u64,
    sample_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct StageSelection {
    stage: TrainingStage,
    held: bool,
    readiness_status: String,
    readiness_reasons: Vec<String>,
}

struct StageStepRunner<'a> {
    scheduler: &'a StageScheduler,
    config: &'a ResearchConfig,
    history: &'a [StepMetrics],
    step: usize,
}

struct StageStepContext {
    fixed_stage: TrainingStage,
    selection: StageSelection,
    weights: EffectiveLossWeights,
    stage_index: usize,
    stage_ramp: f64,
}

struct ObjectiveExecutionRunner;

struct GradientStepRunner<'a> {
    optimizer: &'a mut nn::Optimizer,
    var_store: &'a nn::VarStore,
    config: &'a ResearchConfig,
    step: usize,
    primary_objective_name: &'a str,
    primary_weighted_value: f64,
    auxiliary_metrics: &'a AuxiliaryLossMetrics,
    expected_activity: &'a BTreeMap<&'static str, bool>,
    objective_gradient_diagnostics: Option<ObjectiveGradientDiagnostics>,
}

struct StepRuntimeTracker {
    started_at: Instant,
    memory_before_mb: f64,
}

struct CheckpointStepTrigger {
    checkpoint_every: usize,
}

struct BorrowedExampleSource<'a> {
    examples: &'a [MolecularExample],
}

impl<'a> BorrowedExampleSource<'a> {
    fn new(examples: &'a [MolecularExample]) -> Self {
        Self { examples }
    }
}

impl<'a> MolecularExampleSource for BorrowedExampleSource<'a> {
    type Error = Infallible;

    fn len(&self) -> usize {
        self.examples.len()
    }

    fn get_example(&self, index: usize) -> Result<Option<MolecularExample>, Self::Error> {
        Ok(self.examples.get(index).cloned())
    }

    fn materialized_examples(&self) -> Option<&[MolecularExample]> {
        Some(self.examples)
    }
}

impl TrainingBatchOrderMetadata {
    fn direct_batch(batch_len: usize, sampler_seed: u64) -> Self {
        Self {
            epoch_index: 0,
            sample_order_seed: sample_order_seed_for_epoch(sampler_seed, 0),
            sample_indices: (0..batch_len).collect(),
        }
    }
}

impl<'a> StageStepRunner<'a> {
    fn new(
        scheduler: &'a StageScheduler,
        config: &'a ResearchConfig,
        history: &'a [StepMetrics],
        step: usize,
    ) -> Self {
        Self {
            scheduler,
            config,
            history,
            step,
        }
    }

    fn select(self) -> StageStepContext {
        let fixed_stage = self.scheduler.stage_for_step(self.step);
        let selection = self.select_stage(fixed_stage);
        let weight_step = representative_step_for_stage(self.config, selection.stage);
        let weights = if selection.stage == fixed_stage {
            self.scheduler.weights_for_step(self.step)
        } else {
            self.scheduler.weights_for_step(weight_step)
        };
        let stage_ramp = if selection.stage == fixed_stage {
            self.scheduler.ramp_for_step(self.step)
        } else {
            self.scheduler.ramp_for_step(weight_step)
        };
        StageStepContext {
            fixed_stage,
            stage_index: selection.stage.index(),
            selection,
            weights,
            stage_ramp,
        }
    }

    fn select_stage(&self, fixed_stage: TrainingStage) -> StageSelection {
        let guard = &self.config.training.adaptive_stage_guard;
        if !guard.enabled {
            return StageSelection {
                stage: fixed_stage,
                held: false,
                readiness_status: "disabled".to_string(),
                readiness_reasons: Vec::new(),
            };
        }
        if fixed_stage == TrainingStage::Stage1 {
            return StageSelection {
                stage: fixed_stage,
                held: false,
                readiness_status: "ready".to_string(),
                readiness_reasons: vec![
                    "stage1 bootstrap does not require prior readiness".to_string()
                ],
            };
        }

        let readiness_reasons = stage_readiness_reasons(guard, self.history);
        if readiness_reasons.is_empty() {
            return StageSelection {
                stage: fixed_stage,
                held: false,
                readiness_status: "ready".to_string(),
                readiness_reasons: vec![
                    "recent metrics satisfy adaptive stage readiness checks".to_string()
                ],
            };
        }

        if guard.hold_stages {
            StageSelection {
                stage: previous_stage(fixed_stage),
                held: true,
                readiness_status: "held".to_string(),
                readiness_reasons,
            }
        } else {
            StageSelection {
                stage: fixed_stage,
                held: false,
                readiness_status: "warning".to_string(),
                readiness_reasons,
            }
        }
    }
}

impl ObjectiveExecutionRunner {
    fn weighted_total(
        primary: &Tensor,
        auxiliaries: &AuxiliaryObjectiveTensors,
        weights: EffectiveLossWeights,
    ) -> Tensor {
        primary * weights.primary
            + &auxiliaries.intra_red * weights.intra_red
            + &auxiliaries.probe_core * weights.probe
            + (&auxiliaries.probe_ligand_pharmacophore + &auxiliaries.probe_pocket_pharmacophore)
                * weights.pharmacophore_probe
            + (&auxiliaries.leak_core
                + &auxiliaries.leak_topology_to_geometry
                + &auxiliaries.leak_geometry_to_topology
                + &auxiliaries.leak_pocket_to_geometry)
                * weights.leak
            + (&auxiliaries.leak_topology_to_pocket_role
                + &auxiliaries.leak_geometry_to_pocket_role
                + &auxiliaries.leak_pocket_to_ligand_role)
                * weights.pharmacophore_leakage
            + &auxiliaries.gate * weights.gate
            + &auxiliaries.slot * weights.slot
            + &auxiliaries.consistency * weights.consistency
            + &auxiliaries.pocket_contact * weights.pocket_contact
            + &auxiliaries.pocket_clash * weights.pocket_clash
            + &auxiliaries.pocket_envelope * weights.pocket_envelope
            + &auxiliaries.valence_guardrail * weights.valence_guardrail
            + &auxiliaries.bond_length_guardrail * weights.bond_length_guardrail
    }
}

impl<'a> GradientStepRunner<'a> {
    fn objective_gradient_diagnostics(
        &self,
        global_grad_l2_norm: f64,
        nonfinite_gradient_tensors: usize,
    ) -> ObjectiveGradientDiagnostics {
        self.objective_gradient_diagnostics
            .clone()
            .unwrap_or_else(|| {
                loss_share_objective_gradient_diagnostics(
                    self.step,
                    self.primary_objective_name,
                    self.primary_weighted_value,
                    self.auxiliary_metrics,
                    self.config,
                    global_grad_l2_norm,
                    nonfinite_gradient_tensors,
                )
            })
    }

    fn apply(self, total: &Tensor, nonfinite_loss_terms: Vec<String>) -> GradientHealthMetrics {
        self.optimizer.zero_grad();
        let mut gradient_health;
        if nonfinite_loss_terms.is_empty() && tensor_is_finite(total) {
            total.backward();
            gradient_health = collect_gradient_health(
                self.var_store,
                false,
                self.config.training.gradient_clipping.global_norm,
                nonfinite_loss_terms,
                self.expected_activity,
            );
            if gradient_health.nonfinite_gradient_tensors > 0 {
                gradient_health.optimizer_step_skipped = true;
                gradient_health.objective_families = self.objective_gradient_diagnostics(
                    gradient_health.global_grad_l2_norm,
                    gradient_health.nonfinite_gradient_tensors,
                );
                log::warn!(
                    "skipping optimizer step {} because {} gradient tensors became non-finite",
                    self.step,
                    gradient_health.nonfinite_gradient_tensors
                );
            } else {
                if let Some(max_norm) = self.config.training.gradient_clipping.global_norm {
                    if gradient_health.pre_clip_global_grad_l2_norm > max_norm {
                        self.optimizer.clip_grad_norm(max_norm);
                        gradient_health = collect_gradient_health(
                            self.var_store,
                            false,
                            Some(max_norm),
                            Vec::new(),
                            self.expected_activity,
                        )
                        .with_pre_clip_norm(gradient_health.pre_clip_global_grad_l2_norm);
                        gradient_health.clipped = true;
                    }
                }
                gradient_health.objective_families = self.objective_gradient_diagnostics(
                    gradient_health.global_grad_l2_norm,
                    gradient_health.nonfinite_gradient_tensors,
                );
                self.optimizer.step();
            }
        } else {
            log::warn!(
                "skipping optimizer step {} because loss terms became non-finite: {}",
                self.step,
                nonfinite_loss_terms.join(", ")
            );
            gradient_health = collect_gradient_health(
                self.var_store,
                true,
                self.config.training.gradient_clipping.global_norm,
                nonfinite_loss_terms,
                self.expected_activity,
            );
            gradient_health.objective_families = self.objective_gradient_diagnostics(
                gradient_health.global_grad_l2_norm,
                gradient_health.nonfinite_gradient_tensors,
            );
        }
        gradient_health
    }
}

impl StepRuntimeTracker {
    fn start() -> Self {
        Self {
            started_at: Instant::now(),
            memory_before_mb: current_used_memory_mb(),
        }
    }

    fn finish(
        self,
        batch_size: usize,
        rollout_diagnostics_built: bool,
        rollout_diagnostic_execution_count: usize,
        objective_execution_counts: ObjectiveExecutionCountMetrics,
    ) -> TrainingRuntimeProfileMetrics {
        let step_time_ms = self.started_at.elapsed().as_secs_f64() * 1000.0;
        let examples_per_second = if step_time_ms > 0.0 {
            batch_size as f64 / (step_time_ms / 1000.0)
        } else {
            0.0
        };
        let memory_after_mb = current_used_memory_mb();
        TrainingRuntimeProfileMetrics {
            step_time_ms,
            examples_per_second,
            batch_size,
            forward_batch_count: 1,
            per_example_forward_count: 0,
            forward_execution_mode: "batched_interaction_context".to_string(),
            de_novo_per_example_reason: None,
            rollout_diagnostics_built,
            rollout_diagnostic_execution_count,
            rollout_diagnostics_no_grad: rollout_diagnostics_built,
            memory_before_mb: self.memory_before_mb,
            memory_after_mb,
            memory_delta_mb: memory_after_mb - self.memory_before_mb,
            objective_execution_counts,
        }
    }
}

impl CheckpointStepTrigger {
    fn new(checkpoint_every: usize) -> Self {
        Self { checkpoint_every }
    }

    fn should_save(&self, step: usize) -> bool {
        self.checkpoint_every > 0 && step % self.checkpoint_every == 0
    }
}

impl ResearchTrainer {
    /// Create a new trainer from the shared var store and config.
    pub fn new(
        var_store: &nn::VarStore,
        config: ResearchConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        config
            .validate()
            .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
        Self::new_unvalidated(var_store, config)
    }

    fn new_unvalidated(
        var_store: &nn::VarStore,
        config: ResearchConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let optimizer = nn::Adam::default().build(var_store, config.training.learning_rate)?;
        let scheduler = StageScheduler::new_with_chemistry_warmup(
            config.training.schedule.clone(),
            config.training.loss_weights.clone(),
            config.training.chemistry_warmup.clone(),
        );
        let checkpoints = CheckpointManager::new(config.training.checkpoint_dir.clone());

        Ok(Self {
            optimizer,
            scheduler,
            checkpoints,
            primary_objective: build_primary_objective(&config.training),
            auxiliary_objectives: AuxiliaryObjectiveBlock::new_with_pharmacophore_and_gate_config(
                config.training.loss_weights.slot_sparsity_weight,
                config.training.loss_weights.slot_balance_weight,
                config.training.pharmacophore_probes.clone(),
                config.training.explicit_leakage_probes.clone(),
                config
                    .model
                    .interaction_tuning
                    .gate_regularization_path_weights
                    .clone(),
            ),
            config,
            dataset_validation_fingerprint: None,
            affinity_measurement_weights: BTreeMap::new(),
            step: 0,
            history: Vec::new(),
            last_stage: None,
            restored_optimizer_state: None,
            restored_scheduler_state: None,
            primary_component_scale_ema: BTreeMap::new(),
        })
    }

    #[cfg(test)]
    fn new_unvalidated_for_tests(
        var_store: &nn::VarStore,
        config: ResearchConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_unvalidated(var_store, config)
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
            if let Some(state) = &self.restored_optimizer_state {
                if state.resume_mode == crate::training::ResumeMode::OptimizerExactResume
                    && !state.internal_state_persisted
                {
                    return Err(
                        "checkpoint advertised optimizer_exact_resume but internal_state_persisted is false"
                            .into(),
                    );
                }
            }
            if self.config.training.resume.require_optimizer_exact {
                let exact_available = loaded.metadata.resume_mode
                    == crate::training::ResumeMode::OptimizerExactResume
                    && self.restored_optimizer_state.as_ref().is_some_and(|state| {
                        state.internal_state_persisted && state.exact_resume_supported
                    });
                if !exact_available {
                    return Err(
                        "training.resume.require_optimizer_exact is true but the checkpoint only supports weights-only optimizer metadata; tch optimizer internals are not persisted"
                            .into(),
                    );
                }
            }
            self.restored_scheduler_state = loaded.metadata.scheduler_state.clone();
            if let Some(optimizer_state) = &self.restored_optimizer_state {
                self.optimizer.set_lr(optimizer_state.learning_rate);
                self.optimizer
                    .set_weight_decay(optimizer_state.weight_decay);
            }
            log::info!(
                "checkpoint resume mode: {} (optimizer_internal_state_persisted={})",
                loaded.metadata.resume_mode.as_str(),
                self.restored_optimizer_state
                    .as_ref()
                    .map(|state| state.internal_state_persisted)
                    .unwrap_or(false)
            );
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
        let order_metadata = TrainingBatchOrderMetadata::direct_batch(
            examples.len(),
            self.config.training.data_order.sampler_seed,
        );
        self.train_batch_step_with_order_metadata(var_store, system, examples, order_metadata)
    }

    /// Run training from a source over deterministic mini-batches.
    ///
    /// The active batch order and metadata semantics are identical to in-memory
    /// training because this method reuses the same epoch/seed logic as
    /// `ExampleBatchSampler`.
    pub fn fit_source<S>(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        source: &S,
    ) -> Result<Vec<StepMetrics>, Box<dyn Error>>
    where
        S: MolecularExampleSource,
        S::Error: Error + 'static,
    {
        self.fit_source_with_step_observer(var_store, system, source, |_, _, _| Ok(false))
    }

    /// Train over deterministic mini-batches and invoke an observer after each completed step.
    pub fn fit_source_with_step_observer<S, F>(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        source: &S,
        mut observer: F,
    ) -> Result<Vec<StepMetrics>, Box<dyn Error>>
    where
        S: MolecularExampleSource,
        S::Error: Error + 'static,
        F: FnMut(&mut Self, &nn::VarStore, &StepMetrics) -> Result<bool, Box<dyn Error>>,
    {
        if source.is_empty() {
            return Err("training examples are empty".into());
        }

        let mut steps = Vec::new();
        let batch_size = self.config.data.batch_size.max(1);
        let data_order = self.config.training.data_order.clone();
        let batches_per_epoch =
            ExampleBatchSampler::batches_per_epoch(source.len(), batch_size, data_order.drop_last);
        if batches_per_epoch == 0 {
            return Err("data_order.drop_last leaves no trainable mini-batches".into());
        }

        while self.step < self.config.training.max_steps {
            let mut progressed = false;
            let mut stop_requested_epoch = false;
            let epoch_index = self.step / batches_per_epoch;
            if matches!(data_order.max_epochs, Some(max_epochs) if epoch_index >= max_epochs) {
                break;
            }
            let start_batch_index = self.step % batches_per_epoch;
            let sample_order_seed = sample_order_seed_for_epoch(
                self.config.training.data_order.sampler_seed,
                epoch_index,
            );
            let mut order: Vec<usize> = (0..source.len()).collect();
            if data_order.shuffle {
                use rand::seq::SliceRandom;
                order.shuffle(&mut rand::rngs::StdRng::seed_from_u64(sample_order_seed));
            }
            let mut cursor = 0usize;
            let mut batch_index = 0usize;
            while cursor < order.len() {
                let remaining = order.len() - cursor;
                if data_order.drop_last && remaining < batch_size {
                    break;
                }
                let end = (cursor + batch_size).min(order.len());
                let sample_indices = order[cursor..end].to_vec();
                cursor = end;
                if batch_index < start_batch_index {
                    batch_index += 1;
                    continue;
                }
                let mut examples = Vec::with_capacity(sample_indices.len());
                for index in sample_indices.iter().copied() {
                    let example = source
                        .get_example(index)
                        .map_err(|error| {
                            format!("failed to load training example at index {index}: {error}")
                        })?
                        .ok_or_else(|| format!("missing training example at index {index}"))?
                        .to_device(var_store.device());
                    examples.push(example);
                }

                let order_metadata = TrainingBatchOrderMetadata {
                    epoch_index,
                    sample_order_seed,
                    sample_indices,
                };
                let metrics = self.train_batch_step_with_order_metadata(
                    var_store,
                    system,
                    &examples,
                    order_metadata,
                )?;
                let stop_requested = observer(self, var_store, &metrics)?;
                stop_requested_epoch |= stop_requested;
                steps.push(metrics);
                progressed = true;
                batch_index += 1;
                if self.step >= self.config.training.max_steps || stop_requested {
                    break;
                }
            }
            if !progressed {
                break;
            }
            if self.step >= self.config.training.max_steps || stop_requested_epoch {
                break;
            }
        }
        Ok(steps)
    }

    fn train_batch_step_with_order_metadata(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        examples: &[MolecularExample],
        order_metadata: TrainingBatchOrderMetadata,
    ) -> Result<StepMetrics, Box<dyn std::error::Error>> {
        if examples.is_empty() {
            return Err("cannot train on an empty mini-batch".into());
        }
        let runtime_tracker = StepRuntimeTracker::start();
        let stage_context =
            StageStepRunner::new(&self.scheduler, &self.config, &self.history, self.step).select();
        let stage = stage_context.selection.stage;
        let stage_index = stage_context.stage_index;
        let weights = stage_context.weights;
        let build_rollout_diagnostics = self.config.training.build_rollout_diagnostics;
        let (_, forwards) = system.forward_batch_with_interaction_context_and_rollout_diagnostics(
            examples,
            InteractionExecutionContext {
                training_stage: Some(stage_index),
                training_step: Some(self.step),
                epoch_index: Some(order_metadata.epoch_index),
                sample_order_seed: Some(order_metadata.sample_order_seed),
                rollout_step_index: None,
                flow_t: None,
            },
            build_rollout_diagnostics,
        );
        if self.last_stage != Some(stage) {
            log::info!(
                "entering {:?} at step {} with weights primary={:.4} intra_red={:.4} probe={:.4} pharmacophore_probe={:.4} leak={:.4} pharmacophore_leakage={:.4} gate={:.4} slot={:.4} consistency={:.4} pocket_contact={:.4} pocket_clash={:.4} pocket_envelope={:.4} valence_guardrail={:.4} bond_length_guardrail={:.4}",
                stage,
                self.step,
                weights.primary,
                weights.intra_red,
                weights.probe,
                weights.pharmacophore_probe,
                weights.leak,
                weights.pharmacophore_leakage,
                weights.gate,
                weights.slot,
                weights.consistency,
                weights.pocket_contact,
                weights.pocket_clash,
                weights.pocket_envelope,
                weights.valence_guardrail,
                weights.bond_length_guardrail,
            );
            self.last_stage = Some(stage);
        }

        self.affinity_measurement_weights =
            measurement_weights(examples, self.config.training.affinity_weighting);
        let (primary, primary_components) = compute_primary_objective_batch_with_components(
            self.primary_objective.as_ref(),
            examples,
            &forwards,
        );
        let auxiliary_execution_plan =
            AuxiliaryObjectiveExecutionPlan::from_effective_weights(&weights);
        let auxiliaries = self
            .auxiliary_objectives
            .compute_batch_with_execution_plan_and_step(
                examples,
                &forwards,
                |example| self.affinity_weight_for(example),
                var_store.device(),
                &auxiliary_execution_plan,
                Some(self.step),
            );

        let total = ObjectiveExecutionRunner::weighted_total(&primary, &auxiliaries, weights);

        let primary_value = scalar_or_nan(&primary);
        let (mut auxiliary_metrics, _) = auxiliaries.to_metrics_with_weights_and_execution_plan(
            scalar_or_nan,
            &weights,
            &auxiliary_execution_plan,
        );
        self.update_primary_component_scale_ema(&primary_components);
        let scale_config = &self.config.training.objective_scale_diagnostics;
        let primary_component_scale_report = if scale_config.enabled {
            primary_components.scale_report(
                weights.primary,
                scale_config.warning_ratio,
                scale_config.epsilon,
                scale_config
                    .running_scale_momentum
                    .map(|_| &self.primary_component_scale_ema),
            )
        } else {
            Default::default()
        };
        let primary_branch_schedule = primary_branch_schedule_report(
            &forwards,
            self.step,
            Some(stage_index),
            &primary_components,
            &self.config,
        );
        let objective_execution_counts = objective_execution_counts(
            weights.primary,
            &auxiliary_metrics.auxiliary_objective_report,
        );
        let total_value = scalar_or_nan(&total);
        let primary_weighted_value = primary_value * weights.primary;
        annotate_objective_scale_warnings(&mut auxiliary_metrics, primary_weighted_value);
        let nonfinite_loss_terms = nonfinite_loss_terms(
            self.primary_objective.name(),
            primary_value,
            &auxiliary_metrics,
            total_value,
        );
        let objective_gradient_diagnostics = pre_backward_objective_gradient_diagnostics(
            self.step,
            self.primary_objective.name(),
            &primary,
            primary_weighted_value,
            &auxiliaries,
            &auxiliary_metrics,
            weights,
            &self.config,
            var_store,
        );
        let mut primary_component_provenance = primary_components.provenance_records();
        annotate_primary_flow_component_provenance(
            &mut primary_component_provenance,
            &primary_branch_schedule,
        );
        let expected_activity = expected_gradient_activity(&self.config, &weights);
        let primary_objective_name = self.primary_objective.name().to_string();
        let gradient_health = GradientStepRunner {
            optimizer: &mut self.optimizer,
            var_store,
            config: &self.config,
            step: self.step,
            primary_objective_name: &primary_objective_name,
            primary_weighted_value,
            auxiliary_metrics: &auxiliary_metrics,
            expected_activity: &expected_activity,
            objective_gradient_diagnostics,
        };
        let gradient_health = gradient_health.apply(&total, nonfinite_loss_terms);

        let rollout_diagnostic_execution_count = if build_rollout_diagnostics {
            forwards.len()
        } else {
            0
        };
        let runtime_profile = runtime_tracker.finish(
            examples.len(),
            build_rollout_diagnostics,
            rollout_diagnostic_execution_count,
            objective_execution_counts.clone(),
        );

        let metrics = StepMetrics {
            step: self.step,
            generation_mode: forwards
                .first()
                .map(|forward| forward.generation.generation_mode.as_str().to_string())
                .unwrap_or_else(|| {
                    crate::config::GenerationModeConfig::TargetLigandDenoising
                        .as_str()
                        .to_string()
                }),
            epoch_index: order_metadata.epoch_index,
            sample_order_seed: order_metadata.sample_order_seed,
            batch_sample_indices: order_metadata.sample_indices,
            stage,
            stage_progress: build_stage_progress_metrics(
                &stage_context,
                &self.config,
                &auxiliary_metrics.auxiliary_objective_report,
                objective_execution_counts.clone(),
            ),
            losses: LossBreakdown {
                primary: PrimaryObjectiveMetrics {
                    objective_name: self.primary_objective.name().to_string(),
                    primary_value,
                    effective_weight: weights.primary,
                    weighted_value: primary_weighted_value,
                    enabled: weights.primary.is_finite() && weights.primary > 0.0,
                    decoder_anchored: self.primary_objective.name() != "surrogate_reconstruction",
                    components: primary_components,
                    component_provenance: primary_component_provenance,
                    component_scale_report: primary_component_scale_report,
                    branch_schedule: primary_branch_schedule,
                },
                auxiliaries: auxiliary_metrics,
                total: total_value,
            },
            interaction: InteractionStepMetrics::from_forwards(stage, Some(stage_index), &forwards),
            synchronization: SynchronizationHealthMetrics::from_forwards(&forwards),
            slot_utilization: SlotUtilizationStepMetrics::from_forwards(&forwards)
                .with_stage_index(stage_index),
            gradient_health,
            runtime_profile,
        };

        if CheckpointStepTrigger::new(self.config.training.checkpoint_every).should_save(self.step)
        {
            self.save_checkpoint_for_step(var_store, &metrics)?;
        }
        if self.config.training.log_every > 0 && self.step % self.config.training.log_every == 0 {
            log::info!(
                "step {} [{:?}] total={:.4} primary:{}={:.4} decoder_anchor={} intra_red={:.4} probe={:.4} probe_ligand_pharmacophore={:.4} probe_pocket_pharmacophore={:.4} leak={:.4} leak_core={:.4} leak_similarity_proxy_diagnostic={:.4} leak_explicit_probe_diagnostic={:.4} leak_topology_to_geometry={:.4} leak_geometry_to_topology={:.4} leak_pocket_to_geometry={:.4} leak_topology_to_pocket_role={:.4} leak_geometry_to_pocket_role={:.4} leak_pocket_to_ligand_role={:.4} gate={:.4} slot={:.4} consistency={:.4} pocket_contact={:.4} pocket_clash={:.4} pocket_envelope={:.4} valence_guardrail={:.4} bond_length_guardrail={:.4} mi_topo_geo={:.4} mi_topo_pocket={:.4} mi_geo_pocket={:.4} interaction_gate={:.4} interaction_sparsity={:.4} interaction_entropy={:.4} grad_norm={:.4} grad_nonfinite={} grad_clipped={} optimizer_step_skipped={} sync_mask_mismatch={} sync_slot_mismatch={} sync_frame_mismatch={} stale_context_steps={} refresh_count={} batch_slice_sync_pass={}",
                metrics.step,
                metrics.stage,
                metrics.losses.total,
                metrics.losses.primary.objective_name,
                metrics.losses.primary.primary_value,
                metrics.losses.primary.decoder_anchored,
                metrics.losses.auxiliaries.intra_red,
                metrics.losses.auxiliaries.probe,
                metrics
                    .losses
                    .auxiliaries
                    .probe_ligand_pharmacophore,
                metrics.losses.auxiliaries.probe_pocket_pharmacophore,
                metrics.losses.auxiliaries.leak,
                metrics.losses.auxiliaries.leak_core,
                metrics
                    .losses
                    .auxiliaries
                    .leak_similarity_proxy_diagnostic,
                metrics.losses.auxiliaries.leak_explicit_probe_diagnostic,
                metrics.losses.auxiliaries.leak_topology_to_geometry,
                metrics.losses.auxiliaries.leak_geometry_to_topology,
                metrics.losses.auxiliaries.leak_pocket_to_geometry,
                metrics
                    .losses
                    .auxiliaries
                    .leak_topology_to_pocket_role,
                metrics
                    .losses
                    .auxiliaries
                    .leak_geometry_to_pocket_role,
                metrics.losses.auxiliaries.leak_pocket_to_ligand_role,
                metrics.losses.auxiliaries.gate,
                metrics.losses.auxiliaries.slot,
                metrics.losses.auxiliaries.consistency,
                metrics.losses.auxiliaries.pocket_contact,
                metrics.losses.auxiliaries.pocket_clash,
                metrics.losses.auxiliaries.pocket_envelope,
                metrics.losses.auxiliaries.valence_guardrail,
                metrics.losses.auxiliaries.bond_length_guardrail,
                metrics.losses.auxiliaries.mi_topo_geo,
                metrics.losses.auxiliaries.mi_topo_pocket,
                metrics.losses.auxiliaries.mi_geo_pocket,
                metrics.interaction.mean_gate,
                metrics.interaction.mean_gate_sparsity,
                metrics.interaction.mean_attention_entropy,
                metrics.gradient_health.global_grad_l2_norm,
                metrics.gradient_health.nonfinite_gradient_tensors,
                metrics.gradient_health.clipped,
                metrics.gradient_health.optimizer_step_skipped,
                metrics.synchronization.mask_count_mismatch,
                metrics.synchronization.slot_count_mismatch,
                metrics.synchronization.coordinate_frame_mismatch,
                metrics.synchronization.stale_context_steps,
                metrics.synchronization.refresh_count,
                metrics.synchronization.batch_slice_sync_pass,
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

    /// Train over configured deterministic mini-batches until `max_steps` or
    /// the optional epoch cap is reached.
    pub fn fit(
        &mut self,
        var_store: &nn::VarStore,
        system: &Phase1ResearchSystem,
        examples: &[MolecularExample],
    ) -> Result<Vec<StepMetrics>, Box<dyn std::error::Error>> {
        let source = BorrowedExampleSource::new(examples);
        self.fit_source(var_store, system, &source)
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

    /// Save a standard latest/step checkpoint for a completed step.
    pub fn save_checkpoint_for_step(
        &self,
        var_store: &nn::VarStore,
        metrics: &StepMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.checkpoints.save(
            var_store,
            metrics.step,
            Some(metrics),
            Some(stable_json_hash(&self.config)),
            self.dataset_validation_fingerprint.clone(),
            METRIC_SCHEMA_VERSION,
            RESUME_CONTRACT_VERSION,
            Some(self.optimizer_state_metadata()),
            Some(self.scheduler_state_metadata(metrics.step)),
            Some(determinism_controls_from_config(&self.config)),
            Some(self.backend_training_metadata()),
        )
    }

    /// Save a validation-selected best checkpoint without mutating latest metadata.
    pub fn save_best_checkpoint_for_step(
        &self,
        var_store: &nn::VarStore,
        metrics: &StepMetrics,
        validation_step: usize,
        metric_name: &str,
        metric_value: f64,
        higher_is_better: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.checkpoints.save_best(
            var_store,
            metrics.step,
            Some(metrics),
            Some(stable_json_hash(&self.config)),
            self.dataset_validation_fingerprint.clone(),
            METRIC_SCHEMA_VERSION,
            RESUME_CONTRACT_VERSION,
            Some(self.optimizer_state_metadata()),
            Some(self.scheduler_state_metadata(metrics.step)),
            Some(determinism_controls_from_config(&self.config)),
            Some(self.backend_training_metadata()),
            validation_step,
            metric_name,
            metric_value,
            higher_is_better,
        )
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

    fn update_primary_component_scale_ema(
        &mut self,
        components: &crate::training::PrimaryObjectiveComponentMetrics,
    ) {
        let Some(momentum) = self
            .config
            .training
            .objective_scale_diagnostics
            .running_scale_momentum
        else {
            return;
        };
        let epsilon = self.config.training.objective_scale_diagnostics.epsilon;
        for (component_name, value) in components.observed_component_values() {
            let observed = value.abs().max(epsilon);
            self.primary_component_scale_ema
                .entry(component_name.to_string())
                .and_modify(|scale| *scale = momentum * *scale + (1.0 - momentum) * observed)
                .or_insert(observed);
        }
    }

    fn optimizer_state_metadata(&self) -> OptimizerStateMetadata {
        OptimizerStateMetadata {
            optimizer_kind: "adam".to_string(),
            learning_rate: self.config.training.learning_rate,
            weight_decay: 0.0,
            internal_state_persisted: false,
            resume_mode: crate::training::ResumeMode::WeightsOnlyResume,
            state_persistence_backend: "metadata_only_tch_0_23".to_string(),
            exact_resume_supported: false,
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
            pharmacophore_probe_weight: weights.pharmacophore_probe,
            leak_weight: weights.leak,
            pharmacophore_leakage_weight: weights.pharmacophore_leakage,
            gate_weight: weights.gate,
            slot_weight: weights.slot,
            consistency_weight: weights.consistency,
            pocket_contact_weight: weights.pocket_contact,
            pocket_clash_weight: weights.pocket_clash,
            pocket_envelope_weight: weights.pocket_envelope,
            valence_guardrail_weight: weights.valence_guardrail,
            bond_length_guardrail_weight: weights.bond_length_guardrail,
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
            generation_mode: self
                .config
                .data
                .generation_target
                .generation_mode
                .as_str()
                .to_string(),
            flow_contract_version: crate::models::current_multimodal_flow_contract(
                &self.config.generation_method.flow_matching,
            )
            .flow_contract_version,
            flow_branch_schedule_hash: stable_json_hash(
                &self
                    .config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule,
            ),
            raw_processed_evaluation_contract: "raw_rollout_model_design_contract_v1".to_string(),
            shared_auxiliary_objectives: vec![
                "L_consistency".to_string(),
                "L_intra_red".to_string(),
                "L_probe".to_string(),
                "L_pharmacophore_probe".to_string(),
                "L_leak".to_string(),
                "L_pharmacophore_leakage".to_string(),
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
            || saved.generation_mode != current.generation_mode
            || saved.flow_contract_version != current.flow_contract_version
            || saved.flow_branch_schedule_hash != current.flow_branch_schedule_hash
            || saved.raw_processed_evaluation_contract != current.raw_processed_evaluation_contract
        {
            return Err(format!(
                "checkpoint backend/objective mismatch or replay-contract mismatch: saved {}:{}:{}:{}:{}:{}:{} but current {}:{}:{}:{}:{}:{}:{}",
                saved.backend_id,
                saved.backend_family,
                saved.objective_name,
                saved.generation_mode,
                saved.flow_contract_version,
                saved.flow_branch_schedule_hash,
                saved.raw_processed_evaluation_contract,
                current.backend_id,
                current.backend_family,
                current.objective_name,
                current.generation_mode,
                current.flow_contract_version,
                current.flow_branch_schedule_hash,
                current.raw_processed_evaluation_contract,
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

fn annotate_objective_scale_warnings(
    metrics: &mut AuxiliaryLossMetrics,
    primary_weighted_value: f64,
) {
    let primary_abs = primary_weighted_value.abs().max(1.0e-12);
    for entry in &mut metrics.auxiliary_objective_report.entries {
        if entry.enabled
            && entry.weighted_value.is_finite()
            && entry.weighted_value.abs() > primary_abs * 10.0
        {
            entry.status = "dominant".to_string();
            entry.warning = Some(
                "weighted auxiliary contribution exceeds 10x the weighted primary objective"
                    .to_string(),
            );
        }
    }
}

fn primary_branch_schedule_report(
    forwards: &[ResearchForward],
    training_step: usize,
    stage_index: Option<usize>,
    components: &crate::training::PrimaryObjectiveComponentMetrics,
    config: &ResearchConfig,
) -> PrimaryBranchScheduleReport {
    let Some(flow) = forwards
        .iter()
        .find_map(|forward| forward.generation.flow_matching.as_ref())
    else {
        return PrimaryBranchScheduleReport::default();
    };
    let objective_flow_weight = primary_flow_objective_weight(config);
    let geometry_values = branch_scale_values(
        components.flow_velocity.unwrap_or(0.0) + components.flow_endpoint.unwrap_or(0.0),
        flow.branch_weights.geometry,
        objective_flow_weight,
    );
    let mut entries = vec![PrimaryBranchWeightRecord {
        branch_name: "geometry".to_string(),
        unweighted_value: geometry_values.unweighted,
        effective_weight: flow.branch_weights.geometry,
        schedule_multiplier: config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .geometry
            .effective_multiplier(Some(training_step)),
        weighted_value: geometry_values.weighted,
        optimizer_facing: flow.branch_weights.geometry.is_finite()
            && flow.branch_weights.geometry > 0.0,
        provenance: flow.flow_contract_version.clone(),
        target_matching_policy: Some(flow.target_matching_policy.clone()),
        target_matching_mean_cost: Some(flow.target_matching_mean_cost),
        target_matching_max_cost: Some(flow.target_matching_cost_summary.max_cost),
        target_matching_total_cost: Some(flow.target_matching_cost_summary.total_cost),
        target_matching_coverage: Some(flow.target_matching_coverage),
        target_matching_matched_count: Some(flow.target_matching_cost_summary.matched_count),
        target_matching_unmatched_generated_count: Some(
            flow.target_matching_cost_summary.unmatched_generated_count,
        ),
        target_matching_unmatched_target_count: Some(
            flow.target_matching_cost_summary.unmatched_target_count,
        ),
        target_matching_exact_assignment: Some(flow.target_matching_cost_summary.exact_assignment),
    }];
    if let Some(molecular) = flow.molecular.as_ref() {
        for (branch_name, effective_weight, schedule_multiplier, component_value) in [
            (
                "atom_type",
                molecular.branch_weights.atom_type,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .atom_type
                    .effective_multiplier(Some(training_step)),
                components.flow_atom_type.unwrap_or(0.0),
            ),
            (
                "bond",
                molecular.branch_weights.bond,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .bond
                    .effective_multiplier(Some(training_step)),
                components.flow_bond.unwrap_or(0.0),
            ),
            (
                "topology",
                molecular.branch_weights.topology,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .topology
                    .effective_multiplier(Some(training_step)),
                components.flow_topology.unwrap_or(0.0),
            ),
            (
                "pocket_context",
                molecular.branch_weights.pocket_context,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .pocket_context
                    .effective_multiplier(Some(training_step)),
                components.flow_pocket_context.unwrap_or(0.0),
            ),
            (
                "synchronization",
                molecular.branch_weights.synchronization,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .synchronization
                    .effective_multiplier(Some(training_step)),
                components.flow_synchronization.unwrap_or(0.0),
            ),
        ] {
            let uses_atom_matching = branch_name != "pocket_context";
            let branch_values =
                branch_scale_values(component_value, effective_weight, objective_flow_weight);
            entries.push(PrimaryBranchWeightRecord {
                branch_name: branch_name.to_string(),
                unweighted_value: branch_values.unweighted,
                effective_weight,
                schedule_multiplier,
                weighted_value: branch_values.weighted,
                optimizer_facing: effective_weight.is_finite() && effective_weight > 0.0,
                provenance: format!(
                    "{}:{}",
                    flow.flow_contract_version.as_str(),
                    molecular.target_alignment_policy.as_str()
                ),
                target_matching_policy: uses_atom_matching
                    .then(|| molecular.target_matching_policy.clone()),
                target_matching_mean_cost: uses_atom_matching
                    .then_some(molecular.target_matching_mean_cost),
                target_matching_max_cost: uses_atom_matching
                    .then_some(molecular.target_matching_cost_summary.max_cost),
                target_matching_total_cost: uses_atom_matching
                    .then_some(molecular.target_matching_cost_summary.total_cost),
                target_matching_coverage: uses_atom_matching
                    .then_some(molecular.target_matching_coverage),
                target_matching_matched_count: uses_atom_matching
                    .then_some(molecular.target_matching_cost_summary.matched_count),
                target_matching_unmatched_generated_count: uses_atom_matching.then_some(
                    molecular
                        .target_matching_cost_summary
                        .unmatched_generated_count,
                ),
                target_matching_unmatched_target_count: uses_atom_matching.then_some(
                    molecular
                        .target_matching_cost_summary
                        .unmatched_target_count,
                ),
                target_matching_exact_assignment: uses_atom_matching
                    .then_some(molecular.target_matching_cost_summary.exact_assignment),
            });
        }
    }
    PrimaryBranchScheduleReport {
        observed: true,
        training_step: Some(training_step),
        stage_index,
        source: "generation_method.flow_matching.multi_modal.branch_schedule".to_string(),
        entries,
    }
}

struct BranchScaleValues {
    unweighted: f64,
    weighted: f64,
}

fn primary_flow_objective_weight(config: &ResearchConfig) -> f64 {
    match config.training.primary_objective {
        PrimaryObjectiveConfig::FlowMatching => config.training.flow_matching_loss_weight,
        PrimaryObjectiveConfig::DenoisingFlowMatching => config.training.hybrid_flow_weight,
        _ => 1.0,
    }
}

fn branch_scale_values(
    component_value: f64,
    effective_weight: f64,
    objective_flow_weight: f64,
) -> BranchScaleValues {
    let weighted = if objective_flow_weight.is_finite() && objective_flow_weight > 0.0 {
        component_value / objective_flow_weight
    } else {
        component_value
    };
    let unweighted = if effective_weight.is_finite() && effective_weight > 0.0 {
        weighted / effective_weight
    } else {
        0.0
    };
    BranchScaleValues {
        unweighted,
        weighted,
    }
}

fn annotate_primary_flow_component_provenance(
    records: &mut [crate::training::PrimaryObjectiveComponentProvenance],
    branch_schedule: &PrimaryBranchScheduleReport,
) {
    if !branch_schedule.observed {
        return;
    }
    for record in records {
        let Some(branch_name) = primary_component_branch_name(&record.component_name) else {
            continue;
        };
        if let Some(branch) = branch_schedule
            .entries
            .iter()
            .find(|entry| entry.branch_name == branch_name)
        {
            record.effective_branch_weight = Some(branch.effective_weight);
            record.branch_schedule_source = Some(branch_schedule.source.clone());
        }
    }
}

fn primary_component_branch_name(component_name: &str) -> Option<&'static str> {
    match component_name {
        "flow_velocity" | "flow_endpoint" => Some("geometry"),
        "flow_atom_type" => Some("atom_type"),
        "flow_bond" => Some("bond"),
        "flow_topology" => Some("topology"),
        "flow_pocket_context" => Some("pocket_context"),
        "flow_synchronization" => Some("synchronization"),
        _ => None,
    }
}

fn build_stage_progress_metrics(
    stage_context: &StageStepContext,
    config: &ResearchConfig,
    auxiliary_report: &AuxiliaryObjectiveReport,
    objective_execution_counts: ObjectiveExecutionCountMetrics,
) -> StageProgressMetrics {
    StageProgressMetrics {
        stage_index: stage_context.stage_index,
        fixed_stage_index: stage_context.fixed_stage.index(),
        stage_ramp: stage_context.stage_ramp,
        active_objective_families: active_objective_families(
            stage_context.weights.primary,
            auxiliary_report,
        ),
        adaptive_stage_enabled: config.training.adaptive_stage_guard.enabled,
        adaptive_stage_hold: stage_context.selection.held,
        readiness_status: stage_context.selection.readiness_status.clone(),
        readiness_reasons: stage_context.selection.readiness_reasons.clone(),
        objective_execution_counts,
    }
}

struct ObjectiveGradientFamilyTensor {
    family_name: String,
    weighted_value: f64,
    provenance: String,
    objective: Tensor,
}

fn pre_backward_objective_gradient_diagnostics(
    step: usize,
    primary_objective_name: &str,
    primary: &Tensor,
    primary_weighted_value: f64,
    auxiliaries: &AuxiliaryObjectiveTensors,
    auxiliary_metrics: &AuxiliaryLossMetrics,
    weights: EffectiveLossWeights,
    config: &ResearchConfig,
    var_store: &nn::VarStore,
) -> Option<ObjectiveGradientDiagnostics> {
    let diagnostics = &config.training.objective_gradient_diagnostics;
    if !diagnostics.enabled {
        return Some(ObjectiveGradientDiagnostics {
            enabled: false,
            sampled: false,
            sample_every_steps: diagnostics.sample_every_steps,
            sampling_mode: "disabled".to_string(),
            entries: Vec::new(),
        });
    }
    if diagnostics.sample_every_steps == 0 || step % diagnostics.sample_every_steps != 0 {
        return Some(ObjectiveGradientDiagnostics {
            enabled: true,
            sampled: false,
            sample_every_steps: diagnostics.sample_every_steps,
            sampling_mode: "interval_skipped".to_string(),
            entries: Vec::new(),
        });
    }
    if diagnostics.sampling_mode == ObjectiveGradientSamplingMode::LossShareProxy {
        return None;
    }

    Some(exact_sampled_objective_gradient_diagnostics(
        primary_objective_name,
        primary,
        primary_weighted_value,
        auxiliaries,
        auxiliary_metrics,
        weights,
        config,
        var_store,
    ))
}

fn exact_sampled_objective_gradient_diagnostics(
    primary_objective_name: &str,
    primary: &Tensor,
    primary_weighted_value: f64,
    auxiliaries: &AuxiliaryObjectiveTensors,
    auxiliary_metrics: &AuxiliaryLossMetrics,
    weights: EffectiveLossWeights,
    config: &ResearchConfig,
    var_store: &nn::VarStore,
) -> ObjectiveGradientDiagnostics {
    let families = objective_gradient_family_tensors(
        primary_objective_name,
        primary,
        primary_weighted_value,
        auxiliaries,
        auxiliary_metrics,
        weights,
        config,
    );
    let parameter_tensors = var_store
        .variables()
        .into_values()
        .filter(|tensor| tensor.requires_grad())
        .collect::<Vec<_>>();
    let mut entries = families
        .iter()
        .map(|family| exact_objective_gradient_family_metric(family, &parameter_tensors))
        .collect::<Vec<_>>();
    let norm_total = entries
        .iter()
        .filter_map(|entry| {
            entry
                .grad_l2_norm
                .is_finite()
                .then_some(entry.grad_l2_norm.max(0.0))
        })
        .sum::<f64>();
    if norm_total > 0.0 {
        for entry in &mut entries {
            entry.grad_norm_fraction = entry.grad_l2_norm.max(0.0) / norm_total;
        }
    }

    ObjectiveGradientDiagnostics {
        enabled: true,
        sampled: true,
        sample_every_steps: config
            .training
            .objective_gradient_diagnostics
            .sample_every_steps,
        sampling_mode: "exact_sampled_retained_graph".to_string(),
        entries,
    }
}

fn objective_gradient_family_tensors(
    primary_objective_name: &str,
    primary: &Tensor,
    primary_weighted_value: f64,
    auxiliaries: &AuxiliaryObjectiveTensors,
    auxiliary_metrics: &AuxiliaryLossMetrics,
    weights: EffectiveLossWeights,
    config: &ResearchConfig,
) -> Vec<ObjectiveGradientFamilyTensor> {
    let diagnostics = &config.training.objective_gradient_diagnostics;
    let mut families = Vec::new();
    let primary_family_name = format!("primary:{primary_objective_name}");
    if objective_gradient_family_selected(diagnostics, "primary", &primary_family_name) {
        families.push(ObjectiveGradientFamilyTensor {
            family_name: primary_family_name,
            weighted_value: primary_weighted_value,
            provenance: "primary_objective:exact_autograd".to_string(),
            objective: primary * weights.primary,
        });
    }
    if !diagnostics.include_auxiliary {
        return families;
    }

    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::IntraRed,
        &auxiliaries.intra_red * weights.intra_red,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::Probe,
        &auxiliaries.probe_core * weights.probe,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::PharmacophoreProbe,
        (&auxiliaries.probe_ligand_pharmacophore + &auxiliaries.probe_pocket_pharmacophore)
            * weights.pharmacophore_probe,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::Leak,
        (&auxiliaries.leak_core
            + &auxiliaries.leak_topology_to_geometry
            + &auxiliaries.leak_geometry_to_topology
            + &auxiliaries.leak_pocket_to_geometry)
            * weights.leak,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::PharmacophoreLeakage,
        (&auxiliaries.leak_topology_to_pocket_role
            + &auxiliaries.leak_geometry_to_pocket_role
            + &auxiliaries.leak_pocket_to_ligand_role)
            * weights.pharmacophore_leakage,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::Gate,
        &auxiliaries.gate * weights.gate,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::Slot,
        &auxiliaries.slot * weights.slot,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::Consistency,
        &auxiliaries.consistency * weights.consistency,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::PocketContact,
        &auxiliaries.pocket_contact * weights.pocket_contact,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::PocketClash,
        &auxiliaries.pocket_clash * weights.pocket_clash,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::PocketEnvelope,
        &auxiliaries.pocket_envelope * weights.pocket_envelope,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::ValenceGuardrail,
        &auxiliaries.valence_guardrail * weights.valence_guardrail,
    );
    push_auxiliary_gradient_family(
        &mut families,
        diagnostics,
        auxiliary_metrics,
        AuxiliaryObjectiveFamily::BondLengthGuardrail,
        &auxiliaries.bond_length_guardrail * weights.bond_length_guardrail,
    );
    families
}

fn push_auxiliary_gradient_family(
    families: &mut Vec<ObjectiveGradientFamilyTensor>,
    diagnostics: &crate::config::ObjectiveGradientDiagnosticsConfig,
    auxiliary_metrics: &AuxiliaryLossMetrics,
    family: AuxiliaryObjectiveFamily,
    objective: Tensor,
) {
    let Some(report) = auxiliary_metrics
        .auxiliary_objective_report
        .entries
        .iter()
        .find(|entry| entry.family == family)
    else {
        return;
    };
    if !report.enabled {
        return;
    }
    let short_name = family.as_str();
    let family_name = format!("auxiliary:{short_name}");
    if !objective_gradient_family_selected(diagnostics, short_name, &family_name) {
        return;
    }
    families.push(ObjectiveGradientFamilyTensor {
        family_name,
        weighted_value: report.weighted_value,
        provenance: format!("auxiliary:{}:exact_autograd", report.execution_mode),
        objective,
    });
}

fn objective_gradient_family_selected(
    diagnostics: &crate::config::ObjectiveGradientDiagnosticsConfig,
    short_name: &str,
    family_name: &str,
) -> bool {
    diagnostics.included_families.is_empty()
        || diagnostics.included_families.iter().any(|configured| {
            let configured = configured.trim();
            configured == short_name || configured == family_name
        })
}

fn exact_objective_gradient_family_metric(
    family: &ObjectiveGradientFamilyTensor,
    parameter_tensors: &[Tensor],
) -> ObjectiveGradientFamilyMetrics {
    if !family.weighted_value.is_finite() {
        return objective_gradient_family_metric(
            family,
            0.0,
            0.0,
            "nonfinite_weighted_value",
            Some("weighted objective value is non-finite".to_string()),
        );
    }
    if !family.objective.requires_grad() {
        return objective_gradient_family_metric(
            family,
            0.0,
            0.0,
            "no_gradient",
            Some("objective tensor is detached or constant".to_string()),
        );
    }
    if parameter_tensors.is_empty() {
        return objective_gradient_family_metric(
            family,
            0.0,
            0.0,
            "no_parameters",
            Some("no trainable tensors were available for gradient sampling".to_string()),
        );
    }

    let gradients =
        match Tensor::f_run_backward(&[&family.objective], parameter_tensors, true, false) {
            Ok(gradients) => gradients,
            Err(err) => {
                return objective_gradient_family_metric(
                    family,
                    0.0,
                    0.0,
                    "backward_error",
                    Some(err.to_string()),
                );
            }
        };
    let mut gradient_tensor_count = 0_usize;
    let mut grad_sq_sum = 0.0_f64;
    let mut nonfinite_gradient = false;
    for gradient in gradients {
        if !gradient.defined() || gradient.numel() == 0 {
            continue;
        }
        gradient_tensor_count += 1;
        if !tensor_is_finite(&gradient) {
            nonfinite_gradient = true;
            continue;
        }
        let sq_sum = (&gradient * &gradient).sum(Kind::Float).double_value(&[]);
        if sq_sum.is_finite() && sq_sum >= 0.0 {
            grad_sq_sum += sq_sum;
        }
    }
    if nonfinite_gradient {
        return objective_gradient_family_metric(
            family,
            0.0,
            0.0,
            "nonfinite_gradient",
            Some("sampled objective gradients contained non-finite values".to_string()),
        );
    }
    if gradient_tensor_count == 0 {
        return objective_gradient_family_metric(
            family,
            0.0,
            0.0,
            "no_gradient",
            Some("sampled objective did not reach any trainable tensor".to_string()),
        );
    }
    let grad_l2_norm = grad_sq_sum.max(0.0).sqrt();
    if grad_l2_norm == 0.0 {
        return objective_gradient_family_metric(
            family,
            grad_l2_norm,
            0.0,
            "zero_gradient",
            Some("sampled objective gradients are exactly zero".to_string()),
        );
    }
    objective_gradient_family_metric(family, grad_l2_norm, 0.0, "exact_sampled", None)
}

fn objective_gradient_family_metric(
    family: &ObjectiveGradientFamilyTensor,
    grad_l2_norm: f64,
    grad_norm_fraction: f64,
    status: &str,
    anomaly: Option<String>,
) -> ObjectiveGradientFamilyMetrics {
    ObjectiveGradientFamilyMetrics {
        family_name: family.family_name.clone(),
        weighted_value: family.weighted_value,
        grad_l2_norm,
        grad_norm_fraction,
        status: status.to_string(),
        provenance: family.provenance.clone(),
        anomaly,
    }
}

fn loss_share_objective_gradient_diagnostics(
    step: usize,
    primary_objective_name: &str,
    primary_weighted_value: f64,
    auxiliaries: &AuxiliaryLossMetrics,
    config: &ResearchConfig,
    global_grad_l2_norm: f64,
    nonfinite_gradient_tensors: usize,
) -> ObjectiveGradientDiagnostics {
    let diagnostics = &config.training.objective_gradient_diagnostics;
    if !diagnostics.enabled {
        return ObjectiveGradientDiagnostics {
            enabled: false,
            sampled: false,
            sample_every_steps: diagnostics.sample_every_steps,
            sampling_mode: "disabled".to_string(),
            entries: Vec::new(),
        };
    }
    if diagnostics.sample_every_steps == 0 || step % diagnostics.sample_every_steps != 0 {
        return ObjectiveGradientDiagnostics {
            enabled: true,
            sampled: false,
            sample_every_steps: diagnostics.sample_every_steps,
            sampling_mode: "interval_skipped".to_string(),
            entries: Vec::new(),
        };
    }

    let primary_family_name = format!("primary:{primary_objective_name}");
    let mut families =
        if objective_gradient_family_selected(diagnostics, "primary", &primary_family_name) {
            vec![(
                primary_family_name,
                primary_weighted_value,
                "primary_objective:loss_share_proxy".to_string(),
            )]
        } else {
            Vec::new()
        };
    if diagnostics.include_auxiliary {
        families.extend(
            auxiliaries
                .auxiliary_objective_report
                .entries
                .iter()
                .filter(|entry| {
                    entry.enabled
                        && objective_gradient_family_selected(
                            diagnostics,
                            entry.family.as_str(),
                            &format!("auxiliary:{}", entry.family.as_str()),
                        )
                })
                .map(|entry| {
                    (
                        format!("auxiliary:{}", entry.family.as_str()),
                        entry.weighted_value,
                        format!("auxiliary:{}:loss_share_proxy", entry.execution_mode),
                    )
                }),
        );
    }
    let total_abs = families
        .iter()
        .filter_map(|(_, value, _)| value.is_finite().then_some(value.abs()))
        .sum::<f64>();
    let entries = families
        .into_iter()
        .map(|(family_name, weighted_value, provenance)| {
            let (grad_norm_fraction, grad_l2_norm) =
                if weighted_value.is_finite() && total_abs > 0.0 && global_grad_l2_norm.is_finite()
                {
                    let fraction = weighted_value.abs() / total_abs;
                    (fraction, global_grad_l2_norm * fraction)
                } else {
                    (0.0, 0.0)
                };
            let (status, anomaly) = if !weighted_value.is_finite() {
                (
                    "nonfinite_weighted_value".to_string(),
                    Some("weighted objective value is non-finite".to_string()),
                )
            } else if nonfinite_gradient_tensors > 0 {
                (
                    "nonfinite_gradient".to_string(),
                    Some(format!(
                        "{nonfinite_gradient_tensors} gradient tensors were non-finite"
                    )),
                )
            } else if global_grad_l2_norm == 0.0 {
                (
                    "zero_gradient".to_string(),
                    Some("global post-backward gradient norm is zero".to_string()),
                )
            } else {
                ("loss_share_proxy".to_string(), None)
            };
            ObjectiveGradientFamilyMetrics {
                family_name,
                weighted_value,
                grad_l2_norm,
                grad_norm_fraction,
                status,
                provenance,
                anomaly,
            }
        })
        .collect();

    ObjectiveGradientDiagnostics {
        enabled: true,
        sampled: true,
        sample_every_steps: diagnostics.sample_every_steps,
        sampling_mode: "weighted_loss_share_post_backward_proxy".to_string(),
        entries,
    }
}

fn active_objective_families(
    primary_weight: f64,
    auxiliary_report: &AuxiliaryObjectiveReport,
) -> Vec<String> {
    let mut active = Vec::new();
    if primary_weight.is_finite() && primary_weight > 0.0 {
        active.push("primary".to_string());
    }
    active.extend(
        auxiliary_report
            .entries
            .iter()
            .filter(|entry| entry.enabled)
            .map(|entry| entry.family.as_str().to_string()),
    );
    active
}

fn objective_execution_counts(
    primary_weight: f64,
    auxiliary_report: &AuxiliaryObjectiveReport,
) -> ObjectiveExecutionCountMetrics {
    let primary_enabled_count = usize::from(primary_weight.is_finite() && primary_weight > 0.0);
    let trainable_auxiliary_count = auxiliary_report
        .entries
        .iter()
        .filter(|entry| entry.execution_mode == "trainable")
        .count();
    let detached_diagnostic_count = auxiliary_report
        .entries
        .iter()
        .filter(|entry| entry.execution_mode == "detached_diagnostic")
        .count();
    let skipped_zero_weight_count = auxiliary_report
        .entries
        .iter()
        .filter(|entry| entry.execution_mode == "skipped_zero_weight")
        .count();
    let optimizer_facing_count = primary_enabled_count
        + auxiliary_report
            .entries
            .iter()
            .filter(|entry| {
                entry.execution_mode == "trainable"
                    && entry.enabled
                    && entry.effective_weight.is_finite()
                    && entry.effective_weight > 0.0
            })
            .count();

    ObjectiveExecutionCountMetrics {
        primary_enabled_count,
        trainable_auxiliary_count,
        detached_diagnostic_count,
        skipped_zero_weight_count,
        optimizer_facing_count,
    }
}

fn stage_readiness_reasons(
    guard: &crate::config::AdaptiveStageGuardConfig,
    history: &[StepMetrics],
) -> Vec<String> {
    if history.is_empty() {
        return vec!["no prior step metrics are available for adaptive readiness".to_string()];
    }

    let window_len = guard.readiness_window.min(history.len());
    let window = &history[history.len() - window_len..];
    let latest = window.last().expect("readiness window is non-empty");
    let mut reasons = Vec::new();

    let finite_step_fraction = window
        .iter()
        .filter(|metrics| {
            metrics.losses.total.is_finite()
                && metrics.losses.primary.primary_value.is_finite()
                && metrics.losses.primary.weighted_value.is_finite()
        })
        .count() as f64
        / window.len() as f64;
    if finite_step_fraction < guard.min_finite_step_fraction {
        reasons.push(format!(
            "finite_step_fraction={finite_step_fraction:.3} below required {:.3}",
            guard.min_finite_step_fraction
        ));
    }
    if !latest.losses.primary.primary_value.is_finite() {
        reasons.push("latest primary loss is non-finite".to_string());
    }
    if latest.gradient_health.nonfinite_gradient_tensors > guard.max_nonfinite_gradient_tensors {
        reasons.push(format!(
            "nonfinite_gradient_tensors={} exceeds limit {}",
            latest.gradient_health.nonfinite_gradient_tensors, guard.max_nonfinite_gradient_tensors
        ));
    }
    if guard.require_no_optimizer_skip && latest.gradient_health.optimizer_step_skipped {
        reasons.push("latest optimizer step was skipped".to_string());
    }
    if latest.slot_utilization.collapse_warning_count > guard.max_slot_collapse_warnings {
        reasons.push(format!(
            "slot_collapse_warning_count={} exceeds limit {}",
            latest.slot_utilization.collapse_warning_count, guard.max_slot_collapse_warnings
        ));
    }
    if let Some(minimum) = guard.min_slot_signature_matching_score {
        if latest.slot_utilization.slot_signatures.is_empty() {
            reasons.push(format!(
                "slot_signature_matching_unavailable below required {minimum:.3}"
            ));
        }
        for signature in &latest.slot_utilization.slot_signatures {
            if signature.matching_score.is_finite() && signature.matching_score < minimum {
                reasons.push(format!(
                    "{}_slot_signature_matching_score={:.3} below required {minimum:.3}",
                    signature.modality, signature.matching_score
                ));
            }
        }
    }
    let mean_gate_saturation = if latest.interaction.paths.is_empty() {
        0.0
    } else {
        latest
            .interaction
            .paths
            .iter()
            .map(|path| path.gate_saturation_fraction)
            .sum::<f64>()
            / latest.interaction.paths.len() as f64
    };
    if mean_gate_saturation > guard.max_gate_saturation_fraction {
        reasons.push(format!(
            "mean_gate_saturation_fraction={mean_gate_saturation:.3} exceeds limit {:.3}",
            guard.max_gate_saturation_fraction
        ));
    }
    if let Some(maximum) = guard.max_leakage_diagnostic {
        let leakage_diagnostic = latest
            .losses
            .auxiliaries
            .leak_similarity_proxy_diagnostic
            .max(latest.losses.auxiliaries.leak_explicit_probe_diagnostic)
            .max(latest.losses.auxiliaries.leak);
        if leakage_diagnostic.is_finite() && leakage_diagnostic > maximum {
            reasons.push(format!(
                "leakage_diagnostic={leakage_diagnostic:.3} exceeds limit {maximum:.3}"
            ));
        } else if !leakage_diagnostic.is_finite() {
            reasons.push("leakage_diagnostic is non-finite".to_string());
        }
    }
    if guard.min_primary_loss_improvement_fraction > 0.0 {
        if window.len() < 2 {
            reasons.push(
                "insufficient history for configured primary loss trend readiness".to_string(),
            );
        } else {
            let first = window
                .first()
                .map(|metrics| metrics.losses.primary.primary_value.abs())
                .unwrap_or(0.0);
            let last = latest.losses.primary.primary_value.abs();
            let improvement_fraction = if first <= 1.0e-12 {
                0.0
            } else {
                (first - last) / first
            };
            if improvement_fraction < guard.min_primary_loss_improvement_fraction {
                reasons.push(format!(
                    "primary_loss_improvement_fraction={improvement_fraction:.3} below required {:.3}",
                    guard.min_primary_loss_improvement_fraction
                ));
            }
        }
    }

    reasons
}

fn previous_stage(stage: TrainingStage) -> TrainingStage {
    match stage {
        TrainingStage::Stage1 => TrainingStage::Stage1,
        TrainingStage::Stage2 => TrainingStage::Stage1,
        TrainingStage::Stage3 => TrainingStage::Stage2,
        TrainingStage::Stage4 => TrainingStage::Stage3,
    }
}

fn representative_step_for_stage(config: &ResearchConfig, stage: TrainingStage) -> usize {
    let schedule = &config.training.schedule;
    match stage {
        TrainingStage::Stage1 => 0,
        TrainingStage::Stage2 => schedule.stage1_steps,
        TrainingStage::Stage3 => schedule.stage2_steps,
        TrainingStage::Stage4 => schedule.stage3_steps,
    }
}

fn current_used_memory_mb() -> f64 {
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();
    sys.used_memory() as f64 / (1024.0 * 1024.0)
}

#[derive(Debug, Default)]
struct GradientAccumulator {
    trainable_tensor_count: usize,
    gradient_tensor_count: usize,
    nonfinite_gradient_tensors: usize,
    grad_sq_sum: f64,
    grad_abs_max: f64,
}

impl GradientAccumulator {
    fn observe_variable(&mut self, tensor: &Tensor) {
        self.trainable_tensor_count += 1;
        let grad = tensor.grad();
        if !grad.defined() || grad.numel() == 0 {
            return;
        }
        self.gradient_tensor_count += 1;
        if !tensor_is_finite(&grad) {
            self.nonfinite_gradient_tensors += 1;
            return;
        }
        let sq_sum = (&grad * &grad).sum(Kind::Float).double_value(&[]);
        if sq_sum.is_finite() && sq_sum >= 0.0 {
            self.grad_sq_sum += sq_sum;
        }
        let abs_max = grad.abs().max().double_value(&[]);
        if abs_max.is_finite() {
            self.grad_abs_max = self.grad_abs_max.max(abs_max);
        }
    }

    fn finish(self, module_name: &str, expected_active: bool) -> GradientModuleMetrics {
        let grad_l2_norm = self.grad_sq_sum.max(0.0).sqrt();
        let status = if self.nonfinite_gradient_tensors > 0 {
            "nonfinite_gradient"
        } else if self.gradient_tensor_count == 0 {
            "no_gradient"
        } else if grad_l2_norm > 0.0 {
            "active"
        } else {
            "zero_gradient"
        };
        let inactive = matches!(status, "no_gradient" | "zero_gradient");
        GradientModuleMetrics {
            module_name: module_name.to_string(),
            expected_active,
            trainable_tensor_count: self.trainable_tensor_count,
            gradient_tensor_count: self.gradient_tensor_count,
            nonfinite_gradient_tensors: self.nonfinite_gradient_tensors,
            grad_l2_norm,
            grad_abs_max: self.grad_abs_max,
            status: status.to_string(),
            inactive_expected: inactive && !expected_active,
            inactive_unexpected: inactive && expected_active,
        }
    }
}

impl GradientHealthMetrics {
    fn with_pre_clip_norm(mut self, pre_clip_global_grad_l2_norm: f64) -> Self {
        self.pre_clip_global_grad_l2_norm = pre_clip_global_grad_l2_norm;
        self
    }
}

fn collect_gradient_health(
    var_store: &nn::VarStore,
    optimizer_step_skipped: bool,
    clip_global_norm: Option<f64>,
    nonfinite_loss_terms: Vec<String>,
    expected_activity: &BTreeMap<&'static str, bool>,
) -> GradientHealthMetrics {
    let module_order = [
        "topology_encoder",
        "geometry_encoder",
        "pocket_encoder",
        "slots",
        "gates",
        "cross_modal_interaction",
        "decoder",
        "flow_head",
        "semantic_probes",
        "other",
    ];
    let mut accumulators: BTreeMap<&'static str, GradientAccumulator> = module_order
        .iter()
        .copied()
        .map(|name| (name, GradientAccumulator::default()))
        .collect();

    for (name, tensor) in var_store.variables() {
        let module_name = gradient_module_for_variable(&name);
        accumulators
            .entry(module_name)
            .or_default()
            .observe_variable(&tensor);
    }

    let modules = module_order
        .iter()
        .filter_map(|module_name| {
            accumulators.remove(module_name).map(|accumulator| {
                let expected_active = expected_activity.get(module_name).copied().unwrap_or(true);
                accumulator.finish(module_name, expected_active)
            })
        })
        .collect::<Vec<_>>();
    let global_sq_sum = modules
        .iter()
        .map(|module| module.grad_l2_norm * module.grad_l2_norm)
        .sum::<f64>();
    let global_grad_l2_norm = global_sq_sum.max(0.0).sqrt();
    let global_grad_abs_max = modules
        .iter()
        .map(|module| module.grad_abs_max)
        .fold(0.0, f64::max);
    let nonfinite_gradient_tensors = modules
        .iter()
        .map(|module| module.nonfinite_gradient_tensors)
        .sum();

    GradientHealthMetrics {
        modules,
        pre_clip_global_grad_l2_norm: global_grad_l2_norm,
        global_grad_l2_norm,
        global_grad_abs_max,
        nonfinite_gradient_tensors,
        clipping_enabled: clip_global_norm.is_some(),
        clip_global_norm,
        clipped: false,
        optimizer_step_skipped,
        nonfinite_loss_terms,
        objective_families: ObjectiveGradientDiagnostics::default(),
    }
}

fn expected_gradient_activity(
    config: &ResearchConfig,
    weights: &EffectiveLossWeights,
) -> BTreeMap<&'static str, bool> {
    let focus = config.model.modality_focus;
    let active_modality_count = [
        focus.keep_topology(),
        focus.keep_geometry(),
        focus.keep_pocket(),
    ]
    .into_iter()
    .filter(|active| *active)
    .count();
    let primary = config.training.primary_objective;
    let flow_primary = matches!(
        primary,
        PrimaryObjectiveConfig::FlowMatching | PrimaryObjectiveConfig::DenoisingFlowMatching
    );
    let decoder_primary = matches!(
        primary,
        PrimaryObjectiveConfig::SurrogateReconstruction
            | PrimaryObjectiveConfig::ConditionedDenoising
            | PrimaryObjectiveConfig::DenoisingFlowMatching
    );
    let redundancy_or_slot_active =
        weights.intra_red > 0.0 || weights.slot > 0.0 || weights.consistency > 0.0;
    let probe_active = weights.probe > 0.0
        || weights.pharmacophore_probe > 0.0
        || weights.leak > 0.0
        || weights.pharmacophore_leakage > 0.0;
    let generation_guardrail_active = weights.pocket_contact > 0.0
        || weights.pocket_clash > 0.0
        || weights.pocket_envelope > 0.0
        || weights.valence_guardrail > 0.0
        || weights.bond_length_guardrail > 0.0;
    let interaction_can_learn = active_modality_count >= 2
        && config.model.interaction_mode != CrossAttentionMode::DirectFusionNegativeControl;
    let decoder_expected = decoder_primary || generation_guardrail_active;
    let slots_expected = active_modality_count > 0
        && (decoder_expected || flow_primary || redundancy_or_slot_active);
    let cross_interaction_expected =
        active_modality_count >= 2 && (decoder_expected || flow_primary || weights.gate > 0.0);
    let gates_expected = interaction_can_learn && cross_interaction_expected;
    let flow_backend_expected = config.generation_method.resolved_primary_backend_family()
        == GenerationBackendFamilyConfig::FlowMatching;
    let flow_head_expected = flow_primary || flow_backend_expected;

    BTreeMap::from([
        (
            "topology_encoder",
            modality_expected(focus, PrimaryObjectiveConfig::FlowMatching, primary, true)
                || (focus.keep_topology() && (redundancy_or_slot_active || probe_active)),
        ),
        (
            "geometry_encoder",
            focus.keep_geometry()
                && (decoder_primary
                    || flow_primary
                    || redundancy_or_slot_active
                    || probe_active
                    || generation_guardrail_active),
        ),
        (
            "pocket_encoder",
            modality_expected(focus, PrimaryObjectiveConfig::FlowMatching, primary, false)
                || (focus.keep_pocket()
                    && (redundancy_or_slot_active || probe_active || generation_guardrail_active)),
        ),
        ("slots", slots_expected),
        ("gates", gates_expected),
        ("cross_modal_interaction", cross_interaction_expected),
        ("decoder", decoder_expected),
        ("flow_head", flow_head_expected),
        ("semantic_probes", probe_active),
        ("other", true),
    ])
}

fn modality_expected(
    focus: ModalityFocusConfig,
    flow_only_objective: PrimaryObjectiveConfig,
    primary: PrimaryObjectiveConfig,
    topology: bool,
) -> bool {
    let kept = if topology {
        focus.keep_topology()
    } else {
        focus.keep_pocket()
    };
    kept && primary != flow_only_objective
}

fn gradient_module_for_variable(name: &str) -> &'static str {
    if name.starts_with("topology.") || name.starts_with("topology/") {
        "topology_encoder"
    } else if name.starts_with("geometry.") || name.starts_with("geometry/") {
        "geometry_encoder"
    } else if name.starts_with("pocket.") || name.starts_with("pocket/") {
        "pocket_encoder"
    } else if name.starts_with("slot_") {
        "slots"
    } else if name.starts_with("ligand_decoder.") || name.starts_with("ligand_decoder/") {
        "decoder"
    } else if name.starts_with("flow_matching_head.") || name.starts_with("flow_matching_head/") {
        "flow_head"
    } else if name.starts_with("probes.") || name.starts_with("probes/") {
        "semantic_probes"
    } else if name.contains("gate") {
        "gates"
    } else if name.starts_with("topo_from_")
        || name.starts_with("geo_from_")
        || name.starts_with("pocket_from_")
    {
        "cross_modal_interaction"
    } else {
        "other"
    }
}

fn nonfinite_loss_terms(
    primary_objective_name: &str,
    primary_value: f64,
    auxiliaries: &crate::training::AuxiliaryLossMetrics,
    total_value: f64,
) -> Vec<String> {
    let mut terms = Vec::new();
    if !primary_value.is_finite() {
        terms.push(format!("primary:{primary_objective_name}"));
    }
    for (name, value) in [
        ("auxiliary:intra_red", auxiliaries.intra_red),
        ("auxiliary:probe", auxiliaries.probe),
        (
            "auxiliary:probe_ligand_pharmacophore",
            auxiliaries.probe_ligand_pharmacophore,
        ),
        (
            "auxiliary:probe_pocket_pharmacophore",
            auxiliaries.probe_pocket_pharmacophore,
        ),
        ("auxiliary:leak", auxiliaries.leak),
        ("auxiliary:leak_core", auxiliaries.leak_core),
        (
            "auxiliary:leak_similarity_proxy_diagnostic",
            auxiliaries.leak_similarity_proxy_diagnostic,
        ),
        (
            "auxiliary:leak_explicit_probe_diagnostic",
            auxiliaries.leak_explicit_probe_diagnostic,
        ),
        (
            "auxiliary:leak_probe_fit_loss",
            auxiliaries.leak_probe_fit_loss,
        ),
        (
            "auxiliary:leak_encoder_penalty",
            auxiliaries.leak_encoder_penalty,
        ),
        (
            "auxiliary:leak_topology_to_geometry",
            auxiliaries.leak_topology_to_geometry,
        ),
        (
            "auxiliary:leak_geometry_to_topology",
            auxiliaries.leak_geometry_to_topology,
        ),
        (
            "auxiliary:leak_pocket_to_geometry",
            auxiliaries.leak_pocket_to_geometry,
        ),
        (
            "auxiliary:leak_topology_to_pocket_role",
            auxiliaries.leak_topology_to_pocket_role,
        ),
        (
            "auxiliary:leak_geometry_to_pocket_role",
            auxiliaries.leak_geometry_to_pocket_role,
        ),
        (
            "auxiliary:leak_pocket_to_ligand_role",
            auxiliaries.leak_pocket_to_ligand_role,
        ),
        ("auxiliary:gate", auxiliaries.gate),
        ("auxiliary:slot", auxiliaries.slot),
        ("auxiliary:consistency", auxiliaries.consistency),
        ("auxiliary:pocket_contact", auxiliaries.pocket_contact),
        ("auxiliary:pocket_clash", auxiliaries.pocket_clash),
        ("auxiliary:pocket_envelope", auxiliaries.pocket_envelope),
        ("auxiliary:valence_guardrail", auxiliaries.valence_guardrail),
        (
            "auxiliary:bond_length_guardrail",
            auxiliaries.bond_length_guardrail,
        ),
    ] {
        if !value.is_finite() {
            terms.push(name.to_string());
        }
    }
    if !total_value.is_finite() {
        terms.push("total".to_string());
    }
    terms
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
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct CountingExampleSource {
        examples: Vec<MolecularExample>,
        get_example_calls: Arc<AtomicUsize>,
        materialize_examples: bool,
    }

    impl CountingExampleSource {
        fn new(examples: Vec<MolecularExample>) -> (Self, Arc<AtomicUsize>) {
            let get_example_calls = Arc::new(AtomicUsize::new(0));
            let source = Self {
                examples,
                get_example_calls: get_example_calls.clone(),
                materialize_examples: false,
            };
            (source, get_example_calls)
        }

        fn calls(&self) -> usize {
            self.get_example_calls.load(Ordering::SeqCst)
        }
    }

    impl MolecularExampleSource for CountingExampleSource {
        type Error = std::convert::Infallible;

        fn len(&self) -> usize {
            self.examples.len()
        }

        fn get_example(&self, index: usize) -> Result<Option<MolecularExample>, Self::Error> {
            self.get_example_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.examples.get(index).cloned())
        }

        fn materialized_examples(&self) -> Option<&[MolecularExample]> {
            if self.materialize_examples {
                Some(&self.examples)
            } else {
                None
            }
        }
    }

    #[test]
    fn trainer_constructor_validates_cross_section_config() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline;

        let var_store = nn::VarStore::new(Device::Cpu);
        let error = match ResearchTrainer::new(&var_store, config) {
            Ok(_) => panic!("invalid config should be rejected by ResearchTrainer::new"),
            Err(error) => error.to_string(),
        };

        assert!(error.contains("training.primary_objective=conditioned_denoising"));
        assert!(error.contains("pocket_only_initialization_baseline"));
        assert!(error.contains("surrogate_reconstruction"));
    }

    #[test]
    fn test_only_unvalidated_constructor_is_explicit_escape_hatch() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline;

        let var_store = nn::VarStore::new(Device::Cpu);
        let trainer = ResearchTrainer::new_unvalidated_for_tests(&var_store, config)
            .expect("test-only escape hatch should bypass config validation");

        assert_eq!(trainer.primary_objective.name(), "conditioned_denoising");
    }

    #[test]
    #[should_panic(
        expected = "flow_matching primary objective requires forward.generation.flow_matching"
    )]
    fn flow_matching_objective_fails_loudly_without_flow_record() {
        let mut config = ResearchConfig::default();
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = dataset.examples();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, mut forwards) = system.forward_batch(&examples[..1]);
        forwards[0].generation.flow_matching = None;
        let objective = build_primary_objective(&config.training);

        let _ = compute_primary_objective_batch_with_components(
            objective.as_ref(),
            &examples[..1],
            &forwards,
        );
    }

    fn expected_sample_batches_for_steps(
        examples: &[MolecularExample],
        batch_size: usize,
        shuffle: bool,
        sampler_seed: u64,
        drop_last: bool,
        max_steps: usize,
    ) -> Vec<Vec<usize>> {
        let mut batches = Vec::new();
        for epoch_index in 0usize.. {
            let epoch_batches = ExampleBatchSampler::new(
                examples,
                batch_size,
                shuffle,
                sampler_seed,
                drop_last,
                epoch_index,
            )
            .map(|batch| batch.sample_indices)
            .collect::<Vec<_>>();
            if epoch_batches.is_empty() {
                break;
            }
            for sample_indices in epoch_batches {
                batches.push(sample_indices);
                if batches.len() >= max_steps {
                    return batches;
                }
            }
        }
        batches
    }

    fn gradient_module<'a>(
        health: &'a GradientHealthMetrics,
        module_name: &str,
    ) -> &'a GradientModuleMetrics {
        health
            .modules
            .iter()
            .find(|module| module.module_name == module_name)
            .unwrap_or_else(|| panic!("missing gradient module {module_name}"))
    }

    fn valid_short_schedule_config(mut config: ResearchConfig) -> ResearchConfig {
        if config.training.schedule.stage3_steps > config.training.max_steps {
            config.training.schedule.stage1_steps = config.training.max_steps;
            config.training.schedule.stage2_steps = config.training.max_steps;
            config.training.schedule.stage3_steps = config.training.max_steps;
        }
        config
    }

    fn new_valid_test_trainer(var_store: &nn::VarStore, config: ResearchConfig) -> ResearchTrainer {
        ResearchTrainer::new(var_store, valid_short_schedule_config(config)).unwrap()
    }

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
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();
        assert_eq!(metrics.len(), 3);
        assert_eq!(trainer.step(), 3);
        assert_eq!(metrics[0].epoch_index, 0);
        assert_eq!(metrics[0].sample_order_seed, 0);
        assert_eq!(metrics[0].batch_sample_indices, vec![0, 1]);
        assert_eq!(metrics[1].epoch_index, 0);
        assert_eq!(metrics[1].batch_sample_indices, vec![2, 3]);
        assert_eq!(metrics[2].epoch_index, 1);
        assert_eq!(metrics[2].batch_sample_indices, vec![0, 1]);
    }

    #[test]
    fn fit_source_matches_sampler_indices_and_avoids_materialized_shortcut() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 3;
        config.training.data_order.shuffle = true;
        config.training.data_order.sampler_seed = 17;
        config.training.data_order.drop_last = false;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = dataset.examples().to_vec();
        let (source, _calls) = CountingExampleSource::new(examples.clone());

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config.clone());

        let metrics = trainer.fit_source(&var_store, &system, &source).unwrap();

        let expected = expected_sample_batches_for_steps(
            &examples,
            config.data.batch_size,
            config.training.data_order.shuffle,
            config.training.data_order.sampler_seed,
            config.training.data_order.drop_last,
            config.training.max_steps,
        );
        assert_eq!(metrics.len(), expected.len());
        for (metric, sample_indices) in metrics.iter().zip(expected.iter()) {
            assert_eq!(metric.batch_sample_indices, *sample_indices);
            assert!(!metric.batch_sample_indices.is_empty());
            assert!(metric.losses.total.is_finite());
        }
        let expected_calls: usize = expected.iter().map(|indices| indices.len()).sum();
        assert_eq!(source.calls(), expected_calls);
    }

    #[test]
    fn trainer_reproducibility_records_epoch_seed_and_sample_indices() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 3;
        config.training.data_order.shuffle = true;
        config.training.data_order.sampler_seed = 31;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        let epoch0_seed = crate::data::sample_order_seed_for_epoch(31, 0);
        let epoch1_seed = crate::data::sample_order_seed_for_epoch(31, 1);
        let epoch0_batches = ExampleBatchSampler::new(dataset.examples(), 2, true, 31, false, 0)
            .map(|batch| batch.sample_indices)
            .collect::<Vec<_>>();
        let epoch1_batches = ExampleBatchSampler::new(dataset.examples(), 2, true, 31, false, 1)
            .map(|batch| batch.sample_indices)
            .collect::<Vec<_>>();
        assert_eq!(metrics[0].epoch_index, 0);
        assert_eq!(metrics[0].sample_order_seed, epoch0_seed);
        assert_eq!(metrics[0].batch_sample_indices, epoch0_batches[0]);
        assert_eq!(metrics[1].epoch_index, 0);
        assert_eq!(metrics[1].sample_order_seed, epoch0_seed);
        assert_eq!(metrics[1].batch_sample_indices, epoch0_batches[1]);
        assert_eq!(metrics[2].epoch_index, 1);
        assert_eq!(metrics[2].sample_order_seed, epoch1_seed);
        assert_eq!(metrics[2].batch_sample_indices, epoch1_batches[0]);
    }

    #[test]
    fn fit_data_order_drop_last_and_max_epochs_stop_explicitly() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 3;
        config.training.max_steps = 4;
        config.training.data_order.drop_last = true;
        config.training.data_order.max_epochs = Some(1);
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].batch_sample_indices, vec![0, 1, 2]);
        assert_eq!(trainer.step(), 1);
    }

    #[test]
    fn train_batch_metrics_include_stage_aware_interaction_paths() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        assert_eq!(metrics.interaction.stage, Some(TrainingStage::Stage1));
        assert_eq!(
            metrics.stage_progress.stage_index,
            TrainingStage::Stage1.index()
        );
        assert_eq!(
            metrics.stage_progress.fixed_stage_index,
            TrainingStage::Stage1.index()
        );
        assert!(metrics.stage_progress.stage_ramp.is_finite());
        assert_eq!(metrics.stage_progress.readiness_status, "disabled");
        assert!(!metrics.stage_progress.adaptive_stage_enabled);
        assert!(!metrics.stage_progress.adaptive_stage_hold);
        assert_eq!(
            metrics
                .stage_progress
                .objective_execution_counts
                .primary_enabled_count,
            1
        );
        assert!(metrics.runtime_profile.step_time_ms.is_finite());
        assert!(metrics.runtime_profile.examples_per_second.is_finite());
        assert_eq!(metrics.runtime_profile.batch_size, 2);
        assert_eq!(metrics.runtime_profile.forward_batch_count, 1);
        assert_eq!(metrics.runtime_profile.per_example_forward_count, 0);
        assert_eq!(
            metrics.runtime_profile.forward_execution_mode,
            "batched_interaction_context"
        );
        assert!(metrics.runtime_profile.rollout_diagnostics_built);
        assert_eq!(
            metrics.runtime_profile.rollout_diagnostic_execution_count,
            2
        );
        assert!(metrics.runtime_profile.rollout_diagnostics_no_grad);
        assert_eq!(
            metrics
                .runtime_profile
                .objective_execution_counts
                .primary_enabled_count,
            metrics
                .stage_progress
                .objective_execution_counts
                .primary_enabled_count
        );
        assert!(metrics
            .stage_progress
            .active_objective_families
            .contains(&"primary".to_string()));
        assert_eq!(
            metrics.losses.primary.weighted_value,
            metrics.losses.primary.primary_value * metrics.losses.primary.effective_weight
        );
        let slot_report = metrics
            .losses
            .auxiliaries
            .auxiliary_objective_report
            .entries
            .iter()
            .find(|entry| entry.family.as_str() == "slot")
            .expect("slot report entry should be present");
        assert!(!slot_report.enabled);
        assert_eq!(slot_report.unweighted_value, 0.0);
        assert_eq!(slot_report.execution_mode, "skipped_zero_weight");
        let consistency_report = metrics
            .losses
            .auxiliaries
            .auxiliary_objective_report
            .entries
            .iter()
            .find(|entry| entry.family.as_str() == "consistency")
            .expect("consistency report entry should be present");
        assert!(consistency_report.enabled);
        assert_eq!(consistency_report.execution_mode, "trainable");
        assert_eq!(
            metrics.interaction.stage_index,
            Some(TrainingStage::Stage1.index())
        );
        assert_eq!(metrics.interaction.paths.len(), 6);
        assert!(metrics.interaction.mean_gate.is_finite());
        assert!(metrics.interaction.mean_gate_sparsity.is_finite());
        assert!(metrics.interaction.mean_attention_entropy.is_finite());
        assert!(metrics.slot_utilization.mean_active_slot_count.is_finite());
        assert!(metrics
            .slot_utilization
            .mean_active_slot_fraction
            .is_finite());
        assert!(metrics.slot_utilization.mean_slot_entropy.is_finite());
        assert!(metrics.slot_utilization.dead_slot_fraction.is_finite());
        assert!(metrics.slot_utilization.dead_slot_fraction >= 0.0);
        let bucketed_slots = metrics.slot_utilization.dead_slot_count
            + metrics.slot_utilization.diffuse_slot_count
            + metrics.slot_utilization.saturated_slot_count;
        let observed_slot_observations = metrics
            .slot_utilization
            .slot_signatures
            .iter()
            .map(|signature| signature.slot_count * signature.sample_count)
            .sum::<usize>();
        assert!(bucketed_slots <= observed_slot_observations);
        assert_eq!(
            metrics.slot_utilization.stage_index,
            Some(TrainingStage::Stage1.index())
        );
        let signature_modalities = metrics
            .slot_utilization
            .slot_signatures
            .iter()
            .map(|signature| signature.modality.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            signature_modalities,
            ["geometry", "pocket", "topology"].into_iter().collect()
        );
        for signature in &metrics.slot_utilization.slot_signatures {
            assert_eq!(signature.stage_index, Some(TrainingStage::Stage1.index()));
            assert!(signature.sample_count > 0);
            assert!(signature.slot_count > 0);
            assert!(signature.assignment_entropy.is_finite());
            assert!(signature.semantic_probe_alignment.is_finite());
            assert!(signature.matching_score.is_finite());
            assert_eq!(
                signature.matching_scope,
                "within_step_repeated_signature_proxy"
            );
        }
        for path in &metrics.interaction.paths {
            assert!(!path.path_name.is_empty());
            assert!(!path.path_role.is_empty());
            assert_eq!(
                path.training_stage_index,
                Some(TrainingStage::Stage1.index())
            );
            assert!(path.gate_mean.is_finite());
            assert!(path.gate_abs_mean.is_finite());
            assert!((0.0..=1.0).contains(&path.gate_sparsity));
            assert!((0.0..=1.0).contains(&path.gate_closed_fraction));
            assert!((0.0..=1.0).contains(&path.gate_open_fraction));
            assert!((0.0..=1.0).contains(&path.gate_saturation_fraction));
            assert!(path.gate_element_count > 0);
            assert!(path.gate_entropy.is_finite());
            assert!(path.gate_gradient_proxy.is_finite());
            assert!(path.path_scale.is_finite());
            assert!(!path.gate_status.is_empty());
            assert!(path.attention_entropy.is_finite());
            assert!(path.effective_update_norm.is_finite());
        }
    }

    #[test]
    fn adaptive_stage_guard_can_hold_without_prior_readiness() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 3;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.schedule.stage1_steps = 0;
        config.training.schedule.stage2_steps = 1;
        config.training.schedule.stage3_steps = 2;
        config.training.adaptive_stage_guard.enabled = true;
        config.training.adaptive_stage_guard.hold_stages = true;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        assert_eq!(metrics.stage, TrainingStage::Stage1);
        assert_eq!(
            metrics.stage_progress.fixed_stage_index,
            TrainingStage::Stage2.index()
        );
        assert!(metrics.stage_progress.adaptive_stage_enabled);
        assert!(metrics.stage_progress.adaptive_stage_hold);
        assert_eq!(metrics.stage_progress.readiness_status, "held");
        assert!(metrics
            .stage_progress
            .readiness_reasons
            .iter()
            .any(|reason| reason.contains("no prior step metrics")));
    }

    #[test]
    fn adaptive_stage_guard_advances_when_recent_metrics_are_ready() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 2;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.schedule.stage1_steps = 1;
        config.training.schedule.stage2_steps = 2;
        config.training.schedule.stage3_steps = 2;
        config.training.adaptive_stage_guard.enabled = true;
        config.training.adaptive_stage_guard.hold_stages = true;
        config
            .training
            .adaptive_stage_guard
            .min_finite_step_fraction = 0.0;
        config
            .training
            .adaptive_stage_guard
            .require_no_optimizer_skip = false;
        config
            .training
            .adaptive_stage_guard
            .max_slot_collapse_warnings = usize::MAX;
        config
            .training
            .adaptive_stage_guard
            .max_gate_saturation_fraction = 1.0;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].stage, TrainingStage::Stage1);
        assert_eq!(metrics[1].stage, TrainingStage::Stage2);
        assert!(!metrics[1].stage_progress.adaptive_stage_hold);
        assert_eq!(metrics[1].stage_progress.readiness_status, "ready");
        assert!(metrics[1]
            .stage_progress
            .readiness_reasons
            .iter()
            .any(|reason| reason.contains("satisfy adaptive stage readiness")));
    }

    #[test]
    fn adaptive_stage_guard_holds_on_slot_collapse_warnings() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 2;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.schedule.stage1_steps = 1;
        config.training.schedule.stage2_steps = 2;
        config.training.schedule.stage3_steps = 2;
        config.training.adaptive_stage_guard.enabled = true;
        config.training.adaptive_stage_guard.hold_stages = true;
        config
            .training
            .adaptive_stage_guard
            .min_finite_step_fraction = 0.0;
        config
            .training
            .adaptive_stage_guard
            .require_no_optimizer_skip = false;
        config
            .training
            .adaptive_stage_guard
            .max_slot_collapse_warnings = 0;
        config
            .training
            .adaptive_stage_guard
            .max_gate_saturation_fraction = 1.0;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let mut prior = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();
        prior.slot_utilization.collapse_warning_count = 1;
        trainer.replace_history(vec![prior]);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        assert_eq!(metrics.stage, TrainingStage::Stage1);
        assert_eq!(
            metrics.stage_progress.fixed_stage_index,
            TrainingStage::Stage2.index()
        );
        assert!(metrics.stage_progress.adaptive_stage_hold);
        assert!(metrics
            .stage_progress
            .readiness_reasons
            .iter()
            .any(|reason| reason.contains("slot_collapse_warning_count=1")));
    }

    #[test]
    fn adaptive_stage_guard_holds_on_slot_signature_gate_and_leakage_readiness() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 2;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.schedule.stage1_steps = 1;
        config.training.schedule.stage2_steps = 2;
        config.training.schedule.stage3_steps = 2;
        config.training.adaptive_stage_guard.enabled = true;
        config.training.adaptive_stage_guard.hold_stages = true;
        config
            .training
            .adaptive_stage_guard
            .min_finite_step_fraction = 0.0;
        config
            .training
            .adaptive_stage_guard
            .require_no_optimizer_skip = false;
        config
            .training
            .adaptive_stage_guard
            .max_slot_collapse_warnings = usize::MAX;
        config
            .training
            .adaptive_stage_guard
            .max_gate_saturation_fraction = 0.10;
        config
            .training
            .adaptive_stage_guard
            .min_slot_signature_matching_score = Some(0.75);
        config.training.adaptive_stage_guard.max_leakage_diagnostic = Some(0.10);

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let mut prior = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();
        if let Some(signature) = prior.slot_utilization.slot_signatures.first_mut() {
            signature.matching_score = 0.10;
        }
        if let Some(path) = prior.interaction.paths.first_mut() {
            path.gate_saturation_fraction = 1.0;
        }
        prior.losses.auxiliaries.leak_similarity_proxy_diagnostic = 0.25;
        trainer.replace_history(vec![prior]);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        assert_eq!(metrics.stage, TrainingStage::Stage1);
        assert_eq!(
            metrics.stage_progress.fixed_stage_index,
            TrainingStage::Stage2.index()
        );
        assert!(metrics.stage_progress.adaptive_stage_hold);
        assert!(metrics
            .stage_progress
            .readiness_reasons
            .iter()
            .any(|reason| reason.contains("slot_signature_matching_score")));
        assert!(metrics
            .stage_progress
            .readiness_reasons
            .iter()
            .any(|reason| reason.contains("mean_gate_saturation_fraction")));
        assert!(metrics
            .stage_progress
            .readiness_reasons
            .iter()
            .any(|reason| reason.contains("leakage_diagnostic=0.250")));
    }

    #[test]
    fn train_batch_metrics_report_weighted_gate_path_contributions() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.schedule.stage1_steps = 0;
        config.training.schedule.stage2_steps = 0;
        config.training.schedule.stage3_steps = 0;
        config.training.loss_weights.eta_gate = 1.0;
        config
            .model
            .interaction_tuning
            .gate_regularization_path_weights =
            vec![crate::config::InteractionPathGateRegularizationWeight {
                path: "geo_from_pocket".to_string(),
                weight: 0.0,
            }];

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        let contributions = &metrics.losses.auxiliaries.gate_path_contributions;
        assert_eq!(metrics.interaction.stage, Some(TrainingStage::Stage4));
        assert_eq!(contributions.len(), 6);
        let contribution_sum = contributions
            .iter()
            .map(|contribution| contribution.objective_contribution)
            .sum::<f64>();
        assert!((contribution_sum - metrics.losses.auxiliaries.gate).abs() < 1e-6);
        let geo_from_pocket = contributions
            .iter()
            .find(|contribution| contribution.path_name == "geo_from_pocket")
            .expect("geo_from_pocket gate contribution should be reported");
        assert_eq!(geo_from_pocket.path_weight, 0.0);
        assert_eq!(geo_from_pocket.objective_contribution, 0.0);
        for contribution in contributions {
            assert!(contribution.gate_abs_mean.is_finite());
            assert!(contribution.objective_contribution.is_finite());
            assert!(contribution.effective_loss_weight > 0.0);
            assert!(
                (contribution.optimizer_contribution
                    - contribution.objective_contribution * contribution.effective_loss_weight)
                    .abs()
                    < 1e-9
            );
        }
    }

    #[test]
    fn train_batch_metrics_include_synchronization_health() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        assert_eq!(metrics.synchronization.mask_count_mismatch, 0);
        assert_eq!(metrics.synchronization.slot_count_mismatch, 0);
        assert_eq!(metrics.synchronization.coordinate_frame_mismatch, 0);
        assert!(metrics.synchronization.batch_slice_sync_pass);
    }

    #[test]
    fn training_can_disable_rollout_diagnostics_for_optimizer_forward() {
        fn run_one_step(mut config: ResearchConfig) -> StepMetrics {
            config.data.batch_size = 2;
            config.training.max_steps = 1;
            config.training.checkpoint_every = 100;
            config.training.log_every = 100;

            let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
                .with_pocket_feature_dim(config.model.pocket_feature_dim);
            let var_store = nn::VarStore::new(Device::Cpu);
            let system = Phase1ResearchSystem::new(&var_store.root(), &config);
            let mut trainer = new_valid_test_trainer(&var_store, config);

            trainer
                .train_batch_step(&var_store, &system, &dataset.examples()[..2])
                .unwrap()
        }

        let enabled_metrics = run_one_step(ResearchConfig::default());
        let mut disabled_config = ResearchConfig::default();
        disabled_config.training.build_rollout_diagnostics = false;
        let disabled_metrics = run_one_step(disabled_config);

        assert!(disabled_metrics.losses.primary.primary_value.is_finite());
        assert!(disabled_metrics.losses.primary.weighted_value.is_finite());
        assert!(disabled_metrics.losses.total.is_finite());
        assert!(!disabled_metrics.runtime_profile.rollout_diagnostics_built);
        assert_eq!(
            disabled_metrics
                .runtime_profile
                .rollout_diagnostic_execution_count,
            0
        );
        assert!(!disabled_metrics.runtime_profile.rollout_diagnostics_no_grad);
        assert!(enabled_metrics.runtime_profile.rollout_diagnostics_built);
        assert!(
            enabled_metrics
                .runtime_profile
                .rollout_diagnostic_execution_count
                > disabled_metrics
                    .runtime_profile
                    .rollout_diagnostic_execution_count
        );
    }

    #[test]
    fn default_training_objective_is_denoising_with_rollout_eval_only_metrics() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        assert_eq!(
            metrics.losses.primary.objective_name,
            "conditioned_denoising"
        );
        assert!(metrics.losses.primary.decoder_anchored);
        assert!(metrics.losses.primary.components.rollout.is_none());
        assert!(metrics
            .losses
            .primary
            .components
            .rollout_eval_recovery
            .is_some());
        assert!(metrics
            .losses
            .primary
            .components
            .rollout_eval_pocket_anchor
            .is_some());
        let components = &metrics.losses.primary.components;
        let differentiable_component_sum = components.topology.unwrap()
            + components.geometry.unwrap()
            + components.pocket_anchor.unwrap();
        assert!(
            (metrics.losses.primary.primary_value - differentiable_component_sum).abs() <= 1e-5
        );
        let rollout_eval_records = metrics
            .losses
            .primary
            .component_provenance
            .iter()
            .filter(|record| record.component_name.starts_with("rollout_eval_"))
            .collect::<Vec<_>>();
        assert_eq!(rollout_eval_records.len(), 3);
        assert!(rollout_eval_records
            .iter()
            .any(|record| record.component_name == "rollout_eval_stop"));
        for record in rollout_eval_records {
            assert_eq!(record.role, "evaluation_only");
            assert!(!record.differentiable);
            assert!(!record.optimizer_facing);
        }
        for record in metrics
            .losses
            .primary
            .component_provenance
            .iter()
            .filter(|record| !record.component_name.starts_with("rollout_eval_"))
        {
            assert!(record.differentiable);
            assert!(record.optimizer_facing);
        }
        let scale_report = &metrics.losses.primary.component_scale_report;
        assert!(!scale_report.entries.is_empty());
        assert_eq!(scale_report.normalization_source, "unit_reference");
        for entry in &scale_report.entries {
            assert!(
                (entry.weighted_value - entry.unweighted_value * entry.objective_weight).abs()
                    < 1e-9
            );
            assert!(entry.normalized_value.is_finite());
        }
    }

    #[test]
    fn primary_component_scale_report_warns_on_large_normalized_ratio() {
        let components = crate::training::PrimaryObjectiveComponentMetrics {
            topology: Some(1.0),
            geometry: Some(50.0),
            ..Default::default()
        };

        let report = components.scale_report(2.0, 10.0, 1.0e-12, None);

        assert_eq!(report.entries.len(), 2);
        assert!(report.max_to_min_normalized_ratio >= 50.0);
        assert_eq!(report.warning_count, 1);
        let geometry = report
            .entries
            .iter()
            .find(|entry| entry.component_name == "geometry")
            .unwrap();
        assert_eq!(geometry.weighted_value, 100.0);
        assert_eq!(geometry.status, "dominant");
        assert!(geometry.warning.is_some());
    }

    #[test]
    fn rollout_eval_weight_decay_does_not_change_optimizer_primary_loss() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let example = dataset.examples()[0].clone();
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let forward = system.forward_example(&example);
        let objective = build_primary_objective(&config.training);

        let mut low_decay = example.clone();
        low_decay.decoder_supervision.rollout_eval_step_weight_decay = 0.2;
        let mut high_decay = example;
        high_decay
            .decoder_supervision
            .rollout_eval_step_weight_decay = 0.95;

        assert_ne!(
            low_decay.decoder_supervision.rollout_eval_step_weight(2),
            high_decay.decoder_supervision.rollout_eval_step_weight(2)
        );

        let (low_total, low_components) = compute_primary_objective_batch_with_components(
            objective.as_ref(),
            &[low_decay],
            &[forward.clone()],
        );
        let (high_total, high_components) = compute_primary_objective_batch_with_components(
            objective.as_ref(),
            &[high_decay],
            &[forward],
        );
        let low_primary = scalar_or_nan(&low_total);
        let high_primary = scalar_or_nan(&high_total);

        assert!((low_primary - high_primary).abs() <= 1e-8);
        assert_ne!(
            low_components.rollout_eval_recovery,
            high_components.rollout_eval_recovery
        );
    }

    #[test]
    fn primary_component_provenance_covers_all_objective_families() {
        for primary_objective in [
            crate::config::PrimaryObjectiveConfig::SurrogateReconstruction,
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising,
            crate::config::PrimaryObjectiveConfig::FlowMatching,
            crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching,
        ] {
            let mut config = ResearchConfig::default();
            config.data.batch_size = 2;
            config.training.max_steps = 1;
            config.training.checkpoint_every = 100;
            config.training.log_every = 100;
            config.training.primary_objective = primary_objective;
            if matches!(
                primary_objective,
                crate::config::PrimaryObjectiveConfig::FlowMatching
                    | crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching
            ) {
                config.generation_method.active_method = "flow_matching".to_string();
                config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
                    backend_id: "flow_matching".to_string(),
                    family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
                    trainable: true,
                    ..crate::config::GenerationBackendConfig::default()
                };
            }

            let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
                .with_pocket_feature_dim(config.model.pocket_feature_dim);
            let var_store = nn::VarStore::new(Device::Cpu);
            let system = Phase1ResearchSystem::new(&var_store.root(), &config);
            let mut trainer = new_valid_test_trainer(&var_store, config);
            let metrics = trainer
                .train_batch_step(&var_store, &system, &dataset.examples()[..2])
                .unwrap();
            let provenance = &metrics.losses.primary.component_provenance;

            assert!(
                !provenance.is_empty(),
                "{primary_objective:?} should report primary component provenance"
            );
            match primary_objective {
                crate::config::PrimaryObjectiveConfig::SurrogateReconstruction => {
                    assert!(provenance.iter().all(|record| record.optimizer_facing));
                    assert!(provenance.iter().all(|record| record.differentiable));
                }
                crate::config::PrimaryObjectiveConfig::ConditionedDenoising => {
                    assert!(provenance.iter().any(|record| {
                        record.component_name == "rollout_eval_recovery"
                            && !record.optimizer_facing
                            && !record.differentiable
                            && record.role == "evaluation_only"
                    }));
                    assert!(provenance.iter().any(|record| {
                        record.component_name == "geometry" && record.optimizer_facing
                    }));
                }
                crate::config::PrimaryObjectiveConfig::FlowMatching => {
                    assert!(provenance.iter().all(|record| record.optimizer_facing));
                    assert!(provenance.iter().any(|record| {
                        record.component_name == "flow_velocity" && record.differentiable
                    }));
                }
                crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching => {
                    assert!(provenance.iter().any(|record| {
                        record.component_name == "flow_velocity" && record.optimizer_facing
                    }));
                    assert!(provenance.iter().any(|record| {
                        record.component_name == "rollout_eval_recovery" && !record.optimizer_facing
                    }));
                }
            }
            assert!(metrics
                .losses
                .auxiliaries
                .auxiliary_objective_report
                .entries
                .iter()
                .all(|entry| (entry.enabled && entry.effective_weight > 0.0)
                    || (!entry.enabled && entry.effective_weight == 0.0)));
        }
    }

    #[test]
    fn full_flow_primary_branch_schedule_is_reported_without_auxiliary_stage_changes() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
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
            .enabled_branches = crate::config::FlowBranchKind::ALL.to_vec();
        config
            .generation_method
            .flow_matching
            .multi_modal
            .allow_zero_weight_branch_ablation = true;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .atom_type
            .start_step = 10;
        config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .bond
            .enabled = false;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        let branch_report = &metrics.losses.primary.branch_schedule;
        assert!(branch_report.observed);
        assert_eq!(branch_report.training_step, Some(0));
        assert_eq!(
            branch_report.stage_index,
            Some(TrainingStage::Stage1.index())
        );
        assert!(branch_report
            .source
            .contains("generation_method.flow_matching.multi_modal.branch_schedule"));
        let branch_weight = |name: &str| {
            branch_report
                .entries
                .iter()
                .find(|entry| entry.branch_name == name)
                .map(|entry| entry.effective_weight)
                .unwrap()
        };
        let branch = |name: &str| {
            branch_report
                .entries
                .iter()
                .find(|entry| entry.branch_name == name)
                .unwrap()
        };
        assert_eq!(branch_weight("geometry"), 1.0);
        assert_eq!(branch_weight("atom_type"), 0.0);
        assert_eq!(branch_weight("bond"), 0.0);
        assert_eq!(branch("geometry").schedule_multiplier, 1.0);
        assert_eq!(branch("atom_type").schedule_multiplier, 0.0);
        assert_eq!(branch("bond").schedule_multiplier, 0.0);
        assert!(
            (branch("geometry").weighted_value
                - branch("geometry").unweighted_value * branch("geometry").effective_weight)
                .abs()
                < 1e-6
        );
        assert_eq!(branch("atom_type").weighted_value, 0.0);
        assert_eq!(branch("bond").weighted_value, 0.0);
        assert_eq!(
            branch("geometry").target_matching_policy.as_deref(),
            Some("pad_with_mask")
        );
        assert_eq!(branch("geometry").target_matching_mean_cost, Some(0.0));
        assert_eq!(
            branch("atom_type").target_matching_policy.as_deref(),
            Some("pad_with_mask")
        );
        assert_eq!(branch("atom_type").target_matching_mean_cost, Some(0.0));
        assert_eq!(branch("atom_type").target_matching_coverage, Some(1.0));
        assert_eq!(
            branch("bond").target_matching_policy.as_deref(),
            Some("pad_with_mask")
        );
        assert_eq!(
            branch("topology").target_matching_policy.as_deref(),
            Some("pad_with_mask")
        );
        assert!(branch("pocket_context").target_matching_policy.is_none());
        assert!(metrics
            .losses
            .primary
            .component_provenance
            .iter()
            .any(|record| record.component_name == "flow_atom_type"
                && record.effective_branch_weight == Some(0.0)
                && record.branch_schedule_source.is_some()));
        let consistency = metrics
            .losses
            .auxiliaries
            .auxiliary_objective_report
            .entries
            .iter()
            .find(|entry| entry.family.as_str() == "consistency")
            .unwrap();
        assert!(consistency.enabled);
        assert_eq!(consistency.execution_mode, "trainable");
    }

    #[test]
    fn full_flow_branch_scale_report_covers_all_optimizer_branches() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
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
            .enabled_branches = crate::config::FlowBranchKind::ALL.to_vec();

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        let branch_report = &metrics.losses.primary.branch_schedule;
        let branch_names = branch_report
            .entries
            .iter()
            .map(|entry| entry.branch_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            branch_names,
            vec![
                "geometry",
                "atom_type",
                "bond",
                "topology",
                "pocket_context",
                "synchronization"
            ]
        );
        for entry in &branch_report.entries {
            assert!(entry.optimizer_facing, "{} inactive", entry.branch_name);
            assert!(
                entry.effective_weight > 0.0,
                "{} zero weight",
                entry.branch_name
            );
            assert!(
                entry.schedule_multiplier > 0.0,
                "{} zero schedule multiplier",
                entry.branch_name
            );
            assert!(
                entry.unweighted_value.is_finite() && entry.unweighted_value >= 0.0,
                "{} invalid unweighted value",
                entry.branch_name
            );
            assert!(
                entry.weighted_value.is_finite() && entry.weighted_value > 0.0,
                "{} did not contribute a nonzero weighted value",
                entry.branch_name
            );
            assert!(
                (entry.weighted_value - entry.unweighted_value * entry.effective_weight).abs()
                    < 1e-6,
                "{} weighted/unweighted scale mismatch",
                entry.branch_name
            );
        }
        assert!(!metrics.gradient_health.optimizer_step_skipped);
    }

    #[test]
    fn q14_full_flow_schedule_doc_and_preset_define_staged_activation() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let preset_path = root.join("configs/q14_full_flow_staged_schedule.json");
        let preset: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(preset_path).unwrap()).unwrap();
        let schedule = &preset["branch_schedule"];

        assert_eq!(schedule["geometry"]["start_step"], 0);
        assert!(
            schedule["atom_type"]["start_step"].as_u64().unwrap()
                > schedule["geometry"]["start_step"].as_u64().unwrap()
        );
        assert!(
            schedule["bond"]["start_step"].as_u64().unwrap()
                > schedule["atom_type"]["start_step"].as_u64().unwrap()
        );
        assert!(
            schedule["topology"]["start_step"].as_u64().unwrap()
                > schedule["bond"]["start_step"].as_u64().unwrap()
        );
        assert!(
            schedule["pocket_context"]["start_step"].as_u64().unwrap()
                > schedule["topology"]["start_step"].as_u64().unwrap()
        );
        assert!(
            schedule["synchronization"]["start_step"].as_u64().unwrap()
                > schedule["pocket_context"]["start_step"].as_u64().unwrap()
        );
        for required in [
            "unweighted_value",
            "effective_weight",
            "weighted_value",
            "schedule_multiplier",
        ] {
            assert!(preset["required_branch_report_fields"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field == required));
        }

        let doc = std::fs::read_to_string(root.join("docs/q14_full_flow_schedule.md")).unwrap();
        assert!(doc.contains("Branch Scale Report"));
        assert!(doc.contains("Claim-bearing full-flow configs"));
        assert!(doc.contains("configs/q14_full_flow_staged_schedule.json"));
    }

    #[test]
    fn objective_gradient_diagnostics_are_sparse_and_include_family_provenance() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 2;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.objective_gradient_diagnostics.enabled = true;
        config
            .training
            .objective_gradient_diagnostics
            .sample_every_steps = 2;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        assert!(metrics[0].gradient_health.objective_families.enabled);
        assert!(metrics[0].gradient_health.objective_families.sampled);
        assert_eq!(
            metrics[0].gradient_health.objective_families.sampling_mode,
            "exact_sampled_retained_graph"
        );
        assert!(metrics[0]
            .gradient_health
            .objective_families
            .entries
            .iter()
            .any(|entry| entry.family_name.starts_with("primary:")
                && entry.provenance == "primary_objective:exact_autograd"
                && entry.status == "exact_sampled"
                && entry.grad_l2_norm.is_finite()
                && entry.grad_l2_norm > 0.0));
        assert!(metrics[1].gradient_health.objective_families.enabled);
        assert!(!metrics[1].gradient_health.objective_families.sampled);
        assert!(metrics[1]
            .gradient_health
            .objective_families
            .entries
            .is_empty());
    }

    #[test]
    fn objective_gradient_diagnostics_can_use_loss_share_proxy_fallback() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.objective_gradient_diagnostics.enabled = true;
        config
            .training
            .objective_gradient_diagnostics
            .sample_every_steps = 1;
        config.training.objective_gradient_diagnostics.sampling_mode =
            ObjectiveGradientSamplingMode::LossShareProxy;
        config
            .training
            .objective_gradient_diagnostics
            .include_auxiliary = false;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();
        let diagnostics = &metrics.gradient_health.objective_families;

        assert!(diagnostics.enabled);
        assert!(diagnostics.sampled);
        assert_eq!(
            diagnostics.sampling_mode,
            "weighted_loss_share_post_backward_proxy"
        );
        assert_eq!(diagnostics.entries.len(), 1);
        let primary = &diagnostics.entries[0];
        assert!(primary.family_name.starts_with("primary:"));
        assert_eq!(primary.status, "loss_share_proxy");
        assert_eq!(primary.provenance, "primary_objective:loss_share_proxy");
        assert!(primary.grad_l2_norm.is_finite());
    }

    #[test]
    fn objective_gradient_diagnostics_respect_included_family_allow_list() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.objective_gradient_diagnostics.enabled = true;
        config
            .training
            .objective_gradient_diagnostics
            .sample_every_steps = 1;
        config
            .training
            .objective_gradient_diagnostics
            .included_families = vec!["primary".to_string()];

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();
        let diagnostics = &metrics.gradient_health.objective_families;

        assert_eq!(diagnostics.sampling_mode, "exact_sampled_retained_graph");
        assert_eq!(diagnostics.entries.len(), 1);
        assert!(diagnostics.entries[0].family_name.starts_with("primary:"));
        assert_eq!(diagnostics.entries[0].status, "exact_sampled");
    }

    #[test]
    fn conditioned_denoising_primary_objective_reaches_decoder_gradients() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..2];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(examples);
        let trainer = new_valid_test_trainer(&var_store, config);

        let (primary, components) = compute_primary_objective_batch_with_components(
            trainer.primary_objective.as_ref(),
            examples,
            &forwards,
        );
        primary.backward();
        let decoder_grad_sum: f64 = var_store
            .variables()
            .into_iter()
            .filter(|(name, _)| name.contains("ligand_decoder"))
            .map(|(_, tensor)| {
                let grad = tensor.grad();
                if grad.defined() {
                    grad.abs().sum(Kind::Float).double_value(&[])
                } else {
                    0.0
                }
            })
            .sum();

        assert!(decoder_grad_sum > 0.0);
        assert!(components.rollout.is_none());
        assert!(components.rollout_eval_recovery.is_some());
    }

    #[test]
    fn synchronization_health_detects_synthetic_mismatch() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);

        let mut forward = system.forward_example(&dataset.examples()[0]);
        forward.sync_context.geometry_mask_count += 1;
        forward.sync_context.pocket_slot_count += 1;
        forward.sync_context.coordinate_frame_origin[0] = f32::NAN;

        let metrics = SynchronizationHealthMetrics::from_forwards(&[forward]);

        assert_eq!(metrics.mask_count_mismatch, 1);
        assert_eq!(metrics.slot_count_mismatch, 1);
        assert_eq!(metrics.coordinate_frame_mismatch, 1);
        assert!(!metrics.batch_slice_sync_pass);
    }

    #[test]
    fn flow_matching_metrics_include_flow_time_gate_buckets() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..2])
            .unwrap();

        assert!(metrics
            .interaction
            .paths
            .iter()
            .any(|path| path.flow_t.is_some() && path.flow_time_bucket.is_some()));
        assert!(!metrics.interaction.flow_time_buckets.is_empty());
        for bucket in &metrics.interaction.flow_time_buckets {
            assert!(matches!(bucket.bucket.as_str(), "low" | "mid" | "high"));
            assert!(!bucket.path_name.is_empty());
            assert!(bucket.sample_count > 0);
            assert!(bucket.mean_gate.is_finite());
            assert!((0.0..=1.0).contains(&bucket.mean_gate_sparsity));
        }
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
        let mut trainer = new_valid_test_trainer(&var_store, config.clone());
        trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        let mut resumed_store = nn::VarStore::new(Device::Cpu);
        let _resumed_system = Phase1ResearchSystem::new(&resumed_store.root(), &config);
        let mut resumed_trainer = new_valid_test_trainer(&resumed_store, config);
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
            checkpoint.metadata.resume_mode,
            crate::training::ResumeMode::WeightsOnlyResume
        );
        assert_eq!(
            resumed_trainer
                .restored_optimizer_state()
                .map(|state| (state.internal_state_persisted, state.resume_mode)),
            Some((false, crate::training::ResumeMode::WeightsOnlyResume))
        );
        assert_eq!(
            resumed_trainer
                .restored_optimizer_state()
                .map(|state| state.exact_resume_supported),
            Some(false)
        );
        assert_eq!(
            resumed_trainer
                .restored_scheduler_state()
                .map(|state| state.stage.as_str()),
            Some("Stage1")
        );
    }

    #[test]
    fn resume_can_require_exact_optimizer_state_and_reject_weights_only_checkpoint() {
        let temp = tempfile::tempdir().unwrap();

        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 1;
        config.training.log_every = 100;
        config.training.checkpoint_dir = temp.path().join("checkpoints");

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config.clone());
        trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        config.training.resume.require_optimizer_exact = true;
        let mut resumed_store = nn::VarStore::new(Device::Cpu);
        let _resumed_system = Phase1ResearchSystem::new(&resumed_store.root(), &config);
        let mut resumed_trainer = new_valid_test_trainer(&resumed_store, config);
        let error = resumed_trainer
            .resume_from_latest(&mut resumed_store)
            .unwrap_err()
            .to_string();

        assert!(error.contains("require_optimizer_exact"));
        assert!(error.contains("weights-only optimizer metadata"));
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
        let mut trainer = new_valid_test_trainer(&var_store, config);

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
    fn gradient_health_metrics_cover_modules_and_optional_clipping() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.gradient_clipping.global_norm = Some(0.05);
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();
        let health = &metrics[0].gradient_health;

        assert!(health.clipping_enabled);
        assert_eq!(health.clip_global_norm, Some(0.05));
        assert!(health.global_grad_l2_norm.is_finite());
        assert_eq!(health.nonfinite_gradient_tensors, 0);
        assert!(!health.optimizer_step_skipped);
        assert!(health.nonfinite_loss_terms.is_empty());

        for module_name in [
            "topology_encoder",
            "geometry_encoder",
            "pocket_encoder",
            "slots",
            "gates",
            "decoder",
            "flow_head",
            "semantic_probes",
        ] {
            let module = gradient_module(health, module_name);
            assert!(module.grad_l2_norm.is_finite());
            assert!(module.grad_abs_max.is_finite());
            assert_ne!(module.status, "nonfinite_gradient");
        }

        assert!(health
            .modules
            .iter()
            .any(|module| module.module_name == "decoder" && module.status == "active"));
        let flow_head = gradient_module(health, "flow_head");
        assert!(!flow_head.expected_active);
        assert!(flow_head.inactive_expected);
        assert!(!flow_head.inactive_unexpected);
    }

    #[test]
    fn gradient_health_marks_disabled_modalities_expected_inactive() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::SurrogateReconstruction;
        config.model.modality_focus = crate::config::ModalityFocusConfig::TopologyOnly;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();
        let health = &metrics[0].gradient_health;

        let topology = gradient_module(health, "topology_encoder");
        assert!(topology.expected_active);
        assert!(!topology.inactive_expected);

        for module_name in ["geometry_encoder", "pocket_encoder"] {
            let module = gradient_module(health, module_name);
            assert!(!module.expected_active);
            assert!(module.inactive_expected);
            assert!(!module.inactive_unexpected);
        }
    }

    #[test]
    fn modality_focus_training_keeps_expected_gradients_and_bounded_slot_scale() {
        for (focus, active_modules, inactive_modules) in [
            (
                crate::config::ModalityFocusConfig::All,
                vec!["topology_encoder", "geometry_encoder", "pocket_encoder"],
                vec![],
            ),
            (
                crate::config::ModalityFocusConfig::TopologyOnly,
                vec!["topology_encoder"],
                vec!["geometry_encoder", "pocket_encoder"],
            ),
            (
                crate::config::ModalityFocusConfig::GeometryOnly,
                vec!["geometry_encoder"],
                vec!["topology_encoder", "pocket_encoder"],
            ),
            (
                crate::config::ModalityFocusConfig::PocketOnly,
                vec!["pocket_encoder"],
                vec!["topology_encoder", "geometry_encoder"],
            ),
        ] {
            let mut config = ResearchConfig::default();
            config.data.batch_size = 2;
            config.training.max_steps = 1;
            config.training.checkpoint_every = 100;
            config.training.log_every = 100;
            config.training.primary_objective =
                crate::config::PrimaryObjectiveConfig::SurrogateReconstruction;
            config.model.modality_focus = focus;

            let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
                .with_pocket_feature_dim(config.model.pocket_feature_dim);
            let var_store = nn::VarStore::new(Device::Cpu);
            let system = Phase1ResearchSystem::new(&var_store.root(), &config);
            let mut trainer = new_valid_test_trainer(&var_store, config);

            let metrics = trainer
                .fit(&var_store, &system, dataset.examples())
                .unwrap();
            let step = &metrics[0];
            assert!(
                step.losses.auxiliaries.slot.is_finite(),
                "{focus:?} should report a finite slot objective"
            );
            assert!(
                step.losses.auxiliaries.slot <= 2.1,
                "{focus:?} slot objective should remain bounded after active-modality normalization"
            );
            assert!(step.losses.total.is_finite());
            assert_eq!(step.gradient_health.nonfinite_gradient_tensors, 0);
            let signature_modalities = step
                .slot_utilization
                .slot_signatures
                .iter()
                .map(|signature| signature.modality.as_str())
                .collect::<std::collections::BTreeSet<_>>();
            let expected_modalities = match focus {
                crate::config::ModalityFocusConfig::All => {
                    ["geometry", "pocket", "topology"].into_iter().collect()
                }
                crate::config::ModalityFocusConfig::TopologyOnly => {
                    ["topology"].into_iter().collect()
                }
                crate::config::ModalityFocusConfig::GeometryOnly => {
                    ["geometry"].into_iter().collect()
                }
                crate::config::ModalityFocusConfig::PocketOnly => ["pocket"].into_iter().collect(),
            };
            assert_eq!(
                signature_modalities, expected_modalities,
                "{focus:?} slot signatures should exclude disabled modality branches"
            );

            for module_name in active_modules {
                let module = gradient_module(&step.gradient_health, module_name);
                assert!(
                    module.expected_active,
                    "{focus:?} should expect gradients for {module_name}"
                );
                assert!(
                    !module.inactive_unexpected,
                    "{focus:?} unexpectedly lost gradients for {module_name}"
                );
            }
            for module_name in inactive_modules {
                let module = gradient_module(&step.gradient_health, module_name);
                assert!(
                    !module.expected_active,
                    "{focus:?} should not expect gradients for disabled {module_name}"
                );
                assert!(
                    module.inactive_expected,
                    "{focus:?} should mark disabled {module_name} inactive by design"
                );
            }
        }
    }

    #[test]
    fn gradient_health_marks_direct_fusion_gates_expected_inactive() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::ConditionedDenoising;
        config.model.interaction_mode =
            crate::config::CrossAttentionMode::DirectFusionNegativeControl;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();
        let gates = gradient_module(&metrics[0].gradient_health, "gates");

        assert!(!gates.expected_active);
        assert!(gates.inactive_expected);
        assert!(!gates.inactive_unexpected);
    }

    #[test]
    fn flow_matching_training_remains_finite_on_synthetic_examples() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 3;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.generation_method.active_method = "flow_matching".to_string();
        config.generation_method.primary_backend = crate::config::GenerationBackendConfig {
            backend_id: "flow_matching".to_string(),
            family: crate::config::GenerationBackendFamilyConfig::FlowMatching,
            trainable: true,
            ..crate::config::GenerationBackendConfig::default()
        };

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

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
    fn pocket_only_surrogate_training_is_shape_safe_for_atom_count_mismatch() {
        for atom_count in [1_usize, 12] {
            let mut config = ResearchConfig::default();
            config.data.batch_size = 1;
            config.training.max_steps = 1;
            config.training.checkpoint_every = 100;
            config.training.log_every = 100;
            config.data.generation_target.generation_mode =
                crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline;
            config
                .data
                .generation_target
                .pocket_only_initialization
                .atom_count = atom_count;
            config.training.primary_objective =
                crate::config::PrimaryObjectiveConfig::SurrogateReconstruction;

            let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
                .with_pocket_feature_dim(config.model.pocket_feature_dim);
            let target_atom_count = dataset.examples()[0].topology.atom_types.size()[0] as usize;
            assert_ne!(atom_count, target_atom_count);

            let var_store = nn::VarStore::new(Device::Cpu);
            let system = Phase1ResearchSystem::new(&var_store.root(), &config);
            let mut trainer = new_valid_test_trainer(&var_store, config);

            let metrics = trainer
                .train_batch_step(&var_store, &system, &dataset.examples()[..1])
                .unwrap();

            assert!(metrics.losses.total.is_finite());
            assert!(metrics.losses.primary.primary_value.is_finite());
            assert_eq!(
                metrics.generation_mode,
                crate::config::GenerationModeConfig::PocketOnlyInitializationBaseline.as_str()
            );
        }
    }

    #[test]
    fn de_novo_stage4_training_is_shape_safe_for_mismatched_atom_count() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 100;
        config.training.log_every = 100;
        config.training.schedule.stage1_steps = 0;
        config.training.schedule.stage2_steps = 0;
        config.training.schedule.stage3_steps = 0;
        config.training.loss_weights.upsilon_valence_guardrail = 0.2;
        config.training.loss_weights.phi_bond_length_guardrail = 0.2;
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
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
            .enabled_branches = crate::config::FlowBranchKind::ALL.to_vec();

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let target_atom_count = dataset.examples()[0].topology.atom_types.size()[0] as usize;
        let generated_atom_count = target_atom_count + 4;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = generated_atom_count;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = generated_atom_count;

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

        let metrics = trainer
            .train_batch_step(&var_store, &system, &dataset.examples()[..1])
            .unwrap();

        assert_eq!(metrics.stage, TrainingStage::Stage4);
        assert_eq!(
            metrics.generation_mode,
            crate::config::GenerationModeConfig::DeNovoInitialization.as_str()
        );
        assert!(metrics.losses.total.is_finite());
        assert!(metrics.losses.auxiliaries.probe.is_finite());
        assert!(metrics.losses.auxiliaries.consistency.is_finite());
        assert!(metrics.losses.auxiliaries.valence_guardrail.is_finite());
        assert!(metrics.losses.auxiliaries.bond_length_guardrail.is_finite());
        let geometry_branch = metrics
            .losses
            .primary
            .branch_schedule
            .entries
            .iter()
            .find(|entry| entry.branch_name == "geometry")
            .expect("geometry branch matching provenance");
        assert_eq!(
            geometry_branch.target_matching_policy.as_deref(),
            Some("pad_with_mask")
        );
        assert_eq!(
            geometry_branch.target_matching_matched_count,
            Some(target_atom_count)
        );
        assert_eq!(
            geometry_branch.target_matching_unmatched_generated_count,
            Some(generated_atom_count - target_atom_count)
        );
        assert_eq!(geometry_branch.target_matching_mean_cost, Some(0.0));
        assert!(geometry_branch
            .target_matching_coverage
            .is_some_and(|coverage| coverage > 0.0 && coverage < 1.0));
        assert!(!metrics.gradient_health.optimizer_step_skipped);
    }

    #[test]
    fn trainable_backend_families_have_finite_synthetic_training_smoke() {
        for (backend_id, family, primary_objective) in [
            (
                "flow_matching",
                crate::config::GenerationBackendFamilyConfig::FlowMatching,
                crate::config::PrimaryObjectiveConfig::FlowMatching,
            ),
            (
                "conditioned_denoising",
                crate::config::GenerationBackendFamilyConfig::ConditionedDenoising,
                crate::config::PrimaryObjectiveConfig::ConditionedDenoising,
            ),
        ] {
            let mut config = ResearchConfig::default();
            config.data.batch_size = 2;
            config.training.max_steps = 2;
            config.training.checkpoint_every = 100;
            config.training.log_every = 100;
            config.training.primary_objective = primary_objective;
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
            let mut trainer = new_valid_test_trainer(&var_store, config);

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
        let mut trainer = new_valid_test_trainer(&var_store, config.clone());
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
        incompatible.training.primary_objective =
            crate::config::PrimaryObjectiveConfig::FlowMatching;
        let mut resumed_store = nn::VarStore::new(Device::Cpu);
        let _system = Phase1ResearchSystem::new(&resumed_store.root(), &incompatible);
        let mut resumed = new_valid_test_trainer(&resumed_store, incompatible);

        let error = resumed
            .resume_from_latest(&mut resumed_store)
            .unwrap_err()
            .to_string();
        assert!(error.contains("checkpoint backend/objective mismatch"));
    }

    #[test]
    fn resume_rejects_incompatible_full_flow_branch_schedule() {
        let temp = tempfile::tempdir().unwrap();

        let mut config = ResearchConfig::default();
        config.data.batch_size = 2;
        config.training.max_steps = 1;
        config.training.checkpoint_every = 1;
        config.training.log_every = 100;
        config.training.primary_objective = crate::config::PrimaryObjectiveConfig::FlowMatching;
        config.training.checkpoint_dir = temp.path().join("checkpoints");
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
            .enabled_branches = crate::config::FlowBranchKind::ALL.to_vec();

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config.clone());
        trainer
            .fit(&var_store, &system, dataset.examples())
            .unwrap();

        let mut incompatible = config.clone();
        incompatible
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .bond
            .final_weight_multiplier = 0.5;
        let mut resumed_store = nn::VarStore::new(Device::Cpu);
        let _system = Phase1ResearchSystem::new(&resumed_store.root(), &incompatible);
        let mut resumed = new_valid_test_trainer(&resumed_store, incompatible);

        let error = resumed
            .resume_from_latest(&mut resumed_store)
            .unwrap_err()
            .to_string();
        assert!(error.contains("checkpoint backend/objective mismatch"));
        assert!(error.contains("replay-contract mismatch"));
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
        config.training.loss_weights.tau_pocket_envelope = 0.4;
        config.training.loss_weights.upsilon_valence_guardrail = 0.5;
        config.training.loss_weights.phi_bond_length_guardrail = 0.6;

        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let mut trainer = new_valid_test_trainer(&var_store, config);

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
            .all(|step| step.losses.auxiliaries.pocket_envelope.is_finite()));
        assert!(metrics
            .iter()
            .all(|step| step.losses.auxiliaries.valence_guardrail.is_finite()));
        assert!(metrics.iter().all(|step| step
            .losses
            .auxiliaries
            .bond_length_guardrail
            .is_finite()));
        assert!(metrics
            .iter()
            .any(|step| step.losses.auxiliaries.pocket_contact >= 0.0));
        assert!(metrics.iter().all(|step| {
            step.losses
                .auxiliaries
                .auxiliary_objective_report
                .entries
                .len()
                == 13
        }));
        assert!(metrics.iter().all(|step| {
            step.losses
                .auxiliaries
                .auxiliary_objective_report
                .entries
                .iter()
                .all(|entry| entry.enabled == (entry.effective_weight > 0.0))
        }));
        assert!(metrics.iter().all(|step| step
            .losses
            .auxiliaries
            .auxiliary_objective_report
            .entries
            .iter()
            .all(|entry| entry.weighted_value.is_finite() && !entry.status.is_empty())));
    }

    #[test]
    fn trainer_uses_batched_primary_and_auxiliary_objectives() {
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
        let mut trainer = new_valid_test_trainer(&var_store, config);
        trainer.affinity_measurement_weights =
            measurement_weights(examples, trainer.config.training.affinity_weighting);

        let (primary_batch, primary_components) = compute_primary_objective_batch_with_components(
            trainer.primary_objective.as_ref(),
            examples,
            &forwards,
        );
        let auxiliaries = trainer.auxiliary_objectives.compute_batch(
            examples,
            &forwards,
            |example| trainer.affinity_weight_for(example),
            var_store.device(),
        );
        let denom = examples.len() as f64;
        let mut primary_manual = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            primary_manual += trainer.primary_objective.compute(example, forward);
        }

        assert_close(&primary_batch, &(primary_manual / denom));
        assert!(scalar_or_nan(&auxiliaries.intra_red).is_finite());
        assert!(scalar_or_nan(&auxiliaries.probe).is_finite());
        assert!(scalar_or_nan(&auxiliaries.probe_ligand_pharmacophore).is_finite());
        assert!(scalar_or_nan(&auxiliaries.probe_pocket_pharmacophore).is_finite());
        assert!(scalar_or_nan(&auxiliaries.leak).is_finite());
        assert!(scalar_or_nan(&auxiliaries.leak_topology_to_pocket_role).is_finite());
        assert!(scalar_or_nan(&auxiliaries.leak_geometry_to_pocket_role).is_finite());
        assert!(scalar_or_nan(&auxiliaries.leak_pocket_to_ligand_role).is_finite());
        assert!(scalar_or_nan(&auxiliaries.gate).is_finite());
        assert!(scalar_or_nan(&auxiliaries.slot).is_finite());
        assert!(scalar_or_nan(&auxiliaries.consistency).is_finite());
        assert!(scalar_or_nan(&auxiliaries.pocket_contact).is_finite());
        assert!(scalar_or_nan(&auxiliaries.pocket_clash).is_finite());
        assert!(scalar_or_nan(&auxiliaries.pocket_envelope).is_finite());
        assert!(scalar_or_nan(&auxiliaries.valence_guardrail).is_finite());
        assert!(scalar_or_nan(&auxiliaries.bond_length_guardrail).is_finite());
        assert!(primary_components.topology.is_some());
        assert!(primary_components.geometry.is_some());
        assert!(primary_components.pocket_anchor.is_some());
        assert!(primary_components.rollout.is_none());
        assert!(primary_components.rollout_eval_recovery.is_some());
        assert!(primary_components.rollout_eval_pocket_anchor.is_some());
        assert!(primary_components.flow_velocity.is_none());
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
