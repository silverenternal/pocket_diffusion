//! Primary-objective implementations for the modular research stack.
//!
//! Differentiability contract:
//! - surrogate topology/geometry/pocket reconstruction terms are
//!   tensor-preserving debug losses over encoder and slot outputs.
//! - conditioned denoising topology, coordinate recovery, pairwise distance,
//!   centroid, and pocket-anchor terms are tensor-preserving decoder losses.
//! - sampled rollout records are evaluation-only diagnostics because rollout
//!   commits atom types through argmax/sampling, exports coordinates through
//!   `Vec` records, and stores stop/stability values as scalars.
//! - flow velocity is the direct flow-head loss; endpoint consistency is derived
//!   from the same predicted velocity rather than a separate endpoint head.

use tch::{Kind, Tensor};

use crate::{
    config::{
        PrimaryObjectiveConfig, RolloutTrainingConfig, SparseTopologyCalibrationConfig,
        TrainingConfig,
    },
    data::MolecularExample,
    losses::classification::masked_balanced_bce_with_logits,
    losses::topology_calibration::{SparseTopologyCalibration, SparseTopologyCalibrationScope},
    models::{ResearchForward, TaskDrivenObjective},
    training::metrics::{PrimaryObjectiveComponentMetrics, RolloutTrainingLossMetrics},
};

/// Internal decomposition result for one primary-objective evaluation.
#[derive(Debug)]
pub(crate) struct PrimaryObjectiveComputation {
    /// Primary scalar objective value.
    total: Tensor,
    /// Optional term-level decomposition for reporting.
    components: PrimaryObjectiveComponentMetrics,
}

fn scalar_optional(tensor: &Tensor) -> Option<f64> {
    if tensor.numel() == 0 {
        return None;
    }
    let value = tensor
        .mean(Kind::Float)
        .to_device(tch::Device::Cpu)
        .double_value(&[]);
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

/// Extension interface for primary objectives that expose lightweight component
/// diagnostics alongside the scalar objective.
pub(crate) trait PrimaryObjectiveWithComponents:
    TaskDrivenObjective<ResearchForward>
{
    fn compute_with_components(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> PrimaryObjectiveComputation;

    fn compute_with_components_at_step(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
        _training_step: Option<usize>,
    ) -> PrimaryObjectiveComputation {
        self.compute_with_components(example, forward)
    }
}

/// Reconstruction-style surrogate objective over modality-specific latent paths.
#[derive(Debug, Default, Clone)]
pub struct SurrogateReconstructionObjective;

impl TaskDrivenObjective<ResearchForward> for SurrogateReconstructionObjective {
    fn name(&self) -> &'static str {
        "surrogate_reconstruction"
    }

    fn compute(&self, _example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        self.compute_with_components(_example, forward).total
    }
}

impl PrimaryObjectiveWithComponents for SurrogateReconstructionObjective {
    fn compute_with_components(
        &self,
        _example: &MolecularExample,
        forward: &ResearchForward,
    ) -> PrimaryObjectiveComputation {
        let topo = mse(
            &forward.slots.topology.reconstructed_tokens,
            &forward.encodings.topology.token_embeddings,
        );
        let geo = mse(
            &forward.slots.geometry.reconstructed_tokens,
            &forward.encodings.geometry.token_embeddings,
        );
        let pocket = mse(
            &forward.slots.pocket.reconstructed_tokens,
            &forward.encodings.pocket.token_embeddings,
        );
        let decoder_topology = atom_type_bootstrap_loss(
            &forward.generation.decoded.atom_type_logits,
            &forward.generation.state.partial_ligand.atom_types,
        );
        let decoded_coords = &forward.generation.state.partial_ligand.coords
            + &forward.generation.decoded.coordinate_deltas;
        let decoder_geometry = mse(
            &decoded_coords,
            &forward.generation.state.partial_ligand.coords,
        );

        let total = &topo + &geo + &pocket + 0.1 * (&decoder_topology + &decoder_geometry);
        PrimaryObjectiveComputation {
            total,
            components: PrimaryObjectiveComponentMetrics {
                topology: scalar_optional(&topo),
                geometry: scalar_optional(&geo),
                pocket_anchor: scalar_optional(&pocket),
                rollout: None,
                rollout_eval_recovery: None,
                rollout_eval_pocket_anchor: None,
                rollout_eval_stop: None,
                flow_velocity: None,
                flow_endpoint: None,
                flow_atom_type: None,
                flow_bond: None,
                flow_bond_sparse_negative_rate: None,
                flow_bond_confidence_pressure: None,
                flow_bond_degree_alignment: None,
                flow_topology: None,
                flow_topology_sparse_negative_rate: None,
                flow_topology_confidence_pressure: None,
                flow_topology_degree_alignment: None,
                flow_native_score_calibration: None,
                flow_pocket_context: None,
                flow_synchronization: None,
                ..PrimaryObjectiveComponentMetrics::default()
            },
        }
    }
}

/// Decoder-anchored corruption recovery objective for conditioned ligand generation.
#[derive(Debug, Default, Clone)]
pub struct ConditionedDenoisingObjective;

impl TaskDrivenObjective<ResearchForward> for ConditionedDenoisingObjective {
    fn name(&self) -> &'static str {
        "conditioned_denoising"
    }

    fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        self.compute_with_components(example, forward).total
    }
}

impl PrimaryObjectiveWithComponents for ConditionedDenoisingObjective {
    fn compute_with_components(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> PrimaryObjectiveComputation {
        let supervision = &example.decoder_supervision;
        if !conditioned_denoising_shapes_match(supervision, forward) {
            return zero_primary_computation(forward.generation.decoded.coordinate_deltas.device());
        }
        let preserved_mask = inverse_mask(&supervision.atom_corruption_mask);
        let single_step_topology = masked_atom_type_loss(
            &forward.generation.decoded.atom_type_logits,
            &supervision.target_atom_types,
            &supervision.atom_corruption_mask,
        );
        let topology_preservation = masked_atom_type_loss(
            &forward.generation.decoded.atom_type_logits,
            &supervision.target_atom_types,
            &preserved_mask,
        );
        let single_step_predicted_coords =
            &supervision.noisy_coords + &forward.generation.decoded.coordinate_deltas;
        let single_step_geometry = masked_coord_denoising_loss(
            &single_step_predicted_coords,
            &supervision.target_coords,
            &supervision.atom_corruption_mask,
        );
        let geometry_preservation = masked_coord_denoising_loss(
            &single_step_predicted_coords,
            &supervision.target_coords,
            &preserved_mask,
        );
        let direct_noise_recovery = masked_coord_denoising_loss(
            &forward.generation.decoded.coordinate_deltas,
            &(-&supervision.coordinate_noise),
            &supervision.atom_corruption_mask,
        );
        let single_step_pairwise = pairwise_distance_recovery_loss(
            &single_step_predicted_coords,
            &supervision.target_pairwise_distances,
            &supervision.atom_corruption_mask,
        );
        let single_step_centroid =
            centroid_recovery_loss(&single_step_predicted_coords, &supervision.target_coords);
        let pocket_anchor =
            pocket_anchor_loss(&single_step_predicted_coords, &example.pocket.coords);
        let rollout_eval_recovery =
            rollout_eval_recovery_metric(supervision, &forward.generation.rollout);
        let rollout_eval_pocket_anchor =
            rollout_eval_pocket_anchor_metric(&forward.generation.rollout, &example.pocket.coords);
        let rollout_eval_stop = rollout_eval_stop_metric(supervision, &forward.generation.rollout);

        let total = &single_step_topology
            + &single_step_geometry
            + 0.25 * &topology_preservation
            + 0.15 * &geometry_preservation
            + 0.2 * &direct_noise_recovery
            + 0.2 * &single_step_pairwise
            + 0.15 * &single_step_centroid
            + 0.2 * &pocket_anchor;
        let topology = &single_step_topology + 0.25 * &topology_preservation;
        let geometry = &single_step_geometry
            + 0.15 * &geometry_preservation
            + 0.2 * &direct_noise_recovery
            + 0.2 * &single_step_pairwise
            + 0.15 * &single_step_centroid;
        let pocket_anchor = 0.2 * &pocket_anchor;

        PrimaryObjectiveComputation {
            total,
            components: PrimaryObjectiveComponentMetrics {
                topology: scalar_optional(&topology),
                geometry: scalar_optional(&geometry),
                pocket_anchor: scalar_optional(&pocket_anchor),
                rollout: None,
                rollout_eval_recovery: scalar_optional(&rollout_eval_recovery),
                rollout_eval_pocket_anchor: scalar_optional(&rollout_eval_pocket_anchor),
                rollout_eval_stop: scalar_optional(&rollout_eval_stop),
                flow_velocity: None,
                flow_endpoint: None,
                flow_atom_type: None,
                flow_bond: None,
                flow_bond_sparse_negative_rate: None,
                flow_bond_confidence_pressure: None,
                flow_bond_degree_alignment: None,
                flow_topology: None,
                flow_topology_sparse_negative_rate: None,
                flow_topology_confidence_pressure: None,
                flow_topology_degree_alignment: None,
                flow_native_score_calibration: None,
                flow_pocket_context: None,
                flow_synchronization: None,
                ..PrimaryObjectiveComponentMetrics::default()
            },
        }
    }
}

fn zero_primary_computation(device: tch::Device) -> PrimaryObjectiveComputation {
    PrimaryObjectiveComputation {
        total: Tensor::zeros([1], (Kind::Float, device)),
        components: PrimaryObjectiveComponentMetrics::default(),
    }
}

fn conditioned_denoising_shapes_match(
    supervision: &crate::data::DecoderSupervision,
    forward: &ResearchForward,
) -> bool {
    let atom_count = supervision
        .target_atom_types
        .size()
        .first()
        .copied()
        .unwrap_or(0);
    atom_count > 0
        && supervision
            .corrupted_atom_types
            .size()
            .first()
            .copied()
            .unwrap_or(-1)
            == atom_count
        && supervision
            .atom_corruption_mask
            .size()
            .first()
            .copied()
            .unwrap_or(-1)
            == atom_count
        && supervision.target_coords.size().as_slice() == [atom_count, 3]
        && supervision.noisy_coords.size().as_slice() == [atom_count, 3]
        && supervision.coordinate_noise.size().as_slice() == [atom_count, 3]
        && supervision.target_pairwise_distances.size().as_slice() == [atom_count, atom_count]
        && forward
            .generation
            .decoded
            .atom_type_logits
            .size()
            .first()
            .copied()
            .unwrap_or(-1)
            == atom_count
        && forward
            .generation
            .decoded
            .coordinate_deltas
            .size()
            .as_slice()
            == [atom_count, 3]
}

/// Geometry-only flow-matching objective over predicted velocity fields.
#[derive(Debug, Clone)]
pub struct FlowMatchingObjective {
    weight: f64,
    topology_calibration: SparseTopologyCalibration,
}

impl TaskDrivenObjective<ResearchForward> for FlowMatchingObjective {
    fn name(&self) -> &'static str {
        "flow_matching"
    }

    fn compute(&self, _example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        self.compute_with_components(_example, forward).total
    }
}

impl PrimaryObjectiveWithComponents for FlowMatchingObjective {
    fn compute_with_components(
        &self,
        _example: &MolecularExample,
        forward: &ResearchForward,
    ) -> PrimaryObjectiveComputation {
        self.compute_with_components_at_step(_example, forward, None)
    }

    fn compute_with_components_at_step(
        &self,
        _example: &MolecularExample,
        forward: &ResearchForward,
        training_step: Option<usize>,
    ) -> PrimaryObjectiveComputation {
        let (velocity_loss, endpoint_loss) = flow_matching_velocity_and_endpoint_loss(forward);
        let molecular = molecular_flow_losses(forward, &self.topology_calibration, training_step);
        let geometry_branch_weight = forward
            .generation
            .flow_matching
            .as_ref()
            .map(|flow| flow.branch_weights.geometry)
            .unwrap_or(1.0);
        let scaled_velocity = &velocity_loss * self.weight * geometry_branch_weight;
        let scaled_endpoint = &endpoint_loss * 0.2 * self.weight * geometry_branch_weight;
        let scaled_atom_type = &molecular.atom_type * self.weight;
        let scaled_bond = &molecular.bond * self.weight;
        let scaled_bond_sparse_negative_rate = &molecular.bond_sparse_negative_rate * self.weight;
        let scaled_bond_confidence_pressure = &molecular.bond_confidence_pressure * self.weight;
        let scaled_bond_degree_alignment = &molecular.bond_degree_alignment * self.weight;
        let scaled_topology = &molecular.topology * self.weight;
        let scaled_topology_sparse_negative_rate =
            &molecular.topology_sparse_negative_rate * self.weight;
        let scaled_topology_confidence_pressure =
            &molecular.topology_confidence_pressure * self.weight;
        let scaled_topology_degree_alignment = &molecular.topology_degree_alignment * self.weight;
        let scaled_native_score_calibration =
            &molecular.native_score_calibration.total * self.weight;
        let scaled_native_score_uncapped_raw =
            &molecular.native_score_calibration.uncapped_total * self.weight;
        let native_score_cap_scale = &molecular.native_score_calibration.cap_scale;
        let scaled_native_score_false_positive_margin =
            &molecular.native_score_calibration.false_positive_margin * self.weight;
        let scaled_native_score_false_negative_margin =
            &molecular.native_score_calibration.false_negative_margin * self.weight;
        let scaled_native_score_density_budget =
            &molecular.native_score_calibration.density_budget * self.weight;
        let scaled_native_score_soft_positive_miss =
            &molecular.native_score_calibration.soft_positive_miss * self.weight;
        let scaled_native_score_soft_negative_extraction =
            &molecular.native_score_calibration.soft_negative_extraction * self.weight;
        let scaled_native_score_soft_extraction_budget =
            &molecular.native_score_calibration.soft_extraction_budget * self.weight;
        let scaled_native_score_degree_alignment =
            &molecular.native_score_calibration.degree_alignment * self.weight;
        let scaled_native_score_score_separation =
            &molecular.native_score_calibration.score_separation * self.weight;
        let scaled_pocket_context = &molecular.pocket_context * self.weight;
        let scaled_synchronization = &molecular.synchronization * self.weight;
        PrimaryObjectiveComputation {
            total: &scaled_velocity
                + &scaled_endpoint
                + &scaled_atom_type
                + &scaled_bond
                + &scaled_topology
                + &scaled_native_score_calibration
                + &scaled_pocket_context
                + &scaled_synchronization,
            components: PrimaryObjectiveComponentMetrics {
                topology: None,
                geometry: None,
                pocket_anchor: None,
                rollout: None,
                rollout_eval_recovery: None,
                rollout_eval_pocket_anchor: None,
                rollout_eval_stop: None,
                flow_velocity: scalar_optional(&scaled_velocity),
                flow_endpoint: scalar_optional(&scaled_endpoint),
                flow_atom_type: scalar_optional(&scaled_atom_type),
                flow_bond: scalar_optional(&scaled_bond),
                flow_bond_sparse_negative_rate: scalar_optional(&scaled_bond_sparse_negative_rate),
                flow_bond_confidence_pressure: scalar_optional(&scaled_bond_confidence_pressure),
                flow_bond_degree_alignment: scalar_optional(&scaled_bond_degree_alignment),
                flow_topology: scalar_optional(&scaled_topology),
                flow_topology_sparse_negative_rate: scalar_optional(
                    &scaled_topology_sparse_negative_rate,
                ),
                flow_topology_confidence_pressure: scalar_optional(
                    &scaled_topology_confidence_pressure,
                ),
                flow_topology_degree_alignment: scalar_optional(&scaled_topology_degree_alignment),
                flow_native_score_calibration: scalar_optional(&scaled_native_score_calibration),
                flow_native_score_calibration_uncapped_raw: scalar_optional(
                    &scaled_native_score_uncapped_raw,
                ),
                flow_native_score_calibration_cap_scale: scalar_optional(native_score_cap_scale),
                flow_native_score_calibration_false_positive_margin: scalar_optional(
                    &scaled_native_score_false_positive_margin,
                ),
                flow_native_score_calibration_false_negative_margin: scalar_optional(
                    &scaled_native_score_false_negative_margin,
                ),
                flow_native_score_calibration_density_budget: scalar_optional(
                    &scaled_native_score_density_budget,
                ),
                flow_native_score_calibration_soft_positive_miss: scalar_optional(
                    &scaled_native_score_soft_positive_miss,
                ),
                flow_native_score_calibration_soft_negative_extraction: scalar_optional(
                    &scaled_native_score_soft_negative_extraction,
                ),
                flow_native_score_calibration_soft_extraction_budget: scalar_optional(
                    &scaled_native_score_soft_extraction_budget,
                ),
                flow_native_score_calibration_degree_alignment: scalar_optional(
                    &scaled_native_score_degree_alignment,
                ),
                flow_native_score_calibration_score_separation: scalar_optional(
                    &scaled_native_score_score_separation,
                ),
                flow_pocket_context: scalar_optional(&scaled_pocket_context),
                flow_synchronization: scalar_optional(&scaled_synchronization),
            },
        }
    }
}

/// Hybrid objective combining denoising and flow matching.
#[derive(Debug, Clone)]
pub struct DenoisingFlowMatchingObjective {
    denoising_weight: f64,
    flow_weight: f64,
    topology_calibration: SparseTopologyCalibration,
}

impl TaskDrivenObjective<ResearchForward> for DenoisingFlowMatchingObjective {
    fn name(&self) -> &'static str {
        "denoising_flow_matching"
    }

    fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        self.compute_with_components(example, forward).total
    }
}

impl PrimaryObjectiveWithComponents for DenoisingFlowMatchingObjective {
    fn compute_with_components(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
    ) -> PrimaryObjectiveComputation {
        self.compute_with_components_at_step(example, forward, None)
    }

    fn compute_with_components_at_step(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
        training_step: Option<usize>,
    ) -> PrimaryObjectiveComputation {
        let denoising = ConditionedDenoisingObjective.compute_with_components(example, forward);
        let flow = FlowMatchingObjective {
            weight: self.flow_weight,
            topology_calibration: self.topology_calibration.clone(),
        }
        .compute_with_components_at_step(example, forward, training_step);
        let scaled_denoising_total = denoising.total * self.denoising_weight;
        let mut denoising_components = denoising.components.scale(self.denoising_weight);
        denoising_components.add_assign(&flow.components);

        PrimaryObjectiveComputation {
            total: scaled_denoising_total + flow.total,
            components: denoising_components,
        }
    }
}

/// Build the configured primary objective implementation.
pub(crate) fn build_primary_objective(
    training: &TrainingConfig,
) -> Box<dyn PrimaryObjectiveWithComponents> {
    match training.primary_objective {
        PrimaryObjectiveConfig::SurrogateReconstruction => {
            Box::new(SurrogateReconstructionObjective)
        }
        PrimaryObjectiveConfig::ConditionedDenoising => Box::new(ConditionedDenoisingObjective),
        PrimaryObjectiveConfig::FlowMatching => Box::new(FlowMatchingObjective {
            weight: training.flow_matching_loss_weight,
            topology_calibration: SparseTopologyCalibration::new(
                training.sparse_topology_calibration.clone(),
            ),
        }),
        PrimaryObjectiveConfig::DenoisingFlowMatching => Box::new(DenoisingFlowMatchingObjective {
            denoising_weight: training.hybrid_denoising_weight,
            flow_weight: training.hybrid_flow_weight,
            topology_calibration: SparseTopologyCalibration::new(
                training.sparse_topology_calibration.clone(),
            ),
        }),
    }
}

/// Compute the configured primary objective and return an optional component
/// breakdown alongside the batch scalar.
#[allow(dead_code)] // Compatibility wrapper for tests and callers without trainer-step context.
pub(crate) fn compute_primary_objective_batch_with_components(
    objective: &dyn PrimaryObjectiveWithComponents,
    examples: &[MolecularExample],
    forwards: &[ResearchForward],
) -> (Tensor, PrimaryObjectiveComponentMetrics) {
    compute_primary_objective_batch_with_components_at_step(objective, examples, forwards, None)
}

/// Compute the configured primary objective with trainer-step-aware sub-objectives.
pub(crate) fn compute_primary_objective_batch_with_components_at_step(
    objective: &dyn PrimaryObjectiveWithComponents,
    examples: &[MolecularExample],
    forwards: &[ResearchForward],
    training_step: Option<usize>,
) -> (Tensor, PrimaryObjectiveComponentMetrics) {
    debug_assert_eq!(examples.len(), forwards.len());
    let device = forwards
        .first()
        .map(|forward| forward.generation.decoded.coordinate_deltas.device())
        .or_else(|| {
            examples
                .first()
                .map(|example| example.topology.atom_types.device())
        })
        .unwrap_or(tch::Device::Cpu);
    if examples.is_empty() {
        return (
            Tensor::zeros([1], (Kind::Float, device)),
            PrimaryObjectiveComponentMetrics::default(),
        );
    }

    let mut total = Tensor::zeros([1], (Kind::Float, device));
    let mut components = PrimaryObjectiveComponentMetrics::default();
    for (example, forward) in examples.iter().zip(forwards.iter()) {
        let item = objective.compute_with_components_at_step(example, forward, training_step);
        total += item.total;
        components.add_assign(&item.components);
    }
    let scale = 1.0 / examples.len() as f64;
    (total * scale, components.scale(scale))
}

/// Compute the optional optimizer-facing short-rollout objective.
pub(crate) fn compute_rollout_training_loss(
    config: &RolloutTrainingConfig,
    topology_calibration: &SparseTopologyCalibrationConfig,
    training_step: usize,
    examples: &[MolecularExample],
    forwards: &[ResearchForward],
) -> (Tensor, RolloutTrainingLossMetrics) {
    let device = forwards
        .first()
        .map(|forward| forward.generation.decoded.coordinate_deltas.device())
        .or_else(|| {
            examples
                .first()
                .map(|example| example.topology.atom_types.device())
        })
        .unwrap_or(tch::Device::Cpu);
    let mut metrics = RolloutTrainingLossMetrics {
        enabled: config.enabled,
        active: config.active_at_step(training_step),
        warmup_step: config.warmup_step,
        configured_steps: config.rollout_steps,
        max_batch_examples: config.max_batch_examples,
        detach_policy: config.detach_policy.as_str().to_string(),
        target_source: "generated_rollout_state".to_string(),
        memory_control: format!(
            "bounded_steps={};max_batch_examples={};detach_policy={}",
            config.rollout_steps,
            config.max_batch_examples,
            config.detach_policy.as_str()
        ),
        ..RolloutTrainingLossMetrics::default()
    };
    if !metrics.active || examples.is_empty() {
        return (Tensor::zeros([1], (Kind::Float, device)), metrics);
    }

    let mut total = Tensor::zeros([1], (Kind::Float, device));
    let mut atom_total = 0.0;
    let mut bond_total = 0.0;
    let mut bond_sparse_negative_rate_total = 0.0;
    let mut pocket_total = 0.0;
    let mut clash_total = 0.0;
    let mut endpoint_total = 0.0;
    let mut generated_validity_total = 0.0;
    let mut contributing_examples = 0usize;
    let mut executed_steps = 0usize;
    let topology_calibration = SparseTopologyCalibration::new(topology_calibration.clone());

    for (example, forward) in examples
        .iter()
        .zip(forwards.iter())
        .take(config.max_batch_examples)
    {
        let record = &forward.generation.rollout_training;
        if !record.enabled || !record.mode_allowed || record.steps.is_empty() {
            continue;
        }
        metrics.configured_steps = record.configured_steps;
        metrics.detach_policy = record.detach_policy.clone();
        metrics.target_source = record.target_source.clone();
        metrics.memory_control = record.memory_control.clone();
        let mut example_total = Tensor::zeros([1], (Kind::Float, device));
        let mut example_validity = 0.0;
        executed_steps += record.executed_steps;
        for step in &record.steps {
            let atom = rollout_atom_validity_loss(
                &step.atom_type_logits,
                &example.decoder_supervision.target_atom_types,
                &step.atom_mask,
            ) * config.atom_validity_weight;
            let pocket = rollout_pocket_contact_loss(&step.coords, &example.pocket.coords)
                * config.pocket_contact_weight;
            let clash = rollout_clash_margin_loss(&step.coords, &example.pocket.coords)
                * config.clash_weight;
            let endpoint = rollout_endpoint_consistency_loss(
                &step.coords,
                &example.decoder_supervision.target_coords,
                &step.atom_mask,
            ) * config.endpoint_consistency_weight;
            example_total += &atom + &pocket + &clash + &endpoint;
            atom_total += scalar_or_zero(&atom);
            pocket_total += scalar_or_zero(&pocket);
            clash_total += scalar_or_zero(&clash);
            endpoint_total += scalar_or_zero(&endpoint);
            example_validity += rollout_validity_proxy(&atom, &clash);
        }
        let bond_losses =
            rollout_bond_consistency_loss(forward, &topology_calibration, Some(training_step));
        let bond = &bond_losses.total * config.bond_consistency_weight;
        let bond_sparse_negative_rate =
            &bond_losses.sparse_negative_rate * config.bond_consistency_weight;
        example_total += &bond;
        bond_total += scalar_or_zero(&bond);
        bond_sparse_negative_rate_total += scalar_or_zero(&bond_sparse_negative_rate);
        total += example_total / record.steps.len().max(1) as f64;
        generated_validity_total += example_validity / record.steps.len().max(1) as f64;
        contributing_examples += 1;
    }

    if contributing_examples == 0 {
        return (Tensor::zeros([1], (Kind::Float, device)), metrics);
    }

    let scale = 1.0 / contributing_examples as f64;
    total *= scale;
    metrics.contributing_examples = contributing_examples;
    metrics.executed_steps_mean = executed_steps as f64 / contributing_examples as f64;
    metrics.rollout_state_loss = scalar_or_zero(&total);
    metrics.atom_validity = atom_total * scale;
    metrics.bond_consistency = bond_total * scale;
    metrics.bond_sparse_negative_rate = bond_sparse_negative_rate_total * scale;
    metrics.pocket_contact = pocket_total * scale;
    metrics.clash_margin = clash_total * scale;
    metrics.endpoint_consistency = endpoint_total * scale;
    metrics.generated_state_validity = (generated_validity_total * scale).clamp(0.0, 1.0);
    (total, metrics)
}

fn rollout_atom_validity_loss(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    let atom_count = logits.size().first().copied().unwrap_or(0);
    if atom_count <= 0
        || target.size().first().copied().unwrap_or(-1) != atom_count
        || mask.size().first().copied().unwrap_or(-1) != atom_count
    {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    masked_atom_type_cross_entropy(logits, target, mask)
}

struct RolloutBondConsistencyLoss {
    total: Tensor,
    sparse_negative_rate: Tensor,
}

fn rollout_bond_consistency_loss(
    forward: &ResearchForward,
    calibration: &SparseTopologyCalibration,
    training_step: Option<usize>,
) -> RolloutBondConsistencyLoss {
    let device = forward.generation.decoded.coordinate_deltas.device();
    let Some(flow) = forward.generation.flow_matching.as_ref() else {
        return zero_rollout_bond_consistency_loss(device);
    };
    let Some(molecular) = flow.molecular.as_ref() else {
        return zero_rollout_bond_consistency_loss(device);
    };
    let sparse = calibration.loss(
        SparseTopologyCalibrationScope::Rollout,
        training_step,
        &molecular.bond_exists_logits,
        &molecular.target_adjacency,
        &molecular.pair_mask,
    );
    let total = masked_balanced_bce_with_logits(
        &molecular.bond_exists_logits,
        &molecular.target_adjacency,
        &molecular.pair_mask,
    ) + &sparse.weighted;
    RolloutBondConsistencyLoss {
        total,
        sparse_negative_rate: sparse.weighted,
    }
}

fn zero_rollout_bond_consistency_loss(device: tch::Device) -> RolloutBondConsistencyLoss {
    let zero = Tensor::zeros([1], (Kind::Float, device));
    RolloutBondConsistencyLoss {
        total: zero.shallow_clone(),
        sparse_negative_rate: zero,
    }
}

fn rollout_pocket_contact_loss(coords: &Tensor, pocket_coords: &Tensor) -> Tensor {
    let Some(min_distances) = atom_pocket_min_distances(coords, pocket_coords) else {
        return Tensor::zeros([1], (Kind::Float, coords.device()));
    };
    (min_distances - 4.5).relu().mean(Kind::Float)
}

fn rollout_clash_margin_loss(coords: &Tensor, pocket_coords: &Tensor) -> Tensor {
    let Some(min_distances) = atom_pocket_min_distances(coords, pocket_coords) else {
        return Tensor::zeros([1], (Kind::Float, coords.device()));
    };
    (Tensor::from(1.2f32).to_device(coords.device()) - min_distances)
        .relu()
        .mean(Kind::Float)
}

fn rollout_endpoint_consistency_loss(
    coords: &Tensor,
    target_coords: &Tensor,
    mask: &Tensor,
) -> Tensor {
    if coords.size() != target_coords.size()
        || mask.size().first().copied().unwrap_or(-1) != coords.size().first().copied().unwrap_or(0)
    {
        return Tensor::zeros([1], (Kind::Float, coords.device()));
    }
    masked_coord_denoising_loss(coords, target_coords, mask)
}

fn atom_pocket_min_distances(coords: &Tensor, pocket_coords: &Tensor) -> Option<Tensor> {
    if coords.numel() == 0 || pocket_coords.numel() == 0 {
        return None;
    }
    let diffs = coords.unsqueeze(1) - pocket_coords.to_device(coords.device()).unsqueeze(0);
    let distances = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .clamp_min(1e-12)
        .sqrt();
    Some(distances.min_dim(1, false).0)
}

fn rollout_validity_proxy(atom: &Tensor, clash: &Tensor) -> f64 {
    let burden = scalar_or_zero(atom) + scalar_or_zero(clash);
    if burden.is_finite() {
        1.0 / (1.0 + burden.max(0.0))
    } else {
        0.0
    }
}

fn scalar_or_zero(tensor: &Tensor) -> f64 {
    scalar_optional(tensor).unwrap_or(0.0)
}

fn flow_matching_velocity_and_endpoint_loss(forward: &ResearchForward) -> (Tensor, Tensor) {
    let Some(flow) = forward.generation.flow_matching.as_ref() else {
        panic!(
            "flow_matching primary objective requires forward.generation.flow_matching; validate generation_method.primary_backend.family=flow_matching with training.primary_objective=flow_matching or denoising_flow_matching"
        );
    };
    if flow.predicted_velocity.numel() == 0 || flow.target_velocity.numel() == 0 {
        // Empty tensors are treated as an intentional shape-empty diagnostic
        // path; absent flow records are configuration errors handled above.
        return (
            Tensor::zeros([1], (Kind::Float, flow.predicted_velocity.device())),
            Tensor::zeros([1], (Kind::Float, flow.predicted_velocity.device())),
        );
    }
    let velocity_mse = (&flow.predicted_velocity - &flow.target_velocity)
        .pow_tensor_scalar(2.0)
        .mean_dim([1].as_slice(), false, Kind::Float);
    let velocity_loss = weighted_mean(&velocity_mse, &flow.atom_mask);

    let t = flow.t.clamp(0.0, 1.0);
    let x0_true = &flow.sampled_coords - &flow.target_velocity * t;
    let x1_true = &flow.sampled_coords + &flow.target_velocity * (1.0 - t);
    let x0_pred = &flow.sampled_coords - &flow.predicted_velocity * t;
    let x1_pred = &flow.sampled_coords + &flow.predicted_velocity * (1.0 - t);

    let x0_mse =
        (&x0_pred - &x0_true)
            .pow_tensor_scalar(2.0)
            .mean_dim([1].as_slice(), false, Kind::Float);
    let x1_mse =
        (&x1_pred - &x1_true)
            .pow_tensor_scalar(2.0)
            .mean_dim([1].as_slice(), false, Kind::Float);
    let endpoint_loss =
        weighted_mean(&x0_mse, &flow.atom_mask) + weighted_mean(&x1_mse, &flow.atom_mask);
    (velocity_loss, endpoint_loss)
}

struct MolecularFlowLosses {
    atom_type: Tensor,
    bond: Tensor,
    bond_sparse_negative_rate: Tensor,
    bond_confidence_pressure: Tensor,
    bond_degree_alignment: Tensor,
    topology: Tensor,
    topology_sparse_negative_rate: Tensor,
    topology_confidence_pressure: Tensor,
    topology_degree_alignment: Tensor,
    native_score_calibration: MolecularFlowNativeScoreCalibrationLosses,
    pocket_context: Tensor,
    synchronization: Tensor,
}

struct MolecularFlowNativeScoreCalibrationLosses {
    total: Tensor,
    uncapped_total: Tensor,
    cap_scale: Tensor,
    false_positive_margin: Tensor,
    false_negative_margin: Tensor,
    density_budget: Tensor,
    soft_positive_miss: Tensor,
    soft_negative_extraction: Tensor,
    soft_extraction_budget: Tensor,
    degree_alignment: Tensor,
    score_separation: Tensor,
}

impl MolecularFlowNativeScoreCalibrationLosses {
    fn zero(device: tch::Device) -> Self {
        Self {
            total: Tensor::zeros([1], (Kind::Float, device)),
            uncapped_total: Tensor::zeros([1], (Kind::Float, device)),
            cap_scale: Tensor::zeros([1], (Kind::Float, device)),
            false_positive_margin: Tensor::zeros([1], (Kind::Float, device)),
            false_negative_margin: Tensor::zeros([1], (Kind::Float, device)),
            density_budget: Tensor::zeros([1], (Kind::Float, device)),
            soft_positive_miss: Tensor::zeros([1], (Kind::Float, device)),
            soft_negative_extraction: Tensor::zeros([1], (Kind::Float, device)),
            soft_extraction_budget: Tensor::zeros([1], (Kind::Float, device)),
            degree_alignment: Tensor::zeros([1], (Kind::Float, device)),
            score_separation: Tensor::zeros([1], (Kind::Float, device)),
        }
    }
}

fn molecular_flow_losses(
    forward: &ResearchForward,
    calibration: &SparseTopologyCalibration,
    training_step: Option<usize>,
) -> MolecularFlowLosses {
    let device = forward.generation.decoded.coordinate_deltas.device();
    let Some(flow) = forward.generation.flow_matching.as_ref() else {
        return zero_molecular_flow_losses(device);
    };
    let Some(molecular) = flow.molecular.as_ref() else {
        return zero_molecular_flow_losses(device);
    };
    let zero = || Tensor::zeros([1], (Kind::Float, device));

    let atom_type =
        if molecular.branch_weights.atom_type <= 0.0 || molecular.atom_type_logits.numel() == 0 {
            zero()
        } else {
            masked_atom_type_cross_entropy(
                &molecular.atom_type_logits,
                &molecular.target_atom_types,
                &molecular.target_atom_mask,
            ) * molecular.branch_weights.atom_type
        };

    let (bond, bond_sparse_negative_rate, bond_confidence_pressure, bond_degree_alignment) =
        if molecular.branch_weights.bond <= 0.0 {
            (zero(), zero(), zero(), zero())
        } else {
            let sparse = calibration.loss(
                SparseTopologyCalibrationScope::Flow,
                training_step,
                &molecular.bond_exists_logits,
                &molecular.target_adjacency,
                &molecular.pair_mask,
            );
            let confidence = calibration.confidence_pressure_loss(
                SparseTopologyCalibrationScope::Flow,
                training_step,
                &molecular.bond_exists_logits,
                &molecular.target_adjacency,
                &molecular.pair_mask,
            );
            let degree = calibration.degree_alignment_loss(
                SparseTopologyCalibrationScope::Flow,
                training_step,
                &molecular.bond_exists_logits,
                &molecular.target_adjacency,
                &molecular.pair_mask,
            );
            let weighted_sparse = &sparse.weighted * molecular.branch_weights.bond;
            let weighted_confidence = &confidence.weighted * molecular.branch_weights.bond;
            let weighted_degree = &degree.weighted * molecular.branch_weights.bond;
            let bond_exists = masked_balanced_bce_with_logits(
                &molecular.bond_exists_logits,
                &molecular.target_adjacency,
                &molecular.pair_mask,
            ) + &sparse.weighted
                + &degree.weighted;
            let positive_pair_mask =
                &molecular.pair_mask * molecular.target_adjacency.clamp(0.0, 1.0);
            let bond_type = masked_pair_cross_entropy(
                &molecular.bond_type_logits,
                &molecular.target_bond_types,
                &positive_pair_mask,
            );
            (
                (bond_exists + bond_type + &confidence.weighted) * molecular.branch_weights.bond,
                weighted_sparse,
                weighted_confidence,
                weighted_degree,
            )
        };

    let (
        topology,
        topology_sparse_negative_rate,
        topology_confidence_pressure,
        topology_degree_alignment,
    ) = if molecular.branch_weights.topology <= 0.0 {
        (zero(), zero(), zero(), zero())
    } else {
        let sparse = calibration.loss(
            SparseTopologyCalibrationScope::Flow,
            training_step,
            &molecular.topology_logits,
            &molecular.target_topology,
            &molecular.pair_mask,
        );
        let confidence = calibration.confidence_pressure_loss(
            SparseTopologyCalibrationScope::Flow,
            training_step,
            &molecular.topology_logits,
            &molecular.target_topology,
            &molecular.pair_mask,
        );
        let degree = calibration.degree_alignment_loss(
            SparseTopologyCalibrationScope::Flow,
            training_step,
            &molecular.topology_logits,
            &molecular.target_topology,
            &molecular.pair_mask,
        );
        let weighted_sparse = &sparse.weighted * molecular.branch_weights.topology;
        let weighted_confidence = &confidence.weighted * molecular.branch_weights.topology;
        let weighted_degree = &degree.weighted * molecular.branch_weights.topology;
        let topology_bce = masked_balanced_bce_with_logits(
            &molecular.topology_logits,
            &molecular.target_topology,
            &molecular.pair_mask,
        ) + &sparse.weighted
            + &degree.weighted;
        (
            (topology_bce + &confidence.weighted) * molecular.branch_weights.topology,
            weighted_sparse,
            weighted_confidence,
            weighted_degree,
        )
    };

    let native_score_calibration = molecular_native_score_calibration_losses(
        calibration,
        training_step,
        &molecular.bond_exists_logits,
        &molecular.topology_logits,
        &molecular.target_adjacency,
        &molecular.pair_mask,
        molecular.branch_weights.bond,
        molecular.branch_weights.topology,
    );

    let pocket_context = if molecular.branch_weights.pocket_context <= 0.0
        || molecular.pocket_contact_logits.numel() == 0
        || molecular.target_pocket_contacts.numel() == 0
    {
        zero()
    } else {
        masked_balanced_bce_with_logits(
            &molecular.pocket_contact_logits,
            &molecular.target_pocket_contacts,
            &molecular.pocket_interaction_mask,
        ) * molecular.branch_weights.pocket_context
    };

    let synchronization = if molecular.branch_weights.synchronization <= 0.0 {
        zero()
    } else {
        ((molecular.bond_exists_logits.sigmoid() - molecular.topology_logits.sigmoid())
            .pow_tensor_scalar(2.0)
            * &molecular.pair_mask)
            .sum(Kind::Float)
            / molecular.pair_mask.sum(Kind::Float).clamp_min(1.0)
            * molecular.branch_weights.synchronization
    };

    MolecularFlowLosses {
        atom_type,
        bond,
        bond_sparse_negative_rate,
        bond_confidence_pressure,
        bond_degree_alignment,
        topology,
        topology_sparse_negative_rate,
        topology_confidence_pressure,
        topology_degree_alignment,
        native_score_calibration,
        pocket_context,
        synchronization,
    }
}

fn molecular_native_score_calibration_losses(
    calibration: &SparseTopologyCalibration,
    training_step: Option<usize>,
    bond_logits: &Tensor,
    topology_logits: &Tensor,
    target_adjacency: &Tensor,
    pair_mask: &Tensor,
    bond_branch_weight: f64,
    topology_branch_weight: f64,
) -> MolecularFlowNativeScoreCalibrationLosses {
    let device = bond_logits.device();
    if bond_branch_weight <= 0.0 || topology_branch_weight <= 0.0 {
        return MolecularFlowNativeScoreCalibrationLosses::zero(device);
    }

    let score = calibration.native_score_calibration_loss(
        SparseTopologyCalibrationScope::Flow,
        training_step,
        bond_logits,
        topology_logits,
        target_adjacency,
        pair_mask,
    );
    let graph_branch_weight = 0.5 * (bond_branch_weight + topology_branch_weight);
    let cap_scale =
        (&score.raw.detach() / score.uncapped_raw.detach().clamp_min(1.0e-6)).clamp(0.0, 1.0);
    let Some(components) = score.native_score_components else {
        let mut losses = MolecularFlowNativeScoreCalibrationLosses::zero(device);
        losses.total = &score.weighted * graph_branch_weight;
        losses.uncapped_total = &score.uncapped_raw * score.effective_weight * graph_branch_weight;
        losses.cap_scale = cap_scale;
        return losses;
    };

    MolecularFlowNativeScoreCalibrationLosses {
        total: &score.weighted * graph_branch_weight,
        uncapped_total: &score.uncapped_raw * score.effective_weight * graph_branch_weight,
        cap_scale,
        false_positive_margin: components.false_positive_margin * graph_branch_weight,
        false_negative_margin: components.false_negative_margin * graph_branch_weight,
        density_budget: components.density_budget * graph_branch_weight,
        soft_positive_miss: components.soft_positive_miss * graph_branch_weight,
        soft_negative_extraction: components.soft_negative_extraction * graph_branch_weight,
        soft_extraction_budget: components.soft_extraction_budget * graph_branch_weight,
        degree_alignment: components.degree_alignment * graph_branch_weight,
        score_separation: components.score_separation * graph_branch_weight,
    }
}

fn zero_molecular_flow_losses(device: tch::Device) -> MolecularFlowLosses {
    MolecularFlowLosses {
        atom_type: Tensor::zeros([1], (Kind::Float, device)),
        bond: Tensor::zeros([1], (Kind::Float, device)),
        bond_sparse_negative_rate: Tensor::zeros([1], (Kind::Float, device)),
        bond_confidence_pressure: Tensor::zeros([1], (Kind::Float, device)),
        bond_degree_alignment: Tensor::zeros([1], (Kind::Float, device)),
        topology: Tensor::zeros([1], (Kind::Float, device)),
        topology_sparse_negative_rate: Tensor::zeros([1], (Kind::Float, device)),
        topology_confidence_pressure: Tensor::zeros([1], (Kind::Float, device)),
        topology_degree_alignment: Tensor::zeros([1], (Kind::Float, device)),
        native_score_calibration: MolecularFlowNativeScoreCalibrationLosses::zero(device),
        pocket_context: Tensor::zeros([1], (Kind::Float, device)),
        synchronization: Tensor::zeros([1], (Kind::Float, device)),
    }
}

fn masked_atom_type_cross_entropy(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0 || target.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let log_probs = logits.log_softmax(-1, Kind::Float);
    let target = target
        .to_device(logits.device())
        .to_kind(Kind::Int64)
        .clamp(
            0,
            logits.size().get(1).copied().unwrap_or(1).saturating_sub(1),
        );
    let nll = -log_probs
        .gather(1, &target.unsqueeze(1), false)
        .squeeze_dim(1);
    weighted_mean(&nll, mask)
}

fn masked_pair_cross_entropy(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0 || targets.numel() == 0 || mask.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let size = logits.size();
    if size.len() != 3 || targets.size().as_slice() != [size[0], size[1]] {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let classes = size[2];
    let flat_logits = logits.reshape([size[0] * size[1], classes]);
    let flat_targets = targets.reshape([size[0] * size[1]]).to_kind(Kind::Int64);
    let flat_mask = mask.reshape([size[0] * size[1]]).to_kind(Kind::Float);
    let per_pair = flat_logits.cross_entropy_loss::<Tensor>(
        &flat_targets,
        None,
        tch::Reduction::None,
        -100,
        0.0,
    );
    let denom = flat_mask.sum(Kind::Float).clamp_min(1.0);
    (per_pair * flat_mask).sum(Kind::Float) / denom
}

fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    if pred.numel() == 0 || target.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, pred.device()))
    } else {
        (pred - target).pow_tensor_scalar(2.0).mean(Kind::Float)
    }
}

fn atom_type_bootstrap_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    if logits.numel() == 0 || targets.numel() == 0 {
        Tensor::zeros([1], (Kind::Float, logits.device()))
    } else {
        logits.cross_entropy_for_logits(targets)
    }
}

fn masked_atom_type_loss(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0 || targets.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let atom_count = logits.size().first().copied().unwrap_or(0);
    if targets.size().first().copied().unwrap_or(-1) != atom_count
        || mask.size().first().copied().unwrap_or(-1) != atom_count
    {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }

    let per_atom =
        logits.cross_entropy_loss::<Tensor>(targets, None, tch::Reduction::None, -100, 0.0);
    weighted_mean(&per_atom, mask)
}

fn masked_coord_denoising_loss(predicted: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if predicted.numel() == 0 || target.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted.device()));
    }
    let atom_count = predicted.size().first().copied().unwrap_or(0);
    if predicted.size() != target.size() || mask.size().first().copied().unwrap_or(-1) != atom_count
    {
        return Tensor::zeros([1], (Kind::Float, predicted.device()));
    }
    let per_atom =
        (predicted - target)
            .pow_tensor_scalar(2.0)
            .mean_dim([1].as_slice(), false, Kind::Float);
    weighted_mean(&per_atom, mask)
}

fn pairwise_distance_recovery_loss(
    predicted_coords: &Tensor,
    target_pairwise_distances: &Tensor,
    mask: &Tensor,
) -> Tensor {
    if predicted_coords.numel() == 0 || target_pairwise_distances.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted_coords.device()));
    }
    let atom_count = predicted_coords.size().first().copied().unwrap_or(0);
    if target_pairwise_distances.size().as_slice() != [atom_count, atom_count]
        || mask.size().first().copied().unwrap_or(-1) != atom_count
    {
        return Tensor::zeros([1], (Kind::Float, predicted_coords.device()));
    }
    let predicted_distances = pairwise_distances(predicted_coords);
    let pair_mask = pairwise_mask(mask);
    let per_pair = (predicted_distances - target_pairwise_distances).pow_tensor_scalar(2.0);
    let denom = pair_mask.sum(Kind::Float).clamp_min(1.0);
    (per_pair * pair_mask).sum(Kind::Float) / denom
}

fn pairwise_distances(coords: &Tensor) -> Tensor {
    let diffs = coords.unsqueeze(1) - coords.unsqueeze(0);
    let squared = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float);
    let distances = squared.clamp_min(1e-12).sqrt();
    let count = coords.size().first().copied().unwrap_or(0);
    if count <= 0 {
        distances
    } else {
        let eye = Tensor::eye(count, (Kind::Float, coords.device()));
        let ones = Tensor::ones_like(&eye);
        distances * (&ones - &eye)
    }
}

fn pairwise_mask(mask: &Tensor) -> Tensor {
    let pair_mask = mask.unsqueeze(1) * mask.unsqueeze(0);
    let count = mask.size().first().copied().unwrap_or(0);
    if count <= 0 {
        pair_mask
    } else {
        let eye = Tensor::eye(count, (Kind::Float, mask.device()));
        let ones = Tensor::ones_like(&eye);
        pair_mask * (&ones - &eye)
    }
}

fn inverse_mask(mask: &Tensor) -> Tensor {
    let ones = Tensor::ones_like(mask);
    (&ones - mask).clamp(0.0, 1.0)
}

fn weighted_mean(values: &Tensor, mask: &Tensor) -> Tensor {
    let mask = mask.to_kind(Kind::Float);
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (values * mask).sum(Kind::Float) / denom
}

fn rollout_eval_recovery_metric(
    supervision: &crate::data::DecoderSupervision,
    rollout: &crate::models::GenerationRolloutRecord,
) -> Tensor {
    if rollout.steps.is_empty() {
        return Tensor::zeros([1], (Kind::Float, supervision.target_coords.device()));
    }

    let mut total = Tensor::zeros([1], (Kind::Float, supervision.target_coords.device()));
    let mut weight_sum = 0.0_f64;
    let preserved_mask = inverse_mask(&supervision.atom_corruption_mask);
    for step in &rollout.steps {
        let coords = coords_tensor(&step.coords, supervision.target_coords.device());
        let atom_types = Tensor::from_slice(&step.atom_types)
            .to_kind(Kind::Int64)
            .to_device(supervision.target_atom_types.device());
        let weight = supervision.rollout_eval_step_weight(step.step_index);
        let topology = masked_atom_match_loss(
            &atom_types,
            &supervision.target_atom_types,
            &supervision.atom_corruption_mask,
        );
        let topology_preservation =
            masked_atom_match_loss(&atom_types, &supervision.target_atom_types, &preserved_mask);
        let geometry = masked_coord_denoising_loss(
            &coords,
            &supervision.target_coords,
            &supervision.atom_corruption_mask,
        );
        let geometry_preservation =
            masked_coord_denoising_loss(&coords, &supervision.target_coords, &preserved_mask);
        let pairwise = pairwise_distance_recovery_loss(
            &coords,
            &supervision.target_pairwise_distances,
            &supervision.atom_corruption_mask,
        );
        let stop_target = if step.step_index + 1 >= supervision.rollout_steps {
            1.0
        } else {
            0.0
        };
        let stop_penalty = Tensor::from((step.stop_probability - stop_target).powi(2) as f32)
            .to_device(supervision.target_coords.device());
        let stability_penalty = Tensor::from(
            (step.mean_displacement.powi(2) + step.atom_change_fraction.powi(2)) as f32,
        )
        .to_device(supervision.target_coords.device());
        total += (topology
            + geometry
            + 0.2 * topology_preservation
            + 0.15 * geometry_preservation
            + 0.2 * pairwise
            + 0.05 * stop_penalty
            + 0.02 * stability_penalty)
            * weight;
        weight_sum += weight;
    }

    total / weight_sum.max(1.0)
}

fn rollout_eval_stop_metric(
    supervision: &crate::data::DecoderSupervision,
    rollout: &crate::models::GenerationRolloutRecord,
) -> Tensor {
    if rollout.steps.is_empty() {
        return Tensor::zeros([1], (Kind::Float, supervision.target_coords.device()));
    }

    let mut total = Tensor::zeros([1], (Kind::Float, supervision.target_coords.device()));
    let mut weight_sum = 0.0_f64;
    for step in &rollout.steps {
        let weight = supervision.rollout_eval_step_weight(step.step_index);
        let stop_target = if step.step_index + 1 >= supervision.rollout_steps {
            1.0
        } else {
            0.0
        };
        let stop_penalty = Tensor::from((step.stop_probability - stop_target).powi(2) as f32)
            .to_device(supervision.target_coords.device());
        total += stop_penalty * weight;
        weight_sum += weight;
    }

    total / weight_sum.max(1.0)
}

fn rollout_eval_pocket_anchor_metric(
    rollout: &crate::models::GenerationRolloutRecord,
    pocket_coords: &Tensor,
) -> Tensor {
    if rollout.steps.is_empty() || pocket_coords.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, pocket_coords.device()));
    }

    let mut total = Tensor::zeros([1], (Kind::Float, pocket_coords.device()));
    for step in &rollout.steps {
        let coords = coords_tensor(&step.coords, pocket_coords.device());
        total += pocket_anchor_loss(&coords, pocket_coords);
    }
    total / rollout.steps.len() as f64
}

fn masked_atom_match_loss(predicted: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if predicted.numel() == 0 || target.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, target.device()));
    }
    let atom_count = predicted.size().first().copied().unwrap_or(0);
    if target.size().first().copied().unwrap_or(-1) != atom_count
        || mask.size().first().copied().unwrap_or(-1) != atom_count
    {
        return Tensor::zeros([1], (Kind::Float, target.device()));
    }
    let mismatches = predicted
        .ne_tensor(target)
        .to_kind(Kind::Float)
        .to_device(target.device());
    weighted_mean(&mismatches, mask)
}

fn coords_tensor(coords: &[[f32; 3]], device: tch::Device) -> Tensor {
    if coords.is_empty() {
        Tensor::zeros([0, 3], (Kind::Float, device))
    } else {
        let flat = coords
            .iter()
            .flat_map(|coord| coord.iter().copied())
            .collect::<Vec<_>>();
        Tensor::from_slice(&flat)
            .reshape([coords.len() as i64, 3])
            .to_device(device)
    }
}

fn centroid_recovery_loss(predicted: &Tensor, target: &Tensor) -> Tensor {
    if predicted.numel() == 0 || target.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted.device()));
    }
    let predicted_centroid = predicted.mean_dim([0].as_slice(), false, Kind::Float);
    let target_centroid = target.mean_dim([0].as_slice(), false, Kind::Float);
    (predicted_centroid - target_centroid)
        .pow_tensor_scalar(2.0)
        .mean(Kind::Float)
}

fn pocket_anchor_loss(predicted_coords: &Tensor, pocket_coords: &Tensor) -> Tensor {
    if predicted_coords.numel() == 0 || pocket_coords.numel() == 0 {
        return Tensor::zeros([1], (Kind::Float, predicted_coords.device()));
    }
    let predicted_centroid = predicted_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_centroid = pocket_coords.mean_dim([0].as_slice(), false, Kind::Float);
    let pocket_radius = centered_radius(pocket_coords, &pocket_centroid);
    let centroid_distance = (&predicted_centroid - &pocket_centroid)
        .pow_tensor_scalar(2.0)
        .sum(Kind::Float)
        .sqrt();
    (centroid_distance
        - Tensor::from((pocket_radius * 0.85) as f32).to_device(predicted_coords.device()))
    .relu()
    .pow_tensor_scalar(2.0)
}

fn centered_radius(coords: &Tensor, centroid: &Tensor) -> f64 {
    if coords.numel() == 0 {
        return 0.0;
    }
    (coords - centroid.unsqueeze(0))
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([1].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
        .double_value(&[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::classification::masked_bce_with_logits;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn balanced_bce_upweights_rare_positive_errors() {
        let logits = Tensor::from_slice(&[-2.0_f32, 0.0, 0.0, 0.0]);
        let target = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]);
        let mask = Tensor::ones([4], (Kind::Float, Device::Cpu));

        let plain = masked_bce_with_logits(&logits, &target, &mask).double_value(&[]);
        let balanced = masked_balanced_bce_with_logits(&logits, &target, &mask).double_value(&[]);

        assert!(
            balanced > plain,
            "balanced BCE should emphasize rare positive errors"
        );
    }

    #[test]
    fn balanced_bce_falls_back_when_only_one_class_is_observed() {
        let logits = Tensor::from_slice(&[0.5_f32, -0.5]);
        let target = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([2], (Kind::Float, Device::Cpu));

        let plain = masked_bce_with_logits(&logits, &target, &mask).double_value(&[]);
        let balanced = masked_balanced_bce_with_logits(&logits, &target, &mask).double_value(&[]);

        assert!((balanced - plain).abs() < 1.0e-8);
    }

    #[test]
    fn native_graph_pressure_rewards_confident_connected_target_edges() {
        let target =
            Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).reshape([3, 3]);
        let mask = Tensor::ones([3, 3], (Kind::Float, Device::Cpu))
            - Tensor::eye(3, (Kind::Float, Device::Cpu));
        let weak_logits = Tensor::full([3, 3], -2.0, (Kind::Float, Device::Cpu));
        let confident_logits = &target * 8.0 - 4.0;
        let calibration = native_graph_pressure_test_calibration();

        let weak = calibration
            .confidence_pressure_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &weak_logits,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);
        let confident = calibration
            .confidence_pressure_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &confident_logits,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);

        assert!(
            weak > confident + 0.5,
            "weak target-edge logits should carry substantially more native graph pressure"
        );
    }

    #[test]
    fn native_graph_pressure_is_zero_without_positive_target_edges() {
        let target = Tensor::zeros([3, 3], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([3, 3], (Kind::Float, Device::Cpu))
            - Tensor::eye(3, (Kind::Float, Device::Cpu));
        let logits = Tensor::zeros([3, 3], (Kind::Float, Device::Cpu));
        let calibration = native_graph_pressure_test_calibration();

        let loss = calibration
            .confidence_pressure_loss(
                SparseTopologyCalibrationScope::Flow,
                Some(0),
                &logits,
                &target,
                &mask,
            )
            .raw
            .double_value(&[]);

        assert_eq!(loss, 0.0);
    }

    fn native_graph_pressure_test_calibration() -> SparseTopologyCalibration {
        SparseTopologyCalibration::new(SparseTopologyCalibrationConfig {
            enabled: true,
            start_step: 0,
            warmup_steps: 0,
            probe_weight: 0.0,
            flow_weight: 0.0,
            rollout_weight: 0.0,
            min_rate_scale: 0.05,
            max_raw_loss: 8.0,
            confidence_pressure_weight: 1.0,
            confidence_pressure_max_loss: 100.0,
            degree_alignment_weight: 0.0,
            probe_degree_alignment_weight: 0.0,
            degree_alignment_max_loss: 6.0,
            native_score_calibration_weight: 0.0,
            native_score_calibration_max_loss: 6.0,
            native_score_threshold: 0.55,
        })
    }
}
