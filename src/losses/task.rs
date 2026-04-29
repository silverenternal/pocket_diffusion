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
//! - flow velocity and endpoint terms are tensor-preserving flow-head losses.

use tch::{Kind, Tensor};

use crate::{
    config::{PrimaryObjectiveConfig, TrainingConfig},
    data::MolecularExample,
    models::{ResearchForward, TaskDrivenObjective},
    training::metrics::PrimaryObjectiveComponentMetrics,
};

/// Internal decomposition result for one primary-objective evaluation.
#[derive(Debug)]
pub(crate) struct PrimaryObjectiveComputation {
    /// Primary scalar objective value.
    total: Tensor,
    /// Optional term-level decomposition for reporting.
    components: PrimaryObjectiveComponentMetrics,
}

impl PrimaryObjectiveComponentMetrics {
    fn add_assign(&mut self, other: &Self) {
        add_f64_option(&mut self.topology, other.topology);
        add_f64_option(&mut self.geometry, other.geometry);
        add_f64_option(&mut self.pocket_anchor, other.pocket_anchor);
        add_f64_option(&mut self.rollout, other.rollout);
        add_f64_option(&mut self.rollout_eval_recovery, other.rollout_eval_recovery);
        add_f64_option(
            &mut self.rollout_eval_pocket_anchor,
            other.rollout_eval_pocket_anchor,
        );
        add_f64_option(&mut self.rollout_eval_stop, other.rollout_eval_stop);
        add_f64_option(&mut self.flow_velocity, other.flow_velocity);
        add_f64_option(&mut self.flow_endpoint, other.flow_endpoint);
        add_f64_option(&mut self.flow_atom_type, other.flow_atom_type);
        add_f64_option(&mut self.flow_bond, other.flow_bond);
        add_f64_option(&mut self.flow_topology, other.flow_topology);
        add_f64_option(&mut self.flow_pocket_context, other.flow_pocket_context);
        add_f64_option(&mut self.flow_synchronization, other.flow_synchronization);
    }
}

impl PrimaryObjectiveComponentMetrics {
    fn scale(&self, factor: f64) -> Self {
        Self {
            topology: self.topology.map(|value| value * factor),
            geometry: self.geometry.map(|value| value * factor),
            pocket_anchor: self.pocket_anchor.map(|value| value * factor),
            rollout: self.rollout.map(|value| value * factor),
            rollout_eval_recovery: self.rollout_eval_recovery.map(|value| value * factor),
            rollout_eval_pocket_anchor: self.rollout_eval_pocket_anchor.map(|value| value * factor),
            rollout_eval_stop: self.rollout_eval_stop.map(|value| value * factor),
            flow_velocity: self.flow_velocity.map(|value| value * factor),
            flow_endpoint: self.flow_endpoint.map(|value| value * factor),
            flow_atom_type: self.flow_atom_type.map(|value| value * factor),
            flow_bond: self.flow_bond.map(|value| value * factor),
            flow_topology: self.flow_topology.map(|value| value * factor),
            flow_pocket_context: self.flow_pocket_context.map(|value| value * factor),
            flow_synchronization: self.flow_synchronization.map(|value| value * factor),
        }
    }
}

fn add_f64_option(target: &mut Option<f64>, value: Option<f64>) {
    match (target.as_mut(), value) {
        (Some(left), Some(right)) => *left += right,
        (None, Some(right)) => *target = Some(right),
        _ => {}
    }
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
                flow_topology: None,
                flow_pocket_context: None,
                flow_synchronization: None,
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
                flow_topology: None,
                flow_pocket_context: None,
                flow_synchronization: None,
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
        let (velocity_loss, endpoint_loss) = flow_matching_velocity_and_endpoint_loss(forward);
        let molecular = molecular_flow_losses(forward);
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
        let scaled_topology = &molecular.topology * self.weight;
        let scaled_pocket_context = &molecular.pocket_context * self.weight;
        let scaled_synchronization = &molecular.synchronization * self.weight;
        PrimaryObjectiveComputation {
            total: &scaled_velocity
                + &scaled_endpoint
                + &scaled_atom_type
                + &scaled_bond
                + &scaled_topology
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
                flow_topology: scalar_optional(&scaled_topology),
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
        let denoising = ConditionedDenoisingObjective.compute_with_components(example, forward);
        let flow = FlowMatchingObjective {
            weight: self.flow_weight,
        }
        .compute_with_components(example, forward);
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
        }),
        PrimaryObjectiveConfig::DenoisingFlowMatching => Box::new(DenoisingFlowMatchingObjective {
            denoising_weight: training.hybrid_denoising_weight,
            flow_weight: training.hybrid_flow_weight,
        }),
    }
}

/// Compute the configured primary objective and return an optional component
/// breakdown alongside the batch scalar.
pub(crate) fn compute_primary_objective_batch_with_components(
    objective: &dyn PrimaryObjectiveWithComponents,
    examples: &[MolecularExample],
    forwards: &[ResearchForward],
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
        let item = objective.compute_with_components(example, forward);
        total += item.total;
        components.add_assign(&item.components);
    }
    let scale = 1.0 / examples.len() as f64;
    (
        total * scale,
        PrimaryObjectiveComponentMetrics {
            topology: components.topology.map(|value| value * scale),
            geometry: components.geometry.map(|value| value * scale),
            pocket_anchor: components.pocket_anchor.map(|value| value * scale),
            rollout: components.rollout.map(|value| value * scale),
            rollout_eval_recovery: components.rollout_eval_recovery.map(|value| value * scale),
            rollout_eval_pocket_anchor: components
                .rollout_eval_pocket_anchor
                .map(|value| value * scale),
            rollout_eval_stop: components.rollout_eval_stop.map(|value| value * scale),
            flow_velocity: components.flow_velocity.map(|value| value * scale),
            flow_endpoint: components.flow_endpoint.map(|value| value * scale),
            flow_atom_type: components.flow_atom_type.map(|value| value * scale),
            flow_bond: components.flow_bond.map(|value| value * scale),
            flow_topology: components.flow_topology.map(|value| value * scale),
            flow_pocket_context: components.flow_pocket_context.map(|value| value * scale),
            flow_synchronization: components.flow_synchronization.map(|value| value * scale),
        },
    )
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
    topology: Tensor,
    pocket_context: Tensor,
    synchronization: Tensor,
}

fn molecular_flow_losses(forward: &ResearchForward) -> MolecularFlowLosses {
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

    let bond = if molecular.branch_weights.bond <= 0.0 {
        zero()
    } else {
        let bond_exists = masked_bce_with_logits(
            &molecular.bond_exists_logits,
            &molecular.target_adjacency,
            &molecular.pair_mask,
        );
        let positive_pair_mask = &molecular.pair_mask * molecular.target_adjacency.clamp(0.0, 1.0);
        let bond_type = masked_pair_cross_entropy(
            &molecular.bond_type_logits,
            &molecular.target_bond_types,
            &positive_pair_mask,
        );
        (bond_exists + bond_type) * molecular.branch_weights.bond
    };

    let topology = if molecular.branch_weights.topology <= 0.0 {
        zero()
    } else {
        masked_bce_with_logits(
            &molecular.topology_logits,
            &molecular.target_topology,
            &molecular.pair_mask,
        ) * molecular.branch_weights.topology
    };

    let pocket_context = if molecular.branch_weights.pocket_context <= 0.0
        || molecular.pocket_contact_logits.numel() == 0
        || molecular.target_pocket_contacts.numel() == 0
    {
        zero()
    } else {
        masked_bce_with_logits(
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
        topology,
        pocket_context,
        synchronization,
    }
}

fn zero_molecular_flow_losses(device: tch::Device) -> MolecularFlowLosses {
    MolecularFlowLosses {
        atom_type: Tensor::zeros([1], (Kind::Float, device)),
        bond: Tensor::zeros([1], (Kind::Float, device)),
        topology: Tensor::zeros([1], (Kind::Float, device)),
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

fn masked_bce_with_logits(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || logits.size() != target.size()
    {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let target = target.to_kind(Kind::Float);
    let per_item = logits.clamp_min(0.0) - logits * &target + (-logits.abs()).exp().log1p();
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (per_item * mask).sum(Kind::Float) / denom
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
