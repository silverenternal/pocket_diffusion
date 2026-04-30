use crate::models::{
    interaction::CrossModalInteractionDiagnostics, TargetMatchingCostSummary,
};

/// Compact, scalar sync context for one forward pass across modalities.
#[allow(dead_code)] // Sync metadata is retained for reproducibility reports and audit exports.
#[derive(Debug, Clone)]
pub(crate) struct ModalitySyncContext {
    /// Stable example identifier.
    pub example_id: String,
    /// Stable protein identifier.
    pub protein_id: String,
    /// Number of active ligand topology nodes.
    pub ligand_atom_count: i64,
    /// Number of active pocket atoms.
    pub pocket_atom_count: i64,
    /// Topology modality active mask count.
    pub topology_mask_count: i64,
    /// Geometry modality active mask count.
    pub geometry_mask_count: i64,
    /// Pocket modality active mask count.
    pub pocket_mask_count: i64,
    /// Learned topology slot count.
    pub topology_slot_count: i64,
    /// Learned geometry slot count.
    pub geometry_slot_count: i64,
    /// Learned pocket slot count.
    pub pocket_slot_count: i64,
    /// Source coordinate-frame origin used for ligand/pocket coordinates.
    pub coordinate_frame_origin: [f32; 3],
    /// Device kind string for reproducibility metadata.
    pub device_kind: String,
    /// Optional flow conditioning time used during the forward path.
    pub flow_t: Option<f64>,
    /// Optional rollout step index used by the forward path.
    pub rollout_step_index: Option<usize>,
}

/// Output bundle for the separate modality encoders.
#[derive(Debug, Clone)]
pub(crate) struct EncodedModalities {
    /// Topology encoding.
    pub topology: ModalityEncoding,
    /// Geometry encoding.
    pub geometry: ModalityEncoding,
    /// Pocket/context encoding.
    pub pocket: ModalityEncoding,
}

/// Batched output bundle for the separate modality encoders.
#[derive(Debug, Clone)]
pub(crate) struct BatchedEncodedModalities {
    /// Topology encoding.
    pub topology: BatchedModalityEncoding,
    /// Geometry encoding.
    pub geometry: BatchedModalityEncoding,
    /// Pocket/context encoding.
    pub pocket: BatchedModalityEncoding,
}

/// Slot-decomposed outputs for each modality.
#[derive(Debug, Clone)]
pub(crate) struct DecomposedModalities {
    /// Topology slots.
    pub topology: SlotEncoding,
    /// Geometry slots.
    pub geometry: SlotEncoding,
    /// Pocket/context slots.
    pub pocket: SlotEncoding,
}

/// Batched slot-decomposed outputs for each modality.
#[derive(Debug, Clone)]
pub(crate) struct BatchedDecomposedModalities {
    /// Topology slots.
    pub topology: BatchedSlotEncoding,
    /// Geometry slots.
    pub geometry: BatchedSlotEncoding,
    /// Pocket/context slots.
    pub pocket: BatchedSlotEncoding,
}

/// Directed gated cross-modality interactions.
#[derive(Debug, Clone)]
pub(crate) struct CrossModalInteractions {
    /// Topology receiving information from geometry.
    pub topo_from_geo: CrossAttentionOutput,
    /// Topology receiving information from pocket context.
    pub topo_from_pocket: CrossAttentionOutput,
    /// Geometry receiving information from topology.
    pub geo_from_topo: CrossAttentionOutput,
    /// Geometry receiving information from pocket context.
    pub geo_from_pocket: CrossAttentionOutput,
    /// Pocket receiving information from topology.
    pub pocket_from_topo: CrossAttentionOutput,
    /// Pocket receiving information from geometry.
    pub pocket_from_geo: CrossAttentionOutput,
}

/// Batched directed gated cross-modality interactions.
#[derive(Debug, Clone)]
pub(crate) struct BatchedCrossModalInteractions {
    /// Topology receiving information from geometry.
    pub topo_from_geo: BatchedCrossAttentionOutput,
    /// Topology receiving information from pocket context.
    pub topo_from_pocket: BatchedCrossAttentionOutput,
    /// Geometry receiving information from topology.
    pub geo_from_topo: BatchedCrossAttentionOutput,
    /// Geometry receiving information from pocket context.
    pub geo_from_pocket: BatchedCrossAttentionOutput,
    /// Pocket receiving information from topology.
    pub pocket_from_topo: BatchedCrossAttentionOutput,
    /// Pocket receiving information from geometry.
    pub pocket_from_geo: BatchedCrossAttentionOutput,
}

/// One tensor-preserving optimizer-facing rollout step.
#[derive(Debug)]
pub(crate) struct RolloutTrainingStepRecord {
    /// Zero-based generated step index.
    pub step_index: usize,
    /// Per-atom logits emitted for this generated state.
    pub atom_type_logits: Tensor,
    /// Generated coordinates after this step.
    pub coords: Tensor,
    /// Coordinate delta applied at this step.
    pub coordinate_deltas: Tensor,
    /// Stop/update logit for this generated step.
    pub stop_logit: Tensor,
    /// Active generated atom mask.
    pub atom_mask: Tensor,
    /// Whether the state fed into the following step was detached.
    pub detached_before_next_step: bool,
}

impl Clone for RolloutTrainingStepRecord {
    fn clone(&self) -> Self {
        Self {
            step_index: self.step_index,
            atom_type_logits: self.atom_type_logits.shallow_clone(),
            coords: self.coords.shallow_clone(),
            coordinate_deltas: self.coordinate_deltas.shallow_clone(),
            stop_logit: self.stop_logit.shallow_clone(),
            atom_mask: self.atom_mask.shallow_clone(),
            detached_before_next_step: self.detached_before_next_step,
        }
    }
}

/// Bounded differentiable rollout record used only by optimizer-facing short-rollout losses.
#[derive(Debug, Clone)]
pub(crate) struct RolloutTrainingRecord {
    /// Whether config enabled trainable short-rollout construction for this forward.
    pub enabled: bool,
    /// Whether the active generation mode passed the rollout-training mode gate.
    pub mode_allowed: bool,
    /// Configured bounded step count.
    pub configured_steps: usize,
    /// Number of tensor-preserving records emitted.
    pub executed_steps: usize,
    /// Report label for detach policy.
    pub detach_policy: String,
    /// Compact memory-control note for artifacts and diagnostics.
    pub memory_control: String,
    /// Target/evidence source carried by generated-state losses.
    pub target_source: String,
    /// Per-step tensor records.
    pub steps: Vec<RolloutTrainingStepRecord>,
}

impl RolloutTrainingRecord {
    /// Disabled record used to preserve default training behavior.
    pub(crate) fn disabled(configured_steps: usize, detach_policy: String) -> Self {
        Self {
            enabled: false,
            mode_allowed: false,
            configured_steps,
            executed_steps: 0,
            detach_policy,
            memory_control: "disabled".to_string(),
            target_source: "generated_rollout_state".to_string(),
            steps: Vec::new(),
        }
    }
}

/// Decoder-facing generation bundle produced by the modular backbone.
#[derive(Debug, Clone)]
pub(crate) struct GenerationForward {
    /// Explicit generation-mode contract used for this forward path.
    pub generation_mode: GenerationModeConfig,
    /// Explicit decoder input state with separated modality conditioning.
    pub state: ConditionedGenerationState,
    /// Decoder output for the current ligand draft.
    pub decoded: DecoderOutput,
    /// Iterative rollout trace aligned with the active generation semantics.
    pub rollout: GenerationRolloutRecord,
    /// Optional tensor-preserving short-rollout record for optimizer-facing losses.
    pub rollout_training: RolloutTrainingRecord,
    /// Optional flow-matching training tuple for geometry transport.
    pub flow_matching: Option<FlowMatchingTrainingRecord>,
    /// Explicit pocket-conditioned size and composition prior predictions.
    pub pocket_priors: PocketConditionedPriorOutput,
}

/// Full Phase 2 forward-pass bundle.
#[allow(dead_code)] // Forward bundles expose report-facing diagnostics beyond core training reads.
#[derive(Debug, Clone)]
pub(crate) struct ResearchForward {
    /// Pre-decomposition modality encodings.
    pub encodings: EncodedModalities,
    /// Slot-decomposed modality encodings.
    pub slots: DecomposedModalities,
    /// Per-branch semantic diagnostics.
    pub diagnostics: SemanticDiagnosticsBundle,
    /// Directed gated interactions.
    pub interactions: CrossModalInteractions,
    /// Directed gated interaction diagnostics emitted from forward execution.
    pub interaction_diagnostics: CrossModalInteractionDiagnostics,
    /// Compact synchronization metadata shared by modalities.
    pub sync_context: ModalitySyncContext,
    /// Semantic probe predictions.
    pub probes: ProbeOutputs,
    /// Decoder-facing conditioned generation path.
    pub generation: GenerationForward,
}

/// Optimizer-facing forward record before sampled rollout artifacts are built.
#[allow(dead_code)] // Exposed for trainer/runtime refactors that avoid rollout diagnostics.
#[derive(Debug, Clone)]
pub(crate) struct OptimizerForwardRecord {
    /// Pre-decomposition modality encodings.
    pub encodings: EncodedModalities,
    /// Slot-decomposed modality encodings.
    pub slots: DecomposedModalities,
    /// Per-branch semantic diagnostics.
    pub diagnostics: SemanticDiagnosticsBundle,
    /// Directed gated interactions.
    pub interactions: CrossModalInteractions,
    /// Directed gated interaction diagnostics emitted from forward execution.
    pub interaction_diagnostics: CrossModalInteractionDiagnostics,
    /// Compact synchronization metadata shared by modalities.
    pub sync_context: ModalitySyncContext,
    /// Semantic probe predictions.
    pub probes: ProbeOutputs,
    /// Generation mode active for this record.
    pub generation_mode: GenerationModeConfig,
    /// Decoder input state with separated modality conditioning.
    pub state: ConditionedGenerationState,
    /// Decoder output for the current ligand draft.
    pub decoded: DecoderOutput,
    /// Optional flow-matching training tuple for geometry and molecular transport.
    pub flow_matching: Option<FlowMatchingTrainingRecord>,
    /// Explicit pocket-conditioned size and composition prior predictions.
    pub pocket_priors: PocketConditionedPriorOutput,
    /// Interaction context resolved during optimizer-facing forward construction.
    pub interaction_context: InteractionExecutionContext,
}

/// Compact architecture diagnostics derived from one forward pass.
#[allow(dead_code)] // Constructed by targeted diagnostics/tests and retained for external reports.
#[derive(Debug, Clone)]
pub(crate) struct ResearchForwardDiagnostics {
    /// Branch-level semantic diagnostics.
    pub semantic: SemanticDiagnosticsBundle,
    /// Directed interaction path diagnostics.
    pub interaction: CrossModalInteractionDiagnostics,
    /// Compact synchronization context shared by modality branches.
    pub sync: ModalitySyncContext,
    /// Generation and rollout health summaries.
    pub generation: ResearchForwardGenerationHealth,
}

/// Scalar generation-health probes for one forward pass.
#[allow(dead_code)] // Report-facing health fields are intentionally broader than trainer usage.
#[derive(Debug, Clone)]
pub(crate) struct ResearchForwardGenerationHealth {
    /// Configured rollout budget.
    pub configured_rollout_steps: usize,
    /// Executed rollout steps.
    pub executed_rollout_steps: usize,
    /// Whether rollout terminated before budget exhaustion.
    pub stopped_early: bool,
    /// Mean displacement from the last rollout step.
    pub last_step_mean_displacement: f64,
    /// Fractional atom changes from the last rollout step.
    pub last_step_atom_change_fraction: f64,
    /// Mean atom-change fraction across the rollout.
    pub mean_atom_change_fraction: f64,
    /// Mean coordinate step scale applied across rollout.
    pub mean_coordinate_step_scale: f64,
    /// Whether a flow-matching training tuple is present.
    pub has_flow_matching: bool,
    /// Flow-matching sample time used for the training head.
    pub flow_matching_t: Option<f64>,
    /// Norm of the target velocity used in flow-matching.
    pub flow_target_velocity_norm: Option<f64>,
    /// Norm of the predicted velocity used in flow-matching.
    pub flow_predicted_velocity_norm: Option<f64>,
}

#[allow(dead_code)] // Diagnostics bundle is a compatibility surface for audits and smoke reports.
impl ResearchForward {
    /// Build compact, forward-only architecture diagnostics without cloning large tensors.
    pub(crate) fn diagnostics_bundle(&self) -> ResearchForwardDiagnostics {
        ResearchForwardDiagnostics {
            semantic: self.diagnostics.clone(),
            interaction: self.interaction_diagnostics.clone(),
            sync: self.sync_context.clone(),
            generation: self.generation_health(),
        }
    }

    fn generation_health(&self) -> ResearchForwardGenerationHealth {
        let steps = &self.generation.rollout.steps;
        let step_count = steps.len();
        let last_step = steps.last();
        let mean_atom_change_fraction = if step_count == 0 {
            0.0
        } else {
            steps
                .iter()
                .map(|step| step.atom_change_fraction)
                .sum::<f64>()
                / step_count as f64
        };
        let mean_coordinate_step_scale = if step_count == 0 {
            0.0
        } else {
            steps
                .iter()
                .map(|step| step.coordinate_step_scale)
                .sum::<f64>()
                / step_count as f64
        };

        ResearchForwardGenerationHealth {
            configured_rollout_steps: self.generation.rollout.configured_steps,
            executed_rollout_steps: step_count,
            stopped_early: self.generation.rollout.stopped_early,
            last_step_mean_displacement: last_step
                .map(|step| step.mean_displacement)
                .unwrap_or(0.0),
            last_step_atom_change_fraction: last_step
                .map(|step| step.atom_change_fraction)
                .unwrap_or(0.0),
            mean_atom_change_fraction,
            mean_coordinate_step_scale,
            has_flow_matching: self.generation.flow_matching.is_some(),
            flow_matching_t: self
                .generation
                .flow_matching
                .as_ref()
                .map(|record| record.t),
            flow_target_velocity_norm: self.generation.flow_matching.as_ref().map(|record| {
                record
                    .target_velocity
                    .pow_tensor_scalar(2.0)
                    .sum(Kind::Float)
                    .sqrt()
                    .double_value(&[])
            }),
            flow_predicted_velocity_norm: self.generation.flow_matching.as_ref().map(|record| {
                record
                    .predicted_velocity
                    .pow_tensor_scalar(2.0)
                    .sum(Kind::Float)
                    .sqrt()
                    .double_value(&[])
            }),
        }
    }
}

/// Cached flow-matching tuple used by flow objectives.
#[derive(Debug)]
pub(crate) struct FlowMatchingTrainingRecord {
    /// Predicted velocity field at sampled path location.
    pub predicted_velocity: Tensor,
    /// Supervised target velocity `x1 - x0`.
    pub target_velocity: Tensor,
    /// Coordinates at sampled path location `x_t`.
    pub sampled_coords: Tensor,
    /// Scalar sampled timestep in `[0, 1]`.
    pub t: f64,
    /// Stable x0 source label used by this flow record.
    pub x0_source: String,
    /// Molecular flow contract version active for this record.
    pub flow_contract_version: String,
    /// Atom mask used for weighted reduction.
    pub atom_mask: Tensor,
    /// Target-row matching policy used by optimizer-facing geometry flow losses.
    pub target_matching_policy: String,
    /// Mean generated-to-target matching cost for matched atom rows.
    pub target_matching_mean_cost: f64,
    /// Fraction of generated atom rows backed by matched target supervision.
    pub target_matching_coverage: f64,
    /// Full matching cost and count summary for artifact provenance.
    pub target_matching_cost_summary: TargetMatchingCostSummary,
    /// Effective branch weights after primary branch scheduling.
    pub branch_weights: MolecularFlowBranchWeights,
    /// Optional full molecular flow branch targets and predictions.
    pub molecular: Option<MolecularFlowTrainingRecord>,
}

impl Clone for FlowMatchingTrainingRecord {
    fn clone(&self) -> Self {
        Self {
            predicted_velocity: self.predicted_velocity.shallow_clone(),
            target_velocity: self.target_velocity.shallow_clone(),
            sampled_coords: self.sampled_coords.shallow_clone(),
            t: self.t,
            x0_source: self.x0_source.clone(),
            flow_contract_version: self.flow_contract_version.clone(),
            atom_mask: self.atom_mask.shallow_clone(),
            target_matching_policy: self.target_matching_policy.clone(),
            target_matching_mean_cost: self.target_matching_mean_cost,
            target_matching_coverage: self.target_matching_coverage,
            target_matching_cost_summary: self.target_matching_cost_summary.clone(),
            branch_weights: self.branch_weights,
            molecular: self.molecular.clone(),
        }
    }
}

/// Optimizer-facing training tuple for atom, bond, topology, and pocket/context flow branches.
#[derive(Debug)]
pub(crate) struct MolecularFlowTrainingRecord {
    /// Atom-type categorical logits `[num_atoms, atom_vocab_size]`.
    pub atom_type_logits: Tensor,
    /// Target atom-type labels aligned to the generated draft atom count.
    pub target_atom_types: Tensor,
    /// Bond existence logits `[num_atoms, num_atoms]`.
    pub bond_exists_logits: Tensor,
    /// Target dense adjacency aligned to the generated draft atom count.
    pub target_adjacency: Tensor,
    /// Bond type logits `[num_atoms, num_atoms, bond_vocab_size]`.
    pub bond_type_logits: Tensor,
    /// Target dense bond-type matrix aligned to the generated draft atom count.
    pub target_bond_types: Tensor,
    /// Topology consistency logits `[num_atoms, num_atoms]`.
    pub topology_logits: Tensor,
    /// Target topology matrix aligned to the generated draft atom count.
    pub target_topology: Tensor,
    /// Pair mask excluding diagonal and inactive atoms.
    pub pair_mask: Tensor,
    /// Pocket interaction/contact logits `[num_atoms]`.
    pub pocket_contact_logits: Tensor,
    /// Binary ligand-pocket contact targets `[num_atoms]`.
    pub target_pocket_contacts: Tensor,
    /// Mask for atom-wise pocket interaction/profile losses.
    pub pocket_interaction_mask: Tensor,
    /// Claim-bearing pocket branch target family.
    pub pocket_branch_target_family: String,
    /// Role of the legacy pocket_context reconstruction output.
    pub pocket_context_reconstruction_role: String,
    /// Pocket/context reconstruction predicted by the context branch.
    pub pocket_context_reconstruction: Tensor,
    /// Detached pocket/context target.
    pub target_pocket_context: Tensor,
    /// Per-branch weights resolved from config.
    pub branch_weights: MolecularFlowBranchWeights,
    /// Target alignment policy used to build the target tensors and masks.
    pub target_alignment_policy: String,
    /// Target-row matching policy used by atom-wise molecular branch losses.
    pub target_matching_policy: String,
    /// Mean generated-to-target matching cost for matched atom rows.
    pub target_matching_mean_cost: f64,
    /// Fraction of generated atom rows backed by matched target supervision.
    pub target_matching_coverage: f64,
    /// Full matching cost and count summary for artifact provenance.
    pub target_matching_cost_summary: TargetMatchingCostSummary,
    /// Mask identifying generated atom rows backed by target supervision.
    pub target_atom_mask: Tensor,
    /// Whether all full molecular flow branches were enabled for this record.
    pub full_branch_set_enabled: bool,
}

impl Clone for MolecularFlowTrainingRecord {
    fn clone(&self) -> Self {
        Self {
            atom_type_logits: self.atom_type_logits.shallow_clone(),
            target_atom_types: self.target_atom_types.shallow_clone(),
            bond_exists_logits: self.bond_exists_logits.shallow_clone(),
            target_adjacency: self.target_adjacency.shallow_clone(),
            bond_type_logits: self.bond_type_logits.shallow_clone(),
            target_bond_types: self.target_bond_types.shallow_clone(),
            topology_logits: self.topology_logits.shallow_clone(),
            target_topology: self.target_topology.shallow_clone(),
            pair_mask: self.pair_mask.shallow_clone(),
            pocket_contact_logits: self.pocket_contact_logits.shallow_clone(),
            target_pocket_contacts: self.target_pocket_contacts.shallow_clone(),
            pocket_interaction_mask: self.pocket_interaction_mask.shallow_clone(),
            pocket_branch_target_family: self.pocket_branch_target_family.clone(),
            pocket_context_reconstruction_role: self.pocket_context_reconstruction_role.clone(),
            pocket_context_reconstruction: self.pocket_context_reconstruction.shallow_clone(),
            target_pocket_context: self.target_pocket_context.shallow_clone(),
            branch_weights: self.branch_weights,
            target_alignment_policy: self.target_alignment_policy.clone(),
            target_matching_policy: self.target_matching_policy.clone(),
            target_matching_mean_cost: self.target_matching_mean_cost,
            target_matching_coverage: self.target_matching_coverage,
            target_matching_cost_summary: self.target_matching_cost_summary.clone(),
            target_atom_mask: self.target_atom_mask.shallow_clone(),
            full_branch_set_enabled: self.full_branch_set_enabled,
        }
    }
}

/// Scalar branch weights copied into the forward record for objective evaluation.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MolecularFlowBranchWeights {
    /// Geometry velocity branch weight.
    pub geometry: f64,
    /// Atom-type branch weight.
    pub atom_type: f64,
    /// Bond branch weight.
    pub bond: f64,
    /// Topology branch weight.
    pub topology: f64,
    /// Pocket/context branch weight.
    pub pocket_context: f64,
    /// Cross-branch synchronization weight.
    pub synchronization: f64,
}
