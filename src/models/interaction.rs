//! Cross-modal interaction block for the three semantic branches.
//!
//! This module owns the six directed gated attention paths and keeps explicit,
//! configurable ablation points for each path.

use tch::{nn, Kind, Tensor};

use super::{
    BatchedCrossAttentionOutput, CrossAttentionOutput, CrossModalInteractions,
    DecomposedModalities, GatedCrossAttention,
};
use crate::config::TemporalInteractionPolicyConfig;
use crate::data::{MolecularBatch, MolecularExample};
use crate::models::system::{
    BatchedCrossModalInteractions, BatchedDecomposedModalities as SystemBatchedDecomposedModalities,
};

mod bias;
mod diagnostics;
mod path;

pub(crate) use bias::{
    ligand_pocket_slot_attention_bias_with_scale, LigandPocketSlotAttentionBias,
};
pub(crate) use diagnostics::{
    attach_topology_pocket_pharmacophore_path_diagnostics, interaction_path_diagnostics,
    interaction_path_diagnostics_batched, BatchedCrossModalInteractionDiagnostics,
    CrossModalInteractionDiagnostics, CrossModalInteractionPathDiagnostics,
};
pub(crate) use path::{
    InteractionDiagnosticProvenance, InteractionExecutionContext, InteractionPath,
};

use path::path_scale;

#[cfg(test)]
use path::SUPPORTED_INTERACTION_PATHS;

#[derive(Debug)]
pub struct CrossModalInteractionBlock {
    topo_from_geo: GatedCrossAttention,
    topo_from_pocket: GatedCrossAttention,
    geo_from_topo: GatedCrossAttention,
    geo_from_pocket: GatedCrossAttention,
    pocket_from_topo: GatedCrossAttention,
    pocket_from_geo: GatedCrossAttention,
    geometry_attention_bias_scale: f64,
    chemistry_role_attention_bias_scale: f64,
    disabled_paths: Vec<String>,
    temporal_interaction_policy: TemporalInteractionPolicyConfig,
}

impl CrossModalInteractionBlock {
    /// Construct all directed interaction paths.
    pub fn new(vs: &nn::Path, config: &crate::config::ResearchConfig) -> Self {
        let hidden_dim = config.model.hidden_dim;
        let mode = config.model.interaction_mode;
        let ff_multiplier = config.model.interaction_ff_multiplier;
        let tuning = &config.model.interaction_tuning;
        Self {
            topo_from_geo: GatedCrossAttention::new(
                &(vs / "topo_from_geo"),
                hidden_dim,
                mode,
                ff_multiplier,
                tuning.clone(),
            ),
            topo_from_pocket: GatedCrossAttention::new(
                &(vs / "topo_from_pocket"),
                hidden_dim,
                mode,
                ff_multiplier,
                tuning.clone(),
            ),
            geo_from_topo: GatedCrossAttention::new(
                &(vs / "geo_from_topo"),
                hidden_dim,
                mode,
                ff_multiplier,
                tuning.clone(),
            ),
            geo_from_pocket: GatedCrossAttention::new(
                &(vs / "geo_from_pocket"),
                hidden_dim,
                mode,
                ff_multiplier,
                tuning.clone(),
            ),
            pocket_from_topo: GatedCrossAttention::new(
                &(vs / "pocket_from_topo"),
                hidden_dim,
                mode,
                ff_multiplier,
                tuning.clone(),
            ),
            pocket_from_geo: GatedCrossAttention::new(
                &(vs / "pocket_from_geo"),
                hidden_dim,
                mode,
                ff_multiplier,
                tuning.clone(),
            ),
            geometry_attention_bias_scale: tuning.geometry_attention_bias_scale,
            chemistry_role_attention_bias_scale: tuning.chemistry_role_attention_bias_scale,
            disabled_paths: tuning.disabled_paths.clone(),
            temporal_interaction_policy: config.model.temporal_interaction_policy.clone(),
        }
    }

    #[allow(dead_code)] // Compatibility wrapper for ablations that do not need diagnostics.
    pub(crate) fn forward(&self, slots: &DecomposedModalities) -> CrossModalInteractions {
        self.forward_with_diagnostics(slots).0
    }

    #[allow(dead_code)] // Compatibility wrapper for callers that use default execution context.
    pub(crate) fn forward_with_diagnostics(
        &self,
        slots: &DecomposedModalities,
    ) -> (CrossModalInteractions, CrossModalInteractionDiagnostics) {
        self.forward_with_diagnostics_with_context(slots, InteractionExecutionContext::default())
    }

    pub(crate) fn forward_with_diagnostics_with_context(
        &self,
        slots: &DecomposedModalities,
        context: InteractionExecutionContext,
    ) -> (CrossModalInteractions, CrossModalInteractionDiagnostics) {
        self.forward_with_optional_ligand_pocket_bias(slots, context, None)
    }

    pub(crate) fn forward_example_with_diagnostics_with_context(
        &self,
        example: &MolecularExample,
        slots: &DecomposedModalities,
        context: InteractionExecutionContext,
    ) -> (CrossModalInteractions, CrossModalInteractionDiagnostics) {
        let ligand_pocket_bias = self.ligand_pocket_bias_for_example(example, slots);
        self.forward_with_optional_ligand_pocket_bias(slots, context, Some(&ligand_pocket_bias))
    }

    fn forward_with_optional_ligand_pocket_bias(
        &self,
        slots: &DecomposedModalities,
        context: InteractionExecutionContext,
        ligand_pocket_bias: Option<&LigandPocketSlotAttentionBias>,
    ) -> (CrossModalInteractions, CrossModalInteractionDiagnostics) {
        let geo_from_pocket_bias = ligand_pocket_bias.map(|bias| bias.values.shallow_clone());
        let pocket_from_geo_bias = geo_from_pocket_bias
            .as_ref()
            .map(|bias| bias.transpose(0, 1));
        let chemistry_role_coverage = ligand_pocket_bias.map(|bias| bias.chemistry_role_coverage);

        let topo_from_geo = self.maybe_forward_path(
            InteractionPath::TopologyFromGeometry,
            || self.topo_from_geo.forward(&slots.topology, &slots.geometry),
            || zero_cross_attention_output(&slots.topology.slots, &slots.geometry.slots),
            &context,
        );
        let topo_from_pocket = self.maybe_forward_path(
            InteractionPath::TopologyFromPocket,
            || {
                self.topo_from_pocket
                    .forward(&slots.topology, &slots.pocket)
            },
            || zero_cross_attention_output(&slots.topology.slots, &slots.pocket.slots),
            &context,
        );
        let geo_from_topo = self.maybe_forward_path(
            InteractionPath::GeometryFromTopology,
            || self.geo_from_topo.forward(&slots.geometry, &slots.topology),
            || zero_cross_attention_output(&slots.geometry.slots, &slots.topology.slots),
            &context,
        );
        let geo_from_pocket = self.maybe_forward_path(
            InteractionPath::GeometryFromPocket,
            || {
                self.geo_from_pocket.forward_with_attention_bias(
                    &slots.geometry,
                    &slots.pocket,
                    geo_from_pocket_bias.as_ref(),
                )
            },
            || zero_cross_attention_output(&slots.geometry.slots, &slots.pocket.slots),
            &context,
        );
        let pocket_from_topo = self.maybe_forward_path(
            InteractionPath::PocketFromTopology,
            || {
                self.pocket_from_topo
                    .forward(&slots.pocket, &slots.topology)
            },
            || zero_cross_attention_output(&slots.pocket.slots, &slots.topology.slots),
            &context,
        );
        let pocket_from_geo = self.maybe_forward_path(
            InteractionPath::PocketFromGeometry,
            || {
                self.pocket_from_geo.forward_with_attention_bias(
                    &slots.pocket,
                    &slots.geometry,
                    pocket_from_geo_bias.as_ref(),
                )
            },
            || zero_cross_attention_output(&slots.pocket.slots, &slots.geometry.slots),
            &context,
        );

        let interactions = CrossModalInteractions {
            topo_from_geo,
            topo_from_pocket,
            geo_from_topo,
            geo_from_pocket,
            pocket_from_topo,
            pocket_from_geo,
        };

        let diagnostics = CrossModalInteractionDiagnostics {
            topo_from_geo: interaction_path_diagnostics(
                InteractionPath::TopologyFromGeometry,
                &interactions.topo_from_geo,
                self.path_scale_for(InteractionPath::TopologyFromGeometry, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::PerExample,
            ),
            topo_from_pocket: interaction_path_diagnostics(
                InteractionPath::TopologyFromPocket,
                &interactions.topo_from_pocket,
                self.path_scale_for(InteractionPath::TopologyFromPocket, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::PerExample,
            ),
            geo_from_topo: interaction_path_diagnostics(
                InteractionPath::GeometryFromTopology,
                &interactions.geo_from_topo,
                self.path_scale_for(InteractionPath::GeometryFromTopology, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::PerExample,
            ),
            geo_from_pocket: interaction_path_diagnostics(
                InteractionPath::GeometryFromPocket,
                &interactions.geo_from_pocket,
                self.path_scale_for(InteractionPath::GeometryFromPocket, &context),
                geo_from_pocket_bias.as_ref().map(|bias| {
                    (
                        bias.shallow_clone(),
                        self.geometry_attention_bias_scale,
                        chemistry_role_coverage,
                    )
                }),
                &context,
                InteractionDiagnosticProvenance::PerExample,
            ),
            pocket_from_topo: interaction_path_diagnostics(
                InteractionPath::PocketFromTopology,
                &interactions.pocket_from_topo,
                self.path_scale_for(InteractionPath::PocketFromTopology, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::PerExample,
            ),
            pocket_from_geo: interaction_path_diagnostics(
                InteractionPath::PocketFromGeometry,
                &interactions.pocket_from_geo,
                self.path_scale_for(InteractionPath::PocketFromGeometry, &context),
                pocket_from_geo_bias.as_ref().map(|bias| {
                    (
                        bias.shallow_clone(),
                        self.geometry_attention_bias_scale,
                        chemistry_role_coverage,
                    )
                }),
                &context,
                InteractionDiagnosticProvenance::PerExample,
            ),
        };

        (interactions, diagnostics)
    }

    #[allow(dead_code)] // Compatibility wrapper for batched ablations without diagnostics.
    pub(crate) fn forward_batch(
        &self,
        batch: &MolecularBatch,
        slots: &SystemBatchedDecomposedModalities,
    ) -> BatchedCrossModalInteractions {
        self.forward_batch_with_diagnostics(batch, slots).0
    }

    #[allow(dead_code)] // Compatibility wrapper for batched callers using default context.
    pub(crate) fn forward_batch_with_diagnostics(
        &self,
        batch: &MolecularBatch,
        slots: &SystemBatchedDecomposedModalities,
    ) -> (
        BatchedCrossModalInteractions,
        BatchedCrossModalInteractionDiagnostics,
    ) {
        self.forward_batch_with_diagnostics_with_context(
            batch,
            slots,
            InteractionExecutionContext::default(),
        )
    }

    pub(crate) fn forward_batch_with_diagnostics_with_context(
        &self,
        batch: &MolecularBatch,
        slots: &SystemBatchedDecomposedModalities,
        context: InteractionExecutionContext,
    ) -> (
        BatchedCrossModalInteractions,
        BatchedCrossModalInteractionDiagnostics,
    ) {
        let geo_from_pocket_bias = self.ligand_pocket_bias_for_batch(batch, slots);
        let pocket_from_geo_bias = geo_from_pocket_bias.values.transpose(1, 2);

        let topo_from_geo = self.maybe_forward_batch_path(
            InteractionPath::TopologyFromGeometry,
            || {
                self.topo_from_geo
                    .forward_batch(&slots.topology, &slots.geometry, None)
            },
            || zero_batched_cross_attention_output(&slots.topology.slots, &slots.geometry.slots),
            &context,
        );
        let topo_from_pocket = self.maybe_forward_batch_path(
            InteractionPath::TopologyFromPocket,
            || {
                self.topo_from_pocket
                    .forward_batch(&slots.topology, &slots.pocket, None)
            },
            || zero_batched_cross_attention_output(&slots.topology.slots, &slots.pocket.slots),
            &context,
        );
        let geo_from_topo = self.maybe_forward_batch_path(
            InteractionPath::GeometryFromTopology,
            || {
                self.geo_from_topo
                    .forward_batch(&slots.geometry, &slots.topology, None)
            },
            || zero_batched_cross_attention_output(&slots.geometry.slots, &slots.topology.slots),
            &context,
        );
        let geo_from_pocket = self.maybe_forward_batch_path(
            InteractionPath::GeometryFromPocket,
            || {
                self.geo_from_pocket.forward_batch(
                    &slots.geometry,
                    &slots.pocket,
                    Some(&geo_from_pocket_bias.values),
                )
            },
            || zero_batched_cross_attention_output(&slots.geometry.slots, &slots.pocket.slots),
            &context,
        );
        let pocket_from_topo = self.maybe_forward_batch_path(
            InteractionPath::PocketFromTopology,
            || {
                self.pocket_from_topo
                    .forward_batch(&slots.pocket, &slots.topology, None)
            },
            || zero_batched_cross_attention_output(&slots.pocket.slots, &slots.topology.slots),
            &context,
        );
        let pocket_from_geo = self.maybe_forward_batch_path(
            InteractionPath::PocketFromGeometry,
            || {
                self.pocket_from_geo.forward_batch(
                    &slots.pocket,
                    &slots.geometry,
                    Some(&pocket_from_geo_bias),
                )
            },
            || zero_batched_cross_attention_output(&slots.pocket.slots, &slots.geometry.slots),
            &context,
        );

        let interactions = BatchedCrossModalInteractions {
            topo_from_geo,
            topo_from_pocket,
            geo_from_topo,
            geo_from_pocket,
            pocket_from_topo,
            pocket_from_geo,
        };

        let mut diagnostics = BatchedCrossModalInteractionDiagnostics {
            topo_from_geo: interaction_path_diagnostics_batched(
                InteractionPath::TopologyFromGeometry,
                &interactions.topo_from_geo,
                self.path_scale_for(InteractionPath::TopologyFromGeometry, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::BatchAggregate,
            ),
            topo_from_pocket: interaction_path_diagnostics_batched(
                InteractionPath::TopologyFromPocket,
                &interactions.topo_from_pocket,
                self.path_scale_for(InteractionPath::TopologyFromPocket, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::BatchAggregate,
            ),
            geo_from_topo: interaction_path_diagnostics_batched(
                InteractionPath::GeometryFromTopology,
                &interactions.geo_from_topo,
                self.path_scale_for(InteractionPath::GeometryFromTopology, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::BatchAggregate,
            ),
            geo_from_pocket: interaction_path_diagnostics_batched(
                InteractionPath::GeometryFromPocket,
                &interactions.geo_from_pocket,
                self.path_scale_for(InteractionPath::GeometryFromPocket, &context),
                Some((
                    geo_from_pocket_bias.values.shallow_clone(),
                    self.geometry_attention_bias_scale,
                    Some(geo_from_pocket_bias.chemistry_role_coverage),
                )),
                &context,
                InteractionDiagnosticProvenance::BatchAggregate,
            ),
            pocket_from_topo: interaction_path_diagnostics_batched(
                InteractionPath::PocketFromTopology,
                &interactions.pocket_from_topo,
                self.path_scale_for(InteractionPath::PocketFromTopology, &context),
                None,
                &context,
                InteractionDiagnosticProvenance::BatchAggregate,
            ),
            pocket_from_geo: interaction_path_diagnostics_batched(
                InteractionPath::PocketFromGeometry,
                &interactions.pocket_from_geo,
                self.path_scale_for(InteractionPath::PocketFromGeometry, &context),
                Some((
                    pocket_from_geo_bias.shallow_clone(),
                    self.geometry_attention_bias_scale,
                    Some(geo_from_pocket_bias.chemistry_role_coverage),
                )),
                &context,
                InteractionDiagnosticProvenance::BatchAggregate,
            ),
        };
        attach_topology_pocket_pharmacophore_path_diagnostics(
            &mut diagnostics.topo_from_pocket,
            &mut diagnostics.pocket_from_topo,
            &batch.encoder_inputs.ligand_chemistry_roles,
            &slots.topology.slot_weights,
            &batch.encoder_inputs.pocket_chemistry_roles,
            &slots.pocket.slot_weights,
            &interactions.topo_from_pocket.attention_weights,
            &interactions.pocket_from_topo.attention_weights,
        );

        (interactions, diagnostics)
    }

    fn path_enabled(&self, path: InteractionPath) -> bool {
        !self.disabled_paths.iter().any(|name| name == path.as_str())
    }

    pub(crate) fn uses_flow_time_conditioning(&self) -> bool {
        self.temporal_interaction_policy
            .uses_flow_time_conditioning()
    }

    fn maybe_forward_path<F, G>(
        &self,
        path: InteractionPath,
        if_enabled: F,
        if_disabled: G,
        context: &InteractionExecutionContext,
    ) -> CrossAttentionOutput
    where
        F: FnOnce() -> CrossAttentionOutput,
        G: FnOnce() -> CrossAttentionOutput,
    {
        let scale = self.path_scale_for(path, context);
        if !self.path_enabled(path) || scale <= 0.0 {
            if_disabled()
        } else {
            maybe_scale_attention_output(if_enabled(), scale)
        }
    }

    fn maybe_forward_batch_path<F, G>(
        &self,
        path: InteractionPath,
        if_enabled: F,
        if_disabled: G,
        context: &InteractionExecutionContext,
    ) -> BatchedCrossAttentionOutput
    where
        F: FnOnce() -> BatchedCrossAttentionOutput,
        G: FnOnce() -> BatchedCrossAttentionOutput,
    {
        let scale = self.path_scale_for(path, context);
        if !self.path_enabled(path) || scale <= 0.0 {
            if_disabled()
        } else {
            maybe_scale_attention_output_batch(if_enabled(), scale)
        }
    }

    pub(crate) fn path_scale_for(
        &self,
        path: InteractionPath,
        context: &InteractionExecutionContext,
    ) -> f64 {
        path_scale(&self.temporal_interaction_policy, path, context)
    }

    pub(crate) fn geometry_attention_bias_scale(&self) -> f64 {
        self.geometry_attention_bias_scale
    }

    pub(crate) fn ligand_pocket_bias_for_example(
        &self,
        example: &MolecularExample,
        slots: &DecomposedModalities,
    ) -> LigandPocketSlotAttentionBias {
        let ligand_atoms = example.geometry.coords.size().first().copied().unwrap_or(0);
        let pocket_atoms = example.pocket.coords.size().first().copied().unwrap_or(0);
        let device = example.geometry.coords.device();
        let ligand_mask = Tensor::ones([1, ligand_atoms], (Kind::Float, device));
        let pocket_mask = Tensor::ones([1, pocket_atoms], (Kind::Float, device));
        let ligand_slot_weights = slots.geometry.slot_weights.unsqueeze(0);
        let pocket_slot_weights = slots.pocket.slot_weights.unsqueeze(0);
        let mut bias = ligand_pocket_slot_attention_bias_with_scale(
            &example.geometry.coords.unsqueeze(0),
            &ligand_mask,
            &example.pocket.coords.unsqueeze(0),
            &pocket_mask,
            Some(&example.topology.chemistry_roles.role_vectors.unsqueeze(0)),
            Some(&example.pocket.chemistry_roles.role_vectors.unsqueeze(0)),
            Some(&ligand_slot_weights),
            Some(&pocket_slot_weights),
            slots.geometry.slots.size()[0],
            slots.pocket.slots.size()[0],
            self.chemistry_role_attention_bias_scale,
        );
        bias.values *= self.geometry_attention_bias_scale;
        bias.values = bias.values.squeeze_dim(0);
        bias
    }

    pub(crate) fn ligand_pocket_bias_for_batch(
        &self,
        batch: &MolecularBatch,
        slots: &SystemBatchedDecomposedModalities,
    ) -> LigandPocketSlotAttentionBias {
        let mut bias = ligand_pocket_slot_attention_bias_with_scale(
            &batch.encoder_inputs.ligand_coords,
            &batch.encoder_inputs.ligand_mask,
            &batch.encoder_inputs.pocket_coords,
            &batch.encoder_inputs.pocket_mask,
            Some(&batch.encoder_inputs.ligand_chemistry_roles),
            Some(&batch.encoder_inputs.pocket_chemistry_roles),
            Some(&slots.geometry.slot_weights),
            Some(&slots.pocket.slot_weights),
            slots.geometry.slots.size()[1],
            slots.pocket.slots.size()[1],
            self.chemistry_role_attention_bias_scale,
        );
        bias.values *= self.geometry_attention_bias_scale;
        bias
    }
}

fn maybe_scale_attention_output(output: CrossAttentionOutput, scale: f64) -> CrossAttentionOutput {
    CrossAttentionOutput {
        gate: output.gate,
        forced_open: output.forced_open,
        attended_tokens: output.attended_tokens * scale,
        attention_weights: output.attention_weights,
    }
}

fn maybe_scale_attention_output_batch(
    output: BatchedCrossAttentionOutput,
    scale: f64,
) -> BatchedCrossAttentionOutput {
    BatchedCrossAttentionOutput {
        gate: output.gate,
        forced_open: output.forced_open,
        attended_tokens: output.attended_tokens * scale,
        attention_weights: output.attention_weights,
    }
}

fn zero_cross_attention_output(query_slots: &Tensor, key_slots: &Tensor) -> CrossAttentionOutput {
    CrossAttentionOutput {
        gate: Tensor::zeros([1], (Kind::Float, query_slots.device())),
        forced_open: false,
        attended_tokens: Tensor::zeros(
            [query_slots.size()[0], query_slots.size()[1]],
            (Kind::Float, query_slots.device()),
        ),
        attention_weights: Tensor::zeros(
            [query_slots.size()[0], key_slots.size()[0]],
            (Kind::Float, query_slots.device()),
        ),
    }
}

fn zero_batched_cross_attention_output(
    query_slots: &Tensor,
    key_slots: &Tensor,
) -> BatchedCrossAttentionOutput {
    let batch = query_slots.size()[0];
    let query_len = query_slots.size()[1];
    let key_len = key_slots.size()[1];
    let hidden_dim = query_slots.size()[2];
    BatchedCrossAttentionOutput {
        gate: Tensor::zeros([batch, 1], (Kind::Float, query_slots.device())),
        forced_open: false,
        attended_tokens: Tensor::zeros(
            [batch, query_len, hidden_dim],
            (Kind::Float, query_slots.device()),
        ),
        attention_weights: Tensor::zeros(
            [batch, query_len, key_len],
            (Kind::Float, query_slots.device()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{
            InteractionPathFlowTimeBucketMultiplier, InteractionPathStageMultiplier, ResearchConfig,
        },
        data::{synthetic_phase1_examples, MolecularBatch},
        models::system::BatchedDecomposedModalities,
        models::{GeometrySemanticBranch, PocketSemanticBranch, TopologySemanticBranch},
    };
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn cross_modal_interaction_block_keeps_all_directed_paths_explicit() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };
        let interactions = block.forward(&slots);

        assert_eq!(
            interactions.topo_from_geo.gate.numel(),
            slots.topology.active_slot_mask.numel()
        );
        assert_eq!(
            interactions.topo_from_pocket.gate.numel(),
            slots.topology.active_slot_mask.numel()
        );
        assert_eq!(
            interactions.geo_from_topo.gate.numel(),
            slots.geometry.active_slot_mask.numel()
        );
        assert_eq!(
            interactions.geo_from_pocket.gate.numel(),
            slots.geometry.active_slot_mask.numel()
        );
        assert_eq!(
            interactions.pocket_from_topo.gate.numel(),
            slots.pocket.active_slot_mask.numel()
        );
        assert_eq!(
            interactions.pocket_from_geo.gate.numel(),
            slots.pocket.active_slot_mask.numel()
        );
        assert_eq!(
            interactions.geo_from_pocket.attended_tokens.size()[1],
            config.model.hidden_dim
        );

        let batch = MolecularBatch::collate(&[example]);
        let batched_slots = BatchedDecomposedModalities {
            topology: topology.decompose_batch(&topology.encode_batch(
                &batch.encoder_inputs.atom_types,
                &batch.encoder_inputs.adjacency,
                &batch.encoder_inputs.bond_type_adjacency,
                &batch.encoder_inputs.ligand_mask,
            )),
            geometry: geometry.decompose_batch(&geometry.encode_batch(
                &batch.encoder_inputs.ligand_coords,
                &batch.encoder_inputs.pairwise_distances,
                &batch.encoder_inputs.ligand_mask,
            )),
            pocket: pocket.decompose_batch(&pocket.encode_batch(
                &batch.encoder_inputs.pocket_atom_features,
                &batch.encoder_inputs.pocket_coords,
                &batch.encoder_inputs.pocket_pooled_features,
                &batch.encoder_inputs.pocket_mask,
            )),
        };
        let batched = block.forward_batch(&batch, &batched_slots);
        assert_eq!(batched.geo_from_pocket.gate.size()[0], 1);
        assert_eq!(batched.pocket_from_geo.gate.size()[0], 1);
    }

    #[test]
    fn cross_modal_interaction_block_reports_all_path_diagnostics() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let batch = MolecularBatch::collate(&[example]);
        let batched_slots = BatchedDecomposedModalities {
            topology: topology.decompose_batch(&topology.encode_batch(
                &batch.encoder_inputs.atom_types,
                &batch.encoder_inputs.adjacency,
                &batch.encoder_inputs.bond_type_adjacency,
                &batch.encoder_inputs.ligand_mask,
            )),
            geometry: geometry.decompose_batch(&geometry.encode_batch(
                &batch.encoder_inputs.ligand_coords,
                &batch.encoder_inputs.pairwise_distances,
                &batch.encoder_inputs.ligand_mask,
            )),
            pocket: pocket.decompose_batch(&pocket.encode_batch(
                &batch.encoder_inputs.pocket_atom_features,
                &batch.encoder_inputs.pocket_coords,
                &batch.encoder_inputs.pocket_pooled_features,
                &batch.encoder_inputs.pocket_mask,
            )),
        };
        let (_, diagnostics) = block.forward_batch_with_diagnostics(&batch, &batched_slots);

        let path_names = [
            &diagnostics.topo_from_geo.path_name,
            &diagnostics.topo_from_pocket.path_name,
            &diagnostics.geo_from_topo.path_name,
            &diagnostics.geo_from_pocket.path_name,
            &diagnostics.pocket_from_topo.path_name,
            &diagnostics.pocket_from_geo.path_name,
        ];
        for (expected, observed) in SUPPORTED_INTERACTION_PATHS.iter().zip(path_names.iter()) {
            assert_eq!(expected, observed);
        }
        assert!(diagnostics.topo_from_geo.gate_mean.is_finite());
        assert!(diagnostics.topo_from_geo.attended_norm >= 0.0);
        assert!(diagnostics.topo_from_geo.bias_mean.is_none());
        assert!(diagnostics.geo_from_pocket.bias_mean.is_some());
        assert!(diagnostics.geo_from_pocket.bias_min.unwrap().is_finite());
        assert!(diagnostics.pocket_from_geo.bias_scale.is_some());
        assert!(diagnostics.geo_from_pocket.bias_scale.unwrap() >= 0.0);
        assert!(diagnostics
            .geo_from_pocket
            .chemistry_role_coverage
            .is_some());
        assert!(diagnostics
            .geo_from_pocket
            .chemistry_role_coverage
            .unwrap()
            .is_finite());
        for path in [&diagnostics.topo_from_pocket, &diagnostics.pocket_from_topo] {
            assert_eq!(
                path.pharmacophore_role_provenance.as_deref(),
                Some("heuristic")
            );
            let coverage = path.pharmacophore_role_coverage.unwrap();
            let conflict_rate = path.pharmacophore_role_conflict_rate.unwrap();
            assert!(coverage.is_finite());
            assert!(conflict_rate.is_finite());
            assert!((0.0..=1.0).contains(&coverage));
            assert!((0.0..=1.0).contains(&conflict_rate));
        }
    }

    #[test]
    fn geometry_bias_diagnostics_are_finite_when_masks_are_empty() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let mut batch = MolecularBatch::collate(&[example]);
        batch.encoder_inputs.ligand_mask = Tensor::zeros_like(&batch.encoder_inputs.ligand_mask);
        batch.encoder_inputs.pocket_mask = Tensor::zeros_like(&batch.encoder_inputs.pocket_mask);

        let batched_slots = BatchedDecomposedModalities {
            topology: topology.decompose_batch(&topology.encode_batch(
                &batch.encoder_inputs.atom_types,
                &batch.encoder_inputs.adjacency,
                &batch.encoder_inputs.bond_type_adjacency,
                &batch.encoder_inputs.ligand_mask,
            )),
            geometry: geometry.decompose_batch(&geometry.encode_batch(
                &batch.encoder_inputs.ligand_coords,
                &batch.encoder_inputs.pairwise_distances,
                &batch.encoder_inputs.ligand_mask,
            )),
            pocket: pocket.decompose_batch(&pocket.encode_batch(
                &batch.encoder_inputs.pocket_atom_features,
                &batch.encoder_inputs.pocket_coords,
                &batch.encoder_inputs.pocket_pooled_features,
                &batch.encoder_inputs.pocket_mask,
            )),
        };

        let (_, diagnostics) = block.forward_batch_with_diagnostics(&batch, &batched_slots);
        assert!(diagnostics.geo_from_pocket.bias_mean.unwrap().is_finite());
        assert!(diagnostics.geo_from_pocket.bias_min.unwrap().is_finite());
        assert!(diagnostics.geo_from_pocket.bias_max.unwrap().is_finite());
        assert!(diagnostics
            .geo_from_pocket
            .chemistry_role_coverage
            .unwrap()
            .is_finite());
        assert!(diagnostics.pocket_from_geo.bias_mean.unwrap().is_finite());
        assert!(diagnostics.pocket_from_geo.bias_min.unwrap().is_finite());
        assert!(diagnostics.pocket_from_geo.bias_max.unwrap().is_finite());
        assert!(diagnostics
            .pocket_from_geo
            .chemistry_role_coverage
            .unwrap()
            .is_finite());
    }

    #[test]
    fn chemistry_role_bias_ablation_falls_back_to_distance_contact_bias() {
        let ligand_coords =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 3.0, 0.0, 0.0]).reshape([1, 2, 3]);
        let pocket_coords =
            Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 6.0, 0.0, 0.0]).reshape([1, 2, 3]);
        let ligand_mask = Tensor::ones([1, 2], (Kind::Float, Device::Cpu));
        let pocket_mask = Tensor::ones([1, 2], (Kind::Float, Device::Cpu));
        let ligand_roles = Tensor::from_slice(&[
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
        ])
        .reshape([1, 2, 9]);
        let pocket_roles = Tensor::from_slice(&[
            0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
        ])
        .reshape([1, 2, 9]);
        let ligand_slot_weights = Tensor::from_slice(&[0.8_f32, 0.2]).reshape([1, 2]);
        let pocket_slot_weights = Tensor::from_slice(&[0.3_f32, 0.7]).reshape([1, 2]);

        let distance_only = ligand_pocket_slot_attention_bias_with_scale(
            &ligand_coords,
            &ligand_mask,
            &pocket_coords,
            &pocket_mask,
            None,
            None,
            None,
            None,
            2,
            2,
            0.0,
        );
        let chemistry_disabled = ligand_pocket_slot_attention_bias_with_scale(
            &ligand_coords,
            &ligand_mask,
            &pocket_coords,
            &pocket_mask,
            Some(&ligand_roles),
            Some(&pocket_roles),
            Some(&ligand_slot_weights),
            Some(&pocket_slot_weights),
            2,
            2,
            0.0,
        );
        let chemistry_enabled = ligand_pocket_slot_attention_bias_with_scale(
            &ligand_coords,
            &ligand_mask,
            &pocket_coords,
            &pocket_mask,
            Some(&ligand_roles),
            Some(&pocket_roles),
            Some(&ligand_slot_weights),
            Some(&pocket_slot_weights),
            2,
            2,
            1.0,
        );

        let ablation_delta = (&distance_only.values - &chemistry_disabled.values)
            .abs()
            .max()
            .double_value(&[]);
        let role_delta = (&distance_only.values - &chemistry_enabled.values)
            .abs()
            .max()
            .double_value(&[]);

        assert!(ablation_delta < 1e-8);
        assert!(role_delta > 1e-6);
        assert!(chemistry_enabled.chemistry_role_coverage > 0.0);
    }

    #[test]
    fn interaction_path_mask_zeroes_only_one_disabled_path() {
        let mut config = ResearchConfig::default();
        config.model.interaction_tuning.disabled_paths = vec!["geo_from_pocket".to_string()];
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };
        let (interactions, diagnostics) = block.forward_with_diagnostics(&slots);

        assert!(
            interactions
                .geo_from_pocket
                .gate
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                == 0.0
        );
        assert!(
            interactions
                .geo_from_pocket
                .attended_tokens
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                == 0.0
        );
        assert_eq!(
            diagnostics.geo_from_pocket.gate_mean, 0.0,
            "disabled paths should have zero gate diagnostic"
        );
        assert_eq!(diagnostics.geo_from_pocket.gate_status, "always_closed");
        assert_eq!(diagnostics.geo_from_pocket.gate_closed_fraction, 1.0);
        assert!(diagnostics.geo_from_pocket.gate_warning.is_some());
        assert!(
            interactions
                .topo_from_geo
                .attended_tokens
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                >= 0.0
        );
    }

    #[test]
    fn interaction_path_mask_keeps_other_paths_non_zero() {
        let mut config = ResearchConfig::default();
        config.model.interaction_tuning.disabled_paths = vec!["geo_from_pocket".to_string()];
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };
        let (interactions, _) = block.forward_with_diagnostics(&slots);

        assert!(
            interactions
                .topo_from_geo
                .gate
                .abs()
                .sum(Kind::Float)
                .double_value(&[])
                >= 0.0
        );
    }

    #[test]
    fn interaction_temporal_policy_early_stage_can_zero_targeted_paths() {
        let mut config = ResearchConfig::default();
        config.model.temporal_interaction_policy.stage_multipliers =
            vec![InteractionPathStageMultiplier {
                training_stage: 0,
                path: "geo_from_pocket".to_string(),
                multiplier: 0.0,
            }];
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };
        let (_, diagnostics_baseline) = block.forward_with_diagnostics(&slots);
        let (_, diagnostics_early) = block.forward_with_diagnostics_with_context(
            &slots,
            InteractionExecutionContext {
                training_stage: Some(0),
                rollout_step_index: None,
                flow_t: None,
                ..Default::default()
            },
        );

        assert_eq!(
            diagnostics_baseline.topo_from_geo.provenance,
            diagnostics_early.topo_from_geo.provenance
        );
        assert_eq!(
            diagnostics_baseline.geo_from_topo.provenance,
            diagnostics_early.geo_from_topo.provenance
        );
        assert_eq!(
            diagnostics_baseline.pocket_from_topo.provenance,
            diagnostics_early.pocket_from_topo.provenance
        );
        assert_eq!(
            diagnostics_baseline.pocket_from_geo.provenance,
            diagnostics_early.pocket_from_geo.provenance
        );
        assert!(
            (diagnostics_baseline.topo_from_geo.gate_mean
                - diagnostics_early.topo_from_geo.gate_mean)
                .abs()
                < 1e-8
        );
        assert!(
            (diagnostics_baseline.geo_from_topo.gate_mean
                - diagnostics_early.geo_from_topo.gate_mean)
                .abs()
                < 1e-8
        );
        assert!(
            (diagnostics_baseline.pocket_from_topo.gate_mean
                - diagnostics_early.pocket_from_topo.gate_mean)
                .abs()
                < 1e-8
        );
        assert!(
            (diagnostics_baseline.pocket_from_geo.gate_mean
                - diagnostics_early.pocket_from_geo.gate_mean)
                .abs()
                < 1e-8
        );
        assert!(
            diagnostics_early.geo_from_pocket.gate_mean.abs() < 1e-12,
            "targeted path should be suppressed by temporal policy"
        );
        assert_eq!(diagnostics_early.geo_from_pocket.path_scale, 0.0);
        assert_eq!(diagnostics_early.geo_from_pocket.effective_update_norm, 0.0);
    }

    #[test]
    fn flow_time_context_without_policy_preserves_gate_means() {
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };

        let (_, baseline) = block.forward_with_diagnostics(&slots);
        let (_, with_flow_t) = block.forward_with_diagnostics_with_context(
            &slots,
            InteractionExecutionContext {
                training_stage: None,
                rollout_step_index: None,
                flow_t: Some(0.2),
                ..Default::default()
            },
        );

        assert_eq!(
            with_flow_t.geo_from_pocket.flow_time_bucket.as_deref(),
            Some("low")
        );
        assert!(
            (baseline.geo_from_pocket.gate_mean - with_flow_t.geo_from_pocket.gate_mean).abs()
                < 1e-8
        );
        assert!(
            (baseline.pocket_from_geo.gate_mean - with_flow_t.pocket_from_geo.gate_mean).abs()
                < 1e-8
        );
    }

    #[test]
    fn flow_time_policy_scales_geometry_pocket_update_without_scaling_gate() {
        let mut config = ResearchConfig::default();
        config
            .model
            .temporal_interaction_policy
            .flow_time_bucket_multipliers = vec![InteractionPathFlowTimeBucketMultiplier {
            path: "geo_from_pocket".to_string(),
            start_t: 0.0,
            end_t: 1.0 / 3.0,
            multiplier: 0.5,
        }];
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };

        let (_, baseline) = block.forward_with_diagnostics(&slots);
        let (_, low_flow) = block.forward_with_diagnostics_with_context(
            &slots,
            InteractionExecutionContext {
                training_stage: None,
                rollout_step_index: None,
                flow_t: Some(0.2),
                ..Default::default()
            },
        );

        assert_eq!(
            low_flow.geo_from_pocket.flow_time_bucket.as_deref(),
            Some("low")
        );
        assert_eq!(low_flow.geo_from_pocket.path_scale, 0.5);
        assert!(
            (low_flow.geo_from_pocket.gate_mean - baseline.geo_from_pocket.gate_mean).abs() < 1e-8
        );
        assert!(
            (low_flow.geo_from_pocket.effective_update_norm
                - baseline.geo_from_pocket.effective_update_norm * 0.5)
                .abs()
                < 1e-6
        );
        assert!((low_flow.topo_from_geo.gate_mean - baseline.topo_from_geo.gate_mean).abs() < 1e-8);
    }

    #[test]
    fn temporal_path_scale_above_one_does_not_amplify_gate_value() {
        let mut config = ResearchConfig::default();
        config
            .model
            .temporal_interaction_policy
            .flow_time_bucket_multipliers = vec![InteractionPathFlowTimeBucketMultiplier {
            path: "geo_from_pocket".to_string(),
            start_t: 0.0,
            end_t: 1.0 / 3.0,
            multiplier: 2.0,
        }];
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };

        let (_, baseline) = block.forward_with_diagnostics(&slots);
        let (_, high_flow) = block.forward_with_diagnostics_with_context(
            &slots,
            InteractionExecutionContext {
                training_stage: None,
                rollout_step_index: None,
                flow_t: Some(0.2),
                ..Default::default()
            },
        );

        assert_eq!(high_flow.geo_from_pocket.path_scale, 2.0);
        assert!((0.0..=1.0).contains(&high_flow.geo_from_pocket.gate_mean));
        assert!(
            (high_flow.geo_from_pocket.gate_mean - baseline.geo_from_pocket.gate_mean).abs() < 1e-8
        );
        assert!(
            (high_flow.geo_from_pocket.effective_update_norm
                - baseline.geo_from_pocket.effective_update_norm * 2.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn interaction_path_roles_match_stable_chemistry_interpretation() {
        let expected_roles = [
            ("topo_from_geo", "topology-informed bond plausibility"),
            (
                "topo_from_pocket",
                "pocket-informed ligand chemistry preference",
            ),
            ("geo_from_topo", "topology-constrained conformer geometry"),
            ("geo_from_pocket", "pocket-shaped pose refinement"),
            ("pocket_from_topo", "ligand-chemistry pocket compatibility"),
            ("pocket_from_geo", "pose occupancy feedback"),
        ];
        let config = ResearchConfig::default();
        let var_store = nn::VarStore::new(Device::Cpu);
        let root = var_store.root();
        let topology = TopologySemanticBranch::new(
            &(root.clone() / "topology_branch"),
            config.model.atom_vocab_size,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let geometry = GeometrySemanticBranch::new(
            &(root.clone() / "geometry_branch"),
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let pocket = PocketSemanticBranch::new(
            &(root.clone() / "pocket_branch"),
            config.model.pocket_feature_dim,
            config.model.hidden_dim,
            config.model.num_slots,
        );
        let block = CrossModalInteractionBlock::new(&root, &config);
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);

        let slots = DecomposedModalities {
            topology: topology.decompose(&topology.encode(&example.topology)),
            geometry: geometry.decompose(&geometry.encode(&example.geometry)),
            pocket: pocket.decompose(&pocket.encode(&example.pocket)),
        };
        let (_, diagnostics) = block.forward_with_diagnostics(&slots);

        let observed_roles = [
            (
                &diagnostics.topo_from_geo.path_name[..],
                diagnostics.topo_from_geo.path_role,
            ),
            (
                &diagnostics.topo_from_pocket.path_name[..],
                diagnostics.topo_from_pocket.path_role,
            ),
            (
                &diagnostics.geo_from_topo.path_name[..],
                diagnostics.geo_from_topo.path_role,
            ),
            (
                &diagnostics.geo_from_pocket.path_name[..],
                diagnostics.geo_from_pocket.path_role,
            ),
            (
                &diagnostics.pocket_from_topo.path_name[..],
                diagnostics.pocket_from_topo.path_role,
            ),
            (
                &diagnostics.pocket_from_geo.path_name[..],
                diagnostics.pocket_from_geo.path_role,
            ),
        ];

        for (observed_name, observed_role) in &observed_roles {
            let expected_entry = expected_roles
                .iter()
                .find(|(name, _)| name == observed_name)
                .expect("diagnostic path should be one of known paths");
            assert_eq!(observed_role.to_string(), expected_entry.1.to_string());
        }
    }
}
