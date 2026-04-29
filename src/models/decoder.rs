//! Minimal decoder skeleton for conditioned ligand construction.

use tch::{nn, no_grad, Kind, Tensor};

use crate::config::{DecoderConditioningConfig, DecoderConditioningKind};

use super::{
    ConditionedGenerationState, ConditionedLigandDecoder, DecoderCapabilityDescriptor,
    DecoderOutput,
};

/// Replaceable decoder skeleton that preserves separate topology/geometry conditioning paths.
#[derive(Debug)]
pub struct ModularLigandDecoder {
    atom_embedding: nn::Embedding,
    step_projection: nn::Linear,
    conditioning_projection: nn::Linear,
    topology_projection: nn::Linear,
    geometry_projection: nn::Linear,
    topology_context_projection: nn::Linear,
    geometry_context_projection: nn::Linear,
    coordinate_context_projection: nn::Linear,
    atom_context_gate: nn::Linear,
    geometry_context_gate: nn::Linear,
    atom_query_projection: nn::Linear,
    geometry_query_projection: nn::Linear,
    topology_slot_key: nn::Linear,
    topology_slot_value: nn::Linear,
    geometry_slot_key: nn::Linear,
    geometry_slot_value: nn::Linear,
    pocket_atom_slot_key: nn::Linear,
    pocket_atom_slot_value: nn::Linear,
    pocket_geometry_slot_key: nn::Linear,
    pocket_geometry_slot_value: nn::Linear,
    topology_local_gate: nn::Linear,
    geometry_local_gate: nn::Linear,
    pocket_atom_local_gate: nn::Linear,
    pocket_geometry_local_gate: nn::Linear,
    local_atom_context_projection: nn::Linear,
    local_geometry_context_projection: nn::Linear,
    atom_type_head: nn::Linear,
    coord_delta_head: nn::Linear,
    coord_bias_head: nn::Linear,
    stop_head: nn::Linear,
    hidden_dim: i64,
    conditioning_kind: DecoderConditioningKind,
}

impl ModularLigandDecoder {
    /// Create a simple ligand decoder over modular conditioning state.
    pub fn new(vs: &nn::Path, atom_vocab_size: i64, hidden_dim: i64) -> Self {
        Self::new_with_config(
            vs,
            atom_vocab_size,
            hidden_dim,
            &DecoderConditioningConfig::default(),
        )
    }

    /// Create a decoder with explicit conditioning controls.
    pub fn new_with_config(
        vs: &nn::Path,
        atom_vocab_size: i64,
        hidden_dim: i64,
        config: &DecoderConditioningConfig,
    ) -> Self {
        let atom_embedding = nn::embedding(
            vs / "atom_embed",
            atom_vocab_size,
            hidden_dim,
            Default::default(),
        );
        let topology_projection = nn::linear(
            vs / "topology_proj",
            hidden_dim * 4,
            hidden_dim,
            Default::default(),
        );
        let conditioning_projection = nn::linear(
            vs / "conditioning_proj",
            hidden_dim * 3,
            hidden_dim,
            Default::default(),
        );
        let step_projection = nn::linear(vs / "step_proj", 1, hidden_dim, Default::default());
        let geometry_projection = nn::linear(
            vs / "geometry_proj",
            hidden_dim + 3,
            hidden_dim,
            Default::default(),
        );
        let topology_context_projection = nn::linear(
            vs / "topology_context_proj",
            hidden_dim * 3,
            hidden_dim,
            Default::default(),
        );
        let geometry_context_projection = nn::linear(
            vs / "geometry_context_proj",
            hidden_dim * 3,
            hidden_dim,
            Default::default(),
        );
        let coordinate_context_projection = nn::linear(
            vs / "coordinate_context_proj",
            hidden_dim * 3,
            hidden_dim,
            Default::default(),
        );
        let atom_context_gate = nn::linear(
            vs / "atom_context_gate",
            hidden_dim * 2,
            hidden_dim,
            Default::default(),
        );
        let geometry_context_gate = nn::linear(
            vs / "geometry_context_gate",
            hidden_dim * 2,
            hidden_dim,
            Default::default(),
        );
        let atom_query_projection = nn::linear(
            vs / "atom_query_projection",
            hidden_dim + 3,
            hidden_dim,
            Default::default(),
        );
        let geometry_query_projection = nn::linear(
            vs / "geometry_query_projection",
            hidden_dim + 3,
            hidden_dim,
            Default::default(),
        );
        let topology_slot_key = nn::linear(
            vs / "topology_slot_key",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let topology_slot_value = nn::linear(
            vs / "topology_slot_value",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let geometry_slot_key = nn::linear(
            vs / "geometry_slot_key",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let geometry_slot_value = nn::linear(
            vs / "geometry_slot_value",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let pocket_atom_slot_key = nn::linear(
            vs / "pocket_atom_slot_key",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let pocket_atom_slot_value = nn::linear(
            vs / "pocket_atom_slot_value",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let pocket_geometry_slot_key = nn::linear(
            vs / "pocket_geometry_slot_key",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let pocket_geometry_slot_value = nn::linear(
            vs / "pocket_geometry_slot_value",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );
        let mut topology_local_gate = nn::linear(
            vs / "topology_local_gate",
            hidden_dim * 2,
            1,
            Default::default(),
        );
        let mut geometry_local_gate = nn::linear(
            vs / "geometry_local_gate",
            hidden_dim * 2,
            1,
            Default::default(),
        );
        let mut pocket_atom_local_gate = nn::linear(
            vs / "pocket_atom_local_gate",
            hidden_dim * 2,
            1,
            Default::default(),
        );
        let mut pocket_geometry_local_gate = nn::linear(
            vs / "pocket_geometry_local_gate",
            hidden_dim * 2,
            1,
            Default::default(),
        );
        initialize_gate_bias(&mut topology_local_gate, config.local_gate_initial_bias);
        initialize_gate_bias(&mut geometry_local_gate, config.local_gate_initial_bias);
        initialize_gate_bias(&mut pocket_atom_local_gate, config.local_gate_initial_bias);
        initialize_gate_bias(
            &mut pocket_geometry_local_gate,
            config.local_gate_initial_bias,
        );
        let local_atom_context_projection = nn::linear(
            vs / "local_atom_context_projection",
            hidden_dim * 3,
            hidden_dim,
            Default::default(),
        );
        let local_geometry_context_projection = nn::linear(
            vs / "local_geometry_context_projection",
            hidden_dim * 3,
            hidden_dim,
            Default::default(),
        );
        let atom_type_head = nn::linear(
            vs / "atom_type_head",
            hidden_dim,
            atom_vocab_size,
            Default::default(),
        );
        let coord_delta_head =
            nn::linear(vs / "coord_delta_head", hidden_dim, 3, Default::default());
        let coord_bias_head = nn::linear(vs / "coord_bias_head", hidden_dim, 3, Default::default());
        let stop_head = nn::linear(vs / "stop_head", hidden_dim * 3, 1, Default::default());

        Self {
            atom_embedding,
            step_projection,
            conditioning_projection,
            topology_projection,
            geometry_projection,
            topology_context_projection,
            geometry_context_projection,
            coordinate_context_projection,
            atom_context_gate,
            geometry_context_gate,
            atom_query_projection,
            geometry_query_projection,
            topology_slot_key,
            topology_slot_value,
            geometry_slot_key,
            geometry_slot_value,
            pocket_atom_slot_key,
            pocket_atom_slot_value,
            pocket_geometry_slot_key,
            pocket_geometry_slot_value,
            topology_local_gate,
            geometry_local_gate,
            pocket_atom_local_gate,
            pocket_geometry_local_gate,
            local_atom_context_projection,
            local_geometry_context_projection,
            atom_type_head,
            coord_delta_head,
            coord_bias_head,
            stop_head,
            hidden_dim,
            conditioning_kind: config.kind,
        }
    }
}

impl ConditionedLigandDecoder for ModularLigandDecoder {
    fn capability(&self) -> DecoderCapabilityDescriptor {
        DecoderCapabilityDescriptor::pocket_conditioned_graph_flow()
    }

    fn decode(&self, state: &ConditionedGenerationState) -> DecoderOutput {
        let num_atoms = state.partial_ligand.atom_types.size()[0];
        let device = state.partial_ligand.atom_types.device();

        if num_atoms == 0 {
            let generation_embedding = Tensor::zeros([self.hidden_dim], (Kind::Float, device));
            return DecoderOutput {
                atom_type_logits: Tensor::zeros(
                    [0, self.atom_type_head.ws.size()[0]],
                    (Kind::Float, device),
                ),
                coordinate_deltas: Tensor::zeros([0, 3], (Kind::Float, device)),
                stop_logit: Tensor::zeros([1], (Kind::Float, device)),
                generation_embedding,
            };
        }

        let topology_summary = mean_or_zeros(&state.topology_context);
        let geometry_summary = mean_or_zeros(&state.geometry_context);
        let pocket_summary = mean_or_zeros(&state.pocket_context);
        let step_signal = Tensor::from_slice(&[state.partial_ligand.step_index as f32])
            .to_device(device)
            .unsqueeze(0)
            .apply(&self.step_projection)
            .relu()
            .squeeze_dim(0);
        let conditioning_summary =
            Tensor::cat(&[&topology_summary, &geometry_summary, &pocket_summary], 0);
        let topology_context = Tensor::cat(&[&topology_summary, &pocket_summary, &step_signal], 0)
            .unsqueeze(0)
            .apply(&self.topology_context_projection)
            .relu()
            .squeeze_dim(0);
        let geometry_context = Tensor::cat(&[&geometry_summary, &pocket_summary, &step_signal], 0)
            .unsqueeze(0)
            .apply(&self.geometry_context_projection)
            .relu()
            .squeeze_dim(0);
        let shared_conditioning = conditioning_summary
            .unsqueeze(0)
            .apply(&self.conditioning_projection)
            .relu()
            .squeeze_dim(0);
        let coordinate_context =
            Tensor::cat(&[&geometry_summary, &pocket_summary, &step_signal], 0)
                .unsqueeze(0)
                .apply(&self.coordinate_context_projection)
                .relu()
                .squeeze_dim(0);
        let topology_conditioning = (&topology_context + &shared_conditioning).unsqueeze(0);
        let geometry_conditioning = (&geometry_context + &shared_conditioning).unsqueeze(0);
        let expanded_topology_conditioning =
            topology_conditioning.expand([num_atoms, self.hidden_dim], true);
        let expanded_geometry_conditioning =
            geometry_conditioning.expand([num_atoms, self.hidden_dim], true);

        let atom_embeddings = state.partial_ligand.atom_types.apply(&self.atom_embedding);
        let topology_hidden = Tensor::cat(
            &[
                atom_embeddings.shallow_clone(),
                expanded_topology_conditioning.shallow_clone(),
                (atom_embeddings.shallow_clone() * expanded_topology_conditioning.shallow_clone()),
                (atom_embeddings.shallow_clone() - expanded_topology_conditioning.shallow_clone())
                    .abs(),
            ],
            1,
        )
        .apply(&self.topology_projection)
        .relu();
        let atom_gate = Tensor::cat(
            &[
                topology_hidden.shallow_clone(),
                expanded_topology_conditioning.shallow_clone(),
            ],
            1,
        )
        .apply(&self.atom_context_gate)
        .sigmoid();
        let one_atom_gate = Tensor::ones_like(&atom_gate);
        let mut topology_hidden = topology_hidden * (&one_atom_gate - &atom_gate)
            + expanded_topology_conditioning * atom_gate;
        let geometry_hidden = Tensor::cat(
            &[
                state.partial_ligand.coords.shallow_clone(),
                expanded_geometry_conditioning.shallow_clone(),
            ],
            1,
        )
        .apply(&self.geometry_projection)
        .relu();
        let geometry_gate = Tensor::cat(
            &[
                geometry_hidden.shallow_clone(),
                expanded_geometry_conditioning.shallow_clone(),
            ],
            1,
        )
        .apply(&self.geometry_context_gate)
        .sigmoid();
        let one_geometry_gate = Tensor::ones_like(&geometry_gate);
        let mut geometry_hidden = geometry_hidden * (&one_geometry_gate - &geometry_gate)
            + expanded_geometry_conditioning * geometry_gate;

        if self.conditioning_kind == DecoderConditioningKind::LocalAtomSlotAttention {
            let atom_query = Tensor::cat(
                &[
                    atom_embeddings.shallow_clone(),
                    state.partial_ligand.coords.shallow_clone(),
                ],
                1,
            )
            .apply(&self.atom_query_projection)
            .relu();
            let geometry_query = Tensor::cat(
                &[
                    atom_embeddings.shallow_clone(),
                    state.partial_ligand.coords.shallow_clone(),
                ],
                1,
            )
            .apply(&self.geometry_query_projection)
            .relu();
            let topology_local = gated_slot_attention(
                &atom_query,
                &state.topology_context,
                &self.topology_slot_key,
                &self.topology_slot_value,
                &self.topology_local_gate,
                Some(&state.topology_slot_mask),
                self.hidden_dim,
            );
            let pocket_atom_local = gated_slot_attention(
                &atom_query,
                &state.pocket_context,
                &self.pocket_atom_slot_key,
                &self.pocket_atom_slot_value,
                &self.pocket_atom_local_gate,
                Some(&state.pocket_slot_mask),
                self.hidden_dim,
            );
            let geometry_local = gated_slot_attention(
                &geometry_query,
                &state.geometry_context,
                &self.geometry_slot_key,
                &self.geometry_slot_value,
                &self.geometry_local_gate,
                Some(&state.geometry_slot_mask),
                self.hidden_dim,
            );
            let pocket_geometry_local = gated_slot_attention(
                &geometry_query,
                &state.pocket_context,
                &self.pocket_geometry_slot_key,
                &self.pocket_geometry_slot_value,
                &self.pocket_geometry_local_gate,
                Some(&state.pocket_slot_mask),
                self.hidden_dim,
            );
            let atom_local_context =
                Tensor::cat(&[&topology_hidden, &topology_local, &pocket_atom_local], 1)
                    .apply(&self.local_atom_context_projection)
                    .relu();
            let geometry_local_context = Tensor::cat(
                &[&geometry_hidden, &geometry_local, &pocket_geometry_local],
                1,
            )
            .apply(&self.local_geometry_context_projection)
            .relu();
            topology_hidden = topology_hidden + atom_local_context;
            geometry_hidden = geometry_hidden + geometry_local_context;
        }

        let atom_type_logits = topology_hidden.apply(&self.atom_type_head);
        let coordinate_bias = coordinate_context
            .unsqueeze(0)
            .apply(&self.coord_bias_head)
            .tanh()
            .expand([num_atoms, 3], true);
        let coordinate_deltas = geometry_hidden.apply(&self.coord_delta_head) + coordinate_bias;
        let generation_embedding =
            (topology_hidden + geometry_hidden).mean_dim([0].as_slice(), false, Kind::Float);
        let stop_logit = Tensor::cat(&[topology_context, geometry_context, pocket_summary], 0)
            .unsqueeze(0)
            .apply(&self.stop_head)
            .squeeze_dim(0);

        DecoderOutput {
            atom_type_logits,
            coordinate_deltas,
            stop_logit,
            generation_embedding,
        }
    }
}

fn initialize_gate_bias(gate: &mut nn::Linear, bias: f64) {
    if let Some(bs) = gate.bs.as_mut() {
        no_grad(|| {
            let _ = bs.fill_(bias);
        });
    }
}

fn gated_slot_attention(
    query: &Tensor,
    slots: &Tensor,
    key_projection: &nn::Linear,
    value_projection: &nn::Linear,
    gate_projection: &nn::Linear,
    slot_mask: Option<&Tensor>,
    hidden_dim: i64,
) -> Tensor {
    let num_atoms = query.size()[0];
    if slots.numel() == 0 || slots.size()[0] == 0 {
        return Tensor::zeros([num_atoms, hidden_dim], (Kind::Float, query.device()));
    }
    let keys = slots.apply(key_projection);
    let values = slots.apply(value_projection);
    let scale = (hidden_dim as f64).sqrt();
    let scores = query.matmul(&keys.transpose(0, 1)) / scale;
    let mask = decoder_source_slot_mask(slot_mask, slots.size()[0], query.device());
    let source_available = mask
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .gt(0.0)
        .to_kind(Kind::Float);
    let attended = masked_local_slot_softmax(&scores, &mask).matmul(&values);
    let gate = Tensor::cat(&[query.shallow_clone(), attended.shallow_clone()], 1)
        .apply(gate_projection)
        .sigmoid();
    attended * gate * source_available
}

fn decoder_source_slot_mask(
    slot_mask: Option<&Tensor>,
    slot_count: i64,
    device: tch::Device,
) -> Tensor {
    let Some(mask) = slot_mask else {
        return Tensor::ones([1, slot_count], (Kind::Float, device));
    };
    let mask = mask.to_device(device).to_kind(Kind::Float);
    if mask.size().as_slice() == [slot_count] {
        mask.reshape([1, slot_count])
    } else {
        Tensor::ones([1, slot_count], (Kind::Float, device))
    }
}

fn masked_local_slot_softmax(scores: &Tensor, slot_mask: &Tensor) -> Tensor {
    let mask = slot_mask.clamp(0.0, 1.0);
    let invalid = Tensor::ones_like(&mask) - &mask;
    let masked_scores = scores + invalid * -1.0e9;
    let weights = masked_scores.softmax(-1, Kind::Float) * &mask;
    let denom = weights
        .sum_dim_intlist([1].as_slice(), true, Kind::Float)
        .clamp_min(1e-6);
    weights / denom
}

fn mean_or_zeros(tokens: &Tensor) -> Tensor {
    if tokens.size()[0] == 0 {
        Tensor::zeros([tokens.size()[1]], (Kind::Float, tokens.device()))
    } else {
        tokens.mean_dim([0].as_slice(), false, Kind::Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_preserves_atom_count_and_emits_joint_state() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let decoder = ModularLigandDecoder::new(&vs.root(), 16, 8);
        let state = ConditionedGenerationState {
            example_id: "example-1".to_string(),
            protein_id: "protein-1".to_string(),
            partial_ligand: super::super::PartialLigandState {
                atom_types: Tensor::from_slice(&[0_i64, 1, 2]),
                coords: Tensor::zeros([3, 3], (Kind::Float, tch::Device::Cpu)),
                atom_mask: Tensor::ones([3], (Kind::Float, tch::Device::Cpu)),
                step_index: 0,
            },
            topology_context: Tensor::ones([4, 8], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([4, 8], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([4, 8], (Kind::Float, tch::Device::Cpu)),
            topology_slot_mask: Tensor::ones([4], (Kind::Float, tch::Device::Cpu)),
            geometry_slot_mask: Tensor::ones([4], (Kind::Float, tch::Device::Cpu)),
            pocket_slot_mask: Tensor::ones([4], (Kind::Float, tch::Device::Cpu)),
        };

        let output = decoder.decode(&state);

        assert_eq!(output.atom_type_logits.size(), vec![3, 16]);
        assert_eq!(output.coordinate_deltas.size(), vec![3, 3]);
        assert_eq!(output.stop_logit.size(), vec![1]);
        assert_eq!(output.generation_embedding.size(), vec![8]);
    }

    #[test]
    fn decoder_mean_pooled_baseline_remains_selectable() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let decoder = ModularLigandDecoder::new_with_config(
            &vs.root(),
            16,
            8,
            &DecoderConditioningConfig {
                kind: DecoderConditioningKind::MeanPooled,
                ..DecoderConditioningConfig::default()
            },
        );
        let state = ConditionedGenerationState {
            example_id: "example-1".to_string(),
            protein_id: "protein-1".to_string(),
            partial_ligand: super::super::PartialLigandState {
                atom_types: Tensor::from_slice(&[0_i64, 1, 2]),
                coords: Tensor::zeros([3, 3], (Kind::Float, tch::Device::Cpu)),
                atom_mask: Tensor::ones([3], (Kind::Float, tch::Device::Cpu)),
                step_index: 0,
            },
            topology_context: Tensor::ones([4, 8], (Kind::Float, tch::Device::Cpu)),
            geometry_context: Tensor::ones([4, 8], (Kind::Float, tch::Device::Cpu)),
            pocket_context: Tensor::ones([4, 8], (Kind::Float, tch::Device::Cpu)),
            topology_slot_mask: Tensor::ones([4], (Kind::Float, tch::Device::Cpu)),
            geometry_slot_mask: Tensor::ones([4], (Kind::Float, tch::Device::Cpu)),
            pocket_slot_mask: Tensor::ones([4], (Kind::Float, tch::Device::Cpu)),
        };

        let output = decoder.decode(&state);

        assert_eq!(output.atom_type_logits.size(), vec![3, 16]);
        assert_eq!(output.coordinate_deltas.size(), vec![3, 3]);
    }

    #[test]
    fn decoder_local_slot_softmax_masks_inactive_source_slots() {
        let scores = Tensor::from_slice(&[12.0_f32, 1.0, 10.0, 2.0, 9.0, 3.0]).reshape([3, 2]);
        let mask = Tensor::from_slice(&[0.0_f32, 1.0]).reshape([1, 2]);

        let weights = masked_local_slot_softmax(&scores, &mask);

        assert!(weights.select(1, 0).abs().max().double_value(&[]) < 1e-6);
        assert!((weights.select(1, 1).min().double_value(&[]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decoder_local_slot_softmax_all_masked_source_returns_zero_weights() {
        let scores = Tensor::from_slice(&[12.0_f32, 1.0, 10.0, 2.0, 9.0, 3.0]).reshape([3, 2]);
        let mask = Tensor::zeros([1, 2], (Kind::Float, tch::Device::Cpu));

        let weights = masked_local_slot_softmax(&scores, &mask);

        assert!(weights.abs().sum(Kind::Float).double_value(&[]) < 1e-6);
    }

    fn same_mean_slots_a() -> Tensor {
        Tensor::from_slice(&[
            4.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        .reshape([2, 8])
    }

    fn same_mean_slots_b() -> Tensor {
        Tensor::from_slice(&[
            2.0_f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        .reshape([2, 8])
    }

    fn decoder_state(slots: &Tensor) -> ConditionedGenerationState {
        ConditionedGenerationState {
            example_id: "example-1".to_string(),
            protein_id: "protein-1".to_string(),
            partial_ligand: super::super::PartialLigandState {
                atom_types: Tensor::from_slice(&[0_i64, 1, 2]),
                coords: Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.0, 0.25, 0.0, -0.5, 1.0, 0.5])
                    .reshape([3, 3]),
                atom_mask: Tensor::ones([3], (Kind::Float, tch::Device::Cpu)),
                step_index: 1,
            },
            topology_context: slots.shallow_clone(),
            geometry_context: slots.shallow_clone(),
            pocket_context: slots.shallow_clone(),
            topology_slot_mask: Tensor::ones([2], (Kind::Float, tch::Device::Cpu)),
            geometry_slot_mask: Tensor::ones([2], (Kind::Float, tch::Device::Cpu)),
            pocket_slot_mask: Tensor::ones([2], (Kind::Float, tch::Device::Cpu)),
        }
    }

    fn max_abs_delta(lhs: &Tensor, rhs: &Tensor) -> f64 {
        (lhs - rhs).abs().max().double_value(&[])
    }

    #[test]
    fn decoder_local_slot_conditioning_changes_outputs_with_controlled_mean() {
        tch::manual_seed(23);
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let decoder = ModularLigandDecoder::new_with_config(
            &vs.root(),
            16,
            8,
            &DecoderConditioningConfig {
                kind: DecoderConditioningKind::LocalAtomSlotAttention,
                local_gate_initial_bias: 6.0,
            },
        );
        let output_a = decoder.decode(&decoder_state(&same_mean_slots_a()));
        let output_b = decoder.decode(&decoder_state(&same_mean_slots_b()));

        let output_delta =
            max_abs_delta(&output_a.atom_type_logits, &output_b.atom_type_logits).max(
                max_abs_delta(&output_a.coordinate_deltas, &output_b.coordinate_deltas),
            );
        assert!(
            output_delta > 1e-6,
            "slot-local decoder path should react to slot distribution at fixed mean, got delta {output_delta}"
        );
    }

    #[test]
    fn decoder_mean_pooled_conditioning_ignores_slot_distribution_ablation() {
        tch::manual_seed(23);
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let decoder = ModularLigandDecoder::new_with_config(
            &vs.root(),
            16,
            8,
            &DecoderConditioningConfig {
                kind: DecoderConditioningKind::MeanPooled,
                ..DecoderConditioningConfig::default()
            },
        );
        let output_a = decoder.decode(&decoder_state(&same_mean_slots_a()));
        let output_b = decoder.decode(&decoder_state(&same_mean_slots_b()));

        let output_delta =
            max_abs_delta(&output_a.atom_type_logits, &output_b.atom_type_logits).max(
                max_abs_delta(&output_a.coordinate_deltas, &output_b.coordinate_deltas),
            );
        assert!(
            output_delta < 1e-6,
            "mean-pooled decoder ablation should ignore slot distribution at fixed mean, got delta {output_delta}"
        );
    }
}
