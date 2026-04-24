//! Minimal decoder skeleton for conditioned ligand construction.

use tch::{nn, Kind, Tensor};

use super::{ConditionedGenerationState, ConditionedLigandDecoder, DecoderOutput};

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
    atom_type_head: nn::Linear,
    coord_delta_head: nn::Linear,
    coord_bias_head: nn::Linear,
    stop_head: nn::Linear,
    hidden_dim: i64,
}

impl ModularLigandDecoder {
    /// Create a simple ligand decoder over modular conditioning state.
    pub fn new(vs: &nn::Path, atom_vocab_size: i64, hidden_dim: i64) -> Self {
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
            atom_type_head,
            coord_delta_head,
            coord_bias_head,
            stop_head,
            hidden_dim,
        }
    }
}

impl ConditionedLigandDecoder for ModularLigandDecoder {
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
        let topology_hidden = topology_hidden * (&one_atom_gate - &atom_gate)
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
        let geometry_hidden = geometry_hidden * (&one_geometry_gate - &geometry_gate)
            + expanded_geometry_conditioning * geometry_gate;

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
        };

        let output = decoder.decode(&state);

        assert_eq!(output.atom_type_logits.size(), vec![3, 16]);
        assert_eq!(output.coordinate_deltas.size(), vec![3, 3]);
        assert_eq!(output.stop_logit.size(), vec![1]);
        assert_eq!(output.generation_embedding.size(), vec![8]);
    }
}
