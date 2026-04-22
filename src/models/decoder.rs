//! Minimal decoder skeleton for conditioned ligand construction.

use tch::{nn, Kind, Tensor};

use super::{ConditionedGenerationState, ConditionedLigandDecoder, DecoderOutput};

/// Replaceable decoder skeleton that preserves separate topology/geometry conditioning paths.
#[derive(Debug)]
pub struct ModularLigandDecoder {
    atom_embedding: nn::Embedding,
    topology_projection: nn::Linear,
    geometry_projection: nn::Linear,
    atom_type_head: nn::Linear,
    coord_delta_head: nn::Linear,
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
        let geometry_projection = nn::linear(
            vs / "geometry_proj",
            hidden_dim * 3 + 3,
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
        let stop_head = nn::linear(vs / "stop_head", hidden_dim * 3, 1, Default::default());

        Self {
            atom_embedding,
            topology_projection,
            geometry_projection,
            atom_type_head,
            coord_delta_head,
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
        let conditioning_summary =
            Tensor::cat(&[&topology_summary, &geometry_summary, &pocket_summary], 0);
        let expanded_conditioning = conditioning_summary
            .unsqueeze(0)
            .expand([num_atoms, conditioning_summary.size()[0]], true);

        let atom_embeddings = state.partial_ligand.atom_types.apply(&self.atom_embedding);
        let topology_hidden =
            Tensor::cat(&[atom_embeddings, expanded_conditioning.shallow_clone()], 1)
                .apply(&self.topology_projection)
                .relu();
        let geometry_hidden = Tensor::cat(
            &[
                state.partial_ligand.coords.shallow_clone(),
                expanded_conditioning,
            ],
            1,
        )
        .apply(&self.geometry_projection)
        .relu();

        let atom_type_logits = topology_hidden.apply(&self.atom_type_head);
        let coordinate_deltas = geometry_hidden.apply(&self.coord_delta_head);
        let generation_embedding =
            (topology_hidden + geometry_hidden).mean_dim([0].as_slice(), false, Kind::Float);
        let stop_logit = conditioning_summary
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
