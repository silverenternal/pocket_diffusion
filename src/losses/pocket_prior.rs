//! Pocket-conditioned size and composition prior losses.

use tch::{Kind, Tensor};

use crate::{
    data::MolecularExample,
    models::{PocketConditionedPriorOutput, ResearchForward},
};

/// Target tensors for pocket-conditioned prior supervision.
pub(crate) struct PocketPriorTargets {
    /// Atom-count class target with shape `[1]`.
    pub atom_count_class: Tensor,
    /// Normalized atom-type composition target with shape `[atom_vocab_size]`.
    pub composition: Tensor,
    /// Raw ligand atom count before clamping to the head support.
    pub target_atom_count: usize,
}

/// Decomposed pocket prior loss outputs.
pub(crate) struct PocketPriorAuxOutput {
    /// Combined size and composition loss.
    pub total: Tensor,
    /// Atom-count classification loss.
    pub atom_count: Tensor,
    /// Element/composition distribution loss.
    pub composition: Tensor,
    /// Mean absolute atom-count error in atom units.
    pub atom_count_mae: f64,
}

/// Auxiliary objective for explicit pocket-conditioned generation priors.
#[derive(Debug, Clone, Default)]
pub struct PocketPriorAuxLoss;

impl PocketPriorAuxLoss {
    /// Compute batch-mean size and composition prior losses.
    pub(crate) fn compute_batch(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
    ) -> PocketPriorAuxOutput {
        let device = forwards
            .first()
            .map(|forward| forward.generation.pocket_priors.atom_count_logits.device())
            .or_else(|| {
                examples
                    .first()
                    .map(|example| example.topology.atom_types.device())
            })
            .unwrap_or(tch::Device::Cpu);
        if examples.is_empty() || forwards.is_empty() {
            return zero_output(device);
        }

        let mut total = Tensor::zeros([1], (Kind::Float, device));
        let mut atom_count_total = Tensor::zeros([1], (Kind::Float, device));
        let mut composition_total = Tensor::zeros([1], (Kind::Float, device));
        let mut atom_count_error_sum = 0.0_f64;
        let mut contributing = 0_usize;

        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let output = &forward.generation.pocket_priors;
            let targets = pocket_prior_targets(example, output, device);
            if output.atom_count_logits.numel() == 0 || output.composition_logits.numel() == 0 {
                continue;
            }
            let expected_count = expected_atom_count(output);
            atom_count_error_sum += (expected_count - targets.target_atom_count as f64).abs();
            let atom_count_loss = output
                .atom_count_logits
                .unsqueeze(0)
                .cross_entropy_loss::<Tensor>(
                    &targets.atom_count_class,
                    None,
                    tch::Reduction::Mean,
                    -100,
                    0.0,
                );
            let composition_loss = -(targets.composition
                * output.composition_logits.log_softmax(-1, Kind::Float))
            .sum(Kind::Float);
            total += &atom_count_loss + &composition_loss;
            atom_count_total += atom_count_loss;
            composition_total += composition_loss;
            contributing += 1;
        }

        if contributing == 0 {
            return zero_output(device);
        }
        let scale = 1.0 / contributing as f64;
        PocketPriorAuxOutput {
            total: total * scale,
            atom_count: atom_count_total * scale,
            composition: composition_total * scale,
            atom_count_mae: atom_count_error_sum * scale,
        }
    }
}

pub(crate) fn pocket_prior_targets(
    example: &MolecularExample,
    output: &PocketConditionedPriorOutput,
    device: tch::Device,
) -> PocketPriorTargets {
    let atom_types = &example.topology.atom_types;
    let target_atom_count = atom_types.size().first().copied().unwrap_or(0).max(0) as usize;
    let atom_count_class = target_atom_count.min(output.max_atom_count) as i64;
    let vocab = output.atom_vocab_size.max(1) as usize;
    let mut counts = vec![0.0_f32; vocab];
    for index in 0..target_atom_count {
        let token = atom_types.int64_value(&[index as i64]);
        if token >= 0 {
            let bucket = (token as usize).min(vocab - 1);
            counts[bucket] += 1.0;
        }
    }
    let denom = counts.iter().sum::<f32>().max(1.0);
    for value in &mut counts {
        *value /= denom;
    }
    PocketPriorTargets {
        atom_count_class: Tensor::from_slice(&[atom_count_class]).to_device(device),
        composition: Tensor::from_slice(&counts).to_device(device),
        target_atom_count,
    }
}

fn expected_atom_count(output: &PocketConditionedPriorOutput) -> f64 {
    let probs = output.atom_count_logits.softmax(-1, Kind::Float);
    let classes = Tensor::arange(
        output
            .atom_count_logits
            .size()
            .first()
            .copied()
            .unwrap_or(0)
            .max(0),
        (Kind::Float, output.atom_count_logits.device()),
    );
    (probs * classes).sum(Kind::Float).double_value(&[])
}

fn zero_output(device: tch::Device) -> PocketPriorAuxOutput {
    let zero = Tensor::zeros([1], (Kind::Float, device));
    PocketPriorAuxOutput {
        total: zero.shallow_clone(),
        atom_count: zero.shallow_clone(),
        composition: zero,
        atom_count_mae: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::ResearchConfig, data::synthetic_phase1_examples, models::Phase1ResearchSystem,
    };
    use tch::{nn, Device};

    #[test]
    fn pocket_prior_targets_construct_count_and_composition() {
        let config = ResearchConfig::default();
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let forward = system.forward_example(&example);
        let output = &forward.generation.pocket_priors;

        let targets = pocket_prior_targets(&example, output, Device::Cpu);

        assert_eq!(
            targets.atom_count_class.int64_value(&[0]),
            example.topology.atom_types.size()[0].min(output.max_atom_count as i64)
        );
        let composition_sum = targets.composition.sum(Kind::Float).double_value(&[]);
        assert!((composition_sum - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn pocket_prior_loss_is_finite_on_synthetic_pockets() {
        let config = ResearchConfig::default();
        let example = synthetic_phase1_examples()
            .into_iter()
            .next()
            .unwrap()
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let forward = system.forward_example(&example);

        let loss = PocketPriorAuxLoss.compute_batch(&[example], &[forward]);

        assert!(loss.total.double_value(&[]).is_finite());
        assert!(loss.atom_count.double_value(&[]).is_finite());
        assert!(loss.composition.double_value(&[]).is_finite());
        assert!(loss.atom_count_mae.is_finite());
    }
}
