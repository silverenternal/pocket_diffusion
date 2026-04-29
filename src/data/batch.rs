//! Batch collation with explicit masks and minimal padding.

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use tch::{Device, Kind, Tensor};

use super::MolecularExample;

/// Padded encoder conditioning inputs across all three modalities.
#[derive(Debug)]
pub struct EncoderBatchInputs {
    /// Padded atom types `[batch, max_ligand_atoms]`.
    pub atom_types: Tensor,
    /// Ligand atom mask `[batch, max_ligand_atoms]`.
    pub ligand_mask: Tensor,
    /// Padded ligand coordinates `[batch, max_ligand_atoms, 3]`.
    pub ligand_coords: Tensor,
    /// Padded ligand chemistry-role vectors `[batch, max_ligand_atoms, chemistry_role_dim]`.
    pub ligand_chemistry_roles: Tensor,
    /// Padded pairwise distances `[batch, max_ligand_atoms, max_ligand_atoms]`.
    pub pairwise_distances: Tensor,
    /// Padded dense ligand adjacency `[batch, max_ligand_atoms, max_ligand_atoms]`.
    pub adjacency: Tensor,
    /// Padded dense ligand bond-type matrix `[batch, max_ligand_atoms, max_ligand_atoms]`.
    pub bond_type_adjacency: Tensor,
    /// Padded pocket features `[batch, max_pocket_atoms, pocket_feat_dim]`.
    pub pocket_atom_features: Tensor,
    /// Padded pocket chemistry-role vectors `[batch, max_pocket_atoms, chemistry_role_dim]`.
    pub pocket_chemistry_roles: Tensor,
    /// Padded pocket coordinates `[batch, max_pocket_atoms, 3]`.
    pub pocket_coords: Tensor,
    /// Pocket atom mask `[batch, max_pocket_atoms]`.
    pub pocket_mask: Tensor,
    /// Pooled pocket features `[batch, pocket_feat_dim]`.
    pub pocket_pooled_features: Tensor,
}

impl Clone for EncoderBatchInputs {
    fn clone(&self) -> Self {
        Self {
            atom_types: self.atom_types.shallow_clone(),
            ligand_mask: self.ligand_mask.shallow_clone(),
            ligand_coords: self.ligand_coords.shallow_clone(),
            ligand_chemistry_roles: self.ligand_chemistry_roles.shallow_clone(),
            pairwise_distances: self.pairwise_distances.shallow_clone(),
            adjacency: self.adjacency.shallow_clone(),
            bond_type_adjacency: self.bond_type_adjacency.shallow_clone(),
            pocket_atom_features: self.pocket_atom_features.shallow_clone(),
            pocket_chemistry_roles: self.pocket_chemistry_roles.shallow_clone(),
            pocket_coords: self.pocket_coords.shallow_clone(),
            pocket_mask: self.pocket_mask.shallow_clone(),
            pocket_pooled_features: self.pocket_pooled_features.shallow_clone(),
        }
    }
}

/// Padded decoder-side supervision separated from encoder conditioning.
#[derive(Debug)]
pub struct DecoderBatchTargets {
    /// Clean target atom types `[batch, max_ligand_atoms]`.
    pub target_atom_types: Tensor,
    /// Corrupted decoder input atom types `[batch, max_ligand_atoms]`.
    pub corrupted_atom_types: Tensor,
    /// Corruption mask `[batch, max_ligand_atoms]`.
    pub atom_corruption_mask: Tensor,
    /// Clean target coordinates `[batch, max_ligand_atoms, 3]`.
    pub target_coords: Tensor,
    /// Noisy decoder input coordinates `[batch, max_ligand_atoms, 3]`.
    pub noisy_coords: Tensor,
    /// Coordinate noise delta `[batch, max_ligand_atoms, 3]`.
    pub coordinate_noise: Tensor,
    /// Target pairwise distances `[batch, max_ligand_atoms, max_ligand_atoms]`.
    pub target_pairwise_distances: Tensor,
}

impl Clone for DecoderBatchTargets {
    fn clone(&self) -> Self {
        Self {
            target_atom_types: self.target_atom_types.shallow_clone(),
            corrupted_atom_types: self.corrupted_atom_types.shallow_clone(),
            atom_corruption_mask: self.atom_corruption_mask.shallow_clone(),
            target_coords: self.target_coords.shallow_clone(),
            noisy_coords: self.noisy_coords.shallow_clone(),
            coordinate_noise: self.coordinate_noise.shallow_clone(),
            target_pairwise_distances: self.target_pairwise_distances.shallow_clone(),
        }
    }
}

/// A padded mini-batch across all three modalities.
#[derive(Debug)]
pub struct MolecularBatch {
    /// Example identifiers in batch order.
    pub example_ids: Vec<String>,
    /// Protein identifiers in batch order.
    pub protein_ids: Vec<String>,
    /// Encoder conditioning inputs.
    pub encoder_inputs: EncoderBatchInputs,
    /// Decoder-side generation targets.
    pub decoder_targets: DecoderBatchTargets,
}

impl Clone for MolecularBatch {
    fn clone(&self) -> Self {
        Self {
            example_ids: self.example_ids.clone(),
            protein_ids: self.protein_ids.clone(),
            encoder_inputs: self.encoder_inputs.clone(),
            decoder_targets: self.decoder_targets.clone(),
        }
    }
}

impl MolecularBatch {
    /// Collate examples into a batch using zero padding and explicit masks.
    pub fn collate(examples: &[MolecularExample]) -> Self {
        let device = examples
            .first()
            .map(|example| example.topology.atom_types.device())
            .unwrap_or(Device::Cpu);
        let batch_size = examples.len() as i64;
        let max_ligand_atoms = examples
            .iter()
            .map(|example| example.geometry.coords.size()[0])
            .max()
            .unwrap_or(0);
        let max_pocket_atoms = examples
            .iter()
            .map(|example| example.pocket.coords.size()[0])
            .max()
            .unwrap_or(0);
        let pocket_feature_dim = examples
            .iter()
            .find_map(|example| example.pocket.atom_features.size().get(1).copied())
            .unwrap_or(6);

        let atom_types = Tensor::zeros([batch_size, max_ligand_atoms], (Kind::Int64, device));
        let ligand_mask = Tensor::zeros([batch_size, max_ligand_atoms], (Kind::Float, device));
        let ligand_coords = Tensor::zeros([batch_size, max_ligand_atoms, 3], (Kind::Float, device));
        let ligand_chemistry_roles = Tensor::zeros(
            [
                batch_size,
                max_ligand_atoms,
                crate::data::features::CHEMISTRY_ROLE_FEATURE_DIM,
            ],
            (Kind::Float, device),
        );
        let pairwise_distances = Tensor::zeros(
            [batch_size, max_ligand_atoms, max_ligand_atoms],
            (Kind::Float, device),
        );
        let adjacency = Tensor::zeros(
            [batch_size, max_ligand_atoms, max_ligand_atoms],
            (Kind::Float, device),
        );
        let bond_type_adjacency = Tensor::zeros(
            [batch_size, max_ligand_atoms, max_ligand_atoms],
            (Kind::Int64, device),
        );
        let pocket_atom_features = Tensor::zeros(
            [batch_size, max_pocket_atoms, pocket_feature_dim],
            (Kind::Float, device),
        );
        let pocket_chemistry_roles = Tensor::zeros(
            [
                batch_size,
                max_pocket_atoms,
                crate::data::features::CHEMISTRY_ROLE_FEATURE_DIM,
            ],
            (Kind::Float, device),
        );
        let pocket_coords = Tensor::zeros([batch_size, max_pocket_atoms, 3], (Kind::Float, device));
        let pocket_mask = Tensor::zeros([batch_size, max_pocket_atoms], (Kind::Float, device));
        let pocket_pooled_features =
            Tensor::zeros([batch_size, pocket_feature_dim], (Kind::Float, device));
        let target_atom_types =
            Tensor::zeros([batch_size, max_ligand_atoms], (Kind::Int64, device));
        let corrupted_atom_types =
            Tensor::zeros([batch_size, max_ligand_atoms], (Kind::Int64, device));
        let atom_corruption_mask =
            Tensor::zeros([batch_size, max_ligand_atoms], (Kind::Float, device));
        let target_coords = Tensor::zeros([batch_size, max_ligand_atoms, 3], (Kind::Float, device));
        let noisy_coords = Tensor::zeros([batch_size, max_ligand_atoms, 3], (Kind::Float, device));
        let coordinate_noise =
            Tensor::zeros([batch_size, max_ligand_atoms, 3], (Kind::Float, device));
        let target_pairwise_distances = Tensor::zeros(
            [batch_size, max_ligand_atoms, max_ligand_atoms],
            (Kind::Float, device),
        );

        let mut example_ids = Vec::with_capacity(examples.len());
        let mut protein_ids = Vec::with_capacity(examples.len());

        for (batch_ix, example) in examples.iter().enumerate() {
            example_ids.push(example.example_id.clone());
            protein_ids.push(example.protein_id.clone());

            let ligand_atoms = example.topology.atom_types.size()[0];
            let pocket_atoms = example.pocket.coords.size()[0];

            if ligand_atoms > 0 {
                atom_types
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.topology.atom_types);
                let _ = ligand_mask
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .fill_(1.0);
                ligand_coords
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.geometry.coords);
                ligand_chemistry_roles
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.topology.chemistry_roles.role_vectors);
                pairwise_distances
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .narrow(1, 0, ligand_atoms)
                    .copy_(&example.geometry.pairwise_distances);
                adjacency
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .narrow(1, 0, ligand_atoms)
                    .copy_(&example.topology.adjacency);
                bond_type_adjacency
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .narrow(1, 0, ligand_atoms)
                    .copy_(&dense_bond_type_adjacency(
                        &example.topology,
                        ligand_atoms,
                        device,
                    ));
                target_atom_types
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.decoder_supervision.target_atom_types);
                corrupted_atom_types
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.decoder_supervision.corrupted_atom_types);
                atom_corruption_mask
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.decoder_supervision.atom_corruption_mask);
                target_coords
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.decoder_supervision.target_coords);
                noisy_coords
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.decoder_supervision.noisy_coords);
                coordinate_noise
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .copy_(&example.decoder_supervision.coordinate_noise);
                target_pairwise_distances
                    .get(batch_ix as i64)
                    .narrow(0, 0, ligand_atoms)
                    .narrow(1, 0, ligand_atoms)
                    .copy_(&example.decoder_supervision.target_pairwise_distances);
            }

            if pocket_atoms > 0 {
                pocket_atom_features
                    .get(batch_ix as i64)
                    .narrow(0, 0, pocket_atoms)
                    .copy_(&example.pocket.atom_features);
                pocket_chemistry_roles
                    .get(batch_ix as i64)
                    .narrow(0, 0, pocket_atoms)
                    .copy_(&example.pocket.chemistry_roles.role_vectors);
                pocket_coords
                    .get(batch_ix as i64)
                    .narrow(0, 0, pocket_atoms)
                    .copy_(&example.pocket.coords);
                let _ = pocket_mask
                    .get(batch_ix as i64)
                    .narrow(0, 0, pocket_atoms)
                    .fill_(1.0);
                pocket_pooled_features
                    .get(batch_ix as i64)
                    .copy_(&example.pocket.pooled_features);
            }
        }

        Self {
            example_ids,
            protein_ids,
            encoder_inputs: EncoderBatchInputs {
                atom_types,
                ligand_mask,
                ligand_coords,
                ligand_chemistry_roles,
                pairwise_distances,
                adjacency,
                bond_type_adjacency,
                pocket_atom_features,
                pocket_chemistry_roles,
                pocket_coords,
                pocket_mask,
                pocket_pooled_features,
            },
            decoder_targets: DecoderBatchTargets {
                target_atom_types,
                corrupted_atom_types,
                atom_corruption_mask,
                target_coords,
                noisy_coords,
                coordinate_noise,
                target_pairwise_distances,
            },
        }
    }
}

/// Deterministic sequential mini-batch iterator over borrowed examples.
pub struct ExampleBatchIter<'a> {
    examples: &'a [MolecularExample],
    batch_size: usize,
    cursor: usize,
}

impl<'a> ExampleBatchIter<'a> {
    /// Create a new iterator that yields contiguous mini-batches.
    pub fn new(examples: &'a [MolecularExample], batch_size: usize) -> Self {
        Self {
            examples,
            batch_size: batch_size.max(1),
            cursor: 0,
        }
    }
}

impl<'a> Iterator for ExampleBatchIter<'a> {
    type Item = &'a [MolecularExample];

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.examples.len() {
            return None;
        }
        let start = self.cursor;
        let end = (start + self.batch_size).min(self.examples.len());
        self.cursor = end;
        Some(&self.examples[start..end])
    }
}

/// One sampled mini-batch with explicit epoch and source-index provenance.
pub struct SampledExampleBatch<'a> {
    /// Zero-based epoch index used to produce the sample order.
    pub epoch_index: usize,
    /// Zero-based batch index within the epoch.
    pub batch_index: usize,
    /// Effective seed used to produce this epoch's order.
    pub sample_order_seed: u64,
    /// Source example indices in batch order.
    pub sample_indices: Vec<usize>,
    examples: SampledExampleBatchExamples<'a>,
}

impl<'a> SampledExampleBatch<'a> {
    /// Borrow examples in sampled order.
    pub fn examples(&self) -> &[MolecularExample] {
        match &self.examples {
            SampledExampleBatchExamples::Borrowed(examples) => examples,
            SampledExampleBatchExamples::Owned(examples) => examples,
        }
    }
}

enum SampledExampleBatchExamples<'a> {
    Borrowed(&'a [MolecularExample]),
    Owned(Vec<MolecularExample>),
}

/// Deterministic one-epoch mini-batch sampler.
pub struct ExampleBatchSampler<'a> {
    examples: &'a [MolecularExample],
    batch_size: usize,
    drop_last: bool,
    epoch_index: usize,
    sample_order_seed: u64,
    order: Vec<usize>,
    cursor: usize,
    batch_index: usize,
}

impl<'a> ExampleBatchSampler<'a> {
    /// Build a deterministic one-epoch sampler.
    pub fn new(
        examples: &'a [MolecularExample],
        batch_size: usize,
        shuffle: bool,
        sampler_seed: u64,
        drop_last: bool,
        epoch_index: usize,
    ) -> Self {
        let batch_size = batch_size.max(1);
        let sample_order_seed = sample_order_seed_for_epoch(sampler_seed, epoch_index);
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        if shuffle {
            let mut rng = StdRng::seed_from_u64(sample_order_seed);
            order.shuffle(&mut rng);
        }
        Self {
            examples,
            batch_size,
            drop_last,
            epoch_index,
            sample_order_seed,
            order,
            cursor: 0,
            batch_index: 0,
        }
    }

    /// Number of mini-batches that one epoch will emit.
    pub fn batches_per_epoch(example_count: usize, batch_size: usize, drop_last: bool) -> usize {
        let batch_size = batch_size.max(1);
        if drop_last {
            example_count / batch_size
        } else {
            example_count.div_ceil(batch_size)
        }
    }
}

impl<'a> Iterator for ExampleBatchSampler<'a> {
    type Item = SampledExampleBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.order.len() {
            return None;
        }
        let remaining = self.order.len() - self.cursor;
        if self.drop_last && remaining < self.batch_size {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.order.len());
        let sample_indices = self.order[self.cursor..end].to_vec();
        self.cursor = end;

        let examples = if let Some((start, end)) = contiguous_bounds(&sample_indices) {
            SampledExampleBatchExamples::Borrowed(&self.examples[start..end])
        } else {
            SampledExampleBatchExamples::Owned(
                sample_indices
                    .iter()
                    .map(|index| self.examples[*index].clone())
                    .collect(),
            )
        };
        let batch = SampledExampleBatch {
            epoch_index: self.epoch_index,
            batch_index: self.batch_index,
            sample_order_seed: self.sample_order_seed,
            sample_indices,
            examples,
        };
        self.batch_index += 1;
        Some(batch)
    }
}

/// Effective sampler seed for one epoch.
pub fn sample_order_seed_for_epoch(sampler_seed: u64, epoch_index: usize) -> u64 {
    sampler_seed ^ ((epoch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn contiguous_bounds(indices: &[usize]) -> Option<(usize, usize)> {
    let start = *indices.first()?;
    indices
        .iter()
        .enumerate()
        .all(|(offset, index)| *index == start + offset)
        .then_some((start, start + indices.len()))
}

fn dense_bond_type_adjacency(
    topology: &super::TopologyFeatures,
    ligand_atoms: i64,
    device: Device,
) -> Tensor {
    let dense = Tensor::zeros([ligand_atoms, ligand_atoms], (Kind::Int64, device));
    if ligand_atoms <= 0 || topology.edge_index.size().len() != 2 {
        return dense;
    }
    let edge_count = topology
        .edge_index
        .size()
        .get(1)
        .copied()
        .unwrap_or(0)
        .min(topology.bond_types.size().first().copied().unwrap_or(0));
    for edge_ix in 0..edge_count {
        let src = topology.edge_index.int64_value(&[0, edge_ix]);
        let dst = topology.edge_index.int64_value(&[1, edge_ix]);
        let bond_type = topology.bond_types.int64_value(&[edge_ix]).clamp(0, 7);
        if src < 0 || dst < 0 || src >= ligand_atoms || dst >= ligand_atoms || bond_type == 0 {
            continue;
        }
        let _ = dense.get(src).get(dst).fill_(bond_type);
        let _ = dense.get(dst).get(src).fill_(bond_type);
    }
    dense
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::synthetic_phase1_examples;

    #[test]
    fn batch_iter_respects_batch_size() {
        let examples = synthetic_phase1_examples();
        let batches: Vec<&[MolecularExample]> = ExampleBatchIter::new(&examples, 3).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn batch_iter_sampler_preserves_contiguous_order_by_default() {
        let examples = synthetic_phase1_examples();
        let batches = ExampleBatchSampler::new(&examples, 3, false, 99, false, 0)
            .map(|batch| {
                let example_count = batch.examples().len();
                (
                    batch.epoch_index,
                    batch.batch_index,
                    batch.sample_order_seed,
                    batch.sample_indices,
                    example_count,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], (0, 0, 99, vec![0, 1, 2], 3));
        assert_eq!(batches[1], (0, 1, 99, vec![3], 1));
    }

    #[test]
    fn batch_iter_sampler_reproducibility_replays_seeded_shuffle() {
        let mut examples = synthetic_phase1_examples();
        examples.extend(synthetic_phase1_examples());
        let first = sampled_index_batches(&examples, 3, true, 17, false, 0);
        let replay = sampled_index_batches(&examples, 3, true, 17, false, 0);
        let changed = (18..28)
            .map(|seed| sampled_index_batches(&examples, 3, true, seed, false, 0))
            .any(|order| order != first);

        assert_eq!(first, replay);
        assert!(changed);
        assert_eq!(
            sorted_indices(&first),
            (0..examples.len()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn batch_iter_sampler_reproducibility_changes_order_across_epochs() {
        let mut examples = synthetic_phase1_examples();
        examples.extend(synthetic_phase1_examples());
        let epoch0 = sampled_index_batches(&examples, 4, true, 23, false, 0);
        let epoch0_replay = sampled_index_batches(&examples, 4, true, 23, false, 0);
        let epoch1 = sampled_index_batches(&examples, 4, true, 23, false, 1);

        assert_eq!(epoch0, epoch0_replay);
        assert_ne!(epoch0, epoch1);
        assert_eq!(
            sorted_indices(&epoch1),
            (0..examples.len()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn batch_iter_sampler_drop_last_discards_short_tail() {
        let examples = synthetic_phase1_examples();
        let batches = sampled_index_batches(&examples, 3, false, 0, true, 0);

        assert_eq!(batches, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn collate_preserves_chemistry_role_vectors() {
        let examples = synthetic_phase1_examples();
        let batch = MolecularBatch::collate(&examples[..2]);
        let ligand_atoms = examples[0].topology.atom_types.size()[0];
        let pocket_atoms = examples[0].pocket.coords.size()[0];

        assert_eq!(
            batch.encoder_inputs.ligand_chemistry_roles.size()[2],
            crate::data::features::CHEMISTRY_ROLE_FEATURE_DIM
        );
        assert_eq!(
            batch.encoder_inputs.pocket_chemistry_roles.size()[2],
            crate::data::features::CHEMISTRY_ROLE_FEATURE_DIM
        );
        assert!(
            batch
                .encoder_inputs
                .ligand_chemistry_roles
                .get(0)
                .narrow(0, 0, ligand_atoms)
                .isfinite()
                .all()
                .to_kind(Kind::Int64)
                .int64_value(&[])
                != 0
        );
        assert!(
            batch
                .encoder_inputs
                .pocket_chemistry_roles
                .get(0)
                .narrow(0, 0, pocket_atoms)
                .isfinite()
                .all()
                .to_kind(Kind::Int64)
                .int64_value(&[])
                != 0
        );
    }

    #[test]
    fn collate_preserves_dense_bond_type_adjacency() {
        let examples = synthetic_phase1_examples();
        let batch = MolecularBatch::collate(&examples[..1]);

        assert_eq!(batch.encoder_inputs.bond_type_adjacency.size()[0], 1);
        assert_eq!(
            batch
                .encoder_inputs
                .bond_type_adjacency
                .get(0)
                .get(0)
                .get(1)
                .int64_value(&[]),
            examples[0].topology.bond_types.int64_value(&[0])
        );
        assert_eq!(
            batch
                .encoder_inputs
                .bond_type_adjacency
                .get(0)
                .get(1)
                .get(0)
                .int64_value(&[]),
            examples[0].topology.bond_types.int64_value(&[0])
        );
    }

    fn sampled_index_batches(
        examples: &[MolecularExample],
        batch_size: usize,
        shuffle: bool,
        sampler_seed: u64,
        drop_last: bool,
        epoch_index: usize,
    ) -> Vec<Vec<usize>> {
        ExampleBatchSampler::new(
            examples,
            batch_size,
            shuffle,
            sampler_seed,
            drop_last,
            epoch_index,
        )
        .map(|batch| batch.sample_indices)
        .collect()
    }

    fn sorted_indices(batches: &[Vec<usize>]) -> Vec<usize> {
        let mut indices = batches
            .iter()
            .flat_map(|batch| batch.iter().copied())
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices
    }
}
