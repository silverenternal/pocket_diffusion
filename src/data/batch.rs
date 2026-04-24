//! Batch collation with explicit masks and minimal padding.

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
    /// Padded pairwise distances `[batch, max_ligand_atoms, max_ligand_atoms]`.
    pub pairwise_distances: Tensor,
    /// Padded dense ligand adjacency `[batch, max_ligand_atoms, max_ligand_atoms]`.
    pub adjacency: Tensor,
    /// Padded pocket features `[batch, max_pocket_atoms, pocket_feat_dim]`.
    pub pocket_atom_features: Tensor,
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
            pairwise_distances: self.pairwise_distances.shallow_clone(),
            adjacency: self.adjacency.shallow_clone(),
            pocket_atom_features: self.pocket_atom_features.shallow_clone(),
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
        let pairwise_distances = Tensor::zeros(
            [batch_size, max_ligand_atoms, max_ligand_atoms],
            (Kind::Float, device),
        );
        let adjacency = Tensor::zeros(
            [batch_size, max_ligand_atoms, max_ligand_atoms],
            (Kind::Float, device),
        );
        let pocket_atom_features = Tensor::zeros(
            [batch_size, max_pocket_atoms, pocket_feature_dim],
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
                pairwise_distances,
                adjacency,
                pocket_atom_features,
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
}
