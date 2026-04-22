//! Batch collation with explicit masks and minimal padding.

use tch::{Kind, Tensor};

use super::MolecularExample;

/// A padded mini-batch across all three modalities.
#[derive(Debug)]
pub struct MolecularBatch {
    /// Example identifiers in batch order.
    pub example_ids: Vec<String>,
    /// Protein identifiers in batch order.
    pub protein_ids: Vec<String>,
    /// Padded atom types `[batch, max_ligand_atoms]`.
    pub atom_types: Tensor,
    /// Ligand atom mask `[batch, max_ligand_atoms]`.
    pub ligand_mask: Tensor,
    /// Padded ligand coordinates `[batch, max_ligand_atoms, 3]`.
    pub ligand_coords: Tensor,
    /// Padded pairwise distances `[batch, max_ligand_atoms, max_ligand_atoms]`.
    pub pairwise_distances: Tensor,
    /// Padded pocket features `[batch, max_pocket_atoms, pocket_feat_dim]`.
    pub pocket_atom_features: Tensor,
    /// Padded pocket coordinates `[batch, max_pocket_atoms, 3]`.
    pub pocket_coords: Tensor,
    /// Pocket atom mask `[batch, max_pocket_atoms]`.
    pub pocket_mask: Tensor,
}

impl Clone for MolecularBatch {
    fn clone(&self) -> Self {
        Self {
            example_ids: self.example_ids.clone(),
            protein_ids: self.protein_ids.clone(),
            atom_types: self.atom_types.shallow_clone(),
            ligand_mask: self.ligand_mask.shallow_clone(),
            ligand_coords: self.ligand_coords.shallow_clone(),
            pairwise_distances: self.pairwise_distances.shallow_clone(),
            pocket_atom_features: self.pocket_atom_features.shallow_clone(),
            pocket_coords: self.pocket_coords.shallow_clone(),
            pocket_mask: self.pocket_mask.shallow_clone(),
        }
    }
}

impl MolecularBatch {
    /// Collate examples into a batch using zero padding and explicit masks.
    pub fn collate(examples: &[MolecularExample]) -> Self {
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

        let atom_types = Tensor::zeros(
            [batch_size, max_ligand_atoms],
            (Kind::Int64, tch::Device::Cpu),
        );
        let ligand_mask = Tensor::zeros(
            [batch_size, max_ligand_atoms],
            (Kind::Float, tch::Device::Cpu),
        );
        let ligand_coords = Tensor::zeros(
            [batch_size, max_ligand_atoms, 3],
            (Kind::Float, tch::Device::Cpu),
        );
        let pairwise_distances = Tensor::zeros(
            [batch_size, max_ligand_atoms, max_ligand_atoms],
            (Kind::Float, tch::Device::Cpu),
        );
        let pocket_atom_features = Tensor::zeros(
            [batch_size, max_pocket_atoms, pocket_feature_dim],
            (Kind::Float, tch::Device::Cpu),
        );
        let pocket_coords = Tensor::zeros(
            [batch_size, max_pocket_atoms, 3],
            (Kind::Float, tch::Device::Cpu),
        );
        let pocket_mask = Tensor::zeros(
            [batch_size, max_pocket_atoms],
            (Kind::Float, tch::Device::Cpu),
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
            }
        }

        Self {
            example_ids,
            protein_ids,
            atom_types,
            ligand_mask,
            ligand_coords,
            pairwise_distances,
            pocket_atom_features,
            pocket_coords,
            pocket_mask,
        }
    }
}
