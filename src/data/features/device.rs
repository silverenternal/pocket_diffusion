impl TopologyFeatures {
    /// Move topology tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            atom_types: self.atom_types.to_device(device),
            edge_index: self.edge_index.to_device(device),
            bond_types: self.bond_types.to_device(device),
            adjacency: self.adjacency.to_device(device),
        }
    }
}

impl GeometryFeatures {
    /// Move geometry tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            coords: self.coords.to_device(device),
            pairwise_distances: self.pairwise_distances.to_device(device),
        }
    }
}

impl PocketFeatures {
    /// Move pocket tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            coords: self.coords.to_device(device),
            atom_features: self.atom_features.to_device(device),
            pooled_features: self.pooled_features.to_device(device),
        }
    }

    /// Resize the per-atom and pooled pocket features to match the configured model width.
    pub fn with_feature_dim(&self, target_dim: i64) -> Self {
        Self {
            coords: self.coords.shallow_clone(),
            atom_features: resize_feature_matrix(&self.atom_features, target_dim),
            pooled_features: resize_feature_vector(&self.pooled_features, target_dim),
        }
    }
}

impl DecoderSupervision {
    /// Move decoder supervision tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            target_atom_types: self.target_atom_types.to_device(device),
            corrupted_atom_types: self.corrupted_atom_types.to_device(device),
            atom_corruption_mask: self.atom_corruption_mask.to_device(device),
            target_coords: self.target_coords.to_device(device),
            noisy_coords: self.noisy_coords.to_device(device),
            coordinate_noise: self.coordinate_noise.to_device(device),
            target_pairwise_distances: self.target_pairwise_distances.to_device(device),
            rollout_steps: self.rollout_steps,
            training_step_weight_decay: self.training_step_weight_decay,
            corruption_metadata: self.corruption_metadata.clone(),
        }
    }

    /// Weight assigned to the provided rollout step under the configured decay schedule.
    pub fn rollout_step_weight(&self, step_index: usize) -> f64 {
        self.training_step_weight_decay.powi(step_index as i32)
    }
}
