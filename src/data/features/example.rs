impl MolecularExample {
    /// Build a Phase 1 example from the legacy pocket/ligand structs.
    pub fn from_legacy(
        example_id: impl Into<String>,
        protein_id: impl Into<String>,
        ligand: &Ligand,
        pocket: &Pocket,
    ) -> Self {
        Self::from_legacy_with_targets(
            example_id,
            protein_id,
            ligand,
            pocket,
            ExampleTargets::default(),
        )
    }

    /// Build an example from legacy structs plus optional supervised targets.
    pub fn from_legacy_with_targets(
        example_id: impl Into<String>,
        protein_id: impl Into<String>,
        ligand: &Ligand,
        pocket: &Pocket,
        targets: ExampleTargets,
    ) -> Self {
        Self::from_legacy_with_targets_and_generation(
            example_id,
            protein_id,
            ligand,
            pocket,
            targets,
            &GenerationTargetConfig::default(),
        )
    }

    /// Build an example from legacy structs plus explicit decoder-supervision config.
    pub fn from_legacy_with_targets_and_generation(
        example_id: impl Into<String>,
        protein_id: impl Into<String>,
        ligand: &Ligand,
        pocket: &Pocket,
        targets: ExampleTargets,
        generation_target: &GenerationTargetConfig,
    ) -> Self {
        let topology = topology_from_ligand(ligand);
        let ligand_center = ligand_centroid(ligand);
        let geometry = geometry_from_ligand_centered(ligand, ligand_center);
        let pocket_features = pocket_features_from_pocket_centered(pocket, ligand_center);
        let example_id = example_id.into();
        let protein_id = protein_id.into();
        let decoder_supervision =
            build_decoder_supervision(&example_id, &topology, &geometry, generation_target);
        Self {
            example_id,
            protein_id,
            topology,
            geometry,
            pocket: pocket_features,
            coordinate_frame_origin: [
                ligand_center[0] as f32,
                ligand_center[1] as f32,
                ligand_center[2] as f32,
            ],
            source_pocket_path: None,
            source_ligand_path: None,
            decoder_supervision,
            targets,
        }
    }

    /// Move all modality tensors onto a specific device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            example_id: self.example_id.clone(),
            protein_id: self.protein_id.clone(),
            topology: self.topology.to_device(device),
            geometry: self.geometry.to_device(device),
            pocket: self.pocket.to_device(device),
            coordinate_frame_origin: self.coordinate_frame_origin,
            source_pocket_path: self.source_pocket_path.clone(),
            source_ligand_path: self.source_ligand_path.clone(),
            decoder_supervision: self.decoder_supervision.to_device(device),
            targets: self.targets.clone(),
        }
    }

    /// Resize pocket features to the configured model width by zero-padding or truncation.
    pub fn with_pocket_feature_dim(&self, pocket_feature_dim: i64) -> Self {
        Self {
            example_id: self.example_id.clone(),
            protein_id: self.protein_id.clone(),
            topology: self.topology.clone(),
            geometry: self.geometry.clone(),
            pocket: self.pocket.with_feature_dim(pocket_feature_dim),
            coordinate_frame_origin: self.coordinate_frame_origin,
            source_pocket_path: self.source_pocket_path.clone(),
            source_ligand_path: self.source_ligand_path.clone(),
            decoder_supervision: self.decoder_supervision.clone(),
            targets: self.targets.clone(),
        }
    }

    /// Rebuild decoder-side supervision using the configured corruption process.
    pub fn with_generation_config(&self, generation_target: &GenerationTargetConfig) -> Self {
        Self {
            example_id: self.example_id.clone(),
            protein_id: self.protein_id.clone(),
            topology: self.topology.clone(),
            geometry: self.geometry.clone(),
            pocket: self.pocket.clone(),
            coordinate_frame_origin: self.coordinate_frame_origin,
            source_pocket_path: self.source_pocket_path.clone(),
            source_ligand_path: self.source_ligand_path.clone(),
            decoder_supervision: build_decoder_supervision(
                &self.example_id,
                &self.topology,
                &self.geometry,
                generation_target,
            ),
            targets: self.targets.clone(),
        }
    }
}
