/// Discover a PDBbind-like dataset where each subdirectory contains one PDB and one SDF.
pub fn discover_pdbbind_like_entries(
    root: &Path,
    parsing_mode: ParsingMode,
) -> Result<Vec<ManifestEntry>, DataParseError> {
    let mut entries = Vec::new();

    for dir_entry in fs::read_dir(root)? {
        let dir_entry = dir_entry?;
        let path = dir_entry.path();
        if !path.is_dir() {
            continue;
        }

        let mut pdb_paths = Vec::new();
        let mut sdf_paths = Vec::new();
        for file_entry in fs::read_dir(&path)? {
            let file_entry = file_entry?;
            let file_path = file_entry.path();
            match file_path.extension().and_then(|value| value.to_str()) {
                Some("pdb") => pdb_paths.push(file_path),
                Some("sdf") => sdf_paths.push(file_path),
                _ => {}
            }
        }

        pdb_paths.sort();
        sdf_paths.sort();
        if parsing_mode == ParsingMode::Strict && (pdb_paths.len() != 1 || sdf_paths.len() != 1) {
            return Err(DataParseError::Discovery {
                root: root.to_path_buf(),
                message: format!(
                    "strict mode requires exactly one .pdb and one .sdf in {}; found pdb={} sdf={}",
                    path.display(),
                    pdb_paths.len(),
                    sdf_paths.len()
                ),
            });
        }

        if let (Some(pocket_path), Some(ligand_path)) =
            (pdb_paths.into_iter().next(), sdf_paths.into_iter().next())
        {
            let protein_id = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| DataParseError::Discovery {
                    root: root.to_path_buf(),
                    message: "non-utf8 complex directory name".to_string(),
                })?
                .to_string();
            entries.push(ManifestEntry {
                example_id: protein_id.clone(),
                protein_id,
                pocket_path,
                ligand_path,
                affinity_kcal_mol: None,
                affinity_measurement_type: None,
                affinity_raw_value: None,
                affinity_raw_unit: None,
                affinity_normalization_provenance: None,
                affinity_is_approximate: false,
                affinity_normalization_warning: None,
            });
        }
    }

    entries.sort_by(|left, right| left.example_id.cmp(&right.example_id));
    Ok(entries)
}

/// Load one manifest entry into a `MolecularExample`.
pub fn load_manifest_entry(
    entry: &ManifestEntry,
    pocket_cutoff_angstrom: f32,
    parsing_mode: ParsingMode,
    rotation_augmentation: &crate::config::types::RotationAugmentationConfig,
    rotation_rng: &mut rand::rngs::StdRng,
) -> Result<(MolecularExample, ParsedEntryReport), DataParseError> {
    let mut ligand = load_ligand_from_sdf(&entry.ligand_path)?;
    let center = ligand_center(&ligand);
    let mut pocket_result = load_pocket_from_pdb(
        &entry.pocket_path,
        center,
        pocket_cutoff_angstrom,
        parsing_mode,
    )?;

    let (rotation_augmentation_attempted, rotation_augmentation_applied) =
        apply_rotation_augmentation(
            &mut ligand,
            &mut pocket_result.pocket,
            center,
            rotation_augmentation,
            rotation_rng,
        );

    let mut example = MolecularExample::from_legacy_with_targets(
        entry.example_id.clone(),
        entry.protein_id.clone(),
        &ligand,
        &pocket_result.pocket,
        ExampleTargets {
            affinity_kcal_mol: entry.affinity_kcal_mol,
            affinity_measurement_type: entry.affinity_measurement_type.clone(),
            affinity_raw_value: entry.affinity_raw_value,
            affinity_raw_unit: entry.affinity_raw_unit.clone(),
            affinity_normalization_provenance: entry.affinity_normalization_provenance.clone(),
            affinity_is_approximate: entry.affinity_is_approximate,
            affinity_normalization_warning: entry.affinity_normalization_warning.clone(),
        },
    );
    example.source_pocket_path = Some(entry.pocket_path.clone());
    example.source_ligand_path = Some(entry.ligand_path.clone());
    Ok((
        example,
        ParsedEntryReport {
            parsed_ligand: true,
            parsed_pocket: true,
            used_pocket_fallback: pocket_result.used_fallback,
            rotation_augmentation_attempted,
            rotation_augmentation_applied,
        },
    ))
}

fn apply_rotation_augmentation(
    ligand: &mut Ligand,
    pocket: &mut Pocket,
    center: [f64; 3],
    rotation_augmentation: &crate::config::types::RotationAugmentationConfig,
    rotation_rng: &mut rand::rngs::StdRng,
) -> (bool, bool) {
    if !rotation_augmentation.enabled {
        return (false, false);
    }

    let attempted = true;
    if rotation_rng.gen_range(0.0..1.0) >= f64::from(rotation_augmentation.probability) {
        return (attempted, false);
    }

    let rotation = random_rotation_matrix(rotation_rng);
    rotate_atoms(&mut ligand.atoms, center, rotation);
    rotate_atoms(&mut pocket.atoms, center, rotation);
    (attempted, true)
}

fn random_rotation_matrix(rng: &mut rand::rngs::StdRng) -> [[f64; 3]; 3] {
    // Uniform random quaternion sampling.
    let u1: f64 = rng.gen_range(0.0..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    let u3: f64 = rng.gen_range(0.0..1.0);

    let sqrt_u1 = u1.sqrt();
    let sqrt_one_minus_u1 = (1.0 - u1).sqrt();
    let phi_1 = 2.0 * std::f64::consts::PI * u2;
    let phi_2 = 2.0 * std::f64::consts::PI * u3;

    let x = sqrt_one_minus_u1 * phi_1.sin();
    let y = sqrt_one_minus_u1 * phi_1.cos();
    let z = sqrt_u1 * phi_2.sin();
    let w = sqrt_u1 * phi_2.cos();

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let xw = x * w;
    let yz = y * z;
    let yw = y * w;
    let zw = z * w;

    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
        [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
        [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
    ]
}

fn rotate_atoms(atoms: &mut [Atom], center: [f64; 3], matrix: [[f64; 3]; 3]) {
    for atom in atoms.iter_mut() {
        let rel_x = atom.coords[0] - center[0];
        let rel_y = atom.coords[1] - center[1];
        let rel_z = atom.coords[2] - center[2];

        let rotated_x = matrix[0][0] * rel_x + matrix[0][1] * rel_y + matrix[0][2] * rel_z;
        let rotated_y = matrix[1][0] * rel_x + matrix[1][1] * rel_y + matrix[1][2] * rel_z;
        let rotated_z = matrix[2][0] * rel_x + matrix[2][1] * rel_y + matrix[2][2] * rel_z;

        atom.coords[0] = center[0] + rotated_x;
        atom.coords[1] = center[1] + rotated_y;
        atom.coords[2] = center[2] + rotated_z;
    }
}
