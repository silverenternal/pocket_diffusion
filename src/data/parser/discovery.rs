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
) -> Result<(MolecularExample, ParsedEntryReport), DataParseError> {
    let ligand = load_ligand_from_sdf(&entry.ligand_path)?;
    let center = ligand_center(&ligand);
    let pocket_result = load_pocket_from_pdb(
        &entry.pocket_path,
        center,
        pocket_cutoff_angstrom,
        parsing_mode,
    )?;
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
        },
    ))
}

