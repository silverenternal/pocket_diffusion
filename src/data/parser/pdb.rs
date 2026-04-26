/// Extract a local pocket from a protein structure around the ligand center.
fn load_pocket_from_pdb(
    path: &Path,
    ligand_center: [f64; 3],
    pocket_cutoff_angstrom: f32,
    parsing_mode: ParsingMode,
) -> Result<PocketLoadResult, DataParseError> {
    let content = fs::read_to_string(path)?;
    let mut all_atoms = Vec::new();
    let mut local_atoms = Vec::new();

    for line in content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }
        if line.len() < 54 {
            if parsing_mode == ParsingMode::Strict {
                return Err(DataParseError::InvalidPdb {
                    path: path.to_path_buf(),
                    message: "strict mode rejects ATOM/HETATM lines shorter than 54 columns"
                        .to_string(),
                });
            }
            continue;
        }

        let x = parse_pdb_float(line, 30, 38, path)?;
        let y = parse_pdb_float(line, 38, 46, path)?;
        let z = parse_pdb_float(line, 46, 54, path)?;
        let element = if line.len() >= 78 {
            line[76..78].trim()
        } else if line.len() >= 16 {
            line[12..16].trim()
        } else {
            "C"
        };

        let atom = Atom {
            coords: [x, y, z],
            atom_type: parse_atom_type(element),
            index: all_atoms.len(),
        };
        let distance = euclidean_distance(atom.coords, ligand_center);
        all_atoms.push((distance, atom.clone()));

        if distance <= pocket_cutoff_angstrom as f64 {
            local_atoms.push(atom);
        }
    }

    let mut used_fallback = false;
    if local_atoms.is_empty() {
        if parsing_mode == ParsingMode::Strict {
            return Err(DataParseError::InvalidPdb {
                path: path.to_path_buf(),
                message: "strict mode rejects nearest-atom pocket fallback when no atoms fall within the cutoff".to_string(),
            });
        }
        all_atoms.sort_by(|left, right| left.0.total_cmp(&right.0));
        local_atoms.extend(all_atoms.into_iter().take(64).map(|(_, atom)| atom));
        used_fallback = !local_atoms.is_empty();
    }

    if local_atoms.is_empty() {
        return Err(DataParseError::InvalidPdb {
            path: path.to_path_buf(),
            message: "no atoms were parsed from the structure".to_string(),
        });
    }

    for (index, atom) in local_atoms.iter_mut().enumerate() {
        atom.index = index;
    }

    let pocket_name = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("pocket")
        .to_string();
    Ok(PocketLoadResult {
        pocket: Pocket {
            atoms: local_atoms,
            name: pocket_name,
        },
        used_fallback,
    })
}

/// Compute the ligand center used for pocket extraction.
pub fn ligand_center(ligand: &Ligand) -> [f64; 3] {
    if ligand.atoms.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let (x_sum, y_sum, z_sum) = ligand.atoms.iter().fold((0.0, 0.0, 0.0), |acc, atom| {
        (
            acc.0 + atom.coords[0],
            acc.1 + atom.coords[1],
            acc.2 + atom.coords[2],
        )
    });
    let denom = ligand.atoms.len() as f64;
    [x_sum / denom, y_sum / denom, z_sum / denom]
}

fn parse_pdb_float(
    line: &str,
    start: usize,
    end: usize,
    path: &Path,
) -> Result<f64, DataParseError> {
    line[start..end]
        .trim()
        .parse::<f64>()
        .map_err(|_| DataParseError::InvalidPdb {
            path: path.to_path_buf(),
            message: format!("invalid float slice {}..{}", start, end),
        })
}

fn parse_atom_type(element: &str) -> AtomType {
    match element.trim().to_ascii_uppercase().as_str() {
        "C" => AtomType::Carbon,
        "N" => AtomType::Nitrogen,
        "O" => AtomType::Oxygen,
        "S" => AtomType::Sulfur,
        "H" => AtomType::Hydrogen,
        _ => AtomType::Other,
    }
}

fn euclidean_distance(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    let dz = lhs[2] - rhs[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

