/// Load a ligand from a minimal V2000 SDF file.
pub fn load_ligand_from_sdf(path: &Path) -> Result<Ligand, DataParseError> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    let counts_index = lines
        .iter()
        .position(|line| line.contains("V2000"))
        .ok_or_else(|| DataParseError::InvalidSdf {
            path: path.to_path_buf(),
            message: "missing V2000 counts line".to_string(),
        })?;

    let counts = lines[counts_index];
    if counts.len() < 6 {
        return Err(DataParseError::InvalidSdf {
            path: path.to_path_buf(),
            message: "counts line too short".to_string(),
        });
    }
    let num_atoms =
        counts[0..3]
            .trim()
            .parse::<usize>()
            .map_err(|_| DataParseError::InvalidSdf {
                path: path.to_path_buf(),
                message: "invalid atom count".to_string(),
            })?;
    let num_bonds =
        counts[3..6]
            .trim()
            .parse::<usize>()
            .map_err(|_| DataParseError::InvalidSdf {
                path: path.to_path_buf(),
                message: "invalid bond count".to_string(),
            })?;

    let atom_start = counts_index + 1;
    let bond_start = atom_start + num_atoms;
    if lines.len() < bond_start + num_bonds {
        return Err(DataParseError::InvalidSdf {
            path: path.to_path_buf(),
            message: "file shorter than declared atom/bond counts".to_string(),
        });
    }

    let mut atoms = Vec::with_capacity(num_atoms);
    for (index, line) in lines[atom_start..bond_start].iter().enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(DataParseError::InvalidSdf {
                path: path.to_path_buf(),
                message: format!("atom line {} too short", index),
            });
        }

        let x = fields[0]
            .parse::<f64>()
            .map_err(|_| DataParseError::InvalidSdf {
                path: path.to_path_buf(),
                message: format!("invalid x coordinate for atom {}", index),
            })?;
        let y = fields[1]
            .parse::<f64>()
            .map_err(|_| DataParseError::InvalidSdf {
                path: path.to_path_buf(),
                message: format!("invalid y coordinate for atom {}", index),
            })?;
        let z = fields[2]
            .parse::<f64>()
            .map_err(|_| DataParseError::InvalidSdf {
                path: path.to_path_buf(),
                message: format!("invalid z coordinate for atom {}", index),
            })?;
        let element = fields[3];

        atoms.push(Atom {
            coords: [x, y, z],
            atom_type: parse_atom_type(element),
            index,
        });
    }

    let mut bonds = Vec::with_capacity(num_bonds);
    let mut bond_types = Vec::with_capacity(num_bonds);
    for line in &lines[bond_start..bond_start + num_bonds] {
        let Some((src, dst, bond_type)) = parse_v2000_bond_record(line) else {
            continue;
        };
        if src == 0 || dst == 0 {
            continue;
        }
        if src > num_atoms || dst > num_atoms {
            return Err(DataParseError::InvalidSdf {
                path: path.to_path_buf(),
                message: "bond atom index exceeds declared atom count".to_string(),
            });
        }
        bonds.push((src - 1, dst - 1));
        bond_types.push(bond_type);
    }

    Ok(Ligand {
        atoms,
        bonds,
        bond_types,
        fingerprint: None,
    })
}

#[cfg(test)]
fn parse_v2000_bond_indices(line: &str) -> Option<(usize, usize)> {
    parse_v2000_bond_record(line).map(|(src, dst, _)| (src, dst))
}

fn parse_v2000_bond_record(line: &str) -> Option<(usize, usize, i64)> {
    if line.len() >= 6 {
        let src = line.get(0..3)?.trim().parse::<usize>().ok()?;
        let dst = line.get(3..6)?.trim().parse::<usize>().ok()?;
        let bond_type = line
            .get(6..9)
            .and_then(|value| value.trim().parse::<i64>().ok())
            .unwrap_or(0);
        return Some((src, dst, normalize_sdf_bond_type(bond_type)));
    }

    let mut fields = line.split_whitespace();
    let src = fields.next()?.parse::<usize>().ok()?;
    let dst = fields.next()?.parse::<usize>().ok()?;
    let bond_type = fields
        .next()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0);
    Some((src, dst, normalize_sdf_bond_type(bond_type)))
}

fn normalize_sdf_bond_type(bond_type: i64) -> i64 {
    match bond_type {
        1..=4 => bond_type,
        _ => 0,
    }
}
