/// Load a manifest from disk and resolve all relative file paths from its directory.
pub fn load_manifest(path: &Path) -> Result<DatasetManifest, DataParseError> {
    let content = fs::read_to_string(path)?;
    let mut manifest: DatasetManifest = serde_json::from_str(&content)?;
    let manifest_dir = path.parent().unwrap_or_else(|| Path::new("."));

    for entry in &mut manifest.entries {
        if entry.pocket_path.is_relative() {
            entry.pocket_path = manifest_dir.join(&entry.pocket_path);
        }
        if entry.ligand_path.is_relative() {
            entry.ligand_path = manifest_dir.join(&entry.ligand_path);
        }
    }

    Ok(manifest)
}

/// Load an external affinity table from CSV or TSV.
pub fn load_affinity_labels(path: &Path) -> Result<LoadedAffinityLabels, DataParseError> {
    let content = fs::read_to_string(path)?;
    match path.extension().and_then(|value| value.to_str()) {
        Some("csv") => load_delimited_affinity_labels(path, &content, ','),
        Some("tsv") => load_delimited_affinity_labels(path, &content, '\t'),
        _ => {
            if content
                .lines()
                .next()
                .map(|line| line.contains("affinity_kcal_mol") || line.contains(','))
                .unwrap_or(false)
            {
                load_delimited_affinity_labels(path, &content, ',')
            } else {
                load_pdbbind_index_labels(path, &content)
            }
        }
    }
}

fn load_delimited_affinity_labels(
    path: &Path,
    content: &str,
    delimiter: char,
) -> Result<LoadedAffinityLabels, DataParseError> {
    let mut report = AffinityLabelLoadReport::default();
    let mut lines = content.lines();
    let header = lines
        .next()
        .ok_or_else(|| DataParseError::InvalidLabelTable {
            path: path.to_path_buf(),
            message: "label table is empty".to_string(),
        })?;
    let columns: Vec<String> = header
        .split(delimiter)
        .map(|field| field.trim().to_ascii_lowercase())
        .collect();

    let example_ix = columns.iter().position(|field| field == "example_id");
    let protein_ix = columns.iter().position(|field| field == "protein_id");
    let affinity_ix = columns
        .iter()
        .position(|field| matches!(field.as_str(), "affinity_kcal_mol" | "affinity" | "label"));
    let measurement_ix = columns
        .iter()
        .position(|field| matches!(field.as_str(), "measurement_type" | "affinity_type"));
    let raw_value_ix = columns
        .iter()
        .position(|field| matches!(field.as_str(), "raw_value" | "affinity_value"));
    let raw_unit_ix = columns
        .iter()
        .position(|field| matches!(field.as_str(), "raw_unit" | "affinity_unit"));
    let record_ix = columns
        .iter()
        .position(|field| matches!(field.as_str(), "affinity_record" | "measurement"));

    if affinity_ix.is_none()
        && record_ix.is_none()
        && !(measurement_ix.is_some() && raw_value_ix.is_some())
    {
        return Err(DataParseError::InvalidLabelTable {
            path: path.to_path_buf(),
            message: "missing affinity columns; provide affinity_kcal_mol, affinity_record, or measurement_type/raw_value[/raw_unit]".to_string(),
        });
    }

    let mut labels = Vec::new();
    for (line_ix, line) in lines.enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            report.blank_rows += 1;
            continue;
        }
        if trimmed.starts_with('#') {
            report.comment_rows += 1;
            continue;
        }
        report.rows_seen += 1;
        let fields: Vec<&str> = line.split(delimiter).map(str::trim).collect();
        let example_id = example_ix
            .and_then(|ix| fields.get(ix))
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string());
        let protein_id = protein_ix
            .and_then(|ix| fields.get(ix))
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string());
        if example_id.is_none() && protein_id.is_none() {
            return Err(DataParseError::InvalidLabelTable {
                path: path.to_path_buf(),
                message: format!("line {} has neither example_id nor protein_id", line_ix + 2),
            });
        }

        let parsed = if let Some(ix) = affinity_ix {
            let affinity_str = fields
                .get(ix)
                .ok_or_else(|| DataParseError::InvalidLabelTable {
                    path: path.to_path_buf(),
                    message: format!("missing affinity value on data line {}", line_ix + 2),
                })?;
            let affinity_kcal_mol =
                affinity_str
                    .parse::<f32>()
                    .map_err(|_| DataParseError::InvalidLabelTable {
                        path: path.to_path_buf(),
                        message: format!("invalid affinity value on data line {}", line_ix + 2),
                    })?;
            ParsedAffinityRecord {
                affinity_kcal_mol,
                measurement_type: Some("dG".to_string()),
                raw_value: Some(affinity_kcal_mol),
                raw_unit: Some("kcal/mol".to_string()),
                normalization_provenance: Some("direct_delta_g".to_string()),
                is_approximate: false,
                normalization_warning: None,
            }
        } else if let Some(ix) = record_ix {
            let record = fields
                .get(ix)
                .ok_or_else(|| DataParseError::InvalidLabelTable {
                    path: path.to_path_buf(),
                    message: format!("missing affinity record on data line {}", line_ix + 2),
                })?;
            parse_compact_affinity_field(record).ok_or_else(|| {
                DataParseError::InvalidLabelTable {
                    path: path.to_path_buf(),
                    message: format!("invalid affinity record on data line {}", line_ix + 2),
                }
            })?
        } else {
            let measurement = measurement_ix
                .and_then(|ix| fields.get(ix))
                .copied()
                .ok_or_else(|| DataParseError::InvalidLabelTable {
                    path: path.to_path_buf(),
                    message: format!("missing measurement type on data line {}", line_ix + 2),
                })?;
            let raw_value_str = raw_value_ix
                .and_then(|ix| fields.get(ix))
                .copied()
                .ok_or_else(|| DataParseError::InvalidLabelTable {
                    path: path.to_path_buf(),
                    message: format!("missing raw value on data line {}", line_ix + 2),
                })?;
            let raw_value =
                raw_value_str
                    .parse::<f32>()
                    .map_err(|_| DataParseError::InvalidLabelTable {
                        path: path.to_path_buf(),
                        message: format!("invalid raw value on data line {}", line_ix + 2),
                    })?;
            let raw_unit = raw_unit_ix
                .and_then(|ix| fields.get(ix))
                .copied()
                .filter(|value| !value.is_empty())
                .unwrap_or("M");
            parse_measurement_components(measurement, raw_value, raw_unit).ok_or_else(|| {
                DataParseError::InvalidLabelTable {
                    path: path.to_path_buf(),
                    message: format!(
                        "unsupported measurement components on data line {}",
                        line_ix + 2
                    ),
                }
            })?
        };

        let label = AffinityLabel {
            example_id,
            protein_id,
            affinity_kcal_mol: parsed.affinity_kcal_mol,
            measurement_type: parsed.measurement_type,
            raw_value: parsed.raw_value,
            raw_unit: parsed.raw_unit,
            normalization_provenance: parsed.normalization_provenance,
            is_approximate: parsed.is_approximate,
            normalization_warning: parsed.normalization_warning,
        };
        report.parsed_rows += 1;
        let measurement = label
            .measurement_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        *report.measurement_family_histogram.entry(measurement).or_default() += 1;
        if let Some(provenance) = &label.normalization_provenance {
            report
                .normalization_provenance_values
                .insert(provenance.clone());
        }
        report.approximate_rows += usize::from(label.is_approximate);
        labels.push(label);
    }

    Ok(LoadedAffinityLabels { labels, report })
}

fn load_pdbbind_index_labels(
    path: &Path,
    content: &str,
) -> Result<LoadedAffinityLabels, DataParseError> {
    let mut report = AffinityLabelLoadReport::default();
    let mut labels = Vec::new();

    for (line_ix, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            report.blank_rows += 1;
            continue;
        }
        if line.starts_with('#') {
            report.comment_rows += 1;
            continue;
        }
        report.rows_seen += 1;

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            continue;
        }

        let protein_id = fields[0].to_string();
        let parsed = parse_index_affinity_record(&fields).ok_or_else(|| {
            DataParseError::InvalidLabelTable {
                path: path.to_path_buf(),
                message: format!(
                    "could not parse affinity record on index line {}",
                    line_ix + 1
                ),
            }
        })?;

        let label = AffinityLabel {
            example_id: None,
            protein_id: Some(protein_id),
            affinity_kcal_mol: parsed.affinity_kcal_mol,
            measurement_type: parsed.measurement_type,
            raw_value: parsed.raw_value,
            raw_unit: parsed.raw_unit,
            normalization_provenance: parsed.normalization_provenance,
            is_approximate: parsed.is_approximate,
            normalization_warning: parsed.normalization_warning,
        };
        report.parsed_rows += 1;
        let measurement = label
            .measurement_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        *report.measurement_family_histogram.entry(measurement).or_default() += 1;
        if let Some(provenance) = &label.normalization_provenance {
            report
                .normalization_provenance_values
                .insert(provenance.clone());
        }
        report.approximate_rows += usize::from(label.is_approximate);
        labels.push(label);
    }

    if labels.is_empty() {
        return Err(DataParseError::InvalidLabelTable {
            path: path.to_path_buf(),
            message: "no valid labels found in index file".to_string(),
        });
    }

    Ok(LoadedAffinityLabels { labels, report })
}

#[derive(Debug, Clone)]
struct ParsedAffinityRecord {
    affinity_kcal_mol: f32,
    measurement_type: Option<String>,
    raw_value: Option<f32>,
    raw_unit: Option<String>,
    normalization_provenance: Option<String>,
    is_approximate: bool,
    normalization_warning: Option<String>,
}

fn parse_index_affinity_record(fields: &[&str]) -> Option<ParsedAffinityRecord> {
    for (index, field) in fields.iter().enumerate() {
        if let Some(record) = parse_compact_affinity_field(field) {
            return Some(record);
        }
        if let Some(record) = parse_split_affinity_tokens(fields, index) {
            return Some(record);
        }
    }

    fields
        .iter()
        .rev()
        .find_map(|field| field.parse::<f32>().ok())
        .map(|value| ParsedAffinityRecord {
            affinity_kcal_mol: value,
            measurement_type: Some("dG".to_string()),
            raw_value: Some(value),
            raw_unit: Some("kcal/mol".to_string()),
            normalization_provenance: Some("index_direct_delta_g_fallback".to_string()),
            is_approximate: false,
            normalization_warning: None,
        })
}

fn parse_compact_affinity_field(field: &str) -> Option<ParsedAffinityRecord> {
    let compact = field.trim().trim_end_matches(';').replace(' ', "");
    for prefix in ["Kd=", "Ki=", "IC50=", "EC50=", "pKd=", "pKi="] {
        if let Some(rest) = compact.strip_prefix(prefix) {
            if prefix.starts_with('p') {
                let value = rest.parse::<f32>().ok()?;
                return Some(from_p_measurement(prefix.trim_end_matches('='), value));
            }
            let (value, unit) = split_numeric_and_unit(rest)?;
            return from_concentration_measurement(prefix.trim_end_matches('='), value, unit);
        }
    }
    None
}

fn parse_measurement_components(
    measurement: &str,
    raw_value: f32,
    raw_unit: &str,
) -> Option<ParsedAffinityRecord> {
    let normalized = normalize_measurement_name(measurement);
    if normalized.starts_with('p') {
        return Some(from_p_measurement(normalized, raw_value));
    }
    if normalized == "dG" || normalized == "dg" {
        return Some(ParsedAffinityRecord {
            affinity_kcal_mol: raw_value,
            measurement_type: Some("dG".to_string()),
            raw_value: Some(raw_value),
            raw_unit: Some(raw_unit.to_string()),
            normalization_provenance: Some("direct_delta_g".to_string()),
            is_approximate: false,
            normalization_warning: None,
        });
    }
    from_concentration_measurement(normalized, raw_value, raw_unit)
}

fn parse_split_affinity_tokens(fields: &[&str], index: usize) -> Option<ParsedAffinityRecord> {
    let measurement = normalize_measurement_name(fields.get(index)?);
    if !matches!(measurement, "Kd" | "Ki" | "IC50" | "EC50" | "pKd" | "pKi") {
        return None;
    }

    let next = fields.get(index + 1)?;
    if measurement.starts_with('p') {
        let value = next.trim_end_matches(';').parse::<f32>().ok()?;
        return Some(from_p_measurement(measurement, value));
    }

    if let Some((value, unit)) = split_numeric_and_unit(next) {
        return from_concentration_measurement(measurement, value, unit);
    }

    let value = next.trim_end_matches(';').parse::<f32>().ok()?;
    let unit = fields.get(index + 2).copied().unwrap_or("M");
    from_concentration_measurement(measurement, value, unit)
}

fn normalize_measurement_name(token: &str) -> &str {
    match token.trim_end_matches(':').trim_end_matches('=') {
        "Kd" | "KD" => "Kd",
        "Ki" | "KI" => "Ki",
        "IC50" | "ic50" => "IC50",
        "EC50" | "ec50" => "EC50",
        "pKd" | "PKD" => "pKd",
        "pKi" | "PKI" => "pKi",
        "dG" | "DG" | "dg" => "dG",
        other => other,
    }
}

fn split_numeric_and_unit(token: &str) -> Option<(f32, &str)> {
    let trimmed = token.trim().trim_end_matches(';');
    let cut = trimmed
        .find(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(trimmed.len());
    if cut == 0 || cut == trimmed.len() {
        return None;
    }
    let value = trimmed[..cut].parse::<f32>().ok()?;
    Some((value, &trimmed[cut..]))
}

fn from_p_measurement(measurement: &str, value: f32) -> ParsedAffinityRecord {
    let molar = 10_f32.powf(-value);
    ParsedAffinityRecord {
        affinity_kcal_mol: molar_to_delta_g_kcal(molar),
        measurement_type: Some(measurement.to_string()),
        raw_value: Some(value),
        raw_unit: Some("p".to_string()),
        normalization_provenance: Some(format!("{}_to_delta_g_via_molar", measurement)),
        is_approximate: false,
        normalization_warning: None,
    }
}

fn from_concentration_measurement(
    measurement: &str,
    value: f32,
    unit: &str,
) -> Option<ParsedAffinityRecord> {
    let molar = concentration_to_molar(value, unit)?;
    let is_approximate = matches!(measurement, "IC50" | "EC50");
    let normalization_warning = if is_approximate {
        Some(format!(
            "{measurement} was normalized as a concentration-derived affinity proxy; comparability to thermodynamic binding measures is approximate"
        ))
    } else {
        None
    };
    Some(ParsedAffinityRecord {
        affinity_kcal_mol: molar_to_delta_g_kcal(molar),
        measurement_type: Some(measurement.to_string()),
        raw_value: Some(value),
        raw_unit: Some(unit.to_string()),
        normalization_provenance: Some(format!(
            "{}_{}_to_delta_g_via_molar",
            measurement,
            unit.trim()
        )),
        is_approximate,
        normalization_warning,
    })
}

fn concentration_to_molar(value: f32, unit: &str) -> Option<f32> {
    let normalized = unit.trim().trim_end_matches(';').to_ascii_lowercase();
    let scale = match normalized.as_str() {
        "m" => 1.0,
        "mm" => 1e-3,
        "um" | "μm" => 1e-6,
        "nm" => 1e-9,
        "pm" => 1e-12,
        "fm" => 1e-15,
        _ => return None,
    };
    Some((value * scale).max(1e-15))
}

fn molar_to_delta_g_kcal(molar: f32) -> f32 {
    const R_KCAL_PER_MOL_K: f32 = 0.001_987_204_1;
    const TEMPERATURE_K: f32 = 298.15;
    R_KCAL_PER_MOL_K * TEMPERATURE_K * molar.ln()
}

/// Attach external labels to manifest entries. `example_id` matches take precedence.
pub fn apply_affinity_labels(
    entries: &mut [ManifestEntry],
    labels: &[AffinityLabel],
) -> LabelAttachmentReport {
    let mut by_example: BTreeMap<&str, &AffinityLabel> = BTreeMap::new();
    let mut by_protein: BTreeMap<&str, &AffinityLabel> = BTreeMap::new();
    let mut report = LabelAttachmentReport::default();
    for label in labels {
        if let Some(example_id) = &label.example_id {
            if by_example.insert(example_id.as_str(), label).is_some() {
                report.duplicate_example_id_rows += 1;
            }
        }
        if let Some(protein_id) = &label.protein_id {
            if by_protein.insert(protein_id.as_str(), label).is_some() {
                report.duplicate_protein_id_rows += 1;
            }
        }
    }

    let available_example_ids: BTreeSet<String> =
        entries.iter().map(|entry| entry.example_id.clone()).collect();
    let available_protein_ids: BTreeSet<String> =
        entries.iter().map(|entry| entry.protein_id.clone()).collect();
    for entry in entries {
        if let Some(label) = by_example.get(entry.example_id.as_str()) {
            entry.affinity_kcal_mol = Some(label.affinity_kcal_mol);
            report.example_id_matches += 1;
            entry.affinity_measurement_type = label.measurement_type.clone();
            entry.affinity_raw_value = label.raw_value;
            entry.affinity_raw_unit = label.raw_unit.clone();
            entry.affinity_normalization_provenance = label.normalization_provenance.clone();
            entry.affinity_is_approximate = label.is_approximate;
            entry.affinity_normalization_warning = label.normalization_warning.clone();
        } else if let Some(label) = by_protein.get(entry.protein_id.as_str()) {
            entry.affinity_kcal_mol = Some(label.affinity_kcal_mol);
            report.protein_id_matches += 1;
            entry.affinity_measurement_type = label.measurement_type.clone();
            entry.affinity_raw_value = label.raw_value;
            entry.affinity_raw_unit = label.raw_unit.clone();
            entry.affinity_normalization_provenance = label.normalization_provenance.clone();
            entry.affinity_is_approximate = label.is_approximate;
            entry.affinity_normalization_warning = label.normalization_warning.clone();
        }
    }
    report.unmatched_example_id_rows = by_example
        .keys()
        .filter(|key| !available_example_ids.contains(**key))
        .count();
    report.unmatched_protein_id_rows = by_protein
        .keys()
        .filter(|key| !available_protein_ids.contains(**key))
        .count();

    report
}

