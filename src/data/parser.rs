//! Parsers for synthetic samples and lightweight real-data ingestion.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{ExampleTargets, MolecularExample};
use crate::config::ParsingMode;
use crate::types::{Atom, AtomType, Ligand, Pocket};

/// Errors raised while converting on-disk assets into research examples.
#[derive(Debug, Error)]
pub enum DataParseError {
    #[error("I/O error while reading dataset assets: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid PDB record in {path}: {message}")]
    InvalidPdb { path: PathBuf, message: String },
    #[error("invalid SDF record in {path}: {message}")]
    InvalidSdf { path: PathBuf, message: String },
    #[error("dataset discovery error under {root}: {message}")]
    Discovery { root: PathBuf, message: String },
    #[error("invalid label table at {path}: {message}")]
    InvalidLabelTable { path: PathBuf, message: String },
}

/// Manifest describing a dataset split-agnostic collection of complexes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Entries loaded into the research stack.
    pub entries: Vec<ManifestEntry>,
}

/// One protein-ligand complex entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Stable example identifier.
    pub example_id: String,
    /// Protein identifier used for unseen-pocket splits.
    pub protein_id: String,
    /// Path to the protein pocket source PDB file.
    pub pocket_path: PathBuf,
    /// Path to the ligand structure source SDF file.
    pub ligand_path: PathBuf,
    /// Optional affinity label in kcal/mol.
    #[serde(default)]
    pub affinity_kcal_mol: Option<f32>,
    /// Optional original measurement type before normalization.
    #[serde(default)]
    pub affinity_measurement_type: Option<String>,
    /// Optional original numeric value before normalization.
    #[serde(default)]
    pub affinity_raw_value: Option<f32>,
    /// Optional original unit before normalization.
    #[serde(default)]
    pub affinity_raw_unit: Option<String>,
    /// Optional normalization provenance for the attached affinity target.
    #[serde(default)]
    pub affinity_normalization_provenance: Option<String>,
    /// Whether the normalized target is only approximate.
    #[serde(default)]
    pub affinity_is_approximate: bool,
    /// Optional warning emitted during normalization.
    #[serde(default)]
    pub affinity_normalization_warning: Option<String>,
}

/// One affinity label row loaded from an external index table.
#[derive(Debug, Clone, PartialEq)]
pub struct AffinityLabel {
    /// Optional example identifier key.
    pub example_id: Option<String>,
    /// Optional protein identifier key.
    pub protein_id: Option<String>,
    /// Affinity value in kcal/mol.
    pub affinity_kcal_mol: f32,
    /// Original measurement type before normalization.
    pub measurement_type: Option<String>,
    /// Original numeric value before normalization.
    pub raw_value: Option<f32>,
    /// Original unit before normalization.
    pub raw_unit: Option<String>,
    /// Normalization provenance for the derived internal target.
    pub normalization_provenance: Option<String>,
    /// Whether the normalization path is only approximate.
    pub is_approximate: bool,
    /// Optional warning emitted during normalization.
    pub normalization_warning: Option<String>,
}

/// Parsed labels plus load-time accounting from one external label table.
#[derive(Debug, Clone)]
pub struct LoadedAffinityLabels {
    /// Parsed labels retained from the source table.
    pub labels: Vec<AffinityLabel>,
    /// Structured accounting for the source table.
    pub report: AffinityLabelLoadReport,
}

/// Row-level accounting for one external affinity label table.
#[derive(Debug, Clone, Default)]
pub struct AffinityLabelLoadReport {
    /// Total non-header rows encountered in the source table.
    pub rows_seen: usize,
    /// Blank rows skipped while loading.
    pub blank_rows: usize,
    /// Comment rows skipped while loading.
    pub comment_rows: usize,
    /// Rows retained as parsed affinity labels.
    pub parsed_rows: usize,
    /// Measurement-family histogram for retained labels.
    pub measurement_family_histogram: BTreeMap<String, usize>,
    /// Distinct normalization provenance values observed while loading labels.
    pub normalization_provenance_values: BTreeSet<String>,
    /// Retained labels derived from approximate families such as `IC50` or `EC50`.
    pub approximate_rows: usize,
}

/// Structured dataset validation artifact for one config-driven load.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetValidationReport {
    /// Number of entries discovered before parsing.
    pub discovered_complexes: usize,
    /// Number of examples successfully parsed into the research stack.
    pub parsed_examples: usize,
    /// Number of ligands successfully parsed.
    pub parsed_ligands: usize,
    /// Number of pockets successfully parsed.
    pub parsed_pockets: usize,
    /// Number of examples carrying an attached affinity label after loading.
    pub attached_labels: usize,
    /// Number of labels matched by `example_id`.
    pub example_id_label_matches: usize,
    /// Number of labels matched by `protein_id`.
    pub protein_id_label_matches: usize,
    /// Number of examples left without affinity labels.
    pub unlabeled_examples: usize,
    /// Number of pocket extraction fallback events.
    pub fallback_pocket_extractions: usize,
    /// Number of examples truncated away by `max_examples`.
    #[serde(default)]
    pub truncated_examples: usize,
    /// Number of parsed examples filtered by optional quality criteria.
    #[serde(default)]
    pub quality_filtered_examples: usize,
    /// Number of examples filtered because they lacked affinity labels.
    #[serde(default)]
    pub quality_filtered_unlabeled_examples: usize,
    /// Number of examples filtered by optional ligand atom-count limits.
    #[serde(default)]
    pub quality_filtered_ligand_atom_limit: usize,
    /// Number of examples filtered by optional pocket atom-count limits.
    #[serde(default)]
    pub quality_filtered_pocket_atom_limit: usize,
    /// Number of examples filtered because source structure provenance was missing.
    #[serde(default)]
    pub quality_filtered_missing_source_provenance: usize,
    /// Number of examples filtered because labeled affinity metadata was incomplete.
    #[serde(default)]
    pub quality_filtered_missing_affinity_metadata: usize,
    /// Label coverage after optional quality filtering and truncation.
    #[serde(default)]
    pub retained_label_coverage: f32,
    /// Pocket fallback fraction observed before optional fallback gating.
    #[serde(default)]
    pub observed_fallback_fraction: f32,
    /// Fraction of retained examples carrying source structure provenance.
    #[serde(default)]
    pub retained_source_provenance_coverage: f32,
    /// Number of external label rows loaded.
    pub loaded_label_rows: usize,
    /// Total non-header rows seen in the label table.
    #[serde(default)]
    pub label_table_rows_seen: usize,
    /// Blank rows skipped in the label table.
    #[serde(default)]
    pub label_table_blank_rows: usize,
    /// Comment rows skipped in the label table.
    #[serde(default)]
    pub label_table_comment_rows: usize,
    /// Number of labels normalized through approximate families such as `IC50` or `EC50`.
    pub approximate_affinity_labels: usize,
    /// Histogram of measurement families present in the loaded label table.
    #[serde(default)]
    pub loaded_label_measurement_family_histogram: BTreeMap<String, usize>,
    /// Distinct normalization provenance values seen in the loaded label table.
    #[serde(default)]
    pub loaded_label_normalization_provenance_values: BTreeSet<String>,
    /// Number of retained labeled examples normalized through approximate families.
    #[serde(default)]
    pub retained_approximate_affinity_labels: usize,
    /// Fraction of retained labeled examples that use approximate measurement families.
    #[serde(default)]
    pub retained_approximate_label_fraction: f32,
    /// Number of normalization warnings emitted while loading labels.
    pub affinity_normalization_warnings: usize,
    /// Number of later label rows that overwrote an earlier `example_id` label row.
    #[serde(default)]
    pub duplicate_example_id_label_rows: usize,
    /// Number of later label rows that overwrote an earlier `protein_id` label row.
    #[serde(default)]
    pub duplicate_protein_id_label_rows: usize,
    /// Number of loaded `example_id` label rows that did not attach to any manifest entry.
    #[serde(default)]
    pub unmatched_example_id_label_rows: usize,
    /// Number of loaded `protein_id` label rows that did not attach to any manifest entry.
    #[serde(default)]
    pub unmatched_protein_id_label_rows: usize,
    /// Fraction of retained labeled examples carrying normalization provenance.
    #[serde(default)]
    pub retained_normalization_provenance_coverage: f32,
    /// Number of retained labeled examples missing normalization provenance.
    #[serde(default)]
    pub retained_missing_normalization_provenance: usize,
    /// Number of retained labeled examples missing measurement-family metadata.
    #[serde(default)]
    pub retained_missing_measurement_type: usize,
    /// Histogram of retained measurement families.
    #[serde(default)]
    pub retained_measurement_family_histogram: BTreeMap<String, usize>,
    /// Number of distinct retained measurement families.
    #[serde(default)]
    pub retained_measurement_family_count: usize,
    /// Distinct retained normalization provenance values.
    #[serde(default)]
    pub retained_normalization_provenance_values: BTreeSet<String>,
    /// Warnings emitted by the affinity normalization path.
    pub normalization_warning_messages: Vec<String>,
    /// Active parsing mode used for the dataset load.
    pub parsing_mode: String,
}

/// Report for one parsed manifest entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParsedEntryReport {
    /// Whether ligand parsing succeeded for this entry.
    pub parsed_ligand: bool,
    /// Whether pocket parsing succeeded for this entry.
    pub parsed_pocket: bool,
    /// Whether pocket extraction used the nearest-atom fallback.
    pub used_pocket_fallback: bool,
}

/// Metadata from attaching external affinity labels to manifest entries.
#[derive(Debug, Clone, Copy, Default)]
pub struct LabelAttachmentReport {
    /// Number of labels matched by `example_id`.
    pub example_id_matches: usize,
    /// Number of labels matched by `protein_id`.
    pub protein_id_matches: usize,
    /// Number of duplicate `example_id` rows overwritten during attachment-map construction.
    pub duplicate_example_id_rows: usize,
    /// Number of duplicate `protein_id` rows overwritten during attachment-map construction.
    pub duplicate_protein_id_rows: usize,
    /// Number of loaded `example_id` rows that did not match any manifest entry.
    pub unmatched_example_id_rows: usize,
    /// Number of loaded `protein_id` rows that did not match any manifest entry.
    pub unmatched_protein_id_rows: usize,
}

#[derive(Debug, Clone)]
struct PocketLoadResult {
    pocket: Pocket,
    used_fallback: bool,
}

/// Build a small deterministic synthetic dataset.
pub fn synthetic_phase1_examples() -> Vec<MolecularExample> {
    vec![
        MolecularExample::from_legacy(
            "ex-0",
            "protein-a",
            &toy_ligand(0.0),
            &toy_pocket("protein-a", 0.0),
        ),
        MolecularExample::from_legacy(
            "ex-1",
            "protein-b",
            &toy_ligand(0.4),
            &toy_pocket("protein-b", 0.8),
        ),
        MolecularExample::from_legacy(
            "ex-2",
            "protein-c",
            &toy_ligand(-0.5),
            &toy_pocket("protein-c", -0.2),
        ),
        MolecularExample::from_legacy(
            "ex-3",
            "protein-a",
            &toy_ligand(1.1),
            &toy_pocket("protein-a", 0.6),
        ),
    ]
}

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
    for line in &lines[bond_start..bond_start + num_bonds] {
        let Some((src, dst)) = parse_v2000_bond_indices(line) else {
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
    }

    Ok(Ligand {
        atoms,
        bonds,
        fingerprint: None,
    })
}

fn parse_v2000_bond_indices(line: &str) -> Option<(usize, usize)> {
    if line.len() >= 6 {
        let src = line.get(0..3)?.trim().parse::<usize>().ok()?;
        let dst = line.get(3..6)?.trim().parse::<usize>().ok()?;
        return Some((src, dst));
    }

    let mut fields = line.split_whitespace();
    let src = fields.next()?.parse::<usize>().ok()?;
    let dst = fields.next()?.parse::<usize>().ok()?;
    Some((src, dst))
}

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

fn toy_ligand(offset: f64) -> Ligand {
    Ligand {
        atoms: vec![
            Atom {
                coords: [0.0 + offset, 0.0, 0.0],
                atom_type: AtomType::Carbon,
                index: 0,
            },
            Atom {
                coords: [1.3 + offset, 0.1, 0.0],
                atom_type: AtomType::Nitrogen,
                index: 1,
            },
            Atom {
                coords: [2.1 + offset, 1.0, 0.2],
                atom_type: AtomType::Oxygen,
                index: 2,
            },
        ],
        bonds: vec![(0, 1), (1, 2)],
        fingerprint: None,
    }
}

fn toy_pocket(name: &str, offset: f64) -> Pocket {
    Pocket {
        name: name.to_string(),
        atoms: vec![
            Atom {
                coords: [offset, 0.0, 0.0],
                atom_type: AtomType::Carbon,
                index: 0,
            },
            Atom {
                coords: [0.7 + offset, 1.0, 0.4],
                atom_type: AtomType::Oxygen,
                index: 1,
            },
            Atom {
                coords: [1.5 + offset, -0.8, 0.2],
                atom_type: AtomType::Nitrogen,
                index: 2,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn malformed_sdf_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let sdf_path = temp.path().join("bad.sdf");
        fs::write(&sdf_path, "broken sdf\n").unwrap();

        let err = load_ligand_from_sdf(&sdf_path).unwrap_err();
        assert!(matches!(err, DataParseError::InvalidSdf { .. }));
    }

    #[test]
    fn sdf_bond_parser_handles_contiguous_fixed_width_indices() {
        assert_eq!(
            parse_v2000_bond_indices(" 34100  1  0  0  0"),
            Some((34, 100))
        );
        assert_eq!(parse_v2000_bond_indices("  2  5  1  0  0  0"), Some((2, 5)));
    }

    #[test]
    fn strict_mode_rejects_short_pdb_atom_records() {
        let temp = tempfile::tempdir().unwrap();
        let ligand_path = temp.path().join("ligand.sdf");
        let pocket_path = temp.path().join("bad.pdb");

        fs::write(
            &ligand_path,
            "ligand\n  codex\n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n",
        )
        .unwrap();
        fs::write(&pocket_path, "ATOM      1  C\n").unwrap();

        let entry = ManifestEntry {
            example_id: "ex-1".to_string(),
            protein_id: "prot-1".to_string(),
            pocket_path,
            ligand_path,
            affinity_kcal_mol: None,
            affinity_measurement_type: None,
            affinity_raw_value: None,
            affinity_raw_unit: None,
            affinity_normalization_provenance: None,
            affinity_is_approximate: false,
            affinity_normalization_warning: None,
        };

        let err = load_manifest_entry(&entry, 6.0, ParsingMode::Strict).unwrap_err();
        assert!(matches!(err, DataParseError::InvalidPdb { .. }));
    }
}
