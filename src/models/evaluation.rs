//! Heuristic candidate generation and evaluation helpers for the modular research path.

use std::collections::BTreeMap;

use tch::{Kind, Tensor};

use super::{
    ChemistryValidityEvaluator, DockingEvaluator, ExternalEvaluationReport, ExternalMetricRecord,
    GeneratedCandidateRecord, PocketCompatibilityEvaluator, ResearchForward,
};
use crate::{
    data::MolecularExample,
    types::{Atom, AtomType, CandidateMolecule, Ligand},
};

/// Lightweight chemistry-validity backend for active experiment reporting.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicChemistryValidityEvaluator;

/// Lightweight docking-oriented backend for executable experiment hooks.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicDockingEvaluator;

/// Lightweight pocket-compatibility backend for executable experiment hooks.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicPocketCompatibilityEvaluator;

impl ChemistryValidityEvaluator for HeuristicChemistryValidityEvaluator {
    fn backend_name(&self) -> &'static str {
        "heuristic_validity_v1"
    }

    fn evaluate_chemistry(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        let total = candidates.len().max(1) as f64;
        let valid_fraction = candidates
            .iter()
            .filter(|candidate| basic_validity(candidate))
            .count() as f64
            / total;
        let valence_sanity_fraction = candidates
            .iter()
            .filter(|candidate| valence_sane(candidate))
            .count() as f64
            / total;
        let structural_pass_fraction = candidates
            .iter()
            .filter(|candidate| structural_pass(candidate))
            .count() as f64
            / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("valid_fraction", valid_fraction),
                metric("valence_sanity_fraction", valence_sanity_fraction),
                metric("structural_pass_fraction", structural_pass_fraction),
            ],
        }
    }
}

impl DockingEvaluator for HeuristicDockingEvaluator {
    fn backend_name(&self) -> &'static str {
        "heuristic_docking_hook_v1"
    }

    fn evaluate_docking(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        let total = candidates.len().max(1) as f64;
        let pocket_contact_fraction = candidates
            .iter()
            .filter(|candidate| candidate_contact_pocket(candidate))
            .count() as f64
            / total;
        let mean_centroid_offset = candidates
            .iter()
            .map(centroid_offset_from_pocket)
            .sum::<f64>()
            / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("pocket_contact_fraction", pocket_contact_fraction),
                metric("mean_centroid_offset", mean_centroid_offset),
            ],
        }
    }
}

impl PocketCompatibilityEvaluator for HeuristicPocketCompatibilityEvaluator {
    fn backend_name(&self) -> &'static str {
        "heuristic_pocket_compatibility_v1"
    }

    fn evaluate_pocket_compatibility(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        let total = candidates.len().max(1) as f64;
        let centroid_inside_fraction = candidates
            .iter()
            .filter(|candidate| {
                centroid_offset_from_pocket(candidate) <= candidate.pocket_radius as f64
            })
            .count() as f64
            / total;
        let atom_coverage_fraction =
            candidates.iter().map(atom_coverage_fraction).sum::<f64>() / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("centroid_inside_fraction", centroid_inside_fraction),
                metric("atom_coverage_fraction", atom_coverage_fraction),
            ],
        }
    }
}

/// Build deterministic candidate records from one modular forward pass.
pub(crate) fn generate_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    if num_candidates == 0 {
        return Vec::new();
    }

    let logits = &forward.generation.decoded.atom_type_logits;
    let topk_count = logits.size().get(1).copied().unwrap_or(1).min(2);
    let topk = logits.topk(topk_count, -1, true, true).1;
    let base_coords =
        &example.decoder_supervision.noisy_coords + &forward.generation.decoded.coordinate_deltas;
    let pocket_centroid = tensor_centroid(&example.pocket.coords);
    let pocket_radius = pocket_radius(&example.pocket.coords, pocket_centroid);
    let bond_template = infer_bonds(&tensor_to_coords(&base_coords));

    (0..num_candidates)
        .map(|candidate_ix| {
            let atom_types = if logits.numel() == 0 {
                Vec::new()
            } else {
                let num_atoms = logits.size()[0] as usize;
                (0..num_atoms)
                    .map(|atom_ix| {
                        let choice = ((candidate_ix + atom_ix) as i64).rem_euclid(topk_count);
                        topk.int64_value(&[atom_ix as i64, choice])
                    })
                    .collect()
            };
            let scale = 0.75 + 0.15 * candidate_ix as f32;
            let coords = tensor_to_coords(
                &(example.decoder_supervision.noisy_coords.shallow_clone()
                    + forward.generation.decoded.coordinate_deltas.shallow_clone()
                        * (scale as f64)),
            );
            GeneratedCandidateRecord {
                example_id: example.example_id.clone(),
                protein_id: example.protein_id.clone(),
                molecular_representation: Some(format!(
                    "source=modular;atoms={};bonds={}",
                    atom_types.len(),
                    bond_template.len()
                )),
                atom_types,
                coords,
                inferred_bonds: bond_template.clone(),
                pocket_centroid: [
                    pocket_centroid[0] as f32,
                    pocket_centroid[1] as f32,
                    pocket_centroid[2] as f32,
                ],
                pocket_radius: pocket_radius as f32,
                source: "modular_research_decoder".to_string(),
            }
        })
        .collect()
}

/// Convert generated candidate records into legacy-compatible molecules.
pub(crate) fn candidate_records_to_legacy(
    records: &[GeneratedCandidateRecord],
) -> Vec<CandidateMolecule> {
    records
        .iter()
        .map(|record| CandidateMolecule {
            ligand: Ligand {
                atoms: record
                    .coords
                    .iter()
                    .enumerate()
                    .map(|(index, coords)| Atom {
                        coords: [coords[0] as f64, coords[1] as f64, coords[2] as f64],
                        atom_type: atom_type_from_index(
                            *record
                                .atom_types
                                .get(index)
                                .unwrap_or(&AtomType::Carbon.to_index()),
                        ),
                        index,
                    })
                    .collect(),
                bonds: record.inferred_bonds.clone(),
                fingerprint: None,
            },
            affinity_score: None,
            qed_score: None,
            sa_score: None,
        })
        .collect()
}

/// Convert a backend report into the persisted experiment schema.
pub(crate) fn report_to_metrics(
    report: ExternalEvaluationReport,
    enabled_status: impl Into<String>,
) -> crate::experiments::ReservedBackendMetrics {
    let metrics = report
        .metrics
        .into_iter()
        .map(|metric| (metric.metric_name, metric.value))
        .collect::<BTreeMap<_, _>>();
    crate::experiments::ReservedBackendMetrics {
        available: true,
        backend_name: Some(report.backend_name),
        metrics,
        status: enabled_status.into(),
    }
}

fn metric(name: &str, value: f64) -> ExternalMetricRecord {
    ExternalMetricRecord {
        metric_name: name.to_string(),
        value,
    }
}

fn basic_validity(candidate: &GeneratedCandidateRecord) -> bool {
    !candidate.atom_types.is_empty()
        && candidate.coords.len() == candidate.atom_types.len()
        && candidate
            .coords
            .iter()
            .all(|coord| coord.iter().all(|value| value.is_finite()))
}

fn valence_sane(candidate: &GeneratedCandidateRecord) -> bool {
    if !basic_validity(candidate) {
        return false;
    }
    let mut degrees = vec![0_usize; candidate.atom_types.len()];
    for &(left, right) in &candidate.inferred_bonds {
        if let Some(value) = degrees.get_mut(left) {
            *value += 1;
        }
        if let Some(value) = degrees.get_mut(right) {
            *value += 1;
        }
    }
    candidate
        .atom_types
        .iter()
        .enumerate()
        .all(|(index, atom_type)| degrees[index] <= max_valence(*atom_type))
}

fn structural_pass(candidate: &GeneratedCandidateRecord) -> bool {
    if !basic_validity(candidate) {
        return false;
    }
    for (ix, left) in candidate.coords.iter().enumerate() {
        for right in candidate.coords.iter().skip(ix + 1) {
            let distance = euclidean(left, right);
            if distance < 0.35 || distance > 8.0 {
                return false;
            }
        }
    }
    true
}

fn candidate_contact_pocket(candidate: &GeneratedCandidateRecord) -> bool {
    candidate.coords.iter().any(|coord| {
        euclidean(coord, &candidate.pocket_centroid) <= (candidate.pocket_radius + 2.0) as f64
    })
}

fn centroid_offset_from_pocket(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return f64::INFINITY;
    }
    let centroid = candidate
        .coords
        .iter()
        .fold([0.0_f64; 3], |mut acc, coord| {
            acc[0] += coord[0] as f64;
            acc[1] += coord[1] as f64;
            acc[2] += coord[2] as f64;
            acc
        });
    let denom = candidate.coords.len() as f64;
    let centroid = [
        centroid[0] / denom,
        centroid[1] / denom,
        centroid[2] / denom,
    ];
    euclidean(
        &[centroid[0] as f32, centroid[1] as f32, centroid[2] as f32],
        &candidate.pocket_centroid,
    )
}

fn atom_coverage_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return 0.0;
    }
    candidate
        .coords
        .iter()
        .filter(|coord| {
            euclidean(coord, &candidate.pocket_centroid) <= (candidate.pocket_radius + 3.0) as f64
        })
        .count() as f64
        / candidate.coords.len() as f64
}

fn euclidean(left: &[f32; 3], right: &[f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn infer_bonds(coords: &[[f32; 3]]) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for left in 0..coords.len() {
        for right in (left + 1)..coords.len() {
            let distance = euclidean(&coords[left], &coords[right]);
            if (0.65..=1.95).contains(&distance) {
                bonds.push((left, right));
            }
        }
    }
    bonds
}

fn max_valence(atom_type: i64) -> usize {
    match atom_type {
        0 => 4,
        1 => 3,
        2 => 2,
        3 => 6,
        4 => 1,
        _ => 4,
    }
}

fn atom_type_from_index(index: i64) -> AtomType {
    match index {
        0 => AtomType::Carbon,
        1 => AtomType::Nitrogen,
        2 => AtomType::Oxygen,
        3 => AtomType::Sulfur,
        4 => AtomType::Hydrogen,
        _ => AtomType::Other,
    }
}

fn tensor_to_coords(tensor: &Tensor) -> Vec<[f32; 3]> {
    let num_atoms = tensor.size().first().copied().unwrap_or(0).max(0) as usize;
    (0..num_atoms)
        .map(|atom_ix| {
            [
                tensor.double_value(&[atom_ix as i64, 0]) as f32,
                tensor.double_value(&[atom_ix as i64, 1]) as f32,
                tensor.double_value(&[atom_ix as i64, 2]) as f32,
            ]
        })
        .collect()
}

fn tensor_centroid(coords: &Tensor) -> [f64; 3] {
    if coords.numel() == 0 {
        return [0.0, 0.0, 0.0];
    }
    let centroid = coords.mean_dim([0].as_slice(), false, Kind::Float);
    [
        centroid.double_value(&[0]),
        centroid.double_value(&[1]),
        centroid.double_value(&[2]),
    ]
}

fn pocket_radius(coords: &Tensor, centroid: [f64; 3]) -> f64 {
    let points = tensor_to_coords(coords);
    points
        .iter()
        .map(|coord| {
            let dx = coord[0] as f64 - centroid[0];
            let dy = coord[1] as f64 - centroid[1];
            let dz = coord[2] as f64 - centroid[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0, f64::max)
}
