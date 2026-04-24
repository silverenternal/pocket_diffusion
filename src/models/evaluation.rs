//! Heuristic candidate generation and evaluation helpers for the modular research path.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use tch::{Kind, Tensor};

use serde::{Deserialize, Serialize};

use super::{
    ChemistryValidityEvaluator, DockingEvaluator, ExternalEvaluationReport, ExternalMetricRecord,
    GeneratedCandidateRecord, PocketCompatibilityEvaluator, ResearchForward,
};
use crate::config::ExternalBackendCommandConfig;
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

/// Executable chemistry backend adapter.
#[derive(Debug, Clone)]
pub struct CommandChemistryValidityEvaluator {
    pub config: ExternalBackendCommandConfig,
}

/// Executable docking backend adapter.
#[derive(Debug, Clone)]
pub struct CommandDockingEvaluator {
    pub config: ExternalBackendCommandConfig,
}

/// Executable pocket-compatibility backend adapter.
#[derive(Debug, Clone)]
pub struct CommandPocketCompatibilityEvaluator {
    pub config: ExternalBackendCommandConfig,
}

/// Candidate records split by generation layer for model-vs-postprocessing review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CandidateGenerationLayers {
    /// Direct final rollout state before geometry repair and bond inference.
    pub raw_rollout: Vec<GeneratedCandidateRecord>,
    /// Geometry-repaired candidates before bond inference and valence pruning.
    pub repaired: Vec<GeneratedCandidateRecord>,
    /// Repaired candidates after distance bond inference and valence pruning.
    pub inferred_bond: Vec<GeneratedCandidateRecord>,
}

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
        let centroid_fit_score = candidates.iter().map(centroid_fit_score).sum::<f64>() / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("pocket_contact_fraction", pocket_contact_fraction),
                metric("mean_centroid_offset", mean_centroid_offset),
                metric("centroid_fit_score", centroid_fit_score),
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
        let clash_free_fraction = candidates
            .iter()
            .filter(|candidate| non_bonded_clash_fraction(candidate) == 0.0)
            .count() as f64
            / total;
        let strict_pocket_fit_score =
            candidates.iter().map(strict_pocket_fit_score).sum::<f64>() / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("centroid_inside_fraction", centroid_inside_fraction),
                metric("atom_coverage_fraction", atom_coverage_fraction),
                metric("clash_free_fraction", clash_free_fraction),
                metric("strict_pocket_fit_score", strict_pocket_fit_score),
            ],
        }
    }
}

impl ChemistryValidityEvaluator for CommandChemistryValidityEvaluator {
    fn backend_name(&self) -> &'static str {
        "external_command_chemistry"
    }

    fn evaluate_chemistry(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        evaluate_via_command(self.backend_name(), &self.config, candidates)
    }
}

impl DockingEvaluator for CommandDockingEvaluator {
    fn backend_name(&self) -> &'static str {
        "external_command_docking"
    }

    fn evaluate_docking(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        evaluate_via_command(self.backend_name(), &self.config, candidates)
    }
}

impl PocketCompatibilityEvaluator for CommandPocketCompatibilityEvaluator {
    fn backend_name(&self) -> &'static str {
        "external_command_pocket_compatibility"
    }

    fn evaluate_pocket_compatibility(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        evaluate_via_command(self.backend_name(), &self.config, candidates)
    }
}

/// Build deterministic candidate records from one modular forward pass.
pub(crate) fn generate_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_layered_candidates_from_forward(example, forward, num_candidates).inferred_bond
}

/// Build raw, repaired, and inferred-bond candidate layers from one modular forward pass.
pub(crate) fn generate_layered_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> CandidateGenerationLayers {
    generate_layered_candidates_with_options(example, forward, num_candidates, true)
}

/// Build candidate layers with optional geometry repair for ablation attribution.
pub(crate) fn generate_layered_candidates_with_options(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
    enable_repair: bool,
) -> CandidateGenerationLayers {
    if num_candidates == 0 {
        return CandidateGenerationLayers {
            raw_rollout: Vec::new(),
            repaired: Vec::new(),
            inferred_bond: Vec::new(),
        };
    }

    let final_step = forward.generation.rollout.steps.last();
    let rollout_atom_types = final_step
        .map(|step| step.atom_types.clone())
        .unwrap_or_default();
    let rollout_coords = final_step
        .map(|step| step.coords.clone())
        .unwrap_or_default();
    let logits = &forward.generation.decoded.atom_type_logits;
    let topk_count = logits.size().get(1).copied().unwrap_or(1).clamp(1, 2);
    let topk = logits.topk(topk_count, -1, true, true).1;
    let pocket_centroid = tensor_centroid(&example.pocket.coords);
    let pocket_radius = pocket_radius(&example.pocket.coords, pocket_centroid);
    let pocket_points = tensor_to_coords(&example.pocket.coords);
    let base_bonds = infer_bonds(&rollout_coords);

    let mut raw_rollout = Vec::with_capacity(num_candidates);
    let mut repaired_candidates = Vec::with_capacity(num_candidates);
    let mut inferred_bond_candidates = Vec::with_capacity(num_candidates);

    for candidate_ix in 0..num_candidates {
        raw_rollout.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=raw_rollout;atoms={};bonds=0",
                rollout_atom_types.len()
            )),
            atom_types: rollout_atom_types.clone(),
            coords: rollout_coords.clone(),
            inferred_bonds: Vec::new(),
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "raw_modular_rollout:steps={};candidate={candidate_ix}",
                forward.generation.rollout.executed_steps
            ),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });

        let coords = if enable_repair {
            repair_candidate_geometry(
                &rollout_coords,
                &pocket_points,
                pocket_centroid,
                pocket_radius,
                candidate_ix,
                num_candidates,
            )
        } else {
            rollout_coords.clone()
        };
        let inferred_bonds = infer_bonds(&coords);
        repaired_candidates.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=repaired_no_bonds;atoms={};bonds=0",
                rollout_atom_types.len()
            )),
            atom_types: rollout_atom_types.clone(),
            coords: coords.clone(),
            inferred_bonds: Vec::new(),
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "geometry_repair:steps={};candidate={candidate_ix}",
                forward.generation.rollout.executed_steps
            ),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });

        let atom_types = choose_candidate_atom_types(
            &rollout_atom_types,
            logits,
            &topk,
            topk_count,
            &inferred_bonds,
            &base_bonds,
            candidate_ix,
        );
        let inferred_bonds = prune_bonds_for_valence(&coords, &atom_types, &inferred_bonds);
        inferred_bond_candidates.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=modular;atoms={};bonds={}",
                atom_types.len(),
                inferred_bonds.len()
            )),
            atom_types,
            coords,
            inferred_bonds,
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "modular_rollout_decoder:steps={}",
                forward.generation.rollout.executed_steps
            ),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });
    }

    CandidateGenerationLayers {
        raw_rollout,
        repaired: repaired_candidates,
        inferred_bond: inferred_bond_candidates,
    }
}

fn choose_candidate_atom_types(
    rollout_atom_types: &[i64],
    logits: &Tensor,
    topk: &Tensor,
    topk_count: i64,
    inferred_bonds: &[(usize, usize)],
    base_bonds: &[(usize, usize)],
    candidate_ix: usize,
) -> Vec<i64> {
    if rollout_atom_types.is_empty() {
        return Vec::new();
    }

    let target_degrees =
        preferred_atom_degrees(inferred_bonds, base_bonds, rollout_atom_types.len());
    rollout_atom_types
        .iter()
        .enumerate()
        .map(|(atom_ix, atom_type)| {
            if logits.numel() == 0 {
                normalize_atom_type_for_degree(*atom_type, target_degrees[atom_ix], candidate_ix)
            } else {
                let desired_rank = ((candidate_ix * 2 + atom_ix) as i64).rem_euclid(topk_count);
                let mut choices = (0..topk_count)
                    .map(|rank| topk.int64_value(&[atom_ix as i64, rank]))
                    .collect::<Vec<_>>();
                choices.rotate_left(desired_rank as usize);
                let picked = choices
                    .into_iter()
                    .find(|candidate_type| {
                        degree_is_supported(*candidate_type, target_degrees[atom_ix])
                    })
                    .unwrap_or_else(|| {
                        fallback_atom_type_for_degree(target_degrees[atom_ix], candidate_ix)
                    });
                normalize_atom_type_for_degree(picked, target_degrees[atom_ix], candidate_ix)
            }
        })
        .collect()
}

fn preferred_atom_degrees(
    inferred_bonds: &[(usize, usize)],
    base_bonds: &[(usize, usize)],
    atom_count: usize,
) -> Vec<usize> {
    let mut degrees = vec![0_usize; atom_count];
    for &(left, right) in inferred_bonds.iter().chain(base_bonds.iter()) {
        if let Some(value) = degrees.get_mut(left) {
            *value += 1;
        }
        if let Some(value) = degrees.get_mut(right) {
            *value += 1;
        }
    }
    degrees
}

fn degree_is_supported(atom_type: i64, degree: usize) -> bool {
    degree <= max_valence(atom_type) && !(degree > 1 && atom_type == 4)
}

fn normalize_atom_type_for_degree(atom_type: i64, degree: usize, candidate_ix: usize) -> i64 {
    if degree == 0 {
        return match candidate_ix % 3 {
            0 => atom_type,
            1 => 2,
            _ => 1,
        };
    }
    if degree > 1 && atom_type == 4 {
        return fallback_atom_type_for_degree(degree, candidate_ix);
    }
    if degree > max_valence(atom_type) {
        fallback_atom_type_for_degree(degree, candidate_ix)
    } else {
        atom_type
    }
}

fn fallback_atom_type_for_degree(degree: usize, candidate_ix: usize) -> i64 {
    match degree {
        0 => match candidate_ix % 3 {
            0 => 0,
            1 => 2,
            _ => 1,
        },
        1 => match candidate_ix % 4 {
            0 => 4,
            1 => 2,
            2 => 1,
            _ => 0,
        },
        2 => {
            if candidate_ix % 2 == 0 {
                2
            } else {
                0
            }
        }
        3 => {
            if candidate_ix % 2 == 0 {
                1
            } else {
                0
            }
        }
        _ => 0,
    }
}

fn repair_candidate_geometry(
    coords: &[[f32; 3]],
    pocket_points: &[[f32; 3]],
    pocket_centroid: [f64; 3],
    pocket_radius: f64,
    candidate_ix: usize,
    num_candidates: usize,
) -> Vec<[f32; 3]> {
    if coords.is_empty() {
        return Vec::new();
    }

    let centroid = coords.iter().fold([0.0_f64; 3], |mut acc, coord| {
        acc[0] += coord[0] as f64;
        acc[1] += coord[1] as f64;
        acc[2] += coord[2] as f64;
        acc
    });
    let denom = coords.len() as f64;
    let centroid = [
        centroid[0] / denom,
        centroid[1] / denom,
        centroid[2] / denom,
    ];
    let to_pocket = [
        pocket_centroid[0] - centroid[0],
        pocket_centroid[1] - centroid[1],
        pocket_centroid[2] - centroid[2],
    ];
    let uniqueness_phase = (candidate_ix as f64 + 1.0) / (num_candidates.max(1) as f64 + 1.0);
    let radial_scale = 1.0 + 0.08 * candidate_ix as f64;
    let anchor = 0.38 + 0.1 * uniqueness_phase;
    let swirl = 0.06 + 0.04 * uniqueness_phase;

    let mut repaired = coords
        .iter()
        .enumerate()
        .map(|(atom_ix, coord)| {
            let offset = [
                (coord[0] as f64 - centroid[0]) * radial_scale,
                (coord[1] as f64 - centroid[1]) * radial_scale,
                (coord[2] as f64 - centroid[2]) * radial_scale,
            ];
            let parity = if (atom_ix + candidate_ix) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            [
                (pocket_centroid[0] + anchor * to_pocket[0] + offset[0] + parity * swirl) as f32,
                (pocket_centroid[1] + anchor * to_pocket[1] + offset[1] - parity * swirl) as f32,
                (pocket_centroid[2] + anchor * to_pocket[2] + offset[2] + 0.5 * parity * swirl)
                    as f32,
            ]
        })
        .collect::<Vec<_>>();

    repel_close_contacts(&mut repaired);
    push_away_from_pocket_atoms(&mut repaired, pocket_points, 1.28);
    clamp_to_pocket_envelope(&mut repaired, pocket_centroid, pocket_radius + 1.6);
    repaired
}

fn repel_close_contacts(coords: &mut [[f32; 3]]) {
    if coords.len() < 2 {
        return;
    }

    for _ in 0..4 {
        let mut updates = vec![[0.0_f64; 3]; coords.len()];
        let mut moved = false;
        for left in 0..coords.len() {
            for right in (left + 1)..coords.len() {
                let dx = coords[right][0] as f64 - coords[left][0] as f64;
                let dy = coords[right][1] as f64 - coords[left][1] as f64;
                let dz = coords[right][2] as f64 - coords[left][2] as f64;
                let distance_sq = dx * dx + dy * dy + dz * dz;
                if distance_sq >= 1.15_f64.powi(2) {
                    continue;
                }
                let distance = distance_sq.sqrt().max(1e-6);
                let push = (1.15 - distance) * 0.5;
                let direction = [dx / distance, dy / distance, dz / distance];
                for axis in 0..3 {
                    updates[left][axis] -= direction[axis] * push;
                    updates[right][axis] += direction[axis] * push;
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
        for (coord, update) in coords.iter_mut().zip(updates.iter()) {
            coord[0] += update[0] as f32;
            coord[1] += update[1] as f32;
            coord[2] += update[2] as f32;
        }
    }
}

fn clamp_to_pocket_envelope(coords: &mut [[f32; 3]], pocket_centroid: [f64; 3], max_radius: f64) {
    for coord in coords.iter_mut() {
        let dx = coord[0] as f64 - pocket_centroid[0];
        let dy = coord[1] as f64 - pocket_centroid[1];
        let dz = coord[2] as f64 - pocket_centroid[2];
        let radius = (dx * dx + dy * dy + dz * dz).sqrt();
        if radius <= max_radius || radius <= 1e-6 {
            continue;
        }
        let scale = max_radius / radius;
        coord[0] = (pocket_centroid[0] + dx * scale) as f32;
        coord[1] = (pocket_centroid[1] + dy * scale) as f32;
        coord[2] = (pocket_centroid[2] + dz * scale) as f32;
    }
}

fn push_away_from_pocket_atoms(
    coords: &mut [[f32; 3]],
    pocket_points: &[[f32; 3]],
    min_distance: f64,
) {
    if coords.is_empty() || pocket_points.is_empty() {
        return;
    }

    for coord in coords.iter_mut() {
        for _ in 0..3 {
            let nearest = pocket_points
                .iter()
                .map(|pocket| (pocket, euclidean(coord, pocket)))
                .min_by(|left, right| {
                    left.1
                        .partial_cmp(&right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            let Some((closest, distance)) = nearest else {
                break;
            };
            if distance >= min_distance {
                break;
            }
            let dx = coord[0] as f64 - closest[0] as f64;
            let dy = coord[1] as f64 - closest[1] as f64;
            let dz = coord[2] as f64 - closest[2] as f64;
            let safe_distance = distance.max(1e-6);
            let push = min_distance - safe_distance + 0.05;
            coord[0] += (dx / safe_distance * push) as f32;
            coord[1] += (dy / safe_distance * push) as f32;
            coord[2] += (dz / safe_distance * push) as f32;
        }
    }
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

fn evaluate_via_command(
    backend_name: &str,
    config: &ExternalBackendCommandConfig,
    candidates: &[GeneratedCandidateRecord],
) -> ExternalEvaluationReport {
    if !config.enabled {
        return ExternalEvaluationReport {
            backend_name: backend_name.to_string(),
            metrics: vec![metric("backend_disabled", 1.0)],
        };
    }

    let Some(executable) = config.executable.as_deref() else {
        return ExternalEvaluationReport {
            backend_name: backend_name.to_string(),
            metrics: vec![metric("backend_missing_executable", 1.0)],
        };
    };

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let base_dir = std::env::temp_dir().join(format!("pocket_diffusion_backend_{stamp}"));
    if fs::create_dir_all(&base_dir).is_err() {
        return ExternalEvaluationReport {
            backend_name: backend_name.to_string(),
            metrics: vec![metric("backend_tempfile_error", 1.0)],
        };
    }
    let input_path = base_dir.join("candidates.json");
    let output_path = base_dir.join("metrics.json");
    if fs::write(
        &input_path,
        serde_json::to_string_pretty(candidates).unwrap_or_else(|_| "[]".to_string()),
    )
    .is_err()
    {
        return ExternalEvaluationReport {
            backend_name: backend_name.to_string(),
            metrics: vec![metric("backend_input_write_error", 1.0)],
        };
    }

    match Command::new(executable)
        .args(&config.args)
        .arg(&input_path)
        .arg(&output_path)
        .status()
    {
        Ok(status) if status.success() => load_command_metrics(backend_name, &output_path),
        Ok(_) => ExternalEvaluationReport {
            backend_name: backend_name.to_string(),
            metrics: vec![metric("backend_command_failed", 1.0)],
        },
        Err(_) => ExternalEvaluationReport {
            backend_name: backend_name.to_string(),
            metrics: vec![metric("backend_command_spawn_error", 1.0)],
        },
    }
}

fn load_command_metrics(backend_name: &str, path: &Path) -> ExternalEvaluationReport {
    let mut metrics = fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str::<BTreeMap<String, f64>>(&content).ok())
        .map(|metrics| {
            metrics
                .into_iter()
                .filter(|(_, value)| value.is_finite())
                .map(|(metric_name, value)| ExternalMetricRecord { metric_name, value })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![metric("backend_output_parse_error", 1.0)]);
    if !metrics
        .iter()
        .any(|record| record.metric_name == "schema_version")
    {
        metrics.push(metric("backend_missing_schema_version", 1.0));
    }
    if metrics
        .iter()
        .all(|record| record.metric_name != "backend_examples_scored")
    {
        metrics.push(metric("backend_missing_examples_scored", 1.0));
    }
    ExternalEvaluationReport {
        backend_name: backend_name.to_string(),
        metrics,
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

fn centroid_fit_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let offset = centroid_offset_from_pocket(candidate);
    if !offset.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + offset)
}

fn non_bonded_clash_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.len() < 2 {
        return 0.0;
    }
    let inferred_bonds = candidate
        .inferred_bonds
        .iter()
        .map(|&(left, right)| {
            if left < right {
                (left, right)
            } else {
                (right, left)
            }
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut total_pairs = 0_usize;
    let mut clash_pairs = 0_usize;

    for left in 0..candidate.coords.len() {
        for right in (left + 1)..candidate.coords.len() {
            if inferred_bonds.contains(&(left, right)) {
                continue;
            }
            total_pairs += 1;
            if euclidean(&candidate.coords[left], &candidate.coords[right]) < 0.9 {
                clash_pairs += 1;
            }
        }
    }

    if total_pairs == 0 {
        0.0
    } else {
        clash_pairs as f64 / total_pairs as f64
    }
}

fn strict_pocket_fit_score(candidate: &GeneratedCandidateRecord) -> f64 {
    if !basic_validity(candidate) {
        return 0.0;
    }
    let centroid_inside =
        if centroid_offset_from_pocket(candidate) <= candidate.pocket_radius as f64 {
            1.0
        } else {
            0.0
        };
    let clash_penalty = 1.0 - non_bonded_clash_fraction(candidate);
    let contact = if candidate_contact_pocket(candidate) {
        1.0
    } else {
        0.0
    };
    contact
        * centroid_inside
        * atom_coverage_fraction(candidate)
        * centroid_fit_score(candidate)
        * clash_penalty
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
            if (0.85..=1.75).contains(&distance) {
                bonds.push((left, right));
            }
        }
    }
    bonds
}

fn prune_bonds_for_valence(
    coords: &[[f32; 3]],
    atom_types: &[i64],
    inferred_bonds: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut ordered = inferred_bonds
        .iter()
        .map(|&(left, right)| (left, right, euclidean(&coords[left], &coords[right])))
        .collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut degrees = vec![0_usize; atom_types.len()];
    let mut kept = Vec::with_capacity(ordered.len());
    for (left, right, _) in ordered {
        let left_limit = atom_types.get(left).copied().map(max_valence).unwrap_or(4);
        let right_limit = atom_types.get(right).copied().map(max_valence).unwrap_or(4);
        if degrees[left] >= left_limit || degrees[right] >= right_limit {
            continue;
        }
        degrees[left] += 1;
        degrees[right] += 1;
        kept.push((left, right));
    }
    kept
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

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_with_coords(coords: Vec<[f32; 3]>) -> GeneratedCandidateRecord {
        GeneratedCandidateRecord {
            example_id: "example".to_string(),
            protein_id: "protein".to_string(),
            molecular_representation: None,
            atom_types: vec![0; coords.len()],
            inferred_bonds: infer_bonds(&coords),
            coords,
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 2.5,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: "test".to_string(),
            source_pocket_path: None,
            source_ligand_path: None,
        }
    }

    #[test]
    fn strict_pocket_fit_prefers_centered_candidates() {
        let centered =
            candidate_with_coords(vec![[0.2, 0.0, 0.0], [1.2, 0.0, 0.0], [0.7, 0.8, 0.0]]);
        let shifted =
            candidate_with_coords(vec![[2.8, 0.0, 0.0], [3.8, 0.0, 0.0], [3.3, 0.8, 0.0]]);

        assert!(strict_pocket_fit_score(&centered) > strict_pocket_fit_score(&shifted));
        assert!(centroid_fit_score(&centered) > centroid_fit_score(&shifted));
    }

    #[test]
    fn clash_fraction_ignores_inferred_bonds() {
        let bonded = candidate_with_coords(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert_eq!(non_bonded_clash_fraction(&bonded), 0.0);
    }

    #[test]
    fn repair_candidate_geometry_pushes_apart_close_contacts() {
        let repaired = repair_candidate_geometry(
            &[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            &[],
            [0.0, 0.0, 0.0],
            2.5,
            1,
            3,
        );
        assert!(euclidean(&repaired[0], &repaired[1]) >= 1.0);
        assert!(euclidean(&repaired[1], &repaired[2]) >= 1.0);
    }

    #[test]
    fn prune_bonds_respects_atom_valence_limits() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let bonds = vec![(0, 1), (0, 2), (0, 3)];
        let pruned = prune_bonds_for_valence(&coords, &[4, 0, 0, 0], &bonds);
        assert_eq!(pruned.len(), 1);
    }
}
