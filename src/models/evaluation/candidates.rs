use super::scoring::{atom_type_from_index, euclidean, max_valence};
use super::*;

pub(super) fn choose_candidate_atom_types(
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
        2 if candidate_ix % 2 == 0 => 2,
        2 => 0,
        3 if candidate_ix % 2 == 0 => 1,
        3 => 0,
        _ => 0,
    }
}

pub(super) fn repair_candidate_geometry(
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
                bond_types: vec![0; record.inferred_bonds.len()],
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
