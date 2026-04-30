use super::*;

pub(super) fn metric(name: &str, value: f64) -> ExternalMetricRecord {
    ExternalMetricRecord {
        metric_name: name.to_string(),
        value,
    }
}

pub(super) fn evaluate_via_command(
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
    let backend_input = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            serde_json::to_value(candidate)
                .ok()
                .map(|value| (index, value))
        })
        .map(|(index, mut value)| {
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "candidate_id".to_string(),
                    serde_json::Value::String(format!(
                        "{}:{}:{}",
                        candidate_identity_fragment(&candidates[index].example_id),
                        candidate_identity_fragment(&candidates[index].protein_id),
                        index
                    )),
                );
                object.insert(
                    "candidate_index".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(index as u64)),
                );
            }
            value
        })
        .collect::<Vec<_>>();
    if fs::write(
        &input_path,
        serde_json::to_string_pretty(&backend_input).unwrap_or_else(|_| "[]".to_string()),
    )
    .is_err()
    {
        return ExternalEvaluationReport {
            backend_name: backend_name.to_string(),
            metrics: vec![metric("backend_input_write_error", 1.0)],
        };
    }

    let mut child = match Command::new(executable)
        .args(&config.args)
        .arg(&input_path)
        .arg(&output_path)
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            return ExternalEvaluationReport {
                backend_name: backend_name.to_string(),
                metrics: vec![metric("backend_command_spawn_error", 1.0)],
            };
        }
    };
    let started = Instant::now();
    let timeout = Duration::from_millis(config.timeout_ms);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                return load_command_metrics(backend_name, &output_path);
            }
            Ok(Some(_)) => {
                return ExternalEvaluationReport {
                    backend_name: backend_name.to_string(),
                    metrics: vec![metric("backend_command_failed", 1.0)],
                };
            }
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return ExternalEvaluationReport {
                    backend_name: backend_name.to_string(),
                    metrics: vec![
                        metric("backend_command_timeout", 1.0),
                        metric("backend_timeout_ms", config.timeout_ms as f64),
                    ],
                };
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => {
                return ExternalEvaluationReport {
                    backend_name: backend_name.to_string(),
                    metrics: vec![metric("backend_command_wait_error", 1.0)],
                };
            }
        }
    }
}

fn candidate_identity_fragment(raw: &str) -> String {
    raw.replace('|', "%7C")
}

fn load_command_metrics(backend_name: &str, path: &Path) -> ExternalEvaluationReport {
    let mut metrics = fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .and_then(|payload| parse_backend_output_metrics(&payload))
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

fn parse_backend_output_metrics(payload: &serde_json::Value) -> Option<Vec<ExternalMetricRecord>> {
    let root = payload.as_object()?;
    let has_structured_sections =
        root.contains_key("aggregate_metrics") || root.contains_key("candidate_metrics");
    if !has_structured_sections {
        return Some(
            root.iter()
                .filter_map(|(metric_name, value)| {
                    value
                        .as_f64()
                        .filter(|number| number.is_finite())
                        .map(|number| ExternalMetricRecord {
                            metric_name: metric_name.clone(),
                            value: number,
                        })
                })
                .collect(),
        );
    }

    let mut metrics = Vec::new();
    if let Some(aggregate) = root
        .get("aggregate_metrics")
        .and_then(|value| value.as_object())
    {
        metrics.extend(aggregate.iter().filter_map(|(metric_name, value)| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .map(|number| ExternalMetricRecord {
                    metric_name: metric_name.clone(),
                    value: number,
                })
        }));
    }

    let mut candidate_rows = 0.0;
    if let Some(rows) = root
        .get("candidate_metrics")
        .and_then(|value| value.as_array())
    {
        for row in rows {
            let Some(row_object) = row.as_object() else {
                continue;
            };
            let candidate_id = row_object
                .get("candidate_id")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            let example_id = row_object
                .get("example_id")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            let protein_id = row_object
                .get("protein_id")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            let Some(row_metrics) = row_object
                .get("metrics")
                .and_then(|value| value.as_object())
            else {
                continue;
            };
            candidate_rows += 1.0;
            for (metric_name, value) in row_metrics {
                let Some(number) = value.as_f64().filter(|number| number.is_finite()) else {
                    continue;
                };
                metrics.push(ExternalMetricRecord {
                    metric_name: format!(
                        "candidate_metric|{}|{}|{}|{}",
                        candidate_identity_fragment(example_id),
                        candidate_identity_fragment(protein_id),
                        candidate_identity_fragment(candidate_id),
                        metric_name
                    ),
                    value: number,
                });
            }
        }
    }
    if candidate_rows > 0.0 {
        metrics.push(metric("candidate_metric_rows", candidate_rows));
    }
    Some(metrics)
}
pub(super) fn basic_validity(candidate: &GeneratedCandidateRecord) -> bool {
    !candidate.atom_types.is_empty()
        && candidate.coords.len() == candidate.atom_types.len()
        && candidate
            .coords
            .iter()
            .all(|coord| coord.iter().all(|value| value.is_finite()))
}

pub(super) fn valence_sane(candidate: &GeneratedCandidateRecord) -> bool {
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

pub(super) fn structural_pass(candidate: &GeneratedCandidateRecord) -> bool {
    if !basic_validity(candidate) {
        return false;
    }
    let max_reasonable_span = ((candidate.pocket_radius as f64) * 2.0 + 6.0).max(12.0);
    for (ix, left) in candidate.coords.iter().enumerate() {
        for right in candidate.coords.iter().skip(ix + 1) {
            let distance = euclidean(left, right);
            if distance < 0.35 || distance > max_reasonable_span {
                return false;
            }
        }
    }
    true
}

pub(super) fn candidate_contact_pocket(candidate: &GeneratedCandidateRecord) -> bool {
    candidate.coords.iter().any(|coord| {
        euclidean(coord, &candidate.pocket_centroid) <= (candidate.pocket_radius + 2.0) as f64
    })
}

pub(super) fn centroid_offset_from_pocket(candidate: &GeneratedCandidateRecord) -> f64 {
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

pub(super) fn atom_coverage_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
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

pub(super) fn centroid_fit_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let offset = centroid_offset_from_pocket(candidate);
    if !offset.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + offset)
}

pub(super) fn non_bonded_clash_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
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

pub(super) fn strict_pocket_fit_score(candidate: &GeneratedCandidateRecord) -> f64 {
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

pub(super) fn euclidean(left: &[f32; 3], right: &[f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub(super) fn infer_bonds(coords: &[[f32; 3]]) -> Vec<(usize, usize)> {
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

pub(super) fn prune_bonds_for_valence(
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

pub(super) fn max_valence(atom_type: i64) -> usize {
    match atom_type {
        0 => 4,
        1 => 4,
        2 => 2,
        3 => 6,
        4 => 1,
        _ => 4,
    }
}

pub(super) fn atom_type_from_index(index: i64) -> AtomType {
    match index {
        0 => AtomType::Carbon,
        1 => AtomType::Nitrogen,
        2 => AtomType::Oxygen,
        3 => AtomType::Sulfur,
        4 => AtomType::Hydrogen,
        _ => AtomType::Other,
    }
}

pub(super) fn tensor_to_coords(tensor: &Tensor) -> Vec<[f32; 3]> {
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

pub(super) fn tensor_centroid(coords: &Tensor) -> [f64; 3] {
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

pub(super) fn pocket_radius(coords: &Tensor, centroid: [f64; 3]) -> f64 {
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
