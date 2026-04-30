fn proxy_rerank_candidates(candidates: &[GeneratedCandidateRecord]) -> Vec<GeneratedCandidateRecord> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|left, right| {
        proxy_rerank_score(right)
            .partial_cmp(&proxy_rerank_score(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = (ranked.len() / 2).max(1).min(ranked.len());
    ranked.truncate(keep);
    ranked
}

fn proxy_rerank_score(candidate: &GeneratedCandidateRecord) -> f64 {
    let valid = if candidate_is_valid(candidate) { 1.0 } else { 0.0 };
    let contact = if candidate_has_pocket_contact(candidate) {
        1.0
    } else {
        0.0
    };
    let centroid = 1.0 / (1.0 + candidate_centroid_offset(candidate).max(0.0));
    let clash = 1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0);
    let valence = if valence_sane_proxy(candidate) { 1.0 } else { 0.0 };
    0.25 * valid + 0.25 * contact + 0.2 * centroid + 0.2 * clash + 0.1 * valence
}

#[derive(Debug, Clone)]
struct CalibratedReranker {
    coefficients: BTreeMap<String, f64>,
}

impl CalibratedReranker {
    fn fit(candidates: &[GeneratedCandidateRecord]) -> Self {
        let feature_names = reranker_feature_names();
        if candidates.is_empty() {
            return Self {
                coefficients: default_reranker_coefficients(),
            };
        }
        let features = candidates.iter().map(reranker_features).collect::<Vec<_>>();
        let targets = candidates
            .iter()
            .map(backend_compatible_rerank_target)
            .collect::<Vec<_>>();
        let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let mut means = vec![0.0; feature_names.len()];
        for row in &features {
            for (index, value) in row.iter().enumerate() {
                means[index] += value;
            }
        }
        for mean in &mut means {
            *mean /= features.len() as f64;
        }
        let mut weights = vec![0.0; feature_names.len()];
        for (row, target) in features.iter().zip(targets.iter()) {
            for (index, value) in row.iter().enumerate() {
                weights[index] += (value - means[index]) * (target - target_mean);
            }
        }
        for weight in &mut weights {
            *weight = weight.max(0.0);
        }
        let total = weights.iter().sum::<f64>();
        let coefficients = if total <= 1e-12 {
            default_reranker_coefficients()
        } else {
            feature_names
                .iter()
                .zip(weights.iter())
                .map(|(name, weight)| ((*name).to_string(), weight / total))
                .collect()
        };
        Self { coefficients }
    }

    fn rerank(&self, candidates: &[GeneratedCandidateRecord]) -> Vec<GeneratedCandidateRecord> {
        let mut ranked = candidates.to_vec();
        ranked.sort_by(|left, right| {
            self.score(right)
                .partial_cmp(&self.score(left))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.source.cmp(&right.source))
        });
        let keep = (ranked.len() / 2).max(1).min(ranked.len());
        ranked.truncate(keep);
        ranked
    }

    fn score(&self, candidate: &GeneratedCandidateRecord) -> f64 {
        reranker_feature_names()
            .iter()
            .zip(reranker_features(candidate).iter())
            .map(|(name, value)| self.coefficients.get(*name).copied().unwrap_or(0.0) * value)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }
}

fn reranker_feature_names() -> [&'static str; 6] {
    [
        "valid",
        "valence_sane",
        "pocket_contact",
        "centroid_fit",
        "clash_free",
        "bond_density_fit",
    ]
}

fn reranker_features(candidate: &GeneratedCandidateRecord) -> Vec<f64> {
    vec![
        if candidate_is_valid(candidate) { 1.0 } else { 0.0 },
        if valence_sane_proxy(candidate) { 1.0 } else { 0.0 },
        if candidate_has_pocket_contact(candidate) {
            1.0
        } else {
            0.0
        },
        centroid_fit_feature(candidate),
        1.0 - candidate_clash_fraction(candidate).clamp(0.0, 1.0),
        bond_density_fit_feature(candidate),
    ]
}

fn default_reranker_coefficients() -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("valid".to_string(), 0.22),
        ("valence_sane".to_string(), 0.18),
        ("pocket_contact".to_string(), 0.20),
        ("centroid_fit".to_string(), 0.18),
        ("clash_free".to_string(), 0.17),
        ("bond_density_fit".to_string(), 0.05),
    ])
}

fn backend_compatible_rerank_target(candidate: &GeneratedCandidateRecord) -> f64 {
    let features = reranker_features(candidate);
    (0.24 * features[0]
        + 0.18 * features[1]
        + 0.20 * features[2]
        + 0.20 * features[3]
        + 0.15 * features[4]
        + 0.03 * features[5])
        .clamp(0.0, 1.0)
}

fn centroid_fit_feature(candidate: &GeneratedCandidateRecord) -> f64 {
    let radius = (candidate.pocket_radius as f64).max(1.0);
    (1.0 - candidate_centroid_offset(candidate) / (radius + 2.0)).clamp(0.0, 1.0)
}

fn bond_density_fit_feature(candidate: &GeneratedCandidateRecord) -> f64 {
    let atoms = candidate.atom_types.len();
    if atoms < 2 {
        return 0.0;
    }
    let density = candidate.inferred_bonds.len() as f64 / atoms as f64;
    (1.0 - (density - 1.05).abs() / 1.05).clamp(0.0, 1.0)
}

fn valence_sane_proxy(candidate: &GeneratedCandidateRecord) -> bool {
    if candidate.atom_types.is_empty() {
        return false;
    }
    let mut degrees = vec![0usize; candidate.atom_types.len()];
    for &(left, right) in &candidate.inferred_bonds {
        if left < degrees.len() && right < degrees.len() {
            degrees[left] += 1;
            degrees[right] += 1;
        }
    }
    degrees
        .iter()
        .zip(candidate.atom_types.iter())
        .all(|(degree, atom_type)| *degree <= max_reasonable_valence(*atom_type))
}

fn max_reasonable_valence(atom_type: i64) -> usize {
    match atom_type {
        0 => 4,
        1 => 4,
        2 => 2,
        3 => 6,
        4 => 1,
        _ => 4,
    }
}

fn candidate_is_valid(candidate: &GeneratedCandidateRecord) -> bool {
    !candidate.atom_types.is_empty()
        && candidate.atom_types.len() == candidate.coords.len()
        && candidate
            .coords
            .iter()
            .all(|coord| coord.iter().all(|value| value.is_finite()))
}

fn candidate_has_pocket_contact(candidate: &GeneratedCandidateRecord) -> bool {
    candidate.coords.iter().any(|coord| {
        coord_distance(coord, &candidate.pocket_centroid) <= (candidate.pocket_radius + 2.0) as f64
    })
}

fn candidate_centroid_offset(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return f64::INFINITY;
    }
    let mut centroid = [0.0_f64; 3];
    for coord in &candidate.coords {
        centroid[0] += coord[0] as f64;
        centroid[1] += coord[1] as f64;
        centroid[2] += coord[2] as f64;
    }
    let denom = candidate.coords.len() as f64;
    let centroid = [
        (centroid[0] / denom) as f32,
        (centroid[1] / denom) as f32,
        (centroid[2] / denom) as f32,
    ];
    coord_distance(&centroid, &candidate.pocket_centroid)
}

fn candidate_clash_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.len() < 2 {
        return 0.0;
    }
    let bonds = candidate
        .inferred_bonds
        .iter()
        .map(|&(left, right)| {
            if left < right {
                (left, right)
            } else {
                (right, left)
            }
        })
        .collect::<BTreeSet<_>>();
    let mut total = 0usize;
    let mut clashing = 0usize;
    for left in 0..candidate.coords.len() {
        for right in (left + 1)..candidate.coords.len() {
            if bonds.contains(&(left, right)) {
                continue;
            }
            total += 1;
            if coord_distance(&candidate.coords[left], &candidate.coords[right]) < 0.9 {
                clashing += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        clashing as f64 / total as f64
    }
}

fn coord_distance(left: &[f32; 3], right: &[f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}
