/// Configuration for a separate leakage-probe calibration on frozen representations.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public calibration path is exercised by smoke tests and documented for runs.
pub(crate) struct FrozenLeakageProbeCalibrationConfig {
    /// Fraction of examples used to fit the lightweight held-out probe.
    pub train_fraction: f64,
    /// Feature-prefix capacities to sweep.
    pub capacities: Vec<usize>,
    /// Ridge regularization values to sweep.
    pub regularization: Vec<f64>,
    /// Off-modality routes to evaluate.
    pub routes: Vec<FrozenLeakageProbeRoute>,
}

#[allow(dead_code)] // Used by downstream calibration commands.
impl Default for FrozenLeakageProbeCalibrationConfig {
    fn default() -> Self {
        Self {
            train_fraction: 0.6,
            capacities: vec![1, 4],
            regularization: vec![0.0, 1.0e-3],
            routes: vec![FrozenLeakageProbeRoute::TopologyToGeometry],
        }
    }
}

/// Off-modality route used by frozen leakage probe calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Route variants are part of the calibration sweep contract.
pub(crate) enum FrozenLeakageProbeRoute {
    /// Topology representation predicts a geometry scalar target.
    TopologyToGeometry,
    /// Geometry representation predicts a topology scalar target.
    GeometryToTopology,
    /// Pocket representation predicts a geometry scalar target.
    PocketToGeometry,
}

#[allow(dead_code)] // Helpers are used by the calibration path.
impl FrozenLeakageProbeRoute {
    fn label(self) -> &'static str {
        match self {
            Self::TopologyToGeometry => "topology_to_geometry",
            Self::GeometryToTopology => "geometry_to_topology",
            Self::PocketToGeometry => "pocket_to_geometry",
        }
    }

    fn source_modality(self) -> &'static str {
        match self {
            Self::TopologyToGeometry => "topology",
            Self::GeometryToTopology => "geometry",
            Self::PocketToGeometry => "pocket",
        }
    }

    fn target(self) -> &'static str {
        match self {
            Self::TopologyToGeometry | Self::PocketToGeometry => "geometry_mean_pairwise_distance",
            Self::GeometryToTopology => "topology_edge_density",
        }
    }
}

/// Train lightweight probes on frozen representations and evaluate held-out predictability.
#[allow(dead_code)] // Exposed as a documented smoke/calibration path.
pub(crate) fn calibrate_frozen_leakage_probes(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    config: &FrozenLeakageProbeCalibrationConfig,
) -> FrozenLeakageProbeCalibrationReport {
    calibrate_frozen_leakage_probes_for_split(examples, forwards, "unspecified", config)
}

/// Train lightweight probes for one named evaluation split.
#[allow(dead_code)] // Exposed through evaluation and smoke artifact paths.
pub(crate) fn calibrate_frozen_leakage_probes_for_split(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    split_name: &str,
    config: &FrozenLeakageProbeCalibrationConfig,
) -> FrozenLeakageProbeCalibrationReport {
    debug_assert_eq!(examples.len(), forwards.len());
    let sample_count = examples.len().min(forwards.len());
    let split_name = if split_name.trim().is_empty() {
        "unspecified"
    } else {
        split_name
    };
    let capacities = if config.capacities.is_empty() {
        vec![1]
    } else {
        config.capacities.clone()
    };
    let regularization = if config.regularization.is_empty() {
        vec![0.0]
    } else {
        config.regularization.clone()
    };
    if sample_count < 2 {
        return FrozenLeakageProbeCalibrationReport {
            calibration_status: "insufficient_data".to_string(),
            split_name: split_name.to_string(),
            training_time_signal: "training-time leakage penalty/proxy is not a held-out frozen-probe estimate".to_string(),
            representation_source: "frozen forward encodings".to_string(),
            optimizer_penalty_separated: true,
            trivial_baseline: "heldout_target_mean_mse".to_string(),
            claim_boundary: "requires at least two examples for train/held-out calibration".to_string(),
            ..FrozenLeakageProbeCalibrationReport::default()
        };
    }

    let mut train_count =
        ((sample_count as f64) * config.train_fraction.clamp(0.1, 0.9)).round() as usize;
    train_count = train_count.clamp(1, sample_count - 1);
    let heldout_count = sample_count - train_count;
    let routes = if config.routes.is_empty() {
        vec![FrozenLeakageProbeRoute::TopologyToGeometry]
    } else {
        config.routes.clone()
    };

    let mut route_reports = Vec::new();
    let mut sweep_rows = Vec::new();
    for route in routes {
        let mut features = Vec::with_capacity(sample_count);
        let mut targets = Vec::with_capacity(sample_count);
        for (example, forward) in examples.iter().zip(forwards.iter()).take(sample_count) {
            features.push(frozen_route_features(route, forward));
            targets.push(frozen_route_target(route, example));
        }

        let mut best: Option<FrozenLeakageProbeSweepRow> = None;
        for capacity in capacities.iter().copied() {
            for lambda in regularization.iter().copied() {
                let row = fit_and_score_frozen_probe(
                    route,
                    split_name,
                    &features,
                    &targets,
                    train_count,
                    heldout_count,
                    capacity,
                    lambda.max(0.0),
                );
                if best
                    .as_ref()
                    .is_none_or(|current| row.heldout_mse < current.heldout_mse)
                {
                    best = Some(row.clone());
                }
                sweep_rows.push(row);
            }
        }
        if let Some(best) = best {
            route_reports.push(FrozenLeakageProbeRouteReport {
                route: route.label().to_string(),
                split_name: split_name.to_string(),
                source_modality: route.source_modality().to_string(),
                target: route.target().to_string(),
                train_count,
                heldout_count,
                best_capacity: best.capacity,
                best_regularization: best.regularization,
                heldout_mse: best.heldout_mse,
                baseline_mse: best.baseline_mse,
                improvement_over_baseline: best.improvement_over_baseline,
                predicts_off_modality_target: best.improvement_over_baseline > 0.0,
            });
        }
    }

    FrozenLeakageProbeCalibrationReport {
        calibration_status: "ok".to_string(),
        split_name: split_name.to_string(),
        training_time_signal: "distinct from training-time leakage penalty; fit on frozen representations after the forward pass".to_string(),
        representation_source: "frozen modality pooled embeddings".to_string(),
        optimizer_penalty_separated: true,
        trivial_baseline: "heldout_target_mean_mse".to_string(),
        routes: route_reports,
        capacity_sweep: sweep_rows,
        claim_boundary: "held-out frozen-probe predictability diagnoses recoverable off-modality information; it does not prove semantic independence".to_string(),
        ..FrozenLeakageProbeCalibrationReport::default()
    }
}

#[allow(dead_code)]
fn frozen_route_features(route: FrozenLeakageProbeRoute, forward: &ResearchForward) -> Vec<f64> {
    let tensor = match route {
        FrozenLeakageProbeRoute::TopologyToGeometry => &forward.encodings.topology.pooled_embedding,
        FrozenLeakageProbeRoute::GeometryToTopology => &forward.encodings.geometry.pooled_embedding,
        FrozenLeakageProbeRoute::PocketToGeometry => &forward.encodings.pocket.pooled_embedding,
    };
    let values = tensor_values(tensor, 32);
    if values.is_empty() {
        vec![0.0]
    } else {
        values
    }
}

#[allow(dead_code)]
fn frozen_route_target(
    route: FrozenLeakageProbeRoute,
    example: &crate::data::MolecularExample,
) -> f64 {
    match route {
        FrozenLeakageProbeRoute::TopologyToGeometry | FrozenLeakageProbeRoute::PocketToGeometry => {
            example
                .geometry
                .pairwise_distances
                .mean(tch::Kind::Float)
                .double_value(&[])
        }
        FrozenLeakageProbeRoute::GeometryToTopology => {
            example
                .topology
                .adjacency
                .mean(tch::Kind::Float)
                .double_value(&[])
        }
    }
}

#[allow(dead_code)]
fn tensor_values(tensor: &tch::Tensor, limit: usize) -> Vec<f64> {
    let flat = tensor
        .to_device(tch::Device::Cpu)
        .to_kind(tch::Kind::Double)
        .reshape([-1]);
    let count = flat.numel().min(limit);
    (0..count)
        .map(|index| flat.double_value(&[index as i64]))
        .collect()
}

#[allow(dead_code)]
fn fit_and_score_frozen_probe(
    route: FrozenLeakageProbeRoute,
    split_name: &str,
    features: &[Vec<f64>],
    targets: &[f64],
    train_count: usize,
    heldout_count: usize,
    capacity: usize,
    regularization: f64,
) -> FrozenLeakageProbeSweepRow {
    let feature_dim = features
        .iter()
        .map(Vec::len)
        .min()
        .unwrap_or(1)
        .max(1);
    let used_capacity = capacity.clamp(1, feature_dim);
    let train_features = design_matrix(&features[..train_count], used_capacity);
    let weights = fit_ridge(&train_features, &targets[..train_count], regularization)
        .unwrap_or_else(|| intercept_only_weights(mean(&targets[..train_count]), used_capacity));
    let heldout_features = design_matrix(&features[train_count..], used_capacity);
    let heldout_targets = &targets[train_count..];
    let heldout_mse = mse_for_weights(&heldout_features, heldout_targets, &weights);
    let train_mean = mean(&targets[..train_count]);
    let baseline_mse = heldout_targets
        .iter()
        .map(|target| (target - train_mean).powi(2))
        .sum::<f64>()
        / heldout_count.max(1) as f64;
    let improvement_over_baseline = if baseline_mse > 1.0e-12 {
        (baseline_mse - heldout_mse) / baseline_mse
    } else {
        0.0
    };

    FrozenLeakageProbeSweepRow {
        route: route.label().to_string(),
        split_name: split_name.to_string(),
        capacity: used_capacity,
        regularization,
        train_count,
        heldout_count,
        heldout_mse,
        baseline_mse,
        improvement_over_baseline,
    }
}

#[allow(dead_code)]
fn design_matrix(features: &[Vec<f64>], capacity: usize) -> Vec<Vec<f64>> {
    features
        .iter()
        .map(|row| {
            let mut design = Vec::with_capacity(capacity + 1);
            design.push(1.0);
            for index in 0..capacity {
                design.push(row.get(index).copied().unwrap_or(0.0));
            }
            design
        })
        .collect()
}

#[allow(dead_code)]
fn fit_ridge(features: &[Vec<f64>], targets: &[f64], regularization: f64) -> Option<Vec<f64>> {
    let dim = features.first()?.len();
    let mut normal = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];
    for (row, target) in features.iter().zip(targets.iter()) {
        for i in 0..dim {
            rhs[i] += row[i] * target;
            for j in 0..dim {
                normal[i][j] += row[i] * row[j];
            }
        }
    }
    for (index, row) in normal.iter_mut().enumerate().skip(1) {
        row[index] += regularization;
    }
    solve_linear_system(normal, rhs)
}

#[allow(dead_code)]
fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let dim = rhs.len();
    for pivot in 0..dim {
        let mut best = pivot;
        let mut best_abs = matrix[pivot][pivot].abs();
        for (row, values) in matrix.iter().enumerate().skip(pivot + 1) {
            let candidate = values[pivot].abs();
            if candidate > best_abs {
                best = row;
                best_abs = candidate;
            }
        }
        if best_abs <= 1.0e-12 {
            return None;
        }
        if best != pivot {
            matrix.swap(best, pivot);
            rhs.swap(best, pivot);
        }
        let pivot_value = matrix[pivot][pivot];
        for col in pivot..dim {
            matrix[pivot][col] /= pivot_value;
        }
        rhs[pivot] /= pivot_value;
        for row in 0..dim {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor.abs() <= 1.0e-18 {
                continue;
            }
            for col in pivot..dim {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

#[allow(dead_code)]
fn intercept_only_weights(intercept: f64, capacity: usize) -> Vec<f64> {
    let mut weights = vec![0.0; capacity + 1];
    weights[0] = intercept;
    weights
}

#[allow(dead_code)]
fn mse_for_weights(features: &[Vec<f64>], targets: &[f64], weights: &[f64]) -> f64 {
    features
        .iter()
        .zip(targets.iter())
        .map(|(row, target)| {
            let prediction = row
                .iter()
                .zip(weights.iter())
                .map(|(feature, weight)| feature * weight)
                .sum::<f64>();
            (prediction - target).powi(2)
        })
        .sum::<f64>()
        / targets.len().max(1) as f64
}

#[allow(dead_code)]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod leakage_calibration_tests {
    use super::*;
    use tch::{nn, no_grad, Device};

    #[test]
    fn frozen_leakage_probe_calibration_capacity_sweep_reports_heldout_routes() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let mut examples = dataset.examples()[..4].to_vec();
        for (index, example) in examples.iter_mut().enumerate() {
            let scale = 1.0 + index as f64 * 0.25;
            example.geometry.pairwise_distances =
                example.geometry.pairwise_distances.shallow_clone() * scale;
        }
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = no_grad(|| system.forward_batch(&examples));

        let report = calibrate_frozen_leakage_probes_for_split(
            &examples,
            &forwards,
            "test",
            &FrozenLeakageProbeCalibrationConfig {
                train_fraction: 0.5,
                capacities: vec![1, 4],
                regularization: vec![0.0, 0.1],
                routes: vec![
                    FrozenLeakageProbeRoute::TopologyToGeometry,
                    FrozenLeakageProbeRoute::PocketToGeometry,
                ],
            },
        );

        let best = report.routes.first().expect("route report should exist");
        println!(
            "frozen_leakage_probe_smoke routes={} sweep_rows={} best_route={} heldout_mse={:.6} baseline_mse={:.6} improvement={:.6}",
            report.routes.len(),
            report.capacity_sweep.len(),
            best.route,
            best.heldout_mse,
            best.baseline_mse,
            best.improvement_over_baseline
        );
        assert_eq!(report.calibration_status, "ok");
        assert_eq!(report.split_name, "test");
        assert!(report.optimizer_penalty_separated);
        assert_eq!(report.trivial_baseline, "heldout_target_mean_mse");
        assert_eq!(report.routes.len(), 2);
        assert_eq!(report.capacity_sweep.len(), 8);
        assert!(report
            .training_time_signal
            .contains("distinct from training-time leakage penalty"));
        assert!(report.routes.iter().all(|route| route.heldout_count > 0));
        assert!(report.routes.iter().all(|route| route.split_name == "test"));
        assert!(report
            .capacity_sweep
            .iter()
            .all(|row| row.split_name == "test"));
        assert!(report
            .capacity_sweep
            .iter()
            .any(|row| row.capacity == 1 && (row.regularization - 0.1).abs() < 1.0e-12));
        assert!(report
            .capacity_sweep
            .iter()
            .all(|row| row.baseline_mse.is_finite()));
    }

    #[test]
    fn frozen_leakage_probe_calibration_reports_insufficient_data_with_split_name() {
        let config = ResearchConfig::default();
        let dataset = InMemoryDataset::new(crate::data::synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let examples = &dataset.examples()[..1];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = no_grad(|| system.forward_batch(examples));

        let report = calibrate_frozen_leakage_probes_for_split(
            examples,
            &forwards,
            "test",
            &FrozenLeakageProbeCalibrationConfig::default(),
        );

        assert_eq!(report.calibration_status, "insufficient_data");
        assert_eq!(report.split_name, "test");
        assert!(report.routes.is_empty());
        assert!(report.capacity_sweep.is_empty());
        assert!(report.claim_boundary.contains("at least two examples"));
    }

    #[test]
    fn q14_leakage_calibration_artifact_and_doc_are_parseable() {
        let artifact: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("configs/q14_leakage_calibration.json")
                .expect("q14 leakage calibration artifact should exist"),
        )
        .expect("q14 leakage calibration artifact should be valid json");
        assert_eq!(artifact["schema_version"], 1);
        assert_eq!(
            artifact["calibration_status"],
            "heuristic_smoke_calibrated"
        );
        let reports = artifact["reports"]
            .as_array()
            .expect("calibration reports should be an array");
        for family in [
            "similarity_proxy",
            "explicit_probe_penalties",
            "frozen_probe_audit",
        ] {
            assert!(
                reports
                    .iter()
                    .any(|report| report["metric_family"].as_str() == Some(family)),
                "missing calibration report family {family}"
            );
        }
        let doc = std::fs::read_to_string("docs/q14_leakage_calibration.md")
            .expect("q14 leakage calibration doc should exist");
        assert!(doc.contains("test.leakage_proxy_mean"));
        assert!(doc.contains("0.08"));
        assert!(doc.contains("frozen audit"));
        assert!(doc.contains("heuristic"));
    }
}
