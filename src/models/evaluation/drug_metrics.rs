use super::*;

use std::fs;
use std::path::Path;

/// Drug-discovery-facing metric group used for paper claims and ablation tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrugMetricDomain {
    /// Docking or score-only affinity proxy metrics such as Vina and GNINA affinity.
    Docking,
    /// Drug-likeness and synthetic accessibility metrics such as QED, LogP, and SA.
    DrugLikeness,
    /// Structural validity metrics such as validity and valence sanity.
    ChemistryValidity,
    /// Pocket-contact, clash, and geometric compatibility metrics.
    PocketCompatibility,
    /// Backend coverage and execution health metrics.
    BackendQuality,
    /// Metrics that are retained but should not drive primary claims.
    Auxiliary,
}

/// Expected optimization direction for a drug metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrugMetricDirection {
    /// Larger values are preferred.
    HigherIsBetter,
    /// Smaller values are preferred.
    LowerIsBetter,
    /// The metric is a coverage or health check, not a quality objective.
    Guardrail,
}

/// Stable description of how one backend metric should be interpreted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrugMetricSpec {
    /// Backend metric name.
    pub metric_name: String,
    /// Drug-discovery-facing metric domain.
    pub domain: DrugMetricDomain,
    /// Claim-time optimization direction.
    pub direction: DrugMetricDirection,
    /// Whether this metric may support a primary paper claim.
    pub primary_claim_metric: bool,
}

/// Explicit contract entry for metric classification.
#[allow(dead_code)] // Manifest-backed contract loading is retained for validation and claim audits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrugMetricContractEntry {
    /// Backend metric name before optional transport-side normalization.
    pub metric_name: String,
    /// Domain for domain-level summaries and claim routing.
    pub domain: DrugMetricDomain,
    /// Optimization direction intended by the claim contract.
    pub direction: DrugMetricDirection,
    /// Whether this metric may support a primary paper claim.
    pub primary_claim_metric: bool,
}

/// Contract container persisted in repository JSON assets.
#[allow(dead_code)] // Constructed by serde when validation loads the metric contract manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DrugMetricContract {
    /// Explicit metric-level interpretation rules.
    #[serde(default)]
    pub metrics: Vec<DrugMetricContractEntry>,
}

#[allow(dead_code)] // Private serde envelope for the repository metric-contract manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DrugMetricArtifactManifest {
    #[serde(default)]
    pub metric_contract: DrugMetricContract,
}

/// One numeric backend value with drug-metric semantics attached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugMetricObservation {
    /// Backend that emitted the metric.
    pub backend_name: String,
    /// Metric interpretation contract.
    pub spec: DrugMetricSpec,
    /// Numeric value emitted by the backend.
    pub value: f64,
}

/// Domain-level aggregate for a set of metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrugMetricDomainSummary {
    /// Number of observations in the domain, including non-finite values.
    pub count: usize,
    /// Number of finite observations.
    pub finite_count: usize,
    /// Mean of finite observations.
    pub mean: Option<f64>,
    /// Minimum of finite observations.
    pub min: Option<f64>,
    /// Maximum of finite observations.
    pub max: Option<f64>,
    /// Best finite value according to direction.
    pub best: Option<f64>,
}

/// A concrete reason a backend guardrail check failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrugMetricGuardrailFailure {
    /// Backend that reported the failed metric.
    pub backend_name: String,
    /// Stable metric name that failed.
    pub metric_name: String,
    /// Failing metric value.
    pub metric_value: f64,
    /// Human-readable reason for the failure.
    pub reason: String,
}

/// Thin, backend-agnostic view over chemistry, docking, and pocket reports.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DrugMetricPanel {
    /// Normalized observations from one or more backend reports.
    pub observations: Vec<DrugMetricObservation>,
}

impl DrugMetricPanel {
    /// Build a drug-metric panel from existing backend reports.
    pub fn from_backend_reports(reports: &[ExternalEvaluationReport]) -> Self {
        let observations = reports
            .iter()
            .flat_map(|report| {
                report.metrics.iter().map(|record| DrugMetricObservation {
                    backend_name: report.backend_name.clone(),
                    spec: classify_drug_metric(&record.metric_name),
                    value: record.value,
                })
            })
            .collect();
        Self { observations }
    }

    /// Return all observations in the requested domain.
    pub fn by_domain(&self, domain: DrugMetricDomain) -> Vec<&DrugMetricObservation> {
        self.observations
            .iter()
            .filter(|observation| observation.spec.domain == domain)
            .collect()
    }

    /// Return metrics that can appear in the primary paper claim surface.
    pub fn primary_claim_observations(&self) -> Vec<&DrugMetricObservation> {
        self.observations
            .iter()
            .filter(|observation| observation.spec.primary_claim_metric)
            .collect()
    }

    /// Return a lightweight domain summary for claim diagnostics.
    pub fn summary_for_domain(&self, domain: DrugMetricDomain) -> DrugMetricDomainSummary {
        let observations = self.by_domain(domain);
        let count = observations.len();
        let finite = observations
            .iter()
            .filter(|obs| obs.value.is_finite())
            .collect::<Vec<_>>();
        let finite_count = finite.len();
        let mut min = None;
        let mut max = None;
        let mut sum = 0.0;
        for obs in &finite {
            let value = obs.value;
            if let Some(current_min) = min {
                if value < current_min {
                    min = Some(value);
                }
            } else {
                min = Some(value);
            }
            if let Some(current_max) = max {
                if value > current_max {
                    max = Some(value);
                }
            } else {
                max = Some(value);
            }
            sum += value;
        }
        let mean = if finite_count > 0 {
            Some(sum / (finite_count as f64))
        } else {
            None
        };
        let best = if finite_count == 0 {
            None
        } else {
            let direction = finite[0].spec.direction;
            if direction == DrugMetricDirection::Guardrail {
                None
            } else if finite.iter().all(|obs| obs.spec.direction == direction) {
                let mut best_value = finite[0].value;
                for obs in finite.iter().skip(1) {
                    let value = obs.value;
                    match direction {
                        DrugMetricDirection::HigherIsBetter => {
                            if value > best_value {
                                best_value = value;
                            }
                        }
                        DrugMetricDirection::LowerIsBetter => {
                            if value < best_value {
                                best_value = value;
                            }
                        }
                        DrugMetricDirection::Guardrail => {}
                    }
                }
                Some(best_value)
            } else {
                None
            }
        };

        DrugMetricDomainSummary {
            count,
            finite_count,
            mean,
            min,
            max,
            best,
        }
    }

    /// Return true when required backend health metrics do not report obvious failure.
    pub fn passes_backend_guardrails(&self) -> bool {
        self.guardrail_failures().is_empty()
    }

    /// Return explicit backend guardrail failures with enough evidence for claim-time reporting.
    pub fn guardrail_failures(&self) -> Vec<DrugMetricGuardrailFailure> {
        self.observations
            .iter()
            .filter_map(|observation| {
                if observation.spec.domain != DrugMetricDomain::BackendQuality {
                    return None;
                }
                if !observation.value.is_finite() {
                    return Some(DrugMetricGuardrailFailure {
                        backend_name: observation.backend_name.clone(),
                        metric_name: observation.spec.metric_name.clone(),
                        metric_value: observation.value,
                        reason: "backend quality metric value must be finite".to_string(),
                    });
                }

                let metric_name = observation.spec.metric_name.as_str();
                if (metric_name.contains("missing")
                    || metric_name.contains("failed")
                    || metric_name.contains("error"))
                    && observation.value != 0.0
                {
                    Some(DrugMetricGuardrailFailure {
                        backend_name: observation.backend_name.clone(),
                        metric_name: observation.spec.metric_name.clone(),
                        metric_value: observation.value,
                        reason: "backend quality metric is expected to be zero".to_string(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Classify a backend metric without coupling evaluators to paper-specific tables.
pub fn classify_drug_metric(metric_name: &str) -> DrugMetricSpec {
    let normalized = normalize_metric_name(metric_name);
    classify_drug_metric_legacy(&normalized)
}

/// Classify a metric from the contract, returning None for unknown names.
#[allow(dead_code)] // Exercised by contract-validation tests and kept as the manifest migration path.
pub fn classify_drug_metric_from_contract(
    metric_name: &str,
    contract: &DrugMetricContract,
) -> Option<DrugMetricSpec> {
    let normalized = normalize_metric_name(metric_name);
    contract
        .metrics
        .iter()
        .find(|entry| entry.metric_name == normalized)
        .map(|entry| DrugMetricSpec {
            metric_name: normalized.clone(),
            domain: entry.domain,
            direction: entry.direction,
            primary_claim_metric: entry.primary_claim_metric,
        })
}

/// Load the manifest-embedded metric contract from a filesystem path.
#[allow(dead_code)] // Used by validation-oriented tests; runtime classification still has a fallback.
pub fn load_drug_metric_contract(
    manifest_path: impl AsRef<Path>,
) -> Result<DrugMetricContract, String> {
    let content = fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "failed to read {path}: {error}",
            path = manifest_path.as_ref().display()
        )
    })?;
    let manifest: DrugMetricArtifactManifest = serde_json::from_str(&content)
        .map_err(|error| format!("failed to parse metric contract: {error}"))?;
    Ok(manifest.metric_contract)
}

/// Canonical source metric name for contract lookup and contract-test diagnostics.
fn normalize_metric_name(metric_name: &str) -> String {
    metric_name
        .strip_prefix("candidate_metric|")
        .and_then(|raw| raw.rsplit('|').next())
        .unwrap_or(metric_name)
        .to_string()
}

/// Hard-coded classifier retained as runtime fallback when contracts are not loaded.
fn classify_drug_metric_legacy(metric_name: &str) -> DrugMetricSpec {
    let (domain, direction, primary_claim_metric) = match metric_name {
        "vina_score" | "gnina_affinity" => (
            DrugMetricDomain::Docking,
            DrugMetricDirection::LowerIsBetter,
            true,
        ),
        "gnina_cnn_score" | "qed" | "logp" => (
            DrugMetricDomain::DrugLikeness,
            DrugMetricDirection::HigherIsBetter,
            metric_name == "qed",
        ),
        "sa_score" => (
            DrugMetricDomain::DrugLikeness,
            DrugMetricDirection::LowerIsBetter,
            true,
        ),
        "valid_fraction" | "valence_sanity_fraction" | "structural_pass_fraction" => (
            DrugMetricDomain::ChemistryValidity,
            DrugMetricDirection::HigherIsBetter,
            true,
        ),
        "pocket_contact_fraction"
        | "centroid_inside_fraction"
        | "atom_coverage_fraction"
        | "clash_free_fraction"
        | "strict_pocket_fit_score"
        | "centroid_fit_score"
        | "pharmacophore_role_coverage"
        | "key_residue_contact_coverage" => (
            DrugMetricDomain::PocketCompatibility,
            DrugMetricDirection::HigherIsBetter,
            false,
        ),
        "clash_fraction"
        | "mean_centroid_offset"
        | "role_conflict_rate"
        | "severe_clash_fraction" => (
            DrugMetricDomain::PocketCompatibility,
            DrugMetricDirection::LowerIsBetter,
            false,
        ),
        "valence_violation_fraction" | "bond_length_guardrail_mean" => (
            DrugMetricDomain::ChemistryValidity,
            DrugMetricDirection::LowerIsBetter,
            false,
        ),
        name if name.starts_with("gate_usage_by_chemical_role") => (
            DrugMetricDomain::Auxiliary,
            DrugMetricDirection::Guardrail,
            false,
        ),
        name if name.contains("coverage_fraction")
            || name.contains("success_fraction")
            || name.contains("schema_version")
            || name.contains("examples_scored")
            || name.contains("backend_")
            || name.contains("candidate_metric_rows") =>
        {
            (
                DrugMetricDomain::BackendQuality,
                DrugMetricDirection::Guardrail,
                false,
            )
        }
        _ => (
            DrugMetricDomain::Auxiliary,
            DrugMetricDirection::Guardrail,
            false,
        ),
    };

    DrugMetricSpec {
        metric_name: metric_name.to_string(),
        domain,
        direction,
        primary_claim_metric,
    }
}

#[cfg(test)]
mod drug_metric_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn drug_metric_panel_separates_claim_metrics_from_guardrails() {
        let reports = vec![ExternalEvaluationReport {
            backend_name: "merged_backend".to_string(),
            metrics: vec![
                ExternalMetricRecord {
                    metric_name: "candidate_metric|ex|prot|cand|vina_score".to_string(),
                    value: -7.4,
                },
                ExternalMetricRecord {
                    metric_name: "qed".to_string(),
                    value: 0.63,
                },
                ExternalMetricRecord {
                    metric_name: "backend_missing_schema_version".to_string(),
                    value: 0.0,
                },
            ],
        }];

        let panel = DrugMetricPanel::from_backend_reports(&reports);
        assert_eq!(panel.by_domain(DrugMetricDomain::Docking).len(), 1);
        assert_eq!(panel.primary_claim_observations().len(), 2);
        assert!(panel.passes_backend_guardrails());
        assert_eq!(
            classify_drug_metric("candidate_metric|a|b|c|vina_score").direction,
            DrugMetricDirection::LowerIsBetter
        );
    }

    #[test]
    fn drug_metric_panel_summary_respects_directional_best() {
        let reports = vec![ExternalEvaluationReport {
            backend_name: "vina_backend".to_string(),
            metrics: vec![
                ExternalMetricRecord {
                    metric_name: "vina_score".to_string(),
                    value: -6.0,
                },
                ExternalMetricRecord {
                    metric_name: "vina_score".to_string(),
                    value: -8.0,
                },
            ],
        }];
        let panel = DrugMetricPanel::from_backend_reports(&reports);
        let docking_summary = panel.summary_for_domain(DrugMetricDomain::Docking);
        assert_eq!(docking_summary.count, 2);
        assert_eq!(docking_summary.finite_count, 2);
        assert_eq!(docking_summary.min, Some(-8.0));
        assert_eq!(docking_summary.max, Some(-6.0));
        assert_eq!(docking_summary.best, Some(-8.0));
        assert!((docking_summary.mean.unwrap() + 7.0).abs() < 1e-12);
    }

    #[test]
    fn drug_metric_panel_summary_prefers_higher_for_qed() {
        let reports = vec![ExternalEvaluationReport {
            backend_name: "rdkit_backend".to_string(),
            metrics: vec![
                ExternalMetricRecord {
                    metric_name: "qed".to_string(),
                    value: 0.42,
                },
                ExternalMetricRecord {
                    metric_name: "qed".to_string(),
                    value: 0.73,
                },
            ],
        }];
        let panel = DrugMetricPanel::from_backend_reports(&reports);
        let drug_summary = panel.summary_for_domain(DrugMetricDomain::DrugLikeness);
        assert_eq!(drug_summary.best, Some(0.73));
        assert_eq!(drug_summary.min, Some(0.42));
        assert_eq!(drug_summary.max, Some(0.73));
    }

    #[test]
    fn drug_metric_panel_guardrail_failures_track_backend_and_metric() {
        let reports = vec![ExternalEvaluationReport {
            backend_name: "docking_backend".to_string(),
            metrics: vec![
                ExternalMetricRecord {
                    metric_name: "backend_missing_structure_fraction".to_string(),
                    value: 0.2,
                },
                ExternalMetricRecord {
                    metric_name: "backend_coverage".to_string(),
                    value: 0.7,
                },
            ],
        }];
        let panel = DrugMetricPanel::from_backend_reports(&reports);
        let failures = panel.guardrail_failures();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].backend_name, "docking_backend");
        assert_eq!(
            failures[0].metric_name,
            "backend_missing_structure_fraction"
        );
        assert!(!panel.passes_backend_guardrails());
    }

    fn contract_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("configs/drug_metric_artifact_manifest.json")
    }

    #[test]
    fn drug_metric_contract_matches_legacy_classifier() {
        let contract =
            load_drug_metric_contract(contract_path()).expect("unable to load metric contract");
        for spec in contract.metrics {
            let resolved = classify_drug_metric(&spec.metric_name);
            assert_eq!(
                (
                    resolved.domain,
                    resolved.direction,
                    resolved.primary_claim_metric
                ),
                (spec.domain, spec.direction, spec.primary_claim_metric),
                "contract metric `{}` drifted from Rust fallback rules",
                spec.metric_name
            );
        }
    }

    #[test]
    fn classify_from_contract_prefers_explicit_contract_entry() {
        let contract = DrugMetricContract {
            metrics: vec![DrugMetricContractEntry {
                metric_name: "vina_score".to_string(),
                domain: DrugMetricDomain::ChemistryValidity,
                direction: DrugMetricDirection::HigherIsBetter,
                primary_claim_metric: true,
            }],
        };
        let resolved =
            classify_drug_metric_from_contract("candidate_metric|ex|a|b|vina_score", &contract)
                .expect("contract metric should resolve");
        assert_eq!(resolved.domain, DrugMetricDomain::ChemistryValidity);
        assert_eq!(resolved.direction, DrugMetricDirection::HigherIsBetter);
    }

    #[test]
    fn contract_primary_claim_metric_must_be_known_to_legacy_classifier() {
        let synthetic = DrugMetricContractEntry {
            metric_name: "mystery_affinity_score".to_string(),
            domain: DrugMetricDomain::Docking,
            direction: DrugMetricDirection::LowerIsBetter,
            primary_claim_metric: true,
        };
        let fallback = classify_drug_metric(&synthetic.metric_name);
        assert_ne!(
            (
                fallback.domain,
                fallback.direction,
                fallback.primary_claim_metric
            ),
            (
                synthetic.domain,
                synthetic.direction,
                synthetic.primary_claim_metric
            ),
            "synthetic contract entry illustrates the drift path for unknown primary metrics"
        );
    }

    #[test]
    fn chemistry_collaboration_metrics_are_claim_safe_auxiliary_or_guardrails() {
        let coverage = classify_drug_metric("pharmacophore_role_coverage");
        assert_eq!(coverage.domain, DrugMetricDomain::PocketCompatibility);
        assert_eq!(coverage.direction, DrugMetricDirection::HigherIsBetter);
        assert!(!coverage.primary_claim_metric);

        let valence = classify_drug_metric("valence_violation_fraction");
        assert_eq!(valence.domain, DrugMetricDomain::ChemistryValidity);
        assert_eq!(valence.direction, DrugMetricDirection::LowerIsBetter);
        assert!(!valence.primary_claim_metric);

        let gate = classify_drug_metric("gate_usage_by_chemical_role::pocket_shape");
        assert_eq!(gate.domain, DrugMetricDomain::Auxiliary);
        assert_eq!(gate.direction, DrugMetricDirection::Guardrail);
        assert!(!gate.primary_claim_metric);
    }
}
