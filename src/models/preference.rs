//! Interaction preference contracts built on top of generated candidate layers.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::models::traits::{
    CandidateLayerKind, ExternalEvaluationReport, GeneratedCandidateRecord,
};

/// Current schema version for serialized interaction profiles.
pub const INTERACTION_PROFILE_SCHEMA_VERSION: u32 = 1;
/// Current schema version for serialized preference pairs.
pub const PREFERENCE_PAIR_SCHEMA_VERSION: u32 = 1;
/// Current schema version for preference-aware reranker summaries.
pub const PREFERENCE_RERANKER_SCHEMA_VERSION: u32 = 1;

/// Persisted envelope for split-level interaction-profile artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreferenceProfileArtifact {
    /// Schema version for the artifact envelope.
    pub schema_version: u32,
    /// Split label such as `validation` or `test`.
    pub split: String,
    /// Number of profile records contained in this artifact.
    pub profile_count: usize,
    /// Number of distinct method ids represented by records.
    pub method_id_coverage: usize,
    /// Per-layer profile counts keyed by legacy layer name.
    pub layer_coverage: BTreeMap<String, usize>,
    /// Per-backend availability counts keyed by backend name.
    pub backend_coverage: BTreeMap<String, usize>,
    /// Profile records. Empty means no evidence was available for this split.
    #[serde(default)]
    pub records: Vec<InteractionProfile>,
}

impl PreferenceProfileArtifact {
    /// Build a split-level artifact envelope from profile records.
    pub fn new(split: impl Into<String>, records: Vec<InteractionProfile>) -> Self {
        let mut method_ids = BTreeSet::new();
        let mut layer_coverage = BTreeMap::new();
        let mut backend_coverage = BTreeMap::new();
        for record in &records {
            if let Some(method_id) = &record.method_id {
                method_ids.insert(method_id.clone());
            }
            *layer_coverage
                .entry(record.layer_kind.legacy_field_name().to_string())
                .or_insert(0) += 1;
            for (backend, status) in &record.backend_status {
                if status == "metrics_available" {
                    *backend_coverage.entry(backend.clone()).or_insert(0) += 1;
                }
            }
        }
        Self {
            schema_version: INTERACTION_PROFILE_SCHEMA_VERSION,
            split: split.into(),
            profile_count: records.len(),
            method_id_coverage: method_ids.len(),
            layer_coverage,
            backend_coverage,
            records,
        }
    }
}

/// Persisted envelope for split-level preference-pair artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreferencePairArtifact {
    /// Schema version for the artifact envelope.
    pub schema_version: u32,
    /// Split label such as `validation` or `test`.
    pub split: String,
    /// Number of pair records contained in this artifact.
    pub pair_count: usize,
    /// Number of distinct methods represented by source profiles when known.
    pub method_id_coverage: usize,
    /// Pair counts keyed by preference source.
    pub source_coverage: BTreeMap<String, usize>,
    /// Fraction of pairs with backend-supported source.
    #[serde(default)]
    pub backend_supported_pair_fraction: f64,
    /// Fraction of pairs that remain rule-only.
    #[serde(default)]
    pub rule_only_pair_fraction: f64,
    /// Fraction of pairs without backend-supported source.
    #[serde(default)]
    pub missing_backend_evidence_fraction: f64,
    /// Mean normalized preference strength across persisted pairs.
    #[serde(default)]
    pub mean_preference_strength: f64,
    /// Fraction of pairs where any hard-constraint flag was active.
    #[serde(default)]
    pub hard_constraint_win_fraction: f64,
    /// Pair records. Empty means no pair evidence was available for this split.
    #[serde(default)]
    pub records: Vec<PreferencePair>,
}

impl PreferencePairArtifact {
    /// Build a split-level pair artifact envelope from pair records.
    pub fn new(
        split: impl Into<String>,
        profiles: &[InteractionProfile],
        records: Vec<PreferencePair>,
    ) -> Self {
        let method_id_coverage = profiles
            .iter()
            .filter_map(|profile| profile.method_id.clone())
            .collect::<BTreeSet<_>>()
            .len();
        let mut source_coverage = BTreeMap::new();
        let mut strength_sum = 0.0;
        let mut hard_constraint_wins = 0usize;
        for record in &records {
            *source_coverage
                .entry(preference_source_label(record.preference_source).to_string())
                .or_insert(0) += 1;
            strength_sum += record.preference_strength;
            if record.hard_constraint_flags.values().any(|value| *value) {
                hard_constraint_wins += 1;
            }
        }
        let pair_count = records.len() as f64;
        let backend_pairs = source_coverage
            .get("backend_based")
            .copied()
            .unwrap_or_default() as f64;
        let rule_pairs = source_coverage
            .get("rule_based")
            .copied()
            .unwrap_or_default() as f64;
        Self {
            schema_version: PREFERENCE_PAIR_SCHEMA_VERSION,
            split: split.into(),
            pair_count: records.len(),
            method_id_coverage,
            source_coverage,
            backend_supported_pair_fraction: if pair_count > 0.0 {
                backend_pairs / pair_count
            } else {
                0.0
            },
            rule_only_pair_fraction: if pair_count > 0.0 {
                rule_pairs / pair_count
            } else {
                0.0
            },
            missing_backend_evidence_fraction: if pair_count > 0.0 {
                1.0 - (backend_pairs / pair_count)
            } else {
                0.0
            },
            mean_preference_strength: if pair_count > 0.0 {
                strength_sum / pair_count
            } else {
                0.0
            },
            hard_constraint_win_fraction: if pair_count > 0.0 {
                hard_constraint_wins as f64 / pair_count
            } else {
                0.0
            },
            records,
        }
    }
}

fn preference_source_label(source: PreferenceSource) -> &'static str {
    match source {
        PreferenceSource::RuleBased => "rule_based",
        PreferenceSource::BackendBased => "backend_based",
        PreferenceSource::HumanCurated => "human_curated",
        PreferenceSource::FutureDocking => "future_docking",
        PreferenceSource::FutureExperimental => "future_experimental",
    }
}

/// Compact persisted summary for preference-aware reranking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreferenceRerankerSummaryArtifact {
    /// Schema version for this summary.
    pub schema_version: u32,
    /// Whether preference-aware reranking was enabled.
    pub enabled: bool,
    /// Number of candidate profiles considered.
    pub candidate_count: usize,
    /// Number of candidates selected by the preference reranker.
    pub selected_count: usize,
    /// Rule/feature weights used by the deterministic scorer.
    pub feature_weights: BTreeMap<String, f64>,
    /// Fraction of pairs with rule-based source when pair artifacts exist.
    pub rule_based_pair_fraction: f64,
    /// Fraction of pairs with backend-supported source when pair artifacts exist.
    pub backend_based_pair_fraction: f64,
    /// Number of hard-constraint wins observed in pair evidence.
    pub hard_constraint_wins: usize,
    /// Fraction of pairs with backend-supported source when pair artifacts exist.
    #[serde(default)]
    pub backend_supported_pair_fraction: f64,
    /// Fraction of pairs with rule-only source when pair artifacts exist.
    #[serde(default)]
    pub rule_only_pair_fraction: f64,
    /// Fraction of pairs without backend-supported source.
    #[serde(default)]
    pub missing_backend_evidence_fraction: f64,
    /// Mean normalized preference strength across pair evidence.
    #[serde(default)]
    pub mean_preference_strength: f64,
    /// Fraction of pairs where any hard-constraint flag was active.
    #[serde(default)]
    pub hard_constraint_win_fraction: f64,
    /// Number of soft-preference wins observed in pair evidence.
    pub soft_preference_wins: usize,
}

/// Provenance class for one interaction feature.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum InteractionFeatureProvenance {
    /// Deterministic heuristic computed from candidate payloads.
    HeuristicProxy,
    /// Metric supplied by an external backend.
    ExternalBackend,
    /// Value copied or derived directly from candidate metadata.
    DerivedFromCandidate,
    /// Reserved feature unavailable for the current profile.
    Unavailable,
}

/// Typed value plus provenance for an interaction-profile feature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionFeatureValue {
    /// Feature value. `None` means explicitly unavailable, not zero.
    pub value: Option<f64>,
    /// Evidence source for this feature value.
    pub provenance: InteractionFeatureProvenance,
}

impl InteractionFeatureValue {
    fn available(value: f64, provenance: InteractionFeatureProvenance) -> Self {
        Self {
            value: value.is_finite().then_some(value),
            provenance,
        }
    }

    fn unavailable() -> Self {
        Self {
            value: None,
            provenance: InteractionFeatureProvenance::Unavailable,
        }
    }
}

/// Per-candidate interaction profile used by preference construction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionProfile {
    /// Schema version for artifact compatibility.
    pub schema_version: u32,
    /// Stable candidate identifier within the layer artifact.
    pub candidate_id: String,
    /// Stable example identifier.
    pub example_id: String,
    /// Protein identifier for pocket-conditioned comparison.
    pub protein_id: String,
    /// Candidate source label.
    pub candidate_source: String,
    /// Candidate layer that supplied this profile.
    pub layer_kind: CandidateLayerKind,
    /// Optional method id when emitted by a method-aware generation path.
    #[serde(default)]
    pub method_id: Option<String>,
    /// Whether any external backend metrics were attached.
    pub backend_evidence_available: bool,
    /// Backend status labels keyed by backend name.
    #[serde(default)]
    pub backend_status: BTreeMap<String, String>,
    /// Conservative missing-structure fraction reported by a backend, when available.
    pub backend_missing_structure_fraction: InteractionFeatureValue,
    /// Fraction of candidate atoms contacting the pocket envelope proxy.
    pub pocket_contact_fraction: InteractionFeatureValue,
    /// Non-bonded clash proxy; lower is better.
    pub clash_fraction: InteractionFeatureValue,
    /// Ligand centroid offset from the pocket centroid; lower is better.
    pub centroid_offset: InteractionFeatureValue,
    /// Multiplicative strict pocket-fit proxy.
    pub strict_pocket_fit_score: InteractionFeatureValue,
    /// Fraction of atoms inside the expanded pocket envelope.
    pub atom_coverage_fraction: InteractionFeatureValue,
    /// RDKit validity when a chemistry backend reports it.
    pub rdkit_valid: InteractionFeatureValue,
    /// RDKit sanitization when a chemistry backend reports it.
    pub rdkit_sanitized: InteractionFeatureValue,
    /// RDKit unique-SMILES signal when a chemistry backend reports it.
    pub rdkit_unique_smiles: InteractionFeatureValue,
    /// Bond-density compatibility proxy.
    pub bond_density_fit: InteractionFeatureValue,
    /// Valence sanity proxy.
    pub valence_sane: InteractionFeatureValue,
    /// Reserved hydrophobic-contact proxy.
    pub hydrophobic_contact_proxy: InteractionFeatureValue,
    /// Reserved hydrogen-bond proxy.
    pub hydrogen_bond_proxy: InteractionFeatureValue,
    /// Reserved key-residue-contact proxy.
    pub key_residue_contact_proxy: InteractionFeatureValue,
}

impl InteractionProfile {
    /// Build a deterministic fallback candidate id from layer, method, and position.
    pub fn candidate_id_for(
        layer_kind: CandidateLayerKind,
        method_id: Option<&str>,
        index: usize,
        candidate: &GeneratedCandidateRecord,
    ) -> String {
        format!(
            "{}:{}:{}:{}",
            method_id.unwrap_or("legacy"),
            layer_kind.legacy_field_name(),
            candidate.example_id,
            index
        )
    }
}

/// Context passed to an interaction-profile extractor.
#[derive(Debug, Clone, Copy)]
pub struct InteractionProfileExtractionContext<'a> {
    /// Stable candidate id.
    pub candidate_id: &'a str,
    /// Candidate layer being profiled.
    pub layer_kind: CandidateLayerKind,
    /// Optional method id from method-aware generation.
    pub method_id: Option<&'a str>,
    /// Optional backend reports available to the extraction pass.
    pub backend_reports: &'a [ExternalEvaluationReport],
}

/// Replaceable interface for heuristic, backend-backed, or docking-backed profile extraction.
pub trait InteractionProfileExtractor {
    /// Extract a compact interaction profile for one candidate.
    fn extract_profile(
        &self,
        candidate: &GeneratedCandidateRecord,
        context: InteractionProfileExtractionContext<'_>,
    ) -> InteractionProfile;
}

/// Deterministic extractor using candidate payloads plus optional backend metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicInteractionProfileExtractor;

impl InteractionProfileExtractor for HeuristicInteractionProfileExtractor {
    fn extract_profile(
        &self,
        candidate: &GeneratedCandidateRecord,
        context: InteractionProfileExtractionContext<'_>,
    ) -> InteractionProfile {
        let backend = BackendMetricContext::from_reports(context.backend_reports);
        let backend_missing_structure_fraction = backend
            .metric("backend_missing_structure_fraction")
            .map(|value| {
                InteractionFeatureValue::available(
                    value,
                    InteractionFeatureProvenance::ExternalBackend,
                )
            })
            .unwrap_or_else(InteractionFeatureValue::unavailable);
        let rdkit_valid = backend
            .metric_any(&["rdkit_valid_fraction", "valid_fraction"])
            .map(|value| {
                InteractionFeatureValue::available(
                    value,
                    InteractionFeatureProvenance::ExternalBackend,
                )
            })
            .unwrap_or_else(InteractionFeatureValue::unavailable);
        let rdkit_sanitized = backend
            .metric_any(&["rdkit_sanitized_fraction", "sanitized_fraction"])
            .map(|value| {
                InteractionFeatureValue::available(
                    value,
                    InteractionFeatureProvenance::ExternalBackend,
                )
            })
            .unwrap_or_else(InteractionFeatureValue::unavailable);
        let rdkit_unique_smiles = backend
            .metric_any(&["rdkit_unique_smiles_fraction", "unique_smiles_fraction"])
            .map(|value| {
                InteractionFeatureValue::available(
                    value,
                    InteractionFeatureProvenance::ExternalBackend,
                )
            })
            .unwrap_or_else(InteractionFeatureValue::unavailable);
        InteractionProfile {
            schema_version: INTERACTION_PROFILE_SCHEMA_VERSION,
            candidate_id: context.candidate_id.to_string(),
            example_id: candidate.example_id.clone(),
            protein_id: candidate.protein_id.clone(),
            candidate_source: candidate.source.clone(),
            layer_kind: context.layer_kind,
            method_id: context.method_id.map(str::to_string),
            backend_evidence_available: backend.available,
            backend_status: backend.status,
            backend_missing_structure_fraction,
            pocket_contact_fraction: InteractionFeatureValue::available(
                pocket_contact_fraction(candidate),
                InteractionFeatureProvenance::HeuristicProxy,
            ),
            clash_fraction: InteractionFeatureValue::available(
                clash_fraction(candidate),
                InteractionFeatureProvenance::HeuristicProxy,
            ),
            centroid_offset: InteractionFeatureValue::available(
                centroid_offset(candidate),
                InteractionFeatureProvenance::HeuristicProxy,
            ),
            strict_pocket_fit_score: InteractionFeatureValue::available(
                strict_pocket_fit_score(candidate),
                InteractionFeatureProvenance::HeuristicProxy,
            ),
            atom_coverage_fraction: InteractionFeatureValue::available(
                atom_coverage_fraction(candidate),
                InteractionFeatureProvenance::HeuristicProxy,
            ),
            rdkit_valid,
            rdkit_sanitized,
            rdkit_unique_smiles,
            bond_density_fit: InteractionFeatureValue::available(
                bond_density_fit(candidate),
                InteractionFeatureProvenance::HeuristicProxy,
            ),
            valence_sane: InteractionFeatureValue::available(
                if valence_sane(candidate) { 1.0 } else { 0.0 },
                InteractionFeatureProvenance::HeuristicProxy,
            ),
            hydrophobic_contact_proxy: InteractionFeatureValue::unavailable(),
            hydrogen_bond_proxy: InteractionFeatureValue::unavailable(),
            key_residue_contact_proxy: InteractionFeatureValue::unavailable(),
        }
    }
}

/// Extract profiles for a homogeneous candidate layer.
pub fn extract_interaction_profiles(
    candidates: &[GeneratedCandidateRecord],
    layer_kind: CandidateLayerKind,
    method_id: Option<&str>,
    backend_reports: &[ExternalEvaluationReport],
) -> Vec<InteractionProfile> {
    let extractor = HeuristicInteractionProfileExtractor;
    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let candidate_id =
                InteractionProfile::candidate_id_for(layer_kind, method_id, index, candidate);
            extractor.extract_profile(
                candidate,
                InteractionProfileExtractionContext {
                    candidate_id: &candidate_id,
                    layer_kind,
                    method_id,
                    backend_reports,
                },
            )
        })
        .collect()
}

/// Source label for an explicit preference pair.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceSource {
    /// Rule-based pair built from hard constraints and documented soft preferences.
    RuleBased,
    /// Pair backed by external chemistry, pocket, or docking metrics.
    BackendBased,
    /// Reserved human-curated preference label.
    HumanCurated,
    /// Reserved future docking preference source.
    FutureDocking,
    /// Reserved future experimental outcome source.
    FutureExperimental,
}

/// Structured reason codes for preference pair construction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceReasonCode {
    ValidCandidate,
    RdkitValid,
    RdkitSanitized,
    LowerClash,
    BetterPocketContact,
    LowerCentroidOffset,
    BetterStrictPocketFit,
    BetterAtomCoverage,
    BetterBondDensity,
    ValenceSane,
    BackendEvidenceAvailable,
}

/// Auditable winner/loser pair for future preference learning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreferencePair {
    /// Schema version for artifact compatibility.
    pub schema_version: u32,
    /// Stable pair id.
    pub pair_id: String,
    /// Stable example id.
    pub example_id: String,
    /// Protein id.
    pub protein_id: String,
    /// Winner candidate id.
    pub winner_candidate_id: String,
    /// Loser candidate id.
    pub loser_candidate_id: String,
    /// Structured preference reasons.
    pub preference_reason: Vec<PreferenceReasonCode>,
    /// Evidence source for the preference.
    pub preference_source: PreferenceSource,
    /// Normalized strength in `[0, 1]`.
    pub preference_strength: f64,
    /// Signed feature deltas as winner minus loser.
    pub feature_deltas: BTreeMap<String, f64>,
    /// Hard constraints that affected the decision.
    pub hard_constraint_flags: BTreeMap<String, bool>,
    /// Soft preference flags that affected the decision.
    pub soft_preference_flags: BTreeMap<String, bool>,
    /// Evidence coverage labels and values.
    pub evidence_coverage: BTreeMap<String, f64>,
}

/// Hard and soft controls for pair construction and preference-aware reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceConstructionConfig {
    /// Reject candidates with finite clash fraction above this threshold.
    pub max_clash_fraction: f64,
    /// Require at least this strict pocket-fit score for hard-pass decisions.
    pub min_strict_pocket_fit_score: f64,
    /// Require valence sanity as a hard constraint.
    pub require_valence_sane: bool,
    /// Minimum absolute soft-score margin required to emit a pair.
    pub min_soft_margin: f64,
}

impl Default for PreferenceConstructionConfig {
    fn default() -> Self {
        Self {
            max_clash_fraction: 0.10,
            min_strict_pocket_fit_score: 0.35,
            require_valence_sane: true,
            min_soft_margin: 0.05,
        }
    }
}

/// Builder contract for producing explainable preference pairs from profiles.
pub trait PreferenceDatasetBuilder {
    /// Build pair records from a homogeneous set of profiles.
    fn build_pairs(&self, profiles: &[InteractionProfile]) -> Vec<PreferencePair>;
}

/// Deterministic rule-based preference pair builder.
#[derive(Debug, Clone)]
pub struct RuleBasedPreferenceDatasetBuilder {
    config: PreferenceConstructionConfig,
}

impl RuleBasedPreferenceDatasetBuilder {
    /// Create a builder with explicit thresholds.
    pub fn new(config: PreferenceConstructionConfig) -> Self {
        Self { config }
    }
}

impl Default for RuleBasedPreferenceDatasetBuilder {
    fn default() -> Self {
        Self::new(PreferenceConstructionConfig::default())
    }
}

impl PreferenceDatasetBuilder for RuleBasedPreferenceDatasetBuilder {
    fn build_pairs(&self, profiles: &[InteractionProfile]) -> Vec<PreferencePair> {
        let mut pairs = Vec::new();
        for left in 0..profiles.len() {
            for right in (left + 1)..profiles.len() {
                if let Some(pair) = build_pair(&profiles[left], &profiles[right], &self.config) {
                    pairs.push(pair);
                }
            }
        }
        pairs
    }
}

/// Build deterministic preference pairs grouped by example/protein and bounded per group.
pub fn build_bounded_preference_pairs(
    profiles: &[InteractionProfile],
    config: PreferenceConstructionConfig,
    max_pairs_per_example: usize,
) -> Vec<PreferencePair> {
    let mut grouped: BTreeMap<(&str, &str), Vec<InteractionProfile>> = BTreeMap::new();
    for profile in profiles {
        grouped
            .entry((profile.example_id.as_str(), profile.protein_id.as_str()))
            .or_default()
            .push(profile.clone());
    }
    let builder = RuleBasedPreferenceDatasetBuilder::new(config);
    let mut pairs = Vec::new();
    for (_, mut group) in grouped {
        group.sort_by(|left, right| {
            left.candidate_id
                .cmp(&right.candidate_id)
                .then_with(|| left.layer_kind.cmp(&right.layer_kind))
        });
        let mut group_pairs = builder.build_pairs(&group);
        group_pairs.sort_by(|left, right| {
            right
                .preference_strength
                .partial_cmp(&left.preference_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.pair_id.cmp(&right.pair_id))
        });
        pairs.extend(group_pairs.into_iter().take(max_pairs_per_example));
    }
    pairs
}

/// Interface for preference-aware candidate reranking.
pub trait PreferenceReranker {
    /// Rank candidates from best to worst using extracted interaction profiles.
    fn rerank_profiles<'a>(
        &self,
        profiles: &'a [InteractionProfile],
    ) -> Vec<&'a InteractionProfile>;
}

/// Lightweight deterministic preference-aware reranker over profile features.
#[derive(Debug, Clone, Default)]
pub struct RuleBasedPreferenceReranker {
    config: PreferenceConstructionConfig,
}

impl RuleBasedPreferenceReranker {
    /// Create a reranker with explicit construction/scoring controls.
    pub fn new(config: PreferenceConstructionConfig) -> Self {
        Self { config }
    }

    /// Rerank candidate records by first extracting deterministic profiles.
    pub fn rerank_candidates(
        &self,
        candidates: &[GeneratedCandidateRecord],
        layer_kind: CandidateLayerKind,
        method_id: Option<&str>,
        backend_reports: &[ExternalEvaluationReport],
    ) -> Vec<GeneratedCandidateRecord> {
        let profiles =
            extract_interaction_profiles(candidates, layer_kind, method_id, backend_reports);
        let by_id = candidates
            .iter()
            .zip(profiles.iter())
            .map(|(candidate, profile)| (profile.candidate_id.clone(), candidate))
            .collect::<BTreeMap<_, _>>();
        self.rerank_profiles(&profiles)
            .into_iter()
            .filter_map(|profile| {
                by_id
                    .get(&profile.candidate_id)
                    .map(|candidate| (*candidate).clone())
            })
            .collect()
    }

    /// Expose the deterministic scoring weights used for reviewer summaries.
    pub fn feature_weights() -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("rdkit_valid".to_string(), 0.20),
            ("rdkit_sanitized".to_string(), 0.15),
            ("pocket_contact_fraction".to_string(), 0.15),
            ("strict_pocket_fit_score".to_string(), 0.20),
            ("atom_coverage_fraction".to_string(), 0.10),
            ("clash_free".to_string(), 0.10),
            ("bond_density_fit".to_string(), 0.05),
            ("valence_sane".to_string(), 0.05),
        ])
    }
}

impl PreferenceReranker for RuleBasedPreferenceReranker {
    fn rerank_profiles<'a>(
        &self,
        profiles: &'a [InteractionProfile],
    ) -> Vec<&'a InteractionProfile> {
        let mut ranked = profiles.iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            preference_score(right, &self.config)
                .partial_cmp(&preference_score(left, &self.config))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        });
        ranked
    }
}

/// Reserved trainer interface; current phase intentionally does not implement generator DPO/RL.
pub trait PreferenceTrainer {
    /// Return a stable trainer id for future experiment wiring.
    fn trainer_id(&self) -> &'static str;

    /// Report whether this trainer is active. Current stubs should return false.
    fn is_enabled(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct BackendMetricContext {
    available: bool,
    metrics: BTreeMap<String, f64>,
    status: BTreeMap<String, String>,
}

impl BackendMetricContext {
    fn from_reports(reports: &[ExternalEvaluationReport]) -> Self {
        let mut metrics = BTreeMap::new();
        let mut status = BTreeMap::new();
        for report in reports {
            if report.metrics.is_empty() {
                status.insert(report.backend_name.clone(), "no_metrics".to_string());
                continue;
            }
            status.insert(report.backend_name.clone(), "metrics_available".to_string());
            for metric in &report.metrics {
                if metric.value.is_finite() {
                    metrics.insert(metric.metric_name.clone(), metric.value);
                    metrics.insert(
                        format!("{}.{}", report.backend_name, metric.metric_name),
                        metric.value,
                    );
                }
            }
        }
        Self {
            available: !metrics.is_empty(),
            metrics,
            status,
        }
    }

    fn metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    fn metric_any(&self, names: &[&str]) -> Option<f64> {
        names.iter().find_map(|name| self.metric(name))
    }
}

fn build_pair(
    left: &InteractionProfile,
    right: &InteractionProfile,
    config: &PreferenceConstructionConfig,
) -> Option<PreferencePair> {
    if left.example_id != right.example_id || left.protein_id != right.protein_id {
        return None;
    }
    let left_score = preference_score(left, config);
    let right_score = preference_score(right, config);
    let margin = (left_score - right_score).abs();
    if margin < config.min_soft_margin {
        return None;
    }
    let (winner, loser) = if left_score >= right_score {
        (left, right)
    } else {
        (right, left)
    };
    let feature_deltas = profile_feature_map(winner)
        .into_iter()
        .filter_map(|(name, winner_value)| {
            profile_feature_map(loser)
                .get(&name)
                .map(|loser_value| (name, winner_value - loser_value))
        })
        .collect::<BTreeMap<_, _>>();
    let mut reason_set = BTreeSet::new();
    collect_reasons(&feature_deltas, &mut reason_set);
    let hard_constraint_flags = hard_constraint_flags(winner, loser, config);
    let soft_preference_flags = soft_preference_flags(&feature_deltas);
    let backend_evidence = if winner.backend_evidence_available || loser.backend_evidence_available
    {
        1.0
    } else {
        0.0
    };
    let mut evidence_coverage = BTreeMap::new();
    evidence_coverage.insert("backend_evidence_available".to_string(), backend_evidence);
    evidence_coverage.insert(
        "compared_feature_count".to_string(),
        feature_deltas.len() as f64,
    );
    Some(PreferencePair {
        schema_version: PREFERENCE_PAIR_SCHEMA_VERSION,
        pair_id: format!("{}__vs__{}", winner.candidate_id, loser.candidate_id),
        example_id: winner.example_id.clone(),
        protein_id: winner.protein_id.clone(),
        winner_candidate_id: winner.candidate_id.clone(),
        loser_candidate_id: loser.candidate_id.clone(),
        preference_reason: reason_set.into_iter().collect(),
        preference_source: if backend_evidence > 0.0 {
            PreferenceSource::BackendBased
        } else {
            PreferenceSource::RuleBased
        },
        preference_strength: margin.clamp(0.0, 1.0),
        feature_deltas,
        hard_constraint_flags,
        soft_preference_flags,
        evidence_coverage,
    })
}

fn preference_score(profile: &InteractionProfile, config: &PreferenceConstructionConfig) -> f64 {
    let hard_penalty = if hard_pass(profile, config) {
        0.0
    } else {
        0.35
    };
    let score = 0.20 * feature_or_zero(&profile.rdkit_valid)
        + 0.15 * feature_or_zero(&profile.rdkit_sanitized)
        + 0.15 * feature_or_zero(&profile.pocket_contact_fraction)
        + 0.20 * feature_or_zero(&profile.strict_pocket_fit_score)
        + 0.10 * feature_or_zero(&profile.atom_coverage_fraction)
        + 0.10 * (1.0 - feature_or_zero(&profile.clash_fraction)).clamp(0.0, 1.0)
        + 0.05 * feature_or_zero(&profile.bond_density_fit)
        + 0.05 * feature_or_zero(&profile.valence_sane);
    (score - hard_penalty).clamp(0.0, 1.0)
}

fn hard_pass(profile: &InteractionProfile, config: &PreferenceConstructionConfig) -> bool {
    feature_or_zero(&profile.clash_fraction) <= config.max_clash_fraction
        && feature_or_zero(&profile.strict_pocket_fit_score) >= config.min_strict_pocket_fit_score
        && (!config.require_valence_sane || feature_or_zero(&profile.valence_sane) >= 0.5)
}

fn profile_feature_map(profile: &InteractionProfile) -> BTreeMap<String, f64> {
    BTreeMap::from([
        (
            "pocket_contact_fraction".to_string(),
            feature_or_zero(&profile.pocket_contact_fraction),
        ),
        (
            "clash_fraction".to_string(),
            feature_or_zero(&profile.clash_fraction),
        ),
        (
            "centroid_offset".to_string(),
            feature_or_zero(&profile.centroid_offset),
        ),
        (
            "strict_pocket_fit_score".to_string(),
            feature_or_zero(&profile.strict_pocket_fit_score),
        ),
        (
            "atom_coverage_fraction".to_string(),
            feature_or_zero(&profile.atom_coverage_fraction),
        ),
        (
            "rdkit_valid".to_string(),
            feature_or_zero(&profile.rdkit_valid),
        ),
        (
            "rdkit_sanitized".to_string(),
            feature_or_zero(&profile.rdkit_sanitized),
        ),
        (
            "rdkit_unique_smiles".to_string(),
            feature_or_zero(&profile.rdkit_unique_smiles),
        ),
        (
            "bond_density_fit".to_string(),
            feature_or_zero(&profile.bond_density_fit),
        ),
        (
            "valence_sane".to_string(),
            feature_or_zero(&profile.valence_sane),
        ),
    ])
}

fn collect_reasons(deltas: &BTreeMap<String, f64>, reasons: &mut BTreeSet<PreferenceReasonCode>) {
    if positive_delta(deltas, "rdkit_valid") {
        reasons.insert(PreferenceReasonCode::RdkitValid);
    }
    if positive_delta(deltas, "rdkit_sanitized") {
        reasons.insert(PreferenceReasonCode::RdkitSanitized);
    }
    if negative_delta(deltas, "clash_fraction") {
        reasons.insert(PreferenceReasonCode::LowerClash);
    }
    if positive_delta(deltas, "pocket_contact_fraction") {
        reasons.insert(PreferenceReasonCode::BetterPocketContact);
    }
    if negative_delta(deltas, "centroid_offset") {
        reasons.insert(PreferenceReasonCode::LowerCentroidOffset);
    }
    if positive_delta(deltas, "strict_pocket_fit_score") {
        reasons.insert(PreferenceReasonCode::BetterStrictPocketFit);
    }
    if positive_delta(deltas, "atom_coverage_fraction") {
        reasons.insert(PreferenceReasonCode::BetterAtomCoverage);
    }
    if positive_delta(deltas, "bond_density_fit") {
        reasons.insert(PreferenceReasonCode::BetterBondDensity);
    }
    if positive_delta(deltas, "valence_sane") {
        reasons.insert(PreferenceReasonCode::ValenceSane);
    }
}

fn hard_constraint_flags(
    winner: &InteractionProfile,
    loser: &InteractionProfile,
    config: &PreferenceConstructionConfig,
) -> BTreeMap<String, bool> {
    BTreeMap::from([
        ("winner_hard_pass".to_string(), hard_pass(winner, config)),
        ("loser_hard_pass".to_string(), hard_pass(loser, config)),
        (
            "lower_clash_required".to_string(),
            feature_or_zero(&winner.clash_fraction) <= config.max_clash_fraction
                && feature_or_zero(&loser.clash_fraction) > config.max_clash_fraction,
        ),
        (
            "strict_pocket_fit_required".to_string(),
            feature_or_zero(&winner.strict_pocket_fit_score) >= config.min_strict_pocket_fit_score
                && feature_or_zero(&loser.strict_pocket_fit_score)
                    < config.min_strict_pocket_fit_score,
        ),
    ])
}

fn soft_preference_flags(deltas: &BTreeMap<String, f64>) -> BTreeMap<String, bool> {
    BTreeMap::from([
        (
            "better_contact".to_string(),
            positive_delta(deltas, "pocket_contact_fraction"),
        ),
        (
            "lower_centroid_offset".to_string(),
            negative_delta(deltas, "centroid_offset"),
        ),
        (
            "better_atom_coverage".to_string(),
            positive_delta(deltas, "atom_coverage_fraction"),
        ),
        (
            "better_bond_density".to_string(),
            positive_delta(deltas, "bond_density_fit"),
        ),
    ])
}

fn positive_delta(deltas: &BTreeMap<String, f64>, name: &str) -> bool {
    deltas.get(name).copied().unwrap_or(0.0) > 1e-9
}

fn negative_delta(deltas: &BTreeMap<String, f64>, name: &str) -> bool {
    deltas.get(name).copied().unwrap_or(0.0) < -1e-9
}

fn feature_or_zero(feature: &InteractionFeatureValue) -> f64 {
    feature.value.unwrap_or(0.0)
}

fn valid(candidate: &GeneratedCandidateRecord) -> bool {
    !candidate.atom_types.is_empty()
        && candidate.atom_types.len() == candidate.coords.len()
        && candidate
            .coords
            .iter()
            .all(|coord| coord.iter().all(|value| value.is_finite()))
}

fn valence_sane(candidate: &GeneratedCandidateRecord) -> bool {
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
        1 => 1,
        6 => 4,
        7 => 4,
        8 => 3,
        9 | 17 | 35 | 53 => 1,
        15 => 5,
        16 => 6,
        _ => 4,
    }
}

fn pocket_contact_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return 0.0;
    }
    candidate
        .coords
        .iter()
        .filter(|coord| {
            coord_distance(coord, &candidate.pocket_centroid)
                <= (candidate.pocket_radius + 2.0) as f64
        })
        .count() as f64
        / candidate.coords.len() as f64
}

fn atom_coverage_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
    if candidate.coords.is_empty() {
        return 0.0;
    }
    candidate
        .coords
        .iter()
        .filter(|coord| {
            coord_distance(coord, &candidate.pocket_centroid)
                <= (candidate.pocket_radius + 3.0) as f64
        })
        .count() as f64
        / candidate.coords.len() as f64
}

fn centroid_offset(candidate: &GeneratedCandidateRecord) -> f64 {
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

fn clash_fraction(candidate: &GeneratedCandidateRecord) -> f64 {
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

fn bond_density_fit(candidate: &GeneratedCandidateRecord) -> f64 {
    let atoms = candidate.atom_types.len();
    if atoms < 2 {
        return 0.0;
    }
    let density = candidate.inferred_bonds.len() as f64 / atoms as f64;
    (1.0 - (density - 1.05).abs() / 1.05).clamp(0.0, 1.0)
}

fn strict_pocket_fit_score(candidate: &GeneratedCandidateRecord) -> f64 {
    if !valid(candidate) {
        return 0.0;
    }
    let centroid_inside = if centroid_offset(candidate) <= candidate.pocket_radius as f64 {
        1.0
    } else {
        0.0
    };
    pocket_contact_fraction(candidate).min(1.0)
        * centroid_inside
        * atom_coverage_fraction(candidate)
        * (1.0 - clash_fraction(candidate).clamp(0.0, 1.0))
}

fn coord_distance(left: &[f32; 3], right: &[f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_extraction_marks_proxy_and_backend_sources() {
        let candidate = candidate(vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]);
        let reports = [ExternalEvaluationReport {
            backend_name: "rdkit".to_string(),
            metrics: vec![
                crate::models::ExternalMetricRecord {
                    metric_name: "rdkit_valid_fraction".to_string(),
                    value: 1.0,
                },
                crate::models::ExternalMetricRecord {
                    metric_name: "backend_missing_structure_fraction".to_string(),
                    value: 0.0,
                },
            ],
        }];
        let profiles = extract_interaction_profiles(
            &[candidate],
            CandidateLayerKind::Reranked,
            Some("conditioned_denoising"),
            &reports,
        );
        let profile = &profiles[0];
        assert_eq!(profile.schema_version, INTERACTION_PROFILE_SCHEMA_VERSION);
        assert_eq!(
            profile.pocket_contact_fraction.provenance,
            InteractionFeatureProvenance::HeuristicProxy
        );
        assert_eq!(
            profile.rdkit_valid.provenance,
            InteractionFeatureProvenance::ExternalBackend
        );
        assert_eq!(profile.backend_missing_structure_fraction.value, Some(0.0));
    }

    #[test]
    fn preference_pair_preserves_reasons_and_signed_deltas() {
        let good = extract_interaction_profiles(
            &[candidate(vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])],
            CandidateLayerKind::Reranked,
            None,
            &[],
        )
        .remove(0);
        let bad = extract_interaction_profiles(
            &[candidate(vec![[6.0, 0.0, 0.0], [6.1, 0.0, 0.0]])],
            CandidateLayerKind::Reranked,
            None,
            &[],
        )
        .remove(0);
        let builder = RuleBasedPreferenceDatasetBuilder::default();
        let pairs = builder.build_pairs(&[good, bad]);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].schema_version, PREFERENCE_PAIR_SCHEMA_VERSION);
        assert!(pairs[0]
            .preference_reason
            .contains(&PreferenceReasonCode::BetterStrictPocketFit));
        assert!(pairs[0].feature_deltas["strict_pocket_fit_score"] > 0.0);
    }

    #[test]
    fn preference_pair_artifact_compact_metrics_reflect_source_coverage() {
        let good = extract_interaction_profiles(
            &[candidate(vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])],
            CandidateLayerKind::Reranked,
            None,
            &[],
        )
        .remove(0);
        let bad = extract_interaction_profiles(
            &[candidate(vec![[6.0, 0.0, 0.0], [6.1, 0.0, 0.0]])],
            CandidateLayerKind::Reranked,
            None,
            &[],
        )
        .remove(0);
        let pairs = RuleBasedPreferenceDatasetBuilder::default().build_pairs(&[good, bad]);
        let artifact = PreferencePairArtifact::new("validation", &[], pairs);
        assert_eq!(artifact.pair_count, 1);
        assert_eq!(artifact.rule_only_pair_fraction, 1.0);
        assert_eq!(artifact.backend_supported_pair_fraction, 0.0);
        assert_eq!(artifact.missing_backend_evidence_fraction, 1.0);
        assert!(artifact.mean_preference_strength > 0.0);
    }

    fn candidate(coords: Vec<[f32; 3]>) -> GeneratedCandidateRecord {
        GeneratedCandidateRecord {
            example_id: "ex".to_string(),
            protein_id: "prot".to_string(),
            molecular_representation: None,
            atom_types: vec![6; coords.len()],
            coords,
            inferred_bonds: vec![(0, 1)],
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_radius: 3.0,
            coordinate_frame_origin: [0.0, 0.0, 0.0],
            source: "test".to_string(),
            source_pocket_path: None,
            source_ligand_path: None,
        }
    }
}
