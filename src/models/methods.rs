//! Method registry and concrete generation-method implementations.

use std::collections::{BTreeMap, BTreeSet};

use super::evaluation::{generate_layered_candidates_with_options, CandidateGenerationLayers};
use super::{
    CandidateLayerKind, CandidateLayerOutput, CandidateLayerProvenance, GeneratedCandidateRecord,
    GenerationEvidenceRole, GenerationExecutionMode, GenerationMethodCapability,
    LayeredGenerationOutput, PocketGenerationContext, PocketGenerationMethod,
    PocketGenerationMethodFamily, PocketGenerationMethodMetadata,
};

const CONDITIONED_DENOISING_METHOD_ID: &str = "conditioned_denoising";
const HEURISTIC_RAW_ROLLOUT_METHOD_ID: &str = "heuristic_raw_rollout_no_repair";
const REPAIR_ONLY_METHOD_ID: &str = "pocket_centroid_repair_proxy";
const DETERMINISTIC_PROXY_RERANKER_METHOD_ID: &str = "deterministic_proxy_reranker";
const CALIBRATED_RERANKER_METHOD_ID: &str = "calibrated_reranker";
const FLOW_MATCHING_STUB_METHOD_ID: &str = "flow_matching_stub";
const DIFFUSION_STUB_METHOD_ID: &str = "diffusion_stub";
const AUTOREGRESSIVE_STUB_METHOD_ID: &str = "autoregressive_stub";
const EXTERNAL_WRAPPER_STUB_METHOD_ID: &str = "external_wrapper_stub";

/// Configurable registry surface for generation methods.
#[derive(Debug, Clone, Default)]
pub struct PocketGenerationMethodRegistry;

impl PocketGenerationMethodRegistry {
    /// Stable identifiers that are currently recognized by the registry.
    pub fn registered_method_ids() -> Vec<&'static str> {
        vec![
            CONDITIONED_DENOISING_METHOD_ID,
            HEURISTIC_RAW_ROLLOUT_METHOD_ID,
            REPAIR_ONLY_METHOD_ID,
            DETERMINISTIC_PROXY_RERANKER_METHOD_ID,
            CALIBRATED_RERANKER_METHOD_ID,
            FLOW_MATCHING_STUB_METHOD_ID,
            DIFFUSION_STUB_METHOD_ID,
            AUTOREGRESSIVE_STUB_METHOD_ID,
            EXTERNAL_WRAPPER_STUB_METHOD_ID,
        ]
    }

    /// Whether the provided method identifier is recognized.
    pub fn contains(method_id: &str) -> bool {
        Self::registered_method_ids()
            .iter()
            .any(|registered| *registered == method_id)
    }

    /// Instantiate one registered method.
    pub fn build(method_id: &str) -> Result<Box<dyn PocketGenerationMethod>, String> {
        match method_id {
            CONDITIONED_DENOISING_METHOD_ID => Ok(Box::new(ConditionedDenoisingMethod)),
            HEURISTIC_RAW_ROLLOUT_METHOD_ID => Ok(Box::new(HeuristicRawRolloutMethod)),
            REPAIR_ONLY_METHOD_ID => Ok(Box::new(PocketCentroidRepairMethod)),
            DETERMINISTIC_PROXY_RERANKER_METHOD_ID => Ok(Box::new(DeterministicProxyRerankerMethod)),
            CALIBRATED_RERANKER_METHOD_ID => Ok(Box::new(CalibratedRerankerMethod)),
            FLOW_MATCHING_STUB_METHOD_ID => Ok(Box::new(StubGenerationMethod::flow_matching())),
            DIFFUSION_STUB_METHOD_ID => Ok(Box::new(StubGenerationMethod::diffusion())),
            AUTOREGRESSIVE_STUB_METHOD_ID => Ok(Box::new(StubGenerationMethod::autoregressive())),
            EXTERNAL_WRAPPER_STUB_METHOD_ID => {
                Ok(Box::new(StubGenerationMethod::external_wrapper()))
            }
            _ => Err(format!("unknown generation method `{method_id}`")),
        }
    }

    /// Materialize method metadata for a method id.
    pub fn metadata(method_id: &str) -> Result<PocketGenerationMethodMetadata, String> {
        Self::build(method_id).map(|method| method.metadata())
    }
}

/// Summary row used by method-aware comparison artifacts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct MethodComparisonRow {
    /// Stable method identifier.
    pub method_id: String,
    /// Human-readable method name.
    pub method_name: String,
    /// Method family.
    pub method_family: String,
    /// Evidence role attached to the method.
    pub evidence_role: String,
    /// Whether the method emitted any layer for this split.
    pub available: bool,
    /// Stable legacy field name of the method-native layer when available.
    pub native_layer: Option<String>,
    /// Declared supported layers.
    #[serde(default)]
    pub supported_layers: Vec<String>,
    /// Whether the method is trainable.
    #[serde(default)]
    pub trainable: bool,
    /// Candidate count in the native layer.
    #[serde(default)]
    pub native_candidate_count: usize,
    /// Mean validity gain of repaired over raw layer.
    #[serde(default)]
    pub repair_gain_valid_fraction: Option<f64>,
    /// Mean validity gain of reranked over inferred-bond layer.
    #[serde(default)]
    pub rerank_gain_valid_fraction: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct ConditionedDenoisingMethod;

#[derive(Debug, Clone, Copy)]
struct HeuristicRawRolloutMethod;

#[derive(Debug, Clone, Copy)]
struct PocketCentroidRepairMethod;

#[derive(Debug, Clone, Copy)]
struct DeterministicProxyRerankerMethod;

#[derive(Debug, Clone, Copy)]
struct CalibratedRerankerMethod;

#[derive(Debug, Clone)]
struct StubGenerationMethod {
    metadata: PocketGenerationMethodMetadata,
}

impl StubGenerationMethod {
    fn flow_matching() -> Self {
        Self {
            metadata: stub_metadata(
                FLOW_MATCHING_STUB_METHOD_ID,
                "Flow Matching Stub",
                PocketGenerationMethodFamily::FlowMatching,
                GenerationEvidenceRole::ComparisonOnly,
            ),
        }
    }

    fn diffusion() -> Self {
        Self {
            metadata: stub_metadata(
                DIFFUSION_STUB_METHOD_ID,
                "Diffusion Stub",
                PocketGenerationMethodFamily::Diffusion,
                GenerationEvidenceRole::ComparisonOnly,
            ),
        }
    }

    fn autoregressive() -> Self {
        Self {
            metadata: stub_metadata(
                AUTOREGRESSIVE_STUB_METHOD_ID,
                "Autoregressive Stub",
                PocketGenerationMethodFamily::Autoregressive,
                GenerationEvidenceRole::ComparisonOnly,
            ),
        }
    }

    fn external_wrapper() -> Self {
        Self {
            metadata: PocketGenerationMethodMetadata {
                method_id: EXTERNAL_WRAPPER_STUB_METHOD_ID.to_string(),
                method_name: "External Wrapper Stub".to_string(),
                method_family: PocketGenerationMethodFamily::ExternalWrapper,
                capability: GenerationMethodCapability {
                    external_wrapper: true,
                    stub: true,
                    ..GenerationMethodCapability::default()
                },
                layered_output_support: Vec::new(),
                evidence_role: GenerationEvidenceRole::ExternalWrapper,
                execution_mode: GenerationExecutionMode::Stub,
            },
        }
    }
}

impl PocketGenerationMethod for ConditionedDenoisingMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: CONDITIONED_DENOISING_METHOD_ID.to_string(),
            method_name: "Conditioned Denoising".to_string(),
            method_family: PocketGenerationMethodFamily::ConditionedDenoising,
            capability: GenerationMethodCapability {
                trainable: true,
                batched_generation: true,
                method_native_generation: true,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
                CandidateLayerKind::Reranked,
            ],
            evidence_role: GenerationEvidenceRole::ClaimBearing,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = context.forward else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            &forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let inferred = layers.inferred_bond.clone();
        let proxy = proxy_rerank_candidates(&inferred);
        let calibrated = CalibratedReranker::fit(&inferred).rerank(&inferred);
        layered_output_from_legacy(metadata, layers, Some(proxy), Some(calibrated), true)
    }
}

impl PocketGenerationMethod for HeuristicRawRolloutMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: HEURISTIC_RAW_ROLLOUT_METHOD_ID.to_string(),
            method_name: "Heuristic Raw Rollout".to_string(),
            method_family: PocketGenerationMethodFamily::Heuristic,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: true,
                uses_postprocessors: false,
                repair_layer: false,
                inferred_bond_layer: false,
                reranked_layer: false,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![CandidateLayerKind::RawRollout],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = context.forward else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            &forward,
            context.candidate_limit.max(1),
            false,
        );
        let raw_only = CandidateGenerationLayers {
            raw_rollout: layers.raw_rollout,
            repaired: Vec::new(),
            inferred_bond: Vec::new(),
        };
        layered_output_from_legacy(metadata, raw_only, None, None, true)
    }
}

impl PocketGenerationMethod for PocketCentroidRepairMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: REPAIR_ONLY_METHOD_ID.to_string(),
            method_name: "Pocket Centroid Repair Proxy".to_string(),
            method_family: PocketGenerationMethodFamily::RepairOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: false,
                reranked_layer: false,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = context.forward else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            &forward,
            context.candidate_limit.max(1),
            true,
        );
        let repair_layers = CandidateGenerationLayers {
            raw_rollout: layers.raw_rollout,
            repaired: layers.repaired,
            inferred_bond: Vec::new(),
        };
        layered_output_from_legacy(metadata, repair_layers, None, None, false)
    }
}

impl PocketGenerationMethod for DeterministicProxyRerankerMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: DETERMINISTIC_PROXY_RERANKER_METHOD_ID.to_string(),
            method_name: "Deterministic Proxy Reranker".to_string(),
            method_family: PocketGenerationMethodFamily::RerankerOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
            ],
            evidence_role: GenerationEvidenceRole::ComparisonOnly,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = context.forward else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            &forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let proxy = proxy_rerank_candidates(&layers.inferred_bond);
        layered_output_from_legacy(metadata, layers, Some(proxy), None, false)
    }
}

impl PocketGenerationMethod for CalibratedRerankerMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        PocketGenerationMethodMetadata {
            method_id: CALIBRATED_RERANKER_METHOD_ID.to_string(),
            method_name: "Calibrated Reranker".to_string(),
            method_family: PocketGenerationMethodFamily::RerankerOnly,
            capability: GenerationMethodCapability {
                trainable: false,
                batched_generation: true,
                method_native_generation: false,
                uses_postprocessors: true,
                repair_layer: true,
                inferred_bond_layer: true,
                reranked_layer: true,
                external_wrapper: false,
                stub: false,
            },
            layered_output_support: vec![
                CandidateLayerKind::RawRollout,
                CandidateLayerKind::Repaired,
                CandidateLayerKind::InferredBond,
                CandidateLayerKind::DeterministicProxy,
                CandidateLayerKind::Reranked,
            ],
            evidence_role: GenerationEvidenceRole::ClaimBearing,
            execution_mode: GenerationExecutionMode::Batched,
        }
    }

    fn generate_for_example(&self, context: PocketGenerationContext) -> LayeredGenerationOutput {
        let metadata = self.metadata();
        let Some(forward) = context.forward else {
            return LayeredGenerationOutput::empty(metadata);
        };
        let layers = generate_layered_candidates_with_options(
            &context.example,
            &forward,
            context.candidate_limit.max(1),
            context.enable_repair,
        );
        let inferred = layers.inferred_bond.clone();
        let proxy = proxy_rerank_candidates(&inferred);
        let calibrated = CalibratedReranker::fit(&inferred).rerank(&inferred);
        layered_output_from_legacy(metadata, layers, Some(proxy), Some(calibrated), false)
    }
}

impl PocketGenerationMethod for StubGenerationMethod {
    fn metadata(&self) -> PocketGenerationMethodMetadata {
        self.metadata.clone()
    }

    fn generate_for_example(&self, _context: PocketGenerationContext) -> LayeredGenerationOutput {
        LayeredGenerationOutput::empty(self.metadata())
    }
}

fn stub_metadata(
    method_id: &str,
    method_name: &str,
    family: PocketGenerationMethodFamily,
    evidence_role: GenerationEvidenceRole,
) -> PocketGenerationMethodMetadata {
    PocketGenerationMethodMetadata {
        method_id: method_id.to_string(),
        method_name: method_name.to_string(),
        method_family: family,
        capability: GenerationMethodCapability {
            stub: true,
            ..GenerationMethodCapability::default()
        },
        layered_output_support: Vec::new(),
        evidence_role,
        execution_mode: GenerationExecutionMode::Stub,
    }
}

fn layered_output_from_legacy(
    metadata: PocketGenerationMethodMetadata,
    layers: CandidateGenerationLayers,
    deterministic_proxy: Option<Vec<GeneratedCandidateRecord>>,
    reranked: Option<Vec<GeneratedCandidateRecord>>,
    native_raw: bool,
) -> LayeredGenerationOutput {
    let mut output = LayeredGenerationOutput::empty(metadata.clone());
    if !layers.raw_rollout.is_empty() {
        output.raw_rollout = Some(layer_output(
            &metadata,
            CandidateLayerKind::RawRollout,
            native_raw,
            Vec::new(),
            layers.raw_rollout,
        ));
    }
    if !layers.repaired.is_empty() {
        output.repaired = Some(layer_output(
            &metadata,
            CandidateLayerKind::Repaired,
            false,
            vec!["pocket_centroid_repair".to_string()],
            layers.repaired,
        ));
    }
    if !layers.inferred_bond.is_empty() {
        output.inferred_bond = Some(layer_output(
            &metadata,
            CandidateLayerKind::InferredBond,
            false,
            vec![
                "pocket_centroid_repair".to_string(),
                "distance_bond_inference".to_string(),
                "valence_pruning".to_string(),
            ],
            layers.inferred_bond,
        ));
    }
    if let Some(candidates) = deterministic_proxy.filter(|candidates| !candidates.is_empty()) {
        output.deterministic_proxy = Some(layer_output(
            &metadata,
            CandidateLayerKind::DeterministicProxy,
            false,
            vec![
                "pocket_centroid_repair".to_string(),
                "distance_bond_inference".to_string(),
                "deterministic_proxy_rerank".to_string(),
            ],
            candidates,
        ));
    }
    if let Some(candidates) = reranked.filter(|candidates| !candidates.is_empty()) {
        output.reranked = Some(layer_output(
            &metadata,
            CandidateLayerKind::Reranked,
            false,
            vec![
                "pocket_centroid_repair".to_string(),
                "distance_bond_inference".to_string(),
                "bounded_calibrated_rerank".to_string(),
            ],
            candidates,
        ));
    }
    output
}

fn layer_output(
    metadata: &PocketGenerationMethodMetadata,
    layer_kind: CandidateLayerKind,
    method_native: bool,
    postprocessor_chain: Vec<String>,
    candidates: Vec<GeneratedCandidateRecord>,
) -> CandidateLayerOutput {
    CandidateLayerOutput {
        provenance: CandidateLayerProvenance {
            source_method_id: metadata.method_id.clone(),
            source_method_name: metadata.method_name.clone(),
            source_method_family: metadata.method_family,
            layer_kind,
            legacy_field_name: layer_kind.legacy_field_name().to_string(),
            method_native,
            postprocessor_chain,
            available: true,
        },
        candidates,
    }
}

/// Flatten layered outputs into legacy layer lists for backward-compatible artifact writers.
pub fn flatten_layered_output(
    output: &LayeredGenerationOutput,
) -> (
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
    Vec<GeneratedCandidateRecord>,
) {
    (
        output
            .raw_rollout
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .repaired
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .inferred_bond
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .deterministic_proxy
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
        output
            .reranked
            .as_ref()
            .map(|layer| layer.candidates.clone())
            .unwrap_or_default(),
    )
}

/// Build a compact method comparison row from one layered output.
pub fn summarize_method_output(output: &LayeredGenerationOutput) -> MethodComparisonRow {
    let supported_layers = output
        .metadata
        .layered_output_support
        .iter()
        .map(|layer| layer.legacy_field_name().to_string())
        .collect::<Vec<_>>();
    let native_layer = [
        output.raw_rollout.as_ref(),
        output.repaired.as_ref(),
        output.inferred_bond.as_ref(),
        output.deterministic_proxy.as_ref(),
        output.reranked.as_ref(),
    ]
    .into_iter()
    .flatten()
    .find(|layer| layer.provenance.method_native)
    .map(|layer| layer.provenance.legacy_field_name.clone())
    .or_else(|| {
        if output.metadata.method_family == PocketGenerationMethodFamily::RerankerOnly {
            output
                .reranked
                .as_ref()
                .map(|layer| layer.provenance.legacy_field_name.clone())
        } else {
            None
        }
    });
    let native_candidate_count = native_layer
        .as_ref()
        .and_then(|layer_name| match layer_name.as_str() {
            "raw_rollout" => output.raw_rollout.as_ref(),
            "repaired_candidates" => output.repaired.as_ref(),
            "inferred_bond_candidates" => output.inferred_bond.as_ref(),
            "deterministic_proxy_candidates" => output.deterministic_proxy.as_ref(),
            "reranked_candidates" => output.reranked.as_ref(),
            _ => None,
        })
        .map(|layer| layer.candidates.len())
        .unwrap_or(0);
    let repair_gain_valid_fraction = output
        .raw_rollout
        .as_ref()
        .zip(output.repaired.as_ref())
        .map(|(raw, repaired)| valid_fraction(&repaired.candidates) - valid_fraction(&raw.candidates));
    let rerank_gain_valid_fraction = output
        .inferred_bond
        .as_ref()
        .zip(output.reranked.as_ref())
        .map(|(inferred, reranked)| {
            valid_fraction(&reranked.candidates) - valid_fraction(&inferred.candidates)
        });

    MethodComparisonRow {
        method_id: output.metadata.method_id.clone(),
        method_name: output.metadata.method_name.clone(),
        method_family: format!("{:?}", output.metadata.method_family).to_ascii_lowercase(),
        evidence_role: format!("{:?}", output.metadata.evidence_role).to_ascii_lowercase(),
        available: native_candidate_count > 0
            || output.raw_rollout.is_some()
            || output.repaired.is_some()
            || output.inferred_bond.is_some()
            || output.deterministic_proxy.is_some()
            || output.reranked.is_some(),
        native_layer,
        supported_layers,
        trainable: output.metadata.capability.trainable,
        native_candidate_count,
        repair_gain_valid_fraction,
        rerank_gain_valid_fraction,
    }
}

fn valid_fraction(candidates: &[GeneratedCandidateRecord]) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }
    candidates.iter().filter(|candidate| candidate_is_valid(candidate)).count() as f64
        / candidates.len() as f64
}

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
