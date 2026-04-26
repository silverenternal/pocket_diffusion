const CONDITIONED_DENOISING_METHOD_ID: &str = "conditioned_denoising";
const HEURISTIC_RAW_ROLLOUT_METHOD_ID: &str = "heuristic_raw_rollout_no_repair";
const REPAIR_ONLY_METHOD_ID: &str = "pocket_centroid_repair_proxy";
const DETERMINISTIC_PROXY_RERANKER_METHOD_ID: &str = "deterministic_proxy_reranker";
const CALIBRATED_RERANKER_METHOD_ID: &str = "calibrated_reranker";
const PREFERENCE_AWARE_RERANKER_METHOD_ID: &str = "preference_aware_reranker";
const FLOW_MATCHING_METHOD_ID: &str = "flow_matching";
const AUTOREGRESSIVE_GRAPH_GEOMETRY_METHOD_ID: &str = "autoregressive_graph_geometry";
const ENERGY_GUIDED_REFINEMENT_METHOD_ID: &str = "energy_guided_refinement";
const FLOW_MATCHING_STUB_METHOD_ID: &str = "flow_matching_stub";
const DIFFUSION_STUB_METHOD_ID: &str = "diffusion_stub";
const AUTOREGRESSIVE_STUB_METHOD_ID: &str = "autoregressive_stub";
const EXTERNAL_WRAPPER_STUB_METHOD_ID: &str = "external_wrapper_stub";
const EXTERNAL_WRAPPER_DRY_RUN_METHOD_ID: &str = "external_wrapper_dry_run";

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
            PREFERENCE_AWARE_RERANKER_METHOD_ID,
            FLOW_MATCHING_METHOD_ID,
            AUTOREGRESSIVE_GRAPH_GEOMETRY_METHOD_ID,
            ENERGY_GUIDED_REFINEMENT_METHOD_ID,
            FLOW_MATCHING_STUB_METHOD_ID,
            DIFFUSION_STUB_METHOD_ID,
            AUTOREGRESSIVE_STUB_METHOD_ID,
            EXTERNAL_WRAPPER_STUB_METHOD_ID,
            EXTERNAL_WRAPPER_DRY_RUN_METHOD_ID,
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
            PREFERENCE_AWARE_RERANKER_METHOD_ID => Ok(Box::new(PreferenceAwareRerankerMethod)),
            FLOW_MATCHING_METHOD_ID => Ok(Box::new(FlowMatchingMethod)),
            AUTOREGRESSIVE_GRAPH_GEOMETRY_METHOD_ID => Ok(Box::new(AutoregressiveGraphGeometryMethod)),
            ENERGY_GUIDED_REFINEMENT_METHOD_ID => Ok(Box::new(EnergyGuidedRefinementMethod)),
            FLOW_MATCHING_STUB_METHOD_ID => Ok(Box::new(StubGenerationMethod::flow_matching())),
            DIFFUSION_STUB_METHOD_ID => Ok(Box::new(StubGenerationMethod::diffusion())),
            AUTOREGRESSIVE_STUB_METHOD_ID => Ok(Box::new(StubGenerationMethod::autoregressive())),
            EXTERNAL_WRAPPER_STUB_METHOD_ID => {
                Ok(Box::new(StubGenerationMethod::external_wrapper()))
            }
            EXTERNAL_WRAPPER_DRY_RUN_METHOD_ID => Ok(Box::new(ExternalWrapperDryRunMethod)),
            _ => Err(format!("unknown generation method `{method_id}`")),
        }
    }

    /// Materialize method metadata for a method id.
    pub fn metadata(method_id: &str) -> Result<PocketGenerationMethodMetadata, String> {
        Self::build(method_id).map(|method| method.metadata())
    }

    /// Validate a backend-neutral config entry against registered method metadata.
    pub fn validate_backend_config(
        backend: &crate::config::GenerationBackendConfig,
    ) -> Result<(), String> {
        let metadata = Self::metadata(&backend.backend_id)?;
        let declared_family = config_family_to_method_family(backend.family);
        if metadata.method_family != declared_family {
            return Err(format!(
                "generation backend `{}` declares family {:?} but registry metadata uses {:?}",
                backend.backend_id, backend.family, metadata.method_family
            ));
        }
        if metadata.capability.trainable != backend.trainable && !metadata.capability.stub {
            return Err(format!(
                "generation backend `{}` declares trainable={} but registry metadata uses trainable={}",
                backend.backend_id, backend.trainable, metadata.capability.trainable
            ));
        }
        if backend.external_wrapper.enabled && !metadata.capability.external_wrapper {
            return Err(format!(
                "generation backend `{}` enables external_wrapper but registry method is native",
                backend.backend_id
            ));
        }
        Ok(())
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
    /// Declared parameter count when known for this backend.
    #[serde(default)]
    pub parameter_count: Option<usize>,
    /// Effective sampling steps used by this comparison row when known.
    #[serde(default)]
    pub sampling_steps: Option<usize>,
    /// Wall-clock runtime for this method row in milliseconds when measured.
    #[serde(default)]
    pub wall_time_ms: Option<f64>,
    /// Coarse candidate payload memory estimate in bytes.
    #[serde(default)]
    pub memory_estimate_bytes: Option<usize>,
    /// Layer names that were actually available in this row.
    #[serde(default)]
    pub available_layers: Vec<String>,
    /// Candidate count in the native layer.
    #[serde(default)]
    pub native_candidate_count: usize,
    /// Valid fraction for the native layer when available.
    #[serde(default)]
    pub native_valid_fraction: Option<f64>,
    /// Pocket-contact fraction for the native layer when available.
    #[serde(default)]
    pub native_pocket_contact_fraction: Option<f64>,
    /// Non-bonded clash fraction for the native layer when available.
    #[serde(default)]
    pub native_clash_fraction: Option<f64>,
    /// Mean slot activation carried by this backend request when available.
    #[serde(default)]
    pub slot_activation_mean: Option<f64>,
    /// Mean directed gate activation carried by this backend request when available.
    #[serde(default)]
    pub gate_activation_mean: Option<f64>,
    /// Leakage proxy is shared at split level unless a backend-specific probe is added.
    #[serde(default)]
    pub leakage_proxy_mean: Option<f64>,
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

#[derive(Debug, Clone, Copy)]
struct PreferenceAwareRerankerMethod;

#[derive(Debug, Clone, Copy)]
struct FlowMatchingMethod;

#[derive(Debug, Clone, Copy)]
struct AutoregressiveGraphGeometryMethod;

#[derive(Debug, Clone, Copy)]
struct EnergyGuidedRefinementMethod;

#[derive(Debug, Clone, Copy)]
struct ExternalWrapperDryRunMethod;

#[derive(Debug, Clone)]
struct StubGenerationMethod {
    metadata: PocketGenerationMethodMetadata,
}

fn config_family_to_method_family(
    family: crate::config::GenerationBackendFamilyConfig,
) -> PocketGenerationMethodFamily {
    match family {
        crate::config::GenerationBackendFamilyConfig::ConditionedDenoising => {
            PocketGenerationMethodFamily::ConditionedDenoising
        }
        crate::config::GenerationBackendFamilyConfig::FlowMatching => {
            PocketGenerationMethodFamily::FlowMatching
        }
        crate::config::GenerationBackendFamilyConfig::Diffusion => PocketGenerationMethodFamily::Diffusion,
        crate::config::GenerationBackendFamilyConfig::Autoregressive => {
            PocketGenerationMethodFamily::Autoregressive
        }
        crate::config::GenerationBackendFamilyConfig::EnergyGuidedRefinement => {
            PocketGenerationMethodFamily::RepairOnly
        }
        crate::config::GenerationBackendFamilyConfig::Heuristic => PocketGenerationMethodFamily::Heuristic,
        crate::config::GenerationBackendFamilyConfig::RepairOnly => PocketGenerationMethodFamily::RepairOnly,
        crate::config::GenerationBackendFamilyConfig::RerankerOnly => PocketGenerationMethodFamily::RerankerOnly,
        crate::config::GenerationBackendFamilyConfig::ExternalWrapper => {
            PocketGenerationMethodFamily::ExternalWrapper
        }
    }
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
