/// Encoder and latent architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Shared hidden size used by modality encoders.
    pub hidden_dim: i64,
    /// Slot count upper bound per modality.
    pub num_slots: i64,
    /// Slot activation/decomposition controls for ablations.
    #[serde(default)]
    pub slot_decomposition: SlotDecompositionConfig,
    /// Semantic probe capacity controls for specialization/leakage audits.
    #[serde(default)]
    pub semantic_probes: SemanticProbeConfig,
    /// Atom-type vocabulary size.
    pub atom_vocab_size: i64,
    /// Bond-type vocabulary size.
    pub bond_vocab_size: i64,
    /// Input pocket feature width.
    pub pocket_feature_dim: i64,
    /// Pairwise geometric feature width.
    pub pair_feature_dim: i64,
    /// Topology encoder implementation and message-passing controls.
    #[serde(default)]
    pub topology_encoder: TopologyEncoderConfig,
    /// Geometry encoder implementation and operator-family controls.
    #[serde(default)]
    pub geometry_encoder: GeometryEncoderConfig,
    /// Pocket encoder implementation and local-context controls.
    #[serde(default)]
    pub pocket_encoder: PocketEncoderConfig,
    /// Flow velocity head selected for geometry flow-matching ablations.
    #[serde(default)]
    pub flow_velocity_head: FlowVelocityHeadConfig,
    /// Optional bounded ligand-ligand geometry messages for flow velocity prediction.
    #[serde(default)]
    pub pairwise_geometry: PairwiseGeometryConfig,
    /// Coordinate-preserving bond refinement controls.
    #[serde(default)]
    pub bond_refinement: BondRefinementConfig,
    /// Decoder conditioning style for atom-local versus pooled ablations.
    #[serde(default)]
    pub decoder_conditioning: DecoderConditioningConfig,
    /// Cross-modality interaction block style.
    #[serde(default)]
    pub interaction_mode: CrossAttentionMode,
    /// Feed-forward expansion factor used by the Transformer-style interaction block.
    #[serde(default = "default_interaction_ff_multiplier")]
    pub interaction_ff_multiplier: i64,
    /// Compact tuning knobs for the Transformer-style interaction refinement path.
    #[serde(default)]
    pub interaction_tuning: InteractionTuningConfig,
    /// Optional path-level temporal scheduling of interaction strengths.
    #[serde(default)]
    pub temporal_interaction_policy: TemporalInteractionPolicyConfig,
    /// Optional modality-only negative controls for ablation runs.
    #[serde(default)]
    pub modality_focus: ModalityFocusConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_slots: 8,
            slot_decomposition: SlotDecompositionConfig::default(),
            semantic_probes: SemanticProbeConfig::default(),
            atom_vocab_size: 32,
            bond_vocab_size: 8,
            pocket_feature_dim: 24,
            pair_feature_dim: 8,
            topology_encoder: TopologyEncoderConfig::default(),
            geometry_encoder: GeometryEncoderConfig::default(),
            pocket_encoder: PocketEncoderConfig::default(),
            flow_velocity_head: FlowVelocityHeadConfig::default(),
            pairwise_geometry: PairwiseGeometryConfig::default(),
            bond_refinement: BondRefinementConfig::default(),
            decoder_conditioning: DecoderConditioningConfig::default(),
            interaction_mode: CrossAttentionMode::default(),
            interaction_ff_multiplier: default_interaction_ff_multiplier(),
            interaction_tuning: InteractionTuningConfig::default(),
            temporal_interaction_policy: TemporalInteractionPolicyConfig::default(),
            modality_focus: ModalityFocusConfig::default(),
        }
    }
}

impl ModelConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.hidden_dim <= 0 {
            return Err(ConfigValidationError::new(
                "model.hidden_dim must be greater than zero",
            ));
        }
        if self.num_slots <= 0 {
            return Err(ConfigValidationError::new(
                "model.num_slots must be greater than zero",
            ));
        }
        self.slot_decomposition.validate()?;
        self.semantic_probes.validate()?;
        if self.atom_vocab_size <= 0 || self.bond_vocab_size <= 0 {
            return Err(ConfigValidationError::new(
                "model.atom_vocab_size and model.bond_vocab_size must be greater than zero",
            ));
        }
        if self.pocket_feature_dim <= 0 || self.pair_feature_dim <= 0 {
            return Err(ConfigValidationError::new(
                "model.pocket_feature_dim and model.pair_feature_dim must be greater than zero",
            ));
        }
        self.topology_encoder.validate()?;
        self.geometry_encoder.validate()?;
        self.pocket_encoder.validate()?;
        self.flow_velocity_head.validate()?;
        self.pairwise_geometry.validate()?;
        self.bond_refinement.validate()?;
        self.decoder_conditioning.validate()?;
        if self.interaction_ff_multiplier <= 0 {
            return Err(ConfigValidationError::new(
                "model.interaction_ff_multiplier must be greater than zero",
            ));
        }
        self.interaction_tuning.validate()?;
        self.temporal_interaction_policy.validate()?;
        Ok(())
    }
}

/// Modality focus used only for explicit negative-control ablations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModalityFocusConfig {
    /// Keep topology, geometry, and pocket/context modalities active.
    #[default]
    All,
    /// Keep only topology slots active downstream of the encoder.
    TopologyOnly,
    /// Keep only geometry slots active downstream of the encoder.
    GeometryOnly,
    /// Keep only pocket/context slots active downstream of the encoder.
    PocketOnly,
}

impl ModalityFocusConfig {
    /// Whether topology slots should remain visible to attention and decoder paths.
    pub const fn keep_topology(self) -> bool {
        matches!(self, Self::All | Self::TopologyOnly)
    }

    /// Whether geometry slots should remain visible to attention and decoder paths.
    pub const fn keep_geometry(self) -> bool {
        matches!(self, Self::All | Self::GeometryOnly)
    }

    /// Whether pocket/context slots should remain visible to attention and decoder paths.
    pub const fn keep_pocket(self) -> bool {
        matches!(self, Self::All | Self::PocketOnly)
    }
}

/// Decoder conditioning family for molecule generation ablations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DecoderConditioningKind {
    /// Legacy mean-pooled modality summaries broadcast to every atom.
    MeanPooled,
    /// Atom-level queries attend topology, geometry, and pocket slots through learned gates.
    #[default]
    LocalAtomSlotAttention,
}

/// Decoder-local conditioning controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConditioningConfig {
    /// Conditioning implementation selected for the ligand decoder.
    #[serde(default)]
    pub kind: DecoderConditioningKind,
    /// Initial scalar bias for local attention gates.
    #[serde(default = "default_decoder_local_gate_bias")]
    pub local_gate_initial_bias: f64,
}

impl Default for DecoderConditioningConfig {
    fn default() -> Self {
        Self {
            kind: DecoderConditioningKind::default(),
            local_gate_initial_bias: default_decoder_local_gate_bias(),
        }
    }
}

impl DecoderConditioningConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.local_gate_initial_bias.is_finite() {
            return Err(ConfigValidationError::new(
                "model.decoder_conditioning.local_gate_initial_bias must be finite",
            ));
        }
        Ok(())
    }
}

fn default_decoder_local_gate_bias() -> f64 {
    -1.0
}

/// Capacity controls for semantic and explicit leakage probe heads.
///
/// The default `hidden_layers = 0` preserves the original linear probe
/// baseline. Non-zero hidden layers create a lightweight MLP probe with a
/// configurable hidden width so leakage conclusions can be audited against
/// probe capacity instead of assuming a fixed weak head is sufficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticProbeConfig {
    /// Hidden width used by probe MLPs when `hidden_layers > 0`.
    #[serde(default = "default_semantic_probe_hidden_dim")]
    pub hidden_dim: i64,
    /// Number of ReLU hidden layers before each probe output head.
    #[serde(default)]
    pub hidden_layers: usize,
}

impl Default for SemanticProbeConfig {
    fn default() -> Self {
        Self {
            hidden_dim: default_semantic_probe_hidden_dim(),
            hidden_layers: 0,
        }
    }
}

impl SemanticProbeConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.hidden_dim <= 0 {
            return Err(ConfigValidationError::new(
                "model.semantic_probes.hidden_dim must be greater than zero",
            ));
        }
        Ok(())
    }
}

fn default_semantic_probe_hidden_dim() -> i64 {
    128
}

/// Slot activation and utilization controls shared by all modality branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotDecompositionConfig {
    /// Temperature applied to activation logits before sigmoid gating.
    #[serde(default = "default_slot_activation_temperature")]
    pub activation_temperature: f64,
    /// Activation threshold used for active/dead slot counts.
    #[serde(default = "default_slot_activation_threshold")]
    pub activation_threshold: f64,
    /// Whether active-slot masks are applied to cross-attention and decoder local attention.
    #[serde(default = "default_slot_attention_masking")]
    pub attention_masking: bool,
    /// Minimum highest-activation slots visible to attention when masking is enabled.
    #[serde(default = "default_minimum_visible_slots")]
    pub minimum_visible_slots: i64,
    /// Weight for assignment-mass evidence added to learned slot activation logits.
    #[serde(default = "default_slot_activation_mass_evidence_weight")]
    pub activation_mass_evidence_weight: f64,
    /// Reporting/aggregation window for balance-oriented slot usage summaries.
    #[serde(default = "default_slot_balance_window")]
    pub balance_window: usize,
}

impl Default for SlotDecompositionConfig {
    fn default() -> Self {
        Self {
            activation_temperature: default_slot_activation_temperature(),
            activation_threshold: default_slot_activation_threshold(),
            attention_masking: default_slot_attention_masking(),
            minimum_visible_slots: default_minimum_visible_slots(),
            activation_mass_evidence_weight: default_slot_activation_mass_evidence_weight(),
            balance_window: default_slot_balance_window(),
        }
    }
}

impl SlotDecompositionConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.activation_temperature.is_finite() || self.activation_temperature <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.slot_decomposition.activation_temperature must be finite and positive",
            ));
        }
        if !self.activation_threshold.is_finite()
            || self.activation_threshold <= 0.0
            || self.activation_threshold >= 1.0
        {
            return Err(ConfigValidationError::new(
                "model.slot_decomposition.activation_threshold must be finite and between 0 and 1",
            ));
        }
        if self.balance_window == 0 {
            return Err(ConfigValidationError::new(
                "model.slot_decomposition.balance_window must be greater than zero",
            ));
        }
        if self.minimum_visible_slots < 0 {
            return Err(ConfigValidationError::new(
                "model.slot_decomposition.minimum_visible_slots must be non-negative",
            ));
        }
        if !self.activation_mass_evidence_weight.is_finite()
            || self.activation_mass_evidence_weight < 0.0
        {
            return Err(ConfigValidationError::new(
                "model.slot_decomposition.activation_mass_evidence_weight must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

fn default_slot_activation_temperature() -> f64 {
    1.0
}

fn default_slot_activation_threshold() -> f64 {
    0.5
}

fn default_slot_attention_masking() -> bool {
    true
}

fn default_minimum_visible_slots() -> i64 {
    1
}

fn default_slot_activation_mass_evidence_weight() -> f64 {
    0.5
}

fn default_slot_balance_window() -> usize {
    32
}

/// Selectable topology encoder family for graph-structure ablations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TopologyEncoderKind {
    /// Legacy atom embedding plus degree projection baseline.
    Lightweight,
    /// Residual dense-adjacency graph message passing.
    #[default]
    MessagePassing,
    /// Residual graph message passing with bond-type embeddings on typed edges.
    TypedMessagePassing,
}

/// Graph topology encoder controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyEncoderConfig {
    /// Encoder family.
    #[serde(default)]
    pub kind: TopologyEncoderKind,
    /// Number of residual message-passing layers for `message_passing`.
    #[serde(default = "default_topology_message_passing_layers")]
    pub message_passing_layers: usize,
    /// Residual scale applied to each graph-message update.
    #[serde(default = "default_topology_message_residual_scale")]
    pub residual_scale: f64,
    /// Normalize aggregated neighbor messages by node degree.
    #[serde(default = "default_topology_normalize_messages")]
    pub normalize_messages: bool,
    /// Vocabulary size for typed bond embeddings. Bond type `0` is reserved for unknown.
    #[serde(default = "default_topology_bond_type_vocab_size")]
    pub bond_type_vocab_size: i64,
}

impl Default for TopologyEncoderConfig {
    fn default() -> Self {
        Self {
            kind: TopologyEncoderKind::default(),
            message_passing_layers: default_topology_message_passing_layers(),
            residual_scale: default_topology_message_residual_scale(),
            normalize_messages: default_topology_normalize_messages(),
            bond_type_vocab_size: default_topology_bond_type_vocab_size(),
        }
    }
}

impl TopologyEncoderConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if matches!(
            self.kind,
            TopologyEncoderKind::MessagePassing | TopologyEncoderKind::TypedMessagePassing
        ) && self.message_passing_layers == 0
        {
            return Err(ConfigValidationError::new(
                "model.topology_encoder.message_passing_layers must be greater than zero for message-passing topology encoders",
            ));
        }
        if !self.residual_scale.is_finite() || self.residual_scale < 0.0 {
            return Err(ConfigValidationError::new(
                "model.topology_encoder.residual_scale must be finite and non-negative",
            ));
        }
        if self.bond_type_vocab_size <= 1 {
            return Err(ConfigValidationError::new(
                "model.topology_encoder.bond_type_vocab_size must be greater than one",
            ));
        }
        Ok(())
    }
}

fn default_topology_message_passing_layers() -> usize {
    2
}

fn default_topology_message_residual_scale() -> f64 {
    0.5
}

fn default_topology_normalize_messages() -> bool {
    true
}

fn default_topology_bond_type_vocab_size() -> i64 {
    8
}

/// Selectable geometry operator family for ligand internal geometry ablations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GeometryOperatorKind {
    /// Legacy centroid-centered coordinate projection plus mean distance.
    RawCoordinateProjection,
    /// Translation-invariant E(n) pair-distance kernels aggregated per ligand atom.
    #[default]
    PairDistanceKernel,
    /// Ligand-local-frame pair messages with radial distance and local direction features.
    LocalFramePairMessage,
}

/// Geometry encoder controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryEncoderConfig {
    /// Operator family used before geometry slot decomposition.
    #[serde(default)]
    pub operator: GeometryOperatorKind,
    /// Number of radial distance kernels for `pair_distance_kernel`.
    #[serde(default = "default_geometry_distance_kernel_count")]
    pub distance_kernel_count: i64,
    /// Maximum distance covered by radial kernel centers in Angstrom.
    #[serde(default = "default_geometry_distance_kernel_max_distance")]
    pub distance_kernel_max_distance: f64,
    /// Width of the radial basis functions.
    #[serde(default = "default_geometry_distance_kernel_gamma")]
    pub distance_kernel_gamma: f64,
    /// Residual scale applied to the distance-kernel branch before fusion.
    #[serde(default = "default_geometry_operator_residual_scale")]
    pub residual_scale: f64,
}

impl Default for GeometryEncoderConfig {
    fn default() -> Self {
        Self {
            operator: GeometryOperatorKind::default(),
            distance_kernel_count: default_geometry_distance_kernel_count(),
            distance_kernel_max_distance: default_geometry_distance_kernel_max_distance(),
            distance_kernel_gamma: default_geometry_distance_kernel_gamma(),
            residual_scale: default_geometry_operator_residual_scale(),
        }
    }
}

impl GeometryEncoderConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.distance_kernel_count <= 0 {
            return Err(ConfigValidationError::new(
                "model.geometry_encoder.distance_kernel_count must be greater than zero",
            ));
        }
        if !self.distance_kernel_max_distance.is_finite()
            || self.distance_kernel_max_distance <= 0.0
        {
            return Err(ConfigValidationError::new(
                "model.geometry_encoder.distance_kernel_max_distance must be finite and positive",
            ));
        }
        if !self.distance_kernel_gamma.is_finite() || self.distance_kernel_gamma <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.geometry_encoder.distance_kernel_gamma must be finite and positive",
            ));
        }
        if !self.residual_scale.is_finite() || self.residual_scale < 0.0 {
            return Err(ConfigValidationError::new(
                "model.geometry_encoder.residual_scale must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

fn default_geometry_distance_kernel_count() -> i64 {
    8
}

fn default_geometry_distance_kernel_max_distance() -> f64 {
    8.0
}

fn default_geometry_distance_kernel_gamma() -> f64 {
    0.75
}

fn default_geometry_operator_residual_scale() -> f64 {
    0.5
}

/// Selectable pocket encoder family for context-structure ablations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PocketEncoderKind {
    /// Legacy per-pocket-atom feature and coordinate projection.
    FeatureProjection,
    /// Radius-bounded local pocket message passing over atom/residue tokens.
    #[default]
    LocalMessagePassing,
    /// Ligand-relative local-frame pocket messages using inference-available pocket coordinates.
    LigandRelativeLocalFrame,
}

/// Pocket encoder controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketEncoderConfig {
    /// Encoder family.
    #[serde(default)]
    pub kind: PocketEncoderKind,
    /// Number of residual local message-passing layers.
    #[serde(default = "default_pocket_message_passing_layers")]
    pub message_passing_layers: usize,
    /// Neighbor radius in Angstrom for local pocket aggregation.
    #[serde(default = "default_pocket_neighbor_radius")]
    pub neighbor_radius: f64,
    /// Residual scale applied to each local context update.
    #[serde(default = "default_pocket_message_residual_scale")]
    pub residual_scale: f64,
}

impl Default for PocketEncoderConfig {
    fn default() -> Self {
        Self {
            kind: PocketEncoderKind::default(),
            message_passing_layers: default_pocket_message_passing_layers(),
            neighbor_radius: default_pocket_neighbor_radius(),
            residual_scale: default_pocket_message_residual_scale(),
        }
    }
}

impl PocketEncoderConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if matches!(
            self.kind,
            PocketEncoderKind::LocalMessagePassing | PocketEncoderKind::LigandRelativeLocalFrame
        ) && self.message_passing_layers == 0
        {
            return Err(ConfigValidationError::new(
                "model.pocket_encoder.message_passing_layers must be greater than zero for message-passing pocket encoders",
            ));
        }
        if !self.neighbor_radius.is_finite() || self.neighbor_radius <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.pocket_encoder.neighbor_radius must be finite and positive",
            ));
        }
        if !self.residual_scale.is_finite() || self.residual_scale < 0.0 {
            return Err(ConfigValidationError::new(
                "model.pocket_encoder.residual_scale must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

fn default_pocket_message_passing_layers() -> usize {
    2
}

fn default_pocket_neighbor_radius() -> f64 {
    6.0
}

fn default_pocket_message_residual_scale() -> f64 {
    0.5
}

/// Config for no-coordinate-move bond refinement candidate layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondRefinementConfig {
    /// Emit a coordinate-preserving bond-logit refinement layer.
    #[serde(default = "default_emit_bond_logits_refined")]
    pub emit_bond_logits_refined: bool,
    /// Emit a coordinate-preserving valence-pruned refinement layer.
    #[serde(default = "default_emit_valence_refined")]
    pub emit_valence_refined: bool,
    /// Maximum allowed valence violations before a refined candidate is marked unsafe.
    #[serde(default)]
    pub max_valence_violations: usize,
}

impl Default for BondRefinementConfig {
    fn default() -> Self {
        Self {
            emit_bond_logits_refined: default_emit_bond_logits_refined(),
            emit_valence_refined: default_emit_valence_refined(),
            max_valence_violations: 0,
        }
    }
}

impl BondRefinementConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.emit_bond_logits_refined && self.emit_valence_refined {
            return Err(ConfigValidationError::new(
                "model.bond_refinement.emit_valence_refined requires emit_bond_logits_refined",
            ));
        }
        Ok(())
    }
}

fn default_emit_bond_logits_refined() -> bool {
    true
}

fn default_emit_valence_refined() -> bool {
    true
}

/// Config for bounded ligand pairwise geometry messages in velocity prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseGeometryConfig {
    /// Enable message aggregation.
    #[serde(default)]
    pub enabled: bool,
    /// Neighbor radius in Angstrom.
    #[serde(default = "default_pairwise_geometry_radius")]
    pub radius: f64,
    /// Maximum retained neighbors per atom.
    #[serde(default = "default_pairwise_geometry_max_neighbors")]
    pub max_neighbors: usize,
    /// Conservative residual scale for the message branch.
    #[serde(default = "default_pairwise_geometry_residual_scale")]
    pub residual_scale: f64,
}

impl Default for PairwiseGeometryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: default_pairwise_geometry_radius(),
            max_neighbors: default_pairwise_geometry_max_neighbors(),
            residual_scale: default_pairwise_geometry_residual_scale(),
        }
    }
}

impl PairwiseGeometryConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.radius.is_finite() || self.radius <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.pairwise_geometry.radius must be finite and positive",
            ));
        }
        if self.max_neighbors == 0 {
            return Err(ConfigValidationError::new(
                "model.pairwise_geometry.max_neighbors must be greater than zero",
            ));
        }
        if !self.residual_scale.is_finite() || self.residual_scale < 0.0 {
            return Err(ConfigValidationError::new(
                "model.pairwise_geometry.residual_scale must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

fn default_pairwise_geometry_radius() -> f64 {
    4.5
}

fn default_pairwise_geometry_max_neighbors() -> usize {
    16
}

fn default_pairwise_geometry_residual_scale() -> f64 {
    0.1
}

/// Selectable velocity-head family for geometry flow matching.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FlowVelocityHeadKind {
    /// Baseline pooled-conditioning geometry head.
    #[default]
    Geometry,
    /// EGNN-style scalar-message head with exact rigid-motion equivariant velocities.
    EquivariantGeometry,
    /// Gated atom-to-pocket cross-attention head.
    AtomPocketCrossAttention,
}

/// Config for flow velocity-head ablations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowVelocityHeadConfig {
    /// Velocity-head family.
    #[serde(default)]
    pub kind: FlowVelocityHeadKind,
    /// Initial bias for the local atom-pocket gate; negative is conservative.
    #[serde(default = "default_flow_velocity_gate_initial_bias")]
    pub gate_initial_bias: f64,
}

impl Default for FlowVelocityHeadConfig {
    fn default() -> Self {
        Self {
            kind: FlowVelocityHeadKind::default(),
            gate_initial_bias: default_flow_velocity_gate_initial_bias(),
        }
    }
}

impl FlowVelocityHeadConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.gate_initial_bias.is_finite() {
            return Err(ConfigValidationError::new(
                "model.flow_velocity_head.gate_initial_bias must be finite",
            ));
        }
        Ok(())
    }
}

fn default_flow_velocity_gate_initial_bias() -> f64 {
    -1.0
}

/// Controlled cross-modality interaction style used by the modular stack.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CrossAttentionMode {
    /// Preserve the original gated single-path interaction as the main ablation baseline.
    Lightweight,
    /// Add normalization, residual structure, and feed-forward refinement around gated attention.
    #[default]
    Transformer,
    /// Negative-control ablation that removes learned gate sparsity by forcing paths open.
    DirectFusionNegativeControl,
}

fn default_interaction_ff_multiplier() -> i64 {
    2
}

/// Gate tensor granularity for controlled cross-modality interaction paths.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum InteractionGateMode {
    /// One learned scalar gate per directed interaction path.
    PathScalar,
    /// One learned gate per target slot in each directed interaction path.
    #[default]
    TargetSlot,
}

/// Compact tuning controls for the Transformer-style controlled-interaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionTuningConfig {
    /// Gate tensor granularity for controlled interaction.
    #[serde(default)]
    pub gate_mode: InteractionGateMode,
    /// Temperature applied to gate logits before the sigmoid.
    #[serde(default = "default_interaction_gate_temperature")]
    pub gate_temperature: f64,
    /// Additive bias applied to gate logits before temperature scaling.
    #[serde(default = "default_interaction_gate_bias")]
    pub gate_bias: f64,
    /// Residual scale applied to the gated attention update.
    #[serde(default = "default_attention_residual_scale")]
    pub attention_residual_scale: f64,
    /// Residual scale applied to the Transformer-style feed-forward refinement.
    #[serde(default = "default_ffn_residual_scale")]
    pub ffn_residual_scale: f64,
    /// Whether the refinement feed-forward block should use pre-normalized inputs.
    #[serde(default = "default_transformer_pre_norm")]
    pub transformer_pre_norm: bool,
    /// Multiplier for ligand-pocket geometry bias injected into controlled attention.
    #[serde(default = "default_geometry_attention_bias_scale")]
    pub geometry_attention_bias_scale: f64,
    /// Multiplier for chemistry-role compatibility inside ligand-pocket attention bias.
    #[serde(default = "default_chemistry_role_attention_bias_scale")]
    pub chemistry_role_attention_bias_scale: f64,
    /// Disabled directed interaction paths by stable path name.
    #[serde(default)]
    pub disabled_paths: Vec<String>,
    /// Optional per-path multipliers inside the gate sparsity objective.
    ///
    /// Empty keeps the previous aggregate behavior: every directed path
    /// contributes `gate_abs_mean / 6` to `L_gate`.
    #[serde(default)]
    pub gate_regularization_path_weights: Vec<InteractionPathGateRegularizationWeight>,
}

impl Default for InteractionTuningConfig {
    fn default() -> Self {
        Self {
            gate_mode: InteractionGateMode::default(),
            gate_temperature: default_interaction_gate_temperature(),
            gate_bias: default_interaction_gate_bias(),
            attention_residual_scale: default_attention_residual_scale(),
            ffn_residual_scale: default_ffn_residual_scale(),
            transformer_pre_norm: default_transformer_pre_norm(),
            geometry_attention_bias_scale: default_geometry_attention_bias_scale(),
            chemistry_role_attention_bias_scale: default_chemistry_role_attention_bias_scale(),
            disabled_paths: Vec::new(),
            gate_regularization_path_weights: Vec::new(),
        }
    }
}

impl InteractionTuningConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.gate_temperature.is_finite() || self.gate_temperature <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.gate_temperature must be finite and positive",
            ));
        }
        if !self.gate_bias.is_finite() {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.gate_bias must be finite",
            ));
        }
        if !self.attention_residual_scale.is_finite() || self.attention_residual_scale <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.attention_residual_scale must be finite and positive",
            ));
        }
        if !self.ffn_residual_scale.is_finite() || self.ffn_residual_scale <= 0.0 {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.ffn_residual_scale must be finite and positive",
            ));
        }
        if !self.geometry_attention_bias_scale.is_finite()
            || self.geometry_attention_bias_scale < 0.0
        {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.geometry_attention_bias_scale must be finite and non-negative",
            ));
        }
        if !self.chemistry_role_attention_bias_scale.is_finite()
            || self.chemistry_role_attention_bias_scale < 0.0
        {
            return Err(ConfigValidationError::new(
                "model.interaction_tuning.chemistry_role_attention_bias_scale must be finite and non-negative",
            ));
        }
        for disabled_path in &self.disabled_paths {
            if !is_supported_interaction_path(disabled_path) {
                return Err(ConfigValidationError::new(format!(
                    "model.interaction_tuning.disabled_paths contains unknown path '{disabled_path}'"
                )));
            }
        }
        let mut seen_gate_weight_paths: Vec<&str> = Vec::new();
        for entry in &self.gate_regularization_path_weights {
            if !is_supported_interaction_path(&entry.path) {
                return Err(ConfigValidationError::new(format!(
                    "model.interaction_tuning.gate_regularization_path_weights contains unknown path '{}'",
                    entry.path
                )));
            }
            if !entry.weight.is_finite() || entry.weight < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "model.interaction_tuning.gate_regularization_path_weights requires non-negative finite weights (found {})",
                    entry.weight
                )));
            }
            if seen_gate_weight_paths
                .iter()
                .any(|seen_path| *seen_path == entry.path)
            {
                return Err(ConfigValidationError::new(format!(
                    "model.interaction_tuning.gate_regularization_path_weights contains duplicate path '{}'",
                    entry.path
                )));
            }
            seen_gate_weight_paths.push(&entry.path);
        }
        Ok(())
    }
}

/// Path-specific multiplier inside the gate regularization objective.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionPathGateRegularizationWeight {
    /// Stable directed path identifier.
    pub path: String,
    /// Non-negative multiplier applied before the six-path average.
    pub weight: f64,
}

impl Default for InteractionPathGateRegularizationWeight {
    fn default() -> Self {
        Self {
            path: "geo_from_pocket".to_string(),
            weight: 1.0,
        }
    }
}

fn is_supported_interaction_path(name: &str) -> bool {
    matches!(
        name,
        "topo_from_geo"
            | "topo_from_pocket"
            | "geo_from_topo"
            | "geo_from_pocket"
            | "pocket_from_topo"
            | "pocket_from_geo"
    )
}

/// Stage-level multiplier for one directed interaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPathStageMultiplier {
    /// Training stage index where the multiplier is active.
    pub training_stage: usize,
    /// Stable directed path identifier.
    pub path: String,
    /// Non-negative multiplier applied to the interaction path output.
    pub multiplier: f64,
}

impl Default for InteractionPathStageMultiplier {
    fn default() -> Self {
        Self {
            training_stage: 0,
            path: "topo_from_geo".to_string(),
            multiplier: 1.0,
        }
    }
}

/// Rollout-step bucket multiplier for one directed interaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPathRolloutBucketMultiplier {
    /// Stable directed path identifier.
    pub path: String,
    /// Inclusive lower rollout-step bound.
    pub start_step: usize,
    /// Inclusive upper rollout-step bound.
    pub end_step: usize,
    /// Non-negative multiplier applied in this bucket.
    pub multiplier: f64,
}

impl Default for InteractionPathRolloutBucketMultiplier {
    fn default() -> Self {
        Self {
            path: "topo_from_geo".to_string(),
            start_step: 0,
            end_step: 0,
            multiplier: 1.0,
        }
    }
}

/// Flow-time bucket multiplier for one directed interaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPathFlowTimeBucketMultiplier {
    /// Stable directed path identifier.
    pub path: String,
    /// Inclusive lower normalized flow-time bound.
    pub start_t: f64,
    /// Inclusive upper normalized flow-time bound.
    pub end_t: f64,
    /// Non-negative multiplier applied in this flow-time bucket.
    pub multiplier: f64,
}

impl Default for InteractionPathFlowTimeBucketMultiplier {
    fn default() -> Self {
        Self {
            path: "geo_from_pocket".to_string(),
            start_t: 0.0,
            end_t: 1.0,
            multiplier: 1.0,
        }
    }
}

/// Configurable per-path multipliers for temporal interaction scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInteractionPolicyConfig {
    /// Stage-specific path multipliers.
    #[serde(default)]
    pub stage_multipliers: Vec<InteractionPathStageMultiplier>,
    /// Rollout-step bucket multipliers.
    #[serde(default)]
    pub rollout_bucket_multipliers: Vec<InteractionPathRolloutBucketMultiplier>,
    /// Flow-time bucket multipliers used by flow-matching conditioning.
    #[serde(default)]
    pub flow_time_bucket_multipliers: Vec<InteractionPathFlowTimeBucketMultiplier>,
}

impl Default for TemporalInteractionPolicyConfig {
    fn default() -> Self {
        Self {
            stage_multipliers: Vec::new(),
            rollout_bucket_multipliers: Vec::new(),
            flow_time_bucket_multipliers: Vec::new(),
        }
    }
}

impl TemporalInteractionPolicyConfig {
    /// Whether any path has flow-time-specific modulation enabled.
    pub(crate) fn uses_flow_time_conditioning(&self) -> bool {
        !self.flow_time_bucket_multipliers.is_empty()
    }

    /// Resolve the multiplicative factor for one path given optional context.
    ///
    /// The first-level schedule is stage-specific; rollout buckets can further
    /// override for a given rollout index.
    pub(crate) fn multiplier_for_path(
        &self,
        path: &str,
        training_stage: Option<usize>,
        rollout_step_index: Option<usize>,
        flow_t: Option<f64>,
    ) -> f64 {
        let mut multiplier = 1.0;

        if let Some(training_stage) = training_stage {
            for entry in &self.stage_multipliers {
                if entry.training_stage == training_stage && entry.path == path {
                    multiplier = entry.multiplier;
                }
            }
        }

        if let Some(step_index) = rollout_step_index {
            for entry in &self.rollout_bucket_multipliers {
                if path == entry.path && step_index >= entry.start_step && step_index <= entry.end_step {
                    multiplier = entry.multiplier;
                }
            }
        }

        if let Some(flow_t) = flow_t {
            for entry in &self.flow_time_bucket_multipliers {
                if path == entry.path && flow_t >= entry.start_t && flow_t <= entry.end_t {
                    multiplier = entry.multiplier;
                }
            }
        }

        multiplier
    }

    fn validate_stage_multipliers(&self) -> Result<(), ConfigValidationError> {
        for entry in &self.stage_multipliers {
            if !is_supported_interaction_path(&entry.path) {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.stage_multipliers contains unknown path '{}'",
                    entry.path
                )));
            }
            if !entry.multiplier.is_finite() || entry.multiplier < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.stage_multipliers requires non-negative finite multipliers (found {})",
                    entry.multiplier
                )));
            }
        }
        Ok(())
    }

    fn validate_rollout_bucket_multipliers(&self) -> Result<(), ConfigValidationError> {
        for entry in &self.rollout_bucket_multipliers {
            if !is_supported_interaction_path(&entry.path) {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.rollout_bucket_multipliers contains unknown path '{}'",
                    entry.path
                )));
            }
            if !entry.multiplier.is_finite() || entry.multiplier < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.rollout_bucket_multipliers requires non-negative finite multipliers (found {})",
                    entry.multiplier
                )));
            }
            if entry.end_step < entry.start_step {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.rollout_bucket_multipliers has invalid bucket [{}, {}]",
                    entry.start_step, entry.end_step
                )));
            }
        }
        Ok(())
    }

    fn validate_flow_time_bucket_multipliers(&self) -> Result<(), ConfigValidationError> {
        for entry in &self.flow_time_bucket_multipliers {
            if !is_supported_interaction_path(&entry.path) {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.flow_time_bucket_multipliers contains unknown path '{}'",
                    entry.path
                )));
            }
            if !entry.multiplier.is_finite() || entry.multiplier < 0.0 {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.flow_time_bucket_multipliers requires non-negative finite multipliers (found {})",
                    entry.multiplier
                )));
            }
            if !entry.start_t.is_finite()
                || !entry.end_t.is_finite()
                || !(0.0..=1.0).contains(&entry.start_t)
                || !(0.0..=1.0).contains(&entry.end_t)
                || entry.end_t < entry.start_t
            {
                return Err(ConfigValidationError::new(format!(
                    "model.temporal_interaction_policy.flow_time_bucket_multipliers has invalid bucket [{}, {}]",
                    entry.start_t, entry.end_t
                )));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), ConfigValidationError> {
        self.validate_stage_multipliers()?;
        self.validate_rollout_bucket_multipliers()?;
        self.validate_flow_time_bucket_multipliers()
    }
}

fn default_interaction_gate_temperature() -> f64 {
    1.35
}

fn default_interaction_gate_bias() -> f64 {
    -0.1
}

fn default_attention_residual_scale() -> f64 {
    0.7
}

fn default_ffn_residual_scale() -> f64 {
    0.35
}

fn default_transformer_pre_norm() -> bool {
    true
}

fn default_geometry_attention_bias_scale() -> f64 {
    1.0
}

fn default_chemistry_role_attention_bias_scale() -> f64 {
    0.5
}
