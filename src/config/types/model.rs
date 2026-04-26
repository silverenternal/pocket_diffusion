/// Encoder and latent architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Shared hidden size used by modality encoders.
    pub hidden_dim: i64,
    /// Slot count upper bound per modality.
    pub num_slots: i64,
    /// Atom-type vocabulary size.
    pub atom_vocab_size: i64,
    /// Bond-type vocabulary size.
    pub bond_vocab_size: i64,
    /// Input pocket feature width.
    pub pocket_feature_dim: i64,
    /// Pairwise geometric feature width.
    pub pair_feature_dim: i64,
    /// Cross-modality interaction block style.
    #[serde(default)]
    pub interaction_mode: CrossAttentionMode,
    /// Feed-forward expansion factor used by the Transformer-style interaction block.
    #[serde(default = "default_interaction_ff_multiplier")]
    pub interaction_ff_multiplier: i64,
    /// Compact tuning knobs for the Transformer-style interaction refinement path.
    #[serde(default)]
    pub interaction_tuning: InteractionTuningConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_slots: 8,
            atom_vocab_size: 32,
            bond_vocab_size: 8,
            pocket_feature_dim: 24,
            pair_feature_dim: 8,
            interaction_mode: CrossAttentionMode::default(),
            interaction_ff_multiplier: default_interaction_ff_multiplier(),
            interaction_tuning: InteractionTuningConfig::default(),
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
        if self.interaction_ff_multiplier <= 0 {
            return Err(ConfigValidationError::new(
                "model.interaction_ff_multiplier must be greater than zero",
            ));
        }
        self.interaction_tuning.validate()?;
        Ok(())
    }
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
}

fn default_interaction_ff_multiplier() -> i64 {
    2
}

/// Compact tuning controls for the Transformer-style controlled-interaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionTuningConfig {
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
}

impl Default for InteractionTuningConfig {
    fn default() -> Self {
        Self {
            gate_temperature: default_interaction_gate_temperature(),
            gate_bias: default_interaction_gate_bias(),
            attention_residual_scale: default_attention_residual_scale(),
            ffn_residual_scale: default_ffn_residual_scale(),
            transformer_pre_norm: default_transformer_pre_norm(),
            geometry_attention_bias_scale: default_geometry_attention_bias_scale(),
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
        Ok(())
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

