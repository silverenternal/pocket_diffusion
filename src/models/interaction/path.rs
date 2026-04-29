use crate::config::TemporalInteractionPolicyConfig;

#[cfg(test)]
pub(crate) const SUPPORTED_INTERACTION_PATHS: [&str; 6] = [
    "topo_from_geo",
    "topo_from_pocket",
    "geo_from_topo",
    "geo_from_pocket",
    "pocket_from_topo",
    "pocket_from_geo",
];

#[derive(Debug, Clone, Copy)]
pub(crate) enum InteractionPath {
    TopologyFromGeometry,
    TopologyFromPocket,
    GeometryFromTopology,
    GeometryFromPocket,
    PocketFromTopology,
    PocketFromGeometry,
}

impl InteractionPath {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TopologyFromGeometry => "topo_from_geo",
            Self::TopologyFromPocket => "topo_from_pocket",
            Self::GeometryFromTopology => "geo_from_topo",
            Self::GeometryFromPocket => "geo_from_pocket",
            Self::PocketFromTopology => "pocket_from_topo",
            Self::PocketFromGeometry => "pocket_from_geo",
        }
    }

    pub(crate) fn has_attention_bias(self) -> bool {
        matches!(self, Self::GeometryFromPocket | Self::PocketFromGeometry)
    }

    pub(crate) fn role(self) -> &'static str {
        match self {
            Self::TopologyFromGeometry => "topology-informed bond plausibility",
            Self::TopologyFromPocket => "pocket-informed ligand chemistry preference",
            Self::GeometryFromTopology => "topology-constrained conformer geometry",
            Self::GeometryFromPocket => "pocket-shaped pose refinement",
            Self::PocketFromTopology => "ligand-chemistry pocket compatibility",
            Self::PocketFromGeometry => "pose occupancy feedback",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InteractionDiagnosticProvenance {
    /// Aggregated over an entire batch.
    BatchAggregate,
    /// Computed for one concrete example slice.
    PerExample,
}

impl PartialEq for InteractionDiagnosticProvenance {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::BatchAggregate, Self::BatchAggregate) | (Self::PerExample, Self::PerExample)
        )
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct InteractionExecutionContext {
    /// Optional training-stage index for staged policy and diagnostics.
    pub training_stage: Option<usize>,
    /// Optional global training step index for sampling-related context.
    pub training_step: Option<usize>,
    /// Optional epoch index for sampling- and diagnostics-related context.
    pub epoch_index: Option<usize>,
    /// Optional random-order seed used by this epoch.
    pub sample_order_seed: Option<u64>,
    /// Optional rollout step index for inference-phase diagnostics.
    pub rollout_step_index: Option<usize>,
    /// Optional flow-time value used for flow-matching conditioning.
    pub flow_t: Option<f64>,
}

pub(crate) fn path_scale(
    policy: &TemporalInteractionPolicyConfig,
    path: InteractionPath,
    context: &InteractionExecutionContext,
) -> f64 {
    policy.multiplier_for_path(
        path.as_str(),
        context.training_stage,
        context.rollout_step_index,
        context.flow_t,
    )
}

pub(crate) fn flow_time_bucket_label(flow_t: f64) -> &'static str {
    if flow_t < (1.0 / 3.0) {
        "low"
    } else if flow_t < (2.0 / 3.0) {
        "mid"
    } else {
        "high"
    }
}
