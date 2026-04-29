use tch::{Kind, Tensor};

use super::bias::{
    role_channel, role_tensor_is_unusable, slot_pair_profile, ROLE_ACCEPTOR, ROLE_AROMATIC,
    ROLE_AVAILABLE, ROLE_DONOR, ROLE_HYDROPHOBIC, ROLE_METAL_BINDING, ROLE_NEGATIVE, ROLE_POSITIVE,
};
use super::path::{
    flow_time_bucket_label, InteractionDiagnosticProvenance, InteractionExecutionContext,
    InteractionPath,
};
use crate::models::{BatchedCrossAttentionOutput, CrossAttentionOutput};

#[allow(dead_code)] // Audit/reporting fields are intentionally retained even when not read in lib code.
#[derive(Debug, Clone)]
pub(crate) struct CrossModalInteractionPathDiagnostics {
    /// Stable directed-path name.
    pub path_name: String,
    /// Stable directed-path role for chemistry-level interpretation.
    pub path_role: &'static str,
    /// Mean gate activation in this path.
    pub gate_mean: f64,
    /// Mean absolute gate activation in this path.
    pub gate_abs_mean: f64,
    /// Fraction of gate elements effectively closed.
    pub gate_closed_fraction: f64,
    /// Fraction of gate elements effectively open.
    pub gate_open_fraction: f64,
    /// Fraction of gate elements near either saturation boundary.
    pub gate_saturation_fraction: f64,
    /// Number of gate elements represented by this path diagnostic.
    pub gate_element_count: usize,
    /// Entropy of the gate tensor interpreted as normalized gate mass.
    pub gate_entropy: f64,
    /// Mean sigmoid derivative proxy `gate * (1 - gate)`.
    pub gate_gradient_proxy: f64,
    /// Whether this path's gate was forced open by a negative-control ablation.
    pub forced_open: bool,
    /// Temporal or staged multiplier applied to this path's update, separate from gate value.
    pub path_scale: f64,
    /// Compact status label for active, closed, open, or saturated gates.
    pub gate_status: String,
    /// Warning emitted when a path appears always-open, always-closed, or saturated.
    pub gate_warning: Option<String>,
    /// Mean attention entropy over query tokens.
    pub attention_entropy: f64,
    /// Mean norm of attended tokens.
    pub attended_norm: f64,
    /// Mean norm of the path update after applying temporal path scaling.
    pub effective_update_norm: f64,
    /// Optional geometry-pocket conditioning bias mean.
    pub bias_mean: Option<f64>,
    /// Optional geometry-pocket conditioning bias minimum.
    pub bias_min: Option<f64>,
    /// Optional geometry-pocket conditioning bias maximum.
    pub bias_max: Option<f64>,
    /// Optional geometry-pocket conditioning scale.
    pub bias_scale: Option<f64>,
    /// Optional fraction of valid ligand-pocket pairs with available chemistry-role evidence.
    pub chemistry_role_coverage: Option<f64>,
    /// Optional attention-weighted topology-pocket pharmacophore compatibility coverage.
    pub pharmacophore_role_coverage: Option<f64>,
    /// Optional attention-weighted topology-pocket pharmacophore conflict rate.
    pub pharmacophore_role_conflict_rate: Option<f64>,
    /// Provenance label for pharmacophore diagnostics.
    pub pharmacophore_role_provenance: Option<String>,
    /// Execution provenance used to produce this diagnostic.
    pub provenance: InteractionDiagnosticProvenance,
    /// Optional training stage in which the path was executed.
    pub training_stage: Option<usize>,
    /// Optional rollout step at which the path was executed.
    pub rollout_step_index: Option<usize>,
    /// Optional flow time sampled for this execution.
    pub flow_t: Option<f64>,
    /// Coarse flow-time bucket used for grouped gate diagnostics.
    pub flow_time_bucket: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct CrossModalInteractionDiagnostics {
    pub topo_from_geo: CrossModalInteractionPathDiagnostics,
    pub topo_from_pocket: CrossModalInteractionPathDiagnostics,
    pub geo_from_topo: CrossModalInteractionPathDiagnostics,
    pub geo_from_pocket: CrossModalInteractionPathDiagnostics,
    pub pocket_from_topo: CrossModalInteractionPathDiagnostics,
    pub pocket_from_geo: CrossModalInteractionPathDiagnostics,
}

#[allow(dead_code)] // Batched diagnostics are consumed selectively by validation and future reports.
#[derive(Debug, Clone)]
pub(crate) struct BatchedCrossModalInteractionDiagnostics {
    pub topo_from_geo: CrossModalInteractionPathDiagnostics,
    pub topo_from_pocket: CrossModalInteractionPathDiagnostics,
    pub geo_from_topo: CrossModalInteractionPathDiagnostics,
    pub geo_from_pocket: CrossModalInteractionPathDiagnostics,
    pub pocket_from_topo: CrossModalInteractionPathDiagnostics,
    pub pocket_from_geo: CrossModalInteractionPathDiagnostics,
}

pub(crate) fn interaction_path_diagnostics(
    path: InteractionPath,
    output: &CrossAttentionOutput,
    path_scale: f64,
    bias: Option<(Tensor, f64, Option<f64>)>,
    context: &InteractionExecutionContext,
    provenance: InteractionDiagnosticProvenance,
) -> CrossModalInteractionPathDiagnostics {
    let (bias_mean, bias_min, bias_max, bias_scale, chemistry_role_coverage) =
        if path.has_attention_bias() {
            match bias {
                Some((values, scale, coverage)) if values.numel() > 0 => (
                    Some(values.mean(Kind::Float).double_value(&[])),
                    Some(values.min().double_value(&[])),
                    Some(values.max().double_value(&[])),
                    Some(scale),
                    coverage,
                ),
                _ => (None, None, None, None, None),
            }
        } else {
            (None, None, None, None, None)
        };

    let gate = &output.gate;
    let attention = &output.attention_weights;
    let attended = &output.attended_tokens;
    let gate_stats = gate_distribution_stats(gate);
    let attention_entropy = if attention.numel() == 0 {
        0.0
    } else {
        let normalized_attention = attention.clamp_min(1e-12);
        let entropy = -(normalized_attention.log() * &normalized_attention).sum_dim_intlist(
            [1].as_slice(),
            false,
            Kind::Float,
        );
        entropy.mean(Kind::Float).double_value(&[])
    };
    let attended_norm = if attended.numel() == 0 {
        0.0
    } else {
        attended
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            .sqrt()
            .mean(Kind::Float)
            .double_value(&[])
    };

    CrossModalInteractionPathDiagnostics {
        path_name: path.as_str().to_string(),
        path_role: path.role(),
        gate_mean: gate_stats.mean,
        gate_abs_mean: gate_stats.abs_mean,
        gate_closed_fraction: gate_stats.closed_fraction,
        gate_open_fraction: gate_stats.open_fraction,
        gate_saturation_fraction: gate_stats.saturation_fraction,
        gate_element_count: gate_stats.element_count,
        gate_entropy: gate_stats.entropy,
        gate_gradient_proxy: gate_stats.gradient_proxy,
        forced_open: output.forced_open,
        path_scale,
        gate_status: gate_stats.status,
        gate_warning: gate_stats.warning,
        attention_entropy,
        attended_norm,
        effective_update_norm: attended_norm,
        bias_mean,
        bias_min,
        bias_max,
        bias_scale,
        chemistry_role_coverage,
        pharmacophore_role_coverage: None,
        pharmacophore_role_conflict_rate: None,
        pharmacophore_role_provenance: None,
        provenance,
        training_stage: context.training_stage,
        rollout_step_index: context.rollout_step_index,
        flow_t: context.flow_t,
        flow_time_bucket: context
            .flow_t
            .map(flow_time_bucket_label)
            .map(str::to_string),
    }
}

struct GateDistributionStats {
    mean: f64,
    abs_mean: f64,
    closed_fraction: f64,
    open_fraction: f64,
    saturation_fraction: f64,
    element_count: usize,
    entropy: f64,
    gradient_proxy: f64,
    status: String,
    warning: Option<String>,
}

fn gate_distribution_stats(gate: &Tensor) -> GateDistributionStats {
    if gate.numel() == 0 {
        return GateDistributionStats {
            mean: 0.0,
            abs_mean: 0.0,
            closed_fraction: 1.0,
            open_fraction: 0.0,
            saturation_fraction: 1.0,
            element_count: 0,
            entropy: 0.0,
            gradient_proxy: 0.0,
            status: "always_closed".to_string(),
            warning: Some("gate tensor is empty or disabled for this path".to_string()),
        };
    }

    let mean = gate.mean(Kind::Float).double_value(&[]);
    let abs_mean = gate.abs().mean(Kind::Float).double_value(&[]);
    let closed_fraction = gate
        .le(0.05)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[]);
    let open_fraction = gate
        .ge(0.95)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[]);
    let saturation_fraction = gate
        .le(0.05)
        .logical_or(&gate.ge(0.95))
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[]);
    let element_count = gate.numel();
    let gate_mass = gate.abs();
    let normalized_gate_mass =
        (&gate_mass / gate_mass.sum(Kind::Float).clamp_min(1e-12)).clamp_min(1e-12);
    let entropy =
        (-(&normalized_gate_mass * normalized_gate_mass.log()).sum(Kind::Float)).double_value(&[]);
    let one = Tensor::ones_like(&gate);
    let gradient_proxy = (gate * (one - gate)).mean(Kind::Float).double_value(&[]);
    let (status, warning) = gate_health_status(
        closed_fraction,
        open_fraction,
        saturation_fraction,
        gradient_proxy,
    );
    GateDistributionStats {
        mean,
        abs_mean,
        closed_fraction,
        open_fraction,
        saturation_fraction,
        element_count,
        entropy,
        gradient_proxy,
        status,
        warning,
    }
}

fn gate_health_status(
    closed_fraction: f64,
    open_fraction: f64,
    saturation_fraction: f64,
    gradient_proxy: f64,
) -> (String, Option<String>) {
    if closed_fraction >= 0.98 {
        (
            "always_closed".to_string(),
            Some("gate is effectively always closed for this path".to_string()),
        )
    } else if open_fraction >= 0.98 {
        (
            "always_open".to_string(),
            Some("gate is effectively always open for this path".to_string()),
        )
    } else if saturation_fraction >= 0.80 || gradient_proxy <= 0.02 {
        (
            "saturated".to_string(),
            Some("gate is near a saturation boundary; gradient signal may be weak".to_string()),
        )
    } else {
        ("active".to_string(), None)
    }
}

pub(crate) fn interaction_path_diagnostics_batched(
    path: InteractionPath,
    output: &BatchedCrossAttentionOutput,
    path_scale: f64,
    bias: Option<(Tensor, f64, Option<f64>)>,
    context: &InteractionExecutionContext,
    provenance: InteractionDiagnosticProvenance,
) -> CrossModalInteractionPathDiagnostics {
    let (bias_mean, bias_min, bias_max, bias_scale, chemistry_role_coverage) =
        if path.has_attention_bias() {
            match bias {
                Some((values, scale, coverage)) if values.numel() > 0 => (
                    Some(values.mean(Kind::Float).double_value(&[])),
                    Some(values.min().double_value(&[])),
                    Some(values.max().double_value(&[])),
                    Some(scale),
                    coverage,
                ),
                _ => (None, None, None, None, None),
            }
        } else {
            (None, None, None, None, None)
        };

    let gate = &output.gate;
    let attention = &output.attention_weights;
    let attended = &output.attended_tokens;
    let gate_stats = gate_distribution_stats(gate);
    let attention_entropy = if attention.numel() == 0 {
        0.0
    } else {
        let normalized_attention = attention.clamp_min(1e-12);
        let entropy = -(normalized_attention.log() * &normalized_attention).sum_dim_intlist(
            [2].as_slice(),
            false,
            Kind::Float,
        );
        entropy.mean(Kind::Float).double_value(&[])
    };
    let attended_norm = if attended.numel() == 0 {
        0.0
    } else {
        attended
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([2].as_slice(), false, Kind::Float)
            .sqrt()
            .mean(Kind::Float)
            .double_value(&[])
    };

    CrossModalInteractionPathDiagnostics {
        path_name: path.as_str().to_string(),
        path_role: path.role(),
        gate_mean: gate_stats.mean,
        gate_abs_mean: gate_stats.abs_mean,
        gate_closed_fraction: gate_stats.closed_fraction,
        gate_open_fraction: gate_stats.open_fraction,
        gate_saturation_fraction: gate_stats.saturation_fraction,
        gate_element_count: gate_stats.element_count,
        gate_entropy: gate_stats.entropy,
        gate_gradient_proxy: gate_stats.gradient_proxy,
        forced_open: output.forced_open,
        path_scale,
        gate_status: gate_stats.status,
        gate_warning: gate_stats.warning,
        attention_entropy,
        attended_norm,
        effective_update_norm: attended_norm,
        bias_mean,
        bias_min,
        bias_max,
        bias_scale,
        chemistry_role_coverage,
        pharmacophore_role_coverage: None,
        pharmacophore_role_conflict_rate: None,
        pharmacophore_role_provenance: None,
        provenance,
        training_stage: context.training_stage,
        rollout_step_index: context.rollout_step_index,
        flow_t: context.flow_t,
        flow_time_bucket: context
            .flow_t
            .map(flow_time_bucket_label)
            .map(str::to_string),
    }
}

pub(crate) fn attach_topology_pocket_pharmacophore_path_diagnostics(
    topo_from_pocket: &mut CrossModalInteractionPathDiagnostics,
    pocket_from_topo: &mut CrossModalInteractionPathDiagnostics,
    ligand_roles: &Tensor,
    ligand_slot_weights: &Tensor,
    pocket_roles: &Tensor,
    pocket_slot_weights: &Tensor,
    topo_from_pocket_attention: &Tensor,
    pocket_from_topo_attention: &Tensor,
) {
    let topo_to_pocket = topology_pocket_pharmacophore_diagnostics(
        ligand_roles,
        ligand_slot_weights,
        pocket_roles,
        pocket_slot_weights,
        topo_from_pocket_attention,
    );
    let pocket_to_topo_attention = transpose_pocket_topology_attention(pocket_from_topo_attention);
    let pocket_to_topo = topology_pocket_pharmacophore_diagnostics(
        ligand_roles,
        ligand_slot_weights,
        pocket_roles,
        pocket_slot_weights,
        &pocket_to_topo_attention,
    );

    apply_pharmacophore_diagnostics(topo_from_pocket, topo_to_pocket);
    apply_pharmacophore_diagnostics(pocket_from_topo, pocket_to_topo);
}

fn apply_pharmacophore_diagnostics(
    path: &mut CrossModalInteractionPathDiagnostics,
    diagnostics: PharmacophoreCompatibilityDiagnostics,
) {
    path.pharmacophore_role_coverage = Some(diagnostics.coverage);
    path.pharmacophore_role_conflict_rate = Some(diagnostics.conflict_rate);
    path.pharmacophore_role_provenance = Some(diagnostics.provenance.to_string());
}

#[derive(Debug, Clone, Copy)]
struct PharmacophoreCompatibilityDiagnostics {
    coverage: f64,
    conflict_rate: f64,
    provenance: &'static str,
}

fn topology_pocket_pharmacophore_diagnostics(
    ligand_roles: &Tensor,
    ligand_slot_weights: &Tensor,
    pocket_roles: &Tensor,
    pocket_slot_weights: &Tensor,
    topology_to_pocket_attention: &Tensor,
) -> PharmacophoreCompatibilityDiagnostics {
    let ligand_roles = ensure_batched_roles(ligand_roles);
    let pocket_roles = ensure_batched_roles(pocket_roles);
    if role_tensor_is_unusable(&ligand_roles) || role_tensor_is_unusable(&pocket_roles) {
        return PharmacophoreCompatibilityDiagnostics {
            coverage: 0.0,
            conflict_rate: 0.0,
            provenance: "unavailable",
        };
    }

    let ligand_slot_weights = ensure_batched_slot_weights(ligand_slot_weights);
    let pocket_slot_weights = ensure_batched_slot_weights(pocket_slot_weights);
    let attention = ensure_batched_attention(topology_to_pocket_attention);
    let batch = attention.size()[0];
    let topology_slots = attention.size()[1];
    let pocket_slots = attention.size()[2];
    let slot_profile = slot_pair_profile(
        Some(&ligand_slot_weights),
        Some(&pocket_slot_weights),
        batch,
        topology_slots,
        pocket_slots,
        attention.device(),
    );
    let attention_strength = (attention.to_kind(Kind::Float) * slot_profile).sum_dim_intlist(
        [1, 2].as_slice(),
        false,
        Kind::Float,
    ) / attention
        .to_kind(Kind::Float)
        .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        .clamp_min(1e-6);

    let ligand_profile = role_profile(&ligand_roles);
    let pocket_profile = role_profile(&pocket_roles);
    let compatible = profile_pharmacophore_compatibility(&ligand_profile, &pocket_profile);
    let conflict = profile_pharmacophore_conflict(&ligand_profile, &pocket_profile);
    let ligand_availability =
        role_channel(&ligand_roles, ROLE_AVAILABLE).mean_dim([1, 2].as_slice(), false, Kind::Float);
    let pocket_availability =
        role_channel(&pocket_roles, ROLE_AVAILABLE).mean_dim([1, 2].as_slice(), false, Kind::Float);
    let evidence = ligand_availability.minimum(&pocket_availability);
    let coverage = (compatible * &attention_strength * &evidence)
        .clamp(0.0, 1.0)
        .mean(Kind::Float)
        .double_value(&[]);
    let conflict_rate = (conflict * attention_strength * evidence)
        .clamp(0.0, 1.0)
        .mean(Kind::Float)
        .double_value(&[]);

    PharmacophoreCompatibilityDiagnostics {
        coverage,
        conflict_rate,
        provenance: "heuristic",
    }
}

fn role_profile(roles: &Tensor) -> Tensor {
    let availability = role_channel(roles, ROLE_AVAILABLE);
    (roles.to_kind(Kind::Float) * &availability).sum_dim_intlist([1].as_slice(), false, Kind::Float)
        / availability
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            .clamp_min(1.0)
}

fn profile_channel(profile: &Tensor, channel: i64) -> Tensor {
    profile.narrow(1, channel, 1).squeeze_dim(1)
}

fn profile_pharmacophore_compatibility(ligand_profile: &Tensor, pocket_profile: &Tensor) -> Tensor {
    let hbond = profile_channel(ligand_profile, ROLE_DONOR)
        * profile_channel(pocket_profile, ROLE_ACCEPTOR)
        + profile_channel(ligand_profile, ROLE_ACCEPTOR)
            * profile_channel(pocket_profile, ROLE_DONOR);
    let hydrophobic = profile_channel(ligand_profile, ROLE_HYDROPHOBIC)
        * profile_channel(pocket_profile, ROLE_HYDROPHOBIC);
    let aromatic = profile_channel(ligand_profile, ROLE_AROMATIC)
        * profile_channel(pocket_profile, ROLE_AROMATIC);
    let charge = profile_channel(ligand_profile, ROLE_POSITIVE)
        * profile_channel(pocket_profile, ROLE_NEGATIVE)
        + profile_channel(ligand_profile, ROLE_NEGATIVE)
            * profile_channel(pocket_profile, ROLE_POSITIVE);
    let metal = profile_channel(ligand_profile, ROLE_METAL_BINDING)
        * profile_channel(pocket_profile, ROLE_METAL_BINDING);
    (hbond + hydrophobic + aromatic + charge + metal).clamp(0.0, 1.0)
}

fn profile_pharmacophore_conflict(ligand_profile: &Tensor, pocket_profile: &Tensor) -> Tensor {
    let donor_donor =
        profile_channel(ligand_profile, ROLE_DONOR) * profile_channel(pocket_profile, ROLE_DONOR);
    let acceptor_acceptor = profile_channel(ligand_profile, ROLE_ACCEPTOR)
        * profile_channel(pocket_profile, ROLE_ACCEPTOR);
    let same_positive = profile_channel(ligand_profile, ROLE_POSITIVE)
        * profile_channel(pocket_profile, ROLE_POSITIVE);
    let same_negative = profile_channel(ligand_profile, ROLE_NEGATIVE)
        * profile_channel(pocket_profile, ROLE_NEGATIVE);
    (donor_donor + acceptor_acceptor + same_positive + same_negative).clamp(0.0, 1.0)
}

fn ensure_batched_roles(roles: &Tensor) -> Tensor {
    if roles.size().len() == 2 {
        roles.unsqueeze(0)
    } else {
        roles.shallow_clone()
    }
}

fn ensure_batched_slot_weights(weights: &Tensor) -> Tensor {
    if weights.size().len() == 1 {
        weights.unsqueeze(0)
    } else {
        weights.shallow_clone()
    }
}

fn ensure_batched_attention(attention: &Tensor) -> Tensor {
    if attention.size().len() == 2 {
        attention.unsqueeze(0)
    } else {
        attention.shallow_clone()
    }
}

fn transpose_pocket_topology_attention(attention: &Tensor) -> Tensor {
    match attention.size().len() {
        2 => attention.transpose(0, 1),
        3 => attention.transpose(1, 2),
        _ => attention.shallow_clone(),
    }
}
