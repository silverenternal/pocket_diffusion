/// Build the primary flow branch schedule report for a training step.
pub(crate) fn primary_branch_schedule_report(
    forwards: &[crate::models::ResearchForward],
    training_step: usize,
    stage_index: Option<usize>,
    components: &PrimaryObjectiveComponentMetrics,
    config: &crate::config::ResearchConfig,
) -> PrimaryBranchScheduleReport {
    let Some(flow) = forwards
        .iter()
        .find_map(|forward| forward.generation.flow_matching.as_ref())
    else {
        return PrimaryBranchScheduleReport::default();
    };
    let objective_flow_weight = primary_flow_objective_weight(config);
    let mut component_audits = primary_branch_component_audit_map(components);
    let geometry_values = branch_scale_values(
        components.flow_velocity.unwrap_or(0.0) + components.flow_endpoint.unwrap_or(0.0),
        flow.branch_weights.geometry,
        objective_flow_weight,
    );
    let mut entries = vec![PrimaryBranchWeightRecord {
        branch_name: "geometry".to_string(),
        unweighted_value: geometry_values.unweighted,
        effective_weight: flow.branch_weights.geometry,
        schedule_multiplier: config
            .generation_method
            .flow_matching
            .multi_modal
            .branch_schedule
            .geometry
            .effective_multiplier(Some(training_step)),
        weighted_value: geometry_values.weighted,
        optimizer_facing: flow.branch_weights.geometry.is_finite()
            && flow.branch_weights.geometry > 0.0,
        provenance: flow.flow_contract_version.clone(),
        component_audit: take_branch_component_audit(&mut component_audits, "geometry"),
        target_matching_policy: Some(flow.target_matching_policy.clone()),
        target_matching_mean_cost: Some(flow.target_matching_mean_cost),
        target_matching_max_cost: Some(flow.target_matching_cost_summary.max_cost),
        target_matching_total_cost: Some(flow.target_matching_cost_summary.total_cost),
        target_matching_coverage: Some(flow.target_matching_coverage),
        target_matching_matched_count: Some(flow.target_matching_cost_summary.matched_count),
        target_matching_unmatched_generated_count: Some(
            flow.target_matching_cost_summary.unmatched_generated_count,
        ),
        target_matching_unmatched_target_count: Some(
            flow.target_matching_cost_summary.unmatched_target_count,
        ),
        target_matching_exact_assignment: Some(flow.target_matching_cost_summary.exact_assignment),
    }];
    if let Some(molecular) = flow.molecular.as_ref() {
        for (branch_name, effective_weight, schedule_multiplier, component_value) in [
            (
                "atom_type",
                molecular.branch_weights.atom_type,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .atom_type
                    .effective_multiplier(Some(training_step)),
                components.flow_atom_type.unwrap_or(0.0),
            ),
            (
                "bond",
                molecular.branch_weights.bond,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .bond
                    .effective_multiplier(Some(training_step)),
                components.flow_bond.unwrap_or(0.0),
            ),
            (
                "topology",
                molecular.branch_weights.topology,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .topology
                    .effective_multiplier(Some(training_step)),
                components.flow_topology.unwrap_or(0.0),
            ),
            (
                "pocket_context",
                molecular.branch_weights.pocket_context,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .pocket_context
                    .effective_multiplier(Some(training_step)),
                components.flow_pocket_context.unwrap_or(0.0),
            ),
            (
                "synchronization",
                molecular.branch_weights.synchronization,
                config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
                    .synchronization
                    .effective_multiplier(Some(training_step)),
                components.flow_synchronization.unwrap_or(0.0),
            ),
        ] {
            let branch_values =
                branch_scale_values(component_value, effective_weight, objective_flow_weight);
            entries.push(PrimaryBranchWeightRecord {
                branch_name: branch_name.to_string(),
                unweighted_value: branch_values.unweighted,
                effective_weight,
                schedule_multiplier,
                weighted_value: branch_values.weighted,
                optimizer_facing: effective_weight.is_finite() && effective_weight > 0.0,
                provenance: format!(
                    "{}:{}",
                    flow.flow_contract_version.as_str(),
                    molecular.target_alignment_policy.as_str()
                ),
                component_audit: take_branch_component_audit(&mut component_audits, branch_name),
                target_matching_policy: Some(molecular.target_matching_policy.clone()),
                target_matching_mean_cost: Some(molecular.target_matching_mean_cost),
                target_matching_max_cost: Some(molecular.target_matching_cost_summary.max_cost),
                target_matching_total_cost: Some(molecular.target_matching_cost_summary.total_cost),
                target_matching_coverage: Some(molecular.target_matching_coverage),
                target_matching_matched_count: Some(
                    molecular.target_matching_cost_summary.matched_count,
                ),
                target_matching_unmatched_generated_count: Some(
                    molecular
                        .target_matching_cost_summary
                        .unmatched_generated_count,
                ),
                target_matching_unmatched_target_count: Some(
                    molecular
                        .target_matching_cost_summary
                        .unmatched_target_count,
                ),
                target_matching_exact_assignment: Some(
                    molecular.target_matching_cost_summary.exact_assignment,
                ),
            });
        }
    }
    PrimaryBranchScheduleReport {
        observed: true,
        training_step: Some(training_step),
        stage_index,
        source: "generation_method.flow_matching.multi_modal.branch_schedule".to_string(),
        entries,
    }
}

struct BranchScaleValues {
    unweighted: f64,
    weighted: f64,
}

fn primary_flow_objective_weight(config: &crate::config::ResearchConfig) -> f64 {
    match config.training.primary_objective {
        crate::config::PrimaryObjectiveConfig::FlowMatching => config.training.flow_matching_loss_weight,
        crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching => config.training.hybrid_flow_weight,
        _ => 1.0,
    }
}

fn branch_scale_values(
    component_value: f64,
    effective_weight: f64,
    objective_flow_weight: f64,
) -> BranchScaleValues {
    let weighted = if objective_flow_weight.is_finite() && objective_flow_weight > 0.0 {
        component_value / objective_flow_weight
    } else {
        component_value
    };
    let unweighted = if effective_weight.is_finite() && effective_weight > 0.0 {
        weighted / effective_weight
    } else {
        0.0
    };
    BranchScaleValues {
        unweighted,
        weighted,
    }
}

fn primary_branch_component_audit_map(
    components: &PrimaryObjectiveComponentMetrics,
) -> std::collections::BTreeMap<String, PrimaryBranchComponentAudit> {
    components
        .branch_component_audits()
        .into_iter()
        .map(|audit| (audit.branch_name.clone(), audit))
        .collect()
}

fn take_branch_component_audit(
    audits: &mut std::collections::BTreeMap<String, PrimaryBranchComponentAudit>,
    branch_name: &str,
) -> PrimaryBranchComponentAudit {
    audits
        .remove(branch_name)
        .unwrap_or_else(|| PrimaryBranchComponentAudit::for_branch(branch_name))
}
