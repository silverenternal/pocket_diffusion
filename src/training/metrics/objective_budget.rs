#[derive(Debug, Clone, Copy)]
struct ObjectiveBudgetFamilySpec {
    family: &'static str,
    members: &'static [AuxiliaryObjectiveFamily],
}

const POCKET_INTERACTION_BUDGET_MEMBERS: &[AuxiliaryObjectiveFamily] = &[
    AuxiliaryObjectiveFamily::PocketContact,
    AuxiliaryObjectiveFamily::PocketPairDistance,
    AuxiliaryObjectiveFamily::PocketClash,
    AuxiliaryObjectiveFamily::PocketShapeComplementarity,
    AuxiliaryObjectiveFamily::PocketEnvelope,
    AuxiliaryObjectiveFamily::PocketPrior,
];
const CHEMISTRY_BUDGET_MEMBERS: &[AuxiliaryObjectiveFamily] = &[
    AuxiliaryObjectiveFamily::ValenceGuardrail,
    AuxiliaryObjectiveFamily::BondLengthGuardrail,
    AuxiliaryObjectiveFamily::NonbondedDistanceGuardrail,
    AuxiliaryObjectiveFamily::AngleGuardrail,
];
const REDUNDANCY_BUDGET_MEMBERS: &[AuxiliaryObjectiveFamily] = &[
    AuxiliaryObjectiveFamily::IntraRed,
    AuxiliaryObjectiveFamily::Consistency,
];
const PROBE_BUDGET_MEMBERS: &[AuxiliaryObjectiveFamily] = &[
    AuxiliaryObjectiveFamily::Probe,
    AuxiliaryObjectiveFamily::PharmacophoreProbe,
];
const LEAKAGE_BUDGET_MEMBERS: &[AuxiliaryObjectiveFamily] = &[
    AuxiliaryObjectiveFamily::Leak,
    AuxiliaryObjectiveFamily::PharmacophoreLeakage,
];
const GATE_BUDGET_MEMBERS: &[AuxiliaryObjectiveFamily] = &[AuxiliaryObjectiveFamily::Gate];
const SLOT_BUDGET_MEMBERS: &[AuxiliaryObjectiveFamily] = &[AuxiliaryObjectiveFamily::Slot];

const OBJECTIVE_BUDGET_FAMILY_SPECS: &[ObjectiveBudgetFamilySpec] = &[
    ObjectiveBudgetFamilySpec {
        family: "pocket_interaction",
        members: POCKET_INTERACTION_BUDGET_MEMBERS,
    },
    ObjectiveBudgetFamilySpec {
        family: "chemistry",
        members: CHEMISTRY_BUDGET_MEMBERS,
    },
    ObjectiveBudgetFamilySpec {
        family: "redundancy",
        members: REDUNDANCY_BUDGET_MEMBERS,
    },
    ObjectiveBudgetFamilySpec {
        family: "probe",
        members: PROBE_BUDGET_MEMBERS,
    },
    ObjectiveBudgetFamilySpec {
        family: "leakage",
        members: LEAKAGE_BUDGET_MEMBERS,
    },
    ObjectiveBudgetFamilySpec {
        family: "gate",
        members: GATE_BUDGET_MEMBERS,
    },
    ObjectiveBudgetFamilySpec {
        family: "slot",
        members: SLOT_BUDGET_MEMBERS,
    },
];

/// Stable objective-budget family names in reporting order.
pub fn objective_budget_family_names() -> impl Iterator<Item = &'static str> {
    ["task", "rollout"]
        .into_iter()
        .chain(OBJECTIVE_BUDGET_FAMILY_SPECS.iter().map(|spec| spec.family))
}

/// Build the high-level optimizer objective family budget report.
pub(crate) fn objective_family_budget_report(
    primary_value: f64,
    primary_weight: f64,
    rollout: &RolloutTrainingLossMetrics,
    auxiliary_report: &AuxiliaryObjectiveReport,
    scale_config: &crate::config::ObjectiveScaleDiagnosticsConfig,
) -> ObjectiveFamilyBudgetReport {
    let mut entries = vec![
        ObjectiveFamilyBudgetEntry::new(
            "task",
            primary_value,
            primary_weight,
            primary_value * primary_weight,
            primary_weight.is_finite() && primary_weight > 0.0,
        ),
        ObjectiveFamilyBudgetEntry::new(
            "rollout",
            rollout.rollout_state_loss,
            1.0,
            rollout.rollout_state_loss,
            rollout.active && rollout.rollout_state_loss.is_finite(),
        ),
    ];
    entries.extend(
        OBJECTIVE_BUDGET_FAMILY_SPECS
            .iter()
            .map(|spec| aggregate_auxiliary_budget_family(spec.family, auxiliary_report, spec.members)),
    );

    let mut report = ObjectiveFamilyBudgetReport {
        entries,
        ..ObjectiveFamilyBudgetReport::default()
    };
    report.apply_budget_caps(
        scale_config.family_budget_cap_fraction,
        scale_config.family_budget_action.as_str(),
        scale_config.warning_ratio,
        scale_config.epsilon,
    );
    report
}

fn aggregate_auxiliary_budget_family(
    family: &str,
    auxiliary_report: &AuxiliaryObjectiveReport,
    members: &[AuxiliaryObjectiveFamily],
) -> ObjectiveFamilyBudgetEntry {
    let mut unweighted_value = 0.0;
    let mut weighted_value = 0.0;
    let mut enabled = false;
    for member in members {
        if let Some(entry) = auxiliary_report
            .entries
            .iter()
            .find(|entry| entry.family == *member)
        {
            unweighted_value += entry.unweighted_value;
            weighted_value += entry.weighted_value;
            enabled |= entry.enabled;
        }
    }
    let effective_weight = if unweighted_value.is_finite() && unweighted_value.abs() > 1.0e-12 {
        weighted_value / unweighted_value
    } else if enabled {
        1.0
    } else {
        0.0
    };
    ObjectiveFamilyBudgetEntry::new(
        family,
        unweighted_value,
        effective_weight,
        weighted_value,
        enabled,
    )
}

#[cfg(test)]
mod objective_budget_tests {
    use super::*;

    #[test]
    fn objective_budget_family_registry_is_stable_and_unique() {
        let names = objective_budget_family_names().collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "task",
                "rollout",
                "pocket_interaction",
                "chemistry",
                "redundancy",
                "probe",
                "leakage",
                "gate",
                "slot",
            ]
        );
        let unique = names
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), names.len());
    }
}
