//! Multi-modal flow contract records.
//!
//! Geometry, atom-type, bond, topology, and pocket/context flow branches are
//! optimizer-facing when enabled by config. The contract records keep artifact
//! claims tied to the active branch set.

use serde::{Deserialize, Serialize};

use crate::config::{FlowBranchKind, FlowMatchingConfig};

/// Stable schema version for optimizer-facing molecular flow state records.
pub const MOLECULAR_FLOW_CONTRACT_VERSION: &str = "molecular_flow_contract_v1";

/// Branch set required before artifacts may claim full molecular flow.
pub const REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES: [FlowBranchKind; 5] = [
    FlowBranchKind::Geometry,
    FlowBranchKind::AtomType,
    FlowBranchKind::Bond,
    FlowBranchKind::Topology,
    FlowBranchKind::PocketContext,
];

/// Runtime support state for a modality flow branch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlowBranchSupportStatus {
    /// Branch has an optimizer-facing implementation.
    Implemented,
    /// Branch is an explicit roadmap surface but is not runnable yet.
    PlannedUnsupported,
}

/// Artifact-facing contract for one modality flow branch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultiModalFlowBranchRecord {
    /// Stable branch id.
    pub branch: FlowBranchKind,
    /// Whether the branch is enabled in config.
    pub enabled: bool,
    /// State space transported by this branch.
    pub state_space: String,
    /// Supervision target or self-consistency target.
    pub target: String,
    /// Loss family expected for the branch.
    pub loss_family: String,
    /// Metrics that should be emitted before the branch becomes claim-facing.
    pub metrics: Vec<String>,
    /// Implementation support status.
    pub support_status: FlowBranchSupportStatus,
    /// Reason emitted when a branch is only planned.
    pub unsupported_reason: Option<String>,
}

/// Artifact-facing summary for the current flow implementation boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultiModalFlowContract {
    /// Stable contract version attached to configs and artifacts.
    pub flow_contract_version: String,
    /// Enabled branch labels from config.
    pub enabled_branches: Vec<String>,
    /// Required branch labels not enabled in config.
    pub disabled_branches: Vec<String>,
    /// Per-branch state-space and support metadata.
    pub branches: Vec<MultiModalFlowBranchRecord>,
    /// Whether config requested a full molecular flow claim.
    pub full_molecular_flow_claim_requested: bool,
    /// Whether the implementation can support the requested claim.
    pub full_molecular_flow_claim_allowed: bool,
    /// Stable reason for the claim gate decision.
    pub claim_gate_reason: String,
    /// Explicit target-alignment policy for generated/target atom-count mismatch.
    pub target_alignment_policy: String,
    /// Whether target matching is non-index and claim-safe for de novo full-flow reports.
    #[serde(default)]
    pub target_matching_claim_safe: bool,
}

/// Contract row for one molecular flow state variable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MolecularFlowStateVariableContract {
    /// Stable state variable name.
    pub name: String,
    /// Tensor shape contract using symbolic dimensions.
    pub shape: String,
    /// Whether the variable is available at inference time.
    pub inference_available: bool,
    /// Whether the variable is only a training supervision target.
    pub target_supervised_only: bool,
    /// Rollout update rule or training reduction rule.
    pub update_rule: String,
    /// Artifact fields that expose this state or its diagnostics.
    pub artifact_fields: Vec<String>,
}

/// Full molecular flow state contract for active config metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MolecularFlowStateContract {
    /// Stable contract version.
    pub flow_contract_version: String,
    /// Explicit target alignment policy label.
    pub target_alignment_policy: String,
    /// Per-variable state contracts.
    pub variables: Vec<MolecularFlowStateVariableContract>,
}

/// Build the state-variable contract emitted by full molecular flow artifacts.
pub fn current_molecular_flow_state_contract(
    config: &FlowMatchingConfig,
) -> MolecularFlowStateContract {
    MolecularFlowStateContract {
        flow_contract_version: MOLECULAR_FLOW_CONTRACT_VERSION.to_string(),
        target_alignment_policy: config
            .multi_modal
            .target_alignment_policy
            .as_str()
            .to_string(),
        variables: vec![
            state_variable(
                "geometry",
                "[num_atoms, 3]",
                true,
                false,
                "continuous velocity integration over x_t with endpoint diagnostics",
                &[
                    "flow_matching.sampled_coords",
                    "rollout.steps[].coords",
                    "flow_velocity",
                    "flow_endpoint",
                ],
            ),
            state_variable(
                "atom_type",
                "[num_atoms]",
                true,
                true,
                "categorical logits update draft atom tokens during molecular-flow rollout",
                &["molecular.atom_type_logits", "rollout.steps[].atom_types"],
            ),
            state_variable(
                "bond",
                "[num_atoms, num_atoms]",
                true,
                true,
                "bond existence and type logits feed native graph extraction",
                &[
                    "molecular.bond_exists_logits",
                    "molecular.bond_type_logits",
                    "rollout.steps[].native_graph_provenance.raw_logits_layer",
                    "rollout.steps[].native_bonds",
                    "rollout.steps[].constrained_native_bonds",
                ],
            ),
            state_variable(
                "topology",
                "[num_atoms, num_atoms]",
                true,
                true,
                "topology logits synchronize graph connectivity with bond predictions",
                &["molecular.topology_logits", "flow_topology"],
            ),
            state_variable(
                "pocket_context",
                "[num_atoms]",
                true,
                true,
                "pocket interaction-profile branch predicts ligand-pocket contact compatibility; pocket_context reconstruction is diagnostic only",
                &[
                    "molecular.pocket_contact_logits",
                    "molecular.target_pocket_contacts",
                    "molecular.pocket_branch_target_family",
                    "flow_pocket_context",
                ],
            ),
            state_variable(
                "synchronization",
                "[num_atoms, num_atoms]",
                false,
                false,
                "bond/topology probability agreement reduced under pair mask",
                &["flow_synchronization"],
            ),
        ],
    }
}

/// Build a serializable contract for enabled molecular flow branches.
pub fn current_multimodal_flow_contract(config: &FlowMatchingConfig) -> MultiModalFlowContract {
    let enabled_branches = config
        .multi_modal
        .enabled_branches
        .iter()
        .map(|branch| branch.as_str().to_string())
        .collect::<Vec<_>>();
    let disabled_branches = REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES
        .iter()
        .copied()
        .filter(|branch| !config.multi_modal.enabled_branches.contains(branch))
        .map(|branch| branch.as_str().to_string())
        .collect::<Vec<_>>();
    let branches = REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES
        .iter()
        .copied()
        .map(|branch| {
            branch_record(
                branch,
                config.multi_modal.enabled_branches.contains(&branch),
            )
        })
        .collect::<Vec<_>>();

    let full_molecular_flow_claim_requested = config.multi_modal.claim_full_molecular_flow;
    let all_required_enabled = disabled_branches.is_empty();
    let all_required_implemented = REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES
        .iter()
        .all(|branch| branch.implemented());
    let target_matching_claim_safe = config
        .multi_modal
        .target_alignment_policy
        .claim_safe_for_de_novo();
    let full_molecular_flow_claim_allowed = full_molecular_flow_claim_requested
        && all_required_enabled
        && all_required_implemented
        && target_matching_claim_safe;
    let claim_gate_reason = if !full_molecular_flow_claim_requested {
        "claim_not_requested"
    } else if !all_required_enabled {
        "missing_required_flow_branches"
    } else if !all_required_implemented {
        "planned_flow_branches_unsupported"
    } else if !target_matching_claim_safe {
        "target_matching_policy_not_claim_safe_for_de_novo"
    } else {
        "full_molecular_flow_contract_satisfied"
    };

    MultiModalFlowContract {
        flow_contract_version: MOLECULAR_FLOW_CONTRACT_VERSION.to_string(),
        enabled_branches,
        disabled_branches,
        branches,
        full_molecular_flow_claim_requested,
        full_molecular_flow_claim_allowed,
        claim_gate_reason: claim_gate_reason.to_string(),
        target_alignment_policy: config
            .multi_modal
            .target_alignment_policy
            .as_str()
            .to_string(),
        target_matching_claim_safe,
    }
}

fn state_variable(
    name: &str,
    shape: &str,
    inference_available: bool,
    target_supervised_only: bool,
    update_rule: &str,
    artifact_fields: &[&str],
) -> MolecularFlowStateVariableContract {
    MolecularFlowStateVariableContract {
        name: name.to_string(),
        shape: shape.to_string(),
        inference_available,
        target_supervised_only,
        update_rule: update_rule.to_string(),
        artifact_fields: artifact_fields
            .iter()
            .map(|field| field.to_string())
            .collect(),
    }
}

fn branch_record(branch: FlowBranchKind, enabled: bool) -> MultiModalFlowBranchRecord {
    let (state_space, target, loss_family, metrics) = match branch {
        FlowBranchKind::Geometry => (
            "continuous_3d_coordinates",
            "coordinate_velocity_x1_minus_x0",
            "velocity_mse_plus_endpoint_consistency",
            vec!["flow_velocity_mse", "flow_endpoint_mse", "coordinate_rmsd"],
        ),
        FlowBranchKind::AtomType => (
            "categorical_atom_type_labels",
            "atom_type_categorical_path_or_logits",
            "categorical_flow_cross_entropy_or_ctmc_surrogate",
            vec!["atom_type_accuracy", "atom_type_nll", "valid_atom_fraction"],
        ),
        FlowBranchKind::Bond => (
            "edge_existence_and_bond_order",
            "bond_existence_and_bond_type_path",
            "edge_binary_loss_plus_bond_type_categorical_loss",
            vec!["bond_f1", "bond_type_accuracy", "valence_violation_rate"],
        ),
        FlowBranchKind::Topology => (
            "graph_connectivity_consistency",
            "atom_bond_graph_synchronization",
            "topology_consistency_and_connectivity_loss",
            vec!["graph_validity", "connectivity_rate", "topology_sync_error"],
        ),
        FlowBranchKind::PocketContext => (
            "ligand_pocket_interaction_profile",
            "matched_ligand_pocket_contact_labels",
            "masked_contact_bce",
            vec![
                "pocket_contact_bce",
                "pocket_contact_target_fraction",
                "pocket_interaction_mask_coverage",
            ],
        ),
    };

    let support_status = if branch.implemented() {
        FlowBranchSupportStatus::Implemented
    } else {
        FlowBranchSupportStatus::PlannedUnsupported
    };
    let unsupported_reason = (!branch.implemented()).then(|| {
        "planned branch only; not optimizer-facing and not available for generation claims"
            .to_string()
    });

    MultiModalFlowBranchRecord {
        branch,
        enabled,
        state_space: state_space.to_string(),
        target: target.to_string(),
        loss_family: loss_family.to_string(),
        metrics: metrics.into_iter().map(str::to_string).collect(),
        support_status,
        unsupported_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_multimodal_flow_contract_is_geometry_only() {
        let config = FlowMatchingConfig::default();
        let contract = current_multimodal_flow_contract(&config);

        assert_eq!(
            contract.flow_contract_version,
            MOLECULAR_FLOW_CONTRACT_VERSION
        );
        assert_eq!(contract.enabled_branches, vec!["geometry"]);
        assert_eq!(contract.target_alignment_policy, "pad_with_mask");
        assert!(contract
            .disabled_branches
            .contains(&"atom_type".to_string()));
        assert!(!contract.full_molecular_flow_claim_allowed);
        assert_eq!(contract.claim_gate_reason, "claim_not_requested");
        let geometry = contract
            .branches
            .iter()
            .find(|record| record.branch == FlowBranchKind::Geometry)
            .expect("geometry branch contract");
        assert!(geometry.enabled);
        assert_eq!(
            geometry.support_status,
            FlowBranchSupportStatus::Implemented
        );
    }

    #[test]
    fn full_molecular_flow_claim_allowed_when_all_branches_exist() {
        let mut config = FlowMatchingConfig::default();
        config.multi_modal.enabled_branches = REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES.to_vec();
        config.multi_modal.claim_full_molecular_flow = true;
        config.multi_modal.target_alignment_policy =
            crate::config::FlowTargetAlignmentPolicy::HungarianDistance;
        let contract = current_multimodal_flow_contract(&config);

        assert!(contract.full_molecular_flow_claim_requested);
        assert!(contract.full_molecular_flow_claim_allowed);
        assert!(contract.target_matching_claim_safe);
        assert_eq!(
            contract.claim_gate_reason,
            "full_molecular_flow_contract_satisfied"
        );
        assert!(contract.branches.iter().any(|record| {
            record.branch == FlowBranchKind::Bond
                && record.support_status == FlowBranchSupportStatus::Implemented
        }));
    }

    #[test]
    fn full_molecular_flow_claim_rejects_index_only_target_matching() {
        let mut config = FlowMatchingConfig::default();
        config.multi_modal.enabled_branches = REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES.to_vec();
        config.multi_modal.claim_full_molecular_flow = true;
        config.multi_modal.target_alignment_policy =
            crate::config::FlowTargetAlignmentPolicy::IndexExact;
        let contract = current_multimodal_flow_contract(&config);

        assert!(contract.full_molecular_flow_claim_requested);
        assert!(!contract.full_molecular_flow_claim_allowed);
        assert!(!contract.target_matching_claim_safe);
        assert_eq!(
            contract.claim_gate_reason,
            "target_matching_policy_not_claim_safe_for_de_novo"
        );
    }

    #[test]
    fn molecular_flow_state_contract_covers_all_state_variables() {
        let config = FlowMatchingConfig::default();
        let contract = current_molecular_flow_state_contract(&config);

        assert_eq!(
            contract.flow_contract_version,
            MOLECULAR_FLOW_CONTRACT_VERSION
        );
        for name in [
            "geometry",
            "atom_type",
            "bond",
            "topology",
            "pocket_context",
            "synchronization",
        ] {
            assert!(contract
                .variables
                .iter()
                .any(|variable| variable.name == name));
        }
        assert!(
            contract
                .variables
                .iter()
                .find(|variable| variable.name == "atom_type")
                .unwrap()
                .target_supervised_only
        );
    }

    #[test]
    fn pocket_flow_branch_contract_doc_defines_interaction_profile_targets() {
        let doc = std::fs::read_to_string("docs/q14_pocket_flow_branch_contract.md")
            .expect("pocket flow branch contract doc should exist");

        for required in [
            "pocket_interaction_profile",
            "nearest-pocket distance bins",
            "contact likelihood",
            "ligand-pocket role interaction profile",
            "Inference-time inputs",
            "Training labels must not be fed back into the conditioning state",
            "pocket_context_reconstruction",
            "context_drift_diagnostic",
        ] {
            assert!(
                doc.contains(required),
                "missing required pocket branch contract text: {required}"
            );
        }
    }
}
