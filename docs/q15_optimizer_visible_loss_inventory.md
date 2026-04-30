# Q15 Optimizer-Visible Loss Inventory

Date: 2026-04-29

Purpose: separate optimizer-visible training signal from rollout, repair, and backend diagnostics before adding generation-alignment objectives.

## Primary Objectives

| Surface | Objective family | Code path | Inputs | Target source | Gradient visibility | Config gate | Claim category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `conditioned_denoising` | atom-type recovery | `src/losses/task.rs::ConditionedDenoisingObjective` | decoder atom logits, corruption mask | `reference_ligand` | optimizer-visible | `training.primary_objective=conditioned_denoising` | target-ligand refinement only |
| `conditioned_denoising` | coordinate and pairwise recovery | `src/losses/task.rs::ConditionedDenoisingObjective` | noisy coords, coordinate deltas, pair distances | `reference_ligand` | optimizer-visible | `training.primary_objective=conditioned_denoising` | geometry refinement proxy |
| `conditioned_denoising` | pocket anchor | `src/losses/task.rs::ConditionedDenoisingObjective` | predicted coords, pocket coords | `pocket_geometry` | optimizer-visible | `training.primary_objective=conditioned_denoising` | weak pocket-fit proxy |
| `conditioned_denoising` | rollout recovery metrics | `src/losses/task.rs::rollout_eval_*` | sampled rollout records serialized through atom ids, coords, scalar stop values | `generated_rollout_state` | no-grad diagnostic; not optimizer-facing | `training.build_rollout_diagnostics` | rollout diagnostic only |
| `geometry_flow` | velocity matching | `src/losses/task.rs::FlowMatchingObjective` | sampled flow coords, predicted velocity, target velocity, atom mask | `reference_ligand` | optimizer-visible | `training.primary_objective=flow_matching`, flow backend | flow geometry |
| `geometry_flow` | endpoint consistency | `src/losses/task.rs::flow_matching_velocity_and_endpoint_loss` | sampled flow coords, predicted velocity, target velocity | `reference_ligand` | optimizer-visible, derived from velocity head | `training.flow_matching_loss_weight` | flow geometry |
| `de_novo_full_flow` | atom-type branch | `src/losses/task.rs::molecular_flow_losses` | molecular atom logits, matched target atom types, target atom mask | `reference_ligand` | optimizer-visible when branch weight is active | `generation_method.flow_matching.multi_modal.branch_loss_weights.atom_type` | de novo proxy, reference supervised |
| `de_novo_full_flow` | bond and topology branches | `src/losses/task.rs::molecular_flow_losses` | bond logits, topology logits, target adjacency, target bond types, pair mask | `reference_ligand` | optimizer-visible when branch weights are active | `branch_loss_weights.bond`, `branch_loss_weights.topology` | chemical graph proxy |
| `de_novo_full_flow` | pocket context branch | `src/losses/task.rs::molecular_flow_losses` | pocket-contact logits, target pocket contacts, interaction mask | `pocket_ligand_pair` | optimizer-visible when branch weight is active | `branch_loss_weights.pocket_context` | pocket interaction proxy |
| `de_novo_full_flow` | synchronization branch | `src/losses/task.rs::molecular_flow_losses` | bond and topology probabilities | `generated_rollout_state` | optimizer-visible internal consistency | `branch_loss_weights.synchronization` | branch consistency |

## Auxiliary Objectives

| Family | Code path | Target source | Gradient visibility | Config gate | Claim category |
| --- | --- | --- | --- | --- | --- |
| consistency | `src/losses/consistency.rs`, aggregated by `src/losses/auxiliary.rs` | within-model topology/geometry agreement | optimizer-visible after staged weight | `loss_weights.zeta_consistency` | representation health |
| intra redundancy | `src/losses/redundancy.rs` | modality representations | optimizer-visible after staged weight | `loss_weights.alpha_intra_red` | redundancy reduction |
| semantic probes | `src/losses/probe.rs` | topology, geometry, context, affinity, pharmacophore probe targets | optimizer-visible after staged weight | `loss_weights.gamma_probe`, pharmacophore probe config | specialization proxy |
| leakage | `src/losses/leakage.rs` | off-modality targets and frozen/probe diagnostics | optimizer-visible encoder penalty plus detached diagnostics | `loss_weights.delta_leak`, explicit leakage probe config | leakage control |
| gated interaction | `src/losses/gate.rs` | cross-modal gate activations | optimizer-visible after staged weight | `loss_weights.epsilon_gate` | controlled interaction |
| slot control | `src/losses/auxiliary.rs::SlotControlLoss` | slot activations and assignment balance | optimizer-visible after staged weight | `loss_weights.eta_slot` | slot utilization |
| pocket geometry | `src/losses/consistency.rs::PocketGeometryAuxLoss`, aggregated by `src/losses/auxiliary.rs` | atom-pocket contact, pair distance bins, clash, shape shell, envelope proxies | optimizer-visible after staged weight | `loss_weights.rho_pocket_contact`, `lambda_pocket_pair_distance`, `sigma_pocket_clash`, `omega_pocket_shape_complementarity`, `tau_pocket_envelope` | pocket compatibility proxy |
| chemistry guardrails | `src/losses/consistency.rs::ChemistryGuardrailAuxLoss`, aggregated by `src/losses/auxiliary.rs` | generated atom probabilities, bond-existence probabilities, bond-order probabilities, and coordinate geometry proxies | optimizer-visible after staged weight | `upsilon_valence_guardrail`, `phi_bond_length_guardrail`, `chi_nonbonded_distance_guardrail`, `psi_angle_guardrail`, `chemistry_warmup` | chemical plausibility proxy |

## Missing Generation-Alignment Signals

1. Short rollout training is not optimizer-facing. Current `rollout_eval_*` values come from sampled records and are reported as detached diagnostics.
2. De novo full-flow supervision is still mostly single-reference imitation. Atom, bond, topology, and velocity branches use matched reference ligand targets rather than distributional pocket-conditioned alternatives.
3. Pocket compatibility is now richer but still proxy-based. The current optimizer-visible terms cover contacts, atom-pocket pair distance bins, clash, shape shell, envelope, and optional pharmacophore role probes; they are not a replacement for backend docking or wet-lab interaction evidence.
4. Raw chemical validity is now partially optimizer-visible through native valence-budget, bond-length, nonbonded-distance, and angle proxy guardrails. These remain proxy objectives; final claim-bearing validity still depends on raw-vs-repaired evaluation separation and backend evidence where available.

## Reporting Changes

Training primary component provenance now emits `target_source` alongside `anchor`, `role`, `differentiable`, and `optimizer_facing`. Evaluation train/eval alignment rows also emit `target_source`, with values including `reference_ligand`, `generated_rollout_state`, `pocket_geometry`, `pocket_ligand_pair`, `repair_layer`, and `external_backend`.

The provenance implementation is now registry-backed. The shared primary
component descriptor registry in `src/training/metrics/losses.rs` owns each
component's anchor, target source, differentiability, optimizer-facing status,
branch owner, role, and claim boundary. `src/training/metrics/branch_schedule.rs`
uses the same descriptors to populate `PrimaryBranchComponentAudit`, while
`src/training/metrics/objective_budget.rs` owns objective-family budget grouping.
The full engineering boundary is documented in
[`training_metrics_audit.md`](training_metrics_audit.md).

This means detached rollout diagnostics, native-score cap audit fields, and
optimizer-facing molecular-flow subterms are not separated by ad hoc string
prefix checks in the trainer. They are classified through shared metadata before
being written to `StepMetrics`, objective coverage artifacts, and branch schedule
reports.

Pocket interaction supervision is specified in [`q15_pocket_interaction_supervision_contract.md`](q15_pocket_interaction_supervision_contract.md). Candidate-layer reports now include `layer_name`, `pocket_interaction_provenance`, `pocket_distance_bin_accuracy`, `pocket_contact_precision_proxy`, `pocket_contact_recall_proxy`, and `pocket_role_compatibility_proxy` so raw, repaired, and backend-scored layers are not silently merged.

Chemistry-native constraints are specified in [`q15_chemistry_native_constraints.md`](q15_chemistry_native_constraints.md). Auxiliary reports expose separate `valence_guardrail`, `bond_length_guardrail`, `nonbonded_distance_guardrail`, and `angle_guardrail` families; valence overage and underage are logged as submetrics. Candidate-layer graph reports separately expose `native_valence_violation_fraction`, `native_disconnected_fragment_fraction`, `native_bond_order_conflict_fraction`, and `native_graph_repair_delta_mean`.
