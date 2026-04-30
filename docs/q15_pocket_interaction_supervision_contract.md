# Q15 Pocket Interaction Supervision Contract

Date: 2026-04-29

Purpose: make pocket conditioning richer than binary contact labels while preserving the separate topology, geometry, and pocket/context encoder boundaries.

## Optimizer-Visible Geometry Terms

`src/losses/consistency.rs::PocketGeometryAuxLoss` emits five decomposed objectives, aggregated by `src/losses/auxiliary.rs` and staged by `src/training/scheduler.rs`.

| Objective family | Target/evidence | Config weight | No-op behavior |
| --- | --- | --- | --- |
| `pocket_contact` | nearest ligand atom to pocket distance above `contact_distance` | `training.loss_weights.rho_pocket_contact` | zero when ligand or pocket geometry is empty |
| `pocket_pair_distance` | atom-pocket pair distance-bin centers from reference ligand coordinates | `training.loss_weights.lambda_pocket_pair_distance` | zero when target coordinates, ligand mask, or pocket rows are unavailable |
| `pocket_clash` | steric margin below `clash_distance` for every valid ligand-pocket pair | `training.loss_weights.sigma_pocket_clash` | zero when ligand or pocket geometry is empty |
| `pocket_shape_complementarity` | nearest-pocket shell deviation around `(contact_distance + clash_distance) / 2` | `training.loss_weights.omega_pocket_shape_complementarity` | zero when ligand or pocket geometry is empty |
| `pocket_envelope` | ligand atoms outside the coarse pocket envelope | `training.loss_weights.tau_pocket_envelope` | zero when ligand or pocket geometry is empty |

`pocket_pair_distance` constructs pair targets from reference ligand coordinates and pocket coordinates with explicit masks:

- ligand mask: `forward.generation.state.partial_ligand.atom_mask` aligned to decoded atom rows.
- target coordinate mask: derived through `align_rows(..., LossTargetAlignmentPolicy::PadWithMask, ...)`.
- pocket mask: currently all available pocket coordinate rows, with the same pair-mask path ready for future residue-level masks.
- distance bins: default edges `[2.0, 4.0, 6.0]`, producing four bin-center regression targets.

The pair-distance and shape-complementarity terms run on the teacher-forced decoded state and, when `training.rollout_training.enabled` records bounded generated states, average in those rollout coordinates as optimizer-visible generated-state supervision.

## Pharmacophore Role Contract

Role supervision uses `ChemistryRoleFeatureMatrix` in `src/data/features/types.rs`.

Each ligand atom or pocket atom/residue row has `CHEMISTRY_ROLE_FEATURE_DIM = 9` channels:

| Channel | Meaning |
| --- | --- |
| `donor` | hydrogen-bond donor tendency |
| `acceptor` | hydrogen-bond acceptor tendency |
| `hydrophobic` | hydrophobic contact tendency |
| `aromatic` | aromatic contact tendency |
| `positive` | positive charge tendency |
| `negative` | negative charge tendency |
| `metal_binding` | metal-binding tendency |
| `unknown` | explicit unknown marker |
| `available` | row contains usable role evidence |

The tensor contract stores role values in `role_vectors: [rows, 9]`, row availability in `availability: [rows]`, and provenance in `ChemistryRoleFeatureProvenance::{Heuristic, BackendSupported, Unavailable}`. The in-repo fallback is a deterministic atom-type heuristic in `src/data/features/builders.rs`; unavailable rows set `unknown=1` and `available=0` and must not be treated as negative role labels.

Role losses are optional and config-gated:

- `training.pharmacophore_probes.enable_ligand_role_probe`
- `training.pharmacophore_probes.enable_pocket_role_probe`
- `training.pharmacophore_probes.enable_topology_to_pocket_role_leakage`
- `training.pharmacophore_probes.enable_geometry_to_pocket_role_leakage`
- `training.pharmacophore_probes.enable_pocket_to_ligand_role_leakage`

Same-modality role probes use masked BCE and return zero when no rows are available. Leakage probes are separate auxiliary families and use the configured leakage margin, so role specialization and off-modality leakage control can be staged and ablated independently.

## Layered Evaluation Surface

`CandidateLayerMetrics` now carries pocket-interaction provenance for homogeneous candidate layers:

- `layer_name`
- `pocket_interaction_provenance`
- `pocket_distance_bin_accuracy`
- `pocket_contact_precision_proxy`
- `pocket_contact_recall_proxy`
- `pocket_role_compatibility_proxy`

`src/experiments/unseen_pocket/evaluation/generation_layers.rs::summarize_candidate_layer` computes these per layer and keeps raw model-native, repaired, reranked, and external backend layers separate. The provenance label includes the generation layer and path class, with postprocessor chains called out for repaired or processed states. These fields are proxy metrics for claim gating and reporting; they do not promote repaired/backend-scored candidates into raw-native model quality.

## Verification

Relevant tests:

- `losses::consistency::tests::atom_pocket_distance_bins_respect_atom_and_pocket_masks`
- `losses::consistency::tests::pocket_geometry_pair_distance_and_shape_losses_are_finite`
- `losses::consistency::tests::pocket_geometry_separates_clash_shape_and_missing_pocket_noop`
- `losses::consistency::tests::pocket_geometry_pair_distance_runs_on_rollout_training_states`
- `losses::probe::tests::enabled_pharmacophore_probe_noops_when_role_rows_are_unavailable`
- `losses::probe::tests::supervised_role_example_rewards_compatible_logits`
- `experiments::unseen_pocket::tests::candidate_layer_pocket_interaction_metrics_keep_raw_and_repaired_separate`
