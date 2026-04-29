# Q13 Molecular Flow State Contract

Date: 2026-04-29

Rust contract version: `molecular_flow_contract_v1`.

## State Variables

| Branch | Shape | Inference Available | Target-Supervised Only | Rollout / Loss Rule |
| --- | --- | --- | --- | --- |
| geometry | `[num_atoms, 3]` | yes | no | Coordinate flow integrates predicted velocity over `x_t`; velocity and endpoint losses are masked by the generated/target alignment mask. |
| atom_type | `[num_atoms]` | yes | target labels only | Atom logits update draft atom tokens during rollout; CE loss uses the explicit target-alignment mask. |
| bond | `[num_atoms, num_atoms]` | yes | target adjacency and bond labels only | Bond-existence and bond-type logits feed native graph extraction and masked BCE/CE losses. |
| topology | `[num_atoms, num_atoms]` | yes | target topology matrix only | Topology logits synchronize graph connectivity with bond predictions. |
| pocket_context | `[num_pocket_slots, hidden_dim]` | yes | no | Context branch reconstructs detached conditioning state to control drift. |
| synchronization | `[num_atoms, num_atoms]` | diagnostic/loss only | no | Bond/topology probability agreement is reduced under pair mask. |

## Alignment Policy

The default target-alignment policy is `pad_with_mask`.

- `pad_with_mask`: pads missing target rows with zeros and masks them out of
  velocity, atom-type, bond, and topology reductions.
- `truncate`: uses the leading target rows up to generated atom count.
- `sampled_subgraph`: currently deterministic contiguous smoke subgraph
  selection with the same masking contract.
- `reject_mismatch`: omits molecular-flow supervision when target/generated
  atom counts disagree.
- `smoke_only_modulo_repeat`: legacy repeat semantics retained only for
  explicit smoke/debug configs and rejected for full-flow claim configs.

## Branch Schedule

`generation_method.flow_matching.multi_modal.branch_schedule` supplies
independent `enabled`, `start_step`, `warmup_steps`, and
`final_weight_multiplier` controls for:

- geometry
- atom_type
- bond
- topology
- pocket_context
- synchronization

Effective weights are static `branch_loss_weights` multiplied by the branch
schedule at the trainer global step. This primary branch schedule composes with
the existing auxiliary `StageScheduler`; it does not alter auxiliary stage
semantics.

`docs/q14_full_flow_schedule.md` defines the recommended staged warm-start
preset. Trainer artifacts must report `unweighted_value`, `effective_weight`,
`weighted_value`, and `schedule_multiplier` for every primary flow branch,
including synchronization.

## Artifact Fields

- `FlowMatchingTrainingRecord.flow_contract_version`
- `FlowMatchingTrainingRecord.branch_weights`
- `training_history[].losses.primary.branch_schedule.entries[].unweighted_value`
- `training_history[].losses.primary.branch_schedule.entries[].weighted_value`
- `training_history[].losses.primary.branch_schedule.entries[].schedule_multiplier`
- `MolecularFlowTrainingRecord.target_alignment_policy`
- `MolecularFlowTrainingRecord.target_atom_mask`
- `GenerationRolloutRecord.atom_count_prior_provenance`
- rollout `flow_diagnostics` keys prefixed with `molecular_flow_native_*`

## Claim Boundary

`claim_full_molecular_flow=true` requires geometry, atom type, bond, topology,
and pocket-context branches to be enabled and rejects
`smoke_only_modulo_repeat` alignment. Raw rollout/native graph metrics remain
model-native; repair, pruning, reranking, and backend scoring must stay in
separate processed evidence layers.
