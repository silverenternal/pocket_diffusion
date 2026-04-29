# Q14 Full-Flow Branch Schedule

Date: 2026-04-29

This note defines the recommended warm-start semantics for full molecular flow.
It applies to `generation_method.flow_matching.multi_modal.branch_schedule`.

`enabled_branches` means the branch is present in the molecular-flow contract
and may appear in rollout/artifact diagnostics. `branch_loss_weights` are static
optimizer weights. `branch_schedule` is a staged multiplier on those weights.
Keeping a present branch at zero final optimizer weight is therefore an explicit
negative-control ablation and requires
`allow_zero_weight_branch_ablation=true`.

## Branch Scale Report

Every optimizer-facing flow branch must report:

- `branch_name`
- `unweighted_value`
- `effective_weight`
- `weighted_value`
- `schedule_multiplier`
- `optimizer_facing`
- target matching provenance when the branch consumes matched atom rows

`unweighted_value` is the branch loss before branch scheduling.
`effective_weight` is the static branch loss weight multiplied by the active
schedule multiplier. `weighted_value` is the branch contribution after branch
scheduling and before the staged primary-objective weight.

## Recommended Warm Start

The preset in `configs/q14_full_flow_staged_schedule.json` activates branches in
this order:

| Branch | Start Step | Warmup Steps | Rationale |
| --- | ---: | ---: | --- |
| geometry | 0 | 0 | Establish coordinate transport and endpoint scale first. |
| atom_type | 50 | 50 | Add categorical atom supervision after geometry has a stable signal. |
| bond | 100 | 75 | Add edge supervision once atom tokens are no longer entirely cold. |
| topology | 150 | 75 | Synchronize graph structure after bond logits are active. |
| pocket_context | 200 | 100 | Add pocket interaction-profile pressure after ligand state branches are active. |
| synchronization | 300 | 100 | Penalize cross-branch disagreement after both bond and topology branches exist. |

The exact step values are a conservative smoke/reviewer preset, not a universal
optimization claim. Ablations may move starts or disable branches only when the
config explicitly sets `allow_zero_weight_branch_ablation=true`; artifacts must
preserve the branch scale report above.

## Claim Guardrails

Claim-bearing full-flow configs must keep all required branches implemented,
present in `enabled_branches`, assigned positive static loss weights, enabled
with positive final schedule multipliers, and must have at least one branch
active at step 0. This prevents a nominal full-flow run from reporting an
all-zero primary branch schedule during early training.

Processed graph quality, repair gains, and backend scores remain separate from
raw model-native branch losses and raw rollout metrics.
