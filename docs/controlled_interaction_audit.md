# Controlled Interaction Audit

This audit records the current cross-modality interaction contract. The model
uses six named directed gated-attention paths; it does not use unrestricted
full fusion between topology, geometry, and pocket/context branches.

## Path Contract

| Path | Role | Expected signal | Failure mode | Ablation interpretation | Heuristic bias |
| --- | --- | --- | --- | --- | --- |
| `topo_from_geo` | topology-informed bond plausibility | geometric proximity and conformer context that can disambiguate likely bond structure | geometry dominates topology and turns the topology branch into a distance proxy | disabling tests whether bond/topology predictions need geometry feedback beyond topology encoder features | none |
| `topo_from_pocket` | pocket-informed ligand chemistry preference | pocket chemistry and binding-site role evidence that can bias ligand atom/bond preferences | pocket signal leaks into topology as a shortcut for ligand labels | disabling tests whether pocket chemistry changes topology predictions rather than only downstream scoring | topology-pocket pharmacophore diagnostics only |
| `geo_from_topo` | topology-constrained conformer geometry | graph connectivity and bond-order constraints for coordinate refinement | coordinates ignore graph-implied local constraints | disabling tests whether geometry is relying on topology for conformer plausibility | none |
| `geo_from_pocket` | pocket-shaped pose refinement | ligand-pocket distances and role compatibility for pose updates | geometry collapses toward pocket heuristics or overfits local contacts | disabling tests whether pose improvement is driven by explicit pocket conditioning | ligand-pocket distance/role attention bias |
| `pocket_from_topo` | ligand-chemistry pocket compatibility | ligand topology and pharmacophore role pressure for context compatibility summaries | pocket branch becomes a ligand label copier instead of context encoder | disabling tests whether compatibility diagnostics need topology feedback | topology-pocket pharmacophore diagnostics only |
| `pocket_from_geo` | pose occupancy feedback | ligand pose occupancy and contact geometry for pocket-conditioned summaries | pocket branch overfits generated coordinates and hides geometry leakage | disabling tests whether pocket summaries need pose feedback | ligand-pocket distance/role attention bias |

## Gate Regularization

`model.interaction_tuning.gate_regularization_path_weights` optionally assigns a
non-negative multiplier to any stable path id. Empty config preserves the prior
aggregate objective:

```text
L_gate = mean(abs(gate_path)) over the six directed paths
```

With path weights, each path contributes:

```text
abs_mean(gate_path) * path_weight / 6
```

This keeps the old behavior when every path weight is omitted or equal to `1.0`,
allows a path to be excluded with weight `0.0`, and exposes the exact per-path
objective contribution in training metrics.

## Temporal Policy

Temporal interaction multipliers remain separate from gate values. A multiplier
of `0.0` disables the path update and reports zero gate/update for that step.
Positive multipliers scale the attended update while preserving the gate value
in `[0, 1]`. Rollout records report actual step-bucketed gate summaries; flow
rollouts additionally carry flow-time-conditioned buckets when the policy uses
flow-time multipliers.
