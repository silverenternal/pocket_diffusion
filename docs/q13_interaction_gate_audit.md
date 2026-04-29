# Q13 Directed Interaction Gate Audit

This audit records the claim boundary for controlled cross-modality interaction.

## Directed Paths

The preferred stack keeps six directed paths:

- `topo_from_geo`
- `topo_from_pocket`
- `geo_from_topo`
- `geo_from_pocket`
- `pocket_from_topo`
- `pocket_from_geo`

Each path is emitted as a per-example diagnostic. Training summaries persist
path name, chemistry role, gate mean, closed/open fractions, saturation
fraction, attention entropy, effective update norm, and path scale.

## Q/K/V And Gating

The controlled path follows:

`A(m <- n) = gate(m,n) * Attention(Q_m, K_n, V_n)`

The implementation keeps receiver-side query projections and source-side
key/value projections inside the interaction module. The gate is sigmoid-bounded
in `[0, 1]` for preferred modes. Attention masks are derived from modality slot
visibility, so disabled or padded slots do not become free fusion channels.

## Temporal Policy

Static gates and flow-time policy are separate diagnostics:

- `gate_mean`, `gate_sparsity`, and saturation fields describe learned gate
  behavior.
- `path_scale`, `training_stage_index`, `rollout_step_index`, `flow_t`, and
  `flow_time_bucket` describe temporal or staged policy applied after gating.

This separation keeps flow-time chemistry policies auditable without relabeling
them as learned static gates.

## Negative Control Boundary

`direct_fusion_negative_control` is ablation-only. It forces directed paths open
and is recorded with `forced_open = true`. Claim artifacts now mark this surface
as `direct_fusion_negative_control = true` and set
`preferred_architecture_claim_allowed = false`.

No unrestricted full fusion is part of the preferred architecture surface.
