# Adaptive Staging Design

The fixed staged schedule remains the default training behavior. Adaptive
staging is an optional guard around that fixed schedule, not a replacement for
explicit objective weights.

## Default Contract

- `training.adaptive_stage_guard.enabled = false` preserves fixed step-based
  stage transitions exactly.
- Step metrics still report the fixed stage and the effective stage.
- Objective weights are never hidden: `stage_progress.active_objective_families`
  and `losses.auxiliaries.auxiliary_objective_report` remain the source of
  optimizer-facing weights and execution modes.

## Readiness Signals

When enabled, readiness is evaluated deterministically from recent persisted
step metrics:

- finite primary and total loss fraction over `readiness_window`
- latest primary-loss finiteness
- latest non-finite gradient tensor count
- latest optimizer-step skipped flag
- latest slot collapse warning count
- latest mean interaction gate saturation fraction
- optional primary-loss improvement across the readiness window

Probe-baseline and held-out leakage readiness are not inferred inside the
optimizer loop. They remain evaluation or frozen-probe audit signals and should
gate claim promotion rather than silently alter optimizer weights.

## Hold Versus Warning

`training.adaptive_stage_guard.hold_stages = false` is warning-only mode: fixed
schedule stages and weights remain active, while `stage_progress` records
`readiness_status = "warning"` and deterministic reasons.

`hold_stages = true` uses the same readiness check to hold the effective stage at
the previous stage when the fixed schedule would advance without sufficient
evidence. Metrics record:

- `fixed_stage_index`: fixed-schedule stage
- `stage_index`: effective stage used for weights and interaction context
- `adaptive_stage_hold`
- `readiness_status`
- `readiness_reasons`

This makes adaptive behavior inspectable and reproducible while keeping the
fixed schedule available as the fallback.

## Runtime Profile

Each training step now records `runtime_profile` with wall-clock step time,
examples per second, batch size, used-memory before/after, signed memory delta,
and objective execution counts. The same counts are duplicated in
`stage_progress.objective_execution_counts` for per-stage aggregation.

Resume remains weights-only unless checkpoint metadata explicitly advertises
`optimizer_exact_resume` with persisted optimizer internals. Current Adam moment
buffers are not persisted, so strict optimizer-state-identical replay remains
unsupported by default.
