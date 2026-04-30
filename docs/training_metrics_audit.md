# Training Metrics Audit Boundary

Date: 2026-04-30

This document records the current engineering boundary for training metrics,
objective provenance, branch schedule reporting, and objective-family budget
audits. The goal is to keep training execution, optimizer-facing objectives,
and claim-facing observability decoupled.

## Ownership

`src/training/trainer.rs` owns execution:

- stage selection
- forward construction
- primary and auxiliary tensor computation
- weighted objective assembly
- gradient step and checkpoint triggers
- final `StepMetrics` assembly

It should not own the string rules that explain objective provenance, flow
branch ownership, claim boundaries, or objective-family budget grouping. Those
rules live under `src/training/metrics/`.

## Metrics Registries

The metrics layer has three registry-style boundaries.

| Registry | Code path | Owns | Main consumers |
| --- | --- | --- | --- |
| Primary component registry | `src/training/metrics/losses.rs` | `PrimaryObjectiveComponentSpec`, component anchor, target source, differentiability, optimizer-facing status, branch owner, claim boundary | primary provenance, scale reports, artifact coverage, branch component audit |
| Flow branch schedule builder | `src/training/metrics/branch_schedule.rs` | `PrimaryBranchScheduleReport`, branch weights, matching provenance, `PrimaryBranchComponentAudit` | trainer step metrics, resume/replay metadata, reviewer artifacts |
| Objective-budget registry | `src/training/metrics/objective_budget.rs` | stable budget families and auxiliary-family grouping | `StepMetrics.losses.objective_family_budget_report`, budget warn/clamp diagnostics |

These registries are intentionally separate from the loss implementations in
`src/losses/`. Loss code emits tensor values and scalar component metrics;
metrics code explains how those values should be audited.

## Primary Component Provenance

`PrimaryObjectiveComponentMetrics` is the scalar decomposition of the active
primary objective. It exposes shared methods for:

- observed component traversal
- `add_assign` and `scale` for batch and hybrid objective aggregation
- provenance record construction
- branch component audits
- component scale diagnostics

Each observed component should resolve through
`primary_objective_component_descriptor`. The descriptor is the source of truth
for:

- `anchor`
- `target_source`
- `differentiable`
- `optimizer_facing`
- `role`
- `branch_name`
- `claim_boundary`

Detached rollout diagnostics and native-score cap audit fields are explicitly
non-optimizer-facing even when they are reported next to trainable primary
terms.

## Flow Branch Observability

`PrimaryBranchScheduleReport.entries[]` records configured branch weights and
target-matching provenance. Each entry also carries `component_audit`, which is
derived from the primary component registry rather than from trainer-local
string matching.

`PrimaryBranchComponentAudit` records:

- observed component names assigned to the branch
- observed component count
- optimizer-facing component count
- diagnostic-only component count
- audit-only value sums

The value sums are not used to recompute optimizer totals. They exist so a
reviewer can see that, for example, topology owns both trainable native-score
subterms and diagnostic cap-scale records without double-counting those
subterms as a separate branch loss.

## Objective-Family Budget Report

`objective_family_budget_report` is built in `src/training/metrics/` from:

- primary scalar value and primary weight
- rollout-training metrics
- `AuxiliaryObjectiveReport`
- `ObjectiveScaleDiagnosticsConfig`

The stable budget families are:

- `task`
- `rollout`
- `pocket_interaction`
- `chemistry`
- `redundancy`
- `probe`
- `leakage`
- `gate`
- `slot`

The `OBJECTIVE_BUDGET_FAMILY_SPECS` registry maps auxiliary families into those
budget groups. Budget warnings and clamps are applied after grouping, so
claim-facing reports can tell whether a family was active, dominant, warned, or
clamped without inspecting trainer internals.

## Artifact And Claim Boundaries

Objective coverage artifacts consume the same primary component descriptors as
trainer step metrics. They should not derive claim wording from string prefixes
such as `flow_` or `rollout_eval_`.

Claim-facing interpretation:

- Optimizer-facing means a tensor-preserving value can contribute to the current
  objective when its effective weight is active.
- Diagnostic-only means a value can gate readiness or explain behavior, but it
  is not training evidence by itself.
- Branch schedule presence means the branch contract was observed and reported;
  optimizer-facing branch status still depends on the effective branch weight.
- Processed generation layers remain separate from raw-native model layers.

## Verification

Focused tests for this boundary:

```bash
cargo test primary_component --lib
cargo test objective_budget --lib
cargo test objective_family_budget --lib
cargo test full_flow_branch_scale_report_covers_all_optimizer_branches --lib
cargo test full_flow_primary_branch_schedule_is_reported_without_auxiliary_stage_changes --lib
cargo test --test artifact_compatibility branch_weight_record_serializes_matching_provenance_fields
```

Broad checks used after the current refactor:

```bash
cargo fmt -- --check
jq empty todo.json
cargo check
cargo test --test artifact_compatibility
cargo test --lib
```

