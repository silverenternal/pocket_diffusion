# Q11 Staged Auxiliary Execution Smoke Report

Date: 2026-04-29

Purpose: verify that staged auxiliary execution skips zero-weight trainable graphs while retaining explicit report provenance for trainable, skipped, and detached-diagnostic paths.

Command:

```bash
cargo test staged_auxiliary_execution_plan_reports_stage_dependent_proxy_work -- --nocapture
```

Observed smoke output on the synthetic two-example fixture:

```text
aux_execution_smoke stage1_trainable=1 stage1_skipped=12 stage1_us=16473 stage4_trainable=6 stage4_skipped=7 stage4_us=19327
```

Config summary:

- Stage 1 proxy: primary objective plus `consistency`; all other auxiliary families use `skipped_zero_weight`.
- Stage 4 proxy: `intra_red`, `probe`, `leak`, `gate`, `slot`, and `consistency` are trainable; pharmacophore and chemistry warmup families remain skipped in this smoke fixture.
- Disabled diagnostics can be requested through `AuxiliaryObjectiveExecutionPlan::from_effective_weights_with_detached_diagnostics`, which computes disabled families under `tch::no_grad` and marks report entries as `detached_diagnostic`.

