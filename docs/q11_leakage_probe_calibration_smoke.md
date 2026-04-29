# Q11 Frozen Leakage Probe Calibration Smoke Report

Date: 2026-04-29

Purpose: distinguish training-time leakage penalties from held-out predictability of off-modality targets using separately fit probes on frozen forward representations.

Command:

```bash
cargo test frozen_leakage_probe_calibration_capacity_sweep_reports_heldout_routes -- --nocapture
```

Observed smoke output on a synthetic four-example fixture with perturbed geometry targets:

```text
frozen_leakage_probe_smoke routes=2 sweep_rows=8 best_route=topology_to_geometry heldout_mse=0.309700 baseline_mse=0.309700 improvement=0.000000
```

Calibration contract:

- Source representations are frozen pooled modality embeddings from completed forwards.
- Probe fitting is separate from the main optimizer and uses a small ridge-regression probe.
- The smoke sweep covers two off-modality routes, two feature-prefix capacities, and two regularization values.
- Report fields separate `training_time_signal` from held-out `routes` and `capacity_sweep` metrics.
- A low training leakage penalty is not interpreted as proof of semantic independence unless held-out frozen probes also fail to beat trivial baselines across adequate capacity.
- Claim-facing artifacts additionally expose a role-separated leakage report with
  `optimizer_penalty`, `detached_training_diagnostic`, and `frozen_probe_audit`
  sections.
