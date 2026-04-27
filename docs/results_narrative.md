# Results Narrative and Threats to Validity

## Claim-to-evidence narrative

- **Unseen-pocket generalization on canonical larger-data surface is supported.**  
  `checkpoints/pdbbindpp_real_backends/claim_summary.json` and `checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json` both remain `external_benchmark_backed`, with backend/data/leakage gates passing in `docs/reviewer_refresh_report.json`.
- **Controlled comparison contract is now explicit and machine-checked.**  
  `configs/paper_claim_contract.json` and `tools/claim_regression_gate.py` enforce claim mapping, threshold policy, and required baseline matrix coverage for canonical claim-bearing review.
- **Mechanism evidence is now packaged as first-class artifacts.**  
  `checkpoints/pdbbindpp_real_backends/ablation_matrix_summary.json` + `ablation_delta_table.json` provide direct deltas versus the canonical base; `flow_rollout_diagnostics.json` summarizes flow rollout behavior (displacement, endpoint consistency, gate usage).
- **Multi-seed stability is explicitly aggregated.**  
  `configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json` records per-seed values and aggregate mean/std/CI for claim-facing metrics.

## Failures, caveats, and non-claims

- **No human/experimental preference claim.** Preference evidence remains unavailable for claim wording unless source coverage and schema constraints are explicitly met.
- **Held-out reviewer package is not broad publication benchmark coverage.** Current evidence is stronger than smoke/proxy surfaces but still centered on a local reviewer bundle and selected benchmark paths.
- **Variance remains non-trivial on strict pocket fit across seeds.** Multi-seed summary reports meaningful spread; conclusions should stay phrased as bounded reviewer evidence rather than universal stability.
- **Backend dependence remains operational risk.** Claim-bearing wording requires backend commands and compatible environment availability.

## External-validity constraints

- Data scope and benchmark scope are limited to current curated surfaces and contracts.
- Reported improvements should be interpreted under current staged objectives, configured budgets, and backend adapters.
- Cross-family generalization beyond current held-out setup requires additional benchmark breadth before stronger wording.
