# Train-Eval Alignment Report

`TrainEvalAlignmentReport` is emitted on each unseen-pocket evaluation split and
copied into `claim_summary.json` for the test split. It exists to keep metric
evidence tied to the correct training term, candidate layer, method family, and
backend boundary.

## Metric Rows

Each `metric_rows` entry records:

- `metric_name` and `metric_family`
- `evidence_role`
- `optimizer_facing_terms`
- `optimizer_facing`
- `detached_diagnostic`
- candidate-layer attribution when applicable
- backend slot, backend name, availability, and coverage when applicable
- a claim-boundary note

Raw model-native evidence is kept on `raw_rollout` or `raw_flow`. Processed
candidate metrics such as `constrained_flow`, `repaired`,
`repaired_candidates`, `inferred_bond_candidates`, `deterministic_proxy`, and
`reranked_candidates` are explicitly marked as postprocessed or selected
evidence and must not overwrite raw model fields.

`rollout_eval_*` style rows are detached diagnostics. They are not listed as
optimizer-facing unless a future tensor-preserving rollout objective is
implemented and configured.

## Backend Coverage

`backend_coverage` contains one row per backend slot:

- `chemistry_validity`
- `docking_affinity`
- `pocket_compatibility`

Rows preserve backend availability, backend identity, `examples_scored`,
`candidates_scored`, `missing_structure_fraction`, generic coverage, and
fallback status. Heuristic fallback remains claim-visible through
`heuristic_fallback_labeled` and `fallback_status`.

Claim gates should use these rows instead of inferring coverage from quality
scalars alone.

## Best Metric Review

`best_metric_review` records whether `training.best_metric` is available and
whether it is appropriate for claim-bearing checkpoint selection. The default
`training.best_metric = "auto"` resolves from the full research profile before
selection; de novo full-flow claim configs resolve to `strict_pocket_fit_score`
rather than the smoke health metric.

`finite_forward_fraction` remains valid for smoke runs because it catches broken
forward passes without requiring chemistry or pocket backends. Claim-bearing
configs should prefer quality-aware metrics such as `strict_pocket_fit_score`,
`candidate_valid_fraction`, `distance_probe_rmse`, `affinity_probe_mae`, or
`leakage_proxy_mean`, with the corresponding backend or supervision coverage
available.
