# P10 Final Validation Report

Generated on 2026-04-29.

## Compact End-To-End Smoke

Command:

```bash
cargo run --bin pocket_diffusion -- research experiment --config /tmp/patent_rw_p10_smoke_config.json
cargo run --bin pocket_diffusion -- research train --config /tmp/patent_rw_p10_train_config.json
```

Artifacts:

- `/tmp/patent_rw_p10_smoke_train/checkpoints/training_summary.json`
- `/tmp/patent_rw_p10_smoke/checkpoints/experiment_summary.json`
- `/tmp/patent_rw_p10_smoke/checkpoints/claim_summary.json`
- `/tmp/patent_rw_p10_smoke/checkpoints/dataset_validation_report.json`
- `/tmp/patent_rw_p10_smoke/checkpoints/split_report.json`
- `/tmp/patent_rw_p10_smoke/checkpoints/generation_layers_validation.json`
- `/tmp/patent_rw_p10_smoke/checkpoints/generation_layers_test.json`
- `/tmp/patent_rw_p10_smoke/checkpoints/candidate_metrics_validation.jsonl`
- `/tmp/patent_rw_p10_smoke/checkpoints/candidate_metrics_test.jsonl`

Observed checks:

- Training summary history length: 2 steps.
- Training summary final total loss: `20.312467575073242`.
- Training summary objective coverage keys: `schema_version`, `primary_objective`, `records`.
- Experiment training history length: 2 steps.
- Experiment last training stage: `Stage3`.
- Experiment total losses: `53.27585983276367`, `7.470146656036377`.
- Nonfinite gradient tensors: `0` in both compact train and experiment runs.
- Nonfinite loss terms: `0` in both compact train and experiment runs.
- Training summary validation/test `finite_forward_fraction`: `1.0` / `1.0`.
- Experiment validation/test `finite_forward_fraction`: `1.0` / `1.0`.
- Dataset validation parsed examples: `4`.
- Split leakage checks: no protein overlap and no duplicated example ids.
- `train_eval_alignment.schema_version`: `1` on validation, test, and claim summary.
- Alignment rows distinguish `L_probe` / `L_leak` optimizer-facing terms from detached `rollout_eval_*`, raw model-native, processed-layer, backend, and efficiency diagnostics.
- Best metric review marks `finite_forward_fraction` as `smoke_default`, not claim-bearing quality-aware selection.
- Generation layers preserve `coordinate_frame_contract`, raw native `raw_flow`, and non-native reranked postprocessor chain attribution.
- Candidate metric JSONL includes `raw_flow`, `constrained_flow`, `repaired`, `deterministic_proxy`, and `reranked` layers with coordinate-frame provenance.

Notes:

- This is a compact smoke surface on `examples/datasets/mini_pdbbind`, not claim-bearing evidence.
- The experiment entrypoint persists training history inside `experiment_summary.json`; the standalone train entrypoint was run separately to verify `training_summary.json`.

## Final Repository Verification

Commands:

```bash
cargo fmt --check
cargo test --no-run
cargo test --no-fail-fast
tools/local_ci.sh fast
```

Outcomes:

- `cargo fmt --check`: passed.
- `cargo test --no-run`: passed; all test targets compiled.
- `cargo test --no-fail-fast`: passed; 319 lib tests, 12 main tests, 12 artifact compatibility tests, 4 generation method platform tests, 7 MI integration tests, 15 reviewer tooling tests, 2 smoke trainer/MI tests, and doc tests completed without failures.
- `tools/local_ci.sh fast`: passed.
