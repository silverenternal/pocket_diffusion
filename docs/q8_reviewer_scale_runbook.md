# Q8 Reviewer-Scale Runbook

This runbook is the repeatable path from config validation to claim-boundary
checks for the modular pocket-conditioned generation stack. It separates smoke,
real-data debug, and reviewer-scale evidence. Do not use smoke or capped debug
artifacts for reviewer-scale, backend-supported, docking-supported, or
experimental claims.

## 1. Preflight

Run the fast local gate before refreshing evidence:

```bash
tools/local_ci.sh fast
cargo test --no-run
```

Validate the configs that will be used:

```bash
cargo run -- validate --kind research \
  --config configs/q7_smoke_training_preset.json
cargo run -- validate --kind research \
  --config configs/q7_small_real_debug_training_preset.json
cargo run -- validate --kind research \
  --config configs/q7_reviewer_unseen_pocket_training_preset.json
jq empty configs/q7_core_ablation_matrix.json \
  configs/q7_claim_boundary_contract.json
```

Use `--kind experiment` for unseen-pocket experiment configs such as
`configs/unseen_pocket_pdbbindpp_real_backends.json`.

## 2. Evidence Tiers

| Tier | Command | Claim Use |
| --- | --- | --- |
| Smoke | `cargo run -- research train --config configs/q7_smoke_training_preset.json` | Compile, wiring, numerical-health, and regression checks only. |
| Real-data debug | `cargo run -- research train --config configs/q7_small_real_debug_training_preset.json` | Parser, batching, staged loss, and metric-path debug on capped real data. |
| Reviewer scale | `cargo run --release -- research train --config configs/q7_reviewer_unseen_pocket_training_preset.json` | Held-out-pocket model-design evidence when replay, leakage, raw/processed, and claim gates pass. |

Resume only from compatible artifacts:

```bash
cargo run --release -- research train \
  --config configs/q7_reviewer_unseen_pocket_training_preset.json \
  --resume
cargo run -- replay-check \
  --summary checkpoints/q7_training_presets/reviewer_unseen_pocket/training_summary.json \
  --checkpoint checkpoints/q7_training_presets/reviewer_unseen_pocket/latest.json
```

## 3. Evaluation And Generation Artifacts

Training writes validation/test summaries with no-grad batched evaluation,
resource usage, best-checkpoint metadata, and staged objective reports. Inspect:

```bash
cargo run -- report \
  --artifact-dir checkpoints/q7_training_presets/reviewer_unseen_pocket
```

Refresh candidate layers for the reviewer split when generation artifacts are
needed:

```bash
cargo run --release -- research generate \
  --config configs/q7_reviewer_unseen_pocket_training_preset.json \
  --resume \
  --all-examples \
  --split-label test \
  --num-candidates 4
```

Required artifacts for model-design review:

| Artifact | Role |
| --- | --- |
| `dataset_validation_report.json` | Confirms parser coverage, fallback use, and dataset fingerprint. |
| `training_summary.json` | Training/evaluation metrics, staged losses, validation history, best checkpoint, replay metadata. |
| `latest.json`, `best.json` | Checkpoint metadata and metric schema for resume/evidence compatibility. |
| `claim_summary.json` | Claim-facing summary with raw/processed layer audit, backend status, leakage calibration, and gates. |
| `generation_layers_*.json` | Candidate-level `generation_mode`, `generation_layer`, `model_native_raw`, and postprocessing provenance. |
| `repair_case_audit_*.json` | Raw-vs-repaired help, harm, neutral, raw-failure, and no-repair ablation evidence; repaired cases are postprocessing evidence only. |
| `ablation_matrix_summary.json` | Variant deltas for the configured core ablation matrix. |

Raw model-native claims must use `model_design.raw_model_*`, `raw_rollout`, or
candidate records with `model_native_raw=true`. Processed, repaired,
inferred-bond, reranked, constrained, and backend-scored metrics require their
`postprocessor_chain` and `claim_boundary`. A repaired-layer improvement is a
postprocessing-dependent finding unless the corresponding raw-native metric also
improves and passes its raw gate.

## 4. Ablations And Claim Gates

The core matrix contract is `configs/q7_core_ablation_matrix.json`. It includes
the executable generation-mode refinement boundary, the executable
`pocket_only_initialization_baseline`, and the executable
`de_novo_full_molecular_flow` variant. The pocket-only baseline removes target
ligand atom types and coordinates from decoder initialization and must use the
shape-safe `surrogate_reconstruction` primary objective. It remains a low-claim
initialization baseline. The de novo variant is claim-facing only when it uses
`generation_mode=de_novo_initialization`, the flow-matching backend,
`flow_matching.geometry_only=false`, and the geometry, atom-type, bond,
topology, and pocket/context flow branches.

Run an experiment-level ablation matrix from an unseen-pocket experiment config:

```bash
cargo run --release -- research ablate \
  --config configs/unseen_pocket_pdbbindpp_real_backends.json
```

Then run claim gates for the produced artifact directory:

```bash
python3 tools/claim_regression_gate.py \
  checkpoints/pdbbindpp_real_backends \
  --enforce-backend-thresholds \
  --enforce-data-thresholds
tools/local_ci.sh claim
```

Use `tools/local_ci.sh reviewer` only on machines with the configured backend
tools/data. Missing backends must remain explicit unavailable or failed backend
metadata, not heuristic successes.

## 5. Runtime And Dataset Assumptions

Smoke is CPU-only and should finish quickly. The small real-data debug preset is
intended for minutes-scale iteration on capped manifest data. Reviewer-scale
runs use the full manifest, batch size 8, 5000 steps, periodic validation, and
candidate comparison; record actual wall time, `examples_per_second`,
`evaluation_batch_size`, `forward_batch_count`, and `memory_usage_mb` from the
summary rather than copying hardware-specific expectations.

Reviewer-scale evidence assumes:

- Protein-level train/validation/test split with no train/test pocket leakage.
- Stable `split_seed`, corruption seed, sampling seed, sampler settings, and
  batch size for evidence-compatible comparisons.
- Candidate records retain source pocket/ligand provenance when backend scoring
  requires structure files.
- No de novo wording unless the run uses `de_novo_initialization`, the
  flow-matching backend, `flow_matching.geometry_only=false`, all five
  molecular flow branches, and `claim_context.de_novo_claim_allowed=true`.

## 6. Failure Triage

| Symptom | Check | First Response |
| --- | --- | --- |
| Nonfinite gradients or skipped optimizer steps | `training_history[*].gradient_health`, total loss, stage ramp | Lower learning rate, keep gradient clipping enabled, delay auxiliary stages, inspect the first nonfinite objective family. |
| Slot collapse | `slot_activation_mean`, `slot_assignment_entropy_mean`, `slot_activation_probability_mean`, slot balance losses | Increase slot balance weight, reduce slot sparsity, verify `slot_attention_masking_disabled` is not the claim-bearing variant. |
| Gate saturation | `gate_activation_mean`, `gate_saturation_fraction`, interaction path warnings | Tune gate temperature/bias separately from `eta_gate`; compare `interaction_gate_temperature_high` and `gate_sparsity_disabled`. |
| Leakage regression | `leakage_proxy_mean`, explicit leakage diagnostics, probe baseline comparisons | Keep `delta_leak` active, review `leakage_penalty_disabled`, and avoid no-leakage wording unless probe baselines support it. |
| Raw generation degradation | `model_design.raw_model_*` versus processed fields, `disable_candidate_repair` | Report improvements as postprocessing-dependent when raw metrics degrade; inspect raw candidate records before citing model-native quality. |
| Slow or memory-heavy evaluation | `resource_usage` batch fields and memory delta | Reduce evaluation batch size or candidate count; keep no-grad batched evaluation enabled. |

No artifact in this runbook supports experimental binding affinity, selectivity,
efficacy, human preference, or clinical claims without explicit external
experimental evidence.
