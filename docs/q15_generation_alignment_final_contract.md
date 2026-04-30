# Q15 Generation Alignment Final Contract

Date: 2026-04-30

This document is the Q15 claim boundary for pocket-conditioned molecular
generation. It summarizes which signals are optimizer-facing, which artifacts
are raw model-native evidence, and which ablations are required before using
strong generation wording.

## Architecture Boundary

The model keeps topology, geometry, and pocket/context encoders structurally
separate. Each branch retains slot decomposition, and cross-modality exchange
remains explicit through directed gated attention or a named ablation mode. The
direct-fusion path is a negative control only and must not be described as the
preferred architecture.

Flow-based de novo generation is claim-addressable only when all of these are
visible in config and artifacts:

- `generation_mode=de_novo_initialization`
- `generation_method.primary_backend.family=flow_matching`
- `flow_matching.geometry_only=false`
- enabled `geometry`, `atom_type`, `bond`, `topology`, and `pocket_context`
  flow branches
- non-index matching provenance such as `hungarian_distance`
- raw-native layer metrics before repair, constraints, reranking, or backend
  scoring

## Optimizer-Facing Rollout

`training.rollout_training` is the bounded optimizer-facing rollout path. It is
separate from detached `rollout_eval_*` diagnostics and from serialized
generation artifacts.

The trainable rollout path is default-off and becomes claim-relevant only when
the run records:

- `StepMetrics.losses.rollout_training.enabled=true`
- `StepMetrics.losses.rollout_training.active=true` after warmup
- configured `rollout_steps` in `[1, 3]`
- `detach_policy` and `max_batch_examples`
- generated-state validity and per-term atom, bond, pocket-contact, clash, and
  endpoint contributions

Detached `rollout_eval_*` metrics remain evaluation diagnostics. They can gate
stage promotion and claim review, but they are not a training loss.

## Pocket Interaction Supervision

Pocket compatibility is no longer limited to a binary contact target. The
optimizer-facing pocket-interaction family contains:

- `pocket_contact`
- `pocket_pair_distance`
- `pocket_clash`
- `pocket_shape_complementarity`
- `pocket_envelope`
- `pocket_prior`

The thin contact ablation keeps only the contact term active. The rich
interaction profile enables distance, clash, shape, envelope, and pocket-prior
terms in addition to contact. These objectives are staged and ablation-ready
through `training.loss_weights.*` and the Q15 generation-alignment matrix.

## Equivariant Geometry Flow

The default geometry head remains the MLP-style `model.flow_velocity_head.kind =
geometry`. The exact rigid-motion equivariant head is available through
`equivariant_geometry` with pairwise geometry messages enabled. Reports keep
`FlowHeadAblationDiagnostics.head_kind`, `ablation_label`,
`equivariant_geometry_head`, and runtime fields visible so MLP and equivariant
runs cannot be merged.

The equivariant head may support an equivariance-specific method claim when the
rotation tests pass. It does not by itself prove better molecular generation.
That requires the raw-native evaluation matrix and matched ablations.

## Chemistry-Native Constraints

Chemistry guardrails are optimizer-facing only through native differentiable
objectives, not through post-hoc repair:

- `valence_guardrail`
- `bond_length_guardrail`
- `nonbonded_distance_guardrail`
- `angle_guardrail`

These are controlled by `training.loss_weights.upsilon_valence_guardrail`,
`phi_bond_length_guardrail`, `chi_nonbonded_distance_guardrail`, and
`psi_angle_guardrail`, with warmup stages in `training.chemistry_warmup`.
Raw graph metrics remain separate from constrained, repaired, inferred-bond,
reranked, or backend-scored layers.

## Objective Staging And Scale

The trainer uses staged objective activation:

1. task and consistency
2. low-weight redundancy and pocket/chemistry warmup
3. semantic probes, pharmacophore probes, and leakage controls
4. gate and slot utilization controls

Q15 adds three safeguards around this schedule:

- objective-family budget reports in
  `StepMetrics.losses.objective_family_budget_report`
- adaptive stage promotion gates using raw rollout validity, clash, pocket
  contact, and rollout pocket-anchor diagnostics
- sparse gradient diagnostics with sampled exact or loss-share proxy mode and
  dominant-family marking

Claim-facing training summaries must keep objective families separate:
`task`, `rollout`, `pocket_interaction`, `chemistry`, `redundancy`, `probe`,
`leakage`, `gate`, and `slot`.

The engineering boundary for those reports is centralized in
[`training_metrics_audit.md`](training_metrics_audit.md). Primary component
descriptors, flow branch ownership, branch component audits, and
objective-budget family grouping are owned by `src/training/metrics/`, not by
trainer-local string rules. This matters for claim wording because
optimizer-facing fields, detached diagnostics, native-score cap audit fields,
and branch-schedule provenance are all derived from the same registry-backed
metadata.

## Raw-Native Evaluation

Final generation claims must start from raw model-native evidence. The relevant
artifacts are:

- `evaluation_matrix_report` inside `experiment_summary.json`
- `raw_native_generation_report.json`
- `generation_layers_validation.json`
- `generation_layers_test.json`
- `ablation_matrix_summary.json`

`raw_rollout` and `raw_flow` are model-native layers. `constrained_flow`,
`repaired`, `inferred_bond_candidates`, `deterministic_proxy_candidates`,
`reranked_candidates`, and backend-scored rows are additive processed evidence.
Processed improvements cannot compensate for weak or missing raw-native
validity, pocket contact, clash, or diversity metrics.

## Generation-Alignment Ablation Matrix

The executable Q15 matrix config is:

- `configs/q15_generation_alignment_ablation_matrix.json`

The base row is explicitly named
`q15_generation_alignment_mlp_rollout_chemistry_rich_pocket_base`. It provides
the MLP geometry-flow baseline with rollout training, native chemistry
constraints, and the rich pocket-interaction objective enabled.

The generated variants isolate one primary mechanism where practical:

- `flow_head_equivariant_geometry`
- `rollout_training_disabled`
- `chemistry_native_constraints_disabled`
- `pocket_interaction_thin_contact_loss`
- `direct_fusion_negative_control`

`AblationRunSummary` records `raw_generation_quality`, `runtime`, and
`objective_families` for each executed variant, in addition to validation/test
comparison summaries.

## Claim Boundary

Allowed wording:

- "optimizer-facing bounded short-rollout objective" only when
  `training.rollout_training` is enabled and active in step metrics.
- "rich pocket-interaction supervision" only when contact, pair-distance,
  clash, shape/envelope, or pocket-prior losses are active and reported.
- "equivariant geometry flow head" only for runs with
  `model.flow_velocity_head.kind=equivariant_geometry`.
- "chemistry-native guardrails" only when native valence, bond-length,
  nonbonded-distance, or angle guardrail losses are optimizer-facing.
- "raw-native generation evidence" only from raw model-native layers.

Blocked wording:

- Do not describe repaired, constrained, reranked, or backend-scored metrics as
  raw model quality.
- Do not use de novo benchmark-quality language from smoke data or from a
  matrix that has not been rerun on the target held-out-pocket benchmark.
- Do not present direct fusion as the preferred architecture.
- Do not treat detached rollout diagnostics as optimizer-facing losses.

## Verification

Task-level verification for this contract used:

- `cargo test evaluation`
- `cargo test experiments`
- `cargo test reviewer_tooling`

Focused training-metrics checks for the optimizer-facing/audit boundary are:

- `cargo test primary_component --lib`
- `cargo test objective_budget --lib`
- `cargo test objective_family_budget --lib`
- `cargo test full_flow_branch_scale_report_covers_all_optimizer_branches --lib`
- `cargo test --test artifact_compatibility`

Final repository validation additionally uses the `final` todo profile:

- `jq empty todo.json`
- `cargo fmt --check`
- `git diff --check`
- `cargo test --no-run`
- `cargo test --no-fail-fast`

For a real-use generation review, run the stricter Q15 gate after producing the
main artifact directory and multi-seed summary:

```bash
python3 tools/claim_regression_gate.py checkpoints/<artifact_dir> \
  --enforce-real-generation-readiness \
  --multi-seed-summary checkpoints/<multi_seed_dir>/multi_seed_summary.json
```

The gate fails if evidence is heuristic-only, if split leakage is detected, if
raw-native quality is missing or below configured thresholds, if the standalone
raw-native report is absent, if Q15 generation-alignment ablations are missing,
or if fewer than three seeds are available.
