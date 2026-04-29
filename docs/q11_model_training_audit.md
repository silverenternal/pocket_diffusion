# Q11 Model And Training Design Audit

Date: 2026-04-29

Scope: model architecture, generation semantics, training objectives, staged optimization, leakage/probe design, and evaluation claim boundaries for the Rust pocket-conditioned molecular generation system.

## Executive Summary

The repository has a solid modular skeleton for the stated research direction. Topology, geometry, and pocket/context encoders are structurally separate; cross-modal interaction is explicit and directed; the config layer contains meaningful generation-mode compatibility contracts; and the trainer has a staged loss scheduler.

The largest remaining weaknesses are not missing modules. They are semantic mismatches between what the system records or claims and what the optimizer actually trains. In particular, rollout refresh diagnostics do not record the actual refreshed gates, flow rollout can condition on zero gate summaries under static context, rollout metrics are detached evaluation values rather than trainable rollout losses, and some ablation/auxiliary losses still compute or score disabled paths.

These issues should be fixed before using the current implementation for claim-bearing model comparisons.

## Remediation Notes

- P1 rollout records now build `path_usage.step_bucketed_path_means` from the gate summary actually used at each rollout step, including refreshed standard rollout and flow rollout contexts.
- Flow rollout static-context conditioning now starts from the initial raw interaction gate summary instead of a zero/default gate summary; flow-step diagnostics include `conditioning_gate_input_mean` and per-path conditioning gate values.
- The non-enforcing stop diagnostic now serializes as `guardrail_blockable_stop_flag`; legacy `stop_overridden_flag` remains a read-only serde alias for older artifacts.
- Rollout recovery remains evaluation-only. Config validation rejects `training.enable_trainable_rollout_loss=true` until a tensor-preserving rollout loss exists, and the former `training_step_weight_decay` naming has been narrowed to `rollout_eval_step_weight_decay` with a legacy read alias.
- Slot-control loss now ignores disabled modality branches and normalizes by active modality count, keeping topology-only, geometry-only, pocket-only, and full-modality ablations on a comparable scale.
- Slot attention visibility now supports `model.slot_decomposition.minimum_visible_slots` so early low activations can still expose slots to interaction paths while hard threshold active counts remain diagnostic.
- Auxiliary objectives now use a staged execution plan. Zero-weight families return explicit zero tensors and report `execution_mode=skipped_zero_weight`; optional disabled diagnostics can be computed under `tch::no_grad` as `detached_diagnostic`. A reproducible smoke report is in `docs/q11_staged_auxiliary_execution_smoke.md`.
- Leakage calibration now has a frozen-representation probe path that fits separate held-out ridge probes on completed forwards and records route/capacity-sweep metrics distinct from the training-time leakage penalty. A reproducible smoke report is in `docs/q11_leakage_probe_calibration_smoke.md`.
- Dataset validation now reports `coordinate_frame_contract`, finite-origin counts, ligand-centered frame counts, and `pocket_coordinates_centered_upstream`; translation tests confirm source-coordinate shifts preserve model-frame geometry/pocket tensors while moving only `coordinate_frame_origin`.
- Method comparison rows now expose `selected_metric_layer` beside `method_family`, and native metric-layer selection is tested across base, repair-only, reranker-only, and hybrid outputs so repaired/proxy/reranked layers cannot be mistaken for raw native evidence without explicit method-family provenance.

## Strong Points

- Separate modality branches are explicit in `Phase1ResearchSystem`: topology, geometry, pocket, interaction block, decoder, probes, and flow head are independent fields (`src/models/system/impl.rs:3-22`) and are constructed from separate encoder/decomposer instances (`src/models/system/impl.rs:27-69`).
- Controlled cross-modal interaction exists as six directed gated paths (`src/models/interaction.rs:40-107`), matching the intended `A(m <- n)` design better than unrestricted fusion.
- Config validation already checks generation backend, generation mode, and primary objective compatibility for config-driven entrypoints (`src/config/types/root.rs:55-128`, `src/training/entrypoints.rs:155-160`).
- Staged training weights are represented explicitly by `StageScheduler` and `EffectiveLossWeights` (`src/training/scheduler.rs:7-38`, `src/training/scheduler.rs:84-125`).
- Leakage code now distinguishes optimizer-facing objectives from detached diagnostics (`src/losses/leakage.rs:18-27`, `src/losses/leakage.rs:337-350`), which is a useful foundation for ablations.

## Findings

### Q11-AUD-001: Rollout path usage ignores refreshed gate summaries

Severity: High

In standard rollout, `refresh_generation_context` is called when the refresh policy triggers, but the returned gate summary is ignored (`src/models/system/rollout.rs:26-39`). The final rollout record always calls `path_usage_summary(raw_path_means, steps.len())` (`src/models/system/rollout.rs:131-143`), and `path_usage_summary` repeats the initial raw path means for every step bucket (`src/models/system/rollout.rs:297-306`).

Flow rollout has the same summary problem: the record also uses `path_usage_summary(raw_path_means, steps.len())` even when refresh changes conditioning (`src/models/system/flow_training.rs:175-187`).

Impact: `context_refresh_policy`, `refresh_count`, and step-bucketed path usage can report a refreshed rollout while still showing stale initial gates. This is a claim-risk issue for temporal interaction and path-usage analysis.

Recommendation: Carry a per-step `GenerationGateSummary` alongside each rollout step. Initialize it from the raw interaction summary, replace it on refresh, and build path usage from the actual sequence.

### Q11-AUD-002: Flow rollout uses a zero gate summary before refresh

Severity: High

Flow training conditions the velocity head with the current interaction gates (`src/models/system/flow_training.rs:52-55`). Flow rollout, however, initializes conditioning with `GenerationGateSummary::default()` (`src/models/system/flow_training.rs:84-87`) and only swaps in a real gate summary if the context refresh policy fires (`src/models/system/flow_training.rs:97-110`).

Impact: under a static or late-refresh policy, rollout-time flow velocity prediction receives zero gate features even though initial raw path means are available. This creates a train/inference conditioning mismatch.

Recommendation: initialize flow rollout conditioning with `raw_path_means`, and update it only when refresh produces a new gate summary.

### Q11-AUD-003: Rollout metrics are evaluation-only, not optimizer-facing

Severity: High

`ConditionedDenoisingObjective` computes rollout recovery and pocket-anchor metrics (`src/losses/task.rs:218-221`) but excludes them from `total` (`src/losses/task.rs:223-230`). They are logged as `rollout_eval_*` components only (`src/losses/task.rs:239-248`). The helper reconstructs tensors from serialized rollout records using `Tensor::from_slice` over `Vec` data (`src/losses/task.rs:614-672`, `src/losses/task.rs:708-719`), so it has no gradient path to the decoder.

Impact: configs and narratives can look like multi-step generation is being trained, while the optimizer only sees single-step denoising and auxiliary losses.

Recommendation: either implement a tensor-preserving rollout loss path and make any rollout training weight optimizer-facing, or rename/document rollout metrics as detached evaluation diagnostics only.

### Q11-AUD-004: Slot-control loss scores disabled modalities in ablations

Severity: High

Modality focus can zero disabled slot encodings (`src/models/system/impl.rs:375-395`). The slot-control objective still computes topology, geometry, and pocket penalties unconditionally and divides by three (`src/losses/auxiliary.rs:327-340`).

Impact: topology-only, geometry-only, or pocket-only ablations can receive constant dead-slot/balance penalties from disabled modalities, and active modality loss scale is diluted by a fixed denominator. This pollutes objective values and weakens ablation comparability.

Recommendation: make slot-control active-modality aware. Normalize by the number of enabled modalities and report disabled-modality slots as detached diagnostics.

Remediation: `SlotControlLoss` now skips modality branches whose slot weights are all zero and divides by the number of active branches. Tests cover full and single-modality focus settings and assert disabled branches do not add optimizer-facing penalties.

### Q11-AUD-005: Hard slot masks make attention visibility non-differentiable

Severity: Medium-High

Slot activations are sigmoid gates (`src/models/slot_decomposition.rs:95-109`, `src/models/slot_decomposition.rs:172-184`), but active masks and active counts are hard thresholded with `.gt(threshold)` (`src/models/slot_decomposition.rs:210-250`).

Impact: early random activation differences can hide slots from interaction/redundancy paths, while the threshold itself provides no useful gradient. This is risky for slot discovery and anti-collapse behavior.

Recommendation: use soft masks during warmup or a straight-through estimator, and enforce a minimum visible slot rule during early stages. Keep hard active counts as diagnostics.

Remediation: `SlotDecompositionConfig` now exposes `minimum_visible_slots` with a default of one attention-visible slot per non-empty active modality. Tests cover minimum visibility, post-warmup hard masking with `minimum_visible_slots=0`, and preservation of hard active-slot counts.

### Q11-AUD-006: Auxiliary objectives are computed even when staged weight is zero

Severity: Medium

The trainer computes auxiliary objectives before applying staged effective weights (`src/training/trainer.rs:368-401`). `AuxiliaryObjectiveBlock::compute_batch` computes redundancy, probes, leakage, gate, slot, consistency, pocket geometry, chemistry guardrails, and MI diagnostics unconditionally (`src/losses/auxiliary.rs:427-448`).

Impact: stage-1 or disabled objectives can still allocate tensors and build computation graphs. This wastes runtime and memory and makes profiling stage effects noisy.

Recommendation: split trainable auxiliary computation from detached diagnostics. Only build gradient-carrying losses with positive effective weight; run optional diagnostics under detached/no-grad semantics.

Remediation: the trainer now builds `AuxiliaryObjectiveExecutionPlan` from effective staged weights before auxiliary computation. Report entries include execution provenance, and the smoke report in `docs/q11_staged_auxiliary_execution_smoke.md` records stage-dependent trainable/skipped proxy counts and wall-time observations.

### Q11-AUD-007: Direct trainer construction bypasses config validation

Severity: Medium-High

`ResearchConfig::validate()` enforces generation-mode and objective compatibility (`src/config/types/root.rs:55-128`), and config-driven entrypoints call it (`src/training/entrypoints.rs:155-160`). `ResearchTrainer::new` builds the optimizer, scheduler, objective, and auxiliary block directly without validation (`src/training/trainer.rs:97-130`).

Impact: direct Rust API callers and tests can instantiate invalid combinations that config entrypoints would reject. This can silently produce misleading training behavior.

Recommendation: validate by default in `ResearchTrainer::new`, or add `new_validated` and make the unvalidated constructor crate-private/test-only.

### Q11-AUD-008: Flow objective silently returns zero without flow records

Severity: Medium-High

`FlowMatchingObjective` always builds from training config (`src/losses/task.rs:389-405`). Its core helper returns zero velocity and endpoint losses when `forward.generation.flow_matching` is absent (`src/losses/task.rs:457-475`), and also returns zero for empty flow tensors (`src/losses/task.rs:476-480`).

Impact: if an invalid direct training setup selects `FlowMatching` without a flow backend, the primary objective can become zero rather than failing loudly.

Recommendation: make absent flow records an error under flow objectives, or emit a trainer-level validation failure before the batch is accepted.

### Q11-AUD-009: legacy `stop_overridden` naming did not match behavior

Severity: Medium

Rollout sets `stop_overridden_flag = should_stop && guardrails.any()` (`src/models/system/rollout.rs:66-80`) but still records `stopped: should_stop` and breaks when `should_stop` is true (`src/models/system/rollout.rs:82-104`). The flag means "guardrails would have blocked this stop if enforcement existed", not that stop behavior was actually overridden.

Impact: downstream analysis can misread this as enforced guardrail intervention.

Remediation: new artifacts use `guardrail_blockable_stop_flag`, which states the current non-enforcing diagnostic semantics. Older `stop_overridden_flag` payloads are accepted as a serde alias during deserialization.

### Q11-AUD-010: Pocket local-frame translation contract is under-specified

Severity: Medium

The pocket encoder docs describe a deterministic frame around the ligand-centered pocket origin (`src/models/pocket_encoder.rs:3-8`). The local-frame transform multiplies raw coordinates by the frame without subtracting a centroid or anchor (`src/models/pocket_encoder.rs:268-272`). The current test checks rotation invariance but not translation behavior (`src/models/pocket_encoder.rs:487-518`).

Impact: the implementation may be correct if upstream data is always ligand-centered. If not, translation shifts leak into the representation.

Recommendation: make the coordinate-frame contract explicit in data validation and add a translation test. If ligand-centering is required, fail or warn when raw coordinates are not centered.

Remediation: `DatasetValidationReport` now records the ligand-centered coordinate-frame contract and whether retained pocket coordinates were centered upstream. The data tests cover uniform source-coordinate translation, and the pocket-encoder test explicitly documents that local-frame coordinates are translation-sensitive unless upstream ligand-centering has already been applied.

### Q11-AUD-011: Leakage probes are not yet calibrated leakage estimators

Severity: Medium

The module comment correctly states that similarity is only an early-warning proxy (`src/losses/leakage.rs:1-7`). Explicit leakage routes use current off-modality probe predictions and penalties such as `(tolerance - error).relu()` (`src/losses/leakage.rs:94-200`, `src/losses/leakage.rs:310-320`), with optional adversarial or detached semantics (`src/losses/leakage.rs:337-350`).

Impact: low explicit leakage can mean the probe is weak, not that the representation is clean. This matters for semantic specialization claims.

Recommendation: add a separate leakage-probe calibration protocol: train probes to convergence on frozen representations, report probe capacity sweeps, and use held-out leakage probe accuracy as a diagnostic distinct from optimizer penalties.

Remediation: `FrozenLeakageProbeCalibrationReport` records held-out route performance and capacity-sweep rows for separately fit probes on frozen pooled modality embeddings. The smoke fixture covers multiple routes, capacities, and regularization settings, and documents that low training-time leakage remains diagnostic unless held-out probes also fail to beat trivial baselines.

### Q11-AUD-012: Surrogate reconstruction should remain a bootstrap/debug objective

Severity: Medium

`SurrogateReconstructionObjective` reconstructs modality tokens and adds a small decoder bootstrap against the current partial ligand (`src/losses/task.rs:97-153`). It is useful for shape-safe smoke tests and pocket-only baselines, but it is not a generation-quality objective.

Impact: if results trained primarily with this objective are presented as molecular generation quality, the training signal is weaker than the claim.

Recommendation: keep this objective available, but label it as bootstrap/debug in docs, metrics, and experiment manifests.

## Recommended Execution Order

1. Fix correctness and claim semantics: rollout path usage, flow rollout conditioning, trainer validation, and absent-flow objective failures.
2. Align training signal with claims: decide whether rollout is trainable or explicitly evaluation-only.
3. Make ablations clean: active-modality slot loss and soft/warm-start slot visibility.
4. Improve staged efficiency: avoid building disabled auxiliary graphs.
5. Strengthen semantic specialization evidence: calibrated leakage probes and capacity sweeps.
6. Lock down geometry and evaluation contracts with targeted tests and documentation.

## Verification Guidance

Minimum checks after each implementation task:

- `cargo fmt --check`
- targeted module tests for the touched area
- `jq empty todo.json`

Before claim-bearing experiments:

- `cargo test --no-run`
- `cargo test --no-fail-fast`
- one small unseen-pocket split smoke run
- one modality-focus ablation smoke run
- one flow rollout smoke run when flow backend is enabled

## P0 Re-Audit Verification

Date: 2026-04-29

The Q12 baseline-contract pass rechecked the Q11 remediation surface after the
generation-boundary and method-fallback updates. The following commands passed:

- `cargo fmt --check`
- `cargo test generation_method_platform -- --nocapture`
- `cargo test generation_mode_compatibility_contract_covers_every_variant -- --nocapture`
- `cargo test --lib`
- `jq empty todo.json`

The library test sweep covered the targeted Q11 regression areas: actual
rollout gate usage, flow rollout conditioning from initial gates, loud flow
objective failures without flow records, rollout metrics remaining detached
evaluation diagnostics, active-modality slot loss normalization, minimum visible
slot masks, staged auxiliary execution provenance, frozen leakage-probe
calibration, coordinate-frame translation contracts, method-layer attribution,
trainer validation, and resume/replay boundaries. No new P0 blocker was found.
