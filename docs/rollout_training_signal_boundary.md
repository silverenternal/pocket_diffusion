# Rollout Training Signal Boundary

Date: 2026-04-29

Decision: keep the current multi-step rollout recovery path evaluation-only.

The active rollout records are sampled/committed decoder trajectories. They
contain argmaxed atom choices, clipped coordinate updates, scalar stop
probabilities, guardrail diagnostics, and serialized coordinate records. Those
records are useful for monitoring generation behavior, but they are not a
tensor-preserving unroll suitable for optimizer-facing supervision.

## Current Optimizer-Facing Terms

- `topology`, `geometry`, and `pocket_anchor` in conditioned denoising are
  tensor-preserving primary components.
- `flow_velocity` and `flow_endpoint` are tensor-preserving only for
  flow-compatible primary objectives.
- Staged auxiliary terms are optimizer-facing only when their effective weight
  is positive and their execution mode is `trainable`.

## Detached Diagnostics

- `rollout_eval_recovery`
- `rollout_eval_pocket_anchor`
- `rollout_eval_stop`
- `stopped_early`
- `guardrail_blockable_stop_flag`
- severe clash, valence, and bond-length guardrail flags emitted by rollout
  records

These diagnostics can guide evaluation, search, and guardrail analysis, but they
do not contribute gradients to the decoder or encoders.

## Reserved Trainable Path

The config switch `training.enable_trainable_rollout_loss` remains a reserved
future switch and must validate as unsupported until a bounded differentiable
unroll exists. A future implementation would need:

- fixed maximum rollout steps with explicit memory guards,
- differentiable atom-logit transitions instead of committed argmax-only
  records,
- coordinate updates retained as tensors rather than serialized `Vec` records,
- clear `trainable_rollout_*` metric names distinct from `rollout_eval_*`, and
- gradient reachability tests proving decoder parameters receive rollout-loss
  gradients.

## Coverage Artifacts

Training summaries now persist `objective_coverage`, a per-run report that
records each observed primary component and auxiliary family with:

- `differentiable`,
- `optimizer_facing`,
- observed stage availability, and
- a claim-boundary note.

Rollout evaluation components are marked non-differentiable and
non-optimizer-facing in both step-level `component_provenance` and the per-run
coverage report.
