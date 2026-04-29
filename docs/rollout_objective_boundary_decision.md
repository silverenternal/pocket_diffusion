# Rollout Objective Boundary Decision

## Decision

The current training path keeps rollout-derived losses detached from the
optimizer. `rollout_eval_*` metrics remain evaluation-only diagnostics, and the
config switch `training.enable_trainable_rollout_loss` continues to fail
validation.

## Rationale

The rollout path commits atom identities through argmax or sampling, exports
coordinates into scalar/vector records, and records stop-policy decisions as
non-tensor diagnostics. Treating those records as trainable losses would imply a
differentiable multi-step molecular objective that the current implementation
does not preserve.

Keeping this boundary explicit avoids silently mixing detached evidence with
optimizer-facing terms. Primary objectives may still use tensor-preserving
single-step decoder losses and flow-matching losses. Rollout records can be used
for validation, guardrail reporting, ablations, and reviewer-facing evidence,
but not for gradient updates.

## Contract

- Optimizer-facing primary components are reported with
  `optimizer_facing=true`.
- Rollout diagnostics use the `rollout_eval_*` prefix and are reported with
  `optimizer_facing=false`.
- `training.enable_trainable_rollout_loss=true` is rejected by config
  validation.
- Enabling a trainable rollout objective requires a future task that defines a
  tensor-preserving rollout state, memory/runtime bounds, and tests proving the
  rollout terms remain connected to autograd.
