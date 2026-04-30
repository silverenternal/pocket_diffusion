# Q15 Optimizer-Facing Rollout Training

Date: 2026-04-29

The short-rollout training path is controlled by `training.rollout_training` and is default-off. It is independent from no-grad rollout diagnostics controlled by `training.build_rollout_diagnostics`.

## Config Contract

- `enabled`: builds tensor-preserving short-rollout records and allows rollout-state losses.
- `warmup_step`: first global trainer step where rollout-state losses contribute to the optimizer objective.
- `rollout_steps`: bounded to 1 to 3 when enabled.
- `detach_policy`: `detach_between_steps` by default to prevent unbounded graph growth; `full_graph` is available for small ablations.
- `max_batch_examples`: caps how many examples contribute rollout-state loss.
- `allowed_generation_modes`: optional mode gate. Empty means all active generation modes are allowed.
- loss weights: `atom_validity_weight`, `bond_consistency_weight`, `pocket_contact_weight`, `clash_weight`, and `endpoint_consistency_weight`.

## Report Contract

`StepMetrics.losses.rollout_training` reports:

- `enabled` and `active`, separating configured rollout training from warmup state.
- `teacher_forced_loss`, `rollout_state_loss`, and `teacher_rollout_divergence`.
- `generated_state_validity` as a compact bounded generated-state proxy.
- per-term rollout-state contributions for atom validity, bond consistency, pocket contact, clash margin, and endpoint consistency.
- `memory_control`, `detach_policy`, `configured_steps`, `executed_steps_mean`, and `max_batch_examples`.

The sampled `generation.rollout` artifact remains no-grad diagnostic evidence. The tensor record is `generation.rollout_training` and is emitted only when `training.rollout_training.enabled` passes the mode gate.
