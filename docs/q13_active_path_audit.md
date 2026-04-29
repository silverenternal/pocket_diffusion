# Q13 Active Model And Training Path Audit

Date: 2026-04-29

This audit records the active Rust path for the model, trainer, rollout, and
claim-facing evaluation layers. It is scoped to the current runnable stack and
does not treat smoke artifacts as benchmark-quality evidence.

## Active Config-To-Model Path

- Config validation starts from `ResearchConfig::validate` in
  `src/config/types/root.rs`. Generation-mode compatibility is declared by
  `GenerationModeConfig::compatibility_contract` in
  `src/config/types/generation.rs:77`.
- `Phase1ResearchSystem::new` in `src/models/system/impl.rs:30` constructs the
  topology, geometry, and pocket semantic branches separately, then adds
  `CrossModalInteractionBlock`, `ModularLigandDecoder`, `SemanticProbeHeads`,
  `FlowVelocityHead`, and `FullMolecularFlowHead`.
- Batch collation uses `MolecularBatch::collate` and `MolecularBatchIter` in
  `src/data/batch.rs`; dataset loading and split validation are under
  `src/data/dataset/` and `src/training/metrics/split_impl.rs`.

## Generation Modes

### `target_ligand_denoising`

- Initialization uses `decoder_supervision.corrupted_atom_types` and
  `decoder_supervision.noisy_coords` in
  `Phase1ResearchSystem::initial_partial_ligand_state`
  (`src/models/system/forward.rs:384`).
- Optimizer-facing tensors:
  `decoded.atom_type_logits`, `decoded.coordinate_deltas`, semantic slots,
  probe outputs, and enabled auxiliary losses.
- Detached diagnostics:
  sampled `generation.rollout.steps`, `rollout_eval_*` primary components,
  guardrail flags, and candidate-layer summaries.
- Gradient-receiving components:
  active encoders, slot decomposers, directed interaction block, decoder,
  probes when probe/leak objectives are enabled, and auxiliary-loss paths
  selected by `StageScheduler`.

### `flow_refinement`

- Backend family `flow_matching` resolves default target-ligand denoising mode
  to `flow_refinement` via `GenerationModeConfig::resolved_for_backend`.
- `flow_matching_training_record` in `src/models/system/flow_training.rs:28`
  constructs `x0`, `x1`, `x_t`, velocity targets, and the optional molecular
  branch training record.
- Optimizer-facing tensors:
  `predicted_velocity`, `target_velocity`, endpoint reconstruction tensors, and
  any enabled atom-type, bond, topology, pocket-context, and synchronization
  branch tensors.
- Detached diagnostics:
  rollout coordinates exported through `GenerationRolloutRecord`, flow-head
  scalar diagnostics, native graph extraction diagnostics, and backend layers.
- Gradient-receiving components:
  encoders, slots, directed interaction, selected flow velocity head, full
  molecular flow head when non-geometry branches are enabled, and staged
  auxiliary objectives.

### `pocket_only_initialization_baseline`

- Initialization uses `PocketCentroidScaffoldInitializer` in
  `initial_partial_ligand_state` (`src/models/system/forward.rs:406`), with a
  fixed atom count and uniform atom-type token from config.
- Target ligand tensors remain supervision/evaluation inputs only. The initial
  conditioning state is pocket-derived.
- Optimizer-facing tensors:
  decoder/slot/probe paths under the configured surrogate or denoising
  objectives; rollout records remain detached.
- Gradient-receiving components:
  same model modules as target-ligand denoising when their objective family is
  active; no target ligand tensor is used as an initialization input.

### `de_novo_initialization`

- Initialization uses `DeNovoScaffoldInitializer` and
  `PocketVolumeAtomCountPrior` from `src/models/traits.rs`, called through
  `de_novo_conditioning_modalities` (`src/models/system/impl.rs:383`).
- The topology and geometry encoder inputs are built from the generated
  pocket-conditioned scaffold, while pocket features are recentered through
  `pocket_centered_features`.
- Optimizer-facing tensors:
  geometry flow, atom-type flow, bond flow, topology flow, pocket-context flow,
  synchronization losses, and configured auxiliary objectives.
- Detached diagnostics:
  native graph extraction outputs in rollout steps, graph validity scalars,
  layer summaries, backend scores, and claim reports.
- Gradient-receiving components:
  topology/geometry/pocket encoders, slot decomposers, directed interaction
  block, flow velocity head, full molecular flow head, probes when active, and
  enabled auxiliary objective modules.

## Trainer Path

- `Trainer::train_batch_step_with_order_metadata` in
  `src/training/trainer.rs:404` selects the current stage, forwards the batch
  with `InteractionExecutionContext`, computes primary and auxiliary losses,
  performs the optimizer step, builds metrics, and triggers checkpoints.
- Primary objectives are built in `src/losses/task.rs:432`; component
  provenance is serialized by `PrimaryObjectiveComponentMetrics` in
  `src/training/metrics/losses.rs:50`.
- Auxiliary execution mode is staged in `src/losses/auxiliary.rs`, with
  trainable, detached-diagnostic, and skipped-zero-weight paths represented in
  `AuxiliaryObjectiveReport`.

## Evaluation And Claim Layers

- Candidate layers are generated in
  `src/experiments/unseen_pocket/evaluation/generation_layers.rs:382`.
- `raw_rollout` / `raw_flow` are model-native layers. Repaired, inferred-bond,
  deterministic-proxy, reranked, and backend-scored layers are processed
  evidence and must be labeled separately.
- Artifact compatibility tests in `tests/artifact_compatibility.rs` assert
  model-native raw fields, claim boundaries, and legacy aliases.

## Verification Snapshot

- `jq empty todo.json`: pass.
- `cargo fmt --check`: pass.
- `cargo test --no-fail-fast`: pass.
  Observed test set: 320 library unit tests, 12 binary unit tests,
  40 integration tests, and doc tests with 0 failures.
- Backend-dependent evidence remains labeled as unavailable unless an external
  backend artifact explicitly reports success.
