# Q14 Active Path Review

Date: 2026-04-29

This review records the active model, training, rollout, and evaluation paths
after Q14 hardening. It is scoped to the Rust stack that is constructed from
normal research configs and invoked by the trainer and unseen-pocket experiment
entry points.

The current hardened path keeps topology, geometry, and pocket/context encoders
separate; routes decoder and full-flow heads through slot-local conditioning
unless an explicit ablation selects mean pooling; keeps six directed gated
cross-modal interaction paths explicit; separates optimizer leakage penalties
from frozen held-out leakage audits; and keeps raw-native generation evidence
separate from repaired, inferred-bond, reranked, and backend-scored layers.

## Active Model Construction

- Config validation starts at `ResearchConfig::validate` in
  `src/config/types/root.rs`; generation-mode compatibility is declared by
  `GenerationModeConfig::compatibility_contract` and backend resolution in
  `src/config/types/generation.rs`.
- `Phase1ResearchSystem::new` in `src/models/system/impl.rs:58` constructs
  three separate encoder/slot branches: topology, geometry, and pocket/context.
  These remain grouped in `EncoderStack`, not collapsed into an unrestricted
  fusion encoder.
- The same constructor adds the controlled interaction, generator, and probe
  stacks: `InteractionStack` with `CrossModalInteractionBlock`,
  `GeneratorStack` with `ModularLigandDecoder`, `FlowVelocityHead`, and
  `FullMolecularFlowHead`, and `ProbeStack` with `SemanticProbeHeads`.
- Data reaches the model through `MolecularBatch::collate` in
  `src/data/batch.rs`; dataset loading, validation, and split checks are under
  `src/data/dataset/` and `src/training/metrics/split_impl.rs`.

## Active Forward Path

- Single-example forwarding enters `forward_example_with_interaction_context`
  in `src/models/system/forward.rs:22`. It first builds an
  `OptimizerForwardRecord`, then constructs sampled rollout diagnostics under
  `no_grad` before assembling `ResearchForward`.
- Optimizer-facing construction is in
  `forward_example_optimizer_record_with_interaction_context`
  (`src/models/system/forward.rs:61`). It encodes the three modalities,
  decomposes slots, runs directed cross-modal interaction, computes probes,
  builds `ConditionedGenerationState`, decodes ligand outputs, and creates the
  optional flow-training record.
- Batch forwarding enters `forward_batch` / `forward_batch_with_interaction_context`
  in `src/models/system/forward.rs:174`. The active batched path is used for
  ordinary target-ligand denoising. De novo generation and flow-time-conditioned
  interaction deliberately fall back to per-example forwarding so scaffold
  atom counts and flow times remain explicit.
- Generation state construction is in `build_generation_state`
  (`src/models/system/forward.rs:396`). It combines the initial partial ligand
  with masked topology, geometry, and pocket slot contexts plus directed
  interaction outputs.

## Active Generation Modes

- `target_ligand_denoising`: `initial_partial_ligand_state`
  (`src/models/system/forward.rs:443`) reads
  `decoder_supervision.corrupted_atom_types` and `noisy_coords`; decoder and
  auxiliary losses are optimizer-facing, while rollout steps are diagnostics.
- `flow_refinement`: selected for flow-matching backend configs. The active
  flow record comes from `flow_matching_training_record` in
  `src/models/system/flow_training.rs:29`, which constructs aligned targets,
  velocity targets, molecular branch records, masks, and branch weights.
- `pocket_only_initialization_baseline`: uses
  `PocketCentroidScaffoldInitializer` in `initial_partial_ligand_state`; target
  ligand tensors remain supervision/evaluation inputs.
- `de_novo_initialization`: uses `DeNovoScaffoldInitializer` and
  `PocketVolumeAtomCountPrior` in `initial_partial_ligand_state`, and uses
  `de_novo_conditioning_modalities` in `src/models/system/impl.rs` to encode a
  pocket-conditioned scaffold instead of target ligand topology/geometry.

## Active Loss And Trainer Path

- The trainer path is `ResearchTrainer::train_batch_step_with_order_metadata`
  in `src/training/trainer.rs:642`. It selects the current staged schedule,
  forwards the batch, computes primary and auxiliary objectives, applies the
  optimizer step, emits metrics, and triggers checkpoints.
- Primary objective construction is in `build_primary_objective`
  (`src/losses/task.rs:439`). Flow objectives use
  `flow_matching_velocity_and_endpoint_loss` and `molecular_flow_losses`
  (`src/losses/task.rs:512`, `src/losses/task.rs:558`).
- Auxiliary objective scheduling is centralized in
  `AuxiliaryObjectiveExecutionPlan` and
  `AuxiliaryObjectiveBlock::compute_batch_with_execution_plan`
  (`src/losses/auxiliary.rs:88`, `src/losses/auxiliary.rs:694`). It controls
  trainable, detached-diagnostic, and skipped-zero-weight execution modes for
  redundancy, probes, leakage, gate, slot, consistency, pocket geometry, and
  chemistry guardrails.
- Validation best-checkpoint selection is in `ValidationMonitor` in
  `src/training/entrypoints.rs:315`; training artifacts are persisted by
  `persist_training_artifacts` in `src/training/entrypoints.rs:474`.

## Active Rollout And Evaluation Path

- Non-flow rollout is in `src/models/system/rollout.rs`; flow rollout is in
  `rollout_flow_matching` (`src/models/system/flow_training.rs:102`).
- Native molecular graph extraction for generated flow states uses
  `src/models/flow/native_graph.rs`.
- Unseen-pocket experiment orchestration enters `UnseenPocketExperiment::run`
  in `src/experiments/unseen_pocket/run.rs`.
- Model-native candidate layers are built under
  `src/experiments/unseen_pocket/evaluation/`; claim-facing summaries separate
  raw model-native layers from repaired, inferred-bond, reranked, and backend
  scored layers.

## Legacy Or Demo Paths

- `src/main.rs` contains CLI demo/report helpers and should not be treated as
  the primary optimizer path.
- `src/training/demos.rs`, `src/generator.rs`, and historical
  `src/disentangle/*` code are non-primary unless a test or CLI command invokes
  them explicitly.
- Tests such as `tests/smoke_trainer_mi_integration.rs`,
  `tests/generation_method_platform.rs`, and `tests/artifact_compatibility.rs`
  exercise compatibility and smoke behavior; they are not benchmark evidence.

## Q14 Task Impact Map

| Q14 Area | Active Path Affected |
| --- | --- |
| P1 shape-safe objective contracts | `src/losses/probe.rs`, `src/losses/consistency.rs`, `src/losses/task.rs`, `src/models/system/flow_training.rs` |
| P2 leakage-control redesign | `src/losses/leakage.rs`, `src/losses/auxiliary.rs`, `src/experiments/unseen_pocket/leakage_calibration.rs`, claim diagnostics |
| P3 permutation-invariant matching | `src/models/system/flow_training.rs`, `src/losses/task.rs`, molecular branch metrics |
| P4 pocket-conditioned molecular-flow semantics | `src/models/flow/*`, `src/models/system/flow_training.rs`, `src/losses/task.rs` |
| P5 slot and interaction strengthening | `src/models/slot_decomposition.rs`, `src/models/interaction.rs`, `src/models/interaction/*`, `src/models/decoder.rs`, flow heads |
| P6 trainer/runtime cleanup | `src/models/system/forward.rs`, `src/training/trainer.rs`, `src/training/entrypoints.rs`, training metrics |
| P7 data and coordinate contracts | `src/data/*`, `src/training/metrics/split_impl.rs`, coordinate-frame artifacts |
| P8 evaluation and claim safety | `src/experiments/unseen_pocket/evaluation/*`, claim reports, artifact compatibility tests |
| P9 documentation and final validation | docs, config contracts, final verification profile |

## Q14 Final Smoke Evidence

The compact Q14 final smoke artifacts live under
`checkpoints/q14_final_smoke` and are summarized in
[`q14_final_smoke_summary.md`](q14_final_smoke_summary.md). They cover
conditioned denoising, geometry flow, and de novo full molecular flow. All three
surfaces wrote `training_summary.json`, `experiment_summary.json`,
`claim_summary.json`, and generation-layer artifacts with validation/test
finite-forward fraction `1.0` and total nonfinite gradient tensors `0`.

The de novo smoke records all five molecular-flow branches and
`target_alignment_policy=hungarian_distance`. This is execution and artifact
contract evidence only; broad unseen-pocket generalization and benchmark-quality
de novo generation still require larger held-out-pocket, multi-seed, and
backend-covered evidence.
