# Modular Research Stack Map

This map records the current module boundaries for the pocket-conditioned
molecular generation stack. It is intended to keep implementation, ablation, and
paper claims aligned.

## Model Backbone

- `src/models/semantic.rs`
  - Owns the three modality-specific semantic branches.
  - Keeps topology, geometry, and pocket/context encoders structurally separate.
  - Each branch owns its encoder and soft slot decomposer.

- `src/models/interaction.rs`
  - Owns the controlled cross-modality interaction block.
  - Keeps all six directed gated attention paths explicit:
    topology from geometry, topology from pocket, geometry from topology,
    geometry from pocket, pocket from topology, and pocket from geometry.
  - Path roles, expected signals, failure modes, ablation interpretation, and
    heuristic-biased paths are audited in
    [`controlled_interaction_audit.md`](controlled_interaction_audit.md).

- `src/models/system/`
  - Wires semantic branches, interaction block, probes, decoder state, rollout,
    and flow-matching training records.
  - The module entrypoint is `src/models/system/mod.rs`; no root-level
    `system.rs` shim is required.

## Trait Boundaries

- Encoders
  - Public traits and records live in `src/models/traits.rs`.
  - Implementations should keep topology, geometry, and pocket/context branches
    replaceable without adding unrestricted fusion.
- Slot decomposers
  - Branch-local slot logic belongs with `src/models/semantic.rs`.
  - New slot losses or utilization metrics should surface through diagnostics
    rather than ad hoc logging.
- Interactions
  - Directed gated cross-modal interaction belongs in `src/models/interaction.rs`
    and `src/models/interaction/`.
  - New cross-modal paths must remain named, gated, sparse-regularizable, and
    ablation-addressable.
- Objectives
  - Primary objective variants belong in `src/losses/task.rs`.
  - Auxiliary objective families belong in `src/losses/auxiliary.rs` or focused
    `src/losses/` modules when they become reusable.
- Trainer hooks
  - Scheduling, replay/resume metadata, checkpoint emission, and metrics
    reporting belong under `src/training/`.
  - Trainer changes should preserve reproducibility metadata in
    `src/training/metrics/reproducibility.rs`.
- Datasets
  - Dataset traits and loading boundaries live under `src/data/dataset/`.
  - Feature tensor construction and device movement live under
    `src/data/features/`.
  - Coordinate-frame and target-context leakage contracts are documented in
    [`data_coordinate_contracts.md`](data_coordinate_contracts.md).
- Evaluators
  - Candidate evaluators and drug metric wrappers live under
    `src/models/evaluation/`.
  - Experiment-level evidence aggregation lives under
    `src/experiments/unseen_pocket/evaluation/`.

## Diagnostic Surfaces

- `src/models/semantic.rs`
  - Emits `SemanticBranchDiagnostics` for each branch and a bundled
    `SemanticDiagnosticsBundle` in `ResearchForward.diagnostics`.
  - For each branch, the current fields are:
    - `modality`: branch key (`topology`, `geometry`, `pocket`),
    - `token_count`: number of branch tokens,
    - `slot_count`: number of slots,
    - `active_slot_fraction`: fraction of slots passing a low activation threshold,
    - `slot_entropy`: utilization spread across slots,
    - `pooled_norm`: pooled embedding norm (health + capacity scale),
    - `reconstruction_mse`: optional reconstruction error when target and reconstruction
      shapes align.
  - Interpretation:
    - `active_slot_fraction` + `slot_entropy` are indicators of specialization
      pressure and slot collapse risk (not direct purity guarantees).
    - `reconstruction_mse` is a health signal for slot bottleneck quality and
      reconstruction consistency.
    - Low or unstable values are treated as early warnings for ablation review.
- `docs/slot_semantics_stability.md`
  - Records the current minimum-visible-slot visibility rule, the future
    soft-to-hard schedule, per-slot alignment metrics, signature matching, and
    collapse alarms.
- `src/models/system/types.rs`
  - Emits `ResearchForwardDiagnostics` (compact, forward-level summary) and
    `ResearchForwardGenerationHealth` from `ResearchForward::diagnostics_bundle`.
  - `ResearchForwardGenerationHealth` tracks rollout execution health (configured
    vs executed steps, last-step displacement, step-scale trend, flow presence).
- `src/losses/task.rs` and `src/training/metrics/losses.rs`
  - Primary objective decomposition is recorded in
    `PrimaryObjectiveComponentMetrics` (`topology`, `geometry`, `pocket_anchor`,
    `rollout`, `flow_velocity`, `flow_endpoint`) as part of
    `PrimaryObjectiveMetrics`.

## Objectives

- `src/losses/task.rs`
  - Owns primary task objective variants and now exposes
    `PrimaryObjectiveWithComponents::compute_with_components` so each variant can
    emit term-level decomposition beside the scalar total.
  - Examples: surrogate reconstruction, conditioned denoising, flow matching,
    and hybrid denoising-flow objectives.

- `src/losses/auxiliary.rs`
  - Owns the paper-facing auxiliary objective block.
  - Computes unweighted `L_intra_red`, `L_probe`, `L_leak`, `L_gate`, `L_slot`,
    topology-geometry consistency, pocket contact, pocket clash, pocket
    envelope, valence guardrails, bond-length guardrails, pharmacophore
    role-probe/leakage subterms, and MI diagnostics.

- `src/training/trainer.rs`
  - Owns staged scheduling, weighted objective aggregation, optimizer steps,
    checkpointing, and metric emission.
  - It does not directly own individual auxiliary objective implementations.
  - Optional adaptive stage readiness guards and training runtime profiling are
    documented in [`adaptive_staging_design.md`](adaptive_staging_design.md).

## Generation And Training Boundary

The detailed refinement versus true de novo contract is maintained in
[`generation_objective_boundary.md`](generation_objective_boundary.md).
The rollout optimizer boundary is maintained in
[`rollout_training_signal_boundary.md`](rollout_training_signal_boundary.md).
The molecular-flow branch and target-matching contracts are maintained in
[`q13_molecular_flow_contract.md`](q13_molecular_flow_contract.md),
[`q14_pocket_flow_branch_contract.md`](q14_pocket_flow_branch_contract.md), and
[`q14_full_flow_schedule.md`](q14_full_flow_schedule.md).

- Conditioned denoising/refinement
  - The active decoder-anchored path consumes target-ligand-derived corrupted
    atom types and noisy coordinates during training.
  - It is a ligand-conditioned refinement objective, not evidence of pocket-only
    de novo molecule construction.
- Geometry-first flow matching
  - Geometry-only configs predict coordinate velocity over a fixed ligand atom
    set and should be described as geometry or coordinate flow.
  - They do not support atom-type, bond, topology, or de novo wording unless
    `flow_matching.geometry_only=false` and the full molecular branch contract
    is active.
- Postprocessing and candidate layers
  - `raw_rollout` and `raw_flow` are model-native evidence.
  - `constrained_flow`, `repaired`, `inferred_bond`, `deterministic_proxy`, and
    `reranked` are processed layers and must remain attributed separately from
    raw model behavior.
- De novo full molecular flow
  - The executable path is `generation_mode=de_novo_initialization` with the
    flow-matching backend, `flow_matching.geometry_only=false`, and enabled
    `geometry`, `atom_type`, `bond`, `topology`, and `pocket_context` branches.
  - Topology and geometry encoders consume the generated pocket-conditioned
    scaffold, not target-ligand topology or coordinates. Target ligand fields
    are training supervision only.
  - Claim-facing de novo wording additionally requires non-index target matching
    such as `hungarian_distance`, persisted branch schedule/matching provenance,
    dataset target-context leakage review, raw-native layer metrics, and
    explicit raw-vs-processed attribution.
  - Compact smoke artifacts prove that the path runs; they do not by themselves
    establish benchmark-quality molecular generation.
- Optimizer-facing versus diagnostics
  - Primary losses and staged auxiliary losses are optimizer-facing only when
    their configured effective weights are active and tensor paths remain
    differentiable.
  - Rollout recovery metrics currently use serialized rollout diagnostics and
    are reported as detached `rollout_eval_*` values unless a future
    tensor-preserving rollout objective is implemented.

## Configuration Switches

- `src/config/types/training.rs`
  - Owns staged schedule boundaries (`stage1_steps`, `stage2_steps`,
    `stage3_steps`, `stage4_warmup_steps`) and objective-weight families
    (`alpha_primary`, `beta_intra_red`, `gamma_probe`, `delta_leak`,
    `eta_gate`, `mu_slot`, `nu_consistency`) for staged training ablations.
  - Owns `chemistry_warmup` stage gates. Recommended activation order is:
    keep task/consistency active from stage 1, enable pocket envelope plus
    valence and bond-length guardrails from stage 2, enable pharmacophore
    role probes and role-leakage controls from stage 3, then add gate/slot
    sparsity through the configurable Stage 4 warmup.
  - Exposes explicit leakage-probe flags (`enable_explicit_probes` and per-route
    probes) for controlled off-modality supervision experiments.
  - Exposes `adaptive_stage_guard`, which is disabled by default and can either
    report readiness warnings or hold stage advancement with explicit reasons.
  - Leakage/probe claim boundaries are maintained in
    [`leakage_probe_protocol.md`](leakage_probe_protocol.md). Training artifacts
    must keep optimizer penalties, detached diagnostics, and frozen-probe audits
    as separate evidence roles.
- `src/config/types/model.rs`
  - Owns interaction ablation/regularization switches including interaction mode
    and gating hyper-parameters (`gate_temperature`, `gate_bias`).
  - `interaction_tuning.gate_regularization_path_weights` optionally scales
    individual directed paths inside `L_gate`; empty config preserves the prior
    six-path average.

## Evaluation And Evidence

- `src/models/evaluation/`
  - Owns candidate scoring, evaluator orchestration, and drug-metric wrapping.

- `src/experiments/unseen_pocket/metrics.rs`
  - `ChemistryCollaborationMetrics` records chemistry-role gate usage,
    pharmacophore role coverage/conflict, clash/valence/bond guardrails, and
    key-residue coverage with explicit provenance labels and `null` values for
    unavailable evidence.
  - `TrainEvalAlignmentReport` separates optimizer-facing training terms,
    detached rollout diagnostics, raw versus processed candidate layers,
    method-family selected metric layers, and backend coverage. The detailed
    contract is documented in [`train_eval_alignment.md`](train_eval_alignment.md).
  - `raw_native_evidence`, `processed_generation_evidence`, and
    `postprocessing_repair_audit` are the claim-facing separation between raw
    model-native behavior, constrained/repaired/reranked outputs, and repair
    case evidence. Repaired improvements must not be cited as raw generation
    evidence.

- `configs/drug_metric_artifact_manifest.json`
  - Records claim-facing evidence artifact paths.

- `artifacts/evidence/`
  - Stores generated evidence files rather than leaving claim artifacts in the
    repository root.

## Where To Add Features

- New model capacity should start in the owning branch module:
  `semantic.rs` for modality/slot work, `interaction.rs` or
  `src/models/interaction/` for gated cross-modal exchange, and
  `src/models/flow/` for flow-head variants.
- New training behavior should start in `src/training/` and expose config
  fields under `src/config/types/`; avoid wiring experimental behavior only
  through scripts.
- New evidence or reporting should add a manifest entry in
  `configs/validation_manifest.json` or
  `configs/drug_metric_artifact_manifest.json` before becoming claim-facing.
- New external backend adapters belong in `tools/` and should emit coverage,
  status, backend identity, and failure fields rather than silently imputing
  missing metrics.

## Interaction Observability

  - `src/models/interaction.rs`
  - Emits `CrossModalInteractionDiagnostics` via
    `CrossModalInteractionBlock::forward_with_diagnostics` and
    `forward_batch_with_diagnostics`.
  - Each directed path record includes:
    - `path_name`: stable path identifier,
    - `gate_mean` and `gate_abs_mean`,
    - `attention_entropy`,
    - `attended_norm`,
    - optional bias fields: `bias_mean`, `bias_min`, `bias_max`,
      `bias_scale`.
  - Supported path IDs:
    `topo_from_geo`, `topo_from_pocket`, `geo_from_topo`,
    `geo_from_pocket`, `pocket_from_topo`, `pocket_from_geo`.
  - Stable role labels are emitted in diagnostics as `path_role`:
    - `topo_from_geo` → topology-informed bond plausibility
    - `topo_from_pocket` → pocket-informed ligand chemistry preference
    - `geo_from_topo` → topology-constrained conformer geometry
    - `geo_from_pocket` → pocket-shaped pose refinement
    - `pocket_from_topo` → ligand-chemistry pocket compatibility
    - `pocket_from_geo` → pose occupancy feedback
  - Interpretation:
    - Diagnostics are intended for path-level usage and comparison in ablations.
    - Geometry-pocket bias stats are conditional conditioning diagnostics, not a
      free fusion channel.
    - `AuxiliaryLossMetrics.gate_path_contributions` records each path's raw
      gate magnitude, configured path weight, normalized contribution to
      `L_gate`, and staged optimizer contribution.

## Validation Checks

- `tools/validation_suite.py`
  - Orchestrates command checks and focused validation modules.
- `tools/validation/artifacts.py`
  - Owns shared JSON artifact loading helpers.
- `tools/validation/reporting.py`
  - Owns JSON/Markdown report summaries and keeps the report schema stable.
- `tools/validation/architecture.py`
  - `validate_modular_architecture_boundaries` enforces modular boundaries
    (no root-level `system` shim, required module files present, smoke/q2/q3
    artifact path hygiene).
  - `validate_architecture_module_map` checks that this document retains the
    required source-of-truth anchors.
- `tools/validation/claim_provenance.py`
  - `validate_provenance_safe_pharmacology_claims` rejects claim artifacts that
    promote heuristic-only pharmacophore, key-residue, docking, or affinity
    metrics to backend, docking, or experimental evidence without explicit
    supporting provenance.
- `tools/validation/negative_fixtures.py`
  - Generates temporary negative fixtures for known claim-risk regressions.
- `tools/validation_suite.py` internal artifact checks
  - `validate_drug_metric_contract` validates manifest and contract consistency.
  - `validate_q2_artifacts`, `validate_layer_attribution`,
    `validate_backend_sanity`, and `validate_coordinate_frame_consistency` keep
    experiment- and geometry/coord sanity checks tied to their artifacts.
  - `validate_diagnostic_presence_for_smoke_artifacts` (optional) verifies that
    smoke diagnostics include semantic, interaction, objective, and
    synchronization coverage when such artifacts are present, and fails on
    nonzero synchronization mismatch counters.

## Ablation Handles

- Modality branches: disable or replace topology, geometry, or pocket/context
  paths via `src/models/semantic.rs` and matching model config fields.
- Slot usage: adjust slot count, sparsity, balance, and diagnostics through
  semantic branch config plus `src/losses/auxiliary.rs`.
- Cross-modal paths: disable or replace directed gated paths via
  `src/models/interaction.rs`; each path must keep a stable path ID and role.
- Flow heads: switch `geometry` versus `atom_pocket_cross_attention` through
  flow-head config under `src/models/flow/` and experiment configs.
- Primary objectives: replace objective implementations through
  `src/losses/task.rs`.
- Auxiliary objectives: replace redundancy/probe/leak/gate/slot/chemistry
  groups through `src/losses/auxiliary.rs` or focused loss modules.
- Training stages: use staged loss weights, explicit probe flags, and chemistry
  warmup gates in `src/config/types/training.rs`.
- Dataset boundaries: use `src/data/dataset/` loaders and split config rather
  than embedding source-specific assumptions in model code.
- Evaluator substitutions: add evaluator implementations under
  `src/models/evaluation/` and aggregate claim-facing evidence through
  `src/experiments/unseen_pocket/evaluation/`.
