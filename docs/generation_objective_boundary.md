# Generation Objective Boundary

Date: 2026-04-29

This note separates refinement paths, geometry-only flow, and the executable
de novo full molecular flow path.

## Executable Paths

### Target-Ligand Denoising

- Current role: decoder-anchored conditioned denoising/refinement.
- Initialization: corrupted target-ligand atom types and noisy target-ligand
  coordinates.
- Optimizer signal: `conditioned_denoising` primary objective plus configured
  staged auxiliary objectives.
- Claim boundary: target-ligand-conditioned refinement, not pocket-only de novo
  molecule construction.
- Reusable modules: topology encoder, geometry encoder, pocket encoder, slot
  decomposition, directed gated cross-modal interaction, decoder, probes,
  staged trainer, raw/processed candidate attribution.

### Ligand Refinement

- Current role: fixed-atom refinement of an existing ligand-like input.
- Initialization: externally supplied or target-derived ligand state under a
  refinement contract.
- Claim boundary: refinement of a supplied ligand state. It does not prove atom
  count selection, graph growth, or pocket-only sampling.
- Reusable modules: same fixed-atom decoder path as target-ligand denoising.

### Geometry-First Flow Refinement

- Current role: coordinate flow over a fixed ligand atom set.
- Initialization: current geometry flow may use corrupted coordinates for
  refinement training or configured coordinate noise for flow x0 construction.
- Optimizer signal: flow velocity and optional endpoint losses.
- Claim boundary: coordinate/geometry flow only. It does not generate atom
  types, bond existence, bond type, or topology.
- Reusable modules: geometry flow head, flow rollout diagnostics, gated
  conditioning summaries, trainer objective plumbing, flow artifacts.

### De Novo Full Molecular Flow

- Current role: pocket-conditioned molecular generation without target-ligand
  initialization.
- Required config: `generation_mode=de_novo_initialization`,
  `generation_method.primary_backend.family=flow_matching`,
  `flow_matching.geometry_only=false`, and enabled branches `geometry`,
  `atom_type`, `bond`, `topology`, and `pocket_context`.
- Initialization: pocket-size atom-count policy plus deterministic
  pocket-centered atom-type and coordinate scaffold.
- Conditioning: topology/geometry encoders see the generated scaffold, not the
  target ligand. Pocket features are recentered for this path. Target ligand
  atom types, bonds, and coordinates are training supervision only.
- Optimizer signal: coordinate velocity/endpoint losses plus atom-type
  categorical loss, bond existence/type loss, topology synchronization loss,
  pocket/context representation loss, branch synchronization loss, optional
  bounded short-rollout losses from `training.rollout_training`, rich
  pocket-interaction objectives, and chemistry-native guardrails when their
  staged weights are active.
- Claim boundary: native de novo molecular flow evidence only when the full
  branch set, non-index matching provenance, raw-native layer metrics, and
  dataset leakage gates are persisted. Compact smoke evidence proves execution
  and artifact wiring, not benchmark-quality generation.

### Pocket-Only Initialization Baseline

- Current role: conservative baseline that removes target-ligand coordinates
  from initialization by using configured atom-count and pocket-centroid
  offsets.
- Optimizer signal: shape-safe `surrogate_reconstruction` only.
- Claim boundary: pocket-only fixed-atom initialization baseline, not true de
  novo generation.
- Reusable modules: pocket/context encoder, fixed-atom decoder, artifact
  provenance, config validation.

## Remaining De Novo Evidence Requirements

The executable path now exists, but stronger claims still require persisted
evidence:

- Full-branch config and loss provenance in every claim-bearing run.
- Objective-family budget reports covering task, rollout, pocket-interaction,
  chemistry, redundancy, probe, leakage, gate, and slot families.
- Active rollout-training metrics when optimizer-facing rollout wording is used.
- Dataset leakage gates for target-ligand-centered context dependency.
- Joint evaluation gates for atom type, bond structure, topology validity,
  geometry quality, pocket fit, raw-versus-processed attribution, and efficiency.
- Raw-native generation reports and evaluation matrix rows that keep raw,
  constrained, repaired, reranked, and backend-scored evidence separate.
- Generation-alignment ablations for flow head, rollout training, chemistry
  constraints, pocket-interaction loss richness, and controlled-interaction
  negative control.
- Multi-seed and held-out-pocket evidence showing that native raw flow quality
  does not depend on repaired/reranked postprocessing.
- Claim summaries with `raw_native_evidence` leading
  `processed_generation_evidence`, plus `postprocessing_repair_audit` whenever
  repaired-layer improvements are discussed.

## Input Availability Contract

The machine-readable generation-mode contract in
`src/config/types/generation.rs` exposes the following availability labels:

| Feature family | Availability |
| --- | --- |
| Pocket/context features | `inference_available` |
| Target-ligand atom types | `target_supervision_only` |
| Target-ligand topology/bonds | `target_supervision_only` |
| Target-ligand coordinates | `target_supervision_only` |
| Repaired/reranked/backend candidate layers | `postprocessing_only` |

These labels are deliberately conservative. Even refinement modes may consume a
ligand-like input at inference, but target dataset labels remain
`target_supervision_only` and cannot be treated as pocket-only inference inputs.

## Q15 Alignment Mechanisms

The Q15 final method contract is maintained in
[`q15_generation_alignment_final_contract.md`](q15_generation_alignment_final_contract.md).
The relevant config and report surfaces are:

- `training.rollout_training` for optimizer-facing bounded short-rollout
  losses.
- `training.loss_weights.rho_pocket_contact`,
  `lambda_pocket_pair_distance`, `sigma_pocket_clash`,
  `omega_pocket_shape_complementarity`, `tau_pocket_envelope`, and
  `kappa_pocket_prior` for thin versus rich pocket-interaction supervision.
- `model.flow_velocity_head.kind` and `model.pairwise_geometry` for MLP versus
  equivariant geometry-flow heads.
- `training.loss_weights.upsilon_valence_guardrail`,
  `phi_bond_length_guardrail`, `chi_nonbonded_distance_guardrail`, and
  `psi_angle_guardrail` for chemistry-native constraints.
- `training.objective_scale_diagnostics`,
  `training.objective_gradient_diagnostics`, and
  `training.adaptive_stage_guard` for staged objective scale and promotion
  review.
- `raw_native_generation_report.json`, `evaluation_matrix`, and
  `ablation_matrix_summary.json` for raw-native evaluation and ablation
  evidence.

## Negative Tests And Gates

Required regression coverage:

- `generation_mode_compatibility_contract_covers_every_variant` verifies every
  mode has sources, input availability labels, and compatible objective/backend
  metadata.
- `generation_method_platform_de_novo_generation_mode_requires_flow_contract`
  verifies `de_novo_initialization` requires the flow-matching backend,
  `geometry_only=false`, and the full molecular branch set.
- Full-flow config validation rejects claim-bearing de novo configs with
  index-only matching or disabled/zero-weight required branch schedules.
- `generation_method_platform_allows_pocket_only_initialization_baseline`
  verifies the pocket-only baseline remains executable while
  `permits_de_novo_claims()` stays false.
- Config validation must reject incompatible primary objective and generation
  mode pairs before a trainer or evaluation run starts.
