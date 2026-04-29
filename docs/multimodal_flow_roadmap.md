# Multi-Modal Flow Contract

This document records the boundary between geometry-only flow configs and the
executable full multi-modal molecular flow stack.

## Executable Branches

The default optimizer-facing branch remains `geometry_flow`.

- State space: continuous ligand coordinates.
- Training path: `x_t = (1 - t) * x0 + t * x1`.
- Target: coordinate velocity `x1 - x0`.
- Losses: velocity MSE plus endpoint consistency.
- Inference: ODE-style coordinate integration with Euler or Heun updates.
- Fixed inputs in geometry-only configs: atom count, atom types, and topology
  come from the active generation-mode contract.

When `flow_matching.geometry_only=false` and all branches are enabled, atom
count comes from the pocket-conditioned de novo initializer and atom type, bond,
topology, and pocket/context branches are optimizer-facing.

## x0 Source Semantics

The flow config exposes `use_corrupted_x0`.

- `true`: `x0` is derived from target-ligand corrupted geometry and is labeled
  `target_ligand_corrupted_geometry`.
- `false`: `x0` is a deterministic pocket-centered Gaussian-like initialization
  and is labeled `deterministic_gaussian_noise`.

Training records and rollout records carry this label so artifacts can separate
target-ligand-conditioned refinement from pocket-only initialization work.

## Atom-Type Flow

- State space: categorical atom labels.
- Target: categorical endpoint supervision aligned to the generated draft atom
  count.
- Metrics: atom-type accuracy, negative log likelihood, valid atom fraction,
  and rare-type calibration.
- Coupling: receives pocket and topology context through gated directed
  attention, but must remain ablation-addressable as an atom-type branch.

This branch is implemented by `FullMolecularFlowHead`.

## Bond Flow

- State space: edge existence plus bond type/order.
- Target: binary edge transitions and categorical bond-type transitions.
- Metrics: bond F1, bond-type accuracy, valence violation rate, and edge
  calibration by atom-type pair.
- Coupling: reads atom-type and geometry states through explicit gates so bond
  predictions can depend on chemistry and distance without unrestricted fusion.

This branch is implemented by `FullMolecularFlowHead`. During de novo rollout,
native bond payloads are extracted from a synchronized bond/topology score with
a distance prior, a connectivity pass, and conservative atom-type valence
budgets before they are written to raw model-native artifacts.

## Topology Flow

This branch should coordinate atom and bond branches into a valid graph rather
than duplicate either one.

- State space: graph connectivity and consistency state.
- Targets: connectedness, valence feasibility, ring/fragment constraints where
  available, and synchronization between atom and bond predictions.
- Metrics: graph validity, connectivity rate, topology synchronization error,
  valence feasibility, and fragment count.
- Role: enforce graph-level consistency while keeping atom-type and bond flows
  separately testable.

This branch is implemented as topology/bond synchronization logits plus a
cross-branch synchronization loss. Rollout graph extraction consumes these
topology logits together with bond logits rather than thresholding bond logits
alone.

## Pocket/Context Flow

This is a conditional representation flow, not protein generation.

- State space: pocket-conditioned latent/context representation.
- Target: context consistency under ligand state updates, pocket contact
  preservation, and stable conditioning features.
- Metrics: context drift norm, pocket contact preservation, key-residue coverage,
  and condition-usage entropy.
- Role: keep conditioning synchronized as ligand atom, bond, topology, and
  coordinate states change.

This branch is implemented as a pocket/context representation consistency head.

## Cross-Modal Coupling

The required coupling pattern remains:

`A(m <- n) = gate(m,n) * Attention(Q_m, K_n, V_n)`

Rules:

- Keep modality-specific Q/K/V projections.
- Keep learned gates in `[0, 1]`.
- Log gate statistics and support sparse gate regularization.
- Preserve directed paths and ablation switches.
- Do not replace this with unrestricted full fusion.

The first synchronization objective should be conservative: geometry conditions
on topology and pocket context; bond conditions on atom type and geometry;
topology coordinates atom and bond consistency; pocket/context remains a
conditioning representation branch.

## Config And Artifact Contract

The config surface is:

- `generation_method.flow_matching.multi_modal.enabled_branches`
- `generation_method.flow_matching.multi_modal.branch_loss_weights`
- `generation_method.flow_matching.multi_modal.branch_schedule`
- `generation_method.flow_matching.multi_modal.allow_zero_weight_branch_ablation`
- `generation_method.flow_matching.multi_modal.warm_start_steps`
- `generation_method.flow_matching.multi_modal.claim_full_molecular_flow`

Default enabled branch: `geometry`.

Enabled branches are present in molecular-flow rollout and artifact contracts.
Static branch weights and branch schedules decide optimizer-facing pressure.
Present branches may have zero final optimizer weight only in explicit
`allow_zero_weight_branch_ablation=true` negative-control configs. Setting
`claim_full_molecular_flow = true` requires all required branches to be enabled,
implemented, and assigned positive static and scheduled optimizer weights.
`de_novo_initialization` additionally requires `flow_matching.geometry_only=false`.

Evaluation artifacts expose:

- enabled and disabled flow branches
- per-branch state-space metadata
- implementation support status
- full molecular flow claim gate and reason

## Staged Training Plan

Recommended activation order:

1. Geometry coordinate flow with endpoint consistency.
2. Atom-type categorical flow with low weight and frozen geometry baseline.
3. Bond existence and bond-type flow conditioned on atom and geometry states.
4. Topology synchronization and graph-validity consistency.
5. Pocket interaction-profile flow from matched ligand-pocket contact labels; keep pocket/context reconstruction diagnostic-only unless explicitly re-enabled.
6. Cross-branch synchronization losses and sparse gate regularization.

Each branch needs an independent disable switch, loss weight, metric block, and
ablation label before it becomes part of claim-facing evidence.

## Evaluation Gates

Use separate claim levels:

- `geometry_only_flow`: coordinate velocity branch active; no molecular graph
  generation claim.
- `partial_multimodal_flow`: at least one non-geometry branch enabled, with
  disabled branches listed.
- `full_multimodal_flow`: geometry, atom-type, bond, topology, pocket/context,
  and synchronization gates all implemented and validated.

Raw model outputs must remain separated from processed layers such as repair,
inferred bonds, constraints, external backends, and reranking.
