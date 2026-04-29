# Geometry Flow Matching And Molecular Flow

This repository keeps a **geometry-only flow matching generator** as the default
baseline and also supports config-gated **full molecular flow** for de novo
generation. The multi-modal contract is tracked in
[`multimodal_flow_roadmap.md`](multimodal_flow_roadmap.md).

## Method

For the active `geometry_flow` branch, we construct a linear coordinate path:

- `x_t = (1 - t) * x0 + t * x1`
- `x0`: noisy/corrupted ligand coordinates (configurable source and scale)
- `x1`: target ligand coordinates

The model predicts velocity:

- `v_theta(x_t, t, conditioning) ~= x1 - x0`

Conditioning is explicitly reused from the existing architecture:

- topology context
- geometry context
- pocket context
- gated cross-modal summary

The new `GeometryFlowMatchingHead` adds timestep embedding and fuses it with modality conditioning and coordinates.
It now uses translation-invariant coordinate features (centered at `x0` centroid) plus explicit displacement features (`x_t - x0`) and a lightweight residual+normalization block for stabler velocity prediction.

## Training path

New primary objectives:

- `flow_matching`: tensor-preserving flow-refinement velocity and endpoint objective.
- `denoising_flow_matching`: hybrid training-objective composition that combines conditioned denoising and flow matching; it is not a separate generation mode.

Flow loss is MSE over per-atom velocity:

- `MSE(v_pred, v_target)` where `v_target = x1 - x0`

To stabilize endpoint geometry during training, the objective also applies a low-weight endpoint consistency term reconstructed from `(x_t, v)` at the sampled `t`.

All existing staged auxiliary losses remain unchanged.

## Inference path

When the active backend family is `flow_matching`, rollout uses ODE-style coordinate integration:

- Euler: `x_{t+dt} = x_t + dt * v_theta(x_t, t)`
- Heun: predictor-corrector variant

With `geometry_only=true`, the implementation supports low step counts (e.g.,
10-50), geometry-only transport, and keeps topology fixed. With
`geometry_only=false` and all molecular branches enabled, rollout also updates
atom types and native bond/topology payloads from `FullMolecularFlowHead`.
Native bonds are decoded through a topology-synchronized graph extractor that
combines bond logits, topology logits, coordinate distance priors, connectivity,
and conservative valence budgets before writing raw rollout artifacts.

## Pipeline compatibility

- evaluator path unchanged
- artifact schema remains backward compatible (new flow metrics are additive)
- claim/evidence pipeline unchanged
- output still enters existing repair + rerank stack
- rollout diagnostics artifact can be exported with `python3 tools/flow_rollout_diagnostics.py checkpoints/<surface>` (writes `flow_rollout_diagnostics.json`)

## New config surface

`research.generation_method.flow_matching`:

- `steps`
- `noise_scale`
- `integration_method`: `euler | heun`
- `geometry_only` (`true` for coordinate-flow baseline, `false` for full
  molecular flow)
- `use_corrupted_x0`
- `multi_modal.enabled_branches` (defaults to `["geometry"]`)
- `multi_modal.branch_loss_weights`
- `multi_modal.branch_schedule`
- `multi_modal.allow_zero_weight_branch_ablation` (required when a present
  branch is intentionally kept at zero final optimizer weight)
- `multi_modal.warm_start_steps`
- `multi_modal.claim_full_molecular_flow` (requires every required branch)

Example config: `configs/flow_matching_experiment.json`.

## Current limitations

- lightweight velocity head and deterministic synthetic noise path (v1 baseline)
- not a full diffusion pipeline
