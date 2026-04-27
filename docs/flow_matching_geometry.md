# Geometry Flow Matching (v1)

This repository now includes a **geometry-only flow matching generator** that minimally reuses the existing disentangled + conditioned stack.

## Method

For coordinates, we construct a linear path:

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

- `flow_matching`
- `denoising_flow_matching` (hybrid objective)

Flow loss is MSE over per-atom velocity:

- `MSE(v_pred, v_target)` where `v_target = x1 - x0`

To stabilize endpoint geometry during training, the objective also applies a low-weight endpoint consistency term reconstructed from `(x_t, v)` at the sampled `t`.

All existing staged auxiliary losses remain unchanged.

## Inference path

When the active backend family is `flow_matching`, rollout uses ODE-style coordinate integration:

- Euler: `x_{t+dt} = x_t + dt * v_theta(x_t, t)`
- Heun: predictor-corrector variant

The implementation supports low step counts (e.g., 10-50), geometry-only transport, and keeps topology fixed.

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
- `geometry_only` (must stay `true` in v1)
- `use_corrupted_x0`

Example config: `configs/flow_matching_experiment.json`.

## Current limitations

- geometry-only flow (no topology diffusion/flow yet)
- lightweight velocity head and deterministic synthetic noise path (v1 baseline)
- not a full diffusion pipeline
