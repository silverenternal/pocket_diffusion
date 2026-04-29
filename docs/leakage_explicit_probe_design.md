# Explicit Leakage Probe Design

Current training uses a **similarity-margin leakage proxy** in `src/losses/leakage.rs`.
That proxy is fast and stable, but it is a diagnostic rather than a semantic proof:

- It measures aggregate slot similarity across modalities.
- It does not establish that one branch cannot predict off-modality targets.
- It can still pass when a branch encodes off-modality factors but those factors are not strongly aligned in cosine space.
- It is useful for ablation baselines and lightweight monitoring, not for strict leakage guarantees.

The role-separated protocol is maintained in
[`leakage_probe_protocol.md`](leakage_probe_protocol.md).

## Explicit objective

To move beyond the similarity proxy, the codebase now supports explicit
off-modality leakage routes and a separate frozen-representation calibration
path. The optimizer-facing route remains configurable; held-out calibration is
diagnostic and separate from the main optimizer.

### Target set

The explicit probe targets are defined in `src/losses/probe.rs` as:

- `topology -> geometry`
- `topology -> pocket`
- `geometry -> topology`
- `geometry -> pocket`
- `pocket -> topology`
- `pocket -> geometry`

Each target is controlled by a dedicated configuration flag under
`training.explicit_leakage_probes`.

### Current state

- Configuration is present and off by default:
  `enable_explicit_probes: false`, all target flags false.
- The default behavior still includes the existing similarity-margin
  `LeakageLoss` as a lightweight proxy.
- Explicit routes are reported separately when enabled.
- Frozen calibration is available through the Q11 smoke path documented in
  `docs/q11_leakage_probe_calibration_smoke.md`; it fits separate ridge probes
  on frozen pooled modality embeddings and reports held-out predictability
  distinct from training-time leakage penalties.
- Training and evaluation artifacts should keep `optimizer_penalty`,
  `detached_training_diagnostic`, and `frozen_probe_audit` sections separate.
