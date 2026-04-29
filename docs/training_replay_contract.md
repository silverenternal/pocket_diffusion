# Training Replay Contract

Training artifacts use two compatibility levels.

`strict_replay_compatible` means the training summary and checkpoint metadata
agree on config hash, dataset fingerprint, split seed, corruption seed,
sampling seed, sampler configuration, runtime controls, and optimizer internal
state. This requires `optimizer_exact_resume`; current `tch` optimizer support
does not persist Adam moment buffers, so normal checkpoints should not claim
this level.

Current Rust checkpoints restore model weights, completed step, prior metrics,
scheduler metadata, and optimizer hyperparameters. Adam moment buffers are
metadata-only under `metadata_only_tch_0_23`; runs requiring exact optimizer
continuation must fail unless a future checkpoint advertises
`optimizer_exact_resume` with `internal_state_persisted=true`.

`evidence_compatible` means claim-bearing evidence can be compared because the
effective config, dataset fingerprint, and data-order identity fields match.
Weights-only checkpoints can be evidence-compatible while still failing strict
replay. Resume-mode and optimizer-internal-state mismatches are therefore
reported as replay-blocking but not evidence-blocking.

Evidence compatibility is blocked by mismatches in:

- `config_hash`
- `dataset_validation_fingerprint`
- `metric_schema_version`
- `determinism_controls.split_seed`
- `determinism_controls.corruption_seed`
- `determinism_controls.sampling_seed`
- `determinism_controls.generation_mode`
- `determinism_controls.generation_corruption_seed`
- `determinism_controls.generation_sampling_seed`
- `determinism_controls.flow_contract_version`
- `determinism_controls.flow_branch_schedule_hash`
- `determinism_controls.batch_size`
- `determinism_controls.sampler_shuffle`
- `determinism_controls.sampler_seed`
- `determinism_controls.sampler_drop_last`
- `determinism_controls.sampler_max_epochs`

Checkpoint resume additionally rejects backend/objective replay-contract
mismatches before training continues:

- active generation backend id and family
- primary objective name
- generation mode
- molecular flow contract version
- multi-modal flow branch schedule hash
- raw-versus-processed evaluation contract label

Use the CLI check before promoting a resumed run as comparable evidence:

```bash
cargo run --bin pocket_diffusion -- replay-check \
  --summary checkpoints/training_summary.json \
  --checkpoint checkpoints/latest.json
```

Add `--require-strict-replay` only when the comparison requires exact optimizer
continuation rather than evidence-level compatibility.
