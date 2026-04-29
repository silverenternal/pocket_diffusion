# Q7 Training Presets

These configs separate regression smoke checks from runs intended to produce
research-facing evidence. All three presets keep the claim-bearing objective on
`conditioned_denoising`; surrogate reconstruction should remain an explicit
debug choice, not a silent default for Q7 model-design runs.

`ResearchConfig::default()` is a compile/test default, not a research preset.
Use the named JSON presets below for any artifact that may appear in a review
bundle, and keep smoke summaries out of claim-bearing tables.

| Config | Intended use | Dataset scope | Budget | Notes |
| --- | --- | --- | --- | --- |
| `configs/q7_smoke_training_preset.json` | Unit and local CI smoke path | Synthetic, capped at 8 examples | 4 steps, batch size 2 | Fast CPU-only validation. Uses mean-pooled decoder conditioning as a cheap baseline and disables comparison rollout. |
| `configs/q7_small_real_debug_training_preset.json` | Real-data parser and trainer debug | Manifest-backed real-data path capped at 32 examples | 200 steps, batch size 4 | Exercises local atom-slot decoder conditioning, probe capacity, staged auxiliaries, real manifest loading, and global gradient-norm clipping without claiming scale. |
| `configs/q7_reviewer_unseen_pocket_training_preset.json` | Reviewer-scale unseen-pocket surface | Manifest-backed real-data path with no `max_examples` cap | 5000 steps, batch size 8 | Uses the upgraded local conditioning path, longer staged warmup, rotation augmentation, candidate comparison, gradient clipping, and pocket/chemistry guardrail losses. |

Smoke output is suitable for compile, config, and wiring regressions only.
Research or reviewer-facing summaries should use the real-data presets, preserve
the unseen-pocket split seed, and keep generated checkpoints under
`checkpoints/` rather than `configs/checkpoints/`.

Runtime summaries expose `forward_batch_count`,
`per_example_forward_count`, and `de_novo_per_example_reason`. A de novo run
with `evaluation_batch_size=1` may intentionally report per-example forwards so
that conditioning boundaries, flow-time context, and target-supervision
separation remain explicit; this is a correctness note, not an efficiency
claim.

For the end-to-end reviewer workflow, including evaluation artifacts, ablations,
claim gates, and failure triage, use
[`docs/q8_reviewer_scale_runbook.md`](q8_reviewer_scale_runbook.md).
