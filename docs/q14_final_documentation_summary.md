# Q14 Final Documentation Summary

Date: 2026-04-29

This summary records the documentation updates made after the Q14 hardening
work and compact final smoke.

## Updated Method Boundaries

- `docs/architecture_module_map.md` now describes the executable de novo full
  molecular-flow path instead of treating it as future-only. It also points to
  the Q14 branch schedule, pocket-flow branch, and raw-vs-processed evidence
  contracts.
- `docs/q14_active_path_review.md` now records the post-hardening active path
  and links the Q14 final smoke evidence instead of describing the review as a
  pre-hardening freeze.
- `docs/generation_objective_boundary.md` now requires full branch provenance,
  non-index target matching, raw-native layer metrics, dataset leakage review,
  and raw-vs-processed attribution before de novo molecular-flow wording is
  treated as claim-facing.
- `docs/q14_leakage_training_contract.md` now reflects the implemented leakage
  route semantics: `detached_diagnostic`, `adversarial_penalty`, `probe_fit`,
  `encoder_penalty`, and `alternating`. It keeps optimizer penalties separate
  from frozen held-out leakage audits.
- `docs/claim_readiness.md` now explicitly labels
  `checkpoints/q14_final_smoke` as synthetic compact smoke evidence only.

## Updated User-Facing Method Document

- `docs/dataset_training_methods_zh.md` now points to the Q14 final smoke
  configs and artifacts, states the `hungarian_distance` de novo matching
  provenance, and keeps repaired-layer improvements out of raw generation
  evidence.
- Generated formats were refreshed from the Markdown with `pandoc`:
  `docs/dataset_training_methods_zh.html`,
  `docs/dataset_training_methods_zh.docx`, and
  `docs/dataset_training_methods_zh.odt`.

## Evidence Summaries

- `docs/q14_final_smoke_summary.md` summarizes the three compact smoke surfaces:
  conditioned denoising, geometry flow, and de novo full flow.
- The smoke artifact root is `checkpoints/q14_final_smoke`.
- Every surface includes `training_summary.json`, `experiment_summary.json`,
  `claim_summary.json`, `generation_layers_validation.json`,
  `generation_layers_test.json`, repair-case audits, and frozen leakage audits.
- All three smokes recorded validation/test `finite_forward_fraction = 1.0` and
  total nonfinite gradient tensors `0`.
- The de novo full-flow smoke records `target_alignment_policy =
  hungarian_distance` and `target_matching_claim_safe = true`.

## Claim Boundary

The Q14 final smoke is runnable-path and artifact-contract evidence. It must
not be cited as broad unseen-pocket generalization or benchmark-quality de novo
molecular generation. Strong wording still requires larger held-out-pocket,
multi-seed, backend-covered evidence with raw-native metrics leading processed
or repaired layers.
