# Artifact Boundary Audit

Created: 2026-04-28

This audit separates source configuration, generated checkpoints, generated evidence,
generated reports, and curated reviewer artifacts. It is intentionally conservative:
no existing tracked artifacts are deleted by this audit. The follow-up policy task
should move future generated output away from source config paths and only untrack
existing files after a narrow migration plan is accepted.

## Current Inventory

| Location | Current size | Git tracking state | Classification | Policy |
| --- | ---: | --- | --- | --- |
| `configs/*.json` | 172 top-level JSON files present, 97 tracked | Mixed tracked and untracked source-like configs plus generated plan/report configs | Source configs and curated experiment manifests | Keep small hand-authored or curated configs tracked; generated summaries need explicit manifest entries. |
| `configs/checkpoints/` | 1.7G total | 1735 tracked JSON files; 328 ignored `.ot` files | Generated checkpoint metadata and reviewer smoke evidence mixed into config tree | Freeze as legacy reviewer artifact input; do not add new generated outputs here. Migrate future outputs to `checkpoints/` or curated `artifacts/evidence/`. |
| `checkpoints/` | 1.6G total | Ignored by `/checkpoints/` and `*.ot`; 0 tracked files | Generated checkpoints, generated summaries, docking inputs, and local run outputs | Default destination for reproducible local/generated run output. Keep ignored. |
| `artifacts/evidence/` | 956K total | 0 tracked files currently; 3 local evidence files present | Curated evidence candidate location | Track only curated, size-bounded evidence referenced by a manifest. Keep raw/generated bulk output out. |
| `docs/` | 664K total | 41 tracked files plus local generated reports | Curated documentation mixed with generated reports | Hand-maintained docs stay tracked; generated reports should include markers and allow temporary output paths. |

## Largest Artifact Families

`configs/checkpoints/` is the main source/artifact boundary risk:

| Family | Size | Tracked files | Current role | Retention recommendation |
| --- | ---: | ---: | --- | --- |
| `configs/checkpoints/automated_search/` | 1.5G | 1513 | Generated search candidates and ablation summaries for tight geometry pressure. | Treat as legacy reviewer evidence only. Keep temporarily for reproducibility, then migrate selected summaries to `artifacts/evidence/` and move regenerated outputs under ignored `checkpoints/`. |
| `configs/checkpoints/flow_sweep_search/` | 72M | 61 | Generated flow sweep candidates and summaries. | Keep only curated summaries if still referenced; generated reruns should write to ignored `checkpoints/flow_sweep_search/`. |
| `configs/checkpoints/multi_seed_pdbbindpp_real_backends/` | 43M | 34 | Multi-seed real-backend reviewer evidence metadata. | Candidate for curated evidence retention because backend coverage is expensive to reproduce; migrate selected JSON summaries to `artifacts/evidence/`. |
| `configs/checkpoints/multi_seed_pdbbindpp_flow/` | 42M | 34 | Multi-seed flow evidence metadata. | Keep as curated evidence only if referenced by claim docs; otherwise regenerate under ignored checkpoints. |
| `configs/checkpoints/multi_seed_pdbbindpp/` | 35M | 31 | Multi-seed PDBBind++ run metadata. | Same migration recommendation as above. |
| `configs/checkpoints/multi_seed_medium/` | 31M | 31 | Medium-profile multi-seed run metadata. | Same migration recommendation as above. |
| `configs/checkpoints/multi_seed/` | 31M | 31 | Small multi-seed run metadata. | Same migration recommendation as above. |

Root `checkpoints/` is already ignored and contains larger local generated families,
including `pdbbindpp_real_backends` (247M), `q3_pairwise_geometry_public100`
(204M), `q3_best_variant_multiseed` (147M), `flow_ablation_matrix` (145M),
`harder_pressure` (102M), and `claim_matrix` (102M). These should remain local
outputs unless a narrow subset is curated into `artifacts/evidence/`.

## File-Type Findings

- All tracked files under `configs/checkpoints/` are JSON metadata or summaries.
- `.ot` model weights are ignored by `*.ot`, including `.ot` files that sit under
  `configs/checkpoints/`.
- Root-level `checkpoints/` is ignored, so generated JSON, JSONL, docking inputs,
  and model weights under that root do not enter source review by default.
- `artifacts/evidence/` currently contains local Q2 evidence files but no tracked
  evidence manifest governs what should be retained.
- Several docs and configs are generated reports or report inputs; ownership is not
  consistently marked yet.

## Classification

### Source Config

Source configs are small, hand-authored, or intentionally curated JSON files that
define an experiment, manifest, protocol, or validation contract. They belong under
`configs/` and should stay tracked when they are stable inputs rather than run
outputs.

Examples:

- `configs/research_manifest.json`
- `configs/unseen_pocket_manifest.json`
- `configs/*_protocol.json`
- `configs/*_contract.json`
- curated Q2/Q3/Q5 ablation configs

### Generated Checkpoint

Generated checkpoints include model weights, step metadata, latest pointers, run
artifacts, split reports, and generated summaries produced by training or
evaluation runs. The default destination should be ignored `checkpoints/`.

Examples:

- `checkpoints/**`
- `*.ot`
- `configs/checkpoints/**/step-0.json`
- `configs/checkpoints/**/latest.json`
- `configs/checkpoints/**/run_artifacts.json`
- `configs/checkpoints/**/config.snapshot.json`

### Generated Evidence

Generated evidence is output that can support an evaluation claim but should not be
treated as source config. It may be curated into `artifacts/evidence/` when it is
small enough, has stable provenance, and is referenced from a manifest.

Examples:

- candidate metric JSONL files
- backend coverage summaries
- generation layer records
- postprocessing failure fixtures

### Generated Report

Generated reports are Markdown or JSON files emitted by tools. They should be
clearly marked and should support temporary output paths so validation does not
dirty curated docs during normal development.

Examples:

- `docs/validation_suite_report.md`
- `configs/validation_suite_report.json`
- generated Q1/Q2/Q3 report Markdown

### Curated Reviewer Artifact

Curated reviewer artifacts are selected generated outputs retained to make a
reviewer-facing claim auditable. These should be listed in a manifest, kept
size-bounded, and separated from raw checkpoint trees.

Examples:

- selected claim summaries
- selected backend-supported metric tables
- selected public-baseline comparison summaries
- selected layer-attribution records

## Tracking Policy

1. New generated model weights and checkpoint metadata should default to
   `checkpoints/`, which is already ignored.
2. New curated evidence should live under `artifacts/evidence/<family>/` and must
   be listed in `configs/artifact_retention_manifest.json`.
3. New generated reports should either be written to temporary paths or be clearly
   labeled as generated when committed as reviewer artifacts.
4. No new generated checkpoint tree should be added under `configs/checkpoints/`.
5. Existing tracked `configs/checkpoints` JSON families should remain untouched
   until a migration PR selects the summaries that are still needed for claim
   reproducibility.

## Immediate Follow-Up Recommendations

- Update path helpers so default generated output roots point at ignored
  `checkpoints/`, not `configs/checkpoints/`.
- Add validation coverage that reads curated evidence from
  `configs/artifact_retention_manifest.json`.
- Add generated-report markers and overridable validation report paths.
- Decide which `configs/checkpoints` families are still claim-bearing, then migrate
  only those summaries into `artifacts/evidence/`.
