# Paper-Facing Claim Bundle

This file is generated from canonical reviewer artifacts. Update it through `./tools/revalidate_reviewer_bundle.sh` rather than manual editing.

## Claim Map

| Claim | Canonical artifact | Current support |
| --- | --- | --- |
| Held-out-pocket chemistry/generalization on the canonical benchmark surface | `checkpoints/pdbbindpp_real_backends` | `benchmark_evidence.evidence_tier=external_benchmark_backed`, `strict_pocket_fit_score=0.6752`, `leakage_proxy_mean=0.0506` |
| Second larger-data benchmark surface under the same reviewer policy | `checkpoints/lp_pdbbind_refined_real_backends` | `benchmark_evidence.evidence_tier=external_benchmark_backed`, `strict_pocket_fit_score=0.7599`, `leakage_proxy_mean=0.0569` |
| Additional pressure-surface chemistry/generalization evidence beyond the canonical benchmark path | `checkpoints/tight_geometry_pressure` | `benchmark_evidence.evidence_tier=local_benchmark_style`, `strict_pocket_fit_score=0.6178`, `clash_fraction=0.0000` |
| Repository-supported backend gate | `checkpoints/real_backends` | `rdkit_sanitized_fraction=1.0000`, `strict_pocket_fit_score=0.5805` |
| Seed stability for the larger-data reviewer path | `configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json` | `seed_count=3`, `stability_decision=stable enough for larger-data real-backend claim review on the held-out pocket surface` |
| Stronger backend companion profile | `checkpoints/vina_backend` | `reviewer_status=fail`, `docking_input_completeness_fraction=0.0000`, `docking_score_coverage_fraction=0.0000`, `rdkit_sanitized_fraction=1.0000`, `backend_missing_structure_fraction=0.0000` |

## Main Results Table

| Surface | Evidence tier | Pocket fit | Leakage | Clash | Test eps | Candidate valid |
| --- | --- | --- | --- | --- | --- | --- |
| Canonical larger-data | external_benchmark_backed | 0.6752 | 0.0506 | 0.0000 | 60.7536 | 1.0000 |
| LP-PDBBind refined larger-data | external_benchmark_backed | 0.7599 | 0.0569 | 0.0000 | 62.2725 | 1.0000 |
| Tight geometry pressure | local_benchmark_style | 0.6178 | 0.0613 | 0.0000 | 11.2493 | 1.0000 |
| Real-backend gate | n/a | 0.5805 | 0.0778 | 0.0000 | 11.1560 | 1.0000 |

## Benchmark Breadth Table

| Surface | Evidence tier | Pocket fit | Leakage |
| --- | --- | --- | --- |
| checkpoints/claim_matrix | proxy_only | 0.4259 | 0.0723 |
| checkpoints/tight_geometry_pressure | local_benchmark_style | 0.6178 | 0.0613 |
| checkpoints/real_backends | local_benchmark_style | 0.5805 | 0.0778 |
| checkpoints/lp_pdbbind_refined_real_backends | external_benchmark_backed | 0.7599 | 0.0569 |
| checkpoints/pdbbindpp_real_backends | external_benchmark_backed | 0.6752 | 0.0506 |
| checkpoints/vina_backend | local_benchmark_style | 0.3566 | 0.0385 |

## Ablation Table

| Variant | Test pocket fit | Test leakage | Test valid fraction |
| --- | --- | --- | --- |
| objective_surrogate | 0.4940 | 0.0873 | 1.0000 |
| disable_slots | 0.4561 | 0.0506 | 1.0000 |
| disable_cross_attention | 0.5252 | 0.0406 | 1.0000 |
| disable_geometry_interaction_bias | 0.5848 | 0.0729 | 1.0000 |
| disable_rollout_pocket_guidance | 0.3277 | 0.0795 | 1.0000 |
| disable_candidate_repair | 0.3469 | 0.0367 | 1.0000 |
| interaction_lightweight | 0.7081 | 0.0662 | 1.0000 |

## Efficiency Summary Table

| Surface | Test eps | Relative throughput | Pocket fit | Leakage |
| --- | --- | --- | --- | --- |
| compact_regression | 226.1046 | 3.7217x | 0.4259 | 0.0723 |
| real_backend_gate | 11.1560 | 0.1836x | 0.5805 | 0.0778 |
| larger_data_canonical | 60.7536 | 1.0000x | 0.6752 | 0.0506 |
| larger_data_lp_pdbbind_refined | 62.2725 | 1.0250x | 0.7599 | 0.0569 |
| tight_geometry_pressure | 11.2493 | 0.1852x | 0.6178 | 0.0613 |

## Minimum External Communication Set

- Main results: `checkpoints/pdbbindpp_real_backends`, `checkpoints/lp_pdbbind_refined_real_backends`, and `checkpoints/tight_geometry_pressure`.
- Benchmark breadth: The reviewer bundle now carries 6 persisted chemistry/generalization surface(s), including 2 external-benchmark-backed larger-data surface(s) across 2 benchmark dataset(s): lp_pdbbind_refined, pdbbindpp-2020.
- Ablations and leakage review: `checkpoints/claim_matrix/ablation_matrix_summary.json` together with the leakage review sections embedded in the canonical claim summaries.
- Replay policy: bounded replay with explicit drift tolerances; not strict optimizer-state-identical replay Promotion uses bounded replay drift reports, not strict optimizer-state-identical replay.
- Efficiency/stability: cite the larger-data multi-seed summary and the per-surface performance gates in the claim summaries.
- Stronger backend companion: `checkpoints/vina_backend` now records explicit reviewer pass/fail status, Vina availability, input completeness, and docking score coverage, so claim-bearing backend wording no longer relies on reviewer interpretation of partial failures.
- Generator direction: `continue_incremental_hardening` with saturation status `not_yet_evidenced`; do not justify major objective changes from compact-only wins.

## Residual Caveats

- Compact artifacts are smoke/regression evidence, not broad scientific generalization proof.
- The local medium profile currently contains only a five-complex parser smoke surface and is not the designated reviewer data path.
- Real-backend reviewer evidence is now anchored to a documented larger-data surface, but it still depends on the configured external backend commands being available.
- The reviewer bundle now carries explicit external benchmark-dataset coverage on multiple larger-data surfaces, but it remains a local held-out-pocket reviewer package rather than a broad publication-scale benchmark campaign.
- Tight geometry now clears both the clash gate and the leakage reviewer gate on the canonical surface.

## Generator Direction

- Standalone decision artifact: `checkpoints/generator_decision/generator_decision.json`.
- Major-model-change gate: Only revisit major objective changes such as diffusion after larger-data held-out-family artifacts show plateaued quality and acceptable stability without simpler incremental gains.
- Held-out-family direction notes are also summarized in `docs/generator_hardening_report.md`.
- Efficiency tradeoff summary: Compact regression is the fast gate, but larger-data canonical review is the quality-bearing throughput anchor.; Tight geometry improves strict pocket fit over the canonical larger-data surface at materially lower throughput, so geometry pressure remains a real quality-vs-speed tradeoff instead of a free win.; The LP-PDBBind refined reviewer surface now gives a second larger-data external-benchmark check under the same artifact contract, so chemistry wording no longer depends on a single benchmark anchor.; The repository real-backend gate is slower than the larger-data canonical surface and carries weaker leakage behavior, so compact or backend-gate wins alone are not enough to justify generator promotion.

## Refresh Contract

- Canonical entrypoint: `./tools/revalidate_reviewer_bundle.sh`.
- Replay decision: Canonical reviewer replay is treated as bounded deterministic rerun within explicit metric tolerances, not bitwise or optimizer-state-identical strict replay.
- Reviewer refresh artifact: `docs/reviewer_refresh_report.json`.

