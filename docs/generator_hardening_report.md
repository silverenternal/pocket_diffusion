# Generator Hardening Report

This file is generated from `docs/evidence_bundle.json` and should be refreshed through the reviewer revalidation path.

## Direction

- Current direction: `continue_incremental_hardening`.
- Saturation status: `not_yet_evidenced`.
- Primary justification surface: `checkpoints/pdbbindpp_real_backends`.
- Stability surface: `configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json`.
- Standalone decision artifact: `checkpoints/generator_decision/generator_decision.json`.
- Major model change gate: Only revisit major objective changes such as diffusion after larger-data held-out-family artifacts show plateaued quality and acceptable stability without simpler incremental gains.
- Freshness gate: `python3 tools/generator_decision_bundle.py --check` fails if the persisted decision artifact is stale relative to the canonical larger-data surface, tight-geometry pressure surface, larger-data multi-seed summary, or promotion-relevant rollout/objective files.

## Quality And Efficiency Tradeoffs

| Surface | Pocket fit | Leakage | Test eps | Test memory MB | Chemistry tier |
| --- | --- | --- | --- | --- | --- |
| compact_regression | 0.4259 | 0.0723 | 226.1046 | 0.0000 | proxy_only |
| real_backend_gate | 0.5805 | 0.0778 | 11.1560 | 0.0000 | local_benchmark_style |
| larger_data_canonical | 0.6752 | 0.0506 | 60.7536 | 0.0000 | external_benchmark_backed |
| larger_data_lp_pdbbind_refined | 0.7599 | 0.0569 | 62.2725 | 0.0000 | external_benchmark_backed |
| tight_geometry_pressure | 0.6178 | 0.0613 | 11.2493 | 0.0000 | local_benchmark_style |

## Decision Notes

- Use larger held-out-family evidence as the primary justification surface for generator changes, not compact-only wins.
- Keep the current conditioned-denoising plus bounded reranking path as the default while larger-data and tight-geometry surfaces still show non-trivial quality/efficiency tradeoffs.
- Larger-data multi-seed strict pocket fit still varies materially across persisted seeds, which indicates headroom for incremental hardening before major objective replacement.
- Larger-data multi-seed leakage still spans a meaningful range, so robustness work should target calibration and stability before architectural resets.
- Compact regression is the fast gate, but larger-data canonical review is the quality-bearing throughput anchor.
- Tight geometry improves strict pocket fit over the canonical larger-data surface at materially lower throughput, so geometry pressure remains a real quality-vs-speed tradeoff instead of a free win.
- The LP-PDBBind refined reviewer surface now gives a second larger-data external-benchmark check under the same artifact contract, so chemistry wording no longer depends on a single benchmark anchor.
- The repository real-backend gate is slower than the larger-data canonical surface and carries weaker leakage behavior, so compact or backend-gate wins alone are not enough to justify generator promotion.

## Promotion Artifact

- Refresh `checkpoints/generator_decision/generator_decision.json` through the reviewer revalidation path before promoting major generator changes.
