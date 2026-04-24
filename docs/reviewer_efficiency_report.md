# Reviewer Efficiency Report

This file is generated from canonical reviewer artifacts and should be refreshed through `./tools/revalidate_reviewer_bundle.sh`.

## Surface Tradeoffs

| Surface | Test eps | Rel throughput vs larger-data | Test memory MB | Pocket fit | Leakage | Candidate valid | Chemistry tier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_regression | 226.1046 | 3.7217x | 0.0000 | 0.4259 | 0.0723 | 1.0000 | proxy_only |
| real_backend_gate | 11.1560 | 0.1836x | 0.0000 | 0.5805 | 0.0778 | 1.0000 | local_benchmark_style |
| larger_data_canonical | 60.7536 | 1.0000x | 0.0000 | 0.6752 | 0.0506 | 1.0000 | external_benchmark_backed |
| larger_data_lp_pdbbind_refined | 62.2725 | 1.0250x | 0.0000 | 0.7599 | 0.0569 | 1.0000 | external_benchmark_backed |
| tight_geometry_pressure | 11.2493 | 0.1852x | 0.0000 | 0.6178 | 0.0613 | 1.0000 | local_benchmark_style |

## Larger-Data Seed Stability

| Aggregate | Mean | Std | Min | Max | 95% CI low | 95% CI high |
| --- | --- | --- | --- | --- | --- | --- |
| test_examples_per_second | 65.8136 | 2.8013 | 62.0050 | 68.6623 | 57.2901 | 74.3371 |
| strict_pocket_fit_score | 0.6517 | 0.0710 | 0.5554 | 0.7246 | 0.4356 | 0.8677 |
| leakage_proxy_mean | 0.0719 | 0.0286 | 0.0360 | 0.1060 | -0.0151 | 0.1588 |

## Per-Seed Throughput

| Seed | Test eps | Pocket fit | Leakage | Gate activation | Slot activation |
| --- | --- | --- | --- | --- | --- |
| 17 | 68.6623 | 0.5554 | 0.0736 | 0.1672 | 0.6710 |
| 42 | 62.0050 | 0.7246 | 0.1060 | 0.1749 | 0.5887 |
| 101 | 66.7734 | 0.6749 | 0.0360 | 0.1397 | 0.5714 |

## Interpretation

- Compact regression is the fast gate, but larger-data canonical review is the quality-bearing throughput anchor.
- Tight geometry improves strict pocket fit over the canonical larger-data surface at materially lower throughput, so geometry pressure remains a real quality-vs-speed tradeoff instead of a free win.
- The LP-PDBBind refined reviewer surface now gives a second larger-data external-benchmark check under the same artifact contract, so chemistry wording no longer depends on a single benchmark anchor.
- The repository real-backend gate is slower than the larger-data canonical surface and carries weaker leakage behavior, so compact or backend-gate wins alone are not enough to justify generator promotion.
- Larger-data throughput, pocket fit, and leakage are now reported both as single-surface reviewer gates and as multi-seed aggregates so generator changes can be judged on quality-resource tradeoffs rather than isolated snapshots.
