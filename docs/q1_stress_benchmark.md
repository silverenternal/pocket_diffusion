# Q1 Stress Benchmark

The stress subset was materialized by `tools/stress_benchmark.py` from candidate-level artifacts.
Selection is mechanical: an example must satisfy the configured minimum number of stress rules.

- status: materialized
- selected_examples: 6
- selected_candidate_rows: 66
- quantile: 0.15
- min_rules: 2

## Selected Examples

| Example | Pocket Atoms | Ligand Atoms | Rules |
| --- | ---: | ---: | --- |
| `1aaq` | 552 | 89 | low_homology_proxy, ligand_atom_count_extreme |
| `1bnq` | 505 | 42 | low_homology_proxy, pocket_atom_count_extreme |
| `1c1r` | 453 | 34 | low_homology_proxy, pocket_atom_count_extreme, ligand_atom_count_extreme |
| `1g2k` | 624 | 82 | low_homology_proxy, pocket_atom_count_extreme, ligand_atom_count_extreme |
| `1g53` | 521 | 34 | low_homology_proxy, ligand_atom_count_extreme |
| `1h22` | 740 | 78 | low_homology_proxy, pocket_atom_count_extreme |

## Layer Metrics

| Layer | Rows | Examples | Vina Mean | GNINA Mean | QED Mean | Clash Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deterministic_proxy` | 6 | 6 | 143.8828 | 143.9094 | 0.2252 | 0.0511 |
| `inferred_bond` | 18 | 6 | 152.2499 | 152.2380 | 0.1870 | 0.0553 |
| `raw_rollout` | 18 | 6 | 51.8108 | 51.8700 | 0.2971 | 0.0305 |
| `repaired` | 18 | 6 | 73.1304 | 73.1386 | 0.2971 | 0.0553 |
| `reranked` | 6 | 6 | 157.2793 | 157.2532 | 0.1943 | 0.0647 |

## Guardrail

Stress results are reported separately from the primary benchmark and include all candidate rows for selected examples, including failed or low-quality candidates present in the source metrics.
