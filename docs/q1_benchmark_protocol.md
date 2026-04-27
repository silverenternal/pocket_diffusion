# Q1 Primary Benchmark Protocol

Benchmark: `q1_primary_pdbbindpp_unseen_pocket_v1`.

The split is locked by `protein_id` with `split_seed=42`, `val_fraction=0.15`, `test_fraction=0.15`, measurement stratification enabled, and the PDBBind++ manifest and label-table SHA256 hashes recorded in `configs/q1_primary_benchmark_manifest.json`.

| split | examples | unique proteins | labeled fraction |
| --- | ---: | ---: | ---: |
| train | 358 | 358 | 1.0000 |
| val | 77 | 77 | 1.0000 |
| test | 77 | 77 | 0.7792 |

Protein overlap detected: `false`. Duplicate example ids detected: `false`.

Candidate-level scaffold/ligand overlap proxy statistics are stored in `configs/q1_primary_split_report.json`; split-level source-ligand scaffold IDs are not persisted in the legacy split report, so those proxy fields are labeled accordingly.
