# Q1 Runtime Efficiency Table

status: `q3_backend_aware_posthoc_reranker`

| method | candidates | generation runtime seconds | generation runtime status | backend scoring runtime seconds | backend scoring runtime status |
|---|---:|---:|---|---:|---|
| DiffSBDD | 300 | NA | missing | NA | not_measured_in_backend_scripts |
| Pocket2Mol | 300 | NA | not_available_official_precomputed_meta | NA | not_measured_in_backend_scripts |
| TargetDiff | 300 | NA | not_available_official_precomputed_meta | NA | not_measured_in_backend_scripts |

Backend scoring runtime is measured once for the shared layered Vina+GNINA rescoring batch; the same measured backend batch supports all method rows.
