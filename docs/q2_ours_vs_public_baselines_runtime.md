# Q1 Runtime Efficiency Table

status: `observed_q2_ours_plus_public_full100_layered_score_only`

| method | candidates | generation runtime seconds | generation runtime status | backend scoring runtime seconds | backend scoring runtime status |
|---|---:|---:|---|---:|---|
| DiffSBDD | 300 | 4638 | measured_local_public_checkpoint | 348.4 | measured_by_backend_adapters |
| Pocket2Mol | 300 | NA | not_available_official_precomputed_meta | 348.4 | measured_by_backend_adapters |
| TargetDiff | 300 | NA | not_available_official_precomputed_meta | 348.4 | measured_by_backend_adapters |

Backend scoring runtime is measured once for the shared layered Vina+GNINA rescoring batch; the same measured backend batch supports all method rows.
