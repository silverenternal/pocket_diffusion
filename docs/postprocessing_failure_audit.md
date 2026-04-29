# Postprocessing Failure Audit

Repaired and reranked Q1 docking values are postprocessing evidence and must not be used as native model evidence.

## Method Summary

| method | raw Vina | repaired Vina | Δ Vina | raw GNINA | repaired GNINA | Δ GNINA | RMSD | centroid shift | raw contact | repaired contact | repaired large-positive Vina |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pocket2mol_public | -2.109 | 62 | 66.44 | -2.109 | 62 | 64.11 | 23.57 | 22.5 | 0.986 | 0.9474 | 0.89 |
| targetdiff_public | -4.138 | 224.6 | 231.7 | -3.925 | 224.6 | 228.5 | 21.9 | 20.11 | 0.968 | 0.9324 | 0.95 |
| diffsbdd_public | -0.3601 | 184.1 | 192.8 | -0.8585 | 184.1 | 184.9 | 21.95 | 20.25 | 0.9657 | 0.9338 | 0.9 |

## Interpretation

All three public baselines show repaired/reranked Vina and GNINA affinity degradation to large positive score_only values. The most likely issue is the shared postprocessing or docking-input preparation path, not a native-model improvement signal.

Most likely failure modes:
- `pocket2mol_public`: coordinate_movement_or_frame_sensitive_repair (large repaired-minus-raw docking degradation with nontrivial centroid/RMSD displacement).
- `targetdiff_public`: coordinate_movement_or_frame_sensitive_repair (large repaired-minus-raw docking degradation with nontrivial centroid/RMSD displacement).
- `diffsbdd_public`: coordinate_movement_or_frame_sensitive_repair (large repaired-minus-raw docking degradation with nontrivial centroid/RMSD displacement).

Guardrail: repaired/reranked Q1 docking values must not be promoted as raw native model evidence.
