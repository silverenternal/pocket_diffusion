# Q3 Model Improvement Leaderboard

Only scored raw_flow/raw_rollout/no_repair/raw_geometry rows may support native model quality. Constrained, repaired, and reranked rows are separated and never marked as model improvement.

| Method | Layer | Role | Status | Vina | GNINA affinity | GNINA CNN | QED | SA | Clash | Contact | Model improvement |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TargetDiff | `raw_rollout` | `raw_model_native` | `scored` | -4.138 | -3.925 | 0.3862 | 0.4166 | 2.759 | 0.00245 | 0.968 | false |
| Q3 gated repair ablation | `no_repair` | `raw_model_native` | `scored` | -2.202 | -2.297 | 0.4494 | 0.4325 | 2.242 | 0.002659 | 0.9733 | false |
| Pocket2Mol | `raw_rollout` | `raw_model_native` | `scored` | -2.109 | -2.109 | 0.5881 | 0.4411 | 1.508 | 0 | 0.986 | false |
| DiffSBDD | `raw_rollout` | `raw_model_native` | `scored` | -0.3601 | -0.8585 | 0.3738 | 0.4398 | 2.46 | 0.005526 | 0.9657 | false |
| Ours flow_matching | `raw_flow` | `raw_model_native` | `scored` | 24.94 | 24.94 | 0.1377 | 0.4174 | 2.695 | 0.01547 | 0.9431 | false |
| Q3 coordinate-preserving bond refinement | `raw_geometry` | `raw_model_native` | `scored` | 24.94 | 24.94 | 0.1377 | 0.4174 | 2.695 | 0.01547 | 0.9431 | false |
| Q3 pairwise geometry ablation | `raw_flow` | `raw_model_native` | `scored` | 70.46 | 70.46 | 0.06371 | 0.4174 | 2.695 | 0.07989 | 0.9658 | false |
| Q3 local pocket flow head | `raw_flow` | `raw_model_native` | `scored` | 85.48 | 85.47 | 0.05163 | 0.4174 | 2.695 | 0.09597 | 0.9398 | false |
| Q3 pairwise geometry ablation | `raw_flow` | `raw_model_native` | `scored` | 112.4 | 112.4 | 0.07393 | 0.4174 | 2.695 | 0.1163 | 0.946 | false |
| Q3 pairwise geometry ablation | `raw_flow` | `raw_model_native` | `scored` | 113.6 | 113.6 | 0.04056 | 0.4174 | 2.695 | 0.1286 | 0.9604 | false |
| Q3 pairwise geometry ablation | `raw_flow` | `raw_model_native` | `scored` | 116.9 | 116.9 | 0.0414 | 0.4174 | 2.695 | 0.1356 | 0.9418 | false |
| Ours flow_matching | `constrained_flow` | `constrained_sampling` | `scored` | 79.7 | 79.68 | 0.1123 | 0.4732 | 4.092 | 0.01398 | 0.9918 | false |
| Q3 pairwise geometry ablation | `constrained_flow` | `constrained_sampling` | `scored` | 87.7 | 87.69 | 0.1081 | 0.4336 | 4.113 | 0.01086 | 0.9897 | false |
| Q3 pairwise geometry ablation | `constrained_flow` | `constrained_sampling` | `scored` | 92.67 | 92.65 | 0.1147 | 0.459 | 4.167 | 0.01451 | 0.9928 | false |
| Q3 local pocket flow head | `constrained_flow` | `constrained_sampling` | `scored` | 100.2 | 100.2 | 0.07656 | 0.4689 | 3.989 | 0.01898 | 0.9878 | false |
| Q3 pairwise geometry ablation | `constrained_flow` | `constrained_sampling` | `scored` | 103.8 | 103.8 | 0.07967 | 0.4379 | 3.878 | 0.01853 | 0.9916 | false |
| Q3 pairwise geometry ablation | `constrained_flow` | `constrained_sampling` | `scored` | 126.8 | 126.8 | 0.05057 | 0.4391 | 3.788 | 0.03328 | 0.9903 | false |
| Q3 gated repair ablation | `repair_rejected` | `postprocessing` | `scored` | -2.202 | -2.297 | 0.4494 | 0.4325 | 2.242 | 0.002659 | 0.9733 | false |
| Q3 coordinate-preserving bond refinement | `bond_logits_refined` | `postprocessing` | `scored` | 24.94 | 24.94 | 0.1377 | 0.3677 | 2.731 | 0.01547 | 0.9431 | false |
| Q3 coordinate-preserving bond refinement | `valence_refined` | `postprocessing` | `scored` | 24.94 | 24.94 | 0.1377 | 0.2683 | 3.686 | 0.01547 | 0.9431 | false |
| Pocket2Mol | `repaired` | `postprocessing` | `scored` | 62 | 62 | 0.2801 | 0.4411 | 1.508 | 0.05908 | 0.9474 | false |
| Pocket2Mol | `reranked` | `postprocessing` | `scored` | 62 | 62 | 0.2801 | 0.4428 | 1.518 | 0.05908 | 0.9474 | false |
| Q3 coordinate-preserving bond refinement | `repaired` | `postprocessing` | `scored` | 79.7 | NA | NA | 0.4732 | 4.092 | 0.01398 | 0.9918 | false |
| Q3 gated repair ablation | `full_repair` | `postprocessing` | `scored` | 156.9 | 156.9 | 0.2402 | 0.4315 | 2.284 | 0.1034 | 0.9381 | false |
| DiffSBDD | `repaired` | `postprocessing` | `scored` | 184.1 | 184.1 | 0.2395 | 0.4398 | 2.46 | 0.1199 | 0.9338 | false |
| DiffSBDD | `reranked` | `postprocessing` | `scored` | 184.1 | 184.1 | 0.2395 | 0.4274 | 2.509 | 0.1199 | 0.9338 | false |
| TargetDiff | `repaired` | `postprocessing` | `scored` | 224.6 | 224.6 | 0.2009 | 0.4166 | 2.759 | 0.1311 | 0.9324 | false |
| TargetDiff | `reranked` | `postprocessing` | `scored` | 224.6 | 224.6 | 0.2009 | 0.4241 | 2.825 | 0.1311 | 0.9324 | false |
| Q3 gated repair ablation | `gated_repair` | `postprocessing` | `scored` | NA | NA | NA | NA | NA | NA | NA | false |
| Q3 backend-aware reranker | `backend_aware_posthoc` | `postprocessing` | `scored` | NA | NA | NA | NA | NA | NA | NA | false |
