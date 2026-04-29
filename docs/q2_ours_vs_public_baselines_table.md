# Q2 Ours vs Public Baselines

status: `observed_q2_ours_plus_public_full100_layered_score_only`

Scope: 100 official public-test pockets; 1 candidate per pocket per method per layer; unified RDKit, AutoDock Vina score_only, GNINA score_only, and pocket/contact scoring.


## Raw Native Rows

| method | layer | role | pockets | candidates | source | steps | Vina mean | Vina cov | GNINA affinity | GNINA CNN | QED | SA | LogP | TPSA | Lipinski | scaffold novelty | nearest train sim | pocket contact | clash | centroid offset |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffSBDD | raw_rollout | raw_model_native | 100 | 100 | generated_locally_from_public_checkpoint | 1000 | -0.3601 | 0.89 | -0.8585 | 0.3738 | 0.4398 | 2.46 | 5.87 | 190.1 | 0.75 | 1 | 0 | 0.9657 | 0.005526 | NA |
| Ours flow_matching | raw_flow | raw_model_native | 100 | 100 | native_rust_geometry_first_flow_matching | 20 | 24.94 | 1 | 24.94 | 0.1377 | 0.4174 | 2.695 | 5.312 | 209.2 | 0.86 | 1 | 0 | 0.9431 | 0.01547 | 16.44 |
| Pocket2Mol | raw_rollout | raw_model_native | 100 | 100 | official_targetdiff_sampling_results_google_drive | official_public_precomputed_meta | -2.109 | 0.91 | -2.109 | 0.5881 | 0.4411 | 1.508 | 2.086 | 68.88 | 0.08 | 1 | 0 | 0.986 | 0 | NA |
| TargetDiff | raw_rollout | raw_model_native | 100 | 100 | official_targetdiff_sampling_results_google_drive | official_public_precomputed_meta | -4.138 | 0.89 | -3.925 | 0.3862 | 0.4166 | 2.759 | 8.538 | 185.4 | 1.03 | 1 | 0 | 0.968 | 0.00245 | NA |

## Constrained And Postprocessed Rows

| method | layer | role | pockets | candidates | source | steps | Vina mean | Vina cov | GNINA affinity | GNINA CNN | QED | SA | LogP | TPSA | Lipinski | scaffold novelty | nearest train sim | pocket contact | clash | centroid offset |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffSBDD | repaired | postprocessing | 100 | 100 | generated_locally_from_public_checkpoint | 1000 | 184.1 | 1 | 184.1 | 0.2395 | 0.4398 | 2.46 | 5.87 | 190.1 | 0.75 | 1 | 0 | 0.9338 | 0.1199 | NA |
| DiffSBDD | reranked | postprocessing | 100 | 100 | generated_locally_from_public_checkpoint | 1000 | 184.1 | 1 | 184.1 | 0.2395 | 0.4274 | 2.509 | 4.971 | 176.8 | 0.75 | 1 | 0 | 0.9338 | 0.1199 | NA |
| Ours flow_matching | constrained_flow | constrained_sampling | 100 | 100 | native_rust_geometry_first_flow_matching | 20 | 79.7 | 1 | 79.68 | 0.1123 | 0.4732 | 4.092 | 6.928 | 0 | 0.82 | 1 | 0 | 0.9918 | 0.01398 | 15.82 |
| Pocket2Mol | repaired | postprocessing | 100 | 100 | official_targetdiff_sampling_results_google_drive | official_public_precomputed_meta | 62 | 1 | 62 | 0.2801 | 0.4411 | 1.508 | 2.086 | 68.88 | 0.08 | 1 | 0 | 0.9474 | 0.05908 | NA |
| Pocket2Mol | reranked | postprocessing | 100 | 100 | official_targetdiff_sampling_results_google_drive | official_public_precomputed_meta | 62 | 1 | 62 | 0.2801 | 0.4428 | 1.518 | 1.861 | 65.89 | 0.06 | 1 | 0 | 0.9474 | 0.05908 | NA |
| TargetDiff | repaired | postprocessing | 100 | 100 | official_targetdiff_sampling_results_google_drive | official_public_precomputed_meta | 224.6 | 1 | 224.6 | 0.2009 | 0.4166 | 2.759 | 8.538 | 185.4 | 1.03 | 1 | 0 | 0.9324 | 0.1311 | NA |
| TargetDiff | reranked | postprocessing | 100 | 100 | official_targetdiff_sampling_results_google_drive | official_public_precomputed_meta | 224.6 | 1 | 224.6 | 0.2009 | 0.4241 | 2.825 | 7.534 | 172.7 | 0.95 | 1 | 0 | 0.9324 | 0.1311 | NA |

Guardrail: Vina and GNINA values are score_only backend outputs, not experimental binding affinities. Raw native rows are separated from constrained sampling, repaired, and reranked rows.
