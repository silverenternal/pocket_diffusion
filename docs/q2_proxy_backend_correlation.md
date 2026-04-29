# Correlation Plot

Source table: `artifacts/evidence/q2/q2_proxy_backend_correlation.json`
Record count: 1100
Minimum interpretable samples: 20

This artifact is rendered from `correlation_table.json`; unsupported entries indicate missing backend coverage or too few candidate-level samples.

## Pocket Fit Vs Docking

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | unavailable | unavailable | 0 | 0 | unsupported | unsupported | missing_pair |

## Geometry Vs QED

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | clash_fraction | qed | 1100 | 0 | 0.0249 | 0.0058 | interpretable |
| raw_layers | clash_fraction | qed | 400 | 0 | -0.0631 | -0.0520 | interpretable |
| postprocessed_layers | clash_fraction | qed | 600 | 0 | -0.0884 | -0.0740 | interpretable |
| layer:constrained_flow | mean_centroid_offset | qed | 100 | 0 | 0.0212 | 0.0211 | interpretable |
| layer:raw_flow | mean_centroid_offset | qed | 100 | 0 | -0.0644 | 0.0475 | interpretable |
| layer:raw_rollout | clash_fraction | qed | 300 | 0 | -0.0471 | -0.0198 | interpretable |
| layer:repaired | clash_fraction | qed | 300 | 0 | -0.1007 | -0.0807 | interpretable |
| layer:reranked | clash_fraction | qed | 300 | 0 | -0.0776 | -0.0669 | interpretable |

## Geometry Vs SA

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | clash_fraction | sa_score | 1100 | 0 | 0.3605 | 0.4061 | interpretable |
| raw_layers | clash_fraction | sa_score | 400 | 0 | 0.2171 | 0.2581 | interpretable |
| postprocessed_layers | clash_fraction | sa_score | 600 | 0 | 0.2853 | 0.3503 | interpretable |
| layer:constrained_flow | mean_centroid_offset | sa_score | 100 | 0 | 0.0345 | -0.0510 | interpretable |
| layer:raw_flow | mean_centroid_offset | sa_score | 100 | 0 | 0.0401 | -0.0329 | interpretable |
| layer:raw_rollout | clash_fraction | sa_score | 300 | 0 | -0.0646 | 0.0032 | interpretable |
| layer:repaired | clash_fraction | sa_score | 300 | 0 | 0.3233 | 0.3676 | interpretable |
| layer:reranked | clash_fraction | sa_score | 300 | 0 | 0.2517 | 0.3334 | interpretable |

## Clash Vs Docking

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | unavailable | unavailable | 0 | 0 | unsupported | unsupported | missing_pair |

## Interaction Proxies Vs Docking

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | unavailable | unavailable | 0 | 0 | unsupported | unsupported | missing_pair |

## Q2 Proxy Backend Pairs

| Scope | Pair | Left | Right | N | Missing | Pearson | Spearman | Expected | Interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| all | strict_pocket_fit_vs_vina | pocket_contact_fraction | vina_score | 1069 | 31 | 0.1768 | 0.1981 | negative | weak_or_inconclusive |
| all | pocket_contact_vs_vina | pocket_contact_fraction | vina_score | 1069 | 31 | 0.1768 | 0.1981 | negative | weak_or_inconclusive |
| all | centroid_offset_vs_gnina_affinity | centroid_offset | gnina_affinity | 200 | 900 | -0.0540 | -0.0736 | positive | weak_or_inconclusive |
| all | clash_fraction_vs_gnina_cnn_score | clash_fraction | gnina_cnn_score | 1100 | 0 | -0.1433 | -0.1601 | negative | weak_or_inconclusive |
| all | qed_vs_vina | qed | vina_score | 1069 | 31 | -0.2564 | -0.0471 | negative | weak_or_inconclusive |
| all | qed_vs_gnina_affinity | qed | gnina_affinity | 1100 | 0 | -0.2405 | -0.0310 | negative | weak_or_inconclusive |
| all | qed_vs_gnina_cnn_score | qed | gnina_cnn_score | 1100 | 0 | -0.0137 | -0.0058 | positive | weak_or_inconclusive |
| all | sa_vs_vina | sa_score | vina_score | 1069 | 31 | 0.3615 | 0.2735 | positive | usable_expected_direction |
| all | sa_vs_gnina_affinity | sa_score | gnina_affinity | 1100 | 0 | 0.3505 | 0.2444 | positive | weak_or_inconclusive |
| all | sa_vs_gnina_cnn_score | sa_score | gnina_cnn_score | 1100 | 0 | -0.3380 | -0.4027 | negative | usable_expected_direction |
| all | pocket_fit_vs_vina | pocket_contact_fraction | vina_score | 1069 | 31 | 0.1768 | 0.1981 | negative | weak_or_inconclusive |
| all | pocket_fit_vs_gnina_affinity | pocket_contact_fraction | gnina_affinity | 1100 | 0 | 0.1725 | 0.2026 | negative | weak_or_inconclusive |
| all | pocket_fit_vs_gnina_cnn_score | pocket_contact_fraction | gnina_cnn_score | 1100 | 0 | -0.2880 | -0.0780 | positive | weak_or_inconclusive |
| all | pocket_fit_vs_docking_proxy | pocket_contact_fraction | docking_like_score | 1100 | 0 | 0.3463 | 0.3395 | positive | usable_expected_direction |
| all | geometry_vs_qed | clash_fraction | qed | 1100 | 0 | 0.0249 | 0.0058 | negative | weak_or_inconclusive |
| all | geometry_vs_sa | clash_fraction | sa_score | 1100 | 0 | 0.3605 | 0.4061 | positive | usable_expected_direction |
| all | clash_vs_vina | clash_fraction | vina_score | 1069 | 31 | 0.6188 | 0.6784 | positive | usable_expected_direction |
| all | clash_vs_gnina_affinity | clash_fraction | gnina_affinity | 1100 | 0 | 0.6235 | 0.6804 | positive | usable_expected_direction |
| all | clash_vs_gnina_cnn_score | clash_fraction | gnina_cnn_score | 1100 | 0 | -0.1433 | -0.1601 | negative | weak_or_inconclusive |
| all | clash_vs_docking_proxy | clash_fraction | docking_like_score | 1100 | 0 | 0.3006 | 0.3399 | negative | misleading_opposite_direction |
| all | interaction_vs_vina | hydrogen_bond_proxy | vina_score | 1069 | 31 | 0.0248 | 0.1471 | negative | weak_or_inconclusive |
| all | interaction_vs_gnina_affinity | hydrogen_bond_proxy | gnina_affinity | 1100 | 0 | 0.0257 | 0.1505 | negative | weak_or_inconclusive |
| all | interaction_vs_gnina_cnn_score | hydrogen_bond_proxy | gnina_cnn_score | 1100 | 0 | 0.1109 | 0.0931 | positive | weak_or_inconclusive |
| all | interaction_vs_docking_proxy | hydrogen_bond_proxy | docking_like_score | 1100 | 0 | 0.2218 | 0.2642 | positive | usable_expected_direction |
| raw_layers | strict_pocket_fit_vs_vina | pocket_contact_fraction | vina_score | 369 | 31 | -0.0683 | 0.0373 | negative | weak_or_inconclusive |
| raw_layers | pocket_contact_vs_vina | pocket_contact_fraction | vina_score | 369 | 31 | -0.0683 | 0.0373 | negative | weak_or_inconclusive |
| raw_layers | centroid_offset_vs_gnina_affinity | centroid_offset | gnina_affinity | 100 | 300 | -0.1888 | -0.1907 | positive | weak_or_inconclusive |
| raw_layers | clash_fraction_vs_gnina_cnn_score | clash_fraction | gnina_cnn_score | 400 | 0 | -0.0832 | -0.2895 | negative | usable_expected_direction |
| raw_layers | qed_vs_vina | qed | vina_score | 369 | 31 | -0.0698 | 0.0569 | negative | weak_or_inconclusive |
| raw_layers | qed_vs_gnina_affinity | qed | gnina_affinity | 400 | 0 | -0.0399 | 0.0892 | negative | weak_or_inconclusive |
| raw_layers | qed_vs_gnina_cnn_score | qed | gnina_cnn_score | 400 | 0 | 0.0594 | 0.0733 | positive | weak_or_inconclusive |
| raw_layers | sa_vs_vina | sa_score | vina_score | 369 | 31 | 0.2327 | -0.0871 | positive | weak_or_inconclusive |
| raw_layers | sa_vs_gnina_affinity | sa_score | gnina_affinity | 400 | 0 | 0.1940 | -0.1407 | positive | weak_or_inconclusive |
| raw_layers | sa_vs_gnina_cnn_score | sa_score | gnina_cnn_score | 400 | 0 | -0.4808 | -0.5253 | negative | usable_expected_direction |
| raw_layers | pocket_fit_vs_vina | pocket_contact_fraction | vina_score | 369 | 31 | -0.0683 | 0.0373 | negative | weak_or_inconclusive |
| raw_layers | pocket_fit_vs_gnina_affinity | pocket_contact_fraction | gnina_affinity | 400 | 0 | -0.0652 | 0.0540 | negative | weak_or_inconclusive |
| raw_layers | pocket_fit_vs_gnina_cnn_score | pocket_contact_fraction | gnina_cnn_score | 400 | 0 | 0.2056 | 0.2435 | positive | weak_or_inconclusive |
| raw_layers | pocket_fit_vs_docking_proxy | pocket_contact_fraction | docking_like_score | 400 | 0 | 0.2071 | 0.2181 | positive | weak_or_inconclusive |
| raw_layers | geometry_vs_qed | clash_fraction | qed | 400 | 0 | -0.0631 | -0.0520 | negative | weak_or_inconclusive |
| raw_layers | geometry_vs_sa | clash_fraction | sa_score | 400 | 0 | 0.2171 | 0.2581 | positive | usable_expected_direction |
| raw_layers | clash_vs_vina | clash_fraction | vina_score | 369 | 31 | 0.2957 | 0.3793 | positive | usable_expected_direction |
| raw_layers | clash_vs_gnina_affinity | clash_fraction | gnina_affinity | 400 | 0 | 0.3141 | 0.3755 | positive | usable_expected_direction |
| raw_layers | clash_vs_gnina_cnn_score | clash_fraction | gnina_cnn_score | 400 | 0 | -0.0832 | -0.2895 | negative | usable_expected_direction |
| raw_layers | clash_vs_docking_proxy | clash_fraction | docking_like_score | 400 | 0 | -0.0087 | 0.0288 | negative | weak_or_inconclusive |
| raw_layers | interaction_vs_vina | hydrogen_bond_proxy | vina_score | 369 | 31 | 0.0983 | 0.2292 | negative | weak_or_inconclusive |
| raw_layers | interaction_vs_gnina_affinity | hydrogen_bond_proxy | gnina_affinity | 400 | 0 | 0.1043 | 0.2451 | negative | weak_or_inconclusive |
| raw_layers | interaction_vs_gnina_cnn_score | hydrogen_bond_proxy | gnina_cnn_score | 400 | 0 | 0.1108 | 0.0303 | positive | weak_or_inconclusive |
| raw_layers | interaction_vs_docking_proxy | hydrogen_bond_proxy | docking_like_score | 400 | 0 | 0.0352 | 0.0350 | positive | weak_or_inconclusive |
| postprocessed_layers | strict_pocket_fit_vs_vina | pocket_contact_fraction | vina_score | 600 | 0 | 0.3157 | 0.2490 | negative | weak_or_inconclusive |
| postprocessed_layers | pocket_contact_vs_vina | pocket_contact_fraction | vina_score | 600 | 0 | 0.3157 | 0.2490 | negative | weak_or_inconclusive |
| postprocessed_layers | centroid_offset_vs_gnina_affinity | centroid_offset | gnina_affinity | 0 | 600 | low_sample | low_sample | positive | inconclusive |
| postprocessed_layers | clash_fraction_vs_gnina_cnn_score | clash_fraction | gnina_cnn_score | 600 | 0 | -0.0759 | 0.0013 | negative | weak_or_inconclusive |
| postprocessed_layers | qed_vs_vina | qed | vina_score | 600 | 0 | -0.4143 | -0.1974 | negative | weak_or_inconclusive |
| postprocessed_layers | qed_vs_gnina_affinity | qed | gnina_affinity | 600 | 0 | -0.4144 | -0.1973 | negative | weak_or_inconclusive |
| postprocessed_layers | qed_vs_gnina_cnn_score | qed | gnina_cnn_score | 600 | 0 | -0.0333 | -0.0224 | positive | weak_or_inconclusive |
| postprocessed_layers | sa_vs_vina | sa_score | vina_score | 600 | 0 | 0.7951 | 0.7892 | positive | usable_expected_direction |
| postprocessed_layers | sa_vs_gnina_affinity | sa_score | gnina_affinity | 600 | 0 | 0.7951 | 0.7893 | positive | usable_expected_direction |
| postprocessed_layers | sa_vs_gnina_cnn_score | sa_score | gnina_cnn_score | 600 | 0 | -0.1971 | -0.2936 | negative | usable_expected_direction |
| postprocessed_layers | pocket_fit_vs_vina | pocket_contact_fraction | vina_score | 600 | 0 | 0.3157 | 0.2490 | negative | weak_or_inconclusive |
| postprocessed_layers | pocket_fit_vs_gnina_affinity | pocket_contact_fraction | gnina_affinity | 600 | 0 | 0.3157 | 0.2493 | negative | weak_or_inconclusive |
| postprocessed_layers | pocket_fit_vs_gnina_cnn_score | pocket_contact_fraction | gnina_cnn_score | 600 | 0 | -0.5339 | -0.2443 | positive | weak_or_inconclusive |
| postprocessed_layers | pocket_fit_vs_docking_proxy | pocket_contact_fraction | docking_like_score | 600 | 0 | 0.8718 | 0.4427 | positive | usable_expected_direction |
| postprocessed_layers | geometry_vs_qed | clash_fraction | qed | 600 | 0 | -0.0884 | -0.0740 | negative | weak_or_inconclusive |
| postprocessed_layers | geometry_vs_sa | clash_fraction | sa_score | 600 | 0 | 0.2853 | 0.3503 | positive | usable_expected_direction |
| postprocessed_layers | clash_vs_vina | clash_fraction | vina_score | 600 | 0 | 0.5000 | 0.5376 | positive | usable_expected_direction |
| postprocessed_layers | clash_vs_gnina_affinity | clash_fraction | gnina_affinity | 600 | 0 | 0.5001 | 0.5375 | positive | usable_expected_direction |
| postprocessed_layers | clash_vs_gnina_cnn_score | clash_fraction | gnina_cnn_score | 600 | 0 | -0.0759 | 0.0013 | negative | weak_or_inconclusive |
| postprocessed_layers | clash_vs_docking_proxy | clash_fraction | docking_like_score | 600 | 0 | 0.1051 | -0.0981 | negative | weak_or_inconclusive |
| postprocessed_layers | interaction_vs_vina | hydrogen_bond_proxy | vina_score | 600 | 0 | -0.1164 | -0.0255 | negative | weak_or_inconclusive |
| postprocessed_layers | interaction_vs_gnina_affinity | hydrogen_bond_proxy | gnina_affinity | 600 | 0 | -0.1163 | -0.0252 | negative | weak_or_inconclusive |
| postprocessed_layers | interaction_vs_gnina_cnn_score | hydrogen_bond_proxy | gnina_cnn_score | 600 | 0 | 0.0379 | 0.0474 | positive | weak_or_inconclusive |
| postprocessed_layers | interaction_vs_docking_proxy | hydrogen_bond_proxy | docking_like_score | 600 | 0 | 0.2574 | 0.1383 | positive | weak_or_inconclusive |

## Raw Vs Postprocessed Deltas

| Method | Raw Layer | Target Layer | Pairs | dVina | dGNINA | dCNN | dQED | dSA | dClash | dContact |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| diffsbdd_public | raw_rollout | repaired | 100 | 192.7969 | 184.9193 | -0.1343 | 0.0000 | 0.0000 | 0.1144 | -0.0320 |
| diffsbdd_public | raw_rollout | reranked | 100 | 192.7969 | 184.9193 | -0.1343 | -0.0124 | 0.0486 | 0.1144 | -0.0320 |
| flow_matching | raw_flow | constrained_flow | 100 | 54.7538 | 54.7442 | -0.0255 | 0.0558 | 1.3978 | -0.0015 | 0.0487 |
| pocket2mol_public | raw_rollout | repaired | 100 | 66.4391 | 64.1078 | -0.3080 | 0.0000 | 0.0000 | 0.0591 | -0.0386 |
| pocket2mol_public | raw_rollout | reranked | 100 | 66.4391 | 64.1078 | -0.3080 | 0.0018 | 0.0102 | 0.0591 | -0.0386 |
| targetdiff_public | raw_rollout | repaired | 100 | 231.7438 | 228.4786 | -0.1852 | 0.0000 | 0.0000 | 0.1287 | -0.0356 |
| targetdiff_public | raw_rollout | reranked | 100 | 231.7438 | 228.4786 | -0.1852 | 0.0075 | 0.0657 | 0.1287 | -0.0356 |

## Flow Vs Denoising On Binding

Use method-comparison candidate metrics with matched budgets before interpreting this section. Rows remain unsupported until both flow and denoising layers have candidate-level docking or GNINA/Vina coverage.

## Constraint Vs True Model Capability

Interpret raw_flow rows as native model evidence. Interpret repaired and reranked rows as postprocessing evidence unless raw-flow backend metrics are explicitly present.
