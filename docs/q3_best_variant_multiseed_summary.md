# Q3 Best Variant Multi-Seed

Best variant: `pairwise_distance_direction`

| Seed | Vina | GNINA affinity | GNINA CNN | QED | SA | Clash | Contact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 83.2245 | 83.2075 | 0.0559 | 0.4190 | 2.6970 | 0.0939 | 0.9396 |
| 42 | 70.6441 | 70.6465 | 0.0779 | 0.4178 | 2.6946 | 0.0665 | 0.9530 |
| 101 | 72.3683 | 72.3651 | 0.0814 | 0.4350 | 2.6982 | 0.0799 | 0.9509 |

## Aggregate Means

| Metric | Mean | Variance | Std |
|---|---:|---:|---:|
| vina_score | 75.4123 | 31.0107 | 5.5687 |
| gnina_affinity | 75.4064 | 30.9212 | 5.5607 |
| gnina_cnn_score | 0.0717 | 0.0001 | 0.0113 |
| qed | 0.4239 | 0.0001 | 0.0078 |
| sa_score | 2.6966 | 0.0000 | 0.0015 |
| clash_fraction | 0.0801 | 0.0001 | 0.0112 |
| pocket_contact_fraction | 0.9479 | 0.0000 | 0.0059 |

Decision: best Q3 raw-flow multiseed mean does not beat Q2 flow_matching raw_flow on Vina/GNINA score_only; no statistical dominance claim emitted.
