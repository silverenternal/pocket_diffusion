# Q3 Pairwise Geometry Ablation

| Variant | Layer | Vina | GNINA affinity | GNINA CNN | QED | SA | Clash | Contact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_pairwise | raw_flow | 116.8894 | 116.8789 | 0.0414 | 0.4174 | 2.6946 | 0.1356 | 0.9418 |
| no_pairwise | constrained_flow | 92.6658 | 92.6453 | 0.1147 | 0.4590 | 4.1670 | 0.0145 | 0.9928 |
| pairwise_distance_direction | raw_flow | 70.4562 | 70.4571 | 0.0637 | 0.4174 | 2.6946 | 0.0799 | 0.9658 |
| pairwise_distance_direction | constrained_flow | 87.7005 | 87.6853 | 0.1081 | 0.4336 | 4.1130 | 0.0109 | 0.9897 |
| pairwise_with_clash_margin | raw_flow | 113.5645 | 113.5563 | 0.0406 | 0.4174 | 2.6946 | 0.1286 | 0.9604 |
| pairwise_with_clash_margin | constrained_flow | 103.8247 | 103.8101 | 0.0797 | 0.4379 | 3.8782 | 0.0185 | 0.9916 |
| pairwise_plus_local_pocket | raw_flow | 112.4301 | 112.4200 | 0.0739 | 0.4174 | 2.6946 | 0.1163 | 0.9460 |
| pairwise_plus_local_pocket | constrained_flow | 126.8421 | 126.8144 | 0.0506 | 0.4391 | 3.7876 | 0.0333 | 0.9903 |

Best raw_flow variant by Vina/GNINA rank: `pairwise_distance_direction`.
