# Q3 Repair Damage Cases

This is a postprocessing-diagnosis artifact. It does not promote repaired layers as model-native evidence.

## Summary

- paired_layer_comparisons: 1200
- candidate_geometry_records: 1500
- worst_case_count: 20

## Component Counts

- `bond_payload_chemistry_regression`: 196
- `bond_payload_or_conversion`: 1
- `coordinate_movement`: 580
- `mixed_or_inconclusive`: 423

## Layer Damage

| Layer | Candidates | Degraded | Mean Damage | Mean Centroid Shift | Mean Box Shift | Mean Bond Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bond_inference_only | 300 | 0 | 0 | 0 | 0 | 17.37 |
| centroid_only | 300 | 293 | 284.7 | 15.92 | 9.911 | 0 |
| clash_only | 300 | 33 | -0.08412 | 0.000728 | 0.0008271 | 0 |
| full_repair | 300 | 299 | 305.3 | 20.95 | 13.67 | 2.223 |

## Worst Cases

| Rank | Candidate | Layer | Component | Damage | dVina | dGNINA | Centroid Shift | Box Shift | Bond Delta |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `targetdiff_public:centroid_only:072_TNKS1_HUMAN_1099_1319_0:0` | centroid_only | `coordinate_movement` | 1263 | 631.3 | 631.3 | 8.181 | 6.602 | 0 |
| 2 | `targetdiff_public:centroid_only:026_PHKG1_RABIT_6_296_ATPsite_0:0` | centroid_only | `coordinate_movement` | 1169 | 584.5 | 584.3 | 10.59 | 9.229 | 0 |
| 3 | `diffsbdd_public:full_repair:092_CHIB_SERMA_1_499_0:0` | full_repair | `coordinate_movement` | 1149 | 574.3 | 574.3 | 13.06 | 9.793 | 3 |
| 4 | `targetdiff_public:full_repair:092_CHIB_SERMA_1_499_0:0` | full_repair | `coordinate_movement` | 1118 | 559.3 | 559.1 | 14.12 | 10.38 | 3 |
| 5 | `targetdiff_public:full_repair:031_CPXB_BACMB_2_464_0:0` | full_repair | `coordinate_movement` | 1099 | 549.6 | 549.8 | 10.44 | 10.34 | 9 |
| 6 | `diffsbdd_public:centroid_only:092_CHIB_SERMA_1_499_0:0` | centroid_only | `coordinate_movement` | 1095 | 547.5 | 547.1 | 11.82 | 10.04 | 0 |
| 7 | `targetdiff_public:centroid_only:092_CHIB_SERMA_1_499_0:0` | centroid_only | `coordinate_movement` | 1094 | 547.1 | 547 | 12.49 | 10.27 | 0 |
| 8 | `targetdiff_public:centroid_only:031_CPXB_BACMB_2_464_0:0` | centroid_only | `coordinate_movement` | 1091 | 545.6 | 545.5 | 7.614 | 7.095 | 0 |
| 9 | `targetdiff_public:centroid_only:022_NR1H4_HUMAN_258_486_0:0` | centroid_only | `coordinate_movement` | 1089 | 544.5 | 544.6 | 9.096 | 8.162 | 0 |
| 10 | `targetdiff_public:centroid_only:054_NOS1_HUMAN_302_723_0:0` | centroid_only | `coordinate_movement` | 1045 | 522.8 | 522.7 | 16.08 | 10.18 | 0 |
| 11 | `targetdiff_public:full_repair:081_BGL07_ORYSJ_25_504_0:0` | full_repair | `coordinate_movement` | 1033 | 516.5 | 516.3 | 11.31 | 9.828 | 5 |
| 12 | `targetdiff_public:centroid_only:053_NOS1_HUMAN_302_723_0:0` | centroid_only | `coordinate_movement` | 998.2 | 499 | 499.1 | 16.64 | 10.18 | 0 |
| 13 | `targetdiff_public:centroid_only:037_ROCO4_DICDI_1009_1292_0:0` | centroid_only | `coordinate_movement` | 990.8 | 495.4 | 495.4 | 12.02 | 8.936 | 0 |
| 14 | `diffsbdd_public:full_repair:016_M3K14_HUMAN_321_678_0:0` | full_repair | `coordinate_movement` | 978.7 | 489.4 | 489.3 | 10.83 | 9.671 | 5 |
| 15 | `diffsbdd_public:centroid_only:022_NR1H4_HUMAN_258_486_0:0` | centroid_only | `coordinate_movement` | 975.3 | 487.7 | 487.6 | 8.621 | 6.636 | 0 |
| 16 | `targetdiff_public:centroid_only:081_BGL07_ORYSJ_25_504_0:0` | centroid_only | `coordinate_movement` | 975.1 | 487.5 | 487.6 | 10.07 | 8.833 | 0 |
| 17 | `targetdiff_public:centroid_only:076_ABL2_HUMAN_274_551_0:0` | centroid_only | `coordinate_movement` | 933.2 | 466.5 | 466.7 | 11.36 | 9.633 | 0 |
| 18 | `diffsbdd_public:full_repair:022_NR1H4_HUMAN_258_486_0:0` | full_repair | `coordinate_movement` | 928.9 | 464.6 | 464.3 | 11.28 | 9.301 | 5 |
| 19 | `diffsbdd_public:centroid_only:051_MCCF_ECOLX_1_344_0:0` | centroid_only | `coordinate_movement` | 901.1 | 450.5 | 450.6 | 14.63 | 11.69 | 0 |
| 20 | `targetdiff_public:centroid_only:083_BACE2_HUMAN_76_460_0:0` | centroid_only | `coordinate_movement` | 897.7 | 448.8 | 448.9 | 13.92 | 11.09 | 0 |

## Interpretation

The largest candidate-level failures are dominated by coordinate-moving layers. Bond-only changes are tracked separately because they mainly affect chemistry payload and SA/QED rather than pose coordinates.
