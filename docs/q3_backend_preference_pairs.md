# Q3 Backend Preference Pairs

This artifact constructs backend-backed preference data only. It does not enable RL/DPO training and must not be reported as native model improvement.

## Coverage

- pair_count: 134
- method_id_coverage: 4
- backend_supported_pair_fraction: 0.9627

| Class | Count |
| --- | ---: |
| `docking_good_druglike_bad` | 34 |
| `good_docking_druglike` | 40 |
| `high_pocket_fit_bad_docking` | 40 |
| `repair_destroys_docking` | 20 |

## Top Pairs

| Class | Winner | Loser | Strength | dVina | dGNINA | dQED | dSA |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:000_BSD_ASPTE_1_130_0:0` | `targetdiff_public:centroid_only:000_BSD_ASPTE_1_130_0:0` | 1 | -435.9 | -435.9 | 0 | 0 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:001_GLMU_STRPN_2_459_0:0` | `targetdiff_public:centroid_only:001_GLMU_STRPN_2_459_0:0` | 1 | NA | -126.8 | 0 | 0 |
| high_pocket_fit_bad_docking | `diffsbdd_public:clash_only:002_GRK4_HUMAN_1_578_0:0` | `flow_matching:constrained_flow:002_GRK4_HUMAN_1_578_0:0` | 1 | -263.3 | -263.3 | -0.2565 | -1.64 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:003_GSTP1_HUMAN_2_210_0:0` | `targetdiff_public:full_repair:003_GSTP1_HUMAN_2_210_0:0` | 1 | -253.9 | -253.8 | -0.06454 | -0.36 |
| high_pocket_fit_bad_docking | `diffsbdd_public:no_repair:004_GUX1_HYPJE_18_451_0:0` | `diffsbdd_public:full_repair:004_GUX1_HYPJE_18_451_0:0` | 1 | -264.1 | -264.2 | -0.1489 | 0 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:005_HDAC8_HUMAN_1_377_0:0` | `targetdiff_public:centroid_only:005_HDAC8_HUMAN_1_377_0:0` | 1 | -338.1 | -338.1 | 0 | 0 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:006_HDHA_ECOLI_1_255_0:0` | `targetdiff_public:full_repair:006_HDHA_ECOLI_1_255_0:0` | 1 | -169.1 | -169.1 | -0.01747 | 0.03 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:007_HMD_METJA_1_358_0:0` | `diffsbdd_public:full_repair:007_HMD_METJA_1_358_0:0` | 1 | NA | -498.4 | 0.02636 | -1.43 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:008_CCPR_YEAST_69_361_0:0` | `targetdiff_public:centroid_only:008_CCPR_YEAST_69_361_0:0` | 1 | -276.2 | -276.3 | 0 | 0 |
| high_pocket_fit_bad_docking | `diffsbdd_public:no_repair:009_IPMK_HUMAN_49_416_0:0` | `diffsbdd_public:centroid_only:009_IPMK_HUMAN_49_416_0:0` | 1 | -390.7 | -390.7 | 0 | 0 |
| high_pocket_fit_bad_docking | `diffsbdd_public:no_repair:010_CD38_HUMAN_44_300_0:0` | `diffsbdd_public:full_repair:010_CD38_HUMAN_44_300_0:0` | 1 | -337.7 | -337.7 | 0.04515 | 0.06 |
| high_pocket_fit_bad_docking | `diffsbdd_public:no_repair:011_KS6A3_HUMAN_41_357_0:0` | `diffsbdd_public:full_repair:011_KS6A3_HUMAN_41_357_0:0` | 1 | -289.9 | -289.9 | 0 | 0 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:012_CHOD_BREST_46_552_0:0` | `targetdiff_public:full_repair:012_CHOD_BREST_46_552_0:0` | 1 | -155.4 | -155.4 | 0 | 0 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:013_LAT_MYCTU_1_449_0:0` | `diffsbdd_public:centroid_only:013_LAT_MYCTU_1_449_0:0` | 1 | -249 | -249 | -0.009919 | 0.02 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:014_LMBL1_HUMAN_198_526_0:0` | `flow_matching:constrained_flow:014_LMBL1_HUMAN_198_526_0:0` | 1 | -103.6 | -103.7 | -0.05842 | -0.84 |
| high_pocket_fit_bad_docking | `diffsbdd_public:no_repair:015_LMBL1_HUMAN_198_526_0:0` | `targetdiff_public:full_repair:015_LMBL1_HUMAN_198_526_0:0` | 1 | -177 | -177 | 0.1032 | -1.24 |
| high_pocket_fit_bad_docking | `diffsbdd_public:no_repair:016_M3K14_HUMAN_321_678_0:0` | `diffsbdd_public:full_repair:016_M3K14_HUMAN_321_678_0:0` | 1 | -489.4 | -489.3 | 0.01579 | 0.09 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:017_MENE_BACSU_2_486_0:0` | `targetdiff_public:centroid_only:017_MENE_BACSU_2_486_0:0` | 1 | -340.1 | -340.1 | 0 | 0 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:018_NAGZ_VIBCH_1_330_0:0` | `targetdiff_public:centroid_only:018_NAGZ_VIBCH_1_330_0:0` | 1 | -325.6 | -325.6 | 0 | 0 |
| high_pocket_fit_bad_docking | `targetdiff_public:no_repair:019_NEP_HUMAN_54_750_0:0` | `flow_matching:raw_flow:019_NEP_HUMAN_54_750_0:0` | 1 | -68.45 | -68.36 | 0.1412 | 0.38 |
