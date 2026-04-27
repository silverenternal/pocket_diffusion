# GNINA Scoring Summary

- backend: gnina v1.3.2 master:f23dd2b   Built Jul  8 2025.
- mode: score_only
- metrics: GNINA affinity, CNN score, CNN affinity, CNN variance
- input_preparation_report: `configs/docking_input_preparation_report.json`

| split | input_count | scored_count | coverage | affinity_mean | affinity_best | cnn_score_mean | cnn_affinity_mean | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 88 | 88 | 1.0000 | 152.0550 | -5.3450 | 0.2102 | 4.1057 | 0 |
| test | 88 | 88 | 1.0000 | 105.9071 | -10.8700 | 0.0713 | 5.1499 | 0 |

## Outputs
- `validation` JSON: `checkpoints/pdbbindpp_real_backends/candidate_metrics_gnina_validation.json`
- `validation` JSONL: `checkpoints/pdbbindpp_real_backends/candidate_metrics_gnina_validation.jsonl`
- `test` JSON: `checkpoints/pdbbindpp_real_backends/candidate_metrics_gnina_test.json`
- `test` JSONL: `checkpoints/pdbbindpp_real_backends/candidate_metrics_gnina_test.jsonl`
