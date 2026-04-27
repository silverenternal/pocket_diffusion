# Vina Scoring Summary

- backend: AutoDock Vina 6a621d4
- mode: score_only
- grid_policy: candidate_enclosing_score_only_box
- input_preparation_report: `configs/docking_input_preparation_report.json`

| split | input_count | scored_count | coverage | mean | median | best | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 88 | 88 | 1.0000 | 152.0643 | 48.2230 | -5.3330 | 0 |
| test | 88 | 88 | 1.0000 | 105.8949 | 73.5430 | -10.8520 | 0 |

## Outputs
- `validation` JSON: `checkpoints/pdbbindpp_real_backends/candidate_metrics_vina_validation.json`
- `validation` JSONL: `checkpoints/pdbbindpp_real_backends/candidate_metrics_vina_validation.jsonl`
- `test` JSON: `checkpoints/pdbbindpp_real_backends/candidate_metrics_vina_test.json`
- `test` JSONL: `checkpoints/pdbbindpp_real_backends/candidate_metrics_vina_test.jsonl`
