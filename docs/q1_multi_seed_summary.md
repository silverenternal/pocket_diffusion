# Q1 Multi-Seed Summary

Internal seed artifacts are present for the main method family, and aggregate summaries are generated in `configs/multi_seed_drug_level_summary.json`.

Q1 claim status remains partial because public-baseline multi-seed closure is not complete. The single-seed public-baseline surface is now adapted and scored under the same layer-separated backend pipeline:

- summary: `configs/q1_method_comparison_summary.json`
- table: `docs/q1_method_comparison_table.md`
- merged metrics: `checkpoints/q1_public_baselines_full100_layered/merged/candidate_metrics_q1_public_full100_budget1.jsonl`

Missing public-baseline seed counts must remain visible in generated tables. Do not treat the 100-pocket single-seed public-baseline comparison as a replacement for multi-seed evidence.
