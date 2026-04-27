# Paper Submission Checklist

- Run `tools/artifact_audit.py --gate configs docs` before exporting paper-facing reports.
- Run `tools/artifact_audit.py --check-existing` when reviewing whether `configs/artifact_audit_report.json` is stale relative to scanned sources.
- Treat missing method-comparison, multi-seed, ablation, or correlation artifacts as blocking until regenerated from machine-produced outputs.
- For public baselines, cite `configs/q1_method_comparison_summary.json` and `docs/q1_method_comparison_table.md`; do not copy values by hand.
- State explicitly that `raw_rollout` is native baseline evidence, while `repaired` and `reranked` rows are shared deterministic postprocessing evidence.
- Do not describe Vina/GNINA score-only outputs as experimental binding affinities.
