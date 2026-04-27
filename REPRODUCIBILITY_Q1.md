# Q1 Reproducibility Checklist

This package is initialized around versioned artifacts rather than hand-entered tables. The current public-baseline comparison is generated from 900 layer-separated candidate rows, not manually entered values.

Core commands:

```bash
cargo test
python tools/q1_readiness_audit.py --gate
python tools/validation_suite.py --mode quick --timeout 240
python tools/render_paper_tables.py
```

Backend dependencies:

- RDKit for chemistry validity and drug-likeness metrics
- Vina for docking scores
- GNINA for CNN score and affinity metrics

Current public-baseline evidence:

- Pocket2Mol, TargetDiff, and DiffSBDD have 100-pocket matched-budget public-baseline artifacts.
- `raw_rollout`, `repaired`, and `reranked` rows are reported separately.
- RDKit coverage is 1.0000, GNINA coverage is 1.0000, and Vina coverage is 0.9656 for the 900-row layered run.
- `raw_rollout` is native baseline evidence; repaired/reranked rows are shared deterministic postprocessing evidence.

Known remaining blockers:

- public-baseline multi-seed evidence is not complete
- Q1 statistical tests still need final paired main-vs-baseline candidate metric inputs
- the stress benchmark subset exists as an artifact path, but publication wording should still cite its exact generated report

Do not report binding-affinity claims from proxy-only artifacts. Use `configs/q1_reproducibility_manifest.json` as the machine-readable checklist.
