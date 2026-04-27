# Manuscript Outline

## Title

Semantic-preserving minimal-redundancy representation learning for pocket-conditioned molecular generation.

## Claims

1. The model keeps topology, geometry, and pocket context structurally separate while allowing gated cross-modal attention.
2. Slot decomposition and redundancy controls provide interpretable specialization signals.
3. Layered artifacts separate raw generation, repair, and reranking contributions.
4. Unseen-pocket conclusions are claimable only where real backend and public-baseline evidence is complete.

## Results Map

- Primary benchmark: `configs/q1_primary_benchmark_manifest.json`
- Baselines: `configs/q1_baseline_registry.json`
- Method comparison: `configs/q1_method_comparison_summary.json`
- Ablations: `configs/q1_ablation_matrix.json`
- Statistics: `configs/q1_statistical_tests.json`
- Slot/gate analysis: `configs/q1_slot_gate_analysis.json`
- Cases: `configs/q1_case_selection.json`

## Limitations

Public baseline runs, stress subset materialization, and complete paired statistical tests remain required before Q1 submission claims can be promoted.
