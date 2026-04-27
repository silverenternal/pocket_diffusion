# Baseline Comparison

Provenance: `checkpoints/pdbbindpp_real_backends`

| method | role | steps | runtime ms | raw valid | raw contact | raw clash | layers |
|---|---|---:|---:|---:|---:|---:|---|
| conditioned_denoising | claimbearing | 4 | 56.69 | 1 | 1 | 0.004736 | raw_rollout, repaired_candidates, inferred_bond_candidates, deterministic_proxy_candidates, reranked_candidates |
| heuristic_raw_rollout_no_repair | comparisononly | 4 | 9.828 | 1 | 1 | 0.004736 | raw_rollout |
| pocket_centroid_repair_proxy | comparisononly | 4 | 13.3 | NA | NA | NA | raw_rollout, repaired_candidates |
| deterministic_proxy_reranker | comparisononly | 4 | 24.88 | NA | NA | NA | raw_rollout, repaired_candidates, inferred_bond_candidates, deterministic_proxy_candidates |
| calibrated_reranker | claimbearing | 4 | 58.27 | 1 | 1 | 0.0002642 | raw_rollout, repaired_candidates, inferred_bond_candidates, deterministic_proxy_candidates, reranked_candidates |
| flow_matching | comparisononly | 4 | 48.64 | 1 | 1 | 0.001959 | raw_rollout, repaired_candidates, inferred_bond_candidates |
| autoregressive_graph_geometry | comparisononly | 4 | 26.31 | 1 | 1 | 0.001298 | raw_rollout, repaired_candidates, inferred_bond_candidates |
| energy_guided_refinement | comparisononly | 4 | 44.24 | NA | NA | NA | raw_rollout, repaired_candidates, inferred_bond_candidates, deterministic_proxy_candidates |
| preference_aware_reranker | comparisononly | 4 | 43.36 | 1 | 1 | 0.0002642 | raw_rollout, repaired_candidates, inferred_bond_candidates, deterministic_proxy_candidates, reranked_candidates |
