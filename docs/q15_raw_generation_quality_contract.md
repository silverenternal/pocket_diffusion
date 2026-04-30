# Q15 Raw Generation Quality Contract

Date: 2026-04-29

This contract defines which generation layers can support model-training claims and which layers only support downstream pipeline claims.

## Layer Contract

| Layer family | Canonical fields | Required provenance | Claim use |
| --- | --- | --- | --- |
| Raw-native validity | `raw_flow`, `raw_rollout`, `raw_native_graph_extraction` | `generation_path_class=model_native_raw`, `model_native_raw=true`, `target_source=generated_rollout_state` for evaluation rows | May support raw model-native molecular-generation claims when metrics are finite and no repair, constraints, reranking, or external backend scoring has been applied. |
| Raw geometry consistency | `raw_geometry_candidates`, raw coordinate metrics on `raw_rollout` | `generation_path_class=model_native_raw`, empty `postprocessor_chain` | May support model-native geometry claims only before coordinate repair or envelope clamping. |
| Raw pocket compatibility | raw pocket-contact, centroid, clash, and interaction-profile metrics on `raw_rollout` | `candidate_layer=raw_rollout`, `model_native_raw=true`, target/evidence source from generated rollout and pocket geometry | May support pocket-conditioned raw generation claims only if not mixed with repaired or reranked rows. |
| Repaired validity | `repaired`, `repaired_candidates`, `inferred_bond_candidates`, `constrained_flow` | `model_native_raw=false`, nonempty `postprocessor_chain` such as `geometry_repair`, `bond_inference`, or `valence_pruning` | Supports downstream repair or constrained-pipeline claims. It must not overwrite raw-native validity. |
| External scoring | `backend_scored_candidates`, backend metric slots | `generation_path_class=external_backend`, `backend_supported=true`, `target_source=external_backend` | Supports backend scoring claims only when backend coverage and structure provenance are explicit. It does not by itself prove raw-native generation quality. |

## Metric Aggregation Rules

1. Raw and repaired metrics must remain separate structs in `LayeredGenerationMetrics`.
2. `TrainEvalAlignmentMetricRow.target_source` records the source of metric evidence and prevents generated-state, repair-layer, and backend rows from silently merging.
3. `GenerationPathContractRow` is the layer-level schema contract. A layer without a contract cannot be promoted to claim-bearing evidence.
4. `raw_model_valid_fraction`, `raw_native_graph_valid_fraction`, `native_valence_violation_fraction`, `native_disconnected_fragment_fraction`, `native_bond_order_conflict_fraction`, raw pocket compatibility, and raw geometry consistency must cite raw model-native layers.
5. `processed_valid_fraction`, repaired validity, inferred-bond validity, and reranker rows must cite processed layers and carry a postprocessor chain.
6. Backend metrics must retain backend availability and coverage fields. Missing backend coverage is a caution or fail condition, not an implicit raw-model score.

## Final Claim Requirement

Final molecular-generation claims require raw-native evidence first. Repaired, constrained, reranked, and externally scored layers may be reported as additive pipeline evidence, but they cannot compensate for missing or weak raw-native validity, geometry, or pocket-compatibility metrics.
