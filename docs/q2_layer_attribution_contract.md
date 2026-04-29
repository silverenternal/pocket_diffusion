# Q2 Layer Attribution Contract

Q2 artifacts must keep raw model output separate from constrained sampling and postprocessing. The required structural distinction is:

| Layer | Model native | Constrained sampling | Claim role |
| --- | --- | --- | --- |
| `raw_flow` | yes | no | raw model capability |
| `constrained_flow` | no | yes | constrained sampling evidence |
| `raw_rollout` | yes | no | legacy raw model capability |
| `repaired` | no | no | postprocessing evidence |
| `inferred_bond` | no | no | postprocessing evidence |
| `deterministic_proxy` | no | no | proxy analysis only |
| `reranked` | no | no | postprocessing evidence |

Every candidate row should retain `method_id`, `layer`, `candidate_id`, `example_id`, `protein_id`, source pocket and ligand paths when available, `coordinate_frame_origin`, `model_native`, `constrained_sampling`, `postprocessing_steps`, and `claim_allowed`.

`raw_flow` and `constrained_flow` must be emitted and summarized as different layers. Repaired or reranked values can support postprocessing analysis, but they cannot be used as raw model-native capability.

Method-comparison reports must expose the family-aware `selected_metric_layer` together with `method_family`. Base and hybrid generator methods select their raw model-native layer for native metrics; `repair_only` methods may select repaired layers, and `reranker_only` methods may select deterministic-proxy or reranked layers. This invariant keeps repaired/proxy/reranked quality from silently replacing raw native evidence.
