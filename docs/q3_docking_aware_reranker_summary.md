# Q3 Docking-Aware Reranker Summary

The backend-aware reranker is an offline posthoc baseline over already scored candidates. Raw model capability remains reported only from raw_flow/raw_rollout/no_repair rows.

- label: `backend_aware_posthoc`
- examples: 100
- candidate_pool_count: 1100
- raw_model_capability_source: `raw_flow/raw_rollout/no_repair only`

| Selector | Vina | GNINA | CNN | QED | SA | Clash | Contact | Raw-native fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| proxy | 55.22 | 54.65 | 0.1734 | 0.4606 | 3.235 | 0.001738 | 0.9988 | 0.35 |
| backend_aware_posthoc | -6.285 | -6.663 | 0.4287 | 0.3915 | 2.703 | 0 | 0.965 | 1 |

Mean backend-aware minus proxy deltas:

- Vina: -62.11
- GNINA affinity: -61.31
- QED: -0.0691
- SA: -0.5314
