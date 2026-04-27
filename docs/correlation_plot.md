# Correlation Plot

Source table: `configs/correlation_table.json`
Record count: 88
Minimum interpretable samples: 3

This artifact is rendered from `correlation_table.json`; unsupported entries indicate missing backend coverage or too few candidate-level samples.

## Pocket Fit Vs Docking

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | unavailable | unavailable | 0 | 0 | unsupported | unsupported | missing_pair |

## Geometry Vs QED

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | clash_fraction | qed | 88 | 0 | 0.3290 | 0.4543 | interpretable |
| layer:deterministic_proxy | clash_fraction | qed | 8 | 0 | 0.7608 | 0.6385 | interpretable |
| layer:inferred_bond | clash_fraction | qed | 24 | 0 | 0.6143 | 0.5484 | interpretable |
| layer:raw_rollout | mean_centroid_offset | qed | 24 | 0 | 0.4063 | 0.2143 | interpretable |
| layer:repaired | clash_fraction | qed | 24 | 0 | 0.2310 | 0.2595 | interpretable |
| layer:reranked | clash_fraction | qed | 8 | 0 | 0.5195 | 0.5123 | interpretable |

## Geometry Vs SA

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | clash_fraction | sa_score | 88 | 0 | -0.4848 | -0.5982 | interpretable |
| layer:deterministic_proxy | clash_fraction | sa_score | 8 | 0 | -0.8005 | -0.8231 | interpretable |
| layer:inferred_bond | clash_fraction | sa_score | 24 | 0 | -0.7523 | -0.8074 | interpretable |
| layer:raw_rollout | mean_centroid_offset | sa_score | 24 | 0 | -0.7763 | -0.8024 | interpretable |
| layer:repaired | clash_fraction | sa_score | 24 | 0 | -0.6157 | -0.5968 | interpretable |
| layer:reranked | clash_fraction | sa_score | 8 | 0 | -0.7686 | -0.7408 | interpretable |

## Clash Vs Docking

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | unavailable | unavailable | 0 | 0 | unsupported | unsupported | missing_pair |

## Interaction Proxies Vs Docking

| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | unavailable | unavailable | 0 | 0 | unsupported | unsupported | missing_pair |

## Flow Vs Denoising On Binding

Use method-comparison candidate metrics with matched budgets before interpreting this section. Rows remain unsupported until both flow and denoising layers have candidate-level docking or GNINA/Vina coverage.

## Constraint Vs True Model Capability

Interpret raw_flow rows as native model evidence. Interpret repaired and reranked rows as postprocessing evidence unless raw-flow backend metrics are explicitly present.
