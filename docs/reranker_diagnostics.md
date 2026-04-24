# Reranker Diagnostics

This file is generated from canonical reviewer artifacts and should be refreshed through `./tools/revalidate_reviewer_bundle.sh`.

## larger_data_canonical

- Artifact: `checkpoints/pdbbindpp_real_backends`
- Calibration method: `bounded_nonnegative_feature_target_covariance_v1`
- Active coefficients: valence_sane=0.9760, bond_density_fit=0.0170, centroid_fit=0.0070
- Fitted candidate count: 24
- Decision: reranker improves at least one candidate-layer fit metric relative to inferred-bond candidates.

| Layer | Candidates | Valid | Centroid offset | Clash | Atom-seq diversity | Bond-topology diversity |
| --- | --- | --- | --- | --- | --- | --- |
| raw_rollout | 24 | 1.0000 | 1.4647 | 0.0049 | 0.3333 | 0.0417 |
| repaired_candidates | 24 | 1.0000 | 0.6242 | 0.0003 | 0.3333 | 0.0417 |
| inferred_bond_candidates | 24 | 1.0000 | 0.6242 | 0.0001 | 0.6667 | 1.0000 |
| deterministic_proxy_candidates | 12 | 1.0000 | 0.5781 | 0.0002 | 0.5833 | 1.0000 |
| reranked_candidates | 12 | 1.0000 | 0.5812 | 0.0002 | 0.5833 | 1.0000 |

- Reviewer guardrails: judge reranker changes jointly against strict pocket fit, clash fraction, candidate validity, and leakage rather than a single scalar score.
- Leakage review status: `pass` with decision `current leakage weight passes reviewer bounds and preserves physically necessary cross-modality dependence on reviewed variants`.

## tight_geometry_pressure

- Artifact: `checkpoints/tight_geometry_pressure`
- Calibration method: `bounded_nonnegative_feature_target_covariance_v1`
- Active coefficients: bond_density_fit=0.9850, centroid_fit=0.0150
- Fitted candidate count: 3
- Decision: reranker improves at least one candidate-layer fit metric relative to inferred-bond candidates.

| Layer | Candidates | Valid | Centroid offset | Clash | Atom-seq diversity | Bond-topology diversity |
| --- | --- | --- | --- | --- | --- | --- |
| raw_rollout | 3 | 1.0000 | 1.4545 | 0.0000 | 0.3333 | 0.3333 |
| repaired_candidates | 3 | 1.0000 | 0.6237 | 0.0000 | 0.3333 | 0.3333 |
| inferred_bond_candidates | 3 | 1.0000 | 0.6237 | 0.0000 | 0.3333 | 0.6667 |
| deterministic_proxy_candidates | 1 | 1.0000 | 0.6132 | 0.0000 | 1.0000 | 1.0000 |
| reranked_candidates | 1 | 1.0000 | 0.6187 | 0.0000 | 1.0000 | 1.0000 |

- Reviewer guardrails: judge reranker changes jointly against strict pocket fit, clash fraction, candidate validity, and leakage rather than a single scalar score.
- Leakage review status: `pass` with decision `1 reviewed variant(s) regress pocket fit or chemistry; keep leakage at or below the current default until a sweep clears these blockers`.

## real_backend_gate

- Artifact: `checkpoints/real_backends`
- Calibration method: `bounded_nonnegative_feature_target_covariance_v1`
- Active coefficients: bond_density_fit=1.0000
- Fitted candidate count: 3
- Decision: reranker reshuffles candidate tradeoffs without a clean persisted win over the deterministic proxy on current reviewer surfaces.

| Layer | Candidates | Valid | Centroid offset | Clash | Atom-seq diversity | Bond-topology diversity |
| --- | --- | --- | --- | --- | --- | --- |
| raw_rollout | 3 | 1.0000 | 1.6526 | 0.0000 | 0.3333 | 0.3333 |
| repaired_candidates | 3 | 1.0000 | 0.7613 | 0.0000 | 0.3333 | 0.3333 |
| inferred_bond_candidates | 3 | 1.0000 | 0.7613 | 0.0000 | 0.3333 | 1.0000 |
| deterministic_proxy_candidates | 1 | 1.0000 | 0.7228 | 0.0000 | 1.0000 | 1.0000 |
| reranked_candidates | 1 | 1.0000 | 0.7753 | 0.0000 | 1.0000 | 1.0000 |

- Reviewer guardrails: judge reranker changes jointly against strict pocket fit, clash fraction, candidate validity, and leakage rather than a single scalar score.
- Leakage review status: `pass` with decision `current leakage weight passes reviewer bounds and preserves physically necessary cross-modality dependence on reviewed variants`.
