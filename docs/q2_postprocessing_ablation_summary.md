# Q2 Postprocessing Ablation Summary

Ablation layers are score_only postprocessing evidence. no_repair is an unchanged-coordinate baseline; centroid_only, clash_only, bond_inference_only, and full_repair must not be reported as native public-baseline capability.

## Diagnosis

- dominant_layer: `full_repair`
- conclusion: The combined repair path degrades more than any isolated step.
- full_repair_promoted: False

## Component Findings

- `coordinate_movement`: primary_degradation_source; centroid_only changes coordinates without changing bonds; positive Vina/GNINA affinity deltas mean worse score_only docking.
- `clash_handling`: not_primary_degradation_source; clash_only keeps the raw centroid and bond payload while applying local declash moves.
- `bond_inference`: chemistry_regression_without_docking_degradation; bond_inference_only keeps raw coordinates but changes inferred_bonds/molecular representation.
- `docking_input_conversion`: coverage_issue_not_primary_score_degradation; no_repair is an unchanged-coordinate copy scored through the same docking-input path; missing Vina rows are explicit command failures, while GNINA coverage is complete.
- `full_repair`: not_promoted; full_repair combines centroid-anchored repair, declash, envelope clamp, and bond inference and is worse than no_repair on both score_only affinity backends.
- `reranking`: not_needed_to_explain_degradation; The five-layer ablation shows severe degradation before deterministic reranking is applied; legacy reranked rows remain postprocessing evidence and are not promoted.

## Layer Summary

| Layer | Candidates | Vina Cov | GNINA Cov | dVina | dGNINA | dQED | dSA | dClash | dContact | dCentroid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_repair | 300 | 0.8967 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| centroid_only | 300 | 1 | 1 | 148.3 | 148.4 | 0 | 0 | 0.2064 | -0.03675 | -15.92 |
| clash_only | 300 | 0.8967 | 1 | -0.02936 | -0.05779 | 0 | 0 | -0.002659 | 0 | 0.0004756 |
| bond_inference_only | 300 | 0.8967 | 1 | 0 | 0 | -0.01309 | 0.6798 | 0 | 0 | 0 |
| full_repair | 300 | 1 | 1 | 159.1 | 159.2 | -0.001032 | 0.0415 | 0.1007 | -0.03512 | -10.84 |

## Failure Reasons

- `no_repair`: missing_vina_score: 31, vina_command_failed: 31
- `centroid_only`: none: 0
- `clash_only`: missing_vina_score: 31, vina_command_failed: 31
- `bond_inference_only`: missing_vina_score: 31, vina_command_failed: 31
- `full_repair`: none: 0

## Guardrail

Full repair is not promoted unless score_only docking improves without unacceptable QED/SA, clash, or contact regressions.
