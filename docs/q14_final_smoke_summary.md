# Q14 Final Compact Smoke Summary

Date: 2026-04-29

This directory is compact smoke evidence, not benchmark-quality molecular generation evidence. The runs use the synthetic loader, disabled external chemistry/docking backends, and two training steps per surface. Claim-facing interpretation must use the raw-native and raw-vs-processed fields in each `claim_summary.json`.

## Artifact Root

`checkpoints/q14_final_smoke`

Each surface contains:

- `training_summary.json`
- `experiment_summary.json`
- `claim_summary.json`
- `generation_layers_validation.json`
- `generation_layers_test.json`
- `repair_case_audit_validation.json`
- `repair_case_audit_test.json`
- `frozen_leakage_probe_audit.json`

## Smoke Results

| Surface | Objective | Generation Mode | Val finite | Test finite | Nonfinite gradient tensors | Leakage audit | Matching policy | Claim-safe matching |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `conditioned_denoising` | `conditioned_denoising` | `target_ligand_denoising` | 1.0 | 1.0 | 0 | `ok` | `pad_with_mask` | false |
| `geometry_flow` | `flow_matching` | `target_ligand_denoising` | 1.0 | 1.0 | 0 | `insufficient_data` | `pad_with_mask` | false |
| `de_novo_full_flow` | `flow_matching` | `de_novo_initialization` | 1.0 | 1.0 | 0 | `insufficient_data` | `hungarian_distance` | true |

## Raw-Native And Processed Evidence

| Surface | Raw valid | Raw contact | Raw clash | Raw-native gate | Processed layer | Centroid delta | Clash delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `conditioned_denoising` | 1.0 | 1.0 | 0.1667 | pass | `reranked_candidates` | -0.6222 | -0.1667 |
| `geometry_flow` | 1.0 | 1.0 | 0.0 | pass | `inferred_bond_candidates` | -0.6874 | 0.0 |
| `de_novo_full_flow` | 1.0 | 1.0 | 1.0 | pass | `inferred_bond_candidates` | -0.3319 | -1.0 |

The `de_novo_full_flow` smoke records all five molecular-flow branches and a non-index target matching policy:

- `enabled_flow_branches`: `geometry`, `atom_type`, `bond`, `topology`, `pocket_context`
- `target_alignment_policy`: `hungarian_distance`
- `target_matching_claim_safe`: true
- `claim_gate_reason`: `full_molecular_flow_contract_satisfied`

The detailed matching provenance is in:

- `checkpoints/q14_final_smoke/de_novo_full_flow/experiment_summary.json`
- `training_history[].losses.primary.branch_schedule.entries[]`
- `target_matching_policy`, `target_matching_coverage`, `target_matching_mean_cost`, and matched/unmatched counts

## Claim Boundary

Repaired-layer improvements are postprocessing evidence. They must not be cited as raw generation evidence unless the corresponding raw-native metric also supports the statement. The authoritative per-run boundary is:

- `claim_summary.json.raw_native_evidence`
- `claim_summary.json.processed_generation_evidence`
- `claim_summary.json.postprocessing_repair_audit`
- `generation_layers_*.json.repair_case_audit`

The frozen leakage probe audit is present for all three smokes. The compact synthetic splits are too small for held-out calibration in two surfaces, so no-leakage claims remain unsupported by this smoke evidence.
