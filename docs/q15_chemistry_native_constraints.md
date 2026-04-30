# Q15 Chemistry-Native Constraint Contract

Purpose: make chemical plausibility optimizer-visible before deterministic graph constraints, coordinate repair, reranking, or backend scoring.

## Optimizer-Facing Guardrails

`src/losses/consistency.rs::ChemistryGuardrailAuxLoss` now emits structured chemistry objectives:

| Field | Source | Gradient Path | Interpretation |
| --- | --- | --- | --- |
| `valence_guardrail` | generated atom-type probabilities, bond-existence probabilities, bond-order probabilities | atom logits, bond logits, bond-type logits | total valence budget penalty |
| `valence_overage_guardrail` | same as above | diagnostic subterm | expected valence above element-specific capacity |
| `valence_underage_guardrail` | same as above | diagnostic subterm | expected valence below conservative minimum connectivity |
| `bond_length_guardrail` | generated coordinates and likely generated bonds | coordinate velocity proxy and bond logits | bond-length-by-type margin |
| `nonbonded_distance_guardrail` | generated coordinates and likely absent bonds | coordinate velocity proxy and bond logits | non-bonded short-distance margin |
| `angle_guardrail` | generated coordinates and two-hop generated bond probabilities | coordinate velocity proxy and bond logits | broad local-angle plausibility proxy |

When full molecular flow logits are available, guardrails use the model-native generated atom and bond distributions. Legacy decoder/reference-aligned fallback remains available for configurations that do not emit molecular flow branches.

## Staging

Defaults preserve existing behavior because the new loss weights default to zero.

Config weights:

- `training.loss_weights.upsilon_valence_guardrail`
- `training.loss_weights.phi_bond_length_guardrail`
- `training.loss_weights.chi_nonbonded_distance_guardrail`
- `training.loss_weights.psi_angle_guardrail`

Warmup gates:

- `training.chemistry_warmup.valence_guardrail_start_stage`
- `training.chemistry_warmup.bond_length_guardrail_start_stage`
- `training.chemistry_warmup.nonbonded_distance_guardrail_start_stage`
- `training.chemistry_warmup.angle_guardrail_start_stage`

Auxiliary objective reporting exposes `valence_guardrail`, `bond_length_guardrail`, `nonbonded_distance_guardrail`, and `angle_guardrail` as separate families. Valence overage and underage are logged as submetrics.

## Raw Graph Evaluation

Candidate-layer metrics keep raw model-native graph quality separate from constrained and repaired layers:

- `native_graph_valid_fraction`
- `native_valence_violation_fraction`
- `native_disconnected_fragment_fraction`
- `native_bond_order_conflict_fraction`
- `native_graph_repair_delta_mean`

`raw_flow`/`raw_rollout` metrics remain model-native evidence. `constrained_flow`, `repaired`, inferred-bond, reranked, and backend layers remain postprocessed evidence and must not overwrite raw-native claims.

## Verification

Targeted coverage:

- `cargo test losses`
- `cargo test native_graph`
- `cargo test target_matching`
- `cargo test scheduler`
- `cargo test trainer`
- `cargo test training`
- `cargo test evaluation`
- `cargo test experiments`
