# Q15 Equivariant Geometry Flow

Date: 2026-04-29

Purpose: add a rigid-motion-correct geometry flow head without removing the existing MLP baseline or weakening controlled pocket conditioning.

## Head Boundary

All coordinate velocity heads implement `src/models/traits.rs::FlowMatchingHead`:

- input: `FlowState { coords, x0_coords, target_coords, t }`
- conditioning: `ConditioningState { topology_context, geometry_context, pocket_context, gate_summary }`
- output: `VelocityField { velocity: [num_atoms, 3], diagnostics }`

The config surface is `model.flow_velocity_head.kind`:

| Kind | Label | Claim role |
| --- | --- | --- |
| `geometry` | MLP baseline | translation-invariant baseline, rotation consistency is diagnostic only |
| `equivariant_geometry` | EGNN-style scalar-message head | exact rigid-motion equivariant velocity field within numerical tolerance |
| `atom_pocket_cross_attention` | gated atom-pocket attention ablation | controlled pocket-attention ablation, not an exact equivariance claim |

The default remains `geometry`. Switching the default requires the promotion criteria below.

## Equivariant Head

`src/models/flow/equivariant_velocity_head.rs::EquivariantGeometryVelocityHead` predicts vector velocities as scalar weights multiplied by relative vectors:

- self term: scalar function of invariant displacement/radius features times `coords - x0_coords`
- pair term: scalar function of pair distance and invariant conditioning times `coords_j - coords_i`
- bounded neighbor policy: `PairwiseGeometryConfig { radius, max_neighbors, residual_scale }`

The scalar message networks may use topology, geometry, pocket, gate, and time summaries. They do not directly project coordinate components into scalar logits, so translation and rotation behavior follows from the relative-vector construction.

## Molecular Branch Rigid-Motion Contract

The non-coordinate molecular flow branches now consume invariant geometry features:

- atom branches use radii, displacement norms, and coordinate dot products in the ligand-centered frame
- pair branches use distance, squared distance, inverse distance, and a constant channel
- atom type, bond, topology, and pocket-contact logits are expected to stay invariant under a shared rigid transform of `coords` and `x0_coords`
- vector velocity outputs are expected to rotate with the input when the `equivariant_geometry` head is selected

The MLP baseline remains available as an explicit baseline. Its rotation-consistency metric remains a diagnostic, not an exact equivariance claim.

## Ablation Configs

Smoke configs:

- `configs/q15_geometry_flow_mlp_smoke.research.json`
- `configs/q15_geometry_flow_equivariant_smoke.research.json`

For small/full runs, keep the same head settings and scale only data budget, `training.max_steps`, and checkpoint/evaluation intervals. Both head families are reported through `FlowHeadAblationDiagnostics.head_kind`, `ablation_label`, `equivariant_geometry_head`, and existing runtime fields:

- training: `training_history[].runtime_profile.step_time_ms`
- training: `training_history[].runtime_profile.examples_per_second`
- training: `training_history[].runtime_profile.memory_delta_mb`
- evaluation: `resource_usage.examples_per_second`
- evaluation: `resource_usage.memory_usage_mb`

## Promotion Criteria

Do not change the default from `geometry` to `equivariant_geometry` unless all of the following hold on the same dataset split and budget:

1. `cargo test flow_matching`, `cargo test models::flow`, and `cargo test rotation` pass.
2. Rotation consistency for `equivariant_geometry` is within test tolerance and scalar molecular branch rigid-motion tests pass.
3. Validation quality is not worse than the MLP baseline on finite forward fraction and strict pocket-fit proxies.
4. Throughput and memory remain within the configured performance gates or are explicitly accepted as a research tradeoff.
5. The report keeps `head_kind` and `ablation_label` visible, so MLP and equivariant runs are not merged.

## Verification

Relevant tests:

- `models::flow::equivariant_velocity_head::tests::equivariant_geometry_head_predicts_atomwise_velocity`
- `models::flow::equivariant_velocity_head::tests::equivariant_geometry_head_is_translation_invariant`
- `models::flow::equivariant_velocity_head::tests::equivariant_geometry_head_rotates_velocities_consistently`
- `models::flow::molecular::tests::molecular_flow_scalar_branches_are_rigid_motion_invariant`
- `models::system::tests::flow_matching_selects_equivariant_velocity_head_through_config`
- `experiments::unseen_pocket::tests::cross_attention_velocity_head_ablation_labels_are_stable`
