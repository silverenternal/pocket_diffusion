# Q7 Core Ablation Matrix

The core matrix is defined in `configs/q7_core_ablation_matrix.json`. It is
anchored to the reviewer-scale Q7 training preset and keeps
`conditioned_denoising` as the claim-bearing objective.

The matrix separates raw model-native generation from constrained, repaired,
bond-inferred, reranked, and backend-scored layers. Compare variants using the
`model_design` block in split evaluation artifacts before citing downstream
postprocessing gains.

Required design axes:

| Axis | Base | Variant |
| --- | --- | --- |
| Generation mode | `target_ligand_denoising` | `generation_mode_ligand_refinement`; `generation_mode_pocket_only_initialization_baseline`; `de_novo_full_molecular_flow` |
| Topology encoder | `message_passing` | `topology_encoder_lightweight`; `topology_encoder_typed_message_passing` |
| Geometry operator | `pair_distance_kernel` | `geometry_operator_raw_coordinate_projection`; `geometry_operator_local_frame_pair_message` |
| Pocket encoder | `local_message_passing` | `pocket_encoder_feature_projection`; `pocket_encoder_ligand_relative_local_frame` |
| Slots | enabled | `disable_slots` |
| Slot count | configured count | `slot_count_reduced` |
| Slot attention masking | active-slot masks enabled | `slot_attention_masking_disabled` |
| Leakage diagnostics | enabled | `disable_leakage` |
| Leakage gradient | configured `delta_leak` | `leakage_penalty_disabled` |
| Redundancy objective | configured `beta_intra_red` | `redundancy_disabled` |
| Modality focus | all modalities visible downstream | `topology_only`; `geometry_only`; `pocket_only` |
| Semantic probes | enabled | `disable_probes` |
| Gate regularization | configured `eta_gate` | `gate_sparsity_disabled` |
| Gate scale | configured gate temperature | `interaction_gate_temperature_high` |
| Interaction negative control | directed gated cross-attention | `direct_fusion_negative_control` |
| Decoder conditioning | `local_atom_slot_attention` | `decoder_conditioning_mean_pooled` |
| Candidate postprocessing | repair enabled | `disable_candidate_repair` |
| Training schedule | staged warm-start | `staged_schedule_disabled` |

The pocket-only initialization baseline is executable and removes target ligand
atom types and coordinates from decoder initialization by using a configured
atom-count prior plus pocket-centroid offsets. It validates only with the
shape-safe `surrogate_reconstruction` primary objective in the current stack.
It is still a low-claim baseline.
The de novo row is executable only under the full molecular flow contract:
`generation_mode=de_novo_initialization`, flow-matching backend,
`flow_matching.geometry_only=false`, and all geometry/atom-type/bond/topology/
pocket-context branches enabled. Reviewer-scale configs may execute refinement,
pocket-only mode, and de novo mode boundaries, but must keep raw native flow
metrics separate from repaired/reranked layers.

Use quality, geometry, efficiency, slot usage, gate usage, and leakage fields
together. A variant that improves processed quality while hurting raw quality
should be reported as postprocessing-dependent, not as a stronger native model.
The direct-fusion row is only a negative control: it forces directed paths open
for comparison and must not replace the controlled gated interaction design.
The modality-focus rows are also negative controls: they zero non-selected slot
branches after separate encoders and before downstream interaction/decoder use,
so they test dependence on each modality without collapsing the architecture.
The staged-schedule row activates configured objective families from step zero
and should be read as a training-procedure stress test, not as a new model class.
