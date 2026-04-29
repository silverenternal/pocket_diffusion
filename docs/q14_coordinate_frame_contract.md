# Q14 Coordinate Frame Contract

## Contract

The active Rust data path stores ligand, pocket, and generated candidate
coordinates in ligand-centered model-frame coordinates. `coordinate_frame_origin`
is the translation needed to reconstruct source-frame coordinates for artifacts
and backend scoring.

For de novo claim paths, model conditioning does not read target ligand topology
or geometry. The de novo path recenters pocket context by pocket centroid,
constructs topology/geometry from the generated scaffold, and records
`pocket_centroid_centered_conditioning_no_target_ligand_frame`.

## Translation

Uniform source-coordinate translation must not change model-frame ligand
geometry, pocket tensors, de novo conditioning encodings, or flow `x0`
construction. The source-frame shift is carried only by
`coordinate_frame_origin`.

Regression coverage:

- `source_translation_preserves_ligand_centered_model_frame`
- `de_novo_conditioning_and_flow_x0_ignore_source_frame_translation`

## Rotation

Rotation augmentation is treated as a data augmentation and diagnostic surface,
not as a strict equivariance claim. The parser applies the same rotation to
ligand and pocket coordinates when enabled and records attempted/applied counts.
Flow-head rotation consistency remains a finite diagnostic:

- `rotation_augmentation_changes_geometry_but_preserves_distances`
- `rotation_consistency_metric_is_finite_for_geometry_head`

Persisted summaries set
`coordinate_frame.rotation_consistency_role=diagnostic_not_exact_equivariance_claim`.

## Persisted Provenance

Training and unseen-pocket experiment summaries persist a top-level
`coordinate_frame` block with:

- `coordinate_frame_contract`
- `coordinate_frame_artifact_contract`
- `coordinate_frame_origin_valid_examples`
- `ligand_centered_coordinate_frame_examples`
- `pocket_coordinates_centered_upstream`
- `source_coordinate_reconstruction_supported`
- `rotation_augmentation_attempted_examples`
- `rotation_augmentation_applied_examples`
- `rotation_consistency_role`

Candidate and generation-layer artifacts continue to persist
`coordinate_frame_origin` beside generated coordinates so downstream metrics can
separate model-frame coordinates from source-frame reconstruction.
