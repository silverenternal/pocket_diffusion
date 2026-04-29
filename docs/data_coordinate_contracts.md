# Data And Coordinate Contracts

The active data path stores ligand and pocket tensors in a ligand-centered model
frame. `DatasetValidationReport.coordinate_frame_contract` records this as:

`ligand_centered_model_coordinates_with_coordinate_frame_origin`

Candidate artifacts use the same convention: `candidate.coords` are
ligand-centered model-frame coordinates, and `coordinate_frame_origin` is the
translation needed to reconstruct source-frame coordinates. `generation_layers`
and candidate-metric JSONL artifacts carry this coordinate-frame provenance.
Training and unseen-pocket experiment summaries also persist a top-level
`coordinate_frame` provenance block; see `docs/q14_coordinate_frame_contract.md`.

## Target-Context Leakage

Because pocket coordinates are centered using the ligand centroid, pocket/context
tensors can depend on target-ligand-derived centering. This is acceptable for
target-ligand denoising, ligand refinement, and flow refinement contracts, where
the target ligand is explicitly part of the supervision or refinement setup.

For pocket-only baseline claim modes, this dependency is unsafe unless the data
are rebuilt in a pocket-only coordinate frame or rejected by config. The de novo
execution path now recenters pocket features before scaffold construction and
uses generated topology/geometry scaffold features for conditioning; target
ligand tensors remain training supervision only. Dataset validation still
reports the source dependency so claim artifacts can distinguish parser-level
centering from model execution semantics:

- `target_ligand_context_dependency_detected`
- `target_ligand_context_dependency_allowed`
- `target_ligand_context_dependency_rejected`
- `target_ligand_context_leakage_warnings`

Claim-bearing pocket-only and de novo configs can set
`data.quality_filters.reject_target_ligand_context_leakage=true` to fail the
load when retained pocket/context tensors still depend on target-ligand
centering.

## Split Quality Gates

`SplitReport.quality_checks` records weak held-out family counts, pocket-size
and atom-count skew, measurement-family skew, suspicious distribution collapse,
and unavailable metadata fields. `SplitStats` now separates proxy histograms for
protein family, pocket family, ligand scaffold, affinity measurement family, and
pocket-size bins so synthetic datasets can report `unavailable` instead of
pretending those fields were measured. Configs can additionally require minimum
held-out diversity with:

- `data.quality_filters.min_validation_protein_families`
- `data.quality_filters.min_test_protein_families`
- `data.quality_filters.min_validation_pocket_families`
- `data.quality_filters.min_test_pocket_families`
- `data.quality_filters.min_validation_ligand_scaffolds`
- `data.quality_filters.min_test_ligand_scaffolds`
- `data.quality_filters.min_validation_measurement_families`
- `data.quality_filters.min_test_measurement_families`

Training and unseen-pocket experiment entrypoints enforce these configured
thresholds before claim-bearing runs proceed.
