# Q13 Coordinate Frame Audit

## Contract

Model-space ligand geometry is ligand-centered for target-ligand training
examples. De novo conditioning must not use that target-ligand frame; it
re-centers pocket coordinates by the pocket centroid before scaffold
initialization and encoder conditioning.

## Path Audit

| Stage | Code Path | Frame | Target-Ligand Dependency | Claim Status |
| --- | --- | --- | --- | --- |
| Parser pocket extraction | `src/data/parser/pdb.rs`, `src/data/parser/discovery.rs` | Source structure coordinates; local pocket may be selected around ligand center for dataset construction. | Uses target ligand only to define a supervised complex pocket. | Training/dataset construction only. |
| Feature construction | `src/data/features/builders.rs` | Ligand geometry is centered by ligand centroid; pocket features can be centered against the same model origin. | Ligand-centered model frame is target-dependent. | Marked training supervision only for de novo claims. |
| Dataset validation | `src/data/dataset/quality.rs`, `src/data/parser/manifest.rs` | Records finite `coordinate_frame_origin`, ligand-centered checks, artifact reconstruction support, and target-context dependency flags. | Dependency is reported and can be rejected by quality filters. | Claim-bearing de novo configs must use rejection/filter evidence. |
| Batch collation | `src/data/batch.rs` | Carries already-built model-space tensors and masks. | No new centering transform. | Inherits dataset contract. |
| De novo encoder input | `src/models/system/impl.rs` | `de_novo_conditioning_modalities` builds topology/geometry from a generated scaffold and calls `pocket_centered_features`. | Does not read target atom types, target topology, target coordinates, or decoder supervision for conditioning. | Claim-eligible pocket-centroid frame. |
| De novo flow objective | `src/models/system/flow_training.rs` | Conditioning remains pocket-centroid centered; `x1` target tensors are read only for optimizer supervision. | Target tensors affect training targets, not conditioning inputs. | Training supervision only. |
| Rollout artifact | `src/models/traits.rs`, `src/models/system/rollout.rs`, `src/models/system/flow_training.rs` | `conditioning_coordinate_frame` records the active conditioning frame. | De novo emits `pocket_centroid_centered_conditioning_no_target_ligand_frame`. | Required provenance for de novo evidence. |
| Candidate export/evaluation | `src/models/evaluation/evaluators.rs` | Candidate records carry `coordinate_frame_origin` for source-frame reconstruction and pocket-fit metrics. | Evaluation may compare to known targets only as held-out scoring. | Artifact-side provenance retained. |

## Decision

Target-ligand-centered coordinates are allowed for supervised denoising,
refinement, dataset parsing, and optimizer targets. They are not allowed as de
novo conditioning evidence. The de novo optimizer path is regression-tested by
perturbing target atom types, target topology, target geometry, and decoder
supervision while holding pocket tensors fixed.
