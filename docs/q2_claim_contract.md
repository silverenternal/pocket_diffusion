# Q2 Claim Contract

Q2 claim-facing values must come from machine-generated artifacts. Placeholder
or narrative-only approximate values are not allowed in claim artifacts.

Geometry-only flow configs are geometry-first flow matching and must not be
described as full molecular flow. Full molecular flow wording requires the
explicit five-branch contract: geometry, atom-type, bond, topology, and
pocket/context flow.

Vina and GNINA values are score-only backend outputs. They are not experimental
binding affinities.

Metric provenance labels are defined in
`configs/drug_metric_artifact_manifest.json#claim_provenance_contract`.
`heuristic`, `backend_supported`, `docking_supported`, `experimental`, and
`unavailable` are distinct evidence states. `unavailable` evidence must use
`value=null`; zero is reserved for measured or computed zero-valued metrics.

## Layer Groups

| Group | Layers | Claim Use |
| --- | --- | --- |
| raw model-native | `raw_flow`, `raw_rollout`, `no_repair` | native model evidence |
| constrained sampling | `constrained_flow` | constrained sampling evidence |
| postprocessing | `repaired`, `inferred_bond`, `deterministic_proxy`, `reranked`, `centroid_only`, `clash_only`, `bond_inference_only`, `full_repair` | postprocessing evidence only |

Tables must not collapse raw model-native rows with constrained, repaired, or
inferred-bond, deterministic-proxy, or reranked rows. Public-baseline dominance
claims require matched-budget multi-seed public-baseline evidence and are not
supported by the current Q2 artifacts.
