# Q13 Model Boundary Design

Date: 2026-04-29

## Boundary Goal

Keep topology, geometry, and pocket/context encoders structurally separate while
making ownership and ablation surfaces easier to audit. The migration groups
components by responsibility without changing tensor schemas or replacing
directed gated interaction with unrestricted fusion.

## Stack Responsibilities

| Stack | Responsibility | Current Files |
| --- | --- | --- |
| `EncoderStack` | Owns `TopologySemanticBranch`, `GeometrySemanticBranch`, and `PocketSemanticBranch`; each branch keeps its own encoder and slot decomposer. | `src/models/system/impl.rs`, `src/models/topo_encoder.rs`, `src/models/geo_encoder.rs`, `src/models/pocket_encoder.rs`, `src/models/slot_decomposition.rs` |
| `InteractionStack` | Owns the directed `CrossModalInteractionBlock`; all cross-modality exchange remains path-specific and gated. | `src/models/interaction.rs`, `src/models/interaction/*`, `src/models/cross_attention.rs` |
| `GeneratorStack` | Owns the decoder, coordinate flow velocity head, and full molecular flow head. | `src/models/decoder.rs`, `src/models/flow_matching.rs`, `src/models/flow/*` |
| `ProbeStack` | Owns semantic and leakage probe heads. | `src/models/probe_heads.rs`, `src/losses/probe.rs`, `src/losses/leakage.rs` |
| `SystemForwardAssembler` | Forward-phase helper boundary: optimizer-facing records are built before detached rollout records. | `src/models/system/forward.rs`, `src/models/system/types.rs` |

## Migration Checklist

1. Add stack wrapper structs and move `Phase1ResearchSystem` ownership into
   grouped fields.
2. Update internal call sites to use grouped fields.
3. Add `OptimizerForwardRecord` so trainer-facing tensors can be requested
   before rollout diagnostics are built.
4. Keep `ResearchForward` schema compatible by assembling the same final
   `GenerationForward` and rollout records for evaluation paths.
5. Add regression tests for stack ownership, mode contracts, and de novo
   target-ligand isolation.

## Invariants

- There are still exactly three active modality encoder branches.
- Cross-modality paths remain explicit:
  topology-from-geometry, topology-from-pocket, geometry-from-topology,
  geometry-from-pocket, pocket-from-topology, and pocket-from-geometry.
- Direct fusion is allowed only through the explicit negative-control
  interaction mode and remains claim-limited.
- Rollout records are detached diagnostics unless a future task introduces a
  bounded tensor-preserving rollout objective.
