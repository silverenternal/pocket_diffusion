# Q2 Conditioning Compression Audit

The system keeps topology, geometry, and pocket encoders structurally separate. Token-level representations are decomposed into fixed slot tensors, and directed cross-modal interaction uses gated attention over slots.

The compression risk appears at the default baseline flow head. `GeometryFlowMatchingHead` calls `mean_or_zeros` on topology, geometry, and pocket contexts, concatenates the three modality means, and broadcasts that summary to all atoms. This means residue-level or pocket-slot locality can be preserved upstream but erased before velocity prediction.

## Locality Trace

| Stage | Representation | Locality |
| --- | --- | --- |
| Encoders | token embeddings and pooled embeddings | token-level preserved |
| Soft slots | `[num_slots, hidden_dim]` with slot weights | slot-level preserved |
| Gated cross-modal attention | directed slot-to-slot updates | slot-level preserved |
| Generation state | merged modality slot contexts | slot-level preserved |
| Baseline flow head | mean topology, geometry, and pocket summaries | collapsed to modality means |

Mean pooling that affects velocity prediction is in `src/models/flow_matching.rs`: topology, geometry, and pocket contexts are each reduced to one vector. `src/models/cross_attention.rs` also uses mean summaries for scalar gate decisions, but the attended values remain slot-level there.

## Flow-Head Status

The promoted default remains `model.flow_velocity_head.kind = "geometry"`. A local atom-to-pocket-slot attention head is implemented and config-selectable with `model.flow_velocity_head.kind = "atom_pocket_cross_attention"`:

`gate(atom,pocket) * Attention(Q_atom, K_pocket_slot, V_pocket_slot)`

This preserves controlled interaction and avoids collapsing the three encoders into a single fusion encoder. It should be reported as an ablation variant unless a claim-bearing config explicitly selects it. Pairwise ligand geometry messages are similarly config-selectable through `model.pairwise_geometry.enabled` and do not create topology or bond flow.
