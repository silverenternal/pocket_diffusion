# Q14 Pocket Flow Branch Contract

## Purpose

The full molecular flow `pocket_context` branch must not claim pocket-conditioned
generation strength from reconstructing the same internal `pocket_context` tensor
that conditions the generator. That reconstruction can remain a diagnostic for
context drift, but the claim-bearing pocket branch target is an interaction
profile between the generated ligand draft and the inference-available pocket.

This contract defines the branch target family for the next implementation
phase. It keeps target-ligand information as training labels only and keeps the
conditioning path restricted to inference-available pocket context plus the
generated ligand state.

## Branch Target

The claim-bearing pocket branch target is `pocket_interaction_profile`.

For each generated ligand atom row with active target matching, the branch should
predict a compact profile with these target groups:

| Target group | Shape | Training label source | Inference-time inputs | Primary use |
| --- | --- | --- | --- | --- |
| nearest-pocket distance bins | `[num_atoms, distance_bin_count]` | clean ligand target coordinates and pocket coordinates | generated ligand coordinates and pocket coordinates | geometric pocket fit |
| contact likelihood | `[num_atoms]` | whether any pocket atom is within the contact cutoff of the matched target atom | generated ligand coordinates and pocket coordinates | contact formation |
| ligand-pocket role interaction profile | `[num_atoms, role_pair_count]` | ligand atom chemistry role labels crossed with nearest/contacting pocket role labels | generated ligand atom logits or sampled atom type, generated coordinates, and pocket role features | pharmacophore compatibility |
| pocket summary compatibility | `[num_atoms, compatibility_dim]` or pooled `[compatibility_dim]` | derived labels such as hydrophobic/polar/charged/aromatic compatibility near the matched target atom | generated ligand state and pocket role/context features | local semantic pocket fit |

The first implementation may use any nonempty subset of these groups, but it
must include at least one differentiable ligand-pocket compatibility target that
changes when ligand coordinates move relative to the pocket.

## Allowed Inputs And Labels

Optimizer-facing prediction may use:

- generated ligand coordinates at the current flow state
- generated or current draft atom-type logits/tokens
- generated bond/topology logits when needed for local context
- pocket coordinates
- pocket atom features and pocket chemistry role features
- gated pocket/context encodings produced from inference-available pocket input

Training labels may use:

- matched clean ligand target coordinates
- matched target atom types or chemistry role labels
- ligand-pocket distances computed from matched target coordinates and pocket
  coordinates
- contact and role-interaction labels derived from the target complex

Training labels must not be fed back into the conditioning state, rollout
initialization, context refresh, or inference-time generation path.

## Matching And Masks

The branch must reuse the `TargetMatchingResult` selected for the molecular flow
record:

- row labels are indexed by matched target atom rows
- unmatched generated rows are masked out
- coverage, unmatched counts, and matching cost are reported with the branch
- bond/topology pair targets continue to use the same row matching map

If generated atom count exceeds target atom count, the branch must zero or ignore
unmatched rows under the matched atom mask. If target atom count exceeds generated
atom count, unselected target rows are reported as unmatched targets and do not
create synthetic generated rows.

## Loss Contract

The pocket branch loss should be a weighted sum of shape-safe terms:

- distance-bin cross entropy or ordinal regression under the matched atom mask
- contact binary cross entropy under the matched atom mask
- role-interaction binary or categorical loss under the matched atom mask
- optional compatibility regression/classification under the matched atom mask

The current `pocket_context_reconstruction` loss is demoted to
`context_drift_diagnostic` unless and until it is explicitly combined with an
interaction-profile target. It is not sufficient evidence for claim-bearing
pocket-conditioned molecular flow.

## Artifact Contract

Training and evaluation artifacts should expose:

- `pocket_branch_target_family = "pocket_interaction_profile"`
- enabled target groups, for example `nearest_distance_bins` and
  `contact_likelihood`
- distance/contact cutoffs and bin edges
- matching policy, matched coverage, unmatched row counts, and matching cost
- whether `pocket_context_reconstruction` contributed only as a diagnostic
- pocket branch weighted and unweighted losses

Claim summaries may describe full molecular flow as pocket-conditioned only when
the pocket branch includes an interaction-profile target and the run passes the
existing full-flow branch and non-index target matching gates.

## Implementation Boundary

`Q14-P4-01` is this design contract only. Implementation belongs to
`Q14-P4-02`, where `FullMolecularFlowHead` and `molecular_flow_losses` should add
the interaction-profile head and replace or demote the current
`pocket_context_reconstruction` objective.
