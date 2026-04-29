# Interaction Preference Alignment

## Boundary

The preference-alignment layer is additive over the existing generation method platform. The current `conditioned_denoising` path is the first compatible generator, not the only future method. Flow matching, diffusion, autoregressive, energy-guided, and external-wrapper methods can emit the same candidate layers before preference extraction.

Preference alignment is not a generator-level DPO, diffusion, flow-matching, reinforcement-learning, or adversarial trainer in this phase. The current implementation builds evidence objects over candidate pairs while preserving the separate topology, geometry, and pocket encoders plus gated cross-modal interaction.

## Interfaces

- `InteractionProfile`: one candidate-level record with `schema_version`, candidate identifiers, layer/method provenance, feature values, feature provenance, backend coverage, and missing-evidence flags.
- `InteractionProfileExtractor`: replaceable extraction boundary for heuristic-only, RDKit-backed, pocket-backend-backed, docking-backed, or future experimental profiles.
- `PreferencePair`: winner/loser pair with `schema_version`, source, structured reason codes, strength, signed feature deltas, hard constraints, soft preference flags, and evidence coverage.
- `PreferenceDatasetBuilder`: deterministic builder that converts comparable profiles into auditable pairs.
- `PreferenceReranker`: profile-level ranking interface for future preference-aware candidate layers.
- `PreferenceTrainer`: reserved future hook only; current code does not enable generator-level DPO/RL training.

## Preference Sources

- `rule_based`: heuristic and hard-rule pair construction from candidate payloads.
- `backend_based`: pair construction supported by external backend metrics such as RDKit or pocket-contact command adapters.
- `human_curated`: reserved for human labels; unavailable unless human-curated data are loaded.
- `future_docking`: reserved for docking-backed preferences; unavailable unless Vina, gnina, or equivalent docking coverage passes.
- `future_experimental`: reserved for experimental outcome labels; unavailable unless experimental labels are loaded.

Preference weights may help construct pairs or rerank profiles, but weights are not the sole reported evidence. Artifacts must preserve reason codes, signed feature deltas, and evidence-source labels.

## Schema Contracts

All preference artifacts are versioned:

- `InteractionProfile.schema_version >= 1`
- `PreferencePair.schema_version >= 1`
- `preference_profiles_<split>.json.schema_version >= 1`
- `preference_pairs_<split>.json.schema_version >= 1`
- `preference_reranker_summary.json.schema_version >= 1`
- `method_comparison.preference_alignment.schema_version >= 1`

Missing preference artifacts mean preference evidence is unavailable, not failed alignment. Existing `claim_summary.json`, `generation_layers_<split>.json`, `evidence_bundle.json`, and `paper_claim_bundle.md` remain readable when preference fields are absent. Schema bumps are required when field meaning changes, required fields are added, reason-code semantics change, or source labels change.

## Feature Provenance Map

| Feature | Current source | Provenance | Missing semantics |
| --- | --- | --- | --- |
| `pocket_contact_fraction` | candidate/pocket geometry proxy | `heuristic_proxy` | absent candidate geometry means zero proxy support |
| `clash_fraction` | candidate non-bonded distance proxy | `heuristic_proxy` | absent geometry means unavailable/zero hard support |
| `centroid_offset` | candidate centroid to pocket centroid | `heuristic_proxy` | absent coordinates mean unavailable |
| `strict_pocket_fit_score` | contact, centroid, coverage, clash proxy | `heuristic_proxy` | proxy-only; not docking evidence |
| `atom_coverage_fraction` | candidate atoms in expanded pocket envelope | `heuristic_proxy` | absent coordinates mean zero coverage |
| `backend_missing_structure_fraction` | command backend metric | `external_backend` | unavailable unless backend reports it |
| `rdkit_valid`, `rdkit_sanitized`, `rdkit_unique_smiles` | RDKit command backend | `external_backend` | unavailable unless RDKit backend reports it |
| `bond_density_fit` | inferred-bond density proxy | `heuristic_proxy` | proxy-only |
| `valence_sane` | inferred-bond valence proxy | `heuristic_proxy` | proxy-only |
| `hydrophobic_contact_proxy` | planned pocket chemistry proxy | `future_optional` | explicitly unavailable |
| `hydrogen_bond_proxy` | planned pocket chemistry proxy | `future_optional` | explicitly unavailable |
| `key_residue_contact_proxy` | planned residue-aware contact proxy | `future_optional` | explicitly unavailable |

## Q2 Interaction Profile Metric Plan

The Q2 metric plan is machine-readable in `configs/q2_interaction_profile_metric_plan.json`.
It keeps the current fields proxy-labeled unless a future backend supplies residue chemistry
or experimental annotations. The required candidate-level fields are `candidate_id`,
`example_id`, `protein_id`, `method_id`, `layer`, feature values, feature provenance, and
backend coverage.

Q2 core fields are `hydrogen_bond_proxy`, `hydrophobic_contact_proxy`,
`residue_contact_count`, `clash_fraction`, `pocket_contact_fraction`, and
`centroid_offset`. `key_residue_interaction_proxy` is reserved and remains unavailable
unless receptor residue ids and configured key-residue lists are present. Unavailable
features must remain null or absent with explicit status; they must not be rendered as
zero-valued evidence.

Existing calibrated reranker features are reusable for deterministic profile scoring, but are insufficient as preference evidence by themselves because they do not preserve pair reasons, signed deltas, or source labels.

## Module Ownership

- `src/models/preference.rs`: preference data models, extractor traits, rule-based builder, and preference-aware reranker interface.
- `src/models/traits.rs`: existing generation and backend contracts remain the candidate-production boundary.
- `src/models/evaluation/*`: candidate and backend metric helpers remain evaluation utilities.
- `src/experiments/unseen_pocket/evaluation/*`: artifact aggregation and split-level summaries own future profile/pair emission.
- `src/config/types/*`: `preference_alignment` config remains default-off and ablation-friendly.
- Reviewer claim and evidence tools may summarize preference artifacts additively, but must not infer human, docking, or experimental preference from heuristic-only evidence.

## Artifact Plan

Profile extraction can run on `raw_rollout`, `repaired_candidates`, `inferred_bond_candidates`, `deterministic_proxy_candidates`, `reranked_candidates`, and a future `preference_aware_candidates` layer. Existing generation-layer semantics are preserved when profiles are absent.

`preference_pairs_<split>.json` should include only same-example, same-protein comparisons. Hard constraints include invalid payloads, excessive clash, insufficient strict pocket fit, and optional valence sanity. Soft preferences include better pocket contact, lower centroid offset, better atom coverage, better bond-density fit, and backend-supported chemistry fields when available.

## Backend Adapter Contract (Candidate-Level vs Aggregate-Only)

External backend adapters may emit either:

1. aggregate-only metrics (legacy-compatible), or
2. candidate-level rows keyed by `candidate_id` / `example_id` / `protein_id`.

Preferred structured payload:

```json
{
  "schema_version": 1,
  "aggregate_metrics": {
    "schema_version": 1,
    "backend_examples_scored": 128,
    "rdkit_sanitized_fraction": 0.97
  },
  "candidate_metrics": [
    {
      "candidate_id": "exampleA:proteinA:0",
      "example_id": "exampleA",
      "protein_id": "proteinA",
      "metrics": {
        "rdkit_valid_fraction": 1.0,
        "rdkit_sanitized_fraction": 1.0,
        "backend_missing_structure_fraction": 0.0
      }
    }
  ]
}
```

Claim rule: candidate-level rows are the only source that may upgrade per-profile provenance to backend-supported evidence. Aggregate-only metrics stay valid for coverage/caution summaries but must not be promoted into per-candidate backend evidence claims.

## Metrics And Claims

Preference metrics should report profile count, pair count, backend-supported pair fraction, rule-only pair fraction, mean preference strength, hard-constraint win fraction, mean gate/slot usage for selected candidates, and leakage/provenance caveats when available.

Drug-likeness, scaffold novelty, and train-similarity metrics are future metric interfaces unless backed by explicit RDKit/scaffold implementations. They should be reported as unavailable rather than approximated by unrelated scores.

Claim wording must use conservative labels:

- heuristic-only pairs: `rule-based preference proxy`
- backend-supported pairs: `backend-supported preference evidence`
- docking pairs: only after docking backend coverage passes
- human preference: only after human labels are loaded
- experimental preference: only after experimental outcomes are loaded

## Future Generator-Level Roadmap

The Q2 backend-backed preference dataset contract is recorded in
`configs/q2_preference_dataset_contract.json`. It requires each pair to preserve
winner/loser candidate ids, method id, layer, `feature_delta`, `evidence_source`,
and backend coverage. Required pair examples include docking-good/druglike wins,
high-pocket-fit but docking-bad proxy failures, docking-good but SA/QED-unacceptable
cases, and repairs that improve proxies while destroying score-only docking.

Q2 stops at dataset and deterministic reranking contracts. It does not enable
RL, DPO, preference fine-tuning, or generator-level policy optimization.

Generator-level DPO, RL, diffusion fine-tuning, or flow-matching preference training should only be considered after profile and pair artifacts are stable across held-out pockets, backend coverage is sufficient, and calibrated reranking plateaus. Future trainer hooks should consume `PreferencePair` records without bypassing the disentangled topology/geometry/pocket backbone or replacing gated cross-modal interaction with naive fusion.

Use the existing reviewer gate as an explicit onboarding contract before enabling generator-level preference trainers:

```bash
python3 tools/claim_regression_gate.py checkpoints/pdbbindpp_real_backends \
  --enforce-backend-thresholds \
  --enforce-data-thresholds \
  --enforce-publication-readiness \
  --enforce-preference-readiness
```

This strict mode is intentionally opt-in. Default runs remain additive and backward-compatible (missing preference artifacts still mean unavailable evidence), while trainer onboarding can require external-benchmark chemistry tier plus backend-supported preference coverage.

## Implementation Checkpoint

The initial checkpoint is complete when the crate compiles with versioned profile/pair structs, default-off config, backward-compatible method-comparison summaries, schema tests, and this document. Later implementation should emit artifacts only after tests cover old-artifact compatibility and missing-preference semantics.
