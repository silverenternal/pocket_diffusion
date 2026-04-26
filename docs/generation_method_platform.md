# Generation Method Platform

## Scope

This repository now treats conditioned denoising as one `PocketGenerationMethod` implementation inside a shared comparison platform.

The comparison boundary is:

- shared disentangled encoders, slot decomposition, and gated cross-modal interaction stay in the shared backbone
- generation methods own method metadata, layer provenance, and method-native candidate production
- shared postprocessors own geometry repair, bond inference, proxy reranking, and calibrated reranking layers
- backend evaluators own chemistry, docking, and pocket-compatibility scoring
- experiment runners own split management, fair shared execution, method comparison summaries, and artifact persistence
- reviewer/report builders own backward-compatible claim and evidence surfaces

## Method Contract

The stable contract lives in `src/models/traits.rs`:

- `PocketGenerationMethod`
- `PocketGenerationMethodMetadata`
- `GenerationMethodCapability`
- `GenerationEvidenceRole`
- `GenerationExecutionMode`
- `CandidateLayerKind`
- `LayeredGenerationOutput`

The current registry lives in `src/models/methods.rs` and registers:

- `conditioned_denoising`
- `heuristic_raw_rollout_no_repair`
- `pocket_centroid_repair_proxy`
- `deterministic_proxy_reranker`
- `calibrated_reranker`
- `flow_matching_stub`
- `diffusion_stub`
- `autoregressive_stub`
- `external_wrapper_stub`

## Compatibility Boundary

The existing reviewer-facing artifact keys remain stable:

- `raw_rollout`
- `repaired_candidates`
- `inferred_bond_candidates`
- `deterministic_proxy_candidates`
- `reranked_candidates`
- `baseline_comparisons`
- `claim_summary.json`
- `generation_layers_<split>.json`

Method-aware metadata is additive at the artifact boundary:

- `active_method`
- `method_layer_outputs`
- `method_comparison`

## Extraction Targets

The remaining extraction boundary is explicit:

- `src/models/evaluation.rs`
  generation-layer helper functions remain the compatibility bridge for candidate synthesis and backend adapters
- `src/experiments/unseen_pocket.rs`
  split-level aggregation, claim interpretation, and artifact writing remain the compatibility layer around method-aware internals
- `src/experiments/entrypoints.rs`
  config-driven entrypoints should continue to depend on the registry/config surface instead of `Phase1ResearchSystem` assumptions

## Fair Comparison Rules

- all methods run on the same split examples and shared backend surface
- claim-bearing interpretation is driven by each method's `evidence_role`
- layered metrics distinguish method-native output from derived postprocessed layers
- raw-to-repaired and inferred-to-reranked deltas are reported additively

## Future Planning

Planned method families are registered as stubs first:

- flow matching for geometry-heavy continuous refinement
- diffusion for denoising-family baselines beyond the current path
- autoregressive decoding for topology-first categorical refinement
- external wrappers for non-native baselines

Planned backend-agnostic metric interfaces are tracked in method comparison artifacts:

- `chemistry_property_bundle`
- `scaffold_novelty_bundle`

## Interaction Preference Addendum

Interaction preference alignment is an additive layer over the same method contract. Candidate generation remains owned by `PocketGenerationMethod` implementations; preference extraction starts after candidate layers have been produced.

The current compatible path is `conditioned_denoising`, with the same raw, repaired, inferred-bond, deterministic-proxy, and calibrated-reranked layers available to future methods. Preference-aware reranking is a planned layer, not a replacement for calibrated reranking or a hidden generator trainer.

Preference artifacts should be named independently from legacy layer artifacts:

- `preference_profiles_<split>.json`
- `preference_pairs_<split>.json`
- `preference_reranker_summary.json`

Missing preference artifacts mean preference evidence is unavailable. Existing `generation_layers_<split>.json` and method comparison summaries remain valid without them.
