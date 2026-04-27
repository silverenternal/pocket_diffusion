# Repository Audit

## Claim Matrix

| Surface | Claim | Status | Boundary |
| --- | --- | --- | --- |
| Config-driven modular stack | Separate topology / geometry / pocket encoders | `implemented` | Active research path |
| Config-driven modular stack | Slot decomposition and gated cross-modal interaction | `implemented` | Active research path |
| Config-driven modular stack | Staged training with explicit primary + auxiliary objectives | `implemented` | Supports surrogate ablations and decoder-anchored conditioned denoising |
| Config-driven modular stack | Unseen-pocket evaluation | `implemented` | Outputs diagnostics, proxy metrics, active candidate metrics, and comparison summaries |
| Config-driven modular stack | Chemistry-grade generation metrics | `implemented` | Repository-supported RDKit validity contract plus heuristic reference metrics are integrated |
| Config-driven modular stack | Automated cross-surface tuning | `implemented` | Bounded search over interaction, rollout, and selected loss controls with hard regression gates |
| Config-driven modular stack | Diffusion objective | `planned` | Crate name is historical, not descriptive of the current objective |
| Legacy surface | Candidate generation and ranking demo | `legacy` | Compatibility/demo surface, with optional routing through the modular decoder bridge |
| Lightweight data path | Affinity normalization across mixed measurement families | `prototype+guarded` | Still convenience-oriented, but now persists measurement-family mix, approximate-label fraction, normalization-provenance coverage, and affinity-metadata completeness for stronger review surfaces |

## Current Architecture Summary

The repository currently exposes two parallel surfaces:

- A legacy demo and comparison stack centered around `PocketDiffusionPipeline`, `src/dataset.rs`, and older experiment/demo binaries.
- A newer modular research stack centered around `src/config/*`, `src/data/*`, `src/models/*`, `src/losses/*`, `src/training/*`, and `src/experiments/unseen_pocket.rs`.

The modular stack already reflects the intended research direction:

- separate topology, geometry, and pocket encoders
- slot-based decomposition
- gated cross-modal interaction with configurable lightweight and Transformer-style controlled blocks plus compact Transformer tuning knobs
- staged training losses
- config-driven data loading from synthetic, manifest, and PDBbind-like sources
- automated cross-surface tuning that replays compact, harder-pressure, and tight-geometry surfaces with explicit chemistry, clash, uniqueness, and strict-fit gates
- geometry-attribution ablations that separately disable controlled-attention pocket geometry bias, decoder rollout pocket guidance, and candidate repair
- claim summaries that include raw/repaired/inferred/reranked generation layers, slot-stability diagnostics, and leakage-calibration recommendations
- a method-aware comparison boundary where conditioned denoising is one registered generation method instead of the only generation path

It should currently be described as a modular generation-research framework with deterministic decoder supervision, method-aware generation comparison, conditioned-denoising training support, and backend-backed candidate evaluation. The repository still does not expose a true pocket-conditioned diffusion training loop on the actively extended path.

The method-platform boundary is now explicit:

- `src/models/traits.rs` defines the `PocketGenerationMethod` contract, method metadata, and layered output schema
- `src/models/methods.rs` owns the concrete registry and current built-in method implementations or stubs
- `src/experiments/unseen_pocket.rs` remains the compatibility layer that preserves reviewer field names while emitting additive method metadata
- `docs/generation_method_platform.md` is the checked-in migration note for this boundary

The repo is usable for real research iteration, but users still need to respect the documented surface boundary: config-driven research flows are the default path, while legacy/demo commands remain compatibility shims rather than the place for new claim-bearing behavior.

## Legacy vs Modular Boundary

### Modular research stack

These files are the primary research path and should remain the default documented execution surface:

- `src/config/*`
- `src/data/*`
- `src/models/*`
- `src/losses/*`
- `src/training/*`
- `src/experiments/*`

### Legacy compatibility surface

These files provide older demos or comparison utilities and should remain available, but clearly marked as compatibility or exploratory code:

- `src/dataset.rs`
- `src/experiment.rs`
- `src/bin/disentangle_demo.rs`
- legacy demo flow in `src/main.rs`
- `PocketDiffusionPipeline` in `src/lib.rs`

## Concrete Risks

1. The repository still ships a substantial legacy demo surface, so architectural ambiguity can reappear if new features are added directly to legacy modules instead of the modular stack.
2. Legacy demo and comparison paths still use `Device::Cpu` directly, which is acceptable for compatibility demos but should not be copied into config-driven research code.
3. `src/main.rs` is now a thin routing shell with compatibility notices, but it still contains legacy demo orchestration and human-readable reporting logic, so keeping research execution in `src/training/*` and `src/experiments/*` remains important.
4. The repo contains both `src/data/dataset.rs` and legacy `src/dataset.rs`; the boundary is now documented, but future edits need to preserve that distinction.

## Prioritized Improvement Plan

### Completed P0

- Replaced README absolute links with repository-relative links.
- Introduced a single runtime device parser and routed config-driven training and experiments through it.
- Propagated `ModelConfig.pocket_feature_dim` into model construction and normalized dataset examples to the configured width.
- Separated modular research CLI entrypoints from the legacy demo path so the research stack is the obvious default.

### Completed P1

- Refactored training into deterministic mini-batch iteration that respects `DataConfig.batch_size`.
- Added checkpoint discovery, load, and resume support with step restoration.
- Moved config-driven training and dataset inspection orchestration out of `main.rs` into `src/training/entrypoints.rs`.
- Improved staged training logging so stage transitions and active loss weights are visible.

### Completed P2

- Reduced the supported modular surface at the crate root, while keeping a small set of targeted crate-level Clippy allowances for the mixed legacy/modular codebase.
- Marked legacy dataset/pipeline code as compatibility surface in docs and module comments.
- Added smoke tests for config parsing, device parsing, batch iteration, checkpoint save/load, dataset inspection, and short train/eval paths.
- Updated README instructions so the modular research stack is the primary documented workflow, with legacy demo usage clearly separated.
- Moved unseen-pocket config-driven experiment startup behind a module-level entrypoint so `src/main.rs` remains a thin CLI shell.
- Added an explicit `legacy` comparison wrapper so old benchmark utilities have a clearer namespace instead of relying on crate-root discovery.
- Added namespace-stability coverage for modular and legacy entrypoints so the intended import surfaces remain explicit as the crate evolves.
- Added runtime compatibility notices for legacy/demo CLI paths so the binary itself now reinforces the modular research interface boundary and points users at the canonical `research ... --config ...` replacements.
- Added stronger compile-time/doc-discovery signals on crate-root legacy compatibility modules and the legacy pipeline type so new work is steered toward `pocket_diffusion::legacy` or the modular stack instead of the old flat surface.
- Renamed the primary modular training objective as `surrogate_reconstruction` and separated it structurally from auxiliary regularizers in logs and summaries.
- Reworked evaluation summaries so generation-like names were replaced by explicit diagnostic/proxy namespaces.
- Reduced root/module wildcard re-exports so the supported modular surface is explicitly namespaced.
- Added persisted dataset validation and split-audit artifacts so config-driven inspection, training, and unseen-pocket experiments expose discovery counts, label attachment behavior, fallback pocket extraction counts, and explicit leakage checks.

### Remaining Low-Priority Follow-Up

- Keep new experiment or evaluation flows behind module-level entrypoints instead of re-expanding `src/main.rs`; reviewer-critical behavior should continue to land in modular modules first.
- Decide whether legacy comparison modules should eventually move physically under `src/legacy/*` instead of only being re-exported there.
- Add optional full optimizer-state resume if reproducibility requirements expand beyond weight-and-step restoration.

## Capability And Limitation Table

| Area | Current status | Limitation |
| --- | --- | --- |
| Representation learning and generation | Strong modular prototype | Decoder path is research-grade, not a production sampler |
| Evaluation | Diagnostic/proxy plus executable chemistry and pocket-aware backend reporting | RDKit availability still depends on the local Python environment, and downstream scoring remains research-grade rather than production docking |
| Data ingestion | Lightweight real-file path with deterministic generation targets | Parsing remains convenience-oriented rather than assay-grade |
| Legacy generation demo | Still runnable | Compatibility surface, not the primary abstraction for new work |

## Evidence Review

The current persisted evidence is now concrete rather than hypothetical:

- `checkpoints/claim_matrix/claim_summary.json` and `checkpoints/claim_matrix/ablation_matrix_summary.json` exist and are readable.
- `checkpoints/real_backends/claim_summary.json` and `checkpoints/real_backends/experiment_summary.json` exist and record explicit backend-environment status.
- `checkpoints/pdbbindpp_real_backends/claim_summary.json` and `checkpoints/pdbbindpp_real_backends/experiment_summary.json` exist and record the canonical larger-data held-out-pocket reviewer surface.
- `checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json` and `checkpoints/lp_pdbbind_refined_real_backends/experiment_summary.json` exist and record the second larger-data benchmark reviewer surface under the same artifact contract.

The current workspace was rechecked directly on April 24, 2026 rather than inferred only from prior artifacts:

- `cargo test` passed locally.
- `configs/research_manifest.json` validated, inspected cleanly, trained, and ran a full unseen-pocket experiment on the bundled mini dataset.
- `configs/unseen_pocket_claim_matrix.json` reran successfully and remains suitable as the fast generator regression gate.
- `configs/unseen_pocket_real_backends.json` reran successfully with active external chemistry, docking, and pocket-compatibility backends.
- `configs/unseen_pocket_pdbbindpp_real_backends.json` reran successfully in the current environment and refreshed the larger-data canonical reviewer surface.
- `configs/unseen_pocket_lp_pdbbind_refined_real_backends.json` reran successfully in the current environment and refreshed the second larger-data benchmark reviewer surface.

The refreshed evidence after the April 24 canonical refresh is more actionable:

- On April 23, 2026, rerunning the compact gate with [`configs/unseen_pocket_claim_matrix.json`](../configs/unseen_pocket_claim_matrix.json) after a small gate-calibration plus residual-scaling tune refreshed [`checkpoints/claim_matrix/claim_summary.json`](../checkpoints/claim_matrix/claim_summary.json) and flipped both compact splits toward Transformer (`validation tally: transformer 5, lightweight 3, ties 4`; `test tally: transformer 6, lightweight 2, ties 4`) while keeping the compact chemistry and pocket-fit gates fully stable (`candidate_valid_fraction=1.0`, `pocket_contact_fraction=1.0`, `pocket_compatibility_fraction=1.0`).
- The compact interaction review at [`checkpoints/claim_matrix/interaction_mode_review.json`](../checkpoints/claim_matrix/interaction_mode_review.json) now shows the intended local change: on test, Transformer beats lightweight on geometric fit (`mean_centroid_offset=0.8184` vs `0.9957`, `strict_pocket_fit_score=0.5505` vs `0.5015`) and overall reviewed metrics, while lightweight still keeps some specialization/utilization wins.
- The real-backend workflow remains operational as actual chemistry evidence rather than only a contract test. On April 22, 2026, `python3 -c 'import rdkit'` succeeded, and rerunning [`configs/unseen_pocket_real_backends.json`](../configs/unseen_pocket_real_backends.json) kept [`checkpoints/real_backends/experiment_summary.json`](../checkpoints/real_backends/experiment_summary.json) live with explicit backend-environment status.
- The harder-pressure surface at [`configs/unseen_pocket_harder_pressure.json`](../configs/unseen_pocket_harder_pressure.json) now persists [`checkpoints/harder_pressure/interaction_mode_review.json`](../checkpoints/harder_pressure/interaction_mode_review.json) alongside the claim and ablation summaries. On April 23, 2026, rerunning that surface with the same compact/harder tune kept the base chemistry and pocket gates intact on test (`candidate_valid_fraction=1.0`, `pocket_contact_fraction=1.0`, `pocket_compatibility_fraction=1.0`, `rdkit_sanitized_fraction=1.0`, `rdkit_unique_smiles_fraction=1.0`, `clash_fraction=0.0`) and moved the harder-pressure comparison decisively toward Transformer (`validation tally: transformer 7, lightweight 2, ties 3`; `test tally: transformer 7, lightweight 1, ties 4`). On test, Transformer now wins the target geometric-fit metrics (`mean_centroid_offset=1.1526` vs `1.2932`, `strict_pocket_fit_score=0.4391` vs `0.4178`) instead of yielding them to lightweight.
- The canonical larger-data surface at [`checkpoints/pdbbindpp_real_backends/claim_summary.json`](../checkpoints/pdbbindpp_real_backends/claim_summary.json) was rerun again on April 24, 2026 and materially improved over the earlier same-day canonical artifact: test `strict_pocket_fit_score` rose from `0.5005` to `0.6752`, `test.leakage_proxy_mean` dropped from `0.0991` to `0.0506`, `clash_fraction` stayed at `0.0`, and `benchmark_evidence.evidence_tier=external_benchmark_backed` was preserved.
- The second larger-data benchmark surface at [`checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json`](../checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json) was also refreshed on April 24, 2026 and cleared the same reviewer-facing chemistry tier with `parsed_examples=5048`, `retained_label_coverage=1.0`, `test.strict_pocket_fit_score=0.7599`, `test.leakage_proxy_mean=0.0569`, `clash_fraction=0.0`, and `benchmark_evidence.evidence_tier=external_benchmark_backed`.
- The tighter larger-data interaction decision is now explicit. [`checkpoints/pdbbindpp_real_backends_interaction_review/interaction_mode_review.json`](../checkpoints/pdbbindpp_real_backends_interaction_review/interaction_mode_review.json) compares the same larger-data real-backend surface with a lightweight-only ablation under matched non-interaction settings and shows Transformer decisively ahead on test geometry and leakage (`strict_pocket_fit_score=0.9111` vs `0.5234`, `leakage_proxy_mean=0.0647` vs `0.0911`), while both modes keep `candidate_valid_fraction=1.0` and `unique_smiles_fraction=1.0`.
- The tight-geometry surface at [`configs/unseen_pocket_tight_geometry_pressure.json`](../configs/unseen_pocket_tight_geometry_pressure.json) remains the bounded tradeoff surface rather than the geometry-pressure blocker. Its canonical artifact at [`checkpoints/tight_geometry_pressure/claim_summary.json`](../checkpoints/tight_geometry_pressure/claim_summary.json) keeps `strict_pocket_fit_score=0.6178`, `clash_fraction=0.0`, and `test.leakage_proxy_mean=0.0613` with `leakage_calibration.reviewer_status=pass`, while its interaction review still leaves lightweight competitive on fit.
- The shared cross-surface interaction decision artifact at [`checkpoints/interaction_mode_review.json`](../checkpoints/interaction_mode_review.json) now centers the stronger surfaces that matter for reviewer claims. Its persisted recommendation is to promote Transformer as the default larger-data claim path, while retaining lightweight as an explicit tight-geometry ablation instead of pretending there is a universal winner across every surface.
- The automated search path is now available through `research search --config configs/unseen_pocket_claim_matrix.json`. It generates bounded candidates from `interaction_tuning`, rollout controls, and selected loss weights, reruns the three claim-bearing surfaces into `checkpoints/automated_search/`, rejects candidates that regress sanitization, uniqueness, clash fraction, strict pocket fit, chemistry validity, pocket contact, or pocket compatibility, and ranks surviving candidates with a shared multi-surface objective plus the aggregate interaction review. The claim path now persists both the previous fixed-weight deterministic proxy selector and an active bounded calibrated reranker whose non-negative feature coefficients are fit from split-local candidate features to backend-compatible validity, valence, pocket-contact, centroid-fit, clash, and bond-density targets.
- Medium-scale data review now has three distinct roles: [`configs/unseen_pocket_medium_profile.json`](../configs/unseen_pocket_medium_profile.json) remains a parser smoke profile, [`checkpoints/pdbbindpp_real_backends`](../checkpoints/pdbbindpp_real_backends) is the canonical larger-data reviewer surface on the local PDBbind++ benchmark path, and [`checkpoints/lp_pdbbind_refined_real_backends`](../checkpoints/lp_pdbbind_refined_real_backends) is the second larger-data reviewer surface on the LP-PDBBind refined benchmark contract. Both larger-data surfaces clear the local data thresholds for parsed examples, retained label coverage, and held-out family diversity and promote chemistry evidence to the explicit `external_benchmark_backed` tier.
- Seed-level uncertainty now has a first-class entrypoint: `research multi-seed --config configs/unseen_pocket_multi_seed.json`. It varies split, corruption, and sampling seeds, writes independent per-seed artifacts, and persists `multi_seed_summary.json` with mean/std/min/max summaries for validity, pocket fit, uniqueness, slot activation, gate usage, leakage, slot-signature similarity, and throughput.
- Compact artifact gating is available through `tools/claim_regression_gate.py <artifact_dir>` for existing artifacts or `tools/claim_regression_gate.py --run --config configs/unseen_pocket_claim_matrix.json` for the full local run-and-validate path. The gate checks claim, experiment, split, and bundle artifacts for required schema fields, generation-layer summaries, stratification diagnostics, and finite slot/gate/leakage metrics.
- The modular trainer now uses explicit batch loss aggregation APIs for primary, probe, redundancy, leakage, gate, slot, consistency, contact, and clash objectives. Unit coverage checks the batch values against the prior per-example aggregation on synthetic mini-batches, preserving staged-loss semantics while keeping the batched encoder/interaction path active.
- Slot semantics are now tracked with deterministic activation, slot-signature similarity, and probe-alignment summaries for topology, geometry, and pocket slots. Leakage calibration remains conservative: claim reports recommend the current `delta_leak` unless reviewed ablations preserve geometry fit and chemistry validity.
- The pocket-aware command backends continue to keep failure modes explicit through `schema_version`, `backend_examples_scored`, and `backend_missing_structure_fraction`, so degraded chemistry or geometry now shows up as scored evidence rather than silent backend crashes.
- On April 23, 2026, the compact claim gate was rerun end to end with `python3 tools/claim_regression_gate.py --run --config configs/unseen_pocket_claim_matrix.json` and passed after refreshing `checkpoints/claim_matrix`.
- On April 23, 2026, harder-pressure and tight-geometry surfaces were refreshed. Harder-pressure remained clean on backend chemistry and pocket evidence (`rdkit_unique_smiles_fraction=1.0`, `strict_pocket_fit_score=0.5781`, `clash_fraction=0.0556` on test). Tight geometry is now also clean on the canonical reviewer path with `strict_pocket_fit_score=0.6178`, `clash_fraction=0.0`, and `test.leakage_proxy_mean=0.0613`.
- On April 23, 2026, RDKit availability was verified locally and `configs/unseen_pocket_real_backends.json` was rerun into `checkpoints/real_backends`. The refreshed real-backend artifact reports `rdkit_available=1.0`, `rdkit_sanitized_fraction=1.0`, `backend_examples_scored=3`, `backend_missing_structure_fraction=0.0`, `clash_fraction=0.0`, `strict_pocket_fit_score=0.4418`, and backend uniqueness above the local hard gate, so this reviewer surface now passes the repository's backend-threshold validation.
- `configs/unseen_pocket_medium_profile.json` now points at `data/PDBbind_v2020_refined` through the Rust PDBBind-like directory parser and writes validation/split artifacts under `checkpoints/medium_profile`. This local root currently contains only five unlabeled complexes, so it is a parser/scale-up smoke surface rather than a real medium-scale claim dataset.
- Compact and PDBBind-like multi-seed runs were refreshed with three seeds each. The aggregate summaries now include deterministic 95% t-style confidence intervals in addition to mean/std/min/max. The current larger-data seed-stability surface lives at [`configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json`](../configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json), with refreshed test `strict_pocket_fit_score mean=0.6517`, `leakage_proxy_mean mean=0.0719`, and `test_examples_per_second mean=65.8136`.
- The larger-data pocket-compatibility gap is smaller but not fully closed. The refreshed canonical surface improved backend `atom_coverage_fraction` from `0.7901` to `0.8810` without regressing clash, validity, or chemistry evidence. The remaining misses are now concentrated in explicit `backend_failure_examples` dominated by lightweight pocket clashes and a smaller set of chemistry-sanity failures, rather than by missing structure provenance or backend crashes.
- `tools/claim_regression_gate.py` now has an opt-in `--enforce-backend-thresholds` mode covering RDKit availability, sanitized fraction, uniqueness, backend missing-structure fraction, clash fraction, strict pocket fit, and pocket contact. The refreshed `checkpoints/real_backends`, `checkpoints/pdbbindpp_real_backends`, `checkpoints/lp_pdbbind_refined_real_backends`, and `checkpoints/tight_geometry_pressure` artifacts now pass their active backend-threshold gates.
- A local pre-claim CI entrypoint is available at `tools/local_ci.sh`. The default fast mode runs `cargo fmt --check`, `cargo test`, key experiment config validation, Python backend syntax checks, the compact claim artifact gate, the real-backend threshold gate, the second larger-data benchmark gate, and reviewer evidence-bundle generation. The data-readiness gate should now target `checkpoints/pdbbindpp_real_backends` plus `checkpoints/lp_pdbbind_refined_real_backends`; `checkpoints/medium_profile` still fails locally at `parsed_examples=5 < 100` and should be treated only as a parser smoke surface.
- Dataset quality filters are now available through `data.quality_filters` with default-off behavior. Real-data configs can require source structure provenance, atom-count ceilings, minimum label coverage, maximum pocket-fallback fraction, affinity-metadata completeness, bounded approximate-label share, and minimum normalization-provenance coverage; validation artifacts report filtered counts separately from parser failures and max-example truncation.
- Label-table accounting is now explicit in the same validation artifacts: rows seen, blank/comment rows skipped, loaded measurement-family histograms, normalization provenance values, duplicate-key overwrites, and unmatched loaded labels are persisted instead of being left implicit in parser behavior.
- Backend failure examples are now attached to `generation_layers_<split>.json` artifacts under `backend_failure_examples`, with bounded candidate ids, provenance paths, backend status, and failure reasons for command failures, missing structures, chemistry sanity failures, and pocket clashes.
- An optional AutoDock Vina score-only adapter is available at `tools/vina_score_backend.py` with a runnable profile in `configs/unseen_pocket_vina_backend.json`. It reports explicit availability and input-completeness metrics when Vina or PDBQT inputs are absent, persists those metrics under `checkpoints/vina_backend`, and is now carried through the reviewer evidence bundle as an explicit stronger-backend companion profile rather than being allowed to silently degrade to heuristic-only wording.
- The legacy module layout decision is to keep `src/dataset.rs` and `src/experiment.rs` physically in place as compatibility modules and expose them through `src/legacy/`. Moving them now would add churn without improving the modular research path; existing namespace tests preserve the supported compatibility surface.
- Reviewer-facing evidence can now be collected with `tools/evidence_bundle.py`, which writes `docs/evidence_bundle.json` by default from the current compact, backend, larger-data, and multi-seed artifacts and records explicit reviewer-surface anchors. Claim wording rules are documented in `docs/claim_readiness.md`.

The roadmap no longer needs to justify a still-red reviewer bundle. The current evidence now supports a clean local reviewer pass: the compact gate remains stable, two larger-data real-backend surfaces now carry the explicit `external_benchmark_backed` chemistry tier, the tight-geometry surface clears both backend and leakage review, and `docs/evidence_bundle.json` now reports `reviewer_bundle_status=pass`. The remaining caveat is narrower and clearer: the chemistry story is no longer anchored to one benchmark-dataset reviewer path, but it is still a local reviewer package rather than a broad publication-scale benchmark campaign. The data contract is also stricter than before without pretending to be full assay-grade harmonization: retained measurement-family mix, approximate-label share, normalization provenance, and missing-affinity-metadata drops are now explicit in `dataset_validation.json`, while concentration-derived proxy families remain clearly marked as bounded approximations. The project should therefore keep the compact gate as the mandatory fast regression pass, keep harder-pressure and tight-geometry as decision surfaces, and use the dual-benchmark larger-data plus multi-seed reviewer path as the anchor for future claim promotion.

The latest rerun on April 24, 2026 reinforces that status instead of weakening it. The canonical larger-data artifact currently shows `parsed_examples=5316`, `retained_label_coverage=0.9668`, `candidate_valid_fraction=1.0`, `unique_smiles_fraction=1.0`, `strict_pocket_fit_score=0.6752`, `test.leakage_proxy_mean=0.0506`, `clash_fraction=0.0`, and `benchmark_evidence.evidence_tier=external_benchmark_backed`, while the LP-PDBBind refined artifact shows `parsed_examples=5048`, `retained_label_coverage=1.0`, `candidate_valid_fraction=1.0`, `unique_smiles_fraction=1.0`, `strict_pocket_fit_score=0.7599`, `test.leakage_proxy_mean=0.0569`, `clash_fraction=0.0`, and the same `external_benchmark_backed` tier. In practical terms, the repository is no longer merely "promising"; it is already operating as a usable reviewer-facing research stack with explicit fast-gate and stronger-gate surfaces.

Documentation/state housekeeping update (April 26, 2026): the post-preference-artifact execution checklist in `todo.json` has been marked complete and archived, with `todo.json` reduced to a completion marker. Ongoing status should be read from checked-in evidence and guardrail documents (`docs/evidence_bundle.json`, `docs/paper_claim_bundle.md`, `docs/reviewer_refresh_report.json`, `docs/claim_readiness.md`) rather than from the archived execution checklist body.

Diffusion, adversarial, and GAN-style training remain intentionally out of the mainline roadmap. The current evidence supports keeping conditioned denoising plus bounded iterative refinement while data scale, multi-seed stability, real backend scoring, raw-rollout pocket-envelope stability, and the calibrated reranker path are hardened. A diffusion-style objective should only be reconsidered after raw rollout quality and backend-supported reranking plateau on larger held-out pocket families, and only if that plateau is reproduced across multi-seed real-backend surfaces rather than only compact heuristic artifacts. That decision is now persisted independently at `checkpoints/generator_decision/generator_decision.json`.

## Modeling Decision: Diffusion Deferred

As of April 23, 2026, the active generator should remain a conditioned-denoising iterative refinement path rather than being converted into a diffusion objective. The current implementation has explicit rollout supervision, stop calibration, coordinate-delta clipping, momentum refinement, pocket-centroid guidance, and now per-step pocket-envelope projection for raw rollout stability. The reranking path has also moved from a fixed deterministic proxy to a bounded calibrated layer that persists coefficients and reports calibrated-vs-proxy selection metrics.

The evidence gate for revisiting diffusion is deliberately higher than the current compact surface: raw rollout validity, centroid fit, clash behavior, atom stability, calibrated reranker gains, and real-backend chemistry/pocket metrics must plateau on larger held-out pocket families with multi-seed uncertainty. Until that happens, diffusion would add objective and sampler complexity before the simpler failure modes are isolated.

## Architecture Note: Gap To A True Diffusion Generator

The modular stack has three pieces that are worth preserving for future generation work: modality-specific encoders, slot decomposition, and gated cross-modal interaction. It now also has a decoder-facing corruption/denoising contract, iterative rollout logging, and a compact ablation matrix path. What it still does not have is the rest of a production-grade generation system:

- no diffusion objective
- no high-capacity learned sampler beyond a now stronger but still research-grade iterative decoder rollout
- no chemistry-native toolkit bundled in-process
- no production docking protocol validation is bundled in-process; repository-supported external AutoDock Vina and GNINA score-only adapters now produce candidate-level coverage, runtime, and failure metadata

The intended migration path is therefore:

1. Keep the current modular encoders/slots/interactions as the conditioning backbone.
2. Treat `surrogate_reconstruction` as one primary-objective implementation, not as the end state.
3. Extend `conditioned_denoising` and the decoder path into a stronger iterative generator instead of entangling primary-task logic with auxiliary losses.
4. Replace heuristic chemistry/docking adapters with external backends once the generator matures.

## Preserve vs Refactor

### Preserve

- `src/config/*`
- `src/data/parser.rs`
- `src/models/cross_attention.rs`
- `src/models/slot_decomposition.rs`
- `src/losses/*`
- `src/experiments/unseen_pocket.rs`
- legacy `PocketDiffusionPipeline` surface and `src/dataset.rs` compatibility APIs

### Refactor

- `README.md`
- `src/main.rs`
- `src/lib.rs`
- `src/models/system.rs`
- `src/data/features.rs`
- `src/data/batch.rs`
- `src/data/dataset.rs`
- `src/training/entrypoints.rs`
- `src/training/checkpoint.rs`
- `src/training/trainer.rs`

## Expected Outcome

The repository now keeps its legacy demos intact while making the modular research stack the clean, config-faithful default path for inspection, training, experiments, compact conditioned-generation demos, executable backend workflows, and claim-oriented experiment summaries. The remaining work is incremental model/evaluation hardening rather than architectural correction or basic runtime enablement.
