# Claim Readiness Guardrails

This project should separate implementation claims from evidence claims.

## Allowed Now

- The Rust stack implements separate topology, geometry, and pocket encoders.
- Slot decomposition, gated cross-modal interaction, staged losses, split diagnostics, multi-seed summaries, backend command adapters, and the method-aware generation comparison contract are implemented and runnable.
- Compact artifacts may be described as local regression or smoke evidence.
- As of April 24, 2026, the repository may be described as a backend-supported research framework that is ready for real held-out-pocket iteration: the compact gate, repository real-backend gate, canonical larger-data PDBbind++ real-backend surface, and its refreshed three-seed companion all ran successfully in the current workspace.

## Provisional Only

- Medium-profile results are provisional until the configured root contains enough labeled, diverse, held-out pocket families.
- The current reviewer-facing larger-data surfaces are `checkpoints/pdbbindpp_real_backends` and `checkpoints/lp_pdbbind_refined_real_backends`; use `checkpoints/pdbbindpp_profile` only as the heuristic fallback when real backends are unavailable.
- Real-backend chemistry and pocket results are provisional unless backend availability, missing-structure fraction, sanitization, uniqueness, clash, and pocket-fit gates pass in the target environment.
- Real-data claim wording is also provisional unless the retained dataset contract is explicit: source-structure provenance coverage, normalization-provenance coverage, approximate-label fraction, and measurement-family composition must be persisted in `dataset_validation.json`, not inferred informally.
- Label-table handling must also remain explicit for claim-bearing wording: duplicate-key overwrites, unmatched loaded labels, and skipped blank/comment rows should be taken from persisted dataset validation artifacts rather than assumed away.
- The stronger Vina companion surface is claim-bearing only when `checkpoints/vina_backend/claim_summary.json` reports `backend_review.reviewer_status=pass`; if Vina or Vina-ready PDBQT inputs are absent, the same artifact should remain explicitly non-passing rather than being interpreted as heuristic-only chemistry evidence.
- Tight-geometry interpretation now has a clean reviewer pass on the canonical surface: `checkpoints/tight_geometry_pressure` reports `test.leakage_proxy_mean=0.0613`, `strict_pocket_fit_score=0.6178`, `clash_fraction=0.0`, and `leakage_calibration.reviewer_status=pass`.

## Blocked Wording

- Do not claim broad unseen-pocket generalization from mini or five-complex smoke surfaces.
- Do not claim production docking quality from contact/clash/centroid proxy backends.
- Do not claim a diffusion generator unless a diffusion-style objective and sampler are explicitly implemented and evaluated.
- Treat the crate name `pocket_diffusion` as historical compatibility wording only; reviewer-facing text should describe the active system as modular representation learning / conditioned generation unless and until a true diffusion path is added.
- Do not describe conditioned denoising as the repository's only generator interface anymore. Reviewer-facing wording should distinguish the active claim-bearing method from the broader method platform.
- Do not claim strong chemistry novelty or diversity from uniqueness alone. Use the explicit novelty/diversity fields together with `benchmark_evidence`, which now separates proxy-only chemistry summaries, local benchmark-style chemistry aggregates, `reviewer_benchmark_plus`, and the explicit `external_benchmark_backed` tier on the canonical larger-data real-backend PDBbind++ surface.

Proxy-only chemistry evidence is the structural-signature novelty/diversity summary. Local benchmark-style chemistry evidence combines backend-backed sanitization and unique-SMILES quality with held-out-pocket novelty/diversity aggregates. `reviewer_benchmark_plus` adds explicit reviewer checks on parseability, finite conformers, review-layer support, and novelty/diversity support. The current strongest reviewer-facing chemistry evidence is `benchmark_evidence.evidence_tier=external_benchmark_backed` on both `checkpoints/pdbbindpp_real_backends` and `checkpoints/lp_pdbbind_refined_real_backends`: they keep the reviewer benchmark-plus checks and make the external benchmark dataset layer first-class by requiring explicit benchmark dataset labels, passing data thresholds, and passing held-out family coverage. Strong chemistry-facing wording should now cite benchmark breadth rather than only the single strongest surface: the canonical PDBbind++ artifact remains the main anchor, the LP-PDBBind refined artifact is the second larger-data external benchmark surface, and `checkpoints/tight_geometry_pressure` plus `checkpoints/real_backends` remain persisted companion review surfaces summarized in `docs/evidence_bundle.json` and `docs/paper_claim_bundle.md`.

The stronger Vina companion is now explicit about its own status. `checkpoints/vina_backend/claim_summary.json` persists `backend_review.policy_label=vina_claim_bearing_companion_policy` plus `docking_backend_available`, `docking_input_completeness_fraction`, and `docking_score_coverage_fraction`. On a machine without Vina or without receptor/ligand PDBQT inputs, that artifact should be cited as a non-passing stronger-backend companion profile, not as claim-bearing docking evidence. If those prerequisites are present and the persisted `backend_review.reviewer_status` flips to `pass`, the same surface can be cited as claim-bearing backend companion evidence without changing the reviewer contract.

## Promotion Rule

Use stronger claim language only after a reviewer bundle includes passing compact, real-backend, larger-data, multi-seed, performance, leakage, and baseline evidence with explicit limitations.

## Current Repository Level

As of April 24, 2026, the repository clears the internal bar for `Prototype` and also satisfies the local prerequisites for `Claim-ready` wording on the canonical reviewer path, with the usual limitation that this remains a reviewer-facing benchmark package rather than a broad publication bundle.

- Compact gate: [`checkpoints/claim_matrix/claim_summary.json`](../checkpoints/claim_matrix/claim_summary.json) remains the mandatory fast regression surface.
- Real-backend gate: [`checkpoints/real_backends/claim_summary.json`](../checkpoints/real_backends/claim_summary.json) ran locally with `rdkit_available=1.0`, `rdkit_sanitized_fraction=1.0`, and active external chemistry, docking, and pocket backends.
- Canonical larger-data surface: [`checkpoints/pdbbindpp_real_backends/claim_summary.json`](../checkpoints/pdbbindpp_real_backends/claim_summary.json) was rerun locally on April 24, 2026 and preserved `benchmark_evidence.evidence_tier=external_benchmark_backed` with `parsed_examples=5316`, `retained_label_coverage=0.9668`, `candidate_valid_fraction=1.0`, `unique_smiles_fraction=1.0`, `clash_fraction=0.0`, `test.strict_pocket_fit_score=0.6752`, `test.leakage_proxy_mean=0.0506`, and clean held-out splits.
- Second larger-data benchmark surface: [`checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json`](../checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json) was rerun locally on April 24, 2026 and also cleared `benchmark_evidence.evidence_tier=external_benchmark_backed` with `parsed_examples=5048`, `retained_label_coverage=1.0`, `candidate_valid_fraction=1.0`, `unique_smiles_fraction=1.0`, `clash_fraction=0.0`, `test.strict_pocket_fit_score=0.7599`, `test.leakage_proxy_mean=0.0569`, and clean held-out splits.
- Reviewer refresh status: [`docs/reviewer_refresh_report.json`](./reviewer_refresh_report.json) now reports `reviewer_bundle_status=pass`, with `backend_thresholds_passed=true`, `data_thresholds_passed=true`, `leakage_reviewer_status=pass`, and `replay_drift_passed=true` on the canonical larger-data surface.

That means the repository is no longer blocked on "can this stack run the stronger reviewer surface at all?". The remaining caution is narrower: future strong wording should stay anchored to the dual-benchmark reviewer package, with the canonical PDBbind++ path, the LP-PDBBind refined companion benchmark, and the tighter-geometry, real-backend, multi-seed, and interaction-decision surfaces kept in sync.

## Evidence Levels

| Level | Allowed wording | Minimum evidence |
| --- | --- | --- |
| Smoke | `runnable local regression surface` | Config validates, unit tests pass, compact artifact gate passes, and split leakage checks are explicit. |
| Prototype | `backend-supported prototype on local data` | Smoke requirements plus real-backend schema, RDKit availability, missing-structure rate, clash, strict pocket-fit, and uniqueness thresholds pass on the target machine. |
| Claim-ready | `supports unseen-pocket generalization evidence` | Prototype requirements plus larger labeled data, held-out family diversity, multi-seed confidence intervals, baseline deltas, and reviewer bundle audit pass. |

## Thresholds

The current hard gate thresholds are intentionally conservative local-review defaults, not publication thresholds:

- Minimum dataset size for claim-ready language: `>= 100` parsed, provenance-retained complexes with `>= 0.8` retained label coverage.
- Minimum retained source-structure provenance coverage for claim-ready language: `>= 0.95`.
- Minimum retained normalization-provenance coverage for claim-ready language: `>= 0.95` across labeled examples.
- Maximum retained approximate-label fraction for claim-ready language: `<= 0.25` unless the surface is explicitly described as proxy-family-heavy.
- Minimum held-out diversity: `>= 10` validation proxy protein families and `>= 10` test proxy protein families, with no protein overlap or duplicate examples across splits.
- Required backend schema: every enabled external backend must emit `schema_version >= 1` and `backend_examples_scored > 0`.
- Maximum missing structure fraction: `backend_missing_structure_fraction <= 0.0` for claim-bearing backend runs.
- Minimum RDKit availability: `rdkit_available >= 1.0`.
- Minimum sanitized fraction: `rdkit_sanitized_fraction >= 0.95`.
- Minimum uniqueness: `rdkit_unique_smiles_fraction >= 0.5`.
- Maximum clash fraction: `clash_fraction <= 0.1`.
- Minimum strict pocket fit: `strict_pocket_fit_score >= 0.35` or the explicit backend-equivalent `heuristic_strict_pocket_fit_score >= 0.35`.
- Minimum pocket contact: `contact_fraction >= 0.8` or `pocket_contact_fraction >= 0.8`.
- Multi-seed stability: at least three seeds with persisted `confidence95_low` and `confidence95_high` for validity, strict pocket fit, uniqueness, slot activation, gate activation, leakage, and throughput.
- Baseline deltas: claim-ready wording requires no-slot, no-cross, no-pocket or pocket-centroid, surrogate-objective, deterministic-reranker-only, and calibrated-reranker comparisons to be present in `baseline_comparisons` or the ablation matrix.
- Method contract: claim-bearing wording should cite `method_comparison.active_method` when method-aware artifacts are present and should keep comparison-only methods explicit rather than collapsing them into one unnamed generator path.
- Leakage reviewer rule: `leakage_proxy_mean <= 0.08` is a clean pass, `0.08 < leakage_proxy_mean <= 0.12` is caution-only and must be called out explicitly, and `leakage_proxy_mean > 0.12` fails claim-ready review.
- Leakage regression rule: a reviewed ablation increasing `leakage_proxy_mean` by more than `0.03` relative to the base claim surface is at least `caution` and must be called out explicitly. Treat it as `fail` only when the base run is already above the hard reviewer band or when split leakage checks are not clean.
- Leakage split rule: any protein overlap or duplicated example identifiers across train/val/test is an automatic leakage-review failure regardless of scalar proxy values.

The repository now distinguishes four data-contract labels in reviewer-facing wording:

- `smoke-only`: config validation and parsing path checks only
- `parser-only`: source-format and normalization inspection with no claim-bearing interpretation
- `fallback-heavy`: retained data still depends materially on fallback pocket extraction or weak provenance and must be described explicitly as such
- `claim-bearing`: retained label coverage, source provenance, affinity metadata completeness, approximate-label fraction, and normalization-provenance coverage all clear the configured gates

Reviewer-facing validation should use the single canonical revalidation path:

```bash
./tools/revalidate_reviewer_bundle.sh
```

That script validates the compact regression surface, the repository real-backend gate, the canonical larger-data real-backend surface, the LP-PDBBind refined larger-data benchmark surface, the matching tight-geometry surface, runs `tools/reviewer_env_check.py`, records replay drift reports, rebuilds `docs/evidence_bundle.json`, regenerates `docs/paper_claim_bundle.md`, writes `checkpoints/generator_decision/generator_decision.json`, and persists `docs/reviewer_refresh_report.json`. The generated bundle records backend-threshold results, data-threshold results, split-quality warnings, reranker coefficients, baseline labels, config hashes, reviewer-surface anchors, multi-seed summaries, first-class leakage review verdicts, explicit proxy-vs-benchmark chemistry summaries, stronger-backend companion summaries, claim context, backend-environment fingerprints, determinism controls, replay tolerances, packaged-environment anchors, benchmark-breadth summaries, and the canonical revalidation anchors. Training and experiment summaries now persist deterministic controls plus bounded replay tolerances so reviewer artifacts can distinguish strict replay from tolerated bounded reruns; the repository documents bounded replay as the permanent reviewer guarantee, with `continuity_mode=metadata_only_continuation` and `supports_strict_replay=false`, unless a future implementation adds true optimizer-state replay. The larger-data reviewer path should now point to `checkpoints/pdbbindpp_real_backends`, `checkpoints/lp_pdbbind_refined_real_backends`, and `configs/checkpoints/multi_seed_pdbbindpp_real_backends`. If any threshold fails, keep that blocker explicit in reviewer-facing wording.

For a fresh machine, prefer the packaged reviewer environment path:

```bash
./tools/bootstrap_reviewer_env.sh
REVIEWER_PYTHON=.reviewer-env/bin/python ./tools/revalidate_reviewer_bundle.sh
```

That path is now the standard reviewer revalidation workflow, not an optional extra. It makes reviewer backend setup explicit instead of relying on an already-prepared local Python/RDKit installation, and system `python3` should be treated as fallback only. The generated reviewer reports now also persist whether the effective interpreter was the packaged `.reviewer-env/bin/python` path and whether the packaged environment was ready at refresh time.

Generator-facing work should now use the explicit larger-data decision bundle rather than compact-surface intuition. `docs/evidence_bundle.json` records `efficiency_tradeoffs` and `generator_direction`, `checkpoints/generator_decision/generator_decision.json` persists the promotion decision itself, and `docs/generator_hardening_report.md` makes the current rule reviewer-facing: justify generator-quality changes on `checkpoints/pdbbindpp_real_backends` plus `configs/checkpoints/multi_seed_pdbbindpp_real_backends`, and only revisit major objective changes when those larger held-out-family artifacts show a genuine plateau. Promotion review is now machine-gated as well: `python3 tools/generator_decision_bundle.py --check` fails if the persisted decision artifact has not been refreshed against the canonical larger-data surface, the tight-geometry pressure surface, the larger-data multi-seed summary, or the monitored rollout/objective files.
