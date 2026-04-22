# Repository Audit

## Claim Matrix

| Surface | Claim | Status | Boundary |
| --- | --- | --- | --- |
| Config-driven modular stack | Separate topology / geometry / pocket encoders | `implemented` | Active research path |
| Config-driven modular stack | Slot decomposition and gated cross-modal interaction | `implemented` | Active research path |
| Config-driven modular stack | Staged training with explicit primary + auxiliary objectives | `implemented` | Supports surrogate ablations and decoder-anchored conditioned denoising |
| Config-driven modular stack | Unseen-pocket evaluation | `implemented` | Outputs diagnostics, proxy metrics, and active heuristic candidate metrics |
| Config-driven modular stack | Chemistry-grade generation metrics | `prototype` | Heuristic validity / docking-hook / pocket-compatibility adapters are integrated |
| Config-driven modular stack | Diffusion objective | `planned` | Crate name is historical, not descriptive of the current objective |
| Legacy surface | Candidate generation and ranking demo | `legacy` | Compatibility/demo surface, with optional routing through the modular decoder bridge |
| Lightweight data path | Affinity normalization across mixed measurement families | `prototype` | Simplified normalization contract, not a production assay harmonization stack |

## Current Architecture Summary

The repository currently exposes two parallel surfaces:

- A legacy demo and comparison stack centered around `PocketDiffusionPipeline`, `src/dataset.rs`, and older experiment/demo binaries.
- A newer modular research stack centered around `src/config/*`, `src/data/*`, `src/models/*`, `src/losses/*`, `src/training/*`, and `src/experiments/unseen_pocket.rs`.

The modular stack already reflects the intended research direction:

- separate topology, geometry, and pocket encoders
- slot-based decomposition
- gated cross-modal interaction
- staged training losses
- config-driven data loading from synthetic, manifest, and PDBbind-like sources

It should currently be described as a modular generation-research framework with deterministic decoder supervision, conditioned-denoising training support, and heuristic candidate evaluation. The repository still does not expose a true pocket-conditioned diffusion training loop on the actively extended path.

The repo is usable, but the execution path is still ambiguous because config-driven research flows share a monolithic CLI with legacy demo behavior and some config fields are defined but not faithfully consumed.

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
3. `src/main.rs` now has clearer command routing, but it still contains legacy demo orchestration and human-readable reporting logic, so keeping research execution in `src/training/*` and `src/experiments/*` remains important.
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

- Removed crate-level warning suppressions and replaced them with narrower fixes where necessary.
- Marked legacy dataset/pipeline code as compatibility surface in docs and module comments.
- Added smoke tests for config parsing, device parsing, batch iteration, checkpoint save/load, dataset inspection, and short train/eval paths.
- Updated README instructions so the modular research stack is the primary documented workflow, with legacy demo usage clearly separated.
- Moved unseen-pocket config-driven experiment startup behind a module-level entrypoint so `src/main.rs` remains a thin CLI shell.
- Added an explicit `legacy` comparison wrapper so old benchmark utilities have a clearer namespace instead of relying on crate-root discovery.
- Added namespace-stability coverage for modular and legacy entrypoints so the intended import surfaces remain explicit as the crate evolves.
- Added runtime compatibility notices for legacy/demo CLI paths so the binary itself now reinforces the modular research interface boundary and points users at the canonical `research ... --config ...` replacements.
- Renamed the primary modular training objective as `surrogate_reconstruction` and separated it structurally from auxiliary regularizers in logs and summaries.
- Reworked evaluation summaries so generation-like names were replaced by explicit diagnostic/proxy namespaces.
- Reduced root/module wildcard re-exports so the supported modular surface is explicitly namespaced.
- Added persisted dataset validation and split-audit artifacts so config-driven inspection, training, and unseen-pocket experiments expose discovery counts, label attachment behavior, fallback pocket extraction counts, and explicit leakage checks.

### Remaining Low-Priority Follow-Up

- Keep new experiment or evaluation flows behind module-level entrypoints instead of re-expanding `src/main.rs`.
- Decide whether legacy comparison modules should eventually move physically under `src/legacy/*` instead of only being re-exported there.
- Add optional full optimizer-state resume if reproducibility requirements expand beyond weight-and-step restoration.

## Capability And Limitation Table

| Area | Current status | Limitation |
| --- | --- | --- |
| Representation learning and generation | Strong modular prototype | Decoder path is research-grade, not a production sampler |
| Evaluation | Diagnostic/proxy plus heuristic candidate reporting | No external chemistry toolkit or production docking backend |
| Data ingestion | Lightweight real-file path with deterministic generation targets | Parsing remains convenience-oriented rather than assay-grade |
| Legacy generation demo | Still runnable | Compatibility surface, not the primary abstraction for new work |

## Architecture Note: Gap To A True Diffusion Generator

The modular stack has three pieces that are worth preserving for future generation work: modality-specific encoders, slot decomposition, and gated cross-modal interaction. It now also has a decoder-facing corruption/denoising contract plus a minimal candidate emitter. What it still does not have is the rest of a production-grade generation system:

- no diffusion objective
- no iterative sampler with learned rollout policy
- no external chemistry-aware validity layer
- no downstream production docking backend

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

The repository now keeps its legacy demos intact while making the modular research stack the clean, config-faithful default path for inspection, training, experiments, and compact conditioned-generation demos. The remaining work is incremental model/evaluation hardening rather than architectural correction.
