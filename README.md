# pocket_diffusion

Rust-first modular research framework for pocket-conditioned representation learning and method-aware molecular generation comparison, with a legacy generation demo kept for compatibility.

The actively extended path provides:

- separate topology / geometry / pocket encoders
- slot-based structured decomposition
- gated cross-modal interaction with configurable lightweight and Transformer-style controlled blocks
- staged auxiliary losses for redundancy, leakage, gate, and slot control
- config-driven data loading from real file formats

## Capability matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Modular topology / geometry / pocket encoders | `implemented` | Separate encoder path used by config-driven training and experiments |
| Slot decomposition and gated cross-modal interaction | `implemented` | Used by the modular research stack |
| Staged training with auxiliary regularizers | `implemented` | Supports both `surrogate_reconstruction` and decoder-anchored `conditioned_denoising` primary objectives |
| Unseen-pocket split experiments | `implemented` | Now reports active heuristic chemistry-validity, docking-hook, and pocket-compatibility metrics on modular decoder candidates |
| Affinity supervision from lightweight parsing | `prototype` | Uses simplified normalization and lightweight parsing/reporting |
| Distance / affinity probe evaluation | `proxy` | Computed from probe heads on held-out examples |
| Direct candidate generation and ranking | `implemented` | Available through iterative `research generate`, method-aware unseen-pocket evaluation, and backward-compatible layered artifacts |
| Diffusion training objective | `planned` | Crate name is historical; the modular stack does not train a diffusion objective today |
| Chemistry validity / docking / pocket-compatibility backend | `implemented` | Heuristic baselines remain active, and repository-supported executable backends are now provided under `tools/` |

## Current focus

The repository now supports a practical "real-data first" path:

- JSON config loading
- lightweight PDB + SDF parsing
- manifest-driven datasets
- PDBbind-like directory discovery
- external CSV/TSV affinity label tables
- protein-level train/val/test splits for unseen-pocket experiments

The active generation path is now also a method platform:

- stable `PocketGenerationMethod` contracts for trainable, heuristic, reranker-only, and future external-wrapper methods
- additive `generation_method` config selection for active-method and fair-comparison execution
- layered candidate outputs with explicit provenance for raw, repaired, inferred, deterministic-proxy, and calibrated-reranked layers
- backward-compatible claim and layer artifacts that retain existing reviewer field names

The included sample dataset is intentionally small, but it uses actual on-disk `*.pdb` and `*.sdf` files so the parsing, collation, training, and evaluation paths are exercised end to end.

## Current readiness

As of April 24, 2026, the repository is past the "architecture-only prototype" stage.

- `cargo test` passes locally for the current tree.
- The compact config-driven path validates, inspects data, trains, and runs unseen-pocket experiments on the bundled mini dataset.
- The compact claim gate at [`checkpoints/claim_matrix/`](checkpoints/claim_matrix) is usable as the default fast regression surface for generator-facing changes.
- The repository-supported real-backend surface at [`checkpoints/real_backends/`](checkpoints/real_backends) runs successfully on the current machine with `python3` plus RDKit-backed chemistry checks.
- The canonical larger-data held-out-pocket surface at [`checkpoints/pdbbindpp_real_backends/`](checkpoints/pdbbindpp_real_backends) now runs end to end in the current environment with `benchmark_evidence.evidence_tier=external_benchmark_backed`, `test.strict_pocket_fit_score=0.6752`, `test.leakage_proxy_mean=0.0506`, and `backend atom_coverage_fraction=0.8810`.
- A second larger-data benchmark surface at [`checkpoints/lp_pdbbind_refined_real_backends/`](checkpoints/lp_pdbbind_refined_real_backends) now also passes the same reviewer contract with `parsed_examples=5048`, `retained_label_coverage=1.0`, `test.strict_pocket_fit_score=0.7599`, and `test.leakage_proxy_mean=0.0569`.
- The reviewer refresh bundle is currently green again: [`docs/reviewer_refresh_report.json`](docs/reviewer_refresh_report.json) reports `reviewer_bundle_status=pass`.

In practical terms, the project is now suitable for real research iteration, regression gating, and backend-backed held-out-pocket evaluation. It should still be described as a research framework rather than a finished production system or a fully settled paper result.

## Repository status

This repository currently contains two parallel surfaces:

- a legacy `PocketDiffusionPipeline` demo path for direct candidate generation and ranking
- a newer modular research stack for multi-modal representation learning, staged training, method-aware generation, and unseen-pocket evaluation

The modular research path is the actively extended surface and is the one documented below.

For library consumers, the same boundary now exists in code:

- modular research APIs live under `pocket_diffusion::{config,data,models,training,experiments}`
- legacy demo/comparison APIs are grouped under `pocket_diffusion::legacy`

New integrations should prefer module-qualified imports. The crate root is no longer treated as a flat umbrella namespace for the modular stack.
The remaining crate-root compatibility modules and legacy pipeline type are intentionally retained only for backwards compatibility and now carry stronger deprecation/doc-discovery signals; new modeling, data, trainer, and evaluation work should not land there.

### Library import pattern

For new integrations, prefer explicit module-qualified imports:

```rust
use pocket_diffusion::{
    experiments,
    runtime,
    training,
};
```

For older demo or comparison utilities, prefer the `legacy` namespace explicitly:

```rust
use pocket_diffusion::legacy;
```

### Approved landing zones

For new work, keep code in the modular stack:

- modeling and controlled cross-modal interaction: `src/models/*`
- data loading, parsing, and batching: `src/data/*`
- losses and staged optimization: `src/losses/*`, `src/training/*`
- experiments and reviewer-facing evaluation: `src/experiments/*`

Avoid placing new claim-bearing logic in these compatibility surfaces:

- `src/dataset.rs`
- `src/experiment.rs`
- `PocketDiffusionPipeline`
- `legacy-demo` and hidden compatibility CLI aliases

## Quick start

Inspect the sample dataset:

```bash
cargo run --bin pocket_diffusion -- research inspect --config configs/research_manifest.json
```

Run a short staged training job from config:

```bash
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json
```

Resume that training run from the latest checkpoint in `training.checkpoint_dir`:

```bash
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json --resume
```

Run the unseen-pocket experiment from config:

```bash
cargo run --bin pocket_diffusion -- research experiment --config configs/unseen_pocket_manifest.json
```

Run the modular generation demo and emit conditioned candidates plus evaluation artifacts:

```bash
cargo run --bin pocket_diffusion -- research generate --config configs/research_manifest.json --resume --num-candidates 4
```

The default active method remains `conditioned_denoising`, but config-driven experiment surfaces now read `generation_method.active_method` and execute additive method comparisons when enabled.

Run the configured ablation matrix and emit comparison artifacts:

```bash
cargo run --bin pocket_diffusion -- research ablate --config configs/unseen_pocket_manifest.json
```

Run the compact claim-bearing matrix with a dedicated artifact layout:

```bash
cargo run --bin pocket_diffusion -- research ablate --config configs/unseen_pocket_claim_matrix.json
```

Treat this compact claim-matrix run as the default generator regression gate. Generator-facing changes should refresh [`checkpoints/claim_matrix/claim_summary.json`](checkpoints/claim_matrix/claim_summary.json) before broader interpretation or additional evaluation.

Run the repository-supported external chemistry and pocket-aware backends:

```bash
cargo run --bin pocket_diffusion -- research experiment --config configs/unseen_pocket_real_backends.json
```

Inspect a PDBbind-like directory using an `INDEX`-style label file:

```bash
cargo run --bin pocket_diffusion -- research inspect --config configs/research_pdbbind_index.json
```

Validate configs without loading data or starting training:

```bash
cargo run --bin pocket_diffusion -- validate --kind research --config configs/research_manifest.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_manifest.json
```

Re-print a stored run summary from the shared artifact bundle:

```bash
cargo run --bin pocket_diffusion -- report --artifact-dir checkpoints
```

## Config files

Example configs live under [`configs/`](configs).

- [`configs/research_manifest.json`](configs/research_manifest.json) loads a dataset from an explicit manifest
- [`configs/research_pdbbind_dir.json`](configs/research_pdbbind_dir.json) scans a PDBbind-like directory tree
- [`configs/research_pdbbind_index.json`](configs/research_pdbbind_index.json) scans a PDBbind-like directory tree and attaches labels from an `INDEX`-style file
- [`configs/unseen_pocket_manifest.json`](configs/unseen_pocket_manifest.json) runs the full unseen-pocket experiment
- [`configs/unseen_pocket_real_backends.json`](configs/unseen_pocket_real_backends.json) enables the repository-supported executable chemistry and pocket-aware backend workflow
- [`configs/unseen_pocket_claim_matrix.json`](configs/unseen_pocket_claim_matrix.json) runs the compact claim-bearing ablation matrix
- [`configs/unseen_pocket_harder_pressure.json`](configs/unseen_pocket_harder_pressure.json) runs the harder-pressure external-backend surface with interaction-style ablations enabled
- [`configs/unseen_pocket_tight_geometry_pressure.json`](configs/unseen_pocket_tight_geometry_pressure.json) adds a second pressure surface with tighter geometry and rollout constraints
- [`configs/unseen_pocket_lp_pdbbind_refined_real_backends.json`](configs/unseen_pocket_lp_pdbbind_refined_real_backends.json) adds a second larger-data external-benchmark reviewer surface backed by the LP-PDBBind refined split
- [`configs/unseen_pocket_surrogate_objective.json`](configs/unseen_pocket_surrogate_objective.json), [`configs/unseen_pocket_no_slots.json`](configs/unseen_pocket_no_slots.json), and [`configs/unseen_pocket_reduced_cross_modal.json`](configs/unseen_pocket_reduced_cross_modal.json) provide named single-variant ablations

### `DataConfig` fields

- `dataset_format`: one of `synthetic`, `manifest_json`, `pdbbind_like_dir`
- `root_dir`: root directory for dataset discovery
- `manifest_path`: required when `dataset_format` is `manifest_json`
- `label_table_path`: optional CSV/TSV or PDBbind-like index text with `example_id` and/or `protein_id` plus either `affinity_kcal_mol`, `affinity_record`, or `measurement_type/raw_value[/raw_unit]`
- `parsing_mode`: `lightweight` or `strict`
- `pocket_cutoff_angstrom`: ligand-centered pocket extraction radius
- `max_examples`: optional truncation for quick debugging
- `val_fraction`, `test_fraction`, `split_seed`: protein-level split controls
- `stratify_by_measurement`: preserve measurement-family balance when assigning protein groups to train/val/test
- `quality_filters`: optional reproducible real-data gates for claim-bearing surfaces, including `min_label_coverage`, `max_fallback_fraction`, atom-count ceilings, `require_source_structure_provenance`, `require_affinity_metadata`, `max_approximate_label_fraction`, and `min_normalization_provenance_coverage`
- `generation_target`: deterministic decoder supervision plus iterative rollout config with `atom_mask_ratio`, `coordinate_noise_std`, `corruption_seed`, `rollout_steps`, `min_rollout_steps`, `stop_probability_threshold`, `coordinate_step_scale`, `training_step_weight_decay`, rollout `rollout_mode`, and stronger `momentum_refine` controls including `coordinate_momentum`, `atom_momentum`, `atom_commit_temperature`, `max_coordinate_delta_norm`, `stop_delta_threshold`, and `stop_patience`

`lightweight` mode keeps the convenience-oriented behavior used by the sample configs: first discovered `.pdb`/`.sdf` files are accepted and empty-cutoff pocket extraction falls back to nearest atoms. `strict` mode rejects ambiguous discovery and rejects nearest-atom pocket fallback.

For real-data review surfaces, `dataset_validation.json` now records retained measurement-family histograms, retained approximate-label fraction, retained normalization-provenance coverage, retained source-structure provenance coverage, and filter-drop totals for missing affinity metadata. That makes claim-bearing runs distinguishable from parser-only inspection runs without reading code.
The same artifact now also records label-table row accounting: rows seen, blank/comment rows skipped, duplicate `example_id` / `protein_id` label rows overwritten by later rows, and unmatched loaded labels that never attach to any dataset entry.

### `TrainingConfig` fields

- `affinity_weighting`: `none` or `inverse_frequency` for labeled affinity supervision across mixed measurement families
- `primary_objective`: `surrogate_reconstruction` or `conditioned_denoising`

### `GenerationMethodConfig` fields

- `active_method`: primary method id used by config-driven generation and claim-bearing experiment paths
- `comparison_methods`: additive methods executed on the same split and backend surface for fair comparison
- `candidate_count`: requested candidate count per method execution
- `enable_comparison_runner`: enables persisted method-comparison summaries in experiment artifacts

The built-in method ids currently are:

- `conditioned_denoising`
- `heuristic_raw_rollout_no_repair`
- `pocket_centroid_repair_proxy`
- `deterministic_proxy_reranker`
- `calibrated_reranker`
- `flow_matching_stub`
- `diffusion_stub`
- `autoregressive_stub`
- `external_wrapper_stub`

### `ModelConfig` additions

- `interaction_mode`: `lightweight` or `transformer` for controlled cross-modal interaction
- `interaction_ff_multiplier`: feed-forward width multiplier used when `interaction_mode=transformer`

### `UnseenPocketExperimentConfig` additions

- `external_evaluation`: optional external command adapters for chemistry validity, docking, and pocket-compatibility scoring
- `reviewer_benchmark.dataset`: optional stable benchmark label used when a surface should qualify for the explicit `external_benchmark_backed` chemistry tier
- `ablation_matrix`: lightweight config-driven matrix over objective choice, slots, cross-modal interaction reporting, and interaction-style comparisons

### Method-aware artifacts

Existing reviewer-consumed fields remain stable:

- `raw_rollout`
- `repaired_candidates`
- `inferred_bond_candidates`
- `deterministic_proxy_candidates`
- `reranked_candidates`

Method-platform metadata is now added additively:

- `active_method`
- `method_layer_outputs`
- `method_comparison`

The detailed architecture note lives in [docs/generation_method_platform.md](docs/generation_method_platform.md).

### Repository-supported backend workflow

The repository now ships two executable backend helpers under [`tools/`](tools):

- [`tools/rdkit_validity_backend.py`](tools/rdkit_validity_backend.py): chemistry-validity contract that reads candidate JSON, uses RDKit when available, and emits toolkit-backed validity metrics
- [`tools/pocket_contact_backend.py`](tools/pocket_contact_backend.py): pocket-aware downstream scoring contract that reads real source protein structures from candidate provenance and emits contact/clash/coverage metrics

The external backend adapter contract is:

```text
<executable> <args...> <input_candidates.json> <output_metrics.json>
```

`input_candidates.json` contains the generated candidates plus source structure provenance. `output_metrics.json` must be a JSON object mapping metric names to numeric values. When an external backend is enabled, the persisted experiment metrics keep the external metrics and also retain `heuristic_*` reference metrics for side-by-side interpretation.

`tools/rdkit_validity_backend.py` requires an environment where `python3` can import `rdkit`. If RDKit is unavailable, the backend exits successfully and reports `backend_import_error=1.0`, which keeps artifact semantics explicit instead of silently falling back.

The current workspace now satisfies that contract. On April 22, 2026, the repository-local check `python3 -c 'import rdkit'` succeeded after installing Arch's `rdkit` package, and a fresh run of `configs/unseen_pocket_real_backends.json` emitted RDKit-backed chemistry metrics in [`checkpoints/real_backends/claim_summary.json`](checkpoints/real_backends/claim_summary.json), including `rdkit_available=1.0`, `rdkit_parseable_fraction=1.0`, `rdkit_sanitized_fraction=1.0`, and `rdkit_finite_conformer_fraction=1.0`.

`tools/pocket_contact_backend.py` now treats candidates with missing or non-finite coordinates as unscorable rows instead of crashing. Those rows contribute to `backend_missing_structure_fraction`, which keeps backend failure semantics explicit when the generator emits invalid coordinates.

The compact claim-matrix run emits persisted artifacts under [`checkpoints/claim_matrix/`](checkpoints/claim_matrix), including [`claim_summary.json`](checkpoints/claim_matrix/claim_summary.json) and [`ablation_matrix_summary.json`](checkpoints/claim_matrix/ablation_matrix_summary.json). On April 23, 2026, rerunning that surface with the Transformer-style interaction block kept `candidate_valid_fraction=1.0`, `pocket_contact_fraction=1.0`, and `pocket_compatibility_fraction=1.0` for the base run while expanding the persisted review surface to include `interaction_mode`, topology/geometry/pocket specialization scores, and slot/gate/leakage utilization summaries. This compact matrix remains the repository's mandatory first-pass regression gate for generator work because it is fast, finite, and evidence-bearing.

The interaction-style ablation is now explicit on the stronger reviewer surfaces as well as the compact gate. The current aggregate decision artifact at [`checkpoints/interaction_mode_review.json`](checkpoints/interaction_mode_review.json) compares the canonical larger-data real-backend surface against the tight-geometry pressure surface and includes pocket-fit, chemistry-quality, specialization, leakage, coverage, and throughput deltas for Transformer vs lightweight interaction.

On the canonical larger-data surface, the current comparison now clearly favors Transformer as the default claim path. [`checkpoints/pdbbindpp_real_backends_interaction_review/interaction_mode_review.json`](checkpoints/pdbbindpp_real_backends_interaction_review/interaction_mode_review.json) shows test `strict_pocket_fit_score=0.9111` for Transformer vs `0.5234` for lightweight, lower leakage (`0.0647` vs `0.0911`), better centroid fit, and higher throughput-backed chemistry/coverage quality while both modes preserve `candidate_valid_fraction=1.0` and `unique_smiles_fraction=1.0`.

The tight-geometry surface remains a legitimate two-mode ablation rather than a clean Transformer-only win. [`checkpoints/tight_geometry_pressure/interaction_mode_review.json`](checkpoints/tight_geometry_pressure/interaction_mode_review.json) keeps the tradeoff bounded: lightweight still improves strict pocket fit (`0.6575` vs `0.6178`), while Transformer keeps lower leakage (`0.0613` vs `0.0797`) and slightly stronger geometry/pocket specialization. The current default is therefore: use Transformer on the canonical larger-data claim path, but retain lightweight as an explicit pressure-surface ablation.

The real-backend validation run still emits persisted artifacts under [`checkpoints/real_backends/`](checkpoints/real_backends). In the current repository environment, the chemistry backend now reports RDKit-backed metrics instead of `backend_import_error`, while the downstream pocket-aware backends preserve their explicit scoring and missing-structure semantics. On April 22, 2026, rerunning [`configs/unseen_pocket_real_backends.json`](configs/unseen_pocket_real_backends.json) after the same generator-quality pass refreshed [`checkpoints/real_backends/claim_summary.json`](checkpoints/real_backends/claim_summary.json) to `candidate_valid_fraction=1.0`, `pocket_contact_fraction=1.0`, `pocket_compatibility_fraction=1.0`, alongside RDKit-backed chemistry metrics such as `rdkit_parseable_fraction=1.0` and `rdkit_sanitized_fraction=1.0`.

The stronger backend companion now lives at [`configs/unseen_pocket_vina_backend.json`](configs/unseen_pocket_vina_backend.json) and persists artifacts under [`checkpoints/vina_backend/`](checkpoints/vina_backend). Its claim artifact now carries a first-class `backend_review` policy with explicit reviewer pass/fail status, Vina availability, Vina-input completeness, and docking score coverage. When Vina or required PDBQT inputs are unavailable, the persisted report stays non-passing instead of silently degrading into heuristic-only reviewer wording; when those prerequisites are present and all reviewed candidates are scored, the same surface is intended to support claim-bearing backend companion evidence.

The canonical larger-data reviewer surface lives at [`checkpoints/pdbbindpp_real_backends/`](checkpoints/pdbbindpp_real_backends). Its current refreshed artifact keeps `candidate_valid_fraction=1.0`, `unique_smiles_fraction=1.0`, `clash_fraction=0.0`, `benchmark_evidence.evidence_tier=external_benchmark_backed`, and now improves to `test.strict_pocket_fit_score=0.6752` with `test.leakage_proxy_mean=0.0506`. The stronger tier is explicit: it keeps the backend-backed chemistry quality and reviewer benchmark-plus checks, then adds the fact that this surface is the canonical held-out-pocket PDBbind++ benchmark dataset path with passing dataset-size and held-out-family coverage.

On April 24, 2026, the same canonical surface was rerun successfully in the current workspace with [`configs/unseen_pocket_pdbbindpp_real_backends.json`](configs/unseen_pocket_pdbbindpp_real_backends.json). The refreshed artifact again kept `candidate_valid_fraction=1.0`, `pocket_contact_fraction=1.0`, `pocket_compatibility_fraction=1.0`, `unique_smiles_fraction=1.0`, `rdkit_sanitized_fraction=1.0`, `clash_fraction=0.0`, and `benchmark_evidence.evidence_tier=external_benchmark_backed`, with `parsed_examples=5316`, `retained_label_coverage=0.9668`, `val examples=77`, and `test examples=77`. It also reduced the external pocket-backend coverage gap from `atom_coverage_fraction=0.7901` to `0.8810`, with the remaining misses now concentrated in a small set of explicit pocket-clash/chemistry-sanity failure examples recorded in [`generation_layers_test.json`](checkpoints/pdbbindpp_real_backends/generation_layers_test.json).

The second larger-data benchmark reviewer surface now lives at [`checkpoints/lp_pdbbind_refined_real_backends/`](checkpoints/lp_pdbbind_refined_real_backends). It is driven by [`configs/unseen_pocket_lp_pdbbind_refined_real_backends.json`](configs/unseen_pocket_lp_pdbbind_refined_real_backends.json) plus the local LP-PDBBind refined label contract, and its refreshed artifact keeps `benchmark_evidence.evidence_tier=external_benchmark_backed`, `parsed_examples=5048`, `retained_label_coverage=1.0`, `candidate_valid_fraction=1.0`, `unique_smiles_fraction=1.0`, `clash_fraction=0.0`, `test.strict_pocket_fit_score=0.7599`, and `test.leakage_proxy_mean=0.0569`.

The second pressure surface now lives at [`configs/unseen_pocket_tight_geometry_pressure.json`](configs/unseen_pocket_tight_geometry_pressure.json) and persists artifacts under [`checkpoints/tight_geometry_pressure/`](checkpoints/tight_geometry_pressure). On April 23, 2026, the current canonical artifact holds `candidate_valid_fraction=1.0`, `pocket_contact_fraction=1.0`, `pocket_compatibility_fraction=1.0`, `strict_pocket_fit_score=0.6178`, `rdkit_sanitized_fraction=1.0`, `clash_fraction=0.0`, and `leakage_proxy_mean=0.0613` with `leakage_calibration.reviewer_status=pass`.

Reviewer-facing chemistry/generalization breadth is now a true multi-benchmark package rather than a single-anchor bundle. The current reviewer bundle carries two external-benchmark-backed larger-data surfaces, [`checkpoints/pdbbindpp_real_backends/`](checkpoints/pdbbindpp_real_backends) and [`checkpoints/lp_pdbbind_refined_real_backends/`](checkpoints/lp_pdbbind_refined_real_backends), plus the tighter-geometry pressure surface and the repository real-backend gate as persisted companion review surfaces. The breadth summary is regenerated into [`docs/evidence_bundle.json`](docs/evidence_bundle.json) and the paper-facing mapping in [`docs/paper_claim_bundle.md`](docs/paper_claim_bundle.md).

Canonical reviewer revalidation now has a single local entry point:

```bash
./tools/revalidate_reviewer_bundle.sh
```

That workflow validates `checkpoints/claim_matrix`, `checkpoints/real_backends`, `checkpoints/pdbbindpp_real_backends`, `checkpoints/lp_pdbbind_refined_real_backends`, `checkpoints/tight_geometry_pressure`, and `configs/checkpoints/multi_seed_pdbbindpp_real_backends`, runs [`tools/reviewer_env_check.py`](tools/reviewer_env_check.py), records bounded replay drift reports with [`tools/replay_drift_check.py`](tools/replay_drift_check.py), rebuilds [`docs/evidence_bundle.json`](docs/evidence_bundle.json), regenerates the paper-facing claim map in [`docs/paper_claim_bundle.md`](docs/paper_claim_bundle.md), writes the standalone generator decision artifact at [`checkpoints/generator_decision/generator_decision.json`](checkpoints/generator_decision/generator_decision.json), and persists the auditable refresh decision in [`docs/reviewer_refresh_report.json`](docs/reviewer_refresh_report.json). Promotion is now explicitly bounded-replay-based rather than ambiguous strict-replay wording.

The packaged reviewer environment is now the standard revalidation path. The repository ships [`reviewer_env/environment.yml`](reviewer_env/environment.yml) plus [`tools/bootstrap_reviewer_env.sh`](tools/bootstrap_reviewer_env.sh); after bootstrapping, run `REVIEWER_PYTHON=.reviewer-env/bin/python ./tools/revalidate_reviewer_bundle.sh`. Use system `python3` only as a fallback when `.reviewer-env/bin/python` is unavailable or intentionally bypassed.

Remaining generator work is now explicitly evidence-gated. [`docs/generator_hardening_report.md`](docs/generator_hardening_report.md) summarizes the main reviewer surfaces as a quality/efficiency tradeoff table, [`docs/evidence_bundle.json`](docs/evidence_bundle.json) records the underlying tradeoffs and stronger-backend companion summaries, and [`checkpoints/generator_decision/generator_decision.json`](checkpoints/generator_decision/generator_decision.json) persists the promotion decision itself. Major objective changes such as diffusion should not be justified from compact-only wins; they now require larger-data held-out-family evidence plus the matching multi-seed stability surface. The repository now also checks this contract directly with `python3 tools/generator_decision_bundle.py --check`, which fails when the persisted decision artifact is stale relative to the canonical larger-data, tight-geometry, multi-seed, or promotion-relevant rollout/objective inputs.

Config validation is fail-fast for the config-driven research path. Invalid split fractions, missing manifest paths, impossible stage boundaries, empty device strings, and incompatible checkpoint directory settings are rejected before data loading or training starts.

## Dataset layout

### Manifest mode

Manifest mode expects a JSON file like:

```json
{
  "entries": [
    {
      "example_id": "complex-a",
      "protein_id": "protein-a",
      "pocket_path": "protein_a/protein.pdb",
      "ligand_path": "protein_a/ligand.sdf"
    }
  ]
}
```

Relative paths are resolved from the manifest directory.

### External label table

Affinity labels can be provided in a separate CSV/TSV file:

```csv
example_id,protein_id,measurement_type,raw_value,raw_unit
complex-a,protein-a,Kd,35,nM
complex-b,protein-b,IC50,1.2,uM
```

You can also use a single `affinity_kcal_mol` column or a compact `affinity_record` column such as `Kd=35nM`. Matching is applied by `example_id` first, then by `protein_id`. This works for both manifest mode and PDBbind-like directory scan mode.

PDBbind-like index text is also supported. Non-comment lines are treated as whitespace-delimited records, the first token is used as `protein_id`, and the parser recognizes compact or split affinity records such as `Kd=35nM`, `Ki 120 nM`, `IC50=1.2uM`, and `pKd 7.3`. These are normalized to the unified internal `affinity_kcal_mol` target.

Affinity normalization provenance is now recorded per example. Direct `dG` values are tagged as direct energy targets, concentration-style families such as `Kd` and `Ki` are tagged as concentration-derived conversions, and approximate families such as `IC50` and `EC50` emit explicit warnings because they are treated as proxy concentration targets rather than strictly comparable thermodynamic measurements.

The intended data contracts are now:

- `smoke-only`: parsing succeeds, but no claim-bearing quality filters are required
- `parser-only`: use parsing/inspection outputs to debug source files and normalization warnings without interpreting the resulting labels as reviewer-grade evidence
- `fallback-heavy`: retained examples still depend materially on nearest-atom pocket fallback or weak provenance; keep this explicit and avoid claim-ready wording
- `claim-bearing`: require reproducible quality gates on label coverage, source-structure provenance, affinity metadata completeness, approximate-label fraction, and normalization-provenance coverage

For claim-bearing interpretation, reviewer-facing text should rely on the persisted label accounting rather than hand-waving mixed assay inputs: measurement-family histograms, normalization provenance values, duplicate-key overwrites, and unmatched label rows are now first-class parts of the dataset contract.

```text
# pdb_id  resolution  year  affinity_record
1abc  1.80  2020  Kd=35nM
2def  2.10  2020  Ki 120 nM
```

### PDBbind-like directory mode

Directory scan mode expects:

```text
dataset_root/
  complex_001/
    *.pdb
    *.sdf
  complex_002/
    *.pdb
    *.sdf
```

The loader takes the first `*.pdb` and first `*.sdf` found in each complex directory.

## Evaluation

The unseen-pocket experiment now reports four explicit sections:

- `representation_diagnostics`
- `proxy_task_metrics`
- `split_context`
- `resource_usage`
- `real_generation_metrics`

Representative fields include:

- `finite_forward_fraction`
- `unique_complex_fraction`
- `unseen_protein_fraction`
- `distance_probe_rmse`
- `topology_pocket_cosine_alignment`
- `topology_reconstruction_mse`
- `affinity_probe_mae`
- `affinity_probe_rmse`
- per-measurement affinity `MAE/RMSE` breakdown
- slot / gate / leakage diagnostic means

The first four sections are diagnostics and proxy metrics over held-out examples. `real_generation_metrics` is now active on the modular path through heuristic backend adapters that operate on decoder-emitted ligand candidates.

The current modular experiment path reports separate entries for:

- chemistry validity
- docking / affinity rescoring
- downstream pocket compatibility

The extension points still live in trait form under `pocket_diffusion::models`, and the experiment config can now invoke external executables that read generated candidate JSON and emit backend metrics JSON. That keeps heuristic fallbacks active while providing a real executable integration path for chemistry or docking tools.

Claim-bearing experiment runs now use a stable artifact surface:

- `experiment_summary.json`: full machine-readable run summary
- `claim_summary.json`: compact publishability-oriented summary with the main validation/test claim view and ablation deltas
- `ablation_matrix_summary.json`: compact matrix comparison table when `ablation_matrix.enabled=true`
- `run_artifacts.json`: stable pointer bundle for downstream tooling

Evaluation summaries now also persist a `comparison_summary` section with stable fields for:

- primary objective
- ablation variant label
- candidate validity
- pocket-contact / pocket-compatibility fractions
- geometry-quality summaries such as centroid offset and strict pocket-fit score
- uniqueness summaries such as `unique_smiles_fraction` when chemistry backends emit them
- unseen-pocket generalization context

## Reproducibility artifacts

Config-driven `train` and `experiment` runs now write a shared artifact layout inside `training.checkpoint_dir`:

- `config.snapshot.json`
- `dataset_validation_report.json`
- `dataset_validation.json`
- `split_report.json`
- `training_summary.json` or `experiment_summary.json`
- `ablation_matrix_summary.json` when enabled
- `run_artifacts.json`
- checkpoint weights such as `latest.ot`

`run_artifacts.json` is the stable bundle entrypoint used by the `report` command.

Training and experiment summaries now also persist:

- `config_hash`
- `dataset_validation_fingerprint`
- `metric_schema_version`
- explicit `resume_contract`
- `resume_provenance`

Current resume semantics are intentionally explicit: model weights, step index, prior persisted history, and optimizer/scheduler metadata are restored, but `tch` Adam moment buffers are not. Persisted summaries now record deterministic controls, explicit replay tolerances, `resume_contract.continuity_mode=metadata_only_continuation`, and `resume_provenance.strict_replay_achieved=false`, so a resumed run is treated as bounded replay within reviewer metric tolerances rather than strict optimizer-state-identical replay.

The sample configs now enable measurement-stratified protein splits and inverse-frequency affinity weighting so mixed `Kd/Ki/IC50/pKd` labels do not collapse into a single undifferentiated supervision pool.

## Included sample dataset

The repository ships a tiny dataset under [`examples/datasets/mini_pdbbind`](examples/datasets/mini_pdbbind). It is suitable for smoke tests and for validating the file-format ingestion path.

## Binaries

- `pocket_diffusion`: main binary for config-driven inspection, training, experiments, and the legacy generation demo
- `disentangle_demo`: older standalone demo binary kept for local experimentation

Because the crate defines multiple binaries, use `cargo run --bin pocket_diffusion -- ...` for the commands in this README.

## Research CLI

The modular research stack is the primary path for this repository:

```bash
cargo run --bin pocket_diffusion -- research inspect --config configs/research_manifest.json
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json
cargo run --bin pocket_diffusion -- research experiment --config configs/unseen_pocket_manifest.json
cargo run --bin pocket_diffusion -- research generate --config configs/research_manifest.json --resume
cargo run --bin pocket_diffusion -- validate --kind research --config configs/research_manifest.json
cargo run --bin pocket_diffusion -- report --artifact-dir checkpoints
```

To resume from the latest checkpoint in the configured checkpoint directory:

```bash
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json --resume
```

Legacy demo behavior is still available, but it is now an explicit compatibility path:

```bash
cargo run --bin pocket_diffusion -- legacy-demo 10 3
cargo run --bin pocket_diffusion -- legacy-demo 10 3 --modular-bridge
```

For legacy library-side comparison utilities, prefer the explicit namespace:

- `pocket_diffusion::legacy::run_legacy_demo(...)`
- `pocket_diffusion::legacy::run_comparison_experiment(...)`

For config-driven library entrypoints, prefer:

- `pocket_diffusion::training::inspect_dataset_from_config(...)`
- `pocket_diffusion::training::run_training_from_config(...)`
- `pocket_diffusion::experiments::run_experiment_from_config(...)`

Older one-flag demo shortcuts are still accepted for compatibility, but they are not the primary research interface:

- `--phase1`
- `--train-phase3`
- `--phase4`
- `--inspect-config <path>`
- `--train-config <path> [--resume]`
- `--experiment-config <path> [--resume]`

When these compatibility paths are used through the main binary, the CLI now prints an explicit notice with the corresponding `research ... --config ...` replacement so modular research runs are steered back toward the canonical interface.

Config-driven training writes reproducibility artifacts into `training.checkpoint_dir`:

- `dataset_validation_report.json` and `dataset_validation.json` with discovered complexes, parsed examples, label attachment counts, fallback pocket extraction counts, quality-filter totals, and truncation metadata
- step-indexed checkpoints such as `step-10.ot` and `step-10.json`
- rolling `latest.ot` and `latest.json` pointers for resume
- `training_summary.json` for config-driven training runs, including cumulative `training_history`, dataset split sizes, split audit, resume step, and post-train validation/test metrics
- `experiment_summary.json` for unseen-pocket experiment runs, also preserving cumulative `training_history` across resume together with the split audit

The split audit records per-split unique protein counts, labeled fraction, measurement-family histograms, and explicit leakage checks for protein overlap and duplicated example IDs across train/val/test.

## Capability boundaries

The modular stack currently optimizes a configurable primary objective, with `surrogate_reconstruction` preserved for ablations and `conditioned_denoising` available as a decoder-anchored task objective. Auxiliary losses for consistency, redundancy reduction, probes, leakage control, gate sparsity, and slot control are activated in stages around that primary objective.

What is still not present in the modular path today:

- no active diffusion loss
- no iterative sampler beyond the current research demo candidate emitter
- no external chemistry toolkit backend
- no production docking backend

The word `diffusion` remains in the crate/package name for compatibility. In technical terms, the config-driven path is a modular representation-learning framework with conditioned denoising and deterministic rollout supervision, not a validated diffusion objective/sampler. The only generation path outside that modular stack is the explicitly marked `legacy` demo surface.

## Notes

- The current loader supports minimal V2000 SDF parsing and ligand-centered PDB pocket extraction.
- If no atoms fall within the pocket cutoff, the loader falls back to the nearest atoms instead of producing an empty pocket.
- The formal chemistry and pocket metrics now run on an active heuristic backend. Replacing those heuristics with domain tooling remains the next quality step.
- The bundled dataset is intentionally tiny, so split behavior and per-measurement metrics are useful for plumbing validation, not for claiming model quality.
