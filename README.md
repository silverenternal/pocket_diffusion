# pocket_diffusion

Rust-first modular research framework for pocket-conditioned representation learning and method-aware molecular generation.

## Project Map

| Surface | Where to start | Status |
| --- | --- | --- |
| Active Rust research stack | `src/models/`, `src/data/`, `src/training/`, `src/experiments/` | Separate topology/geometry/pocket encoders, slot decomposition, gated cross-modal interaction, staged losses, flow matching, and unseen-pocket experiments |
| Config-driven runs | `configs/research_manifest.json`, `configs/unseen_pocket_manifest.json` | Main path for inspection, training, validation, and experiments |
| Generated evidence | `artifacts/evidence/`, selected `configs/q*_*summary.json`, linked docs under `docs/` | Curated claim artifacts with provenance and validation gates |
| External baseline tooling | `tools/baseline_output_adapters/`, `tools/public_baseline_*`, `tools/*_backend.py` | Public-baseline adapters plus RDKit/Vina/GNINA/pocket backend wrappers |
| Legacy compatibility | `pocket_diffusion -- legacy-demo`, `pocket_diffusion::legacy` | Deprecated compatibility surface; do not use for new research features |

The crate name still says `diffusion` for compatibility. The active path is a modular representation-learning system with conditioned denoising/refinement, de novo pocket-conditioned molecular flow, deterministic rollout diagnostics, and explicit evidence attribution. Full molecular flow is config-gated: `de_novo_initialization` requires `flow_matching.geometry_only=false` plus geometry, atom-type, bond, topology, and pocket/context branches.

## External Dependencies

The core implementation is Rust. Python is used for reviewer tooling, artifact
checks, and optional chemistry/docking backends; it is not the model-training
implementation.

### Required for Rust build and training

| Dependency | Why it is needed | Notes |
| --- | --- | --- |
| Rust stable toolchain + Cargo | Build and run the crate | Edition 2021; install through `rustup` if needed. |
| C/C++ build toolchain | Native dependencies used by `tch`/libtorch | `gcc`/`clang` plus normal linker tools are sufficient on Linux. |
| libtorch / PyTorch C++ runtime | Tensor backend for `tch = 0.23` | Use a libtorch release compatible with the checked-in `Cargo.lock`, or set `LIBTORCH_USE_PYTORCH=1` when using a Python environment that has a compatible `torch` install. |

The default configs run on CPU (`runtime.device=cpu`). CUDA is optional and
must be configured consistently between libtorch/PyTorch and the local driver
stack.

### Required for validation and reviewer tooling

| Dependency | Why it is needed | Install path |
| --- | --- | --- |
| Python 3.12 | Validation scripts, claim gates, backend adapters | `reviewer_env/environment.yml` pins the reviewer Python environment. |
| RDKit | Chemistry validity, QED/logP/scaffold/diversity metrics | `./tools/bootstrap_reviewer_env.sh` creates `.reviewer-env` with RDKit via conda/mamba. |
| `jq` | Shell examples and JSON inspection | Optional for Rust execution, useful for artifact review. |

Create the packaged reviewer environment with:

```bash
./tools/bootstrap_reviewer_env.sh
```

For a lighter environment check:

```bash
.reviewer-env/bin/python tools/reviewer_env_check.py \
  --config configs/unseen_pocket_pdbbindpp_real_backends.json \
  --config configs/unseen_pocket_lp_pdbbind_refined_real_backends.json
```

The checked-in external-evaluation configs use `python3` as the backend
executable. When running those configs with the packaged RDKit environment,
activate `.reviewer-env` so `python3` resolves to that environment, or update
the config executable to `.reviewer-env/bin/python` for the local run.

Run the full reviewer revalidation only when the referenced datasets and
checkpoint artifacts are present:

```bash
REVIEWER_PYTHON=.reviewer-env/bin/python ./tools/revalidate_reviewer_bundle.sh
```

### Optional external backends

| Backend | Used by | Requirement |
| --- | --- | --- |
| AutoDock Vina | `tools/vina_score_backend.py`, `configs/unseen_pocket_vina_backend.json` | A `vina` executable on `PATH`, or `VINA_EXECUTABLE=/path/to/vina`. True Vina scoring also needs receptor/ligand PDBQT inputs. |
| GNINA | `tools/gnina_score_backend.py`, `configs/unseen_pocket_gnina_backend.json` | A `gnina` executable on `PATH`, or `GNINA_EXECUTABLE=/path/to/gnina`. |
| Pandoc | Refresh generated docs such as `docs/dataset_training_methods_zh.html/.docx/.odt` | Not needed for training. |
| Python `torch` | `tools/run_diffsbdd_public_testset.py` public-baseline helper | Not needed for the Rust model unless you choose `LIBTORCH_USE_PYTORCH=1`. |

The backend adapters degrade to explicit availability/input-completeness
metrics when a backend is missing. Missing Vina/GNINA should therefore be read
as missing stronger backend evidence, not as a successful docking run.

## Evidence Boundaries

Claim-facing artifacts include candidate-level RDKit, AutoDock Vina `score_only`, and GNINA `score_only` outputs. These are backend scores, not experimental binding affinities. Proxy metrics such as `docking_like_score` are heuristic diagnostics only.

Layer attribution is strict:

- `raw_rollout` and `raw_flow` are native model evidence.
- `constrained_flow` is constrained-sampling evidence derived from raw flow output.
- `inferred_bond`, `inferred_bond_candidates`, `repaired`, `repaired_candidates`, `deterministic_proxy`, `deterministic_proxy_candidates`, `reranked`, and `reranked_candidates` are postprocessing or selection evidence.

Training/evaluation alignment is explicit in experiment artifacts:

- `L_probe` and `L_leak` are optimizer-facing auxiliary losses when enabled by the staged trainer.
- `training.rollout_training` is the bounded optimizer-facing short-rollout objective when enabled and active; `rollout_eval_*` fields remain detached diagnostics.
- Objective-family budget reports keep `task`, `rollout`, `pocket_interaction`, `chemistry`, `redundancy`, `probe`, `leakage`, `gate`, and `slot` contributions separate.
- `finite_forward_fraction` is a smoke/default health metric; claim-bearing configs should select quality-aware best metrics with availability checks.
- Backend score rows report coverage, missing-structure fraction, fallback use, and candidate counts before they can support stronger wording.

Current public-baseline details live in:

- [`configs/q1_method_comparison_summary.json`](configs/q1_method_comparison_summary.json)
- [`docs/q1_method_comparison_table.md`](docs/q1_method_comparison_table.md)
- [`docs/q1_runtime_efficiency_table.md`](docs/q1_runtime_efficiency_table.md)
- [`docs/postprocessing_failure_audit.md`](docs/postprocessing_failure_audit.md)
- [`docs/q2_claim_contract.md`](docs/q2_claim_contract.md)
- [`docs/q8_reviewer_scale_runbook.md`](docs/q8_reviewer_scale_runbook.md)
- [`docs/q15_generation_alignment_final_contract.md`](docs/q15_generation_alignment_final_contract.md)

The fast validation gate is:

```bash
python tools/validation_suite.py --mode quick --timeout 240
# or
tools/local_ci.sh fast
```

Before using generated molecules in a real review workflow, run the stricter
raw-native gate on the produced artifacts:

```bash
python3 tools/claim_regression_gate.py checkpoints/<artifact_dir> \
  --enforce-real-generation-readiness \
  --multi-seed-summary checkpoints/<multi_seed_dir>/multi_seed_summary.json
```

The same gate is available through local CI when the artifact paths are explicit:

```bash
REAL_GENERATION_ARTIFACT_DIR=checkpoints/<artifact_dir> \
REAL_GENERATION_MULTI_SEED_SUMMARY=checkpoints/<multi_seed_dir>/multi_seed_summary.json \
tools/local_ci.sh real-gen
```

## Quick start

First verify that Rust and libtorch are usable:

```bash
cargo check
```

Then run the bundled mini-dataset path:

```bash
# Inspect dataset
cargo run --bin pocket_diffusion -- research inspect --config configs/research_manifest.json

# Short staged training
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json

# Resume from checkpoint
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json --resume

# Unseen-pocket experiment
cargo run --bin pocket_diffusion -- research experiment --config configs/unseen_pocket_manifest.json

# Validate config (fail-fast, no data loading)
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_manifest.json
```

Paper-quality training (100 steps, 1024 examples):

```bash
./run_paper_training.sh
```

## Generation Demo Boundaries

```bash
cargo run --bin pocket_diffusion -- research generate --config configs/research_manifest.json --num-candidates 2
```

For the demo run, inspect:

- `training.checkpoint_dir/generation_demo_candidates_raw.json` for `raw_flow` candidates (`candidate_layer=raw_flow`, `evidence_role=raw_model_native`).
- `training.checkpoint_dir/generation_demo_candidates_constrained_flow.json` for `constrained_flow` candidates (`candidate_layer=constrained_flow`, `evidence_role=constrained_sampling`).
- `training.checkpoint_dir/generation_demo_summary.json` for the attributed metric blocks.
- `training.checkpoint_dir/generation_demo_candidates.json` for backward-compatible demo candidates (same constrained-flow output).

The demo summary keeps raw metrics and constrained-flow metrics explicit; constrained-flow metrics are used for legacy-facing demo reporting.

## Config files

Key configurations under [`configs/`](configs):

| Config | Purpose |
| --- | --- |
| `research_manifest.json` | Manifest-driven dataset from explicit file list |
| `research_pdbbind_dir.json` | PDBbind-like directory tree scan |
| `unseen_pocket_manifest.json` | Full unseen-pocket experiment |
| `unseen_pocket_real_backends.json` | External chemistry + pocket backends |
| `unseen_pocket_claim_matrix.json` | Compact ablation matrix (regression gate) |
| `q15_generation_alignment_ablation_matrix.json` | Claim-safe generation-alignment matrix over flow head, rollout training, chemistry guardrails, pocket-interaction loss richness, and direct-fusion negative control |
| `flow_matching_experiment.json` | Flow-matching as primary objective |
| `unseen_pocket_pdbbindpp_flow_best_candidate_paper.json` | Paper-quality flow-matching run |
| `q1_method_comparison_summary.json` | Current layer-separated public-baseline performance summary |
| `q1_public_baseline_run_status.json` | Current public-baseline status, coverage, and runtime provenance |
| `q1_baseline_registry.json` | Public baseline source/status registry |

See [Config files (detailed)](#config-files-detailed) below for field documentation.

## Data Requirements

### What is included

The repository includes a tiny smoke dataset:

| Dataset | Path | Role |
| --- | --- | --- |
| Mini PDBbind-like examples | `examples/datasets/mini_pdbbind` | Four small protein-ligand examples for parser checks, smoke training, and CI-style validation. |
| Local PDBbind-like profile data | `data/PDBbind_v2020_refined` | Parser/pressure-test surface if populated locally; a small or unlabeled tree is not a claim-bearing benchmark by itself. |

The full reviewer/claim configs require local real-data assets under `data/`.
Those datasets are not replaced by the mini examples.

### Supported input formats

**Manifest mode** (`dataset_format=manifest_json`) uses an explicit JSON file.
Paths are resolved relative to the manifest file:

```json
{ "entries": [{ "example_id": "x", "protein_id": "p", "pocket_path": "...", "ligand_path": "..." }] }
```

Each entry must point to:

- `pocket_path`: a PDB file containing the protein pocket or receptor context.
- `ligand_path`: an SDF file containing the reference ligand structure.
- `example_id`: stable complex key used for labels and artifacts.
- `protein_id`: grouping key used for unseen-pocket splits.

Optional affinity fields can be embedded directly in the manifest, but the
usual path is a separate `label_table_path`.

**PDBbind-like directory mode** (`dataset_format=pdbbind_like_dir`) scans one
subdirectory per complex:

```
dataset_root/
  complex_001/*.pdb, *.sdf
  complex_002/*.pdb, *.sdf
```

Lightweight parsing picks the first sorted `.pdb` and `.sdf` in each complex
directory. Strict parsing requires exactly one PDB and one SDF per directory.

**Synthetic mode** (`dataset_format=synthetic`) needs no files and is reserved
for smoke tests and focused model diagnostics.

### Affinity label tables

`label_table_path` can be CSV, TSV, or PDBbind INDEX-style text. CSV/TSV rows
must include `example_id` or `protein_id`, plus one of these label forms:

| Label form | Columns | Example |
| --- | --- | --- |
| Direct internal target | `affinity_kcal_mol` or `affinity` or `label` | `-9.4` |
| Compact measurement | `affinity_record` or `measurement` | `Kd=19uM`, `Ki=5.1nM`, `pKd=7.2` |
| Structured measurement | `measurement_type`, `raw_value`, optional `raw_unit` | `Ki`, `5.1`, `nM` |

The loader normalizes supported `Kd`, `Ki`, `IC50`, and `pKd` families into the
internal affinity target and records provenance in `dataset_validation_report.json`.
Claim-bearing configs should require measurement metadata and normalization
coverage through `data.quality_filters`.

### Real benchmark datasets expected by configs

| Config family | Required local assets | Notes |
| --- | --- | --- |
| Mini/smoke configs, e.g. `configs/research_manifest.json`, `configs/unseen_pocket_claim_matrix.json` | `examples/datasets/mini_pdbbind/manifest.json`, `examples/datasets/mini_pdbbind/affinity_labels.csv` | Runs out of the repository with no external dataset. |
| PDBbind++ configs, e.g. `configs/unseen_pocket_pdbbindpp_real_backends.json`, `configs/unseen_pocket_pdbbindpp_flow_canonical.json` | `data/pdbbindpp-2020/manifest.json`, `data/pdbbindpp-2020/affinity_labels.csv`, and structures under `data/pdbbindpp-2020/extracted/pbpp-2020/<pdb_id>/<pdb_id>_pocket.pdb` plus `<pdb_id>_ligand.sdf` | Main larger-data reviewer surface. |
| LP-PDBBind refined configs, e.g. `configs/unseen_pocket_lp_pdbbind_refined_real_backends.json` | `data/lp_pdbbind_refined/manifest.json`, `data/lp_pdbbind_refined/affinity_labels.csv`; the manifest points at the same `data/pdbbindpp-2020/extracted/pbpp-2020/...` structure layout | Second reviewer dataset/label contract. |
| PDBbind-like directory profiles, e.g. `configs/unseen_pocket_medium_profile.json` | A directory tree such as `data/PDBbind_v2020_refined/<pdb_id>/<pdb_id>_protein.pdb` and `<pdb_id>_ligand.sdf` | Useful for parser and scale-up checks; add labels separately if training/evaluation should use affinity supervision. |

Build a PDBbind++ manifest after placing/extracting the upstream structure
files:

```bash
python3 tools/build_pdbbindpp_manifest.py \
  --root data/pdbbindpp-2020/extracted/pbpp-2020 \
  --output data/pdbbindpp-2020/manifest.json
```

Build the compact affinity table from an upstream CSV that contains PDB ids,
refined/general category metadata, and compact `Kd/Ki`-style affinity records:

```bash
python3 tools/build_pdbbindpp_affinity_labels.py \
  --source-csv /path/to/upstream_pdbbindpp_affinity.csv \
  --manifest data/pdbbindpp-2020/manifest.json \
  --output data/pdbbindpp-2020/affinity_labels.csv
```

Validate a data-bearing config before training:

```bash
cargo run --bin pocket_diffusion -- validate --kind experiment \
  --config configs/unseen_pocket_pdbbindpp_real_backends.json

cargo run --bin pocket_diffusion -- research inspect \
  --config configs/research_pdbbindpp_profile.json
```

For a custom dataset, prefer `manifest_json` first. It is more auditable than
directory discovery because every example has explicit source paths and stable
IDs.

### Data contracts

| Contract | Description |
| --- | --- |
| `smoke-only` | Parsing succeeds, no quality gates |
| `parser-only` | Debug source files and normalization warnings |
| `fallback-heavy` | Retained examples depend on fallback/extrapolation |
| `claim-bearing` | Reproducible quality gates on label coverage, provenance, metadata |

Ligand-centered coordinate and pocket extraction are tracked in `dataset_validation_report.json`. Target-ligand refinement configs may allow ligand-centered context but must keep that dependency visible in the report. De novo model execution recenters pocket features for the generated scaffold and uses target ligand fields only as training supervision; claim-bearing dataset configs should still enable `quality_filters.reject_target_ligand_context_leakage=true` to keep retained data contracts strict. Source-coordinate reconstruction support and generation coordinate-frame contracts are also persisted so candidate artifacts cannot silently mix coordinate frames.

## Evaluation

Unseen-pocket experiments report:

- **Representation diagnostics**: `finite_forward_fraction`, `unique_complex_fraction`, `unseen_protein_fraction`
- **Probe metrics**: `distance_probe_rmse`, `topology_pocket_cosine_alignment`, `affinity_probe_mae/rmse`
- **Generation metrics**: chemistry validity, docking rescoring, pocket compatibility
- **Utilization**: slot/gate/leakage diagnostics, semantic specialization scores
- **Train/eval alignment**: objective coverage, detached rollout diagnostics, backend coverage, and best-metric review

External backends (executables reading candidate JSON, emitting metrics JSON):
- [`tools/rdkit_validity_backend.py`](tools/rdkit_validity_backend.py) — chemistry validity via RDKit
- [`tools/pocket_contact_backend.py`](tools/pocket_contact_backend.py) — pocket-aware scoring
- [`tools/vina_score_backend.py`](tools/vina_score_backend.py) — AutoDock Vina `score_only` candidate scores with coverage/runtime metadata
- [`tools/gnina_score_backend.py`](tools/gnina_score_backend.py) — GNINA `score_only` affinity/CNN metrics with coverage/runtime metadata

Public-baseline comparison helpers:

- [`tools/baseline_output_adapters/targetdiff_meta_generation_layers_adapter.py`](tools/baseline_output_adapters/targetdiff_meta_generation_layers_adapter.py) adapts official Pocket2Mol/TargetDiff public sampling metadata.
- [`tools/run_diffsbdd_public_testset.py`](tools/run_diffsbdd_public_testset.py) generates DiffSBDD candidates from the public checkpoint on the same 100-pocket set.
- [`tools/public_baseline_postprocess_layers.py`](tools/public_baseline_postprocess_layers.py) creates explicit repaired/reranked postprocessing layers from raw public-baseline outputs.
- [`tools/public_baseline_method_comparison.py`](tools/public_baseline_method_comparison.py) renders the current layer-separated comparison tables from candidate JSONL artifacts.

## Artifacts

Config-driven runs write to `training.checkpoint_dir`:

| Artifact | Purpose |
| --- | --- |
| `config.snapshot.json` | Exact config used |
| `dataset_validation_report.json` | Discovery/parsing statistics |
| `split_report.json` | Train/val/test assignment |
| `training_summary.json` | Full run summary with history |
| `experiment_summary.json` | Experiment metrics |
| `claim_summary.json` | Compact publishability view |
| `run_artifacts.json` | Stable pointer bundle |
| `latest.ot` / `step-N.ot` | Checkpoint weights |
| `candidate_metrics_<split>.jsonl` | Candidate-level generation/backend metrics with layer attribution |
| `generation_layers_<split>.json` | Layered generation summary and coordinate-frame contract |
| `raw_native_generation_report.json` | Raw-native-first generation claim review with processed evidence kept additive |
| `ablation_matrix_summary.json` | Ablation rows with raw generation quality, runtime, and objective-family behavior |

Q1 public-baseline artifacts:

| Artifact | Purpose |
| --- | --- |
| [`configs/q1_method_comparison_summary.json`](configs/q1_method_comparison_summary.json) | Machine-readable 100-pocket layer-separated public-baseline summary |
| [`docs/q1_method_comparison_table.md`](docs/q1_method_comparison_table.md) | Human-readable method/layer performance table |
| [`configs/q1_public_baseline_full100_layered_metric_coverage.json`](configs/q1_public_baseline_full100_layered_metric_coverage.json) | RDKit/Vina/GNINA coverage for 900 layered candidates |
| [`configs/q1_public_baseline_full100_postprocess_report.json`](configs/q1_public_baseline_full100_postprocess_report.json) | Provenance for repaired/reranked public-baseline rows |
| [`configs/q1_public_baseline_run_status.json`](configs/q1_public_baseline_run_status.json) | Canonical public-baseline run registry/status |

## Reproducibility

- Config hash + dataset fingerprint persisted per run
- Resume restores weights, step index, optimizer/scheduler metadata (not Adam moment buffers)
- Measurement-stratified protein splits + inverse-frequency affinity weighting for mixed `Kd/Ki/IC50` labels
- See [Artifacts](#artifacts) for persisted run metadata and evidence files

## Config files (detailed)

### `DataConfig` fields

| Field | Description |
| --- | --- |
| `dataset_format` | `synthetic`, `manifest_json`, `pdbbind_like_dir` |
| `root_dir` | Root directory for dataset discovery |
| `manifest_path` | Required when `dataset_format=manifest_json` |
| `label_table_path` | Optional CSV/TSV with affinity labels |
| `parsing_mode` | `lightweight` or `strict` |
| `pocket_cutoff_angstrom` | Ligand-centered pocket extraction radius |
| `max_examples` | Optional truncation for debugging |
| `val_fraction`, `test_fraction`, `split_seed` | Protein-level split controls |
| `stratify_by_measurement` | Preserve measurement-family balance in splits |
| `quality_filters` | Reproducible quality gates for claim-bearing surfaces |

`quality_filters` can also enforce held-out diversity thresholds through `min_validation_protein_families`, `min_test_protein_families`, `min_validation_measurement_families`, and `min_test_measurement_families`. Training and unseen-pocket experiment entrypoints enforce those thresholds for claim-bearing runs; dataset inspection reports the same split-quality contract without turning it into a training failure.

### `TrainingConfig` fields

| Field | Description |
| --- | --- |
| `primary_objective` | `surrogate_reconstruction` (bootstrap/debug or shape-safe baseline), `conditioned_denoising` (decoder-anchored denoising/refinement), `flow_matching` (flow-refinement velocity objective), `denoising_flow_matching` (hybrid training-objective composition, not a separate generation mode) |
| `enable_trainable_rollout_loss` | Reserved future switch; currently must remain `false` because rollout recovery is logged only as detached `rollout_eval_*` diagnostics |
| `affinity_weighting` | `none` or `inverse_frequency` |
| `flow_matching_loss_weight` | Scalar weight for flow-only objective |
| `hybrid_denoising_weight`, `hybrid_flow_weight` | Hybrid objective weights |

### `ModelConfig` additions

| Field | Description |
| --- | --- |
| `interaction_mode` | `lightweight` or `transformer` |
| `interaction_ff_multiplier` | Feed-forward width multiplier for Transformer mode |

### Built-in method ids

`conditioned_denoising`, `flow_matching`, `heuristic_raw_rollout_no_repair`, `pocket_centroid_repair_proxy`, `deterministic_proxy_reranker`, `calibrated_reranker`, `autoregressive_graph_geometry`, `energy_guided_refinement`, and stubs for future backends.

## Binaries

| Binary | Purpose |
| --- | --- |
| `pocket_diffusion` | Main binary: inspection, training, experiments, generation |
| `disentangle_demo` | Older standalone demo |

Use `cargo run --bin pocket_diffusion -- ...` for research commands.

Legacy demo behavior available via `cargo run --bin pocket_diffusion -- legacy-demo`.

## Research stack structure

```
src/
  models/        — Encoders, slot decomposition, cross-modal interaction, generation methods
  data/          — Parsing, collation, batching, splits
  losses/        — Primary objectives + auxiliary losses (redundancy, leakage, gate, slot, MI)
  training/      — Staged trainer, scheduler, checkpointing, resume
  experiments/   — Unseen-pocket experiments, external backends, ablation matrix
  config/        — Config types, validation, serialization
```

Library imports:
```rust
use pocket_diffusion::{config, data, models, training, experiments};
```

Legacy namespace (deprecated):
```rust
use pocket_diffusion::legacy;
```

## Capability Boundaries

**Present**: modular encoders, slot decomposition, gated cross-modal interaction, staged training with auxiliary losses, MI monitoring, conditioned denoising/refinement, de novo pocket-conditioned full molecular flow branches, unseen-pocket experiments, RDKit chemistry metrics, AutoDock Vina score-only rescoring, GNINA score-only rescoring, public-baseline adaptation, and layer-separated raw/constrained/repaired/reranked audit tables.

**Not present**: wet-lab validation, MD validation, production-grade docking protocol validation, public-baseline multi-seed closure, or evidence that repaired/reranked postprocessing improves native generation quality. Current Vina/GNINA rows are score-only backend evidence, not experimental binding-affinity measurements.

The word `diffusion` remains in the crate name for compatibility. The config-driven path is a modular representation-learning framework with conditioned denoising/refinement, full molecular flow when explicitly enabled, deterministic rollout diagnostics, and evidence attribution. `geometry_only=true` preserves the older coordinate-flow baseline; `geometry_only=false` plus all five flow branches enables de novo atom-count initialization, atom-type flow, bond flow, topology synchronization, pocket/context flow, and coordinate flow.

## Full config list

| Config | Purpose |
| --- | --- |
| `configs/research_manifest.json` | Manifest-driven dataset |
| `configs/research_pdbbind_dir.json` | PDBbind-like directory scan |
| `configs/research_pdbbind_index.json` | Directory scan + INDEX-style labels |
| `configs/unseen_pocket_manifest.json` | Full unseen-pocket experiment |
| `configs/unseen_pocket_real_backends.json` | External chemistry + pocket backends |
| `configs/unseen_pocket_claim_matrix.json` | Compact ablation matrix |
| `configs/unseen_pocket_harder_pressure.json` | Harder-pressure external-backend surface |
| `configs/unseen_pocket_tight_geometry_pressure.json` | Tighter geometry constraints |
| `configs/unseen_pocket_lp_pdbbind_refined_real_backends.json` | LP-PDBBind refined split |
| `configs/flow_matching_experiment.json` | Flow-matching as primary objective |
| `configs/unseen_pocket_pdbbindpp_flow_canonical.json` | PDBbind++ flow-matching canonical |
| `configs/unseen_pocket_lp_pdbbind_refined_flow_canonical.json` | LP-PDBBind flow-matching |
| `configs/unseen_pocket_multi_seed_pdbbindpp_flow_canonical.json` | Multi-seed flow-matching |
| `configs/flow_matching_claim_contract.json` | Flow claim/threshold contract |
| `configs/flow_canonical_config_family.json` | Config hash/fingerprint references |
| `configs/unseen_pocket_flow_sweep.json` | Flow stabilization sweep (F2.1) |
| `configs/f21_sweep_result_table.json` | Sweep results with optimal weights |
| `configs/unseen_pocket_pdbbindpp_flow_best_candidate.json` | Best-flow candidate (PDBbind++) |
| `configs/unseen_pocket_lp_pdbbind_refined_flow_best_candidate.json` | Best-flow candidate (LP-PDBBind) |
| `configs/unseen_pocket_pdbbindpp_flow_best_candidate_paper.json` | Paper-quality flow-matching |
| `configs/unseen_pocket_flow_ablation_matrix.json` | Flow ablation bundle (F3.1) |
| `configs/q15_generation_alignment_ablation_matrix.json` | Q15 generation-alignment ablation matrix with claim-safe variant labels |
| `configs/f31_ablation_bundle.json` | Ablation results |
| `configs/f32_diagnostics_package.json` | Extended flow diagnostics (F3.2) |
| `configs/unseen_pocket_multi_seed_pdbbindpp_flow.json` | Multi-seed flow (F4.1) |
| `configs/f41_multi_seed_summary.json` | Aggregated stability metrics |
| `configs/unseen_pocket_pdbbindpp_denoising_matched.json` | Matched denoising baseline |
| `configs/unseen_pocket_pdbbindpp_reranker_matched.json` | Matched reranker baseline |
| `configs/f42_method_comparison.json` | Method comparison (F4.2) |
| `configs/f51_reproducibility_manifest.json` | Reproducibility manifest (F5.1) |
| `configs/f52_paper_narrative.json` | Paper narrative (F5.2) |
| `configs/unseen_pocket_vina_backend.json` | Vina backend companion |
| `configs/q1_baseline_registry.json` | Public baseline registry and claim guardrails |
| `configs/q1_method_comparison_summary.json` | Current public-baseline layer-separated performance summary |
| `configs/q1_public_baseline_run_status.json` | Current public-baseline run status, coverage, and runtime |
| `configs/q1_public_baseline_full100_layered_metric_coverage.json` | Coverage report for the 900-row layered public-baseline run |
