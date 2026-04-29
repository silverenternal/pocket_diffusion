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

## Evidence Boundaries

Claim-facing artifacts include candidate-level RDKit, AutoDock Vina `score_only`, and GNINA `score_only` outputs. These are backend scores, not experimental binding affinities. Proxy metrics such as `docking_like_score` are heuristic diagnostics only.

Layer attribution is strict:

- `raw_rollout` and `raw_flow` are native model evidence.
- `constrained_flow` is constrained-sampling evidence derived from raw flow output.
- `inferred_bond_candidates`, `repaired`, `repaired_candidates`, `deterministic_proxy_candidates`, and `reranked_candidates` are postprocessing or selection evidence.

Training/evaluation alignment is explicit in experiment artifacts:

- `L_probe` and `L_leak` are optimizer-facing auxiliary losses when enabled by the staged trainer.
- `rollout_eval_*` fields are detached diagnostics unless a future tensor-preserving trainable rollout objective is implemented.
- `finite_forward_fraction` is a smoke/default health metric; claim-bearing configs should select quality-aware best metrics with availability checks.
- Backend score rows report coverage, missing-structure fraction, fallback use, and candidate counts before they can support stronger wording.

Current public-baseline details live in:

- [`configs/q1_method_comparison_summary.json`](configs/q1_method_comparison_summary.json)
- [`docs/q1_method_comparison_table.md`](docs/q1_method_comparison_table.md)
- [`docs/q1_runtime_efficiency_table.md`](docs/q1_runtime_efficiency_table.md)
- [`docs/postprocessing_failure_audit.md`](docs/postprocessing_failure_audit.md)
- [`docs/q2_claim_contract.md`](docs/q2_claim_contract.md)
- [`docs/q8_reviewer_scale_runbook.md`](docs/q8_reviewer_scale_runbook.md)

The fast validation gate is:

```bash
python tools/validation_suite.py --mode quick --timeout 240
# or
tools/local_ci.sh fast
```

## Quick start

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
| `flow_matching_experiment.json` | Flow-matching as primary objective |
| `unseen_pocket_pdbbindpp_flow_best_candidate_paper.json` | Paper-quality flow-matching run |
| `q1_method_comparison_summary.json` | Current layer-separated public-baseline performance summary |
| `q1_public_baseline_run_status.json` | Current public-baseline status, coverage, and runtime provenance |
| `q1_baseline_registry.json` | Public baseline source/status registry |

See [Config files](#config-files-1) below for full list and field documentation.

## Dataset

### Layout

**Manifest mode** — JSON with entries:
```json
{ "entries": [{ "example_id": "x", "protein_id": "p", "pocket_path": "...", "ligand_path": "..." }] }
```

**PDBbind-like directory mode**:
```
dataset_root/
  complex_001/*.pdb, *.sdf
  complex_002/*.pdb, *.sdf
```

**External labels** — CSV/TSV with `example_id`, `protein_id`, `measurement_type`, `raw_value`, `raw_unit`. Supports `Kd`, `Ki`, `IC50`, `pKd` with automatic normalization.

### Data contracts

| Contract | Description |
| --- | --- |
| `smoke-only` | Parsing succeeds, no quality gates |
| `parser-only` | Debug source files and normalization warnings |
| `fallback-heavy` | Retained examples depend on fallback/extrapolation |
| `claim-bearing` | Reproducible quality gates on label coverage, provenance, metadata |

Ligand-centered coordinate and pocket extraction are tracked in `dataset_validation_report.json`. Target-ligand refinement configs may allow ligand-centered context but must keep that dependency visible in the report. De novo model execution recenters pocket features for the generated scaffold and uses target ligand fields only as training supervision; claim-bearing dataset configs should still enable `quality_filters.reject_target_ligand_context_leakage=true` to keep retained data contracts strict. Source-coordinate reconstruction support and generation coordinate-frame contracts are also persisted so candidate artifacts cannot silently mix coordinate frames.

### Included sample

Tiny dataset under [`examples/datasets/mini_pdbbind`](examples/datasets/mini_pdbbind) for smoke tests and format validation.

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
- See [Reproducibility artifacts](#reproducibility-artifacts-1) for details

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
