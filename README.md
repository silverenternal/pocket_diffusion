# pocket_diffusion

Rust-first modular research framework for pocket-conditioned representation learning and method-aware molecular generation comparison, with a legacy generation demo kept for compatibility.

## Core capabilities

| Capability | Status |
| --- | --- |
| Topology / geometry / pocket encoders | ✅ Separate encoder paths |
| Slot-based structured decomposition | ✅ Fixed upper-bound, soft gating, sparse utilization |
| Gated cross-modal interaction | ✅ Lightweight and Transformer-style modes |
| Staged auxiliary losses | ✅ Redundancy, leakage, gate, slot, consistency, probe |
| MI monitoring (inter-modality decoupling) | ✅ Entropy-based estimator with pairwise tracking |
| Flow-matching generator | ✅ ODE-based velocity prediction, Euler/Heun integration |
| Unseen-pocket split experiments | ✅ Protein-level train/val/test with external backends |
| Affinity supervision | ✅ Mixed Kd/Ki/IC50 with normalization provenance |

## Latest results (2026-04-27)

**Paper-quality training** on PDBbind++ (5316 examples, unseen-pocket split) with flow-matching objective and Transformer interaction:

| Metric | Value |
| --- | --- |
| `candidate_valid_fraction` | 1.0000 |
| `strict_pocket_fit_score` | **0.8396** |
| `pocket_contact_fraction` | 1.0000 |
| `unique_smiles_fraction` | 1.0000 |
| `mean_centroid_offset` | 0.1910 |
| **Semantic specialization** | |
| topology | 0.0570 |
| geometry | **0.8242** |
| pocket | **0.7936** |
| **Utilization diagnostics** | |
| slot activation | 0.5414 |
| gate activation | 0.3477 |
| leakage proxy | 0.0836 |

Config: `configs/unseen_pocket_pdbbindpp_flow_best_candidate_paper.json`

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

### Included sample

Tiny dataset under [`examples/datasets/mini_pdbbind`](examples/datasets/mini_pdbbind) for smoke tests and format validation.

## Evaluation

Unseen-pocket experiments report:

- **Representation diagnostics**: `finite_forward_fraction`, `unique_complex_fraction`, `unseen_protein_fraction`
- **Probe metrics**: `distance_probe_rmse`, `topology_pocket_cosine_alignment`, `affinity_probe_mae/rmse`
- **Generation metrics**: chemistry validity, docking rescoring, pocket compatibility
- **Utilization**: slot/gate/leakage diagnostics, semantic specialization scores

External backends (executables reading candidate JSON, emitting metrics JSON):
- [`tools/rdkit_validity_backend.py`](tools/rdkit_validity_backend.py) — chemistry validity via RDKit
- [`tools/pocket_contact_backend.py`](tools/pocket_contact_backend.py) — pocket-aware scoring

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

### `TrainingConfig` fields

| Field | Description |
| --- | --- |
| `primary_objective` | `surrogate_reconstruction`, `conditioned_denoising`, `flow_matching`, `denoising_flow_matching` |
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

## Capability boundaries

**Present**: modular encoders, slot decomposition, gated cross-modal interaction, staged training with auxiliary losses, MI monitoring, flow-matching generator, unseen-pocket experiments with external backends.

**Not present**: active diffusion loss, iterative sampler beyond demo candidate emitter, production docking backend, external chemistry toolkit beyond heuristic adapters.

The word `diffusion` remains in the crate name for compatibility. The config-driven path is a modular representation-learning framework with conditioned denoising and deterministic rollout supervision.

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
