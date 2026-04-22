# pocket_diffusion

Rust-first modular research framework for pocket-conditioned representation learning, with a legacy generation demo kept for compatibility.

The actively extended path provides:

- separate topology / geometry / pocket encoders
- slot-based structured decomposition
- gated cross-modal interaction
- staged auxiliary losses for redundancy, leakage, gate, and slot control
- config-driven data loading from real file formats

## Capability matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Modular topology / geometry / pocket encoders | `implemented` | Separate encoder path used by config-driven training and experiments |
| Slot decomposition and gated cross-modal interaction | `implemented` | Used by the modular research stack |
| Staged training with auxiliary regularizers | `implemented` | Primary objective is still a surrogate reconstruction objective |
| Unseen-pocket split experiments | `implemented` | Reported metrics are diagnostics and proxy metrics, not chemistry-grade generation metrics |
| Affinity supervision from lightweight parsing | `prototype` | Uses simplified normalization and lightweight parsing/reporting |
| Distance / affinity probe evaluation | `proxy` | Computed from probe heads on held-out examples |
| Direct candidate generation and ranking | `legacy` | Available through `legacy-demo` and `pocket_diffusion::legacy` |
| Diffusion training objective | `planned` | Crate name is historical; the modular stack does not train a diffusion objective today |
| Chemistry validity / docking / pocket-compatibility backend | `planned` | No RDKit/docking backend is integrated in the modular path yet |

## Current focus

The repository now supports a practical "real-data first" path:

- JSON config loading
- lightweight PDB + SDF parsing
- manifest-driven datasets
- PDBbind-like directory discovery
- external CSV/TSV affinity label tables
- protein-level train/val/test splits for unseen-pocket experiments

The included sample dataset is intentionally small, but it uses actual on-disk `*.pdb` and `*.sdf` files so the parsing, collation, training, and evaluation paths are exercised end to end.

## Repository status

This repository currently contains two parallel surfaces:

- a legacy `PocketDiffusionPipeline` demo path for direct candidate generation and ranking
- a newer modular research stack for multi-modal representation learning, staged surrogate training, and unseen-pocket diagnostics

The modular research path is the actively extended surface and is the one documented below.

For library consumers, the same boundary now exists in code:

- modular research APIs live under `pocket_diffusion::{config,data,models,training,experiments}`
- legacy demo/comparison APIs are grouped under `pocket_diffusion::legacy`

New integrations should prefer module-qualified imports. The crate root is no longer treated as a flat umbrella namespace for the modular stack.

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

`lightweight` mode keeps the convenience-oriented behavior used by the sample configs: first discovered `.pdb`/`.sdf` files are accepted and empty-cutoff pocket extraction falls back to nearest atoms. `strict` mode rejects ambiguous discovery and rejects nearest-atom pocket fallback.

### `TrainingConfig` fields

- `affinity_weighting`: `none` or `inverse_frequency` for labeled affinity supervision across mixed measurement families

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

These are diagnostics and proxy metrics over held-out examples. They are not chemistry validity, docking quality, or generation quality metrics.

`real_generation_metrics` is a reserved schema section for future backend adapters. The current implementation keeps placeholder entries for:

- chemistry validity
- docking / affinity rescoring
- downstream pocket compatibility

The extension points live in trait form under `pocket_diffusion::models` so an RDKit-like validator or docking backend can be attached later without renaming the persisted artifact schema again.

## Reproducibility artifacts

Config-driven `train` and `experiment` runs now write a shared artifact layout inside `training.checkpoint_dir`:

- `config.snapshot.json`
- `dataset_validation_report.json`
- `split_report.json`
- `training_summary.json` or `experiment_summary.json`
- `run_artifacts.json`
- checkpoint weights such as `latest.ot`

`run_artifacts.json` is the stable bundle entrypoint used by the `report` command.

Training and experiment summaries now also persist:

- `config_hash`
- `dataset_validation_fingerprint`
- `metric_schema_version`
- explicit `resume_contract`
- `resume_provenance`

Current resume semantics are intentionally explicit: model weights, step index, and prior persisted history are restored, but optimizer state is not. A resumed run is therefore a convenience continuation with documented limits, not a strict deterministic replay guarantee.

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

- `dataset_validation_report.json` with discovered complexes, parsed examples, label attachment counts, fallback pocket extraction counts, and truncation metadata
- step-indexed checkpoints such as `step-10.ot` and `step-10.json`
- rolling `latest.ot` and `latest.json` pointers for resume
- `training_summary.json` for config-driven training runs, including cumulative `training_history`, dataset split sizes, split audit, resume step, and post-train validation/test metrics
- `experiment_summary.json` for unseen-pocket experiment runs, also preserving cumulative `training_history` across resume together with the split audit

The split audit records per-split unique protein counts, labeled fraction, measurement-family histograms, and explicit leakage checks for protein overlap and duplicated example IDs across train/val/test.

## Capability boundaries

The modular stack currently optimizes a configurable primary objective, with the default implementation set to `surrogate_reconstruction`. Auxiliary losses for consistency, redundancy reduction, probes, leakage control, gate sparsity, and slot control are activated in stages around that primary objective.

What is not present in the modular path today:

- no active diffusion loss
- no generation-time decoder loop
- no chemistry-validity backend
- no docking-based pocket compatibility scoring

The word `diffusion` remains in the crate/package name for compatibility. In technical terms, the config-driven path is a modular representation-learning framework, while the direct generation surface is legacy/demo code.

## Notes

- The current loader supports minimal V2000 SDF parsing and ligand-centered PDB pocket extraction.
- If no atoms fall within the pocket cutoff, the loader falls back to the nearest atoms instead of producing an empty pocket.
- The formal metrics are still constrained by the current model outputs. Affinity labels, cheminformatics validity checks, and docking-quality scores would need additional domain tooling beyond the current crate.
- The bundled dataset is intentionally tiny, so split behavior and per-measurement metrics are useful for plumbing validation, not for claiming model quality.
