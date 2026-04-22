# pocket_diffusion

Rust-first research framework for pocket-conditioned molecular generation with:

- separate topology / geometry / pocket encoders
- slot-based structured decomposition
- gated cross-modal interaction
- staged auxiliary losses for redundancy, leakage, gate, and slot control
- config-driven data loading from real file formats

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
- a newer modular research stack for multi-modal representation learning, slot decomposition, staged training, and unseen-pocket evaluation

The modular research path is the actively extended surface and is the one documented below.

## Quick start

Inspect the sample dataset:

```bash
cargo run --bin pocket_diffusion -- --inspect-config configs/research_manifest.json
```

Run a short staged training job from config:

```bash
cargo run --bin pocket_diffusion -- --train-config configs/research_manifest.json
```

Run the unseen-pocket experiment from config:

```bash
cargo run --bin pocket_diffusion -- --experiment-config configs/unseen_pocket_manifest.json
```

Inspect a PDBbind-like directory using an `INDEX`-style label file:

```bash
cargo run --bin pocket_diffusion -- --inspect-config configs/research_pdbbind_index.json
```

## Config files

Example configs live under [configs](/home/hugo/codes/patent_rw/configs).

- [configs/research_manifest.json](/home/hugo/codes/patent_rw/configs/research_manifest.json) loads a dataset from an explicit manifest
- [configs/research_pdbbind_dir.json](/home/hugo/codes/patent_rw/configs/research_pdbbind_dir.json) scans a PDBbind-like directory tree
- [configs/research_pdbbind_index.json](/home/hugo/codes/patent_rw/configs/research_pdbbind_index.json) scans a PDBbind-like directory tree and attaches labels from an `INDEX`-style file
- [configs/unseen_pocket_manifest.json](/home/hugo/codes/patent_rw/configs/unseen_pocket_manifest.json) runs the full unseen-pocket experiment

### `DataConfig` fields

- `dataset_format`: one of `synthetic`, `manifest_json`, `pdbbind_like_dir`
- `root_dir`: root directory for dataset discovery
- `manifest_path`: required when `dataset_format` is `manifest_json`
- `label_table_path`: optional CSV/TSV or PDBbind-like index text with `example_id` and/or `protein_id` plus either `affinity_kcal_mol`, `affinity_record`, or `measurement_type/raw_value[/raw_unit]`
- `pocket_cutoff_angstrom`: ligand-centered pocket extraction radius
- `max_examples`: optional truncation for quick debugging
- `val_fraction`, `test_fraction`, `split_seed`: protein-level split controls
- `stratify_by_measurement`: preserve measurement-family balance when assigning protein groups to train/val/test

### `TrainingConfig` fields

- `affinity_weighting`: `none` or `inverse_frequency` for labeled affinity supervision across mixed measurement families

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

The unseen-pocket experiment reports:

- `validity`
- `uniqueness`
- `novelty`
- `distance_rmse`
- `affinity_alignment`
- `affinity_mae`
- `affinity_rmse`
- `labeled_fraction`
- per-measurement affinity `MAE/RMSE` breakdown
- `reconstruction_mse`
- `slot_usage_mean`
- `gate_usage_mean`
- `leakage_mean`
- memory and timing statistics

`distance_rmse`, `affinity_mae`, and `affinity_rmse` are more grounded than the earlier proxy naming, but the stack still uses lightweight probe heads rather than a full docking or chemistry toolkit.

The sample configs now enable measurement-stratified protein splits and inverse-frequency affinity weighting so mixed `Kd/Ki/IC50/pKd` labels do not collapse into a single undifferentiated supervision pool.

## Included sample dataset

The repository ships a tiny dataset under [examples/datasets/mini_pdbbind](/home/hugo/codes/patent_rw/examples/datasets/mini_pdbbind). It is suitable for smoke tests and for validating the file-format ingestion path.

## Binaries

- `pocket_diffusion`: main binary for config-driven inspection, training, experiments, and the legacy generation demo
- `disentangle_demo`: older standalone demo binary kept for local experimentation

Because the crate defines multiple binaries, use `cargo run --bin pocket_diffusion -- ...` for the commands in this README.

## Notes

- The current loader supports minimal V2000 SDF parsing and ligand-centered PDB pocket extraction.
- If no atoms fall within the pocket cutoff, the loader falls back to the nearest atoms instead of producing an empty pocket.
- The formal metrics are still constrained by the current model outputs. Affinity labels, cheminformatics validity checks, and docking-quality scores would need additional domain tooling beyond the current crate.
- The bundled dataset is intentionally tiny, so split behavior and per-measurement metrics are useful for plumbing validation, not for claiming model quality.
