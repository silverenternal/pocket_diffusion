# Scalable Dataset Loading Boundary

The current production path intentionally remains in memory:

- `InMemoryDataset::load_from_config` parses configured sources and returns a
  materialized `Vec<MolecularExample>`.
- train/validation/test splits clone examples into `DatasetSplits`.
- trainer batching uses `&[MolecularExample]` through `ExampleBatchSampler`.
- evaluation entrypoints pass borrowed slices from `InMemoryDataset::examples`.

This is still the default smoke, reviewer, and small-experiment path.

The scalable extension point is `MolecularExampleSource` in
`src/data/dataset/core.rs`. It exposes stable indexed access that returns owned
`MolecularExample` values, so a future loader can parse records lazily, read
from an indexed shard, or cache decoded examples without forcing every caller
to borrow from a single `Vec`.

Migration plan for a lazy implementation:

1. Keep manifest discovery and validation in Rust.
2. Build an index of stable example keys and source paths.
3. Implement `MolecularExampleSource` for the indexed store.
4. Use `collect_examples_from_source` only at legacy slice-oriented boundaries.
5. Add sampler/evaluator variants that operate on index batches before removing
   materialization from large-run training.

No Python data loader is part of the core training path. Python tooling may
continue to validate artifacts, but model training and dataset parsing should
stay in Rust.
