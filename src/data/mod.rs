//! Data abstractions for pocket-conditioned molecular generation.

pub mod batch;
pub mod dataset;
pub mod features;
pub mod parser;

pub use batch::{DecoderBatchTargets, EncoderBatchInputs, ExampleBatchIter, MolecularBatch};
pub use dataset::{Dataset, DatasetSplits, InMemoryDataset, LoadedDataset};
pub use features::{
    DecoderSupervision, ExampleTargets, GeometryFeatures, MolecularExample, PocketFeatures,
    TopologyFeatures,
};
pub use parser::{
    apply_affinity_labels, discover_pdbbind_like_entries, load_affinity_labels, load_manifest,
    load_manifest_entry, synthetic_phase1_examples, AffinityLabel, DataParseError, DatasetManifest,
    DatasetValidationReport, ManifestEntry,
};
