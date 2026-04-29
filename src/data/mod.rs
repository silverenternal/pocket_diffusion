//! Data abstractions for pocket-conditioned molecular generation.

pub mod batch;
pub mod dataset;
pub mod features;
pub mod parser;

pub use batch::{
    sample_order_seed_for_epoch, DecoderBatchTargets, EncoderBatchInputs, ExampleBatchIter,
    ExampleBatchSampler, MolecularBatch, SampledExampleBatch,
};
pub use dataset::{
    collect_examples_from_source, Dataset, DatasetSplits, InMemoryDataset, LoadedDataset,
    MolecularExampleSource,
};
pub use features::{
    ChemistryRoleFeature, ChemistryRoleFeatureMatrix, ChemistryRoleFeatureProvenance,
    DecoderSupervision, ExampleTargets, GeometryFeatures, MolecularExample, PocketFeatures,
    TopologyFeatures, CHEMISTRY_ROLE_FEATURE_DIM, CHEMISTRY_ROLE_FEATURE_DIM_USIZE,
};
pub use parser::{
    apply_affinity_labels, discover_pdbbind_like_entries, load_affinity_labels, load_manifest,
    load_manifest_entry, synthetic_phase1_examples, AffinityLabel, AffinityLabelLoadReport,
    DataParseError, DatasetManifest, DatasetValidationReport, LoadedAffinityLabels, ManifestEntry,
};
