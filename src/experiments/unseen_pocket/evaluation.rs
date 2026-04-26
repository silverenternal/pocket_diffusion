// Split evaluation implementation kept in the parent module via textual
// includes to preserve existing private helper visibility during the refactor.
include!("evaluation/core.rs");
include!("evaluation/generation_layers.rs");
include!("evaluation/reranking_artifacts.rs");
include!("evaluation/summary.rs");
