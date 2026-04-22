//! Legacy comparison experiment helpers.
//!
//! These wrappers keep the older ndarray-based comparison workflow available
//! under `crate::legacy` without making it look like part of the primary
//! config-driven research stack.

use crate::experiment::{ComparisonExperiment, ComparisonResult, ExperimentConfig};

/// Run the legacy representation comparison experiment.
pub fn run_comparison_experiment(
    config: ExperimentConfig,
) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
    ComparisonExperiment::new(config).run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_comparison_wrapper_builds_runner() {
        let config = ExperimentConfig::default();
        let runner = ComparisonExperiment::new(config.clone());
        let _ = runner;

        assert!(config.use_sample_data);
    }
}
