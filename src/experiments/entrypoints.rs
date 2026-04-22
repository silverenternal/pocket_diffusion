//! Config-driven entrypoints for modular research experiments.

use std::path::Path;

use crate::experiments::{
    load_experiment_config, UnseenPocketExperiment, UnseenPocketExperimentSummary,
};

/// Execute the unseen-pocket experiment from a JSON config path.
pub fn run_experiment_from_config(
    path: impl AsRef<Path>,
    resume: bool,
) -> Result<UnseenPocketExperimentSummary, Box<dyn std::error::Error>> {
    let config = load_experiment_config(path)?;
    UnseenPocketExperiment::run_with_options(config, resume)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::experiments::UnseenPocketExperimentConfig;

    #[test]
    fn config_driven_experiment_writes_summary() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = UnseenPocketExperimentConfig::default();
        config.research.runtime.device = "cpu".to_string();
        config.research.data.batch_size = 2;
        config.research.training.max_steps = 2;
        config.research.training.log_every = 100;
        config.research.training.checkpoint_every = 100;
        config.research.training.checkpoint_dir = temp.path().join("checkpoints");

        let config_path = temp.path().join("experiment_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let summary = run_experiment_from_config(&config_path, false).unwrap();

        assert_eq!(summary.training_history.len(), 2);
        assert!(summary.validation.validity >= 0.0);
        assert!(summary.test.validity >= 0.0);
        assert!(summary
            .config
            .research
            .training
            .checkpoint_dir
            .join("experiment_summary.json")
            .exists());
    }
}
