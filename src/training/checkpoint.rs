//! Minimal checkpoint save/load helpers.

use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tch::nn;

use super::StepMetrics;

/// Small checkpoint metadata file saved alongside model weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Step number associated with the weights file.
    pub step: usize,
    /// Optional metrics snapshot.
    pub metrics: Option<StepMetrics>,
}

/// Writes VarStore weights and metadata into a directory.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    dir: PathBuf,
}

impl CheckpointManager {
    /// Create a checkpoint manager rooted at `dir`.
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Save weights and metadata for a step.
    pub fn save(
        &self,
        var_store: &nn::VarStore,
        step: usize,
        metrics: Option<&StepMetrics>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(&self.dir)?;
        let weights_path = self.dir.join(format!("step-{step}.ot"));
        let metadata_path = self.dir.join(format!("step-{step}.json"));
        var_store.save(weights_path)?;
        let metadata = CheckpointMetadata {
            step,
            metrics: metrics.cloned(),
        };
        fs::write(metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        Ok(())
    }
}
