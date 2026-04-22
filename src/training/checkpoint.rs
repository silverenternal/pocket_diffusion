//! Checkpoint save/load helpers for the modular research trainer.

use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};
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

/// Result of restoring a checkpoint into a var store.
#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    /// Path to the restored weight file.
    pub weights_path: PathBuf,
    /// Path to the restored metadata file.
    pub metadata_path: PathBuf,
    /// Saved metadata.
    pub metadata: CheckpointMetadata,
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
        let latest_weights_path = self.dir.join("latest.ot");
        let latest_metadata_path = self.dir.join("latest.json");
        var_store.save(weights_path)?;
        let metadata = CheckpointMetadata {
            step,
            metrics: metrics.cloned(),
        };
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, &metadata_json)?;
        fs::copy(
            self.dir.join(format!("step-{step}.ot")),
            latest_weights_path,
        )?;
        fs::write(latest_metadata_path, metadata_json)?;
        Ok(())
    }

    /// Return the configured checkpoint directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Discover the latest checkpoint based on the highest saved step.
    pub fn latest_checkpoint(
        &self,
    ) -> Result<Option<LoadedCheckpoint>, Box<dyn std::error::Error>> {
        if !self.dir.exists() {
            return Ok(None);
        }

        let latest_weights_path = self.dir.join("latest.ot");
        let latest_metadata_path = self.dir.join("latest.json");
        if latest_weights_path.exists() && latest_metadata_path.exists() {
            let metadata: CheckpointMetadata =
                serde_json::from_str(&fs::read_to_string(&latest_metadata_path)?)?;
            return Ok(Some(LoadedCheckpoint {
                weights_path: latest_weights_path,
                metadata_path: latest_metadata_path,
                metadata,
            }));
        }

        let mut latest: Option<LoadedCheckpoint> = None;
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.file_name().and_then(|name| name.to_str()) == Some("latest.json") {
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let metadata: CheckpointMetadata = serde_json::from_str(&fs::read_to_string(&path)?)?;
            let weights_path = self.dir.join(format!("step-{}.ot", metadata.step));
            if !weights_path.exists() {
                continue;
            }
            let candidate = LoadedCheckpoint {
                weights_path,
                metadata_path: path,
                metadata,
            };
            let replace = latest
                .as_ref()
                .map(|current| candidate.metadata.step > current.metadata.step)
                .unwrap_or(true);
            if replace {
                latest = Some(candidate);
            }
        }

        Ok(latest)
    }

    /// Load the latest checkpoint into the provided var store.
    pub fn load_latest(
        &self,
        var_store: &mut nn::VarStore,
    ) -> Result<Option<LoadedCheckpoint>, Box<dyn std::error::Error>> {
        let Some(checkpoint) = self.latest_checkpoint()? else {
            return Ok(None);
        };
        var_store.load(&checkpoint.weights_path)?;
        Ok(Some(checkpoint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, no_grad, Device, Kind, Tensor};

    #[test]
    fn save_and_load_latest_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        let manager = CheckpointManager::new(temp.path().join("checkpoints"));

        let vs = nn::VarStore::new(Device::Cpu);
        let root = &vs.root();
        let mut probe = root.var("probe", &[1], nn::Init::Const(0.0));
        no_grad(|| probe.copy_(&Tensor::from_slice(&[2.5f32]).to_kind(Kind::Float)));
        manager.save(&vs, 3, None).unwrap();

        let mut restored = nn::VarStore::new(Device::Cpu);
        let mut restored_probe = restored.root().var("probe", &[1], nn::Init::Const(0.0));
        no_grad(|| restored_probe.copy_(&Tensor::from_slice(&[0.0f32]).to_kind(Kind::Float)));

        let loaded = manager.load_latest(&mut restored).unwrap().unwrap();
        let restored_tensor = restored.root().get("probe").unwrap();

        assert_eq!(loaded.metadata.step, 3);
        assert_eq!(restored_tensor.double_value(&[0]), 2.5);
        assert_eq!(
            loaded
                .metadata_path
                .file_name()
                .and_then(|name| name.to_str()),
            Some("latest.json")
        );
        assert!(manager.dir().join("latest.ot").exists());
        assert!(manager.dir().join("latest.json").exists());
    }
}
