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
    /// Stable hash of the effective config JSON.
    pub config_hash: Option<String>,
    /// Stable fingerprint of the dataset validation report.
    pub dataset_validation_fingerprint: Option<String>,
    /// Metric schema version associated with summary artifacts.
    pub metric_schema_version: u32,
    /// Human-readable resume contract identifier.
    pub resume_contract_version: String,
    /// Optional optimizer-state metadata snapshot.
    #[serde(default)]
    pub optimizer_state: Option<OptimizerStateMetadata>,
    /// Optional scheduler-state snapshot.
    #[serde(default)]
    pub scheduler_state: Option<SchedulerStateMetadata>,
    /// Backend/objective compatibility metadata for model-switch training.
    #[serde(default)]
    pub backend_training: Option<BackendTrainingMetadata>,
}

/// Serializable backend/objective compatibility metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackendTrainingMetadata {
    /// Active backend id used when the checkpoint was saved.
    pub backend_id: String,
    /// Active backend family used when the checkpoint was saved.
    pub backend_family: String,
    /// Primary objective implementation active when the checkpoint was saved.
    pub objective_name: String,
    /// Whether the active backend declares trainable parameters.
    pub trainable_backend: bool,
    /// Shared staged auxiliary objectives kept backend-independent.
    pub shared_auxiliary_objectives: Vec<String>,
}

/// Serializable optimizer resume metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizerStateMetadata {
    /// Optimizer family used for the checkpoint.
    pub optimizer_kind: String,
    /// Learning rate active when the checkpoint was saved.
    pub learning_rate: f64,
    /// Weight decay active when the checkpoint was saved.
    pub weight_decay: f64,
    /// Whether the underlying optimizer moments were persisted.
    ///
    /// The current `tch` integration does not serialize Adam moment buffers,
    /// so claim-bearing runs should leave this `false` and treat resume as a
    /// bounded continuation rather than a strict replay.
    pub internal_state_persisted: bool,
}

/// Serializable scheduler snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchedulerStateMetadata {
    /// Training stage associated with the saved step.
    pub stage: String,
    /// Primary-objective weight at save time.
    pub primary_weight: f64,
    /// Intra-modality redundancy weight at save time.
    pub intra_red_weight: f64,
    /// Probe weight at save time.
    pub probe_weight: f64,
    /// Leakage weight at save time.
    pub leak_weight: f64,
    /// Gate weight at save time.
    pub gate_weight: f64,
    /// Slot weight at save time.
    pub slot_weight: f64,
    /// Consistency weight at save time.
    pub consistency_weight: f64,
    /// Pocket-contact auxiliary weight at save time.
    pub pocket_contact_weight: f64,
    /// Pocket-clash auxiliary weight at save time.
    pub pocket_clash_weight: f64,
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
        config_hash: Option<String>,
        dataset_validation_fingerprint: Option<String>,
        metric_schema_version: u32,
        resume_contract_version: &str,
        optimizer_state: Option<OptimizerStateMetadata>,
        scheduler_state: Option<SchedulerStateMetadata>,
        backend_training: Option<BackendTrainingMetadata>,
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
            config_hash,
            dataset_validation_fingerprint,
            metric_schema_version,
            resume_contract_version: resume_contract_version.to_string(),
            optimizer_state,
            scheduler_state,
            backend_training,
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
        manager
            .save(
                &vs,
                3,
                None,
                Some("cfg-hash".to_string()),
                Some("dataset-hash".to_string()),
                2,
                "weights+history+step",
                Some(OptimizerStateMetadata {
                    optimizer_kind: "adam".to_string(),
                    learning_rate: 1e-3,
                    weight_decay: 0.0,
                    internal_state_persisted: false,
                }),
                Some(SchedulerStateMetadata {
                    stage: "stage1".to_string(),
                    primary_weight: 1.0,
                    intra_red_weight: 0.0,
                    probe_weight: 0.0,
                    leak_weight: 0.0,
                    gate_weight: 0.0,
                    slot_weight: 0.0,
                    consistency_weight: 1.0,
                    pocket_contact_weight: 0.0,
                    pocket_clash_weight: 0.0,
                }),
                Some(BackendTrainingMetadata {
                    backend_id: "conditioned_denoising".to_string(),
                    backend_family: "conditioneddenoising".to_string(),
                    objective_name: "conditioned_denoising".to_string(),
                    trainable_backend: true,
                    shared_auxiliary_objectives: vec!["L_consistency".to_string()],
                }),
            )
            .unwrap();

        let mut restored = nn::VarStore::new(Device::Cpu);
        let mut restored_probe = restored.root().var("probe", &[1], nn::Init::Const(0.0));
        no_grad(|| restored_probe.copy_(&Tensor::from_slice(&[0.0f32]).to_kind(Kind::Float)));

        let loaded = manager.load_latest(&mut restored).unwrap().unwrap();
        let restored_tensor = restored.root().get("probe").unwrap();

        assert_eq!(loaded.metadata.step, 3);
        assert_eq!(loaded.metadata.config_hash.as_deref(), Some("cfg-hash"));
        assert_eq!(
            loaded.metadata.dataset_validation_fingerprint.as_deref(),
            Some("dataset-hash")
        );
        assert_eq!(loaded.metadata.metric_schema_version, 2);
        assert_eq!(
            loaded.metadata.resume_contract_version,
            "weights+history+step"
        );
        assert_eq!(
            loaded.metadata.optimizer_state,
            Some(OptimizerStateMetadata {
                optimizer_kind: "adam".to_string(),
                learning_rate: 1e-3,
                weight_decay: 0.0,
                internal_state_persisted: false,
            })
        );
        assert_eq!(
            loaded.metadata.scheduler_state,
            Some(SchedulerStateMetadata {
                stage: "stage1".to_string(),
                primary_weight: 1.0,
                intra_red_weight: 0.0,
                probe_weight: 0.0,
                leak_weight: 0.0,
                gate_weight: 0.0,
                slot_weight: 0.0,
                consistency_weight: 1.0,
                pocket_contact_weight: 0.0,
                pocket_clash_weight: 0.0,
            })
        );
        assert_eq!(
            loaded
                .metadata
                .backend_training
                .as_ref()
                .map(|metadata| metadata.backend_id.as_str()),
            Some("conditioned_denoising")
        );
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

    #[test]
    fn older_checkpoint_metadata_without_resume_state_still_loads() {
        let temp = tempfile::tempdir().unwrap();
        let manager = CheckpointManager::new(temp.path().join("checkpoints"));
        fs::create_dir_all(manager.dir()).unwrap();
        fs::write(manager.dir().join("latest.ot"), b"placeholder").unwrap();
        fs::write(
            manager.dir().join("latest.json"),
            r#"{
  "step": 7,
  "metrics": null,
  "config_hash": "cfg-hash",
  "dataset_validation_fingerprint": "dataset-hash",
  "metric_schema_version": 2,
  "resume_contract_version": "weights+history+step"
}"#,
        )
        .unwrap();

        let metadata: CheckpointMetadata =
            serde_json::from_str(&fs::read_to_string(manager.dir().join("latest.json")).unwrap())
                .unwrap();

        assert_eq!(metadata.step, 7);
        assert!(metadata.optimizer_state.is_none());
        assert!(metadata.scheduler_state.is_none());
        assert!(metadata.backend_training.is_none());
    }
}
