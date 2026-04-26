/// Build the reproducibility metadata for a config-driven run.
pub fn reproducibility_metadata(
    config: &ResearchConfig,
    dataset_validation: &DatasetValidationReport,
    checkpoint_metadata: Option<&crate::training::CheckpointMetadata>,
) -> ReproducibilityMetadata {
    ReproducibilityMetadata {
        config_hash: stable_json_hash(config),
        dataset_validation_fingerprint: stable_json_hash(dataset_validation),
        metric_schema_version: METRIC_SCHEMA_VERSION,
        artifact_bundle_schema_version: ARTIFACT_BUNDLE_SCHEMA_VERSION,
        determinism_controls: DeterminismControls {
            split_seed: config.data.split_seed,
            corruption_seed: config.data.generation_target.corruption_seed,
            sampling_seed: config.data.generation_target.sampling_seed,
            device: config.runtime.device.clone(),
            data_workers: config.runtime.data_workers,
            tch_intra_op_threads: config.runtime.tch_intra_op_threads,
            tch_inter_op_threads: config.runtime.tch_inter_op_threads,
        },
        replay_tolerance: ReplayTolerance::default(),
        resume_contract: ResumeContract {
            version: RESUME_CONTRACT_VERSION.to_string(),
            restores_model_weights: true,
            restores_step: true,
            restores_history: true,
            restores_optimizer_state: true,
            continuity_mode: ResumeContinuityMode::MetadataOnlyContinuation,
            supports_strict_replay: false,
            notes: "Resume restores model weights, step index, prior persisted training history, and any checkpointed optimizer/scheduler metadata. The underlying tch Adam moment buffers are not persisted, so this remains a bounded reproducibility aid rather than a strict deterministic replay.".to_string(),
        },
        resume_provenance: ResumeProvenance {
            resumed: checkpoint_metadata.is_some(),
            resumed_from_step: checkpoint_metadata.map(|meta| meta.step),
            checkpoint_config_hash: checkpoint_metadata.and_then(|meta| meta.config_hash.clone()),
            checkpoint_dataset_fingerprint: checkpoint_metadata
                .and_then(|meta| meta.dataset_validation_fingerprint.clone()),
            restored_optimizer_state_metadata: checkpoint_metadata
                .and_then(|meta| meta.optimizer_state.as_ref())
                .is_some(),
            restored_scheduler_state_metadata: checkpoint_metadata
                .and_then(|meta| meta.scheduler_state.as_ref())
                .is_some(),
            continuity_mode: if checkpoint_metadata.is_none() {
                ResumeContinuityMode::FreshRun
            } else {
                ResumeContinuityMode::MetadataOnlyContinuation
            },
            strict_replay_achieved: false,
        },
    }
}

/// Compute a stable short hash for a serializable structure.
pub fn stable_json_hash<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_string(value).unwrap_or_else(|_| "<serde-error>".to_string());
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
