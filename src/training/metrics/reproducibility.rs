/// Build the reproducibility metadata for a config-driven run.
pub fn reproducibility_metadata(
    config: &ResearchConfig,
    dataset_validation: &DatasetValidationReport,
    checkpoint_metadata: Option<&crate::training::CheckpointMetadata>,
) -> ReproducibilityMetadata {
    let resume_mode = checkpoint_metadata
        .map(|metadata| metadata.resume_mode)
        .unwrap_or(ResumeMode::FreshRun);
    let optimizer_internal_state_persisted = checkpoint_metadata
        .and_then(|meta| meta.optimizer_state.as_ref())
        .map(|optimizer| optimizer.internal_state_persisted)
        .unwrap_or(false);
    let strict_replay_achieved =
        resume_mode == ResumeMode::OptimizerExactResume && optimizer_internal_state_persisted;
    ReproducibilityMetadata {
        config_hash: stable_json_hash(config),
        dataset_validation_fingerprint: stable_json_hash(dataset_validation),
        metric_schema_version: METRIC_SCHEMA_VERSION,
        artifact_bundle_schema_version: ARTIFACT_BUNDLE_SCHEMA_VERSION,
        determinism_controls: determinism_controls_from_config(config),
        replay_tolerance: ReplayTolerance::default(),
        resume_contract: ResumeContract {
            version: RESUME_CONTRACT_VERSION.to_string(),
            restores_model_weights: true,
            restores_step: true,
            restores_history: true,
            restores_optimizer_state: false,
            resume_mode: ResumeMode::WeightsOnlyResume,
            continuity_mode: ResumeContinuityMode::MetadataOnlyContinuation,
            supports_strict_replay: false,
            notes: "Resume restores model weights, step index, prior persisted training history, scheduler metadata, and optimizer hyperparameters. tch 0.23 exposes no Adam moment-buffer serialization API, so the current contract is weights_only_resume rather than optimizer_exact_resume.".to_string(),
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
            resume_mode,
            continuity_mode: if checkpoint_metadata.is_none() {
                ResumeContinuityMode::FreshRun
            } else if strict_replay_achieved {
                ResumeContinuityMode::FullOptimizerContinuation
            } else {
                ResumeContinuityMode::MetadataOnlyContinuation
            },
            strict_replay_achieved,
        },
    }
}

/// Build the deterministic identity fields that materially affect training replay.
pub fn determinism_controls_from_config(config: &ResearchConfig) -> DeterminismControls {
    DeterminismControls {
        split_seed: config.data.split_seed,
        corruption_seed: config.data.generation_target.corruption_seed,
        sampling_seed: config.data.generation_target.sampling_seed,
        generation_mode: config.data.generation_target.generation_mode.as_str().to_string(),
        generation_corruption_seed: config.data.generation_target.corruption_seed,
        generation_sampling_seed: config.data.generation_target.sampling_seed,
        flow_contract_version: crate::models::current_multimodal_flow_contract(
            &config.generation_method.flow_matching,
        )
        .flow_contract_version,
        flow_branch_schedule_hash: stable_json_hash(
            &config
                .generation_method
                .flow_matching
                .multi_modal
                .branch_schedule,
        ),
        batch_size: config.data.batch_size,
        sampler_shuffle: config.training.data_order.shuffle,
        sampler_seed: config.training.data_order.sampler_seed,
        sampler_drop_last: config.training.data_order.drop_last,
        sampler_max_epochs: config.training.data_order.max_epochs,
        device: config.runtime.device.clone(),
        data_workers: config.runtime.data_workers,
        tch_intra_op_threads: config.runtime.tch_intra_op_threads,
        tch_inter_op_threads: config.runtime.tch_inter_op_threads,
    }
}

/// Compare a persisted training summary against checkpoint metadata.
pub fn training_summary_checkpoint_replay_compatibility(
    summary: &TrainingRunSummary,
    checkpoint: &crate::training::CheckpointMetadata,
) -> ReplayCompatibilityReport {
    replay_compatibility_for_metadata(&summary.reproducibility, checkpoint)
}

/// Compare reproducibility metadata against checkpoint metadata.
pub fn replay_compatibility_for_metadata(
    reproducibility: &ReproducibilityMetadata,
    checkpoint: &crate::training::CheckpointMetadata,
) -> ReplayCompatibilityReport {
    let mut mismatches = Vec::new();

    compare_optional_field(
        &mut mismatches,
        "config_hash",
        Some(reproducibility.config_hash.as_str()),
        checkpoint.config_hash.as_deref(),
        true,
        true,
    );
    compare_optional_field(
        &mut mismatches,
        "dataset_validation_fingerprint",
        Some(reproducibility.dataset_validation_fingerprint.as_str()),
        checkpoint.dataset_validation_fingerprint.as_deref(),
        true,
        true,
    );
    if reproducibility.metric_schema_version != checkpoint.metric_schema_version {
        mismatches.push(ReplayCompatibilityMismatch {
            field: "metric_schema_version".to_string(),
            expected: reproducibility.metric_schema_version.to_string(),
            observed: checkpoint.metric_schema_version.to_string(),
            replay_blocking: true,
            evidence_blocking: true,
        });
    }

    if let Some(checkpoint_controls) = checkpoint.determinism_controls.as_ref() {
        compare_determinism_controls(
            &mut mismatches,
            &reproducibility.determinism_controls,
            checkpoint_controls,
        );
    } else {
        mismatches.push(ReplayCompatibilityMismatch {
            field: "checkpoint.determinism_controls".to_string(),
            expected: "present".to_string(),
            observed: "missing".to_string(),
            replay_blocking: true,
            evidence_blocking: true,
        });
    }

    if checkpoint.resume_mode != ResumeMode::OptimizerExactResume {
        mismatches.push(ReplayCompatibilityMismatch {
            field: "checkpoint.resume_mode".to_string(),
            expected: ResumeMode::OptimizerExactResume.as_str().to_string(),
            observed: checkpoint.resume_mode.as_str().to_string(),
            replay_blocking: true,
            evidence_blocking: false,
        });
    }

    let optimizer_internal_state_persisted = checkpoint
        .optimizer_state
        .as_ref()
        .map(|state| state.internal_state_persisted)
        .unwrap_or(false);
    if !optimizer_internal_state_persisted {
        mismatches.push(ReplayCompatibilityMismatch {
            field: "checkpoint.optimizer_state.internal_state_persisted".to_string(),
            expected: "true".to_string(),
            observed: optimizer_internal_state_persisted.to_string(),
            replay_blocking: true,
            evidence_blocking: false,
        });
    }

    finish_replay_report(mismatches)
}

fn compare_optional_field(
    mismatches: &mut Vec<ReplayCompatibilityMismatch>,
    field: &str,
    expected: Option<&str>,
    observed: Option<&str>,
    replay_blocking: bool,
    evidence_blocking: bool,
) {
    if expected == observed {
        return;
    }
    mismatches.push(ReplayCompatibilityMismatch {
        field: field.to_string(),
        expected: expected.unwrap_or("<missing>").to_string(),
        observed: observed.unwrap_or("<missing>").to_string(),
        replay_blocking,
        evidence_blocking,
    });
}

fn compare_determinism_controls(
    mismatches: &mut Vec<ReplayCompatibilityMismatch>,
    expected: &DeterminismControls,
    observed: &DeterminismControls,
) {
    compare_replay_value(
        mismatches,
        "determinism_controls.split_seed",
        expected.split_seed,
        observed.split_seed,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.corruption_seed",
        expected.corruption_seed,
        observed.corruption_seed,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.sampling_seed",
        expected.sampling_seed,
        observed.sampling_seed,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.generation_mode",
        expected.generation_mode.as_str(),
        observed.generation_mode.as_str(),
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.generation_corruption_seed",
        expected.generation_corruption_seed,
        observed.generation_corruption_seed,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.generation_sampling_seed",
        expected.generation_sampling_seed,
        observed.generation_sampling_seed,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.flow_contract_version",
        expected.flow_contract_version.as_str(),
        observed.flow_contract_version.as_str(),
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.flow_branch_schedule_hash",
        expected.flow_branch_schedule_hash.as_str(),
        observed.flow_branch_schedule_hash.as_str(),
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.batch_size",
        expected.batch_size,
        observed.batch_size,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.sampler_shuffle",
        expected.sampler_shuffle,
        observed.sampler_shuffle,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.sampler_seed",
        expected.sampler_seed,
        observed.sampler_seed,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.sampler_drop_last",
        expected.sampler_drop_last,
        observed.sampler_drop_last,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.sampler_max_epochs",
        expected.sampler_max_epochs,
        observed.sampler_max_epochs,
        true,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.device",
        expected.device.as_str(),
        observed.device.as_str(),
        false,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.data_workers",
        expected.data_workers,
        observed.data_workers,
        false,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.tch_intra_op_threads",
        expected.tch_intra_op_threads,
        observed.tch_intra_op_threads,
        false,
    );
    compare_replay_value(
        mismatches,
        "determinism_controls.tch_inter_op_threads",
        expected.tch_inter_op_threads,
        observed.tch_inter_op_threads,
        false,
    );
}

fn compare_replay_value<T>(
    mismatches: &mut Vec<ReplayCompatibilityMismatch>,
    field: &str,
    expected: T,
    observed: T,
    evidence_blocking: bool,
) where
    T: PartialEq + Serialize,
{
    if expected == observed {
        return;
    }
    mismatches.push(ReplayCompatibilityMismatch {
        field: field.to_string(),
        expected: replay_value(&expected),
        observed: replay_value(&observed),
        replay_blocking: true,
        evidence_blocking,
    });
}

fn replay_value<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<serde-error>".to_string())
}

fn finish_replay_report(mismatches: Vec<ReplayCompatibilityMismatch>) -> ReplayCompatibilityReport {
    let replay_compatible = !mismatches.iter().any(|item| item.replay_blocking);
    let evidence_compatible = !mismatches.iter().any(|item| item.evidence_blocking);
    let class = if replay_compatible {
        ReplayCompatibilityClass::StrictReplayCompatible
    } else if evidence_compatible {
        ReplayCompatibilityClass::EvidenceCompatible
    } else {
        ReplayCompatibilityClass::Incompatible
    };
    let notes = match class {
        ReplayCompatibilityClass::StrictReplayCompatible => vec![
            "Artifacts match replay identity fields and checkpoint metadata records exact optimizer-state persistence.".to_string(),
        ],
        ReplayCompatibilityClass::EvidenceCompatible => vec![
            "Evidence identity fields match, but exact optimizer-state replay is not supported by this checkpoint.".to_string(),
        ],
        ReplayCompatibilityClass::Incompatible => vec![
            "At least one config, dataset, seed, or sampler identity field blocks evidence-level comparison.".to_string(),
        ],
    };
    ReplayCompatibilityReport {
        class,
        replay_compatible,
        evidence_compatible,
        mismatches,
        notes,
    }
}

/// Compute a stable short hash for a serializable structure.
pub fn stable_json_hash<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_string(value).unwrap_or_else(|_| "<serde-error>".to_string());
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{
        CheckpointMetadata, OptimizerStateMetadata, ResumeMode, RESUME_CONTRACT_VERSION,
    };

    fn checkpoint_for(
        reproducibility: &ReproducibilityMetadata,
        resume_mode: ResumeMode,
        internal_state_persisted: bool,
    ) -> CheckpointMetadata {
        CheckpointMetadata {
            step: 3,
            metrics: None,
            config_hash: Some(reproducibility.config_hash.clone()),
            dataset_validation_fingerprint: Some(
                reproducibility.dataset_validation_fingerprint.clone(),
            ),
            metric_schema_version: reproducibility.metric_schema_version,
            resume_contract_version: RESUME_CONTRACT_VERSION.to_string(),
            resume_mode,
            determinism_controls: Some(reproducibility.determinism_controls.clone()),
            optimizer_state: Some(OptimizerStateMetadata {
                optimizer_kind: "adam".to_string(),
                learning_rate: 1e-3,
                weight_decay: 0.0,
                internal_state_persisted,
                resume_mode,
                state_persistence_backend: if internal_state_persisted {
                    "native_optimizer_state".to_string()
                } else {
                    "metadata_only_tch_0_23".to_string()
                },
                exact_resume_supported: internal_state_persisted
                    && resume_mode == ResumeMode::OptimizerExactResume,
            }),
            scheduler_state: None,
            backend_training: None,
        }
    }

    #[test]
    fn reproducibility_metadata_records_sampler_controls() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 5;
        config.data.split_seed = 11;
        config.data.generation_target.corruption_seed = 12;
        config.data.generation_target.sampling_seed = 13;
        config.training.data_order.shuffle = true;
        config.training.data_order.sampler_seed = 17;
        config.training.data_order.drop_last = true;
        config.training.data_order.max_epochs = Some(3);

        let metadata =
            reproducibility_metadata(&config, &DatasetValidationReport::default(), None);

        assert_eq!(metadata.determinism_controls.split_seed, 11);
        assert_eq!(metadata.determinism_controls.corruption_seed, 12);
        assert_eq!(metadata.determinism_controls.sampling_seed, 13);
        assert_eq!(
            metadata.determinism_controls.generation_mode,
            crate::config::GenerationModeConfig::TargetLigandDenoising.as_str()
        );
        assert_eq!(metadata.determinism_controls.generation_corruption_seed, 12);
        assert_eq!(metadata.determinism_controls.generation_sampling_seed, 13);
        assert_eq!(
            metadata.determinism_controls.flow_contract_version,
            crate::models::MOLECULAR_FLOW_CONTRACT_VERSION
        );
        assert_eq!(
            metadata.determinism_controls.flow_branch_schedule_hash,
            stable_json_hash(
                &config
                    .generation_method
                    .flow_matching
                    .multi_modal
                    .branch_schedule
            )
        );
        assert_eq!(metadata.determinism_controls.batch_size, 5);
        assert!(metadata.determinism_controls.sampler_shuffle);
        assert_eq!(metadata.determinism_controls.sampler_seed, 17);
        assert!(metadata.determinism_controls.sampler_drop_last);
        assert_eq!(metadata.determinism_controls.sampler_max_epochs, Some(3));
    }

    #[test]
    fn replay_contract_distinguishes_evidence_from_strict_replay() {
        let config = ResearchConfig::default();
        let reproducibility =
            reproducibility_metadata(&config, &DatasetValidationReport::default(), None);
        let checkpoint =
            checkpoint_for(&reproducibility, ResumeMode::WeightsOnlyResume, false);

        let report = replay_compatibility_for_metadata(&reproducibility, &checkpoint);

        assert_eq!(report.class, ReplayCompatibilityClass::EvidenceCompatible);
        assert!(!report.replay_compatible);
        assert!(report.evidence_compatible);
        assert!(report
            .mismatches
            .iter()
            .any(|item| item.field == "checkpoint.resume_mode" && !item.evidence_blocking));
        assert!(report.mismatches.iter().any(|item| {
            item.field == "checkpoint.optimizer_state.internal_state_persisted"
                && !item.evidence_blocking
        }));
    }

    #[test]
    fn replay_contract_reports_concrete_sampler_seed_mismatch() {
        let config = ResearchConfig::default();
        let reproducibility =
            reproducibility_metadata(&config, &DatasetValidationReport::default(), None);
        let mut checkpoint =
            checkpoint_for(&reproducibility, ResumeMode::OptimizerExactResume, true);
        checkpoint
            .determinism_controls
            .as_mut()
            .unwrap()
            .sampler_seed += 1;

        let report = replay_compatibility_for_metadata(&reproducibility, &checkpoint);

        assert_eq!(report.class, ReplayCompatibilityClass::Incompatible);
        assert!(!report.evidence_compatible);
        assert!(report.mismatches.iter().any(|item| {
            item.field == "determinism_controls.sampler_seed" && item.evidence_blocking
        }));
    }
}
