/// Persisted summary of a config-driven modular training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunSummary {
    /// Research configuration used for the run.
    pub config: ResearchConfig,
    /// Machine-readable dataset validation artifact for the run.
    pub dataset_validation: DatasetValidationReport,
    /// Coordinate-frame provenance persisted explicitly for audit and replay.
    #[serde(default)]
    pub coordinate_frame: CoordinateFrameProvenance,
    /// Dataset sizes observed by the run.
    pub splits: DatasetSplitSizes,
    /// Split-distribution and leakage audit.
    pub split_report: SplitReport,
    /// Optional step that the run resumed from.
    pub resumed_from_step: Option<usize>,
    /// Reproducibility and schema metadata for this run.
    pub reproducibility: ReproducibilityMetadata,
    /// All collected training metrics.
    pub training_history: Vec<StepMetrics>,
    /// Per-run coverage of optimizer-facing and diagnostic objective components.
    #[serde(default)]
    pub objective_coverage: ObjectiveCoverageReport,
    /// Periodic and final validation metric snapshots used for checkpoint selection.
    #[serde(default)]
    pub validation_history: Vec<ValidationHistoryEntry>,
    /// Best checkpoint selected from validation metrics, when available.
    #[serde(default)]
    pub best_checkpoint: Option<BestCheckpointSummary>,
    /// Early-stopping state for this run.
    #[serde(default)]
    pub early_stopping: EarlyStoppingSummary,
    /// Validation metrics evaluated after training.
    pub validation: EvaluationMetrics,
    /// Test metrics evaluated after training.
    pub test: EvaluationMetrics,
}

/// Coordinate-frame provenance shared by training and experiment summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateFrameProvenance {
    /// Dataset-side coordinate-frame contract.
    #[serde(default)]
    pub coordinate_frame_contract: String,
    /// Candidate artifact coordinate-frame contract.
    #[serde(default)]
    pub coordinate_frame_artifact_contract: String,
    /// Retained examples with finite coordinate-frame origins.
    #[serde(default)]
    pub coordinate_frame_origin_valid_examples: usize,
    /// Retained examples passing the ligand-centered model-frame check.
    #[serde(default)]
    pub ligand_centered_coordinate_frame_examples: usize,
    /// Whether retained pocket coordinates were already centered upstream.
    #[serde(default)]
    pub pocket_coordinates_centered_upstream: bool,
    /// Whether source-frame coordinates can be reconstructed from model coordinates plus origin.
    #[serde(default)]
    pub source_coordinate_reconstruction_supported: bool,
    /// Rotation augmentation attempts recorded by data loading.
    #[serde(default)]
    pub rotation_augmentation_attempted_examples: usize,
    /// Rotation augmentation applications recorded by data loading.
    #[serde(default)]
    pub rotation_augmentation_applied_examples: usize,
    /// Role assigned to rotation-consistency measurements.
    #[serde(default = "default_rotation_consistency_role")]
    pub rotation_consistency_role: String,
}

impl Default for CoordinateFrameProvenance {
    fn default() -> Self {
        Self {
            coordinate_frame_contract: String::new(),
            coordinate_frame_artifact_contract: String::new(),
            coordinate_frame_origin_valid_examples: 0,
            ligand_centered_coordinate_frame_examples: 0,
            pocket_coordinates_centered_upstream: false,
            source_coordinate_reconstruction_supported: false,
            rotation_augmentation_attempted_examples: 0,
            rotation_augmentation_applied_examples: 0,
            rotation_consistency_role: default_rotation_consistency_role(),
        }
    }
}

impl CoordinateFrameProvenance {
    /// Build persisted coordinate-frame provenance from dataset validation.
    pub fn from_dataset_validation(validation: &DatasetValidationReport) -> Self {
        Self {
            coordinate_frame_contract: validation.coordinate_frame_contract.clone(),
            coordinate_frame_artifact_contract: validation.coordinate_frame_artifact_contract.clone(),
            coordinate_frame_origin_valid_examples: validation.coordinate_frame_origin_valid_examples,
            ligand_centered_coordinate_frame_examples: validation
                .ligand_centered_coordinate_frame_examples,
            pocket_coordinates_centered_upstream: validation.pocket_coordinates_centered_upstream,
            source_coordinate_reconstruction_supported: validation
                .source_coordinate_reconstruction_supported,
            rotation_augmentation_attempted_examples: validation
                .rotation_augmentation_attempted_examples,
            rotation_augmentation_applied_examples: validation
                .rotation_augmentation_applied_examples,
            rotation_consistency_role: default_rotation_consistency_role(),
        }
    }
}

fn default_rotation_consistency_role() -> String {
    "diagnostic_not_exact_equivariance_claim".to_string()
}

/// Per-run report separating optimizer-facing terms from detached diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveCoverageReport {
    /// Schema version for this report.
    #[serde(default = "default_objective_coverage_schema_version")]
    pub schema_version: u32,
    /// Configured primary objective.
    #[serde(default)]
    pub primary_objective: String,
    /// One record per observed primary or auxiliary component.
    #[serde(default)]
    pub records: Vec<ObjectiveCoverageRecord>,
}

impl Default for ObjectiveCoverageReport {
    fn default() -> Self {
        Self {
            schema_version: default_objective_coverage_schema_version(),
            primary_objective: String::new(),
            records: Vec::new(),
        }
    }
}

fn default_objective_coverage_schema_version() -> u32 {
    1
}

/// Coverage row for one objective component or family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveCoverageRecord {
    /// `primary` or `auxiliary`.
    pub objective_scope: String,
    /// Stable component or family name.
    pub component_name: String,
    /// Source anchor for this component.
    pub anchor: String,
    /// Whether this component is tensor-differentiable in the current implementation.
    pub differentiable: bool,
    /// Whether this component can contribute to optimizer steps.
    pub optimizer_facing: bool,
    /// Stages where this component was observed.
    #[serde(default)]
    pub observed_stages: Vec<String>,
    /// Compact stage availability label.
    pub stage_availability: String,
    /// Human-readable claim boundary note.
    pub claim_boundary_note: String,
}

pub(crate) fn objective_coverage_report(
    config: &ResearchConfig,
    history: &[StepMetrics],
) -> ObjectiveCoverageReport {
    let mut records: BTreeMap<(String, String), ObjectiveCoverageRecord> = BTreeMap::new();

    for step in history {
        let stage_label = format!("{:?}", step.stage);
        for component in &step.losses.primary.component_provenance {
            let key = ("primary".to_string(), component.component_name.clone());
            let record = records.entry(key).or_insert_with(|| ObjectiveCoverageRecord {
                objective_scope: "primary".to_string(),
                component_name: component.component_name.clone(),
                anchor: component.anchor.clone(),
                differentiable: component.differentiable,
                optimizer_facing: component.optimizer_facing,
                observed_stages: Vec::new(),
                stage_availability: String::new(),
                claim_boundary_note: primary_component_claim_boundary(component),
            });
            if !record.observed_stages.contains(&stage_label) {
                record.observed_stages.push(stage_label.clone());
            }
            record.differentiable |= component.differentiable;
            record.optimizer_facing |= component.optimizer_facing;
        }

        for entry in &step
            .losses
            .auxiliaries
            .auxiliary_objective_report
            .entries
        {
            let family = format!("{:?}", entry.family);
            let key = ("auxiliary".to_string(), family.clone());
            let trainable = entry.execution_mode == "trainable";
            let optimizer_facing = trainable && entry.enabled && entry.effective_weight > 0.0;
            let record = records.entry(key).or_insert_with(|| ObjectiveCoverageRecord {
                objective_scope: "auxiliary".to_string(),
                component_name: family.clone(),
                anchor: "staged_auxiliary_objective".to_string(),
                differentiable: trainable,
                optimizer_facing,
                observed_stages: Vec::new(),
                stage_availability: String::new(),
                claim_boundary_note:
                    "staged auxiliary family; optimizer-facing only when execution_mode=trainable and effective_weight>0"
                        .to_string(),
            });
            if !record.observed_stages.contains(&stage_label) {
                record.observed_stages.push(stage_label.clone());
            }
            record.differentiable |= trainable;
            record.optimizer_facing |= optimizer_facing;
        }
    }

    if records.is_empty() {
        seed_primary_objective_coverage(config, &mut records);
    }

    let mut records = records.into_values().collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.objective_scope
            .cmp(&right.objective_scope)
            .then(left.component_name.cmp(&right.component_name))
    });
    for record in &mut records {
        record.observed_stages.sort();
        record.stage_availability = if record.observed_stages.is_empty() {
            "not_observed_in_history".to_string()
        } else {
            record.observed_stages.join(",")
        };
    }

    ObjectiveCoverageReport {
        schema_version: default_objective_coverage_schema_version(),
        primary_objective: config.training.primary_objective.as_str().to_string(),
        records,
    }
}

fn primary_component_claim_boundary(component: &PrimaryObjectiveComponentProvenance) -> String {
    if component.component_name.starts_with("rollout_eval_") {
        return "detached sampled-rollout diagnostic; not optimizer-facing unless a future tensor-preserving trainable_rollout_* objective is implemented".to_string();
    }
    if component.component_name.starts_with("flow_") {
        return "flow component is optimizer-facing only for flow-compatible primary objectives"
            .to_string();
    }
    if component.component_name == "rollout" {
        return "reserved for future tensor-preserving rollout objective".to_string();
    }
    if component.optimizer_facing {
        "tensor-preserving primary objective component".to_string()
    } else {
        "diagnostic primary objective component".to_string()
    }
}

fn seed_primary_objective_coverage(
    config: &ResearchConfig,
    records: &mut BTreeMap<(String, String), ObjectiveCoverageRecord>,
) {
    let components: Vec<&'static str> = match config.training.primary_objective {
        crate::config::PrimaryObjectiveConfig::SurrogateReconstruction => {
            vec!["topology", "geometry", "pocket_anchor"]
        }
        crate::config::PrimaryObjectiveConfig::ConditionedDenoising => [
            "topology",
            "geometry",
            "pocket_anchor",
            "rollout_eval_recovery",
            "rollout_eval_pocket_anchor",
            "rollout_eval_stop",
        ]
        .to_vec(),
        crate::config::PrimaryObjectiveConfig::FlowMatching => {
            vec!["flow_velocity", "flow_endpoint"]
        }
        crate::config::PrimaryObjectiveConfig::DenoisingFlowMatching => [
            "topology",
            "geometry",
            "pocket_anchor",
            "rollout_eval_recovery",
            "rollout_eval_pocket_anchor",
            "rollout_eval_stop",
            "flow_velocity",
            "flow_endpoint",
        ]
        .to_vec(),
    };

    for component_name in components {
        let optimizer_facing = !component_name.starts_with("rollout_eval_");
        let component = PrimaryObjectiveComponentProvenance {
            component_name: (*component_name).to_string(),
            anchor: if component_name.starts_with("flow_") {
                "flow_matching".to_string()
            } else if component_name.starts_with("rollout_eval_") {
                "sampled_rollout_record".to_string()
            } else {
                "decoder_or_surrogate".to_string()
            },
            differentiable: optimizer_facing,
            optimizer_facing,
            role: if optimizer_facing {
                "optimizer_facing".to_string()
            } else {
                "evaluation_only".to_string()
            },
            effective_branch_weight: None,
            branch_schedule_source: None,
        };
        records.insert(
            ("primary".to_string(), (*component_name).to_string()),
            ObjectiveCoverageRecord {
                objective_scope: "primary".to_string(),
                component_name: (*component_name).to_string(),
                anchor: component.anchor.clone(),
                differentiable: component.differentiable,
                optimizer_facing: component.optimizer_facing,
                observed_stages: Vec::new(),
                stage_availability: "not_observed_in_history".to_string(),
                claim_boundary_note: primary_component_claim_boundary(&component),
            },
        );
    }
}

/// One validation checkpoint-selection observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationHistoryEntry {
    /// Completed optimization step associated with the validation run.
    pub step: usize,
    /// Split label, normally `validation`.
    pub split: String,
    /// Selected metric name.
    pub metric_name: String,
    /// Selected metric value.
    pub metric_value: f64,
    /// Whether larger values are better for this metric.
    pub higher_is_better: bool,
    /// Whether this validation improved the best checkpoint.
    pub improved: bool,
    /// Full validation metrics snapshot.
    pub metrics: EvaluationMetrics,
}

/// Summary of the best validation-selected checkpoint artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestCheckpointSummary {
    /// Training step whose weights were selected.
    pub step: usize,
    /// Metric used for selection.
    pub metric_name: String,
    /// Best metric value.
    pub metric_value: f64,
    /// Whether larger values are better for this metric.
    pub higher_is_better: bool,
    /// Path to best checkpoint weights.
    pub weights_path: PathBuf,
    /// Path to best checkpoint metadata.
    pub metadata_path: PathBuf,
}

/// Early-stopping summary persisted with the training run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EarlyStoppingSummary {
    /// Whether early stopping was enabled.
    pub enabled: bool,
    /// Configured patience, if enabled.
    pub patience: Option<usize>,
    /// Whether training stopped before max_steps due to validation stagnation.
    pub stopped_early: bool,
    /// Step that triggered early stopping.
    pub stop_step: Option<usize>,
    /// Best validation step.
    pub best_step: Option<usize>,
    /// Number of consecutive checks without improvement at run end.
    pub checks_without_improvement: usize,
}

/// Shared reproducibility metadata persisted across train and experiment runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityMetadata {
    /// Stable hash of the effective config JSON.
    pub config_hash: String,
    /// Stable hash of the dataset validation artifact JSON.
    pub dataset_validation_fingerprint: String,
    /// Metric schema version used for evaluation artifacts.
    pub metric_schema_version: u32,
    /// Shared bundle schema version used on disk.
    pub artifact_bundle_schema_version: u32,
    /// Explicit deterministic runtime knobs and seeds that affect replay.
    #[serde(default)]
    pub determinism_controls: DeterminismControls,
    /// Bounded replay tolerance used by reviewer-surface drift checks.
    #[serde(default)]
    pub replay_tolerance: ReplayTolerance,
    /// Explicit resume semantics for this run family.
    pub resume_contract: ResumeContract,
    /// Whether the run resumed from a previous checkpoint.
    pub resume_provenance: ResumeProvenance,
}

/// Runtime and seed controls that materially affect replay behavior.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct DeterminismControls {
    /// Protein split seed.
    #[serde(default)]
    pub split_seed: u64,
    /// Decoder corruption seed.
    #[serde(default)]
    pub corruption_seed: u64,
    /// Decoder rollout sampling seed.
    #[serde(default)]
    pub sampling_seed: u64,
    /// Generation mode active for target construction and rollout conditioning.
    #[serde(default)]
    pub generation_mode: String,
    /// Generation-target corruption seed recorded explicitly for replay contracts.
    #[serde(default)]
    pub generation_corruption_seed: u64,
    /// Generation-target sampling seed recorded explicitly for replay contracts.
    #[serde(default)]
    pub generation_sampling_seed: u64,
    /// Molecular-flow contract version for the configured generation backend.
    #[serde(default)]
    pub flow_contract_version: String,
    /// Stable hash of the configured multi-modal flow branch schedule.
    #[serde(default)]
    pub flow_branch_schedule_hash: String,
    /// Effective mini-batch size used by the sampler.
    #[serde(default)]
    pub batch_size: usize,
    /// Whether the sampler shuffles example order for each epoch.
    #[serde(default)]
    pub sampler_shuffle: bool,
    /// Base seed used by the epoch sampler.
    #[serde(default)]
    pub sampler_seed: u64,
    /// Whether the sampler drops the final short batch.
    #[serde(default)]
    pub sampler_drop_last: bool,
    /// Optional cap on sampler epochs.
    #[serde(default)]
    pub sampler_max_epochs: Option<usize>,
    /// Runtime device string.
    #[serde(default)]
    pub device: String,
    /// Configured Rust-side data workers.
    #[serde(default)]
    pub data_workers: usize,
    /// Optional libtorch intra-op thread count.
    #[serde(default)]
    pub tch_intra_op_threads: Option<i32>,
    /// Optional libtorch inter-op thread count.
    #[serde(default)]
    pub tch_inter_op_threads: Option<i32>,
}

/// Reviewer-facing tolerances for bounded replay and drift checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTolerance {
    /// Maximum absolute drift allowed for leakage proxy mean.
    pub leakage_proxy_mean_abs: f64,
    /// Maximum absolute drift allowed for strict pocket fit.
    pub strict_pocket_fit_score_abs: f64,
    /// Maximum absolute drift allowed for clash fraction.
    pub clash_fraction_abs: f64,
    /// Maximum absolute drift allowed for chemistry quality metrics.
    pub chemistry_quality_abs: f64,
}

impl Default for ReplayTolerance {
    fn default() -> Self {
        Self {
            leakage_proxy_mean_abs: 0.01,
            strict_pocket_fit_score_abs: 0.03,
            clash_fraction_abs: 0.02,
            chemistry_quality_abs: 0.05,
        }
    }
}

/// Reviewer-facing continuity class for resumed runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResumeContinuityMode {
    /// Fresh run with no prior checkpoint applied.
    #[default]
    FreshRun,
    /// Weights, step, history, and metadata were restored, but optimizer moments were not.
    MetadataOnlyContinuation,
    /// Full optimizer/scheduler internal state was restored, enabling strict replay claims.
    FullOptimizerContinuation,
}

/// Explicit statement of what a resume operation restores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeContract {
    /// Stable version identifier for the contract.
    pub version: String,
    /// Whether model weights are restored.
    pub restores_model_weights: bool,
    /// Whether the step counter is restored.
    pub restores_step: bool,
    /// Whether training-history summaries are restored.
    pub restores_history: bool,
    /// Whether optimizer state is restored.
    pub restores_optimizer_state: bool,
    /// Explicit resume mode supported by this run family.
    #[serde(default)]
    pub resume_mode: ResumeMode,
    /// Reviewer-facing label for the strongest continuity this contract can support.
    #[serde(default)]
    pub continuity_mode: ResumeContinuityMode,
    /// Whether the run family supports strict deterministic replay when resumed.
    #[serde(default)]
    pub supports_strict_replay: bool,
    /// Human-readable note describing the limits of reproducibility.
    pub notes: String,
}

/// Summary of whether and how the current run resumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeProvenance {
    /// Whether this run resumed from a prior checkpoint.
    pub resumed: bool,
    /// Step loaded from the checkpoint, if any.
    pub resumed_from_step: Option<usize>,
    /// Config hash recorded in the loaded checkpoint, if any.
    pub checkpoint_config_hash: Option<String>,
    /// Dataset validation fingerprint recorded in the loaded checkpoint, if any.
    pub checkpoint_dataset_fingerprint: Option<String>,
    /// Whether optimizer-state metadata was present in the loaded checkpoint.
    #[serde(default)]
    pub restored_optimizer_state_metadata: bool,
    /// Whether scheduler-state metadata was present in the loaded checkpoint.
    #[serde(default)]
    pub restored_scheduler_state_metadata: bool,
    /// Explicit resume mode actually achieved by this run.
    #[serde(default)]
    pub resume_mode: ResumeMode,
    /// Reviewer-facing label for the continuity actually achieved by this run.
    #[serde(default)]
    pub continuity_mode: ResumeContinuityMode,
    /// Whether this specific resumed run qualifies as a strict replay.
    #[serde(default)]
    pub strict_replay_achieved: bool,
}

/// High-level replay/evidence compatibility class for a pair of artifacts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCompatibilityClass {
    /// All replay identity fields match and optimizer internals support exact replay.
    StrictReplayCompatible,
    /// Claim-bearing evidence can be compared, but exact optimizer replay is not supported.
    EvidenceCompatible,
    /// At least one evidence identity field differs or is missing.
    Incompatible,
}

/// One concrete field mismatch found while comparing replay identities.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCompatibilityMismatch {
    /// Dot-path of the mismatched field.
    pub field: String,
    /// Expected value from the training summary or strict replay contract.
    pub expected: String,
    /// Observed value from the checkpoint metadata.
    pub observed: String,
    /// Whether this mismatch blocks strict replay claims.
    pub replay_blocking: bool,
    /// Whether this mismatch blocks evidence-level comparison.
    pub evidence_blocking: bool,
}

/// Replay compatibility report for a training summary and checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCompatibilityReport {
    /// Strongest compatibility class supported by the compared artifacts.
    pub class: ReplayCompatibilityClass,
    /// Whether strict optimizer-state-identical replay can be claimed.
    pub replay_compatible: bool,
    /// Whether claim-bearing evidence can still be compared.
    pub evidence_compatible: bool,
    /// Concrete field mismatches.
    pub mismatches: Vec<ReplayCompatibilityMismatch>,
    /// Human-readable interpretation notes.
    pub notes: Vec<String>,
}

/// Shared on-disk artifact bundle for training and experiment workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunArtifactBundle {
    /// Bundle schema version.
    pub schema_version: u32,
    /// Run kind stored in this bundle.
    pub run_kind: RunKind,
    /// Base directory for persisted outputs.
    pub artifact_dir: PathBuf,
    /// Config hash used to produce the run.
    pub config_hash: String,
    /// Dataset validation fingerprint associated with the run.
    pub dataset_validation_fingerprint: String,
    /// Metric schema version used by summaries.
    pub metric_schema_version: u32,
    /// Reviewer-facing backend environment fingerprint when external backends are configured.
    #[serde(default)]
    pub backend_environment: Option<crate::experiments::BackendEnvironmentReport>,
    /// Paths to individual artifacts.
    pub paths: RunArtifactPaths,
}

/// Research workflow variant represented by a bundle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    /// Config-driven training workflow.
    Training,
    /// Config-driven unseen-pocket experiment workflow.
    Experiment,
}

/// Concrete files persisted for a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunArtifactPaths {
    /// Persisted config snapshot.
    pub config_snapshot: PathBuf,
    /// Dataset validation report.
    pub dataset_validation_report: PathBuf,
    /// Split audit report.
    pub split_report: PathBuf,
    /// Primary run summary file.
    pub run_summary: PathBuf,
    /// Shared run bundle description.
    pub run_bundle: PathBuf,
    /// Latest checkpoint weights path, if any.
    pub latest_checkpoint: Option<PathBuf>,
}
