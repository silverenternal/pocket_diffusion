//! Main binary for the modular research stack and legacy compatibility demos.

use clap::{Args, Parser, Subcommand, ValueEnum};

use pocket_diffusion::{
    config, experiments, legacy,
    training::{self, RunArtifactBundle, RunKind},
};

fn main() {
    env_logger::init();
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    maybe_print_compatibility_notice(&cli.command);
    match cli.command {
        CliCommand::Research { command } => match command {
            ResearchCommand::Inspect(args) => inspect_dataset_from_config(&args.config),
            ResearchCommand::Train(args) => run_training_from_config(&args.config, args.resume),
            ResearchCommand::Experiment(args) => {
                run_experiment_from_config(&args.config, args.resume)
            }
            ResearchCommand::Ablate(args) => run_ablation_matrix_from_config(&args.config),
            ResearchCommand::Search(args) => run_automated_search_from_config(&args.config),
            ResearchCommand::MultiSeed(args) => run_multi_seed_experiment_from_config(&args.config),
            ResearchCommand::Generate(args) => run_generation_demo_from_config(
                &args.config,
                args.resume,
                args.example_id.as_deref(),
                args.num_candidates,
            ),
        },
        CliCommand::Validate(args) => validate_config(args.kind, &args.config),
        CliCommand::Report(args) => report_run(&args.artifact_dir),
        CliCommand::Phase1Demo => {
            training::run_phase1_demo();
            Ok(())
        }
        CliCommand::Phase3Demo => training::run_phase3_training_demo(),
        CliCommand::Phase4Demo => training::run_phase4_experiment_demo(),
        CliCommand::LegacyDemo(args) => {
            legacy::run_legacy_demo(args.num_candidates, args.top_k, args.modular_bridge);
            Ok(())
        }
        CliCommand::CompatInspect(args) => inspect_dataset_from_config(&args.config),
        CliCommand::CompatTrain(args) => run_training_from_config(&args.config, args.resume),
        CliCommand::CompatExperiment(args) => run_experiment_from_config(&args.config, args.resume),
    }
}

#[derive(Debug, Parser)]
#[command(name = "pocket_diffusion")]
#[command(about = "Rust-first modular research CLI with legacy compatibility paths")]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Config-driven modular research workflows.
    Research {
        #[command(subcommand)]
        command: ResearchCommand,
    },
    /// Validate a config without loading data or launching training.
    Validate(ValidateArgs),
    /// Read a shared run artifact bundle and print the stored summary.
    Report(ReportArgs),
    /// Compatibility demo flag for the early inspection surface.
    #[command(hide = true, name = "--phase1")]
    Phase1Demo,
    /// Compatibility demo flag for staged training.
    #[command(hide = true, name = "--train-phase3")]
    Phase3Demo,
    /// Compatibility demo flag for unseen-pocket experiments.
    #[command(hide = true, name = "--phase4")]
    Phase4Demo,
    /// Legacy generation demo kept for compatibility.
    LegacyDemo(LegacyDemoArgs),
    /// Deprecated compatibility alias for `research inspect`.
    #[command(hide = true, name = "--inspect-config")]
    CompatInspect(CompatConfigArgs),
    /// Deprecated compatibility alias for `research train`.
    #[command(hide = true, name = "--train-config")]
    CompatTrain(CompatRunArgs),
    /// Deprecated compatibility alias for `research experiment`.
    #[command(hide = true, name = "--experiment-config")]
    CompatExperiment(CompatRunArgs),
}

#[derive(Debug, Subcommand)]
enum ResearchCommand {
    /// Inspect a dataset from a research config.
    Inspect(ConfigArgs),
    /// Run staged training from a research config.
    Train(RunArgs),
    /// Run the unseen-pocket experiment from an experiment config.
    Experiment(RunArgs),
    /// Execute the configured ablation matrix from an experiment config.
    Ablate(ConfigArgs),
    /// Run bounded automated tuning across claim-bearing surfaces.
    Search(ConfigArgs),
    /// Run repeated seed-level claim surfaces and aggregate stability metrics.
    MultiSeed(ConfigArgs),
    /// Emit conditioned ligand candidates from the modular research path.
    Generate(GenerateArgs),
}

#[derive(Debug, Args)]
struct ConfigArgs {
    #[arg(long)]
    config: String,
}

#[derive(Debug, Args)]
struct RunArgs {
    #[arg(long)]
    config: String,
    #[arg(long, default_value_t = false)]
    resume: bool,
}

#[derive(Debug, Args)]
struct ValidateArgs {
    #[arg(long, value_enum)]
    kind: ConfigKind,
    #[arg(long)]
    config: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConfigKind {
    Research,
    Experiment,
}

#[derive(Debug, Args)]
struct ReportArgs {
    #[arg(long)]
    artifact_dir: String,
}

#[derive(Debug, Args)]
struct LegacyDemoArgs {
    #[arg(default_value_t = 10)]
    num_candidates: usize,
    #[arg(default_value_t = 3)]
    top_k: usize,
    #[arg(long, default_value_t = false)]
    modular_bridge: bool,
}

#[derive(Debug, Args)]
struct GenerateArgs {
    #[arg(long)]
    config: String,
    #[arg(long, default_value_t = false)]
    resume: bool,
    #[arg(long)]
    example_id: Option<String>,
    #[arg(long, default_value_t = 4)]
    num_candidates: usize,
}

#[derive(Debug, Args)]
struct CompatConfigArgs {
    config: String,
}

#[derive(Debug, Args)]
struct CompatRunArgs {
    config: String,
    #[arg(long, default_value_t = false)]
    resume: bool,
}

fn maybe_print_compatibility_notice(command: &CliCommand) {
    if let Some(message) = compatibility_notice(command) {
        eprintln!("note: {message}");
    }
}

fn compatibility_notice(command: &CliCommand) -> Option<String> {
    match command {
        CliCommand::CompatInspect(args) => Some(format!(
            "compatibility flag detected; prefer `research inspect --config {}` for the modular research stack",
            args.config
        )),
        CliCommand::CompatTrain(args) => Some(format!(
            "compatibility flag detected; prefer `research train --config {}{}` for the modular research stack",
            args.config,
            if args.resume { " --resume" } else { "" }
        )),
        CliCommand::CompatExperiment(args) => Some(format!(
            "compatibility flag detected; prefer `research experiment --config {}{}` for the modular research stack",
            args.config,
            if args.resume { " --resume" } else { "" }
        )),
        CliCommand::Phase1Demo => Some(
            "demo compatibility flag detected; prefer `research inspect --config <path>` or `research train --config <path>` for config-driven modular runs".to_string(),
        ),
        CliCommand::Phase3Demo => Some(
            "demo compatibility flag detected; prefer `research train --config <path>` for config-driven staged training".to_string(),
        ),
        CliCommand::Phase4Demo => Some(
            "demo compatibility flag detected; prefer `research experiment --config <path>` for config-driven unseen-pocket evaluation".to_string(),
        ),
        CliCommand::LegacyDemo(args) => Some(format!(
            "legacy demo path detected; this preserves compatibility but is not the primary modular research interface. Current equivalent compatibility command: `legacy-demo {} {}`",
            args.num_candidates, args.top_k
        )),
        CliCommand::Research { .. } | CliCommand::Validate(..) | CliCommand::Report(..) => None,
    }
}

fn run_training_from_config(path: &str, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    let summary = training::run_training_from_config(path, resume)?;
    training::print_training_run(&summary);
    Ok(())
}

fn run_experiment_from_config(path: &str, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    let summary = experiments::run_experiment_from_config(path, resume)?;
    training::print_experiment_run(&summary);
    Ok(())
}

fn run_ablation_matrix_from_config(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let summary = experiments::run_ablation_matrix_from_config(path)?;
    println!("================================================");
    println!("  Config-Driven Ablation Matrix");
    println!("================================================");
    for variant in summary.variants {
        println!(
            "{} [{}] | valid={:?} pocket={:?} unseen={:.4}",
            variant.variant_label,
            variant.test.interaction_mode,
            variant.test.candidate_valid_fraction,
            variant.test.pocket_compatibility_fraction,
            variant.test.unseen_protein_fraction,
        );
    }
    Ok(())
}

fn run_automated_search_from_config(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let summary = experiments::run_automated_search_from_config(path)?;
    training::print_automated_search(&summary);
    Ok(())
}

fn run_multi_seed_experiment_from_config(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let summary = experiments::run_multi_seed_experiment_from_config(path)?;
    println!("================================================");
    println!("  Multi-Seed Experiment Summary");
    println!("================================================");
    println!("artifact root: {}", summary.artifact_root.display());
    println!("seeds: {}", summary.seed_runs.len());
    println!("decision: {}", summary.stability_decision);
    println!(
        "valid mean/std: {:.4}/{:.4}",
        summary.aggregates.candidate_valid_fraction.mean,
        summary.aggregates.candidate_valid_fraction.std
    );
    println!(
        "pocket fit mean/std: {:.4}/{:.4}",
        summary.aggregates.strict_pocket_fit_score.mean,
        summary.aggregates.strict_pocket_fit_score.std
    );
    Ok(())
}

fn run_generation_demo_from_config(
    path: &str,
    resume: bool,
    example_id: Option<&str>,
    num_candidates: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary =
        experiments::run_generation_demo_from_config(path, resume, example_id, num_candidates)?;
    println!("================================================");
    println!("  Config-Driven Modular Generation Demo");
    println!("================================================");
    println!(
        "example: {} | protein: {}",
        summary.example_id, summary.protein_id
    );
    println!("candidates: {}", summary.candidate_count);
    println!("rollout steps: {}", summary.rollout_steps);
    println!("loaded checkpoint: {}", summary.loaded_checkpoint);
    training::print_eval_metrics(&experiments::EvaluationMetrics {
        representation_diagnostics: experiments::RepresentationDiagnostics {
            finite_forward_fraction: 0.0,
            unique_complex_fraction: 0.0,
            unseen_protein_fraction: 0.0,
            distance_probe_rmse: 0.0,
            topology_pocket_cosine_alignment: 0.0,
            topology_reconstruction_mse: 0.0,
            slot_activation_mean: 0.0,
            gate_activation_mean: 0.0,
            leakage_proxy_mean: 0.0,
        },
        proxy_task_metrics: experiments::ProxyTaskMetrics {
            affinity_probe_mae: 0.0,
            affinity_probe_rmse: 0.0,
            labeled_fraction: 0.0,
            affinity_by_measurement: Vec::new(),
        },
        split_context: experiments::SplitContextMetrics {
            example_count: 0,
            unique_complex_count: 0,
            unique_protein_count: 0,
            train_reference_protein_count: 0,
            ligand_atom_count_bins: std::collections::BTreeMap::new(),
            pocket_atom_count_bins: std::collections::BTreeMap::new(),
            measurement_family_histogram: std::collections::BTreeMap::new(),
        },
        resource_usage: experiments::ResourceUsageMetrics {
            memory_usage_mb: 0.0,
            evaluation_time_ms: 0.0,
            examples_per_second: 0.0,
            average_ligand_atoms: 0.0,
            average_pocket_atoms: 0.0,
        },
        real_generation_metrics: summary.real_generation_metrics,
        layered_generation_metrics: experiments::LayeredGenerationMetrics {
            raw_rollout: experiments::CandidateLayerMetrics {
                candidate_count: 0,
                valid_fraction: 0.0,
                pocket_contact_fraction: 0.0,
                mean_centroid_offset: 0.0,
                clash_fraction: 0.0,
                mean_displacement: 0.0,
                atom_change_fraction: 0.0,
                uniqueness_proxy_fraction: 0.0,
                atom_type_sequence_diversity: 0.0,
                bond_topology_diversity: 0.0,
                coordinate_shape_diversity: 0.0,
                novel_atom_type_sequence_fraction: 0.0,
                novel_bond_topology_fraction: 0.0,
                novel_coordinate_shape_fraction: 0.0,
            },
            repaired_candidates: experiments::CandidateLayerMetrics {
                candidate_count: 0,
                valid_fraction: 0.0,
                pocket_contact_fraction: 0.0,
                mean_centroid_offset: 0.0,
                clash_fraction: 0.0,
                mean_displacement: 0.0,
                atom_change_fraction: 0.0,
                uniqueness_proxy_fraction: 0.0,
                atom_type_sequence_diversity: 0.0,
                bond_topology_diversity: 0.0,
                coordinate_shape_diversity: 0.0,
                novel_atom_type_sequence_fraction: 0.0,
                novel_bond_topology_fraction: 0.0,
                novel_coordinate_shape_fraction: 0.0,
            },
            inferred_bond_candidates: experiments::CandidateLayerMetrics {
                candidate_count: 0,
                valid_fraction: 0.0,
                pocket_contact_fraction: 0.0,
                mean_centroid_offset: 0.0,
                clash_fraction: 0.0,
                mean_displacement: 0.0,
                atom_change_fraction: 0.0,
                uniqueness_proxy_fraction: 0.0,
                atom_type_sequence_diversity: 0.0,
                bond_topology_diversity: 0.0,
                coordinate_shape_diversity: 0.0,
                novel_atom_type_sequence_fraction: 0.0,
                novel_bond_topology_fraction: 0.0,
                novel_coordinate_shape_fraction: 0.0,
            },
            reranked_candidates: experiments::CandidateLayerMetrics::default(),
            deterministic_proxy_candidates: experiments::CandidateLayerMetrics::default(),
            reranker_calibration: experiments::RerankerCalibrationReport::default(),
            backend_scored_candidates: std::collections::BTreeMap::new(),
            method_comparison: experiments::MethodComparisonSummary::default(),
        },
        method_comparison: experiments::MethodComparisonSummary::default(),
        comparison_summary: experiments::GenerationQualitySummary {
            primary_objective: "generation_demo".to_string(),
            variant_label: Some("rollout_demo".to_string()),
            interaction_mode: "n/a".to_string(),
            candidate_valid_fraction: None,
            pocket_contact_fraction: None,
            pocket_compatibility_fraction: None,
            mean_centroid_offset: None,
            strict_pocket_fit_score: None,
            unique_smiles_fraction: None,
            unseen_protein_fraction: 0.0,
            topology_specialization_score: 0.0,
            geometry_specialization_score: 0.0,
            pocket_specialization_score: 0.0,
            slot_activation_mean: 0.0,
            gate_activation_mean: 0.0,
            leakage_proxy_mean: 0.0,
        },
        slot_stability: experiments::SlotStabilityMetrics::default(),
        strata: Vec::new(),
    });
    Ok(())
}

fn inspect_dataset_from_config(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let inspection = training::inspect_dataset_from_config(path)?;
    training::print_dataset_inspection(&inspection);
    Ok(())
}

fn validate_config(kind: ConfigKind, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    match kind {
        ConfigKind::Research => {
            let config = config::load_research_config(path)?;
            config.validate()?;
            println!("research config is valid: {path}");
        }
        ConfigKind::Experiment => {
            let config = experiments::load_experiment_config(path)?;
            config.validate()?;
            println!("experiment config is valid: {path}");
        }
    }
    Ok(())
}

fn report_run(artifact_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let bundle_path = std::path::Path::new(artifact_dir).join("run_artifacts.json");
    let bundle: RunArtifactBundle = serde_json::from_str(&std::fs::read_to_string(&bundle_path)?)?;
    match bundle.run_kind {
        RunKind::Training => {
            let summary: training::TrainingRunSummary =
                serde_json::from_str(&std::fs::read_to_string(bundle.paths.run_summary)?)?;
            training::print_training_run(&summary);
        }
        RunKind::Experiment => {
            let summary: experiments::UnseenPocketExperimentSummary =
                serde_json::from_str(&std::fs::read_to_string(bundle.paths.run_summary)?)?;
            training::print_experiment_run(&summary);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse_cli(args: &[&str]) -> CliCommand {
        Cli::try_parse_from(args)
            .unwrap_or_else(|err| panic!("cli parse failed: {err}"))
            .command
    }

    #[test]
    fn parse_research_train_command_with_resume() {
        let command = parse_cli(&[
            "pocket_diffusion",
            "research",
            "train",
            "--config",
            "configs/research_manifest.json",
            "--resume",
        ]);

        match command {
            CliCommand::Research {
                command: ResearchCommand::Train(args),
            } => {
                assert_eq!(args.config, "configs/research_manifest.json");
                assert!(args.resume);
            }
            _ => panic!("expected research train command"),
        }
    }

    #[test]
    fn parse_legacy_demo_command() {
        let command = parse_cli(&["pocket_diffusion", "legacy-demo", "12", "4"]);

        match command {
            CliCommand::LegacyDemo(args) => {
                assert_eq!(args.num_candidates, 12);
                assert_eq!(args.top_k, 4);
                assert!(!args.modular_bridge);
            }
            _ => panic!("expected legacy demo command"),
        }
    }

    #[test]
    fn parse_research_generate_command() {
        let command = parse_cli(&[
            "pocket_diffusion",
            "research",
            "generate",
            "--config",
            "configs/research_manifest.json",
            "--resume",
            "--num-candidates",
            "5",
        ]);

        match command {
            CliCommand::Research {
                command: ResearchCommand::Generate(args),
            } => {
                assert_eq!(args.config, "configs/research_manifest.json");
                assert!(args.resume);
                assert_eq!(args.num_candidates, 5);
            }
            _ => panic!("expected research generate command"),
        }
    }

    #[test]
    fn parse_research_search_command() {
        let command = parse_cli(&[
            "pocket_diffusion",
            "research",
            "search",
            "--config",
            "configs/unseen_pocket_claim_matrix.json",
        ]);

        match command {
            CliCommand::Research {
                command: ResearchCommand::Search(args),
            } => {
                assert_eq!(args.config, "configs/unseen_pocket_claim_matrix.json");
            }
            _ => panic!("expected research search command"),
        }
    }

    #[test]
    fn parse_research_multi_seed_command() {
        let command = parse_cli(&[
            "pocket_diffusion",
            "research",
            "multi-seed",
            "--config",
            "configs/unseen_pocket_multi_seed.json",
        ]);

        match command {
            CliCommand::Research {
                command: ResearchCommand::MultiSeed(args),
            } => {
                assert_eq!(args.config, "configs/unseen_pocket_multi_seed.json");
            }
            _ => panic!("expected research multi-seed command"),
        }
    }

    #[test]
    fn parse_compatibility_train_flag() {
        let command = parse_cli(&[
            "pocket_diffusion",
            "--train-config",
            "configs/research_manifest.json",
        ]);

        match command {
            CliCommand::CompatTrain(args) => {
                assert_eq!(args.config, "configs/research_manifest.json");
                assert!(!args.resume);
            }
            _ => panic!("expected compatibility research train command"),
        }
    }

    #[test]
    fn parse_research_experiment_command() {
        let command = parse_cli(&[
            "pocket_diffusion",
            "research",
            "experiment",
            "--config",
            "configs/unseen_pocket_manifest.json",
        ]);

        match command {
            CliCommand::Research {
                command: ResearchCommand::Experiment(args),
            } => {
                assert_eq!(args.config, "configs/unseen_pocket_manifest.json");
                assert!(!args.resume);
            }
            _ => panic!("expected research experiment command"),
        }
    }

    #[test]
    fn compatibility_notice_is_emitted_for_legacy_paths() {
        let command = CliCommand::Phase3Demo;
        assert!(compatibility_notice(&command).is_some());

        let command = CliCommand::LegacyDemo(LegacyDemoArgs {
            num_candidates: 10,
            top_k: 3,
            modular_bridge: false,
        });
        assert!(compatibility_notice(&command).is_some());
    }

    #[test]
    fn compatibility_notice_tracks_command_origin() {
        let compatibility = parse_cli(&[
            "pocket_diffusion",
            "--inspect-config",
            "configs/research_manifest.json",
        ]);
        let notice = compatibility_notice(&compatibility).unwrap();
        assert!(notice.contains("research inspect --config configs/research_manifest.json"));

        let canonical = parse_cli(&[
            "pocket_diffusion",
            "research",
            "inspect",
            "--config",
            "configs/research_manifest.json",
        ]);
        assert!(compatibility_notice(&canonical).is_none());
    }

    #[test]
    fn compatibility_notice_preserves_resume_suffix() {
        let compatibility = parse_cli(&[
            "pocket_diffusion",
            "--train-config",
            "configs/research_manifest.json",
            "--resume",
        ]);
        let notice = compatibility_notice(&compatibility).unwrap();

        assert!(notice.contains("research train --config configs/research_manifest.json --resume"));
    }

    #[test]
    fn clap_command_factory_builds() {
        let command = Cli::command();
        assert_eq!(command.get_name(), "pocket_diffusion");
    }
}
