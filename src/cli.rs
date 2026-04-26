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
