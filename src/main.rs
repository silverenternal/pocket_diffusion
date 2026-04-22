//! Main binary for the modular research stack and legacy compatibility demos.

use std::env;

use pocket_diffusion::{experiments, legacy, training};

fn main() {
    env_logger::init();
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let command = parse_cli(&args)?;
    maybe_print_compatibility_notice(&command);
    match command {
        CliCommand::ResearchInspect { config, .. } => inspect_dataset_from_config(&config),
        CliCommand::ResearchTrain { config, resume, .. } => {
            run_training_from_config(&config, resume)
        }
        CliCommand::ResearchExperiment { config, resume, .. } => {
            run_experiment_from_config(&config, resume)
        }
        CliCommand::Phase1Demo => {
            training::run_phase1_demo();
            Ok(())
        }
        CliCommand::Phase3Demo => training::run_phase3_training_demo(),
        CliCommand::Phase4Demo => training::run_phase4_experiment_demo(),
        CliCommand::LegacyDemo {
            num_candidates,
            top_k,
        } => {
            legacy::run_legacy_demo(num_candidates, top_k);
            Ok(())
        }
        CliCommand::Help => {
            print_usage();
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandOrigin {
    Canonical,
    Compatibility,
}

enum CliCommand {
    ResearchInspect {
        config: String,
        origin: CommandOrigin,
    },
    ResearchTrain {
        config: String,
        resume: bool,
        origin: CommandOrigin,
    },
    ResearchExperiment {
        config: String,
        resume: bool,
        origin: CommandOrigin,
    },
    Phase1Demo,
    Phase3Demo,
    Phase4Demo,
    LegacyDemo {
        num_candidates: usize,
        top_k: usize,
    },
    Help,
}

fn parse_cli(args: &[String]) -> Result<CliCommand, Box<dyn std::error::Error>> {
    if args.len() <= 1 {
        return Ok(CliCommand::Help);
    }

    if let Some(path) = value_after_flag(args, "--experiment-config") {
        return Ok(CliCommand::ResearchExperiment {
            config: path.to_string(),
            resume: has_flag(args, "--resume"),
            origin: CommandOrigin::Compatibility,
        });
    }
    if let Some(path) = value_after_flag(args, "--train-config") {
        return Ok(CliCommand::ResearchTrain {
            config: path.to_string(),
            resume: has_flag(args, "--resume"),
            origin: CommandOrigin::Compatibility,
        });
    }
    if let Some(path) = value_after_flag(args, "--inspect-config") {
        return Ok(CliCommand::ResearchInspect {
            config: path.to_string(),
            origin: CommandOrigin::Compatibility,
        });
    }
    if has_flag(args, "--phase4") {
        return Ok(CliCommand::Phase4Demo);
    }
    if has_flag(args, "--train-phase3") {
        return Ok(CliCommand::Phase3Demo);
    }
    if has_flag(args, "--phase1") {
        return Ok(CliCommand::Phase1Demo);
    }

    match args.get(1).map(String::as_str) {
        Some("research") => parse_research_command(args),
        Some("legacy-demo") => Ok(CliCommand::LegacyDemo {
            num_candidates: args.get(2).and_then(|v| v.parse().ok()).unwrap_or(10),
            top_k: args.get(3).and_then(|v| v.parse().ok()).unwrap_or(3),
        }),
        Some("--help") | Some("-h") | Some("help") => Ok(CliCommand::Help),
        Some(first) if first.parse::<usize>().is_ok() => Ok(CliCommand::LegacyDemo {
            num_candidates: first.parse().unwrap_or(10),
            top_k: args.get(2).and_then(|v| v.parse().ok()).unwrap_or(3),
        }),
        Some(other) => Err(format!("unrecognized command `{other}`").into()),
        None => Ok(CliCommand::Help),
    }
}

fn parse_research_command(args: &[String]) -> Result<CliCommand, Box<dyn std::error::Error>> {
    let Some(subcommand) = args.get(2).map(String::as_str) else {
        return Err("missing research subcommand; use `inspect`, `train`, or `experiment`".into());
    };
    let Some(config) = value_after_flag(args, "--config") else {
        return Err("missing `--config <path>` for research command".into());
    };

    Ok(match subcommand {
        "inspect" => CliCommand::ResearchInspect {
            config: config.to_string(),
            origin: CommandOrigin::Canonical,
        },
        "train" => CliCommand::ResearchTrain {
            config: config.to_string(),
            resume: has_flag(args, "--resume"),
            origin: CommandOrigin::Canonical,
        },
        "experiment" => CliCommand::ResearchExperiment {
            config: config.to_string(),
            resume: has_flag(args, "--resume"),
            origin: CommandOrigin::Canonical,
        },
        other => return Err(format!("unsupported research subcommand `{other}`").into()),
    })
}

fn maybe_print_compatibility_notice(command: &CliCommand) {
    if let Some(message) = compatibility_notice(command) {
        eprintln!("note: {message}");
    }
}

fn compatibility_notice(command: &CliCommand) -> Option<String> {
    match command {
        CliCommand::ResearchInspect {
            config,
            origin: CommandOrigin::Compatibility,
        } => Some(format!(
            "compatibility flag detected; prefer `research inspect --config {config}` for the modular research stack"
        )),
        CliCommand::ResearchTrain {
            config,
            resume,
            origin: CommandOrigin::Compatibility,
        } => Some(format!(
            "compatibility flag detected; prefer `research train --config {config}{}` for the modular research stack",
            if *resume { " --resume" } else { "" }
        )),
        CliCommand::ResearchExperiment {
            config,
            resume,
            origin: CommandOrigin::Compatibility,
        } => Some(format!(
            "compatibility flag detected; prefer `research experiment --config {config}{}` for the modular research stack",
            if *resume { " --resume" } else { "" }
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
        CliCommand::LegacyDemo {
            num_candidates,
            top_k,
        } => Some(format!(
            "legacy demo path detected; this preserves compatibility but is not the primary modular research interface. Current equivalent compatibility command: `legacy-demo {num_candidates} {top_k}`"
        )),
        CliCommand::ResearchInspect { .. }
        | CliCommand::ResearchTrain { .. }
        | CliCommand::ResearchExperiment { .. }
        | CliCommand::Help => None,
    }
}

fn print_usage() {
    println!("pocket_diffusion");
    println!();
    println!("Modular research path:");
    println!("  pocket_diffusion research inspect --config <path>");
    println!("  pocket_diffusion research train --config <path> [--resume]");
    println!("  pocket_diffusion research experiment --config <path> [--resume]");
    println!();
    println!("Legacy compatibility path:");
    println!("  pocket_diffusion legacy-demo [num_candidates] [top_k]");
    println!();
    println!("Legacy flags kept for compatibility:");
    println!("  --inspect-config <path>");
    println!("  --train-config <path> [--resume]");
    println!("  --experiment-config <path> [--resume]");
    println!("  --phase1");
    println!("  --train-phase3");
    println!("  --phase4");
}

fn run_training_from_config(path: &str, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    let output = training::run_training_from_config(path, resume)?;
    training::print_training_run(&output);
    Ok(())
}

fn run_experiment_from_config(path: &str, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    let summary = experiments::run_experiment_from_config(path, resume)?;
    training::print_experiment_run(&summary);
    Ok(())
}

fn inspect_dataset_from_config(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let inspection = training::inspect_dataset_from_config(path)?;
    training::print_dataset_inspection(&inspection);
    Ok(())
}

fn value_after_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string()).collect()
    }

    #[test]
    fn parse_research_train_command_with_resume() {
        let command = parse_cli(&cli(&[
            "pocket_diffusion",
            "research",
            "train",
            "--config",
            "configs/research_manifest.json",
            "--resume",
        ]))
        .unwrap();

        match command {
            CliCommand::ResearchTrain {
                config,
                resume,
                origin,
            } => {
                assert_eq!(config, "configs/research_manifest.json");
                assert!(resume);
                assert_eq!(origin, CommandOrigin::Canonical);
            }
            _ => panic!("expected research train command"),
        }
    }

    #[test]
    fn parse_legacy_demo_command() {
        let command = parse_cli(&cli(&["pocket_diffusion", "legacy-demo", "12", "4"])).unwrap();

        match command {
            CliCommand::LegacyDemo {
                num_candidates,
                top_k,
            } => {
                assert_eq!(num_candidates, 12);
                assert_eq!(top_k, 4);
            }
            _ => panic!("expected legacy demo command"),
        }
    }

    #[test]
    fn parse_compatibility_train_flag() {
        let command = parse_cli(&cli(&[
            "pocket_diffusion",
            "--train-config",
            "configs/research_manifest.json",
        ]))
        .unwrap();

        match command {
            CliCommand::ResearchTrain {
                config,
                resume,
                origin,
            } => {
                assert_eq!(config, "configs/research_manifest.json");
                assert!(!resume);
                assert_eq!(origin, CommandOrigin::Compatibility);
            }
            _ => panic!("expected compatibility research train command"),
        }
    }

    #[test]
    fn parse_research_experiment_command() {
        let command = parse_cli(&cli(&[
            "pocket_diffusion",
            "research",
            "experiment",
            "--config",
            "configs/unseen_pocket_manifest.json",
        ]))
        .unwrap();

        match command {
            CliCommand::ResearchExperiment {
                config,
                resume,
                origin,
            } => {
                assert_eq!(config, "configs/unseen_pocket_manifest.json");
                assert!(!resume);
                assert_eq!(origin, CommandOrigin::Canonical);
            }
            _ => panic!("expected research experiment command"),
        }
    }

    #[test]
    fn compatibility_notice_is_emitted_for_legacy_paths() {
        let command = CliCommand::Phase3Demo;
        assert!(compatibility_notice(&command).is_some());

        let command = CliCommand::LegacyDemo {
            num_candidates: 10,
            top_k: 3,
        };
        assert!(compatibility_notice(&command).is_some());
    }

    #[test]
    fn compatibility_notice_tracks_command_origin() {
        let compatibility = parse_cli(&cli(&[
            "pocket_diffusion",
            "--inspect-config",
            "configs/research_manifest.json",
        ]))
        .unwrap();
        let notice = compatibility_notice(&compatibility).unwrap();
        assert!(notice.contains("research inspect --config configs/research_manifest.json"));

        let canonical = parse_cli(&cli(&[
            "pocket_diffusion",
            "research",
            "inspect",
            "--config",
            "configs/research_manifest.json",
        ]))
        .unwrap();
        assert!(compatibility_notice(&canonical).is_none());
    }

    #[test]
    fn compatibility_notice_preserves_resume_suffix() {
        let compatibility = parse_cli(&cli(&[
            "pocket_diffusion",
            "--train-config",
            "configs/research_manifest.json",
            "--resume",
        ]))
        .unwrap();
        let notice = compatibility_notice(&compatibility).unwrap();

        assert!(notice.contains("research train --config configs/research_manifest.json --resume"));
    }
}
