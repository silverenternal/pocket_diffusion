//! Main binary for the modular research stack and legacy compatibility demos.

use clap::{Args, Parser, Subcommand, ValueEnum};

use pocket_diffusion::{
    config, experiments, legacy,
    training::{self, RunArtifactBundle, RunKind},
};

include!("cli.rs");

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
            raw_flow: experiments::CandidateLayerMetrics::default(),
            constrained_flow: experiments::CandidateLayerMetrics::default(),
            repaired: experiments::CandidateLayerMetrics::default(),
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
