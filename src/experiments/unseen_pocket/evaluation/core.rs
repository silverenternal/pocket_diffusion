pub fn evaluate_split(
    system: &Phase1ResearchSystem,
    examples: &[crate::data::MolecularExample],
    train_examples: &[crate::data::MolecularExample],
    train_proteins: &std::collections::BTreeSet<&str>,
    research: &ResearchConfig,
    ablation: AblationConfig,
    external_evaluation: &ExternalEvaluationConfig,
    split_label: &str,
    device: tch::Device,
) -> EvaluationMetrics {
    let start = Instant::now();
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();
    let memory_before = sys.used_memory() as f64 / (1024.0 * 1024.0);

    if examples.is_empty() {
        return EvaluationMetrics {
            representation_diagnostics: RepresentationDiagnostics {
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
            proxy_task_metrics: ProxyTaskMetrics {
                affinity_probe_mae: 0.0,
                affinity_probe_rmse: 0.0,
                labeled_fraction: 0.0,
                affinity_by_measurement: Vec::new(),
            },
            split_context: SplitContextMetrics {
                example_count: 0,
                unique_complex_count: 0,
                unique_protein_count: 0,
                train_reference_protein_count: train_proteins.len(),
                ligand_atom_count_bins: BTreeMap::new(),
                pocket_atom_count_bins: BTreeMap::new(),
                measurement_family_histogram: BTreeMap::new(),
            },
            resource_usage: ResourceUsageMetrics {
                memory_usage_mb: memory_before,
                evaluation_time_ms: 0.0,
                examples_per_second: 0.0,
                average_ligand_atoms: 0.0,
                average_pocket_atoms: 0.0,
            },
            real_generation_metrics: disabled_real_generation_metrics(),
            layered_generation_metrics: empty_layered_generation_metrics(),
            method_comparison: MethodComparisonSummary::default(),
            comparison_summary: GenerationQualitySummary {
                primary_objective: primary_objective_label(research.training.primary_objective),
                variant_label: ablation.variant_label.clone(),
                interaction_mode: interaction_mode_label(
                    ablation
                        .interaction_mode_override
                        .unwrap_or(research.model.interaction_mode),
                ),
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
            slot_stability: SlotStabilityMetrics::default(),
            strata: Vec::new(),
        };
    }

    let forwards: Vec<ResearchForward> = examples
        .iter()
        .map(|example| system.forward_example(&example.to_device(device)))
        .collect();

    sys.refresh_memory();
    let memory_after = sys.used_memory() as f64 / (1024.0 * 1024.0);

    let finite_forward_fraction = forwards
        .iter()
        .filter(|forward| {
            tensor_is_finite(&forward.encodings.topology.pooled_embedding)
                && tensor_is_finite(&forward.encodings.geometry.pooled_embedding)
                && tensor_is_finite(&forward.encodings.pocket.pooled_embedding)
        })
        .count() as f64
        / examples.len() as f64;

    let unique_ids = examples
        .iter()
        .map(|example| format!("{}::{}", example.protein_id, example.example_id))
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64;
    let unique_complex_fraction = unique_ids / examples.len() as f64;

    let unseen_protein_fraction = examples
        .iter()
        .filter(|example| !train_proteins.contains(example.protein_id.as_str()))
        .count() as f64
        / examples.len() as f64;

    let distance_probe_rmse = (examples
        .iter()
        .zip(forwards.iter())
        .map(|(example, forward)| {
            let target = example
                .geometry
                .pairwise_distances
                .mean(tch::Kind::Float)
                .double_value(&[]);
            let pred = forward
                .probes
                .geometry_distance_predictions
                .mean(tch::Kind::Float)
                .double_value(&[]);
            let error = pred - target;
            error * error
        })
        .sum::<f64>()
        / examples.len() as f64)
        .sqrt();

    let topology_pocket_cosine_alignment = forwards
        .iter()
        .map(|forward| {
            let topo = &forward.encodings.topology.pooled_embedding;
            let pocket = &forward.encodings.pocket.pooled_embedding;
            (topo * pocket).sum(tch::Kind::Float).double_value(&[])
                / (topo.norm().double_value(&[]) * pocket.norm().double_value(&[])).max(1e-6)
        })
        .sum::<f64>()
        / examples.len() as f64;

    let labeled_examples: Vec<(&crate::data::MolecularExample, &ResearchForward)> = examples
        .iter()
        .zip(forwards.iter())
        .filter(|(example, _)| example.targets.affinity_kcal_mol.is_some())
        .collect();
    let labeled_fraction = labeled_examples.len() as f64 / examples.len() as f64;
    let affinity_probe_mae = if labeled_examples.is_empty() {
        0.0
    } else {
        labeled_examples
            .iter()
            .map(|(example, forward)| {
                let target = example.targets.affinity_kcal_mol.unwrap() as f64;
                let pred = forward.probes.affinity_prediction.double_value(&[]);
                (pred - target).abs()
            })
            .sum::<f64>()
            / labeled_examples.len() as f64
    };
    let affinity_probe_rmse = if labeled_examples.is_empty() {
        0.0
    } else {
        (labeled_examples
            .iter()
            .map(|(example, forward)| {
                let target = example.targets.affinity_kcal_mol.unwrap() as f64;
                let pred = forward.probes.affinity_prediction.double_value(&[]);
                let error = pred - target;
                error * error
            })
            .sum::<f64>()
            / labeled_examples.len() as f64)
            .sqrt()
    };
    let affinity_by_measurement = measurement_breakdown(&labeled_examples);

    let topology_reconstruction_mse = forwards
        .iter()
        .map(|forward| {
            (forward.slots.topology.reconstructed_tokens.shallow_clone()
                - forward.encodings.topology.token_embeddings.shallow_clone())
            .pow_tensor_scalar(2.0)
            .mean(tch::Kind::Float)
            .double_value(&[])
        })
        .sum::<f64>()
        / examples.len() as f64;
    let topology_specialization_score = forwards
        .iter()
        .map(|forward| {
            if forward.probes.topology_adjacency_logits.numel() == 0 {
                0.0
            } else {
                forward
                    .probes
                    .topology_adjacency_logits
                    .sigmoid()
                    .mean(tch::Kind::Float)
                    .double_value(&[])
            }
        })
        .sum::<f64>()
        / examples.len() as f64;
    let geometry_specialization_score = 1.0 / (1.0 + distance_probe_rmse);
    let pocket_feature_rmse = (examples
        .iter()
        .zip(forwards.iter())
        .map(|(example, forward)| {
            let predicted = &forward.probes.pocket_feature_predictions;
            let target = &example.pocket.atom_features;
            if predicted.numel() == 0 || target.numel() == 0 {
                0.0
            } else {
                (predicted - target)
                    .pow_tensor_scalar(2.0)
                    .mean(tch::Kind::Float)
                    .double_value(&[])
            }
        })
        .sum::<f64>()
        / examples.len() as f64)
        .sqrt();
    let pocket_specialization_score = 1.0 / (1.0 + pocket_feature_rmse);

    let slot_activation_mean = if ablation.disable_slots {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let slot_means = [
                    active_slot_fraction(&forward.slots.topology.slot_weights),
                    active_slot_fraction(&forward.slots.geometry.slot_weights),
                    active_slot_fraction(&forward.slots.pocket.slot_weights),
                ];
                slot_means.iter().sum::<f64>() / slot_means.len() as f64
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let gate_activation_mean = if ablation.disable_cross_attention {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                [
                    forward.interactions.topo_from_geo.gate.double_value(&[0]),
                    forward
                        .interactions
                        .topo_from_pocket
                        .gate
                        .double_value(&[0]),
                    forward.interactions.geo_from_topo.gate.double_value(&[0]),
                    forward.interactions.geo_from_pocket.gate.double_value(&[0]),
                    forward
                        .interactions
                        .pocket_from_topo
                        .gate
                        .double_value(&[0]),
                    forward.interactions.pocket_from_geo.gate.double_value(&[0]),
                ]
                .iter()
                .sum::<f64>()
                    / 6.0
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let leakage_proxy_mean = if ablation.disable_leakage {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let topo = mean_slot(&forward.slots.topology.slots);
                let geo = mean_slot(&forward.slots.geometry.slots);
                let pocket = mean_slot(&forward.slots.pocket.slots);
                cosine_similarity(&topo, &geo).abs()
                    + cosine_similarity(&topo, &pocket).abs()
                    + cosine_similarity(&geo, &pocket).abs()
            })
            .sum::<f64>()
            / (examples.len() as f64 * 3.0)
    };
    let slot_stability = if ablation.disable_slots {
        SlotStabilityMetrics::default()
    } else {
        compute_slot_stability(&forwards)
    };

    let unique_protein_count = examples
        .iter()
        .map(|example| example.protein_id.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    let (real_generation_metrics, layered_generation_metrics, method_comparison) =
        evaluate_real_generation_metrics(
            examples,
            train_examples,
            &forwards,
            research,
            &ablation,
            external_evaluation,
            split_label,
        );
    let evaluation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (ligand_atom_count_bins, pocket_atom_count_bins, measurement_family_histogram) =
        split_histograms(examples);
    let average_ligand_atoms = average_ligand_atoms(examples);
    let average_pocket_atoms = average_pocket_atoms(examples);

    EvaluationMetrics {
        representation_diagnostics: RepresentationDiagnostics {
            finite_forward_fraction,
            unique_complex_fraction,
            unseen_protein_fraction,
            distance_probe_rmse,
            topology_pocket_cosine_alignment,
            topology_reconstruction_mse,
            slot_activation_mean,
            gate_activation_mean,
            leakage_proxy_mean,
        },
        proxy_task_metrics: ProxyTaskMetrics {
            affinity_probe_mae,
            affinity_probe_rmse,
            labeled_fraction,
            affinity_by_measurement,
        },
        split_context: SplitContextMetrics {
            example_count: examples.len(),
            unique_complex_count: unique_ids as usize,
            unique_protein_count,
            train_reference_protein_count: train_proteins.len(),
            ligand_atom_count_bins,
            pocket_atom_count_bins,
            measurement_family_histogram,
        },
        resource_usage: ResourceUsageMetrics {
            memory_usage_mb: (memory_after - memory_before).max(0.0),
            evaluation_time_ms,
            examples_per_second: if evaluation_time_ms > 0.0 {
                examples.len() as f64 / (evaluation_time_ms / 1000.0)
            } else {
                0.0
            },
            average_ligand_atoms,
            average_pocket_atoms,
        },
        real_generation_metrics: real_generation_metrics.clone(),
        layered_generation_metrics,
        method_comparison: method_comparison.clone(),
        comparison_summary: build_comparison_summary(
            research,
            &ablation,
            unseen_protein_fraction,
            topology_specialization_score,
            geometry_specialization_score,
            pocket_specialization_score,
            slot_activation_mean,
            gate_activation_mean,
            leakage_proxy_mean,
            &real_generation_metrics,
        ),
        slot_stability,
        strata: build_stratum_metrics(examples, train_proteins),
    }
}

