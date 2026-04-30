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
    let evaluation_batch_size = research.data.batch_size.max(1);

    if examples.is_empty() {
        let mut layered_generation_metrics = empty_layered_generation_metrics();
        layered_generation_metrics.repair_case_audit = RepairCaseAuditReport {
            split_label: split_label.to_string(),
            no_repair_ablation: NoRepairAblationMetrics {
                repair_enabled: false,
                no_repair_layer: "raw_rollout".to_string(),
                no_repair_metrics: CandidateLayerMetrics::default(),
                interpretation:
                    "empty split still reserves raw_rollout as the no-repair baseline layer"
                        .to_string(),
            },
            artifact_name: Some(format!("repair_case_audit_{split_label}.json")),
            claim_boundary:
                "empty split repair audit; repaired layers remain postprocessing evidence and must not be cited as raw generation evidence"
                    .to_string(),
            ..RepairCaseAuditReport::default()
        };
        return EvaluationMetrics {
            representation_diagnostics: RepresentationDiagnostics {
                finite_forward_fraction: 0.0,
                unique_complex_fraction: 0.0,
                unseen_protein_fraction: 0.0,
                distance_probe_rmse: 0.0,
                topology_pocket_cosine_alignment: 0.0,
                topology_reconstruction_mse: 0.0,
                slot_activation_mean: 0.0,
                slot_assignment_entropy_mean: 0.0,
                slot_activation_probability_mean: 0.0,
                attention_visible_slot_fraction: 0.0,
                gate_activation_mean: 0.0,
                leakage_proxy_mean: 0.0,
            },
            proxy_task_metrics: ProxyTaskMetrics {
                affinity_probe_mae: 0.0,
                affinity_probe_rmse: 0.0,
                labeled_fraction: 0.0,
                affinity_by_measurement: Vec::new(),
                probe_baselines: Vec::new(),
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
                memory_usage_mb: 0.0,
                evaluation_time_ms: 0.0,
                examples_per_second: 0.0,
                evaluation_batch_size,
                forward_batch_count: 0,
                per_example_forward_count: 0,
                no_grad: true,
                batched_forward: true,
                de_novo_per_example_reason: None,
                average_ligand_atoms: 0.0,
                average_pocket_atoms: 0.0,
            },
            model_design: ModelDesignEvaluationMetrics::default(),
            real_generation_metrics: disabled_real_generation_metrics(),
            layered_generation_metrics,
            method_comparison: MethodComparisonSummary::default(),
            train_eval_alignment: TrainEvalAlignmentReport::default(),
            chemistry_collaboration: ChemistryCollaborationMetrics::default(),
            frozen_leakage_probe_calibration: FrozenLeakageProbeCalibrationReport {
                calibration_status: "insufficient_data".to_string(),
                split_name: split_label.to_string(),
                training_time_signal:
                    "training-time leakage penalty/proxy is not a held-out frozen-probe estimate"
                        .to_string(),
                representation_source: "frozen forward encodings".to_string(),
                optimizer_penalty_separated: true,
                claim_boundary: "empty split has no held-out frozen-probe audit examples"
                    .to_string(),
                ..FrozenLeakageProbeCalibrationReport::default()
            },
            comparison_summary: GenerationQualitySummary {
                generation_mode: generation_mode_label(research),
                primary_objective: primary_objective_label(research.training.primary_objective),
                primary_objective_provenance: primary_objective_provenance_label(
                    research.training.primary_objective,
                ),
                primary_objective_claim_boundary: primary_objective_claim_boundary_label(
                    research.training.primary_objective,
                ),
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
                chemistry_collaboration: ChemistryCollaborationMetrics::default(),
            },
            slot_stability: SlotStabilityMetrics::default(),
            strata: Vec::new(),
        };
    }

    let (forwards, forward_batch_count) =
        evaluate_forwards_no_grad_batched(system, examples, device, evaluation_batch_size);
    let frozen_leakage_probe_calibration = calibrate_frozen_leakage_probes_for_split(
        examples,
        &forwards,
        split_label,
        &FrozenLeakageProbeCalibrationConfig::default(),
    );

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

    let distance_probe_rmse = geometry_distance_mse(examples, &forwards)
        .map(f64::sqrt)
        .unwrap_or(0.0);

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
    let probe_baselines = compute_probe_baselines(examples, &forwards);

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
                    active_slot_fraction(&forward.slots.topology.slot_activations),
                    active_slot_fraction(&forward.slots.geometry.slot_activations),
                    active_slot_fraction(&forward.slots.pocket.slot_activations),
                ];
                slot_means.iter().sum::<f64>() / slot_means.len() as f64
            })
            .sum::<f64>()
            / examples.len() as f64
    };
    let slot_assignment_entropy_mean = if ablation.disable_slots {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let entropies = [
                    slot_assignment_entropy(&forward.slots.topology.slot_weights),
                    slot_assignment_entropy(&forward.slots.geometry.slot_weights),
                    slot_assignment_entropy(&forward.slots.pocket.slot_weights),
                ];
                entropies.iter().sum::<f64>() / entropies.len() as f64
            })
            .sum::<f64>()
            / examples.len() as f64
    };
    let slot_activation_probability_mean = if ablation.disable_slots {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let activations = [
                    slot_activation_probability(&forward.slots.topology.slot_activations),
                    slot_activation_probability(&forward.slots.geometry.slot_activations),
                    slot_activation_probability(&forward.slots.pocket.slot_activations),
                ];
                activations.iter().sum::<f64>() / activations.len() as f64
            })
            .sum::<f64>()
            / examples.len() as f64
    };
    let attention_visible_slot_fraction = if ablation.disable_slots {
        0.0
    } else {
        forwards
            .iter()
            .map(|forward| {
                let visible = [
                    attention_visible_slot_fraction_for_mask(
                        &forward.slots.topology.active_slot_mask,
                    ),
                    attention_visible_slot_fraction_for_mask(
                        &forward.slots.geometry.active_slot_mask,
                    ),
                    attention_visible_slot_fraction_for_mask(&forward.slots.pocket.active_slot_mask),
                ];
                visible.iter().sum::<f64>() / visible.len() as f64
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let gate_activation_mean = if ablation.disable_cross_attention {
        0.0
    } else {
        forwards.iter().map(mean_cross_modal_gate_activation).sum::<f64>()
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
        compute_slot_stability(examples, &forwards)
    };

    let unique_protein_count = examples
        .iter()
        .map(|example| example.protein_id.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    let (real_generation_metrics, mut layered_generation_metrics, mut method_comparison) =
        evaluate_real_generation_metrics(
            examples,
            train_examples,
            &forwards,
            research,
            &ablation,
            external_evaluation,
            split_label,
        );
    let chemistry_collaboration =
        compute_chemistry_collaboration_metrics(examples, &forwards, &layered_generation_metrics);
    let evaluation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let examples_per_second = if evaluation_time_ms > 0.0 {
        examples.len() as f64 / (evaluation_time_ms / 1000.0)
    } else {
        0.0
    };
    let memory_usage_mb = (memory_after - memory_before).max(0.0);
    let (ligand_atom_count_bins, pocket_atom_count_bins, measurement_family_histogram) =
        split_histograms(examples);
    let average_ligand_atoms = average_ligand_atoms(examples);
    let average_pocket_atoms = average_pocket_atoms(examples);
    let model_design = build_model_design_evaluation_metrics(
        finite_forward_fraction,
        unseen_protein_fraction,
        topology_reconstruction_mse,
        distance_probe_rmse,
        slot_activation_mean,
        gate_activation_mean,
        leakage_proxy_mean,
        &slot_stability,
        &forwards,
        &layered_generation_metrics,
        examples_per_second,
        memory_usage_mb,
    );
    synchronize_method_comparison_evidence(
        &mut layered_generation_metrics,
        &mut method_comparison,
        &model_design,
    );
    let comparison_summary = build_comparison_summary(
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
        &chemistry_collaboration,
    );
    let train_eval_alignment = build_train_eval_alignment_report(
        research,
        &real_generation_metrics,
        &method_comparison,
        &model_design,
        finite_forward_fraction,
        distance_probe_rmse,
        leakage_proxy_mean,
        affinity_probe_mae,
        labeled_fraction,
        examples_per_second,
        &comparison_summary,
    );

    let per_example_forward_count = if evaluation_batch_size == 1 {
        examples.len()
    } else {
        0
    };
    let de_novo_per_example_reason = if per_example_forward_count > 0
        && research
            .data
            .generation_target
            .generation_mode
            == crate::config::GenerationModeConfig::DeNovoInitialization
    {
        Some(
            "batch_size_one_de_novo_eval_keeps_per-example conditioning and target-supervision boundaries explicit"
                .to_string(),
        )
    } else {
        None
    };

    EvaluationMetrics {
        representation_diagnostics: RepresentationDiagnostics {
            finite_forward_fraction,
            unique_complex_fraction,
            unseen_protein_fraction,
            distance_probe_rmse,
            topology_pocket_cosine_alignment,
            topology_reconstruction_mse,
            slot_activation_mean,
            slot_assignment_entropy_mean,
            slot_activation_probability_mean,
            attention_visible_slot_fraction,
            gate_activation_mean,
            leakage_proxy_mean,
        },
        proxy_task_metrics: ProxyTaskMetrics {
            affinity_probe_mae,
            affinity_probe_rmse,
            labeled_fraction,
            affinity_by_measurement,
            probe_baselines,
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
            memory_usage_mb,
            evaluation_time_ms,
            examples_per_second,
            evaluation_batch_size,
            forward_batch_count,
            per_example_forward_count,
            no_grad: true,
            batched_forward: true,
            de_novo_per_example_reason,
            average_ligand_atoms,
            average_pocket_atoms,
        },
        model_design,
        real_generation_metrics: real_generation_metrics.clone(),
        layered_generation_metrics,
        method_comparison: method_comparison.clone(),
        train_eval_alignment,
        chemistry_collaboration: chemistry_collaboration.clone(),
        frozen_leakage_probe_calibration,
        comparison_summary,
        slot_stability,
        strata: build_stratum_metrics(examples, train_proteins),
    }
}

fn evaluate_forwards_no_grad_batched(
    system: &Phase1ResearchSystem,
    examples: &[crate::data::MolecularExample],
    device: tch::Device,
    evaluation_batch_size: usize,
) -> (Vec<ResearchForward>, usize) {
    let batch_size = evaluation_batch_size.max(1);
    let mut forward_batch_count = 0usize;
    let forwards = no_grad(|| {
        let mut outputs = Vec::with_capacity(examples.len());
        for chunk in examples.chunks(batch_size) {
            forward_batch_count += 1;
            let device_examples = chunk
                .iter()
                .map(|example| example.to_device(device))
                .collect::<Vec<_>>();
            let (_, mut chunk_outputs) = system.forward_batch(&device_examples);
            outputs.append(&mut chunk_outputs);
        }
        outputs
    });
    (forwards, forward_batch_count)
}

fn compute_probe_baselines(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Vec<ProbeBaselineMetric> {
    let mut rows = Vec::with_capacity(9);
    let topology_binary_audit = topology_adjacency_binary_audit(examples, forwards);
    let ligand_role_binary_audit = role_probe_binary_audit(
        examples,
        forwards,
        |example| &example.topology.chemistry_roles,
        |forward| &forward.probes.ligand_pharmacophore_role_logits,
    );
    let pocket_role_binary_audit = role_probe_binary_audit(
        examples,
        forwards,
        |example| &example.pocket.chemistry_roles,
        |forward| &forward.probes.pocket_pharmacophore_role_logits,
    );
    let topology_row = probe_baseline_row(
        "topology_adjacency",
        "binary_cross_entropy",
        topology_adjacency_bce(examples, forwards),
        topology_adjacency_trivial_bce(examples),
        topology_adjacency_available_count(examples),
    );
    rows.push(with_binary_probe_audit(topology_row, topology_binary_audit));
    let topology_balanced_row = probe_baseline_row(
        "topology_adjacency_balanced",
        "balanced_binary_cross_entropy",
        topology_adjacency_balanced_bce(examples, forwards),
        topology_adjacency_balanced_trivial_bce(examples),
        topology_adjacency_available_count(examples),
    );
    rows.push(with_binary_probe_audit(
        topology_balanced_row,
        topology_binary_audit,
    ));
    rows.push(probe_baseline_row(
        "geometry_mean_pairwise_distance",
        "mean_squared_error",
        geometry_distance_mse(examples, forwards),
        geometry_distance_trivial_mse(examples),
        geometry_distance_available_count(examples),
    ));
    rows.push(probe_baseline_row(
        "pocket_atom_features",
        "mean_squared_error",
        pocket_feature_mse(examples, forwards),
        pocket_feature_trivial_mse(examples),
        pocket_feature_available_count(examples),
    ));
    let ligand_role_row = probe_baseline_row(
        "ligand_pharmacophore_roles",
        "binary_cross_entropy",
        ligand_pharmacophore_role_bce(examples, forwards),
        ligand_pharmacophore_role_trivial_bce(examples),
        ligand_pharmacophore_available_count(examples),
    );
    rows.push(with_binary_probe_audit(
        ligand_role_row,
        ligand_role_binary_audit,
    ));
    let ligand_role_balanced_row = probe_baseline_row(
        "ligand_pharmacophore_roles_balanced",
        "balanced_binary_cross_entropy",
        ligand_pharmacophore_role_balanced_bce(examples, forwards),
        ligand_pharmacophore_role_balanced_trivial_bce(examples),
        ligand_pharmacophore_available_count(examples),
    );
    rows.push(with_binary_probe_audit(
        ligand_role_balanced_row,
        ligand_role_binary_audit,
    ));
    let pocket_role_row = probe_baseline_row(
        "pocket_pharmacophore_roles",
        "binary_cross_entropy",
        pocket_pharmacophore_role_bce(examples, forwards),
        pocket_pharmacophore_role_trivial_bce(examples),
        pocket_pharmacophore_available_count(examples),
    );
    rows.push(with_binary_probe_audit(
        pocket_role_row,
        pocket_role_binary_audit,
    ));
    let pocket_role_balanced_row = probe_baseline_row(
        "pocket_pharmacophore_roles_balanced",
        "balanced_binary_cross_entropy",
        pocket_pharmacophore_role_balanced_bce(examples, forwards),
        pocket_pharmacophore_role_balanced_trivial_bce(examples),
        pocket_pharmacophore_available_count(examples),
    );
    rows.push(with_binary_probe_audit(
        pocket_role_balanced_row,
        pocket_role_binary_audit,
    ));
    rows.push(with_scalar_probe_audit(
        probe_baseline_row(
            "affinity_scalar",
            "mean_squared_error",
            affinity_probe_mse(examples, forwards),
            affinity_trivial_mse(examples),
            affinity_available_count(examples),
        ),
        affinity_scalar_audit(examples, forwards),
    ));
    rows
}

fn probe_baseline_row(
    target: &str,
    loss_kind: &str,
    observed_loss: Option<f64>,
    trivial_baseline_loss: Option<f64>,
    available_count: usize,
) -> ProbeBaselineMetric {
    let improves_over_trivial = observed_loss
        .zip(trivial_baseline_loss)
        .map(|(observed, baseline)| observed < baseline);
    let supervision_status = if available_count == 0 {
        "unavailable"
    } else if observed_loss.is_some() && trivial_baseline_loss.is_some() {
        "available"
    } else {
        "target_available_prediction_unavailable"
    };
    let interpretation = match improves_over_trivial {
        Some(true) => "probe improves on a target-only trivial baseline".to_string(),
        Some(false) => {
            "probe does not beat the target-only trivial baseline; leakage/specialization conclusions remain capacity-limited".to_string()
        }
        None => "baseline comparison unavailable because the split lacks this target".to_string(),
    };
    ProbeBaselineMetric {
        target: target.to_string(),
        loss_kind: loss_kind.to_string(),
        observed_loss,
        trivial_baseline_loss,
        improves_over_trivial,
        supervision_status: supervision_status.to_string(),
        available_count,
        interpretation,
        ..ProbeBaselineMetric::default()
    }
}

#[derive(Debug, Clone, Copy)]
struct BinaryProbeAudit {
    target_positive_rate: f64,
    prediction_positive_rate: f64,
    positive_rate_gap: f64,
    positive_observed_loss: Option<f64>,
    negative_observed_loss: Option<f64>,
}

fn with_binary_probe_audit(
    mut row: ProbeBaselineMetric,
    audit: Option<BinaryProbeAudit>,
) -> ProbeBaselineMetric {
    if let Some(audit) = audit {
        row.target_positive_rate = Some(audit.target_positive_rate);
        row.prediction_positive_rate = Some(audit.prediction_positive_rate);
        row.positive_rate_gap = Some(audit.positive_rate_gap);
        row.positive_observed_loss = audit.positive_observed_loss;
        row.negative_observed_loss = audit.negative_observed_loss;
    }
    row
}

#[derive(Debug, Clone, Copy)]
struct ScalarProbeAudit {
    target_mean: f64,
    prediction_mean: f64,
    mean_error: f64,
}

fn with_scalar_probe_audit(
    mut row: ProbeBaselineMetric,
    audit: Option<ScalarProbeAudit>,
) -> ProbeBaselineMetric {
    if let Some(audit) = audit {
        row.scalar_target_mean = Some(audit.target_mean);
        row.scalar_prediction_mean = Some(audit.prediction_mean);
        row.scalar_mean_error = Some(audit.mean_error);
    }
    row
}

#[derive(Debug, Default)]
struct BinaryProbeAuditAccumulator {
    target_positive_sum: f64,
    prediction_positive_sum: f64,
    total_count: f64,
    positive_loss_sum: f64,
    positive_count: f64,
    negative_loss_sum: f64,
    negative_count: f64,
}

impl BinaryProbeAuditAccumulator {
    fn add(&mut self, logits: &tch::Tensor, target: &tch::Tensor, mask: &tch::Tensor) {
        if logits.numel() == 0 {
            return;
        }
        let logits = logits.to_kind(tch::Kind::Float);
        let target = target
            .to_device(logits.device())
            .to_kind(tch::Kind::Float)
            .clamp(0.0, 1.0);
        let mask = mask
            .to_device(logits.device())
            .to_kind(tch::Kind::Float)
            .clamp(0.0, 1.0);
        if logits.size() != target.size() || logits.size() != mask.size() {
            return;
        }
        let prediction = logits.sigmoid();
        let positive_mask = &target * &mask;
        let negative_mask = (tch::Tensor::ones_like(&target) - &target) * &mask;
        let loss =
            logits.clamp_min(0.0) - (&logits * &target) + (-logits.abs()).exp().log1p();
        self.total_count += mask.sum(tch::Kind::Float).double_value(&[]);
        self.target_positive_sum += (&target * &mask).sum(tch::Kind::Float).double_value(&[]);
        self.prediction_positive_sum += (&prediction * &mask)
            .sum(tch::Kind::Float)
            .double_value(&[]);
        self.positive_count += positive_mask.sum(tch::Kind::Float).double_value(&[]);
        self.negative_count += negative_mask.sum(tch::Kind::Float).double_value(&[]);
        self.positive_loss_sum += (&loss * &positive_mask)
            .sum(tch::Kind::Float)
            .double_value(&[]);
        self.negative_loss_sum += (&loss * &negative_mask)
            .sum(tch::Kind::Float)
            .double_value(&[]);
    }

    fn finish(self) -> Option<BinaryProbeAudit> {
        if self.total_count <= 0.0 {
            return None;
        }
        let target_positive_rate = self.target_positive_sum / self.total_count;
        let prediction_positive_rate = self.prediction_positive_sum / self.total_count;
        Some(BinaryProbeAudit {
            target_positive_rate,
            prediction_positive_rate,
            positive_rate_gap: (prediction_positive_rate - target_positive_rate).abs(),
            positive_observed_loss: (self.positive_count > 0.0)
                .then_some(self.positive_loss_sum / self.positive_count),
            negative_observed_loss: (self.negative_count > 0.0)
                .then_some(self.negative_loss_sum / self.negative_count),
        })
    }
}

fn slot_assignment_entropy(weights: &tch::Tensor) -> f64 {
    if weights.numel() == 0 {
        return 0.0;
    }
    let normalized = (weights / weights.sum(tch::Kind::Float).clamp_min(1e-12)).clamp_min(1e-12);
    (-(&normalized * normalized.log()).sum(tch::Kind::Float)).double_value(&[])
}

fn slot_activation_probability(activations: &tch::Tensor) -> f64 {
    if activations.numel() == 0 {
        return 0.0;
    }
    activations.mean(tch::Kind::Float).double_value(&[])
}

fn mean_cross_modal_gate_activation(forward: &ResearchForward) -> f64 {
    let gates = [
        gate_tensor_mean(&forward.interactions.topo_from_geo.gate),
        gate_tensor_mean(&forward.interactions.topo_from_pocket.gate),
        gate_tensor_mean(&forward.interactions.geo_from_topo.gate),
        gate_tensor_mean(&forward.interactions.geo_from_pocket.gate),
        gate_tensor_mean(&forward.interactions.pocket_from_topo.gate),
        gate_tensor_mean(&forward.interactions.pocket_from_geo.gate),
    ];
    gates.iter().sum::<f64>() / gates.len() as f64
}

fn gate_tensor_mean(gate: &tch::Tensor) -> f64 {
    if gate.numel() == 0 {
        0.0
    } else {
        gate.mean(tch::Kind::Float).double_value(&[])
    }
}

fn attention_visible_slot_fraction_for_mask(mask: &tch::Tensor) -> f64 {
    if mask.numel() == 0 {
        return 0.0;
    }
    mask.to_kind(tch::Kind::Float)
        .mean(tch::Kind::Float)
        .double_value(&[])
}

fn topology_adjacency_available_count(examples: &[crate::data::MolecularExample]) -> usize {
    examples
        .iter()
        .filter(|example| example.topology.adjacency.numel() > 0)
        .count()
}

fn topology_adjacency_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    mean_available(
        examples
            .iter()
            .zip(forwards.iter())
            .filter_map(|(example, forward)| {
                if example.topology.adjacency.numel() == 0
                    || forward.probes.topology_adjacency_logits.numel() == 0
                    || forward.probes.topology_adjacency_logits.size()
                        != example.topology.adjacency.size()
                {
                    return None;
                }
                Some(
                    forward
                        .probes
                        .topology_adjacency_logits
                        .binary_cross_entropy_with_logits::<tch::Tensor>(
                            &example.topology.adjacency,
                            None,
                            None,
                            tch::Reduction::Mean,
                        )
                        .double_value(&[]),
                )
            }),
    )
}

fn topology_adjacency_balanced_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    mean_available(
        examples
            .iter()
            .zip(forwards.iter())
            .filter_map(|(example, forward)| {
                if example.topology.adjacency.numel() == 0
                    || forward.probes.topology_adjacency_logits.numel() == 0
                    || forward.probes.topology_adjacency_logits.size()
                        != example.topology.adjacency.size()
                {
                    return None;
                }
                let mask = tch::Tensor::ones_like(&example.topology.adjacency);
                Some(
                    crate::losses::classification::masked_balanced_bce_with_logits(
                        &forward.probes.topology_adjacency_logits,
                        &example.topology.adjacency,
                        &mask,
                    )
                    .double_value(&[]),
                )
            }),
    )
}

fn topology_adjacency_trivial_bce(examples: &[crate::data::MolecularExample]) -> Option<f64> {
    let values = examples
        .iter()
        .filter(|example| example.topology.adjacency.numel() > 0)
        .map(|example| {
            (
                example
                    .topology
                    .adjacency
                    .sum(tch::Kind::Float)
                    .double_value(&[]),
                example.topology.adjacency.numel() as f64,
            )
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let positive = values.iter().map(|(sum, _)| *sum).sum::<f64>();
    let count = values.iter().map(|(_, count)| *count).sum::<f64>();
    let p = (positive / count.max(1.0)).clamp(1e-6, 1.0 - 1e-6);
    Some(-(p * p.ln() + (1.0 - p) * (1.0 - p).ln()))
}

fn topology_adjacency_balanced_trivial_bce(
    examples: &[crate::data::MolecularExample],
) -> Option<f64> {
    let (positive, negative) = topology_adjacency_binary_counts(examples)?;
    balanced_binary_trivial_bce(positive, negative)
}

fn topology_adjacency_binary_counts(
    examples: &[crate::data::MolecularExample],
) -> Option<(f64, f64)> {
    let mut positive = 0.0;
    let mut negative = 0.0;
    for example in examples {
        if example.topology.adjacency.numel() == 0 {
            continue;
        }
        let target = example.topology.adjacency.clamp(0.0, 1.0);
        let pos = target.sum(tch::Kind::Float).double_value(&[]);
        positive += pos;
        negative += target.numel() as f64 - pos;
    }
    ((positive + negative) > 0.0).then_some((positive, negative))
}

fn topology_adjacency_binary_audit(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<BinaryProbeAudit> {
    let mut accumulator = BinaryProbeAuditAccumulator::default();
    for (example, forward) in examples.iter().zip(forwards.iter()) {
        let logits = &forward.probes.topology_adjacency_logits;
        if example.topology.adjacency.numel() == 0
            || logits.numel() == 0
            || logits.size() != example.topology.adjacency.size()
        {
            continue;
        }
        let mask = tch::Tensor::ones_like(logits);
        accumulator.add(logits, &example.topology.adjacency, &mask);
    }
    accumulator.finish()
}

fn geometry_distance_available_count(examples: &[crate::data::MolecularExample]) -> usize {
    examples
        .iter()
        .filter(|example| example.geometry.pairwise_distances.numel() > 0)
        .count()
}

fn geometry_distance_mse(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    mean_available(
        examples
            .iter()
            .zip(forwards.iter())
            .filter_map(|(example, forward)| {
                if example.geometry.pairwise_distances.numel() == 0
                    || forward.probes.geometry_distance_predictions.numel() == 0
                {
                    return None;
                }
                let target = example.geometry.pairwise_distances.mean_dim(
                    [1].as_slice(),
                    false,
                    tch::Kind::Float,
                );
                if forward.probes.geometry_distance_predictions.size() != target.size() {
                    return None;
                }
                Some(
                    (forward.probes.geometry_distance_predictions.shallow_clone() - target)
                        .pow_tensor_scalar(2.0)
                        .mean(tch::Kind::Float)
                        .double_value(&[]),
                )
            }),
    )
}

fn geometry_distance_trivial_mse(examples: &[crate::data::MolecularExample]) -> Option<f64> {
    let targets = examples
        .iter()
        .filter(|example| example.geometry.pairwise_distances.numel() > 0)
        .map(|example| {
            example
                .geometry
                .pairwise_distances
                .mean_dim([1].as_slice(), false, tch::Kind::Float)
        })
        .collect::<Vec<_>>();
    tensor_trivial_mse(targets)
}

fn pocket_feature_available_count(examples: &[crate::data::MolecularExample]) -> usize {
    examples
        .iter()
        .filter(|example| example.pocket.atom_features.numel() > 0)
        .count()
}

fn pocket_feature_mse(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    mean_available(
        examples
            .iter()
            .zip(forwards.iter())
            .filter_map(|(example, forward)| {
                if example.pocket.atom_features.numel() == 0
                    || forward.probes.pocket_feature_predictions.numel() == 0
                    || forward.probes.pocket_feature_predictions.size()
                        != example.pocket.atom_features.size()
                {
                    return None;
                }
                Some(
                    (forward.probes.pocket_feature_predictions.shallow_clone()
                        - example.pocket.atom_features.shallow_clone())
                    .pow_tensor_scalar(2.0)
                    .mean(tch::Kind::Float)
                    .double_value(&[]),
                )
            }),
    )
}

fn pocket_feature_trivial_mse(examples: &[crate::data::MolecularExample]) -> Option<f64> {
    tensor_trivial_mse(
        examples
            .iter()
            .filter(|example| example.pocket.atom_features.numel() > 0)
            .map(|example| example.pocket.atom_features.flatten(0, -1))
            .collect(),
    )
}

fn ligand_pharmacophore_available_count(examples: &[crate::data::MolecularExample]) -> usize {
    examples
        .iter()
        .map(|example| available_role_rows(&example.topology.chemistry_roles))
        .sum()
}

fn pocket_pharmacophore_available_count(examples: &[crate::data::MolecularExample]) -> usize {
    examples
        .iter()
        .map(|example| available_role_rows(&example.pocket.chemistry_roles))
        .sum()
}

fn ligand_pharmacophore_role_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    role_probe_bce(
        examples,
        forwards,
        |example| &example.topology.chemistry_roles,
        |forward| &forward.probes.ligand_pharmacophore_role_logits,
    )
}

fn ligand_pharmacophore_role_balanced_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    role_probe_balanced_bce(
        examples,
        forwards,
        |example| &example.topology.chemistry_roles,
        |forward| &forward.probes.ligand_pharmacophore_role_logits,
    )
}

fn pocket_pharmacophore_role_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    role_probe_bce(
        examples,
        forwards,
        |example| &example.pocket.chemistry_roles,
        |forward| &forward.probes.pocket_pharmacophore_role_logits,
    )
}

fn pocket_pharmacophore_role_balanced_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    role_probe_balanced_bce(
        examples,
        forwards,
        |example| &example.pocket.chemistry_roles,
        |forward| &forward.probes.pocket_pharmacophore_role_logits,
    )
}

fn ligand_pharmacophore_role_trivial_bce(
    examples: &[crate::data::MolecularExample],
) -> Option<f64> {
    role_trivial_bce(examples, |example| &example.topology.chemistry_roles)
}

fn ligand_pharmacophore_role_balanced_trivial_bce(
    examples: &[crate::data::MolecularExample],
) -> Option<f64> {
    role_balanced_trivial_bce(examples, |example| &example.topology.chemistry_roles)
}

fn pocket_pharmacophore_role_trivial_bce(
    examples: &[crate::data::MolecularExample],
) -> Option<f64> {
    role_trivial_bce(examples, |example| &example.pocket.chemistry_roles)
}

fn pocket_pharmacophore_role_balanced_trivial_bce(
    examples: &[crate::data::MolecularExample],
) -> Option<f64> {
    role_balanced_trivial_bce(examples, |example| &example.pocket.chemistry_roles)
}

fn role_probe_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    matrix_for_example: fn(
        &crate::data::MolecularExample,
    ) -> &crate::data::ChemistryRoleFeatureMatrix,
    logits_for_forward: fn(&ResearchForward) -> &tch::Tensor,
) -> Option<f64> {
    mean_available(
        examples
            .iter()
            .zip(forwards.iter())
            .filter_map(|(example, forward)| {
                let matrix = matrix_for_example(example);
                let logits = logits_for_forward(forward);
                if available_role_rows(matrix) == 0
                    || logits.numel() == 0
                    || !role_probe_shapes_match(logits, matrix)
                {
                    return None;
                }
                Some(
                    crate::losses::probe::masked_role_bce_with_logits(
                        logits,
                        &matrix.role_vectors,
                        &matrix.availability,
                    )
                    .double_value(&[]),
                )
            }),
    )
}

fn role_probe_balanced_bce(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    matrix_for_example: fn(
        &crate::data::MolecularExample,
    ) -> &crate::data::ChemistryRoleFeatureMatrix,
    logits_for_forward: fn(&ResearchForward) -> &tch::Tensor,
) -> Option<f64> {
    mean_available(
        examples
            .iter()
            .zip(forwards.iter())
            .filter_map(|(example, forward)| {
                let matrix = matrix_for_example(example);
                let logits = logits_for_forward(forward);
                if available_role_rows(matrix) == 0
                    || logits.numel() == 0
                    || !role_probe_shapes_match(logits, matrix)
                {
                    return None;
                }
                Some(
                    crate::losses::probe::masked_balanced_role_bce_with_logits(
                        logits,
                        &matrix.role_vectors,
                        &matrix.availability,
                    )
                    .double_value(&[]),
                )
            }),
    )
}

fn role_probe_shapes_match(
    logits: &tch::Tensor,
    matrix: &crate::data::ChemistryRoleFeatureMatrix,
) -> bool {
    let logits_size = logits.size();
    let role_size = matrix.role_vectors.size();
    let availability_size = matrix.availability.size();
    logits_size.len() == 2
        && role_size.len() == 2
        && availability_size.len() == 1
        && logits_size == role_size
        && availability_size.first().copied() == logits_size.first().copied()
}

fn role_probe_binary_audit(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
    matrix_for_example: fn(
        &crate::data::MolecularExample,
    ) -> &crate::data::ChemistryRoleFeatureMatrix,
    logits_for_forward: fn(&ResearchForward) -> &tch::Tensor,
) -> Option<BinaryProbeAudit> {
    let mut accumulator = BinaryProbeAuditAccumulator::default();
    for (example, forward) in examples.iter().zip(forwards.iter()) {
        let matrix = matrix_for_example(example);
        let logits = logits_for_forward(forward);
        if available_role_rows(matrix) == 0
            || logits.numel() == 0
            || !role_probe_shapes_match(logits, matrix)
        {
            continue;
        }
        let size = logits.size();
        let Some(rows) = size.first().copied() else {
            continue;
        };
        let Some(cols) = size.get(1).copied() else {
            continue;
        };
        let mask = matrix
            .availability
            .to_device(logits.device())
            .to_kind(tch::Kind::Float)
            .unsqueeze(-1)
            .expand([rows, cols], true);
        accumulator.add(logits, &matrix.role_vectors, &mask);
    }
    accumulator.finish()
}

fn role_trivial_bce(
    examples: &[crate::data::MolecularExample],
    matrix_for_example: fn(
        &crate::data::MolecularExample,
    ) -> &crate::data::ChemistryRoleFeatureMatrix,
) -> Option<f64> {
    let mut positive = 0.0;
    let mut count = 0.0;
    for example in examples {
        let matrix = matrix_for_example(example);
        let rows = matrix
            .role_vectors
            .size()
            .first()
            .copied()
            .unwrap_or(0)
            .min(matrix.availability.size().first().copied().unwrap_or(0));
        let cols = matrix.role_vectors.size().get(1).copied().unwrap_or(0);
        for row in 0..rows {
            if matrix.availability.double_value(&[row]) <= 0.0 {
                continue;
            }
            for col in 0..cols {
                positive += matrix.role_vectors.double_value(&[row, col]).clamp(0.0, 1.0);
                count += 1.0;
            }
        }
    }
    if count <= 0.0 {
        return None;
    }
    let p = (positive / count).clamp(1e-6, 1.0 - 1e-6);
    Some(-((positive * p.ln()) + ((count - positive) * (1.0 - p).ln())) / count)
}

fn role_balanced_trivial_bce(
    examples: &[crate::data::MolecularExample],
    matrix_for_example: fn(
        &crate::data::MolecularExample,
    ) -> &crate::data::ChemistryRoleFeatureMatrix,
) -> Option<f64> {
    let (positive, negative) = role_binary_counts(examples, matrix_for_example)?;
    balanced_binary_trivial_bce(positive, negative)
}

fn role_binary_counts(
    examples: &[crate::data::MolecularExample],
    matrix_for_example: fn(
        &crate::data::MolecularExample,
    ) -> &crate::data::ChemistryRoleFeatureMatrix,
) -> Option<(f64, f64)> {
    let mut positive = 0.0;
    let mut negative = 0.0;
    for example in examples {
        let matrix = matrix_for_example(example);
        let rows = matrix
            .role_vectors
            .size()
            .first()
            .copied()
            .unwrap_or(0)
            .min(matrix.availability.size().first().copied().unwrap_or(0));
        let cols = matrix.role_vectors.size().get(1).copied().unwrap_or(0);
        for row in 0..rows {
            if matrix.availability.double_value(&[row]) <= 0.0 {
                continue;
            }
            for col in 0..cols {
                let value = matrix.role_vectors.double_value(&[row, col]).clamp(0.0, 1.0);
                positive += value;
                negative += 1.0 - value;
            }
        }
    }
    ((positive + negative) > 0.0).then_some((positive, negative))
}

fn balanced_binary_trivial_bce(positive: f64, negative: f64) -> Option<f64> {
    if positive + negative <= 0.0 {
        None
    } else if positive <= 0.0 || negative <= 0.0 {
        Some(0.0)
    } else {
        Some(std::f64::consts::LN_2)
    }
}

fn available_role_rows(matrix: &crate::data::ChemistryRoleFeatureMatrix) -> usize {
    let rows = matrix
        .role_vectors
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(matrix.availability.size().first().copied().unwrap_or(0))
        .max(0);
    (0..rows)
        .filter(|row| matrix.availability.double_value(&[*row]) > 0.0)
        .count()
}

fn affinity_available_count(examples: &[crate::data::MolecularExample]) -> usize {
    examples
        .iter()
        .filter(|example| example.targets.affinity_kcal_mol.is_some())
        .count()
}

fn affinity_probe_mse(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<f64> {
    mean_available(
        examples
            .iter()
            .zip(forwards.iter())
            .filter_map(|(example, forward)| {
                let target = example.targets.affinity_kcal_mol?;
                let pred = forward.probes.affinity_prediction.double_value(&[]);
                let error = pred - target as f64;
                Some(error * error)
            }),
    )
}

fn affinity_scalar_audit(
    examples: &[crate::data::MolecularExample],
    forwards: &[ResearchForward],
) -> Option<ScalarProbeAudit> {
    let mut target_sum = 0.0;
    let mut prediction_sum = 0.0;
    let mut error_sum = 0.0;
    let mut count = 0usize;
    for (example, forward) in examples.iter().zip(forwards.iter()) {
        let Some(target) = example.targets.affinity_kcal_mol.map(|value| value as f64) else {
            continue;
        };
        if forward.probes.affinity_prediction.numel() == 0 {
            continue;
        }
        let prediction = forward.probes.affinity_prediction.double_value(&[]);
        if !target.is_finite() || !prediction.is_finite() {
            continue;
        }
        target_sum += target;
        prediction_sum += prediction;
        error_sum += prediction - target;
        count += 1;
    }
    (count > 0).then_some(ScalarProbeAudit {
        target_mean: target_sum / count as f64,
        prediction_mean: prediction_sum / count as f64,
        mean_error: error_sum / count as f64,
    })
}

fn affinity_trivial_mse(examples: &[crate::data::MolecularExample]) -> Option<f64> {
    let targets = examples
        .iter()
        .filter_map(|example| example.targets.affinity_kcal_mol.map(|value| value as f64))
        .collect::<Vec<_>>();
    if targets.is_empty() {
        return None;
    }
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    Some(
        targets
            .iter()
            .map(|target| {
                let error = target - mean;
                error * error
            })
            .sum::<f64>()
            / targets.len() as f64,
    )
}

fn tensor_trivial_mse(targets: Vec<tch::Tensor>) -> Option<f64> {
    if targets.is_empty() {
        return None;
    }
    let flattened = targets
        .into_iter()
        .map(|target| target.flatten(0, -1))
        .collect::<Vec<_>>();
    let refs = flattened.iter().collect::<Vec<_>>();
    let values = tch::Tensor::cat(&refs, 0);
    if values.numel() == 0 {
        return None;
    }
    let mean = values.mean(tch::Kind::Float);
    Some(
        (values - mean)
            .pow_tensor_scalar(2.0)
            .mean(tch::Kind::Float)
            .double_value(&[]),
    )
}

fn mean_available(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}
