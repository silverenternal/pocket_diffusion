use std::fs;
use std::process::Command;

fn run_python(args: &[&str]) {
    let status = Command::new("python3")
        .args(args)
        .status()
        .expect("python3 should be available for reviewer tooling tests");
    assert!(status.success(), "python command failed: {:?}", args);
}

#[test]
fn claim_regression_gate_exposes_model_onboarding_flags() {
    let output = Command::new("python3")
        .args(["tools/claim_regression_gate.py", "--help"])
        .output()
        .expect("python3 should be available for reviewer tooling tests");
    assert!(output.status.success(), "claim gate --help should succeed");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("--enforce-publication-readiness"));
    assert!(text.contains("--enforce-preference-readiness"));
    assert!(text.contains("--min-backend-supported-pair-fraction"));
}

#[test]
fn replay_drift_report_contains_promotion_decision() {
    let output = "/tmp/replay_drift_report_test.json";
    run_python(&[
        "tools/replay_drift_check.py",
        "checkpoints/pdbbindpp_real_backends/claim_summary.json",
        "checkpoints/pdbbindpp_real_backends/claim_summary.json",
        "--output",
        output,
    ]);
    let payload = fs::read_to_string(output).expect("replay drift report should exist");
    assert!(payload.contains("\"promotion_decision\""));
    assert!(payload.contains("\"status\": \"pass\""));
}

#[test]
fn evidence_bundle_and_paper_bundle_regenerate() {
    let evidence = "/tmp/evidence_bundle_test.json";
    let paper = "/tmp/paper_claim_bundle_test.md";
    let refresh = "/tmp/reviewer_refresh_report_test.json";
    let hardening = "/tmp/generator_hardening_report_test.md";
    run_python(&["tools/evidence_bundle.py", "--output", evidence]);
    run_python(&[
        "tools/paper_claim_bundle.py",
        "--bundle",
        evidence,
        "--output",
        paper,
    ]);
    run_python(&[
        "tools/reviewer_refresh_report.py",
        "--bundle",
        evidence,
        "--output",
        refresh,
    ]);
    run_python(&[
        "tools/generator_hardening_report.py",
        "--bundle",
        evidence,
        "--output",
        hardening,
    ]);

    let evidence_payload = fs::read_to_string(evidence).expect("bundle should exist");
    let paper_payload = fs::read_to_string(paper).expect("paper bundle should exist");
    let refresh_payload = fs::read_to_string(refresh).expect("refresh report should exist");
    let hardening_payload = fs::read_to_string(hardening).expect("hardening report should exist");

    assert!(evidence_payload.contains("\"benchmark_breadth\""));
    assert!(evidence_payload.contains("lp_pdbbind_refined_real_backends"));
    assert!(evidence_payload.contains("lp_pdbbind_refined"));
    assert!(evidence_payload.contains("\"efficiency_tradeoffs\""));
    assert!(evidence_payload.contains("\"generator_direction\""));
    assert!(paper_payload.contains("# Paper-Facing Claim Bundle"));
    assert!(refresh_payload.contains("\"promotion_rule\""));
    assert!(refresh_payload.contains("\"continuity_mode\""));
    assert!(refresh_payload.contains("\"supports_strict_replay\": false"));
    assert!(refresh_payload.contains("\"packaged_environment_ready\""));
    assert!(refresh_payload.contains("\"packaged_environment_effective\""));
    assert!(hardening_payload.contains("# Generator Hardening Report"));
}

#[test]
fn generator_decision_artifact_can_be_checked_for_freshness() {
    let output = "/tmp/generator_decision_test.json";
    run_python(&["tools/generator_decision_bundle.py", "--output", output]);
    run_python(&[
        "tools/generator_decision_bundle.py",
        "--output",
        output,
        "--check",
    ]);

    let payload = fs::read_to_string(output).expect("generator decision should exist");
    assert!(payload.contains("\"freshness\""));
    assert!(payload.contains("\"required_surfaces\""));
    assert!(payload.contains("\"monitored_promotion_paths\""));
}

#[test]
fn vina_companion_claim_artifact_persists_backend_review() {
    let payload = fs::read_to_string("checkpoints/vina_backend/claim_summary.json")
        .expect("vina backend claim summary should exist");
    assert!(payload.contains("\"backend_review\""));
    assert!(payload.contains("\"vina_claim_bearing_companion_policy\""));
    assert!(payload.contains("\"docking_input_completeness_fraction\""));
    assert!(payload.contains("\"docking_score_coverage_fraction\""));
}

#[test]
fn vina_backend_emits_drug_level_contract_aliases_when_unavailable() {
    let input = "/tmp/vina_backend_contract_aliases_input.json";
    let output = "/tmp/vina_backend_contract_aliases_output.json";
    fs::write(
        input,
        r#"[{"candidate_id":"c1","example_id":"e1","protein_id":"p1"}]"#,
    )
    .unwrap();
    run_python(&[
        "tools/vina_score_backend.py",
        "--vina-executable",
        "definitely_missing_vina_for_contract_test",
        input,
        output,
    ]);

    let payload = fs::read_to_string(output).expect("vina backend output should exist");
    assert!(payload.contains("\"input_count\": 1.0"));
    assert!(payload.contains("\"scored_count\": 0.0"));
    assert!(payload.contains("\"failure_count\": 1.0"));
    assert!(payload.contains("\"docking_score_coverage_fraction\": 0.0"));
}

#[test]
fn gnina_backend_is_optional_and_reports_unavailable_coverage() {
    let input = "/tmp/gnina_backend_optional_input.json";
    let output = "/tmp/gnina_backend_optional_output.json";
    fs::write(
        input,
        r#"[{"candidate_id":"c1","example_id":"e1","protein_id":"p1"}]"#,
    )
    .unwrap();
    run_python(&[
        "tools/gnina_score_backend.py",
        "--gnina-executable",
        "definitely_missing_gnina_for_contract_test",
        input,
        output,
    ]);

    let payload = fs::read_to_string(output).expect("gnina backend output should exist");
    assert!(payload.contains("\"gnina_available\": 0.0"));
    assert!(payload.contains("\"input_count\": 1.0"));
    assert!(payload.contains("\"scored_count\": 0.0"));
    assert!(payload.contains("\"gnina_score_coverage_fraction\": 0.0"));
}

#[test]
fn rdkit_backend_declares_drug_likeness_coverage() {
    let input = "/tmp/rdkit_drug_likeness_input.json";
    let output = "/tmp/rdkit_drug_likeness_output.json";
    fs::write(
        input,
        r#"[{"candidate_id":"c1","example_id":"e1","protein_id":"p1","atom_types":[0],"coords":[[0.0,0.0,0.0]],"inferred_bonds":[]}]"#,
    )
    .unwrap();
    run_python(&["tools/rdkit_validity_backend.py", input, output]);

    let payload = fs::read_to_string(output).expect("rdkit backend output should exist");
    assert!(payload.contains("\"drug_likeness_coverage_fraction\""));
    assert!(payload.contains("\"backend_examples_scored\""));
}

#[test]
fn rdkit_backend_declares_scaffold_diversity_coverage() {
    let input = "/tmp/rdkit_scaffold_input.json";
    let reference = "/tmp/rdkit_scaffold_reference.json";
    let output = "/tmp/rdkit_scaffold_output.json";
    fs::write(
        input,
        r#"[
          {"candidate_id":"c1","example_id":"e1","protein_id":"p1","atom_types":[0,0,0],"coords":[[0.0,0.0,0.0],[1.4,0.0,0.0],[2.8,0.0,0.0]],"inferred_bonds":[[0,1],[1,2]]},
          {"candidate_id":"c2","example_id":"e1","protein_id":"p1","atom_types":[0,0,1],"coords":[[0.0,0.0,0.0],[1.4,0.0,0.0],[2.8,0.0,0.0]],"inferred_bonds":[[0,1],[1,2]]}
        ]"#,
    )
    .unwrap();
    fs::write(
        reference,
        r#"[{"candidate_id":"r1","example_id":"train","protein_id":"tp","atom_types":[0,0,0],"coords":[[0.0,0.0,0.0],[1.4,0.0,0.0],[2.8,0.0,0.0]],"inferred_bonds":[[0,1],[1,2]]}]"#,
    )
    .unwrap();
    run_python(&[
        "tools/rdkit_validity_backend.py",
        "--reference-candidates",
        reference,
        input,
        output,
    ]);

    let payload = fs::read_to_string(output).expect("rdkit backend output should exist");
    assert!(payload.contains("\"scaffold_metric_coverage_fraction\""));
    assert!(payload.contains("\"scaffold_novelty_fraction\""));
    assert!(payload.contains("\"unique_scaffold_fraction\""));
    assert!(payload.contains("\"pairwise_tanimoto_mean\""));
    assert!(payload.contains("\"nearest_train_similarity\""));
}

#[test]
fn correlation_table_builder_reports_missing_backend_coverage() {
    let input = "/tmp/candidate_metrics_correlation.jsonl";
    let output = "/tmp/correlation_table_test.json";
    fs::write(
        input,
        concat!(
            "{\"layer\":\"raw_flow\",\"metrics\":{\"pocket_contact_fraction\":0.1,\"vina_score\":-5.0,\"clash_fraction\":0.4,\"qed\":0.3,\"sa_score\":4.0,\"hydrogen_bond_proxy\":0.1}}\n",
            "{\"layer\":\"raw_flow\",\"metrics\":{\"pocket_contact_fraction\":0.2,\"vina_score\":-6.0,\"clash_fraction\":0.3,\"qed\":0.4,\"sa_score\":3.0,\"hydrogen_bond_proxy\":0.2}}\n",
            "{\"layer\":\"reranked\",\"metrics\":{\"pocket_contact_fraction\":0.3,\"vina_score\":-7.0,\"clash_fraction\":0.2,\"qed\":0.5,\"sa_score\":2.0,\"hydrogen_bond_proxy\":0.3}}\n"
        ),
    )
    .unwrap();
    run_python(&[
        "tools/correlation_table.py",
        "--allow-low-sample",
        "--output",
        output,
        input,
    ]);

    let payload = fs::read_to_string(output).expect("correlation table should exist");
    assert!(payload.contains("\"pair_id\": \"pocket_fit_vs_vina\""));
    assert!(payload.contains("\"pearson\""));
    assert!(payload.contains("\"spearman\""));
    assert!(payload.contains("\"missing_count\""));
    assert!(payload.contains("\"scope\": \"layer:raw_flow\""));
}

#[test]
fn correlation_markdown_is_rendered_from_table() {
    let table = "/tmp/correlation_table_markdown_input.json";
    let output = "/tmp/correlation_plot_test.md";
    fs::write(
        table,
        r#"{
          "schema_version": 1,
          "record_count": 3,
          "min_samples": 3,
          "metric_pairs": [
            {"scope":"all","pair_id":"pocket_fit_vs_docking","left_metric":"pocket_contact_fraction","right_metric":"vina_score","sample_count":3,"missing_count":0,"pearson":-1.0,"spearman":-1.0,"confidence_note":"interpretable"}
          ]
        }"#,
    )
    .unwrap();
    run_python(&[
        "tools/correlation_plot_markdown.py",
        table,
        "--output",
        output,
    ]);
    let payload = fs::read_to_string(output).expect("correlation markdown should exist");
    assert!(payload.contains("# Correlation Plot"));
    assert!(payload.contains("Pocket Fit Vs Docking"));
    assert!(payload.contains("-1.0000"));
    assert!(payload.contains("Constraint Vs True Model Capability"));
}

#[test]
fn method_comparison_summary_includes_binding_analysis() {
    let artifact_dir = "/tmp/method_comparison_binding_case";
    let output = "/tmp/method_comparison_binding_summary.json";
    let binding = "/tmp/flow_vs_denoising_binding_analysis_test.json";
    let _ = fs::remove_dir_all(artifact_dir);
    fs::create_dir_all(artifact_dir).unwrap();
    fs::write(
        format!("{artifact_dir}/claim_summary.json"),
        r#"{
          "method_comparison": {
            "active_method": {"method_id":"flow_matching"},
            "methods": [
              {"method_id":"flow_matching","method_name":"Flow","native_valid_fraction":0.9,"native_pocket_contact_fraction":0.8,"native_clash_fraction":0.1},
              {"method_id":"conditioned_denoising","method_name":"Denoising","native_valid_fraction":0.8,"native_pocket_contact_fraction":0.7,"native_clash_fraction":0.2}
            ]
          },
          "layered_generation_metrics": {
            "raw_flow": {"candidate_count":1,"valid_fraction":0.9,"pocket_contact_fraction":0.8,"mean_centroid_offset":0.5,"clash_fraction":0.1,"uniqueness_proxy_fraction":1.0},
            "reranked_candidates": {"candidate_count":1,"valid_fraction":1.0,"pocket_contact_fraction":0.9,"mean_centroid_offset":0.4,"clash_fraction":0.0,"uniqueness_proxy_fraction":1.0,"hydrogen_bond_proxy":0.5,"hydrophobic_contact_proxy":0.25,"contact_balance":0.5,"interaction_profile_coverage_fraction":1.0}
          },
          "backend_metrics": {
            "chemistry_validity": {"metrics": {"qed":0.6,"sa_score":2.5,"drug_likeness_coverage_fraction":1.0,"scaffold_metric_coverage_fraction":1.0}},
            "docking_affinity": {"metrics": {"vina_score_mean":-7.0,"docking_score_coverage_fraction":1.0}}
          },
          "reranker_report": {"flow_native_quality":0.4}
        }"#,
    )
    .unwrap();
    run_python(&[
        "tools/method_comparison_summary.py",
        artifact_dir,
        "--output",
        output,
        "--binding-output",
        binding,
    ]);
    let summary_payload = fs::read_to_string(output).expect("method summary should exist");
    assert!(summary_payload.contains("\"raw_native_evidence\""));
    assert!(summary_payload.contains("\"processed_generation_evidence\""));
    assert!(
        summary_payload.find("\"raw_native_evidence\"").unwrap()
            < summary_payload
                .find("\"processed_generation_evidence\"")
                .unwrap()
    );
    assert!(
        summary_payload
            .find("\"processed_generation_evidence\"")
            .unwrap()
            < summary_payload.find("\"methods\"").unwrap()
    );
    let payload = fs::read_to_string(binding).expect("binding analysis should exist");
    assert!(payload
        .contains("\"comparison\": \"flow_matching_vs_conditioned_denoising_binding_metrics\""));
    assert!(payload.contains("\"raw_native_evidence\""));
    assert!(payload.contains("\"flow_native_raw_flow\""));
    assert!(payload.contains("\"reranked_layer\""));
    assert!(payload.contains("\"promotion_decision\""));
}

#[test]
fn ablation_delta_table_emits_metric_rows() {
    let artifact_dir = "/tmp/ablation_delta_metric_rows_case";
    let output = "/tmp/ablation_delta_metric_rows.json";
    let _ = fs::remove_dir_all(artifact_dir);
    fs::create_dir_all(artifact_dir).unwrap();
    fs::write(
        format!("{artifact_dir}/claim_summary.json"),
        r#"{"test":{"candidate_valid_fraction":0.9,"strict_pocket_fit_score":0.5,"mean_centroid_offset":1.0,"leakage_proxy_mean":0.1,"slot_activation_mean":0.4,"gate_activation_mean":0.3}}"#,
    )
    .unwrap();
    fs::write(
        format!("{artifact_dir}/ablation_matrix_summary.json"),
        r#"{"variants":[{"variant_label":"disable_slots","test":{"candidate_valid_fraction":0.8,"strict_pocket_fit_score":0.4,"mean_centroid_offset":1.5,"leakage_proxy_mean":0.08,"slot_activation_mean":0.0,"gate_activation_mean":0.2}}]}"#,
    )
    .unwrap();
    run_python(&[
        "tools/ablation_delta_table.py",
        artifact_dir,
        "--output",
        output,
    ]);
    let payload = fs::read_to_string(output).expect("ablation delta table should exist");
    assert!(payload.contains("\"metric_name\": \"strict_pocket_fit_score\""));
    assert!(payload.contains("\"base_value\""));
    assert!(payload.contains("\"variant_value\""));
    assert!(payload.contains("\"direction_is_better\""));
    assert!(payload.contains("\"evidence_source\": \"test_summary\""));
}

#[test]
fn multi_seed_drug_level_summary_reports_missing_metrics() {
    let root = "/tmp/multi_seed_drug_level_case";
    let output = "/tmp/multi_seed_drug_level_summary.json";
    let _ = fs::remove_dir_all(root);
    fs::create_dir_all(format!("{root}/seed_1")).unwrap();
    fs::create_dir_all(format!("{root}/seed_2")).unwrap();
    fs::write(
        format!("{root}/seed_1/claim_summary.json"),
        r#"{"test":{"candidate_valid_fraction":1.0,"strict_pocket_fit_score":0.5},"backend_metrics":{"chemistry_validity":{"metrics":{"qed":0.6}}},"layered_generation_metrics":{"reranked_candidates":{"hydrogen_bond_proxy":0.2}}}"#,
    )
    .unwrap();
    fs::write(
        format!("{root}/seed_2/claim_summary.json"),
        r#"{"test":{"candidate_valid_fraction":0.8,"strict_pocket_fit_score":0.4},"backend_metrics":{"chemistry_validity":{"metrics":{}}},"layered_generation_metrics":{"reranked_candidates":{}}}"#,
    )
    .unwrap();
    run_python(&[
        "tools/multi_seed_drug_level_summary.py",
        root,
        "--output",
        output,
    ]);
    let payload = fs::read_to_string(output).expect("multi-seed summary should exist");
    assert!(payload.contains("\"seed_count\": 2"));
    assert!(payload.contains("\"candidate_valid_fraction\""));
    assert!(payload.contains("\"missing_seed_count\""));
    assert!(payload.contains("\"qed\""));
}

#[test]
fn evidence_bundle_reads_compact_preference_artifact_summary() {
    let artifact_dir = "/tmp/reviewer_preference_artifacts_case";
    let output = "/tmp/evidence_bundle_preference_summary_test.json";
    let _ = fs::remove_dir_all(artifact_dir);
    fs::create_dir_all(artifact_dir).expect("temp artifact dir should be creatable");

    fs::write(
        format!("{artifact_dir}/preference_profiles_validation.json"),
        r#"{
          "schema_version": 1,
          "split": "validation",
          "profile_count": 2,
          "backend_coverage": {"chemistry_validity": 2},
          "records": []
        }"#,
    )
    .unwrap();
    fs::write(
        format!("{artifact_dir}/preference_pairs_validation.json"),
        r#"{
          "schema_version": 1,
          "split": "validation",
          "pair_count": 1,
          "source_coverage": {"backend_based": 1},
          "backend_supported_pair_fraction": 1.0,
          "rule_only_pair_fraction": 0.0,
          "missing_backend_evidence_fraction": 0.0,
          "mean_preference_strength": 0.8,
          "hard_constraint_win_fraction": 1.0,
          "records": []
        }"#,
    )
    .unwrap();
    fs::write(
        format!("{artifact_dir}/preference_profiles_test.json"),
        r#"{
          "schema_version": 1,
          "split": "test",
          "profile_count": 1,
          "backend_coverage": {"pocket_compatibility": 1},
          "records": []
        }"#,
    )
    .unwrap();
    fs::write(
        format!("{artifact_dir}/preference_pairs_test.json"),
        r#"{
          "schema_version": 1,
          "split": "test",
          "pair_count": 1,
          "source_coverage": {"rule_based": 1},
          "backend_supported_pair_fraction": 0.0,
          "rule_only_pair_fraction": 1.0,
          "missing_backend_evidence_fraction": 1.0,
          "mean_preference_strength": 0.4,
          "hard_constraint_win_fraction": 0.0,
          "records": []
        }"#,
    )
    .unwrap();
    fs::write(
        format!("{artifact_dir}/preference_reranker_summary.json"),
        r#"{
          "schema_version": 1,
          "enabled": true
        }"#,
    )
    .unwrap();

    run_python(&[
        "tools/evidence_bundle.py",
        "--artifact-dir",
        artifact_dir,
        "--output",
        output,
    ]);

    let payload = fs::read_to_string(output).expect("bundle should exist");
    assert!(payload.contains("\"profile_count_by_split\""));
    assert!(payload.contains("\"pair_count_by_split\""));
    assert!(payload.contains("\"source_breakdown\""));
    assert!(payload.contains("\"backend_coverage\""));
    assert!(payload.contains("\"reranker_enabled\": true"));
    assert!(!payload.contains("\"records\""));
}
