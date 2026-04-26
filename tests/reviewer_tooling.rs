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
