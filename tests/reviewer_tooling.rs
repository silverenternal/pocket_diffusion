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
