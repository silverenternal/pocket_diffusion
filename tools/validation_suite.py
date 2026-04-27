#!/usr/bin/env python3
"""Run a documented validation suite and persist readable results."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PYTHON_TOOLS = [
    "tools/q1_readiness_audit.py",
    "tools/prepare_docking_inputs.py",
    "tools/vina_score_backend.py",
    "tools/gnina_score_backend.py",
    "tools/merge_candidate_metric_backends.py",
    "tools/correlation_table.py",
    "tools/correlation_plot_markdown.py",
    "tools/claim_regression_gate.py",
    "tools/evidence_bundle.py",
    "tools/reviewer_env_check.py",
    "tools/replay_drift_check.py",
    "tools/rdkit_validity_backend.py",
    "tools/pocket_contact_backend.py",
    "tools/stress_benchmark.py",
    "tools/public_baseline_method_comparison.py",
    "tools/public_baseline_postprocess_layers.py",
    "tools/run_diffsbdd_public_testset.py",
    "tools/baseline_output_adapters/sdf_generation_layers_adapter.py",
    "tools/baseline_output_adapters/targetdiff_meta_generation_layers_adapter.py",
]

JSON_ARTIFACTS = [
    "todo.json",
    "configs/q1_readiness_audit.json",
    "configs/q1_claim_contract.json",
    "configs/candidate_metric_coverage.json",
    "configs/vina_protocol.json",
    "configs/gnina_protocol.json",
    "configs/q1_primary_benchmark_manifest.json",
    "configs/q1_primary_split_report.json",
    "configs/q1_stress_benchmark_manifest.json",
    "configs/q1_stress_benchmark_metrics.json",
    "configs/q1_public_baseline_run_status.json",
    "configs/q1_public_baseline_metric_coverage.json",
    "configs/q1_public_baseline_full100_metric_coverage.json",
    "configs/q1_public_baseline_full100_layered_metric_coverage.json",
    "configs/q1_public_baseline_full100_postprocess_report.json",
    "configs/q1_diffsbdd_public_full100_generation_report.json",
    "configs/q1_method_comparison_summary.json",
    "configs/correlation_table.json",
]

EXPERIMENT_CONFIGS = [
    "configs/unseen_pocket_pdbbindpp_real_backends.json",
    "configs/unseen_pocket_lp_pdbbind_refined_real_backends.json",
    "configs/unseen_pocket_tight_geometry_pressure.json",
    "configs/unseen_pocket_multi_seed_pdbbindpp_real_backends.json",
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run validation checks and write JSON/Markdown reports.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when required checks fail.")
    parser.add_argument("--output-json", default="configs/validation_suite_report.json")
    parser.add_argument("--output-md", default="docs/validation_suite_report.md")
    parser.add_argument("--timeout", type=int, default=240, help="Per-command timeout in seconds.")
    return parser.parse_args(argv[1:])


def run_command(name, command, timeout, required=True):
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        status = "pass" if completed.returncode == 0 else "fail"
        return {
            "name": name,
            "command": command,
            "required": required,
            "status": status,
            "returncode": completed.returncode,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "required": required,
            "status": "timeout",
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }


def existing(paths):
    return [path for path in paths if Path(path).is_file()]


def build_checks(mode):
    py = sys.executable
    checks = [
        ("cargo fmt", ["cargo", "fmt", "--check"], True),
        (
            "python syntax",
            [py, "-m", "py_compile", *existing(PYTHON_TOOLS)],
            True,
        ),
        (
            "json artifacts",
            [py, "-c", "import json,sys; [json.load(open(p)) for p in sys.argv[1:]]", *existing(JSON_ARTIFACTS)],
            True,
        ),
        ("q1 readiness audit", [py, "tools/q1_readiness_audit.py"], True),
        (
            "q1 readiness gate",
            [py, "tools/q1_readiness_audit.py", "--gate"],
            False,
        ),
    ]
    for config in EXPERIMENT_CONFIGS:
        if Path(config).is_file():
            checks.append(
                (
                    f"validate {Path(config).name}",
                    ["cargo", "run", "--bin", "pocket_diffusion", "--", "validate", "--kind", "experiment", "--config", config],
                    True,
                )
            )
    if mode == "full":
        checks.insert(1, ("cargo test", ["cargo", "test"], True))
    else:
        checks.insert(1, ("cargo test no-run", ["cargo", "test", "--no-run"], True))
    return checks


def summarize(results):
    required = [item for item in results if item["required"]]
    optional = [item for item in results if not item["required"]]
    failed_required = [item for item in required if item["status"] != "pass"]
    failed_optional = [item for item in optional if item["status"] != "pass"]
    return {
        "total_checks": len(results),
        "required_checks": len(required),
        "optional_checks": len(optional),
        "failed_required": len(failed_required),
        "failed_optional": len(failed_optional),
        "status": "pass" if not failed_required else "fail",
    }


def write_markdown(path, report):
    lines = [
        "# Validation Suite Report",
        "",
        f"- status: {report['summary']['status']}",
        f"- mode: {report['mode']}",
        f"- total_checks: {report['summary']['total_checks']}",
        f"- failed_required: {report['summary']['failed_required']}",
        f"- failed_optional: {report['summary']['failed_optional']}",
        "",
        "| Check | Required | Status | Seconds |",
        "| --- | --- | --- | ---: |",
    ]
    for item in report["checks"]:
        lines.append(
            "| `{}` | {} | {} | {:.3f} |".format(
                item["name"], str(item["required"]).lower(), item["status"], item["duration_seconds"]
            )
        )
    lines.extend(["", "## Failed Checks"])
    failed = [item for item in report["checks"] if item["status"] != "pass"]
    if not failed:
        lines.append("- none")
    for item in failed:
        lines.append(f"### {item['name']}")
        lines.append(f"- required: {str(item['required']).lower()}")
        lines.append(f"- command: `{' '.join(item['command'])}`")
        lines.append(f"- returncode: `{item['returncode']}`")
        if item["stderr_tail"]:
            lines.append("")
            lines.append("```text")
            lines.append(item["stderr_tail"].rstrip())
            lines.append("```")
        if item["stdout_tail"]:
            lines.append("")
            lines.append("```text")
            lines.append(item["stdout_tail"].rstrip())
            lines.append("```")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    checks = build_checks(args.mode)
    results = [run_command(name, command, args.timeout, required) for name, command, required in checks]
    report = {
        "schema_version": 1,
        "tool": "validation_suite",
        "mode": args.mode,
        "strict": args.strict,
        "summary": summarize(results),
        "checks": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, report)
    return 1 if args.strict and report["summary"]["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
