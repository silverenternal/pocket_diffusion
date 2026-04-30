#!/usr/bin/env python3
"""Run a documented validation suite and persist readable results."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import importlib.util
import tempfile
from pathlib import Path

from artifact_paths import (
    artifact_default_root,
    drug_metric_artifact_path,
    validate_artifact_retention_manifest,
    validate_drug_metric_manifest,
)
from validation import architecture as architecture_validation
from validation.artifacts import load_json_artifact
from validation import claim_provenance as claim_provenance_validation
from validation import training_replay as training_replay_validation
from validation import negative_fixtures as negative_fixture_validation
from validation import reporting as validation_reporting

Q2_CANDIDATE_METRICS = drug_metric_artifact_path("q2.candidate_metrics")
Q2_PROXY_BACKEND_CORRELATION = drug_metric_artifact_path("q2.proxy_backend_correlation")
Q2_POSTPROCESSING_FAILURE_AUDIT = drug_metric_artifact_path("q2.postprocessing_failure_audit")
Q2_CLAIM_CONTRACT = drug_metric_artifact_path("q2.claim_contract")
Q2_OURS_VS_PUBLIC_BASELINES_SUMMARY = drug_metric_artifact_path("q2.ours_vs_public_baselines_summary")
Q2_POSTPROCESSING_ABLATION_SUMMARY = drug_metric_artifact_path("q2.postprocessing_ablation_summary")
Q3_ROTATION_CONSISTENCY_REPORT = drug_metric_artifact_path("q3.rotation_consistency_report")
Q3_MODEL_IMPROVEMENT_LEADERBOARD = drug_metric_artifact_path("q3.model_improvement_leaderboard")
Q3_NON_DEGRADATION_GATE = drug_metric_artifact_path("q3.non_degradation_gate")

RAW_MODEL_NATIVE_LAYERS = {"raw_flow", "raw_rollout", "raw_geometry", "no_repair"}
CONSTRAINED_SAMPLING_LAYERS = {"constrained_flow"}
POSTPROCESSING_LAYERS = {
    "repaired",
    "inferred_bond",
    "deterministic_proxy",
    "reranked",
    "centroid_only",
    "clash_only",
    "bond_inference_only",
    "full_repair",
    "bond_logits_refined",
    "valence_refined",
    "gated_repair",
    "repair_rejected",
    "backend_aware_posthoc",
}
REQUIRED_LAYER_BOUNDARY_TERMS = {
    "raw_rollout",
    "raw_flow",
    "constrained_flow",
    "repaired",
    "inferred_bond",
    "deterministic_proxy",
    "reranked",
}
GENERATION_ENTRYPOINT_BOUNDARY_TERMS = {
    "raw_flow",
    "constrained_flow",
    "raw_model_native",
    "constrained_sampling",
    "candidate_layer",
    "evidence_role",
    "generation_demo_candidates_raw.json",
    "generation_demo_candidates_constrained_flow.json",
    "generation_demo_candidates.json",
}

PYTHON_TOOLS = [
    "tools/artifact_paths.py",
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
    "tools/q3_model_improvement_leaderboard.py",
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
    Q2_OURS_VS_PUBLIC_BASELINES_SUMMARY,
    Q2_POSTPROCESSING_ABLATION_SUMMARY,
    Q2_CLAIM_CONTRACT,
    "configs/drug_metric_artifact_manifest.json",
    "configs/artifact_retention_manifest.json",
    Q2_PROXY_BACKEND_CORRELATION,
    Q2_POSTPROCESSING_FAILURE_AUDIT,
    Q3_ROTATION_CONSISTENCY_REPORT,
    Q3_MODEL_IMPROVEMENT_LEADERBOARD,
    Q3_NON_DEGRADATION_GATE,
    "configs/q15_generation_alignment_ablation_matrix.json",
]

EXPERIMENT_CONFIGS = [
    "configs/unseen_pocket_pdbbindpp_real_backends.json",
    "configs/unseen_pocket_lp_pdbbind_refined_real_backends.json",
    "configs/unseen_pocket_tight_geometry_pressure.json",
    "configs/unseen_pocket_multi_seed_pdbbindpp_real_backends.json",
]

VALIDATION_MANIFEST_PATH = Path("configs/validation_manifest.json")


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


def _load_validation_manifest():
    if not VALIDATION_MANIFEST_PATH.is_file():
        return None
    try:
        with VALIDATION_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _normalize_manifest_entry(entry, default_required=True):
    if isinstance(entry, str):
        return {"path": entry, "required": default_required}
    if isinstance(entry, dict) and isinstance(entry.get("path"), str):
        normalized = dict(entry)
        normalized["required"] = bool(entry.get("required", default_required))
        return normalized
    return None


def validation_manifest_entries(section, defaults):
    manifest = _load_validation_manifest()
    raw_entries = manifest.get(section) if isinstance(manifest, dict) else None
    if not isinstance(raw_entries, list):
        raw_entries = list(defaults)
    entries = []
    seen = set()
    for raw in raw_entries:
        entry = _normalize_manifest_entry(raw)
        if entry is None:
            continue
        path = entry["path"]
        if path in seen:
            continue
        seen.add(path)
        entries.append(entry)
    return entries


def validation_manifest_paths(section, defaults, *, include_optional=True):
    return [
        entry["path"]
        for entry in validation_manifest_entries(section, defaults)
        if include_optional or entry.get("required", True)
    ]


def validation_required_missing(section, defaults):
    return [
        entry["path"]
        for entry in validation_manifest_entries(section, defaults)
        if entry.get("required", True) and not Path(entry["path"]).is_file()
    ]


def validation_optional_missing(section, defaults):
    return [
        entry["path"]
        for entry in validation_manifest_entries(section, defaults)
        if not entry.get("required", True) and not Path(entry["path"]).is_file()
    ]


def build_checks(mode):
    py = sys.executable
    python_tools = validation_manifest_paths("python_tools", PYTHON_TOOLS)
    json_artifacts = validation_manifest_paths("json_artifacts", JSON_ARTIFACTS)
    experiment_configs = validation_manifest_paths("experiment_configs", EXPERIMENT_CONFIGS)
    checks = [
        ("cargo fmt", ["cargo", "fmt", "--check"], True),
        (
            "python syntax",
            [py, "-m", "py_compile", *existing(python_tools)],
            True,
        ),
        (
            "json artifacts",
            [py, "-c", "import json,sys; [json.load(open(p)) for p in sys.argv[1:]]", *existing(json_artifacts)],
            True,
        ),
        ("q1 readiness audit", [py, "tools/q1_readiness_audit.py"], True),
        (
            "q1 readiness gate",
            [py, "tools/q1_readiness_audit.py", "--gate"],
            False,
        ),
    ]
    for config in experiment_configs:
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
        checks.insert(
            2,
            (
                "drug metric contract drift check",
                [
                    "cargo",
                    "test",
                    "--no-default-features",
                    "drug_metric_contract_matches_legacy_classifier",
                ],
                True,
            ),
        )
    return checks


PLACEHOLDER_RE = re.compile(r"\b0\.XXX\b|~\s*\d+(?:\.\d+)?|~\s*0\.XXX")


def validate_drug_metric_contract():
    started = time.time()
    failures = []
    manifest_path = "configs/drug_metric_artifact_manifest.json"
    if not Path(manifest_path).is_file():
        failures.append("missing drug-metric artifact manifest")
    else:
        failures.extend(validate_drug_metric_manifest(manifest_path))
        manifest = load_json_artifact(manifest_path)
        contract = manifest.get("metric_contract", {})
        metrics = contract.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            failures.append("drug metric contract missing non-empty metrics list")
        else:
            for index, entry in enumerate(metrics):
                for field in ("metric_name", "domain", "direction", "primary_claim_metric"):
                    if field not in entry:
                        failures.append(
                            f"drug metric contract entry {index} missing required field {field}"
                        )
                if entry.get("direction") not in {
                    "HigherIsBetter",
                    "LowerIsBetter",
                    "Guardrail",
                }:
                    failures.append(
                        f"drug metric contract entry {index} has unsupported direction: {entry.get('direction')}"
                    )
                if entry.get("domain") not in {
                    "Docking",
                    "DrugLikeness",
                    "ChemistryValidity",
                    "PocketCompatibility",
                    "BackendQuality",
                    "Auxiliary",
                }:
                    failures.append(
                        f"drug metric contract entry {index} has unsupported domain: {entry.get('domain')}"
                    )

            required_primary_claims = {
                "vina_score",
                "gnina_affinity",
                "qed",
                "sa_score",
                "valid_fraction",
                "valence_sanity_fraction",
                "structural_pass_fraction",
            }
            present = {
                entry.get("metric_name")
                for entry in metrics
                if isinstance(entry.get("metric_name"), str)
            }
            missing = sorted(required_primary_claims - present)
            if missing:
                failures.append(
                    "drug metric contract missing required primary claim metrics: "
                    + ", ".join(missing)
                )

    status = "pass" if not failures else "fail"
    return {
        "name": "drug metric contract config",
        "command": ["internal", "validate_drug_metric_contract"],
        "required": True,
        "status": status,
        "returncode": 0 if status == "pass" else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-20:]),
        "stderr_tail": "",
    }


def validate_validation_manifest():
    started = time.time()
    failures = []
    optional_missing = []
    if not VALIDATION_MANIFEST_PATH.is_file():
        failures.append(f"missing validation manifest: {VALIDATION_MANIFEST_PATH}")
    else:
        try:
            manifest = load_json_artifact(VALIDATION_MANIFEST_PATH)
        except json.JSONDecodeError as exc:
            failures.append(f"{VALIDATION_MANIFEST_PATH} contains invalid JSON ({exc})")
            manifest = {}
        except OSError as exc:
            failures.append(f"{VALIDATION_MANIFEST_PATH} not readable ({exc})")
            manifest = {}

        if manifest.get("schema_version") != 1:
            failures.append("validation manifest schema_version must be 1")
        for section, defaults in (
            ("python_tools", PYTHON_TOOLS),
            ("json_artifacts", JSON_ARTIFACTS),
            ("experiment_configs", EXPERIMENT_CONFIGS),
        ):
            raw_entries = manifest.get(section)
            if not isinstance(raw_entries, list) or not raw_entries:
                failures.append(f"validation manifest {section} must be a non-empty list")
                continue
            for index, raw in enumerate(raw_entries):
                entry = _normalize_manifest_entry(raw)
                if entry is None:
                    failures.append(f"validation manifest {section}[{index}] must be a path entry")
                    continue
                if entry.get("required", True) and not Path(entry["path"]).is_file():
                    failures.append(f"required validation manifest path missing: {entry['path']}")
                if not entry.get("required", True) and not Path(entry["path"]).is_file():
                    optional_missing.append(entry["path"])
            default_set = set(defaults)
            manifest_set = {
                entry["path"]
                for entry in (_normalize_manifest_entry(raw) for raw in raw_entries)
                if entry is not None
            }
            missing_defaults = sorted(default_set - manifest_set)
            if missing_defaults:
                failures.append(f"validation manifest {section} missing default entries: {missing_defaults}")

        optional_checks = manifest.get("optional_checks")
        if optional_checks is not None and not isinstance(optional_checks, list):
            failures.append("validation manifest optional_checks must be a list when present")
        evidence_families = manifest.get("evidence_families")
        if not isinstance(evidence_families, dict):
            failures.append("validation manifest evidence_families must be an object")
        else:
            for family in ("q1", "q2", "q3", "q5", "artifact_contract"):
                if family not in evidence_families:
                    failures.append(f"validation manifest evidence_families missing {family}")

    lines = []
    if optional_missing:
        lines.append("optional missing paths:")
        lines.extend(f"- {path}" for path in optional_missing)
    if failures:
        lines.extend(failures)
    return {
        "name": "validation manifest",
        "command": ["internal", "validate_validation_manifest"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(lines[-80:]),
        "stderr_tail": "",
    }


def validate_artifact_retention_policy():
    started = time.time()
    failures = []
    failures.extend(validate_artifact_retention_manifest("configs/artifact_retention_manifest.json"))

    if artifact_default_root("generated_checkpoint") != "checkpoints/":
        failures.append("generated_checkpoint default root must be checkpoints/")
    if artifact_default_root("generated_evidence") != "artifacts/evidence/":
        failures.append("generated_evidence default root must be artifacts/evidence/")

    ignored_probes = [
        "checkpoints/q6_artifact_policy_probe/example.json",
        "configs/checkpoints/q6_artifact_policy_probe/example.json",
    ]
    for probe in ignored_probes:
        completed = subprocess.run(
            ["git", "check-ignore", probe],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            failures.append(f"generated checkpoint probe is not ignored: {probe}")

    validation_inputs = [
        *validation_manifest_paths("json_artifacts", JSON_ARTIFACTS),
        *validation_manifest_paths("experiment_configs", EXPERIMENT_CONFIGS),
    ]
    forbidden = [
        path for path in validation_inputs if isinstance(path, str) and path.startswith("configs/checkpoints/")
    ]
    if forbidden:
        failures.append(
            "validation required inputs must not depend on configs/checkpoints trees: "
            + ", ".join(forbidden)
        )

    def _normalize_forbidden_path(value):
        if not isinstance(value, str):
            return ""
        normalized = value.replace("\\", "/").strip()
        parts = []
        for part in normalized.split("/"):
            if not part or part == "." or part == "..":
                continue
            parts.append(part)
        return "/".join(parts)

    def _is_forbidden_checkpoint_dir(value):
        if not isinstance(value, str):
            return False
        normalized = _normalize_forbidden_path(value)
        normalized_parts = normalized.split("/")
        for index in range(len(normalized_parts) - 1):
            if (
                normalized_parts[index] == "configs"
                and normalized_parts[index + 1] == "checkpoints"
            ):
                return True
        if normalized == "configs/checkpoints":
            return True
        return False

    def _json_forbidden_checkpoint_dir(payload):
        if isinstance(payload, dict):
            if _is_forbidden_checkpoint_dir(payload.get("checkpoint_dir", "")):
                return True
            training = payload.get("training")
            if isinstance(training, dict) and _is_forbidden_checkpoint_dir(
                training.get("checkpoint_dir", "")
            ):
                return True
            for value in payload.values():
                if isinstance(value, (dict, list)) and _json_forbidden_checkpoint_dir(value):
                    return True
        elif isinstance(payload, list):
            return any(_json_forbidden_checkpoint_dir(value) for value in payload)
        return False

    for config_path in validation_inputs:
        if not isinstance(config_path, str) or not config_path.startswith("configs/"):
            continue
        if not Path(config_path).is_file():
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            failures.append(f"failed to read validation input as JSON: {config_path}")
            continue
        if _json_forbidden_checkpoint_dir(payload):
            failures.append(
                f"validation input contains training.checkpoint_dir under configs/checkpoints: {config_path}"
            )

    return {
        "name": "artifact retention policy",
        "command": ["internal", "validate_artifact_retention_policy"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }


def validate_todo_schema(failures):
    todo = load_json_artifact("todo.json")
    phases = todo.get("phases")
    if not isinstance(phases, list):
        failures.append("todo.json must contain a phases list")
        return
    if not phases:
        return

    ids = set()
    for phase in phases:
        if not isinstance(phase, dict):
            failures.append("todo phase must be an object")
            continue
        for field in ("id", "name", "goal", "tasks"):
            if field not in phase:
                failures.append(f"todo phase missing {field}: {phase.get('id', '<unknown>')}")
        tasks = phase.get("tasks", [])
        if not isinstance(tasks, list):
            failures.append(f"todo phase tasks must be a list: {phase.get('id', '<unknown>')}")
            continue
        for task in tasks:
            if not isinstance(task, dict):
                failures.append(f"todo task must be an object in phase {phase.get('id', '<unknown>')}")
                continue
            task_id = task.get("id")
            if isinstance(task_id, str) and task_id:
                if task_id in ids:
                    failures.append(f"duplicate todo task id: {task_id}")
                ids.add(task_id)
            for field in ("id", "title", "rationale", "actions", "acceptance_criteria", "verification"):
                if field not in task:
                    failures.append(f"todo task missing {field}: {task.get('id', '<unknown>')}")
            for list_field in ("actions", "acceptance_criteria", "verification"):
                value = task.get(list_field)
                if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                    failures.append(
                        f"todo task {task.get('id', '<unknown>')} {list_field} must be a non-empty list of strings"
                    )

    for phase in phases:
        if not isinstance(phase, dict):
            continue
        for task in phase.get("tasks", []):
            if not isinstance(task, dict):
                continue
            dependencies = task.get("dependencies", [])
            if dependencies is None:
                continue
            if not isinstance(dependencies, list):
                failures.append(f"todo task {task.get('id')} dependencies must be a list when present")
                continue
            for dep in dependencies:
                if dep not in ids:
                    failures.append(f"todo task {task.get('id')} has missing dependency {dep}")


def validate_q2_artifacts():
    started = time.time()
    failures = []
    required_paths = [
        Q2_OURS_VS_PUBLIC_BASELINES_SUMMARY,
        Q2_POSTPROCESSING_ABLATION_SUMMARY,
        Q2_PROXY_BACKEND_CORRELATION,
        Q2_POSTPROCESSING_FAILURE_AUDIT,
        Q2_CLAIM_CONTRACT,
    ]
    for path in required_paths:
        if not Path(path).is_file():
            failures.append(f"missing required q2 artifact: {path}")
    if not failures:
        comparison = load_json_artifact(Q2_OURS_VS_PUBLIC_BASELINES_SUMMARY)
        if "claim_guardrail" not in comparison or "score_only" not in str(comparison.get("claim_guardrail")):
            failures.append("q2 comparison summary missing score_only claim_guardrail")
        if not comparison.get("methods"):
            failures.append("q2 comparison summary missing methods")

        ablation = load_json_artifact(Q2_POSTPROCESSING_ABLATION_SUMMARY)
        layers = {row.get("layer") for row in ablation.get("layer_summaries", [])}
        for layer in ("no_repair", "centroid_only", "clash_only", "bond_inference_only", "full_repair"):
            if layer not in layers:
                failures.append(f"q2 postprocessing ablation missing layer {layer}")
        for row in ablation.get("layer_summaries", []):
            if "backend_coverage" not in row or "failure_reasons" not in row:
                failures.append(f"q2 postprocessing ablation layer missing coverage/failures: {row.get('layer')}")

        correlation = load_json_artifact(Q2_PROXY_BACKEND_CORRELATION)
        if not correlation.get("metric_pairs"):
            failures.append("q2 proxy correlation missing metric_pairs")
        for row in correlation.get("metric_pairs", [])[:20]:
            for field in ("direction_expectation", "interpretation", "sample_count", "missing_count"):
                if field not in row:
                    failures.append(f"q2 proxy correlation row missing {field}")
        if not correlation.get("layer_delta_summaries"):
            failures.append("q2 proxy correlation missing layer_delta_summaries")

        claim = load_json_artifact(Q2_CLAIM_CONTRACT)
        groups = claim.get("layer_groups", {})
        if not groups.get("raw_model_native") or not groups.get("constrained_sampling") or not groups.get("postprocessing"):
            failures.append("q2 claim contract missing required layer groups")
        if "claim_provenance_contract" not in str(claim.get("shared_provenance_contract", "")):
            failures.append("q2 claim contract missing shared provenance contract reference")
        missing_raw_layers = {"raw_flow", "raw_rollout"} - set(groups.get("raw_model_native", []))
        missing_constrained_layers = {"constrained_flow"} - set(groups.get("constrained_sampling", []))
        missing_post_layers = {"repaired", "inferred_bond", "deterministic_proxy", "reranked"} - set(
            groups.get("postprocessing", [])
        )
        if missing_raw_layers:
            failures.append(f"q2 claim contract raw_model_native missing {sorted(missing_raw_layers)}")
        if missing_constrained_layers:
            failures.append(f"q2 claim contract constrained_sampling missing {sorted(missing_constrained_layers)}")
        if missing_post_layers:
            failures.append(f"q2 claim contract postprocessing missing {sorted(missing_post_layers)}")
        claim_text = json.dumps(claim, sort_keys=True)
        if "geometry-first flow matching" not in claim_text or "score_only" not in claim_text:
            failures.append("q2 claim contract missing geometry-first or score_only guardrail")
        if PLACEHOLDER_RE.search(claim_text):
            failures.append("q2 claim contract contains placeholder-like numeric text")

        audit = load_json_artifact(Q2_POSTPROCESSING_FAILURE_AUDIT)
        if "method_summaries" not in audit and "method_records" not in audit:
            failures.append("postprocessing failure audit missing method summaries")

    validate_todo_schema(failures)
    status = "pass" if not failures else "fail"
    return {
        "name": "q2 artifact schema validation",
        "command": ["internal", "validate_q2_artifacts"],
        "required": True,
        "status": status,
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }


def validate_claim_layer_row(row, source_key, failures):
    layer = row.get("layer")
    role = row.get("layer_role")
    if not layer:
        failures.append(f"{source_key} layer row missing layer")
        return
    if role is None:
        failures.append(f"{source_key} layer row missing layer_role for {layer}")
        return
    if layer in RAW_MODEL_NATIVE_LAYERS and role == "postprocessing":
        failures.append(f"{source_key} raw/native layer {layer} marked postprocessing")
    if layer in CONSTRAINED_SAMPLING_LAYERS and role == "raw_model_native":
        failures.append(f"{source_key} constrained layer {layer} promoted to raw_model_native")
    if role == "raw_model_native" and layer not in RAW_MODEL_NATIVE_LAYERS:
        failures.append(f"{source_key} non-native layer {layer} promoted to raw_model_native")
    if layer in POSTPROCESSING_LAYERS and role == "raw_model_native":
        failures.append(f"{source_key} postprocessed layer {layer} promoted to raw_model_native")
    if layer in POSTPROCESSING_LAYERS and row.get("model_improvement") is True:
        failures.append(f"{source_key} postprocessed layer {layer} marked model_improvement=true")


def layer_source_is_explicit(candidate):
    source_layer = candidate.get("source_layer")
    if isinstance(source_layer, str) and source_layer:
        return True
    provenance = candidate.get("transformation_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("source_layer"), str):
        return True
    return False


def layer_transform_is_explicit(candidate):
    provenance = candidate.get("transformation_provenance")
    if isinstance(provenance, dict):
        transforms = provenance.get("transformations")
        if isinstance(transforms, list):
            return True
    return False


def validate_layer_attribution():
    started = time.time()
    failures = []

    def check_candidate(candidate, expected_layer, source_key):
        for field in ("candidate_id", "method_id", "layer", "postprocessing_steps"):
            if field not in candidate:
                failures.append(f"{source_key} candidate missing {field}")
        if candidate.get("layer") != expected_layer:
            failures.append(f"{source_key} candidate layer mismatch: {candidate.get('layer')} != {expected_layer}")
        if not layer_source_is_explicit(candidate):
            failures.append(f"{source_key} candidate missing explicit source_layer provenance")
        if expected_layer not in RAW_MODEL_NATIVE_LAYERS and not layer_transform_is_explicit(candidate):
            failures.append(f"{source_key} candidate missing transformation_provenance")

    negative_fixture_failures = []
    validate_claim_layer_row(
        {
            "method_id": "fixture",
            "layer": "reranked",
            "layer_role": "raw_model_native",
            "model_improvement": True,
        },
        "negative_fixture",
        negative_fixture_failures,
    )
    if not negative_fixture_failures:
        failures.append("negative postprocessed promotion fixture was not rejected")

    ours_path = Path("checkpoints/q2_ours_public100/generation_layers_test.json")
    if ours_path.is_file():
        ours = load_json_artifact(ours_path)
        raw = ours.get("raw_flow_candidates", [])
        constrained = ours.get("constrained_flow_candidates", [])
        if not raw or not constrained:
            failures.append("q2 ours generation layers must contain raw_flow and constrained_flow candidates")
        raw_ids = {candidate.get("candidate_id") for candidate in raw}
        constrained_ids = {candidate.get("candidate_id") for candidate in constrained}
        if raw_ids & constrained_ids:
            failures.append("raw_flow and constrained_flow candidate ids must not collapse")
        for candidate in raw[:10]:
            check_candidate(candidate, "raw_flow", "raw_flow")
            if candidate.get("model_native") is not True:
                failures.append("raw_flow candidate must be model_native=true")
        for candidate in constrained[:10]:
            check_candidate(candidate, "constrained_flow", "constrained_flow")
            if candidate.get("model_native") is True:
                failures.append("constrained_flow candidate must not be model_native=true")
    else:
        failures.append("missing q2 ours generation_layers_test.json for layer attribution check")

    postprocessed_layers = {
        "repaired_candidates": "repaired",
        "reranked_candidates": "reranked",
        "centroid_only_candidates": "centroid_only",
        "clash_only_candidates": "clash_only",
        "bond_inference_only_candidates": "bond_inference_only",
        "full_repair_candidates": "full_repair",
    }
    for path in sorted(Path("checkpoints/q2_postprocessing_ablation").glob("*/generation_layers_test.json")):
        artifact = load_json_artifact(path)
        for source_key, layer in postprocessed_layers.items():
            for candidate in artifact.get(source_key, [])[:10]:
                check_candidate(candidate, layer, source_key)
                if candidate.get("model_native") is True:
                    failures.append(f"{source_key} candidate must not be model_native=true")
                if not candidate.get("postprocessing_steps"):
                    failures.append(f"{source_key} candidate missing non-empty postprocessing_steps")

    comparison = Path(Q2_OURS_VS_PUBLIC_BASELINES_SUMMARY)
    if comparison.is_file():
        methods = load_json_artifact(comparison).get("methods", [])
        for row in methods:
            validate_claim_layer_row(row, "q2 method comparison", failures)
        roles = {(row.get("layer"), row.get("layer_role")) for row in methods}
        if ("raw_flow", "raw_model_native") not in roles or ("constrained_flow", "constrained_sampling") not in roles:
            failures.append("method comparison must summarize raw_flow and constrained_flow separately")
        if not any(role == "postprocessing" for _layer, role in roles):
            failures.append("method comparison must keep postprocessing rows separate")
    else:
        failures.append("missing q2 method comparison summary for layer attribution check")

    ablation_summary = Path(Q2_POSTPROCESSING_ABLATION_SUMMARY)
    if ablation_summary.is_file():
        for row in load_json_artifact(ablation_summary).get("layer_summaries", []):
            validate_claim_layer_row(row, "q2 postprocessing ablation", failures)
    else:
        failures.append("missing q2 postprocessing ablation summary for layer attribution check")

    leaderboard = Path(Q3_MODEL_IMPROVEMENT_LEADERBOARD)
    if leaderboard.is_file():
        for row in load_json_artifact(leaderboard).get("rows", []):
            validate_claim_layer_row(row, "q3 model improvement leaderboard", failures)

    for path in ("README.md", "docs/q1_claim_contract.md", "docs/q2_claim_contract.md"):
        text = Path(path).read_text(encoding="utf-8")
        missing = REQUIRED_LAYER_BOUNDARY_TERMS - set(re.findall(r"`([^`]+)`", text))
        if missing:
            failures.append(f"{path} missing explicit layer boundary terms: {sorted(missing)}")

    entrypoint_path = Path("src/experiments/entrypoints.rs")
    if entrypoint_path.is_file():
        entrypoint_text = entrypoint_path.read_text(encoding="utf-8")
        missing = [
            term
            for term in GENERATION_ENTRYPOINT_BOUNDARY_TERMS
            if term not in entrypoint_text
        ]
        if missing:
            failures.append(
                "generation entrypoint missing raw/constrained boundary terms: "
                + ", ".join(sorted(missing))
            )
        if re.search(r"constrained_flow\"[^\n]{0,120}\"raw_model_native", entrypoint_text):
            failures.append(
                "generation entrypoint uses constrained_flow with raw_model_native in close proximity"
            )
    else:
        failures.append("src/experiments/entrypoints.rs missing for generation boundary scan")

    return {
        "name": "layer attribution integrity",
        "command": ["internal", "validate_layer_attribution"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }


def validate_backend_sanity():
    started = time.time()
    failures = []
    notes = []
    backend_jsons = [
        *Path("checkpoints/q2_ours_public100").glob("candidate_metrics_vina.json"),
        *Path("checkpoints/q2_ours_public100").glob("candidate_metrics_gnina.json"),
        *Path("checkpoints/q2_postprocessing_ablation").glob("candidate_metrics_vina.json"),
        *Path("checkpoints/q2_postprocessing_ablation").glob("candidate_metrics_gnina.json"),
    ]
    if not backend_jsons:
        failures.append("no q2 backend JSON reports found")
    for path in backend_jsons:
        payload = load_json_artifact(path)
        aggregate = payload.get("aggregate_metrics", {})
        rows = payload.get("candidate_metrics", [])
        if not rows:
            failures.append(f"{path} missing candidate_metrics rows")
            continue
        is_vina = "vina" in path.name
        coverage_key = "docking_score_coverage_fraction" if is_vina else "gnina_score_coverage_fraction"
        if coverage_key not in aggregate:
            failures.append(f"{path} missing {coverage_key}")
        extreme = 0
        for row in rows:
            metrics = row.get("metrics", {})
            if is_vina:
                success = metrics.get("vina_score_success_fraction")
                if success == 0.0 and "vina_failure_reason" not in metrics:
                    failures.append(f"{path} failed Vina row missing vina_failure_reason")
                score = metrics.get("vina_score")
            else:
                success = metrics.get("gnina_score_success_fraction")
                if success == 0.0 and "gnina_failure_reason" not in metrics:
                    failures.append(f"{path} failed GNINA row missing gnina_failure_reason")
                score = metrics.get("gnina_affinity")
            if isinstance(score, (int, float)) and score > 50.0:
                extreme += 1
        if extreme:
            notes.append(f"{path}: extreme_positive_score_review_count={extreme}")

    rdkit_jsonls = [
        Path(Q2_CANDIDATE_METRICS),
        Path("checkpoints/q2_postprocessing_ablation/merged/candidate_metrics_q1_public_full100_budget1.jsonl"),
    ]
    for path in rdkit_jsonls:
        if not path.is_file():
            failures.append(f"missing merged candidate metrics for RDKit sanity: {path}")
            continue
        checked = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metrics = row.get("metrics", {})
                if not any(key in metrics for key in ("qed", "sa_score", "logp")):
                    failures.append(f"{path} row missing RDKit drug-likeness metrics")
                    break
                checked += 1
                if checked >= 25:
                    break

    return {
        "name": "backend coverage and score sanity",
        "command": ["internal", "validate_backend_sanity"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join((notes + failures)[-80:]),
        "stderr_tail": "",
    }


def load_tool_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_coordinate_frame_consistency():
    started = time.time()
    failures = []
    prepare = load_tool_module("tools/prepare_docking_inputs.py", "prepare_docking_inputs_validation")
    pocket = load_tool_module("tools/pocket_contact_backend.py", "pocket_contact_backend_validation")
    vina = load_tool_module("tools/vina_score_backend.py", "vina_score_backend_validation")
    candidate = {
        "candidate_id": "fixture:raw_flow:x:0",
        "example_id": "fixture",
        "protein_id": "fixture",
        "method_id": "fixture",
        "layer": "raw_flow",
        "coords": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
        "atom_types": [0, 0],
        "coordinate_frame_origin": [10.0, -1.0, 0.5],
        "pocket_centroid": [0.0, 0.0, 0.0],
        "pocket_radius": 6.0,
    }
    expected_first = (11.0, 1.0, 3.5)
    if prepare.shifted_coords(candidate)[0] != expected_first:
        failures.append("prepare_docking_inputs shifted_coords does not apply origin exactly once")
    if pocket.parse_candidate_coords(candidate)[0] != expected_first:
        failures.append("pocket_contact_backend parse_candidate_coords does not apply origin exactly once")
    if pocket.parse_candidate_coords(candidate)[0] != prepare.shifted_coords(candidate)[0]:
        failures.append("prepare_docking_inputs and pocket_contact_backend disagree on active coordinate frame")
    box, reason = vina.docking_box(candidate)
    if reason is not None:
        failures.append(f"vina docking_box unexpectedly failed: {reason}")
    elif (
        abs(box["center"][0] - 11.0) > 1e-6
        or abs(box["center"][1] - 0.0) > 1e-6
        or abs(box["center"][2] - 2.0) > 1e-6
    ):
        failures.append("vina docking_box center is inconsistent with shifted ligand/pocket frame")
    omitted_origin = dict(candidate)
    omitted_origin["coordinate_frame_origin"] = [0.0, 0.0, 0.0]
    double_origin_first = tuple(expected_first[dim] + candidate["coordinate_frame_origin"][dim] for dim in range(3))
    if prepare.shifted_coords(omitted_origin)[0] == expected_first:
        failures.append("coordinate-frame fixture failed to catch omitted coordinate_frame_origin")
    if double_origin_first == expected_first:
        failures.append("coordinate-frame fixture failed to construct double-origin sentinel")

    damage_path = Path("configs/q3_repair_damage_cases.json")
    if damage_path.is_file():
        damage = load_json_artifact(damage_path)
        candidates = load_damage_case_candidates(damage)
        checked = 0
        for case in damage.get("worst_cases", [])[:10]:
            raw = candidates.get(case.get("raw_candidate_id"))
            layer = candidates.get(case.get("candidate_id"))
            if raw is None or layer is None:
                failures.append(f"q3 coordinate fixture missing candidate payload for {case.get('candidate_id')}")
                continue
            geometry = case.get("geometry_delta", {})
            raw_centroid = centroid_from_coords(prepare.shifted_coords(raw))
            layer_centroid = centroid_from_coords(prepare.shifted_coords(layer))
            centroid_shift = euclidean(raw_centroid, layer_centroid)
            expected_shift = geometry.get("centroid_shift")
            if isinstance(expected_shift, (int, float)) and abs(centroid_shift - expected_shift) > 1e-3:
                failures.append(
                    f"q3 fixture centroid shift mismatch for {case.get('candidate_id')}: {centroid_shift} != {expected_shift}"
                )
            raw_box, raw_reason = vina.docking_box(raw)
            layer_box, layer_reason = vina.docking_box(layer)
            if raw_reason or layer_reason:
                failures.append(f"q3 fixture docking box failed for {case.get('candidate_id')}: {raw_reason or layer_reason}")
            else:
                box_shift = euclidean(raw_box["center"], layer_box["center"])
                expected_box_shift = geometry.get("docking_box_center_shift")
                if isinstance(expected_box_shift, (int, float)) and abs(box_shift - expected_box_shift) > 1e-3:
                    failures.append(
                        f"q3 fixture docking box shift mismatch for {case.get('candidate_id')}: {box_shift} != {expected_box_shift}"
                    )
                raw_pocket_coords = pocket.parse_candidate_coords(raw)
                if raw_pocket_coords != prepare.shifted_coords(raw):
                    failures.append(f"q3 fixture backend coordinate-frame disagreement for {case.get('raw_candidate_id')}")
            if isinstance(expected_shift, (int, float)) and expected_shift > 2.0 and centroid_shift <= 2.0:
                failures.append(f"q3 fixture failed to enforce repair centroid-shift bound for {case.get('candidate_id')}")
            checked += 1
        if checked == 0:
            failures.append("q3 coordinate-frame fixture did not check any repair-damage cases")
    else:
        failures.append("missing q3 repair damage cases for coordinate-frame fixtures")

    ablation_path = Path(Q2_POSTPROCESSING_ABLATION_SUMMARY)
    if ablation_path.is_file():
        ablation = load_json_artifact(ablation_path)
        full = next((row for row in ablation.get("layer_summaries", []) if row.get("layer") == "full_repair"), None)
        if not full:
            failures.append("missing full_repair layer for centroid-shift regression")
        else:
            delta = full.get("delta_vs_no_repair", {}).get("centroid_offset")
            if not isinstance(delta, (int, float)) or abs(delta) < 1.0:
                failures.append("full_repair centroid shift regression expected a finite non-trivial delta")
    else:
        failures.append("missing q2 postprocessing ablation summary for centroid-shift regression")

    return {
        "name": "coordinate frame consistency",
        "command": ["internal", "validate_coordinate_frame_consistency"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }


def _is_smoke_artifact(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return "smoke" in normalized and path.suffix.lower() in {".json", ".jsonl"}


def _is_smoke_diagnostic_candidate(path: Path) -> bool:
    normalized = path.as_posix().lower()
    if normalized.startswith("configs/"):
        return False
    if path.name in {
        "claim_summary.json",
        "config.snapshot.json",
        "dataset_validation.json",
        "dataset_validation_report.json",
        "run_artifacts.json",
        "split_report.json",
    }:
        return False
    if path.name.startswith("generation_layers_"):
        return False
    return True


def _iter_smoke_artifacts() -> list[Path]:
    return sorted(
        path
        for path in Path(".").rglob("*")
        if path.is_file() and _is_smoke_artifact(path) and _is_smoke_diagnostic_candidate(path)
    )


def _flatten_json_keys(payload: dict | list, *, include_nested: bool = True) -> set[str]:
    keys: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str):
                    keys.add(key.lower())
                if include_nested:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return keys


SYNC_DIAGNOSTIC_FIELDS = {
    "mask_count_mismatch",
    "slot_count_mismatch",
    "coordinate_frame_mismatch",
    "stale_context_steps",
    "refresh_count",
    "batch_slice_sync_pass",
}

SYNC_MISMATCH_FIELDS = {
    "mask_count_mismatch",
    "slot_count_mismatch",
    "coordinate_frame_mismatch",
}


def _iter_sync_diagnostic_records(node):
    if isinstance(node, dict):
        lowered = {key.lower(): key for key in node if isinstance(key, str)}
        has_direct_sync_fields = any(field in lowered for field in SYNC_DIAGNOSTIC_FIELDS)
        if has_direct_sync_fields:
            yield node

        sync_key = lowered.get("synchronization")
        if sync_key is not None:
            sync_value = node.get(sync_key)
            if isinstance(sync_value, dict):
                yield sync_value
            else:
                yield {"synchronization": sync_value}

        for key, value in node.items():
            if isinstance(key, str) and key.lower() == "synchronization":
                continue
            yield from _iter_sync_diagnostic_records(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_sync_diagnostic_records(item)


def _numeric_sync_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _sync_record_failures(record, source: str) -> list[str]:
    if not isinstance(record, dict):
        return [f"{source} synchronization diagnostics must be a JSON object"]

    lowered = {key.lower(): key for key in record if isinstance(key, str)}
    present = set(lowered) & SYNC_DIAGNOSTIC_FIELDS
    missing = sorted(SYNC_DIAGNOSTIC_FIELDS - present)
    failures = []
    if missing:
        failures.append(f"{source} missing synchronization diagnostic fields: {', '.join(missing)}")

    for field in sorted(SYNC_MISMATCH_FIELDS & present):
        value = _numeric_sync_value(record[lowered[field]])
        if value is None:
            failures.append(f"{source} synchronization field {field} must be a finite scalar")
        elif value > 0.0:
            failures.append(f"{source} synchronization mismatch {field}={record[lowered[field]]}")

    pass_key = lowered.get("batch_slice_sync_pass")
    if pass_key is not None:
        value = record[pass_key]
        if isinstance(value, bool):
            if not value:
                failures.append(f"{source} synchronization mismatch batch_slice_sync_pass=false")
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            if float(value) <= 0.0:
                failures.append(f"{source} synchronization mismatch batch_slice_sync_pass={value}")
        else:
            failures.append(f"{source} synchronization field batch_slice_sync_pass must be boolean or scalar")

    for field in ("stale_context_steps", "refresh_count"):
        key = lowered.get(field)
        if key is not None and _numeric_sync_value(record[key]) is None:
            failures.append(f"{source} synchronization field {field} must be a finite scalar")

    return failures


def _update_smoke_artifact_categories(payload: object, source: str, categories: set[str]) -> list[str]:
    if not isinstance(payload, dict):
        return []

    failures = []
    flattened = _flatten_json_keys(payload)
    if "representation_diagnostics" in flattened or "semantic" in flattened:
        categories.add("semantic")
    if any(
        "interaction_diagnostics" in key
        or "interaction_mode" in key
        or "interaction_review" in key
        or "cross_modal" in key
        or "cross-modal" in key
        or "topo_from_" in key
        or "geo_from_" in key
        or "pocket_from_" in key
        or "gate_activation_mean" in key
        for key in flattened
    ):
        categories.add("interaction")
    if any(
        key == "primary_objective"
        or "objective" in key
        or "topology_specialization_score" in key
        or "geometry_specialization_score" in key
        or "pocket_specialization_score" in key
        for key in flattened
    ):
        categories.add("objective")

    sync_records = list(_iter_sync_diagnostic_records(payload))
    if "synchronization" in flattened or any(field in flattened for field in SYNC_DIAGNOSTIC_FIELDS):
        categories.add("synchronization")
    for index, record in enumerate(sync_records):
        categories.add("synchronization")
        failures.extend(_sync_record_failures(record, f"{source} sync[{index}]"))

    return failures


def _synthetic_sync_mismatch_fixture_detected() -> bool:
    categories = set()
    payload = {
        "semantic": {},
        "interaction_diagnostics": {},
        "primary_objective": "synthetic_sync_fixture",
        "synchronization": {
            "mask_count_mismatch": 1,
            "slot_count_mismatch": 0,
            "coordinate_frame_mismatch": 0,
            "stale_context_steps": 0,
            "refresh_count": 0,
            "batch_slice_sync_pass": False,
        },
    }
    failures = _update_smoke_artifact_categories(payload, "synthetic_sync_mismatch_fixture", categories)
    return "synchronization" in categories and any("mask_count_mismatch" in failure for failure in failures)


def _collect_smoke_artifact_diagnostic_categories(path: Path) -> tuple[set[str], int, list[str]]:
    categories = set()
    row_count = 0
    sync_failures = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                row = line.strip()
                if not row:
                    continue
                payload = json.loads(row)
                row_count += 1
                sync_failures.extend(
                    _update_smoke_artifact_categories(
                        payload,
                        f"{path.as_posix()}:{line_index + 1}",
                        categories,
                    )
                )
                # keep smoke diagnostics smoke-check lightweight.
                if row_count >= 50:
                    break
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            row_count = 1
            sync_failures.extend(
                _update_smoke_artifact_categories(payload, path.as_posix(), categories)
            )

    return categories, row_count, sync_failures


def validate_diagnostic_presence_for_smoke_artifacts():
    started = time.time()
    failures = []
    if not _synthetic_sync_mismatch_fixture_detected():
        failures.append("synthetic synchronization mismatch fixture was not detected")
    smoke_artifacts = _iter_smoke_artifacts()

    if not smoke_artifacts:
        return {
            "name": "smoke diagnostics presence",
            "command": ["internal", "validate_diagnostic_presence_for_smoke_artifacts"],
            "required": False,
            "status": "pass" if not failures else "fail",
            "returncode": 0 if not failures else 1,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": "\n".join(failures[-50:]) or "no smoke artifacts found (optional)",
            "stderr_tail": "",
        }

    required = {"semantic", "interaction", "objective", "synchronization"}
    observed: set[str] = set()
    diagnosed_artifacts = []
    for path in smoke_artifacts:
        try:
            categories, rows, sync_failures = _collect_smoke_artifact_diagnostic_categories(path)
        except json.JSONDecodeError as exc:
            failures.append(f"{path} contains invalid JSON ({exc})")
            continue
        except OSError as exc:
            failures.append(f"{path} not readable ({exc})")
            continue

        if rows and categories:
            diagnosed_artifacts.append((path.as_posix(), categories))
            observed.update(categories)
            if {"semantic", "interaction", "objective"} & categories and "synchronization" not in categories:
                failures.append(f"{path} missing synchronization diagnostics")
            failures.extend(sync_failures)

    if not diagnosed_artifacts:
        return {
            "name": "smoke diagnostics presence",
            "command": ["internal", "validate_diagnostic_presence_for_smoke_artifacts"],
            "required": False,
            "status": "pass" if not failures else "fail",
            "returncode": 0 if not failures else 1,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": "\n".join(failures[-50:]) or "no smoke diagnostics artifacts found (optional)",
            "stderr_tail": "",
        }

    missing = ", ".join(sorted(required - observed))
    if missing:
        failures.append(
            f"smoke diagnostic coverage incomplete; missing categories: {missing}; "
            + "; ".join(f"{path}={sorted(list(cats))}" for path, cats in diagnosed_artifacts)
        )

    return {
        "name": "smoke diagnostics presence",
        "command": ["internal", "validate_diagnostic_presence_for_smoke_artifacts"],
        "required": False,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }


def load_damage_case_candidates(damage):
    layer_fields = {
        "no_repair": "no_repair_candidates",
        "centroid_only": "centroid_only_candidates",
        "clash_only": "clash_only_candidates",
        "bond_inference_only": "bond_inference_only_candidates",
        "full_repair": "full_repair_candidates",
    }
    paths = sorted(
        {
            source
            for case in damage.get("worst_cases", [])
            for source in (
                case.get("source_artifacts", {}).get("raw_generation_layer"),
                case.get("source_artifacts", {}).get("layer_generation_layer"),
            )
            if source
        }
    )
    candidates = {}
    for source in paths:
        path = Path(source)
        if not path.is_file():
            continue
        artifact = load_json_artifact(path)
        for _layer, field in layer_fields.items():
            for candidate in artifact.get(field, []):
                candidate_id = candidate.get("candidate_id")
                if candidate_id:
                    candidates[candidate_id] = candidate
    return candidates


def centroid_from_coords(coords):
    if not coords:
        return (0.0, 0.0, 0.0)
    count = float(len(coords))
    return tuple(sum(coord[dim] for coord in coords) / count for dim in range(3))


def euclidean(left, right):
    return math.sqrt(sum((float(left[dim]) - float(right[dim])) ** 2 for dim in range(3)))


def validate_method_comparison_regression():
    started = time.time()
    failures = []
    with tempfile.TemporaryDirectory(prefix="q2_method_comparison_fixture_") as tmpdir:
        tmp = Path(tmpdir)
        metrics_path = tmp / "candidate_metrics.jsonl"
        rows = [
            {
                "candidate_id": "flow_matching:raw_flow:fixture:0",
                "example_id": "fixture",
                "protein_id": "fixture",
                "split_label": "fixture_split",
                "method_id": "flow_matching",
                "layer": "raw_flow",
                "metrics": {
                    "vina_score": -6.0,
                    "vina_score_success_fraction": 1.0,
                    "gnina_affinity": -6.2,
                    "gnina_cnn_score": 0.55,
                    "qed": 0.4,
                    "sa_score": 2.5,
                    "pocket_contact_fraction": 0.8,
                    "clash_fraction": 0.0,
                    "centroid_offset": 2.0,
                },
            },
            {
                "candidate_id": "flow_matching:constrained_flow:fixture:0",
                "example_id": "fixture",
                "protein_id": "fixture",
                "split_label": "fixture_split",
                "method_id": "flow_matching",
                "layer": "constrained_flow",
                "metrics": {
                    "vina_score_success_fraction": 0.0,
                    "gnina_cnn_score": 0.25,
                    "qed": 0.3,
                    "sa_score": 3.1,
                },
            },
        ]
        metrics_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
        coverage_path = tmp / "coverage.json"
        coverage_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "split_reports": {
                        "fixture_split": {
                            "candidate_count": 2,
                            "vina_score_coverage_fraction": 0.5,
                            "gnina_affinity_coverage_fraction": 0.5,
                            "gnina_cnn_score_coverage_fraction": 1.0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        outputs = []
        for index in range(2):
            summary = tmp / f"summary_{index}.json"
            table = tmp / f"table_{index}.md"
            runtime = tmp / f"runtime_{index}.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/public_baseline_method_comparison.py",
                    "--candidate-metrics",
                    str(metrics_path),
                    "--coverage-json",
                    str(coverage_path),
                    "--summary-json",
                    str(summary),
                    "--table-md",
                    str(table),
                    "--runtime-md",
                    str(runtime),
                    "--split-label",
                    "fixture_split",
                    "--status",
                    "fixture",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                failures.append(f"method comparison fixture failed: {completed.stderr[-500:]}")
                break
            outputs.append((summary.read_text(encoding="utf-8"), table.read_text(encoding="utf-8")))
        if len(outputs) == 2 and outputs[0] != outputs[1]:
            failures.append("method comparison fixture output is not deterministic")
        if outputs and "NA" not in outputs[0][1]:
            failures.append("method comparison fixture did not render missing metrics as NA")

    ablation = load_json_artifact(Q2_POSTPROCESSING_ABLATION_SUMMARY)
    if not ablation.get("layer_summaries") or not ablation.get("diagnosis"):
        failures.append("postprocessing ablation summary missing regression-critical sections")

    return {
        "name": "method comparison and postprocessing ablation regression",
        "command": ["internal", "validate_method_comparison_regression"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }


def validate_q3_non_degradation_gate():
    started = time.time()
    completed = subprocess.run(
        [
            sys.executable,
            "tools/claim_regression_gate.py",
            "--q3-non-degradation-gate",
            Q3_NON_DEGRADATION_GATE,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "name": "q3 non-degradation gate",
        "command": [
            sys.executable,
            "tools/claim_regression_gate.py",
            "--q3-non-degradation-gate",
            Q3_NON_DEGRADATION_GATE,
        ],
        "required": True,
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def validate_flow_head_status_claims():
    started = time.time()
    failures = []
    atom_pocket_configs = []
    for path in sorted(Path("configs").glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "atom_pocket_cross_attention" in text:
            atom_pocket_configs.append(path.as_posix())

    if not atom_pocket_configs:
        failures.append("no config declares atom_pocket_cross_attention flow velocity head")

    target_paths = [
        Path("docs/q2_architecture_audit.md"),
        Path("docs/q2_conditioning_compression_audit.md"),
        Path("README.md"),
        Path("configs/q2_architecture_audit.json"),
        Path("configs/q2_conditioning_compression_audit.json"),
        Path("configs/q2_flow_head_v2_integration_plan.json"),
        Path("configs/atom_to_pocket_cross_attention_velocity_head_plan.json"),
    ]
    stale_patterns = [
        "prototype-only",
        '"status": "prototype_only"',
        "not wired into Phase1ResearchSystem",
        "not wired as the active system head",
        "flow_matching.velocity_head",
        "local_pocket_cross_attention",
        "geometry_baseline",
    ]
    combined = []
    for path in target_paths:
        if not path.is_file():
            failures.append(f"missing flow-head status target: {path.as_posix()}")
            continue
        text = path.read_text(encoding="utf-8")
        combined.append(text)
        for pattern in stale_patterns:
            if pattern in text:
                failures.append(f"{path.as_posix()} contains stale flow-head status phrase `{pattern}`")

    combined_text = "\n".join(combined).lower()
    for required in (
        "config-selectable",
        "atom_pocket_cross_attention",
        "geometry remains the default",
        "coordinate velocity",
    ):
        if required not in combined_text:
            failures.append(f"flow-head status docs missing `{required}`")
    if "topology flow" not in combined_text or "bond flow" not in combined_text:
        failures.append("flow-head status docs must keep topology and bond flow explicitly unclaimed")
    metrics_source = Path("src/experiments/unseen_pocket/metrics.rs")
    if metrics_source.is_file():
        metrics_text = metrics_source.read_text(encoding="utf-8")
        for token in ("FlowHeadAblationDiagnostics", "flow_head_diagnostics"):
            if token not in metrics_text:
                failures.append(f"experiment metrics schema missing `{token}`")
    else:
        failures.append("missing src/experiments/unseen_pocket/metrics.rs")

    ablation_config_path = Path("configs/q3_pairwise_geometry_ablation_config.json")
    if ablation_config_path.is_file():
        ablation = load_json_artifact(ablation_config_path)
        variant_ids = {
            row.get("id")
            for row in ablation.get("variants", [])
            if isinstance(row, dict)
        }
        required_variants = {
            "no_pairwise",
            "local_pocket_attention",
            "pairwise_distance_direction",
            "pairwise_plus_local_pocket",
        }
        missing = sorted(required_variants - variant_ids)
        if missing:
            failures.append(
                "q3 pairwise/local-pocket ablation config missing variants: "
                + ", ".join(missing)
            )
    else:
        failures.append("missing configs/q3_pairwise_geometry_ablation_config.json")

    return {
        "name": "flow-head status claims",
        "command": ["internal", "validate_flow_head_status_claims"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-80:]),
        "stderr_tail": "",
    }


def main(argv):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    checks = build_checks(args.mode)
    results = [run_command(name, command, args.timeout, required) for name, command, required in checks]
    results.append(validate_validation_manifest())
    results.append(validate_drug_metric_contract())
    results.append(validate_q2_artifacts())
    results.append(validate_layer_attribution())
    results.append(architecture_validation.validate_modular_architecture_boundaries())
    results.append(architecture_validation.validate_architecture_module_map())
    results.append(validate_artifact_retention_policy())
    results.append(validate_backend_sanity())
    results.append(validate_coordinate_frame_consistency())
    results.append(validate_diagnostic_presence_for_smoke_artifacts())
    results.append(claim_provenance_validation.validate_provenance_safe_pharmacology_claims())
    results.append(negative_fixture_validation.validate_negative_fixtures())
    results.append(validate_method_comparison_regression())
    results.append(training_replay_validation.validate_training_replay_contract())
    results.append(validate_flow_head_status_claims())
    if Path(Q3_NON_DEGRADATION_GATE).is_file():
        results.append(validate_q3_non_degradation_gate())
    report = {
        "schema_version": 1,
        "tool": "validation_suite",
        "generated_report": True,
        "generated_by": "tools/validation_suite.py",
        "mode": args.mode,
        "strict": args.strict,
        "summary": validation_reporting.summarize(results),
        "checks": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_reporting.write_markdown(args.output_md, report)
    return 1 if args.strict and report["summary"]["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
