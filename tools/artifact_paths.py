#!/usr/bin/env python3
"""Resolve generated research artifact paths from repository manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_PATH = Path("configs/drug_metric_artifact_manifest.json")
DEFAULT_RETENTION_MANIFEST_PATH = Path("configs/artifact_retention_manifest.json")

DEFAULT_ARTIFACTS = {
    "q2.candidate_metrics": "artifacts/evidence/q2/candidate_metrics_q2_ours_public100.jsonl",
    "q2.proxy_backend_correlation": "artifacts/evidence/q2/q2_proxy_backend_correlation.json",
    "q2.postprocessing_failure_audit": "artifacts/evidence/q2/postprocessing_failure_audit.json",
    "q2.claim_contract": "configs/q2_claim_contract.json",
    "q2.ours_vs_public_baselines_summary": "configs/q2_ours_vs_public_baselines_summary.json",
    "q2.postprocessing_ablation_summary": "configs/q2_postprocessing_ablation_summary.json",
    "q3.rotation_consistency_report": "configs/q3_rotation_consistency_report.json",
    "q3.model_improvement_leaderboard": "configs/q3_model_improvement_leaderboard.json",
    "q3.non_degradation_gate": "configs/q3_non_degradation_gate.json",
}


def _validation_failures(raw: dict[str, Any], entry: dict[str, Any], *, path: str) -> list[str]:
    failures = []
    if "artifact_role" not in raw or not isinstance(raw.get("artifact_role"), str) or not raw["artifact_role"]:
        failures.append(f"{path}: missing artifact_role")
    if "provenance_command" not in raw or not isinstance(raw.get("provenance_command"), str):
        failures.append(f"{path}: provenance_command must be a string")
    if "required" not in raw or not isinstance(raw.get("required"), bool):
        failures.append(f"{path}: required must be a boolean")
    if "producer" not in raw or not isinstance(raw.get("producer"), str) or not raw["producer"]:
        failures.append(f"{path}: producer must be a non-empty string")
    validation_consumer = raw.get("validation_consumer")
    if (
        "validation_consumer" not in raw
        or not isinstance(validation_consumer, list)
        or not all(
        isinstance(item, str) and item for item in validation_consumer
        )
    ):
        failures.append(f"{path}: validation_consumer must be a non-empty list of strings")
    return failures


def _coerce_artifact_entry(raw: Any, phase: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Return normalized artifact entry and validation failures."""
    failures = []
    if isinstance(raw, str):
        return {
            "path": raw,
            "artifact_role": None,
            "provenance_command": "",
            "required": False,
            "producer": "",
            "validation_consumer": [],
            "phase": phase,
        }, failures

    if not isinstance(raw, dict):
        failures.append("artifact entry is not a string or object")
        return {"path": ""}, failures

    path = raw.get("path") or raw.get("artifact_path")
    if not isinstance(path, str) or not path:
        failures.append("artifact path must be a non-empty string")
        return {"path": ""}, failures

    normalized = {
        "path": path,
        "artifact_role": raw.get("artifact_role"),
        "provenance_command": raw.get("provenance_command", ""),
        "required": raw.get("required", False),
        "producer": raw.get("producer", ""),
        "validation_consumer": raw.get("validation_consumer", []),
        "phase": phase or raw.get("phase"),
    }
    failures.extend(_validation_failures(raw, normalized, path=f"{path}"))
    return normalized, failures


def _coerce_legacy_artifacts(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts = []
    failures = []
    for phase, entries in manifest.get("phase_artifacts", {}).items():
        if not isinstance(entries, list):
            failures.append(f"phase_artifacts[{phase}] must be a list")
            continue
        for index, raw in enumerate(entries):
            normalized, entry_failures = _coerce_artifact_entry(raw, phase=phase)
            failures.extend(f"phase_artifacts[{phase}][{index}]: {failure}" for failure in entry_failures)
            path = normalized.get("path")
            if path:
                artifacts.append(normalized)
    return artifacts, failures


def load_drug_metric_manifest(path: str | Path = DEFAULT_MANIFEST_PATH):
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_artifact_retention_manifest(path: str | Path = DEFAULT_RETENTION_MANIFEST_PATH):
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_retention_records(
    path: str | Path = DEFAULT_RETENTION_MANIFEST_PATH,
) -> list[dict[str, Any]]:
    manifest = load_artifact_retention_manifest(path)
    records = manifest.get("artifact_families", []) if isinstance(manifest, dict) else []
    return [record for record in records if isinstance(record, dict)]


def artifact_default_root(
    classification: str,
    path: str | Path = DEFAULT_RETENTION_MANIFEST_PATH,
) -> str | None:
    manifest = load_artifact_retention_manifest(path)
    policies = manifest.get("default_policies", {}) if isinstance(manifest, dict) else {}
    policy = policies.get(classification, {}) if isinstance(policies, dict) else {}
    root = policy.get("default_root") if isinstance(policy, dict) else None
    return root if isinstance(root, str) and root else None


def validate_artifact_retention_manifest(
    path: str | Path = DEFAULT_RETENTION_MANIFEST_PATH,
) -> list[str]:
    manifest = load_artifact_retention_manifest(path)
    failures = []
    if not manifest:
        return [f"{path}: manifest is missing or unreadable"]

    policies = manifest.get("default_policies")
    if not isinstance(policies, dict):
        failures.append("artifact retention manifest missing default_policies object")
        policies = {}

    required_classes = {
        "source_config",
        "generated_checkpoint",
        "generated_evidence",
        "generated_report",
        "curated_reviewer_artifact",
    }
    missing_classes = sorted(required_classes - set(policies))
    if missing_classes:
        failures.append(
            "artifact retention manifest missing default policies: "
            + ", ".join(missing_classes)
        )

    records = manifest.get("artifact_families")
    if not isinstance(records, list) or not records:
        failures.append("artifact retention manifest missing non-empty artifact_families list")
        records = []

    classifications = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"artifact_families[{index}] must be an object")
            continue
        for field in ("id", "classification", "paths", "tracking_policy", "retention_reason"):
            if field not in record:
                failures.append(f"artifact_families[{index}] missing {field}")
        classification = record.get("classification")
        if isinstance(classification, str):
            classifications.add(classification)
            if classification not in required_classes:
                failures.append(
                    f"artifact_families[{index}] unsupported classification: {classification}"
                )
        paths = record.get("paths")
        if not isinstance(paths, list) or not all(isinstance(item, str) and item for item in paths):
            failures.append(f"artifact_families[{index}] paths must be a non-empty list of strings")

    missing_record_classes = sorted(required_classes - classifications)
    if missing_record_classes:
        failures.append(
            "artifact retention manifest missing artifact family classifications: "
            + ", ".join(missing_record_classes)
        )
    return failures


def validate_drug_metric_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> list[str]:
    manifest = load_drug_metric_manifest(path)
    failures = []
    if not manifest:
        failures.append(f"{path}: manifest is missing or unreadable")
        return failures

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        failures.append("manifest.artifacts must be a list")
        artifacts = []
    for index, raw in enumerate(artifacts):
        normalized, entry_failures = _coerce_artifact_entry(raw)
        failures.extend(f"artifacts[{index}] {item}" for item in entry_failures)
        if not normalized.get("path"):
            continue
    legacy_artifacts, legacy_failures = _coerce_legacy_artifacts(manifest)
    failures.extend(legacy_failures)
    return failures


def drug_metric_artifact_records(path: str | Path = DEFAULT_MANIFEST_PATH) -> list[dict[str, Any]]:
    manifest = load_drug_metric_manifest(path)
    if not manifest:
        return []

    records = []
    for raw in manifest.get("artifacts", []):
        normalized, _ = _coerce_artifact_entry(raw)
        if normalized.get("path"):
            records.append(normalized)

    legacy, _ = _coerce_legacy_artifacts(manifest)
    records.extend(legacy)
    return records


def drug_metric_artifact_path(key: str, manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> str:
    records = drug_metric_artifact_records(manifest_path)
    for record in records:
        if record.get("artifact_role") == key:
            return record["path"]

    if key in DEFAULT_ARTIFACTS:
        return DEFAULT_ARTIFACTS[key]

    phase, _, artifact_name = key.partition(".")
    for record in records:
        if record.get("phase") != phase:
            continue
        path = record.get("path", "")
        if not isinstance(path, str):
            continue
        if artifact_name == "candidate_metrics" and path.endswith(".jsonl"):
            return path
        if artifact_name == "proxy_backend_correlation" and "proxy_backend_correlation" in path:
            return path
        if artifact_name == "postprocessing_failure_audit" and "postprocessing_failure_audit" in path:
            return path
    if key in DEFAULT_ARTIFACTS:
        return DEFAULT_ARTIFACTS[key]
    raise KeyError(f"artifact role not found in manifest or defaults: {key}")
