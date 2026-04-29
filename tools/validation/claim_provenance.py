"""Claim-provenance checks for pharmacology and chemistry evidence."""

import json
import math
import re
import time
from pathlib import Path


CLAIM_PROVENANCE_FALLBACK_VALUES = {
    "heuristic",
    "backend_supported",
    "docking_supported",
    "experimental",
    "unavailable",
}

CHEMISTRY_COLLABORATION_METRIC_FIELDS = {
    "pharmacophore_role_coverage",
    "role_conflict_rate",
    "severe_clash_fraction",
    "valence_violation_fraction",
    "bond_length_guardrail_mean",
    "key_residue_contact_coverage",
}

PHARMACOLOGY_PROMOTION_RE = re.compile(
    r"\b("
    r"docking[- ]supported|docking[- ]backed|"
    r"experimental[- ]supported|experimental[- ]backed|"
    r"experimental\s+(?:affinity|evidence|outcome)|"
    r"affinity\s+evidence|selectivity|efficacy"
    r")\b",
    re.IGNORECASE,
)

STRONG_DOCKING_METRIC_RE = re.compile(
    r"(^|[_|])(vina_score|gnina_affinity|docking_score|docking_like_score)([_|]|$)",
    re.IGNORECASE,
)


def load_json_artifact(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _claim_artifact_paths() -> list[Path]:
    return sorted(Path("checkpoints").rglob("claim_summary.json"))


def _iter_json_strings(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if isinstance(key, str) else path
            yield from _iter_json_strings(value, child_path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_json_strings(value, f"{path}[{index}]")
    elif isinstance(node, str):
        yield path, node


def _lower_text(value) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _claim_provenance_contract():
    path = Path("configs/drug_metric_artifact_manifest.json")
    if not path.is_file():
        return {}, [f"missing shared claim provenance manifest: {path}"]
    try:
        payload = load_json_artifact(path)
    except Exception as exc:
        return {}, [f"unable to read shared claim provenance manifest {path}: {exc}"]
    contract = payload.get("claim_provenance_contract")
    if not isinstance(contract, dict):
        return {}, [f"{path} missing claim_provenance_contract"]
    return contract, []


def _claim_provenance_values():
    contract, failures = _claim_provenance_contract()
    if failures:
        return set(CLAIM_PROVENANCE_FALLBACK_VALUES)
    values = {
        _lower_text(item.get("value"))
        for item in contract.get("values", [])
        if isinstance(item, dict) and item.get("value")
    }
    return values or set(CLAIM_PROVENANCE_FALLBACK_VALUES)


def _is_finite_or_null(value) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


def _metric_status_text(metric: dict) -> str:
    return _lower_text(metric.get("status"))


def _backend_examples_scored(section: dict) -> float:
    metrics = section.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get("backend_examples_scored")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _backend_section_provenance(section: dict) -> str:
    name = _lower_text(section.get("backend_name"))
    status = _lower_text(section.get("status"))
    if "heuristic" in name or "heuristic" in status:
        return "heuristic"
    if (
        "external" in name
        or "external" in status
        or "backend" in status
        or _backend_examples_scored(section) > 0.0
    ):
        return "backend_supported"
    return "unavailable"


def _backend_runtime_available(payload: dict, logical_name: str) -> bool:
    environment = payload.get("backend_environment")
    if not isinstance(environment, dict):
        return False
    key_by_logical_name = {
        "chemistry_validity": "chemistry_backend",
        "docking_affinity": "docking_backend",
        "pocket_compatibility": "pocket_backend",
    }
    report = environment.get(key_by_logical_name.get(logical_name, ""))
    return isinstance(report, dict) and report.get("runtime_available") is True


def _claim_has_support_for_provenance(payload: dict, provenance: str) -> bool:
    if provenance == "heuristic":
        return True
    if provenance == "unavailable":
        return True
    if provenance == "docking_supported":
        backend_metrics = payload.get("backend_metrics") or {}
        docking = backend_metrics.get("docking_affinity") or {}
        return (
            _backend_section_provenance(docking) == "backend_supported"
            and (
                _backend_runtime_available(payload, "docking_affinity")
                or _backend_examples_scored(docking) > 0.0
            )
        )
    if provenance == "backend_supported":
        backend_metrics = payload.get("backend_metrics") or {}
        for logical_name in ("chemistry_validity", "pocket_compatibility"):
            section = backend_metrics.get(logical_name) or {}
            if (
                _backend_section_provenance(section) == "backend_supported"
                and (
                    _backend_runtime_available(payload, logical_name)
                    or _backend_examples_scored(section) > 0.0
                )
            ):
                return True
        return False
    if provenance == "experimental":
        text = json.dumps(payload, sort_keys=True).lower()
        return "experimental" in text and "experimental_evidence" in text
    return False


def _validate_metric_provenance_record(metric, source: str, failures: list[str], *, required: bool = True):
    if not isinstance(metric, dict):
        failures.append(f"{source} must be an object with value/provenance/status")
        return
    if "value" not in metric:
        failures.append(f"{source} missing value")
    elif not _is_finite_or_null(metric.get("value")):
        failures.append(f"{source}.value must be finite or null")
    if "provenance" not in metric:
        if required:
            failures.append(f"{source} missing provenance")
        return
    provenance = _lower_text(metric.get("provenance"))
    if provenance not in _claim_provenance_values():
        failures.append(f"{source}.provenance has unsupported value `{metric.get('provenance')}`")
        return
    if "status" not in metric or not isinstance(metric.get("status"), str):
        failures.append(f"{source} missing status text")

    value = metric.get("value")
    if value is None and provenance != "unavailable":
        failures.append(f"{source} has null value but provenance `{provenance}` instead of unavailable")
    if value is not None and provenance == "unavailable":
        failures.append(f"{source} has a value but provenance=unavailable")

    status = _metric_status_text(metric)
    if provenance in {"backend_supported", "docking_supported", "experimental"} and "heuristic" in status:
        failures.append(f"{source} promotes heuristic status to provenance={provenance}")


def _validate_chemistry_collaboration_provenance(payload: dict, source: str, failures: list[str]):
    collaboration = payload.get("chemistry_collaboration")
    if collaboration is None:
        return
    if not isinstance(collaboration, dict):
        failures.append(f"{source}.chemistry_collaboration must be an object")
        return

    for field in sorted(CHEMISTRY_COLLABORATION_METRIC_FIELDS):
        if field in collaboration:
            metric = collaboration[field]
            metric_source = f"{source}.chemistry_collaboration.{field}"
            _validate_metric_provenance_record(metric, metric_source, failures)
            if isinstance(metric, dict):
                provenance = _lower_text(metric.get("provenance"))
                if provenance in {"backend_supported", "docking_supported", "experimental"}:
                    if not _claim_has_support_for_provenance(payload, provenance):
                        failures.append(
                            f"{metric_source} uses provenance={provenance} without supporting backend or experimental evidence"
                        )

    gates = collaboration.get("gate_usage_by_chemical_role", [])
    if gates is None:
        gates = []
    if not isinstance(gates, list):
        failures.append(f"{source}.chemistry_collaboration.gate_usage_by_chemical_role must be a list")
        return
    for index, row in enumerate(gates):
        if not isinstance(row, dict):
            failures.append(f"{source}.chemistry_collaboration.gate_usage_by_chemical_role[{index}] must be an object")
            continue
        _validate_metric_provenance_record(
            row.get("gate_mean"),
            f"{source}.chemistry_collaboration.gate_usage_by_chemical_role[{index}].gate_mean",
            failures,
        )


def _validate_backend_metric_provenance(payload: dict, source: str, failures: list[str]):
    backend_metrics = payload.get("backend_metrics")
    if not isinstance(backend_metrics, dict):
        return
    claim_context = payload.get("claim_context") or {}
    real_backend_backed = claim_context.get("real_backend_backed") is True

    for logical_name, section in backend_metrics.items():
        if not isinstance(section, dict):
            failures.append(f"{source}.backend_metrics.{logical_name} must be an object")
            continue
        available = section.get("available") is True
        metrics = section.get("metrics") if isinstance(section.get("metrics"), dict) else {}
        if available or metrics:
            if "backend_name" not in section:
                failures.append(f"{source}.backend_metrics.{logical_name} missing backend_name provenance")
            if "status" not in section or not isinstance(section.get("status"), str):
                failures.append(f"{source}.backend_metrics.{logical_name} missing status provenance")

        name = _lower_text(section.get("backend_name"))
        status = _lower_text(section.get("status"))
        backend_provenance = _backend_section_provenance(section)
        if backend_provenance == "backend_supported":
            coverage_fields = {
                "backend_examples_scored",
                "candidate_metric_rows",
                "backend_missing_structure_fraction",
            }
            if not any(field in metrics for field in coverage_fields):
                failures.append(f"{source}.backend_metrics.{logical_name} missing backend coverage/status metrics")
            for field in coverage_fields:
                if field in metrics and not _is_finite_or_null(metrics.get(field)):
                    failures.append(f"{source}.backend_metrics.{logical_name}.{field} must be finite or null")
        if "heuristic" in name and any(token in status for token in ("external", "docking-supported", "experimental")):
            failures.append(
                f"{source}.backend_metrics.{logical_name} labels heuristic backend `{section.get('backend_name')}` as `{section.get('status')}`"
            )
        if real_backend_backed and logical_name in {"chemistry_validity", "pocket_compatibility"}:
            if backend_provenance == "heuristic":
                failures.append(f"{source}.claim_context.real_backend_backed=true but {logical_name} is heuristic-only")

        if logical_name == "docking_affinity":
            for metric_name, value in metrics.items():
                if not isinstance(metric_name, str) or not STRONG_DOCKING_METRIC_RE.search(metric_name):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    if backend_provenance == "heuristic":
                        failures.append(
                            f"{source}.backend_metrics.docking_affinity.{metric_name} is claim-grade docking evidence from a heuristic backend"
                        )


def _validate_claim_text_provenance(payload: dict, source: str, failures: list[str]):
    claim_context = payload.get("claim_context") or {}
    heuristic_only = claim_context.get("real_backend_backed") is not True
    if not heuristic_only:
        return

    for path, value in _iter_json_strings(payload):
        text = value.lower()
        if not PHARMACOLOGY_PROMOTION_RE.search(text):
            continue
        if "unavailable" in text or "before" in text or "future" in text or "requires" in text:
            continue
        failures.append(f"{source}{path} promotes heuristic-only evidence with text `{value[:160]}`")


def _validate_pharmacology_claim_provenance_payload(payload: object, source: str) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{source} must contain a JSON object"]
    failures: list[str] = []
    _validate_chemistry_collaboration_provenance(payload, source, failures)
    _validate_backend_metric_provenance(payload, source, failures)
    _validate_claim_text_provenance(payload, source, failures)
    return failures


def pharmacology_claim_provenance_failures(payload: object, source: str) -> list[str]:
    """Return claim-provenance failures for one parsed claim summary payload."""
    return _validate_pharmacology_claim_provenance_payload(payload, source)


def _validate_shared_claim_provenance_contract() -> list[str]:
    contract, failures = _claim_provenance_contract()
    if failures:
        return failures
    values = {
        _lower_text(item.get("value"))
        for item in contract.get("values", [])
        if isinstance(item, dict) and item.get("value")
    }
    missing = CLAIM_PROVENANCE_FALLBACK_VALUES - values
    if missing:
        failures.append(f"claim provenance contract missing values: {sorted(missing)}")
    text = json.dumps(contract, sort_keys=True).lower()
    for token in ("heuristic_to_docking_supported", "heuristic_to_experimental", "unavailable_to_zero"):
        if token not in text:
            failures.append(f"claim provenance contract missing forbidden promotion `{token}`")
    if "value=null" not in text:
        failures.append("claim provenance contract must require unavailable evidence to use value=null")
    return failures


def _synthetic_provenance_mislabel_fixture_detected() -> bool:
    payload = {
        "claim_context": {
            "real_backend_backed": False,
            "evidence_mode": "heuristic-only held-out pocket evidence",
        },
        "chemistry_collaboration": {
            "pharmacophore_role_coverage": {
                "value": 0.8,
                "provenance": "docking_supported",
                "status": "heuristic topology-pocket pharmacophore diagnostic",
            },
            "key_residue_contact_coverage": {
                "value": None,
                "provenance": "unavailable",
                "status": "residue identities unavailable",
            },
        },
        "backend_metrics": {
            "docking_affinity": {
                "available": True,
                "backend_name": "heuristic_docking_hook_v1",
                "metrics": {"docking_like_score": 0.9},
                "status": "external docking backend on modular rollout candidates",
            }
        },
        "interpretation": "This is docking-supported evidence.",
    }
    failures = _validate_pharmacology_claim_provenance_payload(
        payload,
        "synthetic_provenance_mislabel_fixture",
    )
    return any("heuristic" in failure and ("docking" in failure or "provenance" in failure) for failure in failures)


def _synthetic_unavailable_metric_fixture_detected() -> bool:
    payload = {
        "claim_context": {
            "real_backend_backed": False,
            "evidence_mode": "heuristic-only held-out pocket evidence",
        },
        "chemistry_collaboration": {
            "key_residue_contact_coverage": {
                "value": 0.0,
                "provenance": "unavailable",
                "status": "residue identities unavailable",
            }
        },
    }
    failures = _validate_pharmacology_claim_provenance_payload(
        payload,
        "synthetic_unavailable_metric_fixture",
    )
    return any("provenance=unavailable" in failure for failure in failures)


def validate_provenance_safe_pharmacology_claims():
    started = time.time()
    failures = _validate_shared_claim_provenance_contract()
    if not _synthetic_provenance_mislabel_fixture_detected():
        failures.append("synthetic pharmacology provenance mislabel fixture was not detected")
    if not _synthetic_unavailable_metric_fixture_detected():
        failures.append("synthetic unavailable metric fixture was not detected")

    for path in _claim_artifact_paths():
        try:
            payload = load_json_artifact(path)
        except json.JSONDecodeError as exc:
            failures.append(f"{path} contains invalid JSON ({exc})")
            continue
        except OSError as exc:
            failures.append(f"{path} not readable ({exc})")
            continue
        failures.extend(_validate_pharmacology_claim_provenance_payload(payload, path.as_posix()))

    return {
        "name": "provenance-safe pharmacology claims",
        "command": ["internal", "validate_provenance_safe_pharmacology_claims"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-80:]),
        "stderr_tail": "",
    }
