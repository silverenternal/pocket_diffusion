"""Training-replay contract checks split out from `validation_suite.py`."""

import json
import time
from pathlib import Path


def _training_replay_report(summary, checkpoint):
    mismatches = []

    def add(field, expected, observed, replay_blocking=True, evidence_blocking=True):
        if expected == observed:
            return
        mismatches.append(
            {
                "field": field,
                "expected": "<missing>" if expected is None else json.dumps(expected, sort_keys=True),
                "observed": "<missing>" if observed is None else json.dumps(observed, sort_keys=True),
                "replay_blocking": replay_blocking,
                "evidence_blocking": evidence_blocking,
            }
        )

    reproducibility = summary.get("reproducibility") or {}
    add("config_hash", reproducibility.get("config_hash"), checkpoint.get("config_hash"))
    add(
        "dataset_validation_fingerprint",
        reproducibility.get("dataset_validation_fingerprint"),
        checkpoint.get("dataset_validation_fingerprint"),
    )
    add(
        "metric_schema_version",
        reproducibility.get("metric_schema_version"),
        checkpoint.get("metric_schema_version"),
    )

    expected_controls = reproducibility.get("determinism_controls")
    observed_controls = checkpoint.get("determinism_controls")
    if not isinstance(expected_controls, dict) or not isinstance(observed_controls, dict):
        add(
            "checkpoint.determinism_controls",
            "present",
            "missing",
            replay_blocking=True,
            evidence_blocking=True,
        )
    else:
        evidence_fields = [
            "split_seed",
            "corruption_seed",
            "sampling_seed",
            "batch_size",
            "sampler_shuffle",
            "sampler_seed",
            "sampler_drop_last",
            "sampler_max_epochs",
        ]
        replay_only_fields = [
            "device",
            "data_workers",
            "tch_intra_op_threads",
            "tch_inter_op_threads",
        ]
        for field in evidence_fields:
            add(
                f"determinism_controls.{field}",
                expected_controls.get(field),
                observed_controls.get(field),
                replay_blocking=True,
                evidence_blocking=True,
            )
        for field in replay_only_fields:
            add(
                f"determinism_controls.{field}",
                expected_controls.get(field),
                observed_controls.get(field),
                replay_blocking=True,
                evidence_blocking=False,
            )

    add(
        "checkpoint.resume_mode",
        "optimizer_exact_resume",
        checkpoint.get("resume_mode"),
        replay_blocking=True,
        evidence_blocking=False,
    )
    optimizer = checkpoint.get("optimizer_state") or {}
    add(
        "checkpoint.optimizer_state.internal_state_persisted",
        True,
        optimizer.get("internal_state_persisted", False),
        replay_blocking=True,
        evidence_blocking=False,
    )

    replay_compatible = not any(item["replay_blocking"] for item in mismatches)
    evidence_compatible = not any(item["evidence_blocking"] for item in mismatches)
    if replay_compatible:
        compatibility_class = "strict_replay_compatible"
    elif evidence_compatible:
        compatibility_class = "evidence_compatible"
    else:
        compatibility_class = "incompatible"
    return {
        "class": compatibility_class,
        "replay_compatible": replay_compatible,
        "evidence_compatible": evidence_compatible,
        "mismatches": mismatches,
    }


def validate_training_replay_contract():
    started = time.time()
    failures = []
    doc_path = Path("docs/training_replay_contract.md")
    if not doc_path.is_file():
        failures.append("missing docs/training_replay_contract.md")
    else:
        doc_text = doc_path.read_text(encoding="utf-8").lower()
        for term in ("strict_replay_compatible", "evidence_compatible", "sampler_seed"):
            if term not in doc_text:
                failures.append(f"training replay contract docs missing `{term}`")

    base_controls = {
        "split_seed": 11,
        "corruption_seed": 12,
        "sampling_seed": 13,
        "batch_size": 4,
        "sampler_shuffle": True,
        "sampler_seed": 17,
        "sampler_drop_last": False,
        "sampler_max_epochs": 2,
        "device": "cpu",
        "data_workers": 0,
        "tch_intra_op_threads": None,
        "tch_inter_op_threads": None,
    }
    summary = {
        "reproducibility": {
            "config_hash": "cfg",
            "dataset_validation_fingerprint": "dataset",
            "metric_schema_version": 6,
            "determinism_controls": dict(base_controls),
        }
    }
    checkpoint = {
        "config_hash": "cfg",
        "dataset_validation_fingerprint": "dataset",
        "metric_schema_version": 6,
        "determinism_controls": dict(base_controls),
        "resume_mode": "weights_only_resume",
        "optimizer_state": {"internal_state_persisted": False},
    }
    evidence_report = _training_replay_report(summary, checkpoint)
    if evidence_report["class"] != "evidence_compatible":
        failures.append(f"weights-only replay fixture classified as {evidence_report['class']}")
    if not any(item["field"] == "checkpoint.resume_mode" for item in evidence_report["mismatches"]):
        failures.append("weights-only replay fixture did not report checkpoint.resume_mode")

    mismatched = json.loads(json.dumps(checkpoint))
    mismatched["determinism_controls"]["sampler_seed"] = 18
    mismatch_report = _training_replay_report(summary, mismatched)
    if mismatch_report["evidence_compatible"]:
        failures.append("sampler_seed mismatch fixture was incorrectly evidence-compatible")
    if not any(
        item["field"] == "determinism_controls.sampler_seed" and item["evidence_blocking"]
        for item in mismatch_report["mismatches"]
    ):
        failures.append("sampler_seed mismatch fixture did not report a concrete evidence-blocking field")

    return {
        "name": "training replay contract",
        "command": ["internal", "validate_training_replay_contract"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }
