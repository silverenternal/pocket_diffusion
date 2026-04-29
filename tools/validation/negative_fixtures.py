"""Negative fixtures that prove validation gates catch claim-risk regressions."""

import json
import tempfile
import time
from pathlib import Path

from validation.claim_provenance import pharmacology_claim_provenance_failures


POSTPROCESSED_LAYERS = {
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


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _postprocessed_promotion_failures(row, source: str) -> list[str]:
    failures = []
    layer = row.get("layer")
    if layer in POSTPROCESSED_LAYERS and row.get("layer_role") == "raw_model_native":
        failures.append(f"{source} postprocessed layer {layer} promoted to raw_model_native")
    if layer in POSTPROCESSED_LAYERS and row.get("model_improvement") is True:
        failures.append(f"{source} postprocessed layer {layer} marked model_improvement=true")
    return failures


def validate_negative_fixtures():
    started = time.time()
    failures = []
    with tempfile.TemporaryDirectory(prefix="pocket_validation_negative_") as tmpdir:
        tmp = Path(tmpdir)

        heuristic_path = tmp / "heuristic_promoted_claim_summary.json"
        _write_json(
            heuristic_path,
            {
                "claim_context": {
                    "real_backend_backed": False,
                    "evidence_mode": "heuristic-only held-out pocket evidence",
                },
                "chemistry_collaboration": {
                    "pharmacophore_role_coverage": {
                        "value": 0.8,
                        "provenance": "docking_supported",
                        "status": "heuristic topology-pocket pharmacophore diagnostic",
                    }
                },
                "backend_metrics": {
                    "docking_affinity": {
                        "available": True,
                        "backend_name": "heuristic_docking_hook_v1",
                        "metrics": {"docking_like_score": 0.9},
                        "status": "external docking backend on modular rollout candidates",
                    }
                },
                "interpretation": "docking-supported evidence",
            },
        )
        heuristic_failures = pharmacology_claim_provenance_failures(
            _load_json(heuristic_path),
            heuristic_path.as_posix(),
        )
        if not any("heuristic" in failure and "docking" in failure for failure in heuristic_failures):
            failures.append("negative fixture did not reject heuristic evidence promoted to docking-supported")

        for fixture in [
            {
                "path": tmp / "postprocessed_native_row_reranked.json",
                "layer": "reranked",
            },
            {
                "path": tmp / "postprocessed_native_row_inferred_bond.json",
                "layer": "inferred_bond",
            },
        ]:
            layer_path = fixture["path"]
            _write_json(
                layer_path,
                {
                    "method_id": "fixture_method",
                    "layer": fixture["layer"],
                    "layer_role": "raw_model_native",
                    "model_improvement": True,
                },
            )
            failures_for_fixture = _postprocessed_promotion_failures(
                _load_json(layer_path),
                layer_path.as_posix(),
            )
            if not any(
                "raw_model_native" in failure for failure in failures_for_fixture
            ):
                failures.append(
                    f"negative fixture did not reject {fixture['layer']} row promoted to raw_model_native"
                )
            if not any(
                "model_improvement=true" in failure
                for failure in failures_for_fixture
            ):
                failures.append(
                    f"negative fixture did not reject {fixture['layer']} row marked model_improvement=true"
                )

        backend_path = tmp / "missing_backend_coverage_claim_summary.json"
        _write_json(
            backend_path,
            {
                "claim_context": {
                    "real_backend_backed": True,
                    "evidence_mode": "real-backend-backed held-out pocket evidence",
                },
                "backend_metrics": {
                    "docking_affinity": {
                        "available": True,
                        "backend_name": "external_command_docking",
                        "status": "external docking backend on modular rollout candidates",
                        "metrics": {"vina_score": -6.0},
                    }
                },
            },
        )
        backend_failures = pharmacology_claim_provenance_failures(
            _load_json(backend_path),
            backend_path.as_posix(),
        )
        if not any("coverage/status metrics" in failure for failure in backend_failures):
            failures.append("negative fixture did not reject backend-supported metrics missing coverage/status")

    return {
        "name": "negative claim/artifact fixtures",
        "command": ["internal", "validate_negative_fixtures"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }
