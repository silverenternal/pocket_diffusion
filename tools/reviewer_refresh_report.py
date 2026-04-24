#!/usr/bin/env python3
"""Persist a reviewer-facing refresh report with drift and promotion decisions."""

import argparse
import json
from pathlib import Path


CANONICAL_SURFACES = [
    "checkpoints/claim_matrix",
    "checkpoints/real_backends",
    "checkpoints/pdbbindpp_real_backends",
    "checkpoints/lp_pdbbind_refined_real_backends",
    "checkpoints/tight_geometry_pressure",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build reviewer refresh report.")
    parser.add_argument("--bundle", default="docs/evidence_bundle.json")
    parser.add_argument("--output", default="docs/reviewer_refresh_report.json")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_lookup(bundle):
    return {artifact.get("artifact_dir"): artifact for artifact in bundle.get("artifact_dirs", [])}


def main():
    args = parse_args()
    bundle = load_json(args.bundle)
    artifacts = artifact_lookup(bundle)
    surfaces = []
    for path in CANONICAL_SURFACES:
        artifact = artifacts.get(path) or {}
        claim = artifact.get("claim") or {}
        drift = artifact.get("replay_drift_report") or {}
        surfaces.append(
            {
                "artifact_dir": path,
                "run_label": claim.get("run_label"),
                "backend_thresholds_passed": all(
                    result.get("passed", False)
                    for result in (claim.get("backend_thresholds") or {}).values()
                ),
                "data_thresholds_passed": all(
                    result.get("passed", False)
                    for result in (artifact.get("data_thresholds") or {}).values()
                )
                if artifact.get("data_thresholds")
                else None,
                "leakage_reviewer_status": (claim.get("leakage_review") or {}).get("reviewer_status"),
                "replay_drift_passed": drift.get("passed"),
                "promotion_decision": (drift.get("promotion_decision") or {}).get("status"),
            }
        )
    report = {
        "schema_version": 1,
        "refresh_entrypoint": "./tools/revalidate_reviewer_bundle.sh",
        "guarantee": "bounded replay with explicit metric tolerances; not strict optimizer-state-identical replay",
        "promotion_rule": "promote canonical reviewer surfaces only when replay drift passes and reviewer thresholds remain green",
        "supports_strict_replay": ((bundle.get("replay_guarantees") or {}).get("supports_strict_replay")),
        "continuity_mode": ((bundle.get("replay_guarantees") or {}).get("continuity_mode")),
        "packaged_environment_default": (
            (bundle.get("reviewer_environment_readiness") or {}).get("default_revalidation_python")
        ),
        "packaged_environment_effective_python": (
            (bundle.get("reviewer_environment_readiness") or {}).get("effective_python")
        ),
        "packaged_environment_effective": (
            (bundle.get("reviewer_environment_readiness") or {}).get("effective_python_is_packaged")
        ),
        "packaged_environment_ready": (
            (bundle.get("reviewer_environment_readiness") or {}).get("ready")
        ),
        "reviewer_bundle_status": bundle.get("reviewer_bundle_status"),
        "surfaces": surfaces,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reviewer refresh report written: {output}")


if __name__ == "__main__":
    main()
