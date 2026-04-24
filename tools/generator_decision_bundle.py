#!/usr/bin/env python3
"""Persist or validate the current generator promotion decision artifact."""

import argparse
import json
import hashlib
from pathlib import Path


REQUIRED_DECISION_INPUTS = [
    "checkpoints/pdbbindpp_real_backends/claim_summary.json",
    "checkpoints/tight_geometry_pressure/claim_summary.json",
    "configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json",
]

# Promotion-relevant source/config paths that should force a refreshed decision
# artifact when the generator objective or rollout contract changes.
MONITORED_PROMOTION_PATHS = [
    "src/config/types.rs",
    "src/experiments/unseen_pocket.rs",
    "src/losses/task.rs",
    "src/models/decoder.rs",
    "src/models/evaluation.rs",
    "src/models/system.rs",
    "src/training/trainer.rs",
    "configs/unseen_pocket_pdbbindpp_real_backends.json",
    "configs/unseen_pocket_tight_geometry_pressure.json",
    "tools/generator_decision_bundle.py",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build or validate generator decision artifact.")
    parser.add_argument("--bundle", default="docs/evidence_bundle.json")
    parser.add_argument("--output", default="checkpoints/generator_decision/generator_decision.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing decision artifact is stale relative to its required inputs.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths(bundle_path):
    return [bundle_path, *REQUIRED_DECISION_INPUTS]


def snapshot(paths):
    tracked = []
    for raw_path in paths:
        path = Path(raw_path)
        tracked.append(
            {
                "path": raw_path,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return tracked


def build_freshness(bundle_path):
    return {
        "required_inputs": snapshot(tracked_paths(bundle_path)),
        "monitored_promotion_paths": snapshot(MONITORED_PROMOTION_PATHS),
    }


def validate_existing(output_path, expected_freshness):
    artifact = load_json(output_path)
    actual = artifact.get("freshness") or {}
    mismatches = []
    for section in ("required_inputs", "monitored_promotion_paths"):
        expected_rows = {
            row["path"]: row
            for row in expected_freshness.get(section, [])
        }
        actual_rows = {
            row.get("path"): row
            for row in actual.get(section, [])
            if row.get("path") is not None
        }
        for path, expected in expected_rows.items():
            current = actual_rows.get(path)
            if current != expected:
                mismatches.append(path)
    if mismatches:
        joined = ", ".join(sorted(mismatches))
        raise SystemExit(
            "generator decision artifact is stale; refresh via "
            "./tools/revalidate_reviewer_bundle.sh or "
            f"python3 tools/generator_decision_bundle.py. mismatched paths: {joined}"
        )
    print(f"generator decision artifact is fresh: {output_path}")


def main():
    args = parse_args()
    expected_freshness = build_freshness(args.bundle)
    if args.check:
        validate_existing(args.output, expected_freshness)
        return

    bundle = load_json(args.bundle)
    direction = bundle.get("generator_direction") or {}
    tradeoffs = bundle.get("efficiency_tradeoffs") or {}
    refresh = bundle.get("refresh_contract") or {}
    breadth = bundle.get("benchmark_breadth") or {}
    decision = {
        "schema_version": 1,
        "current_direction": direction.get("current_direction"),
        "saturation_status": direction.get("saturation_status"),
        "primary_justification_surface": direction.get("primary_justification_surface"),
        "stability_surface": direction.get("stability_surface"),
        "major_model_change_gate": direction.get("major_model_change_gate"),
        "plateau_rules": {
            "canonical_larger_data_surface": "Require larger-data held-out-pocket quality to plateau on checkpoints/pdbbindpp_real_backends before promoting major generator changes.",
            "tight_geometry_surface": "Require tight-geometry pressure evidence to stop showing quality headroom relative to the canonical surface before claiming saturation.",
            "multi_seed_surface": "Require configs/checkpoints/multi_seed_pdbbindpp_real_backends to keep at least three persisted seeds with acceptable stability before promotion.",
        },
        "decision_inputs": {
            "larger_data_multi_seed_fit_range": direction.get("larger_data_multi_seed_fit_range"),
            "larger_data_multi_seed_leakage_range": direction.get("larger_data_multi_seed_leakage_range"),
            "stability_decision": direction.get("stability_decision"),
            "benchmark_breadth_summary": breadth.get("summary_sentence"),
            "required_surfaces": REQUIRED_DECISION_INPUTS,
        },
        "quality_efficiency_tradeoffs": tradeoffs.get("surfaces"),
        "decision_notes": direction.get("reasons"),
        "efficiency_summary": tradeoffs.get("summary"),
        "refresh_contract": {
            "entrypoint": refresh.get("entrypoint"),
            "guarantee": refresh.get("guarantee"),
            "promotion_rule": refresh.get("promotion_rule"),
        },
        "freshness": expected_freshness,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"generator decision artifact written: {output}")


if __name__ == "__main__":
    main()
