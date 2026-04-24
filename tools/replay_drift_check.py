#!/usr/bin/env python3
"""Compare claim-bearing artifacts against explicit replay tolerances."""

import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_TOLERANCES = {
    "leakage_proxy_mean": 0.01,
    "strict_pocket_fit_score": 0.03,
    "clash_fraction": 0.02,
    "rdkit_sanitized_fraction": 0.05,
    "rdkit_unique_smiles_fraction": 0.05,
}

SCHEMA_VERSION = 2


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Check replay drift between two claim artifacts.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output")
    return parser.parse_args(argv[1:])


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sibling_experiment_summary(path):
    candidate = Path(path)
    sibling = candidate.with_name("experiment_summary.json")
    if sibling.is_file():
        return load_json(sibling)
    return None


def metric(claim, path):
    test = claim.get("test", {})
    backend = claim.get("backend_metrics", {})
    chemistry = backend.get("chemistry_validity", {}).get("metrics", {})
    pocket = backend.get("pocket_compatibility", {}).get("metrics", {})
    if path == "leakage_proxy_mean":
        return test.get("leakage_proxy_mean")
    if path == "strict_pocket_fit_score":
        return pocket.get("strict_pocket_fit_score", pocket.get("heuristic_strict_pocket_fit_score"))
    if path == "clash_fraction":
        return pocket.get("clash_fraction")
    if path == "rdkit_sanitized_fraction":
        return chemistry.get("rdkit_sanitized_fraction")
    if path == "rdkit_unique_smiles_fraction":
        return chemistry.get("rdkit_unique_smiles_fraction")
    raise KeyError(path)


def replay_metadata(summary):
    reproducibility = (summary or {}).get("reproducibility") or {}
    return {
        "determinism_controls": reproducibility.get("determinism_controls"),
        "replay_tolerance": reproducibility.get("replay_tolerance"),
        "resume_contract": reproducibility.get("resume_contract"),
    }


def main(argv):
    args = parse_args(argv)
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    baseline_summary = sibling_experiment_summary(args.baseline)
    candidate_summary = sibling_experiment_summary(args.candidate)
    results = {}
    failed = []
    for name, tolerance in DEFAULT_TOLERANCES.items():
        left = metric(baseline, name)
        right = metric(candidate, name)
        if left is None and right is None:
            diff = None
            passed = True
            status = "not_applicable"
        else:
            diff = None if left is None or right is None else abs(float(right) - float(left))
            passed = diff is not None and math.isfinite(diff) and diff <= tolerance
            status = "checked" if diff is not None else "missing_value"
        results[name] = {
            "baseline": left,
            "candidate": right,
            "abs_diff": diff,
            "tolerance": tolerance,
            "passed": passed,
            "status": status,
        }
        if not passed:
            failed.append(name)
    promotion_decision = {
        "status": "pass" if not failed else "fail",
        "baseline_label": baseline.get("run_label"),
        "candidate_label": candidate.get("run_label"),
        "reason": "all tracked replay metrics are within bounded rerun tolerance"
        if not failed
        else "candidate artifact exceeded one or more bounded replay tolerances",
        "promotion_rule": "Promote a refreshed reviewer surface only when replay drift passes and reviewer thresholds remain green.",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_metadata": replay_metadata(baseline_summary),
        "candidate_metadata": replay_metadata(candidate_summary),
        "metrics": results,
        "passed": not failed,
        "failed_metrics": failed,
        "promotion_decision": promotion_decision,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    if failed:
        raise SystemExit(
            "replay drift check failed for: " + ", ".join(failed)
        )


if __name__ == "__main__":
    main(sys.argv)
