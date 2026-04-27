#!/usr/bin/env python3
"""Deterministically select success, failure, and postprocessing-only cases."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_metrics", nargs="+")
    parser.add_argument("--output", default="configs/q1_case_selection.json")
    parser.add_argument("--markdown", default="docs/q1_case_studies.md")
    return parser.parse_args()


def load(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row.setdefault("source_file", path)
                    rows.append(row)
    return rows


def score(row):
    metrics = row.get("metrics") or {}
    valid = float(metrics.get("valid_fraction", metrics.get("candidate_valid_fraction", 0.0)) or 0.0)
    contact = float(metrics.get("pocket_contact_fraction", metrics.get("strict_pocket_fit_score", 0.0)) or 0.0)
    clash = float(metrics.get("clash_fraction", metrics.get("clash_burden", 1.0)) or 0.0)
    qed = metrics.get("qed")
    qed = float(qed) if isinstance(qed, (int, float)) and math.isfinite(float(qed)) else 0.0
    return valid + contact + qed - clash


def compact(row, label):
    return {
        "case_type": label,
        "candidate_id": row.get("candidate_id"),
        "protein_id": row.get("protein_id"),
        "example_id": row.get("example_id"),
        "layer": row.get("layer"),
        "method_id": row.get("method_id"),
        "selection_score": score(row),
        "source_artifacts": row.get("source_artifacts") or [row.get("source_file")],
    }


def main():
    args = parse_args()
    rows = load(args.candidate_metrics)
    by_pocket = defaultdict(list)
    for row in rows:
        by_pocket[row.get("protein_id") or row.get("example_id") or "unknown"].append(row)
    successes = []
    failures = []
    postprocessing = []
    for pocket, pocket_rows in sorted(by_pocket.items()):
        best = max(pocket_rows, key=score)
        worst = min(pocket_rows, key=score)
        successes.append(compact(best, "success"))
        failures.append(compact(worst, "failure"))
        raw = [row for row in pocket_rows if row.get("layer") in ("raw_rollout", "raw_flow")]
        ranked = [row for row in pocket_rows if row.get("layer") == "reranked"]
        if raw and ranked:
            raw_best = max(raw, key=score)
            ranked_best = max(ranked, key=score)
            if score(ranked_best) > score(raw_best):
                case = compact(ranked_best, "postprocessing_improved")
                case["raw_candidate_id"] = raw_best.get("candidate_id")
                case["raw_selection_score"] = score(raw_best)
                postprocessing.append(case)
    payload = {
        "schema_version": 1,
        "selection_rule": "per-pocket score = valid + contact + qed - clash; deterministic sorted pockets",
        "source_files": args.candidate_metrics,
        "cases": {
            "success": sorted(successes, key=lambda row: row["selection_score"], reverse=True)[:5],
            "failure": sorted(failures, key=lambda row: row["selection_score"])[:5],
            "postprocessing_improved": sorted(postprocessing, key=lambda row: row["selection_score"], reverse=True)[:5],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Q1 Case Studies", "", f"Selection rule: {payload['selection_rule']}", ""]
    for label, cases in payload["cases"].items():
        md.extend([f"## {label}", "", "| candidate | pocket | layer | score |", "|---|---|---|---:|"])
        for case in cases:
            md.append(f"| `{case['candidate_id']}` | `{case['protein_id']}` | `{case['layer']}` | {case['selection_score']:.4g} |")
        md.append("")
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text("\n".join(md), encoding="utf-8")
    print(f"case selection written: {output}")


if __name__ == "__main__":
    main()
