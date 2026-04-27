#!/usr/bin/env python3
"""Aggregate slot, gate, and leakage statistics from claim/summary artifacts."""

import argparse
import json
import math
from pathlib import Path


KEYS = [
    "slot_activation_mean",
    "slot_balance_loss",
    "slot_collapse_fraction",
    "gate_activation_mean",
    "gate_sparsity_loss",
    "leakage_proxy_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--output", default="configs/q1_slot_gate_analysis.json")
    parser.add_argument("--markdown", default="docs/q1_slot_gate_analysis.md")
    return parser.parse_args()


def walk_numbers(payload, path=""):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from walk_numbers(value, f"{path}.{key}" if path else key)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from walk_numbers(value, f"{path}[{index}]")
    elif isinstance(payload, (int, float)) and math.isfinite(float(payload)):
        yield path, float(payload)


def main():
    args = parse_args()
    rows = []
    values = {key: [] for key in KEYS}
    for artifact in args.artifacts:
        payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
        found = {}
        for path, value in walk_numbers(payload):
            leaf = path.split(".")[-1].split("[")[0]
            if leaf in KEYS:
                found.setdefault(leaf, []).append(value)
                values[leaf].append(value)
        rows.append({"artifact": artifact, "metrics": {key: sum(vals) / len(vals) for key, vals in found.items()}})
    summary = {}
    for key, vals in values.items():
        summary[key] = {
            "count": len(vals),
            "mean": sum(vals) / len(vals) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    payload = {
        "schema_version": 1,
        "source_artifacts": args.artifacts,
        "summary": summary,
        "artifact_rows": rows,
        "interpretation_guardrail": "Slot/gate/leakage statistics are descriptive unless tied to matched ablations and backend metrics.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Q1 Slot Gate Analysis", "", "| metric | count | mean | min | max |", "|---|---:|---:|---:|---:|"]
    for key, row in summary.items():
        fmt = lambda value: "NA" if value is None else f"{value:.4g}"
        lines.append(f"| {key} | {row['count']} | {fmt(row['mean'])} | {fmt(row['min'])} | {fmt(row['max'])} |")
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"slot/gate analysis written: {output}")


if __name__ == "__main__":
    main()
