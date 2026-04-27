#!/usr/bin/env python3
"""Pocket-level paired bootstrap comparison for candidate metric JSONL files."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-a", required=True, help="candidate_metrics JSONL for candidate method.")
    parser.add_argument("--method-b", required=True, help="candidate_metrics JSONL for baseline method.")
    parser.add_argument("--method-a-layer", default=None, help="Optional layer filter for method A.")
    parser.add_argument("--method-b-layer", default=None, help="Optional layer filter for method B.")
    parser.add_argument("--metric", action="append", default=["vina_score", "gnina_score", "qed", "strict_pocket_fit_score"])
    parser.add_argument("--output", default="configs/q1_statistical_tests.json")
    parser.add_argument("--markdown", default="docs/q1_statistical_tests.md")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_layer(rows, layer):
    if layer is None:
        return rows
    return [row for row in rows if row.get("layer") == layer]


def pocket_metric(rows, metric):
    values = defaultdict(list)
    for row in rows:
        value = (row.get("metrics") or {}).get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[row.get("protein_id") or row.get("example_id") or "unknown"].append(float(value))
    return {key: sum(vals) / len(vals) for key, vals in values.items() if vals}


def summarize(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "ci95": None}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
    std = math.sqrt(var)
    return {"count": len(values), "mean": mean, "std": std, "ci95": 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0}


def compare(metric, rows_a, rows_b, samples, rng):
    a = pocket_metric(rows_a, metric)
    b = pocket_metric(rows_b, metric)
    pockets = sorted(set(a).intersection(b))
    deltas = [a[pocket] - b[pocket] for pocket in pockets]
    boot = []
    if deltas:
        for _ in range(samples):
            boot.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    boot.sort()
    lo = boot[int(0.025 * (len(boot) - 1))] if boot else None
    hi = boot[int(0.975 * (len(boot) - 1))] if boot else None
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    p_two_sided = None
    if boot and mean_delta is not None:
        opposite = sum(1 for value in boot if (value <= 0.0 if mean_delta > 0.0 else value >= 0.0))
        p_two_sided = min(1.0, 2.0 * opposite / len(boot))
    return {
        "metric": metric,
        "test_unit": "protein_id_or_example_id_pocket_aggregate",
        "paired_pocket_count": len(pockets),
        "method_a": summarize([a[pocket] for pocket in pockets]),
        "method_b": summarize([b[pocket] for pocket in pockets]),
        "mean_delta_a_minus_b": mean_delta,
        "bootstrap_ci95": [lo, hi],
        "bootstrap_two_sided_p": p_two_sided,
        "status": "ok" if len(pockets) >= 3 else "insufficient_paired_pockets",
    }


def write_markdown(path, payload):
    lines = [
        "# Q1 Statistical Tests",
        "",
        f"Source A: `{payload['method_a_source']}`",
        f"Source B: `{payload['method_b_source']}`",
        "",
        "| metric | paired pockets | delta A-B | 95% CI | p | status |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in payload["comparisons"]:
        ci = row["bootstrap_ci95"]
        ci_text = "NA" if ci[0] is None else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
        delta = "NA" if row["mean_delta_a_minus_b"] is None else f"{row['mean_delta_a_minus_b']:.4g}"
        pval = "NA" if row["bootstrap_two_sided_p"] is None else f"{row['bootstrap_two_sided_p']:.4g}"
        lines.append(f"| {row['metric']} | {row['paired_pocket_count']} | {delta} | {ci_text} | {pval} | {row['status']} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows_a = filter_layer(load_jsonl(args.method_a), args.method_a_layer)
    rows_b = filter_layer(load_jsonl(args.method_b), args.method_b_layer)
    payload = {
        "schema_version": 1,
        "method_a_source": args.method_a,
        "method_b_source": args.method_b,
        "method_a_layer": args.method_a_layer,
        "method_b_layer": args.method_b_layer,
        "bootstrap_samples": args.bootstrap_samples,
        "claim_guardrail": "Layer or same-file comparisons are diagnostic; public-baseline superiority requires paired external baseline artifacts.",
        "comparisons": [compare(metric, rows_a, rows_b, args.bootstrap_samples, rng) for metric in args.metric],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown, payload)
    print(f"statistical comparison written: {output}")


if __name__ == "__main__":
    main()
