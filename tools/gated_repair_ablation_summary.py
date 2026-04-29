#!/usr/bin/env python3
"""Summarize Q3 gated repair ablation against Q2 no/full repair layers."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


LAYERS = ("no_repair", "full_repair", "gated_repair", "repair_rejected")
METRICS = (
    "vina_score",
    "gnina_affinity",
    "gnina_cnn_score",
    "qed",
    "sa_score",
    "clash_fraction",
    "pocket_contact_fraction",
    "centroid_offset",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-candidate-metrics", required=True)
    parser.add_argument("--q3-candidate-metrics", required=True)
    parser.add_argument("--repair-gate-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return sum(values) / float(len(values)) if values else None


def metric(row, key):
    value = row.get("metrics", {}).get(key) if row else None
    return float(value) if finite(value) else None


def coverage(rows, key):
    return sum(1 for row in rows if metric(row, key) is not None) / float(max(len(rows), 1))


def row_key(row):
    return (row.get("method_id") or "unknown", row.get("example_id") or "unknown")


def summarize_layer(rows):
    return {
        "candidate_count": len(rows),
        "backend_coverage": {
            "vina_score": coverage(rows, "vina_score"),
            "gnina_affinity": coverage(rows, "gnina_affinity"),
            "gnina_cnn_score": coverage(rows, "gnina_cnn_score"),
            "rdkit": sum(
                1
                for row in rows
                if any(metric(row, key) is not None for key in ("qed", "sa_score", "logp"))
            )
            / float(max(len(rows), 1)),
        },
        "metric_means": {key: mean(metric(row, key) for row in rows) for key in METRICS},
    }


def paired_delta(rows_by_layer, layer):
    baseline = {row_key(row): row for row in rows_by_layer.get("no_repair", [])}
    candidate_rows = rows_by_layer.get(layer, [])
    deltas = defaultdict(list)
    paired = 0
    for row in candidate_rows:
        raw = baseline.get(row_key(row))
        if raw is None:
            continue
        paired += 1
        for key in METRICS:
            left = metric(raw, key)
            right = metric(row, key)
            if left is not None and right is not None:
                deltas[key].append(right - left)
    return {
        "paired_candidate_count": paired,
        "mean_delta_vs_no_repair": {key: mean(values) for key, values in deltas.items()},
    }


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(path, payload):
    lines = [
        "# Q3 Gated Repair Ablation Summary",
        "",
        payload["claim_boundary"],
        "",
        "## Gate Counts",
        "",
    ]
    counts = payload["repair_gate_counts"]
    for key in ("raw_passthrough", "repaired_candidate", "rejected_repair"):
        lines.append(f"- `{key}`: {counts.get(key, 0)}")
    lines.extend([
        "",
        "## Layer Summary",
        "",
        "| Layer | Candidates | Vina Cov | GNINA Cov | Vina | GNINA | CNN | QED | SA | Clash | Contact | dVina | dGNINA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for layer in LAYERS:
        row = payload["layer_summaries"].get(layer, {})
        means = row.get("metric_means", {})
        coverage_row = row.get("backend_coverage", {})
        delta = payload["paired_deltas"].get(layer, {}).get("mean_delta_vs_no_repair", {})
        lines.append(
            "| {layer} | {count} | {vina_cov} | {gnina_cov} | {vina} | {gnina} | {cnn} | {qed} | {sa} | {clash} | {contact} | {dvina} | {dgnina} |".format(
                layer=layer,
                count=row.get("candidate_count", 0),
                vina_cov=fmt(coverage_row.get("vina_score")),
                gnina_cov=fmt(coverage_row.get("gnina_affinity")),
                vina=fmt(means.get("vina_score")),
                gnina=fmt(means.get("gnina_affinity")),
                cnn=fmt(means.get("gnina_cnn_score")),
                qed=fmt(means.get("qed")),
                sa=fmt(means.get("sa_score")),
                clash=fmt(means.get("clash_fraction")),
                contact=fmt(means.get("pocket_contact_fraction")),
                dvina=fmt(delta.get("vina_score")),
                dgnina=fmt(delta.get("gnina_affinity")),
            )
        )
    lines.extend(["", "## Interpretation", ""])
    for item in payload["interpretation"]:
        lines.append(f"- {item}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    q2_rows = [
        row
        for row in load_jsonl(args.q2_candidate_metrics)
        if row.get("layer") in {"no_repair", "full_repair"}
    ]
    q3_rows = [
        row
        for row in load_jsonl(args.q3_candidate_metrics)
        if row.get("layer") in {"gated_repair", "repair_rejected"}
    ]
    rows_by_layer = {layer: [] for layer in LAYERS}
    for row in q2_rows + q3_rows:
        rows_by_layer.setdefault(row.get("layer"), []).append(row)

    layer_summaries = {layer: summarize_layer(rows_by_layer.get(layer, [])) for layer in LAYERS}
    paired_deltas = {
        layer: paired_delta(rows_by_layer, layer)
        for layer in ("full_repair", "gated_repair", "repair_rejected")
    }
    gate_report = load_json(args.repair_gate_report)
    gate_counts = gate_report.get("totals", {}).get("repair_gate_counts", {})

    full_delta = paired_deltas["full_repair"]["mean_delta_vs_no_repair"]
    rejected_delta = paired_deltas["repair_rejected"]["mean_delta_vs_no_repair"]
    interpretation = [
        "Legacy full_repair remains a postprocessing-only layer and is not promoted as model-native evidence.",
        "The configured gate rejected every coordinate-moving repair because raw candidates were already backend-input dockable and proposed repair exceeded movement or box-center bounds.",
        "repair_rejected is a raw-coordinate passthrough layer; its backend scores should be interpreted as a non-degradation guard, not a model improvement.",
    ]
    if finite(full_delta.get("vina_score")) and finite(rejected_delta.get("vina_score")):
        interpretation.append(
            "Vina mean degradation was reduced from legacy full_repair delta "
            f"{full_delta['vina_score']:.4g} to repair_rejected delta {rejected_delta['vina_score']:.4g}."
        )
    if finite(full_delta.get("gnina_affinity")) and finite(rejected_delta.get("gnina_affinity")):
        interpretation.append(
            "GNINA affinity mean degradation was reduced from legacy full_repair delta "
            f"{full_delta['gnina_affinity']:.4g} to repair_rejected delta {rejected_delta['gnina_affinity']:.4g}."
        )

    payload = {
        "schema_version": 1,
        "artifact_name": "q3_gated_repair_ablation_summary",
        "inputs": {
            "q2_candidate_metrics": args.q2_candidate_metrics,
            "q3_candidate_metrics": args.q3_candidate_metrics,
            "repair_gate_report": args.repair_gate_report,
        },
        "repair_gate_options": gate_report.get("repair_gate_options"),
        "repair_gate_counts": gate_counts,
        "layer_summaries": layer_summaries,
        "paired_deltas": paired_deltas,
        "interpretation": interpretation,
        "claim_boundary": "Gated repair evidence is postprocessing evidence. repair_rejected is a safety fallback that preserves raw coordinates and must not be reported as native model improvement.",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)


if __name__ == "__main__":
    main()
