#!/usr/bin/env python3
"""Summarize Q3 no-coordinate-move bond refinement scoring."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


LAYERS = ("raw_geometry", "bond_logits_refined", "valence_refined", "repaired")
METRICS = (
    "vina_score",
    "gnina_affinity",
    "gnina_cnn_score",
    "qed",
    "sa_score",
    "lipinski_violations",
    "rdkit_valid_fraction",
    "bond_count",
    "valence_violation_count",
    "clash_fraction",
    "pocket_contact_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--generation-layers", required=True)
    parser.add_argument("--backend-coverage", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def metric(row, name):
    value = row.get("metrics", {}).get(name) if row else None
    return float(value) if finite(value) else None


def layer_summary(rows):
    return {
        "candidate_count": len(rows),
        "backend_coverage": {
            key: sum(1 for row in rows if metric(row, key) is not None) / float(max(len(rows), 1))
            for key in ("vina_score", "gnina_affinity", "gnina_cnn_score", "qed", "sa_score")
        },
        "means": {key: mean(metric(row, key) for row in rows) for key in METRICS},
    }


def row_key(row):
    return (row.get("example_id") or "unknown", row.get("method_id") or "unknown")


def paired_deltas(rows_by_layer, layer):
    raw = {row_key(row): row for row in rows_by_layer.get("raw_geometry", [])}
    deltas = defaultdict(list)
    for row in rows_by_layer.get(layer, []):
        baseline = raw.get(row_key(row))
        if baseline is None:
            continue
        for name in METRICS:
            left = metric(row, name)
            right = metric(baseline, name)
            if left is not None and right is not None:
                deltas[name].append(left - right)
    return {name: mean(values) for name, values in sorted(deltas.items())}


def coords(candidate):
    return candidate.get("coords") or []


def rmsd(left, right):
    if len(left) != len(right) or not left:
        return None
    total = 0.0
    count = 0
    for a, b in zip(left, right):
        if len(a) != 3 or len(b) != 3:
            return None
        total += sum((float(a[axis]) - float(b[axis])) ** 2 for axis in range(3))
        count += 3
    return math.sqrt(total / count)


def coordinate_audit(generation):
    raw = {
        candidate.get("example_id"): candidate
        for candidate in generation.get("raw_geometry_candidates", [])
    }
    audit = {}
    for layer, field in (
        ("bond_logits_refined", "bond_logits_refined_candidates"),
        ("valence_refined", "valence_refined_candidates"),
        ("repaired", "repaired_candidates"),
    ):
        values = []
        for candidate in generation.get(field, []):
            baseline = raw.get(candidate.get("example_id"))
            if baseline is not None:
                values.append(rmsd(coords(baseline), coords(candidate)))
        audit[layer] = {
            "mean_rmsd_to_raw_geometry": mean(values),
            "max_rmsd_to_raw_geometry": max([value for value in values if finite(value)], default=None),
            "coordinate_preserving": all((value is not None and value <= 1e-8) for value in values)
            if values
            else False,
        }
    return audit


def fmt(value):
    return "NA" if value is None else f"{value:.4g}"


def write_md(payload, path):
    lines = [
        "# Q3 Bond Refinement Summary",
        "",
        payload["claim_boundary"],
        "",
        "| Layer | Candidates | Vina cov | GNINA cov | Vina | GNINA | QED | SA | Bonds | Valence viol | Coord preserving |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for layer in LAYERS:
        summary = payload["layer_summaries"].get(layer, {})
        means = summary.get("means", {})
        coverage = summary.get("backend_coverage", {})
        audit = payload["coordinate_audit"].get(layer, {})
        lines.append(
            f"| {layer} | {summary.get('candidate_count', 0)} | "
            f"{fmt(coverage.get('vina_score'))} | {fmt(coverage.get('gnina_affinity'))} | "
            f"{fmt(means.get('vina_score'))} | {fmt(means.get('gnina_affinity'))} | "
            f"{fmt(means.get('qed'))} | {fmt(means.get('sa_score'))} | "
            f"{fmt(means.get('bond_count'))} | {fmt(means.get('valence_violation_count'))} | "
            f"{audit.get('coordinate_preserving')} |"
        )
    lines.extend(["", "## Mean Deltas Vs Raw Geometry", ""])
    for layer, deltas in payload["paired_deltas_vs_raw_geometry"].items():
        lines.append(
            f"- `{layer}`: dVina={fmt(deltas.get('vina_score'))}, "
            f"dGNINA={fmt(deltas.get('gnina_affinity'))}, dQED={fmt(deltas.get('qed'))}, "
            f"dSA={fmt(deltas.get('sa_score'))}, dBonds={fmt(deltas.get('bond_count'))}, "
            f"dValenceViol={fmt(deltas.get('valence_violation_count'))}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = load_jsonl(args.candidate_metrics)
    generation = load_json(args.generation_layers)
    coverage = load_json(args.backend_coverage)
    rows_by_layer = defaultdict(list)
    for row in rows:
        rows_by_layer[row.get("layer")].append(row)
    payload = {
        "artifact_name": "q3_bond_refinement_summary",
        "schema_version": 1,
        "claim_boundary": "This artifact evaluates coordinate-preserving bond refinement separately from the coordinate-moving repaired reference. Vina/GNINA are score_only backends, not experimental affinity.",
        "inputs": {
            "candidate_metrics": args.candidate_metrics,
            "generation_layers": args.generation_layers,
            "backend_coverage": args.backend_coverage,
        },
        "layer_summaries": {layer: layer_summary(rows_by_layer.get(layer, [])) for layer in LAYERS},
        "paired_deltas_vs_raw_geometry": {
            layer: paired_deltas(rows_by_layer, layer)
            for layer in ("bond_logits_refined", "valence_refined", "repaired")
        },
        "coordinate_audit": coordinate_audit(generation),
        "backend_coverage_report": coverage,
        "gnina_repaired_status": "not_scored_in_q3_bond_refinement_run_after_external_gnina_hang_on_coordinate_moving_repaired_reference",
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(payload, args.output_md)


if __name__ == "__main__":
    main()
