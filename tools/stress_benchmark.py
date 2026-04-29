#!/usr/bin/env python3
"""Materialize a mechanical stress subset from generation and metric artifacts."""

import argparse
import json
import math
from pathlib import Path


LAYER_SOURCES = [
    ("raw_geometry_candidates", "raw_geometry"),
    ("raw_rollout_candidates", "raw_rollout"),
    ("bond_logits_refined_candidates", "bond_logits_refined"),
    ("valence_refined_candidates", "valence_refined"),
    ("repaired_candidates", "repaired"),
    ("inferred_bond_candidates", "inferred_bond"),
    ("deterministic_proxy_candidates", "deterministic_proxy"),
    ("reranked_candidates", "reranked"),
]

SUMMARY_METRICS = [
    "vina_score",
    "vina_best_affinity_kcal_mol",
    "gnina_score",
    "gnina_cnn_score",
    "qed",
    "sa_score",
    "rdkit_valid_fraction",
    "pocket_contact_fraction",
    "clash_fraction",
    "mean_centroid_offset",
    "interaction_profile_coverage_fraction",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select and report a hard unseen-pocket stress subset without hand picking."
    )
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--generation-layers", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--quantile", type=float, default=0.15)
    parser.add_argument(
        "--min-rules",
        type=int,
        default=2,
        help="Minimum stress rules triggered per example. Default requires low homology plus one hard-size/chemotype rule.",
    )
    return parser.parse_args(argv)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def count_pdb_atoms(path):
    if not path:
        return None
    try:
        handle = Path(path).open("r", encoding="utf-8")
    except OSError:
        return None
    with handle:
        return sum(1 for line in handle if line.startswith("ATOM") or line.startswith("HETATM"))


def protein_family_proxy(protein_id):
    for separator in ("_", "-", ":", "."):
        protein_id = protein_id.split(separator)[0]
    return protein_id or "unknown"


def generation_examples(path):
    artifact = load_json(path)
    examples = {}
    for source_key, layer in LAYER_SOURCES:
        for candidate in artifact.get(source_key, []):
            example_id = candidate.get("example_id") or candidate.get("protein_id") or "unknown"
            record = examples.setdefault(
                example_id,
                {
                    "example_id": example_id,
                    "protein_id": candidate.get("protein_id") or example_id,
                    "layers": set(),
                    "ligand_atom_counts": [],
                    "pocket_atom_count": None,
                    "source_pocket_path": candidate.get("source_pocket_path"),
                },
            )
            record["layers"].add(layer)
            atoms = candidate.get("atom_types") or []
            if isinstance(atoms, list):
                record["ligand_atom_counts"].append(len(atoms))
            if record["pocket_atom_count"] is None:
                record["pocket_atom_count"] = count_pdb_atoms(candidate.get("source_pocket_path"))
    for record in examples.values():
        counts = record.pop("ligand_atom_counts")
        record["ligand_atom_count"] = max(counts) if counts else None
        record["layers"] = sorted(record["layers"])
        record["protein_family_proxy"] = protein_family_proxy(record["protein_id"])
    return examples


def quantile_bounds(values, quantile):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return None, None
    lower_index = max(0, math.floor((len(values) - 1) * quantile))
    upper_index = min(len(values) - 1, math.ceil((len(values) - 1) * (1.0 - quantile)))
    return values[lower_index], values[upper_index]


def select_examples(split_report, examples, quantile, min_rules):
    train_families = set((split_report.get("train") or {}).get("protein_family_proxy_histogram") or {})
    pocket_low, pocket_high = quantile_bounds(
        [record.get("pocket_atom_count") for record in examples.values()], quantile
    )
    ligand_low, ligand_high = quantile_bounds(
        [record.get("ligand_atom_count") for record in examples.values()], quantile
    )
    selected = {}
    for example_id, record in examples.items():
        rules = []
        if record["protein_family_proxy"] not in train_families:
            rules.append("low_homology_proxy")
        pocket_atoms = record.get("pocket_atom_count")
        if pocket_atoms is not None and (
            (pocket_low is not None and pocket_atoms <= pocket_low)
            or (pocket_high is not None and pocket_atoms >= pocket_high)
        ):
            rules.append("pocket_atom_count_extreme")
        ligand_atoms = record.get("ligand_atom_count")
        if ligand_atoms is not None and (
            (ligand_low is not None and ligand_atoms <= ligand_low)
            or (ligand_high is not None and ligand_atoms >= ligand_high)
        ):
            rules.append("ligand_atom_count_extreme")
        enriched = dict(record)
        enriched["triggered_rules"] = rules
        if len(rules) >= min_rules:
            selected[example_id] = enriched
    return selected, {
        "quantile": quantile,
        "min_rules": min_rules,
        "pocket_atom_count_low": pocket_low,
        "pocket_atom_count_high": pocket_high,
        "ligand_atom_count_low": ligand_low,
        "ligand_atom_count_high": ligand_high,
    }


def finite_metric(row, name):
    value = (row.get("metrics") or {}).get(name)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize_metrics(rows):
    by_layer = {}
    for row in rows:
        layer = row.get("layer") or "unknown"
        bucket = by_layer.setdefault(layer, {"rows": [], "examples": set()})
        bucket["rows"].append(row)
        bucket["examples"].add(row.get("example_id") or "unknown")
    summary = {}
    for layer, bucket in sorted(by_layer.items()):
        layer_rows = bucket["rows"]
        metric_summary = {}
        for metric in SUMMARY_METRICS:
            values = [finite_metric(row, metric) for row in layer_rows]
            available = [value for value in values if value is not None]
            metric_summary[metric] = {
                "mean": mean(available),
                "available_count": len(available),
                "coverage_fraction": len(available) / len(layer_rows) if layer_rows else 0.0,
            }
        summary[layer] = {
            "candidate_rows": len(layer_rows),
            "unique_examples": len(bucket["examples"]),
            "metrics": metric_summary,
        }
    return summary


def write_markdown(path, report):
    lines = [
        "# Q1 Stress Benchmark",
        "",
        "The stress subset was materialized by `tools/stress_benchmark.py` from candidate-level artifacts.",
        "Selection is mechanical: an example must satisfy the configured minimum number of stress rules.",
        "",
        f"- status: {report['status']}",
        f"- selected_examples: {report['selected_example_count']}",
        f"- selected_candidate_rows: {report['selected_candidate_rows']}",
        f"- quantile: {report['selection_thresholds']['quantile']}",
        f"- min_rules: {report['selection_thresholds']['min_rules']}",
        "",
        "## Selected Examples",
        "",
        "| Example | Pocket Atoms | Ligand Atoms | Rules |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in report["selected_examples"]:
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                item["example_id"],
                item.get("pocket_atom_count"),
                item.get("ligand_atom_count"),
                ", ".join(item["triggered_rules"]),
            )
        )
    lines.extend(["", "## Layer Metrics", "", "| Layer | Rows | Examples | Vina Mean | GNINA Mean | QED Mean | Clash Mean |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for layer, item in report["layer_metrics"].items():
        metrics = item["metrics"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                layer,
                item["candidate_rows"],
                item["unique_examples"],
                fmt(metrics["vina_score"]["mean"]),
                fmt(metrics["gnina_score"]["mean"]),
                fmt(metrics["qed"]["mean"]),
                fmt(metrics["clash_fraction"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "Stress results are reported separately from the primary benchmark and include all candidate rows for selected examples, including failed or low-quality candidates present in the source metrics.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def main(argv=None):
    args = parse_args(argv)
    split_report = load_json(args.split_report)
    examples = generation_examples(args.generation_layers)
    selected, thresholds = select_examples(split_report, examples, args.quantile, args.min_rules)
    metric_rows = load_jsonl(args.candidate_metrics)
    selected_rows = [row for row in metric_rows if row.get("example_id") in selected]
    write_jsonl(args.output_jsonl, selected_rows)
    report = {
        "schema_version": 1,
        "benchmark_id": "q1_stress_unseen_pocket_v1",
        "status": "materialized",
        "source_artifacts": {
            "split_report": args.split_report,
            "generation_layers": args.generation_layers,
            "candidate_metrics": args.candidate_metrics,
        },
        "selection_thresholds": thresholds,
        "selected_example_count": len(selected),
        "selected_candidate_rows": len(selected_rows),
        "selected_examples": [selected[key] for key in sorted(selected)],
        "layer_metrics": summarize_metrics(selected_rows),
        "output_candidate_metrics": args.output_jsonl,
        "claim_guardrail": "Stress results are diagnostic and must be reported separately from primary benchmark metrics.",
    }
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(f"stress benchmark materialized: {args.output_json}")
    print(f"stress candidate metrics written: {args.output_jsonl}")


if __name__ == "__main__":
    main()
