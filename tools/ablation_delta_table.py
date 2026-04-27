#!/usr/bin/env python3
"""Build a canonical ablation delta table against the base claim surface."""

import argparse
import json
from pathlib import Path


REQUIRED_VARIANTS = [
    "objective_surrogate",
    "disable_slots",
    "disable_cross_attention",
    "disable_probes",
    "leakage_penalty_disabled",
]

METRICS = [
    ("candidate_valid_fraction", "test_summary", "higher"),
    ("pocket_contact_fraction", "test_summary", "higher"),
    ("pocket_compatibility_fraction", "test_summary", "higher"),
    ("strict_pocket_fit_score", "test_summary", "higher"),
    ("unique_smiles_fraction", "test_summary", "higher"),
    ("mean_centroid_offset", "test_summary", "lower"),
    ("leakage_proxy_mean", "test_summary", "lower"),
    ("slot_activation_mean", "test_summary", "context"),
    ("gate_activation_mean", "test_summary", "context"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _delta(value, baseline):
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return value - baseline
    return None


def direction_is_better(metric_name, delta, direction):
    if delta is None or direction == "context":
        return None
    if direction == "higher":
        return delta >= 0.0
    if direction == "lower":
        return delta <= 0.0
    return None


def row(variant_label, metric_name, base_value, variant_value, direction, layer, source):
    delta = _delta(variant_value, base_value)
    if delta is None:
        return None
    return {
        "variant_label": variant_label,
        "metric_name": metric_name,
        "base_value": base_value,
        "variant_value": variant_value,
        "delta": delta,
        "direction_is_better": direction_is_better(metric_name, delta, direction),
        "layer": layer,
        "evidence_source": source,
    }


def main():
    parser = argparse.ArgumentParser(description="Write ablation delta table artifact.")
    parser.add_argument("artifact_dir", help="Directory containing claim_summary.json and ablation_matrix_summary.json")
    parser.add_argument(
        "--output",
        default="ablation_delta_table.json",
        help="Output file name under artifact_dir unless absolute path is provided.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    claim = load_json(artifact_dir / "claim_summary.json")
    matrix = load_json(artifact_dir / "ablation_matrix_summary.json")

    base_test = claim.get("test") or {}
    base_metrics = {name: base_test.get(name) for name, _, _ in METRICS}

    rows = []
    variants = matrix.get("variants") or []
    seen = set()
    for variant in variants:
        label = variant.get("variant_label")
        if not isinstance(label, str):
            continue
        test = variant.get("test") or {}
        for key, source, direction in METRICS:
            next_row = row(
                label,
                key,
                base_metrics.get(key),
                test.get(key),
                direction,
                "comparison_summary",
                source,
            )
            if next_row is not None:
                rows.append(next_row)
        seen.add(label)

    output = {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "base_test_metrics": base_metrics,
        "required_variants": REQUIRED_VARIANTS,
        "missing_required_variants": [label for label in REQUIRED_VARIANTS if label not in seen],
        "rows": sorted(rows, key=lambda row: (row["variant_label"], row["metric_name"])),
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = artifact_dir / output_path
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"ablation delta table written: {output_path}")


if __name__ == "__main__":
    main()
