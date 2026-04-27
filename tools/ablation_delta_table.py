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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _delta(value, baseline):
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return value - baseline
    return None


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
    base_metrics = {
        "candidate_valid_fraction": base_test.get("candidate_valid_fraction"),
        "strict_pocket_fit_score": base_test.get("strict_pocket_fit_score"),
        "leakage_proxy_mean": base_test.get("leakage_proxy_mean"),
        "slot_activation_mean": base_test.get("slot_activation_mean"),
        "gate_activation_mean": base_test.get("gate_activation_mean"),
    }

    rows = []
    variants = matrix.get("variants") or []
    seen = set()
    for variant in variants:
        label = variant.get("variant_label")
        if not isinstance(label, str):
            continue
        test = variant.get("test") or {}
        row = {"variant_label": label}
        for key, baseline in base_metrics.items():
            value = test.get(key)
            row[key] = value
            row[f"{key}_delta_vs_base"] = _delta(value, baseline)
        rows.append(row)
        seen.add(label)

    output = {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "base_test_metrics": base_metrics,
        "required_variants": REQUIRED_VARIANTS,
        "missing_required_variants": [label for label in REQUIRED_VARIANTS if label not in seen],
        "rows": sorted(rows, key=lambda row: row["variant_label"]),
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
