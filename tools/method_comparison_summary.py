#!/usr/bin/env python3
"""Export a compact method-comparison summary from a claim artifact directory."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Write method comparison summary artifact.")
    parser.add_argument("artifact_dir", help="Artifact directory containing claim_summary.json")
    parser.add_argument(
        "--output",
        default="method_comparison_summary.json",
        help="Output file name written under artifact_dir unless absolute path is provided.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    claim = load_json(artifact_dir / "claim_summary.json")
    method_comparison = claim.get("method_comparison") or {}
    methods = method_comparison.get("methods") or []

    summary = {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "active_method": method_comparison.get("active_method"),
        "method_count": len(methods),
        "methods": [
            {
                "method_id": row.get("method_id"),
                "method_name": row.get("method_name"),
                "evidence_role": row.get("evidence_role"),
                "trainable": row.get("trainable"),
                "sampling_steps": row.get("sampling_steps"),
                "wall_time_ms": row.get("wall_time_ms"),
                "native_valid_fraction": row.get("native_valid_fraction"),
                "native_pocket_contact_fraction": row.get("native_pocket_contact_fraction"),
                "native_clash_fraction": row.get("native_clash_fraction"),
                "available_layers": row.get("available_layers"),
            }
            for row in methods
        ],
        "layered_generation_artifacts": {
            "validation": str(artifact_dir / "generation_layers_validation.json"),
            "test": str(artifact_dir / "generation_layers_test.json"),
        },
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = artifact_dir / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"method comparison summary written: {output}")


if __name__ == "__main__":
    main()
