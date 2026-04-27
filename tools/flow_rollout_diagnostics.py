#!/usr/bin/env python3
"""Generate compact flow-matching rollout diagnostics for reviewer artifacts."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summary(values):
    clean = [value for value in values if isinstance(value, (int, float))]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": sum(clean) / len(clean),
    }


def main():
    parser = argparse.ArgumentParser(description="Write flow rollout diagnostics artifact.")
    parser.add_argument("artifact_dir", help="Artifact directory containing claim_summary.json")
    parser.add_argument(
        "--output",
        default="flow_rollout_diagnostics.json",
        help="Output file name under artifact_dir unless absolute path is provided.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    claim = load_json(artifact_dir / "claim_summary.json")
    method_comparison = claim.get("method_comparison") or {}
    methods = method_comparison.get("methods") or []
    flow_metrics = method_comparison.get("flow_metrics") or {}
    raw_output = flow_metrics.get("raw_output") or {}
    repaired_output = flow_metrics.get("repaired_output") or {}
    versus = flow_metrics.get("versus_conditioned_denoising") or {}

    flow_method = next((row for row in methods if row.get("method_id") == "flow_matching"), {})
    active_id = (method_comparison.get("active_method") or {}).get("method_id")
    active_method = next((row for row in methods if row.get("method_id") == active_id), {})

    raw_disp = _safe_number(raw_output.get("mean_displacement"))
    repaired_disp = _safe_number(repaired_output.get("mean_displacement"))
    trend_values = [
        {"stage": "raw_output", "mean_displacement": raw_disp},
        {"stage": "repaired_output", "mean_displacement": repaired_disp},
    ]

    diagnostics = {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "flow_method_id": "flow_matching",
        "displacement_distribution": {
            "flow_outputs": _summary([raw_disp, repaired_disp]),
            "active_raw_reference": _safe_number(
                ((claim.get("layered_generation_metrics") or {}).get("raw_rollout") or {}).get(
                    "mean_displacement"
                )
            ),
        },
        "endpoint_consistency": {
            "centroid_offset_delta_raw_to_repaired": (
                _safe_number(repaired_output.get("mean_centroid_offset"))
                - _safe_number(raw_output.get("mean_centroid_offset"))
                if _safe_number(repaired_output.get("mean_centroid_offset")) is not None
                and _safe_number(raw_output.get("mean_centroid_offset")) is not None
                else None
            ),
            "clash_fraction_delta_raw_to_repaired": (
                _safe_number(repaired_output.get("clash_fraction"))
                - _safe_number(raw_output.get("clash_fraction"))
                if _safe_number(repaired_output.get("clash_fraction")) is not None
                and _safe_number(raw_output.get("clash_fraction")) is not None
                else None
            ),
            "valid_fraction_delta_raw_to_repaired": (
                _safe_number(repaired_output.get("valid_fraction"))
                - _safe_number(raw_output.get("valid_fraction"))
                if _safe_number(repaired_output.get("valid_fraction")) is not None
                and _safe_number(raw_output.get("valid_fraction")) is not None
                else None
            ),
            "versus_conditioned_denoising": versus,
        },
        "stepwise_displacement_trend": {
            "stages": trend_values,
            "non_increasing": (
                raw_disp is not None and repaired_disp is not None and repaired_disp <= raw_disp
            ),
        },
        "gate_usage_summary": {
            "flow_gate_activation_mean": _safe_number(flow_method.get("gate_activation_mean")),
            "flow_slot_activation_mean": _safe_number(flow_method.get("slot_activation_mean")),
            "active_method_id": active_id,
            "active_gate_activation_mean": _safe_number(active_method.get("gate_activation_mean")),
            "active_slot_activation_mean": _safe_number(active_method.get("slot_activation_mean")),
            "gate_delta_flow_minus_active": (
                _safe_number(flow_method.get("gate_activation_mean"))
                - _safe_number(active_method.get("gate_activation_mean"))
                if _safe_number(flow_method.get("gate_activation_mean")) is not None
                and _safe_number(active_method.get("gate_activation_mean")) is not None
                else None
            ),
            "slot_delta_flow_minus_active": (
                _safe_number(flow_method.get("slot_activation_mean"))
                - _safe_number(active_method.get("slot_activation_mean"))
                if _safe_number(flow_method.get("slot_activation_mean")) is not None
                and _safe_number(active_method.get("slot_activation_mean")) is not None
                else None
            ),
        },
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = artifact_dir / output
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")
    print(f"flow rollout diagnostics written: {output}")


if __name__ == "__main__":
    main()
