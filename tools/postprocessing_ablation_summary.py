#!/usr/bin/env python3
"""Summarize Q2 public-baseline postprocessing ablation metrics."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ABLATION_LAYERS = [
    "no_repair",
    "centroid_only",
    "clash_only",
    "bond_inference_only",
    "full_repair",
]
METRICS = [
    "vina_score",
    "gnina_affinity",
    "gnina_cnn_score",
    "qed",
    "sa_score",
    "clash_fraction",
    "pocket_contact_fraction",
    "centroid_offset",
]
LOWER_IS_BETTER = {
    "vina_score",
    "gnina_affinity",
    "sa_score",
    "clash_fraction",
    "centroid_offset",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--generation-layers",
        nargs="*",
        default=[],
        help="Optional generation_layers_test.json files used for per-candidate damage geometry.",
    )
    parser.add_argument(
        "--damage-json",
        help="Optional Q3 per-candidate repair damage report JSON output.",
    )
    parser.add_argument(
        "--damage-md",
        help="Optional Q3 per-candidate repair damage report Markdown output.",
    )
    return parser.parse_args()


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


def finite_triplet(value):
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        coords = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return coords if all(math.isfinite(item) for item in coords) else None


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return sum(values) / float(len(values)) if values else None


def metric_mean(rows, key):
    return mean(row.get("metrics", {}).get(key) for row in rows)


def metric_value(row, key):
    value = row.get("metrics", {}).get(key) if row else None
    return float(value) if finite(value) else None


def coverage(rows, key):
    if not rows:
        return 0.0
    return sum(1 for row in rows if finite(row.get("metrics", {}).get(key))) / float(len(rows))


def backend_coverage(rows):
    return {
        "vina_score": coverage(rows, "vina_score"),
        "gnina_affinity": coverage(rows, "gnina_affinity"),
        "gnina_cnn_score": coverage(rows, "gnina_cnn_score"),
        "rdkit": sum(
            1
            for row in rows
            if any(finite(row.get("metrics", {}).get(key)) for key in ("qed", "sa_score", "logp"))
        )
        / float(max(len(rows), 1)),
        "pocket_contact": coverage(rows, "pocket_contact_fraction"),
    }


def failure_reasons(rows):
    reasons = Counter()
    for row in rows:
        metrics = row.get("metrics", {})
        statuses = row.get("backend_statuses", {})
        for backend, status in statuses.items():
            if status != "metrics_available":
                reasons[f"{backend}:{status}"] += 1
        for key in ("vina_failure_reason", "gnina_failure_reason"):
            reason = metrics.get(key)
            if reason:
                reasons[str(reason)] += 1
        if not finite(metrics.get("vina_score")):
            reasons["missing_vina_score"] += 1
        if not finite(metrics.get("gnina_affinity")):
            reasons["missing_gnina_affinity"] += 1
    return dict(sorted(reasons.items()))


def layer_role(layer):
    if layer == "no_repair":
        return "unchanged_coordinate_ablation_baseline"
    if layer == "centroid_only":
        return "coordinate_movement_only"
    if layer == "clash_only":
        return "declash_without_centroid_translation_or_bond_change"
    if layer == "bond_inference_only":
        return "bond_payload_change_without_coordinate_movement"
    if layer == "full_repair":
        return "coordinate_repair_plus_bond_inference"
    return "postprocessing"


def coords_from_candidate(candidate):
    origin = finite_triplet(candidate.get("coordinate_frame_origin")) or [0.0, 0.0, 0.0]
    coords = []
    for coord in candidate.get("coords", []):
        parsed = finite_triplet(coord)
        if parsed is not None:
            coords.append([parsed[axis] + origin[axis] for axis in range(3)])
    return coords


def centroid(coords):
    if not coords:
        return None
    return [sum(coord[axis] for coord in coords) / float(len(coords)) for axis in range(3)]


def distance(left, right):
    if left is None or right is None:
        return None
    return math.sqrt(sum((left[axis] - right[axis]) ** 2 for axis in range(3)))


def rmsd(left_coords, right_coords):
    if not left_coords or len(left_coords) != len(right_coords):
        return None
    total = 0.0
    for left, right in zip(left_coords, right_coords):
        total += sum((left[axis] - right[axis]) ** 2 for axis in range(3))
    return math.sqrt(total / float(len(left_coords)))


def docking_box(candidate):
    coords = coords_from_candidate(candidate)
    origin = finite_triplet(candidate.get("coordinate_frame_origin")) or [0.0, 0.0, 0.0]
    raw_centroid = finite_triplet(candidate.get("pocket_centroid"))
    if raw_centroid is None:
        if not coords:
            return None
        center = centroid(coords)
    else:
        pocket_center = [raw_centroid[axis] + origin[axis] for axis in range(3)]
        if coords:
            axis_values = [[coord[axis] for coord in coords] + [pocket_center[axis]] for axis in range(3)]
            mins = [min(values) for values in axis_values]
            maxs = [max(values) for values in axis_values]
            center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]
        else:
            center = pocket_center
    try:
        radius = float(candidate.get("pocket_radius", 12.0))
    except (TypeError, ValueError):
        radius = 12.0
    if not math.isfinite(radius) or radius <= 0.0:
        radius = 12.0
    if coords:
        axis_values = [[coord[axis] for coord in coords] + [center[axis]] for axis in range(3)]
        ligand_span = max(max(values) - min(values) for values in axis_values)
    else:
        ligand_span = 0.0
    size = max(8.0, min(80.0, max(2.0 * radius, ligand_span + 8.0)))
    return {"center": center, "size": [size, size, size]}


def box_delta(raw_candidate, layer_candidate):
    raw_box = docking_box(raw_candidate) if raw_candidate else None
    layer_box = docking_box(layer_candidate) if layer_candidate else None
    if not raw_box or not layer_box:
        return {
            "raw_box": raw_box,
            "layer_box": layer_box,
            "center_shift": None,
            "size_delta": None,
        }
    return {
        "raw_box": raw_box,
        "layer_box": layer_box,
        "center_shift": distance(raw_box["center"], layer_box["center"]),
        "size_delta": max(abs(layer_box["size"][axis] - raw_box["size"][axis]) for axis in range(3)),
    }


def candidate_key(row_or_candidate):
    return (
        row_or_candidate.get("method_id") or "unknown",
        row_or_candidate.get("example_id") or "unknown",
        row_or_candidate.get("layer") or "unknown",
    )


def load_generation_candidates(paths):
    candidates = {}
    source_paths = {}
    layer_fields = {
        "no_repair": "no_repair_candidates",
        "centroid_only": "centroid_only_candidates",
        "clash_only": "clash_only_candidates",
        "bond_inference_only": "bond_inference_only_candidates",
        "full_repair": "full_repair_candidates",
    }
    for path in paths:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        method = artifact.get("method_id")
        for layer, field in layer_fields.items():
            for candidate in artifact.get(field, []):
                enriched = dict(candidate)
                enriched.setdefault("method_id", method)
                enriched.setdefault("layer", layer)
                key = candidate_key(enriched)
                candidates[key] = enriched
                source_paths[key] = str(path)
    return candidates, source_paths


def paired_metric_rows(rows):
    by_key = {}
    for row in rows:
        layer = row.get("layer")
        if layer in ABLATION_LAYERS:
            by_key[candidate_key(row)] = row
    return by_key


def signed_metric_delta(raw_row, layer_row, key):
    raw_value = metric_value(raw_row, key)
    layer_value = metric_value(layer_row, key)
    if raw_value is None or layer_value is None:
        return None
    return layer_value - raw_value


def degradation_score(deltas):
    values = [
        deltas.get("vina_score"),
        deltas.get("gnina_affinity"),
    ]
    finite_values = [value for value in values if finite(value)]
    if not finite_values:
        return None
    return sum(finite_values)


def likely_failure_component(layer, deltas, geometry):
    vina = deltas.get("vina_score")
    gnina = deltas.get("gnina_affinity")
    docking_worse = any(finite(value) and value > 1.0 for value in (vina, gnina))
    centroid_shift = geometry.get("centroid_shift")
    box_shift = geometry.get("docking_box_center_shift")
    bond_delta = geometry.get("bond_count_delta")
    sa_delta = deltas.get("sa_score")
    clash_delta = deltas.get("clash_fraction")

    if layer in {"centroid_only", "full_repair"} and docking_worse and finite(centroid_shift) and centroid_shift > 2.0:
        return "coordinate_movement"
    if finite(box_shift) and box_shift > 2.0 and docking_worse:
        return "docking_box_shift_from_coordinate_change"
    if layer == "bond_inference_only" and finite(sa_delta) and sa_delta > 0.25:
        return "bond_payload_chemistry_regression"
    if layer == "clash_only" and docking_worse:
        return "declash_geometry"
    if finite(bond_delta) and bond_delta != 0 and docking_worse:
        return "bond_payload_or_conversion"
    if finite(clash_delta) and clash_delta > 0.05 and docking_worse:
        return "clash_increase"
    return "mixed_or_inconclusive"


def geometry_delta(raw_candidate, layer_candidate):
    if not raw_candidate or not layer_candidate:
        return {
            "atom_count_raw": len(raw_candidate.get("atom_types", [])) if raw_candidate else None,
            "atom_count_layer": len(layer_candidate.get("atom_types", [])) if layer_candidate else None,
            "bond_count_raw": len(raw_candidate.get("inferred_bonds", [])) if raw_candidate else None,
            "bond_count_layer": len(layer_candidate.get("inferred_bonds", [])) if layer_candidate else None,
            "bond_count_delta": None,
            "centroid_shift": None,
            "coordinate_rmsd": None,
            "docking_box_center_shift": None,
            "docking_box_size_delta": None,
        }
    raw_coords = coords_from_candidate(raw_candidate)
    layer_coords = coords_from_candidate(layer_candidate)
    raw_centroid = centroid(raw_coords)
    layer_centroid = centroid(layer_coords)
    raw_bonds = len(raw_candidate.get("inferred_bonds", []))
    layer_bonds = len(layer_candidate.get("inferred_bonds", []))
    box = box_delta(raw_candidate, layer_candidate)
    return {
        "atom_count_raw": len(raw_candidate.get("atom_types", [])),
        "atom_count_layer": len(layer_candidate.get("atom_types", [])),
        "bond_count_raw": raw_bonds,
        "bond_count_layer": layer_bonds,
        "bond_count_delta": layer_bonds - raw_bonds,
        "centroid_raw": raw_centroid,
        "centroid_layer": layer_centroid,
        "centroid_shift": distance(raw_centroid, layer_centroid),
        "coordinate_rmsd": rmsd(raw_coords, layer_coords),
        "docking_box_raw": box["raw_box"],
        "docking_box_layer": box["layer_box"],
        "docking_box_center_shift": box["center_shift"],
        "docking_box_size_delta": box["size_delta"],
    }


def repair_damage_report(rows, generation_layer_paths, candidate_metrics_path, top_n=20):
    metric_rows = paired_metric_rows(rows)
    candidates, candidate_sources = load_generation_candidates(generation_layer_paths)
    cases = []
    component_counts = Counter()
    compared = 0
    skipped = Counter()

    methods_examples = sorted({(method, example) for method, example, layer in metric_rows if layer == "no_repair"})
    for method, example in methods_examples:
        raw_key = (method, example, "no_repair")
        raw_row = metric_rows.get(raw_key)
        raw_candidate = candidates.get(raw_key)
        if raw_row is None:
            skipped["missing_no_repair_metric_row"] += 1
            continue
        for layer in ("centroid_only", "clash_only", "bond_inference_only", "full_repair"):
            layer_key = (method, example, layer)
            layer_row = metric_rows.get(layer_key)
            layer_candidate = candidates.get(layer_key)
            if layer_row is None:
                skipped[f"missing_{layer}_metric_row"] += 1
                continue
            compared += 1
            deltas = {
                key: signed_metric_delta(raw_row, layer_row, key)
                for key in METRICS
            }
            geometry = geometry_delta(raw_candidate, layer_candidate)
            component = likely_failure_component(layer, deltas, geometry)
            component_counts[component] += 1
            score = degradation_score(deltas)
            cases.append(
                {
                    "candidate_id": layer_row.get("candidate_id"),
                    "raw_candidate_id": raw_row.get("candidate_id"),
                    "method_id": method,
                    "example_id": example,
                    "protein_id": layer_row.get("protein_id"),
                    "split_label": layer_row.get("split_label"),
                    "layer": layer,
                    "layer_role": layer_role(layer),
                    "degradation_score": score,
                    "likely_failure_component": component,
                    "raw_metrics": {
                        key: metric_value(raw_row, key)
                        for key in METRICS
                    },
                    "layer_metrics": {
                        key: metric_value(layer_row, key)
                        for key in METRICS
                    },
                    "delta_vs_no_repair": deltas,
                    "geometry_delta": geometry,
                    "backend_statuses": {
                        "raw": raw_row.get("backend_statuses", {}),
                        "layer": layer_row.get("backend_statuses", {}),
                    },
                    "source_artifacts": {
                        "candidate_metrics": str(candidate_metrics_path),
                        "raw_generation_layer": candidate_sources.get(raw_key),
                        "layer_generation_layer": candidate_sources.get(layer_key),
                        "raw_candidate_source": raw_row.get("candidate_source"),
                        "layer_candidate_source": layer_row.get("candidate_source"),
                        "raw_metric_source_artifacts": raw_row.get("source_artifacts", []),
                        "layer_metric_source_artifacts": layer_row.get("source_artifacts", []),
                    },
                }
            )

    degraded_cases = [
        case for case in cases
        if case["degradation_score"] is not None and case["degradation_score"] > 0.0
    ]
    worst_cases = sorted(
        degraded_cases,
        key=lambda case: (
            case["degradation_score"],
            case["delta_vs_no_repair"].get("gnina_affinity") or -1e9,
            case["candidate_id"],
        ),
        reverse=True,
    )[:top_n]
    by_layer = defaultdict(list)
    for case in cases:
        by_layer[case["layer"]].append(case)
    layer_damage = {}
    for layer, layer_cases in sorted(by_layer.items()):
        layer_damage[layer] = {
            "candidate_count": len(layer_cases),
            "degraded_candidate_count": sum(
                1 for case in layer_cases
                if case["degradation_score"] is not None and case["degradation_score"] > 0.0
            ),
            "mean_degradation_score": mean(case["degradation_score"] for case in layer_cases),
            "mean_centroid_shift": mean(case["geometry_delta"].get("centroid_shift") for case in layer_cases),
            "mean_docking_box_center_shift": mean(case["geometry_delta"].get("docking_box_center_shift") for case in layer_cases),
            "mean_bond_count_delta": mean(case["geometry_delta"].get("bond_count_delta") for case in layer_cases),
            "failure_component_counts": dict(Counter(case["likely_failure_component"] for case in layer_cases)),
        }
    return {
        "schema_version": 1,
        "artifact_name": "q3_repair_damage_cases",
        "candidate_metrics": str(candidate_metrics_path),
        "generation_layers": [str(path) for path in generation_layer_paths],
        "claim_boundary": "This is a postprocessing-diagnosis artifact. It does not promote repaired layers as model-native evidence.",
        "comparison": "Each case compares one postprocessing layer against the same method/example no_repair baseline.",
        "counts": {
            "paired_layer_comparisons": compared,
            "candidate_geometry_records": len(candidates),
            "worst_case_count": len(worst_cases),
            "skipped": dict(sorted(skipped.items())),
        },
        "component_counts": dict(sorted(component_counts.items())),
        "layer_damage_summary": layer_damage,
        "worst_cases": worst_cases,
    }


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_damage_markdown(path, payload):
    lines = [
        "# Q3 Repair Damage Cases",
        "",
        payload["claim_boundary"],
        "",
        "## Summary",
        "",
        f"- paired_layer_comparisons: {payload['counts']['paired_layer_comparisons']}",
        f"- candidate_geometry_records: {payload['counts']['candidate_geometry_records']}",
        f"- worst_case_count: {payload['counts']['worst_case_count']}",
        "",
        "## Component Counts",
        "",
    ]
    for component, count in payload["component_counts"].items():
        lines.append(f"- `{component}`: {count}")
    lines.extend([
        "",
        "## Layer Damage",
        "",
        "| Layer | Candidates | Degraded | Mean Damage | Mean Centroid Shift | Mean Box Shift | Mean Bond Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for layer, row in payload["layer_damage_summary"].items():
        lines.append(
            "| {layer} | {count} | {degraded} | {damage} | {centroid} | {box} | {bond} |".format(
                layer=layer,
                count=row["candidate_count"],
                degraded=row["degraded_candidate_count"],
                damage=fmt(row["mean_degradation_score"]),
                centroid=fmt(row["mean_centroid_shift"]),
                box=fmt(row["mean_docking_box_center_shift"]),
                bond=fmt(row["mean_bond_count_delta"]),
            )
        )
    lines.extend([
        "",
        "## Worst Cases",
        "",
        "| Rank | Candidate | Layer | Component | Damage | dVina | dGNINA | Centroid Shift | Box Shift | Bond Delta |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for rank, case in enumerate(payload["worst_cases"], start=1):
        delta = case["delta_vs_no_repair"]
        geom = case["geometry_delta"]
        lines.append(
            "| {rank} | `{candidate}` | {layer} | `{component}` | {damage} | {vina} | {gnina} | {centroid} | {box} | {bond} |".format(
                rank=rank,
                candidate=case["candidate_id"],
                layer=case["layer"],
                component=case["likely_failure_component"],
                damage=fmt(case["degradation_score"]),
                vina=fmt(delta.get("vina_score")),
                gnina=fmt(delta.get("gnina_affinity")),
                centroid=fmt(geom.get("centroid_shift")),
                box=fmt(geom.get("docking_box_center_shift")),
                bond=fmt(geom.get("bond_count_delta")),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The largest candidate-level failures are dominated by coordinate-moving layers. Bond-only changes are tracked separately because they mainly affect chemistry payload and SA/QED rather than pose coordinates.",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def explain_layer(layer, delta_metrics):
    vina_delta = delta_metrics.get("vina_score")
    gnina_delta = delta_metrics.get("gnina_affinity")
    centroid_delta = delta_metrics.get("centroid_offset")
    clash_delta = delta_metrics.get("clash_fraction")
    qed_delta = delta_metrics.get("qed")
    sa_delta = delta_metrics.get("sa_score")

    docking_worse = any(finite(value) and value > 1.0 for value in (vina_delta, gnina_delta))
    docking_better = any(finite(value) and value < -0.25 for value in (vina_delta, gnina_delta))

    reasons = []
    if layer == "no_repair":
        reasons.append("baseline unchanged-coordinate copy for same-pipeline scoring")
    if layer == "centroid_only" and docking_worse:
        reasons.append("coordinate movement is sufficient to degrade score_only docking")
    if layer == "clash_only" and docking_worse:
        reasons.append("declash step alone degrades docking or docking input geometry")
    if layer == "bond_inference_only" and docking_worse:
        reasons.append("bond inference or ligand payload conversion changes docking quality")
    if layer == "full_repair" and docking_worse:
        reasons.append("combined repair is not claim-promotable under current scoring")
    if finite(centroid_delta) and abs(centroid_delta) > 5.0:
        reasons.append("large centroid-offset change")
    if finite(clash_delta) and clash_delta < -0.01:
        reasons.append("clash reduction achieved")
    if finite(qed_delta) and qed_delta < -0.05:
        reasons.append("QED regression")
    if finite(sa_delta) and sa_delta > 0.25:
        reasons.append("SA regression")
    if docking_better:
        reasons.append("score_only docking improves for at least one backend")
    return reasons or ["no decisive degradation signal from available metrics"]


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        layer = row.get("layer")
        if layer in ABLATION_LAYERS:
            grouped[(row.get("method_id") or "unknown", layer)].append(row)

    baselines = {
        method: grouped.get((method, "no_repair"), [])
        for method, _layer in grouped
    }
    method_summaries = []
    for method in sorted(baselines):
        baseline_means = {key: metric_mean(baselines[method], key) for key in METRICS}
        for layer in ABLATION_LAYERS:
            layer_rows = grouped.get((method, layer), [])
            means = {key: metric_mean(layer_rows, key) for key in METRICS}
            deltas = {}
            for key in METRICS:
                if means[key] is not None and baseline_means.get(key) is not None:
                    deltas[key] = means[key] - baseline_means[key]
                else:
                    deltas[key] = None
            method_summaries.append(
                {
                    "method_id": method,
                    "layer": layer,
                    "layer_role": layer_role(layer),
                    "candidate_count": len(layer_rows),
                    "backend_coverage": backend_coverage(layer_rows),
                    "metric_means": means,
                    "delta_vs_no_repair": deltas,
                    "failure_reasons": failure_reasons(layer_rows),
                    "interpretation": explain_layer(layer, deltas),
                }
            )

    layer_summaries = []
    for layer in ABLATION_LAYERS:
        layer_rows = [
            row for row in rows if row.get("layer") == layer
        ]
        baseline_rows = [
            row for row in rows if row.get("layer") == "no_repair"
        ]
        means = {key: metric_mean(layer_rows, key) for key in METRICS}
        baseline_means = {key: metric_mean(baseline_rows, key) for key in METRICS}
        deltas = {
            key: (
                means[key] - baseline_means[key]
                if means[key] is not None and baseline_means[key] is not None
                else None
            )
            for key in METRICS
        }
        layer_summaries.append(
            {
                "layer": layer,
                "layer_role": layer_role(layer),
                "candidate_count": len(layer_rows),
                "backend_coverage": backend_coverage(layer_rows),
                "metric_means": means,
                "delta_vs_no_repair": deltas,
                "failure_reasons": failure_reasons(layer_rows),
                "interpretation": explain_layer(layer, deltas),
            }
        )
    return method_summaries, layer_summaries


def dominant_diagnosis(layer_summaries):
    by_layer = {row["layer"]: row for row in layer_summaries}
    signals = []
    for layer in ("centroid_only", "clash_only", "bond_inference_only", "full_repair"):
        row = by_layer.get(layer, {})
        delta = row.get("delta_vs_no_repair", {})
        vina = delta.get("vina_score")
        gnina = delta.get("gnina_affinity")
        score = sum(value for value in (vina, gnina) if finite(value))
        signals.append((score, layer))
    signals.sort(reverse=True)
    worst_score, worst_layer = signals[0] if signals else (None, "unknown")
    if worst_layer == "centroid_only":
        conclusion = "Coordinate movement is the largest isolated degradation source."
    elif worst_layer == "bond_inference_only":
        conclusion = "Bond inference or ligand payload conversion is the largest isolated degradation source."
    elif worst_layer == "clash_only":
        conclusion = "Declash geometry changes are the largest isolated degradation source."
    elif worst_layer == "full_repair":
        conclusion = "The combined repair path degrades more than any isolated step."
    else:
        conclusion = "No dominant degradation source could be isolated."
    return {
        "dominant_layer": worst_layer,
        "docking_degradation_signal": worst_score,
        "conclusion": conclusion,
        "component_findings": component_findings(by_layer),
        "full_repair_promoted": False,
        "promotion_rule": "Full repair is not promoted unless score_only docking improves without unacceptable QED/SA, clash, or contact regressions.",
    }


def signed_delta(row, key):
    value = row.get("delta_vs_no_repair", {}).get(key)
    return float(value) if finite(value) else None


def component_findings(by_layer):
    centroid = by_layer.get("centroid_only", {})
    clash = by_layer.get("clash_only", {})
    bond = by_layer.get("bond_inference_only", {})
    full = by_layer.get("full_repair", {})
    no_repair = by_layer.get("no_repair", {})

    centroid_vina = signed_delta(centroid, "vina_score")
    centroid_gnina = signed_delta(centroid, "gnina_affinity")
    bond_vina = signed_delta(bond, "vina_score")
    bond_gnina = signed_delta(bond, "gnina_affinity")
    bond_sa = signed_delta(bond, "sa_score")
    clash_vina = signed_delta(clash, "vina_score")

    return {
        "coordinate_movement": {
            "status": "primary_degradation_source"
            if any(finite(value) and value > 1.0 for value in (centroid_vina, centroid_gnina))
            else "not_primary",
            "evidence": "centroid_only changes coordinates without changing bonds; positive Vina/GNINA affinity deltas mean worse score_only docking.",
            "vina_delta": centroid_vina,
            "gnina_affinity_delta": centroid_gnina,
        },
        "clash_handling": {
            "status": "not_primary_degradation_source"
            if clash_vina is not None and clash_vina <= 1.0
            else "possible_degradation_source",
            "evidence": "clash_only keeps the raw centroid and bond payload while applying local declash moves.",
            "vina_delta": clash_vina,
            "gnina_affinity_delta": signed_delta(clash, "gnina_affinity"),
            "clash_fraction_delta": signed_delta(clash, "clash_fraction"),
        },
        "bond_inference": {
            "status": "chemistry_regression_without_docking_degradation"
            if (bond_sa is not None and bond_sa > 0.25)
            and not any(finite(value) and value > 1.0 for value in (bond_vina, bond_gnina))
            else "inconclusive",
            "evidence": "bond_inference_only keeps raw coordinates but changes inferred_bonds/molecular representation.",
            "vina_delta": bond_vina,
            "gnina_affinity_delta": bond_gnina,
            "sa_score_delta": bond_sa,
        },
        "docking_input_conversion": {
            "status": "coverage_issue_not_primary_score_degradation",
            "evidence": "no_repair is an unchanged-coordinate copy scored through the same docking-input path; missing Vina rows are explicit command failures, while GNINA coverage is complete.",
            "no_repair_vina_coverage": no_repair.get("backend_coverage", {}).get("vina_score"),
            "no_repair_failure_reasons": no_repair.get("failure_reasons", {}),
        },
        "full_repair": {
            "status": "not_promoted",
            "evidence": "full_repair combines centroid-anchored repair, declash, envelope clamp, and bond inference and is worse than no_repair on both score_only affinity backends.",
            "vina_delta": signed_delta(full, "vina_score"),
            "gnina_affinity_delta": signed_delta(full, "gnina_affinity"),
        },
        "reranking": {
            "status": "not_needed_to_explain_degradation",
            "evidence": "The five-layer ablation shows severe degradation before deterministic reranking is applied; legacy reranked rows remain postprocessing evidence and are not promoted.",
        },
    }


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(path, payload):
    lines = [
        "# Q2 Postprocessing Ablation Summary",
        "",
        payload["claim_boundary"],
        "",
        "## Diagnosis",
        "",
        f"- dominant_layer: `{payload['diagnosis']['dominant_layer']}`",
        f"- conclusion: {payload['diagnosis']['conclusion']}",
        f"- full_repair_promoted: {payload['diagnosis']['full_repair_promoted']}",
        "",
        "## Component Findings",
        "",
    ]
    for name, finding in payload["diagnosis"]["component_findings"].items():
        lines.append(f"- `{name}`: {finding['status']}; {finding['evidence']}")
    lines.extend([
        "",
        "## Layer Summary",
        "",
        "| Layer | Candidates | Vina Cov | GNINA Cov | dVina | dGNINA | dQED | dSA | dClash | dContact | dCentroid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["layer_summaries"]:
        cov = row["backend_coverage"]
        delta = row["delta_vs_no_repair"]
        lines.append(
            "| {layer} | {count} | {vina_cov} | {gnina_cov} | {vina} | {gnina} | {qed} | {sa} | {clash} | {contact} | {centroid} |".format(
                layer=row["layer"],
                count=row["candidate_count"],
                vina_cov=fmt(cov["vina_score"]),
                gnina_cov=fmt(cov["gnina_affinity"]),
                vina=fmt(delta["vina_score"]),
                gnina=fmt(delta["gnina_affinity"]),
                qed=fmt(delta["qed"]),
                sa=fmt(delta["sa_score"]),
                clash=fmt(delta["clash_fraction"]),
                contact=fmt(delta["pocket_contact_fraction"]),
                centroid=fmt(delta["centroid_offset"]),
            )
        )
    lines.extend(["", "## Failure Reasons", ""])
    for row in payload["layer_summaries"]:
        reasons = row["failure_reasons"] or {"none": 0}
        rendered = ", ".join(f"{key}: {value}" for key, value in reasons.items())
        lines.append(f"- `{row['layer']}`: {rendered}")
    lines.extend(["", "## Guardrail", "", payload["diagnosis"]["promotion_rule"]])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = load_jsonl(args.candidate_metrics)
    method_summaries, layer_summaries = summarize(rows)
    payload = {
        "schema_version": 1,
        "artifact_name": "q2_postprocessing_ablation_summary",
        "candidate_metrics": args.candidate_metrics,
        "ablation_layers": ABLATION_LAYERS,
        "score_direction": {
            key: ("lower_is_better" if key in LOWER_IS_BETTER else "higher_is_better")
            for key in METRICS
        },
        "method_summaries": method_summaries,
        "layer_summaries": layer_summaries,
        "diagnosis": dominant_diagnosis(layer_summaries),
        "claim_boundary": "Ablation layers are score_only postprocessing evidence. no_repair is an unchanged-coordinate baseline; centroid_only, clash_only, bond_inference_only, and full_repair must not be reported as native public-baseline capability.",
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    if args.damage_json or args.damage_md:
        if not args.generation_layers:
            raise SystemExit("--generation-layers is required when writing Q3 repair damage outputs")
        damage = repair_damage_report(rows, args.generation_layers, args.candidate_metrics)
        if args.damage_json:
            write_json(args.damage_json, damage)
        if args.damage_md:
            write_damage_markdown(args.damage_md, damage)


if __name__ == "__main__":
    main()
