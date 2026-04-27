#!/usr/bin/env python3
"""Optional AutoDock Vina score-only command adapter.

The adapter accepts the repository candidate JSON contract and emits numeric
schema-versioned metrics. It intentionally degrades to explicit availability or
input-completeness metrics when Vina, receptor PDBQT, or ligand PDBQT payloads
are absent, so smoke runs remain reproducible without making Vina mandatory.
"""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


AFFINITY_RE = re.compile(
    r"(?:Affinity|Estimated Free Energy of Binding)\s*:?\s*([-+]?\d+(?:\.\d+)?)"
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Score candidates with AutoDock Vina when available.")
    parser.add_argument("--vina-executable", default=os.environ.get("VINA_EXECUTABLE", "vina"))
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    return parser.parse_args(argv[1:])


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def structured_payload(aggregate_metrics, candidate_metrics):
    return {
        "schema_version": 1.0,
        "aggregate_metrics": aggregate_metrics,
        "candidate_metrics": candidate_metrics,
    }


def attach_contract_counts(payload, input_count, scored_count, failure_count):
    payload["input_count"] = float(input_count)
    payload["scored_count"] = float(scored_count)
    payload["failure_count"] = float(failure_count)
    payload["docking_score_coverage_fraction"] = (
        float(scored_count) / float(max(input_count, 1))
    )
    return payload


def attach_runtime(payload, started_at, input_count):
    elapsed = time.perf_counter() - started_at
    payload["backend_wall_clock_seconds"] = elapsed
    payload["backend_candidates_per_second"] = (
        float(input_count) / elapsed if elapsed > 0.0 else None
    )
    payload["backend_runtime_status"] = "measured_by_backend_adapter"
    return payload


def parse_affinity(output):
    match = AFFINITY_RE.search(output)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def finite_triplet(value):
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        coords = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in coords):
        return None
    return coords


def docking_box(candidate):
    centroid = finite_triplet(candidate.get("pocket_centroid"))
    origin = finite_triplet(candidate.get("coordinate_frame_origin")) or (0.0, 0.0, 0.0)
    coords = []
    for coord in candidate.get("coords", []):
        parsed = finite_triplet(coord)
        if parsed is not None:
            coords.append(tuple(parsed[dim] + origin[dim] for dim in range(3)))
    if centroid is None:
        if not coords:
            return None, "missing_docking_box"
        center = tuple(sum(coord[dim] for coord in coords) / float(len(coords)) for dim in range(3))
    else:
        pocket_center = tuple(centroid[dim] + origin[dim] for dim in range(3))
        if coords:
            axis_values = [[coord[dim] for coord in coords] + [pocket_center[dim]] for dim in range(3)]
            mins = tuple(min(values) for values in axis_values)
            maxs = tuple(max(values) for values in axis_values)
            center = tuple((mins[dim] + maxs[dim]) / 2.0 for dim in range(3))
        else:
            center = pocket_center
    try:
        radius = float(candidate.get("pocket_radius", 12.0))
    except (TypeError, ValueError):
        radius = 12.0
    if not math.isfinite(radius) or radius <= 0.0:
        radius = 12.0
    if coords:
        axis_values = [[coord[dim] for coord in coords] + [center[dim]] for dim in range(3)]
        ligand_span = max(max(values) - min(values) for values in axis_values)
    else:
        ligand_span = 0.0
    size = max(8.0, min(80.0, max(2.0 * radius, ligand_span + 8.0)))
    return {
        "center": center,
        "size": (size, size, size),
    }, None


def looks_like_pdbqt_payload(payload):
    if not payload:
        return False
    markers = ("ROOT", "ENDROOT", "TORSDOF", "ATOM", "HETATM")
    return any(marker in payload for marker in markers)


def ligand_input(candidate, workdir):
    source_ligand = candidate.get("source_ligand_path")
    if source_ligand:
        ligand_source = Path(source_ligand)
        if ligand_source.is_file() and ligand_source.suffix.lower() == ".pdbqt":
            return ligand_source, None
    ligand_payload = candidate.get("molecular_representation")
    if not ligand_payload:
        return None, "missing_ligand_pdbqt_input"
    ligand_path = Path(workdir) / f"{candidate.get('example_id', 'candidate')}.pdbqt"
    ligand_path.write_text(ligand_payload, encoding="utf-8")
    return ligand_path, None


def score_candidate(vina_executable, candidate, workdir):
    receptor = candidate.get("source_pocket_path")
    if not receptor:
        return None, "missing_receptor_path"

    receptor_path = Path(receptor)
    if not receptor_path.is_file():
        return None, "receptor_path_not_found"
    if receptor_path.suffix.lower() != ".pdbqt":
        return None, "receptor_is_not_pdbqt"

    ligand_path, ligand_reason = ligand_input(candidate, workdir)
    if ligand_reason is not None:
        return None, ligand_reason
    box, box_reason = docking_box(candidate)
    if box_reason is not None:
        return None, box_reason

    completed = subprocess.run(
        [
            vina_executable,
            "--score_only",
            "--receptor",
            str(receptor_path),
            "--ligand",
            str(ligand_path),
            "--center_x",
            str(box["center"][0]),
            "--center_y",
            str(box["center"][1]),
            "--center_z",
            str(box["center"][2]),
            "--size_x",
            str(box["size"][0]),
            "--size_y",
            str(box["size"][1]),
            "--size_z",
            str(box["size"][2]),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return None, "vina_command_failed"
    affinity = parse_affinity(completed.stdout + "\n" + completed.stderr)
    if affinity is None:
        return None, "vina_affinity_parse_failed"
    return affinity, None


def main(argv):
    started_at = time.perf_counter()
    args = parse_args(argv)
    candidates = load_candidates(args.input_json)
    total = max(len(candidates), 1)
    counts = Counter()
    candidate_rows = []

    for candidate in candidates:
        receptor = candidate.get("source_pocket_path")
        source_ligand = candidate.get("source_ligand_path")
        ligand_payload = candidate.get("molecular_representation")
        receptor_path = Path(receptor) if receptor else None
        ligand_path = Path(source_ligand) if source_ligand else None
        counts["candidate_count"] += 1
        counts["receptor_path_present"] += int(bool(receptor))
        counts["receptor_path_exists"] += int(bool(receptor_path and receptor_path.is_file()))
        counts["receptor_is_pdbqt"] += int(
            bool(receptor_path and receptor_path.suffix.lower() == ".pdbqt")
        )
        counts["ligand_source_path_present"] += int(bool(source_ligand))
        counts["ligand_source_pdbqt_exists"] += int(
            bool(ligand_path and ligand_path.is_file() and ligand_path.suffix.lower() == ".pdbqt")
        )
        counts["ligand_payload_present"] += int(bool(ligand_payload))
        counts["ligand_payload_looks_like_pdbqt"] += int(looks_like_pdbqt_payload(ligand_payload))
        counts["candidate_with_complete_vina_inputs"] += int(
            bool(
                receptor_path
                and receptor_path.is_file()
                and receptor_path.suffix.lower() == ".pdbqt"
                and (
                    (ligand_path and ligand_path.is_file() and ligand_path.suffix.lower() == ".pdbqt")
                    or bool(ligand_payload)
                )
            )
        )

    if shutil.which(args.vina_executable) is None:
        write_metrics(
            args.output_json,
            structured_payload(
                attach_runtime(attach_contract_counts({
                "schema_version": 1.0,
                "vina_available": 0.0,
                "backend_examples_scored": 0.0,
                "backend_missing_structure_fraction": 1.0,
                "candidate_count": float(counts["candidate_count"]),
                "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
                "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
                "receptor_pdbqt_fraction": counts["receptor_is_pdbqt"] / float(total),
                "ligand_source_pdbqt_fraction": counts["ligand_source_pdbqt_exists"] / float(total),
                "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
                "ligand_payload_pdbqt_like_fraction": counts["ligand_payload_looks_like_pdbqt"] / float(total),
                "candidate_with_complete_vina_inputs_fraction": counts["candidate_with_complete_vina_inputs"] / float(total),
                }, counts["candidate_count"], 0, counts["candidate_count"]), started_at, counts["candidate_count"]),
                [],
            ),
        )
        return

    affinities = []
    failures = 0
    failure_reasons = Counter()
    with tempfile.TemporaryDirectory(prefix="pocket_diffusion_vina_") as workdir:
        for index, candidate in enumerate(candidates):
            affinity, reason = score_candidate(args.vina_executable, candidate, workdir)
            row = {
                "vina_score_success_fraction": 0.0,
                "backend_missing_structure_fraction": 0.0,
            }
            if reason is not None:
                failures += 1
                failure_reasons[reason] += 1
                row["backend_missing_structure_fraction"] = 1.0
                row["vina_failure_reason"] = reason
                candidate_rows.append(
                    {
                        "candidate_id": candidate.get("candidate_id") or f"unknown:{index}",
                        "example_id": candidate.get("example_id") or "unknown",
                        "protein_id": candidate.get("protein_id") or "unknown",
                        "split_label": candidate.get("split_label") or "unknown",
                        "method_id": candidate.get("method_id") or "unknown",
                        "layer": candidate.get("layer") or "unknown",
                        "metrics": row,
                    }
                )
                continue
            affinities.append(affinity)
            row["vina_score_success_fraction"] = 1.0
            row["vina_score"] = affinity
            row["vina_mean_affinity_kcal_mol"] = affinity
            row["vina_best_affinity_kcal_mol"] = affinity
            candidate_rows.append(
                {
                    "candidate_id": candidate.get("candidate_id") or f"unknown:{index}",
                    "example_id": candidate.get("example_id") or "unknown",
                    "protein_id": candidate.get("protein_id") or "unknown",
                    "split_label": candidate.get("split_label") or "unknown",
                    "method_id": candidate.get("method_id") or "unknown",
                    "layer": candidate.get("layer") or "unknown",
                    "metrics": row,
                }
            )

    if not affinities:
        payload = attach_contract_counts({
            "schema_version": 1.0,
            "vina_available": 1.0,
            "backend_examples_scored": 0.0,
            "backend_missing_structure_fraction": failures / float(total),
            "vina_score_success_fraction": 0.0,
            "candidate_count": float(counts["candidate_count"]),
            "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
            "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
            "receptor_pdbqt_fraction": counts["receptor_is_pdbqt"] / float(total),
            "ligand_source_pdbqt_fraction": counts["ligand_source_pdbqt_exists"] / float(total),
            "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
            "ligand_payload_pdbqt_like_fraction": counts["ligand_payload_looks_like_pdbqt"] / float(total),
            "candidate_with_complete_vina_inputs_fraction": counts["candidate_with_complete_vina_inputs"] / float(total),
        }, counts["candidate_count"], 0, failures)
        for reason, count in failure_reasons.items():
            payload[f"vina_{reason}_fraction"] = count / float(total)
        payload["failure_reasons"] = dict(sorted(failure_reasons.items()))
        attach_runtime(payload, started_at, counts["candidate_count"])
        write_metrics(
            args.output_json,
            structured_payload(payload, candidate_rows),
        )
        return

    mean_affinity = sum(affinities) / float(len(affinities))
    payload = {
        "schema_version": 1.0,
        "vina_available": 1.0,
        "backend_examples_scored": float(len(affinities)),
        "backend_missing_structure_fraction": failures / float(total),
        "vina_score_success_fraction": len(affinities) / float(total),
        "vina_mean_affinity_kcal_mol": mean_affinity,
        "vina_best_affinity_kcal_mol": min(affinities),
        "vina_score_mean": mean_affinity,
        "vina_score_median": statistics.median(affinities),
        "vina_score_best": min(affinities),
        "vina_score_std": statistics.pstdev(affinities) if len(affinities) > 1 else 0.0,
        "candidate_count": float(counts["candidate_count"]),
        "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
        "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
        "receptor_pdbqt_fraction": counts["receptor_is_pdbqt"] / float(total),
        "ligand_source_pdbqt_fraction": counts["ligand_source_pdbqt_exists"] / float(total),
        "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
        "ligand_payload_pdbqt_like_fraction": counts["ligand_payload_looks_like_pdbqt"] / float(total),
        "candidate_with_complete_vina_inputs_fraction": counts["candidate_with_complete_vina_inputs"] / float(total),
    }
    attach_contract_counts(payload, counts["candidate_count"], len(affinities), failures)
    for reason, count in failure_reasons.items():
        payload[f"vina_{reason}_fraction"] = count / float(total)
    payload["failure_reasons"] = dict(sorted(failure_reasons.items()))
    attach_runtime(payload, started_at, counts["candidate_count"])
    write_metrics(
        args.output_json,
        structured_payload(payload, candidate_rows),
    )


if __name__ == "__main__":
    main(sys.argv)
