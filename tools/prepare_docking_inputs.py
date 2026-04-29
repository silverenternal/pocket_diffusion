#!/usr/bin/env python3
"""Prepare candidate-preserving PDBQT inputs for docking backends.

This script intentionally records conversion assumptions instead of hiding
them. When OpenBabel/MGLTools-style preparation is unavailable, it emits a
minimal PDBQT representation with zero partial charges and explicit provenance
so downstream Vina/GNINA coverage failures can be attributed correctly.
"""

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


LAYER_SOURCES = [
    ("raw_flow_candidates", "raw_flow"),
    ("constrained_flow_candidates", "constrained_flow"),
    ("raw_geometry_candidates", "raw_geometry"),
    ("raw_rollout_candidates", "raw_rollout"),
    ("bond_logits_refined_candidates", "bond_logits_refined"),
    ("valence_refined_candidates", "valence_refined"),
    ("no_repair_candidates", "no_repair"),
    ("centroid_only_candidates", "centroid_only"),
    ("clash_only_candidates", "clash_only"),
    ("bond_inference_only_candidates", "bond_inference_only"),
    ("full_repair_candidates", "full_repair"),
    ("gated_repair_candidates", "gated_repair"),
    ("repair_rejected_candidates", "repair_rejected"),
    ("repaired_candidates", "repaired"),
    ("inferred_bond_candidates", "inferred_bond"),
    ("deterministic_proxy_candidates", "deterministic_proxy"),
    ("reranked_candidates", "reranked"),
]
ATOM_TYPE_TO_ELEMENT = {
    0: "C",
    1: "N",
    2: "O",
    3: "S",
    4: "H",
    5: "C",
}
PDBQT_ELEMENT_TO_TYPE = {
    "C": "C",
    "N": "N",
    "O": "OA",
    "S": "S",
    "H": "HD",
}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create receptor/ligand PDBQT docking inputs from generation-layer artifacts."
    )
    parser.add_argument("generation_layers", nargs="+", help="generation_layers_*.json files")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/pdbbindpp_real_backends/docking_inputs",
        help="Directory for generated receptor/ligand PDBQT files and candidate input JSON.",
    )
    parser.add_argument(
        "--report-json",
        default="configs/docking_input_preparation_report.json",
        help="Machine-readable conversion report.",
    )
    parser.add_argument(
        "--report-md",
        default="docs/docking_input_preparation.md",
        help="Short human-readable conversion summary.",
    )
    parser.add_argument(
        "--candidate-json",
        default=None,
        help="Optional explicit path for backend-ready candidate JSON.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional comma-separated layer allowlist, for example no_repair,centroid_only.",
    )
    return parser.parse_args(argv[1:])


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def safe_name(value):
    text = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def candidate_id(method_id, layer, candidate, index):
    return "{}:{}:{}:{}".format(
        method_id,
        layer,
        candidate.get("example_id") or "unknown",
        index,
    )


def collect_candidates(path, allowed_layers=None):
    artifact = load_json(path)
    split = artifact.get("split_label") or Path(path).stem.replace("generation_layers_", "")
    method_id = (
        artifact.get("active_method", {}).get("method_id")
        or artifact.get("method_id")
        or "conditioned_denoising"
    )
    rows = []
    for source_key, layer in LAYER_SOURCES:
        if allowed_layers is not None and layer not in allowed_layers:
            continue
        for index, candidate in enumerate(artifact.get(source_key, [])):
            enriched = dict(candidate)
            enriched["candidate_id"] = enriched.get("candidate_id") or candidate_id(
                method_id, layer, enriched, index
            )
            enriched["split_label"] = enriched.get("split_label") or split
            enriched["method_id"] = enriched.get("method_id") or method_id
            enriched["layer"] = enriched.get("layer") or layer
            enriched["source_artifact"] = str(path)
            rows.append(enriched)
    return rows


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


def shifted_coords(candidate):
    origin = finite_triplet(candidate.get("coordinate_frame_origin")) or (0.0, 0.0, 0.0)
    coords = []
    for coord in candidate.get("coords", []):
        parsed = finite_triplet(coord)
        if parsed is None:
            continue
        coords.append(tuple(parsed[dim] + origin[dim] for dim in range(3)))
    return coords


def atom_element(atom_type):
    try:
        index = int(atom_type)
    except (TypeError, ValueError):
        index = 0
    return ATOM_TYPE_TO_ELEMENT.get(index, "C")


def pdbqt_type(element):
    return PDBQT_ELEMENT_TO_TYPE.get((element or "C").upper(), "C")


def ligand_pdbqt_line(index, element, coord):
    atom_name = element[:2].rjust(2)
    return (
        f"HETATM{index:5d} {atom_name:<4s} LIG A   1    "
        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"  1.00  0.00     0.000 {pdbqt_type(element):>2s}"
    )


def receptor_pdbqt_line(source_line, serial):
    record = source_line[:6] if source_line.startswith(("ATOM", "HETATM")) else "ATOM  "
    name = source_line[12:16] if len(source_line) >= 16 else " C  "
    residue = source_line[17:20] if len(source_line) >= 20 else "RES"
    chain = source_line[21:22] if len(source_line) >= 22 else "A"
    residue_id = source_line[22:26] if len(source_line) >= 26 else "   1"
    try:
        x = float(source_line[30:38])
        y = float(source_line[38:46])
        z = float(source_line[46:54])
    except ValueError:
        return None
    element = source_line[76:78].strip() if len(source_line) >= 78 else ""
    if not element:
        element = "".join(ch for ch in name if ch.isalpha())[:1] or "C"
    return (
        f"{record}{serial:5d} {name:<4s} {residue:>3s} {chain:1s}{residue_id:>4s}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00     0.000 {pdbqt_type(element):>2s}"
    )


def prepare_receptor(source_path, output_path):
    source = Path(source_path) if source_path else None
    if not source:
        return None, "missing_receptor_path"
    if not source.is_file():
        return None, "receptor_path_not_found"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".pdbqt":
        shutil.copyfile(source, output_path)
        return {
            "preparation_method": "copy_existing_pdbqt",
            "source_path": str(source),
            "output_path": str(output_path),
            "protonation_assumption": "preserved_from_source",
            "charge_assumption": "preserved_from_source",
        }, None
    atom_lines = []
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                converted = receptor_pdbqt_line(line.rstrip("\n"), len(atom_lines) + 1)
                if converted:
                    atom_lines.append(converted)
    if not atom_lines:
        return None, "receptor_has_no_parseable_atoms"
    output_path.write_text("\n".join(atom_lines) + "\nEND\n", encoding="utf-8")
    return {
        "preparation_method": "minimal_pdb_to_pdbqt",
        "source_path": str(source),
        "output_path": str(output_path),
        "atom_count": len(atom_lines),
        "protonation_assumption": "input PDB protonation preserved; no hydrogens added",
        "charge_assumption": "all partial charges set to 0.000",
        "atom_type_assumption": "PDB element mapped to simple AutoDock-style atom type",
    }, None


def prepare_ligand(candidate, output_path):
    coords = shifted_coords(candidate)
    atom_types = candidate.get("atom_types") or []
    if not coords:
        return None, "missing_or_invalid_candidate_coordinates"
    if atom_types and len(atom_types) != len(coords):
        return None, "atom_type_coordinate_count_mismatch"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ROOT"]
    element_counts = Counter()
    for index, coord in enumerate(coords, start=1):
        element = atom_element(atom_types[index - 1] if atom_types else 0)
        element_counts[element] += 1
        lines.append(ligand_pdbqt_line(index, element, coord))
    lines.extend(["ENDROOT", "TORSDOF 0"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "preparation_method": "generated_coords_minimal_pdbqt",
        "output_path": str(output_path),
        "atom_count": len(coords),
        "element_counts": dict(sorted(element_counts.items())),
        "sanitization_assumption": "candidate coordinates accepted if finite; no chemistry sanitization performed here",
        "hydrogen_assumption": "uses generated atom_types as-is; no hydrogens added",
        "conformer_assumption": "single generated conformer from candidate coords plus coordinate_frame_origin",
        "charge_assumption": "all partial charges set to 0.000",
        "bond_assumption": "no torsion tree beyond ROOT; inferred_bonds retained only in metadata",
        "source_ligand_reference_path": candidate.get("source_ligand_path"),
    }, None


def prepare_candidates(candidates, output_dir):
    receptor_cache = {}
    prepared = []
    records = []
    counts = Counter()
    for candidate in candidates:
        counts["candidate_count"] += 1
        split = safe_name(candidate.get("split_label"))
        method = safe_name(candidate.get("method_id"))
        layer = safe_name(candidate.get("layer"))
        protein = safe_name(candidate.get("protein_id") or candidate.get("example_id"))
        cid = candidate["candidate_id"]
        receptor_key = candidate.get("source_pocket_path") or f"missing:{protein}"
        receptor_path = output_dir / "receptors" / f"{protein}.pdbqt"
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = prepare_receptor(candidate.get("source_pocket_path"), receptor_path)
        receptor_meta, receptor_reason = receptor_cache[receptor_key]
        ligand_path = output_dir / "ligands" / split / method / layer / f"{safe_name(cid)}.pdbqt"
        ligand_meta, ligand_reason = prepare_ligand(candidate, ligand_path)
        failure_reasons = []
        if receptor_reason:
            failure_reasons.append(receptor_reason)
        if ligand_reason:
            failure_reasons.append(ligand_reason)
        status = "prepared" if not failure_reasons else "failed"
        counts[f"status_{status}"] += 1
        for reason in failure_reasons:
            counts[f"failure_{reason}"] += 1
        record = {
            "candidate_id": cid,
            "example_id": candidate.get("example_id"),
            "protein_id": candidate.get("protein_id"),
            "split_label": candidate.get("split_label"),
            "method_id": candidate.get("method_id"),
            "layer": candidate.get("layer"),
            "status": status,
            "failure_reasons": failure_reasons,
            "receptor": receptor_meta,
            "ligand": ligand_meta,
            "source_artifact": candidate.get("source_artifact"),
        }
        records.append(record)
        if status == "prepared":
            enriched = dict(candidate)
            enriched["source_pocket_path"] = receptor_meta["output_path"]
            enriched["source_ligand_path"] = ligand_meta["output_path"]
            enriched["docking_input_preparation"] = {
                "receptor": receptor_meta,
                "ligand": ligand_meta,
            }
            prepared.append(enriched)
    return prepared, records, counts


def report_payload(records, counts, source_artifacts, candidate_json):
    total = max(counts["candidate_count"], 1)
    failures = {
        key.replace("failure_", ""): value
        for key, value in sorted(counts.items())
        if key.startswith("failure_")
    }
    return {
        "schema_version": 1,
        "artifact_name": "docking_input_preparation_report",
        "source_artifacts": source_artifacts,
        "backend_candidate_json": str(candidate_json),
        "conversion_policy": {
            "receptor_policy": "copy existing PDBQT when present; otherwise convert parseable PDB ATOM/HETATM records to minimal PDBQT",
            "ligand_policy": "write generated candidate coordinates to minimal PDBQT while preserving candidate_id",
            "protonation_assumption": "preserve source receptor hydrogens; do not add ligand hydrogens",
            "charge_assumption": "0.000 partial charge for minimal conversions",
            "sanitization_assumption": "finite-coordinate validation only; RDKit sanitization remains a separate metric backend",
        },
        "counts": {
            "candidate_count": counts["candidate_count"],
            "prepared_count": counts["status_prepared"],
            "failure_count": counts["status_failed"],
            "prepared_fraction": counts["status_prepared"] / float(total),
        },
        "failure_reasons": failures,
        "records": records,
    }


def write_markdown(path, report):
    counts = report["counts"]
    lines = [
        "# Docking Input Preparation",
        "",
        f"- candidate_count: {counts['candidate_count']}",
        f"- prepared_count: {counts['prepared_count']}",
        f"- failure_count: {counts['failure_count']}",
        f"- prepared_fraction: {counts['prepared_fraction']:.4f}",
        f"- backend_candidate_json: `{report['backend_candidate_json']}`",
        "",
        "## Assumptions",
        f"- receptor: {report['conversion_policy']['receptor_policy']}",
        f"- ligand: {report['conversion_policy']['ligand_policy']}",
        f"- protonation: {report['conversion_policy']['protonation_assumption']}",
        f"- charges: {report['conversion_policy']['charge_assumption']}",
        f"- sanitization: {report['conversion_policy']['sanitization_assumption']}",
        "",
        "## Failure Reasons",
    ]
    if not report["failure_reasons"]:
        lines.append("- none")
    for reason, count in report["failure_reasons"].items():
        lines.append(f"- {reason}: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    allowed_layers = None
    if args.layers:
        allowed_layers = {item.strip() for item in args.layers.split(",") if item.strip()}
    candidates = []
    for generation_path in args.generation_layers:
        candidates.extend(collect_candidates(generation_path, allowed_layers))
    candidate_json = Path(args.candidate_json) if args.candidate_json else output_dir / "docking_candidates.json"
    prepared, records, counts = prepare_candidates(candidates, output_dir)
    write_json(candidate_json, prepared)
    source_artifacts = [str(path) for path in args.generation_layers]
    report = report_payload(records, counts, source_artifacts, candidate_json)
    write_json(Path(args.report_json), report)
    write_markdown(Path(args.report_md), report)
    return 0 if counts["status_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
