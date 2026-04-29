#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
import tempfile
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
PDB_ATOM_CACHE = {}
PDB_GEOMETRY_CACHE = {}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build candidate_metrics.jsonl from real generation-layer artifacts."
    )
    parser.add_argument("generation_layers", nargs="+")
    parser.add_argument("--output", default="candidate_metrics.jsonl")
    parser.add_argument("--method-id", default=None)
    parser.add_argument("--tool-dir", default="tools")
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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def candidate_id(method_id, layer, candidate):
    return "{}:{}:{}:{}".format(
        method_id,
        layer,
        candidate.get("example_id") or "unknown",
        candidate.get("_layer_index", 0),
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
            enriched["_layer_index"] = index
            enriched["candidate_id"] = enriched.get("candidate_id") or candidate_id(method_id, layer, enriched)
            enriched["_split_label"] = split
            enriched["_layer"] = layer
            enriched["_candidate_source_key"] = source_key
            enriched["_source_artifact"] = str(path)
            rows.append(enriched)
    return rows, split, method_id


def run_backend(command, input_path, output_path):
    completed = subprocess.run(
        command + [str(input_path), str(output_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "backend failed: {}\nstdout:\n{}\nstderr:\n{}".format(
                " ".join(command), completed.stdout, completed.stderr
            )
        )
    return load_json(output_path)


def profile_metrics_by_id(generation_path):
    split = Path(generation_path).stem.replace("generation_layers_", "")
    profile_path = Path(generation_path).with_name(f"preference_profiles_{split}.json")
    if not profile_path.is_file():
        return {}
    profiles = load_json(profile_path)
    result = {}
    for record in profiles.get("records", []):
        metrics = {}
        for key, value in record.items():
            if isinstance(value, dict) and "value" in value:
                raw = value.get("value")
                if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                    metrics[key] = float(raw)
        result[record.get("candidate_id", "")] = metrics
    return result


def parse_pdb_atoms(path):
    if path in PDB_ATOM_CACHE:
        return PDB_ATOM_CACHE[path]
    atoms = []
    if not path:
        return atoms
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError:
        return atoms
    with handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                coords = (
                    float(line[30:38].strip()),
                    float(line[38:46].strip()),
                    float(line[46:54].strip()),
                )
            except ValueError:
                continue
            element = line[76:78].strip().upper() or line[12:14].strip().upper()
            atoms.append((coords, element))
    PDB_ATOM_CACHE[path] = atoms
    return atoms


def distance(left, right):
    return math.sqrt(sum((left[dim] - right[dim]) ** 2 for dim in range(3)))


def pocket_geometry(path, cell_size=4.5):
    if path in PDB_GEOMETRY_CACHE:
        return PDB_GEOMETRY_CACHE[path]
    atoms = parse_pdb_atoms(path)
    coords = [coord for coord, _element in atoms]
    center = tuple(sum(coord[dim] for coord in coords) / float(len(coords)) for dim in range(3)) if coords else (0.0, 0.0, 0.0)
    grid = {}
    for coord, element in atoms:
        key = tuple(int(math.floor(coord[dim] / cell_size)) for dim in range(3))
        grid.setdefault(key, []).append((coord, element))
    geometry = {"atoms": atoms, "center": center, "grid": grid, "cell_size": cell_size}
    PDB_GEOMETRY_CACHE[path] = geometry
    return geometry


def local_pocket_atoms(coord, geometry):
    cell_size = geometry["cell_size"]
    key = tuple(int(math.floor(coord[dim] / cell_size)) for dim in range(3))
    atoms = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                atoms.extend(geometry["grid"].get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
    return atoms


def min_pocket_distance(coord, geometry):
    atoms = local_pocket_atoms(coord, geometry)
    if atoms:
        return min(distance(coord, pocket_coord) for pocket_coord, _element in atoms)
    return distance(coord, geometry["center"])


def shifted_candidate_coords(candidate):
    origin = candidate.get("coordinate_frame_origin")
    if isinstance(origin, list) and len(origin) == 3:
        try:
            origin = tuple(float(value) for value in origin)
        except (TypeError, ValueError):
            origin = (0.0, 0.0, 0.0)
    else:
        origin = (0.0, 0.0, 0.0)
    coords = []
    for coord in candidate.get("coords", []):
        if not isinstance(coord, list) or len(coord) != 3:
            continue
        try:
            coords.append(tuple(float(value) + origin[dim] for dim, value in enumerate(coord)))
        except (TypeError, ValueError):
            continue
    return coords


def interaction_proxy_metrics(candidate):
    coords = shifted_candidate_coords(candidate)
    geometry = pocket_geometry(candidate.get("source_pocket_path"))
    if not coords or not geometry["atoms"]:
        return {
            "interaction_profile_coverage_fraction": 0.0,
        }
    atom_types = candidate.get("atom_types", [])
    hb_contacts = 0
    hydrophobic_contacts = 0
    close_contacts = 0
    clash_contacts = 0
    for index, coord in enumerate(coords):
        atom_type = int(atom_types[index]) if index < len(atom_types) else 0
        min_distance = min_pocket_distance(coord, geometry)
        if min_distance <= 4.5:
            close_contacts += 1
        if min_distance < 1.2:
            clash_contacts += 1
        if atom_type in (1, 2) and min_distance <= 3.5:
            hb_contacts += 1
        if atom_type == 0 and min_distance <= 4.5:
            hydrophobic_contacts += 1
    atom_count = float(max(len(coords), 1))
    hbond = hb_contacts / atom_count
    hydrophobic = hydrophobic_contacts / atom_count
    return {
        "hydrogen_bond_proxy": hbond,
        "hydrophobic_contact_proxy": hydrophobic,
        "residue_contact_count": float(close_contacts),
        "clash_burden": clash_contacts / atom_count,
        "contact_balance": 1.0 - abs(hbond - hydrophobic),
        "interaction_profile_coverage_fraction": 1.0,
    }


def candidate_rows(candidates, backend_payloads, profile_metrics, method_id):
    backend_by_id = []
    for backend_name, payload in backend_payloads.items():
        rows = {}
        for row in payload.get("candidate_metrics", []):
            rows[row.get("candidate_id", "")] = row.get("metrics", {})
        backend_by_id.append((backend_name, rows))

    rows = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        metrics = {}
        metrics.update(profile_metrics.get(cid, {}))
        statuses = {}
        for backend_name, backend_rows in backend_by_id:
            backend_metrics = backend_rows.get(cid)
            statuses[backend_name] = "metrics_available" if backend_metrics else "missing"
            if backend_metrics:
                metrics.update(backend_metrics)
        if "contact_fraction" in metrics and "pocket_contact_fraction" not in metrics:
            metrics["pocket_contact_fraction"] = metrics["contact_fraction"]
        if "centroid_offset" in metrics and "mean_centroid_offset" not in metrics:
            metrics["mean_centroid_offset"] = metrics["centroid_offset"]
        metrics["bond_count"] = float(candidate.get("bond_count", len(candidate.get("inferred_bonds", []))))
        metrics["valence_violation_count"] = float(candidate.get("valence_violation_count", 0))
        metrics.update(interaction_proxy_metrics(candidate))
        rows.append(
            {
                "candidate_id": cid,
                "example_id": candidate.get("example_id") or "unknown",
                "protein_id": candidate.get("protein_id") or "unknown",
                "split_label": candidate.get("_split_label") or "unknown",
                "layer": candidate.get("_layer") or "unknown",
                "method_id": method_id,
                "candidate_source": candidate.get("source") or candidate.get("_candidate_source_key"),
                "metrics": metrics,
                "backend_statuses": statuses,
                "source_artifacts": [candidate.get("_source_artifact")],
            }
        )
    return rows


def main(argv):
    args = parse_args(argv)
    tool_dir = Path(args.tool_dir)
    allowed_layers = None
    if args.layers:
        allowed_layers = {item.strip() for item in args.layers.split(",") if item.strip()}
    all_rows = []
    for generation_path in args.generation_layers:
        candidates, _split, detected_method_id = collect_candidates(generation_path, allowed_layers)
        method_id = args.method_id or detected_method_id
        for candidate in candidates:
            candidate["candidate_id"] = candidate.get("candidate_id") or candidate_id(
                method_id, candidate["_layer"], candidate
            )
        profiles = profile_metrics_by_id(generation_path)
        with tempfile.TemporaryDirectory(prefix="candidate_metrics_") as tmp:
            tmp = Path(tmp)
            input_path = tmp / "candidates.json"
            write_json(input_path, candidates)
            backend_payloads = {
                "pocket_compatibility": run_backend(
                    [sys.executable, str(tool_dir / "pocket_contact_backend.py"), "--mode", "compatibility"],
                    input_path,
                    tmp / "pocket.json",
                ),
                "docking_affinity": run_backend(
                    [sys.executable, str(tool_dir / "pocket_contact_backend.py"), "--mode", "docking"],
                    input_path,
                    tmp / "docking.json",
                ),
                "chemistry_validity": run_backend(
                    [sys.executable, str(tool_dir / "rdkit_validity_backend.py")],
                    input_path,
                    tmp / "rdkit.json",
                ),
            }
        all_rows.extend(candidate_rows(candidates, backend_payloads, profiles, method_id))

    with open(args.output, "w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
