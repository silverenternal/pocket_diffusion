#!/usr/bin/env python3
import json
import math
import sys


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def parse_mode(argv):
    mode = "compatibility"
    index = 1
    while index < len(argv) - 2:
        if argv[index] == "--mode" and index + 1 < len(argv) - 2:
            mode = argv[index + 1]
            index += 2
        else:
            index += 1
    return mode


def parse_pdb_coords(path):
    coords = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            coords.append((x, y, z))
    return coords


def centroid(points):
    if not points:
        return (0.0, 0.0, 0.0)
    count = float(len(points))
    return tuple(sum(point[dim] for point in points) / count for dim in range(3))


def distance(a, b):
    return math.sqrt(sum((a[dim] - b[dim]) ** 2 for dim in range(3)))


def parse_candidate_coords(candidate):
    coords = []
    origin = candidate.get("coordinate_frame_origin")
    if isinstance(origin, (list, tuple)) and len(origin) == 3:
        try:
            origin = tuple(float(value) for value in origin)
        except (TypeError, ValueError):
            origin = (0.0, 0.0, 0.0)
    else:
        origin = (0.0, 0.0, 0.0)
    for coord in candidate.get("coords", []):
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            return None
        try:
            parsed = tuple(float(value) + origin[dim] for dim, value in enumerate(coord))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in parsed):
            return None
        coords.append(parsed)
    return coords


def candidate_metrics(candidate, mode):
    pocket_path = candidate.get("source_pocket_path")
    candidate_coords = parse_candidate_coords(candidate)
    if not pocket_path or not candidate_coords:
        return None

    try:
        pocket_coords = parse_pdb_coords(pocket_path)
    except OSError:
        return None
    if not pocket_coords:
        return None

    pocket_centroid = centroid(pocket_coords)
    pocket_radius = float(candidate.get("pocket_radius", 0.0) or 0.0)

    per_atom_min = [min(distance(atom, pocket_atom) for pocket_atom in pocket_coords) for atom in candidate_coords]
    atom_count = max(len(per_atom_min), 1)
    contact_fraction = sum(1 for value in per_atom_min if value <= 4.5) / atom_count
    clash_fraction = sum(1 for value in per_atom_min if value < 1.2) / atom_count
    centroid_offset = distance(centroid(candidate_coords), pocket_centroid)
    centroid_inside = 1.0 if centroid_offset <= max(pocket_radius, 1.0) else 0.0
    pocket_coverage = sum(1 for value in per_atom_min if value <= 3.5) / atom_count

    if mode == "docking":
        docking_like_score = contact_fraction - 0.5 * clash_fraction - 0.05 * centroid_offset
        return {
            "contact_fraction": contact_fraction,
            "clash_fraction": clash_fraction,
            "mean_min_contact_distance": sum(per_atom_min) / atom_count,
            "docking_like_score": docking_like_score,
        }

    return {
        "centroid_inside_fraction": centroid_inside,
        "atom_coverage_fraction": pocket_coverage,
        "clash_fraction": clash_fraction,
    }


def aggregate(rows):
    if not rows:
        return {
            "schema_version": 1.0,
            "backend_examples_scored": 0.0,
            "backend_missing_structure_fraction": 1.0,
        }

    keys = sorted({key for row in rows for key in row.keys()})
    metrics = {
        "schema_version": 1.0,
        "backend_examples_scored": float(len(rows)),
        "backend_missing_structure_fraction": 0.0,
    }
    for key in keys:
        metrics[key] = sum(row[key] for row in rows) / float(len(rows))
    return metrics


def main(argv):
    if len(argv) < 3:
        raise SystemExit("usage: pocket_contact_backend.py [--mode docking|compatibility] <input.json> <output.json>")

    mode = parse_mode(argv)
    input_path = argv[-2]
    output_path = argv[-1]
    candidates = load_candidates(input_path)

    rows = []
    missing = 0
    for candidate in candidates:
        metrics = candidate_metrics(candidate, mode)
        if metrics is None:
            missing += 1
            continue
        rows.append(metrics)

    aggregated = aggregate(rows)
    total = max(len(candidates), 1)
    aggregated["backend_missing_structure_fraction"] = missing / float(total)
    write_metrics(output_path, aggregated)


if __name__ == "__main__":
    main(sys.argv)
