#!/usr/bin/env python3
"""Create auditable repaired and reranked layers for public baseline artifacts."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_layers", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_triplet(value):
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        coords = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return coords if all(math.isfinite(item) for item in coords) else None


def parse_pocket_atoms(path):
    atoms = []
    if not path:
        return atoms
    try:
        handle = Path(path).open("r", encoding="utf-8", errors="replace")
    except OSError:
        return atoms
    with handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atoms.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    return atoms


def centroid(coords):
    if not coords:
        return [0.0, 0.0, 0.0]
    return [sum(coord[axis] for coord in coords) / float(len(coords)) for axis in range(3)]


def radius(coords, center):
    if not coords:
        return 12.0
    return max(distance(coord, center) for coord in coords)


def distance(left, right):
    return math.sqrt(sum((float(left[axis]) - float(right[axis])) ** 2 for axis in range(3)))


def candidate_coords(candidate):
    coords = []
    for coord in candidate.get("coords", []):
        parsed = finite_triplet(coord)
        if parsed is not None:
            coords.append(parsed)
    return coords


def candidate_pocket_geometry(candidate):
    pocket_atoms = parse_pocket_atoms(candidate.get("source_pocket_path"))
    center = finite_triplet(candidate.get("pocket_centroid")) or centroid(pocket_atoms)
    raw_radius = candidate.get("pocket_radius")
    try:
        pocket_radius = float(raw_radius)
    except (TypeError, ValueError):
        pocket_radius = radius(pocket_atoms, center)
    if not math.isfinite(pocket_radius) or pocket_radius <= 0.0:
        pocket_radius = radius(pocket_atoms, center)
    # Public baseline receptor files may contain a large local protein crop. The
    # backend adapter historically used 12 A when baseline artifacts did not
    # provide a radius, so cap inferred radii to keep repaired/reranked layers
    # under the same matched docking-box budget as raw public baselines.
    pocket_radius = min(pocket_radius, 12.0)
    return pocket_atoms, center, pocket_radius


def repel_close_contacts(coords):
    if len(coords) < 2:
        return coords
    coords = [list(coord) for coord in coords]
    min_dist = 1.15
    for _ in range(4):
        updates = [[0.0, 0.0, 0.0] for _ in coords]
        moved = False
        for left in range(len(coords)):
            for right in range(left + 1, len(coords)):
                delta = [coords[right][axis] - coords[left][axis] for axis in range(3)]
                dist_sq = sum(value * value for value in delta)
                if dist_sq >= min_dist * min_dist:
                    continue
                dist = max(math.sqrt(dist_sq), 1e-6)
                push = (min_dist - dist) * 0.5
                for axis in range(3):
                    direction = delta[axis] / dist
                    updates[left][axis] -= direction * push
                    updates[right][axis] += direction * push
                moved = True
        if not moved:
            break
        for index in range(len(coords)):
            for axis in range(3):
                coords[index][axis] += updates[index][axis]
    return coords


def push_away_from_pocket_atoms(coords, pocket_atoms, min_distance=1.28):
    if not coords or not pocket_atoms:
        return coords
    coords = [list(coord) for coord in coords]
    for coord in coords:
        for _ in range(3):
            closest = min(pocket_atoms, key=lambda pocket: distance(coord, pocket))
            dist = distance(coord, closest)
            if dist >= min_distance:
                break
            safe = max(dist, 1e-6)
            push = min_distance - safe + 0.05
            for axis in range(3):
                coord[axis] += (coord[axis] - closest[axis]) / safe * push
    return coords


def clamp_to_pocket_envelope(coords, center, max_radius):
    out = []
    for coord in coords:
        delta = [coord[axis] - center[axis] for axis in range(3)]
        dist = math.sqrt(sum(value * value for value in delta))
        if dist <= max_radius or dist <= 1e-6:
            out.append(coord)
            continue
        scale = max_radius / dist
        out.append([center[axis] + delta[axis] * scale for axis in range(3)])
    return out


def repair_candidate_geometry(coords, pocket_atoms, pocket_center, pocket_radius, candidate_ix, num_candidates):
    if not coords:
        return []
    mol_center = centroid(coords)
    to_pocket = [pocket_center[axis] - mol_center[axis] for axis in range(3)]
    uniqueness_phase = (candidate_ix + 1.0) / (max(num_candidates, 1) + 1.0)
    radial_scale = 1.0 + 0.08 * candidate_ix
    anchor = 0.38 + 0.1 * uniqueness_phase
    swirl = 0.06 + 0.04 * uniqueness_phase
    repaired = []
    for atom_ix, coord in enumerate(coords):
        offset = [(coord[axis] - mol_center[axis]) * radial_scale for axis in range(3)]
        parity = 1.0 if (atom_ix + candidate_ix) % 2 == 0 else -1.0
        repaired.append([
            pocket_center[0] + anchor * to_pocket[0] + offset[0] + parity * swirl,
            pocket_center[1] + anchor * to_pocket[1] + offset[1] - parity * swirl,
            pocket_center[2] + anchor * to_pocket[2] + offset[2] + 0.5 * parity * swirl,
        ])
    repaired = repel_close_contacts(repaired)
    repaired = push_away_from_pocket_atoms(repaired, pocket_atoms)
    return clamp_to_pocket_envelope(repaired, pocket_center, pocket_radius + 1.6)


def infer_bonds(coords):
    bonds = []
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            dist = distance(coords[left], coords[right])
            if 0.85 <= dist <= 1.75:
                bonds.append([left, right])
    return bonds


def max_valence(atom_type):
    return {0: 4, 1: 3, 2: 2, 3: 6, 4: 1}.get(int(atom_type), 4)


def prune_bonds_for_valence(coords, atom_types, bonds):
    ordered = sorted(bonds, key=lambda bond: distance(coords[bond[0]], coords[bond[1]]))
    degrees = [0 for _ in atom_types]
    kept = []
    for left, right in ordered:
        if left >= len(degrees) or right >= len(degrees):
            continue
        if degrees[left] >= max_valence(atom_types[left]) or degrees[right] >= max_valence(atom_types[right]):
            continue
        degrees[left] += 1
        degrees[right] += 1
        kept.append([left, right])
    return kept


def candidate_is_valid(candidate):
    coords = candidate_coords(candidate)
    return bool(candidate.get("atom_types")) and len(candidate.get("atom_types", [])) == len(coords)


def candidate_centroid_offset(candidate):
    coords = candidate_coords(candidate)
    if not coords:
        return float("inf")
    center = finite_triplet(candidate.get("pocket_centroid")) or [0.0, 0.0, 0.0]
    return distance(centroid(coords), center)


def candidate_has_pocket_contact(candidate):
    center = finite_triplet(candidate.get("pocket_centroid")) or [0.0, 0.0, 0.0]
    radius_value = candidate.get("pocket_radius")
    try:
        pocket_radius = float(radius_value)
    except (TypeError, ValueError):
        pocket_radius = 12.0
    return any(distance(coord, center) <= pocket_radius + 2.0 for coord in candidate_coords(candidate))


def candidate_clash_fraction(candidate):
    coords = candidate_coords(candidate)
    if len(coords) < 2:
        return 0.0
    clashes = 0
    pairs = 0
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            pairs += 1
            clashes += int(distance(coords[left], coords[right]) < 0.85)
    return clashes / float(max(pairs, 1))


def valence_sane(candidate):
    atom_types = candidate.get("atom_types", [])
    degrees = [0 for _ in atom_types]
    for left, right in candidate.get("inferred_bonds", []):
        if left < len(degrees) and right < len(degrees):
            degrees[left] += 1
            degrees[right] += 1
    return bool(atom_types) and all(degree <= max_valence(atom_type) for degree, atom_type in zip(degrees, atom_types))


def proxy_rerank_score(candidate):
    valid = 1.0 if candidate_is_valid(candidate) else 0.0
    contact = 1.0 if candidate_has_pocket_contact(candidate) else 0.0
    centroid_fit = 1.0 / (1.0 + max(candidate_centroid_offset(candidate), 0.0))
    clash_free = 1.0 - max(0.0, min(1.0, candidate_clash_fraction(candidate)))
    valence = 1.0 if valence_sane(candidate) else 0.0
    return 0.25 * valid + 0.25 * contact + 0.2 * centroid_fit + 0.2 * clash_free + 0.1 * valence


def layer_candidate(raw, layer, coords, inferred_bonds, source):
    candidate = dict(raw)
    method = raw.get("method_id") or raw.get("candidate_id", "").split(":")[0] or "unknown"
    example = raw.get("example_id") or "unknown"
    candidate["candidate_id"] = f"{method}:{layer}:{example}:0"
    candidate["layer"] = layer
    candidate["coords"] = [[round(float(value), 6) for value in coord] for coord in coords]
    candidate["inferred_bonds"] = inferred_bonds
    candidate["source"] = source
    candidate["molecular_representation"] = (
        f"source={source};atoms={len(candidate.get('atom_types', []))};bonds={len(inferred_bonds)}"
    )
    candidate["adapter_metadata"] = {
        "postprocess_layer": layer,
        "raw_candidate_id": raw.get("candidate_id"),
        "algorithm": "public_baseline_postprocess_layers",
    }
    return candidate


def process_artifact(path, output_dir):
    artifact = load_json(path)
    method_id = artifact.get("method_id") or artifact.get("active_method", {}).get("method_id") or "unknown"
    split = artifact.get("split_label") or "q1_public_full100_budget1"
    raw_candidates = artifact.get("raw_rollout_candidates", [])
    repaired = []
    rerank_pool = []
    for index, raw in enumerate(raw_candidates):
        raw = dict(raw)
        raw["method_id"] = raw.get("method_id") or method_id
        raw["layer"] = raw.get("layer") or "raw_rollout"
        pocket_atoms, pocket_center, pocket_radius = candidate_pocket_geometry(raw)
        raw["pocket_centroid"] = raw.get("pocket_centroid") or pocket_center
        raw["pocket_radius"] = raw.get("pocket_radius") or pocket_radius
        coords = candidate_coords(raw)
        fixed = repair_candidate_geometry(coords, pocket_atoms, pocket_center, pocket_radius, index, len(raw_candidates))
        bonds = prune_bonds_for_valence(fixed, raw.get("atom_types", []), infer_bonds(fixed))
        repaired_candidate = layer_candidate(
            raw,
            "repaired",
            fixed,
            [],
            f"shared_public_geometry_repair:raw_candidate={raw.get('candidate_id')}",
        )
        repaired.append(repaired_candidate)
        rerank_candidate = layer_candidate(
            raw,
            "reranked",
            fixed,
            bonds,
            f"shared_public_deterministic_proxy_rerank:raw_candidate={raw.get('candidate_id')}",
        )
        rerank_candidate["postprocess_score"] = proxy_rerank_score(rerank_candidate)
        rerank_pool.append(rerank_candidate)
        raw.update({"pocket_centroid": pocket_center, "pocket_radius": pocket_radius})
        raw_candidates[index] = raw
    reranked = sorted(rerank_pool, key=lambda item: (-item.get("postprocess_score", 0.0), item.get("candidate_id", "")))
    output = dict(artifact)
    output["schema_version"] = max(int(output.get("schema_version", 1)), 2)
    output["method_id"] = method_id
    output["split_label"] = split
    output["postprocessing"] = {
        "tool": "public_baseline_postprocess_layers.py",
        "policy": "shared deterministic geometry repair followed by deterministic proxy rerank",
        "input_artifact": str(path),
        "raw_candidate_count": len(raw_candidates),
        "repaired_candidate_count": len(repaired),
        "reranked_candidate_count": len(reranked),
        "claim_boundary": "raw_rollout remains native public baseline output; repaired and reranked are explicit shared postprocessing layers.",
    }
    output["raw_rollout_candidates"] = raw_candidates
    output["repaired_candidates"] = repaired
    output["reranked_candidates"] = reranked
    out_path = Path(output_dir) / method_id / "generation_layers_test.json"
    write_json(out_path, output)
    return {
        "method_id": method_id,
        "input_artifact": str(path),
        "output_artifact": str(out_path),
        "raw_candidate_count": len(raw_candidates),
        "repaired_candidate_count": len(repaired),
        "reranked_candidate_count": len(reranked),
    }


def main():
    args = parse_args()
    records = [process_artifact(path, args.output_dir) for path in args.generation_layers]
    write_json(
        args.report_json,
        {
            "schema_version": 1,
            "artifact_name": "q1_public_baseline_postprocess_report",
            "records": records,
            "totals": {
                "raw_candidate_count": sum(record["raw_candidate_count"] for record in records),
                "repaired_candidate_count": sum(record["repaired_candidate_count"] for record in records),
                "reranked_candidate_count": sum(record["reranked_candidate_count"] for record in records),
            },
            "claim_boundary": "Generated repaired/reranked rows are postprocessing evidence, not native raw baseline evidence.",
        },
    )


if __name__ == "__main__":
    main()
