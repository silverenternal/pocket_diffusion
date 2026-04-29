#!/usr/bin/env python3
"""Create auditable postprocessing layers for public baseline artifacts."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_layers", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--ablation-layers",
        action="store_true",
        help="Emit Q2 postprocessing ablation layers in addition to legacy repaired/reranked layers.",
    )
    parser.add_argument(
        "--repair-gate",
        action="store_true",
        help="Emit Q3 gated_repair and repair_rejected layers with conservative non-degradation checks.",
    )
    parser.add_argument(
        "--no-move-if-raw-dockable",
        action="store_true",
        help="Reject coordinate-moving repair when the raw candidate is already backend-input dockable.",
    )
    parser.add_argument("--max-centroid-shift", type=float, default=2.0)
    parser.add_argument("--max-clash-increase", type=float, default=0.01)
    parser.add_argument("--max-box-center-shift", type=float, default=2.0)
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


def centroid_only_geometry(coords, pocket_center):
    if not coords:
        return []
    mol_center = centroid(coords)
    shift = [pocket_center[axis] - mol_center[axis] for axis in range(3)]
    return [[coord[axis] + shift[axis] for axis in range(3)] for coord in coords]


def clash_only_geometry(coords, pocket_atoms):
    if not coords:
        return []
    return push_away_from_pocket_atoms(repel_close_contacts(coords), pocket_atoms)


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


def docking_box(candidate):
    coords = candidate_coords(candidate)
    origin = finite_triplet(candidate.get("coordinate_frame_origin")) or [0.0, 0.0, 0.0]
    raw_pocket_center = finite_triplet(candidate.get("pocket_centroid"))
    if raw_pocket_center is None:
        if not coords:
            return None
        center = centroid(coords)
    else:
        pocket_center = [raw_pocket_center[axis] + origin[axis] for axis in range(3)]
        if coords:
            axis_values = [[coord[axis] for coord in coords] + [pocket_center[axis]] for axis in range(3)]
            mins = [min(values) for values in axis_values]
            maxs = [max(values) for values in axis_values]
            center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]
        else:
            center = pocket_center
    try:
        pocket_radius = float(candidate.get("pocket_radius", 12.0))
    except (TypeError, ValueError):
        pocket_radius = 12.0
    if not math.isfinite(pocket_radius) or pocket_radius <= 0.0:
        pocket_radius = 12.0
    if coords:
        axis_values = [[coord[axis] for coord in coords] + [center[axis]] for axis in range(3)]
        ligand_span = max(max(values) - min(values) for values in axis_values)
    else:
        ligand_span = 0.0
    size = max(8.0, min(80.0, max(2.0 * pocket_radius, ligand_span + 8.0)))
    return {"center": center, "size": [size, size, size]}


def candidate_with_geometry(raw, layer, coords, inferred_bonds):
    candidate = dict(raw)
    candidate["layer"] = layer
    candidate["coords"] = [[round(float(value), 6) for value in coord] for coord in coords]
    candidate["inferred_bonds"] = inferred_bonds
    return candidate


def repair_gate_decision(raw, repaired, options):
    raw_coords = candidate_coords(raw)
    repaired_coords = candidate_coords(repaired)
    reasons = []
    raw_dockable = candidate_is_valid(raw)
    raw_centroid = centroid(raw_coords) if raw_coords else None
    repaired_centroid = centroid(repaired_coords) if repaired_coords else None
    centroid_shift = (
        distance(raw_centroid, repaired_centroid)
        if raw_centroid is not None and repaired_centroid is not None
        else float("inf")
    )
    raw_clash = candidate_clash_fraction(raw)
    repaired_clash = candidate_clash_fraction(repaired)
    clash_increase = repaired_clash - raw_clash
    raw_box = docking_box(raw)
    repaired_box = docking_box(repaired)
    box_center_shift = (
        distance(raw_box["center"], repaired_box["center"])
        if raw_box is not None and repaired_box is not None
        else float("inf")
    )

    if options.get("no_move_if_raw_dockable") and raw_dockable and centroid_shift > 1e-6:
        reasons.append("raw_dockable_coordinate_move_blocked")
    if centroid_shift > options["max_centroid_shift"]:
        reasons.append("max_centroid_shift_exceeded")
    if clash_increase > options["max_clash_increase"]:
        reasons.append("max_clash_increase_exceeded")
    if options.get("preserve_raw_box_center") and box_center_shift > options["max_box_center_shift"]:
        reasons.append("preserve_raw_box_center_exceeded")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "raw_dockable": raw_dockable,
        "centroid_shift": centroid_shift if math.isfinite(centroid_shift) else None,
        "raw_clash_fraction": raw_clash,
        "repaired_clash_fraction": repaired_clash,
        "clash_increase": clash_increase,
        "raw_box_center": raw_box["center"] if raw_box else None,
        "repaired_box_center": repaired_box["center"] if repaired_box else None,
        "box_center_shift": box_center_shift if math.isfinite(box_center_shift) else None,
        "bounds": dict(options),
    }


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


def candidate_source_layer(candidate):
    layer = candidate.get("layer")
    if layer:
        return str(layer)
    candidate_id = str(candidate.get("candidate_id", ""))
    parts = candidate_id.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return "raw_rollout"


def layer_candidate(raw, layer, coords, inferred_bonds, source, postprocessing_steps=None):
    candidate = dict(raw)
    method = raw.get("method_id") or raw.get("candidate_id", "").split(":")[0] or "unknown"
    example = raw.get("example_id") or "unknown"
    source_layer = candidate_source_layer(raw)
    raw_candidate_id = raw.get("candidate_id")
    candidate["candidate_id"] = f"{method}:{layer}:{example}:0"
    candidate["layer"] = layer
    candidate["source_layer"] = source_layer
    candidate["coords"] = [[round(float(value), 6) for value in coord] for coord in coords]
    candidate["inferred_bonds"] = inferred_bonds
    candidate["source"] = source
    candidate["postprocessing_steps"] = list(postprocessing_steps or [])
    candidate["model_native"] = False
    candidate["claim_allowed"] = False
    candidate["molecular_representation"] = (
        f"source={source};atoms={len(candidate.get('atom_types', []))};bonds={len(inferred_bonds)}"
    )
    candidate["adapter_metadata"] = {
        "postprocess_layer": layer,
        "raw_candidate_id": raw_candidate_id,
        "algorithm": "public_baseline_postprocess_layers",
        "postprocessing_steps": list(postprocessing_steps or []),
    }
    candidate["transformation_provenance"] = {
        "algorithm": "public_baseline_postprocess_layers",
        "source_layer": source_layer,
        "source_candidate_id": raw_candidate_id,
        "transformations": list(postprocessing_steps or []),
    }
    return candidate


ABLATION_LAYER_KEYS = {
    "no_repair": "no_repair_candidates",
    "centroid_only": "centroid_only_candidates",
    "clash_only": "clash_only_candidates",
    "bond_inference_only": "bond_inference_only_candidates",
    "full_repair": "full_repair_candidates",
}
GATED_LAYER_KEYS = {
    "gated_repair": "gated_repair_candidates",
    "repair_rejected": "repair_rejected_candidates",
}
REPAIR_GATE_COUNT_KEYS = ("raw_passthrough", "repaired_candidate", "rejected_repair")


def normalized_repair_gate_counts(counter):
    return {key: int(counter.get(key, 0)) for key in REPAIR_GATE_COUNT_KEYS}


def process_artifact(path, output_dir, emit_ablation_layers=False, repair_gate_options=None):
    artifact = load_json(path)
    method_id = artifact.get("method_id") or artifact.get("active_method", {}).get("method_id") or "unknown"
    split = artifact.get("split_label") or "q1_public_full100_budget1"
    raw_candidates = artifact.get("raw_rollout_candidates", [])
    ablation_layers = {layer: [] for layer in ABLATION_LAYER_KEYS}
    gated_layers = {layer: [] for layer in GATED_LAYER_KEYS}
    repair_gate_counts = Counter()
    repaired = []
    rerank_pool = []
    for index, raw in enumerate(raw_candidates):
        raw = dict(raw)
        raw["method_id"] = raw.get("method_id") or method_id
        raw["layer"] = raw.get("layer") or "raw_rollout"
        raw["source_layer"] = raw.get("source_layer") or raw["layer"]
        pocket_atoms, pocket_center, pocket_radius = candidate_pocket_geometry(raw)
        raw["pocket_centroid"] = raw.get("pocket_centroid") or pocket_center
        raw["pocket_radius"] = raw.get("pocket_radius") or pocket_radius
        raw["transformation_provenance"] = raw.get("transformation_provenance") or {
            "algorithm": "public_baseline_raw_adapter",
            "source_layer": raw["layer"],
            "source_candidate_id": raw.get("candidate_id"),
            "transformations": [],
        }
        coords = candidate_coords(raw)
        raw_bonds = raw.get("inferred_bonds", [])
        if emit_ablation_layers:
            no_repair = layer_candidate(
                raw,
                "no_repair",
                coords,
                raw_bonds,
                f"q2_no_repair_copy:raw_candidate={raw.get('candidate_id')}",
                ["none"],
            )
            no_repair["model_native"] = True
            ablation_layers["no_repair"].append(no_repair)

            centered = centroid_only_geometry(coords, pocket_center)
            ablation_layers["centroid_only"].append(
                layer_candidate(
                    raw,
                    "centroid_only",
                    centered,
                    raw_bonds,
                    f"q2_centroid_translation_only:raw_candidate={raw.get('candidate_id')}",
                    ["centroid_translation"],
                )
            )

            declashed = clash_only_geometry(coords, pocket_atoms)
            ablation_layers["clash_only"].append(
                layer_candidate(
                    raw,
                    "clash_only",
                    declashed,
                    raw_bonds,
                    f"q2_clash_relaxation_only:raw_candidate={raw.get('candidate_id')}",
                    ["internal_clash_relaxation", "pocket_atom_declash"],
                )
            )

            inferred_on_raw = prune_bonds_for_valence(coords, raw.get("atom_types", []), infer_bonds(coords))
            ablation_layers["bond_inference_only"].append(
                layer_candidate(
                    raw,
                    "bond_inference_only",
                    coords,
                    inferred_on_raw,
                    f"q2_bond_inference_only:raw_candidate={raw.get('candidate_id')}",
                    ["distance_bond_inference", "valence_pruning"],
                )
            )

        fixed = repair_candidate_geometry(coords, pocket_atoms, pocket_center, pocket_radius, index, len(raw_candidates))
        bonds = prune_bonds_for_valence(fixed, raw.get("atom_types", []), infer_bonds(fixed))
        if emit_ablation_layers:
            ablation_layers["full_repair"].append(
                layer_candidate(
                    raw,
                    "full_repair",
                    fixed,
                    bonds,
                    f"q2_full_geometry_repair_and_bond_inference:raw_candidate={raw.get('candidate_id')}",
                    [
                        "centroid_anchored_geometry_repair",
                        "internal_clash_relaxation",
                        "pocket_atom_declash",
                        "pocket_envelope_clamp",
                        "distance_bond_inference",
                        "valence_pruning",
                    ],
                )
            )
        if repair_gate_options is not None:
            proposed = candidate_with_geometry(raw, "gated_repair", fixed, bonds)
            decision = repair_gate_decision(raw, proposed, repair_gate_options)
            repair_gate_counts["raw_passthrough"] += int(decision["raw_dockable"])
            if decision["accepted"]:
                gated = layer_candidate(
                    raw,
                    "gated_repair",
                    fixed,
                    bonds,
                    f"q3_gated_repair_accepted:raw_candidate={raw.get('candidate_id')}",
                    [
                        "repair_gate_checked",
                        "centroid_anchored_geometry_repair",
                        "internal_clash_relaxation",
                        "pocket_atom_declash",
                        "pocket_envelope_clamp",
                        "distance_bond_inference",
                        "valence_pruning",
                    ],
                )
                gated["repair_gate"] = decision
                gated_layers["gated_repair"].append(gated)
                repair_gate_counts["repaired_candidate"] += 1
            else:
                rejected = layer_candidate(
                    raw,
                    "repair_rejected",
                    coords,
                    raw_bonds,
                    f"q3_gated_repair_rejected:raw_candidate={raw.get('candidate_id')}",
                    ["repair_gate_checked", "fallback_to_raw_on_failed_sanity"],
                )
                rejected["repair_gate"] = decision
                rejected["repair_rejection_reason"] = decision["reasons"]
                gated_layers["repair_rejected"].append(rejected)
                repair_gate_counts["rejected_repair"] += 1
        repaired_candidate = layer_candidate(
            raw,
            "repaired",
            fixed,
            [],
            f"shared_public_geometry_repair:raw_candidate={raw.get('candidate_id')}",
            ["centroid_anchored_geometry_repair", "internal_clash_relaxation", "pocket_atom_declash", "pocket_envelope_clamp"],
        )
        repaired.append(repaired_candidate)
        rerank_candidate = layer_candidate(
            raw,
            "reranked",
            fixed,
            bonds,
            f"shared_public_deterministic_proxy_rerank:raw_candidate={raw.get('candidate_id')}",
            [
                "centroid_anchored_geometry_repair",
                "internal_clash_relaxation",
                "pocket_atom_declash",
                "pocket_envelope_clamp",
                "distance_bond_inference",
                "valence_pruning",
                "deterministic_proxy_rerank",
            ],
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
        "ablation_layers_enabled": bool(emit_ablation_layers),
        "repair_gate_enabled": repair_gate_options is not None,
        "repair_gate_options": repair_gate_options,
        "repair_gate_counts": normalized_repair_gate_counts(repair_gate_counts),
        "ablation_layer_candidate_counts": {
            layer: len(candidates) for layer, candidates in sorted(ablation_layers.items())
        },
        "gated_layer_candidate_counts": {
            layer: len(candidates) for layer, candidates in sorted(gated_layers.items())
        },
        "claim_boundary": "raw_rollout/no_repair remain native public baseline coordinates; centroid_only, clash_only, bond_inference_only, full_repair, repaired, and reranked are explicit postprocessing layers.",
    }
    output["raw_rollout_candidates"] = raw_candidates
    for layer, source_key in ABLATION_LAYER_KEYS.items():
        if emit_ablation_layers:
            output[source_key] = ablation_layers[layer]
    for layer, source_key in GATED_LAYER_KEYS.items():
        if repair_gate_options is not None:
            output[source_key] = gated_layers[layer]
    output["repaired_candidates"] = repaired
    output["reranked_candidates"] = reranked
    out_path = Path(output_dir) / method_id / "generation_layers_test.json"
    write_json(out_path, output)
    return {
        "method_id": method_id,
        "input_artifact": str(path),
        "output_artifact": str(out_path),
        "raw_candidate_count": len(raw_candidates),
        "ablation_layer_candidate_counts": {
            layer: len(candidates) for layer, candidates in sorted(ablation_layers.items())
        },
        "gated_layer_candidate_counts": {
            layer: len(candidates) for layer, candidates in sorted(gated_layers.items())
        },
        "repair_gate_counts": normalized_repair_gate_counts(repair_gate_counts),
        "repaired_candidate_count": len(repaired),
        "reranked_candidate_count": len(reranked),
    }


def main():
    args = parse_args()
    repair_gate_options = None
    if args.repair_gate:
        repair_gate_options = {
            "no_move_if_raw_dockable": bool(args.no_move_if_raw_dockable),
            "max_centroid_shift": float(args.max_centroid_shift),
            "max_clash_increase": float(args.max_clash_increase),
            "preserve_raw_box_center": True,
            "max_box_center_shift": float(args.max_box_center_shift),
            "fallback_to_raw_on_failed_sanity": True,
        }
    records = [
        process_artifact(
            path,
            args.output_dir,
            emit_ablation_layers=args.ablation_layers,
            repair_gate_options=repair_gate_options,
        )
        for path in args.generation_layers
    ]
    total_gate_counts = Counter()
    for record in records:
        total_gate_counts.update(record.get("repair_gate_counts", {}))
    write_json(
        args.report_json,
        {
            "schema_version": 1,
            "artifact_name": "q1_public_baseline_postprocess_report",
            "records": records,
            "totals": {
                "raw_candidate_count": sum(record["raw_candidate_count"] for record in records),
                "ablation_layer_candidate_counts": {
                    layer: sum(
                        record.get("ablation_layer_candidate_counts", {}).get(layer, 0)
                        for record in records
                    )
                    for layer in sorted(ABLATION_LAYER_KEYS)
                },
                "gated_layer_candidate_counts": {
                    layer: sum(
                        record.get("gated_layer_candidate_counts", {}).get(layer, 0)
                        for record in records
                    )
                    for layer in sorted(GATED_LAYER_KEYS)
                },
                "repair_gate_counts": normalized_repair_gate_counts(total_gate_counts),
                "repaired_candidate_count": sum(record["repaired_candidate_count"] for record in records),
                "reranked_candidate_count": sum(record["reranked_candidate_count"] for record in records),
            },
            "repair_gate_options": repair_gate_options,
            "claim_boundary": "Generated ablation, repaired, and reranked rows are postprocessing evidence except no_repair, which is an unchanged-coordinate copy used as the ablation baseline.",
        },
    )


if __name__ == "__main__":
    main()
