#!/usr/bin/env python3
"""Create no-coordinate-move bond refinement layers from raw generation artifacts."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_layers")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-id", default="flow_matching")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def distance(left, right):
    return math.sqrt(sum((float(left[axis]) - float(right[axis])) ** 2 for axis in range(3)))


def infer_bonds(coords):
    bonds = []
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            dist = distance(coords[left], coords[right])
            if 0.85 <= dist <= 1.75:
                bonds.append([left, right])
    return bonds


def max_valence(atom_type):
    return {
        0: 4,
        1: 3,
        2: 2,
        3: 6,
        4: 1,
        6: 4,
        7: 4,
        8: 3,
        9: 1,
        15: 5,
        16: 6,
        17: 1,
        35: 1,
        53: 1,
    }.get(int(atom_type), 4)


def valence_violation_count(atom_types, bonds):
    degrees = [0 for _ in atom_types]
    for left, right in bonds:
        if left < len(degrees) and right < len(degrees):
            degrees[left] += 1
            degrees[right] += 1
    return sum(1 for degree, atom_type in zip(degrees, atom_types) if degree > max_valence(atom_type))


def prune_bonds_for_valence(coords, atom_types, bonds):
    ordered = sorted(bonds, key=lambda edge: distance(coords[edge[0]], coords[edge[1]]))
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


def layer_candidate(raw, layer, bonds, source):
    candidate = dict(raw)
    atom_types = candidate.get("atom_types", [])
    source_layer = str(raw.get("layer") or "raw_flow")
    raw_candidate_id = raw.get("candidate_id")
    candidate["candidate_id"] = f"{candidate.get('method_id', 'flow_matching')}:{layer}:{candidate.get('example_id', 'unknown')}:0"
    candidate["layer"] = layer
    candidate["source_layer"] = layer if layer == "raw_geometry" else source_layer
    candidate["inferred_bonds"] = bonds
    candidate["bond_count"] = len(bonds)
    candidate["valence_violation_count"] = valence_violation_count(atom_types, bonds)
    candidate["source"] = source
    candidate["model_native"] = layer == "raw_geometry"
    candidate["claim_allowed"] = layer == "raw_geometry"
    candidate["postprocessing_steps"] = [] if layer == "raw_geometry" else [source]
    candidate["molecular_representation"] = (
        f"source={source};atoms={len(atom_types)};bonds={len(bonds)};"
        "coordinates=raw_preserved"
    )
    candidate["adapter_metadata"] = {
        "algorithm": "q3_no_coordinate_move_bond_refinement",
        "raw_candidate_id": raw_candidate_id,
        "coordinates_preserved": True,
    }
    candidate["transformation_provenance"] = {
        "algorithm": "q3_no_coordinate_move_bond_refinement",
        "source_layer": candidate["source_layer"],
        "source_candidate_id": raw_candidate_id,
        "transformations": [] if layer == "raw_geometry" else [source],
        "coordinates_preserved": True,
    }
    return candidate


def repaired_candidate(raw, constrained):
    candidate = dict(constrained)
    candidate["candidate_id"] = f"{raw.get('method_id', 'flow_matching')}:repaired:{raw.get('example_id', 'unknown')}:0"
    candidate["layer"] = "repaired"
    candidate["method_id"] = raw.get("method_id", candidate.get("method_id", "flow_matching"))
    bonds = candidate.get("inferred_bonds") or infer_bonds(candidate.get("coords", []))
    candidate["inferred_bonds"] = bonds
    candidate["bond_count"] = len(bonds)
    candidate["valence_violation_count"] = valence_violation_count(candidate.get("atom_types", []), bonds)
    candidate["source"] = "coordinate_moving_constrained_flow_reference"
    candidate["source_layer"] = "constrained_flow"
    candidate["model_native"] = False
    candidate["claim_allowed"] = False
    candidate["postprocessing_steps"] = ["coordinate_moving_constrained_flow_reference"]
    candidate["transformation_provenance"] = {
        "algorithm": "q3_coordinate_moving_reference",
        "source_layer": "constrained_flow",
        "source_candidate_id": constrained.get("candidate_id"),
        "transformations": ["coordinate_moving_constrained_flow_reference"],
        "coordinates_preserved": False,
    }
    return candidate


def main():
    args = parse_args()
    source = load_json(args.generation_layers)
    raw_candidates = source.get("raw_flow_candidates") or source.get("raw_rollout_candidates") or []
    constrained_by_example = {
        candidate.get("example_id"): candidate
        for candidate in source.get("constrained_flow_candidates", [])
    }
    raw_geometry = []
    bond_logits_refined = []
    valence_refined = []
    repaired = []
    for raw in raw_candidates:
        raw = dict(raw)
        raw["method_id"] = raw.get("method_id") or args.method_id
        raw["layer"] = "raw_geometry"
        raw["candidate_id"] = f"{raw['method_id']}:raw_geometry:{raw.get('example_id', 'unknown')}:0"
        coords = raw.get("coords", [])
        bonds = infer_bonds(coords)
        raw_geometry.append(layer_candidate(raw, "raw_geometry", [], "raw_geometry"))
        bond_logits_refined.append(
            layer_candidate(raw, "bond_logits_refined", bonds, "distance_bond_inference_no_coordinate_move")
        )
        valence_bonds = prune_bonds_for_valence(coords, raw.get("atom_types", []), bonds)
        valence_refined.append(
            layer_candidate(raw, "valence_refined", valence_bonds, "valence_pruning_no_coordinate_move")
        )
        constrained = constrained_by_example.get(raw.get("example_id"))
        if constrained is not None:
            repaired.append(repaired_candidate(raw, constrained))
    payload = {
        "schema_version": 1,
        "artifact_name": "q3_bond_refinement_generation_layers",
        "split_label": source.get("split_label") or "q3_bond_refinement_public100",
        "method_id": args.method_id,
        "claim_boundary": "raw_geometry, bond_logits_refined, and valence_refined preserve raw coordinates; repaired is a coordinate-moving reference and not native model improvement.",
        "source_artifact": args.generation_layers,
        "raw_geometry_candidates": raw_geometry,
        "bond_logits_refined_candidates": bond_logits_refined,
        "valence_refined_candidates": valence_refined,
        "repaired_candidates": repaired,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
