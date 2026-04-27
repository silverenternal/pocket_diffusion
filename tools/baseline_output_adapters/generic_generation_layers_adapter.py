#!/usr/bin/env python3
"""Normalize external baseline candidates into the project generation-layer schema."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON/JSONL/CSV baseline output.")
    parser.add_argument("--output", required=True, help="generation_layers_<split>.json output path.")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--layer", default="raw_rollout")
    parser.add_argument("--budget-candidates", type=int, default=None)
    return parser.parse_args()


def load_rows(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for key in ("candidates", "records", "molecules"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise SystemExit("JSON input must be a list or contain candidates/records/molecules")


def as_float_list(value):
    if isinstance(value, str):
        value = value.replace(";", ",").split(",")
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def normalize_candidate(row, method_id, layer, index):
    coords = row.get("coords") or row.get("coordinates") or []
    if isinstance(coords, str):
        try:
            coords = json.loads(coords)
        except json.JSONDecodeError:
            coords = []
    atom_types = row.get("atom_types") or row.get("atoms") or []
    if isinstance(atom_types, str):
        try:
            atom_types = json.loads(atom_types)
        except json.JSONDecodeError:
            atom_types = [item for item in atom_types.replace(";", ",").split(",") if item]
    candidate_id = row.get("candidate_id") or f"{method_id}:{layer}:{row.get('example_id', 'unknown')}:{index}"
    return {
        "candidate_id": candidate_id,
        "example_id": row.get("example_id") or row.get("pocket_id") or "unknown",
        "protein_id": row.get("protein_id") or row.get("pocket_id") or "unknown",
        "source": row.get("source") or "external_baseline_adapter",
        "smiles": row.get("smiles"),
        "atom_types": atom_types,
        "coords": coords if isinstance(coords, list) else [],
        "coordinate_frame_origin": as_float_list(row.get("coordinate_frame_origin"))[:3],
        "source_pocket_path": row.get("source_pocket_path") or row.get("pocket_path"),
        "adapter_metadata": {
            "original_id": row.get("id") or row.get("candidate_id"),
            "layer": layer,
        },
    }


def main():
    args = parse_args()
    rows = load_rows(args.input)
    if args.budget_candidates is not None:
        rows = rows[: args.budget_candidates]
    candidates = [
        normalize_candidate(row, args.method_id, args.layer, index)
        for index, row in enumerate(rows)
    ]
    layer_key = {
        "raw_rollout": "raw_rollout_candidates",
        "repaired": "repaired_candidates",
        "inferred_bond": "inferred_bond_candidates",
        "deterministic_proxy": "deterministic_proxy_candidates",
        "reranked": "reranked_candidates",
    }.get(args.layer, "raw_rollout_candidates")
    payload = {
        "schema_version": 1,
        "method_id": args.method_id,
        "split_label": args.split,
        "adapter": "generic_generation_layers_adapter",
        "budget_check": {
            "requested_candidate_count": args.budget_candidates,
            "emitted_candidate_count": len(candidates),
            "budget_matched": args.budget_candidates is None or len(candidates) == args.budget_candidates,
        },
        layer_key: candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"baseline generation layer artifact written: {output}")


if __name__ == "__main__":
    main()
