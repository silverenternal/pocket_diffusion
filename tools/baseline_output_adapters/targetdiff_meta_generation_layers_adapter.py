#!/usr/bin/env python3
"""Convert TargetDiff public sampling meta files into generation-layer artifacts."""

import argparse
import json
import re
from pathlib import Path


ELEMENT_TYPES = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "P": 4,
    "F": 5,
    "CL": 6,
    "BR": 7,
    "I": 8,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meta_pt", help="Official TargetDiff sampling_results/*.pt file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--split", default="targetdiff_public_overlap13")
    parser.add_argument("--budget-candidates-per-pocket", type=int, default=1)
    parser.add_argument(
        "--local-root",
        default="data/pdbbindpp-2020/extracted/pbpp-2020",
        help="Local receptor/ligand root used for unified rescoring.",
    )
    parser.add_argument(
        "--test-set-root",
        default=None,
        help=(
            "Official TargetDiff test_set root. When provided, receptor/ligand "
            "paths are resolved from ligand_filename before falling back to local-root."
        ),
    )
    parser.add_argument(
        "--require-local-pocket",
        action="store_true",
        help="Drop official pockets that cannot be mapped to local receptor/ligand files.",
    )
    return parser.parse_args()


def import_torch():
    import torch  # pylint: disable=import-outside-toplevel

    return torch


def atom_type(symbol):
    return ELEMENT_TYPES.get(str(symbol).upper(), 9)


def infer_reference_id(ligand_filename):
    match = re.search(r"_rec_([0-9A-Za-z]{4})_", ligand_filename or "")
    if match:
        return match.group(1).lower()
    stem = Path(ligand_filename or "unknown").stem
    return stem[:4].lower() if len(stem) >= 4 else "unknown"


def infer_test_set_paths(test_set_root, ligand_filename):
    if not test_set_root or not ligand_filename or "/" not in ligand_filename:
        return None
    root = Path(test_set_root)
    ligand_path = root / ligand_filename
    match = re.match(r"(.+)/(.+?_rec)_", ligand_filename)
    if not match:
        return None
    pocket_id = match.group(1)
    receptor_path = root / pocket_id / f"{match.group(2)}.pdb"
    return {
        "pocket_id": pocket_id,
        "protein_id": pocket_id,
        "source_pocket_path": receptor_path if receptor_path.is_file() else None,
        "source_ligand_path": ligand_path if ligand_path.is_file() else None,
    }


def local_paths(local_root, protein_id):
    root = Path(local_root) / protein_id
    pocket = root / f"{protein_id}_pocket.pdb"
    ligand = root / f"{protein_id}_ligand.sdf"
    if not ligand.is_file():
        ligand = root / f"{protein_id}_ligand.mol2"
    return pocket, ligand


def resolved_paths(source_row, local_root, test_set_root):
    ligand_filename = source_row.get("ligand_filename")
    official = infer_test_set_paths(test_set_root, ligand_filename)
    if official:
        return official
    protein_id = infer_reference_id(ligand_filename)
    pocket_path, ligand_path = local_paths(local_root, protein_id)
    return {
        "pocket_id": protein_id,
        "protein_id": protein_id,
        "source_pocket_path": pocket_path if pocket_path.is_file() else None,
        "source_ligand_path": ligand_path if ligand_path.is_file() else None,
    }


def molecule_record(
    mol,
    method_id,
    split,
    pocket_index,
    candidate_index,
    source_row,
    local_root,
    test_set_root,
):
    ligand_filename = source_row.get("ligand_filename")
    paths = resolved_paths(source_row, local_root, test_set_root)
    protein_id = paths["protein_id"]
    example_id = f"{pocket_index:03d}_{paths['pocket_id']}"
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    coords = []
    for atom_index in range(mol.GetNumAtoms()):
        if conf is None:
            coords.append([0.0, 0.0, 0.0])
        else:
            pos = conf.GetAtomPosition(atom_index)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
    official_vina = source_row.get("vina") if isinstance(source_row.get("vina"), dict) else {}
    return {
        "candidate_id": f"{method_id}:raw_rollout:{example_id}:{candidate_index}",
        "example_id": example_id,
        "protein_id": protein_id,
        "split_label": split,
        "method_id": method_id,
        "layer": "raw_rollout",
        "source": "targetdiff_public_sampling_results",
        "smiles": source_row.get("smiles"),
        "atom_types": [atom_type(atom.GetSymbol()) for atom in mol.GetAtoms()],
        "coords": coords,
        "coordinate_frame_origin": [0.0, 0.0, 0.0],
        "source_pocket_path": str(paths["source_pocket_path"]) if paths["source_pocket_path"] else None,
        "source_ligand_path": str(paths["source_ligand_path"]) if paths["source_ligand_path"] else None,
        "adapter_metadata": {
            "official_ligand_filename": ligand_filename,
            "official_pocket_index": pocket_index,
            "official_candidate_index": candidate_index,
            "official_vina": official_vina,
            "path_resolution": "targetdiff_test_set" if test_set_root else "local_root",
        },
    }


def main():
    args = parse_args()
    torch = import_torch()
    meta = torch.load(args.meta_pt, map_location="cpu")
    records = []
    skipped_missing_local = 0
    skipped_empty = 0
    for pocket_index, pocket_rows in enumerate(meta):
        if not pocket_rows:
            skipped_empty += 1
            continue
        emitted_for_pocket = 0
        for source_row in pocket_rows:
            mol = source_row.get("mol") if isinstance(source_row, dict) else None
            if mol is None:
                continue
            paths = resolved_paths(source_row, args.local_root, args.test_set_root)
            if args.require_local_pocket and not (
                paths["source_pocket_path"] and paths["source_ligand_path"]
            ):
                skipped_missing_local += 1
                break
            records.append(
                molecule_record(
                    mol,
                    args.method_id,
                    args.split,
                    pocket_index,
                    emitted_for_pocket,
                    source_row,
                    args.local_root,
                    args.test_set_root,
                )
            )
            emitted_for_pocket += 1
            if emitted_for_pocket >= args.budget_candidates_per_pocket:
                break

    payload = {
        "schema_version": 1,
        "method_id": args.method_id,
        "split_label": args.split,
        "adapter": "targetdiff_meta_generation_layers_adapter",
        "source_meta_pt": args.meta_pt,
        "test_set_root": args.test_set_root,
        "budget_check": {
            "requested_candidates_per_pocket": args.budget_candidates_per_pocket,
            "emitted_candidate_count": len(records),
            "skipped_empty_pockets": skipped_empty,
            "skipped_missing_local_pockets": skipped_missing_local,
            "require_local_pocket": args.require_local_pocket,
        },
        "raw_rollout_candidates": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"TargetDiff meta generation layer artifact written: {output}")


if __name__ == "__main__":
    main()
