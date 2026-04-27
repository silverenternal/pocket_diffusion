#!/usr/bin/env python3
"""Convert generated SDF molecules into the generation-layer artifact schema."""

import argparse
import json
from pathlib import Path

from rdkit import Chem


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

LAYER_KEYS = {
    "raw_rollout": "raw_rollout_candidates",
    "repaired": "repaired_candidates",
    "inferred_bond": "inferred_bond_candidates",
    "deterministic_proxy": "deterministic_proxy_candidates",
    "reranked": "reranked_candidates",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdf", help="Generated SDF file, possibly containing multiple molecules.")
    parser.add_argument("--output", required=True, help="Output generation_layers_<split>.json path.")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--protein-id", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--layer", default="raw_rollout", choices=sorted(LAYER_KEYS))
    parser.add_argument("--source-pocket-path", default=None)
    parser.add_argument("--source-ligand-path", default=None)
    parser.add_argument("--budget-candidates", type=int, default=None)
    return parser.parse_args()


def atom_type(symbol):
    return ELEMENT_TYPES.get(symbol.upper(), 9)


def molecule_record(mol, args, index):
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    coords = []
    for atom_index in range(mol.GetNumAtoms()):
        if conf is None:
            coords.append([0.0, 0.0, 0.0])
        else:
            pos = conf.GetAtomPosition(atom_index)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
    atom_types = [atom_type(atom.GetSymbol()) for atom in mol.GetAtoms()]
    return {
        "candidate_id": f"{args.method_id}:{args.layer}:{args.example_id}:{index}",
        "example_id": args.example_id,
        "protein_id": args.protein_id or args.example_id,
        "source": "sdf_generation_layers_adapter",
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=True) if mol.GetNumAtoms() else None,
        "atom_types": atom_types,
        "coords": coords,
        "coordinate_frame_origin": [0.0, 0.0, 0.0],
        "source_pocket_path": args.source_pocket_path,
        "source_ligand_path": args.source_ligand_path,
        "adapter_metadata": {
            "input_sdf": str(args.sdf),
            "layer": args.layer,
            "sdf_index": index,
        },
    }


def main():
    args = parse_args()
    supplier = Chem.SDMolSupplier(str(args.sdf), sanitize=False, removeHs=False)
    records = []
    for index, mol in enumerate(supplier):
        if mol is None:
            continue
        records.append(molecule_record(mol, args, index))
        if args.budget_candidates is not None and len(records) >= args.budget_candidates:
            break

    layer_key = LAYER_KEYS[args.layer]
    payload = {
        "schema_version": 1,
        "method_id": args.method_id,
        "split_label": args.split,
        "adapter": "sdf_generation_layers_adapter",
        "budget_check": {
            "requested_candidate_count": args.budget_candidates,
            "emitted_candidate_count": len(records),
            "budget_matched": args.budget_candidates is None or len(records) == args.budget_candidates,
        },
        layer_key: records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SDF generation layer artifact written: {output}")


if __name__ == "__main__":
    main()
