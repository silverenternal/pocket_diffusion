#!/usr/bin/env python3
import json
import math
import sys


ATOM_NUMBERS = {
    0: 6,
    1: 7,
    2: 8,
    3: 16,
    4: 1,
}


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def build_molecule(candidate, Chem):
    mol = Chem.RWMol()
    for atom_type in candidate.get("atom_types", []):
        atomic_num = ATOM_NUMBERS.get(int(atom_type), 6)
        mol.AddAtom(Chem.Atom(atomic_num))

    atom_count = len(candidate.get("atom_types", []))
    for left, right in candidate.get("inferred_bonds", []):
        left = int(left)
        right = int(right)
        if 0 <= left < atom_count and 0 <= right < atom_count and left != right:
            if mol.GetBondBetweenAtoms(left, right) is None:
                mol.AddBond(left, right, Chem.BondType.SINGLE)

    conformer = Chem.Conformer(atom_count)
    for atom_ix, coords in enumerate(candidate.get("coords", [])):
        if atom_ix >= atom_count:
            break
        x, y, z = [float(value) for value in coords]
        conformer.SetAtomPosition(atom_ix, (x, y, z))
    mol.AddConformer(conformer, assignId=True)
    return mol.GetMol()


def main(argv):
    if len(argv) < 3:
        raise SystemExit("usage: rdkit_validity_backend.py <input.json> <output.json>")

    input_path = argv[-2]
    output_path = argv[-1]
    candidates = load_candidates(input_path)

    try:
        from rdkit import Chem
    except Exception:
        write_metrics(
            output_path,
            {
                "schema_version": 1.0,
                "rdkit_available": 0.0,
                "backend_import_error": 1.0,
                "backend_examples_scored": 0.0,
                "backend_missing_structure_fraction": 1.0,
            },
        )
        return

    total = max(len(candidates), 1)
    parseable = 0
    sanitized = 0
    unique_smiles = set()
    finite_conformers = 0

    for candidate in candidates:
        try:
            mol = build_molecule(candidate, Chem)
            parseable += 1
        except Exception:
            continue

        coords = candidate.get("coords", [])
        if all(
            len(coord) == 3 and all(math.isfinite(float(value)) for value in coord)
            for coord in coords
        ):
            finite_conformers += 1

        try:
            Chem.SanitizeMol(mol)
            sanitized += 1
        except Exception:
            continue

        try:
            unique_smiles.add(Chem.MolToSmiles(mol))
        except Exception:
            pass

    write_metrics(
        output_path,
        {
            "schema_version": 1.0,
            "rdkit_available": 1.0,
            "backend_examples_scored": float(len(candidates)),
            "backend_missing_structure_fraction": 0.0,
            "rdkit_parseable_fraction": parseable / total,
            "rdkit_sanitized_fraction": sanitized / total,
            "rdkit_unique_smiles_fraction": len(unique_smiles) / total,
            "rdkit_finite_conformer_fraction": finite_conformers / total,
        },
    )


if __name__ == "__main__":
    main(sys.argv)
