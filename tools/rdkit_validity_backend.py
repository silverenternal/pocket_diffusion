#!/usr/bin/env python3
import argparse
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


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="RDKit chemistry, drug-likeness, scaffold, and diversity backend"
    )
    parser.add_argument(
        "--reference-candidates",
        help=(
            "Optional JSON file containing split-local training/reference candidates. "
            "Scaffold novelty and nearest-train similarity are computed against this file."
        ),
    )
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    return parser.parse_args(argv[1:])


def sa_score_proxy(mol, Descriptors, rdMolDescriptors):
    heavy_atoms = max(Descriptors.HeavyAtomCount(mol), 1)
    rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    hetero = rdMolDescriptors.CalcNumHBA(mol) + rdMolDescriptors.CalcNumHBD(mol)
    raw = 1.0 + 0.08 * heavy_atoms + 0.18 * rotors + 0.12 * rings - 0.03 * hetero
    return max(1.0, min(10.0, raw))


def lipinski_violations(mw, logp, hbd, hba):
    return int(mw > 500.0) + int(logp > 5.0) + int(hbd > 5) + int(hba > 10)


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


def load_reference_molecules(path, Chem):
    if not path:
        return []
    try:
        references = load_candidates(path)
    except Exception:
        return []
    molecules = []
    for candidate in references:
        try:
            mol = build_molecule(candidate, Chem)
            Chem.SanitizeMol(mol)
            molecules.append(mol)
        except Exception:
            continue
    return molecules


def molecule_scaffold_smiles(mol, MurckoScaffold):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=scaffold)


def molecule_fingerprint(mol, AllChem):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def mean_pairwise_tanimoto(fingerprints, DataStructs):
    if len(fingerprints) < 2:
        return 0.0
    total = 0.0
    count = 0
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            total += float(DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right]))
            count += 1
    return total / float(count) if count else 0.0


def nearest_similarity(fingerprint, reference_fingerprints, DataStructs):
    if not reference_fingerprints:
        return 0.0
    return max(
        float(DataStructs.TanimotoSimilarity(fingerprint, reference))
        for reference in reference_fingerprints
    )


def main(argv):
    args = parse_args(argv)
    input_path = args.input_json
    output_path = args.output_json
    candidates = load_candidates(input_path)

    try:
        from rdkit import Chem
    except Exception:
        write_metrics(
            output_path,
            {
                "schema_version": 1.0,
                "aggregate_metrics": {
                    "schema_version": 1.0,
                    "rdkit_available": 0.0,
                    "backend_import_error": 1.0,
                    "backend_examples_scored": 0.0,
                    "backend_missing_structure_fraction": 1.0,
                    "drug_likeness_coverage_fraction": 0.0,
                    "scaffold_metric_coverage_fraction": 0.0,
                },
                "candidate_metrics": [],
            },
        )
        return

    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        write_metrics(
            output_path,
            {
                "schema_version": 1.0,
                "aggregate_metrics": {
                    "schema_version": 1.0,
                    "rdkit_available": 0.0,
                    "backend_import_error": 1.0,
                    "backend_examples_scored": 0.0,
                    "backend_missing_structure_fraction": 1.0,
                    "drug_likeness_coverage_fraction": 0.0,
                    "scaffold_metric_coverage_fraction": 0.0,
                },
                "candidate_metrics": [],
            },
        )
        return

    reference_molecules = load_reference_molecules(args.reference_candidates, Chem)
    reference_scaffolds = set()
    reference_fingerprints = []
    for mol in reference_molecules:
        try:
            reference_scaffolds.add(molecule_scaffold_smiles(mol, MurckoScaffold))
            reference_fingerprints.append(molecule_fingerprint(mol, AllChem))
        except Exception:
            continue

    total = max(len(candidates), 1)
    parseable = 0
    sanitized = 0
    unique_smiles = set()
    finite_conformers = 0
    drug_like_count = 0
    qed_values = []
    sa_values = []
    logp_values = []
    tpsa_values = []
    mw_values = []
    hbd_values = []
    hba_values = []
    rotatable_values = []
    lipinski_values = []
    scaffold_count = 0
    novel_scaffold_count = 0
    scaffolds = set()
    fingerprints = []
    nearest_train_values = []
    candidate_metrics = []

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id") or "unknown"
        example_id = candidate.get("example_id") or "unknown"
        protein_id = candidate.get("protein_id") or "unknown"
        row = {
            "rdkit_parseable_fraction": 0.0,
            "rdkit_valid_fraction": 0.0,
            "rdkit_sanitized_fraction": 0.0,
            "rdkit_unique_smiles_fraction": 0.0,
            "rdkit_finite_conformer_fraction": 0.0,
            "backend_missing_structure_fraction": 0.0,
        }
        try:
            mol = build_molecule(candidate, Chem)
            parseable += 1
            row["rdkit_parseable_fraction"] = 1.0
            row["rdkit_valid_fraction"] = 1.0
        except Exception:
            candidate_metrics.append(
                {
                    "candidate_id": candidate_id,
                    "example_id": example_id,
                    "protein_id": protein_id,
                    "metrics": row,
                }
            )
            continue

        coords = candidate.get("coords", [])
        if all(
            len(coord) == 3 and all(math.isfinite(float(value)) for value in coord)
            for coord in coords
        ):
            finite_conformers += 1
            row["rdkit_finite_conformer_fraction"] = 1.0

        try:
            Chem.SanitizeMol(mol)
            sanitized += 1
            row["rdkit_sanitized_fraction"] = 1.0
        except Exception:
            candidate_metrics.append(
                {
                    "candidate_id": candidate_id,
                    "example_id": example_id,
                    "protein_id": protein_id,
                    "metrics": row,
                }
            )
            continue

        try:
            unique_smiles.add(Chem.MolToSmiles(mol))
            row["rdkit_unique_smiles_fraction"] = 1.0
        except Exception:
            pass
        try:
            qed = float(QED.qed(mol))
            sa = float(sa_score_proxy(mol, Descriptors, rdMolDescriptors))
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            mw = float(Descriptors.MolWt(mol))
            hbd = float(Lipinski.NumHDonors(mol))
            hba = float(Lipinski.NumHAcceptors(mol))
            rotatable = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
            lipinski = float(lipinski_violations(mw, logp, hbd, hba))
            row.update(
                {
                    "drug_likeness_coverage_fraction": 1.0,
                    "qed": qed,
                    "raw_qed": qed,
                    "sa_score": sa,
                    "raw_sa": sa,
                    "logp": logp,
                    "tpsa": tpsa,
                    "molecular_weight": mw,
                    "hbd": hbd,
                    "hba": hba,
                    "rotatable_bonds": rotatable,
                    "lipinski_violations": lipinski,
                }
            )
            drug_like_count += 1
            qed_values.append(qed)
            sa_values.append(sa)
            logp_values.append(logp)
            tpsa_values.append(tpsa)
            mw_values.append(mw)
            hbd_values.append(hbd)
            hba_values.append(hba)
            rotatable_values.append(rotatable)
            lipinski_values.append(lipinski)
        except Exception:
            row["drug_likeness_coverage_fraction"] = 0.0
        try:
            scaffold = molecule_scaffold_smiles(mol, MurckoScaffold)
            fingerprint = molecule_fingerprint(mol, AllChem)
            nearest_train = nearest_similarity(fingerprint, reference_fingerprints, DataStructs)
            scaffold_count += 1
            scaffolds.add(scaffold)
            fingerprints.append(fingerprint)
            nearest_train_values.append(nearest_train)
            is_novel_scaffold = not reference_scaffolds or scaffold not in reference_scaffolds
            if is_novel_scaffold:
                novel_scaffold_count += 1
            row.update(
                {
                    "scaffold_metric_coverage_fraction": 1.0,
                    "scaffold_novelty_fraction": 1.0 if is_novel_scaffold else 0.0,
                    "unique_scaffold_fraction": 1.0,
                    "nearest_train_similarity": nearest_train,
                    "reference_scaffold_count": float(len(reference_scaffolds)),
                }
            )
        except Exception:
            row["scaffold_metric_coverage_fraction"] = 0.0
        candidate_metrics.append(
            {
                "candidate_id": candidate_id,
                "example_id": example_id,
                "protein_id": protein_id,
                "metrics": row,
            }
        )

    write_metrics(
        output_path,
        {
            "schema_version": 1.0,
            "aggregate_metrics": {
                "schema_version": 1.0,
                "rdkit_available": 1.0,
                "backend_examples_scored": float(len(candidates)),
                "backend_missing_structure_fraction": 0.0,
                "rdkit_parseable_fraction": parseable / total,
                "rdkit_valid_fraction": parseable / total,
                "rdkit_sanitized_fraction": sanitized / total,
                "rdkit_unique_smiles_fraction": len(unique_smiles) / total,
                "rdkit_finite_conformer_fraction": finite_conformers / total,
                "drug_likeness_coverage_fraction": drug_like_count / total,
                "qed": mean(qed_values),
                "raw_qed": mean(qed_values),
                "sa_score": mean(sa_values),
                "raw_sa": mean(sa_values),
                "logp": mean(logp_values),
                "tpsa": mean(tpsa_values),
                "molecular_weight": mean(mw_values),
                "hbd": mean(hbd_values),
                "hba": mean(hba_values),
                "rotatable_bonds": mean(rotatable_values),
                "lipinski_violations": mean(lipinski_values),
                "scaffold_metric_coverage_fraction": scaffold_count / total,
                "scaffold_novelty_fraction": novel_scaffold_count / total,
                "unique_scaffold_fraction": len(scaffolds) / total,
                "pairwise_tanimoto_mean": mean_pairwise_tanimoto(fingerprints, DataStructs),
                "nearest_train_similarity": mean(nearest_train_values),
                "reference_scaffold_count": float(len(reference_scaffolds)),
            },
            "candidate_metrics": candidate_metrics,
        },
    )


if __name__ == "__main__":
    main(sys.argv)
