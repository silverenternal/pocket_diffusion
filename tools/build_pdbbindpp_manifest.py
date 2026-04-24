#!/usr/bin/env python3
"""Build a manifest for the Hugging Face PDBbind++ 2020 refined-set layout."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a manifest_json dataset for PDBbind++ 2020.")
    parser.add_argument(
        "--root",
        default="data/pdbbindpp-2020/extracted/pbpp-2020",
        help="Directory containing one subdirectory per PDB id.",
    )
    parser.add_argument(
        "--output",
        default="data/pdbbindpp-2020/manifest.json",
        help="Manifest path to write.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    entries = []
    missing = []

    for complex_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pdb_id = complex_dir.name
        pocket = complex_dir / f"{pdb_id}_pocket.pdb"
        ligand = complex_dir / f"{pdb_id}_ligand.sdf"
        if not pocket.is_file() or not ligand.is_file():
            missing.append(pdb_id)
            continue
        entries.append(
            {
                "example_id": pdb_id,
                "protein_id": pdb_id,
                "pocket_path": str(pocket.relative_to(output.parent)),
                "ligand_path": str(ligand.relative_to(output.parent)),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"manifest written: {output}")
    print(f"entries: {len(entries)}")
    print(f"missing_pairs: {len(missing)}")
    if missing:
        print("first_missing:", ",".join(missing[:20]))


if __name__ == "__main__":
    main()
