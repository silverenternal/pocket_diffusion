#!/usr/bin/env python3
"""Build a compact affinity label table for the local PDBbind++ 2020 manifest."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an affinity label CSV keyed by PDB id for the local PDBbind++ manifest."
    )
    parser.add_argument(
        "--source-csv",
        required=True,
        help="Upstream CSV containing PDB ids and compact affinity records.",
    )
    parser.add_argument(
        "--manifest",
        default="data/pdbbindpp-2020/manifest.json",
        help="Local manifest_json file whose example ids should be covered.",
    )
    parser.add_argument(
        "--output",
        default="data/pdbbindpp-2020/affinity_labels.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_manifest_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["example_id"].strip().lower()
        for entry in payload.get("entries", [])
        if entry.get("example_id")
    }


def compact_affinity_record(row: dict[str, str]) -> str | None:
    record = (row.get("kd/ki") or row.get("Kd/Ki") or "").strip()
    if not record:
        value = (row.get("value") or "").strip()
        if not value:
            return None
        return f"pKd={value}"
    normalized = record.replace(" ", "")
    if any(token in normalized for token in ("<", ">", "~")):
        return None
    return normalized


def is_refined_row(row: dict[str, str]) -> bool:
    category = (row.get("category") or "").strip().lower()
    return category == "refined"


def main():
    args = parse_args()
    source_csv = Path(args.source_csv)
    manifest = Path(args.manifest)
    output = Path(args.output)

    manifest_ids = load_manifest_ids(manifest)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        id_field = reader.fieldnames[0] if reader.fieldnames else None
        if id_field is None:
            raise SystemExit("source csv is missing a leading PDB id column")

        for row in reader:
            if not is_refined_row(row):
                continue
            example_id = (row.get(id_field) or "").strip().lower()
            if not example_id or example_id not in manifest_ids or example_id in seen:
                continue
            affinity_record = compact_affinity_record(row)
            if not affinity_record:
                continue
            rows.append(
                {
                    "example_id": example_id,
                    "affinity_record": affinity_record,
                }
            )
            seen.add(example_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "affinity_record"])
        writer.writeheader()
        writer.writerows(rows)

    missing = sorted(manifest_ids.difference(seen))
    print(f"label table written: {output}")
    print(f"manifest ids: {len(manifest_ids)}")
    print(f"labels written: {len(rows)}")
    print(f"missing manifest ids: {len(missing)}")
    if missing:
        print("first missing ids:", ",".join(missing[:20]))


if __name__ == "__main__":
    main()
