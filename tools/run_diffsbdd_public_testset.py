#!/usr/bin/env python3
"""Run the public DiffSBDD checkpoint on the official TargetDiff test set."""

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsbdd-root", default="external_baselines/DiffSBDD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta-pt", required=True)
    parser.add_argument("--test-set-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260427)
    return parser.parse_args()


def official_paths(test_set_root, ligand_filename):
    if not ligand_filename or "/" not in ligand_filename:
        raise ValueError(f"cannot parse ligand filename: {ligand_filename!r}")
    match = re.match(r"(.+)/(.+?_rec)_", ligand_filename)
    if not match:
        raise ValueError(f"cannot parse receptor filename from: {ligand_filename!r}")
    root = Path(test_set_root)
    pocket_id = match.group(1)
    receptor_path = root / pocket_id / f"{match.group(2)}.pdb"
    ligand_path = root / ligand_filename
    if not receptor_path.is_file():
        raise FileNotFoundError(receptor_path)
    if not ligand_path.is_file():
        raise FileNotFoundError(ligand_path)
    return pocket_id, receptor_path, ligand_path


def load_diffsbdd(diffsbdd_root, checkpoint):
    root = Path(diffsbdd_root).resolve()
    sys.path.insert(0, str(root))
    from openbabel import openbabel  # pylint: disable=import-outside-toplevel
    from lightning_modules import LigandPocketDDPM  # pylint: disable=import-outside-toplevel

    openbabel.obErrorLog.StopLogging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LigandPocketDDPM.load_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device)
    model.eval()
    return model, device


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Imported after sys.path is set by load_diffsbdd.
    model, device = load_diffsbdd(args.diffsbdd_root, args.checkpoint)
    import utils  # pylint: disable=import-outside-toplevel

    meta = torch.load(args.meta_pt, map_location="cpu")
    rows = []
    start_all = perf_counter()
    total = len(meta) if args.limit is None else min(args.limit, len(meta))
    for pocket_index, pocket_rows in enumerate(meta[:total]):
        ligand_filename = pocket_rows[0].get("ligand_filename", "") if pocket_rows else ""
        pocket_start = perf_counter()
        record = {
            "official_pocket_index": pocket_index,
            "official_ligand_filename": ligand_filename,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "timesteps": args.timesteps,
            "status": "pending",
        }
        try:
            pocket_id, receptor_path, ligand_path = official_paths(args.test_set_root, ligand_filename)
            outfile = output_dir / f"{pocket_index:03d}_{pocket_id}_raw.sdf"
            record.update(
                {
                    "pocket_id": pocket_id,
                    "source_pocket_path": str(receptor_path),
                    "source_ligand_path": str(ligand_path),
                    "output_sdf": str(outfile),
                }
            )
            if args.skip_existing and outfile.is_file() and outfile.stat().st_size > 0:
                record["status"] = "skipped_existing"
            else:
                molecules = []
                attempts = 0
                while len(molecules) < args.n_samples and attempts < args.max_attempts:
                    attempts += 1
                    batch = model.generate_ligands(
                        str(receptor_path),
                        args.batch_size,
                        ref_ligand=str(ligand_path),
                        sanitize=False,
                        largest_frag=True,
                        relax_iter=0,
                        timesteps=args.timesteps,
                    )
                    molecules.extend(batch)
                molecules = molecules[: args.n_samples]
                record["attempts"] = attempts
                record["molecule_count"] = len(molecules)
                if len(molecules) == args.n_samples:
                    utils.write_sdf_file(outfile, molecules)
                    record["status"] = "pass"
                else:
                    record["status"] = "failed_budget"
        except Exception as exc:  # pylint: disable=broad-except
            record["status"] = "error"
            record["error"] = repr(exc)
        record["runtime_seconds"] = perf_counter() - pocket_start
        rows.append(record)
        report = {
            "schema_version": 1,
            "method_id": "diffsbdd_public",
            "device": device,
            "checkpoint": args.checkpoint,
            "meta_pt": args.meta_pt,
            "test_set_root": args.test_set_root,
            "output_dir": str(output_dir),
            "seed": args.seed,
            "requested_pocket_count": total,
            "completed_pocket_count": sum(row["status"] in ("pass", "skipped_existing") for row in rows),
            "failed_pocket_count": sum(row["status"] not in ("pass", "skipped_existing") for row in rows),
            "elapsed_seconds": perf_counter() - start_all,
            "rows": rows,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"[{pocket_index + 1}/{total}] {record.get('pocket_id', 'unknown')} "
            f"{record['status']} {record['runtime_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
