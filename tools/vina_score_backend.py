#!/usr/bin/env python3
"""Optional AutoDock Vina score-only command adapter.

The adapter accepts the repository candidate JSON contract and emits numeric
schema-versioned metrics. It intentionally degrades to explicit availability or
input-completeness metrics when Vina, receptor PDBQT, or ligand PDBQT payloads
are absent, so smoke runs remain reproducible without making Vina mandatory.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


AFFINITY_RE = re.compile(r"Affinity:\s*([-+]?\d+(?:\.\d+)?)")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Score candidates with AutoDock Vina when available.")
    parser.add_argument("--vina-executable", default=os.environ.get("VINA_EXECUTABLE", "vina"))
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    return parser.parse_args(argv[1:])


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def parse_affinity(output):
    match = AFFINITY_RE.search(output)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def looks_like_pdbqt_payload(payload):
    if not payload:
        return False
    markers = ("ROOT", "ENDROOT", "TORSDOF", "ATOM", "HETATM")
    return any(marker in payload for marker in markers)


def ligand_input(candidate, workdir):
    source_ligand = candidate.get("source_ligand_path")
    if source_ligand:
        ligand_source = Path(source_ligand)
        if ligand_source.is_file() and ligand_source.suffix.lower() == ".pdbqt":
            return ligand_source, None
    ligand_payload = candidate.get("molecular_representation")
    if not ligand_payload:
        return None, "missing_ligand_pdbqt_input"
    ligand_path = Path(workdir) / f"{candidate.get('example_id', 'candidate')}.pdbqt"
    ligand_path.write_text(ligand_payload, encoding="utf-8")
    return ligand_path, None


def score_candidate(vina_executable, candidate, workdir):
    receptor = candidate.get("source_pocket_path")
    if not receptor:
        return None, "missing_receptor_path"

    receptor_path = Path(receptor)
    if not receptor_path.is_file():
        return None, "receptor_path_not_found"
    if receptor_path.suffix.lower() != ".pdbqt":
        return None, "receptor_is_not_pdbqt"

    ligand_path, ligand_reason = ligand_input(candidate, workdir)
    if ligand_reason is not None:
        return None, ligand_reason

    completed = subprocess.run(
        [
            vina_executable,
            "--score_only",
            "--receptor",
            str(receptor_path),
            "--ligand",
            str(ligand_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return None, "vina_command_failed"
    affinity = parse_affinity(completed.stdout + "\n" + completed.stderr)
    if affinity is None:
        return None, "vina_affinity_parse_failed"
    return affinity, None


def main(argv):
    args = parse_args(argv)
    candidates = load_candidates(args.input_json)
    total = max(len(candidates), 1)
    counts = Counter()

    for candidate in candidates:
        receptor = candidate.get("source_pocket_path")
        source_ligand = candidate.get("source_ligand_path")
        ligand_payload = candidate.get("molecular_representation")
        receptor_path = Path(receptor) if receptor else None
        ligand_path = Path(source_ligand) if source_ligand else None
        counts["candidate_count"] += 1
        counts["receptor_path_present"] += int(bool(receptor))
        counts["receptor_path_exists"] += int(bool(receptor_path and receptor_path.is_file()))
        counts["receptor_is_pdbqt"] += int(
            bool(receptor_path and receptor_path.suffix.lower() == ".pdbqt")
        )
        counts["ligand_source_path_present"] += int(bool(source_ligand))
        counts["ligand_source_pdbqt_exists"] += int(
            bool(ligand_path and ligand_path.is_file() and ligand_path.suffix.lower() == ".pdbqt")
        )
        counts["ligand_payload_present"] += int(bool(ligand_payload))
        counts["ligand_payload_looks_like_pdbqt"] += int(looks_like_pdbqt_payload(ligand_payload))
        counts["candidate_with_complete_vina_inputs"] += int(
            bool(
                receptor_path
                and receptor_path.is_file()
                and receptor_path.suffix.lower() == ".pdbqt"
                and (
                    (ligand_path and ligand_path.is_file() and ligand_path.suffix.lower() == ".pdbqt")
                    or bool(ligand_payload)
                )
            )
        )

    if shutil.which(args.vina_executable) is None:
        write_metrics(
            args.output_json,
            {
                "schema_version": 1.0,
                "vina_available": 0.0,
                "backend_examples_scored": 0.0,
                "backend_missing_structure_fraction": 1.0,
                "candidate_count": float(counts["candidate_count"]),
                "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
                "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
                "receptor_pdbqt_fraction": counts["receptor_is_pdbqt"] / float(total),
                "ligand_source_pdbqt_fraction": counts["ligand_source_pdbqt_exists"] / float(total),
                "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
                "ligand_payload_pdbqt_like_fraction": counts["ligand_payload_looks_like_pdbqt"] / float(total),
                "candidate_with_complete_vina_inputs_fraction": counts["candidate_with_complete_vina_inputs"] / float(total),
            },
        )
        return

    affinities = []
    failures = 0
    failure_reasons = Counter()
    with tempfile.TemporaryDirectory(prefix="pocket_diffusion_vina_") as workdir:
        for candidate in candidates:
            affinity, reason = score_candidate(args.vina_executable, candidate, workdir)
            if reason is not None:
                failures += 1
                failure_reasons[reason] += 1
                continue
            affinities.append(affinity)

    if not affinities:
        payload = {
            "schema_version": 1.0,
            "vina_available": 1.0,
            "backend_examples_scored": 0.0,
            "backend_missing_structure_fraction": failures / float(total),
            "vina_score_success_fraction": 0.0,
            "candidate_count": float(counts["candidate_count"]),
            "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
            "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
            "receptor_pdbqt_fraction": counts["receptor_is_pdbqt"] / float(total),
            "ligand_source_pdbqt_fraction": counts["ligand_source_pdbqt_exists"] / float(total),
            "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
            "ligand_payload_pdbqt_like_fraction": counts["ligand_payload_looks_like_pdbqt"] / float(total),
            "candidate_with_complete_vina_inputs_fraction": counts["candidate_with_complete_vina_inputs"] / float(total),
        }
        for reason, count in failure_reasons.items():
            payload[f"vina_{reason}_fraction"] = count / float(total)
        write_metrics(
            args.output_json,
            payload,
        )
        return

    payload = {
        "schema_version": 1.0,
        "vina_available": 1.0,
        "backend_examples_scored": float(len(affinities)),
        "backend_missing_structure_fraction": failures / float(total),
        "vina_score_success_fraction": len(affinities) / float(total),
        "vina_mean_affinity_kcal_mol": sum(affinities) / float(len(affinities)),
        "vina_best_affinity_kcal_mol": min(affinities),
        "candidate_count": float(counts["candidate_count"]),
        "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
        "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
        "receptor_pdbqt_fraction": counts["receptor_is_pdbqt"] / float(total),
        "ligand_source_pdbqt_fraction": counts["ligand_source_pdbqt_exists"] / float(total),
        "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
        "ligand_payload_pdbqt_like_fraction": counts["ligand_payload_looks_like_pdbqt"] / float(total),
        "candidate_with_complete_vina_inputs_fraction": counts["candidate_with_complete_vina_inputs"] / float(total),
    }
    for reason, count in failure_reasons.items():
        payload[f"vina_{reason}_fraction"] = count / float(total)
    write_metrics(
        args.output_json,
        payload,
    )


if __name__ == "__main__":
    main(sys.argv)
