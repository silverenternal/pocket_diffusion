#!/usr/bin/env python3
"""Optional GNINA score-only command adapter."""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


SCORE_RES = {
    "gnina_affinity": re.compile(r"Affinity:\s*([-+]?\d+(?:\.\d+)?)"),
    "gnina_cnn_score": re.compile(r"CNNscore:\s*([-+]?\d+(?:\.\d+)?)"),
    "gnina_cnn_affinity": re.compile(r"CNNaffinity:\s*([-+]?\d+(?:\.\d+)?)"),
    "gnina_cnn_variance": re.compile(r"CNNvariance:\s*([-+]?\d+(?:\.\d+)?)"),
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Score candidates with GNINA when available.")
    parser.add_argument("--gnina-executable", default=os.environ.get("GNINA_EXECUTABLE", "gnina"))
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    return parser.parse_args(argv[1:])


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def structured_payload(aggregate_metrics, candidate_metrics):
    return {
        "schema_version": 1.0,
        "aggregate_metrics": aggregate_metrics,
        "candidate_metrics": candidate_metrics,
    }


def attach_contract_counts(payload, input_count, scored_count, failure_count):
    payload["input_count"] = float(input_count)
    payload["scored_count"] = float(scored_count)
    payload["failure_count"] = float(failure_count)
    payload["gnina_score_coverage_fraction"] = float(scored_count) / float(max(input_count, 1))
    return payload


def attach_runtime(payload, started_at, input_count):
    elapsed = time.perf_counter() - started_at
    payload["backend_wall_clock_seconds"] = elapsed
    payload["backend_candidates_per_second"] = (
        float(input_count) / elapsed if elapsed > 0.0 else None
    )
    payload["backend_runtime_status"] = "measured_by_backend_adapter"
    return payload


def parse_scores(output):
    scores = {}
    for key, regex in SCORE_RES.items():
        match = regex.search(output)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if math.isfinite(value):
            scores[key] = value
    return scores


def ligand_input(candidate, workdir):
    source_ligand = candidate.get("source_ligand_path")
    if source_ligand:
        ligand_source = Path(source_ligand)
        if ligand_source.is_file():
            return ligand_source, None
    payload = candidate.get("molecular_representation")
    if not payload:
        return None, "missing_ligand_input"
    ligand_path = Path(workdir) / f"{candidate.get('example_id', 'candidate')}.pdbqt"
    ligand_path.write_text(payload, encoding="utf-8")
    return ligand_path, None


def score_candidate(gnina_executable, candidate, workdir):
    receptor = candidate.get("source_pocket_path")
    if not receptor:
        return None, "missing_receptor_path"
    receptor_path = Path(receptor)
    if not receptor_path.is_file():
        return None, "receptor_path_not_found"
    ligand_path, ligand_reason = ligand_input(candidate, workdir)
    if ligand_reason is not None:
        return None, ligand_reason
    completed = subprocess.run(
        [
            gnina_executable,
            "--score_only",
            "-r",
            str(receptor_path),
            "-l",
            str(ligand_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return None, "gnina_command_failed"
    scores = parse_scores(completed.stdout + "\n" + completed.stderr)
    if not scores:
        return None, "gnina_score_parse_failed"
    return scores, None


def main(argv):
    started_at = time.perf_counter()
    args = parse_args(argv)
    candidates = load_candidates(args.input_json)
    total = max(len(candidates), 1)
    counts = Counter()
    rows = []
    for candidate in candidates:
        receptor = candidate.get("source_pocket_path")
        ligand = candidate.get("source_ligand_path")
        counts["candidate_count"] += 1
        counts["receptor_path_present"] += int(bool(receptor))
        counts["receptor_path_exists"] += int(bool(receptor and Path(receptor).is_file()))
        counts["ligand_source_path_present"] += int(bool(ligand))
        counts["ligand_source_path_exists"] += int(bool(ligand and Path(ligand).is_file()))
        counts["ligand_payload_present"] += int(bool(candidate.get("molecular_representation")))

    if shutil.which(args.gnina_executable) is None:
        payload = attach_contract_counts(
            {
                "schema_version": 1.0,
                "gnina_available": 0.0,
                "backend_examples_scored": 0.0,
                "backend_missing_structure_fraction": 1.0,
                "candidate_count": float(counts["candidate_count"]),
                "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
                "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
                "ligand_source_path_present_fraction": counts["ligand_source_path_present"] / float(total),
                "ligand_source_path_exists_fraction": counts["ligand_source_path_exists"] / float(total),
                "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
            },
            counts["candidate_count"],
            0,
            counts["candidate_count"],
        )
        attach_runtime(payload, started_at, counts["candidate_count"])
        write_metrics(args.output_json, structured_payload(payload, []))
        return

    affinities = []
    cnn_scores = []
    cnn_affinities = []
    failures = 0
    failure_reasons = Counter()
    with tempfile.TemporaryDirectory(prefix="pocket_diffusion_gnina_") as workdir:
        for index, candidate in enumerate(candidates):
            scores, reason = score_candidate(args.gnina_executable, candidate, workdir)
            row = {"gnina_score_success_fraction": 0.0, "backend_missing_structure_fraction": 0.0}
            if reason is not None:
                failures += 1
                failure_reasons[reason] += 1
                row["backend_missing_structure_fraction"] = 1.0
                row["gnina_failure_reason"] = reason
            else:
                row["gnina_score_success_fraction"] = 1.0
                row.update(scores)
                if "gnina_affinity" in scores:
                    affinities.append(scores["gnina_affinity"])
                    row["gnina_score"] = scores["gnina_affinity"]
                if "gnina_cnn_score" in scores:
                    cnn_scores.append(scores["gnina_cnn_score"])
                if "gnina_cnn_affinity" in scores:
                    cnn_affinities.append(scores["gnina_cnn_affinity"])
            rows.append(
                {
                    "candidate_id": candidate.get("candidate_id") or f"unknown:{index}",
                    "example_id": candidate.get("example_id") or "unknown",
                    "protein_id": candidate.get("protein_id") or "unknown",
                    "split_label": candidate.get("split_label") or "unknown",
                    "method_id": candidate.get("method_id") or "unknown",
                    "layer": candidate.get("layer") or "unknown",
                    "metrics": row,
                }
            )

    payload = attach_contract_counts(
        {
            "schema_version": 1.0,
            "gnina_available": 1.0,
            "backend_examples_scored": float(len(affinities) if affinities else len(cnn_scores)),
            "backend_missing_structure_fraction": failures / float(total),
            "gnina_score_success_fraction": (total - failures) / float(total),
            "candidate_count": float(counts["candidate_count"]),
            "receptor_path_present_fraction": counts["receptor_path_present"] / float(total),
            "receptor_path_exists_fraction": counts["receptor_path_exists"] / float(total),
            "ligand_source_path_present_fraction": counts["ligand_source_path_present"] / float(total),
            "ligand_source_path_exists_fraction": counts["ligand_source_path_exists"] / float(total),
            "ligand_payload_present_fraction": counts["ligand_payload_present"] / float(total),
        },
        counts["candidate_count"],
        total - failures,
        failures,
    )
    if affinities:
        payload["gnina_affinity_mean"] = sum(affinities) / float(len(affinities))
        payload["gnina_affinity_median"] = statistics.median(affinities)
        payload["gnina_affinity_best"] = min(affinities)
        payload["gnina_affinity_std"] = statistics.pstdev(affinities) if len(affinities) > 1 else 0.0
        payload["gnina_score_mean"] = payload["gnina_affinity_mean"]
        payload["gnina_score_best"] = payload["gnina_affinity_best"]
    if cnn_scores:
        payload["gnina_cnn_score_mean"] = sum(cnn_scores) / float(len(cnn_scores))
        payload["gnina_cnn_score_median"] = statistics.median(cnn_scores)
        payload["gnina_cnn_score_best"] = max(cnn_scores)
        payload["gnina_cnn_score_std"] = statistics.pstdev(cnn_scores) if len(cnn_scores) > 1 else 0.0
    if cnn_affinities:
        payload["gnina_cnn_affinity_mean"] = sum(cnn_affinities) / float(len(cnn_affinities))
        payload["gnina_cnn_affinity_median"] = statistics.median(cnn_affinities)
        payload["gnina_cnn_affinity_best"] = min(cnn_affinities)
        payload["gnina_cnn_affinity_std"] = statistics.pstdev(cnn_affinities) if len(cnn_affinities) > 1 else 0.0
    for reason, count in failure_reasons.items():
        payload[f"gnina_{reason}_fraction"] = count / float(total)
    payload["failure_reasons"] = dict(sorted(failure_reasons.items()))
    attach_runtime(payload, started_at, counts["candidate_count"])
    write_metrics(args.output_json, structured_payload(payload, rows))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
