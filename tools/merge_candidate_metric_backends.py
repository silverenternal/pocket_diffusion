#!/usr/bin/env python3
"""Merge proxy/RDKit candidate metrics with real docking backend metrics."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


VINA_PREFIXES = ("vina_",)
GNINA_PREFIXES = ("gnina_",)
RDKIT_KEYS = {
    "qed",
    "raw_qed",
    "sa_score",
    "raw_sa",
    "logp",
    "tpsa",
    "molecular_weight",
    "hbd",
    "hba",
    "rotatable_bonds",
    "lipinski_violations",
    "rdkit_sanitized",
    "rdkit_sanitized_fraction",
    "rdkit_valid",
    "rdkit_valid_fraction",
    "rdkit_unique_smiles",
    "rdkit_unique_smiles_fraction",
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Merge candidate metric JSONL backend outputs.")
    parser.add_argument("--base-jsonl", required=True, help="Base candidate_metrics JSONL")
    parser.add_argument("--vina-jsonl", action="append", default=[], help="Vina candidate metric JSONL")
    parser.add_argument("--gnina-jsonl", action="append", default=[], help="GNINA candidate metric JSONL")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/pdbbindpp_real_backends",
        help="Directory for candidate_metrics_<split>.jsonl outputs.",
    )
    parser.add_argument(
        "--coverage-json",
        default="configs/candidate_metric_coverage.json",
        help="Machine-readable backend coverage report.",
    )
    return parser.parse_args(argv[1:])


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            row["_source_file"] = path
            rows.append(row)
    return rows


def index_backend(paths, backend_name):
    by_id = {}
    collisions = []
    for path in paths:
        for row in load_jsonl(path):
            cid = row.get("candidate_id")
            if not cid:
                collisions.append({"path": path, "reason": "missing_candidate_id"})
                continue
            if cid in by_id:
                collisions.append({"candidate_id": cid, "reason": f"duplicate_{backend_name}_row"})
                continue
            by_id[cid] = row
    return by_id, collisions


def metric_provenance(metric_name):
    if metric_name.startswith(VINA_PREFIXES):
        return "vina"
    if metric_name.startswith(GNINA_PREFIXES):
        return "gnina"
    if metric_name in RDKIT_KEYS:
        return "rdkit"
    if metric_name == "docking_like_score":
        return "heuristic_proxy"
    if any(token in metric_name for token in ("contact", "clash", "centroid", "pocket", "hydrogen_bond", "hydrophobic")):
        return "geometry_or_interaction_proxy"
    return "base_candidate_metric"


def finite_metric_count(row, keys):
    metrics = row.get("metrics", {})
    return sum(
        1
        for key in keys
        if isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
    )


def merge_rows(base_rows, vina_by_id, gnina_by_id):
    by_split = defaultdict(list)
    collisions = []
    seen = set()
    for row in base_rows:
        cid = row.get("candidate_id")
        if not cid:
            collisions.append({"reason": "base_row_missing_candidate_id"})
            continue
        if cid in seen:
            collisions.append({"candidate_id": cid, "reason": "duplicate_base_row"})
            continue
        seen.add(cid)
        merged = {key: value for key, value in row.items() if not key.startswith("_")}
        metrics = dict(row.get("metrics", {}))
        backend_statuses = dict(row.get("backend_statuses", {}))
        source_artifacts = list(row.get("source_artifacts", []))
        metric_sources = {name: metric_provenance(name) for name in metrics}

        vina = vina_by_id.get(cid)
        if vina:
            metrics.update(vina.get("metrics", {}))
            backend_statuses["vina"] = "metrics_available"
            source_artifacts.append(vina["_source_file"])
        else:
            backend_statuses["vina"] = "missing"
        gnina = gnina_by_id.get(cid)
        if gnina:
            metrics.update(gnina.get("metrics", {}))
            backend_statuses["gnina"] = "metrics_available"
            source_artifacts.append(gnina["_source_file"])
        else:
            backend_statuses["gnina"] = "missing"

        for name in metrics:
            metric_sources[name] = metric_provenance(name)
        merged["metrics"] = metrics
        merged["backend_statuses"] = backend_statuses
        merged["metric_provenance"] = metric_sources
        merged["source_artifacts"] = sorted(set(source_artifacts))
        split = merged.get("split_label") or "unknown"
        by_split[split].append(merged)
    return by_split, collisions


def write_outputs(by_split, output_dir):
    output_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in sorted(by_split.items()):
        path = output_dir / f"candidate_metrics_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        output_paths[split] = str(path)
    return output_paths


def coverage_report(by_split, output_paths, collisions, vina_by_id, gnina_by_id):
    split_reports = {}
    for split, rows in sorted(by_split.items()):
        total = max(len(rows), 1)
        counts = Counter()
        for row in rows:
            metrics = row.get("metrics", {})
            counts["vina"] += int("vina_score" in metrics)
            counts["gnina_affinity"] += int("gnina_affinity" in metrics)
            counts["gnina_cnn_score"] += int("gnina_cnn_score" in metrics)
            counts["rdkit"] += int(finite_metric_count(row, ["qed", "sa_score", "logp"]) >= 1)
            counts["proxy_docking_like"] += int("docking_like_score" in metrics)
        split_reports[split] = {
            "candidate_count": len(rows),
            "vina_score_coverage_fraction": counts["vina"] / float(total),
            "gnina_affinity_coverage_fraction": counts["gnina_affinity"] / float(total),
            "gnina_cnn_score_coverage_fraction": counts["gnina_cnn_score"] / float(total),
            "rdkit_metric_coverage_fraction": counts["rdkit"] / float(total),
            "proxy_docking_like_fraction": counts["proxy_docking_like"] / float(total),
            "output_path": output_paths[split],
        }
    return {
        "schema_version": 1,
        "artifact_name": "candidate_metric_coverage",
        "split_reports": split_reports,
        "backend_row_counts": {
            "vina": len(vina_by_id),
            "gnina": len(gnina_by_id),
        },
        "collision_count": len(collisions),
        "collisions": collisions,
    }


def main(argv):
    args = parse_args(argv)
    base_rows = load_jsonl(args.base_jsonl)
    vina_by_id, vina_collisions = index_backend(args.vina_jsonl, "vina")
    gnina_by_id, gnina_collisions = index_backend(args.gnina_jsonl, "gnina")
    by_split, merge_collisions = merge_rows(base_rows, vina_by_id, gnina_by_id)
    collisions = vina_collisions + gnina_collisions + merge_collisions
    output_paths = write_outputs(by_split, Path(args.output_dir))
    report = coverage_report(by_split, output_paths, collisions, vina_by_id, gnina_by_id)
    coverage_path = Path(args.coverage_json)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if collisions else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
