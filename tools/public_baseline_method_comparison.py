#!/usr/bin/env python3
"""Summarize public-baseline matched-budget metrics from candidate JSONL."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHOD_NAMES = {
    "diffsbdd_public": "DiffSBDD",
    "pocket2mol_public": "Pocket2Mol",
    "targetdiff_public": "TargetDiff",
}
METHOD_SOURCES = {
    "diffsbdd_public": "generated_locally_from_public_checkpoint",
    "pocket2mol_public": "official_targetdiff_sampling_results_google_drive",
    "targetdiff_public": "official_targetdiff_sampling_results_google_drive",
}
METHOD_STEPS = {
    "diffsbdd_public": 1000,
    "pocket2mol_public": "official_public_precomputed_meta",
    "targetdiff_public": "official_public_precomputed_meta",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--diffsbdd-generation-report", default=None)
    parser.add_argument("--docking-input-report", default=None)
    parser.add_argument("--postprocess-report", default=None)
    parser.add_argument("--vina-report-json", default=None)
    parser.add_argument("--gnina-report-json", default=None)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--table-md", required=True)
    parser.add_argument("--runtime-md", required=True)
    parser.add_argument("--split-label", required=True)
    parser.add_argument("--status", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values):
    vals = [float(value) for value in values if finite(value)]
    return sum(vals) / len(vals) if vals else None


def median(values):
    vals = [float(value) for value in values if finite(value)]
    return statistics.median(vals) if vals else None


def metric_values(rows, name):
    return [row.get("metrics", {}).get(name) for row in rows]


def method_summary(method_id, layer, rows):
    examples = {row.get("example_id") for row in rows}
    return {
        "method_id": method_id,
        "method_name": METHOD_NAMES.get(method_id, method_id),
        "layer": layer,
        "source": METHOD_SOURCES.get(method_id, "unknown"),
        "sampling_steps": METHOD_STEPS.get(method_id),
        "matched_candidates_per_pocket": 1,
        "pocket_count": len(examples),
        "candidate_count": len(rows),
        "rdkit_valid_fraction_mean": mean(metric_values(rows, "rdkit_valid_fraction")),
        "rdkit_coverage_fraction": mean(metric_values(rows, "drug_likeness_coverage_fraction")),
        "qed_mean": mean(metric_values(rows, "qed")),
        "sa_score_mean": mean(metric_values(rows, "sa_score")),
        "pocket_contact_fraction_mean": mean(metric_values(rows, "pocket_contact_fraction")),
        "vina_score_mean": mean(metric_values(rows, "vina_score")),
        "vina_score_median": median(metric_values(rows, "vina_score")),
        "vina_coverage_fraction": mean(metric_values(rows, "vina_score_success_fraction")),
        "gnina_affinity_mean": mean(metric_values(rows, "gnina_affinity")),
        "gnina_cnn_score_mean": mean(metric_values(rows, "gnina_cnn_score")),
        "gnina_coverage_fraction": mean(metric_values(rows, "gnina_score_success_fraction")),
    }


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}g}"
    return str(value)


def write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path, lines):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def backend_runtime(report):
    if not report:
        return None
    aggregate = report.get("aggregate_metrics", {})
    seconds = aggregate.get("backend_wall_clock_seconds")
    if seconds is None:
        return None
    return {
        "seconds": seconds,
        "candidates_per_second": aggregate.get("backend_candidates_per_second"),
        "status": aggregate.get("backend_runtime_status") or "measured",
        "candidate_count": aggregate.get("candidate_count"),
        "scored_count": aggregate.get("scored_count"),
    }


def combined_backend_runtime(vina_report, gnina_report):
    vina = backend_runtime(vina_report)
    gnina = backend_runtime(gnina_report)
    measured = [item for item in (vina, gnina) if item]
    if not measured:
        return {
            "status": "not_measured_in_backend_scripts",
            "total_seconds": None,
            "backends": {
                "vina": vina,
                "gnina": gnina,
            },
        }
    total_seconds = sum(float(item["seconds"]) for item in measured)
    candidate_count = max(float(item.get("candidate_count") or 0.0) for item in measured)
    return {
        "status": "measured_by_backend_adapters",
        "total_seconds": total_seconds,
        "candidate_count": candidate_count,
        "candidates_per_second": candidate_count / total_seconds if total_seconds > 0.0 else None,
        "backends": {
            "vina": vina,
            "gnina": gnina,
        },
    }


def runtime_rows(diffsbdd_report, backend_runtime_summary):
    rows = [
        {
            "method_id": "diffsbdd_public",
            "generation_runtime_seconds": diffsbdd_report.get("elapsed_seconds") if diffsbdd_report else None,
            "generation_runtime_status": "measured_local_public_checkpoint" if diffsbdd_report else "missing",
            "backend_scoring_runtime_seconds": backend_runtime_summary.get("total_seconds"),
            "backend_scoring_runtime_status": backend_runtime_summary.get("status"),
        },
        {
            "method_id": "pocket2mol_public",
            "generation_runtime_seconds": None,
            "generation_runtime_status": "not_available_official_precomputed_meta",
            "backend_scoring_runtime_seconds": backend_runtime_summary.get("total_seconds"),
            "backend_scoring_runtime_status": backend_runtime_summary.get("status"),
        },
        {
            "method_id": "targetdiff_public",
            "generation_runtime_seconds": None,
            "generation_runtime_status": "not_available_official_precomputed_meta",
            "backend_scoring_runtime_seconds": backend_runtime_summary.get("total_seconds"),
            "backend_scoring_runtime_status": backend_runtime_summary.get("status"),
        },
    ]
    return rows


def unique_ordered(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def main():
    args = parse_args()
    rows = load_jsonl(args.candidate_metrics)
    coverage = load_json(args.coverage_json)
    diff_report = load_json(args.diffsbdd_generation_report) if args.diffsbdd_generation_report else None
    docking_report = load_json(args.docking_input_report) if args.docking_input_report else None
    postprocess_report = load_json(args.postprocess_report) if args.postprocess_report else None
    vina_report = load_json(args.vina_report_json) if args.vina_report_json else None
    gnina_report = load_json(args.gnina_report_json) if args.gnina_report_json else None
    backend_runtime_summary = combined_backend_runtime(vina_report, gnina_report)

    by_method = defaultdict(list)
    observed_layers = []
    for row in rows:
        if row.get("split_label") == args.split_label:
            key = (row.get("method_id"), row.get("layer"))
            by_method[key].append(row)
            if row.get("layer") not in observed_layers:
                observed_layers.append(row.get("layer"))

    methods = [
        method_summary(method_id, layer, by_method[(method_id, layer)])
        for method_id, layer in sorted(by_method)
    ]
    runtimes = runtime_rows(diff_report, backend_runtime_summary)
    summary = {
        "schema_version": 1,
        "artifact_name": "q1_method_comparison_summary",
        "status": args.status,
        "split_label": args.split_label,
        "claim_guardrail": (
            "This artifact supports a full 100-pocket public-baseline matched-budget comparison "
            "with unified RDKit, Vina, and GNINA rescoring. Raw_rollout rows are native public "
            "baseline outputs; repaired and reranked rows are explicit shared postprocessing "
            "layers generated from those raw outputs."
        ),
        "scope": {
            "public_meta_pockets_available": 100,
            "rescored_public_pockets": 100,
            "matched_candidates_per_pocket": 1,
            "candidate_count_total": len(rows),
            "layers": observed_layers,
            "methods": unique_ordered(
                [METHOD_NAMES.get(method.get("method_id"), method.get("method_id")) for method in methods]
            ),
            "unified_rescoring": [
                "RDKit",
                "AutoDock Vina score_only",
                "GNINA score_only",
                "pocket/contact proxies",
            ],
            "public_baseline_layer_attribution": (
                "raw_rollout is native baseline evidence; repaired/reranked are postprocessing evidence."
            ),
        },
        "input_artifacts": {
            "merged_candidate_metrics": args.candidate_metrics,
            "coverage_report": args.coverage_json,
            "docking_input_report": args.docking_input_report,
            "diffsbdd_generation_report": args.diffsbdd_generation_report,
            "vina_report": args.vina_report_json,
            "gnina_report": args.gnina_report_json,
            "postprocess_report": args.postprocess_report,
        },
        "coverage": coverage.get("split_reports", {}).get(args.split_label),
        "methods": methods,
        "runtime": {
            "rows": runtimes,
            "backend_runtime_status": backend_runtime_summary.get("status"),
            "backend_runtime": backend_runtime_summary,
            "docking_input_prepared_count": (docking_report or {}).get("counts", {}).get("prepared_count"),
            "postprocess_totals": (postprocess_report or {}).get("totals"),
        },
    }
    write_json(args.summary_json, summary)

    lines = [
        "# Q1 Public Baseline Method Comparison",
        "",
        f"status: `{args.status}`",
        "",
        "Scope: 100 official public-test pockets; 1 candidate per pocket per method per layer; unified RDKit, Vina, GNINA, and pocket/contact scoring.",
        "",
        "| method | layer | pockets | candidates | source | steps | RDKit valid | QED | Vina mean | Vina coverage | GNINA affinity | GNINA CNN score |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            f"| {method['method_name']} | {method['layer']} | {method['pocket_count']} | {method['candidate_count']} | "
            f"{method['source']} | {method['sampling_steps']} | "
            f"{fmt(method['rdkit_valid_fraction_mean'])} | {fmt(method['qed_mean'])} | "
            f"{fmt(method['vina_score_mean'])} | {fmt(method['vina_coverage_fraction'])} | "
            f"{fmt(method['gnina_affinity_mean'])} | "
            f"{fmt(method['gnina_cnn_score_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Guardrail: raw_rollout rows are native public-baseline outputs. Repaired and reranked rows are generated by the shared deterministic postprocessing pipeline and must be interpreted as postprocessing evidence.",
        ]
    )
    write_text(args.table_md, lines)

    runtime_lines = [
        "# Q1 Runtime Efficiency Table",
        "",
        f"status: `{args.status}`",
        "",
        "| method | candidates | generation runtime seconds | generation runtime status | backend scoring runtime seconds | backend scoring runtime status |",
        "|---|---:|---:|---|---:|---|",
    ]
    by_method_count = defaultdict(int)
    for method in methods:
        by_method_count[method["method_id"]] += method["candidate_count"]
    for row in runtimes:
        runtime_lines.append(
            f"| {METHOD_NAMES.get(row['method_id'], row['method_id'])} | {by_method_count.get(row['method_id'], 0)} | "
            f"{fmt(row['generation_runtime_seconds'])} | {row['generation_runtime_status']} | "
            f"{fmt(row.get('backend_scoring_runtime_seconds'))} | {row.get('backend_scoring_runtime_status')} |"
        )
    runtime_lines.extend(
        [
            "",
            "Backend scoring runtime is measured once for the shared layered Vina+GNINA rescoring batch; the same measured backend batch supports all method rows.",
        ]
    )
    write_text(args.runtime_md, runtime_lines)
    print(f"public baseline summary written: {args.summary_json}")


if __name__ == "__main__":
    main()
