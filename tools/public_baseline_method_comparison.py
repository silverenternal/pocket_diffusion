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
    "flow_matching": "Ours flow_matching",
    "pocket2mol_public": "Pocket2Mol",
    "targetdiff_public": "TargetDiff",
}
METHOD_SOURCES = {
    "diffsbdd_public": "generated_locally_from_public_checkpoint",
    "flow_matching": "native_rust_geometry_first_flow_matching",
    "pocket2mol_public": "official_targetdiff_sampling_results_google_drive",
    "targetdiff_public": "official_targetdiff_sampling_results_google_drive",
}
METHOD_STEPS = {
    "diffsbdd_public": 1000,
    "flow_matching": 20,
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
    parser.add_argument("--backend-aware-reranker-summary-json", default=None)
    parser.add_argument("--backend-aware-reranker-summary-md", default=None)
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


def metric_values_any(rows, names):
    values = []
    for row in rows:
        metrics = row.get("metrics", {})
        value = None
        for name in names:
            if name in metrics:
                value = metrics.get(name)
                break
        values.append(value)
    return values


def layer_role(layer):
    if layer in {"raw_flow", "raw_rollout", "no_repair"}:
        return "raw_model_native"
    if layer == "constrained_flow":
        return "constrained_sampling"
    return "postprocessing"


def method_summary(method_id, layer, rows):
    examples = {row.get("example_id") for row in rows}
    return {
        "method_id": method_id,
        "method_name": METHOD_NAMES.get(method_id, method_id),
        "layer": layer,
        "layer_role": layer_role(layer),
        "source": METHOD_SOURCES.get(method_id, "unknown"),
        "sampling_steps": METHOD_STEPS.get(method_id),
        "matched_candidates_per_pocket": 1,
        "pocket_count": len(examples),
        "candidate_count": len(rows),
        "rdkit_valid_fraction_mean": mean(metric_values(rows, "rdkit_valid_fraction")),
        "rdkit_coverage_fraction": mean(metric_values(rows, "drug_likeness_coverage_fraction")),
        "qed_mean": mean(metric_values(rows, "qed")),
        "sa_score_mean": mean(metric_values(rows, "sa_score")),
        "logp_mean": mean(metric_values(rows, "logp")),
        "tpsa_mean": mean(metric_values(rows, "tpsa")),
        "lipinski_violations_mean": mean(metric_values(rows, "lipinski_violations")),
        "scaffold_novelty_fraction_mean": mean(metric_values(rows, "scaffold_novelty_fraction")),
        "nearest_train_similarity_mean": mean(metric_values(rows, "nearest_train_similarity")),
        "pocket_contact_fraction_mean": mean(metric_values(rows, "pocket_contact_fraction")),
        "clash_fraction_mean": mean(metric_values(rows, "clash_fraction")),
        "centroid_offset_mean": mean(
            metric_values_any(rows, ["mean_centroid_offset", "centroid_offset"])
        ),
        "vina_score_mean": mean(metric_values(rows, "vina_score")),
        "vina_score_median": median(metric_values(rows, "vina_score")),
        "vina_coverage_fraction": mean(metric_values(rows, "vina_score_success_fraction")),
        "gnina_affinity_mean": mean(metric_values(rows, "gnina_affinity")),
        "gnina_cnn_score_mean": mean(metric_values(rows, "gnina_cnn_score")),
        "gnina_coverage_fraction": mean(metric_values(rows, "gnina_score_success_fraction")),
    }


def metric(row, name):
    value = row.get("metrics", {}).get(name)
    return float(value) if finite(value) else None


def metric_or(row, name, default):
    value = metric(row, name)
    return default if value is None else value


def proxy_selection_score(row):
    contact = metric_or(row, "pocket_contact_fraction", 0.0)
    strict_fit = metric_or(row, "strict_pocket_fit_score", contact)
    clash_free = 1.0 - max(0.0, min(1.0, metric_or(row, "clash_fraction", 1.0)))
    centroid = metric_or(row, "centroid_offset", metric_or(row, "mean_centroid_offset", 50.0))
    centroid_fit = 1.0 / (1.0 + max(centroid, 0.0))
    valid = metric_or(row, "rdkit_valid_fraction", 1.0)
    return (
        0.25 * contact
        + 0.20 * strict_fit
        + 0.20 * clash_free
        + 0.20 * centroid_fit
        + 0.15 * valid
    )


def backend_aware_posthoc_score(row):
    vina = metric(row, "vina_score")
    gnina = metric(row, "gnina_affinity")
    docking_values = [value for value in (vina, gnina) if value is not None]
    if not docking_values:
        return None
    docking = sum(docking_values) / len(docking_values)
    cnn = metric_or(row, "gnina_cnn_score", 0.0)
    qed = metric_or(row, "qed", 0.0)
    sa = metric_or(row, "sa_score", 10.0)
    clash = metric_or(row, "clash_fraction", 1.0)
    return docking - 2.0 * cnn - 3.0 * qed + 0.25 * sa + 5.0 * clash


def summarize_selected_rows(rows):
    return {
        "candidate_count": len(rows),
        "method_layer_counts": dict(
            sorted(
                {
                    f"{method}:{layer}": count
                    for (method, layer), count in defaultdict(int, {
                    }).items()
                }.items()
            )
        ),
        "vina_score_mean": mean(metric_values(rows, "vina_score")),
        "gnina_affinity_mean": mean(metric_values(rows, "gnina_affinity")),
        "gnina_cnn_score_mean": mean(metric_values(rows, "gnina_cnn_score")),
        "qed_mean": mean(metric_values(rows, "qed")),
        "sa_score_mean": mean(metric_values(rows, "sa_score")),
        "clash_fraction_mean": mean(metric_values(rows, "clash_fraction")),
        "pocket_contact_fraction_mean": mean(metric_values(rows, "pocket_contact_fraction")),
        "raw_model_native_fraction": (
            sum(1 for row in rows if layer_role(row.get("layer")) == "raw_model_native")
            / float(max(len(rows), 1))
        ),
    }


def selected_row_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[(row.get("method_id") or "unknown", row.get("layer") or "unknown")] += 1
    return dict(sorted((f"{method}:{layer}", count) for (method, layer), count in counts.items()))


def build_backend_aware_reranker_summary(rows, split_label):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("split_label") != split_label:
            continue
        grouped[(row.get("example_id") or "unknown", row.get("protein_id") or "unknown")].append(row)

    proxy_selected = []
    backend_selected = []
    examples = []
    for key, group in sorted(grouped.items()):
        proxy = max(group, key=lambda row: (proxy_selection_score(row), row.get("candidate_id") or ""))
        backend_candidates = [
            (backend_aware_posthoc_score(row), row)
            for row in group
            if backend_aware_posthoc_score(row) is not None
        ]
        if not backend_candidates:
            continue
        backend_candidates.sort(key=lambda item: (item[0], item[1].get("candidate_id") or ""))
        backend = backend_candidates[0][1]
        proxy_selected.append(proxy)
        backend_selected.append(backend)
        examples.append(
            {
                "example_id": key[0],
                "protein_id": key[1],
                "proxy_candidate_id": proxy.get("candidate_id"),
                "backend_aware_candidate_id": backend.get("candidate_id"),
                "proxy_layer": proxy.get("layer"),
                "backend_aware_layer": backend.get("layer"),
                "backend_aware_label": "backend_aware_posthoc",
                "delta_backend_minus_proxy": {
                    "vina_score": (
                        metric(backend, "vina_score") - metric(proxy, "vina_score")
                        if metric(backend, "vina_score") is not None and metric(proxy, "vina_score") is not None
                        else None
                    ),
                    "gnina_affinity": (
                        metric(backend, "gnina_affinity") - metric(proxy, "gnina_affinity")
                        if metric(backend, "gnina_affinity") is not None and metric(proxy, "gnina_affinity") is not None
                        else None
                    ),
                    "qed": (
                        metric(backend, "qed") - metric(proxy, "qed")
                        if metric(backend, "qed") is not None and metric(proxy, "qed") is not None
                        else None
                    ),
                    "sa_score": (
                        metric(backend, "sa_score") - metric(proxy, "sa_score")
                        if metric(backend, "sa_score") is not None and metric(proxy, "sa_score") is not None
                        else None
                    ),
                },
            }
        )

    proxy_summary = summarize_selected_rows(proxy_selected)
    backend_summary = summarize_selected_rows(backend_selected)
    proxy_summary["method_layer_counts"] = selected_row_counts(proxy_selected)
    backend_summary["method_layer_counts"] = selected_row_counts(backend_selected)
    return {
        "schema_version": 1,
        "artifact_name": "q3_docking_aware_reranker_summary",
        "split_label": split_label,
        "label": "backend_aware_posthoc",
        "claim_boundary": (
            "The backend-aware reranker is an offline posthoc baseline over already scored "
            "candidates. Raw model capability remains reported only from raw_flow/raw_rollout/no_repair rows."
        ),
        "candidate_pool_count": sum(len(group) for group in grouped.values()),
        "example_count": len(backend_selected),
        "proxy_reranker": proxy_summary,
        "backend_aware_posthoc": backend_summary,
        "mean_delta_backend_aware_minus_proxy": {
            name: mean(
                example["delta_backend_minus_proxy"].get(name)
                for example in examples
            )
            for name in ("vina_score", "gnina_affinity", "qed", "sa_score")
        },
        "raw_model_capability_source": "raw_flow/raw_rollout/no_repair only",
        "examples": examples,
    }


def write_backend_aware_reranker_md(path, payload):
    lines = [
        "# Q3 Docking-Aware Reranker Summary",
        "",
        payload["claim_boundary"],
        "",
        f"- label: `{payload['label']}`",
        f"- examples: {payload['example_count']}",
        f"- candidate_pool_count: {payload['candidate_pool_count']}",
        f"- raw_model_capability_source: `{payload['raw_model_capability_source']}`",
        "",
        "| Selector | Vina | GNINA | CNN | QED | SA | Clash | Contact | Raw-native fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in [
        ("proxy", payload["proxy_reranker"]),
        ("backend_aware_posthoc", payload["backend_aware_posthoc"]),
    ]:
        lines.append(
            f"| {name} | {fmt(row['vina_score_mean'])} | {fmt(row['gnina_affinity_mean'])} | "
            f"{fmt(row['gnina_cnn_score_mean'])} | {fmt(row['qed_mean'])} | "
            f"{fmt(row['sa_score_mean'])} | {fmt(row['clash_fraction_mean'])} | "
            f"{fmt(row['pocket_contact_fraction_mean'])} | {fmt(row['raw_model_native_fraction'])} |"
        )
    delta = payload["mean_delta_backend_aware_minus_proxy"]
    lines.extend(
        [
            "",
            "Mean backend-aware minus proxy deltas:",
            "",
            f"- Vina: {fmt(delta.get('vina_score'))}",
            f"- GNINA affinity: {fmt(delta.get('gnina_affinity'))}",
            f"- QED: {fmt(delta.get('qed'))}",
            f"- SA: {fmt(delta.get('sa_score'))}",
        ]
    )
    write_text(path, lines)


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
        "artifact_name": "public_baseline_method_comparison_summary",
        "status": args.status,
        "split_label": args.split_label,
        "claim_guardrail": (
            "This artifact supports a full 100-pocket public-baseline matched-budget comparison "
            "with unified RDKit, AutoDock Vina score_only, and GNINA score_only rescoring. "
            "Raw_rollout/raw_flow rows are native outputs; constrained_flow, repaired, and "
            "reranked rows are explicit constrained or postprocessing layers generated from "
            "those raw outputs. Backend score_only values are not experimental affinities."
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
    if args.backend_aware_reranker_summary_json or args.backend_aware_reranker_summary_md:
        reranker_summary = build_backend_aware_reranker_summary(rows, args.split_label)
        summary["backend_aware_reranker"] = {
            "label": reranker_summary["label"],
            "summary_json": args.backend_aware_reranker_summary_json,
            "summary_md": args.backend_aware_reranker_summary_md,
            "claim_boundary": reranker_summary["claim_boundary"],
        }
        if args.backend_aware_reranker_summary_json:
            write_json(args.backend_aware_reranker_summary_json, reranker_summary)
        if args.backend_aware_reranker_summary_md:
            write_backend_aware_reranker_md(args.backend_aware_reranker_summary_md, reranker_summary)
    write_json(args.summary_json, summary)

    lines = [
        "# Q2 Ours vs Public Baselines",
        "",
        f"status: `{args.status}`",
        "",
        "Scope: 100 official public-test pockets; 1 candidate per pocket per method per layer; unified RDKit, AutoDock Vina score_only, GNINA score_only, and pocket/contact scoring.",
        "",
    ]
    header = (
        "| method | layer | role | pockets | candidates | source | steps | Vina mean | Vina cov | "
        "GNINA affinity | GNINA CNN | QED | SA | LogP | TPSA | Lipinski | scaffold novelty | "
        "nearest train sim | pocket contact | clash | centroid offset |"
    )
    divider = "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for title, role_names in [
        ("Raw Native Rows", {"raw_model_native"}),
        ("Constrained And Postprocessed Rows", {"constrained_sampling", "postprocessing"}),
    ]:
        lines.extend(["", f"## {title}", "", header, divider])
        for method in [row for row in methods if row["layer_role"] in role_names]:
            lines.append(
                f"| {method['method_name']} | {method['layer']} | {method['layer_role']} | "
                f"{method['pocket_count']} | {method['candidate_count']} | "
                f"{method['source']} | {method['sampling_steps']} | "
                f"{fmt(method['vina_score_mean'])} | {fmt(method['vina_coverage_fraction'])} | "
                f"{fmt(method['gnina_affinity_mean'])} | {fmt(method['gnina_cnn_score_mean'])} | "
                f"{fmt(method['qed_mean'])} | {fmt(method['sa_score_mean'])} | "
                f"{fmt(method['logp_mean'])} | {fmt(method['tpsa_mean'])} | "
                f"{fmt(method['lipinski_violations_mean'])} | "
                f"{fmt(method['scaffold_novelty_fraction_mean'])} | "
                f"{fmt(method['nearest_train_similarity_mean'])} | "
                f"{fmt(method['pocket_contact_fraction_mean'])} | "
                f"{fmt(method['clash_fraction_mean'])} | {fmt(method['centroid_offset_mean'])} |"
            )
    lines.extend(
        [
            "",
            "Guardrail: Vina and GNINA values are score_only backend outputs, not experimental binding affinities. Raw native rows are separated from constrained sampling, repaired, and reranked rows.",
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
