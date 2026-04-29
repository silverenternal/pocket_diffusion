#!/usr/bin/env python3
"""Build a guarded Q3 model-improvement leaderboard from scored artifacts."""

import argparse
import json
from pathlib import Path


METRIC_FIELDS = [
    "vina_score",
    "gnina_affinity",
    "gnina_cnn_score",
    "qed",
    "sa_score",
    "clash_fraction",
    "pocket_contact_fraction",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-summary", default="configs/q2_ours_vs_public_baselines_summary.json")
    parser.add_argument("--bond-summary", default="configs/q3_bond_refinement_summary.json")
    parser.add_argument("--repair-summary", default="configs/q3_gated_repair_ablation_summary.json")
    parser.add_argument("--reranker-summary", default="configs/q3_docking_aware_reranker_summary.json")
    parser.add_argument("--local-pocket-summary", default="configs/q3_local_pocket_flow_summary.json")
    parser.add_argument(
        "--pairwise-summary", default="configs/q3_pairwise_geometry_ablation_summary.json"
    )
    parser.add_argument("--output-json", default="configs/q3_model_improvement_leaderboard.json")
    parser.add_argument("--output-md", default="docs/q3_model_improvement_leaderboard.md")
    return parser.parse_args(argv)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(row, name):
    if f"{name}_mean" in row:
        return row.get(f"{name}_mean")
    return row.get(name)


def q2_row(row):
    return {
        "method_id": row.get("method_id"),
        "method_name": row.get("method_name"),
        "layer": row.get("layer"),
        "layer_role": row.get("layer_role"),
        "candidate_count": row.get("candidate_count"),
        "coverage": {
            "vina_score": row.get("vina_coverage_fraction"),
            "gnina_affinity": row.get("gnina_coverage_fraction"),
            "gnina_cnn_score": row.get("gnina_coverage_fraction"),
            "rdkit": row.get("rdkit_coverage_fraction"),
        },
        "metrics": {name: metric(row, name) for name in METRIC_FIELDS},
        "runtime": {
            "sampling_steps": row.get("sampling_steps"),
            "matched_candidates_per_pocket": row.get("matched_candidates_per_pocket"),
        },
        "source_artifact": "configs/q2_ours_vs_public_baselines_summary.json",
        "model_improvement": False,
        "status": "scored",
    }


def layer_summary_row(method_id, method_name, layer, role, summary, source_artifact):
    means = summary.get("means") or summary.get("metric_means") or {}
    return {
        "method_id": method_id,
        "method_name": method_name,
        "layer": layer,
        "layer_role": role,
        "candidate_count": summary.get("candidate_count"),
        "coverage": summary.get("backend_coverage", {}),
        "metrics": {name: means.get(name) for name in METRIC_FIELDS},
        "runtime": {"sampling_steps": None, "matched_candidates_per_pocket": None},
        "source_artifact": source_artifact,
        "model_improvement": False,
        "status": "scored",
    }


def variant_layer_row(method_id, method_name, variant, layer_summary, source_artifact):
    layer = layer_summary.get("layer")
    role = "raw_model_native" if layer in {"raw_flow", "raw_rollout"} else "constrained_sampling"
    return {
        "method_id": method_id,
        "method_name": method_name,
        "variant_id": variant.get("variant_id"),
        "layer": layer,
        "layer_role": role,
        "candidate_count": layer_summary.get("candidate_count"),
        "coverage": {},
        "metrics": {
            name: layer_summary.get("metrics_mean", {}).get(name) for name in METRIC_FIELDS
        },
        "runtime": {"sampling_steps": 20, "matched_candidates_per_pocket": 1},
        "source_artifact": source_artifact,
        "model_improvement": False,
        "status": "scored",
    }


def variant_summary_rows(method_id, method_name, summary_path):
    path = Path(summary_path)
    if not path.is_file():
        return []
    summary = load_json(path)
    rows = []
    for variant in summary.get("variants", []):
        for layer_summary in variant.get("layers", []):
            rows.append(
                variant_layer_row(method_id, method_name, variant, layer_summary, summary_path)
            )
    return rows


def unavailable_row(method_id, method_name, layer, role, expected_artifact, reason):
    return {
        "method_id": method_id,
        "method_name": method_name,
        "layer": layer,
        "layer_role": role,
        "candidate_count": 0,
        "coverage": {},
        "metrics": {name: None for name in METRIC_FIELDS},
        "runtime": {"sampling_steps": None, "matched_candidates_per_pocket": None},
        "source_artifact": expected_artifact,
        "model_improvement": False,
        "status": "unavailable",
        "unavailable_reason": reason,
    }


def sort_key(row):
    role_rank = {"raw_model_native": 0, "constrained_sampling": 1, "postprocessing": 2}.get(
        row.get("layer_role"), 3
    )
    status_rank = 0 if row.get("status") == "scored" else 1
    vina = row.get("metrics", {}).get("vina_score")
    return (role_rank, status_rank, float("inf") if vina is None else vina)


def build_leaderboard(args):
    rows = []
    q2 = load_json(args.q2_summary)
    for row in q2.get("methods", []):
        rows.append(q2_row(row))

    bond_path = Path(args.bond_summary)
    if bond_path.is_file():
        bond = load_json(bond_path)
        for layer, summary in bond.get("layer_summaries", {}).items():
            role = "raw_model_native" if layer == "raw_geometry" else "postprocessing"
            rows.append(
                layer_summary_row(
                    "q3_bond_refinement",
                    "Q3 coordinate-preserving bond refinement",
                    layer,
                    role,
                    summary,
                    args.bond_summary,
                )
            )

    repair_path = Path(args.repair_summary)
    if repair_path.is_file():
        repair = load_json(repair_path)
        for layer, summary in repair.get("layer_summaries", {}).items():
            role = "raw_model_native" if layer == "no_repair" else "postprocessing"
            rows.append(
                layer_summary_row(
                    "q3_gated_repair",
                    "Q3 gated repair ablation",
                    layer,
                    role,
                    summary,
                    args.repair_summary,
                )
            )

    reranker_path = Path(args.reranker_summary)
    if reranker_path.is_file():
        reranker = load_json(reranker_path)
        summary = reranker.get("backend_aware_posthoc", {})
        rows.append(
            layer_summary_row(
                "q3_backend_aware_reranker",
                "Q3 backend-aware reranker",
                "backend_aware_posthoc",
                "postprocessing",
                {"candidate_count": summary.get("candidate_count"), "metric_means": summary},
                args.reranker_summary,
            )
        )

    local_rows = variant_summary_rows(
        "q3_local_pocket_flow",
        "Q3 local pocket flow head",
        args.local_pocket_summary,
    )
    rows.extend(local_rows)
    pairwise_rows = variant_summary_rows(
        "q3_pairwise_geometry",
        "Q3 pairwise geometry ablation",
        args.pairwise_summary,
    )
    rows.extend(pairwise_rows)
    if not local_rows:
        rows.append(
            unavailable_row(
                "q3_local_pocket_flow",
                "Q3 local pocket flow head",
                "raw_flow",
                "raw_model_native",
                args.local_pocket_summary,
                "public100 training/scoring artifact not present in this workspace",
            )
        )
    if not pairwise_rows:
        rows.append(
            unavailable_row(
                "q3_pairwise_geometry",
                "Q3 pairwise geometry ablation",
                "raw_flow",
                "raw_model_native",
                args.pairwise_summary,
                "public100 pairwise ablation scoring artifact not present in this workspace",
            )
        )
    rows.sort(key=sort_key)
    return {
        "artifact_name": "q3_model_improvement_leaderboard",
        "schema_version": 1,
        "claim_boundary": (
            "Only scored raw_flow/raw_rollout/no_repair/raw_geometry rows may support native model "
            "quality. Constrained, repaired, and reranked rows are separated and never marked as "
            "model improvement."
        ),
        "ranking_policy": {
            "primary_order": ["raw_model_native", "constrained_sampling", "postprocessing"],
            "within_group": "lower Vina score first when available",
            "unavailable_rows": "listed for dependency tracking but excluded from improvement claims",
        },
        "rows": rows,
    }


def write_markdown(path, leaderboard):
    lines = [
        "# Q3 Model Improvement Leaderboard",
        "",
        leaderboard["claim_boundary"],
        "",
        "| Method | Layer | Role | Status | Vina | GNINA affinity | GNINA CNN | QED | SA | Clash | Contact | Model improvement |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in leaderboard["rows"]:
        metrics = row["metrics"]
        fmt = lambda value: "NA" if value is None else f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
        lines.append(
            "| {} | `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["method_name"],
                row["layer"],
                row["layer_role"],
                row["status"],
                fmt(metrics.get("vina_score")),
                fmt(metrics.get("gnina_affinity")),
                fmt(metrics.get("gnina_cnn_score")),
                fmt(metrics.get("qed")),
                fmt(metrics.get("sa_score")),
                fmt(metrics.get("clash_fraction")),
                fmt(metrics.get("pocket_contact_fraction")),
                str(row["model_improvement"]).lower(),
            )
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    leaderboard = build_leaderboard(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(leaderboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, leaderboard)


if __name__ == "__main__":
    main()
