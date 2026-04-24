#!/usr/bin/env python3
"""Generate reviewer-facing reranker and candidate-layer diagnostics."""

import argparse
import json
from pathlib import Path


SURFACES = [
    ("larger_data_canonical", "checkpoints/pdbbindpp_real_backends"),
    ("tight_geometry_pressure", "checkpoints/tight_geometry_pressure"),
    ("real_backend_gate", "checkpoints/real_backends"),
]

LAYERS = [
    "raw_rollout",
    "repaired_candidates",
    "inferred_bond_candidates",
    "deterministic_proxy_candidates",
    "reranked_candidates",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build reranker diagnostics.")
    parser.add_argument("--output", default="docs/reranker_diagnostics.md")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def calibration_summary(coefficients):
    if not coefficients:
        return "n/a"
    ordered = sorted(coefficients.items(), key=lambda item: item[1], reverse=True)
    active = [f"{name}={value:.4f}" for name, value in ordered if value and value > 0.0]
    return ", ".join(active) if active else "all coefficients are zero"


def reranker_decision(metrics):
    reranked = metrics.get("reranked_candidates") or {}
    deterministic = metrics.get("deterministic_proxy_candidates") or {}
    inferred = metrics.get("inferred_bond_candidates") or {}
    if not reranked:
        return "reranked layer missing"
    if deterministic and reranked == deterministic:
        return "reranker currently ties deterministic proxy selection on persisted metrics; keep it documented as bounded calibration rather than a distinct quality win."
    if inferred:
        centroid_gain = (inferred.get("mean_centroid_offset") or 0.0) - (
            reranked.get("mean_centroid_offset") or 0.0
        )
        clash_delta = (inferred.get("clash_fraction") or 0.0) - (
            reranked.get("clash_fraction") or 0.0
        )
        if centroid_gain > 0 or clash_delta > 0:
            return "reranker improves at least one candidate-layer fit metric relative to inferred-bond candidates."
    return "reranker reshuffles candidate tradeoffs without a clean persisted win over the deterministic proxy on current reviewer surfaces."


def build_markdown():
    lines = [
        "# Reranker Diagnostics",
        "",
        "This file is generated from canonical reviewer artifacts and should be refreshed through `./tools/revalidate_reviewer_bundle.sh`.",
    ]

    for label, artifact_dir in SURFACES:
        claim = load_json(Path(artifact_dir) / "claim_summary.json")
        layers_doc = load_json(Path(artifact_dir) / "generation_layers_test.json")
        layered_metrics = layers_doc.get("layered_metrics") or {}
        calibration = layered_metrics.get("reranker_calibration") or {}
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- Artifact: `{artifact_dir}`",
                f"- Calibration method: `{calibration.get('method')}`",
                f"- Active coefficients: {calibration_summary(calibration.get('coefficients') or {})}",
                f"- Fitted candidate count: {fmt(calibration.get('fitted_candidate_count'), digits=0)}",
                f"- Decision: {reranker_decision(layered_metrics)}",
                "",
                "| Layer | Candidates | Valid | Centroid offset | Clash | Atom-seq diversity | Bond-topology diversity |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for layer in LAYERS:
            metrics = layered_metrics.get(layer) or {}
            lines.append(
                "| "
                + layer
                + " | "
                + fmt(metrics.get("candidate_count"), digits=0)
                + " | "
                + fmt(metrics.get("valid_fraction"))
                + " | "
                + fmt(metrics.get("mean_centroid_offset"))
                + " | "
                + fmt(metrics.get("clash_fraction"))
                + " | "
                + fmt(metrics.get("atom_type_sequence_diversity"))
                + " | "
                + fmt(metrics.get("bond_topology_diversity"))
                + " |"
            )
        leakage = (claim.get("leakage_calibration") or {})
        lines.extend(
            [
                "",
                "- Reviewer guardrails: judge reranker changes jointly against strict pocket fit, clash fraction, candidate validity, and leakage rather than a single scalar score.",
                "- Leakage review status: `"
                + str(leakage.get("reviewer_status"))
                + "` with decision `"
                + str(leakage.get("decision"))
                + "`.",
            ]
        )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(), encoding="utf-8")
    print(f"reranker diagnostics written: {output}")


if __name__ == "__main__":
    main()
