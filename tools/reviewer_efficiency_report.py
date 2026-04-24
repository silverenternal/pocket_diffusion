#!/usr/bin/env python3
"""Generate a reviewer-facing efficiency and resource report."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build reviewer efficiency report.")
    parser.add_argument("--bundle", default="docs/evidence_bundle.json")
    parser.add_argument(
        "--multi-seed",
        default="configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json",
    )
    parser.add_argument("--output", default="docs/reviewer_efficiency_report.md")
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


def build_markdown(bundle, multi_seed):
    tradeoffs = bundle.get("efficiency_tradeoffs") or {}
    surfaces = tradeoffs.get("surfaces") or []
    aggregates = multi_seed.get("aggregates") or {}
    seed_runs = multi_seed.get("seed_runs") or []
    throughput = aggregates.get("test_examples_per_second") or {}
    pocket_fit = aggregates.get("strict_pocket_fit_score") or {}
    leakage = aggregates.get("leakage_proxy_mean") or {}

    lines = [
        "# Reviewer Efficiency Report",
        "",
        "This file is generated from canonical reviewer artifacts and should be refreshed through `./tools/revalidate_reviewer_bundle.sh`.",
        "",
        "## Surface Tradeoffs",
        "",
        "| Surface | Test eps | Rel throughput vs larger-data | Test memory MB | Pocket fit | Leakage | Candidate valid | Chemistry tier |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in surfaces:
        lines.append(
            "| "
            + str(row.get("label"))
            + " | "
            + fmt(row.get("test_examples_per_second"))
            + " | "
            + fmt(row.get("relative_test_throughput_vs_larger_data"))
            + "x | "
            + fmt(row.get("test_memory_mb"))
            + " | "
            + fmt(row.get("strict_pocket_fit_score"))
            + " | "
            + fmt(row.get("leakage_proxy_mean"))
            + " | "
            + fmt(row.get("candidate_valid_fraction"))
            + " | "
            + str(row.get("chemistry_evidence_tier") or "n/a")
            + " |"
        )

    lines.extend(
        [
            "",
            "## Larger-Data Seed Stability",
            "",
            "| Aggregate | Mean | Std | Min | Max | 95% CI low | 95% CI high |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "| test_examples_per_second | "
            + fmt(throughput.get("mean"))
            + " | "
            + fmt(throughput.get("std"))
            + " | "
            + fmt(throughput.get("min"))
            + " | "
            + fmt(throughput.get("max"))
            + " | "
            + fmt(throughput.get("confidence95_low"))
            + " | "
            + fmt(throughput.get("confidence95_high"))
            + " |",
            "| strict_pocket_fit_score | "
            + fmt(pocket_fit.get("mean"))
            + " | "
            + fmt(pocket_fit.get("std"))
            + " | "
            + fmt(pocket_fit.get("min"))
            + " | "
            + fmt(pocket_fit.get("max"))
            + " | "
            + fmt(pocket_fit.get("confidence95_low"))
            + " | "
            + fmt(pocket_fit.get("confidence95_high"))
            + " |",
            "| leakage_proxy_mean | "
            + fmt(leakage.get("mean"))
            + " | "
            + fmt(leakage.get("std"))
            + " | "
            + fmt(leakage.get("min"))
            + " | "
            + fmt(leakage.get("max"))
            + " | "
            + fmt(leakage.get("confidence95_low"))
            + " | "
            + fmt(leakage.get("confidence95_high"))
            + " |",
            "",
            "## Per-Seed Throughput",
            "",
            "| Seed | Test eps | Pocket fit | Leakage | Gate activation | Slot activation |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in seed_runs:
        lines.append(
            "| "
            + str(row.get("seed"))
            + " | "
            + fmt(row.get("test_examples_per_second"))
            + " | "
            + fmt(row.get("strict_pocket_fit_score"))
            + " | "
            + fmt(row.get("leakage_proxy_mean"))
            + " | "
            + fmt(row.get("gate_activation_mean"))
            + " | "
            + fmt(row.get("slot_activation_mean"))
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    for note in tradeoffs.get("summary") or []:
        lines.append(f"- {note}")
    lines.append(
        "- Larger-data throughput, pocket fit, and leakage are now reported both as single-surface reviewer gates and as multi-seed aggregates so generator changes can be judged on quality-resource tradeoffs rather than isolated snapshots."
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    bundle = load_json(args.bundle)
    multi_seed = load_json(args.multi_seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(bundle, multi_seed), encoding="utf-8")
    print(f"reviewer efficiency report written: {output}")


if __name__ == "__main__":
    main()
