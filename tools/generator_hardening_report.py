#!/usr/bin/env python3
"""Generate a concise generator hardening report from the evidence bundle."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a generator hardening report.")
    parser.add_argument("--bundle", default="docs/evidence_bundle.json")
    parser.add_argument("--output", default="docs/generator_hardening_report.md")
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


def build_markdown(bundle):
    direction = bundle.get("generator_direction") or {}
    tradeoffs = bundle.get("efficiency_tradeoffs") or {}
    rows = tradeoffs.get("surfaces") or []
    lines = [
        "# Generator Hardening Report",
        "",
        "This file is generated from `docs/evidence_bundle.json` and should be refreshed through the reviewer revalidation path.",
        "",
        "## Direction",
        "",
        f"- Current direction: `{direction.get('current_direction')}`.",
        f"- Saturation status: `{direction.get('saturation_status')}`.",
        f"- Primary justification surface: `{direction.get('primary_justification_surface')}`.",
        f"- Stability surface: `{direction.get('stability_surface')}`.",
        "- Standalone decision artifact: `checkpoints/generator_decision/generator_decision.json`.",
        f"- Major model change gate: {direction.get('major_model_change_gate')}",
        "- Freshness gate: `python3 tools/generator_decision_bundle.py --check` fails if the persisted decision artifact is stale relative to the canonical larger-data surface, tight-geometry pressure surface, larger-data multi-seed summary, or promotion-relevant rollout/objective files.",
        "",
        "## Quality And Efficiency Tradeoffs",
        "",
        "| Surface | Pocket fit | Leakage | Test eps | Test memory MB | Chemistry tier |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + str(row.get("label"))
            + " | "
            + fmt(row.get("strict_pocket_fit_score"))
            + " | "
            + fmt(row.get("leakage_proxy_mean"))
            + " | "
            + fmt(row.get("test_examples_per_second"))
            + " | "
            + fmt(row.get("test_memory_mb"))
            + " | "
            + str(row.get("chemistry_evidence_tier") or "n/a")
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
        ]
    )
    for note in direction.get("reasons") or []:
        lines.append(f"- {note}")
    for note in tradeoffs.get("summary") or []:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Promotion Artifact",
            "",
            "- Refresh `checkpoints/generator_decision/generator_decision.json` through the reviewer revalidation path before promoting major generator changes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    bundle = load_json(args.bundle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(bundle), encoding="utf-8")
    print(f"generator hardening report written: {output}")


if __name__ == "__main__":
    main()
