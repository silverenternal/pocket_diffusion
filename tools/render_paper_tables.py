#!/usr/bin/env python3
"""Render manuscript tables from versioned JSON artifacts."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-comparison", default="configs/method_comparison_summary.json")
    parser.add_argument("--ablation", default="configs/ablation_delta_table.json")
    parser.add_argument("--multi-seed", default="configs/multi_seed_drug_level_summary.json")
    parser.add_argument("--out-dir", default="paper/tables")
    return parser.parse_args()


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_baselines(payload, out_dir):
    lines = [
        "# Baseline Comparison",
        "",
        f"Provenance: `{payload.get('artifact_dir', 'unknown')}`",
        "",
        "| method | role | steps | runtime ms | raw valid | raw contact | raw clash | layers |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("methods", []):
        layers = ", ".join(row.get("available_layers") or [])
        val = lambda key: "NA" if row.get(key) is None else f"{row.get(key):.4g}"
        lines.append(
            f"| {row.get('method_id')} | {row.get('evidence_role')} | {row.get('sampling_steps')} | "
            f"{val('wall_time_ms')} | {val('native_valid_fraction')} | {val('native_pocket_contact_fraction')} | "
            f"{val('native_clash_fraction')} | {layers} |"
        )
    write(out_dir / "baselines.md", lines)
    write(out_dir / "main_results.md", lines)


def render_ablations(payload, out_dir):
    lines = [
        "# Ablations",
        "",
        f"Provenance: `{payload.get('artifact_dir', 'unknown')}`",
        "",
        "| variant | metric | base | variant | delta | layer |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| {row.get('variant_label')} | {row.get('metric_name')} | {row.get('base_value'):.4g} | "
            f"{row.get('variant_value'):.4g} | {row.get('delta'):.4g} | {row.get('layer')} |"
        )
    write(out_dir / "ablations.md", lines)


def render_multiseed(payload, out_dir):
    lines = [
        "# Multi-Seed Summary",
        "",
        f"Provenance: `{payload.get('multi_seed_dir', 'unknown')}`",
        "",
        "| metric | count | mean | std | ci95 | missing seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, row in (payload.get("aggregates") or {}).items():
        fmt = lambda value: "NA" if value is None else f"{value:.4g}"
        lines.append(
            f"| {metric} | {row.get('count')} | {fmt(row.get('mean'))} | {fmt(row.get('std'))} | "
            f"{fmt(row.get('ci95_half_width'))} | {row.get('missing_seed_count')} |"
        )
    write(out_dir / "multi_seed.md", lines)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    method = load(args.method_comparison)
    ablation = load(args.ablation)
    multiseed = load(args.multi_seed)
    if method:
        render_baselines(method, out_dir)
    if ablation:
        render_ablations(ablation, out_dir)
    if multiseed:
        render_multiseed(multiseed, out_dir)
    print(f"paper tables rendered under: {out_dir}")


if __name__ == "__main__":
    main()
