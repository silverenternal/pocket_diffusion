#!/usr/bin/env python3
import argparse
import json


SECTIONS = [
    ("Pocket Fit Vs Docking", "pocket_fit_vs_docking"),
    ("Geometry Vs QED", "geometry_vs_qed"),
    ("Geometry Vs SA", "geometry_vs_sa"),
    ("Clash Vs Docking", "clash_vs_docking"),
    ("Interaction Proxies Vs Docking", "interaction_vs_docking"),
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render correlation_table.json as Markdown.")
    parser.add_argument("correlation_table")
    parser.add_argument("--output", default="docs/correlation_plot.md")
    return parser.parse_args(argv[1:])


def fmt(value, note=""):
    if value is None:
        if note == "constant_metric":
            return "constant_metric"
        if str(note).startswith("too_few_samples"):
            return "low_sample"
        return "unavailable"
    return f"{float(value):.4f}"


def rows_for(table, pair_id):
    return [row for row in table.get("metric_pairs", []) if row.get("pair_id") == pair_id]


def render_section(title, pair_id, table):
    lines = [f"## {title}", ""]
    lines.append("| Scope | Left | Right | N | Missing | Pearson | Spearman | Note |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows_for(table, pair_id):
        lines.append(
            "| {scope} | {left} | {right} | {n} | {missing} | {pearson} | {spearman} | {note} |".format(
                scope=row.get("scope", "unknown"),
                left=row.get("left_metric", ""),
                right=row.get("right_metric", ""),
                n=row.get("sample_count", 0),
                missing=row.get("missing_count", 0),
                pearson=fmt(row.get("pearson"), row.get("confidence_note", "")),
                spearman=fmt(row.get("spearman"), row.get("confidence_note", "")),
                note=row.get("confidence_note", ""),
            )
        )
    if not rows_for(table, pair_id):
        lines.append("| all | unavailable | unavailable | 0 | 0 | unsupported | unsupported | missing_pair |")
    lines.append("")
    return lines


def render(table, source):
    lines = [
        "# Correlation Plot",
        "",
        f"Source table: `{source}`",
        f"Record count: {table.get('record_count', 0)}",
        f"Minimum interpretable samples: {table.get('min_samples', 0)}",
        "",
        "This artifact is rendered from `correlation_table.json`; unsupported entries indicate missing backend coverage or too few candidate-level samples.",
        "",
    ]
    for title, pair_id in SECTIONS:
        lines.extend(render_section(title, pair_id, table))
    lines.extend(
        [
            "## Q2 Proxy Backend Pairs",
            "",
            "| Scope | Pair | Left | Right | N | Missing | Pearson | Spearman | Expected | Interpretation |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in table.get("metric_pairs", []):
        if row.get("scope") not in ("all", "raw_layers", "postprocessed_layers"):
            continue
        lines.append(
            "| {scope} | {pair} | {left} | {right} | {n} | {missing} | {pearson} | {spearman} | {expected} | {interp} |".format(
                scope=row.get("scope", "unknown"),
                pair=row.get("pair_id", ""),
                left=row.get("left_metric", ""),
                right=row.get("right_metric", ""),
                n=row.get("sample_count", 0),
                missing=row.get("missing_count", 0),
                pearson=fmt(row.get("pearson"), row.get("confidence_note", "")),
                spearman=fmt(row.get("spearman"), row.get("confidence_note", "")),
                expected=row.get("direction_expectation", ""),
                interp=row.get("interpretation", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Vs Postprocessed Deltas",
            "",
            "| Method | Raw Layer | Target Layer | Pairs | dVina | dGNINA | dCNN | dQED | dSA | dClash | dContact |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in table.get("layer_delta_summaries", []):
        deltas = row.get("metric_deltas", {})
        def delta(metric):
            return fmt(deltas.get(metric, {}).get("mean_delta"))

        lines.append(
            "| {method} | {raw} | {target} | {pairs} | {vina} | {gnina} | {cnn} | {qed} | {sa} | {clash} | {contact} |".format(
                method=row.get("method_id", "unknown"),
                raw=row.get("raw_layer", ""),
                target=row.get("target_layer", ""),
                pairs=row.get("pair_count", 0),
                vina=delta("vina_score"),
                gnina=delta("gnina_affinity"),
                cnn=delta("gnina_cnn_score"),
                qed=delta("qed"),
                sa=delta("sa_score"),
                clash=delta("clash_fraction"),
                contact=delta("pocket_contact_fraction"),
            )
        )
    lines.append("")
    lines.extend(
        [
            "## Flow Vs Denoising On Binding",
            "",
            "Use method-comparison candidate metrics with matched budgets before interpreting this section. Rows remain unsupported until both flow and denoising layers have candidate-level docking or GNINA/Vina coverage.",
            "",
            "## Constraint Vs True Model Capability",
            "",
            "Interpret raw_flow rows as native model evidence. Interpret repaired and reranked rows as postprocessing evidence unless raw-flow backend metrics are explicitly present.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv):
    args = parse_args(argv)
    with open(args.correlation_table, "r", encoding="utf-8") as handle:
        table = json.load(handle)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(render(table, args.correlation_table))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
