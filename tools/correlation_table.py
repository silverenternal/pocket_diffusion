#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import defaultdict


METRIC_PAIRS = [
    ("pocket_fit_vs_vina", ["pocket_contact_fraction", "strict_pocket_fit_score", "contact_fraction"], ["vina_score", "vina_best_affinity_kcal_mol"]),
    ("pocket_fit_vs_gnina_affinity", ["pocket_contact_fraction", "strict_pocket_fit_score", "contact_fraction"], ["gnina_affinity", "gnina_score"]),
    ("pocket_fit_vs_gnina_cnn_score", ["pocket_contact_fraction", "strict_pocket_fit_score", "contact_fraction"], ["gnina_cnn_score"]),
    ("pocket_fit_vs_docking_proxy", ["pocket_contact_fraction", "strict_pocket_fit_score", "contact_fraction"], ["docking_like_score", "docking_score"]),
    ("geometry_vs_qed", ["mean_centroid_offset", "centroid_offset", "clash_fraction"], ["qed", "raw_qed"]),
    ("geometry_vs_sa", ["mean_centroid_offset", "centroid_offset", "clash_fraction"], ["sa_score", "raw_sa"]),
    ("clash_vs_vina", ["clash_fraction"], ["vina_score", "vina_best_affinity_kcal_mol"]),
    ("clash_vs_gnina_affinity", ["clash_fraction"], ["gnina_affinity", "gnina_score"]),
    ("clash_vs_gnina_cnn_score", ["clash_fraction"], ["gnina_cnn_score"]),
    ("clash_vs_docking_proxy", ["clash_fraction"], ["docking_like_score", "docking_score"]),
    ("interaction_vs_vina", ["hydrogen_bond_proxy", "hydrophobic_contact_proxy", "contact_balance"], ["vina_score", "vina_best_affinity_kcal_mol"]),
    ("interaction_vs_gnina_affinity", ["hydrogen_bond_proxy", "hydrophobic_contact_proxy", "contact_balance"], ["gnina_affinity", "gnina_score"]),
    ("interaction_vs_gnina_cnn_score", ["hydrogen_bond_proxy", "hydrophobic_contact_proxy", "contact_balance"], ["gnina_cnn_score"]),
    ("interaction_vs_docking_proxy", ["hydrogen_bond_proxy", "hydrophobic_contact_proxy", "contact_balance"], ["docking_like_score", "docking_score"]),
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Build deterministic candidate-metric correlation tables.")
    parser.add_argument("candidate_metrics", nargs="+", help="candidate_metrics.jsonl files")
    parser.add_argument("--output", default="correlation_table.json")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--allow-low-sample", action="store_true")
    return parser.parse_args(argv[1:])


def load_records(paths):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
                record.setdefault("source_file", path)
                records.append(record)
    return records


def metric_value(record, aliases):
    metrics = record.get("metrics", {})
    for name in aliases:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value), name
    return None, None


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(vx * vy)
    if denom == 0.0:
        return None
    return cov / denom


def ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            result[ordered[k][0]] = rank
        i = j
    return result


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def build_row(scope, pair_id, left_aliases, right_aliases, records, min_samples):
    xs = []
    ys = []
    left_names = defaultdict(int)
    right_names = defaultdict(int)
    missing = 0
    for record in records:
        left, left_name = metric_value(record, left_aliases)
        right, right_name = metric_value(record, right_aliases)
        if left is None or right is None:
            missing += 1
            continue
        xs.append(left)
        ys.append(right)
        left_names[left_name] += 1
        right_names[right_name] += 1
    sample_count = len(xs)
    enough = sample_count >= min_samples
    pearson_value = pearson(xs, ys) if enough else None
    spearman_value = spearman(xs, ys) if enough else None
    if not enough:
        confidence_note = f"too_few_samples_min_{min_samples}"
    elif pearson_value is None or spearman_value is None:
        confidence_note = "constant_metric"
    else:
        confidence_note = "interpretable"
    return {
        "scope": scope,
        "pair_id": pair_id,
        "left_metric": max(left_names, key=left_names.get) if left_names else left_aliases[0],
        "right_metric": max(right_names, key=right_names.get) if right_names else right_aliases[0],
        "sample_count": sample_count,
        "missing_count": missing,
        "pearson": pearson_value,
        "spearman": spearman_value,
        "confidence_note": confidence_note,
    }


def build_table(records, min_samples):
    by_layer = defaultdict(list)
    for record in records:
        by_layer[record.get("layer", "unknown")].append(record)
    scopes = {"all": records}
    scopes.update({f"layer:{layer}": layer_records for layer, layer_records in sorted(by_layer.items())})
    rows = []
    for scope, scope_records in scopes.items():
        for pair_id, left_aliases, right_aliases in METRIC_PAIRS:
            rows.append(build_row(scope, pair_id, left_aliases, right_aliases, scope_records, min_samples))
    return {
        "schema_version": 1,
        "record_count": len(records),
        "min_samples": min_samples,
        "metric_pairs": rows,
    }


def main(argv):
    args = parse_args(argv)
    records = load_records(args.candidate_metrics)
    table = build_table(records, args.min_samples)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not args.allow_low_sample:
        low = [
            row for row in table["metric_pairs"]
            if row["scope"] == "all" and row["sample_count"] < args.min_samples
        ]
        if low:
            print(
                f"correlation table has {len(low)} required all-scope pair(s) below min_samples={args.min_samples}",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
