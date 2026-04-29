#!/usr/bin/env python3
"""Build backend-backed preference pairs from scored Q2/Q3 candidate artifacts."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


METRICS = (
    "vina_score",
    "gnina_affinity",
    "gnina_cnn_score",
    "qed",
    "sa_score",
    "pocket_contact_fraction",
    "strict_pocket_fit_score",
    "clash_fraction",
    "centroid_offset",
    "rdkit_valid_fraction",
    "rdkit_sanitized_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", nargs="+", required=True)
    parser.add_argument("--repair-damage-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-pairs-per-class", type=int, default=40)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric(row, key):
    value = row.get("metrics", {}).get(key)
    return float(value) if finite(value) else None


def value(row, key, default=None):
    result = metric(row, key)
    return default if result is None else result


def docking_score(row):
    values = [metric(row, "vina_score"), metric(row, "gnina_affinity")]
    values = [item for item in values if item is not None]
    return sum(values) / len(values) if values else None


def druglike_score(row):
    qed = value(row, "qed", 0.0)
    sa = value(row, "sa_score", 10.0)
    valid = value(row, "rdkit_valid_fraction", 1.0)
    sanitized = value(row, "rdkit_sanitized_fraction", 1.0)
    return 0.45 * qed + 0.25 * max(0.0, 1.0 - sa / 10.0) + 0.15 * valid + 0.15 * sanitized


def pair_candidate(row):
    return {
        "candidate_id": row.get("candidate_id") or "unknown",
        "method_id": row.get("method_id") or "unknown",
        "layer": row.get("layer") or "unknown",
    }


def feature_deltas(winner, loser):
    deltas = {}
    for key in METRICS:
        winner_value = metric(winner, key)
        loser_value = metric(loser, key)
        if winner_value is not None and loser_value is not None:
            deltas[key] = winner_value - loser_value
    winner_docking = docking_score(winner)
    loser_docking = docking_score(loser)
    if winner_docking is not None and loser_docking is not None:
        deltas["mean_docking_score"] = winner_docking - loser_docking
    deltas["druglike_score"] = druglike_score(winner) - druglike_score(loser)
    return deltas


def backend_coverage(*rows):
    coverage = {}
    for key in ("vina_score", "gnina_affinity", "gnina_cnn_score", "qed", "sa_score"):
        coverage[key] = sum(1 for row in rows if metric(row, key) is not None) / float(len(rows))
    return coverage


def make_pair(preference_class, evidence_source, winner, loser, strength):
    return {
        "schema_version": 1,
        "pair_id": (
            f"{preference_class}:{winner.get('candidate_id')}__beats__"
            f"{loser.get('candidate_id')}"
        ),
        "example_id": winner.get("example_id") or loser.get("example_id") or "unknown",
        "protein_id": winner.get("protein_id") or loser.get("protein_id") or "unknown",
        "winner": pair_candidate(winner),
        "loser": pair_candidate(loser),
        "preference_class": preference_class,
        "evidence_source": evidence_source,
        "preference_strength": max(0.0, min(float(strength), 1.0)),
        "feature_deltas": feature_deltas(winner, loser),
        "backend_coverage": backend_coverage(winner, loser),
    }


def best_by(rows, key_fn, reverse=False):
    scored = [(key_fn(row), row) for row in rows]
    scored = [(score, row) for score, row in scored if score is not None and finite(score)]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=reverse)
    return scored[0][1]


def build_proxy_failure_pairs(grouped, max_count):
    pairs = []
    for _, rows in grouped.items():
        candidates = [row for row in rows if docking_score(row) is not None]
        if len(candidates) < 2:
            continue
        loser = best_by(
            [
                row
                for row in candidates
                if value(row, "pocket_contact_fraction", 0.0) >= 0.8
                and docking_score(row) is not None
                and docking_score(row) > 5.0
            ],
            docking_score,
            reverse=True,
        )
        winner = best_by(
            [
                row
                for row in candidates
                if loser is not None
                and row.get("candidate_id") != loser.get("candidate_id")
                and docking_score(row) is not None
                and docking_score(row) + 5.0 < docking_score(loser)
            ],
            docking_score,
        )
        if winner and loser:
            strength = min((docking_score(loser) - docking_score(winner)) / 50.0, 1.0)
            pairs.append(
                make_pair(
                    "high_pocket_fit_bad_docking",
                    "vina_gnina_proxy_disagreement",
                    winner,
                    loser,
                    strength,
                )
            )
    return sorted(pairs, key=lambda pair: -pair["preference_strength"])[:max_count]


def build_good_docking_druglike_pairs(grouped, max_count):
    pairs = []
    for _, rows in grouped.items():
        candidates = [row for row in rows if docking_score(row) is not None]
        if len(candidates) < 2:
            continue
        winner = best_by(candidates, lambda row: docking_score(row) - 5.0 * druglike_score(row))
        loser = best_by(
            [
                row
                for row in candidates
                if winner is not None
                and row.get("candidate_id") != winner.get("candidate_id")
                and docking_score(row) is not None
                and docking_score(row) > docking_score(winner) + 2.0
            ],
            lambda row: docking_score(row) - 2.0 * druglike_score(row),
            reverse=True,
        )
        if winner and loser:
            strength = min(
                ((docking_score(loser) - docking_score(winner)) / 30.0)
                + max(druglike_score(winner) - druglike_score(loser), 0.0),
                1.0,
            )
            pairs.append(
                make_pair(
                    "good_docking_druglike",
                    "vina_gnina_rdkit",
                    winner,
                    loser,
                    strength,
                )
            )
    return sorted(pairs, key=lambda pair: -pair["preference_strength"])[:max_count]


def build_docking_good_druglike_bad_pairs(grouped, max_count):
    pairs = []
    for _, rows in grouped.items():
        candidates = [row for row in rows if docking_score(row) is not None]
        if len(candidates) < 2:
            continue
        loser = best_by(
            [
                row
                for row in candidates
                if druglike_score(row) < 0.55
                and (value(row, "qed", 0.0) < 0.35 or value(row, "sa_score", 10.0) > 3.0)
            ],
            docking_score,
        )
        winner = best_by(
            [
                row
                for row in candidates
                if loser is not None
                and row.get("candidate_id") != loser.get("candidate_id")
                and druglike_score(row) > druglike_score(loser) + 0.08
                and docking_score(row) <= docking_score(loser) + 10.0
            ],
            lambda row: -druglike_score(row),
        )
        if winner and loser:
            strength = min(max(druglike_score(winner) - druglike_score(loser), 0.0) * 2.0, 1.0)
            pairs.append(
                make_pair(
                    "docking_good_druglike_bad",
                    "vina_gnina_rdkit_tradeoff",
                    winner,
                    loser,
                    strength,
                )
            )
    return sorted(pairs, key=lambda pair: -pair["preference_strength"])[:max_count]


def synthetic_row_from_repair_case(case, which):
    metrics_key = f"{which}_metrics"
    candidate_key = "raw_candidate_id" if which == "raw" else "candidate_id"
    return {
        "candidate_id": case.get(candidate_key),
        "example_id": case.get("example_id"),
        "protein_id": case.get("protein_id"),
        "method_id": case.get("method_id"),
        "layer": "no_repair" if which == "raw" else case.get("layer"),
        "metrics": case.get(metrics_key, {}),
    }


def build_repair_failure_pairs(repair_damage, max_count):
    pairs = []
    for case in repair_damage.get("worst_cases", []):
        if case.get("likely_failure_component") not in {
            "coordinate_movement",
            "docking_box_shift_from_coordinate_change",
            "bond_payload_or_conversion",
        }:
            continue
        raw = synthetic_row_from_repair_case(case, "raw")
        layer = synthetic_row_from_repair_case(case, "layer")
        delta = case.get("delta_vs_no_repair", {})
        strength = min(max(float(delta.get("vina_score") or 0.0), float(delta.get("gnina_affinity") or 0.0)) / 100.0, 1.0)
        pair = make_pair(
            "repair_destroys_docking",
            "repair_damage_cases",
            raw,
            layer,
            strength,
        )
        pair["geometry_delta"] = case.get("geometry_delta", {})
        pair["likely_failure_component"] = case.get("likely_failure_component")
        pairs.append(pair)
    return sorted(pairs, key=lambda pair: -pair["preference_strength"])[:max_count]


def write_markdown(payload, path):
    lines = [
        "# Q3 Backend Preference Pairs",
        "",
        payload["claim_boundary"],
        "",
        "## Coverage",
        "",
        f"- pair_count: {payload['pair_count']}",
        f"- method_id_coverage: {payload['method_id_coverage']}",
        f"- backend_supported_pair_fraction: {payload['backend_supported_pair_fraction']:.4g}",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    for name, count in sorted(payload["class_coverage"].items()):
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Top Pairs",
            "",
            "| Class | Winner | Loser | Strength | dVina | dGNINA | dQED | dSA |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair in payload["records"][:20]:
        delta = pair["feature_deltas"]
        lines.append(
            "| {klass} | `{winner}` | `{loser}` | {strength:.4g} | {vina} | {gnina} | {qed} | {sa} |".format(
                klass=pair["preference_class"],
                winner=pair["winner"]["candidate_id"],
                loser=pair["loser"]["candidate_id"],
                strength=pair["preference_strength"],
                vina=format_value(delta.get("vina_score")),
                gnina=format_value(delta.get("gnina_affinity")),
                qed=format_value(delta.get("qed")),
                sa=format_value(delta.get("sa_score")),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value):
    return "NA" if value is None else f"{value:.4g}"


def main():
    args = parse_args()
    rows = []
    for path in args.candidate_metrics:
        rows.extend(load_jsonl(path))
    repair_damage = load_json(args.repair_damage_json)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("example_id") or "unknown", row.get("protein_id") or "unknown")].append(row)

    records = []
    records.extend(build_proxy_failure_pairs(grouped, args.max_pairs_per_class))
    records.extend(build_good_docking_druglike_pairs(grouped, args.max_pairs_per_class))
    records.extend(build_docking_good_druglike_bad_pairs(grouped, args.max_pairs_per_class))
    records.extend(build_repair_failure_pairs(repair_damage, args.max_pairs_per_class))

    class_coverage = Counter(pair["preference_class"] for pair in records)
    method_ids = {
        pair[side]["method_id"]
        for pair in records
        for side in ("winner", "loser")
        if pair[side].get("method_id")
    }
    backend_pairs = sum(
        1
        for pair in records
        if all(pair["backend_coverage"].get(key, 0.0) > 0.0 for key in ("vina_score", "gnina_affinity"))
    )
    payload = {
        "artifact_name": "q3_backend_preference_pairs",
        "schema_version": 1,
        "split": "q2_public100_and_q3_repair_damage",
        "claim_boundary": "This artifact constructs backend-backed preference data only. It does not enable RL/DPO training and must not be reported as native model improvement.",
        "inputs": {
            "candidate_metrics": args.candidate_metrics,
            "repair_damage_json": args.repair_damage_json,
        },
        "pair_count": len(records),
        "method_id_coverage": len(method_ids),
        "class_coverage": dict(sorted(class_coverage.items())),
        "backend_supported_pair_fraction": backend_pairs / float(max(len(records), 1)),
        "required_classes_present": {
            "proxy_failure": class_coverage.get("high_pocket_fit_bad_docking", 0) > 0,
            "repair_failure": class_coverage.get("repair_destroys_docking", 0) > 0,
        },
        "records": records,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.output_md)


if __name__ == "__main__":
    main()
