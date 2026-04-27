#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


METRIC_PATHS = {
    "candidate_valid_fraction": ["test", "candidate_valid_fraction"],
    "strict_pocket_fit_score": ["test", "strict_pocket_fit_score"],
    "unique_smiles_fraction": ["test", "unique_smiles_fraction"],
    "qed": ["backend_metrics", "chemistry_validity", "metrics", "qed"],
    "sa_score": ["backend_metrics", "chemistry_validity", "metrics", "sa_score"],
    "drug_likeness_coverage_fraction": ["backend_metrics", "chemistry_validity", "metrics", "drug_likeness_coverage_fraction"],
    "docking_score_coverage_fraction": ["backend_metrics", "docking_affinity", "metrics", "docking_score_coverage_fraction"],
    "vina_score_mean": ["backend_metrics", "docking_affinity", "metrics", "vina_score_mean"],
    "scaffold_novelty_fraction": ["layered_generation_metrics", "reranked_candidates", "scaffold_novelty_fraction"],
    "unique_scaffold_fraction": ["layered_generation_metrics", "reranked_candidates", "unique_scaffold_fraction"],
    "hydrogen_bond_proxy": ["layered_generation_metrics", "reranked_candidates", "hydrogen_bond_proxy"],
    "hydrophobic_contact_proxy": ["layered_generation_metrics", "reranked_candidates", "hydrophobic_contact_proxy"],
    "contact_balance": ["layered_generation_metrics", "reranked_candidates", "contact_balance"],
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Aggregate drug-level metrics across multi-seed claim artifacts.")
    parser.add_argument("multi_seed_dir")
    parser.add_argument("--output", default="multi_seed_drug_level_summary.json")
    return parser.parse_args(argv[1:])


def get_path(payload, path):
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def aggregate(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "ci95_half_width": None,
        }
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci95_half_width": 1.96 * std / math.sqrt(n) if n > 1 else 0.0,
    }


def main(argv):
    args = parse_args(argv)
    root = Path(args.multi_seed_dir)
    seed_paths = sorted(root.glob("seed_*/claim_summary.json"))
    seeds = []
    values_by_metric = {metric: [] for metric in METRIC_PATHS}
    missing_by_metric = {metric: [] for metric in METRIC_PATHS}
    for path in seed_paths:
        claim = json.loads(path.read_text(encoding="utf-8"))
        seed = path.parent.name.removeprefix("seed_")
        row = {"seed": seed, "artifact": str(path), "metrics": {}, "missing_metrics": []}
        for metric, metric_path in METRIC_PATHS.items():
            value = get_path(claim, metric_path)
            if value is None:
                row["missing_metrics"].append(metric)
                missing_by_metric[metric].append(seed)
            else:
                row["metrics"][metric] = value
                values_by_metric[metric].append(value)
        seeds.append(row)
    summary = {
        "schema_version": 1,
        "multi_seed_dir": str(root),
        "seed_count": len(seeds),
        "seeds": seeds,
        "aggregates": {
            metric: {
                **aggregate(values),
                "missing_seed_count": len(missing_by_metric[metric]),
                "missing_seeds": missing_by_metric[metric],
            }
            for metric, values in values_by_metric.items()
        },
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"multi-seed drug-level summary written: {output}")


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
