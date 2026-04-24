#!/usr/bin/env python3
"""Compact artifact gate for claim-matrix regression runs.

Usage:
  tools/claim_regression_gate.py <artifact_dir>
  tools/claim_regression_gate.py --run --config configs/unseen_pocket_claim_matrix.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_CLAIM_FIELDS = {
    "validation",
    "test",
    "backend_metrics",
    "layered_generation_metrics",
    "chemistry_novelty_diversity",
    "reranker_report",
    "slot_stability",
    "leakage_calibration",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise SystemExit(f"claim regression gate failed: {message}")


def finite_number(value):
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run and/or validate compact claim artifacts.")
    parser.add_argument("artifact_dir", nargs="?", help="Artifact directory to validate.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the configured compact claim experiment before validating artifacts.",
    )
    parser.add_argument(
        "--config",
        default="configs/unseen_pocket_claim_matrix.json",
        help="Experiment config used by --run and artifact-dir inference.",
    )
    parser.add_argument(
        "--cargo",
        default="cargo",
        help="Cargo executable used by --run.",
    )
    parser.add_argument(
        "--enforce-backend-thresholds",
        action="store_true",
        help="Require real-backend chemistry and pocket metrics to satisfy claim thresholds.",
    )
    parser.add_argument(
        "--enforce-data-thresholds",
        action="store_true",
        help="Require claim-ready real-data size, label coverage, and held-out family counts.",
    )
    parser.add_argument("--min-rdkit-available", type=float, default=1.0)
    parser.add_argument("--min-rdkit-sanitized-fraction", type=float, default=0.95)
    parser.add_argument("--min-unique-fraction", type=float, default=0.5)
    parser.add_argument("--max-backend-missing-structure-fraction", type=float, default=0.0)
    parser.add_argument("--max-clash-fraction", type=float, default=0.1)
    parser.add_argument("--min-strict-pocket-fit", type=float, default=0.35)
    parser.add_argument("--min-pocket-contact", type=float, default=0.8)
    parser.add_argument("--min-parsed-complexes", type=int, default=100)
    parser.add_argument("--min-retained-label-coverage", type=float, default=0.8)
    parser.add_argument("--min-heldout-family-count", type=int, default=10)
    return parser.parse_args(argv[1:])


def artifact_dir_from_config(config_path):
    config = load_json(config_path)
    try:
        return Path(config["research"]["training"]["checkpoint_dir"])
    except KeyError as exc:
        raise SystemExit(f"claim regression gate failed: config missing checkpoint_dir: {exc}") from exc


def run_experiment(args):
    command = [
        args.cargo,
        "run",
        "--bin",
        "pocket_diffusion",
        "--",
        "research",
        "experiment",
        "--config",
        args.config,
    ]
    completed = subprocess.run(command, check=False)
    require(completed.returncode == 0, f"experiment command failed with exit code {completed.returncode}")


def metric(report, section, name, fallback=None):
    metrics = report["backend_metrics"][section].get("metrics", {})
    if name in metrics:
        return metrics[name]
    if fallback and fallback in metrics:
        return metrics[fallback]
    return None


def require_metric_at_least(report, section, name, threshold, fallback=None):
    value = metric(report, section, name, fallback)
    require(value is not None, f"missing backend metric {section}.{name}")
    require(finite_number(value), f"backend metric {section}.{name} is not finite")
    require(value >= threshold, f"{section}.{name}={value} below threshold {threshold}")


def require_metric_at_most(report, section, name, threshold, fallback=None):
    value = metric(report, section, name, fallback)
    require(value is not None, f"missing backend metric {section}.{name}")
    require(finite_number(value), f"backend metric {section}.{name} is not finite")
    require(value <= threshold, f"{section}.{name}={value} above threshold {threshold}")


def validate_backend_thresholds(claim, args):
    require(
        claim["backend_metrics"]["chemistry_validity"].get("available") is True,
        "chemistry backend is unavailable under backend-threshold mode",
    )
    require(
        claim["backend_metrics"]["pocket_compatibility"].get("available") is True,
        "pocket backend is unavailable under backend-threshold mode",
    )
    require_metric_at_least(
        claim,
        "chemistry_validity",
        "rdkit_available",
        args.min_rdkit_available,
    )
    require_metric_at_least(
        claim,
        "chemistry_validity",
        "rdkit_sanitized_fraction",
        args.min_rdkit_sanitized_fraction,
        "valid_fraction",
    )
    require_metric_at_least(
        claim,
        "chemistry_validity",
        "rdkit_unique_smiles_fraction",
        args.min_unique_fraction,
        "unique_smiles_fraction",
    )
    for section in ("chemistry_validity", "docking_affinity", "pocket_compatibility"):
        if "backend_examples_scored" in claim["backend_metrics"][section].get("metrics", {}):
            require_metric_at_most(
                claim,
                section,
                "backend_missing_structure_fraction",
                args.max_backend_missing_structure_fraction,
            )
    require_metric_at_most(
        claim,
        "pocket_compatibility",
        "clash_fraction",
        args.max_clash_fraction,
    )
    require_metric_at_least(
        claim,
        "pocket_compatibility",
        "strict_pocket_fit_score",
        args.min_strict_pocket_fit,
        "heuristic_strict_pocket_fit_score",
    )
    require_metric_at_least(
        claim,
        "docking_affinity",
        "contact_fraction",
        args.min_pocket_contact,
        "pocket_contact_fraction",
    )


def validate_data_thresholds(validation, split, args):
    require(validation is not None, "missing dataset_validation_report.json under data-threshold mode")
    parsed = validation.get("parsed_examples")
    require(isinstance(parsed, int), "dataset_validation_report.parsed_examples missing")
    require(
        parsed >= args.min_parsed_complexes,
        f"parsed_examples={parsed} below threshold {args.min_parsed_complexes}",
    )

    label_coverage = validation.get("retained_label_coverage")
    require(
        finite_number(label_coverage),
        "dataset_validation_report.retained_label_coverage missing or non-finite",
    )
    require(
        label_coverage >= args.min_retained_label_coverage,
        f"retained_label_coverage={label_coverage} below threshold {args.min_retained_label_coverage}",
    )

    for split_name in ("val", "test"):
        histogram = split.get(split_name, {}).get("protein_family_proxy_histogram", {})
        family_count = len(histogram)
        require(
            family_count >= args.min_heldout_family_count,
            f"{split_name} protein family count={family_count} below threshold {args.min_heldout_family_count}",
        )


def validate_artifact_dir(artifact_dir, args):
    claim_path = artifact_dir / "claim_summary.json"
    experiment_path = artifact_dir / "experiment_summary.json"
    split_path = artifact_dir / "split_report.json"
    bundle_path = artifact_dir / "run_artifacts.json"
    validation_path = artifact_dir / "dataset_validation_report.json"

    for path in (claim_path, experiment_path, split_path, bundle_path):
        require(path.is_file(), f"missing required artifact {path}")

    claim = load_json(claim_path)
    experiment = load_json(experiment_path)
    split = load_json(split_path)
    bundle = load_json(bundle_path)
    validation = load_json(validation_path) if validation_path.is_file() else None

    missing = REQUIRED_CLAIM_FIELDS.difference(claim)
    require(not missing, f"claim_summary missing fields: {sorted(missing)}")
    require(bundle.get("schema_version", 0) >= 1, "run_artifacts schema_version must be >= 1")
    require(experiment.get("reproducibility", {}).get("metric_schema_version", 0) >= 4, "metric schema is stale")

    test = claim["test"]
    require(test.get("unseen_protein_fraction", 0.0) >= 0.0, "unseen protein fraction missing")
    require(finite_number(test.get("slot_activation_mean", 0.0)), "slot activation is not finite")
    require(finite_number(test.get("gate_activation_mean", 0.0)), "gate activation is not finite")
    require(finite_number(test.get("leakage_proxy_mean", 0.0)), "leakage proxy is not finite")

    layered = claim["layered_generation_metrics"]
    for layer in ("raw_rollout", "repaired_candidates", "inferred_bond_candidates", "reranked_candidates"):
        require(layer in layered, f"missing generation layer {layer}")
        require("valid_fraction" in layered[layer], f"missing {layer}.valid_fraction")
        require(
            "atom_type_sequence_diversity" in layered[layer],
            f"missing {layer}.atom_type_sequence_diversity",
        )
        require(
            "novel_atom_type_sequence_fraction" in layered[layer],
            f"missing {layer}.novel_atom_type_sequence_fraction",
        )

    chemistry_novelty = claim["chemistry_novelty_diversity"]
    for field in (
        "review_layer",
        "atom_type_sequence_diversity",
        "bond_topology_diversity",
        "coordinate_shape_diversity",
        "novel_atom_type_sequence_fraction",
        "novel_bond_topology_fraction",
        "novel_coordinate_shape_fraction",
        "interpretation",
    ):
        require(field in chemistry_novelty, f"chemistry_novelty_diversity missing {field}")
    benchmark = chemistry_novelty.get("benchmark_evidence", {})
    if benchmark:
        require(
            benchmark.get("evidence_tier")
            in {
                "proxy_only",
                "local_benchmark_style",
                "reviewer_benchmark_plus",
                "external_benchmark_backed",
            },
            "benchmark_evidence.evidence_tier is not a recognized reviewer tier",
        )

    for split_name in ("train", "val", "test"):
        row = split.get(split_name, {})
        require("ligand_atom_count_bins" in row, f"split {split_name} missing ligand bins")
        require("pocket_atom_count_bins" in row, f"split {split_name} missing pocket bins")
        require("protein_family_proxy_histogram" in row, f"split {split_name} missing family strata")

    quality = split.get("quality_checks")
    require(isinstance(quality, dict), "split report missing quality_checks")
    for field in (
        "weak_val_family_count",
        "weak_test_family_count",
        "severe_atom_count_skew_detected",
        "measurement_family_skew_detected",
        "warnings",
    ):
        require(field in quality, f"split quality_checks missing {field}")

    if args.enforce_backend_thresholds:
        validate_backend_thresholds(claim, args)
    if args.enforce_data_thresholds:
        validate_data_thresholds(validation, split, args)

    print(f"claim regression gate passed: {artifact_dir}")


def main(argv):
    args = parse_args(argv)
    config_path = Path(args.config)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else artifact_dir_from_config(config_path)
    if args.run:
        run_experiment(args)
    validate_artifact_dir(artifact_dir, args)


if __name__ == "__main__":
    main(sys.argv)
