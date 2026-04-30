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


DEFAULT_GATE_THRESHOLDS = {
    "min_rdkit_available": 1.0,
    "min_rdkit_sanitized_fraction": 0.95,
    "min_unique_fraction": 0.5,
    "max_backend_missing_structure_fraction": 0.0,
    "max_clash_fraction": 0.1,
    "min_strict_pocket_fit": 0.35,
    "min_pocket_contact": 0.8,
    "min_candidate_valid_fraction": 0.95,
    "max_leakage_proxy_mean": 0.08,
    "min_parsed_complexes": 100,
    "min_retained_label_coverage": 0.8,
    "min_heldout_family_count": 10,
}


DEFAULT_REAL_GENERATION_THRESHOLDS = {
    "min_raw_native_valid_fraction": 0.1,
    "min_raw_native_graph_valid_fraction": 0.1,
    "min_raw_native_pocket_contact_fraction": 0.5,
    "max_raw_native_clash_fraction": 0.2,
    "min_raw_native_unique_fraction": 0.1,
    "min_real_generation_seed_count": 3,
}


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
        "--claim-contract",
        default="configs/paper_claim_contract.json",
        help="Claim/threshold contract consumed by claim gate checks.",
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
    parser.add_argument("--min-rdkit-available", type=float, default=DEFAULT_GATE_THRESHOLDS["min_rdkit_available"])
    parser.add_argument(
        "--min-rdkit-sanitized-fraction",
        type=float,
        default=DEFAULT_GATE_THRESHOLDS["min_rdkit_sanitized_fraction"],
    )
    parser.add_argument("--min-unique-fraction", type=float, default=DEFAULT_GATE_THRESHOLDS["min_unique_fraction"])
    parser.add_argument(
        "--max-backend-missing-structure-fraction",
        type=float,
        default=DEFAULT_GATE_THRESHOLDS["max_backend_missing_structure_fraction"],
    )
    parser.add_argument("--max-clash-fraction", type=float, default=DEFAULT_GATE_THRESHOLDS["max_clash_fraction"])
    parser.add_argument(
        "--min-strict-pocket-fit",
        type=float,
        default=DEFAULT_GATE_THRESHOLDS["min_strict_pocket_fit"],
    )
    parser.add_argument("--min-pocket-contact", type=float, default=DEFAULT_GATE_THRESHOLDS["min_pocket_contact"])
    parser.add_argument(
        "--min-candidate-valid-fraction",
        type=float,
        default=DEFAULT_GATE_THRESHOLDS["min_candidate_valid_fraction"],
    )
    parser.add_argument(
        "--max-leakage-proxy-mean",
        type=float,
        default=DEFAULT_GATE_THRESHOLDS["max_leakage_proxy_mean"],
    )
    parser.add_argument("--min-parsed-complexes", type=int, default=DEFAULT_GATE_THRESHOLDS["min_parsed_complexes"])
    parser.add_argument(
        "--min-retained-label-coverage",
        type=float,
        default=DEFAULT_GATE_THRESHOLDS["min_retained_label_coverage"],
    )
    parser.add_argument(
        "--min-heldout-family-count",
        type=int,
        default=DEFAULT_GATE_THRESHOLDS["min_heldout_family_count"],
    )
    parser.add_argument(
        "--strict-preference-gate",
        action="store_true",
        help="Enforce strict preference schema/source/wording checks when preference artifacts are expected.",
    )
    parser.add_argument(
        "--strict-preference-min-pair-count",
        type=int,
        default=1,
        help="Minimum total preference pairs required in strict preference mode.",
    )
    parser.add_argument(
        "--enforce-publication-readiness",
        action="store_true",
        help=(
            "Require external benchmark-backed chemistry evidence tier on top of "
            "the selected claim/data/backend threshold checks."
        ),
    )
    parser.add_argument(
        "--enforce-preference-readiness",
        action="store_true",
        help=(
            "Require usable preference evidence for generator-level alignment onboarding "
            "(schema, counts, backend-supported coverage, and source hygiene)."
        ),
    )
    parser.add_argument(
        "--enforce-real-generation-readiness",
        action="store_true",
        help=(
            "Require Q15 real-use molecular generation evidence: raw-native quality, "
            "non-leaky splits, real backend coverage, generation-alignment ablations, "
            "and multi-seed support."
        ),
    )
    parser.add_argument(
        "--multi-seed-summary",
        default=None,
        help="Multi-seed summary JSON required by --enforce-real-generation-readiness.",
    )
    parser.add_argument(
        "--min-raw-native-valid-fraction",
        type=float,
        default=DEFAULT_REAL_GENERATION_THRESHOLDS["min_raw_native_valid_fraction"],
    )
    parser.add_argument(
        "--min-raw-native-graph-valid-fraction",
        type=float,
        default=DEFAULT_REAL_GENERATION_THRESHOLDS["min_raw_native_graph_valid_fraction"],
    )
    parser.add_argument(
        "--min-raw-native-pocket-contact-fraction",
        type=float,
        default=DEFAULT_REAL_GENERATION_THRESHOLDS["min_raw_native_pocket_contact_fraction"],
    )
    parser.add_argument(
        "--max-raw-native-clash-fraction",
        type=float,
        default=DEFAULT_REAL_GENERATION_THRESHOLDS["max_raw_native_clash_fraction"],
    )
    parser.add_argument(
        "--min-raw-native-unique-fraction",
        type=float,
        default=DEFAULT_REAL_GENERATION_THRESHOLDS["min_raw_native_unique_fraction"],
    )
    parser.add_argument(
        "--min-real-generation-seed-count",
        type=int,
        default=DEFAULT_REAL_GENERATION_THRESHOLDS["min_real_generation_seed_count"],
    )
    parser.add_argument(
        "--min-preference-profile-count",
        type=int,
        default=50,
        help="Minimum preference profile_count required under --enforce-preference-readiness.",
    )
    parser.add_argument(
        "--min-preference-pair-count",
        type=int,
        default=100,
        help="Minimum preference_pair_count required under --enforce-preference-readiness.",
    )
    parser.add_argument(
        "--min-backend-supported-pair-fraction",
        type=float,
        default=0.2,
        help="Minimum backend_supported_pair_fraction required under --enforce-preference-readiness.",
    )
    parser.add_argument(
        "--min-backend-based-source-count",
        type=int,
        default=1,
        help="Minimum backend_based source count required under --enforce-preference-readiness.",
    )
    parser.add_argument(
        "--q3-non-degradation-gate",
        default=None,
        help="Validate a Q3 non-degradation gate JSON instead of a claim artifact directory.",
    )
    return parser.parse_args(argv[1:])


def artifact_dir_from_config(config_path):
    config = load_json(config_path)
    try:
        return Path(config["research"]["training"]["checkpoint_dir"])
    except KeyError as exc:
        raise SystemExit(f"claim regression gate failed: config missing checkpoint_dir: {exc}") from exc


def _require_contract_number(thresholds, key, integer=False):
    value = thresholds.get(key)
    if integer:
        require(isinstance(value, int), f"claim contract threshold `{key}` must be an integer")
    else:
        require(finite_number(value), f"claim contract threshold `{key}` must be finite")
    return value


def load_claim_contract(path):
    require(path.is_file(), f"missing claim contract file: {path}")
    contract = load_json(path)
    require(contract.get("schema_version", 0) >= 1, "claim contract schema_version must be >= 1")
    claims = contract.get("claims")
    require(isinstance(claims, list) and claims, "claim contract must include non-empty claims")
    for index, entry in enumerate(claims):
        prefix = f"claim contract claims[{index}]"
        require(isinstance(entry.get("id"), str) and entry["id"], f"{prefix}.id must be a non-empty string")
        require(
            isinstance(entry.get("wording"), str) and entry["wording"],
            f"{prefix}.wording must be a non-empty string",
        )
        surfaces = entry.get("surfaces")
        require(isinstance(surfaces, list) and surfaces, f"{prefix}.surfaces must be non-empty")
        for surface in surfaces:
            require(isinstance(surface, str) and surface, f"{prefix}.surfaces must contain non-empty strings")
        metrics = entry.get("required_metrics")
        require(isinstance(metrics, list) and metrics, f"{prefix}.required_metrics must be non-empty")
        for metric_entry in metrics:
            require(
                isinstance(metric_entry.get("path"), str) and metric_entry["path"],
                f"{prefix}.required_metrics.path must be non-empty",
            )
            has_bound = any(bound in metric_entry for bound in ("min", "max", "equals"))
            require(
                has_bound,
                f"{prefix}.required_metrics entries must define at least one of min/max/equals",
            )

    thresholds = contract.get("thresholds")
    require(isinstance(thresholds, dict), "claim contract missing thresholds section")
    for key, default_value in DEFAULT_GATE_THRESHOLDS.items():
        _require_contract_number(thresholds, key, integer=isinstance(default_value, int))

    promotion = contract.get("promotion_decision")
    require(isinstance(promotion, dict), "claim contract missing promotion_decision")
    require(
        isinstance(promotion.get("criteria"), list) and promotion["criteria"],
        "claim contract promotion_decision.criteria must be non-empty",
    )

    baseline_matrix = contract.get("baseline_matrix")
    require(isinstance(baseline_matrix, dict), "claim contract missing baseline_matrix")
    surfaces = baseline_matrix.get("surfaces")
    require(isinstance(surfaces, list) and surfaces, "claim contract baseline_matrix.surfaces must be non-empty")
    for index, surface in enumerate(surfaces):
        require(
            isinstance(surface, str) and surface,
            f"claim contract baseline_matrix.surfaces[{index}] must be a non-empty string",
        )
    required_methods = baseline_matrix.get("required_methods")
    require(
        isinstance(required_methods, list) and required_methods,
        "claim contract baseline_matrix.required_methods must be non-empty",
    )
    for index, method in enumerate(required_methods):
        prefix = f"claim contract baseline_matrix.required_methods[{index}]"
        require(
            isinstance(method.get("method_id"), str) and method["method_id"],
            f"{prefix}.method_id must be a non-empty string",
        )
        require(
            isinstance(method.get("evidence_role"), str) and method["evidence_role"],
            f"{prefix}.evidence_role must be a non-empty string",
        )
    required_metric_fields = baseline_matrix.get("required_metric_fields")
    require(
        isinstance(required_metric_fields, list) and required_metric_fields,
        "claim contract baseline_matrix.required_metric_fields must be non-empty",
    )
    for index, field in enumerate(required_metric_fields):
        require(
            isinstance(field, str) and field,
            f"claim contract baseline_matrix.required_metric_fields[{index}] must be a non-empty string",
        )
    return contract


def apply_contract_threshold_overrides(args, contract):
    thresholds = contract["thresholds"]
    for field, default_value in DEFAULT_GATE_THRESHOLDS.items():
        current = getattr(args, field)
        if current == default_value:
            setattr(args, field, thresholds[field])


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
    test = claim.get("test", {})
    leakage_proxy = test.get("leakage_proxy_mean")
    require(finite_number(leakage_proxy), "test.leakage_proxy_mean is missing or non-finite")
    require(
        leakage_proxy <= args.max_leakage_proxy_mean,
        f"test.leakage_proxy_mean={leakage_proxy} above threshold {args.max_leakage_proxy_mean}",
    )
    candidate_valid_fraction = test.get("candidate_valid_fraction")
    require(finite_number(candidate_valid_fraction), "test.candidate_valid_fraction is missing or non-finite")
    require(
        candidate_valid_fraction >= args.min_candidate_valid_fraction,
        (
            f"test.candidate_valid_fraction={candidate_valid_fraction} "
            f"below threshold {args.min_candidate_valid_fraction}"
        ),
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


def iter_strings(payload):
    if isinstance(payload, str):
        yield payload
        return
    if isinstance(payload, dict):
        for value in payload.values():
            yield from iter_strings(value)
        return
    if isinstance(payload, list):
        for value in payload:
            yield from iter_strings(value)


def strict_preference_checks(artifact_dir, claim, args):
    profiles = []
    pairs = []
    schema_versions = []
    for split in ("validation", "test"):
        profile_path = artifact_dir / f"preference_profiles_{split}.json"
        pair_path = artifact_dir / f"preference_pairs_{split}.json"
        require(profile_path.is_file(), f"strict preference gate missing {profile_path.name}")
        require(pair_path.is_file(), f"strict preference gate missing {pair_path.name}")
        profile = load_json(profile_path)
        pair = load_json(pair_path)
        profiles.append(profile)
        pairs.append(pair)
        schema_versions.append(profile.get("schema_version", 0))
        schema_versions.append(pair.get("schema_version", 0))

    summary_path = artifact_dir / "preference_reranker_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        schema_versions.append(summary.get("schema_version", 0))

    for version in schema_versions:
        require(
            isinstance(version, (int, float)) and version >= 1,
            "strict preference gate requires preference artifact schema_version >= 1",
        )

    total_pairs = sum(int((payload or {}).get("pair_count", 0)) for payload in pairs)
    require(
        total_pairs >= args.strict_preference_min_pair_count,
        f"strict preference gate requires at least {args.strict_preference_min_pair_count} pairs",
    )

    source_coverage = {}
    for payload in pairs:
        for source, count in ((payload or {}).get("source_coverage") or {}).items():
            source_coverage[source] = source_coverage.get(source, 0) + int(count)
    allowed_sources = {
        "rule_based",
        "backend_based",
        "human_curated",
        "future_docking",
        "future_experimental",
    }
    unknown_sources = sorted(set(source_coverage) - allowed_sources)
    require(not unknown_sources, f"strict preference gate unknown source labels: {unknown_sources}")

    require(
        source_coverage.get("human_curated", 0) == 0,
        "strict preference gate blocks human_curated source without explicit human-labeled evidence contract",
    )
    require(
        source_coverage.get("future_experimental", 0) == 0,
        "strict preference gate blocks future_experimental source without experimental outcome evidence",
    )

    if source_coverage.get("future_docking", 0) > 0:
        docking = (claim.get("backend_metrics") or {}).get("docking_affinity") or {}
        docking_metrics = docking.get("metrics") or {}
        scored = docking_metrics.get("backend_examples_scored", 0.0)
        require(
            isinstance(scored, (int, float)) and scored > 0,
            "strict preference gate blocks future_docking source without docking backend coverage",
        )

    claim_text = "\n".join(value.lower() for value in iter_strings(claim))
    forbidden = {
        "human-aligned": source_coverage.get("human_curated", 0) > 0,
        "experimental preference": source_coverage.get("future_experimental", 0) > 0,
        "docking preference trained": source_coverage.get("future_docking", 0) > 0,
    }
    for phrase, allowed in forbidden.items():
        if phrase in claim_text:
            require(
                allowed,
                f"strict preference gate forbids wording `{phrase}` without matching preference evidence source coverage",
            )


def validate_publication_readiness(claim):
    chemistry = claim.get("chemistry_novelty_diversity") or {}
    benchmark = chemistry.get("benchmark_evidence") or {}
    tier = benchmark.get("evidence_tier")
    require(
        tier == "external_benchmark_backed",
        "publication-readiness gate requires chemistry benchmark evidence_tier=external_benchmark_backed",
    )


def validate_preference_readiness(claim, args):
    preference = (claim.get("method_comparison") or {}).get("preference_alignment")
    require(
        isinstance(preference, dict) and preference,
        "preference-readiness gate requires method_comparison.preference_alignment",
    )
    require(
        preference.get("schema_version", 0) >= 1,
        "preference-readiness gate requires preference_alignment.schema_version >= 1",
    )
    require(
        preference.get("artifact_interpretation") != "unavailable",
        "preference-readiness gate requires available preference artifacts (not unavailable)",
    )
    profile_count = preference.get("profile_count")
    pair_count = preference.get("preference_pair_count")
    backend_fraction = preference.get("backend_supported_pair_fraction")
    source_breakdown = preference.get("source_breakdown") or {}

    pair_artifact_name = preference.get("preference_pair_artifact")
    if not isinstance(pair_artifact_name, str) or not pair_artifact_name:
        pair_artifact_name = "preference_pairs_test.json"
    pair_artifact_path = Path(claim.get("artifact_dir", ".")).joinpath(pair_artifact_name)
    if not pair_artifact_path.is_file():
        pair_artifact_path = Path("checkpoints").joinpath(pair_artifact_name)
    if pair_artifact_path.is_file():
        pair_payload = load_json(pair_artifact_path)
        if not isinstance(pair_count, int):
            pair_count = pair_payload.get("pair_count")
        if not finite_number(backend_fraction):
            backend_fraction = pair_payload.get("backend_supported_pair_fraction")
        if not source_breakdown:
            source_breakdown = pair_payload.get("source_coverage") or {}

    profile_artifact_name = preference.get("profile_artifact")
    if not isinstance(profile_artifact_name, str) or not profile_artifact_name:
        profile_artifact_name = "preference_profiles_test.json"
    profile_artifact_path = Path(claim.get("artifact_dir", ".")).joinpath(profile_artifact_name)
    if not profile_artifact_path.is_file():
        profile_artifact_path = Path("checkpoints").joinpath(profile_artifact_name)
    if profile_artifact_path.is_file() and not isinstance(profile_count, int):
        profile_payload = load_json(profile_artifact_path)
        profile_count = profile_payload.get("profile_count")
    require(
        isinstance(profile_count, int),
        "preference-readiness gate requires integer preference_alignment.profile_count",
    )
    require(
        isinstance(pair_count, int),
        "preference-readiness gate requires integer preference_alignment.preference_pair_count",
    )
    require(
        profile_count >= args.min_preference_profile_count,
        (
            "preference-readiness gate requires "
            f"profile_count >= {args.min_preference_profile_count} (got {profile_count})"
        ),
    )
    require(
        pair_count >= args.min_preference_pair_count,
        (
            "preference-readiness gate requires "
            f"preference_pair_count >= {args.min_preference_pair_count} (got {pair_count})"
        ),
    )
    require(
        finite_number(backend_fraction),
        "preference-readiness gate requires finite backend_supported_pair_fraction",
    )
    require(
        backend_fraction >= args.min_backend_supported_pair_fraction,
        (
            "preference-readiness gate requires "
            f"backend_supported_pair_fraction >= {args.min_backend_supported_pair_fraction} "
            f"(got {backend_fraction})"
        ),
    )
    backend_based = int(source_breakdown.get("backend_based", 0))
    require(
        backend_based >= args.min_backend_based_source_count,
        (
            "preference-readiness gate requires "
            f"source_breakdown.backend_based >= {args.min_backend_based_source_count} "
            f"(got {backend_based})"
        ),
    )
    require(
        int(source_breakdown.get("human_curated", 0)) == 0,
        "preference-readiness gate blocks human_curated source without explicit human-label contract",
    )
    require(
        int(source_breakdown.get("future_experimental", 0)) == 0,
        "preference-readiness gate blocks future_experimental source without experimental outcomes",
    )


def _require_finite_at_least(value, threshold, label):
    require(finite_number(value), f"{label} is missing or non-finite")
    require(value >= threshold, f"{label}={value} below threshold {threshold}")


def _require_finite_at_most(value, threshold, label):
    require(finite_number(value), f"{label} is missing or non-finite")
    require(value <= threshold, f"{label}={value} above threshold {threshold}")


def _require_nonheuristic_backend(claim, section):
    backend = (claim.get("backend_metrics") or {}).get(section) or {}
    require(backend.get("available") is True, f"real-generation gate requires available {section} backend")
    backend_name = str(backend.get("backend_name") or "").lower()
    status = str(backend.get("status") or "").lower()
    require(backend_name, f"real-generation gate requires {section}.backend_name")
    require(
        "heuristic" not in backend_name and "heuristic" not in status,
        f"real-generation gate rejects heuristic-only {section} backend `{backend.get('backend_name')}`",
    )
    metrics = backend.get("metrics") or {}
    if "backend_examples_scored" in metrics:
        _require_finite_at_least(
            metrics.get("backend_examples_scored"),
            1.0,
            f"backend_metrics.{section}.backend_examples_scored",
        )


def _require_clean_split_leakage(split):
    leakage = split.get("leakage_checks") or {}
    for key in (
        "protein_overlap_detected",
        "pocket_overlap_detected",
        "pocket_family_proxy_overlap_detected",
        "duplicate_example_ids_detected",
    ):
        if key in leakage:
            require(leakage.get(key) is False, f"real-generation gate split leakage check failed: {key}")
    for key in (
        "train_val_protein_overlap",
        "train_test_protein_overlap",
        "val_test_protein_overlap",
        "train_val_pocket_overlap",
        "train_test_pocket_overlap",
        "val_test_pocket_overlap",
        "duplicated_example_ids",
    ):
        if key in leakage:
            require(leakage.get(key) == 0, f"real-generation gate split leakage count nonzero: {key}")


def _evaluation_matrix_test_row(experiment):
    matrix = experiment.get("evaluation_matrix") or {}
    rows = matrix.get("rows") or []
    for row in rows:
        if row.get("split_type") == "unseen_pocket_test" or row.get("split_label") == "unseen_pocket_test":
            return row
    return None


def _validate_evaluation_matrix(experiment, args):
    row = _evaluation_matrix_test_row(experiment)
    require(row is not None, "real-generation gate requires evaluation_matrix unseen_pocket_test row")
    require(row.get("evaluation_status") == "evaluated", "unseen_pocket_test evaluation row must be evaluated")
    quality = row.get("quality") or {}
    efficiency = row.get("efficiency") or {}
    _require_finite_at_least(
        quality.get("raw_valid_fraction"),
        args.min_raw_native_valid_fraction,
        "evaluation_matrix.unseen_pocket_test.quality.raw_valid_fraction",
    )
    _require_finite_at_least(
        quality.get("raw_pocket_contact_fraction"),
        args.min_raw_native_pocket_contact_fraction,
        "evaluation_matrix.unseen_pocket_test.quality.raw_pocket_contact_fraction",
    )
    _require_finite_at_most(
        quality.get("raw_clash_fraction"),
        args.max_raw_native_clash_fraction,
        "evaluation_matrix.unseen_pocket_test.quality.raw_clash_fraction",
    )
    _require_finite_at_least(
        efficiency.get("examples_per_second"),
        0.0,
        "evaluation_matrix.unseen_pocket_test.efficiency.examples_per_second",
    )
    require(
        efficiency.get("no_grad") is True,
        "evaluation_matrix.unseen_pocket_test.efficiency.no_grad must be true",
    )


def _validate_raw_native_generation_report(artifact_dir, args):
    report_path = artifact_dir / "raw_native_generation_report.json"
    require(report_path.is_file(), "real-generation gate requires raw_native_generation_report.json")
    report = load_json(report_path)
    require(report.get("schema_version", 0) >= 1, "raw_native_generation_report schema_version must be >= 1")
    raw = report.get("raw_native") or {}
    processed = report.get("processed") or {}
    eligibility = report.get("claim_eligibility") or {}
    require(raw.get("model_native_raw") is True, "raw_native_generation_report.raw_native must be model-native")
    require(
        processed.get("model_native_raw") is False,
        "raw_native_generation_report.processed must remain additive processed evidence",
    )
    require(
        eligibility.get("processed_evidence_additive_only") is True,
        "raw_native_generation_report claim eligibility must keep processed evidence additive-only",
    )
    require(
        eligibility.get("status") in {"supported", "weakly_supported"},
        "raw_native_generation_report does not support raw-native generation claims",
    )
    _require_finite_at_least(
        raw.get("valid_fraction"),
        args.min_raw_native_valid_fraction,
        "raw_native_generation_report.raw_native.valid_fraction",
    )
    _require_finite_at_least(
        raw.get("pocket_contact_fraction"),
        args.min_raw_native_pocket_contact_fraction,
        "raw_native_generation_report.raw_native.pocket_contact_fraction",
    )
    _require_finite_at_most(
        raw.get("clash_fraction"),
        args.max_raw_native_clash_fraction,
        "raw_native_generation_report.raw_native.clash_fraction",
    )
    _require_finite_at_least(
        raw.get("validity_conditioned_unique_fraction"),
        args.min_raw_native_unique_fraction,
        "raw_native_generation_report.raw_native.validity_conditioned_unique_fraction",
    )
    require(
        isinstance(report.get("objective_families"), list) and report["objective_families"],
        "raw_native_generation_report requires non-empty objective_families",
    )


def _validate_generation_alignment_ablation_matrix(artifact_dir, args):
    matrix_path = artifact_dir / "ablation_matrix_summary.json"
    require(matrix_path.is_file(), "real-generation gate requires ablation_matrix_summary.json")
    matrix = load_json(matrix_path)
    variants = matrix.get("variants") or []
    require(isinstance(variants, list) and variants, "ablation matrix variants must be non-empty")
    by_label = {variant.get("variant_label"): variant for variant in variants if isinstance(variant, dict)}
    required_labels = {
        "flow_head_equivariant_geometry",
        "rollout_training_disabled",
        "chemistry_native_constraints_disabled",
        "pocket_interaction_thin_contact_loss",
        "direct_fusion_negative_control",
    }
    missing = sorted(required_labels.difference(by_label))
    require(not missing, f"real-generation gate missing Q15 ablation variants: {missing}")
    for label in sorted(required_labels):
        variant = by_label[label]
        raw = variant.get("raw_generation_quality") or {}
        runtime = variant.get("runtime") or {}
        objective_families = variant.get("objective_families") or []
        _require_finite_at_least(
            raw.get("raw_valid_fraction"),
            0.0,
            f"ablation_matrix.{label}.raw_generation_quality.raw_valid_fraction",
        )
        _require_finite_at_least(
            runtime.get("examples_per_second"),
            0.0,
            f"ablation_matrix.{label}.runtime.examples_per_second",
        )
        require(
            isinstance(objective_families, list) and objective_families,
            f"ablation_matrix.{label}.objective_families must be non-empty",
        )
        families = {entry.get("family") for entry in objective_families if isinstance(entry, dict)}
        require("task" in families, f"ablation_matrix.{label}.objective_families missing task family")


def _multi_seed_count(summary):
    seed_count = summary.get("seed_count")
    if isinstance(seed_count, int):
        return seed_count
    seeds = summary.get("seeds")
    if isinstance(seeds, list):
        return len(seeds)
    per_seed = summary.get("per_seed_summary")
    if isinstance(per_seed, list):
        return len(per_seed)
    results = summary.get("multi_seed_results")
    if isinstance(results, list):
        return len(results)
    return 0


def _validate_multi_seed_summary(args):
    require(args.multi_seed_summary, "real-generation gate requires --multi-seed-summary")
    path = Path(args.multi_seed_summary)
    require(path.is_file(), f"missing multi-seed summary: {path}")
    summary = load_json(path)
    seed_count = _multi_seed_count(summary)
    require(
        seed_count >= args.min_real_generation_seed_count,
        f"real-generation gate requires at least {args.min_real_generation_seed_count} seeds (got {seed_count})",
    )


def validate_real_generation_readiness(artifact_dir, claim, experiment, split, args):
    raw = claim.get("raw_native_evidence")
    require(isinstance(raw, dict) and raw, "real-generation gate requires raw_native_evidence")
    require(raw.get("model_native_raw") is True, "raw_native_evidence must be model-native raw")
    require(
        raw.get("raw_model_layer") in {"raw_rollout", "raw_flow", "raw_native_graph_extraction"},
        f"raw_native_evidence uses unsupported raw_model_layer={raw.get('raw_model_layer')}",
    )
    require(raw.get("candidate_count", 0) > 0, "raw_native_evidence requires raw candidates")
    _require_finite_at_least(
        raw.get("valid_fraction"),
        args.min_raw_native_valid_fraction,
        "raw_native_evidence.valid_fraction",
    )
    _require_finite_at_least(
        raw.get("native_graph_valid_fraction"),
        args.min_raw_native_graph_valid_fraction,
        "raw_native_evidence.native_graph_valid_fraction",
    )
    _require_finite_at_least(
        raw.get("pocket_contact_fraction"),
        args.min_raw_native_pocket_contact_fraction,
        "raw_native_evidence.pocket_contact_fraction",
    )
    _require_finite_at_most(
        raw.get("clash_fraction"),
        args.max_raw_native_clash_fraction,
        "raw_native_evidence.clash_fraction",
    )
    _require_clean_split_leakage(split)
    for section in ("chemistry_validity", "docking_affinity", "pocket_compatibility"):
        _require_nonheuristic_backend(claim, section)
    _validate_evaluation_matrix(experiment, args)
    _validate_raw_native_generation_report(artifact_dir, args)
    _validate_generation_alignment_ablation_matrix(artifact_dir, args)
    _validate_multi_seed_summary(args)


def _value_at_path(payload, dotted_path):
    current = payload
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _contract_surface_labels(artifact_dir):
    return {
        str(artifact_dir).replace("\\", "/"),
        artifact_dir.name,
        f"checkpoints/{artifact_dir.name}",
    }


def _normalize_token(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def validate_baseline_matrix(contract, artifact_dir, claim):
    matrix = contract.get("baseline_matrix", {})
    configured_surfaces = set(matrix.get("surfaces", []))
    if configured_surfaces and not (configured_surfaces & _contract_surface_labels(artifact_dir)):
        return
    required_methods = matrix.get("required_methods", [])
    required_metric_fields = matrix.get("required_metric_fields", [])
    method_comparison = claim.get("method_comparison") or {}
    methods = method_comparison.get("methods") or []
    require(isinstance(methods, list) and methods, "claim missing method_comparison.methods")
    by_id = {row.get("method_id"): row for row in methods if isinstance(row, dict)}
    active_method = (method_comparison.get("active_method") or {}).get("method_id")
    require(isinstance(active_method, str) and active_method, "claim missing method_comparison.active_method.method_id")
    active_row = by_id.get(active_method)
    require(active_row is not None, "active method is absent from method_comparison.methods")

    required_ids = [entry["method_id"] for entry in required_methods]
    for entry in required_methods:
        method_id = entry["method_id"]
        row = by_id.get(method_id)
        require(row is not None, f"baseline matrix required method missing: {method_id}")
        require(row.get("available") is True, f"baseline matrix method `{method_id}` is unavailable")
        expected_role = _normalize_token(entry["evidence_role"])
        actual_role = _normalize_token(row.get("evidence_role", ""))
        require(
            actual_role == expected_role,
            f"baseline matrix method `{method_id}` evidence_role `{row.get('evidence_role')}` != `{entry['evidence_role']}`",
        )
        if entry.get("trainable") is not None:
            require(
                bool(row.get("trainable")) is bool(entry["trainable"]),
                f"baseline matrix method `{method_id}` trainable mismatch",
            )
        for field in required_metric_fields:
            value = row.get(field)
            require(
                value is not None,
                f"baseline matrix method `{method_id}` missing required metric field `{field}`",
            )

    if matrix.get("require_matched_sampling_steps", False):
        target_steps = active_row.get("sampling_steps")
        require(isinstance(target_steps, int), "active method sampling_steps missing or non-integer")
        for method_id in required_ids:
            steps = by_id.get(method_id, {}).get("sampling_steps")
            require(
                isinstance(steps, int) and steps == target_steps,
                f"baseline matrix method `{method_id}` sampling_steps={steps} expected {target_steps}",
            )


def validate_claim_contract_mappings(contract, artifact_dir, claim):
    surface_labels = _contract_surface_labels(artifact_dir)
    claims = contract.get("claims", [])
    for entry in claims:
        surfaces = set(entry.get("surfaces", []))
        if not (surfaces & surface_labels):
            continue
        for metric_entry in entry.get("required_metrics", []):
            metric_path = metric_entry["path"]
            value = _value_at_path(claim, metric_path)
            require(value is not None, f"claim `{entry['id']}` missing required metric `{metric_path}`")
            if "equals" in metric_entry:
                require(
                    value == metric_entry["equals"],
                    f"claim `{entry['id']}` metric `{metric_path}` expected {metric_entry['equals']} got {value}",
                )
            if "min" in metric_entry:
                require(
                    finite_number(value) and value >= metric_entry["min"],
                    f"claim `{entry['id']}` metric `{metric_path}` below minimum {metric_entry['min']}",
                )
            if "max" in metric_entry:
                require(
                    finite_number(value) and value <= metric_entry["max"],
                    f"claim `{entry['id']}` metric `{metric_path}` above maximum {metric_entry['max']}",
                )


def validate_raw_native_claim_gate(claim):
    raw = claim.get("raw_native_evidence")
    if not isinstance(raw, dict) or not raw:
        return
    require(raw.get("valid_fraction") is not None, "raw_native_evidence missing valid_fraction")
    require(
        raw.get("native_graph_valid_fraction") is not None,
        "raw_native_evidence missing native_graph_valid_fraction",
    )
    require(raw.get("mean_centroid_offset") is not None, "raw_native_evidence missing mean_centroid_offset")
    require(raw.get("pocket_contact_fraction") is not None, "raw_native_evidence missing pocket_contact_fraction")
    require(raw.get("leakage_proxy_mean") is not None, "raw_native_evidence missing leakage_proxy_mean")
    gate = raw.get("raw_native_gate") or {}
    if gate:
        require(
            gate.get("processed_metrics_excluded") is True,
            "raw_native_gate must exclude processed/repaired/reranked metrics",
        )
        require(
            gate.get("passed") is not False,
            "raw_native_gate failed independently of processed metrics: "
            + "; ".join(str(reason) for reason in gate.get("failed_reasons", [])),
        )


def validate_artifact_dir(artifact_dir, args, claim_contract):
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
    validate_raw_native_claim_gate(claim)

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

    preference = claim.get("method_comparison", {}).get("preference_alignment")
    if preference:
        require(
            preference.get("schema_version", 0) >= 1,
            "method_comparison.preference_alignment schema_version must be >= 1",
        )
        require(
            preference.get("missing_artifacts_mean_unavailable") is True,
            "missing preference artifacts must mean unavailable evidence, not failed alignment",
        )
        require(
            isinstance(preference.get("profile_count", 0), int),
            "preference_alignment.profile_count must be an integer",
        )
        require(
            isinstance(preference.get("preference_pair_count", 0), int),
            "preference_alignment.preference_pair_count must be an integer",
        )
        if args.strict_preference_gate:
            strict_preference_checks(artifact_dir, claim, args)
    elif args.strict_preference_gate:
        require(False, "strict preference gate requires method_comparison.preference_alignment")

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
    if args.enforce_publication_readiness:
        validate_publication_readiness(claim)
    if args.enforce_preference_readiness:
        validate_preference_readiness(claim, args)
    if args.enforce_real_generation_readiness:
        validate_real_generation_readiness(artifact_dir, claim, experiment, split, args)
    validate_baseline_matrix(claim_contract, artifact_dir, claim)
    validate_claim_contract_mappings(claim_contract, artifact_dir, claim)

    print(f"claim regression gate passed: {artifact_dir}")


def _gate_number(value):
    return finite_number(value) or value is None


def validate_q3_non_degradation_gate(path):
    gate = load_json(path)
    require(gate.get("schema_version", 0) >= 1, "q3 gate schema_version must be >= 1")
    leaderboard_path = Path(gate.get("leaderboard_path", ""))
    require(leaderboard_path.is_file(), f"missing q3 leaderboard: {leaderboard_path}")
    leaderboard = load_json(leaderboard_path)
    rows = leaderboard.get("rows")
    require(isinstance(rows, list) and rows, "q3 leaderboard must contain rows")

    thresholds = gate.get("thresholds", {})
    max_vina_delta = thresholds.get("max_constrained_vina_degradation", 5.0)
    max_gnina_delta = thresholds.get("max_constrained_gnina_degradation", 5.0)
    require(finite_number(max_vina_delta), "max_constrained_vina_degradation must be finite")
    require(finite_number(max_gnina_delta), "max_constrained_gnina_degradation must be finite")

    failures = []
    warnings = []
    raw_by_method = {}
    for row in rows:
        role = row.get("layer_role")
        metrics = row.get("metrics", {})
        if row.get("model_improvement") is True and role != "raw_model_native":
            failures.append(f"{row.get('method_id')}:{row.get('layer')} marks non-raw row as model improvement")
        if row.get("layer") in {"repaired", "reranked", "backend_aware_posthoc", "full_repair", "repair_rejected"}:
            if row.get("model_improvement") is True:
                failures.append(f"{row.get('method_id')}:{row.get('layer')} repaired/reranked row is model improvement")
        for metric_name, value in metrics.items():
            if not _gate_number(value):
                failures.append(f"{row.get('method_id')}:{row.get('layer')} metric {metric_name} is not finite/null")
        if role == "raw_model_native" and row.get("status") == "scored":
            raw_by_method.setdefault(row.get("method_id"), row)

    for row in rows:
        if row.get("layer_role") != "constrained_sampling" or row.get("status") != "scored":
            continue
        raw = raw_by_method.get(row.get("method_id"))
        if not raw:
            continue
        raw_metrics = raw.get("metrics", {})
        metrics = row.get("metrics", {})
        vina_delta = _numeric_delta(metrics.get("vina_score"), raw_metrics.get("vina_score"))
        gnina_delta = _numeric_delta(metrics.get("gnina_affinity"), raw_metrics.get("gnina_affinity"))
        if vina_delta is not None and vina_delta > max_vina_delta:
            warnings.append(
                f"{row.get('method_id')} constrained_flow Vina degradation {vina_delta:.4g} exceeds {max_vina_delta:.4g}"
            )
        if gnina_delta is not None and gnina_delta > max_gnina_delta:
            warnings.append(
                f"{row.get('method_id')} constrained_flow GNINA degradation {gnina_delta:.4g} exceeds {max_gnina_delta:.4g}"
            )

    for claim in gate.get("claim_table", []):
        evidence_type = claim.get("evidence_type")
        claim_type = claim.get("claim_type")
        if evidence_type == "proxy_only" and claim_type == "docking_improvement":
            failures.append(f"claim {claim.get('id')} marks proxy-only gain as docking improvement")
        text = json.dumps(claim, sort_keys=True)
        if "0.XXX" in text or "~" in text:
            failures.append(f"claim {claim.get('id')} contains placeholder-like numeric text")

    report = {
        "schema_version": 1,
        "artifact_name": "q3_non_degradation_gate_report",
        "status": "fail" if failures else "pass",
        "failures": failures,
        "warnings": warnings,
        "leaderboard_path": str(leaderboard_path),
    }
    output_path = Path(gate.get("output_report", "configs/q3_non_degradation_gate_report.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require(not failures, "q3 non-degradation gate failed: " + "; ".join(failures[:5]))
    print(f"q3 non-degradation gate passed: {path}")


def _numeric_delta(value, baseline):
    if finite_number(value) and finite_number(baseline):
        return value - baseline
    return None


def main(argv):
    args = parse_args(argv)
    if args.q3_non_degradation_gate:
        validate_q3_non_degradation_gate(Path(args.q3_non_degradation_gate))
        return
    claim_contract = load_claim_contract(Path(args.claim_contract))
    apply_contract_threshold_overrides(args, claim_contract)
    config_path = Path(args.config)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else artifact_dir_from_config(config_path)
    if args.run:
        run_experiment(args)
    validate_artifact_dir(artifact_dir, args, claim_contract)


if __name__ == "__main__":
    main(sys.argv)
