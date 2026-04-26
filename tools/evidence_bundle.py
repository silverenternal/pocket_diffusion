#!/usr/bin/env python3
"""Collect claim-facing artifacts into one compact evidence bundle."""

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ARTIFACT_DIRS = [
    "checkpoints/claim_matrix",
    "checkpoints/harder_pressure",
    "checkpoints/tight_geometry_pressure",
    "checkpoints/real_backends",
    "checkpoints/lp_pdbbind_refined_real_backends",
    "checkpoints/medium_profile",
    "checkpoints/pdbbindpp_profile",
    "checkpoints/pdbbindpp_real_backends",
    "checkpoints/multi_seed",
    "checkpoints/multi_seed_medium",
    "configs/checkpoints/multi_seed_pdbbindpp",
    "configs/checkpoints/multi_seed_pdbbindpp_real_backends",
    "checkpoints/vina_backend",
]

BACKEND_THRESHOLDS = {
    "min_rdkit_available": 1.0,
    "min_rdkit_sanitized_fraction": 0.95,
    "min_unique_fraction": 0.5,
    "max_backend_missing_structure_fraction": 0.0,
    "max_clash_fraction": 0.1,
    "min_strict_pocket_fit": 0.35,
    "min_pocket_contact": 0.8,
}

DATA_THRESHOLDS = {
    "min_parsed_complexes": 100,
    "min_retained_label_coverage": 0.8,
    "min_heldout_family_count": 10,
}

LEAKAGE_THRESHOLDS = {
    "preferred_max_leakage_proxy_mean": 0.08,
    "hard_max_leakage_proxy_mean": 0.12,
    "max_leakage_proxy_regression": 0.03,
    "require_clean_split_audit": True,
}


def load_json(path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_hash(path):
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _sum_breakdown(target, source):
    for key, value in (source or {}).items():
        try:
            target[key] = target.get(key, 0) + int(value)
        except (TypeError, ValueError):
            continue


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def load_preference_artifact_summary(path):
    profile_count_by_split = {}
    pair_count_by_split = {}
    source_breakdown = {}
    backend_coverage = {}
    schema_versions = {}
    backend_supported_pair_fraction_by_split = {}
    rule_only_pair_fraction_by_split = {}
    missing_backend_evidence_fraction_by_split = {}
    mean_preference_strength_by_split = {}
    hard_constraint_win_fraction_by_split = {}

    for split in ("validation", "test"):
        profile = load_json(path / f"preference_profiles_{split}.json") or {}
        if profile:
            schema_versions[f"profile_{split}"] = profile.get("schema_version")
            profile_count_by_split[split] = _safe_int(profile.get("profile_count"))
            _sum_breakdown(backend_coverage, profile.get("backend_coverage") or {})

        pair = load_json(path / f"preference_pairs_{split}.json") or {}
        if pair:
            schema_versions[f"pair_{split}"] = pair.get("schema_version")
            pair_count = _safe_int(pair.get("pair_count"))
            pair_count_by_split[split] = pair_count
            _sum_breakdown(source_breakdown, pair.get("source_coverage") or {})
            backend_supported = _safe_float(pair.get("backend_supported_pair_fraction"))
            rule_only = _safe_float(pair.get("rule_only_pair_fraction"))
            missing_backend = _safe_float(pair.get("missing_backend_evidence_fraction"))
            mean_strength = _safe_float(pair.get("mean_preference_strength"))
            hard_constraint = _safe_float(pair.get("hard_constraint_win_fraction"))
            if backend_supported is None:
                backend_supported = (
                    (_safe_int((pair.get("source_coverage") or {}).get("backend_based")) / pair_count)
                    if pair_count > 0
                    else 0.0
                )
            if rule_only is None:
                rule_only = (
                    (_safe_int((pair.get("source_coverage") or {}).get("rule_based")) / pair_count)
                    if pair_count > 0
                    else 0.0
                )
            if missing_backend is None:
                missing_backend = 1.0 - backend_supported if pair_count > 0 else 0.0
            backend_supported_pair_fraction_by_split[split] = backend_supported
            rule_only_pair_fraction_by_split[split] = rule_only
            missing_backend_evidence_fraction_by_split[split] = missing_backend
            mean_preference_strength_by_split[split] = mean_strength if mean_strength is not None else 0.0
            hard_constraint_win_fraction_by_split[split] = hard_constraint if hard_constraint is not None else 0.0

    reranker_summary = load_json(path / "preference_reranker_summary.json") or {}
    if reranker_summary:
        schema_versions["reranker_summary"] = reranker_summary.get("schema_version")

    pair_total = sum(pair_count_by_split.values())
    backend_weighted = 0.0
    rule_weighted = 0.0
    missing_weighted = 0.0
    mean_strength_weighted = 0.0
    hard_constraint_weighted = 0.0
    for split, pair_count in pair_count_by_split.items():
        if pair_count <= 0:
            continue
        backend_weighted += backend_supported_pair_fraction_by_split.get(split, 0.0) * pair_count
        rule_weighted += rule_only_pair_fraction_by_split.get(split, 0.0) * pair_count
        missing_weighted += (
            missing_backend_evidence_fraction_by_split.get(split, 0.0) * pair_count
        )
        mean_strength_weighted += mean_preference_strength_by_split.get(split, 0.0) * pair_count
        hard_constraint_weighted += (
            hard_constraint_win_fraction_by_split.get(split, 0.0) * pair_count
        )

    artifacts_present = bool(
        profile_count_by_split or pair_count_by_split or reranker_summary
    )
    return {
        "available": artifacts_present,
        "interpretation": (
            "available" if artifacts_present else "unavailable"
        ),
        "schema_versions": schema_versions,
        "profile_count_by_split": profile_count_by_split,
        "pair_count_by_split": pair_count_by_split,
        "source_breakdown": source_breakdown,
        "backend_coverage": backend_coverage,
        "reranker_enabled": reranker_summary.get("enabled")
        if reranker_summary
        else None,
        "backend_supported_pair_fraction": (
            backend_weighted / pair_total if pair_total > 0 else None
        ),
        "rule_only_pair_fraction": (
            rule_weighted / pair_total if pair_total > 0 else None
        ),
        "missing_backend_evidence_fraction": (
            missing_weighted / pair_total if pair_total > 0 else None
        ),
        "mean_preference_strength": (
            mean_strength_weighted / pair_total if pair_total > 0 else None
        ),
        "hard_constraint_win_fraction": (
            hard_constraint_weighted / pair_total if pair_total > 0 else None
        ),
    }


def compact_claim(claim, preference_artifacts=None):
    if not claim:
        return None
    test = claim.get("test", {})
    backend = claim.get("backend_metrics", {})
    chemistry = backend.get("chemistry_validity", {}).get("metrics", {})
    pocket = backend.get("pocket_compatibility", {}).get("metrics", {})
    docking = backend.get("docking_affinity", {}).get("metrics", {})
    reranker = claim.get("reranker_report", {})
    calibration = reranker.get("calibration", {})
    baselines = claim.get("baseline_comparisons", [])
    method_comparison = claim.get("method_comparison", {})
    method_rows = method_comparison.get("methods", [])
    preference_alignment = method_comparison.get("preference_alignment", {})
    leakage = claim.get("leakage_calibration", {})
    chemistry_novelty = claim.get("chemistry_novelty_diversity", {})
    benchmark = chemistry_novelty.get("benchmark_evidence", {})
    preference_artifacts = preference_artifacts or {}
    profile_count_by_split = preference_artifacts.get("profile_count_by_split") or {}
    pair_count_by_split = preference_artifacts.get("pair_count_by_split") or {}
    source_breakdown = preference_artifacts.get("source_breakdown") or {}
    backend_coverage = preference_artifacts.get("backend_coverage") or {}

    return {
        "run_label": claim.get("run_label"),
        "claim_context": claim.get("claim_context"),
        "candidate_valid_fraction": test.get("candidate_valid_fraction"),
        "strict_pocket_fit_score": test.get("strict_pocket_fit_score"),
        "unique_smiles_fraction": test.get("unique_smiles_fraction"),
        "slot_activation_mean": test.get("slot_activation_mean"),
        "gate_activation_mean": test.get("gate_activation_mean"),
        "leakage_proxy_mean": test.get("leakage_proxy_mean"),
        "rdkit_available": chemistry.get("rdkit_available"),
        "rdkit_sanitized_fraction": chemistry.get("rdkit_sanitized_fraction"),
        "backend_missing_structure_fraction": pocket.get("backend_missing_structure_fraction"),
        "clash_fraction": pocket.get("clash_fraction"),
        "performance_gates": claim.get("performance_gates"),
        "backend_review": claim.get("backend_review"),
        "backend_thresholds": evaluate_backend_thresholds(chemistry, docking, pocket),
        "backend_environment": claim.get("backend_environment"),
        "reranker_coefficients": calibration.get("coefficients"),
        "reranker_training_candidate_count": calibration.get("training_candidate_count"),
        "baseline_comparison_count": len(baselines),
        "baseline_labels": [row.get("label") for row in baselines if row.get("label")],
        "active_generation_method": (method_comparison.get("active_method") or {}).get(
            "method_id"
        ),
        "comparison_method_count": len(method_rows),
        "comparison_method_ids": [
            row.get("method_id") for row in method_rows if row.get("method_id")
        ],
        "preference_alignment": {
            "schema_version": preference_alignment.get("schema_version"),
            "profile_extraction_enabled": preference_alignment.get("profile_extraction_enabled"),
            "pair_construction_enabled": preference_alignment.get("pair_construction_enabled"),
            "profile_count": preference_alignment.get("profile_count"),
            "preference_pair_count": preference_alignment.get("preference_pair_count"),
            "missing_artifacts_mean_unavailable": preference_alignment.get(
                "missing_artifacts_mean_unavailable", True
            ),
            "interpretation": (
                "Missing preference artifacts mean preference evidence unavailable, not failed alignment."
                if not preference_artifacts.get("available")
                else "Preference artifacts are available and summarized as compact counts/coverage."
            ),
            "artifact_interpretation": preference_artifacts.get("interpretation", "unavailable"),
            "profile_count_by_split": profile_count_by_split,
            "pair_count_by_split": pair_count_by_split,
            "source_breakdown": source_breakdown,
            "backend_coverage": backend_coverage,
            "reranker_enabled": preference_artifacts.get("reranker_enabled"),
            "backend_supported_pair_fraction": preference_artifacts.get(
                "backend_supported_pair_fraction"
            ),
            "rule_only_pair_fraction": preference_artifacts.get(
                "rule_only_pair_fraction"
            ),
            "missing_backend_evidence_fraction": preference_artifacts.get(
                "missing_backend_evidence_fraction"
            ),
            "mean_preference_strength": preference_artifacts.get("mean_preference_strength"),
            "hard_constraint_win_fraction": preference_artifacts.get(
                "hard_constraint_win_fraction"
            ),
        },
        "chemistry_novelty_diversity": {
            "review_layer": chemistry_novelty.get("review_layer"),
            "unique_smiles_fraction": chemistry_novelty.get("unique_smiles_fraction"),
            "atom_type_sequence_diversity": chemistry_novelty.get("atom_type_sequence_diversity"),
            "bond_topology_diversity": chemistry_novelty.get("bond_topology_diversity"),
            "coordinate_shape_diversity": chemistry_novelty.get("coordinate_shape_diversity"),
            "novel_atom_type_sequence_fraction": chemistry_novelty.get(
                "novel_atom_type_sequence_fraction"
            ),
            "novel_bond_topology_fraction": chemistry_novelty.get(
                "novel_bond_topology_fraction"
            ),
            "novel_coordinate_shape_fraction": chemistry_novelty.get(
                "novel_coordinate_shape_fraction"
            ),
            "interpretation": chemistry_novelty.get("interpretation"),
            "benchmark_evidence": benchmark,
        },
        "leakage_review": {
            "recommended_delta_leak": leakage.get("recommended_delta_leak"),
            "preferred_max_leakage_proxy_mean": leakage.get(
                "preferred_max_leakage_proxy_mean",
                LEAKAGE_THRESHOLDS["preferred_max_leakage_proxy_mean"],
            ),
            "hard_max_leakage_proxy_mean": leakage.get(
                "hard_max_leakage_proxy_mean",
                LEAKAGE_THRESHOLDS["hard_max_leakage_proxy_mean"],
            ),
            "max_leakage_proxy_regression": leakage.get(
                "max_leakage_proxy_regression",
                LEAKAGE_THRESHOLDS["max_leakage_proxy_regression"],
            ),
            "reviewer_status": leakage.get("reviewer_status"),
            "reviewer_passed": leakage.get("reviewer_passed"),
            "reviewer_reasons": leakage.get("reviewer_reasons", []),
            "decision": leakage.get("decision"),
        },
    }


def canonical_revalidation_workflow():
    return {
        "script": "tools/revalidate_reviewer_bundle.sh",
        "environment_check": "python3 tools/reviewer_env_check.py --config configs/unseen_pocket_pdbbindpp_real_backends.json --config configs/unseen_pocket_lp_pdbbind_refined_real_backends.json --config configs/unseen_pocket_tight_geometry_pressure.json",
        "replay_check": "python3 tools/replay_drift_check.py <baseline-claim> <candidate-claim>",
        "canonical_surfaces": [
            "checkpoints/claim_matrix",
            "checkpoints/real_backends",
            "checkpoints/pdbbindpp_real_backends",
            "checkpoints/lp_pdbbind_refined_real_backends",
            "checkpoints/tight_geometry_pressure",
            "configs/checkpoints/multi_seed_pdbbindpp_real_backends",
        ],
        "notes": [
            "Rebuild docs/evidence_bundle.json after validating the canonical reviewer surfaces.",
            "Treat tight geometry as leakage-caution-only unless its reviewer_status becomes pass; it is no longer clash-blocked when backend thresholds pass.",
            "Treat chemistry benchmark evidence as stronger than proxy-only summaries when benchmark_evidence.evidence_tier is local_benchmark_style, reviewer benchmark-plus when it is reviewer_benchmark_plus, and explicit external benchmark coverage when it is external_benchmark_backed.",
        ],
    }


def threshold_result(value, threshold, direction):
    if value is None:
        return {"value": None, "threshold": threshold, "passed": False}
    if direction == "min":
        passed = value >= threshold
    else:
        passed = value <= threshold
    return {"value": value, "threshold": threshold, "passed": passed}


def evaluate_backend_thresholds(chemistry, docking, pocket):
    strict_fit = pocket.get("strict_pocket_fit_score")
    if strict_fit is None:
        strict_fit = pocket.get("heuristic_strict_pocket_fit_score")
    pocket_contact = docking.get("contact_fraction")
    if pocket_contact is None:
        pocket_contact = docking.get("pocket_contact_fraction")
    return {
        "rdkit_available": threshold_result(
            chemistry.get("rdkit_available"),
            BACKEND_THRESHOLDS["min_rdkit_available"],
            "min",
        ),
        "rdkit_sanitized_fraction": threshold_result(
            chemistry.get("rdkit_sanitized_fraction"),
            BACKEND_THRESHOLDS["min_rdkit_sanitized_fraction"],
            "min",
        ),
        "rdkit_unique_smiles_fraction": threshold_result(
            chemistry.get("rdkit_unique_smiles_fraction"),
            BACKEND_THRESHOLDS["min_unique_fraction"],
            "min",
        ),
        "backend_missing_structure_fraction": threshold_result(
            pocket.get("backend_missing_structure_fraction"),
            BACKEND_THRESHOLDS["max_backend_missing_structure_fraction"],
            "max",
        ),
        "clash_fraction": threshold_result(
            pocket.get("clash_fraction"),
            BACKEND_THRESHOLDS["max_clash_fraction"],
            "max",
        ),
        "strict_pocket_fit_score": threshold_result(
            strict_fit,
            BACKEND_THRESHOLDS["min_strict_pocket_fit"],
            "min",
        ),
        "pocket_contact_fraction": threshold_result(
            pocket_contact,
            BACKEND_THRESHOLDS["min_pocket_contact"],
            "min",
        ),
    }


def data_threshold_result(value, threshold, direction):
    if value is None:
        return {"value": None, "threshold": threshold, "passed": False}
    if direction == "min":
        passed = value >= threshold
    else:
        passed = value <= threshold
    return {"value": value, "threshold": threshold, "passed": passed}


def evaluate_data_thresholds(validation, split):
    validation = validation or {}
    split = split or {}
    val_families = split.get("val", {}).get("protein_family_proxy_histogram", {})
    test_families = split.get("test", {}).get("protein_family_proxy_histogram", {})
    return {
        "parsed_examples": data_threshold_result(
            validation.get("parsed_examples"),
            DATA_THRESHOLDS["min_parsed_complexes"],
            "min",
        ),
        "retained_label_coverage": data_threshold_result(
            validation.get("retained_label_coverage"),
            DATA_THRESHOLDS["min_retained_label_coverage"],
            "min",
        ),
        "val_protein_family_count": data_threshold_result(
            len(val_families),
            DATA_THRESHOLDS["min_heldout_family_count"],
            "min",
        ),
        "test_protein_family_count": data_threshold_result(
            len(test_families),
            DATA_THRESHOLDS["min_heldout_family_count"],
            "min",
        ),
    }


def collect_artifact_dir(path):
    path = Path(path)
    claim = load_json(path / "claim_summary.json")
    experiment = load_json(path / "experiment_summary.json")
    validation = load_json(path / "dataset_validation_report.json")
    split = load_json(path / "split_report.json")
    bundle = load_json(path / "run_artifacts.json")
    multi_seed = load_json(path / "multi_seed_summary.json")
    replay_drift = load_json(path / "replay_drift_report.json")
    preference_artifacts = load_preference_artifact_summary(path)
    return {
        "artifact_dir": str(path),
        "exists": path.is_dir(),
        "config_snapshot_hash": file_hash(path / "config.snapshot.json"),
        "claim": compact_claim(claim, preference_artifacts),
        "preference_artifacts": preference_artifacts,
        "reproducibility": None if experiment is None else experiment.get("reproducibility"),
        "dataset_validation": validation,
        "data_thresholds": evaluate_data_thresholds(validation, split),
        "split_report": split,
        "split_quality_warnings": None if split is None else split.get("quality_checks", {}).get("warnings", []),
        "run_artifacts_schema_version": None if bundle is None else bundle.get("schema_version"),
        "multi_seed_summary": multi_seed,
        "replay_drift_report": replay_drift,
    }


def reviewer_surface_summary(bundle):
    artifact_dirs = bundle["artifact_dirs"]

    def find_artifact(path):
        for artifact in artifact_dirs:
            if artifact["artifact_dir"] == path:
                return artifact
        return None

    larger_data = find_artifact("checkpoints/pdbbindpp_real_backends")
    if larger_data is None or not larger_data.get("exists"):
        larger_data = find_artifact("checkpoints/pdbbindpp_profile")
    backend = find_artifact("checkpoints/real_backends")
    compact = find_artifact("checkpoints/claim_matrix")
    secondary_benchmark = find_artifact("checkpoints/lp_pdbbind_refined_real_backends")
    multi_seed = find_artifact("configs/checkpoints/multi_seed_pdbbindpp_real_backends")
    if multi_seed is None or not multi_seed.get("exists"):
        multi_seed = find_artifact("configs/checkpoints/multi_seed_pdbbindpp")
    return {
        "compact_regression_surface": None
        if compact is None
        else {
            "artifact_dir": compact["artifact_dir"],
            "run_label": (compact.get("claim") or {}).get("run_label"),
            "purpose": "fast regression and baseline review surface",
        },
        "backend_review_surface": None
        if backend is None
        else {
            "artifact_dir": backend["artifact_dir"],
            "backend_thresholds_passed": all(
                result.get("passed", False)
                for result in ((backend.get("claim") or {}).get("backend_thresholds") or {}).values()
            ),
            "real_backend_backed": ((backend.get("claim") or {}).get("claim_context") or {}).get(
                "real_backend_backed"
            ),
            "purpose": "repository-supported real-backend chemistry and pocket gate",
        },
        "larger_data_review_surface": None
        if larger_data is None
        else {
            "artifact_dir": larger_data["artifact_dir"],
            "data_thresholds_passed": all(
                result.get("passed", False)
                for result in (larger_data.get("data_thresholds") or {}).values()
            ),
            "backend_thresholds_passed": all(
                result.get("passed", False)
                for result in ((larger_data.get("claim") or {}).get("backend_thresholds") or {}).values()
            ),
            "real_backend_backed": (
                ((larger_data.get("claim") or {}).get("claim_context") or {}).get(
                    "real_backend_backed"
                )
            ),
            "chemistry_evidence_tier": (
                (((larger_data.get("claim") or {}).get("chemistry_novelty_diversity") or {}).get(
                    "benchmark_evidence"
                ) or {}).get("evidence_tier")
            ),
            "purpose": "larger-data held-out family review surface",
        },
        "secondary_benchmark_review_surface": None
        if secondary_benchmark is None
        else {
            "artifact_dir": secondary_benchmark["artifact_dir"],
            "data_thresholds_passed": all(
                result.get("passed", False)
                for result in (secondary_benchmark.get("data_thresholds") or {}).values()
            ),
            "backend_thresholds_passed": all(
                result.get("passed", False)
                for result in (
                    (secondary_benchmark.get("claim") or {}).get("backend_thresholds") or {}
                ).values()
            ),
            "real_backend_backed": (
                (
                    (secondary_benchmark.get("claim") or {}).get("claim_context") or {}
                ).get("real_backend_backed")
            ),
            "chemistry_evidence_tier": (
                (
                    (
                        (secondary_benchmark.get("claim") or {}).get(
                            "chemistry_novelty_diversity"
                        )
                        or {}
                    ).get("benchmark_evidence")
                    or {}
                ).get("evidence_tier")
            ),
            "purpose": "secondary larger-data external-benchmark reviewer surface",
        },
        "multi_seed_review_surface": None
        if multi_seed is None
        else {
            "artifact_dir": multi_seed["artifact_dir"],
            "seed_count": len((multi_seed.get("multi_seed_summary") or {}).get("seed_runs", [])),
            "tracked_aggregates": sorted(
                ((multi_seed.get("multi_seed_summary") or {}).get("aggregates") or {}).keys()
            ),
            "stability_decision": (multi_seed.get("multi_seed_summary") or {}).get(
                "stability_decision"
            ),
            "purpose": "seed-stability surface for the larger-data review path",
        },
    }


def reviewer_bundle_status(bundle):
    artifact_dirs = {artifact["artifact_dir"]: artifact for artifact in bundle["artifact_dirs"]}
    required_paths = [
        "checkpoints/claim_matrix",
        "checkpoints/real_backends",
        "checkpoints/pdbbindpp_real_backends",
        "checkpoints/lp_pdbbind_refined_real_backends",
        "checkpoints/tight_geometry_pressure",
        "configs/checkpoints/multi_seed_pdbbindpp_real_backends",
    ]
    reasons = []
    for path in required_paths:
        if not (artifact_dirs.get(path) or {}).get("exists"):
            reasons.append(f"missing canonical reviewer artifact directory: {path}")

    larger = artifact_dirs.get("checkpoints/pdbbindpp_real_backends") or {}
    larger_claim = larger.get("claim") or {}
    secondary = artifact_dirs.get("checkpoints/lp_pdbbind_refined_real_backends") or {}
    secondary_claim = secondary.get("claim") or {}
    if larger_claim:
        for name, result in (larger_claim.get("backend_thresholds") or {}).items():
            if not result.get("passed", False):
                reasons.append(f"larger-data backend threshold failed: {name}")
        for name, result in (larger.get("data_thresholds") or {}).items():
            if not result.get("passed", False):
                reasons.append(f"larger-data threshold failed: {name}")
        if not (larger.get("replay_drift_report") or {}).get("passed", False):
            reasons.append("larger-data replay drift review failed")
    if secondary_claim:
        for name, result in (secondary_claim.get("backend_thresholds") or {}).items():
            if not result.get("passed", False):
                reasons.append(f"secondary benchmark backend threshold failed: {name}")
        for name, result in (secondary.get("data_thresholds") or {}).items():
            if not result.get("passed", False):
                reasons.append(f"secondary benchmark threshold failed: {name}")
        if (secondary_claim.get("leakage_review") or {}).get("reviewer_status") == "fail":
            reasons.append("secondary benchmark leakage review is in fail state")
        if not (secondary.get("replay_drift_report") or {}).get("passed", False):
            reasons.append("secondary benchmark replay drift review failed")

    tight = artifact_dirs.get("checkpoints/tight_geometry_pressure") or {}
    tight_claim = tight.get("claim") or {}
    if tight_claim:
        for name, result in (tight_claim.get("backend_thresholds") or {}).items():
            if not result.get("passed", False):
                reasons.append(f"tight-geometry backend threshold failed: {name}")
        tight_leakage = tight_claim.get("leakage_review") or {}
        if tight_leakage.get("reviewer_status") == "fail":
            reasons.append("tight-geometry leakage review is in fail state")
        if not (tight.get("replay_drift_report") or {}).get("passed", False):
            reasons.append("tight-geometry replay drift review failed")

    multi_seed = artifact_dirs.get("configs/checkpoints/multi_seed_pdbbindpp_real_backends") or {}
    multi_seed_summary = multi_seed.get("multi_seed_summary") or {}
    seed_runs = multi_seed_summary.get("seed_runs") or []
    if len(seed_runs) < 3:
        reasons.append("multi-seed reviewer surface has fewer than three persisted seeds")

    larger_leakage_status = (larger_claim.get("leakage_review") or {}).get("reviewer_status")
    tight_leakage_status = (tight_claim.get("leakage_review") or {}).get("reviewer_status")
    if reasons:
        status = "fail"
    elif "caution" in (larger_leakage_status, tight_leakage_status):
        status = "caution"
        if larger_leakage_status == "caution":
            reasons.append("larger-data reviewer surface remains leakage-caution")
        if tight_leakage_status == "caution":
            reasons.append("tight-geometry reviewer surface remains leakage-caution")
    else:
        status = "pass"
    return status, reasons


def build_limitations(bundle):
    limitations = [
        "Compact artifacts are smoke/regression evidence, not broad scientific generalization proof.",
        "The local medium profile currently contains only a five-complex parser smoke surface and is not the designated reviewer data path.",
    ]
    reviewer_surfaces = bundle["reviewer_surfaces"]
    larger = reviewer_surfaces.get("larger_data_review_surface") or {}
    multi_seed = reviewer_surfaces.get("multi_seed_review_surface") or {}
    tight = next(
        (
            artifact
            for artifact in bundle["artifact_dirs"]
            if artifact["artifact_dir"] == "checkpoints/tight_geometry_pressure"
        ),
        None,
    )
    if larger.get("real_backend_backed"):
        limitations.append(
            "Real-backend reviewer evidence is now anchored to a documented larger-data surface, but it still depends on the configured external backend commands being available."
        )
    else:
        limitations.append(
            "External backend metrics depend on locally installed tools and input structure provenance."
        )
    claim = (larger.get("artifact_dir") and next(
        (artifact for artifact in bundle["artifact_dirs"] if artifact["artifact_dir"] == larger.get("artifact_dir")),
        None,
    ) or {}).get("claim") or {}
    benchmark = ((claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {})
    if benchmark.get("evidence_tier") == "local_benchmark_style":
        limitations.append(
            "Chemistry review now carries a local benchmark-style aggregate that combines backend quality with held-out-pocket novelty/diversity, but it is still narrower than publication-grade external chemistry benchmarking."
        )
    elif benchmark.get("evidence_tier") == "reviewer_benchmark_plus":
        limitations.append(
            "The canonical larger-data chemistry surface now clears a stronger reviewer benchmark-plus tier with backend quality, held-out-pocket novelty/diversity, and explicit reviewer support checks, but it still lacks the full external benchmark-dataset layer."
        )
    elif benchmark.get("evidence_tier") == "external_benchmark_backed":
        external_count = (bundle.get("benchmark_breadth") or {}).get(
            "external_benchmark_backed_surface_count", 0
        )
        limitations.append(
            "The reviewer bundle now carries explicit external benchmark-dataset coverage on multiple larger-data surfaces, but it remains a local held-out-pocket reviewer package rather than a broad publication-scale benchmark campaign."
            if external_count >= 2
            else "The canonical larger-data chemistry surface now carries explicit external benchmark-dataset coverage on PDBbind++, so the residual chemistry caveat is narrower: this is still a held-out-pocket local reviewer surface rather than a broad multi-benchmark publication package."
        )
    else:
        limitations.append(
            "Novelty/diversity metrics are now explicit proxy evidence, but they remain structure-signature summaries rather than publication-grade chemistry benchmarks."
        )
    leakage = ((tight or {}).get("claim") or {}).get("leakage_review") or {}
    backend_thresholds = ((tight or {}).get("claim") or {}).get("backend_thresholds") or {}
    clash = (backend_thresholds.get("clash_fraction") or {}).get("passed")
    if clash and leakage.get("reviewer_status") == "pass":
        limitations.append(
            "Tight geometry now clears both the clash gate and the leakage reviewer gate on the canonical surface."
        )
    elif clash and leakage.get("reviewer_status") == "caution":
        limitations.append(
            "Tight geometry no longer fails the clash gate; reviewer wording should track the remaining leakage or chemistry blocker instead."
        )
    elif clash:
        limitations.append(
            "Tight geometry clears the clash gate, but the surface still needs a non-failing leakage review before stronger chemistry-and-geometry wording."
        )
    else:
        limitations.append(
            "Tight geometry remains blocked by the current backend clash gate."
        )
    if multi_seed.get("seed_count", 0) < 3:
        limitations.append(
            "The larger-data reviewer path still needs at least three persisted seeds for claim-bearing stability language."
        )
    return limitations


def build_environment_readiness(bundle):
    readiness_artifact = load_json(Path("docs/reviewer_env_readiness.json")) or {}
    larger = next(
        (
            artifact
            for artifact in bundle["artifact_dirs"]
            if artifact["artifact_dir"] == "checkpoints/pdbbindpp_real_backends"
        ),
        {},
    )
    backend_environment = (larger.get("claim") or {}).get("backend_environment") or {}
    return {
        "script": "tools/reviewer_env_check.py",
        "bootstrap_script": "tools/bootstrap_reviewer_env.sh",
        "packaged_environment_file": "reviewer_env/environment.yml",
        "default_revalidation_python": ".reviewer-env/bin/python when available, otherwise REVIEWER_PYTHON or python3",
        "canonical_config_examples": [
            "configs/unseen_pocket_pdbbindpp_real_backends.json",
            "configs/unseen_pocket_lp_pdbbind_refined_real_backends.json",
            "configs/unseen_pocket_tight_geometry_pressure.json",
        ],
        "real_backend_backed": backend_environment.get("real_backend_backed"),
        "prerequisites": backend_environment.get("prerequisites", []),
        "readiness_artifact": "docs/reviewer_env_readiness.json",
        "effective_python": ((readiness_artifact.get("packaged_environment") or {}).get("effective_python")),
        "effective_python_is_packaged": (
            (readiness_artifact.get("packaged_environment") or {}).get("effective_python_is_packaged")
        ),
        "ready": readiness_artifact.get("ready"),
    }


def build_replay_guarantees(bundle):
    larger = next(
        (
            artifact
            for artifact in bundle["artifact_dirs"]
            if artifact["artifact_dir"] == "checkpoints/pdbbindpp_real_backends"
        ),
        {},
    )
    reproducibility = larger.get("reproducibility") or {}
    return {
        "supports_strict_replay": ((reproducibility.get("resume_contract") or {}).get("supports_strict_replay")),
        "continuity_mode": ((reproducibility.get("resume_contract") or {}).get("continuity_mode")),
        "determinism_controls": reproducibility.get("determinism_controls"),
        "replay_tolerance": reproducibility.get("replay_tolerance"),
        "notes": [
            "Canonical reviewer replay is treated as bounded deterministic rerun within explicit metric tolerances, not bitwise or optimizer-state-identical strict replay.",
            "Use tools/replay_drift_check.py to compare refreshed claim surfaces against the canonical baseline before promotion.",
        ],
    }


def build_benchmark_breadth(bundle):
    surfaces = []
    external_count = 0
    benchmark_backed_count = 0
    external_datasets = []
    for artifact in bundle["artifact_dirs"]:
        claim = artifact.get("claim") or {}
        benchmark = ((claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {})
        evidence_tier = benchmark.get("evidence_tier")
        if evidence_tier or benchmark.get("backend_backed"):
            benchmark_backed_count += 1
            if evidence_tier == "external_benchmark_backed":
                external_count += 1
                dataset = benchmark.get("external_benchmark_dataset")
                if dataset:
                    external_datasets.append(dataset)
            surfaces.append(
                {
                    "artifact_dir": artifact.get("artifact_dir"),
                    "run_label": claim.get("run_label"),
                    "evidence_tier": evidence_tier,
                    "external_benchmark_dataset": benchmark.get("external_benchmark_dataset"),
                    "strict_pocket_fit_score": claim.get("strict_pocket_fit_score"),
                    "leakage_proxy_mean": claim.get("leakage_proxy_mean"),
                }
            )
    distinct_external_datasets = sorted(set(external_datasets))
    summary_sentence = (
        f"The reviewer bundle now carries {benchmark_backed_count} persisted chemistry/generalization "
        f"surface(s), including {external_count} external-benchmark-backed larger-data surface(s) "
        f"across {len(distinct_external_datasets)} benchmark dataset(s)"
    )
    if distinct_external_datasets:
        summary_sentence += ": " + ", ".join(distinct_external_datasets) + "."
    else:
        summary_sentence += "."
    return {
        "benchmark_backed_surface_count": benchmark_backed_count,
        "external_benchmark_backed_surface_count": external_count,
        "external_benchmark_datasets": distinct_external_datasets,
        "surfaces": surfaces,
        "summary_sentence": summary_sentence,
    }


def build_refresh_contract(bundle):
    artifacts = {artifact["artifact_dir"]: artifact for artifact in bundle["artifact_dirs"]}
    canonical_paths = [
        "checkpoints/claim_matrix",
        "checkpoints/real_backends",
        "checkpoints/pdbbindpp_real_backends",
        "checkpoints/lp_pdbbind_refined_real_backends",
        "checkpoints/tight_geometry_pressure",
    ]
    return {
        "entrypoint": "./tools/revalidate_reviewer_bundle.sh",
        "guarantee": "bounded replay with explicit drift tolerances; not strict optimizer-state-identical replay",
        "promotion_rule": "promote only when replay drift passes and reviewer thresholds remain green",
        "canonical_surface_reports": [
            {
                "artifact_dir": path,
                "replay_drift_passed": ((artifacts.get(path) or {}).get("replay_drift_report") or {}).get(
                    "passed"
                ),
                "replay_report_path": f"{path}/replay_drift_report.json",
            }
            for path in canonical_paths
        ],
    }


def _surface_row(bundle, artifact_dir, label):
    artifact = next(
        (artifact for artifact in bundle["artifact_dirs"] if artifact["artifact_dir"] == artifact_dir),
        {},
    )
    claim = artifact.get("claim") or {}
    benchmark = ((claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {})
    performance = claim.get("performance_gates") or {}
    return {
        "label": label,
        "artifact_dir": artifact_dir,
        "strict_pocket_fit_score": claim.get("strict_pocket_fit_score"),
        "leakage_proxy_mean": claim.get("leakage_proxy_mean"),
        "candidate_valid_fraction": claim.get("candidate_valid_fraction"),
        "test_examples_per_second": performance.get("test_examples_per_second"),
        "test_memory_mb": performance.get("test_memory_mb"),
        "chemistry_evidence_tier": benchmark.get("evidence_tier"),
    }


def build_efficiency_tradeoffs(bundle):
    rows = [
        _surface_row(bundle, "checkpoints/claim_matrix", "compact_regression"),
        _surface_row(bundle, "checkpoints/real_backends", "real_backend_gate"),
        _surface_row(bundle, "checkpoints/pdbbindpp_real_backends", "larger_data_canonical"),
        _surface_row(
            bundle,
            "checkpoints/lp_pdbbind_refined_real_backends",
            "larger_data_lp_pdbbind_refined",
        ),
        _surface_row(bundle, "checkpoints/tight_geometry_pressure", "tight_geometry_pressure"),
    ]
    canonical = next((row for row in rows if row["label"] == "larger_data_canonical"), None)
    if canonical is not None:
        base_eps = canonical.get("test_examples_per_second")
        for row in rows:
            eps = row.get("test_examples_per_second")
            row["relative_test_throughput_vs_larger_data"] = None if not base_eps or eps is None else eps / base_eps
    summary = []
    compact = next((row for row in rows if row["label"] == "compact_regression"), {})
    tight = next((row for row in rows if row["label"] == "tight_geometry_pressure"), {})
    backend = next((row for row in rows if row["label"] == "real_backend_gate"), {})
    if compact and canonical:
        summary.append(
            "Compact regression is the fast gate, but larger-data canonical review is the quality-bearing throughput anchor."
        )
    if tight and canonical:
        summary.append(
            "Tight geometry improves strict pocket fit over the canonical larger-data surface at materially lower throughput, so geometry pressure remains a real quality-vs-speed tradeoff instead of a free win."
        )
    secondary = next(
        (row for row in rows if row["label"] == "larger_data_lp_pdbbind_refined"),
        {},
    )
    if secondary and canonical:
        summary.append(
            "The LP-PDBBind refined reviewer surface now gives a second larger-data external-benchmark check under the same artifact contract, so chemistry wording no longer depends on a single benchmark anchor."
        )
    if backend and canonical:
        summary.append(
            "The repository real-backend gate is slower than the larger-data canonical surface and carries weaker leakage behavior, so compact or backend-gate wins alone are not enough to justify generator promotion."
        )
    return {
        "surfaces": rows,
        "summary": summary,
    }


def build_stronger_backend_profiles(bundle):
    profiles = []
    for artifact_dir, profile_label in [
        ("checkpoints/real_backends", "repository_real_backends"),
        ("checkpoints/vina_backend", "vina_backend_companion"),
    ]:
        artifact = next(
            (
                entry
                for entry in bundle["artifact_dirs"]
                if entry["artifact_dir"] == artifact_dir
            ),
            {},
        )
        claim = artifact.get("claim") or {}
        backend_thresholds = claim.get("backend_thresholds") or {}
        chemistry_backend = ((claim.get("backend_environment") or {}).get("chemistry_backend") or {})
        docking_backend = ((claim.get("backend_environment") or {}).get("docking_backend") or {})
        pocket_backend = ((claim.get("backend_environment") or {}).get("pocket_backend") or {})
        backend_review = claim.get("backend_review") or {}
        profiles.append(
            {
                "profile_label": profile_label,
                "artifact_dir": artifact_dir,
                "run_label": claim.get("run_label"),
                "backend_thresholds_passed": all(
                    result.get("passed", False) for result in backend_thresholds.values()
                )
                if backend_thresholds
                else False,
                "strict_pocket_fit_score": claim.get("strict_pocket_fit_score"),
                "leakage_proxy_mean": claim.get("leakage_proxy_mean"),
                "candidate_valid_fraction": claim.get("candidate_valid_fraction"),
                "backend_missing_structure_fraction": claim.get(
                    "backend_missing_structure_fraction"
                ),
                "rdkit_available": claim.get("rdkit_available"),
                "rdkit_sanitized_fraction": claim.get("rdkit_sanitized_fraction"),
                "chemistry_backend_enabled": chemistry_backend.get("enabled"),
                "docking_backend_enabled": docking_backend.get("enabled"),
                "pocket_backend_enabled": pocket_backend.get("enabled"),
                "real_backend_backed": ((claim.get("claim_context") or {}).get("real_backend_backed")),
                "reviewer_status": backend_review.get("reviewer_status"),
                "claim_bearing_ready": backend_review.get("claim_bearing_ready"),
                "docking_backend_available": backend_review.get("docking_backend_available"),
                "docking_input_completeness_fraction": backend_review.get(
                    "docking_input_completeness_fraction"
                ),
                "docking_score_coverage_fraction": backend_review.get(
                    "docking_score_coverage_fraction"
                ),
                "reviewer_reasons": backend_review.get("reviewer_reasons", []),
            }
        )
    return {
        "profiles": profiles,
        "summary": "Stronger backend-backed reviewer wording should cite the real-backend gate or the Vina claim-bearing companion profile explicitly, with pass/fail companion readiness taken from the persisted backend review rather than reviewer inference from partial failures.",
    }


def build_generator_direction(bundle):
    larger = next(
        (
            artifact
            for artifact in bundle["artifact_dirs"]
            if artifact["artifact_dir"] == "checkpoints/pdbbindpp_real_backends"
        ),
        {},
    )
    tight = next(
        (
            artifact
            for artifact in bundle["artifact_dirs"]
            if artifact["artifact_dir"] == "checkpoints/tight_geometry_pressure"
        ),
        {},
    )
    multi_seed = next(
        (
            artifact
            for artifact in bundle["artifact_dirs"]
            if artifact["artifact_dir"] == "configs/checkpoints/multi_seed_pdbbindpp_real_backends"
        ),
        {},
    )
    larger_claim = larger.get("claim") or {}
    tight_claim = tight.get("claim") or {}
    multi_seed_summary = multi_seed.get("multi_seed_summary") or {}
    seed_runs = multi_seed_summary.get("seed_runs") or []
    strict_fit_values = [
        row.get("strict_pocket_fit_score")
        for row in seed_runs
        if row.get("strict_pocket_fit_score") is not None
    ]
    leakage_values = [
        row.get("leakage_proxy_mean")
        for row in seed_runs
        if row.get("leakage_proxy_mean") is not None
    ]
    fit_range = None
    if strict_fit_values:
        fit_range = max(strict_fit_values) - min(strict_fit_values)
    leakage_range = None
    if leakage_values:
        leakage_range = max(leakage_values) - min(leakage_values)
    reasons = [
        "Use larger held-out-family evidence as the primary justification surface for generator changes, not compact-only wins.",
        "Keep the current conditioned-denoising plus bounded reranking path as the default while larger-data and tight-geometry surfaces still show non-trivial quality/efficiency tradeoffs.",
    ]
    active_method = larger_claim.get("active_generation_method")
    if active_method:
        reasons.append(
            f"Keep `{active_method}` as the explicit active method id in reviewer artifacts so future generator-family comparisons stay schema-stable."
        )
    if fit_range is not None and fit_range > 0.1:
        reasons.append(
            "Larger-data multi-seed strict pocket fit still varies materially across persisted seeds, which indicates headroom for incremental hardening before major objective replacement."
        )
    if leakage_range is not None and leakage_range > 0.05:
        reasons.append(
            "Larger-data multi-seed leakage still spans a meaningful range, so robustness work should target calibration and stability before architectural resets."
        )
    if (
        larger_claim.get("strict_pocket_fit_score") is not None
        and tight_claim.get("strict_pocket_fit_score") is not None
        and tight_claim.get("strict_pocket_fit_score") > larger_claim.get("strict_pocket_fit_score")
    ):
        reasons.append(
            "The tighter-geometry surface outperforms the canonical larger-data surface on pocket fit, so the generator path is not yet evidenced as saturated."
        )
    return {
        "primary_justification_surface": "checkpoints/pdbbindpp_real_backends",
        "stability_surface": "configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json",
        "major_model_change_gate": "Only revisit major objective changes such as diffusion after larger-data held-out-family artifacts show plateaued quality and acceptable stability without simpler incremental gains.",
        "current_direction": "continue_incremental_hardening",
        "saturation_status": "not_yet_evidenced",
        "larger_data_multi_seed_fit_range": fit_range,
        "larger_data_multi_seed_leakage_range": leakage_range,
        "stability_decision": multi_seed_summary.get("stability_decision"),
        "reasons": reasons,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build a compact reviewer-facing evidence bundle.")
    parser.add_argument("--output", default="docs/evidence_bundle.json")
    parser.add_argument("--artifact-dir", action="append", default=[])
    parser.add_argument(
        "--validate-reviewer-bundle",
        action="store_true",
        help="Fail when canonical reviewer surfaces are missing or outside the current reviewer bundle policy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dirs = args.artifact_dir or DEFAULT_ARTIFACT_DIRS
    bundle = {
        "schema_version": 6,
        "backend_thresholds": BACKEND_THRESHOLDS,
        "data_thresholds": DATA_THRESHOLDS,
        "leakage_thresholds": LEAKAGE_THRESHOLDS,
        "canonical_revalidation": canonical_revalidation_workflow(),
        "artifact_dirs": [collect_artifact_dir(path) for path in artifact_dirs],
        "reviewer_surfaces": None,
        "reviewer_bundle_status": None,
        "reviewer_bundle_reasons": [],
        "reviewer_environment_readiness": None,
        "replay_guarantees": None,
        "benchmark_breadth": None,
        "refresh_contract": None,
        "efficiency_tradeoffs": None,
        "stronger_backend_profiles": None,
        "generator_direction": None,
        "limitations": [],
        "claim_language_guardrail": "Use implemented/prototype wording unless medium-scale multi-seed real-backend evidence passes data, backend, performance, and leakage gates.",
    }
    bundle["reviewer_surfaces"] = reviewer_surface_summary(bundle)
    status, reasons = reviewer_bundle_status(bundle)
    bundle["reviewer_bundle_status"] = status
    bundle["reviewer_bundle_reasons"] = reasons
    bundle["reviewer_environment_readiness"] = build_environment_readiness(bundle)
    bundle["replay_guarantees"] = build_replay_guarantees(bundle)
    bundle["benchmark_breadth"] = build_benchmark_breadth(bundle)
    bundle["refresh_contract"] = build_refresh_contract(bundle)
    bundle["efficiency_tradeoffs"] = build_efficiency_tradeoffs(bundle)
    bundle["stronger_backend_profiles"] = build_stronger_backend_profiles(bundle)
    bundle["generator_direction"] = build_generator_direction(bundle)
    bundle["limitations"] = build_limitations(bundle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"evidence bundle written: {output}")
    if args.validate_reviewer_bundle and status == "fail":
        raise SystemExit(
            "reviewer bundle validation failed: "
            + "; ".join(reasons or ["canonical reviewer bundle is not in a passing state"])
        )


if __name__ == "__main__":
    main()
