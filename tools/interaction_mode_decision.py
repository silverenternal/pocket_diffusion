#!/usr/bin/env python3
"""Summarize transformer vs lightweight interaction tradeoffs on reviewer surfaces."""

import json
from pathlib import Path


SURFACES = [
    {
        "label": "larger_data_canonical",
        "artifact_dir": "checkpoints/pdbbindpp_real_backends_interaction_review",
    },
    {
        "label": "tight_geometry_pressure",
        "artifact_dir": "checkpoints/tight_geometry_pressure",
    },
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_sub(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def claim_metrics(claim):
    chemistry = (
        claim.get("backend_metrics", {})
        .get("chemistry_validity", {})
        .get("metrics", {})
    )
    pocket = (
        claim.get("backend_metrics", {})
        .get("pocket_compatibility", {})
        .get("metrics", {})
    )
    performance = claim.get("performance_gates", {})
    chemistry_novelty = claim.get("chemistry_novelty_diversity", {})
    benchmark = chemistry_novelty.get("benchmark_evidence", {})
    test = claim.get("test", {})
    return {
        "strict_pocket_fit_score": test.get("strict_pocket_fit_score"),
        "leakage_proxy_mean": test.get("leakage_proxy_mean"),
        "candidate_valid_fraction": test.get("candidate_valid_fraction"),
        "unique_smiles_fraction": test.get("unique_smiles_fraction"),
        "topology_specialization_score": test.get("topology_specialization_score"),
        "geometry_specialization_score": test.get("geometry_specialization_score"),
        "pocket_specialization_score": test.get("pocket_specialization_score"),
        "slot_activation_mean": test.get("slot_activation_mean"),
        "gate_activation_mean": test.get("gate_activation_mean"),
        "test_examples_per_second": performance.get("test_examples_per_second"),
        "rdkit_sanitized_fraction": chemistry.get("rdkit_sanitized_fraction"),
        "rdkit_unique_smiles_fraction": chemistry.get("rdkit_unique_smiles_fraction"),
        "clash_fraction": pocket.get("clash_fraction"),
        "atom_coverage_fraction": pocket.get("atom_coverage_fraction"),
        "chemistry_evidence_tier": benchmark.get("evidence_tier"),
    }


def winner(delta, higher_is_better=True):
    if delta is None:
        return "unknown"
    if abs(delta) < 1e-12:
        return "tie"
    if higher_is_better:
        return "transformer" if delta > 0 else "lightweight"
    return "transformer" if delta < 0 else "lightweight"


def summarize_surface(label: str, artifact_dir: Path):
    review = load_json(artifact_dir / "interaction_mode_review.json")
    transformer_claim = load_json(artifact_dir / "claim_summary.json")
    lightweight_claim = load_json(
        artifact_dir / "ablations" / "interaction_lightweight" / "claim_summary.json"
    )
    transformer = claim_metrics(transformer_claim)
    lightweight = claim_metrics(lightweight_claim)
    deltas = {
        "strict_pocket_fit_score": safe_sub(
            transformer["strict_pocket_fit_score"], lightweight["strict_pocket_fit_score"]
        ),
        "leakage_proxy_mean": safe_sub(
            transformer["leakage_proxy_mean"], lightweight["leakage_proxy_mean"]
        ),
        "test_examples_per_second": safe_sub(
            transformer["test_examples_per_second"], lightweight["test_examples_per_second"]
        ),
        "clash_fraction": safe_sub(
            transformer["clash_fraction"], lightweight["clash_fraction"]
        ),
        "atom_coverage_fraction": safe_sub(
            transformer["atom_coverage_fraction"], lightweight["atom_coverage_fraction"]
        ),
        "topology_specialization_score": safe_sub(
            transformer["topology_specialization_score"],
            lightweight["topology_specialization_score"],
        ),
        "geometry_specialization_score": safe_sub(
            transformer["geometry_specialization_score"],
            lightweight["geometry_specialization_score"],
        ),
        "pocket_specialization_score": safe_sub(
            transformer["pocket_specialization_score"],
            lightweight["pocket_specialization_score"],
        ),
    }
    return {
        "surface_label": label,
        "artifact_dir": str(artifact_dir),
        "interaction_review_test_tally": review.get("test", {}).get("tally"),
        "transformer": transformer,
        "lightweight": lightweight,
        "deltas_transformer_minus_lightweight": deltas,
        "metric_wins": {
            "strict_pocket_fit_score": winner(deltas["strict_pocket_fit_score"], True),
            "leakage_proxy_mean": winner(deltas["leakage_proxy_mean"], False),
            "test_examples_per_second": winner(deltas["test_examples_per_second"], True),
            "clash_fraction": winner(deltas["clash_fraction"], False),
            "atom_coverage_fraction": winner(deltas["atom_coverage_fraction"], True),
            "topology_specialization_score": winner(
                deltas["topology_specialization_score"], True
            ),
            "geometry_specialization_score": winner(
                deltas["geometry_specialization_score"], True
            ),
            "pocket_specialization_score": winner(
                deltas["pocket_specialization_score"], True
            ),
        },
    }


def main():
    summaries = [
        summarize_surface(surface["label"], Path(surface["artifact_dir"]))
        for surface in SURFACES
    ]
    recommendation = (
        "Promote transformer as the default larger-data claim path: it is decisively better on "
        "the canonical larger-data real-backend surface, while tight-geometry remains a retained "
        "two-mode ablation because its tradeoff is still bounded."
    )
    payload = {
        "schema_version": 1,
        "review_root": "./checkpoints",
        "surfaces": summaries,
        "recommendation": recommendation,
    }
    output = Path("checkpoints/interaction_mode_review.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"interaction mode decision written: {output}")


if __name__ == "__main__":
    main()
