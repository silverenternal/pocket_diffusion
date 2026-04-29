#!/usr/bin/env python3
"""Export a compact method-comparison summary from a claim artifact directory."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(metrics, *names):
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def layer_metrics(claim, name):
    return ((claim.get("layered_generation_metrics") or {}).get(name)) or {}


def backend_group(claim, group):
    return (((claim.get("backend_metrics") or {}).get(group) or {}).get("metrics")) or {}


def gain(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def raw_native_evidence(claim):
    existing = claim.get("raw_native_evidence")
    if isinstance(existing, dict) and existing:
        return existing
    raw_flow = layer_metrics(claim, "raw_flow") or layer_metrics(claim, "raw_rollout")
    model = claim.get("model_design") or {}
    return {
        "schema_version": 1,
        "evidence_role": "model_native_raw_first",
        "raw_model_layer": model.get("raw_model_layer") or ("raw_flow" if raw_flow else "unavailable"),
        "model_native_raw": True if raw_flow else False,
        "candidate_count": raw_flow.get("candidate_count"),
        "valid_fraction": metric(raw_flow, "valid_fraction") if raw_flow else model.get("raw_model_valid_fraction"),
        "native_graph_valid_fraction": metric(raw_flow, "native_graph_valid_fraction"),
        "native_bond_count_mean": metric(raw_flow, "native_bond_count_mean"),
        "mean_centroid_offset": metric(raw_flow, "mean_centroid_offset"),
        "mean_displacement": metric(raw_flow, "mean_displacement"),
        "clash_fraction": metric(raw_flow, "clash_fraction") if raw_flow else model.get("raw_model_clash_fraction"),
        "pocket_contact_fraction": metric(raw_flow, "pocket_contact_fraction") if raw_flow else model.get("raw_model_pocket_contact_fraction"),
        "gate_activation_mean": (claim.get("test") or {}).get("gate_activation_mean"),
        "leakage_proxy_mean": (claim.get("test") or {}).get("leakage_proxy_mean"),
        "claim_boundary": "raw model-native output before repair, constraints, reranking, or backend scoring",
    }


def processed_generation_evidence(claim):
    existing = claim.get("processed_generation_evidence")
    if isinstance(existing, dict) and existing:
        return existing
    model = claim.get("model_design") or {}
    processed_layer = model.get("processed_layer") or "reranked_candidates"
    processed = layer_metrics(claim, processed_layer) or layer_metrics(claim, "reranked_candidates")
    return {
        "schema_version": 1,
        "evidence_role": "additive_processed_or_reranked_evidence",
        "processed_layer": processed_layer,
        "model_native_raw": processed_layer in {"raw_flow", "raw_rollout"},
        "candidate_count": processed.get("candidate_count"),
        "valid_fraction": metric(processed, "valid_fraction") if processed else model.get("processed_valid_fraction"),
        "pocket_contact_fraction": metric(processed, "pocket_contact_fraction") if processed else model.get("processed_pocket_contact_fraction"),
        "mean_centroid_offset": metric(processed, "mean_centroid_offset"),
        "clash_fraction": metric(processed, "clash_fraction") if processed else model.get("processed_clash_fraction"),
        "postprocessor_chain": model.get("processed_postprocessor_chain") or [],
        "claim_boundary": model.get("processed_claim_boundary") or "processed layer evidence is additive and not raw-native model capability",
    }


def build_binding_analysis(artifact_dir, claim, method_comparison):
    methods = method_comparison.get("methods") or []
    flow = next((row for row in methods if row.get("method_id") == "flow_matching"), None)
    denoising = next((row for row in methods if row.get("method_id") == "conditioned_denoising"), None)
    raw_flow = layer_metrics(claim, "raw_flow") or layer_metrics(claim, "raw_rollout")
    reranked = layer_metrics(claim, "reranked_candidates")
    chemistry = backend_group(claim, "chemistry_validity")
    docking = backend_group(claim, "docking_affinity")
    interaction_metrics = {
        "hydrogen_bond_proxy": metric(reranked, "hydrogen_bond_proxy"),
        "hydrophobic_contact_proxy": metric(reranked, "hydrophobic_contact_proxy"),
        "contact_balance": metric(reranked, "contact_balance"),
    }
    backend_coverage = {
        "docking_score_coverage_fraction": metric(docking, "docking_score_coverage_fraction", "gnina_score_coverage_fraction"),
        "drug_likeness_coverage_fraction": metric(chemistry, "drug_likeness_coverage_fraction"),
        "scaffold_metric_coverage_fraction": metric(chemistry, "scaffold_metric_coverage_fraction"),
        "interaction_profile_coverage_fraction": metric(reranked, "interaction_profile_coverage_fraction"),
    }
    flow_native = {
        "valid_fraction": metric(raw_flow, "valid_fraction"),
        "pocket_contact_fraction": metric(raw_flow, "pocket_contact_fraction"),
        "clash_fraction": metric(raw_flow, "clash_fraction"),
        "flow_native_quality": ((claim.get("reranker_report") or {}).get("flow_native_quality")),
    }
    reranked_layer = {
        "valid_fraction": metric(reranked, "valid_fraction"),
        "pocket_contact_fraction": metric(reranked, "pocket_contact_fraction"),
        "clash_fraction": metric(reranked, "clash_fraction"),
        "qed": metric(chemistry, "qed", "raw_qed"),
        "sa_score": metric(chemistry, "sa_score", "raw_sa"),
        "scaffold_novelty_fraction": metric(reranked, "scaffold_novelty_fraction"),
        "unique_scaffold_fraction": metric(reranked, "unique_scaffold_fraction"),
        **interaction_metrics,
    }
    flow_vs_denoising = None
    if flow and denoising:
        flow_vs_denoising = {
            "native_valid_fraction_delta": gain(flow.get("native_valid_fraction"), denoising.get("native_valid_fraction")),
            "native_pocket_contact_fraction_delta": gain(flow.get("native_pocket_contact_fraction"), denoising.get("native_pocket_contact_fraction")),
            "native_clash_fraction_delta": gain(flow.get("native_clash_fraction"), denoising.get("native_clash_fraction")),
        }
    coverage_values = [value for value in backend_coverage.values() if value is not None]
    sufficient_backend_coverage = bool(coverage_values) and min(coverage_values) > 0.0
    return {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "comparison": "flow_matching_vs_conditioned_denoising_binding_metrics",
        "matched_budget_required": True,
        "raw_native_evidence": raw_native_evidence(claim),
        "processed_generation_evidence": processed_generation_evidence(claim),
        "flow_native_raw_flow": flow_native,
        "reranked_layer": reranked_layer,
        "backend_coverage": backend_coverage,
        "flow_vs_denoising_native_deltas": flow_vs_denoising,
        "promotion_decision": (
            "unsupported_backend_coverage"
            if not sufficient_backend_coverage
            else "review_required_do_not_promote_from_postprocessing_only"
        ),
        "interpretation": (
            "Binding deltas are separated between raw_flow and reranked layers. "
            "Do not promote flow matching when gains exist only after repair or reranking."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Write method comparison summary artifact.")
    parser.add_argument("artifact_dir", help="Artifact directory containing claim_summary.json")
    parser.add_argument(
        "--output",
        default="method_comparison_summary.json",
        help="Output file name written under artifact_dir unless absolute path is provided.",
    )
    parser.add_argument(
        "--binding-output",
        default=None,
        help="Optional flow-vs-denoising binding analysis JSON path.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    claim = load_json(artifact_dir / "claim_summary.json")
    method_comparison = claim.get("method_comparison") or {}
    methods = method_comparison.get("methods") or []

    binding_analysis = build_binding_analysis(artifact_dir, claim, method_comparison)
    summary = {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "raw_native_evidence": raw_native_evidence(claim),
        "processed_generation_evidence": processed_generation_evidence(claim),
        "active_method": method_comparison.get("active_method"),
        "method_count": len(methods),
        "methods": [
            {
                "method_id": row.get("method_id"),
                "method_name": row.get("method_name"),
                "evidence_role": row.get("evidence_role"),
                "trainable": row.get("trainable"),
                "sampling_steps": row.get("sampling_steps"),
                "wall_time_ms": row.get("wall_time_ms"),
                "native_valid_fraction": row.get("native_valid_fraction"),
                "native_pocket_contact_fraction": row.get("native_pocket_contact_fraction"),
                "native_clash_fraction": row.get("native_clash_fraction"),
                "available_layers": row.get("available_layers"),
            }
            for row in methods
        ],
        "layered_generation_artifacts": {
            "validation": str(artifact_dir / "generation_layers_validation.json"),
            "test": str(artifact_dir / "generation_layers_test.json"),
        },
        "flow_vs_denoising_binding_analysis": binding_analysis,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = artifact_dir / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"method comparison summary written: {output}")
    if args.binding_output:
        binding_output = Path(args.binding_output)
        if not binding_output.is_absolute():
            binding_output = artifact_dir / binding_output
        binding_output.parent.mkdir(parents=True, exist_ok=True)
        with binding_output.open("w", encoding="utf-8") as handle:
            json.dump(binding_analysis, handle, indent=2)
            handle.write("\n")
        print(f"flow-vs-denoising binding analysis written: {binding_output}")


if __name__ == "__main__":
    main()
