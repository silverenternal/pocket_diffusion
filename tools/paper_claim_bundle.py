#!/usr/bin/env python3
"""Generate a concise paper-facing claim bundle from the reviewer evidence bundle."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a paper-facing claim bundle.")
    parser.add_argument("--bundle", default="docs/evidence_bundle.json")
    parser.add_argument("--output", default="docs/paper_claim_bundle.md")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def claim_for(bundle, artifact_dir):
    for artifact in bundle.get("artifact_dirs", []):
        if artifact.get("artifact_dir") == artifact_dir:
            return artifact
    return {}


def load_optional_json(path):
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(bundle):
    larger = claim_for(bundle, "checkpoints/pdbbindpp_real_backends")
    secondary_benchmark = claim_for(bundle, "checkpoints/lp_pdbbind_refined_real_backends")
    tight = claim_for(bundle, "checkpoints/tight_geometry_pressure")
    backend = claim_for(bundle, "checkpoints/real_backends")
    multi_seed = claim_for(bundle, "configs/checkpoints/multi_seed_pdbbindpp_real_backends")
    breadth = bundle.get("benchmark_breadth") or {}
    replay = bundle.get("replay_guarantees") or {}
    refresh = bundle.get("refresh_contract") or {}
    hardening = bundle.get("generator_direction") or {}
    tradeoffs = bundle.get("efficiency_tradeoffs") or {}
    stronger_profiles = bundle.get("stronger_backend_profiles") or {}
    limitations = bundle.get("limitations") or []
    ablations = load_optional_json("checkpoints/claim_matrix/ablation_matrix_summary.json")

    larger_claim = larger.get("claim") or {}
    secondary_benchmark_claim = secondary_benchmark.get("claim") or {}
    tight_claim = tight.get("claim") or {}
    backend_claim = backend.get("claim") or {}
    multi_seed_summary = multi_seed.get("multi_seed_summary") or {}
    ablation_variants = ablations.get("variants") or []

    lines = [
        "# Paper-Facing Claim Bundle",
        "",
        "This file is generated from canonical reviewer artifacts. Update it through `./tools/revalidate_reviewer_bundle.sh` rather than manual editing.",
        "",
        "## Claim Map",
        "",
        "| Claim | Canonical artifact | Current support |",
        "| --- | --- | --- |",
        "| Held-out-pocket chemistry/generalization on the canonical benchmark surface | `checkpoints/pdbbindpp_real_backends` | `benchmark_evidence.evidence_tier="
        + str((((larger_claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {}).get("evidence_tier")))
        + "`, `strict_pocket_fit_score="
        + fmt(larger_claim.get("strict_pocket_fit_score"))
        + "`, `leakage_proxy_mean="
        + fmt(larger_claim.get("leakage_proxy_mean"))
        + "` |",
        "| Second larger-data benchmark surface under the same reviewer policy | `checkpoints/lp_pdbbind_refined_real_backends` | `benchmark_evidence.evidence_tier="
        + str((((secondary_benchmark_claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {}).get("evidence_tier")))
        + "`, `strict_pocket_fit_score="
        + fmt(secondary_benchmark_claim.get("strict_pocket_fit_score"))
        + "`, `leakage_proxy_mean="
        + fmt(secondary_benchmark_claim.get("leakage_proxy_mean"))
        + "` |",
        "| Additional pressure-surface chemistry/generalization evidence beyond the canonical benchmark path | `checkpoints/tight_geometry_pressure` | `benchmark_evidence.evidence_tier="
        + str((((tight_claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {}).get("evidence_tier")))
        + "`, `strict_pocket_fit_score="
        + fmt(tight_claim.get("strict_pocket_fit_score"))
        + "`, `clash_fraction="
        + fmt(tight_claim.get("clash_fraction"))
        + "` |",
        "| Repository-supported backend gate | `checkpoints/real_backends` | `rdkit_sanitized_fraction="
        + fmt(backend_claim.get("rdkit_sanitized_fraction"))
        + "`, `strict_pocket_fit_score="
        + fmt(backend_claim.get("strict_pocket_fit_score"))
        + "` |",
        "| Seed stability for the larger-data reviewer path | `configs/checkpoints/multi_seed_pdbbindpp_real_backends/multi_seed_summary.json` | `seed_count="
        + str(len(multi_seed_summary.get("seed_runs") or []))
        + "`, `stability_decision="
        + str(multi_seed_summary.get("stability_decision"))
        + "` |",
        "| Stronger backend companion profile | `checkpoints/vina_backend` | `reviewer_status="
        + str(
            (
                next(
                    (
                        profile
                        for profile in stronger_profiles.get("profiles", [])
                        if profile.get("artifact_dir") == "checkpoints/vina_backend"
                    ),
                    {},
                ).get("reviewer_status")
            )
        )
        + "`, `docking_input_completeness_fraction="
        + fmt(
            (
                next(
                    (
                        profile
                        for profile in stronger_profiles.get("profiles", [])
                        if profile.get("artifact_dir") == "checkpoints/vina_backend"
                    ),
                    {},
                ).get("docking_input_completeness_fraction")
            )
        )
        + "`, `docking_score_coverage_fraction="
        + fmt(
            (
                next(
                    (
                        profile
                        for profile in stronger_profiles.get("profiles", [])
                        if profile.get("artifact_dir") == "checkpoints/vina_backend"
                    ),
                    {},
                ).get("docking_score_coverage_fraction")
            )
        )
        + "`, `rdkit_sanitized_fraction="
        + fmt(
            (
                next(
                    (
                        profile
                        for profile in stronger_profiles.get("profiles", [])
                        if profile.get("artifact_dir") == "checkpoints/vina_backend"
                    ),
                    {},
                ).get("rdkit_sanitized_fraction")
            )
        )
        + "`, `backend_missing_structure_fraction="
        + fmt(
            (
                next(
                    (
                        profile
                        for profile in stronger_profiles.get("profiles", [])
                        if profile.get("artifact_dir") == "checkpoints/vina_backend"
                    ),
                    {},
                ).get("backend_missing_structure_fraction")
            )
        )
        + "` |",
        "",
        "## Main Results Table",
        "",
        "| Surface | Evidence tier | Pocket fit | Leakage | Clash | Test eps | Candidate valid |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| Canonical larger-data | "
        + str((((larger_claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {}).get("evidence_tier")) or "n/a")
        + " | "
        + fmt(larger_claim.get("strict_pocket_fit_score"))
        + " | "
        + fmt(larger_claim.get("leakage_proxy_mean"))
        + " | "
        + fmt(larger_claim.get("clash_fraction"))
        + " | "
        + fmt(((larger_claim.get("performance_gates") or {}).get("test_examples_per_second")))
        + " | "
        + fmt(larger_claim.get("candidate_valid_fraction"))
        + " |",
        "| LP-PDBBind refined larger-data | "
        + str((((secondary_benchmark_claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {}).get("evidence_tier")) or "n/a")
        + " | "
        + fmt(secondary_benchmark_claim.get("strict_pocket_fit_score"))
        + " | "
        + fmt(secondary_benchmark_claim.get("leakage_proxy_mean"))
        + " | "
        + fmt(secondary_benchmark_claim.get("clash_fraction"))
        + " | "
        + fmt(((secondary_benchmark_claim.get("performance_gates") or {}).get("test_examples_per_second")))
        + " | "
        + fmt(secondary_benchmark_claim.get("candidate_valid_fraction"))
        + " |",
        "| Tight geometry pressure | "
        + str((((tight_claim.get("chemistry_novelty_diversity") or {}).get("benchmark_evidence") or {}).get("evidence_tier")) or "n/a")
        + " | "
        + fmt(tight_claim.get("strict_pocket_fit_score"))
        + " | "
        + fmt(tight_claim.get("leakage_proxy_mean"))
        + " | "
        + fmt(tight_claim.get("clash_fraction"))
        + " | "
        + fmt(((tight_claim.get("performance_gates") or {}).get("test_examples_per_second")))
        + " | "
        + fmt(tight_claim.get("candidate_valid_fraction"))
        + " |",
        "| Real-backend gate | n/a | "
        + fmt(backend_claim.get("strict_pocket_fit_score"))
        + " | "
        + fmt(backend_claim.get("leakage_proxy_mean"))
        + " | "
        + fmt(backend_claim.get("clash_fraction"))
        + " | "
        + fmt(((backend_claim.get("performance_gates") or {}).get("test_examples_per_second")))
        + " | "
        + fmt(backend_claim.get("candidate_valid_fraction"))
        + " |",
        "",
        "## Benchmark Breadth Table",
        "",
        "| Surface | Evidence tier | Pocket fit | Leakage |",
        "| --- | --- | --- | --- |",
    ]
    for artifact in breadth.get("surfaces") or []:
        lines.append(
            "| "
            + str(artifact.get("artifact_dir"))
            + " | "
            + str(artifact.get("evidence_tier") or "n/a")
            + " | "
            + fmt(artifact.get("strict_pocket_fit_score"))
            + " | "
            + fmt(artifact.get("leakage_proxy_mean"))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Ablation Table",
            "",
            "| Variant | Test pocket fit | Test leakage | Test valid fraction |",
            "| --- | --- | --- | --- |",
        ]
    )
    for variant in ablation_variants[:8]:
        test = variant.get("test") or {}
        lines.append(
            "| "
            + str(variant.get("variant_label"))
            + " | "
            + fmt(test.get("strict_pocket_fit_score"))
            + " | "
            + fmt(test.get("leakage_proxy_mean"))
            + " | "
            + fmt(test.get("candidate_valid_fraction"))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Efficiency Summary Table",
            "",
            "| Surface | Test eps | Relative throughput | Pocket fit | Leakage |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in tradeoffs.get("surfaces") or []:
        lines.append(
            "| "
            + str(row.get("label"))
            + " | "
            + fmt(row.get("test_examples_per_second"))
            + " | "
            + fmt(row.get("relative_test_throughput_vs_larger_data"))
            + "x | "
            + fmt(row.get("strict_pocket_fit_score"))
            + " | "
            + fmt(row.get("leakage_proxy_mean"))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Minimum External Communication Set",
            "",
            "- Main results: `checkpoints/pdbbindpp_real_backends`, `checkpoints/lp_pdbbind_refined_real_backends`, and `checkpoints/tight_geometry_pressure`.",
            "- Benchmark breadth: "
            + str(breadth.get("summary_sentence", "n/a")),
            "- Ablations and leakage review: `checkpoints/claim_matrix/ablation_matrix_summary.json` together with the leakage review sections embedded in the canonical claim summaries.",
            "- Replay policy: "
            + str(refresh.get("guarantee", "n/a"))
            + " Promotion uses bounded replay drift reports, not strict optimizer-state-identical replay.",
            "- Efficiency/stability: cite the larger-data multi-seed summary and the per-surface performance gates in the claim summaries.",
            "- Stronger backend companion: `checkpoints/vina_backend` now records explicit reviewer pass/fail status, Vina availability, input completeness, and docking score coverage, so claim-bearing backend wording no longer relies on reviewer interpretation of partial failures.",
            "- Generator direction: `"
            + str(hardening.get("current_direction"))
            + "` with saturation status `"
            + str(hardening.get("saturation_status"))
            + "`; do not justify major objective changes from compact-only wins.",
            "",
            "## Residual Caveats",
            "",
        ]
    )
    for limitation in limitations:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "## Generator Direction",
            "",
            "- Standalone decision artifact: `checkpoints/generator_decision/generator_decision.json`.",
            "- Major-model-change gate: " + str(hardening.get("major_model_change_gate")),
            "- Held-out-family direction notes are also summarized in `docs/generator_hardening_report.md`.",
            "- Efficiency tradeoff summary: " + "; ".join(tradeoffs.get("summary", [])),
            "",
            "## Refresh Contract",
            "",
            "- Canonical entrypoint: `./tools/revalidate_reviewer_bundle.sh`.",
            "- Replay decision: "
            + str(replay.get("notes", ["n/a"])[0]),
            "- Reviewer refresh artifact: `docs/reviewer_refresh_report.json`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    bundle = load_json(args.bundle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(bundle), encoding="utf-8")
    print(f"paper claim bundle written: {output}")


if __name__ == "__main__":
    main()
