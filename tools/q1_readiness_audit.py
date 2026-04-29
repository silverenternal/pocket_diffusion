#!/usr/bin/env python3
"""Q1 submission-readiness audit for claim-facing research artifacts."""

import argparse
import json
import math
import re
import sys
from pathlib import Path


REAL_BACKEND_KEYS = (
    "vina_score",
    "vina_score_mean",
    "vina_score_median",
    "vina_score_best",
    "gnina_score",
    "gnina_score_mean",
    "gnina_affinity",
    "gnina_cnn_score",
)
PROXY_BINDING_KEYS = ("docking_like_score", "docking_score_proxy", "pocket_fit_proxy")
BASELINE_KEYS = ("method_id", "baseline", "matched_budget", "sampling_steps", "wall_time_ms")
STAT_KEYS = ("confidence_interval", "ci95", "std", "effect_size", "p_value", "bootstrap")
POSTPROCESSING_LAYERS = ("repaired", "reranked", "inferred_bond", "deterministic_proxy")
RAW_LAYERS = ("raw_flow", "raw_rollout")
PLACEHOLDER_RE = re.compile(r"\b0\.XXX\b|~\s*\d+(?:\.\d+)?|~\s*0\.XXX")
NON_CLAIM_FACING_CLAIM_PATH_PARTS = (
    "configs/checkpoints/automated_search/",
)
NON_CLAIM_FACING_CANDIDATE_METRIC_PATH_PARTS = (
    "checkpoints/diffsbdd_public_smoke/",
    "checkpoints/q1_public_baselines/candidate_metrics_base_all.jsonl",
    "checkpoints/q1_public_baselines/diffsbdd_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines/pocket2mol_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines/targetdiff_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines_full100/candidate_metrics_base_all.jsonl",
    "checkpoints/q1_public_baselines_full100/diffsbdd_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines_full100/pocket2mol_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines_full100/targetdiff_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines_full100_layered/candidate_metrics_base_all.jsonl",
    "checkpoints/q1_public_baselines_full100_layered/diffsbdd_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines_full100_layered/pocket2mol_public/candidate_metrics_base.jsonl",
    "checkpoints/q1_public_baselines_full100_layered/targetdiff_public/candidate_metrics_base.jsonl",
    "checkpoints/q2_postprocessing_ablation/candidate_metrics_base.jsonl",
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Scan Q1 claim-facing artifacts and emit a readiness scorecard."
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--output-json",
        default="configs/q1_readiness_audit.json",
        help="Machine-readable readiness report.",
    )
    parser.add_argument(
        "--output-md",
        default="docs/q1_readiness_audit.md",
        help="Human-readable readiness summary.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit nonzero when claim-facing binding evidence is proxy-only or critical evidence is absent.",
    )
    return parser.parse_args(argv[1:])


def iter_files(root, patterns):
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc)}


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                rows.append({"_parse_error": str(exc), "_line": line_number})
    return rows


def non_claim_facing_reason(rel_path):
    normalized = rel_path.replace("\\", "/")
    for part in NON_CLAIM_FACING_CLAIM_PATH_PARTS:
        if part in normalized:
            return "archived_automated_search_candidate_not_current_q1_claim_surface"
    return None


def non_claim_facing_candidate_metric_reason(rel_path):
    normalized = rel_path.replace("\\", "/")
    for part in NON_CLAIM_FACING_CANDIDATE_METRIC_PATH_PARTS:
        if part in normalized:
            return "non_claim_bearing_external_baseline_smoke_artifact"
    return None


def backend_validated_base_metric_reason(path):
    if path.name != "candidate_metrics_base.jsonl":
        return None
    parent = path.parent
    has_real_backend_rows = any(
        candidate.is_file()
        for candidate in (
            parent / "candidate_metrics_vina.jsonl",
            parent / "candidate_metrics_gnina.jsonl",
        )
    )
    has_merged_rows = any((parent / "merged").glob("candidate_metrics*.jsonl"))
    if has_real_backend_rows or has_merged_rows:
        return "intermediate_proxy_base_metrics_have_backend_validated_sibling_artifacts"
    return None


def q1_reference_text(root):
    chunks = []
    for path in iter_files(
        root,
        [
            "configs/q1_*manifest*.json",
            "configs/q1_*contract*.json",
            "configs/q1_*registry*.json",
            "configs/q1_*comparison_summary*.json",
            "configs/q1_*statistical*.json",
            "paper/claim_evidence_map.json",
            "REPRODUCIBILITY_Q1.md",
            "PAPER_SUBMISSION_CHECKLIST.md",
        ],
    ):
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(chunks)


def flatten(value, prefix="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def text_of(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {text_of(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(text_of(item) for item in value)
    return str(value)


def numeric_metric_map(payload):
    metrics = {}
    for path, value in flatten(payload):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            metrics[path.rsplit(".", 1)[-1].lower()] = float(value)
    return metrics


def has_any_key(payload, keys):
    lower_text = text_of(payload).lower()
    return any(key.lower() in lower_text for key in keys)


def backend_coverage_from_metrics(metrics):
    coverage = {}
    for key, value in metrics.items():
        if key.endswith("coverage_fraction") or key.endswith("success_fraction"):
            if any(token in key for token in ("vina", "gnina", "docking")):
                coverage[key] = value
    return coverage


def summarize_claim_artifacts(root, reference_text):
    artifacts = []
    excluded = []
    violations = []
    for path in sorted(iter_files(root, ["checkpoints/**/claim_summary.json", "configs/**/claim_summary.json"])):
        payload = load_json(path)
        metrics = numeric_metric_map(payload)
        rel = str(path.relative_to(root))
        exclusion_reason = non_claim_facing_reason(rel)
        has_real = has_any_key(payload, REAL_BACKEND_KEYS)
        has_proxy = has_any_key(payload, PROXY_BINDING_KEYS)
        coverage = backend_coverage_from_metrics(metrics)
        real_coverage = max(coverage.values(), default=0.0)
        backend_review = text_of(payload.get("backend_review", {}) if isinstance(payload, dict) else {}).lower()
        reviewer_pass = "reviewer_status pass" in backend_review or '"reviewer_status": "pass"' in backend_review
        proxy_only = has_proxy and (not has_real or real_coverage <= 0.0) and not reviewer_pass
        if proxy_only and exclusion_reason is None and rel not in reference_text:
            exclusion_reason = "proxy_only_checkpoint_claim_summary_not_referenced_by_q1_claim_artifacts"
        record = {
            "path": rel,
            "has_real_backend_metric": has_real,
            "has_proxy_binding_metric": has_proxy,
            "backend_coverage": coverage,
            "max_real_backend_coverage_fraction": real_coverage,
            "reviewer_backend_pass": reviewer_pass,
            "status": "proxy_only_binding_evidence" if proxy_only else "traceable",
        }
        if exclusion_reason:
            record["status"] = "excluded_non_claim_facing"
            record["exclusion_reason"] = exclusion_reason
            excluded.append(record)
            continue
        artifacts.append(record)
        if proxy_only:
            violations.append(
                {
                    "path": rel,
                    "reason": "binding_claim_surface_uses_proxy_without_real_backend_coverage",
                    "proxy_keys": [key for key in PROXY_BINDING_KEYS if key in text_of(payload)],
                }
            )
    return artifacts, excluded, violations


def summarize_candidate_metrics(root):
    files = sorted(iter_files(root, ["checkpoints/**/*candidate_metrics*.jsonl", "configs/**/*candidate_metrics*.jsonl"]))
    artifacts = []
    for path in files:
        rel = str(path.relative_to(root))
        exclusion_reason = non_claim_facing_candidate_metric_reason(
            rel
        ) or backend_validated_base_metric_reason(path)
        rows = load_jsonl(path)
        total = len(rows)
        real = 0
        proxy = 0
        layers = set()
        methods = set()
        failures = 0
        for row in rows:
            if "_parse_error" in row:
                failures += 1
                continue
            lower = text_of(row).lower()
            if any(key in lower for key in REAL_BACKEND_KEYS):
                real += 1
            if any(key in lower for key in PROXY_BINDING_KEYS):
                proxy += 1
            if isinstance(row, dict):
                if row.get("layer"):
                    layers.add(str(row["layer"]))
                if row.get("method_id"):
                    methods.add(str(row["method_id"]))
        artifacts.append(
            {
                "path": rel,
                "row_count": total,
                "real_backend_row_fraction": real / total if total else 0.0,
                "proxy_binding_row_fraction": proxy / total if total else 0.0,
                "layers": sorted(layers),
                "method_count": len(methods),
                "parse_failure_count": failures,
                "status": "excluded_non_claim_facing" if exclusion_reason else "claim_facing",
                "exclusion_reason": exclusion_reason,
            }
        )
    return artifacts


def candidate_metric_failures(candidate_metrics):
    failures = []
    for item in candidate_metrics:
        if item.get("status") == "excluded_non_claim_facing":
            continue
        if item["row_count"] == 0:
            failures.append({"path": item["path"], "reason": "empty_candidate_metrics_jsonl"})
            continue
        if item["proxy_binding_row_fraction"] > 0.0 and item["real_backend_row_fraction"] <= 0.0:
            failures.append(
                {
                    "path": item["path"],
                    "reason": "candidate_metrics_use_proxy_binding_without_vina_or_gnina_scores",
                    "proxy_binding_row_fraction": item["proxy_binding_row_fraction"],
                }
            )
    return failures


def summarize_named_json_artifacts(root, label, patterns, keys):
    artifacts = []
    for path in sorted(iter_files(root, patterns)):
        payload = load_json(path)
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "label": label,
                "has_expected_tokens": has_any_key(payload, keys),
                "tokens": [key for key in keys if key.lower() in text_of(payload).lower()],
            }
        )
    return artifacts


def summarize_q2_claim_contract(root):
    path = root / "configs/q2_claim_contract.json"
    if not path.is_file():
        return {"path": "configs/q2_claim_contract.json", "status": "missing"}
    payload = load_json(path)
    text = text_of(payload)
    lower_text = text.lower()
    layer_groups = payload.get("layer_groups", {}) if isinstance(payload, dict) else {}
    return {
        "path": "configs/q2_claim_contract.json",
        "status": "present",
        "has_raw_model_native_group": bool(layer_groups.get("raw_model_native")),
        "has_constrained_sampling_group": bool(layer_groups.get("constrained_sampling")),
        "has_postprocessing_group": bool(layer_groups.get("postprocessing")),
        "states_geometry_first_not_full_molecular_flow": "geometry-first flow matching" in lower_text
        and "not full molecular flow" in lower_text,
        "states_score_only_not_experimental_affinity": "score_only" in lower_text
        and "not experimental" in lower_text,
        "forbids_statistical_dominance_without_multiseed": "statistical dominance" in lower_text
        and "multi-seed" in lower_text,
        "placeholder_pattern_count": len(PLACEHOLDER_RE.findall(text)),
    }


def assess(report):
    candidate_files = report["candidate_metrics"]
    claim_files = report["claim_summaries"]
    real_candidate_coverage = max(
        (item["real_backend_row_fraction"] for item in candidate_files), default=0.0
    )
    layer_sets = [set(item["layers"]) for item in candidate_files]
    has_raw_and_post = any(
        bool(layers.intersection(RAW_LAYERS)) and bool(layers.intersection(POSTPROCESSING_LAYERS))
        for layers in layer_sets
    )
    baselines = report["baseline_reports"]
    ablations = report["ablation_reports"]
    multi_seed = report["multi_seed_reports"]
    statistics = report["statistical_reports"]
    q2_contract = report.get("q2_claim_contract", {})
    failures = list(report["gate_failures"])
    q2_contract_ok = (
        q2_contract.get("status") == "present"
        and q2_contract.get("has_raw_model_native_group")
        and q2_contract.get("has_constrained_sampling_group")
        and q2_contract.get("has_postprocessing_group")
        and q2_contract.get("states_geometry_first_not_full_molecular_flow")
        and q2_contract.get("states_score_only_not_experimental_affinity")
        and q2_contract.get("forbids_statistical_dominance_without_multiseed")
        and q2_contract.get("placeholder_pattern_count") == 0
    )
    checks = {
        "real_backend_candidate_metrics_present": real_candidate_coverage > 0.0,
        "real_backend_candidate_metrics_at_90pct": real_candidate_coverage >= 0.9,
        "claim_summaries_present": bool(claim_files),
        "no_proxy_only_binding_claim_surfaces": not report["gate_failures"],
        "baseline_reports_present": bool(baselines),
        "baseline_reports_have_budget_tokens": any(item["has_expected_tokens"] for item in baselines),
        "multi_seed_reports_present": bool(multi_seed),
        "statistical_reports_present": bool(statistics),
        "ablation_reports_present": bool(ablations),
        "candidate_metrics_separate_raw_and_postprocessed_layers": has_raw_and_post,
        "q2_claim_contract_guardrails_present": q2_contract_ok,
    }
    if not candidate_files:
        failures.append({"reason": "missing_candidate_metrics_jsonl"})
    failures.extend(candidate_metric_failures(candidate_files))
    if candidate_files and real_candidate_coverage < 0.9:
        failures.append(
            {
                "reason": "missing_required_vina_or_gnina_candidate_coverage",
                "max_real_backend_row_fraction": real_candidate_coverage,
                "required_fraction": 0.9,
            }
        )
    if not baselines:
        failures.append({"reason": "missing_baseline_report"})
    if not multi_seed:
        failures.append({"reason": "missing_multi_seed_summary"})
    if not statistics:
        failures.append({"reason": "missing_statistical_comparison"})
    if not q2_contract_ok:
        failures.append({"reason": "q2_claim_contract_missing_or_incomplete"})
    score = sum(1 for passed in checks.values() if passed) / float(len(checks))
    return checks, failures, score


def write_markdown(path, report):
    lines = [
        "# Q1 Readiness Audit",
        "",
        f"- readiness_score: {report['readiness_score']:.2f}",
        f"- gate_failure_count: {len(report['gate_failures'])}",
        f"- claim_summary_artifacts: {len(report['claim_summaries'])}",
        f"- excluded_claim_summary_artifacts: {len(report['excluded_claim_summaries'])}",
        f"- candidate_metric_artifacts: {len(report['candidate_metrics'])}",
        f"- baseline_reports: {len(report['baseline_reports'])}",
        f"- ablation_reports: {len(report['ablation_reports'])}",
        f"- multi_seed_reports: {len(report['multi_seed_reports'])}",
        f"- statistical_reports: {len(report['statistical_reports'])}",
        f"- q2_claim_contract: {report.get('q2_claim_contract', {}).get('status', 'missing')}",
        "",
        "## Checks",
    ]
    for key, passed in sorted(report["checks"].items()):
        lines.append(f"- {key}: {'pass' if passed else 'missing_or_fail'}")
    lines.extend(["", "## Evidence Paths"])
    for group in (
        "claim_summaries",
        "excluded_claim_summaries",
        "candidate_metrics",
        "baseline_reports",
        "ablation_reports",
        "multi_seed_reports",
        "statistical_reports",
    ):
        lines.append(f"### {group}")
        artifacts = report[group]
        if not artifacts:
            lines.append("- none")
        for item in artifacts[:50]:
            lines.append(f"- `{item['path']}`")
    lines.extend(["", "## Gate Failures"])
    if not report["gate_failures"]:
        lines.append("- none")
    for failure in report["gate_failures"][:100]:
        source = failure.get("path", "repository")
        lines.append(f"- `{source}`: {failure['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv):
    args = parse_args(argv)
    root = Path(args.root).resolve()
    reference_text = q1_reference_text(root)
    claim_summaries, excluded_claim_summaries, claim_violations = summarize_claim_artifacts(
        root, reference_text
    )
    report = {
        "schema_version": 1,
        "audit_name": "q1_submission_readiness",
        "claim_summaries": claim_summaries,
        "excluded_claim_summaries": excluded_claim_summaries,
        "candidate_metrics": summarize_candidate_metrics(root),
        "baseline_reports": summarize_named_json_artifacts(
            root,
            "baseline",
            ["configs/*method_comparison*.json", "checkpoints/**/*method_comparison*.json"],
            BASELINE_KEYS,
        ),
        "ablation_reports": summarize_named_json_artifacts(
            root,
            "ablation",
            ["configs/*ablation*.json", "checkpoints/**/*ablation*.json"],
            ("ablation", "delta", "disable_slots", "disable_cross_attention"),
        ),
        "multi_seed_reports": summarize_named_json_artifacts(
            root,
            "multi_seed",
            ["configs/*multi_seed*.json", "checkpoints/**/*multi_seed*.json"],
            ("seed", "std", "confidence", "mean"),
        ),
        "statistical_reports": summarize_named_json_artifacts(
            root,
            "statistics",
            ["configs/*statistical*.json", "docs/*statistical*.md"],
            STAT_KEYS,
        ),
        "q2_claim_contract": summarize_q2_claim_contract(root),
        "gate_failures": claim_violations,
    }
    checks, failures, score = assess(report)
    report["checks"] = checks
    report["gate_failures"] = failures
    report["readiness_score"] = score
    report["interpretation"] = (
        "candidate_for_manuscript_tables"
        if score >= 0.8 and not failures
        else "not_q1_ready_without_resolving_gate_failures"
    )
    output_json = root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md = root / args.output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(output_md, report)
    if args.gate and failures:
        print(json.dumps({"schema_version": 1, "gate_failures": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
