#!/usr/bin/env python3
"""Audit claim-facing artifacts for placeholders and evidence coverage."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


PLACEHOLDER_RE = re.compile(
    r"\b(?:TBD|TODO|planned-only|placeholder|0\.XXX|x\.xxx|N/A)\b",
    re.IGNORECASE,
)
APPROXIMATE_CLAIM_RE = re.compile(
    r"\b(?:around|about|approx(?:imately)?|near|roughly|~)\s+[-+]?\d+(?:\.\d+)?",
    re.IGNORECASE,
)
ARTIFACT_PATH_RE = re.compile(
    r"(?P<path>(?:configs|checkpoints|docs|tools)/[A-Za-z0-9_./-]+(?:\.json|\.md|\.py|\.sh)?)"
)
COVERAGE_GROUPS = {
    "docking": ("docking", "vina", "gnina", "affinity"),
    "drug_likeness": ("qed", "sa_score", "logp", "tpsa", "lipinski", "molecular_weight"),
    "scaffold": ("scaffold", "tanimoto", "nearest_train"),
    "interaction": ("interaction", "hydrogen_bond", "hydrophobic", "contact", "clash"),
    "multi_seed": ("multi_seed", "seed_count"),
    "ablation": ("ablation", "delta"),
    "method_comparison": ("method_comparison", "method_id"),
}
REQUIRED_GATE_ARTIFACTS = {
    "method_comparison": ["configs/f42_method_comparison.json"],
    "multi_seed": ["configs/f41_multi_seed_summary.json"],
    "ablation_delta": ["configs/f31_ablation_bundle.json"],
    "correlation": ["configs/correlation_table.json", "docs/correlation_plot.md"],
}
METRIC_TOKEN_RE = re.compile(
    r"\b(?:vina_score|qed|sa_score|logp|tpsa|lipinski|scaffold|tanimoto|"
    r"strict_pocket_fit|centroid_offset|candidate_valid_fraction|unique_smiles_fraction)\b",
    re.IGNORECASE,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Audit claim-facing JSON and markdown artifacts.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve artifact paths.",
    )
    parser.add_argument(
        "--output-json",
        default="configs/artifact_audit_report.json",
        help="Machine-readable audit report path.",
    )
    parser.add_argument(
        "--output-md",
        default="docs/artifact_audit_summary.md",
        help="Short markdown summary path.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to scan. Defaults to configs, docs, and tools.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Fail when required paper-facing artifacts are missing or any placeholder flag is found.",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Validate an existing audit report against current source hashes without rewriting it.",
    )
    return parser.parse_args(argv[1:])


def iter_files(root, paths):
    selected = paths or ["configs", "docs", "tools"]
    for raw in selected:
        path = root / raw
        if path.is_file():
            if should_scan_file(path) and path.suffix in {".json", ".md", ".py", ".sh"}:
                yield path
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if should_scan_file(child) and child.is_file() and child.suffix in {".json", ".md", ".py", ".sh"}:
                    yield child


def should_scan_file(path):
    if "__pycache__" in path.parts:
        return False
    if path.name == "artifact_audit.py":
        return False
    if path.name.startswith("artifact_audit") and path.suffix in {".json", ".md"}:
        return False
    return True


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path, root):
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_digest(path),
    }


def json_paths(value, prefix="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from json_paths(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from json_paths(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def flatten_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def coverage_for_text(text):
    lower = text.lower()
    return {
        group: any(token in lower for token in tokens)
        for group, tokens in COVERAGE_GROUPS.items()
    }


def classify_artifact(flags, coverage):
    if flags:
        return "narrative_only"
    covered = sum(1 for present in coverage.values() if present)
    if covered >= 5:
        return "claim_ready"
    if covered:
        return "partial"
    return "narrative_only"


def scan_json(path, root):
    flags = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path.relative_to(root)),
            "kind": "json",
            "identity": artifact_identity(path, root),
            "classification": "narrative_only",
            "coverage": {group: False for group in COVERAGE_GROUPS},
            "flags": [{"path": "$", "reason": f"json_parse_error: {exc}"}],
        }

    for json_path, value in json_paths(payload):
        if isinstance(value, str):
            for match in PLACEHOLDER_RE.finditer(value):
                flags.append(
                    {
                        "path": json_path,
                        "reason": "placeholder_or_planned_value",
                        "excerpt": match.group(0),
                    }
                )
            for match in APPROXIMATE_CLAIM_RE.finditer(value):
                flags.append(
                    {
                        "path": json_path,
                        "reason": "approximate_numeric_claim",
                        "excerpt": match.group(0),
                    }
                )
    text = flatten_text(payload)
    coverage = coverage_for_text(text)
    return {
        "path": str(path.relative_to(root)),
        "kind": "json",
        "identity": artifact_identity(path, root),
        "classification": classify_artifact(flags, coverage),
        "coverage": coverage,
        "flags": flags,
    }


def scan_text(path, root):
    flags = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        for match in PLACEHOLDER_RE.finditer(line):
            flags.append(
                {
                    "line": line_number,
                    "reason": "placeholder_or_planned_value",
                    "excerpt": match.group(0),
                }
            )
        for match in APPROXIMATE_CLAIM_RE.finditer(line):
            flags.append(
                {
                    "line": line_number,
                    "reason": "approximate_numeric_claim",
                    "excerpt": match.group(0),
                }
            )
        for match in ARTIFACT_PATH_RE.finditer(line):
            artifact = root / match.group("path")
            if not artifact.exists():
                flags.append(
                    {
                        "line": line_number,
                        "reason": "missing_referenced_artifact",
                        "excerpt": match.group("path"),
                    }
                )
    coverage = coverage_for_text(text)
    return {
        "path": str(path.relative_to(root)),
        "kind": path.suffix.lstrip(".") or "text",
        "identity": artifact_identity(path, root),
        "classification": classify_artifact(flags, coverage),
        "coverage": coverage,
        "flags": flags,
    }


def summarize(records):
    coverage = {
        group: sum(1 for record in records if record["coverage"].get(group))
        for group in COVERAGE_GROUPS
    }
    classes = {}
    for record in records:
        classes[record["classification"]] = classes.get(record["classification"], 0) + 1
    return {
        "schema_version": 1,
        "artifact_count": len(records),
        "flag_count": sum(len(record["flags"]) for record in records),
        "classification_counts": classes,
        "coverage_counts": coverage,
    }


def metric_keys_from_json_artifacts(records, root):
    keys = set()
    for record in records:
        if record["kind"] != "json":
            continue
        path = root / record["path"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for json_path, value in json_paths(payload):
            if isinstance(value, (int, float)) and value == value:
                keys.add(json_path.rsplit(".", 1)[-1].lower())
    return keys


def narrative_metric_flags(records, root):
    metric_keys = metric_keys_from_json_artifacts(records, root)
    flags = []
    for record in records:
        if record["kind"] != "md":
            continue
        path = root / record["path"]
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not re.search(r"\d", line):
                continue
            for match in METRIC_TOKEN_RE.finditer(line):
                token = match.group(0).lower()
                if token not in metric_keys:
                    flags.append(
                        {
                            "path": record["path"],
                            "line": line_number,
                            "reason": "metric_absent_from_scanned_json",
                            "excerpt": token,
                        }
                    )
    return flags


def gate_failures(report, root):
    failures = []
    for group, paths in REQUIRED_GATE_ARTIFACTS.items():
        present = [path for path in paths if (root / path).is_file()]
        if not present:
            failures.append(
                {
                    "reason": "missing_required_artifact",
                    "group": group,
                    "paths": paths,
                }
            )
    for record in report["artifacts"]:
        for flag in record["flags"]:
            failures.append({"source_file": record["path"], **flag})
    failures.extend(narrative_metric_flags(report["artifacts"], root))
    return failures


def existing_report_failures(report, root):
    failures = []
    for record in report.get("artifacts", []):
        identity = record.get("identity") or {}
        rel_path = identity.get("path") or record.get("path")
        if not rel_path:
            continue
        path = root / rel_path
        if not path.exists():
            failures.append({"path": rel_path, "reason": "audited_source_missing"})
            continue
        current = artifact_identity(path, root)
        if current["sha256"] != identity.get("sha256") or current["mtime_ns"] != identity.get("mtime_ns"):
            failures.append({"path": rel_path, "reason": "audit_report_stale"})
    return failures


def write_markdown(path, report):
    lines = [
        "# Artifact Audit Summary",
        "",
        f"- artifacts_scanned: {report['summary']['artifact_count']}",
        f"- flags: {report['summary']['flag_count']}",
    ]
    for name, count in sorted(report["summary"]["classification_counts"].items()):
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Coverage")
    for group, count in sorted(report["summary"]["coverage_counts"].items()):
        lines.append(f"- {group}: {count}")
    lines.append("")
    lines.append("## Flagged Artifacts")
    flagged = [record for record in report["artifacts"] if record["flags"]]
    if not flagged:
        lines.append("- none")
    for record in flagged[:100]:
        lines.append(f"- `{record['path']}`: {len(record['flags'])} flag(s)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv):
    args = parse_args(argv)
    root = Path(args.root).resolve()
    output_json = root / args.output_json
    if args.check_existing:
        if not output_json.is_file():
            print(f"missing audit report: {output_json}", file=sys.stderr)
            return 1
        report = json.loads(output_json.read_text(encoding="utf-8"))
        failures = existing_report_failures(report, root)
        if failures:
            print(json.dumps({"schema_version": 1, "gate_failures": failures}, indent=2), file=sys.stderr)
            return 1
        return 0

    records = []
    for path in sorted(set(iter_files(root, args.paths))):
        if path.suffix == ".json":
            records.append(scan_json(path, root))
        else:
            records.append(scan_text(path, root))

    report = {"schema_version": 1, "summary": summarize(records), "artifacts": records}
    failures = gate_failures(report, root) if args.gate else []
    if failures:
        report["gate_failures"] = failures
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md = root / args.output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(output_md, report)
    if failures:
        print(json.dumps({"schema_version": 1, "gate_failures": failures}, indent=2), file=sys.stderr)
        return 1
    return 1 if report["summary"]["flag_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
