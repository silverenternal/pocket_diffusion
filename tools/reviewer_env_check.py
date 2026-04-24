#!/usr/bin/env python3
"""Machine-checkable readiness check for reviewer-facing backend surfaces."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Check reviewer backend environment readiness.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Experiment config(s) whose backend commands should be validated.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path.",
    )
    return parser.parse_args(argv[1:])


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_python():
    completed = subprocess.run(
        [sys.executable, "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "available": completed.returncode == 0,
        "detail": (completed.stdout or "").strip(),
        "executable": sys.executable,
    }


def check_rdkit():
    spec = importlib.util.find_spec("rdkit")
    return {
        "available": spec is not None,
        "detail": "importable" if spec is not None else "missing python module `rdkit`",
    }


def command_check(executable, args):
    executable_path = shutil.which(executable) if executable else None
    script_status = []
    for candidate in args:
        if candidate.endswith(".py"):
            script = Path(candidate)
            script_status.append(
                {
                    "path": candidate,
                    "exists": script.is_file(),
                }
            )
    return {
        "executable": executable,
        "resolved_executable": executable_path,
        "available": executable_path is not None,
        "args": args,
        "script_status": script_status,
    }


def config_check(path):
    config = load_json(path)
    evaluation = config.get("external_evaluation", {})
    return {
        "config": path,
        "surface_label": config.get("surface_label"),
        "chemistry_backend": command_check(
            evaluation.get("chemistry_backend", {}).get("executable"),
            evaluation.get("chemistry_backend", {}).get("args", []),
        ),
        "docking_backend": command_check(
            evaluation.get("docking_backend", {}).get("executable"),
            evaluation.get("docking_backend", {}).get("args", []),
        ),
        "pocket_backend": command_check(
            evaluation.get("pocket_backend", {}).get("executable"),
            evaluation.get("pocket_backend", {}).get("args", []),
        ),
    }


def packaged_environment_status():
    env_file = Path("reviewer_env/environment.yml")
    bootstrap = Path("tools/bootstrap_reviewer_env.sh")
    local_python = Path(".reviewer-env/bin/python")
    active_python = Path(sys.executable)
    return {
        "environment_file": str(env_file),
        "environment_file_exists": env_file.is_file(),
        "bootstrap_script": str(bootstrap),
        "bootstrap_script_exists": bootstrap.is_file(),
        "local_python": str(local_python),
        "local_python_exists": local_python.is_file(),
        "effective_python": str(active_python),
        "effective_python_is_packaged": local_python.is_file()
        and active_python.resolve() == local_python.resolve(),
        "recommended_bootstrap_command": "./tools/bootstrap_reviewer_env.sh",
    }


def main(argv):
    args = parse_args(argv)
    report = {
        "python": check_python(),
        "rdkit": check_rdkit(),
        "configs": [config_check(path) for path in args.config],
        "packaged_environment": packaged_environment_status(),
    }
    report["ready"] = report["python"]["available"] and report["rdkit"]["available"] and all(
        backend["available"] and all(item["exists"] for item in backend["script_status"])
        for config in report["configs"]
        for backend in (
            config["chemistry_backend"],
            config["docking_backend"],
            config["pocket_backend"],
        )
        if backend["executable"] is not None
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    if not report["ready"]:
        raise SystemExit("reviewer environment readiness failed")


if __name__ == "__main__":
    main(sys.argv)
