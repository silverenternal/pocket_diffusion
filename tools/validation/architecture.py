"""Architecture-boundary validation checks."""

import time
from pathlib import Path


def validate_modular_architecture_boundaries():
    started = time.time()
    failures = []

    if Path("src/models/system.rs").is_file():
        failures.append("architecture boundary regression: src/models/system.rs exists (expected directory module)")

    required_modules = [
        "src/models/semantic.rs",
        "src/models/interaction.rs",
        "src/losses/auxiliary.rs",
        "src/models/evaluation/mod.rs",
    ]
    for module in required_modules:
        if not Path(module).is_file():
            failures.append(f"required module missing: {module}")

    for path in Path(".").glob("*q2*.jsonl"):
        if "artifacts/evidence" not in str(path):
            failures.append(f"root-level q2 artifact JSONL should be under artifacts/evidence: {path}")
    for path in Path(".").glob("*q3*.jsonl"):
        if "artifacts/evidence" not in str(path):
            failures.append(f"root-level q3 artifact JSONL should be under artifacts/evidence: {path}")

    return {
        "name": "architecture boundary checks",
        "command": ["internal", "validate_modular_architecture_boundaries"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }


def validate_architecture_module_map():
    started = time.time()
    failures = []
    path = Path("docs/architecture_module_map.md")
    if not path.is_file():
        failures.append("missing docs/architecture_module_map.md")
        text = ""
    else:
        text = path.read_text(encoding="utf-8")

    for anchor in ("Model Backbone", "Trait Boundaries", "Objectives", "Validation Checks", "Ablation Handles"):
        if f"## {anchor}" not in text:
            failures.append(f"architecture module map missing section: {anchor}")
    haystack = text.lower()
    for token in (
        "topology, geometry, and pocket/context encoders structurally separate",
        "directed gated cross-modal interaction",
        "src/data/dataset/",
        "src/models/evaluation/",
        "configs/validation_manifest.json",
    ):
        if token not in haystack:
            failures.append(f"architecture module map missing token: {token}")

    return {
        "name": "architecture module map",
        "command": ["internal", "validate_architecture_module_map"],
        "required": True,
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": "\n".join(failures[-50:]),
        "stderr_tail": "",
    }
