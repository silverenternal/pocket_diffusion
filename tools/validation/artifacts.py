"""Artifact-loading helpers for validation modules."""

import json
from pathlib import Path


def load_json_artifact(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
