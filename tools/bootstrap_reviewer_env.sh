#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_PREFIX="${REVIEWER_ENV_PREFIX:-${ROOT_DIR}/.reviewer-env}"
ENV_FILE="${ROOT_DIR}/reviewer_env/environment.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing reviewer environment file: ${ENV_FILE}" >&2
  exit 1
fi

if command -v micromamba >/dev/null 2>&1; then
  micromamba create -y -p "${ENV_PREFIX}" -f "${ENV_FILE}"
elif command -v mamba >/dev/null 2>&1; then
  mamba env create -y -p "${ENV_PREFIX}" -f "${ENV_FILE}"
elif command -v conda >/dev/null 2>&1; then
  conda env create -y -p "${ENV_PREFIX}" -f "${ENV_FILE}"
else
  echo "need one of: micromamba, mamba, conda" >&2
  exit 1
fi

echo "reviewer environment ready at ${ENV_PREFIX}"
echo "use REVIEWER_PYTHON=${ENV_PREFIX}/bin/python ./tools/revalidate_reviewer_bundle.sh"
