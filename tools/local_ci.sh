#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_REVIEWER_PYTHON="python3"
if [[ -x ".reviewer-env/bin/python" ]]; then
  DEFAULT_REVIEWER_PYTHON=".reviewer-env/bin/python"
fi
REVIEWER_PYTHON="${REVIEWER_PYTHON:-${DEFAULT_REVIEWER_PYTHON}}"

usage() {
  cat <<'USAGE'
Usage: tools/local_ci.sh [fast|claim|reviewer|full]

Modes:
  fast      Fast pre-review gate. Runs formatting, compile-oriented tests, and
            the manifest-driven quick validation suite. This is the default.
  claim     Runs fast plus compact claim/evidence gates that do not require
            regenerating reviewer artifacts.
  reviewer  Runs claim plus reviewer environment and evidence-bundle checks.
            Requires the configured backend tools/data to be available when
            enforcing backend/data thresholds.
  full      Alias for reviewer, then runs the compact claim experiment refresh.

Environment:
  REVIEWER_PYTHON                 Python used for reviewer tools.
  STRICT_MODEL_ONBOARDING_GATE=1  Add publication/preference readiness gates.

Backend behavior:
  Missing backend executables or failed backend commands must be represented as
  coverage/status/failure metadata in artifacts. Do not treat unavailable
  backend evidence as a heuristic metric success.
USAGE
}

MODE="${1:-fast}"
case "${MODE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  fast|claim|reviewer|full)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

run_fast() {
  cargo fmt --check
  cargo test --test reviewer_tooling \
    correlation_table_builder_reports_missing_backend_coverage
  "${REVIEWER_PYTHON}" tools/validation_suite.py --mode quick --timeout 240
}

run_claim() {
  run_fast
  "${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/claim_matrix
  if [[ -f configs/q3_non_degradation_gate.json ]]; then
    "${REVIEWER_PYTHON}" tools/claim_regression_gate.py \
      --q3-non-degradation-gate configs/q3_non_degradation_gate.json
  fi
  jq empty configs/validation_manifest.json configs/artifact_retention_manifest.json
}

run_reviewer() {
  run_claim
  "${REVIEWER_PYTHON}" tools/reviewer_env_check.py \
    --config configs/unseen_pocket_pdbbindpp_real_backends.json \
    --config configs/unseen_pocket_lp_pdbbind_refined_real_backends.json \
    --config configs/unseen_pocket_tight_geometry_pressure.json
  "${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/real_backends --enforce-backend-thresholds
  "${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/pdbbindpp_profile --enforce-data-thresholds
  "${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/pdbbindpp_real_backends \
    --enforce-backend-thresholds \
    --enforce-data-thresholds
  "${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/lp_pdbbind_refined_real_backends \
    --enforce-backend-thresholds \
    --enforce-data-thresholds
  if [[ "${STRICT_MODEL_ONBOARDING_GATE:-0}" == "1" ]]; then
    "${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/pdbbindpp_real_backends \
      --enforce-backend-thresholds \
      --enforce-data-thresholds \
      --enforce-publication-readiness \
      --enforce-preference-readiness
  fi
  "${REVIEWER_PYTHON}" tools/evidence_bundle.py --validate-reviewer-bundle
}

case "${MODE}" in
  fast)
    run_fast
    ;;
  claim)
    run_claim
    ;;
  reviewer)
    run_reviewer
    ;;
  full)
    run_reviewer
    "${REVIEWER_PYTHON}" tools/claim_regression_gate.py --run --config configs/unseen_pocket_claim_matrix.json
    ;;
esac
