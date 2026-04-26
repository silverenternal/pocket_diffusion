#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fast}"

DEFAULT_REVIEWER_PYTHON="python3"
if [[ -x ".reviewer-env/bin/python" ]]; then
  DEFAULT_REVIEWER_PYTHON=".reviewer-env/bin/python"
fi
REVIEWER_PYTHON="${REVIEWER_PYTHON:-${DEFAULT_REVIEWER_PYTHON}}"

cargo fmt --check
cargo test

cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_claim_matrix.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_real_backends.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_medium_profile.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_pdbbindpp_profile.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_pdbbindpp_real_backends.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_lp_pdbbind_refined_real_backends.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_multi_seed.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_multi_seed_medium.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_multi_seed_pdbbindpp.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_multi_seed_pdbbindpp_real_backends.json
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_vina_backend.json

"${REVIEWER_PYTHON}" -m py_compile \
  tools/claim_regression_gate.py \
  tools/evidence_bundle.py \
  tools/generator_decision_bundle.py \
  tools/generator_hardening_report.py \
  tools/paper_claim_bundle.py \
  tools/reviewer_refresh_report.py \
  tools/reviewer_env_check.py \
  tools/replay_drift_check.py \
  tools/rdkit_validity_backend.py \
  tools/pocket_contact_backend.py \
  tools/vina_score_backend.py

"${REVIEWER_PYTHON}" tools/reviewer_env_check.py \
  --config configs/unseen_pocket_pdbbindpp_real_backends.json \
  --config configs/unseen_pocket_lp_pdbbind_refined_real_backends.json \
  --config configs/unseen_pocket_tight_geometry_pressure.json

"${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/claim_matrix
"${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/real_backends --enforce-backend-thresholds
"${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/pdbbindpp_profile --enforce-data-thresholds
"${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/pdbbindpp_real_backends --enforce-backend-thresholds --enforce-data-thresholds
"${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/lp_pdbbind_refined_real_backends --enforce-backend-thresholds --enforce-data-thresholds
if [[ "${STRICT_MODEL_ONBOARDING_GATE:-0}" == "1" ]]; then
  "${REVIEWER_PYTHON}" tools/claim_regression_gate.py checkpoints/pdbbindpp_real_backends \
    --enforce-backend-thresholds \
    --enforce-data-thresholds \
    --enforce-publication-readiness \
    --enforce-preference-readiness
fi
"${REVIEWER_PYTHON}" tools/replay_drift_check.py \
  checkpoints/pdbbindpp_real_backends/claim_summary.json \
  checkpoints/pdbbindpp_real_backends/claim_summary.json
"${REVIEWER_PYTHON}" tools/replay_drift_check.py \
  checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json \
  checkpoints/lp_pdbbind_refined_real_backends/claim_summary.json
"${REVIEWER_PYTHON}" tools/evidence_bundle.py
"${REVIEWER_PYTHON}" tools/generator_decision_bundle.py --check
"${REVIEWER_PYTHON}" tools/generator_hardening_report.py
"${REVIEWER_PYTHON}" tools/paper_claim_bundle.py
"${REVIEWER_PYTHON}" tools/reviewer_refresh_report.py

if [[ "${MODE}" == "full" ]]; then
  "${REVIEWER_PYTHON}" tools/claim_regression_gate.py --run --config configs/unseen_pocket_claim_matrix.json
fi
