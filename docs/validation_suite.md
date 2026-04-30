# Validation Suite

Run the documented validation bundle with:

```bash
python tools/validation_suite.py
```

The script writes:

- `configs/validation_suite_report.json`
- `docs/validation_suite_report.md`

These default report files are generated reviewer artifacts. The Markdown report
starts with a generated-file marker, and the JSON report includes
`generated_report=true` plus `generated_by=tools/validation_suite.py`.

For normal local checks, write reports to temporary paths so curated report files
do not become dirty during unrelated work:

```bash
python tools/validation_suite.py \
  --mode quick \
  --timeout 240 \
  --output-json /tmp/pocket_diffusion_validation.json \
  --output-md /tmp/pocket_diffusion_validation.md
```

Use `--mode full` to run full `cargo test` instead of `cargo test --no-run`.
`tools/local_ci.sh fast` is now intended to run at least one executable reviewer
tooling regression (`correlation_table_builder_reports_missing_backend_coverage`)
in addition to quick manifest validation.
Use `--strict` when a nonzero exit code should block on required failures. The
current Q1 readiness gate passes on the layer-separated public-baseline
artifacts; optional checks remain recorded separately so missing noncritical
extras do not mask required failures.

## Training Metrics Audit Checks

When changing training metrics, primary objective decomposition, flow branch
schedule reporting, or objective scale diagnostics, run the focused checks below
in addition to the normal format and compile gates:

```bash
cargo test primary_component --lib
cargo test objective_budget --lib
cargo test objective_family_budget --lib
cargo test full_flow_branch_scale_report_covers_all_optimizer_branches --lib
cargo test full_flow_primary_branch_schedule_is_reported_without_auxiliary_stage_changes --lib
cargo test --test artifact_compatibility
```

These checks protect the registry-backed audit boundaries documented in
[`training_metrics_audit.md`](training_metrics_audit.md): primary component
provenance, branch component audits, objective-family budget grouping, and
backward-compatible persisted artifacts.

## Local CI Workflow

Use `tools/local_ci.sh` for contributor-facing gates:

| Gate | Command | Expected use | Backend/data requirements |
| --- | --- | --- | --- |
| Fast local gate | `tools/local_ci.sh fast` | Normal pre-review check; usually minutes on a warm build | No external docking backend required; runs a targeted reviewer tooling executable regression and quick manifest validation |
| Claim gate | `tools/local_ci.sh claim` | Before editing claim contracts, reports, or evidence manifests | Existing compact claim artifacts under `checkpoints/`; no regeneration required |
| Reviewer/evidence gate | `tools/local_ci.sh reviewer` | Before promoting reviewer-facing evidence or backend/data claims | Packaged reviewer Python preferred; configured RDKit/pocket/Vina/GNINA-style backends and data must be available when thresholds are enforced |
| Real-generation gate | `REAL_GENERATION_ARTIFACT_DIR=checkpoints/<artifact_dir> REAL_GENERATION_MULTI_SEED_SUMMARY=checkpoints/<multi_seed_dir>/multi_seed_summary.json tools/local_ci.sh real-gen` | Before using generated molecules in a real molecular-generation review workflow | Requires raw-native artifacts, Q15 ablation matrix summary, clean split-leakage checks, non-heuristic backend coverage, and at least three seeds |
| Full refresh gate | `tools/local_ci.sh full` | Explicit compact claim experiment refresh | Same as reviewer, plus the compact claim matrix run |

Backend-unavailable behavior is part of the contract: missing executables,
failed backend commands, or missing structures must be recorded as coverage,
status, and failure metadata. They must not be converted into heuristic success
metrics or zero-valued claim evidence.

## Validation Manifest

`configs/validation_manifest.json` is the source of truth for validation inputs:

- `python_tools`: Python files compiled by the syntax check.
- `json_artifacts`: required and optional JSON claim/config artifacts.
- `experiment_configs`: experiment configs passed to `cargo run -- validate`.
- `evidence_families`: short labels for Q1/Q2/Q3/Q5/Q15 and shared artifact contracts.

To add a new artifact family, add its paths to `json_artifacts` with a `family`
label and set `required=false` until the artifact is expected in every checkout.
Required missing paths fail validation; optional missing paths are recorded in
the JSON report without failing quick validation.

Focused helpers live under `tools/validation/`:

- `reporting.py` renders stable JSON/Markdown summaries.
- `artifacts.py` owns shared artifact loading helpers.
- `architecture.py` owns architecture-boundary checks.
- `claim_provenance.py` owns pharmacology and chemistry provenance checks.
- `negative_fixtures.py` creates temporary fixtures for known claim-risk failure modes.

Keep `tools/validation_suite.py` as the CLI orchestration layer. New artifact
families should add focused validation code under `tools/validation/` when the
check is reusable or grows beyond simple command wiring.
