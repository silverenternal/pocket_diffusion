# Q14 Baseline Verification

Date: 2026-04-29

Commands run from `/home/hugo/codes/patent_rw` after the active-path review and
risk reproduction audit were prepared.

| Command | Status | Notes |
| --- | --- | --- |
| `jq empty todo.json` | pass | The Q14 todo file was valid before removing completed P0 tasks. |
| `cargo fmt --check` | pass | Existing Rust formatting was clean. |
| `cargo test --no-run` | pass | Test binaries compiled successfully in the local test profile. |

Compiled test executables:

- `src/lib.rs`
- `src/bin/disentangle_demo.rs`
- `src/main.rs`
- `tests/artifact_compatibility.rs`
- `tests/generation_method_platform.rs`
- `tests/integration_mi_monitor.rs`
- `tests/reviewer_tooling.rs`
- `tests/smoke_trainer_mi_integration.rs`

Known baseline boundaries:

- This is a compile and smoke-readiness baseline, not an end-to-end quality
  benchmark.
- The risk reproductions in `docs/q14_baseline_risk_reproductions.md` remain
  planned fixes, not successful molecular-generation claims.
- External docking, chemistry, and pocket scoring evidence remains unavailable
  unless a backend-specific artifact reports it.
- Existing dirty worktree changes were treated as user-owned context and were
  not reverted.
