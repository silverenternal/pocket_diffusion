# Q13 Baseline Verification Snapshot

Date: 2026-04-29

Commands run from `/home/hugo/codes/patent_rw`:

| Command | Status | Notes |
| --- | --- | --- |
| `jq empty todo.json` | pass | JSON was valid before task cleanup. |
| `cargo fmt --check` | pass | Existing Rust formatting was clean. |
| `cargo test --no-fail-fast` | pass | 320 library unit tests, 12 binary unit tests, 40 integration tests, and doc tests passed. |

Known evidence boundaries:

- The verification is a compact local CPU/test-profile snapshot, not a
  reviewer-scale benchmark.
- External docking/chemistry backend evidence is only successful when a
  backend-specific artifact says so; otherwise it remains unavailable.
- Existing dirty worktree changes were treated as user-owned context and were
  not reverted.
