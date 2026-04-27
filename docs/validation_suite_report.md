# Validation Suite Report

- status: pass
- mode: quick
- total_checks: 10
- failed_required: 0
- failed_optional: 0

| Check | Required | Status | Seconds |
| --- | --- | --- | ---: |
| `cargo fmt` | true | pass | 0.106 |
| `cargo test no-run` | true | pass | 0.086 |
| `python syntax` | true | pass | 0.072 |
| `json artifacts` | true | pass | 0.015 |
| `q1 readiness audit` | true | pass | 0.386 |
| `q1 readiness gate` | false | pass | 0.388 |
| `validate unseen_pocket_pdbbindpp_real_backends.json` | true | pass | 0.244 |
| `validate unseen_pocket_lp_pdbbind_refined_real_backends.json` | true | pass | 0.240 |
| `validate unseen_pocket_tight_geometry_pressure.json` | true | pass | 0.246 |
| `validate unseen_pocket_multi_seed_pdbbindpp_real_backends.json` | true | pass | 0.242 |

## Failed Checks
- none
