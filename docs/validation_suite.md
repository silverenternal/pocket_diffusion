# Validation Suite

Run the documented validation bundle with:

```bash
python tools/validation_suite.py
```

The script writes:

- `configs/validation_suite_report.json`
- `docs/validation_suite_report.md`

Use `--mode full` to run full `cargo test` instead of `cargo test --no-run`. Use `--strict` when a nonzero exit code should block on required failures. The current Q1 readiness gate passes on the layer-separated public-baseline artifacts; optional checks remain recorded separately so missing noncritical extras do not mask required failures.
