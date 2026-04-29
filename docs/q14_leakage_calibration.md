# Q14 Leakage Calibration

Date: 2026-04-29

This calibration separates three leakage signals that must not be merged in
claim wording:

- similarity proxy diagnostics from modality representations
- optimizer-facing explicit probe penalties
- held-out frozen-probe audit performance

The machine-readable threshold table is
[`configs/q14_leakage_calibration.json`](../configs/q14_leakage_calibration.json).

## Thresholds

| Signal | Warning | Hard | Source |
| --- | ---: | ---: | --- |
| `test.leakage_proxy_mean` | `0.08` | `0.12` | Empirical smoke boundary from persisted mini-real claim summaries; still heuristic until larger held-out sweeps exist. |
| `auxiliary.leak_encoder_penalty` | `> 0.0` | `> 0.1` | Heuristic for the new Q14 split probe-fit/encoder-penalty routes. |
| frozen audit `improvement_over_baseline` | `> 0.05` | `> 0.2` | Heuristic smoke threshold for separately fit frozen ridge probes. |

`max_leakage_proxy_regression = 0.03` remains the recommended reviewer-facing
regression warning threshold for ablation comparisons.

## Evidence Basis

The current thresholds are intentionally conservative. The similarity-proxy
band is anchored to existing mini-real claim summaries:

- `checkpoints/pdbbindpp_real_backends/experiment_summary.json` reports
  `test.leakage_proxy_mean = 0.07348963423207786`, below the preferred warning
  threshold but close enough to require cautious wording without frozen audit
  evidence.
- `checkpoints/tight_geometry_pressure/claim_summary.json` already uses the
  same `0.08 / 0.12` reviewer band and reports a pass status under that band.

The explicit probe and frozen-audit thresholds are smoke-calibrated heuristics.
They should not be described as population-calibrated until enough held-out
real split rows are available.

## Claim Policy

No-leakage wording requires all of the following:

- similarity proxy below `0.08`
- explicit encoder penalty at `0.0` after its warmup stage
- frozen-probe audit status `ok`
- frozen audit improvement over baseline at or below `0.05`
- sufficient held-out rows for every claim-facing route

If the frozen audit is `not_run` or `insufficient_data`, the result may only be
described as a bounded training/evaluation diagnostic. It is not evidence that
off-modality information is absent.
