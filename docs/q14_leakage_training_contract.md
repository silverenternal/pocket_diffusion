# Q14 Leakage Training Contract

Date: 2026-04-29

This contract defines how leakage-related training signals may be interpreted.
It separates optimizer objectives, detached diagnostics, and claim-facing
held-out audits so low leakage loss is not confused with proof that
off-modality information is absent.

## Evidence Roles

| Role | Optimizer-Facing | Claim Use |
| --- | --- | --- |
| `optimizer_penalty` | Yes, when the staged leakage weight is positive and execution mode is trainable. | Training regularizer only; not sufficient for no-leakage wording. |
| `detached_training_diagnostic` | No. Values are detached from the optimizer graph. | Monitoring and ablation evidence only. |
| `frozen_probe_audit` | No. Fit/evaluated after forward passes on frozen representations. | Required claim-facing evidence for recoverable off-modality information. |

## Allowed Modes

| Mode | Intended Gradient Semantics | Current Status | Allowed Claim |
| --- | --- | --- | --- |
| `detached_diagnostic` | Compute route values under detached/no-gradient semantics and return an explicit zero optimizer objective. | Implemented by `ExplicitLeakageProbeTrainingSemantics::DetachedDiagnostic`. | Diagnostic only. |
| `adversarial_penalty` | Penalize source representations when wrong-modality prediction becomes too accurate. | Implemented legacy/default route. It can update the active model path and is optimizer-facing. | Training regularizer only; no held-out no-leakage claim by itself. |
| `probe_fit` | Detach source encoder features and train only leakage probe parameters to predict off-modality targets. | Implemented by `ExplicitLeakageProbeTrainingSemantics::ProbeFit`. | Probe-capacity measurement only unless evaluated on held-out frozen representations. |
| `encoder_penalty` | Freeze or detach fitted probe parameters and penalize encoder/source features against the fixed probe. | Implemented by `ExplicitLeakageProbeTrainingSemantics::EncoderPenalty`. | Training regularizer only; must cite frozen-probe audit for no-leakage wording. |
| `alternating` | Alternate probe-fit and encoder-penalty phases with explicit route status in metrics. | Implemented by `ExplicitLeakageProbeTrainingSemantics::Alternating`. | Training protocol evidence plus held-out audit; not standalone proof. |

The default safe interpretation is conservative: training-time leakage values
are optimizer or diagnostic signals, while no-leakage wording requires a
`frozen_probe_audit` artifact that reports held-out performance against trivial
baselines across adequate probe capacity.

## Required Metrics

Every leakage-bearing training or evaluation artifact should preserve:

- route or family name
- execution mode
- effective staged weight
- optimizer-facing penalty value
- detached diagnostic value
- frozen audit status when available
- claim-boundary text explaining the difference between training penalties and
  held-out probe evidence

For future `probe_fit`, `encoder_penalty`, or `alternating` modes, metrics must
separately report:

- probe prediction loss
- encoder penalty
- active phase or route status
- whether probe parameters received gradients
- whether encoder/source modality parameters received gradients

## Config Surface

The current config surface lives under `training.explicit_leakage_probes`:

- `enable_explicit_probes`
- `training_semantics`
- route flags such as `topology_to_geometry_probe` and
  `pocket_to_geometry_probe`

As of Q14 hardening, the executable `training_semantics` values are
`detached_diagnostic`, `adversarial_penalty`, `probe_fit`, `encoder_penalty`,
and `alternating`. Claim summaries still must keep the training route status
separate from held-out frozen-probe calibration; a passing or low optimizer
penalty is not no-leakage evidence by itself.

## Claim Boundary

Permitted wording:

- "training uses an optimizer-facing leakage regularizer"
- "detached leakage diagnostics are bounded on this run"
- "held-out frozen probes did/did not improve over a trivial baseline"

Not permitted from training loss alone:

- "the representation contains no off-modality information"
- "leakage is absent"
- "low leakage loss proves semantic independence"
