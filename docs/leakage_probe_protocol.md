# Leakage And Probe Protocol

This protocol separates three different evidence roles that should not be
merged in claims or artifacts.

The Q14 executable and planned training-mode contract is maintained in
[`q14_leakage_training_contract.md`](q14_leakage_training_contract.md).
The Q14 threshold table and claim wording policy are maintained in
[`q14_leakage_calibration.md`](q14_leakage_calibration.md).

## Evidence Roles

1. `optimizer_penalty`

   Training-time leakage losses that may contribute to the optimizer objective
   when their effective staged weights are positive and the execution mode is
   `trainable`.

2. `detached_training_diagnostic`

   Values computed from the same forward pass but detached from optimization.
   These are useful for monitoring and ablations, but they do not prove that
   off-modality information is absent.

3. `frozen_probe_audit`

   Held-out probes fit after the forward pass on frozen representations. This is
   the claim-facing leakage audit path because it tests whether off-modality
   targets are recoverable by separately trained probes.

Training summaries and leakage calibration artifacts expose these sections
separately through the leakage role report. Low optimizer penalties or low
training diagnostics are not sufficient for no-leakage wording unless the
frozen-probe audit also fails to beat trivial baselines across adequate capacity.
Calibration artifacts also persist `optimizer_penalty_separated=true` and the
`trivial_baseline` label so reviewer-facing reports cannot confuse
optimizer-facing penalties with held-out frozen-probe evidence.

## Frozen Probe Calibration

The frozen audit fits lightweight ridge probes on frozen pooled modality
embeddings. The report records:

- route
- source modality
- target family
- train and held-out counts
- probe capacity
- regularization
- held-out MSE
- target-mean baseline MSE
- improvement over the baseline

The audit is diagnostic evidence. It does not enforce independence by itself.
It answers a narrower question: can a separately trained probe recover the
off-modality target from a frozen source representation?

## Same-Modality Probe Baselines

Same-modality probe rows compare observed probe losses to trivial target-only
baselines for:

- topology adjacency
- geometry mean pairwise distance
- pocket atom features
- ligand pharmacophore roles
- pocket pharmacophore roles
- affinity scalar, when labels are available

Unavailable targets must be marked as `supervision_status = unavailable`; they
must not be serialized as zero-success evidence.

## Gradient-Flow Options

The current executable options are:

- `adversarial_penalty`: the penalty can backpropagate to the model when the
  staged leakage weight is active.
- `detached_diagnostic`: route values are reported but the optimizer receives an
  explicit zero objective from those routes.
- `probe_fit`: source features are detached and only explicit leakage probe
  heads are fit to off-modality targets.
- `encoder_penalty`: probe parameters are detached and the source encoders
  receive the encoder-facing leakage penalty.
- `alternating`: trainer steps alternate between `probe_fit` and
  `encoder_penalty`.

These modes must preserve the gradient-routing tests:

- probe parameters receive predictive gradients during the probe-fit phase
- source encoder parameters receive adversarial or reversal gradients only in
  the encoder phase
- detached diagnostic mode produces no optimizer-facing leakage gradient
- the whole protocol remains disableable for ablations

No-leakage claims still require the held-out frozen-probe audit and the
threshold policy in `q14_leakage_calibration.md`; training routes alone are
bounded diagnostics rather than semantic guarantees.
