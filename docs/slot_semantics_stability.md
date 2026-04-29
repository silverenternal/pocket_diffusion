# Slot Semantics And Stability

This document records the current slot visibility and evidence protocol.

## Visibility Schedule

The current implementation keeps the optimizer path conservative:

- Slot activation gates remain differentiable sigmoid values.
- Hard active-slot counts remain diagnostics.
- Attention visibility can keep a configurable minimum number of slots visible
  through `model.slot_decomposition.minimum_visible_slots`.
- The current hard-mask behavior is recoverable with
  `attention_masking = true` and `minimum_visible_slots = 0`.
- Full stage-aware soft-to-hard scheduling is not enabled yet because the
  decomposer does not receive trainer stage context.

Recommended future stage behavior:

1. Early warmup: use soft visibility or minimum-visible slots so low random
   activations do not hide all slots from cross-attention or decoder local
   attention.
2. Middle stages: decay the minimum-visible rule while retaining activation
   probabilities for redundancy and slot-balance losses.
3. Late stages: use hard visibility for attention masks while keeping hard
   active counts as diagnostics only.

Effects by component:

- Cross-attention: visibility controls which slots can send or receive directed
  gated attention updates.
- Decoder local attention: visibility controls which modality slots can be used
  for atom-local conditioning.
- Redundancy loss: should use activation gates and active-modality masks, not
  hidden disabled slots.
- Slot metrics: must report hard active count, attention-visible fraction,
  assignment entropy, and dominant-slot fraction separately.

## Per-Slot Alignment

Evaluation artifacts now report per-slot alignment vectors for:

- topology slots against adjacency density
- geometry slots against inverse mean pairwise distance
- pocket slots against pocket feature magnitude

These are lightweight semantic alignment diagnostics. They avoid unstable
mutual-information estimators and should be interpreted as specialization
proxies rather than direct semantic labels.

## Signature Matching

Slot semantics can permute across seeds or repeated runs. The matching protocol
uses cosine similarity over slot signatures and greedily matches the strongest
pairs above a threshold.

Reported fields include:

- matched slot count
- unmatched left and right slots
- mean matched similarity
- collapse warning flag

Within a single evaluation split, the current artifact reports a deterministic
first-half versus second-half repeated-signature proxy. Multi-seed runs can use
the same matching function across run summaries.

Training step summaries now also persist active-modality slot signature
summaries. Each record includes slot activation, attention visibility,
assignment entropy, semantic-probe alignment, a bounded mean signature vector,
and a within-step repeated-signature matching proxy. Modality-focused ablations
that zero a branch are skipped so disabled topology, geometry, or pocket slots do
not pollute active-modality summaries.

## Collapse Alarms

Evaluation artifacts include per-modality collapse warnings:

- `dead`: no active or attention-visible slots
- `saturated`: nearly all slots active
- `single_slot_dominated`: one slot carries most assignment mass
- `balanced`: conservative diagnostic bounds are satisfied

These warnings do not change optimizer behavior. They are intended for trainer
readiness checks and claim review.
