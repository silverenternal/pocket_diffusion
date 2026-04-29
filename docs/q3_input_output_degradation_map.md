# Q3 Input-To-Output Degradation Map

This audit treats poor metrics as the final symptom, not the diagnosis. The
pipeline must be inspected from input coordinates through model conditioning,
flow sampling, decoding, repair, reranking, and backend export.

## Main Degradation Paths

1. Coordinate-frame or docking-box mismatch can make an otherwise unchanged
   candidate score badly. This must be checked before interpreting model
   changes.
2. Separate topology, geometry, and pocket encoders can lose local information
   through slot collapse or over-pooled summaries.
3. Current flow velocity prediction can underuse local atom-pocket contacts,
   so raw coordinates may look globally pocket-adjacent while remaining locally
   incompatible with receptor atoms.
4. Constrained sampling can damage a raw pose if it applies hard geometric
   correction without a non-degradation gate.
5. Bond inference and chemistry refinement should be audited separately from
   coordinate movement. Bond-only changes mainly affect validity, SA, QED, and
   backend molecular payloads.
6. Repair is currently the clearest postprocessing risk. Centroid translation
   alone can cause severe docking degradation, so every repair step needs
   per-candidate deltas and rejection reasons.
7. Proxy reranking is unsafe unless backend correlation is demonstrated on
   held-out scored candidates.

## Required Stage Observables

- Input/export: ligand centroid, pocket centroid, coordinate frame origin,
  docking box center, docking box size.
- Encoder/slot: active slot fractions, slot balance, leakage probe scores.
- Cross-modal conditioning: gate means, atom-pocket attention entropy,
  pocket-token utilization.
- Flow: velocity norm, per-step coordinate delta, nearest pocket atom distance,
  nearest ligand-neighbor distance, rotation consistency error.
- Constraints: raw-to-constrained centroid shift, pair-distance delta, clash
  delta, atom-type entropy.
- Chemistry: bond count, valence violations, sanitization status,
  coordinate max absolute delta, SA/QED deltas.
- Repair: raw-to-repair centroid shift, RMSD, clash/contact deltas, acceptance
  or rejection reason.
- Reranking/export: proxy component scores, proxy/backend rank correlation,
  exported atom and coordinate counts, model_native flag.

## Practical Priority

The next Q3 work should start with this order:

1. Verify coordinate frame and backend export.
2. Compare raw_flow to constrained_flow before repair.
3. Split coordinate-preserving chemistry refinement from coordinate-moving
   repair.
4. Add model changes only where the pipeline audit shows native raw_flow loss.
5. Use proxy or preference methods only after backend-backed correlation checks.
