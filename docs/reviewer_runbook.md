# Reviewer Runbook

The reviewer-scale workflow is maintained in
[`q8_reviewer_scale_runbook.md`](q8_reviewer_scale_runbook.md). Use that file
for config validation, training, evaluation, ablation, claim-gate, and artifact
refresh commands.

Generation-mode wording follows `configs/q7_claim_boundary_contract.json`:
`target_ligand_denoising`, `ligand_refinement`, `flow_refinement`,
`pocket_only_initialization_baseline`, and `de_novo_initialization` are
supported execution labels. Only `de_novo_initialization` supports de novo
wording, and only when paired with the flow-matching backend,
`flow_matching.geometry_only=false`, and all five molecular flow branches. The
pocket-only baseline remains a low-claim initialization baseline and should use
the shape-safe `surrogate_reconstruction` primary objective.

When reviewing model quality, read raw-native results first:
`model_design.raw_model_*`, `model_design.raw_native_*`, and
`layered_generation_metrics.raw_rollout` are the pre-repair model outputs.
Processed layers must be interpreted through their `postprocessor_chain` and
`claim_boundary`; backend score-only metrics are docking-supported scores, not
experimental binding affinity.
