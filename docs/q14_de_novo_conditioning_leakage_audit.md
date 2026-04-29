# Q14 De Novo Conditioning Leakage Audit

This audit records the de novo conditioning boundary after P7-01. The claim-bearing
de novo path must use only inference-time pocket/context inputs and internally
initialized ligand state for conditioning. Target ligand tensors remain
supervision labels only.

## Audited Paths

| Path | Contract | Evidence |
| --- | --- | --- |
| `de_novo_conditioning_modalities` | Builds topology and geometry from `initial_partial_ligand_state`; centers pocket context from `example.pocket`; does not read target atom types, target topology, target coordinates, noisy target coordinates, or target pair distances. | `de_novo_optimizer_conditioning_ignores_target_ligand_tensors` perturbs all target ligand tensors while holding pocket context fixed and verifies unchanged modality encodings, initial state, contexts, and decoder outputs. |
| Context refresh | Refreshes from the generated partial ligand state plus pocket context; no target ligand tensor is required for refreshed conditioning. | `de_novo_rollout_refresh_and_x0_ignore_target_ligand_tensors` enables every-step refresh and verifies identical rollout atom/coordinate states under target-ligand perturbation. |
| Flow `x0` construction | De novo and pocket-only modes cannot use `decoder_supervision.noisy_coords`, even when the generated scaffold atom count equals the target ligand atom count. The flow source is reported as `conditioning_scaffold_deterministic_noise_no_target_ligand`. | `de_novo_flow_targets_can_change_without_changing_conditioning` forces shape-coincident de novo scaffolds and proves reconstructed `x0` is unchanged while the supervised flow target and loss change. |
| Rollout initialization | Uses deterministic pocket-conditioned scaffold initialization and records `pocket_centroid_centered_conditioning_no_target_ligand_frame`. | Rollout provenance fields record de novo atom-count, topology, geometry, and coordinate-frame source labels. |
| Rollout guardrail diagnostics | No-target initialization modes compute valence guardrails from generated/scaffold adjacency rather than target ligand adjacency. Pocket clash and pharmacophore conflict diagnostics use pocket context only. | Target-ligand perturbation leaves rollout valence flags and generated states unchanged in the de novo refresh audit. |
| Data validation | Ligand-centered pocket/context tensors are reported as target-context leakage for de novo runs and may be rejected before training. | `de_novo_context_leakage_is_reported_and_optionally_rejected` covers report and rejection behavior. |

## Allowed Target-Ligand Uses

Training may still read target ligand tensors for optimizer-facing supervision:

- flow target coordinates `x1`
- atom-type, bond, topology, and pocket-interaction labels
- shape-safe target masks and matching provenance
- evaluation-only comparisons explicitly labeled as target supervision

Changing target labels is therefore expected to change supervised targets and
losses. It must not change de novo conditioning encodings, generated initial
state, refreshed conditioning state, or rollout initialization.

## Non-Claim Exceptions

`de_novo_initialization.dataset_calibrated_atom_count` is allowed only as an
explicit debug or ablation prior. It fixes the scaffold atom count from config
and is reported through `atom_count_prior_provenance=dataset_calibrated`; it must
not be described as claim-bearing pocket-conditioned atom-count inference.

Smoke tests or legacy synthetic datasets may still carry ligand-centered pocket
coordinates. For claim-bearing de novo evaluation, enable
`data.quality_filters.reject_target_ligand_context_leakage=true` or provide a
pocket-only coordinate frame before training/evaluation.
