# Q7 Claim Boundaries

The machine-readable contract is
`configs/q7_claim_boundary_contract.json`. It applies to the Q7 model,
training, evaluation, and ablation updates.

Evidence tiers:

| Tier | Claim use |
| --- | --- |
| `synthetic_smoke` | Compile, wiring, numerical-health, and regression checks only. |
| `real_data_debug` | Parser, trainer, and metric-path debugging on capped manifest data. |
| `reviewer_scale_unseen_pocket` | Held-out-pocket reviewer evidence when split, backend, leakage, replay, and model-design checks are present. |
| `heuristic_metric` | Diagnostic proxy only with explicit heuristic provenance. |
| `backend_supported` | Backend-supported chemistry or pocket metric only with candidate-level coverage and provenance. |
| `docking_supported` | Score-only docking or pocket-fit evidence with backend availability, input completeness, and score coverage. |
| `experimental` | Reserved for explicit experimental evidence; currently unavailable in this repository. |

Raw model-native quality must come from `model_design.raw_model_*`, the
`model_design.raw_native_*` graph metrics, and `raw_rollout`. Native graph
metrics include raw bond count, component count, valence-violation fraction,
topology/bond payload sync, atom-type entropy, and raw native graph validity.
They are computed before repair, inferred-bond replacement, valence pruning, or
reranking. Processed, constrained, repaired, inferred-bond, reranked, and
backend-scored fields may only be cited with their postprocessing chain and
claim-boundary note.
Candidate records carry this attribution directly through `generation_mode`,
`generation_layer`, `generation_path_class`, `model_native_raw`,
`postprocessor_chain`, and `claim_boundary`. Claim summaries also persist
`layer_provenance_audit`; model-design language is claim-safe only when that
audit passes.

Supported generation-mode labels are `target_ligand_denoising`,
`ligand_refinement`, `flow_refinement`,
`pocket_only_initialization_baseline`, and `de_novo_initialization`. The
pocket-only baseline uses a
configured atom-count prior and pocket-centroid coordinate offsets for decoder
initialization and currently validates only with the shape-safe
`surrogate_reconstruction` objective; it is a conservative initialization
baseline, not evidence for true de novo molecular generation.
`surrogate_reconstruction` is a bootstrap/debug objective for smoke tests and
shape-safe baselines, not a generation-quality objective. `denoising_flow_matching`
is a hybrid training-objective composition over denoising and flow terms; reports
persist `primary_objective_provenance` and `primary_objective_claim_boundary` so
objective source cannot be read as a separate generation mode.
`de_novo_initialization` is executable only with the flow-matching backend,
`flow_matching.geometry_only=false`, and all five molecular flow branches
enabled. De novo wording also requires `claim_context.de_novo_claim_allowed`,
the pocket-centered conditioning frame, and no target-ligand tensor use as
conditioning input; target ligand tensors are training supervision only.

The Q7 reviewer-scale surface must include the `model_design` evaluation block
and the core ablation variants listed in `configs/q7_core_ablation_matrix.json`
before using model-design language about encoders, geometry operators, slots,
gates, leakage, or decoder conditioning.

The `direct_fusion_negative_control` ablation is claim-facing only as a
negative control. It forces directed interaction paths open for comparison and
must not be described as the preferred model architecture.

The `topology_only`, `geometry_only`, and `pocket_only` ablations are
claim-facing only as modality-dependence controls. They preserve separate
encoders and zero non-selected slot branches downstream, so they do not support
claims that the architecture should collapse to a single modality. The
`staged_schedule_disabled` ablation is a training-procedure stress test and
must not be cited as evidence that staged training is unnecessary unless the
reviewer-scale matrix shows matching raw quality, leakage, slot, and gate
metrics.

Backend score-only metrics, including Vina/GNINA-style docking scores, may be
described only as docking-supported score evidence with backend/input coverage.
They must not be described as experimental binding affinity. No Q7 artifact
currently supports experimental binding affinity, selectivity, efficacy, or
human preference claims.
