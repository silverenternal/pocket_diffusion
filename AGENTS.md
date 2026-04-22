# AGENTS

## Scope

This repository is a Rust-first research codebase for pocket-conditioned molecular generation.

All substantial code changes should preserve these project-level constraints:

- Prefer Rust for core implementation and experimentation infrastructure.
- Optimize for runtime efficiency, memory efficiency, and numerical stability.
- Keep module boundaries explicit and ablation-friendly.
- Preserve physically necessary cross-modality dependencies; do not force strict independence.
- Maintain compilable, incrementally usable code at each implementation phase.

## Research Objective

Implement a modular framework for:

`semantic-preserving minimal-redundancy representation learning for pocket-conditioned molecular generation`

The intended system should support:

- automatic structured decomposition of entangled molecular information
- minimal redundancy within each modality
- controlled cross-modality interaction
- semantic specialization via auxiliary supervision
- staged training and reproducible ablations
- generalization to unseen protein pockets

## Enduring Architecture Constraints

### Modalities

Keep three encoders structurally separate:

1. topology encoder
2. geometry encoder
3. pocket/context encoder

Do not collapse them into a single unrestricted fusion encoder.

### Slot-Based Decomposition

Each modality representation should support learned slot decomposition with:

- fixed upper-bound slot count
- soft activation or gating
- sparse utilization
- automatically learned semantics

This is structured decomposition, not per-sample MoE routing.

### Intra-Modality Redundancy Reduction

Support stable redundancy-reduction objectives inside each modality:

- covariance decorrelation
- predictability penalty
- optional lightweight dependence penalty such as HSIC

Do not make unstable mutual-information estimators the central optimization mechanism.

### Semantic Probes And Leakage Control

Support lightweight semantic probes that preserve specialization:

- topology -> adjacency / bond structure targets
- geometry -> distance / geometric targets
- context -> pocket feature targets

Also support leakage probes to penalize excessive prediction of off-modality targets.

### Controlled Cross-Modality Interaction

Cross-modality information exchange should remain gated and explicit.

Preferred pattern:

`A(m <- n) = gate(m,n) * Attention(Q_m, K_n, V_n)`

Requirements:

- modality-specific Q/K/V projections
- learned gates in `[0, 1]`
- sparse gate regularization
- no naive unrestricted full fusion

### Slot Utilization Control

Support:

- slot sparsity losses
- anti-collapse / balance losses
- activation statistics logging

## Training Constraints

Use staged training rather than enabling all regularizers from step zero.

Recommended progression:

1. `L_task + L_consistency`
2. add low-weight `L_intra_red`
3. add `L_probe + L_leak`
4. add `L_gate + L_slot`

All auxiliary objectives should be configurable, warm-startable, and easy to disable for ablation.

## Data And Evaluation

The data layer should support:

- protein-ligand examples with graph, geometry, and pocket context features
- unseen-pocket train/val/test splitting
- efficient batching
- experiment-level evaluation hooks

Important evaluation directions:

- unseen protein pocket generalization
- geometric constraint handling
- efficiency analysis
- ablation studies

Metrics should include task quality, geometric consistency, efficiency, slot usage, gate usage, and leakage statistics when available.

## Software Design Rules

- Prefer static dispatch and generics where practical.
- Avoid unnecessary cloning and intermediate allocations.
- Document all public APIs.
- Keep components replaceable through traits and clear interfaces.
- Do not over-centralize logic into one file.
- Do not replace controlled interaction with naive fusion for convenience.
- Do not rewrite core modeling logic into Python.

Important trait boundaries should remain easy to swap:

- encoders
- slot decomposers
- cross-modal interaction modules
- loss terms
- trainer hooks
- datasets

## Delivery Strategy

When extending the system, work in phases:

1. project structure, config, data structs, core traits, encoder skeletons
2. slot decomposition, gated cross-attention, probe heads
3. loss modules, staged trainer, logging, checkpointing
4. unseen-pocket experiments, metrics, ablations, evaluation

At each phase:

- keep the crate compiling
- keep interfaces clean
- avoid placeholder-only architecture with no runnable path

## Priority Order

When tradeoffs arise, prioritize:

1. correctness of the research constraints above
2. runtime and memory efficiency
3. modularity for ablation and experimentation
4. maintainability of the Rust codebase
