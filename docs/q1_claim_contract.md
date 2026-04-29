# Q1 Claim Contract

This contract freezes manuscript vocabulary around traceable metric sources.

Binding and docking claims require real backend evidence from Vina or GNINA candidate-level metrics with backend provenance and coverage. `docking_like_score` and other pocket-fit proxies may appear only as heuristic analysis or failure-analysis signals.

Raw model capability must be reported from `raw_flow` or `raw_rollout` layers. `constrained_flow` is constrained-sampling evidence. `repaired`, `inferred_bond`, `deterministic_proxy`, and `reranked` layers are postprocessed evidence and must be labeled separately in tables and prose.

Allowed metric sources and provenance requirements are machine-readable in `configs/q1_claim_contract.json`. The local gate command is:

```bash
python tools/q1_readiness_audit.py --gate
```
