# Q13 Model And Training Risk Register

Date: 2026-04-29

## Claim-Safety Risks

| Rank | Risk | Affected Files | Failure Mode | Mitigation Task |
| --- | --- | --- | --- | --- |
| 1 | Target tensors leak into de novo conditioning | `src/models/system/forward.rs`, `src/data/features/*`, `src/data/batch.rs` | De novo claims accidentally depend on target ligand atom, topology, or coordinate tensors | `q13-p4-01`, `q13-p4-02` |
| 2 | Raw-native and processed evidence are conflated | `src/experiments/unseen_pocket/evaluation/*`, `src/models/evaluation/*`, claim docs | Postprocessed or backend-scored improvements are described as native model improvements | `q13-p6-01`, `q13-p6-02`, `q13-p6-04` |
| 3 | Full molecular flow wording exceeds implemented branch evidence | `src/models/flow/*`, `src/models/system/flow_training.rs`, configs | Geometry-only or partially enabled branch runs are promoted as full molecular flow | `q13-p2-01`, `q13-p2-03`, `q13-p6-03` |
| 4 | Frozen leakage evidence is confused with optimizer leakage penalties | `src/losses/leakage.rs`, `src/experiments/unseen_pocket/leakage_calibration.rs` | Probe quality drops but non-leakage claims are still made | `q13-p4-04` |
| 5 | Unseen-pocket splits permit identity overlap | `src/training/metrics/split_impl.rs`, `src/data/dataset/*` | Held-out-pocket claims are invalidated by protein, pocket, or example leakage | `q13-p4-03` |

## Engineering-Maintenance Risks

| Rank | Risk | Affected Files | Failure Mode | Mitigation Task |
| --- | --- | --- | --- | --- |
| 1 | `Phase1ResearchSystem` owns too many responsibilities | `src/models/system/*` | Refactors silently change active components or ablation meaning | `q13-p1-01`, `q13-p1-02`, `q13-p1-03`, `q13-p1-04` |
| 2 | Full-flow branch losses have weak schedule semantics | `src/config/types/root.rs`, `src/models/system/flow_training.rs`, `src/losses/task.rs` | Atom, bond, topology, context, and sync objectives destabilize early training | `q13-p2-03`, `q13-p3-01`, `q13-p3-02` |
| 3 | Objective scale diagnostics are too coarse | `src/losses/task.rs`, `src/training/metrics/losses.rs`, `src/training/trainer.rs` | A large branch or auxiliary term dominates updates without clear attribution | `q13-p3-01`, `q13-p3-03`, `q13-p3-05` |
| 4 | Trainer step remains hard to audit | `src/training/trainer.rs` | Scheduling, forward, loss, optimizer, metrics, and checkpoint logic become coupled | `q13-p7-01`, `q13-p7-03`, `q13-p7-04` |
| 5 | Slot and gate semantics can drift across ablations | `src/models/slot_decomposition.rs`, `src/models/interaction/*`, `src/losses/auxiliary.rs` | Dead slots, saturated gates, or direct-fusion controls pollute preferred-architecture claims | `q13-p5-01`, `q13-p5-02`, `q13-p5-03`, `q13-p5-04` |

## Evidence Required Before Strong Claims

- Held-out-pocket raw-native generation metrics must be available and separated
  from repair, pruning, reranking, and backend-scored layers.
- De novo initialization must show no target-ligand conditioning dependence.
- Full molecular flow claims require geometry, atom type, bond, topology, and
  pocket-context branches enabled with schedule and contract metadata.
- Slot/gate specialization claims require stability summaries, collapse
  warnings, and directed-gate diagnostics across seeds or ablations.
- Checkpoint replay claims must state whether optimizer internals are exact or
  metadata-only.
