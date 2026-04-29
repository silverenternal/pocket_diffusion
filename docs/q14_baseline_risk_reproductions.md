# Q14 Baseline Risk Reproductions

Date: 2026-04-29

These are compact reproductions and diagnostics for the highest-risk Q14
training semantics. They record the current behavior before fixes, not the
desired final claim state.

## Reproduction Matrix

| Risk | Minimal Reproduction | Current Behavior | Expected Fixed Behavior |
| --- | --- | --- | --- |
| De novo auxiliary probe shape mismatch | Configure `generation_mode=de_novo_initialization` with generated atom count different from target ligand atom count, run enough steps to enter Stage3 so `L_probe` is active, then compute `ProbeLoss::compute_weighted_components`. Active code paths: `src/models/system/forward.rs:174`, `src/losses/probe.rs:215`. | Topology adjacency BCE and geometry distance MSE currently compare probe tensors directly with target ligand tensors. When generated rows differ from target rows, these paths can panic through tensor shape mismatch instead of reporting a masked/skip reason. | Probe losses use a shared alignment contract. Exact-shape target-ligand denoising remains unchanged; de novo mismatches either mask aligned rows or emit explicit zero-valued skipped diagnostics with provenance. |
| Index-based de novo target alignment | Run de novo full molecular flow with `target_alignment_policy=pad_with_mask` or `truncate` on a target whose atom order is permuted. Active code paths: `src/models/system/flow_training.rs:29`, `src/losses/task.rs:558`. | Molecular flow target rows are aligned by leading row/index policy. `pad_with_mask` makes shape handling safe, but the supervision is still order-dependent and can teach atom ordering artifacts. | A molecular target matching abstraction provides permutation-invariant matching for geometry, atom type, bond, and topology targets, with matching provenance emitted in training/evaluation artifacts. |
| Leakage probe semantics ambiguity | Enable explicit leakage probes and compare `ExplicitLeakageProbeTrainingSemantics::AdversarialPenalty` against detached diagnostic mode. Existing diagnostics: `src/losses/leakage.rs:268`, `src/losses/leakage.rs:908`, `src/experiments/unseen_pocket/leakage_calibration.rs:67`. | The code now separates optimizer penalty, detached diagnostic, and frozen-probe audit sections, but optimizer-facing explicit leakage penalties can still update the same probe/representation path. Low training-time leakage loss is therefore not, by itself, held-out non-leakage evidence. | Probe fitting and encoder penalty are split. Claim summaries rely on held-out frozen leakage probes, while optimizer penalties remain clearly labeled training objectives. |
| Batched rollout graph construction | Compare `forward_batch` against per-example `forward_example` for target-ligand denoising and de novo flow-time contexts. Existing checks: `src/models/system/tests.rs:675`, `src/models/system/tests.rs:837`, `src/training/trainer.rs:658`. | Target-ligand denoising uses batched encoders and then constructs rollout records during per-example assembly. De novo and flow-time-conditioned interaction intentionally use per-example forwarding. Some batched-path rollout diagnostics historically risked being confused with optimizer-facing graph construction; the current single-example path wraps rollout in `no_grad`, while the batched assembly still needs P6 confirmation. | All sampled rollout diagnostics are constructed under no-grad semantics and reported as detached diagnostics; optimizer-facing records can be requested independently of rollout artifacts. |

## Stage3/Stage4 Diagnostic Scenario

A compact Stage3/Stage4 smoke should use a synthetic mini-dataset and a schedule
with `stage1_steps=0`, `stage2_steps=0`, and either `stage3_steps=0` for Stage4
or one step in Stage3. The config should use:

- `generation_mode=de_novo_initialization`
- flow-matching backend with full molecular branches enabled
- generated atom count intentionally different from the target ligand count
- `training.disable_probe=false`
- `training.disable_leakage=false`
- `training.disable_gate=false`
- `training.disable_slot=false`

Current expected result before P1/P6:

- Stage3/Stage4 routing activates probe, leakage, gate, and slot objectives.
- Molecular flow primary branches are shape-safe through existing target masks.
- Same-modality `ProbeLoss` remains the highest-risk panic/mis-supervision
  point for mismatched atom counts.
- Leakage diagnostics must be interpreted as training-time signals unless the
  frozen held-out leakage calibration artifact is present.

## Existing Baseline Evidence

- `src/models/system/tests.rs:837` checks that de novo full molecular flow uses
  pocket scaffolds and records molecular branches.
- `src/models/system/tests.rs:957` and `src/models/system/tests.rs:1067`
  check that de novo optimizer conditioning is isolated from target ligand
  tensors and that target changes do not change conditioning tensors.
- `src/training/trainer.rs:3454` checks a pocket-only mismatch case, but it is
  not sufficient for de novo Stage3/Stage4 probe objectives.
- `src/experiments/unseen_pocket/leakage_calibration.rs:379` checks frozen
  leakage probe calibration on held-out routes.

These diagnostics justify Q14 P1 through P6. They should not be presented as
model-quality or de novo molecular-generation benchmark claims.
