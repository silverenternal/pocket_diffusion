# Q1 Case Studies

Selection rule: per-pocket score = valid + contact + qed - clash; deterministic sorted pockets

## success

| candidate | pocket | layer | score |
|---|---|---|---:|
| `conditioned_denoising:raw_rollout:1g53:0` | `1g53` | `raw_rollout` | 1.359 |
| `conditioned_denoising:repaired:1g74:18` | `1g74` | `repaired` | 1.343 |
| `conditioned_denoising:repaired:1aaq:3` | `1aaq` | `repaired` | 1.324 |
| `conditioned_denoising:raw_rollout:1bnq:9` | `1bnq` | `raw_rollout` | 1.314 |
| `conditioned_denoising:raw_rollout:1c1r:21` | `1c1r` | `raw_rollout` | 1.273 |

## failure

| candidate | pocket | layer | score |
|---|---|---|---:|
| `conditioned_denoising:inferred_bond:1a4w:14` | `1a4w` | `inferred_bond` | 0.9949 |
| `conditioned_denoising:inferred_bond:1c1r:23` | `1c1r` | `inferred_bond` | 1.029 |
| `conditioned_denoising:inferred_bond:1aaq:4` | `1aaq` | `inferred_bond` | 1.029 |
| `conditioned_denoising:inferred_bond:1g2k:8` | `1g2k` | `inferred_bond` | 1.056 |
| `conditioned_denoising:inferred_bond:1h22:17` | `1h22` | `inferred_bond` | 1.06 |

## postprocessing_improved

| candidate | pocket | layer | score |
|---|---|---|---:|
