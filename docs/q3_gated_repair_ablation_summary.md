# Q3 Gated Repair Ablation Summary

Gated repair evidence is postprocessing evidence. repair_rejected is a safety fallback that preserves raw coordinates and must not be reported as native model improvement.

## Gate Counts

- `raw_passthrough`: 300
- `repaired_candidate`: 0
- `rejected_repair`: 300

## Layer Summary

| Layer | Candidates | Vina Cov | GNINA Cov | Vina | GNINA | CNN | QED | SA | Clash | Contact | dVina | dGNINA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_repair | 300 | 0.8967 | 1 | -2.202 | -2.297 | 0.4494 | 0.4325 | 2.242 | 0.002659 | 0.9733 | NA | NA |
| full_repair | 300 | 1 | 1 | 156.9 | 156.9 | 0.2402 | 0.4315 | 2.284 | 0.1034 | 0.9381 | 162.9 | 159.2 |
| gated_repair | 0 | 0 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| repair_rejected | 300 | 0.8967 | 1 | -2.202 | -2.297 | 0.4494 | 0.4325 | 2.242 | 0.002659 | 0.9733 | 0 | 0 |

## Interpretation

- Legacy full_repair remains a postprocessing-only layer and is not promoted as model-native evidence.
- The configured gate rejected every coordinate-moving repair because raw candidates were already backend-input dockable and proposed repair exceeded movement or box-center bounds.
- repair_rejected is a raw-coordinate passthrough layer; its backend scores should be interpreted as a non-degradation guard, not a model improvement.
- Vina mean degradation was reduced from legacy full_repair delta 162.9 to repair_rejected delta 0.
- GNINA affinity mean degradation was reduced from legacy full_repair delta 159.2 to repair_rejected delta 0.
