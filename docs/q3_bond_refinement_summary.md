# Q3 Bond Refinement Summary

This artifact evaluates coordinate-preserving bond refinement separately from the coordinate-moving repaired reference. Vina/GNINA are score_only backends, not experimental affinity.

| Layer | Candidates | Vina cov | GNINA cov | Vina | GNINA | QED | SA | Bonds | Valence viol | Coord preserving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw_geometry | 100 | 1 | 1 | 24.94 | 24.94 | 0.4174 | 2.695 | 0 | 0 | None |
| bond_logits_refined | 100 | 1 | 1 | 24.94 | 24.94 | 0.3677 | 2.731 | 26.18 | 2.38 | True |
| valence_refined | 100 | 1 | 1 | 24.94 | 24.94 | 0.2683 | 3.686 | 23.24 | 0 | True |
| repaired | 100 | 1 | 0 | 79.7 | NA | 0.4732 | 4.092 | 24.69 | 0 | False |

## Mean Deltas Vs Raw Geometry

- `bond_logits_refined`: dVina=0, dGNINA=0, dQED=-0.1204, dSA=0.5445, dBonds=26.18, dValenceViol=2.38
- `valence_refined`: dVina=0, dGNINA=0, dQED=-0.1491, dSA=0.9912, dBonds=23.24, dValenceViol=0
- `repaired`: dVina=54.75, dGNINA=NA, dQED=0.05582, dSA=1.398, dBonds=24.69, dValenceViol=0
