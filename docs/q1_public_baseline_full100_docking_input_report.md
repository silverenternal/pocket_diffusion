# Docking Input Preparation

- candidate_count: 300
- prepared_count: 300
- failure_count: 0
- prepared_fraction: 1.0000
- backend_candidate_json: `checkpoints/q1_public_baselines_full100/docking_inputs/docking_candidates.json`

## Assumptions
- receptor: copy existing PDBQT when present; otherwise convert parseable PDB ATOM/HETATM records to minimal PDBQT
- ligand: write generated candidate coordinates to minimal PDBQT while preserving candidate_id
- protonation: preserve source receptor hydrogens; do not add ligand hydrogens
- charges: 0.000 partial charge for minimal conversions
- sanitization: finite-coordinate validation only; RDKit sanitization remains a separate metric backend

## Failure Reasons
- none
