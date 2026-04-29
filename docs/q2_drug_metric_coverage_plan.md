# Q2 Drug Metric Coverage Plan

Existing backends already cover the core Q2 table fields. Table scripts should aggregate these fields rather than reimplement chemistry or docking logic.

| Metric family | Fields | Status |
| --- | --- | --- |
| Vina score_only | `vina_score`, `vina_score_mean`, `vina_score_success_fraction`, `vina_failure_reason` | implemented |
| GNINA score_only | `gnina_affinity`, `gnina_affinity_mean`, `gnina_cnn_score`, `gnina_cnn_score_mean` | implemented |
| Drug-likeness | `qed`, `sa_score`, `logp`, `tpsa`, `lipinski_violations` | implemented |
| Diversity and novelty | `scaffold_novelty_fraction`, `unique_scaffold_fraction`, `pairwise_tanimoto_mean`, `nearest_train_similarity`, `rdkit_unique_smiles_fraction` | implemented |
| Pocket geometry | `contact_fraction`, `clash_fraction`, `centroid_offset`, `centroid_inside_fraction`, `mean_min_contact_distance` | implemented |
| Runtime and failures | `backend_runtime_seconds`, `input_count`, `scored_count`, `failure_count`, `failure_reasons` | implemented for scoring/merge contracts |
| PAINS and Brenk | no current fields | feasible extension pending RDKit support check |

`sa_score` is a local proxy implementation in `tools/rdkit_validity_backend.py`, so Q2 tables should label it as a proxy SA score. Vina and GNINA outputs are score-only backend values and must not be described as experimental binding affinity.

Missing values should render as `NA` with coverage, never as zero-filled placeholders.
