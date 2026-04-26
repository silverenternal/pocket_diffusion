/// Lightweight chemistry-validity backend for active experiment reporting.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicChemistryValidityEvaluator;

/// Lightweight docking-oriented backend for executable experiment hooks.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicDockingEvaluator;

/// Lightweight pocket-compatibility backend for executable experiment hooks.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicPocketCompatibilityEvaluator;

/// Executable chemistry backend adapter.
#[derive(Debug, Clone)]
pub struct CommandChemistryValidityEvaluator {
    pub config: ExternalBackendCommandConfig,
}

/// Executable docking backend adapter.
#[derive(Debug, Clone)]
pub struct CommandDockingEvaluator {
    pub config: ExternalBackendCommandConfig,
}

/// Executable pocket-compatibility backend adapter.
#[derive(Debug, Clone)]
pub struct CommandPocketCompatibilityEvaluator {
    pub config: ExternalBackendCommandConfig,
}

/// Candidate records split by generation layer for model-vs-postprocessing review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CandidateGenerationLayers {
    /// Direct final rollout state before geometry repair and bond inference.
    pub raw_rollout: Vec<GeneratedCandidateRecord>,
    /// Geometry-repaired candidates before bond inference and valence pruning.
    pub repaired: Vec<GeneratedCandidateRecord>,
    /// Repaired candidates after distance bond inference and valence pruning.
    pub inferred_bond: Vec<GeneratedCandidateRecord>,
}

impl ChemistryValidityEvaluator for HeuristicChemistryValidityEvaluator {
    fn backend_name(&self) -> &'static str {
        "heuristic_validity_v1"
    }

    fn evaluate_chemistry(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        let total = candidates.len().max(1) as f64;
        let valid_fraction = candidates
            .iter()
            .filter(|candidate| basic_validity(candidate))
            .count() as f64
            / total;
        let valence_sanity_fraction = candidates
            .iter()
            .filter(|candidate| valence_sane(candidate))
            .count() as f64
            / total;
        let structural_pass_fraction = candidates
            .iter()
            .filter(|candidate| structural_pass(candidate))
            .count() as f64
            / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("valid_fraction", valid_fraction),
                metric("valence_sanity_fraction", valence_sanity_fraction),
                metric("structural_pass_fraction", structural_pass_fraction),
            ],
        }
    }
}

impl DockingEvaluator for HeuristicDockingEvaluator {
    fn backend_name(&self) -> &'static str {
        "heuristic_docking_hook_v1"
    }

    fn evaluate_docking(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        let total = candidates.len().max(1) as f64;
        let pocket_contact_fraction = candidates
            .iter()
            .filter(|candidate| candidate_contact_pocket(candidate))
            .count() as f64
            / total;
        let mean_centroid_offset = candidates
            .iter()
            .map(centroid_offset_from_pocket)
            .sum::<f64>()
            / total;
        let centroid_fit_score = candidates.iter().map(centroid_fit_score).sum::<f64>() / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("pocket_contact_fraction", pocket_contact_fraction),
                metric("mean_centroid_offset", mean_centroid_offset),
                metric("centroid_fit_score", centroid_fit_score),
            ],
        }
    }
}

impl PocketCompatibilityEvaluator for HeuristicPocketCompatibilityEvaluator {
    fn backend_name(&self) -> &'static str {
        "heuristic_pocket_compatibility_v1"
    }

    fn evaluate_pocket_compatibility(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        let total = candidates.len().max(1) as f64;
        let centroid_inside_fraction = candidates
            .iter()
            .filter(|candidate| {
                centroid_offset_from_pocket(candidate) <= candidate.pocket_radius as f64
            })
            .count() as f64
            / total;
        let atom_coverage_fraction =
            candidates.iter().map(atom_coverage_fraction).sum::<f64>() / total;
        let clash_free_fraction = candidates
            .iter()
            .filter(|candidate| non_bonded_clash_fraction(candidate) == 0.0)
            .count() as f64
            / total;
        let strict_pocket_fit_score =
            candidates.iter().map(strict_pocket_fit_score).sum::<f64>() / total;

        ExternalEvaluationReport {
            backend_name: self.backend_name().to_string(),
            metrics: vec![
                metric("centroid_inside_fraction", centroid_inside_fraction),
                metric("atom_coverage_fraction", atom_coverage_fraction),
                metric("clash_free_fraction", clash_free_fraction),
                metric("strict_pocket_fit_score", strict_pocket_fit_score),
            ],
        }
    }
}

impl ChemistryValidityEvaluator for CommandChemistryValidityEvaluator {
    fn backend_name(&self) -> &'static str {
        "external_command_chemistry"
    }

    fn evaluate_chemistry(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        evaluate_via_command(self.backend_name(), &self.config, candidates)
    }
}

impl DockingEvaluator for CommandDockingEvaluator {
    fn backend_name(&self) -> &'static str {
        "external_command_docking"
    }

    fn evaluate_docking(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        evaluate_via_command(self.backend_name(), &self.config, candidates)
    }
}

impl PocketCompatibilityEvaluator for CommandPocketCompatibilityEvaluator {
    fn backend_name(&self) -> &'static str {
        "external_command_pocket_compatibility"
    }

    fn evaluate_pocket_compatibility(
        &self,
        candidates: &[GeneratedCandidateRecord],
    ) -> ExternalEvaluationReport {
        evaluate_via_command(self.backend_name(), &self.config, candidates)
    }
}

/// Build deterministic candidate records from one modular forward pass.
pub(crate) fn generate_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_layered_candidates_from_forward(example, forward, num_candidates).inferred_bond
}

/// Build raw, repaired, and inferred-bond candidate layers from one modular forward pass.
pub(crate) fn generate_layered_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> CandidateGenerationLayers {
    generate_layered_candidates_with_options(example, forward, num_candidates, true)
}

/// Build candidate layers with optional geometry repair for ablation attribution.
pub(crate) fn generate_layered_candidates_with_options(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
    enable_repair: bool,
) -> CandidateGenerationLayers {
    if num_candidates == 0 {
        return CandidateGenerationLayers {
            raw_rollout: Vec::new(),
            repaired: Vec::new(),
            inferred_bond: Vec::new(),
        };
    }

    let final_step = forward.generation.rollout.steps.last();
    let rollout_atom_types = final_step
        .map(|step| step.atom_types.clone())
        .unwrap_or_default();
    let rollout_coords = final_step
        .map(|step| step.coords.clone())
        .unwrap_or_default();
    let logits = &forward.generation.decoded.atom_type_logits;
    let topk_count = logits.size().get(1).copied().unwrap_or(1).clamp(1, 2);
    let topk = logits.topk(topk_count, -1, true, true).1;
    let pocket_centroid = tensor_centroid(&example.pocket.coords);
    let pocket_radius = pocket_radius(&example.pocket.coords, pocket_centroid);
    let pocket_points = tensor_to_coords(&example.pocket.coords);
    let base_bonds = infer_bonds(&rollout_coords);

    let mut raw_rollout = Vec::with_capacity(num_candidates);
    let mut repaired_candidates = Vec::with_capacity(num_candidates);
    let mut inferred_bond_candidates = Vec::with_capacity(num_candidates);

    for candidate_ix in 0..num_candidates {
        raw_rollout.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=raw_rollout;atoms={};bonds=0",
                rollout_atom_types.len()
            )),
            atom_types: rollout_atom_types.clone(),
            coords: rollout_coords.clone(),
            inferred_bonds: Vec::new(),
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "raw_modular_rollout:steps={};candidate={candidate_ix}",
                forward.generation.rollout.executed_steps
            ),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });

        let coords = if enable_repair {
            repair_candidate_geometry(
                &rollout_coords,
                &pocket_points,
                pocket_centroid,
                pocket_radius,
                candidate_ix,
                num_candidates,
            )
        } else {
            rollout_coords.clone()
        };
        let inferred_bonds = infer_bonds(&coords);
        repaired_candidates.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=repaired_no_bonds;atoms={};bonds=0",
                rollout_atom_types.len()
            )),
            atom_types: rollout_atom_types.clone(),
            coords: coords.clone(),
            inferred_bonds: Vec::new(),
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "geometry_repair:steps={};candidate={candidate_ix}",
                forward.generation.rollout.executed_steps
            ),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });

        let atom_types = choose_candidate_atom_types(
            &rollout_atom_types,
            logits,
            &topk,
            topk_count,
            &inferred_bonds,
            &base_bonds,
            candidate_ix,
        );
        let inferred_bonds = prune_bonds_for_valence(&coords, &atom_types, &inferred_bonds);
        inferred_bond_candidates.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=modular;atoms={};bonds={}",
                atom_types.len(),
                inferred_bonds.len()
            )),
            atom_types,
            coords,
            inferred_bonds,
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "modular_rollout_decoder:steps={}",
                forward.generation.rollout.executed_steps
            ),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });
    }

    CandidateGenerationLayers {
        raw_rollout,
        repaired: repaired_candidates,
        inferred_bond: inferred_bond_candidates,
    }
}

