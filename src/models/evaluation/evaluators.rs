use super::candidates::{choose_candidate_atom_types, repair_candidate_geometry};
use super::scoring::*;
use super::*;
use crate::models::CandidateLayerKind;

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
    /// Raw final geometry before coordinate movement or bond-payload refinement.
    pub raw_geometry: Vec<GeneratedCandidateRecord>,
    /// Direct final rollout state before geometry repair and bond inference.
    pub raw_rollout: Vec<GeneratedCandidateRecord>,
    /// Raw-coordinate candidates with refined bond payloads before valence pruning.
    pub bond_logits_refined: Vec<GeneratedCandidateRecord>,
    /// Raw-coordinate candidates after conservative valence pruning.
    pub valence_refined: Vec<GeneratedCandidateRecord>,
    /// Geometry-repaired candidates before bond inference and valence pruning.
    pub repaired: Vec<GeneratedCandidateRecord>,
    /// Repaired candidates after distance bond inference and valence pruning.
    pub inferred_bond: Vec<GeneratedCandidateRecord>,
}

impl CandidateGenerationLayers {
    fn empty() -> Self {
        Self {
            raw_geometry: Vec::new(),
            raw_rollout: Vec::new(),
            bond_logits_refined: Vec::new(),
            valence_refined: Vec::new(),
            repaired: Vec::new(),
            inferred_bond: Vec::new(),
        }
    }

    fn extend(&mut self, other: Self) {
        self.raw_geometry.extend(other.raw_geometry);
        self.raw_rollout.extend(other.raw_rollout);
        self.bond_logits_refined.extend(other.bond_logits_refined);
        self.valence_refined.extend(other.valence_refined);
        self.repaired.extend(other.repaired);
        self.inferred_bond.extend(other.inferred_bond);
    }
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
#[allow(dead_code)]
pub fn generate_raw_rollout_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_layered_candidates_from_forward(example, forward, num_candidates).raw_rollout
}

/// Build explicit constrained-flow candidates from one modular forward pass.
#[allow(dead_code)]
pub fn generate_constrained_flow_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_layered_candidates_from_forward(example, forward, num_candidates).inferred_bond
}

/// Build explicit repaired-coordinate candidates from one modular forward pass.
#[allow(dead_code)]
pub fn generate_repaired_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_layered_candidates_from_forward(example, forward, num_candidates).repaired
}

/// Build explicit inferred-bond candidates from one modular forward pass.
#[allow(dead_code)]
pub fn generate_inferred_bond_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_layered_candidates_from_forward(example, forward, num_candidates).inferred_bond
}

/// Build explicit claim-facing candidates from one modular forward pass.
pub fn generate_claim_facing_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_inferred_bond_candidates_from_forward(example, forward, num_candidates)
}

#[deprecated(
    note = "Compatibility wrapper retained for legacy callers; use `generate_claim_facing_candidates_from_forward` (or one explicit layer helper) instead."
)]
#[allow(dead_code)]
pub fn generate_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> Vec<GeneratedCandidateRecord> {
    generate_claim_facing_candidates_from_forward(example, forward, num_candidates)
}

/// Build raw, repaired, and inferred-bond candidate layers from one modular forward pass.
pub(crate) fn generate_layered_candidates_from_forward(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
) -> CandidateGenerationLayers {
    generate_layered_candidates_with_options(example, forward, num_candidates, true)
}

/// Build candidate layers from multiple de novo initialization samples for one pocket.
pub(crate) fn generate_layered_candidates_from_generation_samples(
    example: &MolecularExample,
    forwards: &[ResearchForward],
    num_candidates: usize,
) -> CandidateGenerationLayers {
    if forwards.is_empty() {
        return CandidateGenerationLayers::empty();
    }
    let requested = num_candidates.max(forwards.len()).max(1);
    let per_sample_base = requested / forwards.len();
    let remainder = requested % forwards.len();
    let mut layers = CandidateGenerationLayers::empty();
    for (sample_index, forward) in forwards.iter().enumerate() {
        let count = per_sample_base + usize::from(sample_index < remainder);
        if count > 0 {
            layers.extend(generate_layered_candidates_from_forward(
                example, forward, count,
            ));
        }
    }
    layers
}

/// Build candidate layers with optional geometry repair for ablation attribution.
pub(crate) fn generate_layered_candidates_with_options(
    example: &MolecularExample,
    forward: &ResearchForward,
    num_candidates: usize,
    enable_repair: bool,
) -> CandidateGenerationLayers {
    if num_candidates == 0 {
        return CandidateGenerationLayers::empty();
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
    let native_bonds = final_step
        .map(|step| step.native_bonds.clone())
        .unwrap_or_default();
    let native_bond_types = final_step
        .map(|step| step.native_bond_types.clone())
        .unwrap_or_default();
    let constrained_native_bond_types = final_step
        .map(|step| step.constrained_native_bond_types.clone())
        .unwrap_or_default();
    let native_graph_provenance = final_step
        .map(|step| step.native_graph_provenance.clone())
        .unwrap_or_default();
    let constrained_native_bonds = final_step
        .map(|step| step.constrained_native_bonds.clone())
        .unwrap_or_default();
    let base_bonds = if !constrained_native_bonds.is_empty() {
        constrained_native_bonds.clone()
    } else if native_bonds.is_empty() {
        infer_bonds(&rollout_coords)
    } else {
        native_bonds.clone()
    };
    let base_bond_types = if !constrained_native_bond_types.is_empty() {
        constrained_native_bond_types.clone()
    } else {
        native_bond_types.clone()
    };
    let raw_rollout_bonds = native_bonds.clone();
    let raw_rollout_bond_types = native_bond_types.clone();
    let generation_mode = forward.generation.generation_mode.as_str().to_string();
    let sample_source_fragment = format!(
        "sample={};sample_count={};sample_seed={}",
        forward.generation.rollout.sample_index,
        forward.generation.rollout.sample_count,
        forward
            .generation
            .rollout
            .sample_seed
            .map(|seed| seed.to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    let mut raw_geometry = Vec::with_capacity(num_candidates);
    let mut raw_rollout = Vec::with_capacity(num_candidates);
    let mut bond_logits_refined = Vec::with_capacity(num_candidates);
    let mut valence_refined = Vec::with_capacity(num_candidates);
    let mut repaired_candidates = Vec::with_capacity(num_candidates);
    let mut inferred_bond_candidates = Vec::with_capacity(num_candidates);

    for candidate_ix in 0..num_candidates {
        raw_geometry.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=raw_geometry;atoms={};bonds=0",
                rollout_atom_types.len()
            )),
            atom_types: rollout_atom_types.clone(),
            coords: rollout_coords.clone(),
            inferred_bonds: Vec::new(),
            bond_count: 0,
            valence_violation_count: 0,
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "raw_geometry:steps={};candidate={candidate_ix};{sample_source_fragment}",
                forward.generation.rollout.executed_steps
            ),
            generation_mode: generation_mode.clone(),
            generation_layer: CandidateLayerKind::RawGeometry
                .canonical_generation_layer()
                .to_string(),
            generation_path_class: CandidateLayerKind::RawGeometry
                .generation_path_class()
                .to_string(),
            model_native_raw: CandidateLayerKind::RawGeometry.is_model_native_raw(),
            postprocessor_chain: Vec::new(),
            claim_boundary: CandidateLayerKind::RawGeometry.claim_boundary().to_string(),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });

        raw_rollout.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=raw_rollout;native_decoder=valence_connected;atoms={};bonds={};native_bond_types={};raw_to_constrained_delta={};valence_downgrades={};guardrail_delta={}",
                rollout_atom_types.len(),
                raw_rollout_bonds.len(),
                raw_rollout_bond_types.len(),
                native_graph_provenance.raw_to_constrained_removed_bond_count,
                native_graph_provenance.valence_guardrail_downgrade_count,
                native_graph_provenance.guardrail_trigger_count
            )),
            atom_types: rollout_atom_types.clone(),
            coords: rollout_coords.clone(),
            inferred_bonds: raw_rollout_bonds.clone(),
            bond_count: raw_rollout_bonds.len(),
            valence_violation_count: valence_violation_count(
                &rollout_atom_types,
                &raw_rollout_bonds,
            ),
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "raw_modular_rollout:steps={};candidate={candidate_ix};{sample_source_fragment}",
                forward.generation.rollout.executed_steps
            ),
            generation_mode: generation_mode.clone(),
            generation_layer: CandidateLayerKind::RawRollout
                .canonical_generation_layer()
                .to_string(),
            generation_path_class: CandidateLayerKind::RawRollout
                .generation_path_class()
                .to_string(),
            model_native_raw: CandidateLayerKind::RawRollout.is_model_native_raw(),
            postprocessor_chain: Vec::new(),
            claim_boundary: CandidateLayerKind::RawRollout.claim_boundary().to_string(),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });

        let bond_refined_atom_types = choose_candidate_atom_types(
            &rollout_atom_types,
            logits,
            &topk,
            topk_count,
            &base_bonds,
            &base_bonds,
            candidate_ix,
        );
        bond_logits_refined.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=bond_logits_refined;atoms={};bonds={};native_bond_types={};raw_to_constrained_delta={};valence_downgrades={};guardrail_delta={}",
                bond_refined_atom_types.len(),
                base_bonds.len(),
                base_bond_types.len(),
                native_graph_provenance.raw_to_constrained_removed_bond_count,
                native_graph_provenance.valence_guardrail_downgrade_count,
                native_graph_provenance.guardrail_trigger_count
            )),
            atom_types: bond_refined_atom_types.clone(),
            coords: rollout_coords.clone(),
            inferred_bonds: base_bonds.clone(),
            bond_count: base_bonds.len(),
            valence_violation_count: valence_violation_count(&bond_refined_atom_types, &base_bonds),
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "bond_logits_refined:no_coordinate_move;steps={};candidate={candidate_ix};{sample_source_fragment}",
                forward.generation.rollout.executed_steps
            ),
            generation_mode: generation_mode.clone(),
            generation_layer: CandidateLayerKind::BondLogitsRefined
                .canonical_generation_layer()
                .to_string(),
            generation_path_class: CandidateLayerKind::BondLogitsRefined
                .generation_path_class()
                .to_string(),
            model_native_raw: CandidateLayerKind::BondLogitsRefined.is_model_native_raw(),
            postprocessor_chain: vec!["bond_logits_refinement_no_coordinate_move".to_string()],
            claim_boundary: CandidateLayerKind::BondLogitsRefined
                .claim_boundary()
                .to_string(),
            source_pocket_path: example
                .source_pocket_path
                .as_ref()
                .map(|path| path.display().to_string()),
            source_ligand_path: example
                .source_ligand_path
                .as_ref()
                .map(|path| path.display().to_string()),
        });

        let valence_bonds =
            prune_bonds_for_valence(&rollout_coords, &bond_refined_atom_types, &base_bonds);
        valence_refined.push(GeneratedCandidateRecord {
            example_id: example.example_id.clone(),
            protein_id: example.protein_id.clone(),
            molecular_representation: Some(format!(
                "source=valence_refined;atoms={};bonds={}",
                bond_refined_atom_types.len(),
                valence_bonds.len()
            )),
            atom_types: bond_refined_atom_types.clone(),
            coords: rollout_coords.clone(),
            inferred_bonds: valence_bonds.clone(),
            bond_count: valence_bonds.len(),
            valence_violation_count: valence_violation_count(
                &bond_refined_atom_types,
                &valence_bonds,
            ),
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "valence_refined:no_coordinate_move;steps={};candidate={candidate_ix};{sample_source_fragment}",
                forward.generation.rollout.executed_steps
            ),
            generation_mode: generation_mode.clone(),
            generation_layer: CandidateLayerKind::ValenceRefined
                .canonical_generation_layer()
                .to_string(),
            generation_path_class: CandidateLayerKind::ValenceRefined
                .generation_path_class()
                .to_string(),
            model_native_raw: CandidateLayerKind::ValenceRefined.is_model_native_raw(),
            postprocessor_chain: vec![
                "bond_logits_refinement_no_coordinate_move".to_string(),
                "valence_pruning_no_coordinate_move".to_string(),
            ],
            claim_boundary: CandidateLayerKind::ValenceRefined
                .claim_boundary()
                .to_string(),
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
            bond_count: 0,
            valence_violation_count: 0,
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "geometry_repair:steps={};candidate={candidate_ix};{sample_source_fragment}",
                forward.generation.rollout.executed_steps
            ),
            generation_mode: generation_mode.clone(),
            generation_layer: CandidateLayerKind::Repaired
                .canonical_generation_layer()
                .to_string(),
            generation_path_class: CandidateLayerKind::Repaired
                .generation_path_class()
                .to_string(),
            model_native_raw: CandidateLayerKind::Repaired.is_model_native_raw(),
            postprocessor_chain: vec!["pocket_centroid_repair".to_string()],
            claim_boundary: CandidateLayerKind::Repaired.claim_boundary().to_string(),
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
        let bond_count = inferred_bonds.len();
        let valence_violation_count = valence_violation_count(&atom_types, &inferred_bonds);
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
            bond_count,
            valence_violation_count,
            pocket_centroid: [
                pocket_centroid[0] as f32,
                pocket_centroid[1] as f32,
                pocket_centroid[2] as f32,
            ],
            pocket_radius: pocket_radius as f32,
            coordinate_frame_origin: example.coordinate_frame_origin,
            source: format!(
                "modular_rollout_decoder:steps={};candidate={candidate_ix};{sample_source_fragment}",
                forward.generation.rollout.executed_steps
            ),
            generation_mode: generation_mode.clone(),
            generation_layer: CandidateLayerKind::InferredBond
                .canonical_generation_layer()
                .to_string(),
            generation_path_class: CandidateLayerKind::InferredBond
                .generation_path_class()
                .to_string(),
            model_native_raw: CandidateLayerKind::InferredBond.is_model_native_raw(),
            postprocessor_chain: vec![
                "pocket_centroid_repair".to_string(),
                "distance_bond_inference".to_string(),
                "valence_pruning".to_string(),
            ],
            claim_boundary: CandidateLayerKind::InferredBond
                .claim_boundary()
                .to_string(),
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
        raw_geometry,
        raw_rollout,
        bond_logits_refined,
        valence_refined,
        repaired: repaired_candidates,
        inferred_bond: inferred_bond_candidates,
    }
}

fn valence_violation_count(atom_types: &[i64], bonds: &[(usize, usize)]) -> usize {
    let mut degrees = vec![0_usize; atom_types.len()];
    for &(left, right) in bonds {
        if left < degrees.len() && right < degrees.len() {
            degrees[left] += 1;
            degrees[right] += 1;
        }
    }
    degrees
        .iter()
        .zip(atom_types.iter())
        .filter(|(degree, atom_type)| **degree > max_valence(**atom_type))
        .count()
}
