//! Semantic probe supervision for specialized modality paths.

use tch::{Kind, Reduction, Tensor};

use crate::{config::PharmacophoreProbeConfig, data::MolecularExample, models::ResearchForward};

use super::alignment::{align_rows, align_square_matrix, align_vector, LossTargetAlignmentPolicy};

/// Modality that owns a semantic probe source representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeSourceModality {
    /// Ligand topology branch.
    Topology,
    /// Ligand geometry branch.
    Geometry,
    /// Pocket/context branch.
    Pocket,
    /// Joint pooled multi-modal state.
    Joint,
}

/// How missing supervision is handled for a probe target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeSupervisionPolicy {
    /// Target is constructed from required encoder inputs.
    RequiredInput,
    /// Target rows are masked by an explicit availability tensor.
    AvailabilityMasked,
    /// Target is optional; missing values contribute an explicit zero tensor.
    OptionalScalar,
}

/// Same-modality semantic probe targets in stable report order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameModalityProbeTarget {
    /// Topology branch predicts ligand adjacency/bond proximity structure.
    TopologyAdjacency,
    /// Geometry branch predicts per-atom mean pairwise ligand distance.
    GeometryMeanPairwiseDistance,
    /// Pocket branch reconstructs pocket atom/residue feature rows.
    PocketAtomFeatures,
    /// Topology branch predicts ligand pharmacophore role rows when available.
    LigandPharmacophoreRoles,
    /// Pocket branch predicts pocket pharmacophore role rows when available.
    PocketPharmacophoreRoles,
    /// Joint pooled state predicts optional affinity scalar.
    AffinityScalar,
}

/// Ordered same-modality probe target contract.
pub const SAME_MODALITY_PROBE_TARGETS: [SameModalityProbeTarget; 6] = [
    SameModalityProbeTarget::TopologyAdjacency,
    SameModalityProbeTarget::GeometryMeanPairwiseDistance,
    SameModalityProbeTarget::PocketAtomFeatures,
    SameModalityProbeTarget::LigandPharmacophoreRoles,
    SameModalityProbeTarget::PocketPharmacophoreRoles,
    SameModalityProbeTarget::AffinityScalar,
];

impl SameModalityProbeTarget {
    /// All same-modality targets in stable report order.
    pub const fn all() -> &'static [SameModalityProbeTarget; 6] {
        &SAME_MODALITY_PROBE_TARGETS
    }

    /// Source modality whose representation is supervised by this target.
    pub const fn source_modality(&self) -> ProbeSourceModality {
        match self {
            Self::TopologyAdjacency | Self::LigandPharmacophoreRoles => {
                ProbeSourceModality::Topology
            }
            Self::GeometryMeanPairwiseDistance => ProbeSourceModality::Geometry,
            Self::PocketAtomFeatures | Self::PocketPharmacophoreRoles => {
                ProbeSourceModality::Pocket
            }
            Self::AffinityScalar => ProbeSourceModality::Joint,
        }
    }

    /// Missing-supervision policy for this target.
    pub const fn supervision_policy(&self) -> ProbeSupervisionPolicy {
        match self {
            Self::TopologyAdjacency
            | Self::GeometryMeanPairwiseDistance
            | Self::PocketAtomFeatures => ProbeSupervisionPolicy::RequiredInput,
            Self::LigandPharmacophoreRoles | Self::PocketPharmacophoreRoles => {
                ProbeSupervisionPolicy::AvailabilityMasked
            }
            Self::AffinityScalar => ProbeSupervisionPolicy::OptionalScalar,
        }
    }

    /// Stable metric/report key.
    pub const fn metric_key(&self) -> &'static str {
        match self {
            Self::TopologyAdjacency => "topology_adjacency",
            Self::GeometryMeanPairwiseDistance => "geometry_mean_pairwise_distance",
            Self::PocketAtomFeatures => "pocket_atom_features",
            Self::LigandPharmacophoreRoles => "ligand_pharmacophore_roles",
            Self::PocketPharmacophoreRoles => "pocket_pharmacophore_roles",
            Self::AffinityScalar => "affinity_scalar",
        }
    }
}

/// Explicit off-modality leakage probe directions used by future explicit wiring.
///
/// The current training path does not consume these targets. They are defined as
/// an explicit contract for upcoming task loss modules and config surface work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffModalityLeakageProbeTarget {
    /// Topology branch predicts geometric targets.
    TopologyToGeometry,
    /// Topology branch predicts pocket targets.
    TopologyToPocket,
    /// Geometry branch predicts topological targets.
    GeometryToTopology,
    /// Geometry branch predicts pocket targets.
    GeometryToPocket,
    /// Pocket branch predicts topological targets.
    PocketToTopology,
    /// Pocket branch predicts geometric targets.
    PocketToGeometry,
}

/// Ordered list of explicit off-modality probe directions.
pub const OFF_MODALITY_LEAKAGE_PROBE_TARGETS: [OffModalityLeakageProbeTarget; 6] = [
    OffModalityLeakageProbeTarget::TopologyToGeometry,
    OffModalityLeakageProbeTarget::TopologyToPocket,
    OffModalityLeakageProbeTarget::GeometryToTopology,
    OffModalityLeakageProbeTarget::GeometryToPocket,
    OffModalityLeakageProbeTarget::PocketToTopology,
    OffModalityLeakageProbeTarget::PocketToGeometry,
];

impl OffModalityLeakageProbeTarget {
    /// Stable configuration flag name for this target.
    pub const fn config_flag(&self) -> &'static str {
        match self {
            Self::TopologyToGeometry => "topology_to_geometry_probe",
            Self::TopologyToPocket => "topology_to_pocket_probe",
            Self::GeometryToTopology => "geometry_to_topology_probe",
            Self::GeometryToPocket => "geometry_to_pocket_probe",
            Self::PocketToTopology => "pocket_to_topology_probe",
            Self::PocketToGeometry => "pocket_to_geometry_probe",
        }
    }

    /// Human-readable description for diagnostics and docs.
    pub const fn description(&self) -> &'static str {
        match self {
            Self::TopologyToGeometry => "topology -> geometry",
            Self::TopologyToPocket => "topology -> pocket",
            Self::GeometryToTopology => "geometry -> topology",
            Self::GeometryToPocket => "geometry -> pocket",
            Self::PocketToTopology => "pocket -> topology",
            Self::PocketToGeometry => "pocket -> geometry",
        }
    }

    /// All explicit probe targets in stable order.
    pub const fn all() -> &'static [OffModalityLeakageProbeTarget; 6] {
        &OFF_MODALITY_LEAKAGE_PROBE_TARGETS
    }
}

/// Decomposed semantic probe losses used for scalar metrics.
pub(crate) struct ProbeLossTensors {
    /// Core semantic probe objective before optional pharmacophore subterms.
    pub core: Tensor,
    /// Total semantic probe objective.
    pub total: Tensor,
    /// Ligand pharmacophore role supervision subterm.
    pub ligand_pharmacophore: Tensor,
    /// Pocket pharmacophore role supervision subterm.
    pub pocket_pharmacophore: Tensor,
}

/// Lightweight supervision over topology, geometry, and pocket probe heads.
#[derive(Debug, Clone)]
pub struct ProbeLoss {
    pharmacophore: PharmacophoreProbeConfig,
}

impl Default for ProbeLoss {
    fn default() -> Self {
        Self {
            pharmacophore: PharmacophoreProbeConfig::default(),
        }
    }
}

impl ProbeLoss {
    /// Create probe loss wiring from training config.
    pub fn new(pharmacophore: PharmacophoreProbeConfig) -> Self {
        Self { pharmacophore }
    }

    /// Compute the semantic probe objective for one example.
    #[allow(dead_code)]
    pub(crate) fn compute(&self, example: &MolecularExample, forward: &ResearchForward) -> Tensor {
        self.compute_weighted(example, forward, 1.0)
    }

    /// Compute the semantic probe objective with an optional affinity weight.
    pub(crate) fn compute_weighted(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
        affinity_weight: f64,
    ) -> Tensor {
        self.compute_weighted_components(example, forward, affinity_weight)
            .total
    }

    /// Compute the semantic probe objective with pharmacophore subterm decomposition.
    pub(crate) fn compute_weighted_components(
        &self,
        example: &MolecularExample,
        forward: &ResearchForward,
        affinity_weight: f64,
    ) -> ProbeLossTensors {
        let topo_loss = topology_adjacency_probe_loss(
            &forward.probes.topology_adjacency_logits,
            &example.topology.adjacency,
        );
        let geo_loss = geometry_distance_probe_loss(
            &forward.probes.geometry_distance_predictions,
            &example.geometry.pairwise_distances,
        );
        let pocket_loss = pocket_feature_probe_loss(
            &forward.probes.pocket_feature_predictions,
            &example.pocket.atom_features,
        );

        let affinity_loss = if forward.probes.affinity_prediction.numel() == 0 {
            Tensor::zeros(
                [1],
                (Kind::Float, forward.probes.affinity_prediction.device()),
            )
        } else if let Some(target_affinity) = example.targets.affinity_kcal_mol {
            (forward.probes.affinity_prediction.shallow_clone()
                - Tensor::from(target_affinity as f64)
                    .to_kind(Kind::Float)
                    .to_device(forward.probes.affinity_prediction.device()))
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
                * affinity_weight
        } else {
            Tensor::zeros(
                [1],
                (Kind::Float, forward.probes.affinity_prediction.device()),
            )
        };

        let ligand_pharmacophore = if self.pharmacophore.enable_ligand_role_probe {
            aligned_role_bce_with_logits(
                &forward.probes.ligand_pharmacophore_role_logits,
                &example.topology.chemistry_roles.role_vectors,
                &example.topology.chemistry_roles.availability,
                "probe.ligand_pharmacophore_roles",
            )
        } else {
            Tensor::zeros(
                [1],
                (Kind::Float, forward.probes.affinity_prediction.device()),
            )
        };
        let pocket_pharmacophore = if self.pharmacophore.enable_pocket_role_probe {
            aligned_role_bce_with_logits(
                &forward.probes.pocket_pharmacophore_role_logits,
                &example.pocket.chemistry_roles.role_vectors,
                &example.pocket.chemistry_roles.availability,
                "probe.pocket_pharmacophore_roles",
            )
        } else {
            Tensor::zeros(
                [1],
                (Kind::Float, forward.probes.affinity_prediction.device()),
            )
        };

        let core = topo_loss + geo_loss + pocket_loss + affinity_loss;
        let total = core.shallow_clone()
            + ligand_pharmacophore.shallow_clone()
            + pocket_pharmacophore.shallow_clone();

        ProbeLossTensors {
            core,
            total,
            ligand_pharmacophore,
            pocket_pharmacophore,
        }
    }

    /// Compute mean semantic probe objective and role-probe subterms over a mini-batch.
    pub(crate) fn compute_batch_weighted_components(
        &self,
        examples: &[MolecularExample],
        forwards: &[ResearchForward],
        affinity_weight_for: impl Fn(&MolecularExample) -> f64,
    ) -> ProbeLossTensors {
        debug_assert_eq!(examples.len(), forwards.len());
        let device = forwards
            .first()
            .map(|forward| forward.probes.affinity_prediction.device())
            .or_else(|| {
                examples
                    .first()
                    .map(|example| example.topology.atom_types.device())
            })
            .unwrap_or(tch::Device::Cpu);
        if examples.is_empty() {
            let zero = Tensor::zeros([1], (Kind::Float, device));
            return ProbeLossTensors {
                core: zero.shallow_clone(),
                total: zero.shallow_clone(),
                ligand_pharmacophore: zero.shallow_clone(),
                pocket_pharmacophore: zero,
            };
        }

        let mut total = Tensor::zeros([1], (Kind::Float, device));
        let mut core = Tensor::zeros([1], (Kind::Float, device));
        let mut ligand_pharmacophore = Tensor::zeros([1], (Kind::Float, device));
        let mut pocket_pharmacophore = Tensor::zeros([1], (Kind::Float, device));
        for (example, forward) in examples.iter().zip(forwards.iter()) {
            let components =
                self.compute_weighted_components(example, forward, affinity_weight_for(example));
            core += components.core.to_device(device);
            total += components.total.to_device(device);
            ligand_pharmacophore += components.ligand_pharmacophore.to_device(device);
            pocket_pharmacophore += components.pocket_pharmacophore.to_device(device);
        }
        let denom = examples.len() as f64;
        ProbeLossTensors {
            core: core / denom,
            total: total / denom,
            ligand_pharmacophore: ligand_pharmacophore / denom,
            pocket_pharmacophore: pocket_pharmacophore / denom,
        }
    }
}

fn topology_adjacency_probe_loss(logits: &Tensor, target_adjacency: &Tensor) -> Tensor {
    let device = logits.device();
    if logits.numel() == 0 || logits.dim() != 2 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let rows = logits.size().first().copied().unwrap_or(0).max(0);
    if logits.size().get(1).copied().unwrap_or(-1) != rows {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let Some(aligned) = align_square_matrix(
        target_adjacency,
        rows,
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "probe.topology_adjacency",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    masked_bce_with_logits(logits, &aligned.values, &aligned.mask)
}

fn geometry_distance_probe_loss(
    predictions: &Tensor,
    target_pairwise_distances: &Tensor,
) -> Tensor {
    let device = predictions.device();
    if predictions.numel() == 0 || predictions.dim() != 1 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    if target_pairwise_distances.numel() == 0 || target_pairwise_distances.dim() != 2 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let rows = predictions.size().first().copied().unwrap_or(0).max(0);
    let target_distances = target_pairwise_distances.mean_dim([1].as_slice(), false, Kind::Float);
    let Some(aligned) = align_vector(
        &target_distances,
        rows,
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "probe.geometry_mean_pairwise_distance",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let per_row = (predictions - aligned.values).pow_tensor_scalar(2.0);
    weighted_mean(&per_row, &aligned.mask)
}

fn pocket_feature_probe_loss(predictions: &Tensor, target_features: &Tensor) -> Tensor {
    let device = predictions.device();
    if predictions.numel() == 0 || predictions.dim() != 2 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let rows = predictions.size().first().copied().unwrap_or(0).max(0);
    let feature_dim = predictions.size().get(1).copied().unwrap_or(0).max(0);
    let Some(aligned) = align_rows(
        target_features,
        rows,
        &[feature_dim],
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        "probe.pocket_atom_features",
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let per_row = (predictions - aligned.values)
        .pow_tensor_scalar(2.0)
        .mean_dim([1].as_slice(), false, Kind::Float);
    weighted_mean(&per_row, &aligned.mask)
}

pub(crate) fn masked_role_bce_with_logits(
    logits: &Tensor,
    target_roles: &Tensor,
    availability: &Tensor,
) -> Tensor {
    aligned_role_bce_with_logits(logits, target_roles, availability, "role_bce")
}

fn aligned_role_bce_with_logits(
    logits: &Tensor,
    target_roles: &Tensor,
    availability: &Tensor,
    label: &str,
) -> Tensor {
    let device = logits.device();
    if logits.numel() == 0
        || target_roles.numel() == 0
        || availability.numel() == 0
        || logits.dim() != 2
    {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let rows = logits.size().first().copied().unwrap_or(0).max(0);
    let cols = logits.size().get(1).copied().unwrap_or(0).max(0);
    if rows <= 0 || cols <= 0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }
    let Some(aligned_roles) = align_rows(
        target_roles,
        rows,
        &[cols],
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        label,
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let Some(aligned_availability) = align_vector(
        availability,
        rows,
        Kind::Float,
        device,
        LossTargetAlignmentPolicy::PadWithMask,
        format!("{label}.availability"),
    ) else {
        return Tensor::zeros([1], (Kind::Float, device));
    };
    let mask = (aligned_roles.mask * aligned_availability.values * aligned_availability.mask)
        .to_kind(Kind::Float)
        .unsqueeze(-1);
    let available = mask.sum(Kind::Float).double_value(&[]);
    if available <= 0.0 {
        return Tensor::zeros([1], (Kind::Float, device));
    }

    let loss = logits.binary_cross_entropy_with_logits::<Tensor>(
        &aligned_roles.values,
        None,
        None,
        Reduction::None,
    );
    (loss * &mask).sum(Kind::Float) / (available * cols as f64)
}

fn masked_bce_with_logits(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    if logits.numel() == 0
        || target.numel() == 0
        || mask.numel() == 0
        || logits.size() != target.size()
        || logits.size() != mask.size()
    {
        return Tensor::zeros([1], (Kind::Float, logits.device()));
    }
    let target = target.to_kind(Kind::Float);
    let per_item = logits.clamp_min(0.0) - logits * &target + (-logits.abs()).exp().log1p();
    weighted_mean(&per_item, mask)
}

fn weighted_mean(values: &Tensor, mask: &Tensor) -> Tensor {
    let mask = mask.to_kind(Kind::Float);
    let denom = mask.sum(Kind::Float).clamp_min(1.0);
    (values * mask).sum(Kind::Float) / denom
}

#[cfg(test)]
mod tests {
    use super::{
        masked_role_bce_with_logits, OffModalityLeakageProbeTarget, ProbeLoss, ProbeSourceModality,
        ProbeSupervisionPolicy, SameModalityProbeTarget, OFF_MODALITY_LEAKAGE_PROBE_TARGETS,
        SAME_MODALITY_PROBE_TARGETS,
    };
    use tch::{nn, Device, Kind, Tensor};

    use crate::{
        config::{PharmacophoreProbeConfig, ResearchConfig},
        data::{synthetic_phase1_examples, InMemoryDataset},
        models::Phase1ResearchSystem,
    };

    #[test]
    fn off_modality_probe_targets_are_stable_and_exhaustive() {
        assert_eq!(OFF_MODALITY_LEAKAGE_PROBE_TARGETS.len(), 6);
        let expected = [
            OffModalityLeakageProbeTarget::TopologyToGeometry,
            OffModalityLeakageProbeTarget::TopologyToPocket,
            OffModalityLeakageProbeTarget::GeometryToTopology,
            OffModalityLeakageProbeTarget::GeometryToPocket,
            OffModalityLeakageProbeTarget::PocketToTopology,
            OffModalityLeakageProbeTarget::PocketToGeometry,
        ];
        assert_eq!(OFF_MODALITY_LEAKAGE_PROBE_TARGETS, expected);
        assert_eq!(
            OffModalityLeakageProbeTarget::TopologyToGeometry.config_flag(),
            "topology_to_geometry_probe"
        );
        assert_eq!(
            OffModalityLeakageProbeTarget::PocketToGeometry.description(),
            "pocket -> geometry"
        );
    }

    #[test]
    fn same_modality_probe_targets_define_stable_supervision_contracts() {
        assert_eq!(SAME_MODALITY_PROBE_TARGETS.len(), 6);
        assert_eq!(SameModalityProbeTarget::all(), &SAME_MODALITY_PROBE_TARGETS);
        assert_eq!(
            SameModalityProbeTarget::TopologyAdjacency.source_modality(),
            ProbeSourceModality::Topology
        );
        assert_eq!(
            SameModalityProbeTarget::GeometryMeanPairwiseDistance.source_modality(),
            ProbeSourceModality::Geometry
        );
        assert_eq!(
            SameModalityProbeTarget::PocketAtomFeatures.source_modality(),
            ProbeSourceModality::Pocket
        );
        assert_eq!(
            SameModalityProbeTarget::LigandPharmacophoreRoles.supervision_policy(),
            ProbeSupervisionPolicy::AvailabilityMasked
        );
        assert_eq!(
            SameModalityProbeTarget::PocketPharmacophoreRoles.supervision_policy(),
            ProbeSupervisionPolicy::AvailabilityMasked
        );
        assert_eq!(
            SameModalityProbeTarget::AffinityScalar.supervision_policy(),
            ProbeSupervisionPolicy::OptionalScalar
        );
        assert_eq!(
            SameModalityProbeTarget::TopologyAdjacency.metric_key(),
            "topology_adjacency"
        );
    }

    #[test]
    fn pharmacophore_role_probe_terms_are_config_gated() {
        let mut config = ResearchConfig::default();
        config.data.batch_size = 1;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let example = &dataset.examples()[0];
        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let forward = system.forward_example(example);

        let disabled = ProbeLoss::default().compute_weighted_components(example, &forward, 1.0);
        assert_eq!(disabled.ligand_pharmacophore.double_value(&[]), 0.0);
        assert_eq!(disabled.pocket_pharmacophore.double_value(&[]), 0.0);

        let enabled = ProbeLoss::new(PharmacophoreProbeConfig {
            enable_ligand_role_probe: true,
            enable_pocket_role_probe: true,
            ..PharmacophoreProbeConfig::default()
        })
        .compute_weighted_components(example, &forward, 1.0);
        assert!(enabled.ligand_pharmacophore.double_value(&[]).is_finite());
        assert!(enabled.pocket_pharmacophore.double_value(&[]).is_finite());
        assert!(enabled.total.double_value(&[]).is_finite());
    }

    #[test]
    fn probe_loss_is_shape_safe_for_de_novo_atom_count_mismatch() {
        let mut config = ResearchConfig::default();
        config.data.generation_target.generation_mode =
            crate::config::GenerationModeConfig::DeNovoInitialization;
        let dataset = InMemoryDataset::new(synthetic_phase1_examples())
            .with_pocket_feature_dim(config.model.pocket_feature_dim);
        let example = &dataset.examples()[0];
        let target_atom_count = example.topology.atom_types.size()[0] as usize;
        let generated_atom_count = target_atom_count + 3;
        config
            .data
            .generation_target
            .de_novo_initialization
            .min_atom_count = generated_atom_count;
        config
            .data
            .generation_target
            .de_novo_initialization
            .max_atom_count = generated_atom_count;

        let var_store = nn::VarStore::new(Device::Cpu);
        let system = Phase1ResearchSystem::new(&var_store.root(), &config);
        let (_, forwards) = system.forward_batch(&dataset.examples()[..1]);
        let forward = &forwards[0];
        assert_ne!(
            forward.probes.topology_adjacency_logits.size(),
            example.topology.adjacency.size()
        );

        let loss = ProbeLoss::new(PharmacophoreProbeConfig {
            enable_ligand_role_probe: true,
            enable_pocket_role_probe: true,
            ..PharmacophoreProbeConfig::default()
        })
        .compute_weighted_components(example, forward, 1.0);

        assert!(is_finite_scalar(&loss.core));
        assert!(is_finite_scalar(&loss.ligand_pharmacophore));
        assert!(is_finite_scalar(&loss.pocket_pharmacophore));
        assert!(is_finite_scalar(&loss.total));
        loss.total.backward();
    }

    #[test]
    fn masked_role_bce_ignores_unavailable_rows() {
        let logits = Tensor::zeros([2, 3], (Kind::Float, Device::Cpu));
        let targets = Tensor::ones([2, 3], (Kind::Float, Device::Cpu));
        let availability = Tensor::from_slice(&[0.0f32, 0.0]).to_kind(Kind::Float);
        let loss = masked_role_bce_with_logits(&logits, &targets, &availability);
        assert_eq!(loss.double_value(&[]), 0.0);
    }

    fn is_finite_scalar(tensor: &Tensor) -> bool {
        tensor
            .isfinite()
            .all()
            .to_kind(Kind::Int64)
            .int64_value(&[])
            != 0
    }
}
