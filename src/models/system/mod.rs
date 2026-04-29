//! Phase 2 system wiring for separate encoders, slots, controlled interaction, and probes.
//!
//! The implementation is split into type bundles, system methods, rollout
//! helpers, slicing helpers, and tests while preserving the original module API.

use tch::{nn, no_grad, Kind, Tensor};

mod flow_training;
mod slicing;

use self::flow_training::{
    atom_change_fraction, clip_coordinate_delta_norm, constrain_to_pocket_envelope,
    flow_matching_t_from_example, per_atom_displacement_mean, pocket_guidance_delta,
    resolved_generation_backend_family, tensor_to_coords, tensor_to_i64_vec,
};
use self::slicing::{
    merge_slot_contexts, slice_cross_modal_interactions, slice_decomposed_modalities,
    slice_encoded_modalities,
};
use super::{
    BatchedCrossAttentionOutput, BatchedModalityEncoding, BatchedSlotEncoding,
    ConditionedGenerationState, CrossAttentionOutput, CrossModalInteractionBlock,
    DeNovoScaffoldInitializer, DecoderOutput, FlowVelocityHead, FullMolecularFlowHead,
    GenerationGateSummary, GenerationPathUsageSummary, GenerationRolloutRecord,
    GenerationStepPathUsageSummary, GenerationStepRecord, GeometryEncoderImpl,
    GeometryFlowMatchingHead, GeometrySemanticBranch, ModalityEncoding, ModularLigandDecoder,
    PartialLigandState, PocketCentroidScaffoldInitializer, PocketEncoderImpl,
    PocketInitializationContext, PocketSemanticBranch, PocketVolumeAtomCountPrior, ProbeOutputs,
    SemanticProbeHeads, SlotEncoding, SoftSlotDecomposer, TopologyEncoderImpl,
    TopologySemanticBranch,
};
use crate::models::interaction::{
    attach_topology_pocket_pharmacophore_path_diagnostics, interaction_path_diagnostics,
    InteractionDiagnosticProvenance, InteractionExecutionContext, InteractionPath,
};
use crate::models::semantic::SemanticDiagnosticsBundle;
use crate::models::traits::{
    ConditionedLigandDecoder, FixedAtomCountPrior, NativeGraphLayerProvenance, ScaffoldInitializer,
};
use crate::{
    config::{
        FlowVelocityHeadKind, GenerationBackendFamilyConfig, GenerationModeConfig,
        GenerationRolloutMode, GenerationTargetConfig, InferenceContextRefreshPolicy,
        ResearchConfig,
    },
    data::{
        features::{
            chemistry_role_features_from_atom_types, GeometryFeatures, PocketFeatures,
            TopologyFeatures,
        },
        MolecularBatch, MolecularExample,
    },
    types::AtomType,
};

include!("types.rs");
include!("impl.rs");
include!("forward.rs");
include!("sync.rs");
include!("rollout.rs");
include!("tests.rs");
