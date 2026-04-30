//! Flow-matching variants and ablation-friendly velocity heads.

pub mod cross_attention_velocity_head;
pub mod equivariant_velocity_head;
pub mod molecular;
pub mod multimodal;
pub mod native_graph;

pub use cross_attention_velocity_head::{
    AtomPocketCrossAttentionVelocityConfig, AtomPocketCrossAttentionVelocityHead, FlowVelocityHead,
};
pub use equivariant_velocity_head::{
    EquivariantGeometryVelocityConfig, EquivariantGeometryVelocityHead,
};
pub use molecular::{FullMolecularFlowHead, MolecularFlowInput, MolecularFlowPrediction};
pub use multimodal::{
    current_molecular_flow_state_contract, current_multimodal_flow_contract,
    FlowBranchSupportStatus, MolecularFlowStateContract, MolecularFlowStateVariableContract,
    MultiModalFlowBranchRecord, MultiModalFlowContract, MOLECULAR_FLOW_CONTRACT_VERSION,
    REQUIRED_FULL_MOLECULAR_FLOW_BRANCHES,
};
pub use native_graph::{
    predicted_native_graph_from_flow, predicted_native_graph_from_flow_with_config,
    NativeGraphExtractionConfig, NativeGraphExtractionResult, NATIVE_GRAPH_EXTRACTOR_VERSION,
};
