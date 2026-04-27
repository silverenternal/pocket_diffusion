//! Flow-matching variants and ablation-friendly velocity heads.

pub mod cross_attention_velocity_head;

pub use cross_attention_velocity_head::{
    AtomPocketCrossAttentionVelocityConfig, AtomPocketCrossAttentionVelocityHead,
};
