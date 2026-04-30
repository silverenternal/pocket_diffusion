//! Shared native graph extraction-score composition.

/// Bond-existence branch contribution to the native extraction score.
pub(crate) const NATIVE_SCORE_BOND_WEIGHT: f64 = 0.60;
/// Topology branch contribution to the native extraction score.
pub(crate) const NATIVE_SCORE_TOPOLOGY_WEIGHT: f64 = 0.40;

/// Weights used to combine branch probabilities into the native extraction score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NativeScoreWeights {
    /// Bond-existence branch contribution.
    pub bond: f64,
    /// Topology branch contribution.
    pub topology: f64,
}

impl Default for NativeScoreWeights {
    fn default() -> Self {
        Self {
            bond: NATIVE_SCORE_BOND_WEIGHT,
            topology: NATIVE_SCORE_TOPOLOGY_WEIGHT,
        }
    }
}

/// Default score weights used by native graph extraction and its calibration loss.
pub(crate) const DEFAULT_NATIVE_SCORE_WEIGHTS: NativeScoreWeights = NativeScoreWeights {
    bond: NATIVE_SCORE_BOND_WEIGHT,
    topology: NATIVE_SCORE_TOPOLOGY_WEIGHT,
};

/// Combine branch probabilities into the native extraction score.
pub(crate) fn combined_native_score(bond_probability: f64, topology_probability: f64) -> f64 {
    DEFAULT_NATIVE_SCORE_WEIGHTS.bond * bond_probability
        + DEFAULT_NATIVE_SCORE_WEIGHTS.topology * topology_probability
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_native_score_weights_are_normalized() {
        let total = DEFAULT_NATIVE_SCORE_WEIGHTS.bond + DEFAULT_NATIVE_SCORE_WEIGHTS.topology;
        assert!((total - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn combined_native_score_uses_shared_weights() {
        let score = combined_native_score(0.8, 0.2);
        assert!((score - 0.56).abs() < 1.0e-12);
    }
}
