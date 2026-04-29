//! Bounded ligand pairwise geometry message passing.

/// Configuration for radius-then-top-k local geometry messages.
#[derive(Debug, Clone)]
pub struct PairwiseGeometryConfig {
    /// Maximum distance considered for neighbor candidates.
    pub radius: f32,
    /// Maximum retained neighbors per atom after radius filtering.
    pub max_neighbors: usize,
    /// Conservative residual scale applied when messages are injected into a velocity head.
    pub residual_scale: f64,
}

impl Default for PairwiseGeometryConfig {
    fn default() -> Self {
        Self {
            radius: 4.5,
            max_neighbors: 16,
            residual_scale: 0.1,
        }
    }
}

/// A local geometry message from one ligand atom to another.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseGeometryMessage {
    /// Source atom index.
    pub source: usize,
    /// Target atom index.
    pub target: usize,
    /// Euclidean atom distance.
    pub distance: f32,
    /// Unit direction from source to target.
    pub direction: [f32; 3],
    /// Clash margin relative to a conservative 1.2 Angstrom threshold.
    pub clash_margin: f32,
}

/// Stateless bounded O(Nk) message builder.
#[derive(Debug, Clone)]
pub struct PairwiseGeometryMessagePassing {
    config: PairwiseGeometryConfig,
}

impl PairwiseGeometryMessagePassing {
    /// Create a message builder.
    pub fn new(config: PairwiseGeometryConfig) -> Self {
        Self { config }
    }

    /// Build radius-filtered, top-k nearest messages for each atom.
    pub fn build_messages(&self, coords: &[[f32; 3]]) -> Vec<PairwiseGeometryMessage> {
        let mut messages = Vec::new();
        for source in 0..coords.len() {
            let mut neighbors = Vec::new();
            for target in 0..coords.len() {
                if source == target {
                    continue;
                }
                let delta = [
                    coords[target][0] - coords[source][0],
                    coords[target][1] - coords[source][1],
                    coords[target][2] - coords[source][2],
                ];
                let distance =
                    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                if distance <= self.config.radius {
                    neighbors.push((target, distance, delta));
                }
            }
            neighbors.sort_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for (target, distance, delta) in neighbors.into_iter().take(self.config.max_neighbors) {
                let inv = if distance > 0.0 { 1.0 / distance } else { 0.0 };
                messages.push(PairwiseGeometryMessage {
                    source,
                    target,
                    distance,
                    direction: [delta[0] * inv, delta[1] * inv, delta[2] * inv],
                    clash_margin: 1.2 - distance,
                });
            }
        }
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_neighbor_count_for_large_ligand() {
        let coords: Vec<[f32; 3]> = (0..64).map(|idx| [idx as f32 * 0.2, 0.0, 0.0]).collect();
        let mp = PairwiseGeometryMessagePassing::new(PairwiseGeometryConfig {
            radius: 100.0,
            max_neighbors: 3,
            residual_scale: 0.1,
        });
        let messages = mp.build_messages(&coords);
        assert!(messages.len() <= coords.len() * 3);
    }

    #[test]
    fn translation_invariance_smoke() {
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let shifted: Vec<[f32; 3]> = coords
            .iter()
            .map(|coord| [coord[0] + 5.0, coord[1] - 3.0, coord[2] + 1.0])
            .collect();
        let mp = PairwiseGeometryMessagePassing::new(PairwiseGeometryConfig::default());
        let base = mp.build_messages(&coords);
        let next = mp.build_messages(&shifted);
        assert_eq!(base.len(), next.len());
        for (left, right) in base.iter().zip(next.iter()) {
            assert!((left.distance - right.distance).abs() < 1e-6);
            assert_eq!(left.source, right.source);
            assert_eq!(left.target, right.target);
        }
    }

    #[test]
    fn empty_and_single_atom_paths_are_empty() {
        let mp = PairwiseGeometryMessagePassing::new(PairwiseGeometryConfig::default());
        assert!(mp.build_messages(&[]).is_empty());
        assert!(mp.build_messages(&[[0.0, 0.0, 0.0]]).is_empty());
    }
}
