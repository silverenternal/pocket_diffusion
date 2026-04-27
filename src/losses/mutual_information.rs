//! Mutual Information Monitoring for Semantic Decoupling
//!
//! Tracks inter-modality MI to verify that topology, geometry, and pocket
//! encodings maintain semantic separation during training.

use tch::{Kind, Tensor};

use crate::models::ResearchForward;

/// Tracks mutual information between modalities to verify decoupling.
#[derive(Debug, Clone)]
pub struct MutualInformationMonitor {
    /// Number of bins for entropy discretization.
    pub num_bins: usize,
}

impl Default for MutualInformationMonitor {
    fn default() -> Self {
        Self { num_bins: 32 }
    }
}

impl MutualInformationMonitor {
    /// Create new MI monitor with given bin count.
    pub fn new(num_bins: usize) -> Self {
        Self { num_bins }
    }

    /// Compute mutual information between two modality embeddings using entropy-based estimator.
    ///
    /// MI(X; Y) = H(X) + H(Y) - H(X, Y)
    ///
    /// Where H denotes entropy, estimated via binning.
    pub fn compute_mi(&self, x: &Tensor, y: &Tensor) -> f64 {
        if x.numel() == 0 || y.numel() == 0 {
            return 0.0;
        }

        // Flatten to vectors
        let x_flat = x.view([-1]);
        let y_flat = y.view([-1]);

        // Ensure same batch size
        if x_flat.size()[0] != y_flat.size()[0] {
            return 0.0;
        }

        let n = x_flat.size()[0] as usize;
        if n < 2 {
            return 0.0;
        }

        // Normalize to [0, 1]
        let x_min = x_flat.min();
        let x_max = x_flat.max();
        let x_range = &x_max - &x_min + 1e-8;
        let x_norm = (&x_flat - &x_min) / &x_range;

        let y_min = y_flat.min();
        let y_max = y_flat.max();
        let y_range = &y_max - &y_min + 1e-8;
        let y_norm = (&y_flat - &y_min) / &y_range;

        // Compute marginal entropies via binning
        let h_x = self.compute_entropy(&x_norm, n);
        let h_y = self.compute_entropy(&y_norm, n);

        // Compute joint entropy
        let h_xy = self.compute_joint_entropy(&x_norm, &y_norm, n);

        // MI = H(X) + H(Y) - H(X,Y)
        let mi = h_x + h_y - h_xy;

        // Clamp to [0, inf)
        mi.max(0.0)
    }

    /// Compute entropy of a normalized distribution via histogram.
    fn compute_entropy(&self, x: &Tensor, n: usize) -> f64 {
        if n < 2 {
            return 0.0;
        }

        // Bin the data
        let bin_indices = (x * (self.num_bins as f64)).to_kind(Kind::Int64);
        let bin_indices = bin_indices
            .clamp_min(0)
            .clamp_max((self.num_bins - 1) as i64);

        let mut bin_counts = vec![0usize; self.num_bins];
        for i in 0..n {
            let bin_idx = bin_indices.int64_value(&[i as i64]) as usize;
            if bin_idx < self.num_bins {
                bin_counts[bin_idx] += 1;
            }
        }

        // Compute entropy from bin counts
        let mut entropy = 0.0;
        for count in bin_counts {
            if count > 0 {
                let p = count as f64 / n as f64;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Compute joint entropy of two normalized distributions.
    fn compute_joint_entropy(&self, x: &Tensor, y: &Tensor, n: usize) -> f64 {
        if n < 2 {
            return 0.0;
        }

        // Bin both variables
        let x_bins = (x * (self.num_bins as f64))
            .to_kind(Kind::Int64)
            .clamp_min(0)
            .clamp_max((self.num_bins - 1) as i64);
        let y_bins = (y * (self.num_bins as f64))
            .to_kind(Kind::Int64)
            .clamp_min(0)
            .clamp_max((self.num_bins - 1) as i64);

        // Count joint occurrences
        let mut joint_counts = vec![vec![0usize; self.num_bins]; self.num_bins];
        for i in 0..n {
            let x_bin = x_bins.int64_value(&[i as i64]) as usize;
            let y_bin = y_bins.int64_value(&[i as i64]) as usize;
            if x_bin < self.num_bins && y_bin < self.num_bins {
                joint_counts[x_bin][y_bin] += 1;
            }
        }

        // Compute joint entropy
        let mut entropy = 0.0;
        for row in joint_counts {
            for count in row {
                if count > 0 {
                    let p = count as f64 / n as f64;
                    entropy -= p * p.log2();
                }
            }
        }

        entropy
    }

    /// Compute all three pairwise MIs: (Topo, Geo), (Topo, Pocket), (Geo, Pocket).
    pub(crate) fn compute_all_mi(&self, forward: &ResearchForward) -> (f64, f64, f64) {
        let topo_slots = &forward.slots.topology.slots;
        let geo_slots = &forward.slots.geometry.slots;
        let pocket_slots = &forward.slots.pocket.slots;

        let mi_topo_geo = self.compute_mi(topo_slots, geo_slots);
        let mi_topo_pocket = self.compute_mi(topo_slots, pocket_slots);
        let mi_geo_pocket = self.compute_mi(geo_slots, pocket_slots);

        (mi_topo_geo, mi_topo_pocket, mi_geo_pocket)
    }
}

/// Extensible decoupling quality report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecouplingQualityReport {
    /// Mutual information between topology and geometry.
    pub mi_topo_geo: f64,
    /// Mutual information between topology and pocket.
    pub mi_topo_pocket: f64,
    /// Mutual information between geometry and pocket.
    pub mi_geo_pocket: f64,
    /// Mean MI (indicator of overall decoupling).
    pub mi_mean: f64,
}

impl DecouplingQualityReport {
    /// Create from computed MI values.
    pub fn new(mi_topo_geo: f64, mi_topo_pocket: f64, mi_geo_pocket: f64) -> Self {
        let mi_mean = (mi_topo_geo + mi_topo_pocket + mi_geo_pocket) / 3.0;
        Self {
            mi_topo_geo,
            mi_topo_pocket,
            mi_geo_pocket,
            mi_mean,
        }
    }
}
