//! L3 bound certificates: partition-to-bound bridges.
//!
//! This module provides certificates based on the L3 (Level 3) decomposition
//! bound theory. Given a partition P = {B_1, ..., B_k} and dual solution y*,
//! the gap between the LP relaxation and the decomposition bound is bounded by:
//!
//!   z_LP - z_D(P) ≤ Σ_{e ∈ E_cross(P)} |y*_e| * (n_e(P) - 1)
//!
//! Specializations for Benders decomposition and Dantzig-Wolfe decomposition
//! provide tighter bounds by exploiting problem structure.

pub mod benders_cert;
pub mod dw_cert;
pub mod partition_bound;

pub use benders_cert::BendersCertificate;
pub use dw_cert::DWCertificate;
pub use partition_bound::L3PartitionCertificate;

use serde::{Deserialize, Serialize};

/// Summary of an L3 bound computation across multiple certificates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L3BoundSummary {
    pub partition_bound: Option<f64>,
    pub benders_bound: Option<f64>,
    pub dw_bound: Option<f64>,
    pub tightest_bound: f64,
    pub tightest_source: String,
    pub num_crossing_edges: usize,
    pub num_coupling_vars: usize,
    pub num_linking_constraints: usize,
}

impl L3BoundSummary {
    /// Creates a summary from optional certificates.
    pub fn from_certificates(
        partition: Option<&L3PartitionCertificate>,
        benders: Option<&BendersCertificate>,
        dw: Option<&DWCertificate>,
    ) -> Self {
        let partition_bound = partition.map(|c| c.total_bound);
        let benders_bound = benders.map(|c| c.current_bound());
        let dw_bound = dw.map(|c| c.current_bound());

        let mut tightest = f64::MAX;
        let mut source = "none".to_string();

        if let Some(pb) = partition_bound {
            if pb < tightest {
                tightest = pb;
                source = "partition".to_string();
            }
        }
        if let Some(bb) = benders_bound {
            if bb < tightest {
                tightest = bb;
                source = "benders".to_string();
            }
        }
        if let Some(db) = dw_bound {
            if db < tightest {
                tightest = db;
                source = "dw".to_string();
            }
        }

        if tightest == f64::MAX {
            tightest = f64::INFINITY;
        }

        Self {
            partition_bound,
            benders_bound,
            dw_bound,
            tightest_bound: tightest,
            tightest_source: source,
            num_crossing_edges: partition.map_or(0, |c| c.crossing_edges.len()),
            num_coupling_vars: benders.map_or(0, |c| c.coupling_variables.len()),
            num_linking_constraints: dw.map_or(0, |c| c.linking_constraints.len()),
        }
    }

    /// Whether any bound was computed.
    pub fn has_bound(&self) -> bool {
        self.tightest_bound.is_finite()
    }

    /// Ratio of tightest to loosest bound (measures how much structure helps).
    pub fn tightening_ratio(&self) -> Option<f64> {
        let bounds: Vec<f64> = [self.partition_bound, self.benders_bound, self.dw_bound]
            .iter()
            .filter_map(|b| *b)
            .collect();
        if bounds.len() < 2 {
            return None;
        }
        let max = bounds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = bounds.iter().cloned().fold(f64::INFINITY, f64::min);
        if max.abs() < 1e-15 {
            return None;
        }
        Some(min / max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_no_certificates() {
        let summary = L3BoundSummary::from_certificates(None, None, None);
        assert!(!summary.has_bound());
        assert_eq!(summary.tightest_source, "none");
    }

    #[test]
    fn test_summary_tightening_ratio_insufficient() {
        let summary = L3BoundSummary {
            partition_bound: Some(10.0),
            benders_bound: None,
            dw_bound: None,
            tightest_bound: 10.0,
            tightest_source: "partition".to_string(),
            num_crossing_edges: 5,
            num_coupling_vars: 0,
            num_linking_constraints: 0,
        };
        assert!(summary.tightening_ratio().is_none());
    }

    #[test]
    fn test_summary_tightening_ratio() {
        let summary = L3BoundSummary {
            partition_bound: Some(10.0),
            benders_bound: Some(5.0),
            dw_bound: None,
            tightest_bound: 5.0,
            tightest_source: "benders".to_string(),
            num_crossing_edges: 5,
            num_coupling_vars: 3,
            num_linking_constraints: 0,
        };
        let ratio = summary.tightening_ratio().unwrap();
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_summary_has_bound() {
        let summary = L3BoundSummary {
            partition_bound: Some(7.5),
            benders_bound: None,
            dw_bound: None,
            tightest_bound: 7.5,
            tightest_source: "partition".to_string(),
            num_crossing_edges: 2,
            num_coupling_vars: 0,
            num_linking_constraints: 0,
        };
        assert!(summary.has_bound());
    }
}
