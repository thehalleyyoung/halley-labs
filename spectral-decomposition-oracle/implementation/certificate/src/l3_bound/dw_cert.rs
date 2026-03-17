//! L3-C Dantzig-Wolfe specialization certificate.
//!
//! For DW decomposition at iteration t:
//!   z_LP - z_DW^(t) ≤ Σ_{i linking} |μ_i^(t)| * (|blocks(i)| - 1)
//!
//! where μ_i^(t) is the master dual for linking constraint i at iteration t
//! and |blocks(i)| is the number of blocks that constraint i involves.

use crate::error::{CertificateError, CertificateResult};
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A linking constraint in a DW decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingConstraint {
    pub constraint_index: usize,
    pub constraint_name: String,
    pub block_memberships: Vec<usize>,
    pub num_blocks: usize,
    pub current_dual: f64,
    pub current_contribution: f64,
    pub rhs_value: f64,
}

impl LinkingConstraint {
    pub fn new(
        index: usize,
        name: impl Into<String>,
        blocks: Vec<usize>,
        rhs: f64,
    ) -> Self {
        let num_blocks = blocks.len();
        Self {
            constraint_index: index,
            constraint_name: name.into(),
            block_memberships: blocks,
            num_blocks,
            current_dual: 0.0,
            current_contribution: 0.0,
            rhs_value: rhs,
        }
    }

    /// Update dual value and recompute contribution.
    /// Contribution = |μ_i| * (|blocks(i)| - 1)
    pub fn update_dual(&mut self, dual: f64) {
        self.current_dual = dual;
        let factor = if self.num_blocks > 1 {
            (self.num_blocks - 1) as f64
        } else {
            0.0
        };
        self.current_contribution = dual.abs() * factor;
    }

    /// Generic (k-1) factor (uses total number of blocks in problem, not per-constraint).
    pub fn generic_contribution(&self, total_blocks: usize) -> f64 {
        if total_blocks <= 1 {
            return 0.0;
        }
        self.current_dual.abs() * (total_blocks - 1) as f64
    }

    /// Tightening ratio: specific (|blocks(i)|-1) vs generic (k-1).
    pub fn tightening_ratio(&self, total_blocks: usize) -> Option<f64> {
        let generic = self.generic_contribution(total_blocks);
        if generic.abs() < 1e-15 {
            return None;
        }
        Some(self.current_contribution / generic)
    }
}

/// Record of the DW bound at a single iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWIterationRecord {
    pub iteration: usize,
    pub bound_value: f64,
    pub master_objective: f64,
    pub best_primal: f64,
    pub gap: f64,
    pub num_columns_added: usize,
    pub worst_linking_constraint: Option<usize>,
    pub worst_contribution: f64,
    pub dual_norm: f64,
    pub bound_generic: f64,
    pub tightening_achieved: f64,
}

/// DW dual convergence tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWConvergenceTracker {
    pub history: Vec<DWIterationRecord>,
    pub dual_history: Vec<Vec<f64>>,
    pub dual_stability_score: f64,
    pub convergence_rate: Option<f64>,
    pub is_converged: bool,
    pub tolerance: f64,
}

impl DWConvergenceTracker {
    pub fn new(tolerance: f64) -> Self {
        Self {
            history: Vec::new(),
            dual_history: Vec::new(),
            dual_stability_score: 0.0,
            convergence_rate: None,
            is_converged: false,
            tolerance,
        }
    }

    /// Record an iteration and update convergence info.
    pub fn record_iteration(&mut self, record: DWIterationRecord, duals: Vec<f64>) {
        self.history.push(record);
        self.dual_history.push(duals);
        self.update_stability();
        self.update_convergence();
    }

    fn update_stability(&mut self) {
        let n = self.dual_history.len();
        if n < 2 {
            self.dual_stability_score = 0.0;
            return;
        }
        let prev = &self.dual_history[n - 2];
        let curr = &self.dual_history[n - 1];

        let len = prev.len().min(curr.len());
        if len == 0 {
            self.dual_stability_score = 1.0;
            return;
        }

        let mut max_change = 0.0f64;
        for i in 0..len {
            let change = (curr[i] - prev[i]).abs();
            let scale = prev[i].abs().max(1.0);
            max_change = max_change.max(change / scale);
        }

        // Score: 1.0 = perfectly stable, 0.0 = highly unstable
        self.dual_stability_score = (-max_change).exp();
    }

    fn update_convergence(&mut self) {
        let n = self.history.len();
        if n < 3 {
            return;
        }

        let mut improvements = Vec::new();
        for i in 1..n {
            let improvement =
                (self.history[i - 1].bound_value - self.history[i].bound_value).abs();
            improvements.push(improvement);
        }

        if improvements.len() >= 2 {
            let mut ratios = Vec::new();
            for i in 1..improvements.len() {
                if improvements[i - 1] > 1e-15 {
                    ratios.push(improvements[i] / improvements[i - 1]);
                }
            }
            if !ratios.is_empty() {
                self.convergence_rate =
                    Some(ratios.iter().sum::<f64>() / ratios.len() as f64);
            }
        }

        let last = self.history.last().unwrap();
        if last.bound_value < self.tolerance && last.gap < self.tolerance {
            self.is_converged = true;
        }
    }

    pub fn num_iterations(&self) -> usize {
        self.history.len()
    }

    pub fn best_bound(&self) -> Option<f64> {
        self.history
            .iter()
            .map(|r| r.bound_value)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.min(v))))
    }

    /// Average tightening achieved across iterations.
    pub fn average_tightening(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().map(|r| r.tightening_achieved).sum();
        sum / self.history.len() as f64
    }
}

/// L3-C DW specialization certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWCertificate {
    pub id: String,
    pub created_at: String,
    pub linking_constraints: Vec<LinkingConstraint>,
    pub master_duals_per_iteration: Vec<Vec<f64>>,
    pub block_membership_per_constraint: IndexMap<usize, Vec<usize>>,
    pub per_constraint_contribution: Vec<f64>,
    pub convergence: DWConvergenceTracker,
    pub total_blocks: usize,
    pub current_iteration: usize,
    pub metadata: IndexMap<String, String>,
}

impl DWCertificate {
    /// Create a new DW certificate.
    pub fn new(
        linking: Vec<LinkingConstraint>,
        total_blocks: usize,
        convergence_tolerance: f64,
    ) -> Self {
        let membership: IndexMap<usize, Vec<usize>> = linking
            .iter()
            .map(|lc| (lc.constraint_index, lc.block_memberships.clone()))
            .collect();
        let per_constraint = vec![0.0; linking.len()];

        Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            linking_constraints: linking,
            master_duals_per_iteration: Vec::new(),
            block_membership_per_constraint: membership,
            per_constraint_contribution: per_constraint,
            convergence: DWConvergenceTracker::new(convergence_tolerance),
            total_blocks,
            current_iteration: 0,
            metadata: IndexMap::new(),
        }
    }

    /// Compute the DW bound at a given iteration.
    ///
    /// bound = Σ_{i linking} |μ_i^(t)| * (|blocks(i)| - 1)
    pub fn compute_at_iteration(
        &mut self,
        master_duals: &[f64],
        master_objective: f64,
        best_primal: f64,
        num_columns_added: usize,
    ) -> CertificateResult<f64> {
        if master_duals.len() != self.linking_constraints.len() {
            return Err(CertificateError::incomplete_data(
                "master_duals",
                format!(
                    "expected {} values, got {}",
                    self.linking_constraints.len(),
                    master_duals.len()
                ),
            ));
        }

        for (i, &d) in master_duals.iter().enumerate() {
            if d.is_nan() {
                return Err(CertificateError::numerical_precision(
                    format!("master dual {} is NaN", i),
                    d,
                    0.0,
                ));
            }
        }

        let mut bound = 0.0;
        let mut bound_generic = 0.0;
        let mut worst_idx = None;
        let mut worst_contribution = 0.0f64;

        for (i, lc) in self.linking_constraints.iter_mut().enumerate() {
            lc.update_dual(master_duals[i]);
            self.per_constraint_contribution[i] = lc.current_contribution;
            bound += lc.current_contribution;
            bound_generic += lc.generic_contribution(self.total_blocks);

            if lc.current_contribution > worst_contribution {
                worst_contribution = lc.current_contribution;
                worst_idx = Some(lc.constraint_index);
            }
        }

        self.master_duals_per_iteration
            .push(master_duals.to_vec());

        let dual_norm: f64 = master_duals.iter().map(|d| d * d).sum::<f64>().sqrt();

        let gap = if best_primal.is_finite() && master_objective.is_finite() {
            (best_primal - master_objective).abs() / best_primal.abs().max(1.0)
        } else {
            f64::INFINITY
        };

        let tightening = if bound_generic > 1e-15 {
            1.0 - (bound / bound_generic)
        } else {
            0.0
        };

        let record = DWIterationRecord {
            iteration: self.current_iteration,
            bound_value: bound,
            master_objective,
            best_primal,
            gap,
            num_columns_added,
            worst_linking_constraint: worst_idx,
            worst_contribution,
            dual_norm,
            bound_generic,
            tightening_achieved: tightening,
        };

        self.convergence
            .record_iteration(record, master_duals.to_vec());
        self.current_iteration += 1;

        Ok(bound)
    }

    /// Current bound value.
    pub fn current_bound(&self) -> f64 {
        self.convergence
            .history
            .last()
            .map(|r| r.bound_value)
            .unwrap_or(f64::INFINITY)
    }

    /// Compare (|blocks(i)|-1) tightening vs generic (k-1) bound.
    pub fn tightening_analysis(&self) -> IndexMap<String, f64> {
        let mut analysis = IndexMap::new();

        let specific: f64 = self.per_constraint_contribution.iter().sum();
        let generic: f64 = self
            .linking_constraints
            .iter()
            .map(|lc| lc.generic_contribution(self.total_blocks))
            .sum();

        analysis.insert("specific_bound".to_string(), specific);
        analysis.insert("generic_bound".to_string(), generic);

        if generic > 1e-15 {
            analysis.insert("tightening_ratio".to_string(), specific / generic);
            analysis.insert("tightening_percent".to_string(), (1.0 - specific / generic) * 100.0);
        }

        let avg_blocks_per_constraint = if self.linking_constraints.is_empty() {
            0.0
        } else {
            self.linking_constraints
                .iter()
                .map(|lc| lc.num_blocks as f64)
                .sum::<f64>()
                / self.linking_constraints.len() as f64
        };
        analysis.insert("avg_blocks_per_constraint".to_string(), avg_blocks_per_constraint);
        analysis.insert("total_blocks".to_string(), self.total_blocks as f64);

        analysis
    }

    /// Track dual convergence across iterations.
    pub fn dual_convergence_profile(&self) -> Vec<f64> {
        if self.master_duals_per_iteration.len() < 2 {
            return Vec::new();
        }

        let mut changes = Vec::new();
        for i in 1..self.master_duals_per_iteration.len() {
            let prev = &self.master_duals_per_iteration[i - 1];
            let curr = &self.master_duals_per_iteration[i];
            let len = prev.len().min(curr.len());
            let change: f64 = (0..len)
                .map(|j| (curr[j] - prev[j]).powi(2))
                .sum::<f64>()
                .sqrt();
            changes.push(change);
        }
        changes
    }

    /// Identify linking constraints with highest contribution.
    pub fn worst_linking_constraints(&self, n: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .linking_constraints
            .iter()
            .map(|lc| (lc.constraint_index, lc.current_contribution))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(n);
        indexed
    }

    /// Stability of the certificate: high stability = duals converged.
    pub fn stability_score(&self) -> f64 {
        self.convergence.dual_stability_score
    }

    /// Whether dual oscillation is detected (unstable pricing).
    pub fn detect_oscillation(&self, window: usize) -> bool {
        let profile = self.dual_convergence_profile();
        if profile.len() < window + 1 {
            return false;
        }
        let recent = &profile[profile.len() - window..];
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        let cv = if mean.abs() > 1e-15 {
            variance.sqrt() / mean
        } else {
            0.0
        };
        cv > 0.5
    }

    /// Summary statistics.
    pub fn summary_stats(&self) -> IndexMap<String, f64> {
        let mut stats = IndexMap::new();
        stats.insert("num_linking_constraints".to_string(), self.linking_constraints.len() as f64);
        stats.insert("total_blocks".to_string(), self.total_blocks as f64);
        stats.insert("current_bound".to_string(), self.current_bound());
        stats.insert("iterations".to_string(), self.current_iteration as f64);
        stats.insert("stability".to_string(), self.stability_score());
        if let Some(rate) = self.convergence.convergence_rate {
            stats.insert("convergence_rate".to_string(), rate);
        }
        let ta = self.tightening_analysis();
        for (k, v) in ta {
            stats.insert(format!("tightening_{}", k), v);
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linking(n: usize) -> Vec<LinkingConstraint> {
        (0..n)
            .map(|i| LinkingConstraint::new(i, format!("link{}", i), vec![0, 1], 10.0))
            .collect()
    }

    #[test]
    fn test_linking_constraint_creation() {
        let lc = LinkingConstraint::new(0, "link0", vec![0, 1, 2], 5.0);
        assert_eq!(lc.num_blocks, 3);
        assert_eq!(lc.rhs_value, 5.0);
    }

    #[test]
    fn test_linking_constraint_update_dual() {
        let mut lc = LinkingConstraint::new(0, "link0", vec![0, 1, 2], 5.0);
        lc.update_dual(-4.0);
        // contribution = 4.0 * (3-1) = 8.0
        assert!((lc.current_contribution - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_linking_constraint_generic() {
        let mut lc = LinkingConstraint::new(0, "link0", vec![0, 1], 5.0);
        lc.update_dual(3.0);
        let generic = lc.generic_contribution(5);
        // generic = 3.0 * (5-1) = 12.0
        assert!((generic - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_linking_tightening_ratio() {
        let mut lc = LinkingConstraint::new(0, "link0", vec![0, 1], 5.0);
        lc.update_dual(3.0);
        let ratio = lc.tightening_ratio(5).unwrap();
        // specific = 3.0 * 1 = 3.0, generic = 3.0 * 4 = 12.0
        assert!((ratio - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_dw_certificate_creation() {
        let lcs = make_linking(3);
        let cert = DWCertificate::new(lcs, 4, 1e-6);
        assert_eq!(cert.linking_constraints.len(), 3);
        assert_eq!(cert.total_blocks, 4);
    }

    #[test]
    fn test_compute_at_iteration() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 3, 1e-6);
        let bound = cert
            .compute_at_iteration(&[2.0, 3.0], 90.0, 100.0, 5)
            .unwrap();
        // |2.0| * (2-1) + |3.0| * (2-1) = 2.0 + 3.0 = 5.0
        assert!((bound - 5.0).abs() < 1e-10);
        assert_eq!(cert.current_iteration, 1);
    }

    #[test]
    fn test_compute_wrong_size() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 3, 1e-6);
        let result = cert.compute_at_iteration(&[1.0], 90.0, 100.0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_tightening_analysis() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 5, 1e-6);
        let _ = cert.compute_at_iteration(&[2.0, 3.0], 90.0, 100.0, 5);
        let analysis = cert.tightening_analysis();
        assert!(analysis.contains_key("specific_bound"));
        assert!(analysis.contains_key("generic_bound"));
        let specific = analysis["specific_bound"];
        let generic = analysis["generic_bound"];
        assert!(specific <= generic);
    }

    #[test]
    fn test_dual_convergence_profile() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 3, 1e-6);
        let _ = cert.compute_at_iteration(&[2.0, 3.0], 90.0, 100.0, 5);
        let _ = cert.compute_at_iteration(&[2.1, 3.1], 91.0, 100.0, 3);
        let profile = cert.dual_convergence_profile();
        assert_eq!(profile.len(), 1);
        assert!(profile[0] > 0.0);
    }

    #[test]
    fn test_worst_linking_constraints() {
        let lcs = make_linking(3);
        let mut cert = DWCertificate::new(lcs, 3, 1e-6);
        let _ = cert.compute_at_iteration(&[1.0, 5.0, 2.0], 90.0, 100.0, 3);
        let worst = cert.worst_linking_constraints(2);
        assert_eq!(worst.len(), 2);
        assert_eq!(worst[0].0, 1);
    }

    #[test]
    fn test_stability_score() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 3, 1e-6);
        let _ = cert.compute_at_iteration(&[2.0, 3.0], 90.0, 100.0, 5);
        let _ = cert.compute_at_iteration(&[2.0, 3.0], 91.0, 100.0, 3);
        assert!(cert.stability_score() > 0.9);
    }

    #[test]
    fn test_detect_oscillation_no() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 3, 1e-6);
        for i in 0..10 {
            let v = 2.0 + i as f64 * 0.01;
            let _ = cert.compute_at_iteration(&[v, v], 90.0, 100.0, 1);
        }
        assert!(!cert.detect_oscillation(5));
    }

    #[test]
    fn test_summary_stats() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 3, 1e-6);
        let _ = cert.compute_at_iteration(&[1.0, 1.0], 90.0, 100.0, 5);
        let stats = cert.summary_stats();
        assert!(stats.contains_key("current_bound"));
        assert!(stats.contains_key("stability"));
    }

    #[test]
    fn test_convergence_tracker_average_tightening() {
        let lcs = make_linking(2);
        let mut cert = DWCertificate::new(lcs, 5, 1e-6);
        for _ in 0..5 {
            let _ = cert.compute_at_iteration(&[2.0, 3.0], 90.0, 100.0, 3);
        }
        let avg = cert.convergence.average_tightening();
        assert!(avg >= 0.0);
    }
}
