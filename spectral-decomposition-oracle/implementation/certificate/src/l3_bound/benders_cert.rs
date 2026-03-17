//! L3-C Benders specialization certificate.
//!
//! For Benders decomposition at iteration t:
//!   z_LP - z_Benders^(t) ≤ Σ_{j coupling} |r_j^(t)| * |blocks(j)|
//!
//! where r_j^(t) is the reduced cost of coupling variable j at iteration t
//! and |blocks(j)| is the number of subproblems that variable j appears in.

use crate::error::{CertificateError, CertificateResult};
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A coupling variable in a Benders decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingVariable {
    pub variable_index: usize,
    pub variable_name: String,
    pub block_memberships: Vec<usize>,
    pub num_blocks: usize,
    pub current_reduced_cost: f64,
    pub current_contribution: f64,
}

impl CouplingVariable {
    /// Create a new coupling variable.
    pub fn new(
        index: usize,
        name: impl Into<String>,
        blocks: Vec<usize>,
    ) -> Self {
        let num_blocks = blocks.len();
        Self {
            variable_index: index,
            variable_name: name.into(),
            block_memberships: blocks,
            num_blocks,
            current_reduced_cost: 0.0,
            current_contribution: 0.0,
        }
    }

    /// Update reduced cost and recompute contribution.
    pub fn update_reduced_cost(&mut self, rc: f64) {
        self.current_reduced_cost = rc;
        self.current_contribution = rc.abs() * self.num_blocks as f64;
    }
}

/// Record of the bound at a single Benders iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationRecord {
    pub iteration: usize,
    pub bound_value: f64,
    pub master_objective: f64,
    pub best_primal: f64,
    pub gap: f64,
    pub num_cuts_added: usize,
    pub worst_coupling_var: Option<usize>,
    pub worst_contribution: f64,
    pub total_reduced_cost_norm: f64,
}

/// Convergence tracking for Benders bound certificates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceTracker {
    pub history: Vec<IterationRecord>,
    pub convergence_rate: Option<f64>,
    pub estimated_iterations_remaining: Option<usize>,
    pub is_converged: bool,
    pub convergence_tolerance: f64,
}

impl ConvergenceTracker {
    pub fn new(tolerance: f64) -> Self {
        Self {
            history: Vec::new(),
            convergence_rate: None,
            estimated_iterations_remaining: None,
            is_converged: false,
            convergence_tolerance: tolerance,
        }
    }

    /// Add a new iteration record and update convergence estimates.
    pub fn record_iteration(&mut self, record: IterationRecord) {
        let prev_bound = self.history.last().map(|r| r.bound_value);
        self.history.push(record);

        if self.history.len() >= 2 {
            self.update_convergence_rate();
        }

        if let Some(prev) = prev_bound {
            let current = self.history.last().unwrap().bound_value;
            let improvement = (prev - current).abs();
            if improvement < self.convergence_tolerance && current < self.convergence_tolerance {
                self.is_converged = true;
            }
        }
    }

    fn update_convergence_rate(&mut self) {
        let n = self.history.len();
        if n < 3 {
            return;
        }

        // Compute average ratio of successive improvements
        let mut ratios = Vec::new();
        for i in 2..n {
            let prev_improvement =
                (self.history[i - 1].bound_value - self.history[i - 2].bound_value).abs();
            let curr_improvement =
                (self.history[i].bound_value - self.history[i - 1].bound_value).abs();
            if prev_improvement > 1e-15 {
                ratios.push(curr_improvement / prev_improvement);
            }
        }

        if ratios.is_empty() {
            return;
        }

        let avg_ratio: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
        self.convergence_rate = Some(avg_ratio);

        // Estimate remaining iterations based on geometric convergence
        if avg_ratio < 1.0 && avg_ratio > 0.0 {
            let current_bound = self.history.last().unwrap().bound_value;
            if current_bound > self.convergence_tolerance {
                let log_ratio = avg_ratio.ln();
                let log_target = (self.convergence_tolerance / current_bound).ln();
                let remaining = (log_target / log_ratio).ceil() as usize;
                self.estimated_iterations_remaining = Some(remaining.min(10000));
            }
        }
    }

    /// Smoothed bound using exponential moving average.
    pub fn smoothed_bound(&self, alpha: f64) -> Vec<f64> {
        let alpha = alpha.clamp(0.0, 1.0);
        let mut smoothed = Vec::with_capacity(self.history.len());
        let mut ema = 0.0;
        for (i, record) in self.history.iter().enumerate() {
            if i == 0 {
                ema = record.bound_value;
            } else {
                ema = alpha * record.bound_value + (1.0 - alpha) * ema;
            }
            smoothed.push(ema);
        }
        smoothed
    }

    /// Number of iterations recorded.
    pub fn num_iterations(&self) -> usize {
        self.history.len()
    }

    /// Best (lowest) bound achieved.
    pub fn best_bound(&self) -> Option<f64> {
        self.history
            .iter()
            .map(|r| r.bound_value)
            .fold(None, |acc, v| {
                Some(acc.map_or(v, |a: f64| a.min(v)))
            })
    }
}

/// L3-C Benders specialization certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendersCertificate {
    pub id: String,
    pub created_at: String,
    pub coupling_variables: Vec<CouplingVariable>,
    pub reduced_costs_per_iteration: Vec<Vec<f64>>,
    pub block_membership: IndexMap<usize, Vec<usize>>,
    pub per_variable_contribution: Vec<f64>,
    pub convergence: ConvergenceTracker,
    pub current_iteration: usize,
    pub metadata: IndexMap<String, String>,
}

impl BendersCertificate {
    /// Create a new Benders certificate with initial coupling variable info.
    pub fn new(
        coupling_vars: Vec<CouplingVariable>,
        convergence_tolerance: f64,
    ) -> Self {
        let block_membership: IndexMap<usize, Vec<usize>> = coupling_vars
            .iter()
            .map(|cv| (cv.variable_index, cv.block_memberships.clone()))
            .collect();
        let per_var = vec![0.0; coupling_vars.len()];

        Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            coupling_variables: coupling_vars,
            reduced_costs_per_iteration: Vec::new(),
            block_membership,
            per_variable_contribution: per_var,
            convergence: ConvergenceTracker::new(convergence_tolerance),
            current_iteration: 0,
            metadata: IndexMap::new(),
        }
    }

    /// Compute the bound at a Benders iteration.
    ///
    /// bound = Σ_{j coupling} |r_j^(t)| * |blocks(j)|
    pub fn compute_at_iteration(
        &mut self,
        reduced_costs: &[f64],
        master_objective: f64,
        best_primal: f64,
    ) -> CertificateResult<f64> {
        if reduced_costs.len() != self.coupling_variables.len() {
            return Err(CertificateError::incomplete_data(
                "reduced_costs",
                format!(
                    "expected {} values, got {}",
                    self.coupling_variables.len(),
                    reduced_costs.len()
                ),
            ));
        }

        for (i, rc) in reduced_costs.iter().enumerate() {
            if rc.is_nan() {
                return Err(CertificateError::numerical_precision(
                    format!("reduced cost {} is NaN", i),
                    *rc,
                    0.0,
                ));
            }
        }

        let mut bound = 0.0;
        let mut worst_var = None;
        let mut worst_contribution = 0.0f64;

        for (i, cv) in self.coupling_variables.iter_mut().enumerate() {
            cv.update_reduced_cost(reduced_costs[i]);
            self.per_variable_contribution[i] = cv.current_contribution;
            bound += cv.current_contribution;

            if cv.current_contribution > worst_contribution {
                worst_contribution = cv.current_contribution;
                worst_var = Some(cv.variable_index);
            }
        }

        self.reduced_costs_per_iteration
            .push(reduced_costs.to_vec());

        let rc_norm: f64 = reduced_costs.iter().map(|r| r * r).sum::<f64>().sqrt();

        let gap = if best_primal.is_finite() && master_objective.is_finite() {
            (best_primal - master_objective).abs()
                / (best_primal.abs().max(1.0))
        } else {
            f64::INFINITY
        };

        let record = IterationRecord {
            iteration: self.current_iteration,
            bound_value: bound,
            master_objective,
            best_primal,
            gap,
            num_cuts_added: 0,
            worst_coupling_var: worst_var,
            worst_contribution,
            total_reduced_cost_norm: rc_norm,
        };

        self.convergence.record_iteration(record);
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

    /// Track convergence of the bound across iterations.
    pub fn convergence_summary(&self) -> IndexMap<String, f64> {
        let mut summary = IndexMap::new();
        summary.insert("num_iterations".to_string(), self.current_iteration as f64);
        summary.insert("current_bound".to_string(), self.current_bound());

        if let Some(rate) = self.convergence.convergence_rate {
            summary.insert("convergence_rate".to_string(), rate);
        }
        if let Some(remaining) = self.convergence.estimated_iterations_remaining {
            summary.insert("est_remaining_iters".to_string(), remaining as f64);
        }
        if let Some(best) = self.convergence.best_bound() {
            summary.insert("best_bound".to_string(), best);
        }
        summary.insert(
            "is_converged".to_string(),
            if self.convergence.is_converged {
                1.0
            } else {
                0.0
            },
        );
        summary
    }

    /// Identify worst coupling variables contributing most to the bound.
    pub fn worst_coupling_variables(&self, n: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .coupling_variables
            .iter()
            .map(|cv| (cv.variable_index, cv.current_contribution))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(n);
        indexed
    }

    /// Average reduced cost magnitude over all iterations.
    pub fn average_reduced_cost_history(&self) -> Vec<f64> {
        self.reduced_costs_per_iteration
            .iter()
            .map(|rcs| {
                if rcs.is_empty() {
                    0.0
                } else {
                    rcs.iter().map(|r| r.abs()).sum::<f64>() / rcs.len() as f64
                }
            })
            .collect()
    }

    /// Fraction of total bound from top N coupling variables.
    pub fn coupling_concentration(&self, n: usize) -> f64 {
        let total: f64 = self.per_variable_contribution.iter().sum();
        if total.abs() < 1e-15 {
            return 0.0;
        }
        let mut sorted = self.per_variable_contribution.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top_sum: f64 = sorted.iter().take(n).sum();
        top_sum / total
    }

    /// Check if bound has plateaued (no significant improvement in last n iterations).
    pub fn has_plateaued(&self, window: usize, tolerance: f64) -> bool {
        let n = self.convergence.history.len();
        if n < window + 1 {
            return false;
        }
        let recent = &self.convergence.history[n - window..];
        let first = recent.first().unwrap().bound_value;
        let last = recent.last().unwrap().bound_value;
        (first - last).abs() < tolerance
    }

    /// Total number of blocks across all coupling variables.
    pub fn total_block_involvement(&self) -> usize {
        self.coupling_variables
            .iter()
            .map(|cv| cv.num_blocks)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coupling_vars(n: usize) -> Vec<CouplingVariable> {
        (0..n)
            .map(|i| CouplingVariable::new(i, format!("x{}", i), vec![0, 1]))
            .collect()
    }

    #[test]
    fn test_coupling_variable_creation() {
        let cv = CouplingVariable::new(0, "x0", vec![0, 1, 2]);
        assert_eq!(cv.num_blocks, 3);
        assert_eq!(cv.variable_index, 0);
    }

    #[test]
    fn test_coupling_variable_update() {
        let mut cv = CouplingVariable::new(0, "x0", vec![0, 1]);
        cv.update_reduced_cost(-3.0);
        assert!((cv.current_contribution - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_benders_certificate_creation() {
        let vars = make_coupling_vars(3);
        let cert = BendersCertificate::new(vars, 1e-6);
        assert_eq!(cert.coupling_variables.len(), 3);
        assert_eq!(cert.current_iteration, 0);
    }

    #[test]
    fn test_compute_at_iteration() {
        let vars = make_coupling_vars(2);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        let bound = cert.compute_at_iteration(&[1.0, 2.0], 90.0, 100.0).unwrap();
        // |1.0| * 2 + |2.0| * 2 = 2.0 + 4.0 = 6.0
        assert!((bound - 6.0).abs() < 1e-10);
        assert_eq!(cert.current_iteration, 1);
    }

    #[test]
    fn test_compute_wrong_size() {
        let vars = make_coupling_vars(2);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        let result = cert.compute_at_iteration(&[1.0], 90.0, 100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_nan_reduced_cost() {
        let vars = make_coupling_vars(2);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        let result = cert.compute_at_iteration(&[f64::NAN, 1.0], 90.0, 100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_convergence_tracking() {
        let vars = make_coupling_vars(2);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        for i in 0..10 {
            let scale = 1.0 / (i as f64 + 1.0);
            let _ = cert.compute_at_iteration(&[scale, scale * 0.5], 90.0 + scale, 100.0);
        }
        assert_eq!(cert.convergence.num_iterations(), 10);
        assert!(cert.convergence.best_bound().is_some());
    }

    #[test]
    fn test_worst_coupling_variables() {
        let vars = make_coupling_vars(3);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        let _ = cert.compute_at_iteration(&[1.0, 5.0, 2.0], 90.0, 100.0);
        let worst = cert.worst_coupling_variables(2);
        assert_eq!(worst.len(), 2);
        assert_eq!(worst[0].0, 1); // variable 1 with rc=5.0 is worst
    }

    #[test]
    fn test_coupling_concentration() {
        let vars = make_coupling_vars(4);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        let _ = cert.compute_at_iteration(&[10.0, 1.0, 1.0, 1.0], 90.0, 100.0);
        let conc = cert.coupling_concentration(1);
        assert!(conc > 0.7);
    }

    #[test]
    fn test_has_plateaued() {
        let vars = make_coupling_vars(2);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        for _ in 0..5 {
            let _ = cert.compute_at_iteration(&[1.0, 1.0], 90.0, 100.0);
        }
        assert!(cert.has_plateaued(3, 0.01));
    }

    #[test]
    fn test_average_reduced_cost_history() {
        let vars = make_coupling_vars(2);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        let _ = cert.compute_at_iteration(&[2.0, 4.0], 90.0, 100.0);
        let _ = cert.compute_at_iteration(&[1.0, 3.0], 91.0, 100.0);
        let avg = cert.average_reduced_cost_history();
        assert_eq!(avg.len(), 2);
        assert!((avg[0] - 3.0).abs() < 1e-10);
        assert!((avg[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_convergence_summary() {
        let vars = make_coupling_vars(2);
        let mut cert = BendersCertificate::new(vars, 1e-6);
        let _ = cert.compute_at_iteration(&[1.0, 1.0], 90.0, 100.0);
        let summary = cert.convergence_summary();
        assert!(summary.contains_key("num_iterations"));
        assert!(summary.contains_key("current_bound"));
    }

    #[test]
    fn test_smoothed_bound() {
        let mut tracker = ConvergenceTracker::new(1e-6);
        for i in 0..5 {
            tracker.record_iteration(IterationRecord {
                iteration: i,
                bound_value: 10.0 / (i as f64 + 1.0),
                master_objective: 90.0,
                best_primal: 100.0,
                gap: 0.1,
                num_cuts_added: 1,
                worst_coupling_var: None,
                worst_contribution: 0.0,
                total_reduced_cost_norm: 1.0,
            });
        }
        let smoothed = tracker.smoothed_bound(0.5);
        assert_eq!(smoothed.len(), 5);
        // Smoothed values should be between bounds
        assert!(smoothed[0] > 0.0);
    }

    #[test]
    fn test_total_block_involvement() {
        let vars = vec![
            CouplingVariable::new(0, "x0", vec![0, 1]),
            CouplingVariable::new(1, "x1", vec![0, 1, 2]),
        ];
        let cert = BendersCertificate::new(vars, 1e-6);
        assert_eq!(cert.total_block_involvement(), 5);
    }
}
