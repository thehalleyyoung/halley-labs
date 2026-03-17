//! Benders decomposition module.
//!
//! Implements classical Benders decomposition with Magnanti-Wong cut
//! strengthening, multi-cut generation, and cut pool management.

pub mod cuts;
pub mod decomposition;

use serde::{Deserialize, Serialize};

// Re-export key types for convenience.
pub use cuts::{BendersCut, CutPool, CutType};
pub use decomposition::{BendersDecomposition, SubproblemResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Benders decomposition solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendersConfig {
    /// Maximum number of Benders iterations.
    pub max_iterations: usize,
    /// Relative optimality gap tolerance.
    pub gap_tolerance: f64,
    /// Wall-clock time limit in seconds.
    pub time_limit: f64,
    /// Whether to use Magnanti-Wong cut strengthening.
    pub use_magnanti_wong: bool,
    /// Whether to use multi-cut (one θ per block) vs single-cut.
    pub multi_cut: bool,
    /// Maximum optimality/feasibility cuts added per round.
    pub max_cuts_per_round: usize,
    /// How often (in iterations) to clean up the cut pool.
    pub cut_cleanup_frequency: usize,
    /// Maximum age before a cut is eligible for removal.
    pub cut_age_limit: usize,
    /// Whether to warm-start sub-problems with previous basis.
    pub warm_start: bool,
    /// Verbose logging.
    pub verbose: bool,
}

impl Default for BendersConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            gap_tolerance: 1e-6,
            time_limit: 3600.0,
            use_magnanti_wong: true,
            multi_cut: true,
            max_cuts_per_round: 100,
            cut_cleanup_frequency: 50,
            cut_age_limit: 100,
            warm_start: true,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Status & result
// ---------------------------------------------------------------------------

/// Termination status for the Benders solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BendersStatus {
    /// Proven optimal within gap tolerance.
    Optimal,
    /// Relative gap closed below tolerance.
    GapClosed,
    /// Maximum iteration count reached.
    IterationLimit,
    /// Wall-clock time limit exceeded.
    TimeLimit,
    /// Problem (or a subproblem) is infeasible.
    Infeasible,
    /// Numerical difficulties prevented convergence.
    NumericalError,
}

/// Summary of one cut-generation round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutRoundInfo {
    pub iteration: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub cuts_added: usize,
    pub gap: f64,
}

/// Result returned by the Benders decomposition solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendersResult {
    /// Termination status.
    pub status: BendersStatus,
    /// Best known lower bound (from master).
    pub lower_bound: f64,
    /// Best known upper bound (from subproblems).
    pub upper_bound: f64,
    /// Relative optimality gap.
    pub gap: f64,
    /// Number of Benders iterations performed.
    pub iterations: usize,
    /// Total optimality cuts generated.
    pub num_optimality_cuts: usize,
    /// Total feasibility cuts generated.
    pub num_feasibility_cuts: usize,
    /// Incumbent complicating-variable solution.
    pub master_solution: Vec<f64>,
    /// Wall-clock time in seconds.
    pub time_seconds: f64,
    /// Per-round convergence history.
    pub cut_history: Vec<CutRoundInfo>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the relative gap given lower and upper bounds.
pub fn relative_gap(lb: f64, ub: f64) -> f64 {
    if ub.is_infinite() || lb.is_infinite() {
        return f64::INFINITY;
    }
    let denom = f64::max(1.0, ub.abs());
    (ub - lb).abs() / denom
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = BendersConfig::default();
        assert_eq!(cfg.max_iterations, 1000);
        assert!((cfg.gap_tolerance - 1e-6).abs() < 1e-15);
        assert!((cfg.time_limit - 3600.0).abs() < 1e-12);
        assert!(cfg.use_magnanti_wong);
        assert!(cfg.multi_cut);
        assert_eq!(cfg.max_cuts_per_round, 100);
        assert_eq!(cfg.cut_cleanup_frequency, 50);
        assert_eq!(cfg.cut_age_limit, 100);
        assert!(cfg.warm_start);
        assert!(!cfg.verbose);
    }

    #[test]
    fn test_benders_status_eq() {
        assert_eq!(BendersStatus::Optimal, BendersStatus::Optimal);
        assert_ne!(BendersStatus::Optimal, BendersStatus::Infeasible);
    }

    #[test]
    fn test_relative_gap_basic() {
        let gap = relative_gap(90.0, 100.0);
        assert!((gap - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_relative_gap_small_values() {
        // When |ub| < 1, denominator is clamped to 1.
        let gap = relative_gap(0.0, 0.5);
        assert!((gap - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_relative_gap_negative() {
        let gap = relative_gap(-110.0, -100.0);
        assert!((gap - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_relative_gap_infinite() {
        let gap = relative_gap(0.0, f64::INFINITY);
        assert!(gap.is_infinite());
    }

    #[test]
    fn test_relative_gap_equal() {
        let gap = relative_gap(42.0, 42.0);
        assert!(gap < 1e-15);
    }

    #[test]
    fn test_cut_round_info_construction() {
        let info = CutRoundInfo {
            iteration: 5,
            lower_bound: 10.0,
            upper_bound: 12.0,
            cuts_added: 3,
            gap: 0.2,
        };
        assert_eq!(info.iteration, 5);
        assert_eq!(info.cuts_added, 3);
    }

    #[test]
    fn test_benders_result_construction() {
        let result = BendersResult {
            status: BendersStatus::Optimal,
            lower_bound: 100.0,
            upper_bound: 100.0,
            gap: 0.0,
            iterations: 12,
            num_optimality_cuts: 10,
            num_feasibility_cuts: 2,
            master_solution: vec![1.0, 2.0],
            time_seconds: 0.5,
            cut_history: vec![],
        };
        assert_eq!(result.status, BendersStatus::Optimal);
        assert_eq!(result.iterations, 12);
        assert_eq!(result.master_solution.len(), 2);
    }

    #[test]
    fn test_config_clone() {
        let cfg = BendersConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.max_iterations, cfg2.max_iterations);
        assert!((cfg.gap_tolerance - cfg2.gap_tolerance).abs() < 1e-15);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = BendersConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: BendersConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.max_iterations, cfg2.max_iterations);
        assert!((cfg.gap_tolerance - cfg2.gap_tolerance).abs() < 1e-15);
    }
}
