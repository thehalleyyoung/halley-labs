// ---------------------------------------------------------------------------
// Dantzig-Wolfe decomposition module
// ---------------------------------------------------------------------------

pub mod column;
pub mod decomposition;

use serde::{Deserialize, Serialize};

pub use column::{ColumnPool, DWColumn};
pub use decomposition::DWDecomposition;

// ---------------------------------------------------------------------------
// Stabilization strategy
// ---------------------------------------------------------------------------

/// Dual stabilization techniques for column generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DWStabilization {
    /// No stabilization.
    None,
    /// Du Merle smoothing with parameter alpha in (0,1).
    DuSmoothing { alpha: f64 },
    /// Box-step method with trust-region radius delta.
    BoxStep { delta: f64 },
    /// Wentges smoothing with parameter alpha in (0,1).
    Wentges { alpha: f64 },
}

impl Default for DWStabilization {
    fn default() -> Self {
        DWStabilization::None
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Dantzig-Wolfe decomposition solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWConfig {
    pub max_iterations: usize,
    pub gap_tolerance: f64,
    pub time_limit: f64,
    pub max_columns_per_iter: usize,
    pub column_age_limit: usize,
    pub column_cleanup_frequency: usize,
    pub stabilization: DWStabilization,
    pub initial_smoothing_alpha: f64,
    pub phase_one_max_iter: usize,
    pub verbose: bool,
}

impl Default for DWConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            gap_tolerance: 1e-6,
            time_limit: 3600.0,
            max_columns_per_iter: 50,
            column_age_limit: 200,
            column_cleanup_frequency: 50,
            stabilization: DWStabilization::None,
            initial_smoothing_alpha: 0.3,
            phase_one_max_iter: 100,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Status & results
// ---------------------------------------------------------------------------

/// Termination status of DW decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DWStatus {
    Optimal,
    GapClosed,
    IterationLimit,
    TimeLimit,
    Infeasible,
    NumericalError,
}

/// Per-iteration column generation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnRoundInfo {
    pub iteration: usize,
    pub lower_bound: f64,
    pub num_new_columns: usize,
    pub min_reduced_cost: f64,
}

/// Result returned by the DW decomposition solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWResult {
    pub status: DWStatus,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub gap: f64,
    pub iterations: usize,
    pub num_columns_generated: usize,
    pub master_solution: Vec<f64>,
    pub time_seconds: f64,
    pub column_history: Vec<ColumnRoundInfo>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = DWConfig::default();
        assert_eq!(cfg.max_iterations, 1000);
        assert!((cfg.gap_tolerance - 1e-6).abs() < 1e-15);
        assert!((cfg.time_limit - 3600.0).abs() < 1e-12);
        assert_eq!(cfg.max_columns_per_iter, 50);
        assert_eq!(cfg.column_age_limit, 200);
        assert_eq!(cfg.column_cleanup_frequency, 50);
        assert!((cfg.initial_smoothing_alpha - 0.3).abs() < 1e-12);
        assert_eq!(cfg.phase_one_max_iter, 100);
        assert!(!cfg.verbose);
    }

    #[test]
    fn test_default_stabilization() {
        let s = DWStabilization::default();
        assert!(matches!(s, DWStabilization::None));
    }

    #[test]
    fn test_stabilization_variants() {
        let a = DWStabilization::DuSmoothing { alpha: 0.5 };
        let b = DWStabilization::BoxStep { delta: 1.0 };
        let c = DWStabilization::Wentges { alpha: 0.7 };
        assert!(matches!(a, DWStabilization::DuSmoothing { .. }));
        assert!(matches!(b, DWStabilization::BoxStep { .. }));
        assert!(matches!(c, DWStabilization::Wentges { .. }));
    }

    #[test]
    fn test_dw_status_eq() {
        assert_eq!(DWStatus::Optimal, DWStatus::Optimal);
        assert_ne!(DWStatus::Optimal, DWStatus::Infeasible);
    }

    #[test]
    fn test_column_round_info() {
        let info = ColumnRoundInfo {
            iteration: 5,
            lower_bound: -10.0,
            num_new_columns: 3,
            min_reduced_cost: -0.5,
        };
        assert_eq!(info.iteration, 5);
        assert_eq!(info.num_new_columns, 3);
    }

    #[test]
    fn test_dw_result_construction() {
        let result = DWResult {
            status: DWStatus::Optimal,
            lower_bound: -7.0,
            upper_bound: -7.0,
            gap: 0.0,
            iterations: 12,
            num_columns_generated: 25,
            master_solution: vec![1.0, 2.0, 3.0],
            time_seconds: 0.5,
            column_history: vec![],
        };
        assert_eq!(result.status, DWStatus::Optimal);
        assert_eq!(result.iterations, 12);
        assert_eq!(result.master_solution.len(), 3);
    }

    #[test]
    fn test_config_custom() {
        let cfg = DWConfig {
            max_iterations: 500,
            gap_tolerance: 1e-8,
            time_limit: 60.0,
            max_columns_per_iter: 10,
            column_age_limit: 100,
            column_cleanup_frequency: 25,
            stabilization: DWStabilization::DuSmoothing { alpha: 0.4 },
            initial_smoothing_alpha: 0.4,
            phase_one_max_iter: 50,
            verbose: true,
        };
        assert_eq!(cfg.max_iterations, 500);
        assert!(cfg.verbose);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = DWConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: DWConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg2.max_iterations, cfg.max_iterations);
        assert!((cfg2.gap_tolerance - cfg.gap_tolerance).abs() < 1e-15);
    }

    #[test]
    fn test_status_serde_roundtrip() {
        let status = DWStatus::GapClosed;
        let json = serde_json::to_string(&status).unwrap();
        let status2: DWStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status2, DWStatus::GapClosed);
    }

    #[test]
    fn test_result_serde_roundtrip() {
        let result = DWResult {
            status: DWStatus::IterationLimit,
            lower_bound: -5.0,
            upper_bound: -3.0,
            gap: 2.0,
            iterations: 100,
            num_columns_generated: 50,
            master_solution: vec![0.5, 0.5],
            time_seconds: 1.23,
            column_history: vec![ColumnRoundInfo {
                iteration: 0,
                lower_bound: -10.0,
                num_new_columns: 2,
                min_reduced_cost: -1.0,
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: DWResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result2.status, DWStatus::IterationLimit);
        assert_eq!(result2.column_history.len(), 1);
    }
}
