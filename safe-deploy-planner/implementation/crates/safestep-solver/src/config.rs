// Solver configuration types and defaults.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Strategy for solver restarts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RestartStrategy {
    /// Luby sequence restarts.
    Luby { base_interval: u64 },
    /// Geometric growth restarts.
    Geometric { initial: u64, factor: f64 },
    /// Fixed interval restarts.
    Fixed { interval: u64 },
    /// No restarts.
    Never,
}

impl Default for RestartStrategy {
    fn default() -> Self {
        RestartStrategy::Luby { base_interval: 100 }
    }
}

/// Strategy for clause deletion / garbage collection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClauseDeletionStrategy {
    /// Delete clauses with activity below threshold.
    Activity { threshold: f64 },
    /// Keep only clauses with LBD at most the given value.
    Lbd { max_lbd: u32 },
    /// Combined activity + LBD strategy.
    Combined { activity_threshold: f64, max_lbd: u32 },
}

impl Default for ClauseDeletionStrategy {
    fn default() -> Self {
        ClauseDeletionStrategy::Combined {
            activity_threshold: 1e-8,
            max_lbd: 30,
        }
    }
}

/// Phase selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhasePolicy {
    /// Always pick positive.
    Positive,
    /// Always pick negative.
    Negative,
    /// Save and reuse last phase.
    PhaseSaving,
    /// Random phase selection.
    Random,
}

impl Default for PhasePolicy {
    fn default() -> Self {
        PhasePolicy::PhaseSaving
    }
}

/// Variable selection heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarSelectionHeuristic {
    /// VSIDS (Variable State Independent Decaying Sum).
    Vsids,
    /// Pick the first unassigned variable.
    Sequential,
    /// Random selection.
    Random,
}

impl Default for VarSelectionHeuristic {
    fn default() -> Self {
        VarSelectionHeuristic::Vsids
    }
}

/// Comprehensive solver configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// VSIDS activity decay factor (0.0 – 1.0). Lower decays faster.
    pub vsids_decay: f64,
    /// Initial VSIDS activity increment.
    pub vsids_increment: f64,
    /// Restart strategy.
    pub restart_strategy: RestartStrategy,
    /// Clause deletion strategy.
    pub clause_deletion: ClauseDeletionStrategy,
    /// Maximum number of learned clauses before GC triggers.
    pub max_learned_clauses: usize,
    /// Growth factor for max_learned_clauses after each GC.
    pub learned_clause_growth: f64,
    /// Phase selection policy.
    pub phase_policy: PhasePolicy,
    /// Variable selection heuristic.
    pub var_selection: VarSelectionHeuristic,
    /// Theory propagation frequency (check every N propagations).
    pub theory_propagation_freq: u32,
    /// Solver timeout.
    pub timeout: Option<Duration>,
    /// Memory limit in bytes.
    pub memory_limit: Option<usize>,
    /// Enable proof logging.
    pub proof_logging: bool,
    /// Random seed.
    pub random_seed: u64,
    /// Clause minimization during conflict analysis.
    pub minimize_learned: bool,
    /// Binary clause optimization.
    pub binary_clause_optimization: bool,
    /// Luby restart unit run.
    pub luby_unit_run: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            vsids_decay: 0.95,
            vsids_increment: 1.0,
            restart_strategy: RestartStrategy::default(),
            clause_deletion: ClauseDeletionStrategy::default(),
            max_learned_clauses: 20_000,
            learned_clause_growth: 1.1,
            phase_policy: PhasePolicy::default(),
            var_selection: VarSelectionHeuristic::default(),
            theory_propagation_freq: 1,
            timeout: None,
            memory_limit: None,
            proof_logging: false,
            random_seed: 42,
            minimize_learned: true,
            binary_clause_optimization: true,
            luby_unit_run: 100,
        }
    }
}

impl SolverConfig {
    /// Configuration tuned for small instances (< 100 variables).
    pub fn small_instance() -> Self {
        Self {
            vsids_decay: 0.90,
            vsids_increment: 1.0,
            restart_strategy: RestartStrategy::Fixed { interval: 50 },
            clause_deletion: ClauseDeletionStrategy::Activity { threshold: 1e-10 },
            max_learned_clauses: 1_000,
            learned_clause_growth: 1.5,
            phase_policy: PhasePolicy::PhaseSaving,
            var_selection: VarSelectionHeuristic::Vsids,
            theory_propagation_freq: 1,
            timeout: Some(Duration::from_secs(10)),
            memory_limit: None,
            proof_logging: false,
            random_seed: 42,
            minimize_learned: true,
            binary_clause_optimization: true,
            luby_unit_run: 50,
        }
    }

    /// Configuration tuned for large instances (> 10k variables).
    pub fn large_instance() -> Self {
        Self {
            vsids_decay: 0.99,
            vsids_increment: 1.0,
            restart_strategy: RestartStrategy::Luby { base_interval: 512 },
            clause_deletion: ClauseDeletionStrategy::Combined {
                activity_threshold: 1e-6,
                max_lbd: 50,
            },
            max_learned_clauses: 200_000,
            learned_clause_growth: 1.05,
            phase_policy: PhasePolicy::PhaseSaving,
            var_selection: VarSelectionHeuristic::Vsids,
            theory_propagation_freq: 5,
            timeout: Some(Duration::from_secs(300)),
            memory_limit: Some(4 * 1024 * 1024 * 1024), // 4 GB
            proof_logging: false,
            random_seed: 42,
            minimize_learned: true,
            binary_clause_optimization: true,
            luby_unit_run: 512,
        }
    }

    /// Configuration for deployment verification (bounded model checking).
    pub fn deployment_verification() -> Self {
        Self {
            vsids_decay: 0.95,
            vsids_increment: 1.0,
            restart_strategy: RestartStrategy::Geometric {
                initial: 100,
                factor: 1.5,
            },
            clause_deletion: ClauseDeletionStrategy::Lbd { max_lbd: 20 },
            max_learned_clauses: 50_000,
            learned_clause_growth: 1.1,
            phase_policy: PhasePolicy::PhaseSaving,
            var_selection: VarSelectionHeuristic::Vsids,
            theory_propagation_freq: 1,
            timeout: Some(Duration::from_secs(60)),
            memory_limit: Some(2 * 1024 * 1024 * 1024), // 2 GB
            proof_logging: true,
            random_seed: 42,
            minimize_learned: true,
            binary_clause_optimization: true,
            luby_unit_run: 100,
        }
    }

    /// Configuration for incremental / online solving.
    pub fn incremental() -> Self {
        Self {
            vsids_decay: 0.95,
            vsids_increment: 1.0,
            restart_strategy: RestartStrategy::Luby { base_interval: 64 },
            clause_deletion: ClauseDeletionStrategy::Activity { threshold: 1e-8 },
            max_learned_clauses: 30_000,
            learned_clause_growth: 1.15,
            phase_policy: PhasePolicy::PhaseSaving,
            var_selection: VarSelectionHeuristic::Vsids,
            theory_propagation_freq: 1,
            timeout: None,
            memory_limit: None,
            proof_logging: false,
            random_seed: 42,
            minimize_learned: true,
            binary_clause_optimization: true,
            luby_unit_run: 64,
        }
    }

    /// Validate the configuration values.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vsids_decay <= 0.0 || self.vsids_decay >= 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "vsids_decay".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        if self.vsids_increment <= 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "vsids_increment".into(),
                reason: "must be positive".into(),
            });
        }
        if self.max_learned_clauses == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_learned_clauses".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.learned_clause_growth < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "learned_clause_growth".into(),
                reason: "must be >= 1.0".into(),
            });
        }
        if let RestartStrategy::Geometric { factor, .. } = self.restart_strategy {
            if factor <= 1.0 {
                return Err(ConfigError::InvalidValue {
                    field: "restart_strategy.geometric.factor".into(),
                    reason: "must be > 1.0".into(),
                });
            }
        }
        Ok(())
    }

    /// Compute the Luby sequence value for the given index.
    pub fn luby_value(i: u64) -> u64 {
        let mut size = 1u64;
        let mut seq = 1u64;
        while size < i + 1 {
            seq = size + 1;
            size = 2 * size + 1;
        }
        let mut i = i;
        while size - 1 != i {
            size = (size - 1) >> 1;
            if i >= size {
                i -= size;
            }
        }
        // At this point seq is no longer correct; re-derive via recursion-style.
        // Simple recursive definition:
        luby_recursive(i + 1)
    }
}

fn luby_recursive(i: u64) -> u64 {
    if i <= 0 {
        return 1;
    }
    // Find k such that 2^k - 1 >= i
    let mut k = 1u32;
    while (1u64 << k) - 1 < i {
        k += 1;
    }
    if (1u64 << k) - 1 == i {
        1u64 << (k - 1)
    } else {
        luby_recursive(i - (1u64 << (k - 1)) + 1)
    }
}

/// Configuration-related errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("invalid config value for `{field}`: {reason}")]
    InvalidValue { field: String, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = SolverConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_small_instance_config() {
        let cfg = SolverConfig::small_instance();
        assert!(cfg.validate().is_ok());
        assert!(cfg.timeout.is_some());
    }

    #[test]
    fn test_large_instance_config() {
        let cfg = SolverConfig::large_instance();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.max_learned_clauses, 200_000);
    }

    #[test]
    fn test_deployment_verification_config() {
        let cfg = SolverConfig::deployment_verification();
        assert!(cfg.validate().is_ok());
        assert!(cfg.proof_logging);
    }

    #[test]
    fn test_incremental_config() {
        let cfg = SolverConfig::incremental();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_invalid_vsids_decay() {
        let mut cfg = SolverConfig::default();
        cfg.vsids_decay = 0.0;
        assert!(cfg.validate().is_err());
        cfg.vsids_decay = 1.0;
        assert!(cfg.validate().is_err());
        cfg.vsids_decay = -0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_vsids_increment() {
        let mut cfg = SolverConfig::default();
        cfg.vsids_increment = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_max_learned() {
        let mut cfg = SolverConfig::default();
        cfg.max_learned_clauses = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_growth_factor() {
        let mut cfg = SolverConfig::default();
        cfg.learned_clause_growth = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_geometric_factor() {
        let mut cfg = SolverConfig::default();
        cfg.restart_strategy = RestartStrategy::Geometric {
            initial: 100,
            factor: 0.5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_luby_sequence() {
        // Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
        let expected = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                SolverConfig::luby_value(i as u64),
                exp,
                "luby({}) expected {} got {}",
                i,
                exp,
                SolverConfig::luby_value(i as u64)
            );
        }
    }

    #[test]
    fn test_restart_strategy_serde() {
        let strategy = RestartStrategy::Luby { base_interval: 100 };
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: RestartStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(strategy, deserialized);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = SolverConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: SolverConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.vsids_decay, deserialized.vsids_decay);
        assert_eq!(cfg.max_learned_clauses, deserialized.max_learned_clauses);
    }
}
