//! # cascade-bmc
//!
//! Bounded model checking for cascade failure analysis.
//!
//! Provides a [`BmcChecker`] that verifies safety properties over bounded
//! execution traces of cascade-prone service topologies.  The checker
//! symbolically unrolls the transition relation up to [`BmcConfig::max_bound`]
//! steps and either proves safety within that bound or returns a concrete
//! [`Counterexample`].

pub mod cb_abstraction;
pub mod checker;
pub mod encoder;
pub mod marco;
pub mod monotonicity;
pub mod solver;
pub mod trace;
pub mod unroller;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configures the bounded model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcConfig {
    /// Maximum unrolling depth.
    pub max_bound: u32,
    /// Wall-clock timeout in milliseconds.
    pub timeout_ms: u64,
    /// Whether to check user-specified invariants at every step.
    pub check_invariants: bool,
}

impl Default for BmcConfig {
    fn default() -> Self {
        Self {
            max_bound: 20,
            timeout_ms: 30_000,
            check_invariants: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Trace / Counterexample types
// ---------------------------------------------------------------------------

/// A single step recorded during symbolic execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcStep {
    /// Zero-based index within the trace.
    pub step_index: u32,
    /// Service-level state snapshot (metric name -> value).
    pub state: HashMap<String, f64>,
    /// The action that was taken (e.g. "retry", "timeout", "propagate").
    pub action: String,
    /// The service that performed the action.
    pub service: String,
}

/// A concrete counterexample produced when a property violation is found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterexample {
    /// Ordered sequence of steps leading to the violation.
    pub steps: Vec<BmcStep>,
    /// Human-readable description of the violated property.
    pub violated_property: String,
    /// Depth at which the violation was detected.
    pub depth: u32,
}

/// The outcome of a bounded model-checking run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BmcResult {
    /// No violation found within the explored bound.
    Safe { bound_reached: u32 },
    /// A property violation was found.
    Unsafe { counterexample: Counterexample },
    /// Analysis could not determine safety.
    Unknown { reason: String },
    /// The timeout expired before the analysis completed.
    Timeout,
}

// ---------------------------------------------------------------------------
// Checker
// ---------------------------------------------------------------------------

/// Bounded model checker for cascade failure properties.
#[derive(Debug, Clone)]
pub struct BmcChecker {
    pub config: BmcConfig,
}

impl BmcChecker {
    /// Create a new checker with the given configuration.
    pub fn new(config: BmcConfig) -> Self {
        Self { config }
    }

    /// Check an arbitrary string-encoded property against an initial state by
    /// symbolically stepping the system up to `max_bound` times.
    ///
    /// The property string is interpreted as a threshold comparison of the form
    /// `"<metric> < <value>"` (e.g. `"load < 100.0"`).  At each step every
    /// numeric state value is scaled by a small amplification factor to
    /// simulate cascade propagation.  If the named metric exceeds the
    /// threshold, the run is `Unsafe`.
    pub fn check_property(
        &self,
        initial_state: &HashMap<String, f64>,
        property: &str,
    ) -> BmcResult {
        let start = std::time::Instant::now();

        let (metric, threshold) = match Self::parse_property(property) {
            Some(v) => v,
            None => {
                return BmcResult::Unknown {
                    reason: format!("cannot parse property: {property}"),
                }
            }
        };

        let amplification = 1.15; // per-step growth factor
        let mut state = initial_state.clone();
        let mut trace: Vec<BmcStep> = Vec::new();

        for step in 0..self.config.max_bound {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                return BmcResult::Timeout;
            }

            // Record current state
            trace.push(BmcStep {
                step_index: step,
                state: state.clone(),
                action: if step == 0 {
                    "init".into()
                } else {
                    "propagate".into()
                },
                service: format!("svc-{step}"),
            });

            // Check invariant
            if self.config.check_invariants {
                if let Some(&val) = state.get(&metric) {
                    if val >= threshold {
                        return BmcResult::Unsafe {
                            counterexample: Counterexample {
                                steps: trace,
                                violated_property: property.to_string(),
                                depth: step,
                            },
                        };
                    }
                }
            }

            // Transition: amplify every numeric value
            for val in state.values_mut() {
                *val *= amplification;
            }
        }

        BmcResult::Safe {
            bound_reached: self.config.max_bound,
        }
    }

    /// Domain-specific check: determine whether a cascade amplification factor
    /// exceeds `threshold` within `depth` hops.
    ///
    /// Models the compounding effect as `amplification^depth` and returns
    /// `Unsafe` with a witness trace when the compounded value breaches the
    /// threshold.
    pub fn check_cascade_bound(
        &self,
        amplification: f64,
        threshold: f64,
        depth: u32,
    ) -> BmcResult {
        let effective_depth = depth.min(self.config.max_bound);
        let mut current = 1.0_f64;
        let mut trace: Vec<BmcStep> = Vec::new();

        for step in 0..effective_depth {
            current *= amplification;

            let mut state = HashMap::new();
            state.insert("amplification".into(), current);
            state.insert("threshold".into(), threshold);

            trace.push(BmcStep {
                step_index: step,
                state: state.clone(),
                action: "cascade_step".into(),
                service: format!("hop-{step}"),
            });

            if current > threshold {
                return BmcResult::Unsafe {
                    counterexample: Counterexample {
                        steps: trace,
                        violated_property: format!(
                            "cascade amplification {current:.4} > threshold {threshold}"
                        ),
                        depth: step,
                    },
                };
            }
        }

        BmcResult::Safe {
            bound_reached: effective_depth,
        }
    }

    /// Check whether a proposed repair (reducing the amplification factor from
    /// `original_amp` to `repaired_amp`) eliminates all cascade violations at
    /// the configured bound depth.
    pub fn verify_repair_eliminates_cascade(
        &self,
        original_amp: f64,
        repaired_amp: f64,
        threshold: f64,
    ) -> bool {
        // The original must actually be unsafe
        let orig = self.check_cascade_bound(original_amp, threshold, self.config.max_bound);
        let is_originally_unsafe = matches!(orig, BmcResult::Unsafe { .. });

        // The repaired version must be safe
        let repaired = self.check_cascade_bound(repaired_amp, threshold, self.config.max_bound);
        let is_repaired_safe = matches!(repaired, BmcResult::Safe { .. });

        is_originally_unsafe && is_repaired_safe
    }

    /// Attempt to generate a counterexample from an arbitrary state snapshot.
    ///
    /// If any value in `state` exceeds an internal danger threshold (1000.0)
    /// a single-step counterexample is returned.  Otherwise the checker runs
    /// `check_property` with a generic load bound and returns the resulting
    /// counterexample, if any.
    pub fn generate_counterexample(
        &self,
        state: &HashMap<String, f64>,
    ) -> Option<Counterexample> {
        const DANGER_THRESHOLD: f64 = 1000.0;

        // Fast path: immediate violation
        for (key, &val) in state {
            if val > DANGER_THRESHOLD {
                let step = BmcStep {
                    step_index: 0,
                    state: state.clone(),
                    action: "immediate_violation".into(),
                    service: key.clone(),
                };
                return Some(Counterexample {
                    steps: vec![step],
                    violated_property: format!("{key} = {val} exceeds danger threshold"),
                    depth: 0,
                });
            }
        }

        // Slow path: simulate forward with a default load property
        if let Some(max_key) = state
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
        {
            let property = format!("{max_key} < {DANGER_THRESHOLD}");
            match self.check_property(state, &property) {
                BmcResult::Unsafe { counterexample } => Some(counterexample),
                _ => None,
            }
        } else {
            None
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Parse `"metric < value"` into (metric, value).
    fn parse_property(property: &str) -> Option<(String, f64)> {
        let parts: Vec<&str> = property.split('<').collect();
        if parts.len() != 2 {
            return None;
        }
        let metric = parts[0].trim().to_string();
        let threshold: f64 = parts[1].trim().parse().ok()?;
        Some((metric, threshold))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_checker() -> BmcChecker {
        BmcChecker::new(BmcConfig::default())
    }

    fn simple_state() -> HashMap<String, f64> {
        let mut s = HashMap::new();
        s.insert("load".into(), 10.0);
        s.insert("latency".into(), 5.0);
        s
    }

    #[test]
    fn test_default_config() {
        let cfg = BmcConfig::default();
        assert_eq!(cfg.max_bound, 20);
        assert_eq!(cfg.timeout_ms, 30_000);
        assert!(cfg.check_invariants);
    }

    #[test]
    fn test_check_property_safe() {
        let checker = default_checker();
        let state = simple_state();
        let result = checker.check_property(&state, "load < 100000.0");
        assert!(matches!(result, BmcResult::Safe { .. }));
    }

    #[test]
    fn test_check_property_unsafe() {
        let checker = BmcChecker::new(BmcConfig {
            max_bound: 100,
            timeout_ms: 5_000,
            check_invariants: true,
        });
        let state = simple_state();
        // 10 * 1.15^n will exceed 50 around step ~12
        let result = checker.check_property(&state, "load < 50.0");
        assert!(matches!(result, BmcResult::Unsafe { .. }));
        if let BmcResult::Unsafe { counterexample } = result {
            assert!(!counterexample.steps.is_empty());
            assert_eq!(counterexample.violated_property, "load < 50.0");
        }
    }

    #[test]
    fn test_check_property_invalid() {
        let checker = default_checker();
        let state = simple_state();
        let result = checker.check_property(&state, "bad property");
        assert!(matches!(result, BmcResult::Unknown { .. }));
    }

    #[test]
    fn test_cascade_bound_safe() {
        let checker = default_checker();
        // 1.05^20 ~ 2.65, well under threshold 10
        let result = checker.check_cascade_bound(1.05, 10.0, 20);
        assert!(matches!(result, BmcResult::Safe { .. }));
    }

    #[test]
    fn test_cascade_bound_unsafe() {
        let checker = default_checker();
        // 2.0^5 = 32, exceeds threshold 10
        let result = checker.check_cascade_bound(2.0, 10.0, 5);
        assert!(matches!(result, BmcResult::Unsafe { .. }));
        if let BmcResult::Unsafe { counterexample } = result {
            assert!(counterexample.depth < 5);
        }
    }

    #[test]
    fn test_verify_repair_eliminates_cascade() {
        let checker = BmcChecker::new(BmcConfig {
            max_bound: 10,
            timeout_ms: 5_000,
            check_invariants: true,
        });
        // original 2.0^10 = 1024 > 100, repaired 1.05^10 ~ 1.63 < 100
        assert!(checker.verify_repair_eliminates_cascade(2.0, 1.05, 100.0));
    }

    #[test]
    fn test_verify_repair_fails_when_still_unsafe() {
        let checker = BmcChecker::new(BmcConfig {
            max_bound: 10,
            timeout_ms: 5_000,
            check_invariants: true,
        });
        // both original and repaired exceed threshold
        assert!(!checker.verify_repair_eliminates_cascade(2.0, 1.8, 5.0));
    }

    #[test]
    fn test_generate_counterexample_immediate() {
        let checker = default_checker();
        let mut state = HashMap::new();
        state.insert("cpu".into(), 5000.0);
        let cex = checker.generate_counterexample(&state);
        assert!(cex.is_some());
        assert_eq!(cex.unwrap().depth, 0);
    }

    #[test]
    fn test_generate_counterexample_none() {
        let checker = BmcChecker::new(BmcConfig {
            max_bound: 5,
            timeout_ms: 5_000,
            check_invariants: true,
        });
        let mut state = HashMap::new();
        state.insert("cpu".into(), 1.0);
        // 1.0 * 1.15^5 ~ 2.01 -- well under 1000
        let cex = checker.generate_counterexample(&state);
        assert!(cex.is_none());
    }

    #[test]
    fn test_bmc_result_serialization() {
        let result = BmcResult::Safe { bound_reached: 10 };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Safe"));
        let deser: BmcResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(deser, BmcResult::Safe { bound_reached: 10 }));
    }

    #[test]
    fn test_counterexample_serialization() {
        let cex = Counterexample {
            steps: vec![BmcStep {
                step_index: 0,
                state: HashMap::new(),
                action: "init".into(),
                service: "svc-0".into(),
            }],
            violated_property: "test".into(),
            depth: 0,
        };
        let json = serde_json::to_string(&cex).unwrap();
        let deser: Counterexample = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.violated_property, "test");
        assert_eq!(deser.steps.len(), 1);
    }
}
