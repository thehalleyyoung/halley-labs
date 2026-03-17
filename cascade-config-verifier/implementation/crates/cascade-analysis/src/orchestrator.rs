//! Orchestrator — drives the two-tier analysis pipeline.
//!
//! Selects and sequences Tier 1 (fast graph) and Tier 2 (deep BMC) analyses
//! based on the chosen [`AnalysisMode`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::tier1::{Tier1Analyzer, Tier1Config, Tier1Result};
use crate::tier2::{Tier2Analyzer, Tier2Config, Tier2Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which combination of analysis tiers to run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisMode {
    /// Run only Tier 1 (fast graph analysis).
    FastOnly,
    /// Run only Tier 2 (deep BMC).
    DeepOnly,
    /// Run Tier 1 first; if risks are found, follow up with Tier 2.
    TwoTier,
    /// Run both tiers unconditionally plus repair synthesis.
    FullWithRepair,
}

impl Default for AnalysisMode {
    fn default() -> Self {
        Self::TwoTier
    }
}

/// Top-level analysis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub mode: AnalysisMode,
    pub tier1: Tier1Config,
    pub tier2: Option<Tier2Config>,
    pub repair_enabled: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            mode: AnalysisMode::TwoTier,
            tier1: Tier1Config::default(),
            tier2: Some(Tier2Config::default()),
            repair_enabled: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Combined analysis output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub tier1_result: Option<Tier1Result>,
    pub tier2_result: Option<Tier2Result>,
    pub total_duration_ms: u64,
    pub mode_used: AnalysisMode,
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Stateless orchestrator that drives the two-tier analysis pipeline.
#[derive(Debug, Clone)]
pub struct AnalysisOrchestrator;

impl AnalysisOrchestrator {
    pub fn new() -> Self {
        Self
    }

    /// Execute the analysis pipeline.
    ///
    /// # Arguments
    /// * `adjacency` – `(src, dst, retry_count, timeout_ms, weight)` tuples.
    /// * `capacities` – per-service capacity map.
    /// * `deadlines`  – per-source deadline map.
    /// * `service_names` – ordered list of all service names.
    /// * `config` – orchestration configuration.
    pub fn run(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        deadlines: &HashMap<String, u64>,
        service_names: &[String],
        config: &AnalysisConfig,
    ) -> AnalysisResult {
        let start = Instant::now();

        match config.mode {
            AnalysisMode::FastOnly => {
                let t1 = self.run_tier1(adjacency, capacities, deadlines, &config.tier1);
                AnalysisResult {
                    tier1_result: Some(t1),
                    tier2_result: None,
                    total_duration_ms: start.elapsed().as_millis() as u64,
                    mode_used: AnalysisMode::FastOnly,
                }
            }
            AnalysisMode::DeepOnly => {
                let t2_cfg = config.tier2.clone().unwrap_or_default();
                let t2 = self.run_tier2(adjacency, capacities, service_names, &t2_cfg);
                AnalysisResult {
                    tier1_result: None,
                    tier2_result: Some(t2),
                    total_duration_ms: start.elapsed().as_millis() as u64,
                    mode_used: AnalysisMode::DeepOnly,
                }
            }
            AnalysisMode::TwoTier => {
                let t1 = self.run_tier1(adjacency, capacities, deadlines, &config.tier1);
                let t2 = if self.should_run_tier2(&t1) {
                    let t2_cfg = config.tier2.clone().unwrap_or_default();
                    Some(self.run_tier2(adjacency, capacities, service_names, &t2_cfg))
                } else {
                    None
                };
                AnalysisResult {
                    tier1_result: Some(t1),
                    tier2_result: t2,
                    total_duration_ms: start.elapsed().as_millis() as u64,
                    mode_used: AnalysisMode::TwoTier,
                }
            }
            AnalysisMode::FullWithRepair => {
                let t1 = self.run_tier1(adjacency, capacities, deadlines, &config.tier1);
                let t2_cfg = config.tier2.clone().unwrap_or_default();
                let t2 = self.run_tier2(adjacency, capacities, service_names, &t2_cfg);
                // Repair synthesis would be invoked here in a full implementation.
                // For now we record the results from both tiers.
                AnalysisResult {
                    tier1_result: Some(t1),
                    tier2_result: Some(t2),
                    total_duration_ms: start.elapsed().as_millis() as u64,
                    mode_used: AnalysisMode::FullWithRepair,
                }
            }
        }
    }

    /// Determine whether Tier 1 findings warrant a Tier 2 deep dive.
    pub fn should_run_tier2(&self, tier1: &Tier1Result) -> bool {
        !tier1.risky_paths.is_empty()
            || !tier1.timeout_violations.is_empty()
            || !tier1.fan_in_risks.is_empty()
    }

    // ----- private helpers -----

    fn run_tier1(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        deadlines: &HashMap<String, u64>,
        config: &Tier1Config,
    ) -> Tier1Result {
        let analyzer = Tier1Analyzer::new();
        analyzer.analyze(adjacency, capacities, deadlines, config)
    }

    fn run_tier2(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        service_names: &[String],
        config: &Tier2Config,
    ) -> Tier2Result {
        let analyzer = Tier2Analyzer::new();
        analyzer.analyze(adjacency, capacities, service_names, config)
    }
}

impl Default for AnalysisOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_adjacency() -> Vec<(String, String, u32, u64, u64)> {
        vec![
            ("A".into(), "B".into(), 3, 1000, 1),
            ("B".into(), "C".into(), 3, 1000, 1),
            ("C".into(), "D".into(), 3, 1000, 1),
        ]
    }

    fn sample_capacities() -> HashMap<String, u64> {
        [("A", 5), ("B", 5), ("C", 5), ("D", 5)]
            .iter()
            .map(|(k, v)| (k.to_string(), *v as u64))
            .collect()
    }

    fn sample_deadlines() -> HashMap<String, u64> {
        HashMap::from([("A".into(), 5000u64)])
    }

    fn sample_names() -> Vec<String> {
        vec!["A".into(), "B".into(), "C".into(), "D".into()]
    }

    #[test]
    fn test_fast_only_mode() {
        let orch = AnalysisOrchestrator::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::FastOnly,
            ..Default::default()
        };
        let result = orch.run(
            &sample_adjacency(),
            &sample_capacities(),
            &sample_deadlines(),
            &sample_names(),
            &config,
        );
        assert!(result.tier1_result.is_some());
        assert!(result.tier2_result.is_none());
        assert_eq!(result.mode_used, AnalysisMode::FastOnly);
    }

    #[test]
    fn test_deep_only_mode() {
        let orch = AnalysisOrchestrator::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::DeepOnly,
            ..Default::default()
        };
        let result = orch.run(
            &sample_adjacency(),
            &sample_capacities(),
            &sample_deadlines(),
            &sample_names(),
            &config,
        );
        assert!(result.tier1_result.is_none());
        assert!(result.tier2_result.is_some());
        assert_eq!(result.mode_used, AnalysisMode::DeepOnly);
    }

    #[test]
    fn test_two_tier_triggers_tier2_when_risks_found() {
        let orch = AnalysisOrchestrator::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::TwoTier,
            tier1: Tier1Config {
                amplification_threshold: 1.0, // very low → will find risks
                ..Default::default()
            },
            tier2: Some(Tier2Config::default()),
            repair_enabled: false,
        };
        let result = orch.run(
            &sample_adjacency(),
            &sample_capacities(),
            &sample_deadlines(),
            &sample_names(),
            &config,
        );
        assert!(result.tier1_result.is_some());
        // Should have found risks and triggered Tier 2.
        assert!(result.tier2_result.is_some());
    }

    #[test]
    fn test_two_tier_skips_tier2_when_no_risks() {
        let adj = vec![("A".into(), "B".into(), 0u32, 100u64, 1u64)];
        let orch = AnalysisOrchestrator::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::TwoTier,
            tier1: Tier1Config {
                amplification_threshold: 999.0,
                timeout_threshold_ms: 999_999,
                ..Default::default()
            },
            tier2: Some(Tier2Config::default()),
            repair_enabled: false,
        };
        let deadlines = HashMap::from([("A".into(), 999_999u64)]);
        let names = vec!["A".into(), "B".into()];
        let result = orch.run(&adj, &sample_capacities(), &deadlines, &names, &config);
        assert!(result.tier1_result.is_some());
        assert!(result.tier2_result.is_none());
    }

    #[test]
    fn test_full_with_repair_mode() {
        let orch = AnalysisOrchestrator::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::FullWithRepair,
            ..Default::default()
        };
        let result = orch.run(
            &sample_adjacency(),
            &sample_capacities(),
            &sample_deadlines(),
            &sample_names(),
            &config,
        );
        assert!(result.tier1_result.is_some());
        assert!(result.tier2_result.is_some());
        assert_eq!(result.mode_used, AnalysisMode::FullWithRepair);
    }

    #[test]
    fn test_should_run_tier2_empty_results() {
        let orch = AnalysisOrchestrator::new();
        let empty = Tier1Result {
            risky_paths: vec![],
            timeout_violations: vec![],
            fan_in_risks: vec![],
            duration_ms: 0,
        };
        assert!(!orch.should_run_tier2(&empty));
    }

    #[test]
    fn test_should_run_tier2_with_risks() {
        let orch = AnalysisOrchestrator::new();
        let result = Tier1Result {
            risky_paths: vec![crate::tier1::AmplificationRisk {
                path: vec!["A".into(), "B".into()],
                amplification_factor: 10.0,
                capacity: 100,
                severity: "high".into(),
            }],
            timeout_violations: vec![],
            fan_in_risks: vec![],
            duration_ms: 0,
        };
        assert!(orch.should_run_tier2(&result));
    }

    #[test]
    fn test_should_run_tier2_with_timeout_violations() {
        let orch = AnalysisOrchestrator::new();
        let result = Tier1Result {
            risky_paths: vec![],
            timeout_violations: vec![crate::tier1::TimeoutViolation {
                path: vec!["A".into(), "B".into()],
                total_timeout_ms: 10_000,
                deadline_ms: 5000,
                excess_ms: 5000,
            }],
            fan_in_risks: vec![],
            duration_ms: 0,
        };
        assert!(orch.should_run_tier2(&result));
    }

    #[test]
    fn test_result_has_duration() {
        let orch = AnalysisOrchestrator::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::FastOnly,
            ..Default::default()
        };
        let result = orch.run(
            &sample_adjacency(),
            &sample_capacities(),
            &sample_deadlines(),
            &sample_names(),
            &config,
        );
        assert!(result.total_duration_ms < 10_000);
    }

    #[test]
    fn test_empty_topology() {
        let orch = AnalysisOrchestrator::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::TwoTier,
            ..Default::default()
        };
        let result = orch.run(&[], &HashMap::new(), &HashMap::new(), &[], &config);
        let t1 = result.tier1_result.unwrap();
        assert!(t1.risky_paths.is_empty());
        assert!(result.tier2_result.is_none());
    }

    #[test]
    fn test_default_config() {
        let cfg = AnalysisConfig::default();
        assert_eq!(cfg.mode, AnalysisMode::TwoTier);
        assert!(!cfg.repair_enabled);
        assert!(cfg.tier2.is_some());
    }
}
