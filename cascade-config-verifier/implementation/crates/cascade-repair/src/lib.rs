//! # cascade-repair
//!
//! Repair synthesis for cascade failures in the CascadeVerify project.
//!
//! This crate provides algorithms for synthesizing minimal configuration
//! repairs that fix retry-amplification and timeout-chain violations
//! detected by the CascadeVerify analysis pipeline.

pub mod config_diff;
pub mod constraint_builder;
pub mod explanation;
pub mod strategy;
pub mod synthesizer;
pub mod validator;

use cascade_graph::RtigGraph as ServiceGraph;
use serde::{Deserialize, Serialize};

pub use config_diff::{ConfigDiff, ConfigDiffGenerator, DiffHunk, DiffLine, FileChange};
pub use constraint_builder::{
    AmplificationConstraint, BoundConstraint, ConstraintBuilder, ConsistencyConstraint,
    DeviationObjective, RepairVariable, RepairWeights, TimeoutConstraint,
};
pub use explanation::{Counterfactual, Explanation, ExplanationDetail, ExplanationGenerator};
pub use strategy::{
    compare_strategies, GreedyStrategy, MinimalChangeStrategy, MinimalDeviationStrategy,
    NaiveDefaultStrategy, RepairStrategy, StrategyComparison, UniformReductionStrategy,
};
pub use synthesizer::{
    ParameterBounds, RepairObjective, RepairResult, RepairStatistics, RiskyPathInfo,
};
pub use validator::{RepairValidator, ValidationIssue, ValidationResult};

// ---------------------------------------------------------------------------
// Core types (kept from the original lib.rs)
// ---------------------------------------------------------------------------

/// Configuration for repair synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairConfig {
    pub max_retry_reduction: u32,
    pub max_timeout_adjustment_ms: u64,
    pub budget: f64,
    pub preserve_functionality: bool,
    pub max_changes: usize,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_retry_reduction: 3,
            max_timeout_adjustment_ms: 10_000,
            budget: 100.0,
            preserve_functionality: true,
            max_changes: 20,
        }
    }
}

/// Type of repair action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepairActionType {
    ReduceRetries { from: u32, to: u32 },
    AdjustTimeout { from_ms: u64, to_ms: u64 },
    AddCircuitBreaker { threshold: u32 },
    AddRateLimit { rps: f64 },
    IncreaseCapacity { from: f64, to: f64 },
}

/// A single repair action targeting one edge parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairAction {
    pub service: String,
    pub edge: Option<(String, String)>,
    pub action_type: RepairActionType,
    pub description: String,
    pub deviation: f64,
}

impl RepairAction {
    pub fn reduce_retries(source: &str, target: &str, from: u32, to: u32) -> Self {
        Self {
            service: source.to_string(),
            edge: Some((source.to_string(), target.to_string())),
            action_type: RepairActionType::ReduceRetries { from, to },
            description: format!(
                "Reduce retries on {}->{} from {} to {}",
                source, target, from, to
            ),
            deviation: (from - to) as f64,
        }
    }

    pub fn adjust_timeout(source: &str, target: &str, from_ms: u64, to_ms: u64) -> Self {
        Self {
            service: source.to_string(),
            edge: Some((source.to_string(), target.to_string())),
            action_type: RepairActionType::AdjustTimeout { from_ms, to_ms },
            description: format!(
                "Adjust timeout on {}->{} from {}ms to {}ms",
                source, target, from_ms, to_ms
            ),
            deviation: (from_ms as f64 - to_ms as f64).abs() / from_ms.max(1) as f64,
        }
    }
}

/// A complete repair plan.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepairPlan {
    pub actions: Vec<RepairAction>,
    pub total_deviation: f64,
    pub affected_services: Vec<String>,
    pub feasible: bool,
}

impl RepairPlan {
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn add_action(&mut self, action: RepairAction) {
        if !self.affected_services.contains(&action.service) {
            self.affected_services.push(action.service.clone());
        }
        self.total_deviation += action.deviation;
        self.actions.push(action);
    }

    /// Alias for `total_deviation` used by the synthesizer module.
    pub fn total_cost(&self) -> f64 {
        self.total_deviation
    }
}

/// Graph-aware repair synthesizer that operates on [`ServiceGraph`] directly.
#[derive(Debug)]
pub struct GraphRepairSynthesizer;

impl GraphRepairSynthesizer {
    pub fn new() -> Self {
        Self
    }

    /// Synthesize a repair plan for the given graph and failure sets.
    pub fn synthesize(
        &self,
        graph: &ServiceGraph,
        failure_services: &[Vec<String>],
        config: &RepairConfig,
    ) -> RepairPlan {
        let mut plan = RepairPlan { feasible: true, ..Default::default() };
        for failure_set in failure_services {
            for svc_id in failure_set {
                let incoming = graph.incoming_edges(svc_id);
                for edge in incoming {
                    if edge.retry_count > 0 && plan.actions.len() < config.max_changes {
                        let new_retries =
                            edge.retry_count.saturating_sub(config.max_retry_reduction);
                        let action = RepairAction::reduce_retries(
                            edge.source.as_str(),
                            edge.target.as_str(),
                            edge.retry_count,
                            new_retries,
                        );
                        if plan.total_deviation + action.deviation <= config.budget {
                            plan.add_action(action);
                        }
                    }
                }
            }
        }
        plan
    }
}

impl Default for GraphRepairSynthesizer {
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

    // -- RepairConfig -------------------------------------------------------

    #[test]
    fn test_repair_config_default() {
        let cfg = RepairConfig::default();
        assert_eq!(cfg.max_retry_reduction, 3);
        assert_eq!(cfg.max_timeout_adjustment_ms, 10_000);
        assert_eq!(cfg.budget, 100.0);
        assert!(cfg.preserve_functionality);
        assert_eq!(cfg.max_changes, 20);
    }

    #[test]
    fn test_repair_config_serde_roundtrip() {
        let cfg = RepairConfig {
            max_retry_reduction: 5,
            max_timeout_adjustment_ms: 20_000,
            budget: 50.0,
            preserve_functionality: false,
            max_changes: 10,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let deser: RepairConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.max_retry_reduction, 5);
        assert_eq!(deser.budget, 50.0);
        assert!(!deser.preserve_functionality);
    }

    // -- RepairActionType ---------------------------------------------------

    #[test]
    fn test_action_type_reduce_retries_serde() {
        let at = RepairActionType::ReduceRetries { from: 5, to: 2 };
        let json = serde_json::to_string(&at).unwrap();
        assert!(json.contains("ReduceRetries"));
        let deser: RepairActionType = serde_json::from_str(&json).unwrap();
        if let RepairActionType::ReduceRetries { from, to } = deser {
            assert_eq!(from, 5);
            assert_eq!(to, 2);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_action_type_adjust_timeout_serde() {
        let at = RepairActionType::AdjustTimeout {
            from_ms: 30000,
            to_ms: 15000,
        };
        let json = serde_json::to_string(&at).unwrap();
        let deser: RepairActionType = serde_json::from_str(&json).unwrap();
        if let RepairActionType::AdjustTimeout { from_ms, to_ms } = deser {
            assert_eq!(from_ms, 30000);
            assert_eq!(to_ms, 15000);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_action_type_add_circuit_breaker_serde() {
        let at = RepairActionType::AddCircuitBreaker { threshold: 5 };
        let json = serde_json::to_string(&at).unwrap();
        let deser: RepairActionType = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            deser,
            RepairActionType::AddCircuitBreaker { threshold: 5 }
        ));
    }

    #[test]
    fn test_action_type_add_rate_limit_serde() {
        let at = RepairActionType::AddRateLimit { rps: 100.5 };
        let json = serde_json::to_string(&at).unwrap();
        let deser: RepairActionType = serde_json::from_str(&json).unwrap();
        if let RepairActionType::AddRateLimit { rps } = deser {
            assert!((rps - 100.5).abs() < f64::EPSILON);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_action_type_increase_capacity_serde() {
        let at = RepairActionType::IncreaseCapacity {
            from: 100.0,
            to: 200.0,
        };
        let json = serde_json::to_string(&at).unwrap();
        let deser: RepairActionType = serde_json::from_str(&json).unwrap();
        if let RepairActionType::IncreaseCapacity { from, to } = deser {
            assert_eq!(from, 100.0);
            assert_eq!(to, 200.0);
        } else {
            panic!("wrong variant");
        }
    }

    // -- RepairAction -------------------------------------------------------

    #[test]
    fn test_reduce_retries_action() {
        let action = RepairAction::reduce_retries("frontend", "api", 5, 2);
        assert_eq!(action.service, "frontend");
        assert_eq!(
            action.edge,
            Some(("frontend".to_string(), "api".to_string()))
        );
        assert_eq!(action.deviation, 3.0);
        assert!(action.description.contains("frontend"));
        assert!(action.description.contains("api"));
        assert!(action.description.contains("5"));
        assert!(action.description.contains("2"));
    }

    #[test]
    fn test_reduce_retries_to_zero() {
        let action = RepairAction::reduce_retries("a", "b", 3, 0);
        assert_eq!(action.deviation, 3.0);
    }

    #[test]
    fn test_adjust_timeout_action() {
        let action = RepairAction::adjust_timeout("gateway", "backend", 30000, 15000);
        assert_eq!(action.service, "gateway");
        assert!(action.description.contains("30000ms"));
        assert!(action.description.contains("15000ms"));
        // deviation = |30000 - 15000| / 30000 = 0.5
        assert!((action.deviation - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adjust_timeout_increase() {
        let action = RepairAction::adjust_timeout("a", "b", 5000, 10000);
        assert_eq!(action.service, "a");
        // deviation = |5000 - 10000| / 5000 = 1.0
        assert!((action.deviation - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_repair_action_serde_roundtrip() {
        let action = RepairAction::reduce_retries("x", "y", 4, 1);
        let json = serde_json::to_string(&action).unwrap();
        let deser: RepairAction = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.service, "x");
        assert_eq!(deser.deviation, 3.0);
    }

    // -- RepairPlan ---------------------------------------------------------

    #[test]
    fn test_repair_plan_default() {
        let plan = RepairPlan::default();
        assert!(plan.is_empty());
        assert!(!plan.feasible);
        assert_eq!(plan.total_cost(), 0.0);
        assert!(plan.affected_services.is_empty());
    }

    #[test]
    fn test_repair_plan_add_action() {
        let mut plan = RepairPlan::default();
        plan.add_action(RepairAction::reduce_retries("a", "b", 5, 2));
        assert!(!plan.is_empty());
        assert_eq!(plan.actions.len(), 1);
        assert_eq!(plan.total_deviation, 3.0);
        assert_eq!(plan.affected_services, vec!["a".to_string()]);
    }

    #[test]
    fn test_repair_plan_multiple_actions_same_service() {
        let mut plan = RepairPlan::default();
        plan.add_action(RepairAction::reduce_retries("a", "b", 5, 2));
        plan.add_action(RepairAction::reduce_retries("a", "c", 3, 1));
        assert_eq!(plan.actions.len(), 2);
        // "a" should appear only once in affected_services
        assert_eq!(plan.affected_services, vec!["a".to_string()]);
        assert_eq!(plan.total_deviation, 5.0);
    }

    #[test]
    fn test_repair_plan_multiple_services() {
        let mut plan = RepairPlan::default();
        plan.add_action(RepairAction::reduce_retries("a", "b", 3, 1));
        plan.add_action(RepairAction::reduce_retries("c", "d", 4, 2));
        assert_eq!(plan.affected_services.len(), 2);
        assert!(plan.affected_services.contains(&"a".to_string()));
        assert!(plan.affected_services.contains(&"c".to_string()));
    }

    #[test]
    fn test_repair_plan_total_cost() {
        let mut plan = RepairPlan::default();
        plan.add_action(RepairAction::reduce_retries("a", "b", 5, 3));
        plan.add_action(RepairAction::adjust_timeout("c", "d", 10000, 5000));
        assert_eq!(plan.total_cost(), plan.total_deviation);
    }

    #[test]
    fn test_repair_plan_serde_roundtrip() {
        let mut plan = RepairPlan {
            feasible: true,
            ..Default::default()
        };
        plan.add_action(RepairAction::reduce_retries("a", "b", 5, 2));
        let json = serde_json::to_string(&plan).unwrap();
        let deser: RepairPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.actions.len(), 1);
        assert!(deser.feasible);
        assert_eq!(deser.total_deviation, 3.0);
    }

    // -- GraphRepairSynthesizer ---------------------------------------------

    #[test]
    fn test_synthesizer_default() {
        let s = GraphRepairSynthesizer::default();
        let _ = format!("{:?}", s);
    }

    #[test]
    fn test_synthesizer_new() {
        let s = GraphRepairSynthesizer::new();
        let _ = format!("{:?}", s);
    }

    #[test]
    fn test_synthesizer_empty_graph() {
        let synth = GraphRepairSynthesizer::new();
        let graph = ServiceGraph::new();
        let config = RepairConfig::default();
        let plan = synth.synthesize(&graph, &[], &config);
        assert!(plan.is_empty());
        assert!(plan.feasible);
    }

    #[test]
    fn test_synthesizer_no_failures() {
        use cascade_graph::rtig::build_chain;
        let synth = GraphRepairSynthesizer::new();
        let graph = build_chain(&["A", "B", "C"], 3);
        let config = RepairConfig::default();
        let plan = synth.synthesize(&graph, &[], &config);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_synthesizer_single_failure() {
        use cascade_graph::rtig::build_chain;
        let synth = GraphRepairSynthesizer::new();
        let graph = build_chain(&["A", "B", "C"], 3);
        let config = RepairConfig::default();
        let failures = vec![vec!["B".to_string()]];
        let plan = synth.synthesize(&graph, &failures, &config);
        assert!(plan.feasible);
        // Should reduce retries on A->B edge
        assert!(!plan.is_empty());
    }

    #[test]
    fn test_synthesizer_respects_max_changes() {
        use cascade_graph::rtig::build_chain;
        let synth = GraphRepairSynthesizer::new();
        let graph = build_chain(&["A", "B", "C", "D", "E"], 5);
        let config = RepairConfig {
            max_changes: 1,
            ..Default::default()
        };
        let failures = vec![vec![
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
        ]];
        let plan = synth.synthesize(&graph, &failures, &config);
        assert!(plan.actions.len() <= 1);
    }

    #[test]
    fn test_synthesizer_respects_budget() {
        use cascade_graph::rtig::build_chain;
        let synth = GraphRepairSynthesizer::new();
        let graph = build_chain(&["A", "B", "C"], 3);
        let config = RepairConfig {
            budget: 0.5,
            ..Default::default()
        };
        let failures = vec![vec!["B".to_string(), "C".to_string()]];
        let plan = synth.synthesize(&graph, &failures, &config);
        assert!(plan.total_deviation <= config.budget);
    }
}
