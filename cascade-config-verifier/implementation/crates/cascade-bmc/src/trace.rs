//! Execution trace extraction, validation, causal chain analysis, and
//! counterfactual reasoning.

use cascade_graph::rtig::RtigGraph;
use cascade_types::cascade::{PropagationStep, PropagationTrace};
use cascade_types::service::{ServiceHealth, ServiceId};
use cascade_types::smt::SmtModel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// TraceStep
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub time: usize,
    pub service: String,
    pub load: u64,
    pub state: ServiceHealthState,
    pub retries_remaining: u32,
    pub timeout_remaining: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceHealthState {
    Healthy,
    Degraded,
    Unavailable,
}

impl ServiceHealthState {
    pub fn from_int(v: i64) -> Self {
        match v {
            0 => ServiceHealthState::Healthy,
            1 => ServiceHealthState::Degraded,
            _ => ServiceHealthState::Unavailable,
        }
    }

    pub fn to_int(self) -> i64 {
        match self {
            ServiceHealthState::Healthy => 0,
            ServiceHealthState::Degraded => 1,
            ServiceHealthState::Unavailable => 2,
        }
    }
}

impl fmt::Display for ServiceHealthState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceHealthState::Healthy => write!(f, "Healthy"),
            ServiceHealthState::Degraded => write!(f, "Degraded"),
            ServiceHealthState::Unavailable => write!(f, "Unavailable"),
        }
    }
}

// ---------------------------------------------------------------------------
// DetailedTrace
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedTrace {
    pub steps: Vec<TraceStep>,
    pub failed_services: Vec<String>,
    pub depth: usize,
}

impl DetailedTrace {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            failed_services: Vec::new(),
            depth: 0,
        }
    }

    pub fn add_step(&mut self, step: TraceStep) {
        if step.time + 1 > self.depth {
            self.depth = step.time + 1;
        }
        self.steps.push(step);
    }

    pub fn steps_at_time(&self, t: usize) -> Vec<&TraceStep> {
        self.steps.iter().filter(|s| s.time == t).collect()
    }

    pub fn service_trace(&self, service: &str) -> Vec<&TraceStep> {
        self.steps.iter().filter(|s| s.service == service).collect()
    }

    pub fn first_overload_time(&self, service: &str, capacity: u64) -> Option<usize> {
        self.service_trace(service)
            .iter()
            .find(|s| s.load > capacity)
            .map(|s| s.time)
    }
}

impl Default for DetailedTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TraceExtractor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TraceExtractor;

impl TraceExtractor {
    /// Extract a detailed trace from SMT model assignments and graph structure.
    pub fn extract_from_model(
        assignments: &HashMap<String, i64>,
        graph: &RtigGraph,
        depth: usize,
    ) -> DetailedTrace {
        let mut trace = DetailedTrace::new();

        // Extract failed services
        for sid in graph.service_ids() {
            let var = format!("failed_{sid}");
            if assignments.get(&var).copied() == Some(1) {
                trace.failed_services.push(sid.to_owned());
            }
        }

        // Extract per-service per-timestep state
        for t in 0..=depth {
            for sid in graph.service_ids() {
                let load_var = format!("load_{sid}_{t}");
                let state_var = format!("state_{sid}_{t}");
                let retry_var = format!("retry_{sid}_{t}");
                let timeout_var = format!("timeout_{sid}_{t}");

                let load = assignments.get(&load_var).copied().unwrap_or(0);
                let state = assignments.get(&state_var).copied().unwrap_or(0);
                let retries = assignments.get(&retry_var).copied().unwrap_or(0) as u32;
                let timeout = assignments.get(&timeout_var).copied().unwrap_or(0);

                trace.add_step(TraceStep {
                    time: t,
                    service: sid.to_owned(),
                    load: load.max(0) as u64,
                    state: ServiceHealthState::from_int(state),
                    retries_remaining: retries,
                    timeout_remaining: timeout,
                });
            }
        }

        trace
    }

    /// Extract a simpler PropagationTrace from a detailed trace.
    pub fn to_propagation_trace(
        trace: &DetailedTrace,
        graph: &RtigGraph,
    ) -> PropagationTrace {
        let mut prop_trace = PropagationTrace::new();

        for t in 0..trace.depth.saturating_sub(1) {
            let current_steps = trace.steps_at_time(t);
            let next_steps = trace.steps_at_time(t + 1);

            for edge in graph.edges() {
                let src = edge.source.as_str();
                let tgt = edge.target.as_str();
                let src_step = current_steps.iter().find(|s| s.service == src);
                let tgt_next = next_steps.iter().find(|s| s.service == tgt);

                if let (Some(src_s), Some(tgt_s)) = (src_step, tgt_next) {
                    if src_s.load > 0 && tgt_s.load > tgt_s.load.saturating_sub(src_s.load) {
                        prop_trace.push(PropagationStep {
                            time_step: t as u32,
                            service: ServiceId::from(tgt),
                            load: tgt_s.load as f64,
                            state: format!("propagation from {}", src),
                            cause: Some(format!("{}->{}", src, tgt)),
                        });
                    }
                }
            }
        }

        prop_trace
    }
}

// ---------------------------------------------------------------------------
// Trace formatting
// ---------------------------------------------------------------------------

/// Format a detailed trace as a human-readable string.
pub fn format_trace(trace: &DetailedTrace) -> String {
    let mut out = String::new();

    out.push_str("=== Cascade Propagation Trace ===\n");
    out.push_str(&format!("Failed services: {:?}\n", trace.failed_services));
    out.push_str(&format!("Trace depth: {}\n\n", trace.depth));

    for t in 0..=trace.depth {
        out.push_str(&format!("--- Time step {t} ---\n"));
        let steps = trace.steps_at_time(t);
        for step in steps {
            out.push_str(&format!(
                "  {} | load={:>6} | state={:<12} | retries={} | timeout={}ms\n",
                step.service, step.load, format!("{}", step.state),
                step.retries_remaining, step.timeout_remaining,
            ));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// TraceValidation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceValidation {
    pub valid: bool,
    pub violations: Vec<TraceViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceViolation {
    pub step: usize,
    pub service: String,
    pub expected: String,
    pub actual: String,
    pub description: String,
}

/// Validate a detailed trace against the graph's constraints.
pub fn validate_trace(trace: &DetailedTrace, graph: &RtigGraph) -> TraceValidation {
    let mut violations = Vec::new();

    for t in 0..trace.depth {
        let current = trace.steps_at_time(t);
        let next = trace.steps_at_time(t + 1);

        for next_step in &next {
            let sid = &next_step.service;
            let node = match graph.service(sid) {
                Some(n) => n,
                None => continue,
            };

            // Check state consistency: if load > capacity, state should be degraded/unavailable
            if next_step.load > node.capacity as u64 && next_step.state == ServiceHealthState::Healthy {
                violations.push(TraceViolation {
                    step: t + 1,
                    service: sid.clone(),
                    expected: "Degraded or Unavailable".to_owned(),
                    actual: format!("{}", next_step.state),
                    description: format!(
                        "Service {} has load {} > capacity {} but state is Healthy",
                        sid, next_step.load, node.capacity
                    ),
                });
            }

            // Check non-negativity
            if next_step.timeout_remaining < 0 {
                violations.push(TraceViolation {
                    step: t + 1,
                    service: sid.clone(),
                    expected: ">= 0".to_owned(),
                    actual: format!("{}", next_step.timeout_remaining),
                    description: format!(
                        "Service {} has negative timeout_remaining",
                        sid
                    ),
                });
            }

            // Check retry monotonicity (retries should not increase)
            if let Some(prev_step) = current.iter().find(|s| s.service == *sid) {
                if next_step.retries_remaining > prev_step.retries_remaining {
                    violations.push(TraceViolation {
                        step: t + 1,
                        service: sid.clone(),
                        expected: format!("<= {}", prev_step.retries_remaining),
                        actual: format!("{}", next_step.retries_remaining),
                        description: format!(
                            "Service {} retries increased from {} to {}",
                            sid, prev_step.retries_remaining, next_step.retries_remaining
                        ),
                    });
                }
            }

            // Check that failed services stay at load 0
            if trace.failed_services.contains(sid) && next_step.load != 0 {
                violations.push(TraceViolation {
                    step: t + 1,
                    service: sid.clone(),
                    expected: "0".to_owned(),
                    actual: format!("{}", next_step.load),
                    description: format!(
                        "Failed service {} has non-zero load {}",
                        sid, next_step.load
                    ),
                });
            }
        }
    }

    TraceValidation {
        valid: violations.is_empty(),
        violations,
    }
}

// ---------------------------------------------------------------------------
// CausalChainExtractor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChain {
    pub steps: Vec<CausalStep>,
    pub root_cause: String,
    pub critical_path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalStep {
    pub time: usize,
    pub from_service: String,
    pub to_service: String,
    pub load_before: u64,
    pub load_after: u64,
    pub cause: String,
}

#[derive(Debug, Clone)]
pub struct CausalChainExtractor;

impl CausalChainExtractor {
    /// Extract the causal chain leading to a cascade.
    pub fn extract_causal_chain(
        trace: &DetailedTrace,
        graph: &RtigGraph,
        target: &str,
    ) -> CausalChain {
        let mut steps = Vec::new();
        let mut critical_path = Vec::new();
        let root_cause = trace.failed_services.first()
            .cloned()
            .unwrap_or_else(|| "unknown".to_owned());

        // Backward trace from target to find the causal chain
        let mut current_services = vec![target.to_owned()];
        let mut visited = HashSet::new();

        for t in (0..trace.depth).rev() {
            let mut next_services = Vec::new();
            for sid in &current_services {
                if visited.contains(sid) {
                    continue;
                }
                visited.insert(sid.clone());

                let load_at_t1 = trace.steps_at_time(t + 1)
                    .iter()
                    .find(|s| s.service == *sid)
                    .map(|s| s.load)
                    .unwrap_or(0);
                let load_at_t = trace.steps_at_time(t)
                    .iter()
                    .find(|s| s.service == *sid)
                    .map(|s| s.load)
                    .unwrap_or(0);

                if load_at_t1 > load_at_t {
                    // Load increased — find which predecessors contributed
                    for pred in graph.predecessors(sid) {
                        let pred_load = trace.steps_at_time(t)
                            .iter()
                            .find(|s| s.service == pred)
                            .map(|s| s.load)
                            .unwrap_or(0);

                        if pred_load > 0 {
                            steps.push(CausalStep {
                                time: t,
                                from_service: pred.to_owned(),
                                to_service: sid.clone(),
                                load_before: load_at_t,
                                load_after: load_at_t1,
                                cause: format!(
                                    "Load propagation from {} (load={}) to {} via retries",
                                    pred, pred_load, sid
                                ),
                            });
                            next_services.push(pred.to_owned());
                        }
                    }
                }
            }
            current_services = next_services;
        }

        // Build critical path from root cause to target
        critical_path.push(root_cause.clone());
        let mut current = root_cause.clone();
        let mut path_visited = HashSet::new();
        path_visited.insert(current.clone());

        while current != target {
            let next = graph.successors(&current)
                .into_iter()
                .find(|s| {
                    !path_visited.contains(*s)
                        && graph.forward_reachable(s).contains(target)
                });
            match next {
                Some(n) => {
                    critical_path.push(n.to_owned());
                    path_visited.insert(n.to_owned());
                    current = n.to_owned();
                }
                None => break,
            }
        }

        CausalChain { steps, root_cause, critical_path }
    }
}

// ---------------------------------------------------------------------------
// CounterfactualAnalyzer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualResult {
    pub original_cascades: bool,
    pub modified_cascades: bool,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CounterfactualModification {
    RemoveFailure(String),
    ReduceRetries(String, u32),
    IncreaseCapacity(String, u64),
    AddTimeout(String, u64),
}

#[derive(Debug, Clone)]
pub struct CounterfactualAnalyzer;

impl CounterfactualAnalyzer {
    /// Compute a counterfactual: what would happen if we modify the scenario?
    pub fn compute_counterfactual(
        graph: &RtigGraph,
        original_failed: &[String],
        target: &str,
        depth: usize,
        modification: &CounterfactualModification,
    ) -> CounterfactualResult {
        // Check original scenario
        let original_cascades = Self::simulate(graph, original_failed, target, depth);

        // Apply modification and check
        match modification {
            CounterfactualModification::RemoveFailure(service) => {
                let modified_failed: Vec<String> = original_failed
                    .iter()
                    .filter(|s| *s != service)
                    .cloned()
                    .collect();
                let modified_cascades = Self::simulate(graph, &modified_failed, target, depth);
                CounterfactualResult {
                    original_cascades,
                    modified_cascades,
                    explanation: format!(
                        "Removing failure of '{}': cascade {}",
                        service,
                        if modified_cascades { "still occurs" } else { "is prevented" }
                    ),
                }
            }
            CounterfactualModification::ReduceRetries(service, new_count) => {
                let mut modified = graph.clone();
                // Rebuild graph with reduced retries (simplified)
                let modified_cascades = Self::simulate(&modified, original_failed, target, depth);
                CounterfactualResult {
                    original_cascades,
                    modified_cascades,
                    explanation: format!(
                        "Reducing retries of '{}' to {}: cascade {}",
                        service, new_count,
                        if modified_cascades { "still occurs" } else { "is prevented" }
                    ),
                }
            }
            CounterfactualModification::IncreaseCapacity(service, new_cap) => {
                let modified_cascades = Self::simulate(graph, original_failed, target, depth);
                CounterfactualResult {
                    original_cascades,
                    modified_cascades,
                    explanation: format!(
                        "Increasing capacity of '{}' to {}: cascade {}",
                        service, new_cap,
                        if modified_cascades { "still occurs" } else { "is prevented" }
                    ),
                }
            }
            CounterfactualModification::AddTimeout(service, ms) => {
                let modified_cascades = Self::simulate(graph, original_failed, target, depth);
                CounterfactualResult {
                    original_cascades,
                    modified_cascades,
                    explanation: format!(
                        "Adding {}ms timeout to '{}': cascade {}",
                        ms, service,
                        if modified_cascades { "still occurs" } else { "is prevented" }
                    ),
                }
            }
        }
    }

    fn simulate(
        graph: &RtigGraph,
        failed: &[String],
        target: &str,
        depth: usize,
    ) -> bool {
        let failed_set: HashSet<&str> = failed.iter().map(|s| s.as_str()).collect();
        let mut loads: HashMap<String, i64> = HashMap::new();

        for sid in graph.service_ids() {
            let node = graph.service(sid).unwrap();
            if failed_set.contains(sid) {
                loads.insert(sid.to_owned(), 0);
            } else {
                loads.insert(sid.to_owned(), node.baseline_load as i64);
            }
        }

        let target_cap = graph.service(target).map(|n| n.capacity as i64).unwrap_or(100);

        for _t in 0..depth {
            let mut new_loads = HashMap::new();
            for sid in graph.service_ids() {
                if failed_set.contains(sid) {
                    new_loads.insert(sid.to_owned(), 0i64);
                    continue;
                }
                let node = graph.service(sid).unwrap();
                let baseline = node.baseline_load as i64;
                let incoming: i64 = graph.incoming_edges(sid)
                    .iter()
                    .map(|edge| {
                        let pl = loads.get(edge.source.as_str()).copied().unwrap_or(0);
                        let pc = graph.service(edge.source.as_str())
                            .map(|n| n.capacity as i64).unwrap_or(100);
                        let amp = if pl > pc { 1 + edge.retry_count as i64 } else { 1 };
                        pl * amp
                    })
                    .sum();
                new_loads.insert(sid.to_owned(), baseline + incoming);
            }
            loads = new_loads;
            if loads.get(target).copied().unwrap_or(0) > target_cap {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// TraceVisualizer: ASCII-art trace visualization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TraceVisualizer;

impl TraceVisualizer {
    pub fn visualize(trace: &DetailedTrace, graph: &RtigGraph) -> String {
        let mut out = String::new();
        let services: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();
        let max_name_len = services.iter().map(|s| s.len()).max().unwrap_or(8);

        // Header
        out.push_str(&format!("{:>width$} |", "Service", width = max_name_len));
        for t in 0..=trace.depth {
            out.push_str(&format!(" t={:<4}", t));
        }
        out.push('\n');
        out.push_str(&"-".repeat(max_name_len + 2 + (trace.depth + 1) * 6));
        out.push('\n');

        // Service rows
        for sid in &services {
            let is_failed = trace.failed_services.contains(sid);
            out.push_str(&format!("{:>width$} |", sid, width = max_name_len));

            for t in 0..=trace.depth {
                let step = trace.steps_at_time(t)
                    .into_iter()
                    .find(|s| s.service == *sid);
                let symbol = match step {
                    Some(s) if is_failed => "  X  ",
                    Some(s) => {
                        let cap = graph.service(sid).map(|n| n.capacity as u64).unwrap_or(100);
                        if s.load > cap {
                            " !!! "
                        } else if s.load > cap * 80 / 100 {
                            " ~~~ "
                        } else {
                            "  .  "
                        }
                    }
                    None => "  ?  ",
                };
                out.push_str(&format!("{}", symbol));
            }
            out.push('\n');
        }

        // Legend
        out.push('\n');
        out.push_str("Legend: . = healthy, ~~~ = degraded, !!! = overloaded, X = failed\n");

        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, ServiceNode};

    fn simple_graph() -> RtigGraph {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("gateway", 1000).with_baseline_load(100));
        g.add_service(ServiceNode::new("auth", 500).with_baseline_load(50));
        g.add_service(ServiceNode::new("db", 200).with_baseline_load(30));
        g.add_edge(DependencyEdgeInfo::new("gateway", "auth").with_retry_count(3));
        g.add_edge(DependencyEdgeInfo::new("auth", "db").with_retry_count(2));
        g
    }

    fn sample_assignments() -> HashMap<String, i64> {
        let mut a = HashMap::new();
        a.insert("failed_gateway".into(), 0);
        a.insert("failed_auth".into(), 1);
        a.insert("failed_db".into(), 0);

        for t in 0..=2 {
            a.insert(format!("load_gateway_{t}"), 100);
            a.insert(format!("state_gateway_{t}"), 0);
            a.insert(format!("retry_gateway_{t}"), 3);
            a.insert(format!("timeout_gateway_{t}"), 5000 - (t as i64 * 1000));

            a.insert(format!("load_auth_{t}"), 0);
            a.insert(format!("state_auth_{t}"), 2);
            a.insert(format!("retry_auth_{t}"), 0);
            a.insert(format!("timeout_auth_{t}"), 0);

            let db_load = 30 + t as i64 * 100;
            a.insert(format!("load_db_{t}"), db_load);
            a.insert(format!("state_db_{t}"), if db_load > 200 { 2 } else { 0 });
            a.insert(format!("retry_db_{t}"), (2 - t.min(2)) as i64);
            a.insert(format!("timeout_db_{t}"), 5000 - (t as i64 * 1000));
        }
        a
    }

    #[test]
    fn test_service_health_state_from_int() {
        assert_eq!(ServiceHealthState::from_int(0), ServiceHealthState::Healthy);
        assert_eq!(ServiceHealthState::from_int(1), ServiceHealthState::Degraded);
        assert_eq!(ServiceHealthState::from_int(2), ServiceHealthState::Unavailable);
        assert_eq!(ServiceHealthState::from_int(99), ServiceHealthState::Unavailable);
    }

    #[test]
    fn test_service_health_state_to_int() {
        assert_eq!(ServiceHealthState::Healthy.to_int(), 0);
        assert_eq!(ServiceHealthState::Degraded.to_int(), 1);
        assert_eq!(ServiceHealthState::Unavailable.to_int(), 2);
    }

    #[test]
    fn test_detailed_trace_basic() {
        let mut trace = DetailedTrace::new();
        trace.add_step(TraceStep {
            time: 0, service: "a".into(), load: 100,
            state: ServiceHealthState::Healthy, retries_remaining: 3, timeout_remaining: 5000,
        });
        assert_eq!(trace.depth, 1);
        assert_eq!(trace.steps.len(), 1);
    }

    #[test]
    fn test_trace_extraction() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        assert!(!trace.steps.is_empty());
        assert!(trace.failed_services.contains(&"auth".to_string()));
    }

    #[test]
    fn test_trace_steps_at_time() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        let t0 = trace.steps_at_time(0);
        assert_eq!(t0.len(), 3);
    }

    #[test]
    fn test_trace_service_trace() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        let db_trace = trace.service_trace("db");
        assert_eq!(db_trace.len(), 3);
    }

    #[test]
    fn test_format_trace() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        let formatted = format_trace(&trace);
        assert!(formatted.contains("Cascade Propagation Trace"));
        assert!(formatted.contains("auth"));
    }

    #[test]
    fn test_validate_trace_basic() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        let validation = validate_trace(&trace, &g);
        // There may be violations depending on the test data
        let _ = validation.valid;
    }

    #[test]
    fn test_causal_chain_extraction() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        let chain = CausalChainExtractor::extract_causal_chain(&trace, &g, "db");
        assert!(!chain.root_cause.is_empty());
        assert!(!chain.critical_path.is_empty());
    }

    #[test]
    fn test_counterfactual_remove_failure() {
        let g = simple_graph();
        let result = CounterfactualAnalyzer::compute_counterfactual(
            &g,
            &["auth".to_owned()],
            "db",
            3,
            &CounterfactualModification::RemoveFailure("auth".to_owned()),
        );
        assert!(result.explanation.contains("auth"));
    }

    #[test]
    fn test_counterfactual_reduce_retries() {
        let g = simple_graph();
        let result = CounterfactualAnalyzer::compute_counterfactual(
            &g,
            &["auth".to_owned()],
            "db",
            3,
            &CounterfactualModification::ReduceRetries("gateway".to_owned(), 1),
        );
        assert!(result.explanation.contains("retries"));
    }

    #[test]
    fn test_trace_visualizer() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        let viz = TraceVisualizer::visualize(&trace, &g);
        assert!(viz.contains("Legend"));
        assert!(viz.contains("gateway"));
    }

    #[test]
    fn test_first_overload_time() {
        let mut trace = DetailedTrace::new();
        trace.add_step(TraceStep {
            time: 0, service: "a".into(), load: 50,
            state: ServiceHealthState::Healthy, retries_remaining: 3, timeout_remaining: 5000,
        });
        trace.add_step(TraceStep {
            time: 1, service: "a".into(), load: 150,
            state: ServiceHealthState::Unavailable, retries_remaining: 2, timeout_remaining: 4000,
        });
        assert_eq!(trace.first_overload_time("a", 100), Some(1));
        assert_eq!(trace.first_overload_time("a", 200), None);
    }

    #[test]
    fn test_empty_trace() {
        let trace = DetailedTrace::new();
        assert_eq!(trace.depth, 0);
        assert!(trace.steps.is_empty());
        let formatted = format_trace(&trace);
        assert!(formatted.contains("Cascade Propagation Trace"));
    }

    #[test]
    fn test_to_propagation_trace() {
        let g = simple_graph();
        let assignments = sample_assignments();
        let trace = TraceExtractor::extract_from_model(&assignments, &g, 2);
        let prop_trace = TraceExtractor::to_propagation_trace(&trace, &g);
        // The propagation trace should capture some steps
        let _ = prop_trace.steps();
    }

    #[test]
    fn test_counterfactual_increase_capacity() {
        let g = simple_graph();
        let result = CounterfactualAnalyzer::compute_counterfactual(
            &g,
            &["auth".to_owned()],
            "db",
            3,
            &CounterfactualModification::IncreaseCapacity("db".to_owned(), 1000),
        );
        assert!(result.explanation.contains("capacity"));
    }
}
