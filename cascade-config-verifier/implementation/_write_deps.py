#!/usr/bin/env python3
"""Write dependency crate stubs and cascade-analysis files."""
import os

BASE = "/Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/cascade-config-verifier/implementation"

def write_file(rel_path, content):
    full = os.path.join(BASE, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    print(f"  wrote {rel_path} ({os.path.getsize(full)} bytes)")

# ─── cascade-bmc/src/lib.rs ───
write_file("crates/cascade-bmc/src/lib.rs", r"""//! Bounded model checking for cascade failure analysis.

use cascade_graph::{ServiceGraph, DependencyEdgeInfo};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcConfig {
    pub max_steps: usize,
    pub timeout_ms: u64,
    pub use_monotonicity: bool,
    pub use_antichain: bool,
    pub cone_of_influence: bool,
}

impl Default for BmcConfig {
    fn default() -> Self {
        Self { max_steps: 10, timeout_ms: 30_000, use_monotonicity: true, use_antichain: true, cone_of_influence: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BmcStatus { Safe, Unsafe, Unknown, Timeout }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSet {
    pub services: Vec<String>,
    pub size: usize,
}

impl FailureSet {
    pub fn new(services: Vec<String>) -> Self { let size = services.len(); Self { services, size } }
    pub fn contains(&self, service: &str) -> bool { self.services.iter().any(|s| s == service) }
    pub fn is_subset_of(&self, other: &FailureSet) -> bool { self.services.iter().all(|s| other.contains(s)) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceEvent {
    RequestReceived,
    RetryTriggered { attempt: u32 },
    TimeoutExpired { timeout_ms: u64 },
    CapacityExceeded { load: f64, capacity: f64 },
    CircuitBroken,
    ServiceDegraded,
    CascadeTriggered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub time_ms: u64,
    pub service: String,
    pub event: TraceEvent,
    pub load: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationTrace {
    pub steps: Vec<TraceStep>,
    pub failure_set: FailureSet,
    pub total_duration_ms: u64,
    pub affected_services: Vec<String>,
}

impl PropagationTrace {
    pub fn new(failure_set: FailureSet) -> Self {
        Self { steps: Vec::new(), failure_set, total_duration_ms: 0, affected_services: Vec::new() }
    }
    pub fn add_step(&mut self, step: TraceStep) {
        if step.time_ms > self.total_duration_ms { self.total_duration_ms = step.time_ms; }
        if !self.affected_services.contains(&step.service) { self.affected_services.push(step.service.clone()); }
        self.steps.push(step);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BmcStatistics {
    pub solver_calls: usize,
    pub explored_states: usize,
    pub pruned_states: usize,
    pub duration_ms: u64,
    pub max_depth_reached: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcResult {
    pub status: BmcStatus,
    pub failure_sets: Vec<FailureSet>,
    pub traces: Vec<PropagationTrace>,
    pub statistics: BmcStatistics,
}

#[derive(Debug, Clone)]
pub struct BmcSolver {
    graph: Arc<ServiceGraph>,
}

impl BmcSolver {
    pub fn new(graph: Arc<ServiceGraph>) -> Self { Self { graph } }
    pub fn graph(&self) -> &ServiceGraph { &self.graph }

    pub fn check(&self, config: &BmcConfig) -> BmcResult {
        let start = std::time::Instant::now();
        let mut failure_sets = Vec::new();
        let mut traces = Vec::new();
        let mut stats = BmcStatistics::default();
        let node_ids: Vec<String> = self.graph.service_ids().iter().map(|s| s.to_string()).collect();
        for id in &node_ids {
            stats.solver_calls += 1;
            stats.explored_states += 1;
            if let Some(node) = self.graph.service(id) {
                let incoming = self.graph.incoming_edges(id);
                let total_amp: f64 = incoming.iter().map(|e| e.amplification_factor()).sum();
                if total_amp > node.capacity as f64 {
                    let fs = FailureSet::new(vec![id.clone()]);
                    let trace = self.simulate_failure(&fs);
                    traces.push(trace);
                    failure_sets.push(fs);
                }
            }
        }
        stats.duration_ms = start.elapsed().as_millis() as u64;
        let status = if failure_sets.is_empty() { BmcStatus::Safe } else { BmcStatus::Unsafe };
        BmcResult { status, failure_sets, traces, statistics: stats }
    }

    pub fn find_minimal_failure_sets(&self, config: &BmcConfig, max_size: usize) -> Vec<FailureSet> {
        let result = self.check(config);
        let mut minimal = Vec::new();
        for fs in &result.failure_sets {
            if fs.size <= max_size {
                let dominated = minimal.iter().any(|m: &FailureSet| m.is_subset_of(fs));
                if !dominated {
                    minimal.retain(|m: &FailureSet| !fs.is_subset_of(m));
                    minimal.push(fs.clone());
                }
            }
        }
        minimal
    }

    pub fn generate_trace(&self, failure_set: &FailureSet) -> PropagationTrace {
        self.simulate_failure(failure_set)
    }

    fn simulate_failure(&self, failure_set: &FailureSet) -> PropagationTrace {
        let mut trace = PropagationTrace::new(failure_set.clone());
        let mut time = 0u64;
        for svc in &failure_set.services {
            trace.add_step(TraceStep {
                time_ms: time, service: svc.clone(),
                event: TraceEvent::ServiceDegraded, load: 0.0,
            });
        }
        for svc in &failure_set.services {
            for edge in self.graph.incoming_edges(svc) {
                time += edge.timeout_ms;
                for attempt in 0..edge.retry_count {
                    trace.add_step(TraceStep {
                        time_ms: time,
                        service: edge.source.as_str().to_string(),
                        event: TraceEvent::RetryTriggered { attempt: attempt + 1 },
                        load: (attempt + 1) as f64,
                    });
                    time += edge.timeout_ms;
                }
                trace.add_step(TraceStep {
                    time_ms: time,
                    service: edge.source.as_str().to_string(),
                    event: TraceEvent::TimeoutExpired { timeout_ms: edge.timeout_ms },
                    load: edge.amplification_factor(),
                });
            }
        }
        trace
    }
}
""")

# ─── cascade-maxsat/src/lib.rs ───
write_file("crates/cascade-maxsat/src/lib.rs", r"""//! MaxSAT solving for repair synthesis in CascadeVerify.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxSatStatus { Optimal, Satisfiable, Unsatisfiable, Timeout, Unknown }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Literal { pub var: usize, pub negated: bool }

impl Literal {
    pub fn pos(var: usize) -> Self { Self { var, negated: false } }
    pub fn neg(var: usize) -> Self { Self { var, negated: true } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clause { pub literals: Vec<Literal> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardClause { pub clause: Clause }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftClause { pub clause: Clause, pub weight: u64 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaxSatFormula {
    pub hard_clauses: Vec<HardClause>,
    pub soft_clauses: Vec<SoftClause>,
    pub num_vars: usize,
}

impl MaxSatFormula {
    pub fn new() -> Self { Self::default() }
    pub fn add_hard(&mut self, clause: Clause) {
        self.hard_clauses.push(HardClause { clause });
    }
    pub fn add_soft(&mut self, clause: Clause, weight: u64) {
        self.soft_clauses.push(SoftClause { clause, weight });
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Model { pub assignments: Vec<bool> }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStatistics {
    pub iterations: usize,
    pub sat_calls: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxSatResult {
    pub status: MaxSatStatus,
    pub model: Option<Model>,
    pub cost: u64,
    pub statistics: SolverStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub timeout_ms: u64,
    pub max_iterations: usize,
}

impl Default for SolverConfig {
    fn default() -> Self { Self { timeout_ms: 30_000, max_iterations: 10_000 } }
}

pub struct MaxSatSolver { config: SolverConfig }

impl MaxSatSolver {
    pub fn new(config: SolverConfig) -> Self { Self { config } }

    pub fn solve(&self, formula: &MaxSatFormula) -> MaxSatResult {
        let model = Model { assignments: vec![false; formula.num_vars] };
        let cost: u64 = formula.soft_clauses.iter().map(|c| c.weight).sum();
        MaxSatResult {
            status: MaxSatStatus::Satisfiable,
            model: Some(model),
            cost,
            statistics: SolverStatistics::default(),
        }
    }
}

impl Default for MaxSatSolver {
    fn default() -> Self { Self::new(SolverConfig::default()) }
}
""")

# ─── cascade-repair/src/lib.rs ───
write_file("crates/cascade-repair/src/lib.rs", r"""//! Repair synthesis for cascade failures.

use cascade_graph::ServiceGraph;
use serde::{Deserialize, Serialize};

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
        Self { max_retry_reduction: 3, max_timeout_adjustment_ms: 10_000, budget: 100.0, preserve_functionality: true, max_changes: 20 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepairActionType {
    ReduceRetries { from: u32, to: u32 },
    AdjustTimeout { from_ms: u64, to_ms: u64 },
    AddCircuitBreaker { threshold: u32 },
    AddRateLimit { rps: f64 },
    IncreaseCapacity { from: f64, to: f64 },
}

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
            description: format!("Reduce retries on {}->{} from {} to {}", source, target, from, to),
            deviation: (from - to) as f64,
        }
    }
    pub fn adjust_timeout(source: &str, target: &str, from_ms: u64, to_ms: u64) -> Self {
        Self {
            service: source.to_string(),
            edge: Some((source.to_string(), target.to_string())),
            action_type: RepairActionType::AdjustTimeout { from_ms, to_ms },
            description: format!("Adjust timeout on {}->{} from {}ms to {}ms", source, target, from_ms, to_ms),
            deviation: (from_ms as f64 - to_ms as f64).abs() / from_ms.max(1) as f64,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepairPlan {
    pub actions: Vec<RepairAction>,
    pub total_deviation: f64,
    pub affected_services: Vec<String>,
    pub feasible: bool,
}

impl RepairPlan {
    pub fn is_empty(&self) -> bool { self.actions.is_empty() }
    pub fn add_action(&mut self, action: RepairAction) {
        if !self.affected_services.contains(&action.service) {
            self.affected_services.push(action.service.clone());
        }
        self.total_deviation += action.deviation;
        self.actions.push(action);
    }
}

#[derive(Debug)]
pub struct RepairSynthesizer;

impl RepairSynthesizer {
    pub fn new() -> Self { Self }
    pub fn synthesize(&self, graph: &ServiceGraph, failure_services: &[Vec<String>], config: &RepairConfig) -> RepairPlan {
        let mut plan = RepairPlan { feasible: true, ..Default::default() };
        for failure_set in failure_services {
            for svc_id in failure_set {
                let incoming = graph.incoming_edges(svc_id);
                for edge in incoming {
                    if edge.retry_count > 0 && plan.actions.len() < config.max_changes {
                        let new_retries = edge.retry_count.saturating_sub(config.max_retry_reduction);
                        let action = RepairAction::reduce_retries(
                            edge.source.as_str(), edge.target.as_str(), edge.retry_count, new_retries,
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

impl Default for RepairSynthesizer {
    fn default() -> Self { Self::new() }
}
""")

print("All dependency stubs written successfully")
