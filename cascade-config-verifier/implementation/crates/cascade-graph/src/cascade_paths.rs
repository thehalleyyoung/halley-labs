//! Cascade path composition, simulation, bottleneck detection, and convergence.

use crate::path_analysis::{PathComposer, PathEnumerator};
use crate::rtig::RtigGraph;
use cascade_types::service::ServiceId;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SimulationState
// ---------------------------------------------------------------------------

/// Health status of a service during simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceHealth {
    Healthy,
    Degraded,
    Unavailable,
}

/// Configuration for the propagation simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub max_steps: usize,
    pub convergence_epsilon: f64,
    pub degraded_threshold: f64,
    pub unavailable_threshold: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            max_steps: 50,
            convergence_epsilon: 1e-6,
            degraded_threshold: 0.8,
            unavailable_threshold: 1.0,
        }
    }
}

/// Snapshot of load and health at one discrete time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub loads: IndexMap<ServiceId, f64>,
    pub health: IndexMap<ServiceId, ServiceHealth>,
    pub time_step: usize,
}

impl SimulationState {
    pub fn new(graph: &RtigGraph) -> Self {
        let mut loads = IndexMap::new();
        let mut health = IndexMap::new();
        for svc in graph.services() {
            loads.insert(svc.clone(), 1.0);
            health.insert(svc, ServiceHealth::Healthy);
        }
        Self { loads, health, time_step: 0 }
    }
}

// ---------------------------------------------------------------------------
// LoadPropagator
// ---------------------------------------------------------------------------

pub struct LoadPropagator;

impl LoadPropagator {
    /// Propagate one discrete time step: each service's load is its base load
    /// plus the amplified load flowing in from predecessors.
    pub fn propagate_one_step(graph: &RtigGraph, state: &SimulationState, config: &SimulationConfig) -> SimulationState {
        let mut new_loads = IndexMap::new();
        for svc in graph.services() {
            let base = state.loads.get(&svc).copied().unwrap_or(1.0);
            let incoming: f64 = graph
                .get_predecessors(&svc)
                .iter()
                .map(|pred| {
                    let pred_load = state.loads.get(pred).copied().unwrap_or(0.0);
                    let amp = graph
                        .get_edge_policy(pred, &svc)
                        .map(|p| p.amplification_factor() as f64)
                        .unwrap_or(1.0);
                    pred_load * amp
                })
                .sum();
            new_loads.insert(svc.clone(), base + incoming);
        }
        let mut new_health = IndexMap::new();
        for svc in graph.services() {
            let load = new_loads.get(&svc).copied().unwrap_or(0.0);
            let h = if load > config.unavailable_threshold * 100.0 {
                ServiceHealth::Unavailable
            } else if load > config.degraded_threshold * 100.0 {
                ServiceHealth::Degraded
            } else {
                ServiceHealth::Healthy
            };
            new_health.insert(svc, h);
        }
        SimulationState {
            loads: new_loads,
            health: new_health,
            time_step: state.time_step + 1,
        }
    }
}

// ---------------------------------------------------------------------------
// PropagationSimulator
// ---------------------------------------------------------------------------

pub struct PropagationSimulator;

impl PropagationSimulator {
    /// Run a discrete-step simulation.
    pub fn simulate(
        graph: &RtigGraph,
        failed_services: &[ServiceId],
        config: &SimulationConfig,
    ) -> Vec<SimulationState> {
        let mut state = SimulationState::new(graph);
        // Mark failed services.
        for f in failed_services {
            state.loads.insert(f.clone(), 0.0);
            state.health.insert(f.clone(), ServiceHealth::Unavailable);
        }
        let mut history = vec![state.clone()];
        for _ in 0..config.max_steps {
            let next = LoadPropagator::propagate_one_step(graph, history.last().unwrap(), config);
            history.push(next);
        }
        history
    }

    /// Run with default config.
    pub fn simulate_default(graph: &RtigGraph, failed_services: &[ServiceId]) -> Vec<SimulationState> {
        Self::simulate(graph, failed_services, &SimulationConfig::default())
    }
}

// ---------------------------------------------------------------------------
// CascadePathComposition
// ---------------------------------------------------------------------------

/// Composition details for a single cascade path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadePathComposition {
    pub path: Vec<ServiceId>,
    pub amplification_per_hop: Vec<f64>,
    pub cumulative_amplification: Vec<f64>,
    pub total_latency_ms: u64,
}

// ---------------------------------------------------------------------------
// CriticalPathFinder
// ---------------------------------------------------------------------------

pub struct CriticalPathFinder;

impl CriticalPathFinder {
    /// Find the path with the highest cumulative amplification.
    pub fn find_critical_path(graph: &RtigGraph) -> Option<CascadePathComposition> {
        let roots = PathEnumerator::find_roots(graph);
        let leaves = PathEnumerator::find_leaves(graph);
        let mut best: Option<(Vec<ServiceId>, f64)> = None;

        for root in &roots {
            for leaf in &leaves {
                for path in PathEnumerator::enumerate_paths(graph, root, leaf) {
                    let amp = PathComposer::amplification_factor(graph, &path);
                    if best.as_ref().map_or(true, |(_, a)| amp > *a) {
                        best = Some((path, amp));
                    }
                }
            }
        }

        best.map(|(path, _)| Self::compose_path(graph, &path))
    }

    fn compose_path(graph: &RtigGraph, path: &[ServiceId]) -> CascadePathComposition {
        let per_hop = PathComposer::per_hop_amplification(graph, path);
        let mut cumulative = Vec::new();
        let mut cum = 1.0f64;
        for &a in &per_hop {
            cum *= a;
            cumulative.push(cum);
        }
        let total_latency_ms = PathComposer::timeout_budget(graph, path);
        CascadePathComposition {
            path: path.to_vec(),
            amplification_per_hop: per_hop,
            cumulative_amplification: cumulative,
            total_latency_ms,
        }
    }

    /// Find the top-K critical paths.
    pub fn find_top_k(graph: &RtigGraph, k: usize) -> Vec<CascadePathComposition> {
        let roots = PathEnumerator::find_roots(graph);
        let leaves = PathEnumerator::find_leaves(graph);
        let mut all: Vec<(Vec<ServiceId>, f64)> = Vec::new();

        for root in &roots {
            for leaf in &leaves {
                for path in PathEnumerator::enumerate_paths(graph, root, leaf) {
                    let amp = PathComposer::amplification_factor(graph, &path);
                    all.push((path, amp));
                }
            }
        }
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.into_iter()
            .take(k)
            .map(|(path, _)| Self::compose_path(graph, &path))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// BottleneckFinder
// ---------------------------------------------------------------------------

pub struct BottleneckFinder;

impl BottleneckFinder {
    /// Score each service by incoming amplified load relative to fan-in.
    pub fn find_bottlenecks(graph: &RtigGraph) -> Vec<(ServiceId, f64)> {
        let mut scores: Vec<(ServiceId, f64)> = graph
            .services()
            .into_iter()
            .map(|svc| {
                let preds = graph.get_predecessors(&svc);
                let incoming_amp: f64 = preds
                    .iter()
                    .map(|pred| {
                        graph
                            .get_edge_policy(pred, &svc)
                            .map(|p| p.amplification_factor() as f64)
                            .unwrap_or(1.0)
                    })
                    .sum();
                let fan_in = preds.len().max(1) as f64;
                (svc, incoming_amp / fan_in)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Find services where the fan-in exceeds a threshold.
    pub fn high_fan_in_services(graph: &RtigGraph, threshold: usize) -> Vec<(ServiceId, usize)> {
        graph
            .services()
            .into_iter()
            .filter_map(|svc| {
                let fi = graph.get_predecessors(&svc).len();
                if fi >= threshold {
                    Some((svc, fi))
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ConvergenceChecker
// ---------------------------------------------------------------------------

pub struct ConvergenceChecker;

impl ConvergenceChecker {
    /// Return the first time step at which the simulation has converged
    /// (max load change < epsilon).
    pub fn check_convergence(states: &[SimulationState], epsilon: f64) -> Option<usize> {
        for i in 1..states.len() {
            let prev = &states[i - 1];
            let curr = &states[i];
            let max_diff: f64 = curr
                .loads
                .iter()
                .map(|(k, &v)| (v - prev.loads.get(k).copied().unwrap_or(0.0)).abs())
                .fold(0.0f64, f64::max);
            if max_diff < epsilon {
                return Some(i);
            }
        }
        None
    }

    /// Check whether all services eventually become healthy.
    pub fn all_healthy_at_end(states: &[SimulationState]) -> bool {
        states
            .last()
            .map(|s| s.health.values().all(|h| *h == ServiceHealth::Healthy))
            .unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtig::{build_chain, build_diamond, RtigGraph};
    use cascade_types::policy::{ResiliencePolicy, RetryPolicy};
    use cascade_types::service::ServiceId;

    fn sid(s: &str) -> ServiceId {
        ServiceId::new(s)
    }

    fn retry_chain() -> RtigGraph {
        let mut g = RtigGraph::new();
        for n in &["A", "B", "C"] {
            g.add_service(sid(n));
        }
        let p_ab = ResiliencePolicy::empty().with_retry(RetryPolicy::new(2));
        let p_bc = ResiliencePolicy::empty().with_retry(RetryPolicy::new(3));
        g.add_dependency(&sid("A"), &sid("B"), p_ab);
        g.add_dependency(&sid("B"), &sid("C"), p_bc);
        g
    }

    #[test]
    fn test_simulation_state_new() {
        let g = build_chain(&["A", "B", "C"], 1);
        let state = SimulationState::new(&g);
        assert_eq!(state.loads.len(), 3);
        assert_eq!(state.time_step, 0);
    }

    #[test]
    fn test_propagate_one_step() {
        let g = build_chain(&["A", "B", "C"], 1);
        let state = SimulationState::new(&g);
        let config = SimulationConfig::default();
        let next = LoadPropagator::propagate_one_step(&g, &state, &config);
        assert_eq!(next.time_step, 1);
        assert_eq!(next.loads.len(), 3);
    }

    #[test]
    fn test_simulate_default() {
        let g = build_chain(&["A", "B", "C"], 1);
        let states = PropagationSimulator::simulate_default(&g, &[]);
        assert_eq!(states.len(), 51); // initial + 50 steps
    }

    #[test]
    fn test_simulate_with_failure() {
        let g = build_chain(&["A", "B", "C"], 1);
        let states = PropagationSimulator::simulate_default(&g, &[sid("B")]);
        assert_eq!(states[0].health[&sid("B")], ServiceHealth::Unavailable);
    }

    #[test]
    fn test_critical_path() {
        let g = retry_chain();
        let cp = CriticalPathFinder::find_critical_path(&g);
        assert!(cp.is_some());
        let cp = cp.unwrap();
        assert_eq!(cp.path.len(), 3);
        assert_eq!(cp.amplification_per_hop.len(), 2);
    }

    #[test]
    fn test_top_k_critical() {
        let g = build_diamond(2);
        let top = CriticalPathFinder::find_top_k(&g, 5);
        assert!(top.len() >= 1);
    }

    #[test]
    fn test_bottleneck_finder() {
        let g = retry_chain();
        let bn = BottleneckFinder::find_bottlenecks(&g);
        assert_eq!(bn.len(), 3);
    }

    #[test]
    fn test_high_fan_in() {
        let g = build_diamond(1);
        let hi = BottleneckFinder::high_fan_in_services(&g, 2);
        assert_eq!(hi.len(), 1); // exit has fan_in=2
        assert_eq!(hi[0].0, sid("D"));
    }

    #[test]
    fn test_convergence_checker() {
        let g = build_chain(&["A", "B"], 1);
        let config = SimulationConfig { max_steps: 5, ..Default::default() };
        let states = PropagationSimulator::simulate(&g, &[], &config);
        // Should converge quickly for a simple chain with constant load.
        let conv = ConvergenceChecker::check_convergence(&states, 1e-3);
        assert!(conv.is_some());
    }

    #[test]
    fn test_all_healthy_at_end() {
        let g = build_chain(&["A", "B"], 1);
        let states = PropagationSimulator::simulate_default(&g, &[]);
        assert!(ConvergenceChecker::all_healthy_at_end(&states));
    }

    #[test]
    fn test_empty_graph_simulation() {
        let g = RtigGraph::new();
        let states = PropagationSimulator::simulate_default(&g, &[]);
        assert_eq!(states.len(), 51);
    }

    #[test]
    fn test_cascade_path_composition() {
        let g = retry_chain();
        let cp = CriticalPathFinder::find_critical_path(&g).unwrap();
        let cum = &cp.cumulative_amplification;
        assert_eq!(cum.len(), 2);
        // First hop: 3, cumulative = 3
        assert!((cum[0] - 3.0).abs() < 1e-9);
        // Second hop: 4, cumulative = 12
        assert!((cum[1] - 12.0).abs() < 1e-9);
    }
}
