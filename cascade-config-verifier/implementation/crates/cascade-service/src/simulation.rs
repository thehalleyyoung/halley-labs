//! Discrete event simulation of cascade failures in service meshes.
//! Propagates load through the graph step by step, applying retry and
//! timeout policies, and tracks service degradation and cascades.

use std::collections::HashMap;

use cascade_graph::rtig::RtigGraph;
use cascade_types::service::ServiceHealth;
use serde::{Deserialize, Serialize};

use crate::mesh::ServiceMesh;

// ── Configuration ───────────────────────────────────────────────────

/// Simulation parameters.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub duration_steps: usize,
    pub failure_injection: Vec<FailureEvent>,
    pub record_trace: bool,
    /// Load threshold fraction (0.0–1.0) above which a service degrades.
    pub degradation_threshold: f64,
    /// Load threshold fraction above which a service is considered failed.
    pub failure_threshold: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            duration_steps: 50,
            failure_injection: Vec::new(),
            record_trace: true,
            degradation_threshold: 0.8,
            failure_threshold: 1.2,
        }
    }
}

/// A failure to inject at a specific time step.
#[derive(Debug, Clone)]
pub struct FailureEvent {
    pub time_step: usize,
    pub service: String,
    pub mode: FailureMode,
}

/// How a service fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureMode {
    /// Immediate total failure – service becomes unavailable.
    HardFailure,
    /// Gradual degradation – increased latency and error rate.
    SlowDegradation,
    /// Partial failure – some percentage of requests fail.
    PartialFailure { error_pct: u32 },
}

// ── Events ──────────────────────────────────────────────────────────

/// Simulation events emitted during a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimEvent {
    RequestSent {
        from: String,
        to: String,
        load: u64,
    },
    RequestFailed {
        from: String,
        to: String,
        reason: String,
    },
    RetryTriggered {
        from: String,
        to: String,
        attempt: u32,
    },
    TimeoutExpired {
        service: String,
        timeout_ms: u64,
    },
    ServiceDegraded {
        service: String,
        utilization: f64,
    },
    ServiceFailed {
        service: String,
        utilization: f64,
    },
    ServiceRecovered {
        service: String,
    },
    CascadeDetected {
        origin: String,
        affected: Vec<String>,
    },
}

// ── State ───────────────────────────────────────────────────────────

/// Per-service state during simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceSimState {
    pub health: ServiceHealth,
    pub current_load: u64,
    pub capacity: u64,
    pub error_rate: f64,
    pub injected_failure: Option<FailureMode>,
}

impl ServiceSimState {
    fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return if self.current_load > 0 { 2.0 } else { 0.0 };
        }
        self.current_load as f64 / self.capacity as f64
    }

    fn is_available(&self) -> bool {
        self.health != ServiceHealth::Unavailable
    }

    fn effective_error_rate(&self) -> f64 {
        match self.injected_failure {
            Some(FailureMode::HardFailure) => 1.0,
            Some(FailureMode::PartialFailure { error_pct }) => error_pct as f64 / 100.0,
            Some(FailureMode::SlowDegradation) => self.error_rate + 0.1,
            None => self.error_rate,
        }
    }
}

/// Snapshot of the entire mesh at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshState {
    pub service_states: HashMap<String, ServiceSimState>,
}

impl MeshState {
    fn from_mesh(mesh: &ServiceMesh) -> Self {
        let mut states = HashMap::new();
        for svc in &mesh.services {
            states.insert(
                svc.id.clone(),
                ServiceSimState {
                    health: ServiceHealth::Healthy,
                    current_load: svc.baseline_load,
                    capacity: svc.capacity,
                    error_rate: 0.0,
                    injected_failure: None,
                },
            );
        }
        MeshState {
            service_states: states,
        }
    }

    fn failed_services(&self) -> Vec<String> {
        self.service_states
            .iter()
            .filter(|(_, s)| s.health == ServiceHealth::Unavailable)
            .map(|(id, _)| id.clone())
            .collect()
    }

    fn degraded_services(&self) -> Vec<String> {
        self.service_states
            .iter()
            .filter(|(_, s)| s.health == ServiceHealth::Degraded)
            .map(|(id, _)| id.clone())
            .collect()
    }
}

/// One step of the simulation trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStep {
    pub time: usize,
    pub service_states: HashMap<String, ServiceSimState>,
    pub loads: HashMap<String, u64>,
    pub events: Vec<SimEvent>,
}

/// Full trace of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationTrace {
    pub steps: Vec<SimulationStep>,
}

/// Final result of a simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub trace: SimulationTrace,
    pub final_state: MeshState,
    pub cascaded: bool,
    pub time_to_cascade: Option<usize>,
    pub max_affected: usize,
}

// ── MeshSimulator ───────────────────────────────────────────────────

/// Discrete event simulator for service meshes.
pub struct MeshSimulator;

impl MeshSimulator {
    /// Run a simulation with the given configuration.
    pub fn simulate(mesh: &ServiceMesh, config: &SimulationConfig) -> SimulationResult {
        let graph = mesh.graph();
        let mut state = MeshState::from_mesh(mesh);
        let mut trace = SimulationTrace { steps: Vec::new() };
        let mut cascaded = false;
        let mut time_to_cascade: Option<usize> = None;
        let mut max_affected = 0usize;

        // Pre-index failure events by time step
        let mut injections: HashMap<usize, Vec<&FailureEvent>> = HashMap::new();
        for fe in &config.failure_injection {
            injections.entry(fe.time_step).or_default().push(fe);
        }

        for step in 0..config.duration_steps {
            let mut events = Vec::new();

            // 1. Inject failures
            if let Some(faults) = injections.get(&step) {
                for fault in faults {
                    if let Some(ss) = state.service_states.get_mut(&fault.service) {
                        ss.injected_failure = Some(fault.mode);
                        match fault.mode {
                            FailureMode::HardFailure => {
                                ss.health = ServiceHealth::Unavailable;
                                events.push(SimEvent::ServiceFailed {
                                    service: fault.service.clone(),
                                    utilization: ss.utilization(),
                                });
                            }
                            FailureMode::SlowDegradation => {
                                ss.health = ServiceHealth::Degraded;
                                events.push(SimEvent::ServiceDegraded {
                                    service: fault.service.clone(),
                                    utilization: ss.utilization(),
                                });
                            }
                            FailureMode::PartialFailure { .. } => {
                                ss.health = ServiceHealth::Degraded;
                            }
                        }
                    }
                }
            }

            // 2. Propagate load
            let (new_state, load_events) =
                Self::simulate_step(&state, &graph, config);
            events.extend(load_events);
            state = new_state;

            // 3. Check for cascade
            let failed = state.failed_services();
            let affected = failed.len() + state.degraded_services().len();
            max_affected = max_affected.max(affected);

            if failed.len() >= 2 && !cascaded {
                cascaded = true;
                time_to_cascade = Some(step);
                events.push(SimEvent::CascadeDetected {
                    origin: failed.first().cloned().unwrap_or_default(),
                    affected: failed.clone(),
                });
            }

            if config.record_trace {
                let loads: HashMap<String, u64> = state
                    .service_states
                    .iter()
                    .map(|(id, s)| (id.clone(), s.current_load))
                    .collect();
                trace.steps.push(SimulationStep {
                    time: step,
                    service_states: state.service_states.clone(),
                    loads,
                    events,
                });
            }
        }

        SimulationResult {
            trace,
            final_state: state,
            cascaded,
            time_to_cascade,
            max_affected,
        }
    }

    /// Simulate one time step: propagate loads, apply retries, check thresholds.
    fn simulate_step(
        state: &MeshState,
        graph: &RtigGraph,
        config: &SimulationConfig,
    ) -> (MeshState, Vec<SimEvent>) {
        let mut new_state = state.clone();
        let mut events = Vec::new();

        // Reset loads to baseline before propagation
        for (id, ss) in new_state.service_states.iter_mut() {
            if ss.health != ServiceHealth::Unavailable {
                let baseline = graph.service(id).map(|n| n.baseline_load as u64).unwrap_or(0);
                ss.current_load = baseline;
            }
        }

        // Propagate load through the graph
        let sorted: Vec<String> = graph.topological_sort()
            .map(|v| v.into_iter().map(|id| id.to_string()).collect())
            .unwrap_or_else(|| {
                graph.service_ids().iter().map(|s| s.to_string()).collect()
            });

        for svc in &sorted {
            let my_load = new_state
                .service_states
                .get(svc.as_str())
                .map(|s| s.current_load)
                .unwrap_or(0);

            let my_available = new_state
                .service_states
                .get(svc.as_str())
                .map(|s| s.is_available())
                .unwrap_or(false);

            if !my_available || my_load == 0 {
                continue;
            }

            for edge in graph.outgoing_edges(svc) {
                let target = edge.target.as_str();
                let target_available = new_state
                    .service_states
                    .get(target)
                    .map(|s| s.is_available())
                    .unwrap_or(false);

                let target_error_rate = new_state
                    .service_states
                    .get(target)
                    .map(|s| s.effective_error_rate())
                    .unwrap_or(0.0);

                let base_load = (my_load as f64 * edge.amplification_factor_f64().max(1.0)) as u64;

                if target_available {
                    events.push(SimEvent::RequestSent {
                        from: svc.clone(),
                        to: target.to_string(),
                        load: base_load,
                    });

                    // Retries on errors
                    let retry_load = if target_error_rate > 0.0 {
                        let retries = edge.retry_count.min(5);
                        let retry_extra =
                            (base_load as f64 * target_error_rate * retries as f64) as u64;
                        for attempt in 1..=retries {
                            events.push(SimEvent::RetryTriggered {
                                from: svc.clone(),
                                to: target.to_string(),
                                attempt,
                            });
                        }
                        retry_extra
                    } else {
                        0
                    };

                    if let Some(ts) = new_state.service_states.get_mut(target) {
                        ts.current_load += base_load + retry_load;
                    }
                } else {
                    events.push(SimEvent::RequestFailed {
                        from: svc.clone(),
                        to: target.to_string(),
                        reason: "target unavailable".into(),
                    });

                    // All retries will fail, but still generate load on caller
                    let retries = edge.retry_count.min(5);
                    for attempt in 1..=retries {
                        events.push(SimEvent::RetryTriggered {
                            from: svc.clone(),
                            to: target.to_string(),
                            attempt,
                        });
                    }
                }
            }
        }

        // 4. Update health based on load
        for (id, ss) in new_state.service_states.iter_mut() {
            if ss.injected_failure == Some(FailureMode::HardFailure) {
                continue; // Already hard-failed
            }

            let util = ss.utilization();
            if util > config.failure_threshold {
                if ss.health != ServiceHealth::Unavailable {
                    ss.health = ServiceHealth::Unavailable;
                    ss.error_rate = 1.0;
                    events.push(SimEvent::ServiceFailed {
                        service: id.clone(),
                        utilization: util,
                    });
                }
            } else if util > config.degradation_threshold {
                if ss.health == ServiceHealth::Healthy {
                    ss.health = ServiceHealth::Degraded;
                    ss.error_rate = (util - config.degradation_threshold)
                        / (config.failure_threshold - config.degradation_threshold);
                    events.push(SimEvent::ServiceDegraded {
                        service: id.clone(),
                        utilization: util,
                    });
                }
            } else if ss.health == ServiceHealth::Degraded
                && ss.injected_failure.is_none()
            {
                ss.health = ServiceHealth::Healthy;
                ss.error_rate = 0.0;
                events.push(SimEvent::ServiceRecovered {
                    service: id.clone(),
                });
            }
        }

        (new_state, events)
    }

    /// Detect if the simulation reached a steady state.
    pub fn detect_steady_state(trace: &SimulationTrace) -> Option<usize> {
        if trace.steps.len() < 3 {
            return None;
        }

        for i in 2..trace.steps.len() {
            let prev = &trace.steps[i - 1];
            let cur = &trace.steps[i];

            let all_same = prev
                .service_states
                .iter()
                .all(|(id, prev_s)| {
                    cur.service_states
                        .get(id)
                        .map(|cur_s| {
                            cur_s.health == prev_s.health
                                && cur_s.current_load == prev_s.current_load
                        })
                        .unwrap_or(false)
                });

            if all_same {
                return Some(i);
            }
        }
        None
    }

    /// Run simulation and return a concise summary.
    pub fn simulate_summary(
        mesh: &ServiceMesh,
        config: &SimulationConfig,
    ) -> SimulationSummary {
        let result = Self::simulate(mesh, config);
        let total_steps = result.trace.steps.len();
        let steady = Self::detect_steady_state(&result.trace);
        let final_failed = result.final_state.failed_services();
        let final_degraded = result.final_state.degraded_services();

        let total_events: usize = result
            .trace
            .steps
            .iter()
            .map(|s| s.events.len())
            .sum();

        SimulationSummary {
            total_steps,
            cascaded: result.cascaded,
            time_to_cascade: result.time_to_cascade,
            max_affected: result.max_affected,
            final_failed_count: final_failed.len(),
            final_degraded_count: final_degraded.len(),
            steady_state_step: steady,
            total_events,
        }
    }
}

/// Concise simulation outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSummary {
    pub total_steps: usize,
    pub cascaded: bool,
    pub time_to_cascade: Option<usize>,
    pub max_affected: usize,
    pub final_failed_count: usize,
    pub final_degraded_count: usize,
    pub steady_state_step: Option<usize>,
    pub total_events: usize,
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshBuilder;
    use cascade_graph::rtig::ServiceNode;
    use cascade_types::topology::DependencyType;

    fn stable_mesh() -> ServiceMesh {
        MeshBuilder::new()
            .add_node(ServiceNode::new("gw", 1000).with_baseline_load(100))
            .unwrap()
            .add_node(ServiceNode::new("api", 500).with_baseline_load(50))
            .unwrap()
            .add_node(ServiceNode::new("db", 300).with_baseline_load(30))
            .unwrap()
            .add_dependency_with_retries("gw", "api", DependencyType::Synchronous, 2, 2000)
            .unwrap()
            .add_dependency_with_retries("api", "db", DependencyType::Synchronous, 2, 1000)
            .unwrap()
            .build()
            .unwrap()
    }

    fn tight_mesh() -> ServiceMesh {
        MeshBuilder::new()
            .add_node(ServiceNode::new("entry", 100).with_baseline_load(80))
            .unwrap()
            .add_node(ServiceNode::new("svc-a", 50).with_baseline_load(40))
            .unwrap()
            .add_node(ServiceNode::new("svc-b", 30).with_baseline_load(20))
            .unwrap()
            .add_dependency_with_retries("entry", "svc-a", DependencyType::Synchronous, 3, 1000)
            .unwrap()
            .add_dependency_with_retries("svc-a", "svc-b", DependencyType::Synchronous, 3, 500)
            .unwrap()
            .build()
            .unwrap()
    }

    fn cascade_mesh() -> ServiceMesh {
        // Designed to cascade: small capacities, high retries
        MeshBuilder::new()
            .add_node(ServiceNode::new("a", 100).with_baseline_load(60))
            .unwrap()
            .add_node(ServiceNode::new("b", 40).with_baseline_load(20))
            .unwrap()
            .add_node(ServiceNode::new("c", 20).with_baseline_load(10))
            .unwrap()
            .add_dependency_with_retries("a", "b", DependencyType::Synchronous, 5, 1000)
            .unwrap()
            .add_dependency_with_retries("b", "c", DependencyType::Synchronous, 5, 500)
            .unwrap()
            .build()
            .unwrap()
    }

    // ── No-failure scenarios ────

    #[test]
    fn stable_mesh_no_cascade() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 20,
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        assert!(!result.cascaded);
        assert!(result.time_to_cascade.is_none());
    }

    #[test]
    fn stable_mesh_reaches_steady_state() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 30,
            record_trace: true,
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        let ss = MeshSimulator::detect_steady_state(&result.trace);
        assert!(ss.is_some());
    }

    #[test]
    fn stable_mesh_all_healthy_at_end() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 10,
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        assert!(result.final_state.failed_services().is_empty());
    }

    // ── Failure injection ──────

    #[test]
    fn inject_hard_failure() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 10,
            failure_injection: vec![FailureEvent {
                time_step: 2,
                service: "db".to_string(),
                mode: FailureMode::HardFailure,
            }],
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        assert!(result
            .final_state
            .service_states
            .get("db")
            .map(|s| s.health == ServiceHealth::Unavailable)
            .unwrap_or(false));
    }

    #[test]
    fn inject_partial_failure() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 10,
            failure_injection: vec![FailureEvent {
                time_step: 1,
                service: "api".to_string(),
                mode: FailureMode::PartialFailure { error_pct: 50 },
            }],
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        // Api should be degraded but possibly not failed
        let api = &result.final_state.service_states["api"];
        assert!(api.health == ServiceHealth::Degraded || api.health == ServiceHealth::Unavailable);
    }

    #[test]
    fn inject_slow_degradation() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 10,
            failure_injection: vec![FailureEvent {
                time_step: 0,
                service: "db".to_string(),
                mode: FailureMode::SlowDegradation,
            }],
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        let db = &result.final_state.service_states["db"];
        assert!(db.health != ServiceHealth::Healthy || db.error_rate > 0.0);
    }

    // ── Cascade scenarios ──────

    #[test]
    fn cascade_detected() {
        let mesh = cascade_mesh();
        let config = SimulationConfig {
            duration_steps: 20,
            failure_injection: vec![FailureEvent {
                time_step: 0,
                service: "c".to_string(),
                mode: FailureMode::HardFailure,
            }],
            degradation_threshold: 0.8,
            failure_threshold: 1.0,
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        // With c hard-failed and 5 retries, load amplification should
        // cascade through b
        assert!(result.cascaded || result.max_affected >= 2);
    }

    #[test]
    fn cascade_with_tight_capacity() {
        let mesh = tight_mesh();
        let config = SimulationConfig {
            duration_steps: 15,
            failure_injection: vec![FailureEvent {
                time_step: 0,
                service: "svc-b".to_string(),
                mode: FailureMode::HardFailure,
            }],
            degradation_threshold: 0.7,
            failure_threshold: 1.0,
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        assert!(result.max_affected >= 1);
    }

    // ── Trace / summary ────────

    #[test]
    fn trace_recorded() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 5,
            record_trace: true,
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        assert_eq!(result.trace.steps.len(), 5);
    }

    #[test]
    fn trace_not_recorded_when_disabled() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 5,
            record_trace: false,
            ..Default::default()
        };
        let result = MeshSimulator::simulate(&mesh, &config);
        assert!(result.trace.steps.is_empty());
    }

    #[test]
    fn simulation_summary() {
        let mesh = stable_mesh();
        let config = SimulationConfig {
            duration_steps: 10,
            ..Default::default()
        };
        let summary = MeshSimulator::simulate_summary(&mesh, &config);
        assert_eq!(summary.total_steps, 10);
        assert!(!summary.cascaded);
        assert!(summary.total_events > 0);
    }

    #[test]
    fn simulation_summary_cascade() {
        let mesh = cascade_mesh();
        let config = SimulationConfig {
            duration_steps: 20,
            failure_injection: vec![FailureEvent {
                time_step: 0,
                service: "c".to_string(),
                mode: FailureMode::HardFailure,
            }],
            failure_threshold: 1.0,
            ..Default::default()
        };
        let summary = MeshSimulator::simulate_summary(&mesh, &config);
        assert!(summary.final_failed_count >= 1);
    }

    #[test]
    fn detect_steady_state_short_trace() {
        let trace = SimulationTrace {
            steps: vec![SimulationStep {
                time: 0,
                service_states: HashMap::new(),
                loads: HashMap::new(),
                events: vec![],
            }],
        };
        assert!(MeshSimulator::detect_steady_state(&trace).is_none());
    }
}
