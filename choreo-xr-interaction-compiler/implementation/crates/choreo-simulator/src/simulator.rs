//! Headless simulation loop that advances entities, monitors predicates,
//! and records state changes for offline testing and CI integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::trajectory::vec3_distance;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level knobs for a headless simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Physics time-step in seconds.
    pub time_step: f64,
    /// Hard ceiling on simulated time (seconds).
    pub max_duration: f64,
    /// Seed used for any pseudo-random behaviour.
    pub random_seed: u64,
    /// If `true`, skip any rendering or visual feedback paths.
    pub headless: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            time_step: 1.0 / 60.0,
            max_duration: 10.0,
            random_seed: 42,
            headless: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A single state-machine transition that fired during simulation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateChange {
    pub from_state: String,
    pub to_state: String,
    pub transition: String,
    pub time: f64,
}

/// Snapshot returned from a single simulation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStepResult {
    /// Current simulation time after this step.
    pub time: f64,
    /// State-machine transitions that fired during this step.
    pub state_changes: Vec<StateChange>,
    /// Identifiers of events that were emitted.
    pub events_fired: Vec<String>,
    /// Predicates whose truth value changed, together with the new value.
    pub predicates_changed: Vec<(String, bool)>,
}

/// Aggregate statistics collected over a full simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStats {
    pub avg_step_time_us: f64,
    pub max_step_time_us: f64,
    pub predicates_evaluated: u64,
    pub transitions_fired: u64,
}

impl Default for SimulationStats {
    fn default() -> Self {
        Self {
            avg_step_time_us: 0.0,
            max_step_time_us: 0.0,
            predicates_evaluated: 0,
            transitions_fired: 0,
        }
    }
}

/// Final report produced by `run_for_duration`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub final_time: f64,
    pub total_steps: u64,
    pub state_trace: Vec<StateChange>,
    pub event_count: u64,
    pub final_state: String,
    pub errors: Vec<String>,
    pub statistics: SimulationStats,
}

// ---------------------------------------------------------------------------
// Simulation entity
// ---------------------------------------------------------------------------

/// A lightweight runtime entity tracked by the headless simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimEntity {
    pub id: String,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    /// Quaternion stored as `[x, y, z, w]`.
    pub rotation: [f64; 4],
    /// Arbitrary key-value metadata attached to the entity.
    pub properties: HashMap<String, String>,
}

impl SimEntity {
    pub fn new(id: &str, position: [f64; 3]) -> Self {
        Self {
            id: id.to_string(),
            position,
            velocity: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            properties: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Proximity predicate
// ---------------------------------------------------------------------------

/// A named proximity predicate that fires when two entities are within a
/// configurable distance threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityPredicate {
    pub name: String,
    pub entity_a: String,
    pub entity_b: String,
    pub threshold: f64,
}

// ---------------------------------------------------------------------------
// Region predicate
// ---------------------------------------------------------------------------

/// A named region predicate that fires when an entity is inside an
/// axis-aligned bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionPredicate {
    pub name: String,
    pub entity: String,
    pub min: [f64; 3],
    pub max: [f64; 3],
}

// ---------------------------------------------------------------------------
// Transition rule
// ---------------------------------------------------------------------------

/// A lightweight transition rule evaluated each step.  When the named
/// predicate holds, the automaton transitions from `from_state` to
/// `to_state`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRule {
    pub name: String,
    pub from_state: String,
    pub to_state: String,
    pub predicate: String,
    pub require_value: bool,
}

// ---------------------------------------------------------------------------
// HeadlessSimulator
// ---------------------------------------------------------------------------

pub struct HeadlessSimulator {
    config: SimulationConfig,
    entities: HashMap<String, SimEntity>,
    current_time: f64,
    state_trace: Vec<StateChange>,
    event_log: Vec<String>,
    step_count: u64,
    current_automaton_state: Option<String>,
    predicates: HashMap<String, bool>,
    proximity_predicates: Vec<ProximityPredicate>,
    region_predicates: Vec<RegionPredicate>,
    transition_rules: Vec<TransitionRule>,
    step_times_us: Vec<f64>,
    predicates_evaluated: u64,
    transitions_fired: u64,
    errors: Vec<String>,
}

impl HeadlessSimulator {
    // ---- construction ------------------------------------------------------

    pub fn new(config: SimulationConfig) -> Self {
        Self {
            config,
            entities: HashMap::new(),
            current_time: 0.0,
            state_trace: Vec::new(),
            event_log: Vec::new(),
            step_count: 0,
            current_automaton_state: None,
            predicates: HashMap::new(),
            proximity_predicates: Vec::new(),
            region_predicates: Vec::new(),
            transition_rules: Vec::new(),
            step_times_us: Vec::new(),
            predicates_evaluated: 0,
            transitions_fired: 0,
            errors: Vec::new(),
        }
    }

    // ---- entity management -------------------------------------------------

    /// Add an entity with the given position and zero velocity.
    pub fn add_entity(&mut self, id: &str, position: [f64; 3]) {
        self.entities
            .insert(id.to_string(), SimEntity::new(id, position));
    }

    /// Remove an entity by id, returning `true` if it existed.
    pub fn remove_entity(&mut self, id: &str) -> bool {
        self.entities.remove(id).is_some()
    }

    /// Set the velocity vector for an existing entity.
    pub fn set_velocity(&mut self, id: &str, velocity: [f64; 3]) {
        if let Some(e) = self.entities.get_mut(id) {
            e.velocity = velocity;
        }
    }

    /// Set the rotation quaternion for an existing entity.
    pub fn set_rotation(&mut self, id: &str, rotation: [f64; 4]) {
        if let Some(e) = self.entities.get_mut(id) {
            e.rotation = rotation;
        }
    }

    /// Set a named property on an entity.
    pub fn set_entity_property(&mut self, id: &str, key: &str, value: &str) {
        if let Some(e) = self.entities.get_mut(id) {
            e.properties.insert(key.to_string(), value.to_string());
        }
    }

    /// Set the position of an existing entity.
    pub fn set_entity_position(&mut self, id: &str, position: [f64; 3]) {
        if let Some(e) = self.entities.get_mut(id) {
            e.position = position;
        }
    }

    /// Retrieve the current position of an entity.
    pub fn get_entity_position(&self, id: &str) -> Option<[f64; 3]> {
        self.entities.get(id).map(|e| e.position)
    }

    /// Retrieve a reference to a simulation entity.
    pub fn get_entity(&self, id: &str) -> Option<&SimEntity> {
        self.entities.get(id)
    }

    /// Return the number of entities currently tracked.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    // ---- predicate management ----------------------------------------------

    /// Manually set or override a boolean predicate.
    pub fn set_predicate(&mut self, name: &str, value: bool) {
        self.predicates.insert(name.to_string(), value);
    }

    /// Query the current value of a predicate.
    pub fn get_predicate(&self, name: &str) -> Option<bool> {
        self.predicates.get(name).copied()
    }

    /// Register a proximity-based predicate.
    pub fn add_proximity_predicate(
        &mut self,
        name: &str,
        entity_a: &str,
        entity_b: &str,
        threshold: f64,
    ) {
        self.proximity_predicates.push(ProximityPredicate {
            name: name.to_string(),
            entity_a: entity_a.to_string(),
            entity_b: entity_b.to_string(),
            threshold,
        });
        self.predicates.insert(name.to_string(), false);
    }

    /// Register a region-based predicate.
    pub fn add_region_predicate(
        &mut self,
        name: &str,
        entity: &str,
        min: [f64; 3],
        max: [f64; 3],
    ) {
        self.region_predicates.push(RegionPredicate {
            name: name.to_string(),
            entity: entity.to_string(),
            min,
            max,
        });
        self.predicates.insert(name.to_string(), false);
    }

    /// Register a state-machine transition rule.
    pub fn add_transition_rule(
        &mut self,
        name: &str,
        from_state: &str,
        to_state: &str,
        predicate: &str,
        require_value: bool,
    ) {
        self.transition_rules.push(TransitionRule {
            name: name.to_string(),
            from_state: from_state.to_string(),
            to_state: to_state.to_string(),
            predicate: predicate.to_string(),
            require_value,
        });
    }

    /// Set the current automaton state label.
    pub fn set_automaton_state(&mut self, state: &str) {
        self.current_automaton_state = Some(state.to_string());
    }

    /// Get the current automaton state label.
    pub fn get_automaton_state(&self) -> Option<&str> {
        self.current_automaton_state.as_deref()
    }

    // ---- proximity ---------------------------------------------------------

    /// Check whether two entities are within `threshold` of each other.
    pub fn check_proximity(&self, a: &str, b: &str, threshold: f64) -> bool {
        let pa = match self.entities.get(a) {
            Some(e) => e.position,
            None => return false,
        };
        let pb = match self.entities.get(b) {
            Some(e) => e.position,
            None => return false,
        };
        vec3_distance(&pa, &pb) < threshold
    }

    // ---- simulation stepping -----------------------------------------------

    /// Advance the simulation by `delta_t` seconds.
    ///
    /// 1. Integrate entity positions by velocity.
    /// 2. Evaluate all registered predicates and detect changes.
    /// 3. Fire any applicable transition rules.
    /// 4. Record state changes and events.
    pub fn step_simulation(&mut self, delta_t: f64) -> SimulationStepResult {
        let t_start = std::time::Instant::now();

        // 1. Integrate positions.
        let ids: Vec<String> = self.entities.keys().cloned().collect();
        for id in &ids {
            if let Some(e) = self.entities.get_mut(id) {
                e.position[0] += e.velocity[0] * delta_t;
                e.position[1] += e.velocity[1] * delta_t;
                e.position[2] += e.velocity[2] * delta_t;
            }
        }

        // 2. Evaluate predicates.
        let mut predicates_changed: Vec<(String, bool)> = Vec::new();
        let mut events_fired: Vec<String> = Vec::new();

        // Proximity predicates.
        for pp in &self.proximity_predicates {
            self.predicates_evaluated += 1;
            let new_val = self.entities.get(&pp.entity_a).and_then(|ea| {
                self.entities.get(&pp.entity_b).map(|eb| {
                    vec3_distance(&ea.position, &eb.position) < pp.threshold
                })
            }).unwrap_or(false);

            let old_val = self.predicates.get(&pp.name).copied().unwrap_or(false);
            if new_val != old_val {
                predicates_changed.push((pp.name.clone(), new_val));
                let ev_name = if new_val {
                    format!("{}:entered", pp.name)
                } else {
                    format!("{}:exited", pp.name)
                };
                events_fired.push(ev_name);
            }
        }

        // Region predicates.
        for rp in &self.region_predicates {
            self.predicates_evaluated += 1;
            let new_val = self.entities.get(&rp.entity).map(|e| {
                e.position[0] >= rp.min[0]
                    && e.position[0] <= rp.max[0]
                    && e.position[1] >= rp.min[1]
                    && e.position[1] <= rp.max[1]
                    && e.position[2] >= rp.min[2]
                    && e.position[2] <= rp.max[2]
            }).unwrap_or(false);

            let old_val = self.predicates.get(&rp.name).copied().unwrap_or(false);
            if new_val != old_val {
                predicates_changed.push((rp.name.clone(), new_val));
                let ev_name = if new_val {
                    format!("{}:entered", rp.name)
                } else {
                    format!("{}:exited", rp.name)
                };
                events_fired.push(ev_name);
            }
        }

        // Apply predicate updates.
        for (name, val) in &predicates_changed {
            self.predicates.insert(name.clone(), *val);
        }

        // 3. Fire transition rules.
        let mut state_changes: Vec<StateChange> = Vec::new();
        let new_time = self.current_time + delta_t;

        if let Some(ref current_st) = self.current_automaton_state.clone() {
            for rule in &self.transition_rules {
                if rule.from_state != *current_st {
                    continue;
                }
                let pred_val = self.predicates.get(&rule.predicate).copied().unwrap_or(false);
                if pred_val == rule.require_value {
                    let change = StateChange {
                        from_state: current_st.clone(),
                        to_state: rule.to_state.clone(),
                        transition: rule.name.clone(),
                        time: new_time,
                    };
                    state_changes.push(change.clone());
                    self.state_trace.push(change);
                    self.current_automaton_state = Some(rule.to_state.clone());
                    self.transitions_fired += 1;
                    events_fired.push(format!("transition:{}", rule.name));
                    break; // Only one transition per step.
                }
            }
        }

        // 4. Bookkeeping.
        self.current_time = new_time;
        self.step_count += 1;
        for ev in &events_fired {
            self.event_log.push(ev.clone());
        }

        let elapsed_us = t_start.elapsed().as_secs_f64() * 1_000_000.0;
        self.step_times_us.push(elapsed_us);

        SimulationStepResult {
            time: self.current_time,
            state_changes,
            events_fired,
            predicates_changed,
        }
    }

    /// Run the simulation for the given `duration` using the configured
    /// time step, returning the aggregate result.
    pub fn run_for_duration(&mut self, duration: f64) -> SimulationResult {
        let dt = self.config.time_step;
        let mut elapsed = 0.0;
        while elapsed < duration - 1e-12 {
            let step_dt = dt.min(duration - elapsed);
            self.step_simulation(step_dt);
            elapsed += step_dt;
        }
        self.get_result()
    }

    /// Run the simulation up to `max_duration` from the config.
    pub fn run(&mut self) -> SimulationResult {
        let dur = self.config.max_duration;
        self.run_for_duration(dur)
    }

    // ---- reset -------------------------------------------------------------

    /// Reset the simulator to its initial state, clearing all entities,
    /// predicates, traces, and resetting time to zero.
    pub fn reset(&mut self) {
        self.entities.clear();
        self.current_time = 0.0;
        self.state_trace.clear();
        self.event_log.clear();
        self.step_count = 0;
        self.current_automaton_state = None;
        self.predicates.clear();
        self.proximity_predicates.clear();
        self.region_predicates.clear();
        self.transition_rules.clear();
        self.step_times_us.clear();
        self.predicates_evaluated = 0;
        self.transitions_fired = 0;
        self.errors.clear();
    }

    // ---- result construction -----------------------------------------------

    /// Build the current `SimulationResult` snapshot.
    pub fn get_result(&self) -> SimulationResult {
        let stats = self.compute_stats();
        SimulationResult {
            final_time: self.current_time,
            total_steps: self.step_count,
            state_trace: self.state_trace.clone(),
            event_count: self.event_log.len() as u64,
            final_state: self
                .current_automaton_state
                .clone()
                .unwrap_or_default(),
            errors: self.errors.clone(),
            statistics: stats,
        }
    }

    /// Return the current simulation time.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Return the total number of steps executed so far.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Return the event log.
    pub fn event_log(&self) -> &[String] {
        &self.event_log
    }

    // ---- internal statistics -----------------------------------------------

    fn compute_stats(&self) -> SimulationStats {
        if self.step_times_us.is_empty() {
            return SimulationStats::default();
        }
        let sum: f64 = self.step_times_us.iter().sum();
        let avg = sum / self.step_times_us.len() as f64;
        let max = self
            .step_times_us
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        SimulationStats {
            avg_step_time_us: avg,
            max_step_time_us: max,
            predicates_evaluated: self.predicates_evaluated,
            transitions_fired: self.transitions_fired,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch simulation helpers
// ---------------------------------------------------------------------------

/// Run a batch of simulations with varying time-steps and return all results.
pub fn run_batch(
    configs: &[SimulationConfig],
    setup: impl Fn(&mut HeadlessSimulator),
) -> Vec<SimulationResult> {
    configs
        .iter()
        .map(|cfg| {
            let mut sim = HeadlessSimulator::new(cfg.clone());
            setup(&mut sim);
            sim.run()
        })
        .collect()
}

/// Run a convergence study: execute the same scenario at progressively
/// smaller time-steps and return results for comparison.
pub fn convergence_study(
    base_config: &SimulationConfig,
    refinements: &[f64],
    setup: impl Fn(&mut HeadlessSimulator),
) -> Vec<SimulationResult> {
    refinements
        .iter()
        .map(|&dt| {
            let mut cfg = base_config.clone();
            cfg.time_step = dt;
            let mut sim = HeadlessSimulator::new(cfg);
            setup(&mut sim);
            sim.run()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-entity scenario runner
// ---------------------------------------------------------------------------

/// Convenience: create a simulator, add a grid of `n×n` entities with the
/// given spacing, and return the simulator ready to run.
pub fn setup_grid_simulation(
    config: SimulationConfig,
    n: usize,
    spacing: f64,
) -> HeadlessSimulator {
    let mut sim = HeadlessSimulator::new(config);
    for row in 0..n {
        for col in 0..n {
            let id = format!("e_{}_{}", row, col);
            let x = col as f64 * spacing;
            let z = row as f64 * spacing;
            sim.add_entity(&id, [x, 0.0, z]);
        }
    }
    sim
}

/// Create a simulator with two entities moving toward each other along the
/// X axis.  Useful for testing proximity-predicate triggers.
pub fn setup_approach_simulation(
    config: SimulationConfig,
    speed: f64,
    initial_separation: f64,
) -> HeadlessSimulator {
    let half = initial_separation / 2.0;
    let mut sim = HeadlessSimulator::new(config);
    sim.add_entity("left", [-half, 0.0, 0.0]);
    sim.add_entity("right", [half, 0.0, 0.0]);
    sim.set_velocity("left", [speed, 0.0, 0.0]);
    sim.set_velocity("right", [-speed, 0.0, 0.0]);
    sim
}

// ---------------------------------------------------------------------------
// Deterministic LCG helper (same as benchmark)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn lcg_f64(state: &mut u64, lo: f64, hi: f64) -> f64 {
    let raw = lcg_next(state);
    let t = (raw >> 11) as f64 / ((1u64 << 53) as f64);
    lo + t * (hi - lo)
}

/// Create a simulator with `count` entities at random positions inside the
/// cube `[-bound, bound]³`.
pub fn setup_random_simulation(
    config: SimulationConfig,
    count: usize,
    bound: f64,
) -> HeadlessSimulator {
    let mut sim = HeadlessSimulator::new(config.clone());
    let mut rng = config.random_seed;
    for i in 0..count {
        let x = lcg_f64(&mut rng, -bound, bound);
        let y = lcg_f64(&mut rng, -bound, bound);
        let z = lcg_f64(&mut rng, -bound, bound);
        sim.add_entity(&format!("r_{}", i), [x, y, z]);
    }
    sim
}

// ---------------------------------------------------------------------------
// Simulation recorder
// ---------------------------------------------------------------------------

/// Records every step result for later analysis or replay.
pub struct SimulationRecorder {
    steps: Vec<SimulationStepResult>,
}

impl SimulationRecorder {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn record(&mut self, step: SimulationStepResult) {
        self.steps.push(step);
    }

    pub fn steps(&self) -> &[SimulationStepResult] {
        &self.steps
    }

    pub fn total_events(&self) -> usize {
        self.steps.iter().map(|s| s.events_fired.len()).sum()
    }

    pub fn total_state_changes(&self) -> usize {
        self.steps.iter().map(|s| s.state_changes.len()).sum()
    }

    pub fn predicate_flip_count(&self) -> usize {
        self.steps.iter().map(|s| s.predicates_changed.len()).sum()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.steps).unwrap_or_default()
    }
}

impl Default for SimulationRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Step iterator
// ---------------------------------------------------------------------------

/// An iterator that yields `SimulationStepResult` each time it is advanced.
pub struct SimulationStepIterator<'a> {
    sim: &'a mut HeadlessSimulator,
    remaining: f64,
}

impl<'a> SimulationStepIterator<'a> {
    pub fn new(sim: &'a mut HeadlessSimulator, duration: f64) -> Self {
        Self {
            sim,
            remaining: duration,
        }
    }
}

impl<'a> Iterator for SimulationStepIterator<'a> {
    type Item = SimulationStepResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining <= 1e-12 {
            return None;
        }
        let dt = self.sim.config.time_step.min(self.remaining);
        self.remaining -= dt;
        Some(self.sim.step_simulation(dt))
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a `SimulationResult` to a pretty-printed JSON string.
pub fn result_to_json(result: &SimulationResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_default()
}

/// Deserialize a `SimulationResult` from a JSON string.
pub fn result_from_json(json: &str) -> Result<SimulationResult, serde_json::Error> {
    serde_json::from_str(json)
}

/// Produce a one-line human-readable summary of a simulation result.
pub fn result_summary(result: &SimulationResult) -> String {
    format!(
        "t={:.4}s  steps={}  transitions={}  events={}  state={}",
        result.final_time,
        result.total_steps,
        result.statistics.transitions_fired,
        result.event_count,
        result.final_state,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // ---- basic entity management -------------------------------------------

    #[test]
    fn add_and_query_entity() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [1.0, 2.0, 3.0]);
        let pos = sim.get_entity_position("a").unwrap();
        assert_eq!(pos, [1.0, 2.0, 3.0]);
        assert_eq!(sim.entity_count(), 1);
    }

    #[test]
    fn remove_entity() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0; 3]);
        assert!(sim.remove_entity("a"));
        assert!(!sim.remove_entity("a"));
        assert_eq!(sim.entity_count(), 0);
    }

    #[test]
    fn missing_entity_returns_none() {
        let sim = HeadlessSimulator::new(SimulationConfig::default());
        assert!(sim.get_entity_position("nope").is_none());
    }

    // ---- velocity integration ----------------------------------------------

    #[test]
    fn velocity_integration() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0, 0.0, 0.0]);
        sim.set_velocity("a", [1.0, 0.0, 0.0]);
        sim.step_simulation(1.0);
        let pos = sim.get_entity_position("a").unwrap();
        assert!(approx(pos[0], 1.0, 1e-9));
    }

    #[test]
    fn multiple_steps_accumulate() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0, 0.0, 0.0]);
        sim.set_velocity("a", [2.0, -1.0, 0.5]);
        for _ in 0..10 {
            sim.step_simulation(0.1);
        }
        let pos = sim.get_entity_position("a").unwrap();
        assert!(approx(pos[0], 2.0, 1e-6));
        assert!(approx(pos[1], -1.0, 1e-6));
        assert!(approx(pos[2], 0.5, 1e-6));
    }

    // ---- proximity ---------------------------------------------------------

    #[test]
    fn check_proximity_within_threshold() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0, 0.0, 0.0]);
        sim.add_entity("b", [0.5, 0.0, 0.0]);
        assert!(sim.check_proximity("a", "b", 1.0));
        assert!(!sim.check_proximity("a", "b", 0.1));
    }

    #[test]
    fn proximity_with_missing_entity() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0; 3]);
        assert!(!sim.check_proximity("a", "ghost", 10.0));
    }

    // ---- proximity predicate events ----------------------------------------

    #[test]
    fn proximity_predicate_fires_event() {
        let cfg = SimulationConfig {
            time_step: 0.1,
            max_duration: 5.0,
            ..SimulationConfig::default()
        };
        let mut sim = HeadlessSimulator::new(cfg);
        sim.add_entity("a", [0.0, 0.0, 0.0]);
        sim.add_entity("b", [3.0, 0.0, 0.0]);
        sim.set_velocity("a", [1.0, 0.0, 0.0]);
        sim.add_proximity_predicate("near_ab", "a", "b", 1.5);

        // Step until the predicate fires.
        let mut entered = false;
        for _ in 0..50 {
            let result = sim.step_simulation(0.1);
            for (name, val) in &result.predicates_changed {
                if name == "near_ab" && *val {
                    entered = true;
                }
            }
            if entered {
                break;
            }
        }
        assert!(entered, "proximity predicate should have fired");
        assert_eq!(sim.get_predicate("near_ab"), Some(true));
    }

    // ---- region predicate events -------------------------------------------

    #[test]
    fn region_predicate_fires_event() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [-5.0, 0.0, 0.0]);
        sim.set_velocity("a", [1.0, 0.0, 0.0]);
        sim.add_region_predicate("zone", "a", [0.0, -1.0, -1.0], [2.0, 1.0, 1.0]);

        let mut entered = false;
        for _ in 0..100 {
            let result = sim.step_simulation(0.1);
            for (name, val) in &result.predicates_changed {
                if name == "zone" && *val {
                    entered = true;
                }
            }
            if entered {
                break;
            }
        }
        assert!(entered, "region predicate should have fired");
    }

    // ---- transition rules --------------------------------------------------

    #[test]
    fn transition_rule_fires_on_predicate() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0, 0.0, 0.0]);
        sim.add_entity("b", [3.0, 0.0, 0.0]);
        sim.set_velocity("a", [1.0, 0.0, 0.0]);

        sim.add_proximity_predicate("close", "a", "b", 1.5);
        sim.set_automaton_state("idle");
        sim.add_transition_rule("grab_rule", "idle", "grabbing", "close", true);

        let mut transitioned = false;
        for _ in 0..50 {
            let result = sim.step_simulation(0.1);
            if !result.state_changes.is_empty() {
                assert_eq!(result.state_changes[0].from_state, "idle");
                assert_eq!(result.state_changes[0].to_state, "grabbing");
                transitioned = true;
                break;
            }
        }
        assert!(transitioned, "transition should have fired");
        assert_eq!(sim.get_automaton_state(), Some("grabbing"));
    }

    // ---- run_for_duration --------------------------------------------------

    #[test]
    fn run_for_duration_basic() {
        let cfg = SimulationConfig {
            time_step: 0.01,
            max_duration: 10.0,
            ..SimulationConfig::default()
        };
        let mut sim = HeadlessSimulator::new(cfg);
        sim.add_entity("a", [0.0; 3]);
        sim.set_velocity("a", [1.0, 0.0, 0.0]);
        let result = sim.run_for_duration(1.0);
        assert!(approx(result.final_time, 1.0, 1e-6));
        assert!(result.total_steps >= 99);
        let pos = sim.get_entity_position("a").unwrap();
        assert!(approx(pos[0], 1.0, 0.02));
    }

    // ---- reset -------------------------------------------------------------

    #[test]
    fn reset_clears_state() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [1.0; 3]);
        sim.step_simulation(0.1);
        sim.reset();
        assert_eq!(sim.entity_count(), 0);
        assert!(approx(sim.current_time(), 0.0, 1e-12));
        assert_eq!(sim.step_count(), 0);
    }

    // ---- result serialization ----------------------------------------------

    #[test]
    fn result_json_round_trip() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0; 3]);
        let result = sim.run_for_duration(0.5);
        let json = result_to_json(&result);
        let decoded = result_from_json(&json).unwrap();
        assert_eq!(decoded.total_steps, result.total_steps);
    }

    // ---- convergence study -------------------------------------------------

    #[test]
    fn convergence_study_produces_results() {
        let cfg = SimulationConfig {
            time_step: 0.1,
            max_duration: 1.0,
            ..SimulationConfig::default()
        };
        let results = convergence_study(&cfg, &[0.1, 0.05, 0.01], |sim| {
            sim.add_entity("a", [0.0; 3]);
            sim.set_velocity("a", [1.0, 0.0, 0.0]);
        });
        assert_eq!(results.len(), 3);
        // Finer time-steps should yield more steps.
        assert!(results[2].total_steps > results[0].total_steps);
    }

    // ---- batch runner ------------------------------------------------------

    #[test]
    fn batch_run() {
        let cfgs: Vec<SimulationConfig> = (1..=3)
            .map(|i| SimulationConfig {
                time_step: 0.1 / i as f64,
                max_duration: 1.0,
                ..SimulationConfig::default()
            })
            .collect();
        let results = run_batch(&cfgs, |sim| {
            sim.add_entity("a", [0.0; 3]);
        });
        assert_eq!(results.len(), 3);
    }

    // ---- grid simulation ---------------------------------------------------

    #[test]
    fn grid_simulation_setup() {
        let sim = setup_grid_simulation(SimulationConfig::default(), 3, 2.0);
        assert_eq!(sim.entity_count(), 9);
    }

    // ---- approach simulation -----------------------------------------------

    #[test]
    fn approach_simulation_entities_converge() {
        let cfg = SimulationConfig {
            time_step: 0.01,
            max_duration: 10.0,
            ..SimulationConfig::default()
        };
        let mut sim = setup_approach_simulation(cfg, 1.0, 4.0);
        sim.add_proximity_predicate("close", "left", "right", 1.0);
        let result = sim.run_for_duration(2.5);
        // After 2 seconds the gap of 4 shrinks by 2*1.0*2=4 → entities meet.
        assert!(result.event_count > 0);
    }

    // ---- recorder ----------------------------------------------------------

    #[test]
    fn recorder_captures_steps() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0; 3]);
        let mut rec = SimulationRecorder::new();
        for _ in 0..5 {
            let step = sim.step_simulation(0.1);
            rec.record(step);
        }
        assert_eq!(rec.steps().len(), 5);
    }

    // ---- result summary ----------------------------------------------------

    #[test]
    fn summary_string_format() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0; 3]);
        let result = sim.run_for_duration(0.5);
        let s = result_summary(&result);
        assert!(s.contains("t="));
        assert!(s.contains("steps="));
    }

    // ---- step iterator -----------------------------------------------------

    #[test]
    fn step_iterator_yields_correct_count() {
        let cfg = SimulationConfig {
            time_step: 0.25,
            max_duration: 10.0,
            ..SimulationConfig::default()
        };
        let mut sim = HeadlessSimulator::new(cfg);
        sim.add_entity("a", [0.0; 3]);
        let iter = SimulationStepIterator::new(&mut sim, 1.0);
        let steps: Vec<_> = iter.collect();
        assert_eq!(steps.len(), 4);
    }

    // ---- random simulation -------------------------------------------------

    #[test]
    fn random_simulation_entity_count() {
        let cfg = SimulationConfig {
            random_seed: 123,
            ..SimulationConfig::default()
        };
        let sim = setup_random_simulation(cfg, 20, 10.0);
        assert_eq!(sim.entity_count(), 20);
    }

    // ---- config default ----------------------------------------------------

    #[test]
    fn default_config_is_headless() {
        let cfg = SimulationConfig::default();
        assert!(cfg.headless);
        assert!(cfg.time_step > 0.0);
        assert!(cfg.max_duration > 0.0);
    }

    // ---- entity properties -------------------------------------------------

    #[test]
    fn entity_properties() {
        let mut sim = HeadlessSimulator::new(SimulationConfig::default());
        sim.add_entity("a", [0.0; 3]);
        sim.set_entity_property("a", "type", "hand");
        let e = sim.get_entity("a").unwrap();
        assert_eq!(e.properties.get("type").unwrap(), "hand");
    }

    // ---- statistics --------------------------------------------------------

    #[test]
    fn statistics_populated_after_run() {
        let cfg = SimulationConfig {
            time_step: 0.1,
            max_duration: 1.0,
            ..SimulationConfig::default()
        };
        let mut sim = HeadlessSimulator::new(cfg);
        sim.add_entity("a", [0.0; 3]);
        sim.add_entity("b", [5.0, 0.0, 0.0]);
        sim.add_proximity_predicate("p", "a", "b", 1.0);
        let result = sim.run_for_duration(1.0);
        assert!(result.statistics.predicates_evaluated > 0);
    }
}
