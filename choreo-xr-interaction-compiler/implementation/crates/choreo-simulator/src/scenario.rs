//! Scenario descriptions and a builder API for scripted simulation tests.
//!
//! A `Scenario` bundles initial entity placement, timed scripted actions,
//! and expected-outcome assertions into a single reusable test case.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Scripted actions
// ---------------------------------------------------------------------------

/// A single discrete action that can be scheduled within a scenario.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScriptedAction {
    MoveTo {
        entity: String,
        position: [f64; 3],
        duration: f64,
    },
    SetVelocity {
        entity: String,
        velocity: [f64; 3],
    },
    PerformGesture {
        gesture: String,
        hand: String,
    },
    Wait {
        duration: f64,
    },
    SetPredicate {
        name: String,
        value: bool,
    },
    Assert(ScenarioCondition),
    Log(String),
    SpawnEntity {
        id: String,
        position: [f64; 3],
    },
    RemoveEntity {
        id: String,
    },
}

// ---------------------------------------------------------------------------
// Conditions
// ---------------------------------------------------------------------------

/// An observable condition that can be checked at any point during a scenario.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScenarioCondition {
    EntityNear {
        a: String,
        b: String,
        threshold: f64,
    },
    EntityInRegion {
        entity: String,
        min: [f64; 3],
        max: [f64; 3],
    },
    PredicateEquals {
        name: String,
        value: bool,
    },
    StateEquals {
        state: String,
    },
    TimeGreaterThan(f64),
    TimeLessThan(f64),
}

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

/// A complete test scenario: initial state, timed actions, and expectations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    /// Entity id → initial position.
    pub initial_entities: Vec<(String, [f64; 3])>,
    /// `(time, action)` pairs sorted by time.
    pub actions: Vec<(f64, ScriptedAction)>,
    /// Conditions that must hold at the end of the scenario.
    pub expected_outcomes: Vec<ScenarioCondition>,
}

impl Scenario {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            initial_entities: Vec::new(),
            actions: Vec::new(),
            expected_outcomes: Vec::new(),
        }
    }

    /// Total scripted duration (latest action timestamp).
    pub fn duration(&self) -> f64 {
        self.actions
            .iter()
            .map(|(t, _)| *t)
            .fold(0.0f64, f64::max)
    }

    /// Number of distinct entity ids referenced in initial placement.
    pub fn entity_count(&self) -> usize {
        self.initial_entities.len()
    }

    /// Sort the action list by time.
    pub fn sort_actions(&mut self) {
        self.actions.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Extract all unique entity ids that appear anywhere in the scenario
    /// (initial placement, scripted actions, or conditions).
    pub fn all_entity_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let mut push = |s: &str| {
            if seen.insert(s.to_string()) {
                ids.push(s.to_string());
            }
        };

        for (id, _) in &self.initial_entities {
            push(id);
        }
        for (_, action) in &self.actions {
            match action {
                ScriptedAction::MoveTo { entity, .. } => push(entity),
                ScriptedAction::SetVelocity { entity, .. } => push(entity),
                ScriptedAction::SpawnEntity { id, .. } => push(id),
                ScriptedAction::RemoveEntity { id } => push(id),
                ScriptedAction::Assert(cond) => match cond {
                    ScenarioCondition::EntityNear { a, b, .. } => {
                        push(a);
                        push(b);
                    }
                    ScenarioCondition::EntityInRegion { entity, .. } => push(entity),
                    _ => {}
                },
                _ => {}
            }
        }
        for cond in &self.expected_outcomes {
            match cond {
                ScenarioCondition::EntityNear { a, b, .. } => {
                    push(a);
                    push(b);
                }
                ScenarioCondition::EntityInRegion { entity, .. } => push(entity),
                _ => {}
            }
        }
        ids
    }

    /// Count the total number of scripted actions.
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }
}

// ---------------------------------------------------------------------------
// Scenario builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing `Scenario` instances.
pub struct ScenarioBuilder {
    name: String,
    description: String,
    initial_entities: Vec<(String, [f64; 3])>,
    actions: Vec<(f64, ScriptedAction)>,
    expected_outcomes: Vec<ScenarioCondition>,
    current_time: f64,
}

impl ScenarioBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            initial_entities: Vec::new(),
            actions: Vec::new(),
            expected_outcomes: Vec::new(),
            current_time: 0.0,
        }
    }

    pub fn description(mut self, d: &str) -> Self {
        self.description = d.to_string();
        self
    }

    pub fn add_entity(mut self, id: &str, pos: [f64; 3]) -> Self {
        self.initial_entities.push((id.to_string(), pos));
        self
    }

    /// Set the implicit time-stamp for subsequent actions.
    pub fn at_time(mut self, t: f64) -> Self {
        self.current_time = t;
        self
    }

    pub fn move_to(mut self, entity: &str, pos: [f64; 3], duration: f64) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::MoveTo {
                entity: entity.to_string(),
                position: pos,
                duration,
            },
        ));
        self
    }

    pub fn set_velocity(mut self, entity: &str, velocity: [f64; 3]) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::SetVelocity {
                entity: entity.to_string(),
                velocity,
            },
        ));
        self
    }

    pub fn perform_gesture(mut self, gesture: &str, hand: &str) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::PerformGesture {
                gesture: gesture.to_string(),
                hand: hand.to_string(),
            },
        ));
        self
    }

    pub fn wait(mut self, duration: f64) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::Wait { duration },
        ));
        self.current_time += duration;
        self
    }

    pub fn set_predicate(mut self, name: &str, value: bool) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::SetPredicate {
                name: name.to_string(),
                value,
            },
        ));
        self
    }

    pub fn spawn_entity(mut self, id: &str, position: [f64; 3]) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::SpawnEntity {
                id: id.to_string(),
                position,
            },
        ));
        self
    }

    pub fn remove_entity(mut self, id: &str) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::RemoveEntity {
                id: id.to_string(),
            },
        ));
        self
    }

    pub fn log(mut self, msg: &str) -> Self {
        self.actions
            .push((self.current_time, ScriptedAction::Log(msg.to_string())));
        self
    }

    pub fn assert_near(mut self, a: &str, b: &str, threshold: f64) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::Assert(ScenarioCondition::EntityNear {
                a: a.to_string(),
                b: b.to_string(),
                threshold,
            }),
        ));
        self
    }

    pub fn assert_in_region(
        mut self,
        entity: &str,
        min: [f64; 3],
        max: [f64; 3],
    ) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::Assert(ScenarioCondition::EntityInRegion {
                entity: entity.to_string(),
                min,
                max,
            }),
        ));
        self
    }

    pub fn assert_predicate(mut self, name: &str, value: bool) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::Assert(ScenarioCondition::PredicateEquals {
                name: name.to_string(),
                value,
            }),
        ));
        self
    }

    pub fn assert_state(mut self, state: &str) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::Assert(ScenarioCondition::StateEquals {
                state: state.to_string(),
            }),
        ));
        self
    }

    pub fn assert_time_greater_than(mut self, t: f64) -> Self {
        self.actions.push((
            self.current_time,
            ScriptedAction::Assert(ScenarioCondition::TimeGreaterThan(t)),
        ));
        self
    }

    pub fn expect(mut self, condition: ScenarioCondition) -> Self {
        self.expected_outcomes.push(condition);
        self
    }

    pub fn build(mut self) -> Scenario {
        self.actions.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        Scenario {
            name: self.name,
            description: self.description,
            initial_entities: self.initial_entities,
            actions: self.actions,
            expected_outcomes: self.expected_outcomes,
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario library
// ---------------------------------------------------------------------------

/// A named collection of scenarios for batch execution.
pub struct ScenarioLibrary {
    scenarios: Vec<Scenario>,
}

impl ScenarioLibrary {
    pub fn new() -> Self {
        Self {
            scenarios: Vec::new(),
        }
    }

    pub fn add(&mut self, scenario: Scenario) {
        self.scenarios.push(scenario);
    }

    pub fn get(&self, name: &str) -> Option<&Scenario> {
        self.scenarios.iter().find(|s| s.name == name)
    }

    pub fn list_names(&self) -> Vec<&str> {
        self.scenarios.iter().map(|s| s.name.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    pub fn scenarios(&self) -> &[Scenario] {
        &self.scenarios
    }

    // ---- built-in scenarios ------------------------------------------------

    /// A hand entity approaches an object, reaches grab distance, and grabs.
    pub fn builtin_grab_scenario() -> Scenario {
        ScenarioBuilder::new("builtin_grab")
            .description("Hand approaches object and grabs it")
            .add_entity("hand", [0.0, 1.0, 0.0])
            .add_entity("object", [1.0, 1.0, 0.0])
            .at_time(0.0)
            .set_velocity("hand", [0.5, 0.0, 0.0])
            .at_time(1.5)
            .assert_near("hand", "object", 0.3)
            .at_time(1.6)
            .perform_gesture("grab", "right")
            .at_time(2.0)
            .set_predicate("grabbed", true)
            .expect(ScenarioCondition::EntityNear {
                a: "hand".into(),
                b: "object".into(),
                threshold: 0.5,
            })
            .expect(ScenarioCondition::PredicateEquals {
                name: "grabbed".into(),
                value: true,
            })
            .build()
    }

    /// Gaze cursor dwells on a target for a fixed period, then selects.
    pub fn builtin_gaze_dwell_scenario() -> Scenario {
        ScenarioBuilder::new("builtin_gaze_dwell")
            .description("Gaze at target, dwell, and select")
            .add_entity("gaze_cursor", [0.0, 1.5, 2.0])
            .add_entity("target", [0.0, 1.5, 2.0])
            .at_time(0.0)
            .log("gaze starts on target")
            .set_predicate("gaze_on_target", true)
            .at_time(0.5)
            .assert_predicate("gaze_on_target", true)
            .at_time(1.0)
            .log("dwell time reached")
            .set_predicate("dwell_complete", true)
            .at_time(1.1)
            .set_predicate("selected", true)
            .expect(ScenarioCondition::PredicateEquals {
                name: "selected".into(),
                value: true,
            })
            .expect(ScenarioCondition::PredicateEquals {
                name: "dwell_complete".into(),
                value: true,
            })
            .build()
    }

    /// Two entities approach each other along the X axis until they are
    /// within a proximity threshold.
    pub fn builtin_proximity_scenario() -> Scenario {
        ScenarioBuilder::new("builtin_proximity")
            .description("Two entities approach each other")
            .add_entity("a", [-3.0, 0.0, 0.0])
            .add_entity("b", [3.0, 0.0, 0.0])
            .at_time(0.0)
            .set_velocity("a", [1.0, 0.0, 0.0])
            .set_velocity("b", [-1.0, 0.0, 0.0])
            .at_time(2.5)
            .assert_near("a", "b", 1.5)
            .expect(ScenarioCondition::EntityNear {
                a: "a".into(),
                b: "b".into(),
                threshold: 1.5,
            })
            .build()
    }

    /// Create a library pre-loaded with all built-in scenarios.
    pub fn with_builtins() -> Self {
        let mut lib = Self::new();
        lib.add(Self::builtin_grab_scenario());
        lib.add(Self::builtin_gaze_dwell_scenario());
        lib.add(Self::builtin_proximity_scenario());
        lib
    }
}

impl Default for ScenarioLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Parametric scenario
// ---------------------------------------------------------------------------

/// A scenario template whose numeric parameters can be swept over a range
/// of values.  `generate_variants` produces the Cartesian product of all
/// parameter combinations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricScenario {
    pub base: Scenario,
    /// Parameter name → list of values to sweep.
    pub parameters: HashMap<String, Vec<f64>>,
}

impl ParametricScenario {
    pub fn new(base: Scenario) -> Self {
        Self {
            base,
            parameters: HashMap::new(),
        }
    }

    pub fn add_parameter(&mut self, name: &str, values: Vec<f64>) {
        self.parameters.insert(name.to_string(), values);
    }

    /// Generate the Cartesian product of all parameter values.
    ///
    /// Each combination produces a new `Scenario` whose name is suffixed
    /// with a human-readable parameter tag and whose description contains
    /// the parameter values.
    pub fn generate_variants(&self) -> Vec<Scenario> {
        let keys: Vec<&String> = self.parameters.keys().collect();
        if keys.is_empty() {
            return vec![self.base.clone()];
        }

        let value_lists: Vec<&Vec<f64>> = keys.iter().map(|k| &self.parameters[*k]).collect();
        let combos = cartesian_product(&value_lists);

        combos
            .iter()
            .enumerate()
            .map(|(idx, combo)| {
                let mut s = self.base.clone();
                let mut suffix = String::new();
                for (i, key) in keys.iter().enumerate() {
                    if !suffix.is_empty() {
                        suffix.push('_');
                    }
                    suffix.push_str(&format!("{}={:.2}", key, combo[i]));
                }
                s.name = format!("{}_{}", self.base.name, suffix);
                s.description = format!(
                    "{} [variant {} — {}]",
                    self.base.description, idx, suffix
                );
                s
            })
            .collect()
    }

    /// Total number of variants that would be generated.
    pub fn variant_count(&self) -> usize {
        if self.parameters.is_empty() {
            return 1;
        }
        self.parameters.values().map(|v| v.len().max(1)).product()
    }
}

/// Compute the Cartesian product of a list of value slices.
fn cartesian_product(lists: &[&Vec<f64>]) -> Vec<Vec<f64>> {
    if lists.is_empty() {
        return vec![vec![]];
    }
    let mut result: Vec<Vec<f64>> = vec![vec![]];
    for list in lists {
        let mut next = Vec::new();
        for existing in &result {
            for val in *list {
                let mut combo = existing.clone();
                combo.push(*val);
                next.push(combo);
            }
        }
        result = next;
    }
    result
}

// ---------------------------------------------------------------------------
// Scenario validation helpers
// ---------------------------------------------------------------------------

/// Check that all entity ids referenced in actions also appear in the
/// initial entity list (or are spawned before use).
pub fn validate_entity_references(scenario: &Scenario) -> Vec<String> {
    let mut errors = Vec::new();
    let mut known: std::collections::HashSet<String> = scenario
        .initial_entities
        .iter()
        .map(|(id, _)| id.clone())
        .collect();

    for (time, action) in &scenario.actions {
        match action {
            ScriptedAction::SpawnEntity { id, .. } => {
                known.insert(id.clone());
            }
            ScriptedAction::MoveTo { entity, .. }
            | ScriptedAction::SetVelocity { entity, .. } => {
                if !known.contains(entity) {
                    errors.push(format!(
                        "t={:.2}: entity '{}' referenced before spawn",
                        time, entity
                    ));
                }
            }
            ScriptedAction::RemoveEntity { id } => {
                if !known.contains(id) {
                    errors.push(format!(
                        "t={:.2}: removing unknown entity '{}'",
                        time, id
                    ));
                }
                known.remove(id);
            }
            ScriptedAction::Assert(ScenarioCondition::EntityNear { a, b, .. }) => {
                if !known.contains(a) {
                    errors.push(format!(
                        "t={:.2}: assert_near references unknown entity '{}'",
                        time, a
                    ));
                }
                if !known.contains(b) {
                    errors.push(format!(
                        "t={:.2}: assert_near references unknown entity '{}'",
                        time, b
                    ));
                }
            }
            ScriptedAction::Assert(ScenarioCondition::EntityInRegion {
                entity, ..
            }) => {
                if !known.contains(entity) {
                    errors.push(format!(
                        "t={:.2}: assert_in_region references unknown entity '{}'",
                        time, entity
                    ));
                }
            }
            _ => {}
        }
    }
    errors
}

/// Check that all actions are sorted by time.
pub fn validate_action_ordering(scenario: &Scenario) -> bool {
    scenario
        .actions
        .windows(2)
        .all(|w| w[0].0 <= w[1].0 + 1e-12)
}

/// Check that a scenario has at least one entity and one expected outcome.
pub fn validate_completeness(scenario: &Scenario) -> Vec<String> {
    let mut errors = Vec::new();
    if scenario.initial_entities.is_empty() {
        errors.push("scenario has no initial entities".to_string());
    }
    if scenario.expected_outcomes.is_empty() {
        errors.push("scenario has no expected outcomes".to_string());
    }
    errors
}

// ---------------------------------------------------------------------------
// Scenario diff
// ---------------------------------------------------------------------------

/// Compare two scenarios by name, entity count, action count, and
/// expected outcome count.  Returns a list of differences.
pub fn diff_scenarios(a: &Scenario, b: &Scenario) -> Vec<String> {
    let mut diffs = Vec::new();
    if a.name != b.name {
        diffs.push(format!(
            "name: '{}' vs '{}'",
            a.name, b.name
        ));
    }
    if a.initial_entities.len() != b.initial_entities.len() {
        diffs.push(format!(
            "entity count: {} vs {}",
            a.initial_entities.len(),
            b.initial_entities.len()
        ));
    }
    if a.actions.len() != b.actions.len() {
        diffs.push(format!(
            "action count: {} vs {}",
            a.actions.len(),
            b.actions.len()
        ));
    }
    if a.expected_outcomes.len() != b.expected_outcomes.len() {
        diffs.push(format!(
            "expected outcomes: {} vs {}",
            a.expected_outcomes.len(),
            b.expected_outcomes.len()
        ));
    }
    diffs
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- builder API -------------------------------------------------------

    #[test]
    fn builder_basic() {
        let s = ScenarioBuilder::new("test")
            .description("a test scenario")
            .add_entity("hand", [0.0, 1.0, 0.0])
            .add_entity("obj", [1.0, 1.0, 0.0])
            .at_time(0.0)
            .set_velocity("hand", [1.0, 0.0, 0.0])
            .at_time(1.0)
            .assert_near("hand", "obj", 0.5)
            .expect(ScenarioCondition::EntityNear {
                a: "hand".into(),
                b: "obj".into(),
                threshold: 0.5,
            })
            .build();
        assert_eq!(s.name, "test");
        assert_eq!(s.initial_entities.len(), 2);
        assert_eq!(s.actions.len(), 2);
        assert_eq!(s.expected_outcomes.len(), 1);
    }

    #[test]
    fn builder_wait_advances_time() {
        let s = ScenarioBuilder::new("wait_test")
            .add_entity("a", [0.0; 3])
            .at_time(1.0)
            .wait(2.0)
            .log("after wait")
            .build();
        // The wait is at t=1.0, the log should be at t=3.0.
        assert_eq!(s.actions.len(), 2);
        let log_time = s.actions.iter().find(|(_, a)| matches!(a, ScriptedAction::Log(_))).unwrap().0;
        assert!((log_time - 3.0).abs() < 1e-12);
    }

    #[test]
    fn builder_actions_sorted() {
        let s = ScenarioBuilder::new("order_test")
            .add_entity("a", [0.0; 3])
            .at_time(2.0)
            .log("second")
            .at_time(0.5)
            .log("first")
            .build();
        assert!(s.actions[0].0 <= s.actions[1].0);
    }

    #[test]
    fn builder_spawn_and_remove() {
        let s = ScenarioBuilder::new("spawn_test")
            .add_entity("a", [0.0; 3])
            .at_time(1.0)
            .spawn_entity("b", [1.0, 0.0, 0.0])
            .at_time(2.0)
            .remove_entity("b")
            .build();
        assert_eq!(s.actions.len(), 2);
    }

    #[test]
    fn builder_assert_variants() {
        let s = ScenarioBuilder::new("asserts")
            .add_entity("a", [0.0; 3])
            .at_time(0.0)
            .assert_in_region("a", [-1.0; 3], [1.0; 3])
            .assert_predicate("p", true)
            .assert_state("idle")
            .assert_time_greater_than(0.0)
            .build();
        assert_eq!(s.actions.len(), 4);
    }

    // ---- serialization round-trip ------------------------------------------

    #[test]
    fn scenario_json_round_trip() {
        let original = ScenarioBuilder::new("json_test")
            .description("serialization test")
            .add_entity("x", [1.0, 2.0, 3.0])
            .at_time(0.0)
            .set_velocity("x", [0.0, 1.0, 0.0])
            .at_time(1.0)
            .assert_near("x", "x", 0.0)
            .expect(ScenarioCondition::TimeGreaterThan(0.5))
            .build();
        let json = original.to_json();
        let decoded = Scenario::from_json(&json).unwrap();
        assert_eq!(decoded.name, original.name);
        assert_eq!(decoded.initial_entities.len(), original.initial_entities.len());
        assert_eq!(decoded.actions.len(), original.actions.len());
        assert_eq!(
            decoded.expected_outcomes.len(),
            original.expected_outcomes.len()
        );
    }

    #[test]
    fn scenario_all_action_types_serialize() {
        let s = ScenarioBuilder::new("all_types")
            .add_entity("e", [0.0; 3])
            .at_time(0.0)
            .move_to("e", [1.0, 0.0, 0.0], 1.0)
            .set_velocity("e", [0.5, 0.0, 0.0])
            .perform_gesture("pinch", "left")
            .wait(0.5)
            .set_predicate("p", true)
            .log("hello")
            .spawn_entity("f", [2.0, 0.0, 0.0])
            .remove_entity("f")
            .build();
        let json = s.to_json();
        let decoded = Scenario::from_json(&json).unwrap();
        assert_eq!(decoded.actions.len(), s.actions.len());
    }

    // ---- library builtins --------------------------------------------------

    #[test]
    fn builtin_grab_scenario_valid() {
        let s = ScenarioLibrary::builtin_grab_scenario();
        assert_eq!(s.name, "builtin_grab");
        assert!(!s.initial_entities.is_empty());
        assert!(!s.expected_outcomes.is_empty());
        assert!(validate_action_ordering(&s));
    }

    #[test]
    fn builtin_gaze_dwell_scenario_valid() {
        let s = ScenarioLibrary::builtin_gaze_dwell_scenario();
        assert_eq!(s.name, "builtin_gaze_dwell");
        assert!(!s.initial_entities.is_empty());
        assert!(!s.expected_outcomes.is_empty());
    }

    #[test]
    fn builtin_proximity_scenario_valid() {
        let s = ScenarioLibrary::builtin_proximity_scenario();
        assert_eq!(s.name, "builtin_proximity");
        assert_eq!(s.initial_entities.len(), 2);
        assert!(!s.expected_outcomes.is_empty());
    }

    #[test]
    fn library_with_builtins() {
        let lib = ScenarioLibrary::with_builtins();
        assert_eq!(lib.len(), 3);
        assert!(lib.get("builtin_grab").is_some());
        assert!(lib.get("builtin_gaze_dwell").is_some());
        assert!(lib.get("builtin_proximity").is_some());
        assert!(lib.get("nonexistent").is_none());
    }

    #[test]
    fn library_add_and_list() {
        let mut lib = ScenarioLibrary::new();
        assert!(lib.is_empty());
        lib.add(Scenario::new("custom"));
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.list_names(), vec!["custom"]);
    }

    // ---- parametric scenarios ----------------------------------------------

    #[test]
    fn parametric_single_param() {
        let base = ScenarioBuilder::new("param_test")
            .add_entity("a", [0.0; 3])
            .build();
        let mut ps = ParametricScenario::new(base);
        ps.add_parameter("speed", vec![1.0, 2.0, 3.0]);
        let variants = ps.generate_variants();
        assert_eq!(variants.len(), 3);
        assert_eq!(ps.variant_count(), 3);
        // Each variant has a unique name.
        let names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn parametric_two_params_cartesian() {
        let base = ScenarioBuilder::new("cart")
            .add_entity("a", [0.0; 3])
            .build();
        let mut ps = ParametricScenario::new(base);
        ps.add_parameter("x", vec![1.0, 2.0]);
        ps.add_parameter("y", vec![10.0, 20.0, 30.0]);
        let variants = ps.generate_variants();
        assert_eq!(variants.len(), 6);
        assert_eq!(ps.variant_count(), 6);
    }

    #[test]
    fn parametric_no_params() {
        let base = Scenario::new("single");
        let ps = ParametricScenario::new(base.clone());
        let variants = ps.generate_variants();
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].name, "single");
    }

    #[test]
    fn parametric_serialization() {
        let base = ScenarioBuilder::new("ser")
            .add_entity("a", [0.0; 3])
            .build();
        let mut ps = ParametricScenario::new(base);
        ps.add_parameter("p", vec![1.0, 2.0]);
        let json = serde_json::to_string(&ps).unwrap();
        let decoded: ParametricScenario = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.generate_variants().len(), 2);
    }

    // ---- validation --------------------------------------------------------

    #[test]
    fn validate_entity_refs_pass() {
        let s = ScenarioBuilder::new("valid")
            .add_entity("a", [0.0; 3])
            .add_entity("b", [1.0, 0.0, 0.0])
            .at_time(0.0)
            .set_velocity("a", [1.0, 0.0, 0.0])
            .assert_near("a", "b", 1.0)
            .build();
        let errs = validate_entity_references(&s);
        assert!(errs.is_empty(), "errors: {:?}", errs);
    }

    #[test]
    fn validate_entity_refs_missing() {
        let s = ScenarioBuilder::new("invalid")
            .add_entity("a", [0.0; 3])
            .at_time(0.0)
            .set_velocity("ghost", [1.0, 0.0, 0.0])
            .build();
        let errs = validate_entity_references(&s);
        assert!(!errs.is_empty());
    }

    #[test]
    fn validate_entity_refs_spawned_later() {
        let s = ScenarioBuilder::new("spawn_order")
            .add_entity("a", [0.0; 3])
            .at_time(0.0)
            .spawn_entity("b", [1.0, 0.0, 0.0])
            .at_time(1.0)
            .set_velocity("b", [0.0, 1.0, 0.0])
            .build();
        let errs = validate_entity_references(&s);
        assert!(errs.is_empty(), "errors: {:?}", errs);
    }

    #[test]
    fn validate_action_ordering_pass() {
        let s = ScenarioBuilder::new("ordered")
            .add_entity("a", [0.0; 3])
            .at_time(0.0)
            .log("first")
            .at_time(1.0)
            .log("second")
            .build();
        assert!(validate_action_ordering(&s));
    }

    #[test]
    fn validate_completeness_pass() {
        let s = ScenarioBuilder::new("complete")
            .add_entity("a", [0.0; 3])
            .expect(ScenarioCondition::TimeGreaterThan(0.0))
            .build();
        assert!(validate_completeness(&s).is_empty());
    }

    #[test]
    fn validate_completeness_fail() {
        let s = Scenario::new("empty");
        let errs = validate_completeness(&s);
        assert_eq!(errs.len(), 2);
    }

    // ---- scenario helpers --------------------------------------------------

    #[test]
    fn scenario_duration() {
        let s = ScenarioBuilder::new("dur")
            .add_entity("a", [0.0; 3])
            .at_time(0.0)
            .log("start")
            .at_time(5.0)
            .log("end")
            .build();
        assert!((s.duration() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn scenario_all_entity_ids() {
        let s = ScenarioBuilder::new("ids")
            .add_entity("a", [0.0; 3])
            .add_entity("b", [1.0, 0.0, 0.0])
            .at_time(1.0)
            .spawn_entity("c", [2.0, 0.0, 0.0])
            .build();
        let ids = s.all_entity_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
        assert!(ids.contains(&"c".to_string()));
    }

    #[test]
    fn scenario_action_count() {
        let s = ScenarioBuilder::new("count")
            .add_entity("a", [0.0; 3])
            .at_time(0.0)
            .log("one")
            .log("two")
            .build();
        assert_eq!(s.action_count(), 2);
    }

    // ---- diff --------------------------------------------------------------

    #[test]
    fn diff_identical() {
        let s = ScenarioBuilder::new("same")
            .add_entity("a", [0.0; 3])
            .build();
        let diffs = diff_scenarios(&s, &s);
        assert!(diffs.is_empty());
    }

    #[test]
    fn diff_different_names() {
        let a = Scenario::new("alpha");
        let b = Scenario::new("beta");
        let diffs = diff_scenarios(&a, &b);
        assert!(!diffs.is_empty());
        assert!(diffs[0].contains("name"));
    }
}
