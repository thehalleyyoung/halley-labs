//! JSON export / import backend.
//!
//! Serialises a [`SpatialEventAutomaton`] into a portable JSON
//! representation and can reconstruct the automaton from that JSON.

use std::collections::{HashMap, HashSet};

use choreo_automata::automaton::{
    AutomatonKind, AutomatonMetadata, AutomatonStatistics, SpatialEventAutomaton,
    State, Transition,
};
use choreo_automata::{
    Action, Guard, Span, StateId, TransitionId,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{CodeGenerator, CodegenError, CodegenResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Options for the JSON exporter.
#[derive(Debug, Clone)]
pub struct JsonExporterConfig {
    /// Emit pretty-printed (indented) JSON.
    pub pretty_print: bool,
    /// Include the automaton metadata block in the output.
    pub include_metadata: bool,
}

impl Default for JsonExporterConfig {
    fn default() -> Self {
        Self {
            pretty_print: true,
            include_metadata: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Intermediate serde types
// ---------------------------------------------------------------------------

/// JSON-serialisable representation of a single state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonState {
    pub id: u32,
    pub name: String,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub is_error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invariant: Option<Guard>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub on_entry: Vec<Action>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub on_exit: Vec<Action>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// JSON-serialisable representation of a single transition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonTransition {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub guard: Guard,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<Action>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// JSON-serialisable representation of the automaton metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonMetadata {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Top-level JSON-serialisable representation of an automaton.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonAutomaton {
    pub states: Vec<JsonState>,
    pub transitions: Vec<JsonTransition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_state: Option<u32>,
    pub accepting_states: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonMetadata>,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn state_to_json(state: &State) -> JsonState {
    JsonState {
        id: state.id.0,
        name: state.name.clone(),
        is_initial: state.is_initial,
        is_accepting: state.is_accepting,
        is_error: state.is_error,
        invariant: state.invariant.clone(),
        on_entry: state.on_entry.clone(),
        on_exit: state.on_exit.clone(),
        metadata: state.metadata.clone(),
    }
}

fn transition_to_json(tr: &Transition) -> JsonTransition {
    JsonTransition {
        id: tr.id.0,
        source: tr.source.0,
        target: tr.target.0,
        guard: tr.guard.clone(),
        actions: tr.actions.clone(),
        priority: tr.priority,
        metadata: tr.metadata.clone(),
    }
}

fn json_state_to_state(js: &JsonState) -> State {
    let mut s = State::new(StateId(js.id), &js.name);
    s.is_initial = js.is_initial;
    s.is_accepting = js.is_accepting;
    s.is_error = js.is_error;
    s.invariant = js.invariant.clone();
    s.on_entry = js.on_entry.clone();
    s.on_exit = js.on_exit.clone();
    s.metadata = js.metadata.clone();
    s
}

fn json_transition_to_transition(jt: &JsonTransition) -> Transition {
    let mut t = Transition::new(
        TransitionId(jt.id),
        StateId(jt.source),
        StateId(jt.target),
        jt.guard.clone(),
        jt.actions.clone(),
    );
    t.priority = jt.priority;
    t.metadata = jt.metadata.clone();
    t
}

// ---------------------------------------------------------------------------
// JsonExporter
// ---------------------------------------------------------------------------

/// JSON code generator / exporter for [`SpatialEventAutomaton`].
pub struct JsonExporter {
    pub config: JsonExporterConfig,
}

impl JsonExporter {
    pub fn new() -> Self {
        Self {
            config: JsonExporterConfig::default(),
        }
    }

    pub fn with_config(config: JsonExporterConfig) -> Self {
        Self { config }
    }

    /// Export the automaton to a [`serde_json::Value`].
    pub fn export(&self, automaton: &SpatialEventAutomaton) -> CodegenResult<serde_json::Value> {
        let ja = self.to_json_automaton(automaton);
        serde_json::to_value(&ja).map_err(|e| CodegenError::Serialization(e.to_string()))
    }

    /// Import a [`SpatialEventAutomaton`] from a JSON string.
    pub fn import(json: &str) -> CodegenResult<SpatialEventAutomaton> {
        let ja: JsonAutomaton =
            serde_json::from_str(json).map_err(|e| CodegenError::Serialization(e.to_string()))?;
        Ok(Self::from_json_automaton(&ja))
    }

    /// Import a [`SpatialEventAutomaton`] from a [`serde_json::Value`].
    pub fn import_value(value: &serde_json::Value) -> CodegenResult<SpatialEventAutomaton> {
        let ja: JsonAutomaton = serde_json::from_value(value.clone())
            .map_err(|e| CodegenError::Serialization(e.to_string()))?;
        Ok(Self::from_json_automaton(&ja))
    }

    fn to_json_automaton(&self, automaton: &SpatialEventAutomaton) -> JsonAutomaton {
        let states: Vec<JsonState> = automaton.states.values().map(|s| state_to_json(s)).collect();
        let transitions: Vec<JsonTransition> = automaton
            .transitions
            .values()
            .map(|t| transition_to_json(t))
            .collect();

        let initial_state = automaton.initial_state.map(|id| id.0);
        let accepting_states: Vec<u32> = automaton.accepting_states.iter().map(|id| id.0).collect();

        let metadata = if self.config.include_metadata {
            let kind_str = match automaton.kind {
                AutomatonKind::DFA => "DFA",
                AutomatonKind::NFA => "NFA",
            };
            Some(JsonMetadata {
                name: automaton.metadata.name.clone(),
                description: automaton.metadata.description.clone(),
                tags: automaton.metadata.tags.clone(),
                kind: Some(kind_str.into()),
            })
        } else {
            None
        };

        JsonAutomaton {
            states,
            transitions,
            initial_state,
            accepting_states,
            metadata,
        }
    }

    fn from_json_automaton(ja: &JsonAutomaton) -> SpatialEventAutomaton {
        let name = ja
            .metadata
            .as_ref()
            .map(|m| m.name.clone())
            .unwrap_or_else(|| "imported".into());

        let mut automaton = SpatialEventAutomaton::new(&name);

        if let Some(ref meta) = ja.metadata {
            automaton.metadata.description = meta.description.clone();
            automaton.metadata.tags = meta.tags.clone();
            if let Some(ref kind_str) = meta.kind {
                automaton.kind = match kind_str.as_str() {
                    "NFA" => AutomatonKind::NFA,
                    _ => AutomatonKind::DFA,
                };
            }
        }

        // Reconstruct states
        let mut max_state_id: u32 = 0;
        for js in &ja.states {
            let state = json_state_to_state(js);
            automaton.add_state(state);
            if js.id >= max_state_id {
                max_state_id = js.id + 1;
            }
        }
        automaton.next_state_id = max_state_id;

        // Reconstruct transitions
        let mut max_transition_id: u32 = 0;
        for jt in &ja.transitions {
            let transition = json_transition_to_transition(jt);
            automaton.add_transition(transition);
            if jt.id >= max_transition_id {
                max_transition_id = jt.id + 1;
            }
        }
        automaton.next_transition_id = max_transition_id;

        // Force initial_state and accepting_states from the JSON data
        automaton.initial_state = ja.initial_state.map(StateId);
        automaton.accepting_states = ja.accepting_states.iter().map(|id| StateId(*id)).collect();

        automaton
    }
}

impl Default for JsonExporter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator impl
// ---------------------------------------------------------------------------

impl CodeGenerator for JsonExporter {
    fn generate(&self, automaton: &SpatialEventAutomaton) -> CodegenResult<String> {
        if automaton.states.is_empty() {
            return Err(CodegenError::Config(
                "cannot generate JSON for an empty automaton".into(),
            ));
        }

        let ja = self.to_json_automaton(automaton);

        let json_str = if self.config.pretty_print {
            serde_json::to_string_pretty(&ja)
        } else {
            serde_json::to_string(&ja)
        };

        json_str.map_err(|e| CodegenError::Serialization(e.to_string()))
    }

    fn name(&self) -> &str {
        "JSON"
    }

    fn file_extension(&self) -> &str {
        "json"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_automata::automaton::{
        AutomatonKind, AutomatonMetadata, AutomatonStatistics, SpatialEventAutomaton,
        State, Transition,
    };
    use choreo_automata::{
        Action, EventKind, Guard, Span, StateId, TransitionId,
        EntityId, RegionId, SpatialPredicate, TimerId, Value, VarId,
    };
    use indexmap::IndexMap;
    use std::collections::{HashMap, HashSet};

    fn two_state_automaton() -> SpatialEventAutomaton {
        let s0 = StateId(0);
        let s1 = StateId(1);
        let t0 = TransitionId(0);

        let mut states = IndexMap::new();
        let mut st0 = State::new(s0, "idle");
        st0.is_initial = true;
        let mut st1 = State::new(s1, "active");
        st1.is_accepting = true;
        states.insert(s0, st0);
        states.insert(s1, st1);

        let mut transitions = IndexMap::new();
        transitions.insert(
            t0,
            Transition::new(
                t0,
                s0,
                s1,
                Guard::Event(EventKind::GrabStart),
                vec![Action::EmitEvent(EventKind::TouchStart)],
            ),
        );

        SpatialEventAutomaton {
            states,
            transitions,
            initial_state: Some(s0),
            accepting_states: HashSet::from([s1]),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata {
                name: "test".into(),
                source_span: Some(Span::empty()),
                description: "A test automaton".into(),
                statistics: AutomatonStatistics::default(),
                tags: vec!["demo".into()],
            },
            kind: AutomatonKind::DFA,
            next_state_id: 2,
            next_transition_id: 1,
        }
    }

    fn complex_automaton() -> SpatialEventAutomaton {
        let s0 = StateId(0);
        let s1 = StateId(1);
        let s2 = StateId(2);
        let t0 = TransitionId(0);
        let t1 = TransitionId(1);

        let mut states = IndexMap::new();
        let mut st0 = State::new(s0, "idle");
        st0.is_initial = true;
        st0.on_entry = vec![Action::PlayFeedback("enter_idle".into())];

        let mut st1 = State::new(s1, "hover");
        st1.metadata.insert("region".into(), "lobby".into());

        let mut st2 = State::new(s2, "grabbed");
        st2.is_accepting = true;
        st2.is_error = false;

        states.insert(s0, st0);
        states.insert(s1, st1);
        states.insert(s2, st2);

        let guard = Guard::And(vec![
            Guard::Event(EventKind::GazeEnter),
            Guard::Spatial(SpatialPredicate::Inside {
                entity: EntityId("hand".into()),
                region: RegionId("zone1".into()),
            }),
        ]);

        let mut transitions = IndexMap::new();
        transitions.insert(
            t0,
            Transition::new(t0, s0, s1, guard, vec![
                Action::StartTimer(TimerId("dwell".into())),
            ]),
        );
        transitions.insert(
            t1,
            Transition::new(
                t1,
                s1,
                s2,
                Guard::Event(EventKind::GrabStart),
                vec![
                    Action::SetVar {
                        var: VarId("grabbed".into()),
                        value: Value::Bool(true),
                    },
                ],
            ),
        );

        SpatialEventAutomaton {
            states,
            transitions,
            initial_state: Some(s0),
            accepting_states: HashSet::from([s2]),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata {
                name: "complex".into(),
                source_span: None,
                description: "A complex test".into(),
                statistics: AutomatonStatistics::default(),
                tags: vec!["xr".into(), "test".into()],
            },
            kind: AutomatonKind::DFA,
            next_state_id: 3,
            next_transition_id: 2,
        }
    }

    #[test]
    fn test_generate_json_basic() {
        let exporter = JsonExporter::new();
        let json = exporter.generate(&two_state_automaton()).unwrap();
        assert!(json.contains("\"idle\""));
        assert!(json.contains("\"active\""));
        assert!(json.contains("GrabStart"));
    }

    #[test]
    fn test_generate_compact_json() {
        let exporter = JsonExporter::with_config(JsonExporterConfig {
            pretty_print: false,
            include_metadata: true,
        });
        let json = exporter.generate(&two_state_automaton()).unwrap();
        // Compact JSON should not have leading spaces for indentation
        assert!(!json.contains("  \"states\""));
        assert!(json.contains("\"states\""));
    }

    #[test]
    fn test_export_value() {
        let exporter = JsonExporter::new();
        let aut = two_state_automaton();
        let value = exporter.export(&aut).unwrap();
        assert!(value.is_object());
        assert!(value["states"].is_array());
        assert_eq!(value["states"].as_array().unwrap().len(), 2);
        assert!(value["transitions"].is_array());
        assert_eq!(value["initial_state"], serde_json::json!(0));
    }

    #[test]
    fn test_export_includes_metadata() {
        let exporter = JsonExporter::new();
        let aut = two_state_automaton();
        let value = exporter.export(&aut).unwrap();
        let meta = &value["metadata"];
        assert_eq!(meta["name"], "test");
        assert_eq!(meta["description"], "A test automaton");
    }

    #[test]
    fn test_export_excludes_metadata() {
        let exporter = JsonExporter::with_config(JsonExporterConfig {
            pretty_print: true,
            include_metadata: false,
        });
        let aut = two_state_automaton();
        let value = exporter.export(&aut).unwrap();
        assert!(value.get("metadata").is_none() || value["metadata"].is_null());
    }

    #[test]
    fn test_import_from_string() {
        let exporter = JsonExporter::new();
        let aut = two_state_automaton();
        let json = exporter.generate(&aut).unwrap();

        let imported = JsonExporter::import(&json).unwrap();
        assert_eq!(imported.states.len(), 2);
        assert_eq!(imported.transitions.len(), 1);
        assert_eq!(imported.initial_state, Some(StateId(0)));
        assert!(imported.accepting_states.contains(&StateId(1)));
    }

    #[test]
    fn test_import_from_value() {
        let exporter = JsonExporter::new();
        let aut = two_state_automaton();
        let value = exporter.export(&aut).unwrap();

        let imported = JsonExporter::import_value(&value).unwrap();
        assert_eq!(imported.states.len(), 2);
        assert_eq!(imported.transitions.len(), 1);
    }

    #[test]
    fn test_round_trip_basic() {
        let exporter = JsonExporter::new();
        let original = two_state_automaton();

        let json = exporter.generate(&original).unwrap();
        let imported = JsonExporter::import(&json).unwrap();

        // Verify structural equivalence
        assert_eq!(imported.states.len(), original.states.len());
        assert_eq!(imported.transitions.len(), original.transitions.len());
        assert_eq!(imported.initial_state, original.initial_state);
        assert_eq!(imported.accepting_states, original.accepting_states);
        assert_eq!(imported.metadata.name, original.metadata.name);

        // Verify state properties
        for (id, orig_state) in original.states.iter() {
            let imp_state = imported.state(*id).expect("state should exist");
            assert_eq!(imp_state.name, orig_state.name);
            assert_eq!(imp_state.is_initial, orig_state.is_initial);
            assert_eq!(imp_state.is_accepting, orig_state.is_accepting);
            assert_eq!(imp_state.is_error, orig_state.is_error);
        }

        // Verify transition properties
        for (id, orig_tr) in original.transitions.iter() {
            let imp_tr = imported.transition(*id).expect("transition should exist");
            assert_eq!(imp_tr.source, orig_tr.source);
            assert_eq!(imp_tr.target, orig_tr.target);
            assert_eq!(imp_tr.guard, orig_tr.guard);
            assert_eq!(imp_tr.actions, orig_tr.actions);
        }
    }

    #[test]
    fn test_round_trip_complex() {
        let exporter = JsonExporter::new();
        let original = complex_automaton();

        let json = exporter.generate(&original).unwrap();
        let imported = JsonExporter::import(&json).unwrap();

        assert_eq!(imported.states.len(), 3);
        assert_eq!(imported.transitions.len(), 2);
        assert_eq!(imported.initial_state, Some(StateId(0)));
        assert!(imported.accepting_states.contains(&StateId(2)));

        // Check on_entry actions survived round-trip
        let s0 = imported.state(StateId(0)).unwrap();
        assert_eq!(s0.on_entry.len(), 1);

        // Check metadata survived round-trip
        let s1 = imported.state(StateId(1)).unwrap();
        assert_eq!(s1.metadata.get("region"), Some(&"lobby".into()));

        // Check automaton-level metadata
        assert_eq!(imported.metadata.name, "complex");
        assert_eq!(imported.metadata.description, "A complex test");
        assert_eq!(imported.metadata.tags, vec!["xr".to_string(), "test".to_string()]);
    }

    #[test]
    fn test_round_trip_preserves_kind() {
        let mut aut = two_state_automaton();
        aut.kind = AutomatonKind::NFA;

        let exporter = JsonExporter::new();
        let json = exporter.generate(&aut).unwrap();
        let imported = JsonExporter::import(&json).unwrap();
        assert_eq!(imported.kind, AutomatonKind::NFA);
    }

    #[test]
    fn test_round_trip_preserves_ids() {
        let exporter = JsonExporter::new();
        let original = two_state_automaton();

        let json = exporter.generate(&original).unwrap();
        let imported = JsonExporter::import(&json).unwrap();

        assert_eq!(imported.next_state_id, original.next_state_id);
        assert_eq!(imported.next_transition_id, original.next_transition_id);
    }

    #[test]
    fn test_empty_automaton_errors() {
        let exporter = JsonExporter::new();
        let empty = SpatialEventAutomaton {
            states: IndexMap::new(),
            transitions: IndexMap::new(),
            initial_state: None,
            accepting_states: HashSet::new(),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata {
                name: "empty".into(),
                source_span: None,
                description: String::new(),
                statistics: AutomatonStatistics::default(),
                tags: Vec::new(),
            },
            kind: AutomatonKind::DFA,
            next_state_id: 0,
            next_transition_id: 0,
        };
        assert!(exporter.generate(&empty).is_err());
    }

    #[test]
    fn test_import_invalid_json() {
        let result = JsonExporter::import("{ invalid json }}}");
        assert!(result.is_err());
    }

    #[test]
    fn test_json_state_serde() {
        let js = JsonState {
            id: 0,
            name: "test".into(),
            is_initial: true,
            is_accepting: false,
            is_error: false,
            invariant: None,
            on_entry: vec![],
            on_exit: vec![],
            metadata: HashMap::new(),
        };
        let serialized = serde_json::to_string(&js).unwrap();
        let deserialized: JsonState = serde_json::from_str(&serialized).unwrap();
        assert_eq!(js, deserialized);
    }

    #[test]
    fn test_json_transition_serde() {
        let jt = JsonTransition {
            id: 0,
            source: 0,
            target: 1,
            guard: Guard::True,
            actions: vec![Action::Noop],
            priority: 5,
            metadata: HashMap::new(),
        };
        let serialized = serde_json::to_string(&jt).unwrap();
        let deserialized: JsonTransition = serde_json::from_str(&serialized).unwrap();
        assert_eq!(jt, deserialized);
    }

    #[test]
    fn test_accepting_states_in_json() {
        let exporter = JsonExporter::new();
        let aut = two_state_automaton();
        let value = exporter.export(&aut).unwrap();
        let accepting = value["accepting_states"].as_array().unwrap();
        assert!(accepting.contains(&serde_json::json!(1)));
    }

    #[test]
    fn test_file_extension() {
        let exporter = JsonExporter::new();
        assert_eq!(exporter.file_extension(), "json");
        assert_eq!(exporter.name(), "JSON");
    }
}
