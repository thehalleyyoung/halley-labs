//! Graphviz DOT format generation backend.
//!
//! Produces a DOT-language graph for visualising the automaton.  States are
//! coloured by type (initial=green, accepting=double outline, error=red),
//! transitions are labelled with guard / action annotations, and parallel
//! composition regions are rendered as clusters.

use std::collections::{HashMap, HashSet};

use choreo_automata::automaton::{SpatialEventAutomaton, State, Transition};
use choreo_automata::{Action, Guard, SpatialPredicate, StateId, TemporalGuardExpr};

use crate::template::CodeBuffer;
use crate::{CodeGenerator, CodegenError, CodegenResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Options for the generated DOT output.
#[derive(Debug, Clone)]
pub struct DotGraphConfig {
    /// Graph label shown at the top.
    pub graph_label: Option<String>,
    /// Use left-to-right layout instead of top-to-bottom.
    pub left_to_right: bool,
    /// Colour initial states green.
    pub color_initial: bool,
    /// Use double-circle for accepting states.
    pub double_accepting: bool,
    /// Colour error states red.
    pub color_error: bool,
    /// Show guard expressions on transition labels.
    pub show_guards: bool,
    /// Show action lists on transition labels.
    pub show_actions: bool,
    /// Group states into clusters by a metadata key.
    pub cluster_key: Option<String>,
    /// Font name.
    pub font: String,
    /// Font size.
    pub font_size: u32,
}

impl Default for DotGraphConfig {
    fn default() -> Self {
        Self {
            graph_label: None,
            left_to_right: true,
            color_initial: true,
            double_accepting: true,
            color_error: true,
            show_guards: true,
            show_actions: true,
            cluster_key: None,
            font: "Helvetica".into(),
            font_size: 12,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn state_label(state: &State) -> String {
    if state.name.is_empty() {
        format!("s{}", state.id.0)
    } else {
        state.name.clone()
    }
}

fn state_node_id(sid: StateId) -> String {
    format!("s{}", sid.0)
}

fn guard_label(guard: &Guard) -> String {
    match guard {
        Guard::True => String::new(),
        Guard::False => "false".into(),
        Guard::Spatial(sp) => spatial_label(sp),
        Guard::Temporal(tg) => temporal_label(tg),
        Guard::Event(ek) => format!("{}", ek),
        Guard::And(gs) => {
            let parts: Vec<String> = gs.iter().map(guard_label).collect();
            parts.join(" ∧ ")
        }
        Guard::Or(gs) => {
            let parts: Vec<String> = gs.iter().map(guard_label).collect();
            parts.join(" ∨ ")
        }
        Guard::Not(g) => format!("¬{}", guard_label(g)),
    }
}

fn spatial_label(sp: &SpatialPredicate) -> String {
    match sp {
        SpatialPredicate::Inside { entity, region } => {
            format!("{} ∈ {}", entity.0, region.0)
        }
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            threshold,
        } => format!("d({},{}) < {:.2}", entity_a.0, entity_b.0, threshold),
        SpatialPredicate::GazeAt { entity, target } => {
            format!("gaze({},{})", entity.0, target.0)
        }
        SpatialPredicate::Contact { entity_a, entity_b } => {
            format!("contact({},{})", entity_a.0, entity_b.0)
        }
        SpatialPredicate::Grasping { hand, object } => {
            format!("grasp({},{})", hand.0, object.0)
        }
        SpatialPredicate::Not(inner) => format!("¬{}", spatial_label(inner)),
        SpatialPredicate::And(ps) => {
            let parts: Vec<String> = ps.iter().map(spatial_label).collect();
            parts.join(" ∧ ")
        }
        SpatialPredicate::Or(ps) => {
            let parts: Vec<String> = ps.iter().map(spatial_label).collect();
            parts.join(" ∨ ")
        }
        SpatialPredicate::Named(id) => id.0.clone(),
    }
}

fn temporal_label(tg: &TemporalGuardExpr) -> String {
    match tg {
        TemporalGuardExpr::TimerElapsed { timer, threshold } => {
            format!("timer({}) ≥ {:.2}s", timer.0, threshold.0)
        }
        TemporalGuardExpr::WithinInterval(iv) => {
            format!("t ∈ [{:.2},{:.2}]", iv.start.0, iv.end.0)
        }
        TemporalGuardExpr::Named(id) => id.0.clone(),
        TemporalGuardExpr::And(es) => {
            let parts: Vec<String> = es.iter().map(temporal_label).collect();
            parts.join(" ∧ ")
        }
        TemporalGuardExpr::Or(es) => {
            let parts: Vec<String> = es.iter().map(temporal_label).collect();
            parts.join(" ∨ ")
        }
        TemporalGuardExpr::Not(e) => format!("¬{}", temporal_label(e)),
    }
}

fn action_label(action: &Action) -> String {
    match action {
        Action::StartTimer(id) => format!("start({})", id.0),
        Action::StopTimer(id) => format!("stop({})", id.0),
        Action::EmitEvent(ek) => format!("emit({})", ek),
        Action::SetVar { var, value } => format!("{} := {:?}", var.0, value),
        Action::PlayFeedback(fb) => format!("feedback({})", fb),
        Action::Highlight { entity, style } => format!("highlight({},{})", entity.0, style),
        Action::ClearHighlight(entity) => format!("unhighlight({})", entity.0),
        Action::MoveEntity { entity, target } => {
            format!("move({},{:.1},{:.1},{:.1})", entity.0, target.x, target.y, target.z)
        }
        Action::Custom(s) => format!("custom({})", s),
        Action::Noop => String::new(),
    }
}

// ---------------------------------------------------------------------------
// DotGraphGenerator
// ---------------------------------------------------------------------------

/// Generates Graphviz DOT source for an automaton.
pub struct DotGraphGenerator {
    pub config: DotGraphConfig,
}

impl DotGraphGenerator {
    pub fn new() -> Self {
        Self {
            config: DotGraphConfig::default(),
        }
    }

    pub fn with_config(config: DotGraphConfig) -> Self {
        Self { config }
    }

    fn emit_header(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        buf.line("digraph automaton {");
        buf.indent();

        if self.config.left_to_right {
            buf.line("rankdir=LR;");
        }
        buf.line(&format!(
            "fontname=\"{}\"; fontsize={};",
            self.config.font, self.config.font_size
        ));
        buf.line(&format!(
            "node [fontname=\"{}\", fontsize={}];",
            self.config.font, self.config.font_size
        ));
        buf.line(&format!(
            "edge [fontname=\"{}\", fontsize={}];",
            self.config.font,
            self.config.font_size.saturating_sub(2)
        ));

        if let Some(label) = &self.config.graph_label {
            buf.line(&format!("label=\"{}\";", dot_escape(label)));
            buf.line("labelloc=t;");
        } else {
            buf.line(&format!(
                "label=\"{}\";",
                dot_escape(&automaton.metadata.name)
            ));
            buf.line("labelloc=t;");
        }
        buf.blank();
    }

    fn emit_invisible_start(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        if let Some(init) = &automaton.initial_state {
            buf.line("__start [shape=none, label=\"\", width=0, height=0];");
            buf.line(&format!("__start -> {};", state_node_id(*init)));
            buf.blank();
        }
    }

    fn emit_states(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        for (_, state) in automaton.states.iter() {
            let nid = state_node_id(state.id);
            let label = dot_escape(&state_label(state));
            let mut attrs = vec![format!("label=\"{}\"", label)];

            // Shape
            if state.is_accepting && self.config.double_accepting {
                attrs.push("shape=doublecircle".into());
            } else {
                attrs.push("shape=circle".into());
            }

            // Colour
            if state.is_initial && self.config.color_initial {
                attrs.push("style=filled".into());
                attrs.push("fillcolor=\"#c8e6c9\"".into());
            }
            if state.is_error && self.config.color_error {
                attrs.push("style=filled".into());
                attrs.push("fillcolor=\"#ffcdd2\"".into());
                attrs.push("fontcolor=\"#b71c1c\"".into());
            }

            buf.line(&format!("{} [{}];", nid, attrs.join(", ")));
        }
        buf.blank();
    }

    fn emit_transitions(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        for (_, tr) in automaton.transitions.iter() {
            let src = state_node_id(tr.source);
            let tgt = state_node_id(tr.target);

            let mut label_parts: Vec<String> = Vec::new();

            if self.config.show_guards {
                let gl = guard_label(&tr.guard);
                if !gl.is_empty() {
                    label_parts.push(gl);
                }
            }

            if self.config.show_actions {
                let action_labels: Vec<String> = tr
                    .actions
                    .iter()
                    .map(action_label)
                    .filter(|s| !s.is_empty())
                    .collect();
                if !action_labels.is_empty() {
                    label_parts.push(format!("/ {}", action_labels.join(", ")));
                }
            }

            let label = dot_escape(&label_parts.join("\\n"));
            buf.line(&format!("{src} -> {tgt} [label=\"{label}\"];"));
        }
        buf.blank();
    }

    fn emit_clusters(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        let key = match &self.config.cluster_key {
            Some(k) => k.clone(),
            None => return,
        };

        let mut groups: HashMap<String, Vec<StateId>> = HashMap::new();
        let mut ungrouped: Vec<StateId> = Vec::new();

        for (_, state) in automaton.states.iter() {
            if let Some(val) = state.metadata.get(&key) {
                groups.entry(val.clone()).or_default().push(state.id);
            } else {
                ungrouped.push(state.id);
            }
        }

        for (idx, (group_name, state_ids)) in groups.iter().enumerate() {
            buf.line(&format!("subgraph cluster_{} {{", idx));
            buf.indent();
            buf.line(&format!("label=\"{}\";", dot_escape(group_name)));
            buf.line("style=dashed;");
            buf.line("color=\"#9e9e9e\";");
            for sid in state_ids {
                buf.line(&format!("{};", state_node_id(*sid)));
            }
            buf.dedent();
            buf.line("}");
            buf.blank();
        }
    }
}

impl Default for DotGraphGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator trait impl
// ---------------------------------------------------------------------------

impl CodeGenerator for DotGraphGenerator {
    fn generate(&self, automaton: &SpatialEventAutomaton) -> CodegenResult<String> {
        if automaton.states.is_empty() {
            return Err(CodegenError::Config(
                "cannot generate DOT for an empty automaton".into(),
            ));
        }

        let mut buf = CodeBuffer::new("  ");

        self.emit_header(&mut buf, automaton);
        self.emit_invisible_start(&mut buf, automaton);
        self.emit_clusters(&mut buf, automaton);
        self.emit_states(&mut buf, automaton);
        self.emit_transitions(&mut buf, automaton);

        buf.dedent();
        buf.line("}");

        Ok(buf.finish())
    }

    fn name(&self) -> &str {
        "Graphviz DOT"
    }

    fn file_extension(&self) -> &str {
        "dot"
    }
}

// ---------------------------------------------------------------------------
// Standalone helper: generate a DOT subgraph for parallel composition
// ---------------------------------------------------------------------------

/// Render multiple automata as a single DOT file with each automaton in its
/// own cluster subgraph.  Useful for visualising parallel composition.
pub fn generate_parallel_composition_dot(
    automata: &[(&str, &SpatialEventAutomaton)],
    config: &DotGraphConfig,
) -> CodegenResult<String> {
    let mut buf = CodeBuffer::new("  ");

    buf.line("digraph parallel_composition {");
    buf.indent();

    if config.left_to_right {
        buf.line("rankdir=LR;");
    }
    buf.line(&format!(
        "fontname=\"{}\"; fontsize={};",
        config.font, config.font_size
    ));
    buf.line(&format!(
        "node [fontname=\"{}\", fontsize={}];",
        config.font, config.font_size
    ));
    buf.line("label=\"Parallel Composition\"; labelloc=t;");
    buf.blank();

    for (idx, (name, aut)) in automata.iter().enumerate() {
        let prefix = format!("c{}", idx);
        buf.line(&format!("subgraph cluster_{} {{", idx));
        buf.indent();
        buf.line(&format!("label=\"{}\";", dot_escape(name)));
        buf.line("style=solid; color=\"#1565c0\";");

        // States
        for (_, state) in aut.states.iter() {
            let nid = format!("{}_{}", prefix, state.id.0);
            let label = dot_escape(&state_label(state));
            let mut attrs = vec![format!("label=\"{}\"", label)];
            if state.is_accepting {
                attrs.push("shape=doublecircle".into());
            } else {
                attrs.push("shape=circle".into());
            }
            if state.is_initial {
                attrs.push("style=filled".into());
                attrs.push("fillcolor=\"#c8e6c9\"".into());
            }
            if state.is_error {
                attrs.push("style=filled".into());
                attrs.push("fillcolor=\"#ffcdd2\"".into());
            }
            buf.line(&format!("{} [{}];", nid, attrs.join(", ")));
        }

        // Invisible start
        if let Some(init) = &aut.initial_state {
            let start = format!("{}_start", prefix);
            buf.line(&format!(
                "{start} [shape=none, label=\"\", width=0, height=0];"
            ));
            buf.line(&format!(
                "{start} -> {}_{};",
                prefix, init.0
            ));
        }

        // Transitions
        for (_, tr) in aut.transitions.iter() {
            let src = format!("{}_{}", prefix, tr.source.0);
            let tgt = format!("{}_{}", prefix, tr.target.0);
            let gl = guard_label(&tr.guard);
            buf.line(&format!(
                "{src} -> {tgt} [label=\"{}\"];",
                dot_escape(&gl)
            ));
        }

        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    buf.dedent();
    buf.line("}");

    Ok(buf.finish())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_automata::automaton::{
        AutomatonMetadata, AutomatonStatistics, State, Transition,
    };
    use choreo_automata::{
        Action, AutomatonKind, EntityId, EventKind, Guard, RegionId, Span,
        StateId, TransitionId,
    };
    use indexmap::IndexMap;

    fn three_state() -> SpatialEventAutomaton {
        let s0 = StateId(0);
        let s1 = StateId(1);
        let s2 = StateId(2);

        let mut states = IndexMap::new();
        let mut st0 = State::new(s0, "idle");
        st0.is_initial = true;
        let st1 = State::new(s1, "hover");
        let mut st2 = State::new(s2, "grabbed");
        st2.is_accepting = true;
        states.insert(s0, st0);
        states.insert(s1, st1);
        states.insert(s2, st2);

        let mut transitions = IndexMap::new();
        transitions.insert(
            TransitionId(0),
            Transition::new(
                TransitionId(0),
                s0,
                s1,
                Guard::Event(EventKind::GazeEnter),
                vec![],
            ),
        );
        transitions.insert(
            TransitionId(1),
            Transition::new(
                TransitionId(1),
                s1,
                s2,
                Guard::Event(EventKind::GrabStart),
                vec![Action::EmitEvent(EventKind::TouchStart)],
            ),
        );
        transitions.insert(
            TransitionId(2),
            Transition::new(
                TransitionId(2),
                s1,
                s0,
                Guard::Event(EventKind::GazeExit),
                vec![],
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
                name: "grab_interaction".into(),
                source_span: Some(Span::empty()),
                description: String::new(),
                statistics: AutomatonStatistics::default(),
                tags: Vec::new(),
            },
            kind: AutomatonKind::DFA,
            next_state_id: 3,
            next_transition_id: 3,
        }
    }

    #[test]
    fn test_generates_digraph() {
        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&three_state()).unwrap();
        assert!(dot.starts_with("digraph automaton {"));
        assert!(dot.contains('}'));
    }

    #[test]
    fn test_states_appear_as_nodes() {
        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&three_state()).unwrap();
        assert!(dot.contains("s0 ["));
        assert!(dot.contains("s1 ["));
        assert!(dot.contains("s2 ["));
    }

    #[test]
    fn test_transitions_appear_as_edges() {
        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&three_state()).unwrap();
        assert!(dot.contains("s0 -> s1"));
        assert!(dot.contains("s1 -> s2"));
        assert!(dot.contains("s1 -> s0"));
    }

    #[test]
    fn test_initial_state_colored() {
        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&three_state()).unwrap();
        // s0 is initial, should have green fill
        let s0_line = dot.lines().find(|l| l.contains("s0 [")).unwrap();
        assert!(s0_line.contains("#c8e6c9"));
    }

    #[test]
    fn test_accepting_double_circle() {
        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&three_state()).unwrap();
        let s2_line = dot.lines().find(|l| l.contains("s2 [")).unwrap();
        assert!(s2_line.contains("doublecircle"));
    }

    #[test]
    fn test_error_state_red() {
        let s0 = StateId(0);
        let mut states = IndexMap::new();
        let mut st0 = State::new(s0, "error");
        st0.is_initial = true;
        st0.is_error = true;
        states.insert(s0, st0);

        let aut = SpatialEventAutomaton {
            states,
            transitions: IndexMap::new(),
            initial_state: Some(s0),
            accepting_states: HashSet::new(),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata {
                name: "err".into(),
                source_span: None,
                description: String::new(),
                statistics: AutomatonStatistics::default(),
                tags: Vec::new(),
            },
            kind: AutomatonKind::DFA,
            next_state_id: 1,
            next_transition_id: 0,
        };

        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&aut).unwrap();
        assert!(dot.contains("#ffcdd2"));
    }

    #[test]
    fn test_invisible_start_node() {
        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&three_state()).unwrap();
        assert!(dot.contains("__start"));
        assert!(dot.contains("__start -> s0"));
    }

    #[test]
    fn test_guard_label_formatting() {
        assert_eq!(guard_label(&Guard::True), "");
        let g = Guard::Spatial(SpatialPredicate::Inside {
            entity: EntityId("hand".into()),
            region: RegionId("zone".into()),
        });
        assert_eq!(guard_label(&g), "hand ∈ zone");
    }

    #[test]
    fn test_show_actions_on_edge() {
        let gen = DotGraphGenerator::new();
        let dot = gen.generate(&three_state()).unwrap();
        // Transition t1 (s1->s2) has EmitEvent action
        let edge_line = dot.lines().find(|l| l.contains("s1 -> s2")).unwrap();
        assert!(edge_line.contains("emit("));
    }

    #[test]
    fn test_no_guards_option() {
        let gen = DotGraphGenerator::with_config(DotGraphConfig {
            show_guards: false,
            ..Default::default()
        });
        let dot = gen.generate(&three_state()).unwrap();
        // Event labels should not appear on edges
        assert!(!dot.contains("gaze_enter"));
    }

    #[test]
    fn test_cluster_grouping() {
        let s0 = StateId(0);
        let s1 = StateId(1);
        let mut states = IndexMap::new();
        let mut st0 = State::new(s0, "a");
        st0.is_initial = true;
        st0.metadata
            .insert("region".into(), "group1".into());
        let mut st1 = State::new(s1, "b");
        st1.metadata
            .insert("region".into(), "group1".into());
        states.insert(s0, st0);
        states.insert(s1, st1);

        let aut = SpatialEventAutomaton {
            states,
            transitions: IndexMap::new(),
            initial_state: Some(s0),
            accepting_states: HashSet::new(),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata {
                name: "cluster_test".into(),
                source_span: None,
                description: String::new(),
                statistics: AutomatonStatistics::default(),
                tags: Vec::new(),
            },
            kind: AutomatonKind::DFA,
            next_state_id: 2,
            next_transition_id: 0,
        };

        let gen = DotGraphGenerator::with_config(DotGraphConfig {
            cluster_key: Some("region".into()),
            ..Default::default()
        });
        let dot = gen.generate(&aut).unwrap();
        assert!(dot.contains("subgraph cluster_"));
        assert!(dot.contains("group1"));
    }

    #[test]
    fn test_parallel_composition_dot() {
        let aut1 = three_state();
        let aut2 = three_state();
        let automata = vec![("left", &aut1), ("right", &aut2)];
        let config = DotGraphConfig::default();
        let dot = generate_parallel_composition_dot(&automata, &config).unwrap();
        assert!(dot.contains("subgraph cluster_0"));
        assert!(dot.contains("subgraph cluster_1"));
        assert!(dot.contains("\"left\""));
        assert!(dot.contains("\"right\""));
    }

    #[test]
    fn test_file_extension() {
        let gen = DotGraphGenerator::new();
        assert_eq!(gen.file_extension(), "dot");
        assert_eq!(gen.name(), "Graphviz DOT");
    }

    #[test]
    fn test_empty_automaton_errors() {
        let gen = DotGraphGenerator::new();
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
        assert!(gen.generate(&empty).is_err());
    }
}
