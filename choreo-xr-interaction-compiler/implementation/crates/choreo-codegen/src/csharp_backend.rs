//! C# code generation backend targeting Unity / MRTK.
//!
//! Generates a MonoBehaviour-derived class that implements the compiled
//! state machine with MRTK-style event handlers and a Unity `Update` loop
//! for guard evaluation.

use std::collections::HashMap;

use choreo_automata::automaton::{SpatialEventAutomaton, State, Transition};
use choreo_automata::{
    Action, EventKind, Guard, SpatialPredicate, StateId, TemporalGuardExpr, Value,
};

use crate::template::CodeBuffer;
use crate::{CodeGenerator, CodegenError, CodegenResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Options for the generated C# source.
#[derive(Debug, Clone)]
pub struct CSharpCodegenConfig {
    pub class_name: String,
    /// C# namespace for the generated class.
    pub namespace: Option<String>,
    /// Generate MonoBehaviour base class.
    pub mono_behaviour: bool,
    /// Generate MRTK event handler stubs.
    pub mrtk_events: bool,
    /// Use Unity-style `Update` loop for guard polling.
    pub update_loop: bool,
}

impl Default for CSharpCodegenConfig {
    fn default() -> Self {
        Self {
            class_name: "InteractionStateMachine".into(),
            namespace: None,
            mono_behaviour: true,
            mrtk_events: true,
            update_loop: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cs_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else if ch == ' ' || ch == '-' {
            out.push('_');
        }
    }
    if out.is_empty() {
        "Unnamed".into()
    } else {
        // PascalCase first char
        let mut chars = out.chars();
        chars
            .next()
            .map(|c| c.to_uppercase().to_string())
            .unwrap_or_default()
            + chars.as_str()
    }
}

fn cs_pascal(name: &str) -> String {
    name.split(|c: char| c == '_' || c == ' ' || c == '-')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().to_string() + c.as_str(),
            }
        })
        .collect()
}

fn state_cs_name(state: &State) -> String {
    if state.name.is_empty() {
        format!("S{}", state.id.0)
    } else {
        cs_pascal(&state.name)
    }
}

fn guard_to_cs(guard: &Guard) -> String {
    match guard {
        Guard::True => "true".into(),
        Guard::False => "false".into(),
        Guard::Spatial(sp) => spatial_pred_cs(sp),
        Guard::Temporal(tg) => temporal_guard_cs(tg),
        Guard::Event(ek) => format!("eventName == \"{}\"", ek),
        Guard::And(gs) => {
            let parts: Vec<String> = gs.iter().map(guard_to_cs).collect();
            if parts.is_empty() {
                "true".into()
            } else {
                format!("({})", parts.join(" && "))
            }
        }
        Guard::Or(gs) => {
            let parts: Vec<String> = gs.iter().map(guard_to_cs).collect();
            if parts.is_empty() {
                "false".into()
            } else {
                format!("({})", parts.join(" || "))
            }
        }
        Guard::Not(g) => format!("!({})", guard_to_cs(g)),
    }
}

fn spatial_pred_cs(sp: &SpatialPredicate) -> String {
    match sp {
        SpatialPredicate::Inside { entity, region } => {
            format!("scene.IsInside(\"{}\", \"{}\")", entity.0, region.0)
        }
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            threshold,
        } => format!(
            "scene.Distance(\"{}\", \"{}\") < {:.4}f",
            entity_a.0, entity_b.0, threshold
        ),
        SpatialPredicate::GazeAt { entity, target } => {
            format!("scene.IsGazingAt(\"{}\", \"{}\")", entity.0, target.0)
        }
        SpatialPredicate::Contact { entity_a, entity_b } => {
            format!("scene.AreInContact(\"{}\", \"{}\")", entity_a.0, entity_b.0)
        }
        SpatialPredicate::Grasping { hand, object } => {
            format!("scene.IsGrasping(\"{}\", \"{}\")", hand.0, object.0)
        }
        SpatialPredicate::Not(inner) => format!("!({})", spatial_pred_cs(inner)),
        SpatialPredicate::And(ps) => {
            let parts: Vec<String> = ps.iter().map(spatial_pred_cs).collect();
            format!("({})", parts.join(" && "))
        }
        SpatialPredicate::Or(ps) => {
            let parts: Vec<String> = ps.iter().map(spatial_pred_cs).collect();
            format!("({})", parts.join(" || "))
        }
        SpatialPredicate::Named(id) => format!("scene.EvaluatePredicate(\"{}\")", id.0),
    }
}

fn temporal_guard_cs(tg: &TemporalGuardExpr) -> String {
    match tg {
        TemporalGuardExpr::TimerElapsed { timer, threshold } => {
            format!("timers.HasElapsed(\"{}\", {:.4}f)", timer.0, threshold.0)
        }
        TemporalGuardExpr::WithinInterval(iv) => {
            format!(
                "(Time.time >= {:.4}f && Time.time <= {:.4}f)",
                iv.start.0, iv.end.0
            )
        }
        TemporalGuardExpr::Named(id) => format!("EvalTemporal(\"{}\")", id.0),
        TemporalGuardExpr::And(es) => {
            let parts: Vec<String> = es.iter().map(temporal_guard_cs).collect();
            format!("({})", parts.join(" && "))
        }
        TemporalGuardExpr::Or(es) => {
            let parts: Vec<String> = es.iter().map(temporal_guard_cs).collect();
            format!("({})", parts.join(" || "))
        }
        TemporalGuardExpr::Not(e) => format!("!({})", temporal_guard_cs(e)),
    }
}

fn action_to_cs(action: &Action) -> String {
    match action {
        Action::StartTimer(id) => format!("timers.Start(\"{}\");", id.0),
        Action::StopTimer(id) => format!("timers.Stop(\"{}\");", id.0),
        Action::EmitEvent(ek) => format!("EmitEvent(\"{}\");", ek),
        Action::SetVar { var, value } => format!("vars[\"{}\"] = {};", var.0, value_to_cs(value)),
        Action::PlayFeedback(fb) => format!("PlayFeedback(\"{}\");", fb),
        Action::Highlight { entity, style } => {
            format!("Highlight(\"{}\", \"{}\");", entity.0, style)
        }
        Action::ClearHighlight(entity) => format!("ClearHighlight(\"{}\");", entity.0),
        Action::MoveEntity { entity, target } => format!(
            "MoveEntity(\"{}\", new Vector3({:.4}f, {:.4}f, {:.4}f));",
            entity.0, target.x, target.y, target.z
        ),
        Action::Custom(s) => format!("CustomAction(\"{}\");", s),
        Action::Noop => "// noop".into(),
    }
}

fn value_to_cs(v: &Value) -> String {
    match v {
        Value::Bool(b) => format!("{}", b),
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => format!("{:.6}f", f),
        Value::Str(s) => format!("\"{}\"", s),
    }
}

// ---------------------------------------------------------------------------
// CSharpCodeGenerator
// ---------------------------------------------------------------------------

/// Generates a Unity / MRTK C# class implementing the state machine.
pub struct CSharpCodeGenerator {
    pub config: CSharpCodegenConfig,
}

impl CSharpCodeGenerator {
    pub fn new() -> Self {
        Self {
            config: CSharpCodegenConfig::default(),
        }
    }

    pub fn with_config(config: CSharpCodegenConfig) -> Self {
        Self { config }
    }

    fn emit_enum(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        buf.line(&format!("public enum {}State", self.config.class_name));
        buf.line("{");
        buf.indent();
        for (_, state) in automaton.states.iter() {
            buf.line(&format!("{},", state_cs_name(state)));
        }
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_class(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let cls = &self.config.class_name;
        let base = if self.config.mono_behaviour {
            " : MonoBehaviour"
        } else {
            ""
        };

        buf.line(&format!("public class {cls}{base}"));
        buf.line("{");
        buf.indent();

        // Fields
        buf.line(&format!(
            "public {cls}State CurrentState {{ get; private set; }}"
        ));
        buf.blank();

        let initial = automaton
            .initial_state
            .as_ref()
            .and_then(|id| automaton.states.get(id))
            .map(|s| state_cs_name(s))
            .unwrap_or_else(|| "S0".into());

        // Start / constructor
        if self.config.mono_behaviour {
            buf.line("void Start()");
            buf.line("{");
            buf.indent();
            buf.line(&format!("CurrentState = {cls}State.{initial};"));
            buf.line("OnEnterState(CurrentState);");
            buf.dedent();
            buf.line("}");
        } else {
            buf.line(&format!("public {cls}()"));
            buf.line("{");
            buf.indent();
            buf.line(&format!("CurrentState = {cls}State.{initial};"));
            buf.dedent();
            buf.line("}");
        }
        buf.blank();

        // ProcessEvent
        self.emit_process_event(buf, automaton);

        // EvaluateTransition
        self.emit_evaluate_transition(buf, automaton);

        // Entry/exit hooks
        self.emit_entry_exit_hooks(buf, automaton);

        // Update loop
        if self.config.update_loop && self.config.mono_behaviour {
            self.emit_update_loop(buf, automaton);
        }

        // MRTK event handlers
        if self.config.mrtk_events {
            self.emit_mrtk_handlers(buf, automaton);
        }

        buf.dedent();
        buf.line("}");
    }

    fn emit_process_event(
        &self,
        buf: &mut CodeBuffer,
        _automaton: &SpatialEventAutomaton,
    ) {
        let cls = &self.config.class_name;
        buf.line("public bool ProcessEvent(string eventName)");
        buf.line("{");
        buf.indent();
        buf.line("var next = EvaluateTransition(eventName);");
        buf.line("if (next.HasValue)");
        buf.line("{");
        buf.indent();
        buf.line("OnExitState(CurrentState);");
        buf.line(&format!("{cls}State prev = CurrentState;"));
        buf.line("CurrentState = next.Value;");
        buf.line("OnEnterState(CurrentState);");
        buf.line("return true;");
        buf.dedent();
        buf.line("}");
        buf.line("return false;");
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_evaluate_transition(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let cls = &self.config.class_name;

        buf.line(&format!(
            "private {cls}State? EvaluateTransition(string eventName)"
        ));
        buf.line("{");
        buf.indent();
        buf.line("switch (CurrentState)");
        buf.line("{");
        buf.indent();

        let mut by_source: HashMap<StateId, Vec<&Transition>> = HashMap::new();
        for (_, tr) in automaton.transitions.iter() {
            by_source.entry(tr.source).or_default().push(tr);
        }

        for (_, state) in automaton.states.iter() {
            buf.line(&format!(
                "case {cls}State.{name}:",
                name = state_cs_name(state)
            ));
            buf.indent();
            if let Some(transitions) = by_source.get(&state.id) {
                for tr in transitions {
                    let target_name = automaton
                        .states
                        .get(&tr.target)
                        .map(|s| state_cs_name(s))
                        .unwrap_or_else(|| format!("S{}", tr.target.0));

                    let guard_code = guard_to_cs(&tr.guard);
                    if guard_code == "true" {
                        buf.line(&format!("return {cls}State.{target_name};"));
                    } else {
                        buf.line(&format!("if ({guard_code})"));
                        buf.indent();
                        buf.line(&format!("return {cls}State.{target_name};"));
                        buf.dedent();
                    }
                }
            }
            buf.line("break;");
            buf.dedent();
        }

        buf.dedent();
        buf.line("}");
        buf.line("return null;");
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_entry_exit_hooks(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let cls = &self.config.class_name;

        // OnEnterState
        buf.line(&format!("private void OnEnterState({cls}State state)"));
        buf.line("{");
        buf.indent();
        buf.line("switch (state)");
        buf.line("{");
        buf.indent();
        for (_, state) in automaton.states.iter() {
            if !state.on_entry.is_empty() {
                buf.line(&format!(
                    "case {cls}State.{name}:",
                    name = state_cs_name(state)
                ));
                buf.indent();
                for action in &state.on_entry {
                    buf.line(&action_to_cs(action));
                }
                buf.line("break;");
                buf.dedent();
            }
        }
        buf.dedent();
        buf.line("}");
        buf.dedent();
        buf.line("}");
        buf.blank();

        // OnExitState
        buf.line(&format!("private void OnExitState({cls}State state)"));
        buf.line("{");
        buf.indent();
        buf.line("switch (state)");
        buf.line("{");
        buf.indent();
        for (_, state) in automaton.states.iter() {
            if !state.on_exit.is_empty() {
                buf.line(&format!(
                    "case {cls}State.{name}:",
                    name = state_cs_name(state)
                ));
                buf.indent();
                for action in &state.on_exit {
                    buf.line(&action_to_cs(action));
                }
                buf.line("break;");
                buf.dedent();
            }
        }
        buf.dedent();
        buf.line("}");
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_update_loop(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        // Collect transitions with non-event, non-trivial guards (spatial/temporal)
        let polling_transitions: Vec<&Transition> = automaton
            .transitions
            .values()
            .filter(|t| has_polled_guard(&t.guard))
            .collect();

        if polling_transitions.is_empty() {
            return;
        }

        let cls = &self.config.class_name;

        buf.line("void Update()");
        buf.line("{");
        buf.indent();
        buf.line("// Poll spatial/temporal guards each frame");
        buf.line("switch (CurrentState)");
        buf.line("{");
        buf.indent();

        let mut by_source: HashMap<StateId, Vec<&&Transition>> = HashMap::new();
        for tr in &polling_transitions {
            by_source.entry(tr.source).or_default().push(tr);
        }

        for (sid, trs) in &by_source {
            let sname = automaton
                .states
                .get(sid)
                .map(|s| state_cs_name(s))
                .unwrap_or_else(|| format!("S{}", sid.0));
            buf.line(&format!("case {cls}State.{sname}:"));
            buf.indent();
            for tr in trs {
                let target_name = automaton
                    .states
                    .get(&tr.target)
                    .map(|s| state_cs_name(s))
                    .unwrap_or_else(|| format!("S{}", tr.target.0));
                let guard_code = guard_to_cs(&tr.guard);
                buf.line(&format!("if ({guard_code})"));
                buf.line("{");
                buf.indent();
                buf.line(&format!(
                    "OnExitState(CurrentState); CurrentState = {cls}State.{target_name}; OnEnterState(CurrentState);"
                ));
                buf.line("return;");
                buf.dedent();
                buf.line("}");
            }
            buf.line("break;");
            buf.dedent();
        }

        buf.dedent();
        buf.line("}");
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_mrtk_handlers(
        &self,
        buf: &mut CodeBuffer,
        _automaton: &SpatialEventAutomaton,
    ) {
        buf.line("// --- MRTK Event Handlers ---");
        buf.blank();

        let handlers = [
            ("OnFocusEnter", "gaze_enter"),
            ("OnFocusExit", "gaze_exit"),
            ("OnPointerDown", "touch_start"),
            ("OnPointerUp", "touch_end"),
            ("OnManipulationStarted", "grab_start"),
            ("OnManipulationEnded", "grab_end"),
        ];

        for (method, event) in &handlers {
            buf.line(&format!("public void {method}()"));
            buf.line("{");
            buf.indent();
            buf.line(&format!("ProcessEvent(\"{event}\");"));
            buf.dedent();
            buf.line("}");
            buf.blank();
        }
    }
}

/// Returns `true` if the guard contains spatial or temporal sub-guards
/// that should be polled in `Update()`.
fn has_polled_guard(guard: &Guard) -> bool {
    match guard {
        Guard::Spatial(_) | Guard::Temporal(_) => true,
        Guard::And(gs) | Guard::Or(gs) => gs.iter().any(has_polled_guard),
        Guard::Not(g) => has_polled_guard(g),
        _ => false,
    }
}

impl Default for CSharpCodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator trait impl
// ---------------------------------------------------------------------------

impl CodeGenerator for CSharpCodeGenerator {
    fn generate(&self, automaton: &SpatialEventAutomaton) -> CodegenResult<String> {
        if automaton.states.is_empty() {
            return Err(CodegenError::Config(
                "cannot generate C# for an empty automaton".into(),
            ));
        }

        let mut buf = CodeBuffer::new("    ");

        buf.line("// Auto-generated by choreo-codegen (C# backend)");
        buf.line("// DO NOT EDIT – regenerate from the interaction specification.");
        buf.blank();
        buf.line("using System;");
        buf.line("using System.Collections.Generic;");
        if self.config.mono_behaviour {
            buf.line("using UnityEngine;");
        }
        if self.config.mrtk_events {
            buf.line("using Microsoft.MixedReality.Toolkit.Input;");
            buf.line("using Microsoft.MixedReality.Toolkit.UI;");
        }
        buf.blank();

        if let Some(ns) = &self.config.namespace {
            buf.line(&format!("namespace {ns}"));
            buf.line("{");
            buf.indent();
        }

        self.emit_enum(&mut buf, automaton);
        self.emit_class(&mut buf, automaton);

        if self.config.namespace.is_some() {
            buf.dedent();
            buf.line("}");
        }

        Ok(buf.finish())
    }

    fn name(&self) -> &str {
        "C#"
    }

    fn file_extension(&self) -> &str {
        "cs"
    }
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
        Action, AutomatonKind, EventKind, Guard, Span, StateId, TransitionId,
    };
    use indexmap::IndexMap;
    use std::collections::HashSet;

    fn two_state() -> SpatialEventAutomaton {
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
                description: String::new(),
                statistics: AutomatonStatistics::default(),
                tags: Vec::new(),
            },
            kind: AutomatonKind::DFA,
            next_state_id: 2,
            next_transition_id: 1,
        }
    }

    #[test]
    fn test_generates_class() {
        let gen = CSharpCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("public class InteractionStateMachine"));
        assert!(code.contains("enum InteractionStateMachineState"));
    }

    #[test]
    fn test_generates_using_directives() {
        let gen = CSharpCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("using UnityEngine;"));
        assert!(code.contains("using Microsoft.MixedReality.Toolkit"));
    }

    #[test]
    fn test_generates_states() {
        let gen = CSharpCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("Idle"));
        assert!(code.contains("Active"));
    }

    #[test]
    fn test_generates_process_event() {
        let gen = CSharpCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("ProcessEvent(string eventName)"));
    }

    #[test]
    fn test_mrtk_handlers_present() {
        let gen = CSharpCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("OnFocusEnter"));
        assert!(code.contains("OnManipulationStarted"));
    }

    #[test]
    fn test_no_mrtk() {
        let gen = CSharpCodeGenerator::with_config(CSharpCodegenConfig {
            mrtk_events: false,
            ..Default::default()
        });
        let code = gen.generate(&two_state()).unwrap();
        assert!(!code.contains("OnFocusEnter"));
    }

    #[test]
    fn test_namespace_wrapping() {
        let gen = CSharpCodeGenerator::with_config(CSharpCodegenConfig {
            namespace: Some("Choreo.Generated".into()),
            ..Default::default()
        });
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("namespace Choreo.Generated"));
    }

    #[test]
    fn test_guard_to_cs_basic() {
        assert_eq!(guard_to_cs(&Guard::True), "true");
        assert_eq!(guard_to_cs(&Guard::False), "false");
    }

    #[test]
    fn test_guard_to_cs_and_or() {
        let g = Guard::And(vec![Guard::True, Guard::False]);
        assert!(guard_to_cs(&g).contains("&&"));
        let g = Guard::Or(vec![Guard::True, Guard::False]);
        assert!(guard_to_cs(&g).contains("||"));
    }

    #[test]
    fn test_file_extension() {
        let gen = CSharpCodeGenerator::new();
        assert_eq!(gen.file_extension(), "cs");
        assert_eq!(gen.name(), "C#");
    }

    #[test]
    fn test_empty_automaton_errors() {
        let gen = CSharpCodeGenerator::new();
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

    #[test]
    fn test_with_spatial_guard_generates_update() {
        use choreo_automata::{EntityId, RegionId};
        let s0 = StateId(0);
        let s1 = StateId(1);

        let mut states = IndexMap::new();
        let mut st0 = State::new(s0, "outside");
        st0.is_initial = true;
        let st1 = State::new(s1, "inside");
        states.insert(s0, st0);
        states.insert(s1, st1);

        let mut transitions = IndexMap::new();
        transitions.insert(
            TransitionId(0),
            Transition::new(
                TransitionId(0),
                s0,
                s1,
                Guard::Spatial(SpatialPredicate::Inside {
                    entity: EntityId("hand".into()),
                    region: RegionId("zone".into()),
                }),
                vec![],
            ),
        );

        let aut = SpatialEventAutomaton {
            states,
            transitions,
            initial_state: Some(s0),
            accepting_states: HashSet::new(),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata {
                name: "spatial_test".into(),
                source_span: None,
                description: String::new(),
                statistics: AutomatonStatistics::default(),
                tags: Vec::new(),
            },
            kind: AutomatonKind::DFA,
            next_state_id: 2,
            next_transition_id: 1,
        };

        let gen = CSharpCodeGenerator::new();
        let code = gen.generate(&aut).unwrap();
        assert!(code.contains("void Update()"));
        assert!(code.contains("IsInside"));
    }
}
