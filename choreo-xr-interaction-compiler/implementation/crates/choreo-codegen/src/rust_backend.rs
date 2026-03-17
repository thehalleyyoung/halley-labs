//! Rust code generation backend.
//!
//! Produces a standalone Rust module containing a state enum, a transition
//! evaluator, guard helpers, action dispatch, and a complete `step` loop
//! that mirrors the compiled [`SpatialEventAutomaton`].

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

/// Options that customise the generated Rust source.
#[derive(Debug, Clone)]
pub struct RustCodegenConfig {
    /// Name used for the generated state enum and machine struct.
    pub machine_name: String,
    /// Emit `#[derive(Serialize, Deserialize)]` on generated types.
    pub serde_derives: bool,
    /// Generate `#![no_std]`-compatible code.
    pub no_std: bool,
    /// Include inline doc-comments.
    pub doc_comments: bool,
    /// Visibility prefix (`pub`, `pub(crate)`, or empty).
    pub visibility: String,
}

impl Default for RustCodegenConfig {
    fn default() -> Self {
        Self {
            machine_name: "Interaction".into(),
            serde_derives: false,
            no_std: false,
            doc_comments: true,
            visibility: "pub".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers – name sanitisation
// ---------------------------------------------------------------------------

fn sanitize_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else if ch == ' ' || ch == '-' {
            out.push('_');
        } else if i == 0 && ch.is_ascii_digit() {
            out.push('_');
            out.push(ch);
        }
    }
    if out.is_empty() {
        out.push_str("unnamed");
    }
    out
}

fn to_pascal_case(name: &str) -> String {
    sanitize_ident(name)
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().to_string() + &c.as_str().to_lowercase()
                }
            }
        })
        .collect()
}

fn state_variant_name(state: &State) -> String {
    if state.name.is_empty() {
        format!("S{}", state.id.0)
    } else {
        to_pascal_case(&state.name)
    }
}

// ---------------------------------------------------------------------------
// Guard codegen
// ---------------------------------------------------------------------------

fn guard_to_rust(guard: &Guard) -> String {
    match guard {
        Guard::True => "true".into(),
        Guard::False => "false".into(),
        Guard::Spatial(sp) => spatial_predicate_to_rust(sp),
        Guard::Temporal(tg) => temporal_guard_to_rust(tg),
        Guard::Event(ek) => format!("event_matches(event, \"{}\")", ek),
        Guard::And(gs) => {
            let parts: Vec<String> = gs.iter().map(|g| guard_to_rust(g)).collect();
            if parts.is_empty() {
                "true".into()
            } else {
                format!("({})", parts.join(" && "))
            }
        }
        Guard::Or(gs) => {
            let parts: Vec<String> = gs.iter().map(|g| guard_to_rust(g)).collect();
            if parts.is_empty() {
                "false".into()
            } else {
                format!("({})", parts.join(" || "))
            }
        }
        Guard::Not(g) => format!("!({})", guard_to_rust(g)),
    }
}

fn spatial_predicate_to_rust(sp: &SpatialPredicate) -> String {
    match sp {
        SpatialPredicate::Inside { entity, region } => {
            format!("scene.is_inside(\"{}\", \"{}\")", entity.0, region.0)
        }
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            threshold,
        } => format!(
            "scene.proximity(\"{}\", \"{}\") < {:.4}",
            entity_a.0, entity_b.0, threshold
        ),
        SpatialPredicate::GazeAt { entity, target } => {
            format!("scene.gaze_at(\"{}\", \"{}\")", entity.0, target.0)
        }
        SpatialPredicate::Contact { entity_a, entity_b } => {
            format!("scene.contact(\"{}\", \"{}\")", entity_a.0, entity_b.0)
        }
        SpatialPredicate::Grasping { hand, object } => {
            format!("scene.grasping(\"{}\", \"{}\")", hand.0, object.0)
        }
        SpatialPredicate::Not(inner) => {
            format!("!({})", spatial_predicate_to_rust(inner))
        }
        SpatialPredicate::And(preds) => {
            let parts: Vec<String> = preds.iter().map(|p| spatial_predicate_to_rust(p)).collect();
            format!("({})", parts.join(" && "))
        }
        SpatialPredicate::Or(preds) => {
            let parts: Vec<String> = preds.iter().map(|p| spatial_predicate_to_rust(p)).collect();
            format!("({})", parts.join(" || "))
        }
        SpatialPredicate::Named(id) => {
            format!("scene.predicate(\"{}\")", id.0)
        }
    }
}

fn temporal_guard_to_rust(tg: &TemporalGuardExpr) -> String {
    match tg {
        TemporalGuardExpr::TimerElapsed { timer, threshold } => {
            format!("timers.elapsed(\"{}\") >= {:.4}", timer.0, threshold.0)
        }
        TemporalGuardExpr::WithinInterval(iv) => {
            format!(
                "(current_time >= {:.4} && current_time <= {:.4})",
                iv.start.0, iv.end.0
            )
        }
        TemporalGuardExpr::Named(id) => {
            format!("temporal_pred(\"{}\")", id.0)
        }
        TemporalGuardExpr::And(exprs) => {
            let parts: Vec<String> = exprs.iter().map(|e| temporal_guard_to_rust(e)).collect();
            format!("({})", parts.join(" && "))
        }
        TemporalGuardExpr::Or(exprs) => {
            let parts: Vec<String> = exprs.iter().map(|e| temporal_guard_to_rust(e)).collect();
            format!("({})", parts.join(" || "))
        }
        TemporalGuardExpr::Not(e) => {
            format!("!({})", temporal_guard_to_rust(e))
        }
    }
}

// ---------------------------------------------------------------------------
// Action codegen
// ---------------------------------------------------------------------------

fn action_to_rust(action: &Action) -> String {
    match action {
        Action::StartTimer(id) => format!("timers.start(\"{}\");", id.0),
        Action::StopTimer(id) => format!("timers.stop(\"{}\");", id.0),
        Action::EmitEvent(ek) => format!("ctx.emit_event(\"{}\");", ek),
        Action::SetVar { var, value } => {
            format!("vars.set(\"{}\", {});", var.0, value_to_rust(value))
        }
        Action::PlayFeedback(fb) => format!("ctx.play_feedback(\"{}\");", fb),
        Action::Highlight { entity, style } => {
            format!("ctx.highlight(\"{}\", \"{}\");", entity.0, style)
        }
        Action::ClearHighlight(entity) => {
            format!("ctx.clear_highlight(\"{}\");", entity.0)
        }
        Action::MoveEntity { entity, target } => {
            format!(
                "ctx.move_entity(\"{}\", ({:.4}, {:.4}, {:.4}));",
                entity.0, target.x, target.y, target.z
            )
        }
        Action::Custom(s) => format!("ctx.custom(\"{}\");", s),
        Action::Noop => "/* noop */".into(),
    }
}

fn value_to_rust(v: &Value) -> String {
    match v {
        Value::Bool(b) => format!("{}", b),
        Value::Int(n) => format!("{}_i64", n),
        Value::Float(f) => format!("{:.6}_f64", f),
        Value::Str(s) => format!("\"{}\"", s),
    }
}

// ---------------------------------------------------------------------------
// RustCodeGenerator
// ---------------------------------------------------------------------------

/// Code generator that emits Rust source from a
/// [`SpatialEventAutomaton`].
pub struct RustCodeGenerator {
    pub config: RustCodegenConfig,
}

impl RustCodeGenerator {
    pub fn new() -> Self {
        Self {
            config: RustCodegenConfig::default(),
        }
    }

    pub fn with_config(config: RustCodegenConfig) -> Self {
        Self { config }
    }

    // -- State enum ---------------------------------------------------------

    fn emit_state_enum(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let vis = &self.config.visibility;
        let name = &self.config.machine_name;

        if self.config.doc_comments {
            buf.line(&format!("/// States of the `{}` state machine.", name));
        }

        let mut derives = vec!["Debug", "Clone", "Copy", "PartialEq", "Eq", "Hash"];
        if self.config.serde_derives {
            derives.push("Serialize");
            derives.push("Deserialize");
        }
        buf.line(&format!("#[derive({})]", derives.join(", ")));

        buf.open_brace(&format!("{vis} enum {name}State"));
        for (_, state) in automaton.states.iter() {
            let variant = state_variant_name(state);
            if self.config.doc_comments && !state.metadata.is_empty() {
                if let Some(desc) = state.metadata.get("description") {
                    buf.line(&format!("/// {}", desc));
                }
            }
            buf.line(&format!("{},", variant));
        }
        buf.close_brace();
        buf.blank();

        // is_accepting helper
        let accepting: Vec<String> = automaton
            .states
            .values()
            .filter(|s| s.is_accepting)
            .map(|s| format!("Self::{}", state_variant_name(s)))
            .collect();

        buf.open_brace(&format!("impl {name}State"));
        if self.config.doc_comments {
            buf.line("/// Returns `true` when this state is an accepting state.");
        }
        if accepting.is_empty() {
            buf.line(&format!("{vis} fn is_accepting(&self) -> bool {{ false }}"));
        } else {
            buf.line(&format!(
                "{vis} fn is_accepting(&self) -> bool {{ matches!(self, {}) }}",
                accepting.join(" | ")
            ));
        }
        buf.blank();

        let error_states: Vec<String> = automaton
            .states
            .values()
            .filter(|s| s.is_error)
            .map(|s| format!("Self::{}", state_variant_name(s)))
            .collect();
        if self.config.doc_comments {
            buf.line("/// Returns `true` when this state is an error state.");
        }
        if error_states.is_empty() {
            buf.line(&format!("{vis} fn is_error(&self) -> bool {{ false }}"));
        } else {
            buf.line(&format!(
                "{vis} fn is_error(&self) -> bool {{ matches!(self, {}) }}",
                error_states.join(" | ")
            ));
        }
        buf.close_brace();
        buf.blank();
    }

    // -- Event enum (from the alphabet) -------------------------------------

    fn emit_event_enum(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let vis = &self.config.visibility;
        let name = &self.config.machine_name;

        let alphabet = automaton.alphabet();
        if alphabet.is_empty() {
            return;
        }

        if self.config.doc_comments {
            buf.line(&format!(
                "/// Events accepted by the `{}` state machine.",
                name
            ));
        }

        let mut derives = vec!["Debug", "Clone", "PartialEq", "Eq", "Hash"];
        if self.config.serde_derives {
            derives.push("Serialize");
            derives.push("Deserialize");
        }
        buf.line(&format!("#[derive({})]", derives.join(", ")));
        buf.open_brace(&format!("{vis} enum {name}Event"));
        for ek in &alphabet {
            buf.line(&format!("{},", event_kind_variant(ek)));
        }
        buf.close_brace();
        buf.blank();
    }

    // -- Machine struct & impl ----------------------------------------------

    fn emit_machine_struct(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let vis = &self.config.visibility;
        let name = &self.config.machine_name;

        let initial = automaton
            .initial_state
            .as_ref()
            .and_then(|id| automaton.states.get(id))
            .map(|s| state_variant_name(s))
            .unwrap_or_else(|| "S0".into());

        if self.config.doc_comments {
            buf.line(&format!("/// Runtime instance of the `{}` state machine.", name));
        }
        buf.open_brace(&format!("{vis} struct {name}Machine"));
        buf.line(&format!("state: {name}State,"));
        buf.close_brace();
        buf.blank();

        buf.open_brace(&format!("impl {name}Machine"));

        // new()
        if self.config.doc_comments {
            buf.line("/// Create a new machine in its initial state.");
        }
        buf.open_brace(&format!("{vis} fn new() -> Self"));
        buf.line(&format!("Self {{ state: {name}State::{initial} }}"));
        buf.close_brace();
        buf.blank();

        // current_state()
        buf.line(&format!(
            "{vis} fn current_state(&self) -> {name}State {{ self.state }}"
        ));
        buf.blank();

        // is_accepting()
        buf.line(&format!(
            "{vis} fn is_accepting(&self) -> bool {{ self.state.is_accepting() }}"
        ));
        buf.blank();

        // step()
        self.emit_step_function(buf, automaton);

        buf.close_brace(); // impl end
        buf.blank();
    }

    // -- Step function with match arms --------------------------------------

    fn emit_step_function(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let vis = &self.config.visibility;
        let name = &self.config.machine_name;

        if self.config.doc_comments {
            buf.line("/// Advance the machine by processing the given event string.");
            buf.line("/// Returns `true` if a transition was taken.");
        }

        buf.open_brace(&format!("{vis} fn step(&mut self, event: &str) -> bool"));

        buf.open_brace("let next = match (self.state, event)");

        // Group transitions by source state for cleaner output.
        let mut by_source: HashMap<StateId, Vec<&Transition>> = HashMap::new();
        for (_, tr) in automaton.transitions.iter() {
            by_source.entry(tr.source).or_default().push(tr);
        }

        for (_, state) in automaton.states.iter() {
            if let Some(transitions) = by_source.get(&state.id) {
                let variant = state_variant_name(state);
                for tr in transitions {
                    let target = automaton
                        .states
                        .get(&tr.target)
                        .map(|s| state_variant_name(s))
                        .unwrap_or_else(|| format!("S{}", tr.target.0));

                    let event_pat = guard_event_pattern(&tr.guard);

                    if tr.guard.is_trivially_true() || matches!(tr.guard, Guard::Event(_)) {
                        buf.line(&format!(
                            "({name}State::{variant}, {event_pat}) => Some({name}State::{target}),"
                        ));
                    } else {
                        let guard_expr = guard_to_rust(&tr.guard);
                        buf.line(&format!(
                            "({name}State::{variant}, {event_pat}) if {guard_expr} => Some({name}State::{target}),",
                        ));
                    }
                }
            }
        }
        buf.line("_ => None,");
        buf.close_brace(); // match end
        buf.line(";");
        buf.blank();

        // Entry/exit actions
        buf.open_brace("if let Some(s) = next");
        buf.line("self.state = s;");
        buf.line("true");
        buf.close_brace();
        buf.open_brace("else");
        buf.line("false");
        buf.close_brace();

        buf.close_brace(); // fn step end
    }

    // -- Guard function codegen ---------------------------------------------

    fn emit_guard_helpers(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let name = &self.config.machine_name;

        // Collect all distinct guards
        let mut guard_fns: Vec<(String, String)> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (_, tr) in automaton.transitions.iter() {
            let code = guard_to_rust(&tr.guard);
            if code != "true" && code != "false" && seen.insert(code.clone()) {
                let fn_name = format!("guard_t{}", tr.id.0);
                guard_fns.push((fn_name, code));
            }
        }

        if guard_fns.is_empty() {
            return;
        }

        if self.config.doc_comments {
            buf.line(&format!(
                "/// Guard evaluation helpers for `{}`.",
                name
            ));
        }

        buf.open_brace(&format!("mod {}_guards", name.to_lowercase()));
        for (fn_name, expr) in &guard_fns {
            buf.line("#[inline]");
            buf.open_brace(&format!("pub fn {}() -> bool", fn_name));
            buf.line(expr);
            buf.close_brace();
        }
        buf.close_brace();
        buf.blank();
    }

    // -- Action dispatch ----------------------------------------------------

    fn emit_action_dispatch(
        &self,
        buf: &mut CodeBuffer,
        automaton: &SpatialEventAutomaton,
    ) {
        let name = &self.config.machine_name;

        let has_actions = automaton
            .transitions
            .values()
            .any(|t| !t.actions.is_empty())
            || automaton
                .states
                .values()
                .any(|s| !s.on_entry.is_empty() || !s.on_exit.is_empty());

        if !has_actions {
            return;
        }

        if self.config.doc_comments {
            buf.line(&format!(
                "/// Action dispatch for `{}` transitions.",
                name
            ));
        }

        buf.open_brace(&format!(
            "mod {}_actions",
            name.to_lowercase()
        ));

        // Per-transition action functions
        for (_, tr) in automaton.transitions.iter() {
            if tr.actions.is_empty() {
                continue;
            }
            buf.open_brace(&format!("pub fn transition_t{}()", tr.id.0));
            for action in &tr.actions {
                buf.line(&action_to_rust(action));
            }
            buf.close_brace();
        }

        // Per-state entry/exit actions
        for (_, state) in automaton.states.iter() {
            if !state.on_entry.is_empty() {
                buf.open_brace(&format!(
                    "pub fn on_enter_{}()",
                    sanitize_ident(&state.name).to_lowercase()
                ));
                for action in &state.on_entry {
                    buf.line(&action_to_rust(action));
                }
                buf.close_brace();
            }
            if !state.on_exit.is_empty() {
                buf.open_brace(&format!(
                    "pub fn on_exit_{}()",
                    sanitize_ident(&state.name).to_lowercase()
                ));
                for action in &state.on_exit {
                    buf.line(&action_to_rust(action));
                }
                buf.close_brace();
            }
        }

        buf.close_brace();
        buf.blank();
    }

    // -- Full state machine loop  -------------------------------------------

    fn emit_run_loop(
        &self,
        buf: &mut CodeBuffer,
        _automaton: &SpatialEventAutomaton,
    ) {
        let vis = &self.config.visibility;
        let name = &self.config.machine_name;

        if self.config.doc_comments {
            buf.line(&format!(
                "/// Run the `{}` machine in a loop, consuming events from an iterator.",
                name
            ));
        }

        buf.open_brace(&format!(
            "{vis} fn run_{machine}(events: impl IntoIterator<Item = impl AsRef<str>>) -> {name}State",
            vis = vis,
            machine = name.to_lowercase(),
            name = name,
        ));
        buf.line(&format!("let mut machine = {name}Machine::new();"));
        buf.open_brace("for event in events");
        buf.line("machine.step(event.as_ref());");
        buf.open_brace("if machine.current_state().is_error()");
        buf.line("break;");
        buf.close_brace();
        buf.close_brace();
        buf.line("machine.current_state()");
        buf.close_brace();
        buf.blank();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the event string pattern for a match arm from a guard.
fn guard_event_pattern(guard: &Guard) -> String {
    match guard {
        Guard::Event(ek) => format!("\"{}\"", ek),
        Guard::And(gs) => {
            for g in gs {
                if let Guard::Event(ek) = g {
                    return format!("\"{}\"", ek);
                }
            }
            "_".into()
        }
        _ => "_".into(),
    }
}

/// Convert an EventKind to a Rust enum variant name.
fn event_kind_variant(ek: &EventKind) -> String {
    match ek {
        EventKind::GazeEnter => "GazeEnter".into(),
        EventKind::GazeExit => "GazeExit".into(),
        EventKind::GazeDwell => "GazeDwell".into(),
        EventKind::GrabStart => "GrabStart".into(),
        EventKind::GrabEnd => "GrabEnd".into(),
        EventKind::TouchStart => "TouchStart".into(),
        EventKind::TouchEnd => "TouchEnd".into(),
        EventKind::ProximityEnter => "ProximityEnter".into(),
        EventKind::ProximityExit => "ProximityExit".into(),
        EventKind::GestureRecognised(g) => to_pascal_case(&format!("gesture_{}", g)),
        EventKind::ButtonPress(b) => to_pascal_case(&format!("press_{}", b)),
        EventKind::ButtonRelease(b) => to_pascal_case(&format!("release_{}", b)),
        EventKind::Timer(t) => to_pascal_case(&format!("timer_{}", t.0)),
        EventKind::Custom(c) => to_pascal_case(&format!("custom_{}", c)),
        EventKind::Epsilon => "Epsilon".into(),
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator impl
// ---------------------------------------------------------------------------

impl CodeGenerator for RustCodeGenerator {
    fn generate(&self, automaton: &SpatialEventAutomaton) -> CodegenResult<String> {
        if automaton.states.is_empty() {
            return Err(CodegenError::Config(
                "cannot generate code for an empty automaton".into(),
            ));
        }

        let mut buf = CodeBuffer::new("    ");

        // Header
        buf.line("// Auto-generated by choreo-codegen (Rust backend)");
        buf.line("// DO NOT EDIT – regenerate from the interaction specification.");
        buf.blank();

        if self.config.no_std {
            buf.line("#![no_std]");
            buf.blank();
        }

        if self.config.serde_derives {
            buf.line("use serde::{Serialize, Deserialize};");
            buf.blank();
        }

        self.emit_state_enum(&mut buf, automaton);
        self.emit_event_enum(&mut buf, automaton);
        self.emit_machine_struct(&mut buf, automaton);
        self.emit_guard_helpers(&mut buf, automaton);
        self.emit_action_dispatch(&mut buf, automaton);
        self.emit_run_loop(&mut buf, automaton);

        Ok(buf.finish())
    }

    fn name(&self) -> &str {
        "Rust"
    }

    fn file_extension(&self) -> &str {
        "rs"
    }
}

impl Default for RustCodeGenerator {
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
    use choreo_automata::automaton::{
        AutomatonMetadata, AutomatonStatistics, SpatialEventAutomaton, State, Transition,
    };
    use choreo_automata::{
        Action, AutomatonKind, EventKind, Guard, Span, StateId, TransitionId,
    };
    use indexmap::IndexMap;
    use std::collections::{HashMap, HashSet};

    /// Build a minimal two-state automaton for testing.
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
    fn test_generate_contains_state_enum() {
        let gen = RustCodeGenerator::new();
        let code = gen.generate(&two_state_automaton()).unwrap();
        assert!(code.contains("enum InteractionState"));
        assert!(code.contains("Idle"));
        assert!(code.contains("Active"));
    }

    #[test]
    fn test_generate_contains_machine_struct() {
        let gen = RustCodeGenerator::new();
        let code = gen.generate(&two_state_automaton()).unwrap();
        assert!(code.contains("struct InteractionMachine"));
        assert!(code.contains("fn new()"));
        assert!(code.contains("fn step("));
    }

    #[test]
    fn test_generate_contains_run_loop() {
        let gen = RustCodeGenerator::new();
        let code = gen.generate(&two_state_automaton()).unwrap();
        assert!(code.contains("fn run_interaction("));
    }

    #[test]
    fn test_serde_derives() {
        let gen = RustCodeGenerator::with_config(RustCodegenConfig {
            serde_derives: true,
            ..Default::default()
        });
        let code = gen.generate(&two_state_automaton()).unwrap();
        assert!(code.contains("Serialize"));
        assert!(code.contains("Deserialize"));
    }

    #[test]
    fn test_no_std_flag() {
        let gen = RustCodeGenerator::with_config(RustCodegenConfig {
            no_std: true,
            ..Default::default()
        });
        let code = gen.generate(&two_state_automaton()).unwrap();
        assert!(code.contains("#![no_std]"));
    }

    #[test]
    fn test_empty_automaton_errors() {
        let gen = RustCodeGenerator::new();
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
    fn test_guard_to_rust_trivial() {
        assert_eq!(guard_to_rust(&Guard::True), "true");
        assert_eq!(guard_to_rust(&Guard::False), "false");
    }

    #[test]
    fn test_guard_to_rust_event() {
        let g = Guard::Event(EventKind::GrabStart);
        let code = guard_to_rust(&g);
        assert!(code.contains("event_matches"));
        assert!(code.contains("grab_start"));
    }

    #[test]
    fn test_guard_to_rust_and() {
        let g = Guard::And(vec![Guard::True, Guard::False]);
        let code = guard_to_rust(&g);
        assert!(code.contains("&&"));
    }

    #[test]
    fn test_guard_to_rust_or() {
        let g = Guard::Or(vec![Guard::True, Guard::False]);
        let code = guard_to_rust(&g);
        assert!(code.contains("||"));
    }

    #[test]
    fn test_guard_to_rust_not() {
        let g = Guard::Not(Box::new(Guard::True));
        assert_eq!(guard_to_rust(&g), "!(true)");
    }

    #[test]
    fn test_action_to_rust_variants() {
        use choreo_automata::{TimerId, VarId};
        assert!(action_to_rust(&Action::StartTimer(TimerId("t1".into()))).contains("start"));
        assert!(action_to_rust(&Action::StopTimer(TimerId("t1".into()))).contains("stop"));
        assert!(action_to_rust(&Action::EmitEvent(EventKind::GazeDwell)).contains("emit_event"));
        assert!(action_to_rust(&Action::SetVar {
            var: VarId("x".into()),
            value: Value::Int(42),
        })
        .contains("42"));
        assert!(action_to_rust(&Action::Custom("beep".into())).contains("beep"));
        assert!(action_to_rust(&Action::Noop).contains("noop"));
    }

    #[test]
    fn test_sanitize_ident() {
        assert_eq!(sanitize_ident("hello world"), "hello_world");
        assert_eq!(sanitize_ident("foo-bar"), "foo_bar");
        assert_eq!(sanitize_ident(""), "unnamed");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("idle_state"), "IdleState");
        assert_eq!(to_pascal_case("grab-start"), "GrabStart");
    }

    #[test]
    fn test_multiple_transitions_generate() {
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

        let t0 = TransitionId(0);
        let t1 = TransitionId(1);
        let mut transitions = IndexMap::new();
        transitions.insert(
            t0,
            Transition::new(t0, s0, s1, Guard::Event(EventKind::GazeEnter), vec![]),
        );
        transitions.insert(
            t1,
            Transition::new(t1, s1, s2, Guard::Event(EventKind::GrabStart), vec![]),
        );

        let aut = SpatialEventAutomaton {
            states,
            transitions,
            initial_state: Some(s0),
            accepting_states: HashSet::from([s2]),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata {
                name: "multi".into(),
                source_span: None,
                description: String::new(),
                statistics: AutomatonStatistics::default(),
                tags: Vec::new(),
            },
            kind: AutomatonKind::DFA,
            next_state_id: 3,
            next_transition_id: 2,
        };

        let gen = RustCodeGenerator::new();
        let code = gen.generate(&aut).unwrap();
        assert!(code.contains("Idle"));
        assert!(code.contains("Hover"));
        assert!(code.contains("Grabbed"));
    }

    #[test]
    fn test_file_extension() {
        let gen = RustCodeGenerator::new();
        assert_eq!(gen.file_extension(), "rs");
        assert_eq!(gen.name(), "Rust");
    }
}
