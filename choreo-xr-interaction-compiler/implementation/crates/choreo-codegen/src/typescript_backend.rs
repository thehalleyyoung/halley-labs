//! TypeScript code generation backend targeting WebXR.
//!
//! Produces a TypeScript module with a state enum, a machine class,
//! promise-based async event processing, and WebXR session integration stubs.

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

/// Options for the generated TypeScript source.
#[derive(Debug, Clone)]
pub struct TypeScriptCodegenConfig {
    pub class_name: String,
    /// Emit WebXR session lifecycle hooks.
    pub webxr_hooks: bool,
    /// Use `async` / `await` patterns in generated step function.
    pub async_step: bool,
    /// Generate JSDoc comments.
    pub jsdoc: bool,
}

impl Default for TypeScriptCodegenConfig {
    fn default() -> Self {
        Self {
            class_name: "InteractionMachine".into(),
            webxr_hooks: true,
            async_step: true,
            jsdoc: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ts_pascal(name: &str) -> String {
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

fn state_ts_name(state: &State) -> String {
    if state.name.is_empty() {
        format!("S{}", state.id.0)
    } else {
        ts_pascal(&state.name)
    }
}

fn guard_to_ts(guard: &Guard) -> String {
    match guard {
        Guard::True => "true".into(),
        Guard::False => "false".into(),
        Guard::Spatial(sp) => spatial_pred_ts(sp),
        Guard::Temporal(tg) => temporal_guard_ts(tg),
        Guard::Event(ek) => format!("event === \"{}\"", ek),
        Guard::And(gs) => {
            let parts: Vec<String> = gs.iter().map(guard_to_ts).collect();
            if parts.is_empty() {
                "true".into()
            } else {
                format!("({})", parts.join(" && "))
            }
        }
        Guard::Or(gs) => {
            let parts: Vec<String> = gs.iter().map(guard_to_ts).collect();
            if parts.is_empty() {
                "false".into()
            } else {
                format!("({})", parts.join(" || "))
            }
        }
        Guard::Not(g) => format!("!({})", guard_to_ts(g)),
    }
}

fn spatial_pred_ts(sp: &SpatialPredicate) -> String {
    match sp {
        SpatialPredicate::Inside { entity, region } => {
            format!("this.scene.isInside(\"{}\", \"{}\")", entity.0, region.0)
        }
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            threshold,
        } => format!(
            "this.scene.distance(\"{}\", \"{}\") < {:.4}",
            entity_a.0, entity_b.0, threshold
        ),
        SpatialPredicate::GazeAt { entity, target } => {
            format!("this.scene.gazeAt(\"{}\", \"{}\")", entity.0, target.0)
        }
        SpatialPredicate::Contact { entity_a, entity_b } => {
            format!(
                "this.scene.contact(\"{}\", \"{}\")",
                entity_a.0, entity_b.0
            )
        }
        SpatialPredicate::Grasping { hand, object } => {
            format!("this.scene.grasping(\"{}\", \"{}\")", hand.0, object.0)
        }
        SpatialPredicate::Not(inner) => format!("!({})", spatial_pred_ts(inner)),
        SpatialPredicate::And(ps) => {
            let parts: Vec<String> = ps.iter().map(spatial_pred_ts).collect();
            format!("({})", parts.join(" && "))
        }
        SpatialPredicate::Or(ps) => {
            let parts: Vec<String> = ps.iter().map(spatial_pred_ts).collect();
            format!("({})", parts.join(" || "))
        }
        SpatialPredicate::Named(id) => {
            format!("this.scene.evaluatePredicate(\"{}\")", id.0)
        }
    }
}

fn temporal_guard_ts(tg: &TemporalGuardExpr) -> String {
    match tg {
        TemporalGuardExpr::TimerElapsed { timer, threshold } => {
            format!(
                "this.timers.get(\"{}\")! >= {:.4}",
                timer.0, threshold.0
            )
        }
        TemporalGuardExpr::WithinInterval(iv) => {
            format!(
                "(performance.now() / 1000 >= {:.4} && performance.now() / 1000 <= {:.4})",
                iv.start.0, iv.end.0
            )
        }
        TemporalGuardExpr::Named(id) => {
            format!("this.evalTemporal(\"{}\")", id.0)
        }
        TemporalGuardExpr::And(es) => {
            let parts: Vec<String> = es.iter().map(temporal_guard_ts).collect();
            format!("({})", parts.join(" && "))
        }
        TemporalGuardExpr::Or(es) => {
            let parts: Vec<String> = es.iter().map(temporal_guard_ts).collect();
            format!("({})", parts.join(" || "))
        }
        TemporalGuardExpr::Not(e) => format!("!({})", temporal_guard_ts(e)),
    }
}

fn action_to_ts(action: &Action) -> String {
    match action {
        Action::StartTimer(id) => {
            format!("this.timers.set(\"{}\", performance.now());", id.0)
        }
        Action::StopTimer(id) => format!("this.timers.delete(\"{}\");", id.0),
        Action::EmitEvent(ek) => format!("this.emit(\"{}\");", ek),
        Action::SetVar { var, value } => {
            format!("this.vars.set(\"{}\", {});", var.0, value_to_ts(value))
        }
        Action::PlayFeedback(fb) => format!("this.playFeedback(\"{}\");", fb),
        Action::Highlight { entity, style } => {
            format!("this.highlight(\"{}\", \"{}\");", entity.0, style)
        }
        Action::ClearHighlight(entity) => {
            format!("this.clearHighlight(\"{}\");", entity.0)
        }
        Action::MoveEntity { entity, target } => format!(
            "this.moveEntity(\"{}\", {{ x: {:.4}, y: {:.4}, z: {:.4} }});",
            entity.0, target.x, target.y, target.z
        ),
        Action::Custom(s) => format!("this.customAction(\"{}\");", s),
        Action::Noop => "/* noop */".into(),
    }
}

fn value_to_ts(v: &Value) -> String {
    match v {
        Value::Bool(b) => format!("{}", b),
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => format!("{}", f),
        Value::Str(s) => format!("\"{}\"", s),
    }
}

// ---------------------------------------------------------------------------
// TypeScriptCodeGenerator
// ---------------------------------------------------------------------------

/// Generates TypeScript classes for WebXR environments.
pub struct TypeScriptCodeGenerator {
    pub config: TypeScriptCodegenConfig,
}

impl TypeScriptCodeGenerator {
    pub fn new() -> Self {
        Self {
            config: TypeScriptCodegenConfig::default(),
        }
    }

    pub fn with_config(config: TypeScriptCodegenConfig) -> Self {
        Self { config }
    }

    fn emit_state_enum(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        let cls = &self.config.class_name;
        if self.config.jsdoc {
            buf.line(&format!("/** States for the {} state machine. */", cls));
        }
        buf.line(&format!("export enum {}State {{", cls));
        buf.indent();
        for (_, state) in automaton.states.iter() {
            buf.line(&format!("{} = \"{}\",", state_ts_name(state), state_ts_name(state)));
        }
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_class(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        let cls = &self.config.class_name;

        let initial = automaton
            .initial_state
            .as_ref()
            .and_then(|id| automaton.states.get(id))
            .map(|s| state_ts_name(s))
            .unwrap_or_else(|| "S0".into());

        if self.config.jsdoc {
            buf.line(&format!("/** Runtime instance of the {} state machine. */", cls));
        }
        buf.line(&format!("export class {} {{", cls));
        buf.indent();

        // Fields
        buf.line(&format!("private state: {cls}State;"));
        buf.line("private readonly timers: Map<string, number> = new Map();");
        buf.line("private readonly vars: Map<string, unknown> = new Map();");
        buf.line("private readonly listeners: Array<(event: string) => void> = [];");
        buf.line("private scene: any;");
        buf.blank();

        // Constructor
        buf.line("constructor(scene?: any) {");
        buf.indent();
        buf.line(&format!("this.state = {cls}State.{initial};"));
        buf.line("this.scene = scene ?? {};");
        buf.dedent();
        buf.line("}");
        buf.blank();

        // currentState getter
        buf.line(&format!("get currentState(): {cls}State {{"));
        buf.indent();
        buf.line("return this.state;");
        buf.dedent();
        buf.line("}");
        buf.blank();

        // isAccepting
        self.emit_is_accepting(buf, automaton);

        // on / emit
        buf.line("on(listener: (event: string) => void): void {");
        buf.indent();
        buf.line("this.listeners.push(listener);");
        buf.dedent();
        buf.line("}");
        buf.blank();

        buf.line("private emit(event: string): void {");
        buf.indent();
        buf.line("for (const listener of this.listeners) { listener(event); }");
        buf.dedent();
        buf.line("}");
        buf.blank();

        // step
        self.emit_step(buf, automaton);

        // evaluate
        self.emit_evaluate(buf, automaton);

        // entry/exit hooks
        self.emit_entry_exit(buf, automaton);

        // feedback stubs
        self.emit_stubs(buf);

        // WebXR hooks
        if self.config.webxr_hooks {
            self.emit_webxr_hooks(buf);
        }

        buf.dedent();
        buf.line("}");
    }

    fn emit_is_accepting(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        let cls = &self.config.class_name;
        let accepting: Vec<String> = automaton
            .states
            .values()
            .filter(|s| s.is_accepting)
            .map(|s| format!("{cls}State.{}", state_ts_name(s)))
            .collect();

        buf.line("get isAccepting(): boolean {");
        buf.indent();
        if accepting.is_empty() {
            buf.line("return false;");
        } else {
            let cond = accepting
                .iter()
                .map(|a| format!("this.state === {}", a))
                .collect::<Vec<_>>()
                .join(" || ");
            buf.line(&format!("return {};", cond));
        }
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_step(&self, buf: &mut CodeBuffer, _automaton: &SpatialEventAutomaton) {
        let cls = &self.config.class_name;
        if self.config.async_step {
            buf.line("async step(event: string): Promise<boolean> {");
        } else {
            buf.line("step(event: string): boolean {");
        }
        buf.indent();
        buf.line(&format!(
            "const next: {cls}State | undefined = this.evaluate(event);"
        ));
        buf.line("if (next !== undefined) {");
        buf.indent();
        buf.line("this.onExit(this.state);");
        buf.line("this.state = next;");
        buf.line("this.onEnter(this.state);");
        buf.line("return true;");
        buf.dedent();
        buf.line("}");
        buf.line("return false;");
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_evaluate(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        let cls = &self.config.class_name;
        buf.line(&format!(
            "private evaluate(event: string): {cls}State | undefined {{"
        ));
        buf.indent();
        buf.line("switch (this.state) {");
        buf.indent();

        let mut by_source: HashMap<StateId, Vec<&Transition>> = HashMap::new();
        for (_, tr) in automaton.transitions.iter() {
            by_source.entry(tr.source).or_default().push(tr);
        }

        for (_, state) in automaton.states.iter() {
            buf.line(&format!(
                "case {cls}State.{}:",
                state_ts_name(state)
            ));
            buf.indent();
            if let Some(transitions) = by_source.get(&state.id) {
                for tr in transitions {
                    let target_name = automaton
                        .states
                        .get(&tr.target)
                        .map(|s| state_ts_name(s))
                        .unwrap_or_else(|| format!("S{}", tr.target.0));

                    let guard_code = guard_to_ts(&tr.guard);
                    if guard_code == "true" {
                        buf.line(&format!("return {cls}State.{target_name};"));
                    } else {
                        buf.line(&format!(
                            "if ({guard_code}) return {cls}State.{target_name};"
                        ));
                    }
                }
            }
            buf.line("break;");
            buf.dedent();
        }

        buf.dedent();
        buf.line("}");
        buf.line("return undefined;");
        buf.dedent();
        buf.line("}");
        buf.blank();
    }

    fn emit_entry_exit(&self, buf: &mut CodeBuffer, automaton: &SpatialEventAutomaton) {
        let cls = &self.config.class_name;

        // onEnter
        buf.line(&format!("private onEnter(state: {cls}State): void {{"));
        buf.indent();
        buf.line("switch (state) {");
        buf.indent();
        for (_, state) in automaton.states.iter() {
            if !state.on_entry.is_empty() {
                buf.line(&format!(
                    "case {cls}State.{}:",
                    state_ts_name(state)
                ));
                buf.indent();
                for action in &state.on_entry {
                    buf.line(&action_to_ts(action));
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

        // onExit
        buf.line(&format!("private onExit(state: {cls}State): void {{"));
        buf.indent();
        buf.line("switch (state) {");
        buf.indent();
        for (_, state) in automaton.states.iter() {
            if !state.on_exit.is_empty() {
                buf.line(&format!(
                    "case {cls}State.{}:",
                    state_ts_name(state)
                ));
                buf.indent();
                for action in &state.on_exit {
                    buf.line(&action_to_ts(action));
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

    fn emit_stubs(&self, buf: &mut CodeBuffer) {
        buf.line("private playFeedback(_name: string): void { /* override */ }");
        buf.line("private highlight(_entity: string, _style: string): void { /* override */ }");
        buf.line("private clearHighlight(_entity: string): void { /* override */ }");
        buf.line("private moveEntity(_entity: string, _pos: {x:number,y:number,z:number}): void { /* override */ }");
        buf.line("private customAction(_name: string): void { /* override */ }");
        buf.line("private evalTemporal(_id: string): boolean { return false; }");
        buf.blank();
    }

    fn emit_webxr_hooks(&self, buf: &mut CodeBuffer) {
        if self.config.jsdoc {
            buf.line("/** Bind to an XRSession for automatic event dispatch. */");
        }
        buf.line("bindToXRSession(session: XRSession): void {");
        buf.indent();
        buf.line("session.addEventListener('select', () => this.step('select'));");
        buf.line("session.addEventListener('selectstart', () => this.step('grab_start'));");
        buf.line("session.addEventListener('selectend', () => this.step('grab_end'));");
        buf.line("session.addEventListener('squeeze', () => this.step('squeeze'));");
        buf.line("session.addEventListener('squeezestart', () => this.step('touch_start'));");
        buf.line("session.addEventListener('squeezeend', () => this.step('touch_end'));");
        buf.dedent();
        buf.line("}");
        buf.blank();

        if self.config.jsdoc {
            buf.line("/** Run a per-frame polling loop for spatial/temporal guards. */");
        }
        buf.line("startFrameLoop(session: XRSession): void {");
        buf.indent();
        buf.line("const loop = (_time: DOMHighResTimeStamp, _frame: XRFrame) => {");
        buf.indent();
        buf.line("// Spatial/temporal guards can be polled here");
        buf.line("session.requestAnimationFrame(loop);");
        buf.dedent();
        buf.line("};");
        buf.line("session.requestAnimationFrame(loop);");
        buf.dedent();
        buf.line("}");
        buf.blank();
    }
}

impl Default for TypeScriptCodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator trait impl
// ---------------------------------------------------------------------------

impl CodeGenerator for TypeScriptCodeGenerator {
    fn generate(&self, automaton: &SpatialEventAutomaton) -> CodegenResult<String> {
        if automaton.states.is_empty() {
            return Err(CodegenError::Config(
                "cannot generate TypeScript for an empty automaton".into(),
            ));
        }

        let mut buf = CodeBuffer::new("  ");

        buf.line("// Auto-generated by choreo-codegen (TypeScript backend)");
        buf.line("// DO NOT EDIT – regenerate from the interaction specification.");
        buf.blank();

        self.emit_state_enum(&mut buf, automaton);
        self.emit_class(&mut buf, automaton);

        Ok(buf.finish())
    }

    fn name(&self) -> &str {
        "TypeScript"
    }

    fn file_extension(&self) -> &str {
        "ts"
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
    fn test_generates_enum_and_class() {
        let gen = TypeScriptCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("export enum InteractionMachineState"));
        assert!(code.contains("export class InteractionMachine"));
    }

    #[test]
    fn test_generates_states() {
        let gen = TypeScriptCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("Idle"));
        assert!(code.contains("Active"));
    }

    #[test]
    fn test_async_step() {
        let gen = TypeScriptCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("async step(event: string): Promise<boolean>"));
    }

    #[test]
    fn test_sync_step() {
        let gen = TypeScriptCodeGenerator::with_config(TypeScriptCodegenConfig {
            async_step: false,
            ..Default::default()
        });
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("step(event: string): boolean"));
        assert!(!code.contains("async step"));
    }

    #[test]
    fn test_webxr_hooks() {
        let gen = TypeScriptCodeGenerator::new();
        let code = gen.generate(&two_state()).unwrap();
        assert!(code.contains("bindToXRSession"));
        assert!(code.contains("startFrameLoop"));
        assert!(code.contains("XRSession"));
    }

    #[test]
    fn test_no_webxr_hooks() {
        let gen = TypeScriptCodeGenerator::with_config(TypeScriptCodegenConfig {
            webxr_hooks: false,
            ..Default::default()
        });
        let code = gen.generate(&two_state()).unwrap();
        assert!(!code.contains("bindToXRSession"));
    }

    #[test]
    fn test_guard_to_ts_basics() {
        assert_eq!(guard_to_ts(&Guard::True), "true");
        assert_eq!(guard_to_ts(&Guard::False), "false");
        let g = Guard::Event(EventKind::GrabStart);
        assert!(guard_to_ts(&g).contains("event === \"grab_start\""));
    }

    #[test]
    fn test_guard_to_ts_compound() {
        let g = Guard::And(vec![Guard::True, Guard::False]);
        assert!(guard_to_ts(&g).contains("&&"));
        let g = Guard::Or(vec![Guard::True, Guard::False]);
        assert!(guard_to_ts(&g).contains("||"));
        let g = Guard::Not(Box::new(Guard::True));
        assert_eq!(guard_to_ts(&g), "!(true)");
    }

    #[test]
    fn test_file_extension() {
        let gen = TypeScriptCodeGenerator::new();
        assert_eq!(gen.file_extension(), "ts");
        assert_eq!(gen.name(), "TypeScript");
    }

    #[test]
    fn test_empty_automaton_errors() {
        let gen = TypeScriptCodeGenerator::new();
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
    fn test_action_to_ts_variants() {
        use choreo_automata::{TimerId, VarId};
        assert!(action_to_ts(&Action::StartTimer(TimerId("t1".into()))).contains("timers.set"));
        assert!(action_to_ts(&Action::StopTimer(TimerId("t1".into()))).contains("timers.delete"));
        assert!(action_to_ts(&Action::EmitEvent(EventKind::GazeDwell)).contains("emit"));
        assert!(action_to_ts(&Action::SetVar {
            var: VarId("x".into()),
            value: Value::Int(42),
        })
        .contains("42"));
        assert!(action_to_ts(&Action::Custom("beep".into())).contains("customAction"));
    }
}
