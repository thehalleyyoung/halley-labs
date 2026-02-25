//! QCTL_F Model Checking Engine for CABER
//!
//! Implements bottom-up CTL model checking over Kripke structures with
//! probabilistic extensions, quantitative satisfaction degrees, witness
//! generation, and counterexample construction.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CompOp
// ---------------------------------------------------------------------------

/// Comparison operators for probabilistic thresholds.
#[derive(Clone, Debug, PartialEq)]
pub enum CompOp {
    Ge,
    Gt,
    Le,
    Lt,
    Eq,
}

impl CompOp {
    /// Evaluate `lhs op rhs`.
    pub fn eval(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            CompOp::Ge => lhs >= rhs,
            CompOp::Gt => lhs > rhs,
            CompOp::Le => lhs <= rhs,
            CompOp::Lt => lhs < rhs,
            CompOp::Eq => (lhs - rhs).abs() < 1e-12,
        }
    }

    pub fn render(&self) -> &str {
        match self {
            CompOp::Ge => ">=",
            CompOp::Gt => ">",
            CompOp::Le => "<=",
            CompOp::Lt => "<",
            CompOp::Eq => "==",
        }
    }
}

impl fmt::Display for CompOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ---------------------------------------------------------------------------
// CTLFormula
// ---------------------------------------------------------------------------

/// CTL formula with probabilistic extension.
#[derive(Clone, Debug, PartialEq)]
pub enum CTLFormula {
    True,
    False,
    Atom(String),
    Not(Box<CTLFormula>),
    And(Box<CTLFormula>, Box<CTLFormula>),
    Or(Box<CTLFormula>, Box<CTLFormula>),
    Implies(Box<CTLFormula>, Box<CTLFormula>),
    EX(Box<CTLFormula>),
    AX(Box<CTLFormula>),
    EU(Box<CTLFormula>, Box<CTLFormula>),
    AU(Box<CTLFormula>, Box<CTLFormula>),
    EG(Box<CTLFormula>),
    AG(Box<CTLFormula>),
    EF(Box<CTLFormula>),
    AF(Box<CTLFormula>),
    /// P[op threshold](φ) – probabilistic operator.
    Prob(CompOp, f64, Box<CTLFormula>),
}

impl CTLFormula {
    /// Pretty-print the formula.
    pub fn render(&self) -> String {
        match self {
            CTLFormula::True => "true".to_string(),
            CTLFormula::False => "false".to_string(),
            CTLFormula::Atom(p) => p.clone(),
            CTLFormula::Not(f) => format!("¬({})", f.render()),
            CTLFormula::And(a, b) => format!("({} ∧ {})", a.render(), b.render()),
            CTLFormula::Or(a, b) => format!("({} ∨ {})", a.render(), b.render()),
            CTLFormula::Implies(a, b) => format!("({} → {})", a.render(), b.render()),
            CTLFormula::EX(f) => format!("EX({})", f.render()),
            CTLFormula::AX(f) => format!("AX({})", f.render()),
            CTLFormula::EU(a, b) => format!("E[{} U {}]", a.render(), b.render()),
            CTLFormula::AU(a, b) => format!("A[{} U {}]", a.render(), b.render()),
            CTLFormula::EG(f) => format!("EG({})", f.render()),
            CTLFormula::AG(f) => format!("AG({})", f.render()),
            CTLFormula::EF(f) => format!("EF({})", f.render()),
            CTLFormula::AF(f) => format!("AF({})", f.render()),
            CTLFormula::Prob(op, t, f) => format!("P[{}{}]({})", op.render(), t, f.render()),
        }
    }

    /// Depth of the formula tree.
    pub fn depth(&self) -> usize {
        match self {
            CTLFormula::True | CTLFormula::False | CTLFormula::Atom(_) => 0,
            CTLFormula::Not(f)
            | CTLFormula::EX(f)
            | CTLFormula::AX(f)
            | CTLFormula::EG(f)
            | CTLFormula::AG(f)
            | CTLFormula::EF(f)
            | CTLFormula::AF(f)
            | CTLFormula::Prob(_, _, f) => 1 + f.depth(),
            CTLFormula::And(a, b)
            | CTLFormula::Or(a, b)
            | CTLFormula::Implies(a, b)
            | CTLFormula::EU(a, b)
            | CTLFormula::AU(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    /// Number of subformulas (nodes in the AST).
    pub fn size(&self) -> usize {
        match self {
            CTLFormula::True | CTLFormula::False | CTLFormula::Atom(_) => 1,
            CTLFormula::Not(f)
            | CTLFormula::EX(f)
            | CTLFormula::AX(f)
            | CTLFormula::EG(f)
            | CTLFormula::AG(f)
            | CTLFormula::EF(f)
            | CTLFormula::AF(f)
            | CTLFormula::Prob(_, _, f) => 1 + f.size(),
            CTLFormula::And(a, b)
            | CTLFormula::Or(a, b)
            | CTLFormula::Implies(a, b)
            | CTLFormula::EU(a, b)
            | CTLFormula::AU(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Collect all atomic proposition names used in the formula.
    pub fn atoms(&self) -> Vec<String> {
        let mut set = HashSet::new();
        self.collect_atoms(&mut set);
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    fn collect_atoms(&self, set: &mut HashSet<String>) {
        match self {
            CTLFormula::True | CTLFormula::False => {}
            CTLFormula::Atom(p) => {
                set.insert(p.clone());
            }
            CTLFormula::Not(f)
            | CTLFormula::EX(f)
            | CTLFormula::AX(f)
            | CTLFormula::EG(f)
            | CTLFormula::AG(f)
            | CTLFormula::EF(f)
            | CTLFormula::AF(f)
            | CTLFormula::Prob(_, _, f) => {
                f.collect_atoms(set);
            }
            CTLFormula::And(a, b)
            | CTLFormula::Or(a, b)
            | CTLFormula::Implies(a, b)
            | CTLFormula::EU(a, b)
            | CTLFormula::AU(a, b) => {
                a.collect_atoms(set);
                b.collect_atoms(set);
            }
        }
    }

    // --- Helper constructors ------------------------------------------------

    pub fn atom(s: &str) -> Self {
        CTLFormula::Atom(s.to_string())
    }

    pub fn not(f: CTLFormula) -> Self {
        CTLFormula::Not(Box::new(f))
    }

    pub fn and(f: CTLFormula, g: CTLFormula) -> Self {
        CTLFormula::And(Box::new(f), Box::new(g))
    }

    pub fn or(f: CTLFormula, g: CTLFormula) -> Self {
        CTLFormula::Or(Box::new(f), Box::new(g))
    }

    pub fn implies(f: CTLFormula, g: CTLFormula) -> Self {
        CTLFormula::Implies(Box::new(f), Box::new(g))
    }

    pub fn ex(f: CTLFormula) -> Self {
        CTLFormula::EX(Box::new(f))
    }

    pub fn ax(f: CTLFormula) -> Self {
        CTLFormula::AX(Box::new(f))
    }

    pub fn eu(f: CTLFormula, g: CTLFormula) -> Self {
        CTLFormula::EU(Box::new(f), Box::new(g))
    }

    pub fn au(f: CTLFormula, g: CTLFormula) -> Self {
        CTLFormula::AU(Box::new(f), Box::new(g))
    }

    pub fn eg(f: CTLFormula) -> Self {
        CTLFormula::EG(Box::new(f))
    }

    pub fn ag(f: CTLFormula) -> Self {
        CTLFormula::AG(Box::new(f))
    }

    pub fn ef(f: CTLFormula) -> Self {
        CTLFormula::EF(Box::new(f))
    }

    pub fn af(f: CTLFormula) -> Self {
        CTLFormula::AF(Box::new(f))
    }

    pub fn prob_ge(p: f64, f: CTLFormula) -> Self {
        CTLFormula::Prob(CompOp::Ge, p, Box::new(f))
    }

    pub fn prob_le(p: f64, f: CTLFormula) -> Self {
        CTLFormula::Prob(CompOp::Le, p, Box::new(f))
    }
}

impl fmt::Display for CTLFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ---------------------------------------------------------------------------
// KripkeModel
// ---------------------------------------------------------------------------

/// A Kripke structure with probabilistic transitions.
#[derive(Clone, Debug)]
pub struct KripkeModel {
    pub num_states: usize,
    /// `transitions[s]` is a list of `(target, probability)`.
    pub transitions: Vec<Vec<(usize, f64)>>,
    /// Atomic propositions that hold in each state.
    pub labels: Vec<Vec<String>>,
    pub initial_states: Vec<usize>,
    pub state_names: Vec<String>,
}

impl KripkeModel {
    pub fn new(num_states: usize) -> Self {
        let state_names: Vec<String> = (0..num_states).map(|i| format!("s{}", i)).collect();
        KripkeModel {
            num_states,
            transitions: vec![Vec::new(); num_states],
            labels: vec![Vec::new(); num_states],
            initial_states: if num_states > 0 { vec![0] } else { vec![] },
            state_names,
        }
    }

    pub fn add_transition(&mut self, source: usize, target: usize, prob: f64) {
        assert!(source < self.num_states, "source out of range");
        assert!(target < self.num_states, "target out of range");
        self.transitions[source].push((target, prob));
    }

    pub fn add_label(&mut self, state: usize, label: &str) {
        assert!(state < self.num_states, "state out of range");
        if !self.labels[state].contains(&label.to_string()) {
            self.labels[state].push(label.to_string());
        }
    }

    pub fn set_state_name(&mut self, state: usize, name: &str) {
        assert!(state < self.num_states, "state out of range");
        self.state_names[state] = name.to_string();
    }

    pub fn successors(&self, state: usize) -> &[(usize, f64)] {
        &self.transitions[state]
    }

    /// Compute predecessors by scanning all transitions.
    pub fn predecessors(&self, state: usize) -> Vec<(usize, f64)> {
        let mut preds = Vec::new();
        for s in 0..self.num_states {
            for &(t, p) in &self.transitions[s] {
                if t == state {
                    preds.push((s, p));
                }
            }
        }
        preds
    }

    pub fn has_label(&self, state: usize, label: &str) -> bool {
        self.labels[state].iter().any(|l| l == label)
    }

    pub fn states_with_label(&self, label: &str) -> Vec<usize> {
        (0..self.num_states)
            .filter(|&s| self.has_label(s, label))
            .collect()
    }

    pub fn is_terminal(&self, state: usize) -> bool {
        self.transitions[state].is_empty()
    }

    /// BFS from `state`, returning all reachable state indices (including `state`).
    pub fn reachable_from(&self, state: usize) -> Vec<usize> {
        let mut visited = vec![false; self.num_states];
        let mut queue = VecDeque::new();
        visited[state] = true;
        queue.push_back(state);
        let mut result = Vec::new();
        while let Some(s) = queue.pop_front() {
            result.push(s);
            for &(t, _) in &self.transitions[s] {
                if !visited[t] {
                    visited[t] = true;
                    queue.push_back(t);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// WitnessType, CheckerWitness, CheckerCounterexample
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum WitnessType {
    Atomic,
    Conjunction,
    Disjunction,
    ExistentialNext,
    UniversalNext,
    ExistentialUntil,
    UniversalUntil,
    ExistentialGlobally,
    UniversalGlobally,
    Probabilistic,
}

#[derive(Clone, Debug)]
pub struct CheckerWitness {
    pub state: usize,
    pub formula: String,
    pub witness_type: WitnessType,
    pub children: Vec<CheckerWitness>,
    /// Path witness for temporal formulas.
    pub trace: Option<Vec<usize>>,
}

impl CheckerWitness {
    pub fn render(&self) -> String {
        self.render_indent(0)
    }

    fn render_indent(&self, indent: usize) -> String {
        let pad = " ".repeat(indent);
        let mut s = format!(
            "{}[s{}] {:?} satisfies {}",
            pad, self.state, self.witness_type, self.formula
        );
        if let Some(ref trace) = self.trace {
            let path: Vec<String> = trace.iter().map(|t| format!("s{}", t)).collect();
            s.push_str(&format!(" via path [{}]", path.join(" -> ")));
        }
        s.push('\n');
        for c in &self.children {
            s.push_str(&c.render_indent(indent + 2));
        }
        s
    }

    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }
}

#[derive(Clone, Debug)]
pub struct CheckerCounterexample {
    pub state: usize,
    pub formula: String,
    pub trace: Vec<usize>,
    pub explanation: String,
}

impl CheckerCounterexample {
    pub fn render(&self) -> String {
        let path: Vec<String> = self.trace.iter().map(|t| format!("s{}", t)).collect();
        format!(
            "Counterexample at s{}: {} not satisfied.\nTrace: [{}]\n{}",
            self.state,
            self.formula,
            path.join(" -> "),
            self.explanation,
        )
    }

    pub fn length(&self) -> usize {
        self.trace.len()
    }
}

// ---------------------------------------------------------------------------
// ComplexityTracker
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ComplexityTracker {
    pub states_visited: usize,
    pub fixpoint_iterations_total: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub formulas_checked: usize,
}

impl ComplexityTracker {
    pub fn new() -> Self {
        ComplexityTracker {
            states_visited: 0,
            fixpoint_iterations_total: 0,
            cache_hits: 0,
            cache_misses: 0,
            formulas_checked: 0,
        }
    }

    pub fn record_visit(&mut self) {
        self.states_visited += 1;
    }

    pub fn record_fixpoint(&mut self, iters: usize) {
        self.fixpoint_iterations_total += iters;
    }

    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    pub fn summary(&self) -> String {
        format!(
            "Complexity: {} states visited, {} fixpoint iterations, {} cache hits, {} cache misses, {} formulas checked",
            self.states_visited,
            self.fixpoint_iterations_total,
            self.cache_hits,
            self.cache_misses,
            self.formulas_checked,
        )
    }
}

impl Default for ComplexityTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ModelCheckConfig
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ModelCheckConfig {
    pub max_fixpoint_iterations: usize,
    pub epsilon: f64,
    pub early_termination: bool,
    pub generate_witnesses: bool,
    pub cache_results: bool,
}

impl Default for ModelCheckConfig {
    fn default() -> Self {
        ModelCheckConfig {
            max_fixpoint_iterations: 10000,
            epsilon: 1e-10,
            early_termination: true,
            generate_witnesses: false,
            cache_results: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelCheckResult, StateCheckResult
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ModelCheckResult {
    pub formula: String,
    pub satisfying_states: Vec<usize>,
    pub satisfaction_map: Vec<bool>,
    pub satisfaction_degrees: Vec<f64>,
    pub witness: Option<CheckerWitness>,
    pub counterexample: Option<CheckerCounterexample>,
    pub fixpoint_iterations: usize,
    pub computation_time_ms: f64,
}

impl ModelCheckResult {
    pub fn num_satisfying(&self) -> usize {
        self.satisfying_states.len()
    }

    pub fn fraction_satisfying(&self) -> f64 {
        if self.satisfaction_map.is_empty() {
            0.0
        } else {
            self.satisfying_states.len() as f64 / self.satisfaction_map.len() as f64
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "{}: {}/{} states satisfy ({}%), time={:.3}ms, fixpoint_iters={}",
            self.formula,
            self.num_satisfying(),
            self.satisfaction_map.len(),
            (self.fraction_satisfying() * 100.0) as u64,
            self.computation_time_ms,
            self.fixpoint_iterations,
        )
    }
}

#[derive(Clone, Debug)]
pub struct StateCheckResult {
    pub state: usize,
    pub satisfied: bool,
    pub degree: f64,
    pub witness: Option<CheckerWitness>,
}

// ---------------------------------------------------------------------------
// QCTLFModelChecker
// ---------------------------------------------------------------------------

/// Main model checking engine.
pub struct QCTLFModelChecker {
    pub model: KripkeModel,
    pub config: ModelCheckConfig,
    label_cache: HashMap<String, Vec<bool>>,
    complexity: ComplexityTracker,
}

impl QCTLFModelChecker {
    pub fn new(model: KripkeModel, config: ModelCheckConfig) -> Self {
        QCTLFModelChecker {
            model,
            config,
            label_cache: HashMap::new(),
            complexity: ComplexityTracker::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Top-level checking
    // -----------------------------------------------------------------------

    /// Check which states of the model satisfy `formula`.
    pub fn check(&mut self, formula: &CTLFormula) -> ModelCheckResult {
        let start = Instant::now();
        let fixpoint_before = self.complexity.fixpoint_iterations_total;

        let sat_map = self.label_states(formula);

        let satisfying_states: Vec<usize> = sat_map
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        let degrees: Vec<f64> = (0..self.model.num_states)
            .map(|s| {
                if sat_map[s] {
                    self.compute_sat_degree(s, formula)
                } else {
                    0.0
                }
            })
            .collect();

        let fixpoint_iters = self.complexity.fixpoint_iterations_total - fixpoint_before;

        // Optionally generate witness/counterexample for the first initial state.
        let witness = if self.config.generate_witnesses {
            if let Some(&init) = self.model.initial_states.first() {
                if sat_map[init] {
                    self.generate_witness(init, formula)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let counterexample = if self.config.generate_witnesses {
            if let Some(&init) = self.model.initial_states.first() {
                if !sat_map[init] {
                    self.generate_counterexample(init, formula)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        self.complexity.formulas_checked += 1;

        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0;

        ModelCheckResult {
            formula: formula.render(),
            satisfying_states,
            satisfaction_map: sat_map,
            satisfaction_degrees: degrees,
            witness,
            counterexample,
            fixpoint_iterations: fixpoint_iters,
            computation_time_ms: time_ms,
        }
    }

    /// Check whether a specific state satisfies `formula`.
    pub fn check_state(&mut self, state: usize, formula: &CTLFormula) -> StateCheckResult {
        let sat_map = self.label_states(formula);
        let satisfied = sat_map[state];
        let degree = self.compute_sat_degree(state, formula);
        let witness = if satisfied && self.config.generate_witnesses {
            self.generate_witness(state, formula)
        } else {
            None
        };
        StateCheckResult {
            state,
            satisfied,
            degree,
            witness,
        }
    }

    // -----------------------------------------------------------------------
    // Core labeling algorithm (bottom-up recursive)
    // -----------------------------------------------------------------------

    /// Recursively label states, returning a boolean vector indexed by state.
    pub fn label_states(&mut self, formula: &CTLFormula) -> Vec<bool> {
        let key = formula.render();

        // Check cache
        if self.config.cache_results {
            if let Some(cached) = self.label_cache.get(&key) {
                self.complexity.record_cache_hit();
                return cached.clone();
            }
            self.complexity.record_cache_miss();
        }

        let n = self.model.num_states;

        let result = match formula {
            CTLFormula::True => vec![true; n],

            CTLFormula::False => vec![false; n],

            CTLFormula::Atom(p) => {
                let mut v = vec![false; n];
                for s in 0..n {
                    self.complexity.record_visit();
                    if self.model.has_label(s, p) {
                        v[s] = true;
                    }
                }
                v
            }

            CTLFormula::Not(inner) => {
                let inner_sat = self.label_states(inner);
                inner_sat.iter().map(|b| !b).collect()
            }

            CTLFormula::And(a, b) => {
                let sa = self.label_states(a);
                let sb = self.label_states(b);
                sa.iter().zip(sb.iter()).map(|(x, y)| *x && *y).collect()
            }

            CTLFormula::Or(a, b) => {
                let sa = self.label_states(a);
                let sb = self.label_states(b);
                sa.iter().zip(sb.iter()).map(|(x, y)| *x || *y).collect()
            }

            CTLFormula::Implies(a, b) => {
                // a → b  ≡  ¬a ∨ b
                let sa = self.label_states(a);
                let sb = self.label_states(b);
                sa.iter().zip(sb.iter()).map(|(x, y)| !x || *y).collect()
            }

            CTLFormula::EX(inner) => {
                let inner_sat = self.label_states(inner);
                self.label_ex(&inner_sat)
            }

            CTLFormula::AX(inner) => {
                let inner_sat = self.label_states(inner);
                self.label_ax(&inner_sat)
            }

            CTLFormula::EU(phi, psi) => {
                let phi_sat = self.label_states(phi);
                let psi_sat = self.label_states(psi);
                self.label_eu(&phi_sat, &psi_sat)
            }

            CTLFormula::AU(phi, psi) => {
                let phi_sat = self.label_states(phi);
                let psi_sat = self.label_states(psi);
                self.label_au(&phi_sat, &psi_sat)
            }

            CTLFormula::EG(inner) => {
                let inner_sat = self.label_states(inner);
                self.label_eg(&inner_sat)
            }

            CTLFormula::AG(inner) => {
                let inner_sat = self.label_states(inner);
                self.label_ag(&inner_sat)
            }

            CTLFormula::EF(inner) => {
                // EF(φ) = E[true U φ]
                let phi_sat = self.label_states(inner);
                let true_sat = vec![true; n];
                self.label_eu(&true_sat, &phi_sat)
            }

            CTLFormula::AF(inner) => {
                // AF(φ) = A[true U φ]
                let phi_sat = self.label_states(inner);
                let true_sat = vec![true; n];
                self.label_au(&true_sat, &phi_sat)
            }

            CTLFormula::Prob(op, threshold, inner) => {
                let inner_sat = self.label_states(inner);
                let mut result = vec![false; n];
                for s in 0..n {
                    self.complexity.record_visit();
                    let prob = self.compute_probability_with_sat(s, inner, &inner_sat);
                    if op.eval(prob, *threshold) {
                        result[s] = true;
                    }
                }
                result
            }
        };

        if self.config.cache_results {
            self.label_cache.insert(key, result.clone());
        }

        result
    }

    // -----------------------------------------------------------------------
    // EX / AX
    // -----------------------------------------------------------------------

    /// EX(φ): states with at least one successor satisfying φ.
    fn label_ex(&self, inner_sat: &[bool]) -> Vec<bool> {
        let n = self.model.num_states;
        let mut result = vec![false; n];
        for s in 0..n {
            for &(t, _) in self.model.successors(s) {
                if inner_sat[t] {
                    result[s] = true;
                    break;
                }
            }
        }
        result
    }

    /// AX(φ): states where ALL successors satisfy φ.
    /// A state with no successors (terminal) vacuously satisfies AX(φ).
    fn label_ax(&self, inner_sat: &[bool]) -> Vec<bool> {
        let n = self.model.num_states;
        let mut result = vec![false; n];
        for s in 0..n {
            let succs = self.model.successors(s);
            if succs.is_empty() {
                // Terminal states vacuously satisfy AX.
                result[s] = true;
            } else {
                result[s] = succs.iter().all(|&(t, _)| inner_sat[t]);
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // EU (least fixpoint, backward BFS)
    // -----------------------------------------------------------------------

    /// E[φ U ψ]: least fixpoint via backward BFS.
    /// sat = states(ψ), then repeatedly add s if s ∈ states(φ) and some successor
    /// of s is in sat.
    fn label_eu(&mut self, phi_sat: &[bool], psi_sat: &[bool]) -> Vec<bool> {
        let n = self.model.num_states;
        let mut sat = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Initialize with states satisfying ψ.
        for s in 0..n {
            if psi_sat[s] {
                sat[s] = true;
                queue.push_back(s);
            }
        }

        // Backward BFS: for each newly added state, check predecessors.
        let mut iterations = 0;
        while let Some(t) = queue.pop_front() {
            iterations += 1;
            if iterations > self.config.max_fixpoint_iterations {
                break;
            }
            // Scan all predecessors of t.
            for s in 0..n {
                if !sat[s] && phi_sat[s] {
                    if self.model.successors(s).iter().any(|&(succ, _)| succ == t) {
                        sat[s] = true;
                        queue.push_back(s);
                    }
                }
            }
        }

        self.complexity.record_fixpoint(iterations);
        sat
    }

    // -----------------------------------------------------------------------
    // AU (least fixpoint)
    // -----------------------------------------------------------------------

    /// A[φ U ψ]: least fixpoint.
    /// sat = states(ψ), then repeatedly add s if s ∈ states(φ) and ALL
    /// successors of s are in sat.
    fn label_au(&mut self, phi_sat: &[bool], psi_sat: &[bool]) -> Vec<bool> {
        let n = self.model.num_states;
        let mut sat = vec![false; n];

        // Initialize with states satisfying ψ.
        for s in 0..n {
            if psi_sat[s] {
                sat[s] = true;
            }
        }

        // Iterate until fixpoint.
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.config.max_fixpoint_iterations {
                break;
            }
            let mut changed = false;
            for s in 0..n {
                if !sat[s] && phi_sat[s] {
                    let succs = self.model.successors(s);
                    if !succs.is_empty() && succs.iter().all(|&(t, _)| sat[t]) {
                        sat[s] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        self.complexity.record_fixpoint(iterations);
        sat
    }

    // -----------------------------------------------------------------------
    // EG (greatest fixpoint)
    // -----------------------------------------------------------------------

    /// EG(φ): greatest fixpoint.
    /// sat = states(φ), then iteratively remove states that have no successor
    /// in sat.
    fn label_eg(&mut self, phi_sat: &[bool]) -> Vec<bool> {
        let n = self.model.num_states;
        let mut sat: Vec<bool> = phi_sat.to_vec();

        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.config.max_fixpoint_iterations {
                break;
            }
            let mut changed = false;
            for s in 0..n {
                if sat[s] {
                    let succs = self.model.successors(s);
                    // Must have at least one successor still in sat.
                    let has_succ_in_sat = succs.iter().any(|&(t, _)| sat[t]);
                    if !has_succ_in_sat {
                        sat[s] = false;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        self.complexity.record_fixpoint(iterations);
        sat
    }

    // -----------------------------------------------------------------------
    // AG (greatest fixpoint)
    // -----------------------------------------------------------------------

    /// AG(φ): greatest fixpoint.
    /// sat = states(φ), then iteratively remove states that have any successor
    /// not in sat.
    fn label_ag(&mut self, phi_sat: &[bool]) -> Vec<bool> {
        let n = self.model.num_states;
        let mut sat: Vec<bool> = phi_sat.to_vec();

        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.config.max_fixpoint_iterations {
                break;
            }
            let mut changed = false;
            for s in 0..n {
                if sat[s] {
                    let succs = self.model.successors(s);
                    // All successors must be in sat (terminal states are fine).
                    if !succs.is_empty() && !succs.iter().all(|&(t, _)| sat[t]) {
                        sat[s] = false;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        self.complexity.record_fixpoint(iterations);
        sat
    }

    // -----------------------------------------------------------------------
    // Quantitative satisfaction degree
    // -----------------------------------------------------------------------

    /// Compute quantitative satisfaction degree in [0,1] for a state.
    pub fn compute_sat_degree(&self, state: usize, formula: &CTLFormula) -> f64 {
        match formula {
            CTLFormula::True => 1.0,
            CTLFormula::False => 0.0,
            CTLFormula::Atom(p) => {
                if self.model.has_label(state, p) {
                    1.0
                } else {
                    0.0
                }
            }
            CTLFormula::Not(f) => 1.0 - self.compute_sat_degree(state, f),
            CTLFormula::And(a, b) => {
                let da = self.compute_sat_degree(state, a);
                let db = self.compute_sat_degree(state, b);
                da.min(db)
            }
            CTLFormula::Or(a, b) => {
                let da = self.compute_sat_degree(state, a);
                let db = self.compute_sat_degree(state, b);
                da.max(db)
            }
            CTLFormula::Implies(a, b) => {
                let da = self.compute_sat_degree(state, a);
                let db = self.compute_sat_degree(state, b);
                (1.0 - da).max(db)
            }
            CTLFormula::EX(f) => {
                // Max over successors, weighted by transition probability.
                let succs = self.model.successors(state);
                if succs.is_empty() {
                    return 0.0;
                }
                succs
                    .iter()
                    .map(|&(t, p)| p * self.compute_sat_degree(t, f))
                    .fold(0.0_f64, f64::max)
            }
            CTLFormula::AX(f) => {
                let succs = self.model.successors(state);
                if succs.is_empty() {
                    return 1.0; // vacuously true
                }
                succs
                    .iter()
                    .map(|&(t, p)| p * self.compute_sat_degree(t, f))
                    .fold(1.0_f64, f64::min)
            }
            CTLFormula::EU(_, _)
            | CTLFormula::AU(_, _)
            | CTLFormula::EG(_)
            | CTLFormula::AG(_)
            | CTLFormula::EF(_)
            | CTLFormula::AF(_) => {
                // For temporal operators, we use the labeling result: if the
                // state is labeled, degree is based on path probability;
                // otherwise 0. We approximate by a bounded probabilistic
                // computation.
                // Cache lookup (read-only here; label_states must have been
                // called already in check()).
                let key = formula.render();
                if let Some(sat) = self.label_cache.get(&key) {
                    if sat[state] {
                        // Approximate degree by summing transition probabilities
                        // to satisfying successors, clamped to [0,1].
                        self.approximate_temporal_degree(state, &sat)
                    } else {
                        0.0
                    }
                } else {
                    // Fallback: Boolean
                    0.0
                }
            }
            CTLFormula::Prob(_, _, _) => {
                // Degree is the probability itself, clamped to [0,1].
                let key = formula.render();
                if let Some(sat) = self.label_cache.get(&key) {
                    if sat[state] {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
        }
    }

    /// Approximate temporal satisfaction degree: probability of reaching /
    /// staying in satisfying states from `state`.
    fn approximate_temporal_degree(&self, state: usize, sat: &[bool]) -> f64 {
        let succs = self.model.successors(state);
        if succs.is_empty() {
            return if sat[state] { 1.0 } else { 0.0 };
        }
        let total_prob: f64 = succs
            .iter()
            .filter(|&&(t, _)| sat[t])
            .map(|&(_, p)| p)
            .sum();
        total_prob.min(1.0).max(if sat[state] { 0.5 } else { 0.0 })
    }

    // -----------------------------------------------------------------------
    // Probability computation
    // -----------------------------------------------------------------------

    /// Compute probability that a random path from `state` satisfies `formula`.
    pub fn compute_probability(&mut self, state: usize, formula: &CTLFormula) -> f64 {
        let sat = self.label_states(formula);
        self.compute_probability_with_sat(state, formula, &sat)
    }

    /// Internal: compute probability using pre-computed satisfaction vector.
    fn compute_probability_with_sat(
        &self,
        state: usize,
        formula: &CTLFormula,
        sat: &[bool],
    ) -> f64 {
        match formula {
            CTLFormula::True => 1.0,
            CTLFormula::False => 0.0,
            CTLFormula::Atom(_) => {
                if sat[state] {
                    1.0
                } else {
                    0.0
                }
            }
            CTLFormula::EX(inner) => {
                // P(EX φ) at state s = sum of p(s,t) for t satisfying φ.
                let inner_key = inner.render();
                let inner_sat = self.label_cache.get(&inner_key).cloned().unwrap_or_else(|| sat.to_vec());
                let succs = self.model.successors(state);
                succs
                    .iter()
                    .filter(|&&(t, _)| inner_sat[t])
                    .map(|&(_, p)| p)
                    .sum::<f64>()
                    .min(1.0)
            }
            CTLFormula::AX(inner) => {
                let inner_key = inner.render();
                let inner_sat = self.label_cache.get(&inner_key).cloned().unwrap_or_else(|| sat.to_vec());
                let succs = self.model.successors(state);
                if succs.is_empty() {
                    return 1.0;
                }
                if succs.iter().all(|&(t, _)| inner_sat[t]) {
                    1.0
                } else {
                    0.0
                }
            }
            CTLFormula::EU(_, _) | CTLFormula::EF(_) => {
                // Iterative probability computation for reachability.
                self.iterative_reach_probability(state, sat)
            }
            CTLFormula::EG(_) => {
                // Probability of staying in satisfying states.
                self.iterative_persist_probability(state, sat)
            }
            CTLFormula::Prob(_, _, inner) => {
                // Recurse into inner formula.
                let inner_sat_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_sat_key) {
                    self.compute_probability_with_sat(state, inner, inner_sat)
                } else {
                    if sat[state] {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
            _ => {
                if sat[state] {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Iterative value-iteration for reachability probability.
    /// Computes P(reach sat from state) by iterating:
    ///   x_s = sum_{t} p(s,t) * x_t   if s not in sat
    ///   x_s = 1                       if s in sat
    fn iterative_reach_probability(&self, target_state: usize, sat: &[bool]) -> f64 {
        let n = self.model.num_states;
        let mut x = vec![0.0_f64; n];
        // States already satisfying the target get probability 1.
        for s in 0..n {
            if sat[s] {
                x[s] = 1.0;
            }
        }

        let max_iter = self.config.max_fixpoint_iterations.min(1000);
        for _ in 0..max_iter {
            let mut max_delta: f64 = 0.0;
            for s in 0..n {
                if sat[s] {
                    continue;
                }
                let succs = self.model.successors(s);
                if succs.is_empty() {
                    continue;
                }
                let new_val: f64 = succs.iter().map(|&(t, p)| p * x[t]).sum();
                let delta = (new_val - x[s]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                x[s] = new_val;
            }
            if max_delta < self.config.epsilon {
                break;
            }
        }

        x[target_state]
    }

    /// Iterative value-iteration for persistence probability (EG).
    fn iterative_persist_probability(&self, target_state: usize, sat: &[bool]) -> f64 {
        let n = self.model.num_states;
        let mut x = vec![0.0_f64; n];
        for s in 0..n {
            if sat[s] {
                x[s] = 1.0;
            }
        }

        let max_iter = self.config.max_fixpoint_iterations.min(1000);
        for _ in 0..max_iter {
            let mut max_delta: f64 = 0.0;
            for s in 0..n {
                if !sat[s] {
                    x[s] = 0.0;
                    continue;
                }
                let succs = self.model.successors(s);
                if succs.is_empty() {
                    // Terminal state in sat: persists trivially.
                    continue;
                }
                let new_val: f64 = succs
                    .iter()
                    .map(|&(t, p)| if sat[t] { p * x[t] } else { 0.0 })
                    .sum();
                let delta = (new_val - x[s]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                x[s] = new_val;
            }
            if max_delta < self.config.epsilon {
                break;
            }
        }

        x[target_state]
    }

    // -----------------------------------------------------------------------
    // Witness generation
    // -----------------------------------------------------------------------

    /// Build a witness tree demonstrating why `state` satisfies `formula`.
    pub fn generate_witness(
        &self,
        state: usize,
        formula: &CTLFormula,
    ) -> Option<CheckerWitness> {
        let key = formula.render();

        // Verify the state actually satisfies the formula.
        if let Some(sat) = self.label_cache.get(&key) {
            if !sat[state] {
                return None;
            }
        }

        match formula {
            CTLFormula::True => Some(CheckerWitness {
                state,
                formula: key,
                witness_type: WitnessType::Atomic,
                children: vec![],
                trace: None,
            }),

            CTLFormula::False => None,

            CTLFormula::Atom(_) => Some(CheckerWitness {
                state,
                formula: key,
                witness_type: WitnessType::Atomic,
                children: vec![],
                trace: None,
            }),

            CTLFormula::Not(inner) => {
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::Atomic,
                    children: self
                        .generate_witness(state, inner)
                        .into_iter()
                        .collect(),
                    trace: None,
                })
            }

            CTLFormula::And(a, b) => {
                let ca = self.generate_witness(state, a);
                let cb = self.generate_witness(state, b);
                let mut children = Vec::new();
                if let Some(wa) = ca {
                    children.push(wa);
                }
                if let Some(wb) = cb {
                    children.push(wb);
                }
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::Conjunction,
                    children,
                    trace: None,
                })
            }

            CTLFormula::Or(a, b) => {
                // Pick whichever sub-formula is satisfied.
                let ca = self.generate_witness(state, a);
                if ca.is_some() {
                    return Some(CheckerWitness {
                        state,
                        formula: key,
                        witness_type: WitnessType::Disjunction,
                        children: ca.into_iter().collect(),
                        trace: None,
                    });
                }
                let cb = self.generate_witness(state, b);
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::Disjunction,
                    children: cb.into_iter().collect(),
                    trace: None,
                })
            }

            CTLFormula::Implies(a, b) => {
                // a → b. Witness: either ¬a or b.
                let cb = self.generate_witness(state, b);
                if cb.is_some() {
                    return Some(CheckerWitness {
                        state,
                        formula: key,
                        witness_type: WitnessType::Disjunction,
                        children: cb.into_iter().collect(),
                        trace: None,
                    });
                }
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::Disjunction,
                    children: vec![],
                    trace: None,
                })
            }

            CTLFormula::EX(inner) => {
                // Find a successor that satisfies inner.
                let inner_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    for &(t, _) in self.model.successors(state) {
                        if inner_sat[t] {
                            let child = self.generate_witness(t, inner);
                            return Some(CheckerWitness {
                                state,
                                formula: key,
                                witness_type: WitnessType::ExistentialNext,
                                children: child.into_iter().collect(),
                                trace: Some(vec![state, t]),
                            });
                        }
                    }
                }
                None
            }

            CTLFormula::AX(inner) => {
                // All successors satisfy inner.
                let mut children = Vec::new();
                let mut trace = vec![state];
                for &(t, _) in self.model.successors(state) {
                    trace.push(t);
                    if let Some(w) = self.generate_witness(t, inner) {
                        children.push(w);
                    }
                }
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::UniversalNext,
                    children,
                    trace: Some(trace),
                })
            }

            CTLFormula::EU(phi, psi) => {
                // Build a path witness: find a path from state through phi-states
                // to a psi-state.
                let phi_key = phi.render();
                let psi_key = psi.render();
                let phi_sat = self.label_cache.get(&phi_key);
                let psi_sat = self.label_cache.get(&psi_key);
                if let (Some(ps), Some(qs)) = (phi_sat, psi_sat) {
                    if let Some(path) = self.find_eu_path(state, ps, qs) {
                        return Some(CheckerWitness {
                            state,
                            formula: key,
                            witness_type: WitnessType::ExistentialUntil,
                            children: vec![],
                            trace: Some(path),
                        });
                    }
                }
                // Fallback if psi holds at state.
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::ExistentialUntil,
                    children: vec![],
                    trace: Some(vec![state]),
                })
            }

            CTLFormula::AU(phi, psi) => {
                let phi_key = phi.render();
                let psi_key = psi.render();
                let phi_sat = self.label_cache.get(&phi_key);
                let psi_sat = self.label_cache.get(&psi_key);
                if let (Some(ps), Some(qs)) = (phi_sat, psi_sat) {
                    if let Some(path) = self.find_eu_path(state, ps, qs) {
                        return Some(CheckerWitness {
                            state,
                            formula: key,
                            witness_type: WitnessType::UniversalUntil,
                            children: vec![],
                            trace: Some(path),
                        });
                    }
                }
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::UniversalUntil,
                    children: vec![],
                    trace: Some(vec![state]),
                })
            }

            CTLFormula::EG(inner) => {
                let inner_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    let cycle = self.find_cycle_through(state, inner_sat);
                    return Some(CheckerWitness {
                        state,
                        formula: key,
                        witness_type: WitnessType::ExistentialGlobally,
                        children: vec![],
                        trace: Some(cycle),
                    });
                }
                None
            }

            CTLFormula::AG(inner) => {
                let inner_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    let cycle = self.find_cycle_through(state, inner_sat);
                    return Some(CheckerWitness {
                        state,
                        formula: key,
                        witness_type: WitnessType::UniversalGlobally,
                        children: vec![],
                        trace: Some(cycle),
                    });
                }
                None
            }

            CTLFormula::EF(inner) => {
                let inner_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    let true_sat = vec![true; self.model.num_states];
                    if let Some(path) = self.find_eu_path(state, &true_sat, inner_sat) {
                        return Some(CheckerWitness {
                            state,
                            formula: key,
                            witness_type: WitnessType::ExistentialUntil,
                            children: vec![],
                            trace: Some(path),
                        });
                    }
                }
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::ExistentialUntil,
                    children: vec![],
                    trace: Some(vec![state]),
                })
            }

            CTLFormula::AF(inner) => {
                let inner_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    let true_sat = vec![true; self.model.num_states];
                    if let Some(path) = self.find_eu_path(state, &true_sat, inner_sat) {
                        return Some(CheckerWitness {
                            state,
                            formula: key,
                            witness_type: WitnessType::UniversalUntil,
                            children: vec![],
                            trace: Some(path),
                        });
                    }
                }
                Some(CheckerWitness {
                    state,
                    formula: key,
                    witness_type: WitnessType::UniversalUntil,
                    children: vec![],
                    trace: Some(vec![state]),
                })
            }

            CTLFormula::Prob(_, _, _) => Some(CheckerWitness {
                state,
                formula: key,
                witness_type: WitnessType::Probabilistic,
                children: vec![],
                trace: None,
            }),
        }
    }

    /// BFS to find a path from `start` through phi-states to a psi-state.
    fn find_eu_path(
        &self,
        start: usize,
        phi_sat: &[bool],
        psi_sat: &[bool],
    ) -> Option<Vec<usize>> {
        if psi_sat[start] {
            return Some(vec![start]);
        }
        if !phi_sat[start] {
            return None;
        }

        let n = self.model.num_states;
        let mut visited = vec![false; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(s) = queue.pop_front() {
            for &(t, _) in self.model.successors(s) {
                if !visited[t] {
                    visited[t] = true;
                    parent[t] = Some(s);
                    if psi_sat[t] {
                        // Reconstruct path.
                        let mut path = vec![t];
                        let mut cur = t;
                        while let Some(p) = parent[cur] {
                            path.push(p);
                            cur = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    if phi_sat[t] {
                        queue.push_back(t);
                    }
                }
            }
        }
        None
    }

    /// Find a cycle (lasso) through states satisfying `sat`, starting from `start`.
    fn find_cycle_through(&self, start: usize, sat: &[bool]) -> Vec<usize> {
        // DFS to find a cycle.
        let mut path = vec![start];
        let mut visited = HashSet::new();
        visited.insert(start);
        let mut current = start;
        let bound = self.model.num_states * 2;

        for _ in 0..bound {
            let mut found_next = false;
            for &(t, _) in self.model.successors(current) {
                if sat[t] {
                    if visited.contains(&t) {
                        // Cycle found: append the repeated state.
                        path.push(t);
                        return path;
                    }
                    path.push(t);
                    visited.insert(t);
                    current = t;
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }
        }
        path
    }

    // -----------------------------------------------------------------------
    // Counterexample generation
    // -----------------------------------------------------------------------

    /// Build a counterexample showing why `state` violates `formula`.
    pub fn generate_counterexample(
        &self,
        state: usize,
        formula: &CTLFormula,
    ) -> Option<CheckerCounterexample> {
        let key = formula.render();

        // Verify the state does NOT satisfy the formula.
        if let Some(sat) = self.label_cache.get(&key) {
            if sat[state] {
                return None; // state satisfies formula, no counterexample
            }
        }

        match formula {
            CTLFormula::True => None, // true is always satisfied

            CTLFormula::False => Some(CheckerCounterexample {
                state,
                formula: key,
                trace: vec![state],
                explanation: "false is never satisfied".to_string(),
            }),

            CTLFormula::Atom(p) => Some(CheckerCounterexample {
                state,
                formula: key.clone(),
                trace: vec![state],
                explanation: format!(
                    "State s{} does not have atomic proposition '{}'",
                    state, p
                ),
            }),

            CTLFormula::Not(inner) => {
                // ¬φ fails means φ holds. Show witness for φ.
                Some(CheckerCounterexample {
                    state,
                    formula: key,
                    trace: vec![state],
                    explanation: format!(
                        "State s{} satisfies {}, so ¬({}) fails",
                        state,
                        inner.render(),
                        inner.render()
                    ),
                })
            }

            CTLFormula::And(a, b) => {
                // Identify which conjunct fails.
                let a_key = a.render();
                let b_key = b.render();
                let a_fails = self
                    .label_cache
                    .get(&a_key)
                    .map(|s| !s[state])
                    .unwrap_or(false);
                let failing = if a_fails { &a_key } else { &b_key };
                Some(CheckerCounterexample {
                    state,
                    formula: key,
                    trace: vec![state],
                    explanation: format!(
                        "State s{} fails conjunct {}",
                        state, failing
                    ),
                })
            }

            CTLFormula::Or(a, b) => Some(CheckerCounterexample {
                state,
                formula: key,
                trace: vec![state],
                explanation: format!(
                    "State s{} satisfies neither {} nor {}",
                    state,
                    a.render(),
                    b.render()
                ),
            }),

            CTLFormula::Implies(a, b) => Some(CheckerCounterexample {
                state,
                formula: key,
                trace: vec![state],
                explanation: format!(
                    "State s{} satisfies {} but not {}",
                    state,
                    a.render(),
                    b.render()
                ),
            }),

            CTLFormula::EX(inner) => {
                // No successor satisfies inner.
                Some(CheckerCounterexample {
                    state,
                    formula: key,
                    trace: vec![state],
                    explanation: format!(
                        "No successor of s{} satisfies {}",
                        state,
                        inner.render()
                    ),
                })
            }

            CTLFormula::AX(inner) => {
                // Find a successor that violates inner.
                let inner_key = inner.render();
                let mut bad_succ = state;
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    for &(t, _) in self.model.successors(state) {
                        if !inner_sat[t] {
                            bad_succ = t;
                            break;
                        }
                    }
                }
                Some(CheckerCounterexample {
                    state,
                    formula: key,
                    trace: vec![state, bad_succ],
                    explanation: format!(
                        "Successor s{} of s{} does not satisfy {}",
                        bad_succ,
                        state,
                        inner.render()
                    ),
                })
            }

            CTLFormula::EU(_, psi) => Some(CheckerCounterexample {
                state,
                formula: key,
                trace: vec![state],
                explanation: format!(
                    "No path from s{} reaches a state satisfying {}",
                    state,
                    psi.render()
                ),
            }),

            CTLFormula::AU(_, psi) => {
                // Find a path that avoids psi.
                let psi_key = psi.render();
                if let Some(psi_sat) = self.label_cache.get(&psi_key) {
                    let path = self.find_violating_path(state, psi_sat);
                    return Some(CheckerCounterexample {
                        state,
                        formula: key,
                        trace: path,
                        explanation: format!(
                            "Found a path from s{} that never reaches {}",
                            state,
                            psi.render()
                        ),
                    });
                }
                Some(CheckerCounterexample {
                    state,
                    formula: key,
                    trace: vec![state],
                    explanation: format!(
                        "Not all paths from s{} reach {}",
                        state,
                        psi.render()
                    ),
                })
            }

            CTLFormula::EG(inner) => Some(CheckerCounterexample {
                state,
                formula: key,
                trace: vec![state],
                explanation: format!(
                    "No infinite path from s{} stays within states satisfying {}",
                    state,
                    inner.render()
                ),
            }),

            CTLFormula::AG(inner) => {
                // Find a reachable state that violates inner.
                let inner_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    let true_sat = vec![true; self.model.num_states];
                    let neg: Vec<bool> = inner_sat.iter().map(|b| !b).collect();
                    if let Some(path) = self.find_eu_path(state, &true_sat, &neg) {
                        return Some(CheckerCounterexample {
                            state,
                            formula: key,
                            trace: path.clone(),
                            explanation: format!(
                                "State s{} reachable from s{} violates {}",
                                path.last().unwrap_or(&state),
                                state,
                                inner.render()
                            ),
                        });
                    }
                }
                Some(CheckerCounterexample {
                    state,
                    formula: key,
                    trace: vec![state],
                    explanation: format!(
                        "Some reachable state from s{} violates {}",
                        state,
                        inner.render()
                    ),
                })
            }

            CTLFormula::EF(inner) => Some(CheckerCounterexample {
                state,
                formula: key,
                trace: vec![state],
                explanation: format!(
                    "No state satisfying {} is reachable from s{}",
                    inner.render(),
                    state
                ),
            }),

            CTLFormula::AF(inner) => {
                let inner_key = inner.render();
                if let Some(inner_sat) = self.label_cache.get(&inner_key) {
                    let path = self.find_violating_path(state, inner_sat);
                    return Some(CheckerCounterexample {
                        state,
                        formula: key,
                        trace: path,
                        explanation: format!(
                            "Found a path from s{} that avoids {}",
                            state,
                            inner.render()
                        ),
                    });
                }
                Some(CheckerCounterexample {
                    state,
                    formula: key,
                    trace: vec![state],
                    explanation: format!(
                        "Some path from s{} never reaches {}",
                        state,
                        inner.render()
                    ),
                })
            }

            CTLFormula::Prob(op, threshold, inner) => Some(CheckerCounterexample {
                state,
                formula: key,
                trace: vec![state],
                explanation: format!(
                    "Probability of {} at s{} does not satisfy {} {}",
                    inner.render(),
                    state,
                    op.render(),
                    threshold
                ),
            }),
        }
    }

    /// Find a path from `start` that avoids states in `target_sat` (for
    /// AU/AF counterexamples).
    fn find_violating_path(&self, start: usize, target_sat: &[bool]) -> Vec<usize> {
        let n = self.model.num_states;
        let mut visited = vec![false; n];
        let mut path = vec![start];
        visited[start] = true;
        let mut current = start;

        let bound = n * 2;
        for _ in 0..bound {
            if target_sat[current] {
                break;
            }
            let mut found = false;
            for &(t, _) in self.model.successors(current) {
                if !visited[t] && !target_sat[t] {
                    visited[t] = true;
                    path.push(t);
                    current = t;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        path
    }

    // -----------------------------------------------------------------------
    // Accessor
    // -----------------------------------------------------------------------

    pub fn complexity(&self) -> &ComplexityTracker {
        &self.complexity
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a simple 3-state model:
    ///
    ///   s0 --1.0--> s1 --1.0--> s2 (terminal)
    ///   labels: s0={a}, s1={b}, s2={c}
    fn simple_chain() -> KripkeModel {
        let mut m = KripkeModel::new(3);
        m.add_transition(0, 1, 1.0);
        m.add_transition(1, 2, 1.0);
        m.add_label(0, "a");
        m.add_label(1, "b");
        m.add_label(2, "c");
        m
    }

    /// Build a model with a cycle:
    ///
    ///   s0 --1.0--> s1 --1.0--> s0  (cycle)
    ///   s0 --1.0--> s2 (terminal, branch)
    ///
    /// Note: s0 has two outgoing edges.
    ///   labels: s0={a}, s1={a,b}, s2={c}
    fn cycle_model() -> KripkeModel {
        let mut m = KripkeModel::new(3);
        m.add_transition(0, 1, 0.5);
        m.add_transition(0, 2, 0.5);
        m.add_transition(1, 0, 1.0);
        m.add_label(0, "a");
        m.add_label(1, "a");
        m.add_label(1, "b");
        m.add_label(2, "c");
        m
    }

    /// Model with dead-end:
    ///
    ///   s0 --1.0--> s1 (terminal)
    ///   s0 --1.0--> s2 --1.0--> s3 (terminal)
    ///   labels: s0={start}, s1={dead}, s2={mid}, s3={goal}
    fn deadend_model() -> KripkeModel {
        let mut m = KripkeModel::new(4);
        m.add_transition(0, 1, 0.5);
        m.add_transition(0, 2, 0.5);
        m.add_transition(2, 3, 1.0);
        m.add_label(0, "start");
        m.add_label(1, "dead");
        m.add_label(2, "mid");
        m.add_label(3, "goal");
        m
    }

    /// Probabilistic branching model:
    ///
    ///   s0 --0.7--> s1 (ok)
    ///   s0 --0.3--> s2 (fail)
    ///   labels: s0={init}, s1={ok}, s2={fail}
    fn prob_model() -> KripkeModel {
        let mut m = KripkeModel::new(3);
        m.add_transition(0, 1, 0.7);
        m.add_transition(0, 2, 0.3);
        m.add_label(0, "init");
        m.add_label(1, "ok");
        m.add_label(2, "fail");
        m
    }

    fn default_checker(model: KripkeModel) -> QCTLFModelChecker {
        QCTLFModelChecker::new(model, ModelCheckConfig::default())
    }

    fn witness_checker(model: KripkeModel) -> QCTLFModelChecker {
        QCTLFModelChecker::new(
            model,
            ModelCheckConfig {
                generate_witnesses: true,
                ..Default::default()
            },
        )
    }

    // -----------------------------------------------------------------------
    // 1. Simple atom check
    // -----------------------------------------------------------------------

    #[test]
    fn test_atom_check() {
        let model = simple_chain();
        let mut checker = default_checker(model);

        let result = checker.check(&CTLFormula::atom("a"));
        assert_eq!(result.satisfying_states, vec![0]);
        assert!(result.satisfaction_map[0]);
        assert!(!result.satisfaction_map[1]);
        assert!(!result.satisfaction_map[2]);

        let result_b = checker.check(&CTLFormula::atom("b"));
        assert_eq!(result_b.satisfying_states, vec![1]);

        let result_c = checker.check(&CTLFormula::atom("c"));
        assert_eq!(result_c.satisfying_states, vec![2]);

        // Non-existent prop
        let result_x = checker.check(&CTLFormula::atom("x"));
        assert!(result_x.satisfying_states.is_empty());
    }

    // -----------------------------------------------------------------------
    // 2. Negation
    // -----------------------------------------------------------------------

    #[test]
    fn test_negation() {
        let model = simple_chain();
        let mut checker = default_checker(model);

        // ¬a should be {s1, s2}
        let result = checker.check(&CTLFormula::not(CTLFormula::atom("a")));
        assert_eq!(result.satisfying_states, vec![1, 2]);

        // ¬¬a should be {s0}
        let result2 =
            checker.check(&CTLFormula::not(CTLFormula::not(CTLFormula::atom("a"))));
        assert_eq!(result2.satisfying_states, vec![0]);
    }

    // -----------------------------------------------------------------------
    // 3. And / Or / Implies
    // -----------------------------------------------------------------------

    #[test]
    fn test_and_or_implies() {
        let model = cycle_model(); // s0={a}, s1={a,b}, s2={c}
        let mut checker = default_checker(model);

        // a ∧ b: only s1
        let result_and = checker.check(&CTLFormula::and(
            CTLFormula::atom("a"),
            CTLFormula::atom("b"),
        ));
        assert_eq!(result_and.satisfying_states, vec![1]);

        // a ∨ c: s0, s1, s2
        let result_or = checker.check(&CTLFormula::or(
            CTLFormula::atom("a"),
            CTLFormula::atom("c"),
        ));
        assert_eq!(result_or.satisfying_states, vec![0, 1, 2]);

        // a → b: ¬a ∨ b = {s1(a∧b), s2(¬a)}  but s0 has a and not b → false
        let result_impl = checker.check(&CTLFormula::implies(
            CTLFormula::atom("a"),
            CTLFormula::atom("b"),
        ));
        assert_eq!(result_impl.satisfying_states, vec![1, 2]);
    }

    // -----------------------------------------------------------------------
    // 4. EX / AX on simple model
    // -----------------------------------------------------------------------

    #[test]
    fn test_ex_ax() {
        let model = simple_chain(); // s0->s1->s2(terminal)
        let mut checker = default_checker(model);

        // EX(b): states with a successor labeled b → s0
        let result_ex = checker.check(&CTLFormula::ex(CTLFormula::atom("b")));
        assert_eq!(result_ex.satisfying_states, vec![0]);

        // EX(c): successor labeled c → s1
        let result_ex2 = checker.check(&CTLFormula::ex(CTLFormula::atom("c")));
        assert_eq!(result_ex2.satisfying_states, vec![1]);

        // AX(b): states where ALL successors have b.
        // s0's only succ is s1(b) → yes
        // s1's only succ is s2(c) → no
        // s2 is terminal → AX vacuously true
        let result_ax = checker.check(&CTLFormula::ax(CTLFormula::atom("b")));
        assert_eq!(result_ax.satisfying_states, vec![0, 2]);
    }

    // -----------------------------------------------------------------------
    // 5. EU reachability
    // -----------------------------------------------------------------------

    #[test]
    fn test_eu_reachability() {
        let model = simple_chain(); // s0->s1->s2
        let mut checker = default_checker(model);

        // E[true U c]: can reach a c-state → s0, s1, s2
        let result = checker.check(&CTLFormula::eu(CTLFormula::True, CTLFormula::atom("c")));
        assert_eq!(result.satisfying_states, vec![0, 1, 2]);

        // E[true U a]: only s0 has a, and you're already there
        let result2 = checker.check(&CTLFormula::eu(CTLFormula::True, CTLFormula::atom("a")));
        assert_eq!(result2.satisfying_states, vec![0]);

        // E[b U c]: must stay in b-states until c.
        // s1 has b and succ s2 has c → s1 satisfies.
        // s0 does not have b → cannot start from s0.
        let result3 = checker.check(&CTLFormula::eu(
            CTLFormula::atom("b"),
            CTLFormula::atom("c"),
        ));
        assert_eq!(result3.satisfying_states, vec![1, 2]);
    }

    // -----------------------------------------------------------------------
    // 6. AU
    // -----------------------------------------------------------------------

    #[test]
    fn test_au() {
        let model = simple_chain(); // s0->s1->s2
        let mut checker = default_checker(model);

        // A[true U c]: all paths from each state eventually reach c.
        // s2: already c → yes
        // s1: only path goes to s2(c) → yes
        // s0: only path s0->s1->s2 → yes
        let result = checker.check(&CTLFormula::au(CTLFormula::True, CTLFormula::atom("c")));
        assert_eq!(result.satisfying_states, vec![0, 1, 2]);

        // In deadend_model: s0->s1(dead, terminal) and s0->s2->s3(goal)
        // A[true U goal]: s3 yes, s2 yes(only succ s3), s1 no(terminal, not goal),
        // s0 no(s1 branch fails)
        let dm = deadend_model();
        let mut checker2 = default_checker(dm);
        let result2 = checker2.check(&CTLFormula::au(
            CTLFormula::True,
            CTLFormula::atom("goal"),
        ));
        assert_eq!(result2.satisfying_states, vec![2, 3]);
    }

    // -----------------------------------------------------------------------
    // 7. EG / AG on cycles
    // -----------------------------------------------------------------------

    #[test]
    fn test_eg_ag_cycles() {
        let model = cycle_model(); // s0->s1->s0 cycle, s0->s2 terminal
        let mut checker = default_checker(model);

        // EG(a): can we stay in a-states forever?
        // s0 and s1 both have a, and s0->s1->s0 is a cycle of a-states.
        // So s0 and s1 satisfy EG(a).
        let result_eg = checker.check(&CTLFormula::eg(CTLFormula::atom("a")));
        assert_eq!(result_eg.satisfying_states, vec![0, 1]);

        // AG(a): ALL paths stay in a forever.
        // s0 has a path s0->s2 which does not have a → s0 fails.
        // s1 has only s1->s0, and s0 fails → s1 fails.
        let result_ag = checker.check(&CTLFormula::ag(CTLFormula::atom("a")));
        // Neither s0 nor s1 satisfy AG(a) because s0 can go to s2 which lacks a.
        assert!(result_ag.satisfying_states.is_empty());
    }

    // -----------------------------------------------------------------------
    // 8. EF / AF derived operators
    // -----------------------------------------------------------------------

    #[test]
    fn test_ef_af() {
        let model = simple_chain(); // s0->s1->s2
        let mut checker = default_checker(model);

        // EF(c) = E[true U c]
        let result_ef = checker.check(&CTLFormula::ef(CTLFormula::atom("c")));
        assert_eq!(result_ef.satisfying_states, vec![0, 1, 2]);

        // AF(c) = A[true U c]
        let result_af = checker.check(&CTLFormula::af(CTLFormula::atom("c")));
        assert_eq!(result_af.satisfying_states, vec![0, 1, 2]);

        // In deadend_model: EF(goal) vs AF(goal)
        let dm = deadend_model();
        let mut checker2 = default_checker(dm);

        // EF(goal): s0 can reach s3 via s0->s2->s3 → yes
        let result_ef2 = checker2.check(&CTLFormula::ef(CTLFormula::atom("goal")));
        assert!(result_ef2.satisfaction_map[0]);
        assert!(result_ef2.satisfaction_map[2]);
        assert!(result_ef2.satisfaction_map[3]);

        // AF(goal): s0 has a path to s1(dead, terminal, not goal) → fails
        let result_af2 = checker2.check(&CTLFormula::af(CTLFormula::atom("goal")));
        assert!(!result_af2.satisfaction_map[0]);
        assert!(result_af2.satisfaction_map[2]);
        assert!(result_af2.satisfaction_map[3]);
    }

    // -----------------------------------------------------------------------
    // 9. Probabilistic operator
    // -----------------------------------------------------------------------

    #[test]
    fn test_probabilistic_operator() {
        let model = prob_model(); // s0--0.7-->s1(ok), s0--0.3-->s2(fail)
        let mut checker = default_checker(model);

        // P[>=0.5](EX(ok)): probability of EX(ok) at s0 is 0.7 >= 0.5 → true
        let formula = CTLFormula::Prob(
            CompOp::Ge,
            0.5,
            Box::new(CTLFormula::ex(CTLFormula::atom("ok"))),
        );
        let result = checker.check(&formula);
        assert!(result.satisfaction_map[0]);

        // P[>=0.9](EX(ok)): 0.7 < 0.9 → false at s0
        let formula2 = CTLFormula::Prob(
            CompOp::Ge,
            0.9,
            Box::new(CTLFormula::ex(CTLFormula::atom("ok"))),
        );
        let result2 = checker.check(&formula2);
        assert!(!result2.satisfaction_map[0]);

        // P[<=0.5](EX(fail)): probability of EX(fail) at s0 is 0.3 <= 0.5 → true
        let formula3 = CTLFormula::Prob(
            CompOp::Le,
            0.5,
            Box::new(CTLFormula::ex(CTLFormula::atom("fail"))),
        );
        let result3 = checker.check(&formula3);
        assert!(result3.satisfaction_map[0]);
    }

    // -----------------------------------------------------------------------
    // 10. Satisfaction degree computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_satisfaction_degree() {
        let model = simple_chain();
        let mut checker = default_checker(model);

        // Atom: degree is 1.0 if holds, 0.0 otherwise.
        let f = CTLFormula::atom("a");
        checker.check(&f); // populate cache
        assert!((checker.compute_sat_degree(0, &f) - 1.0).abs() < 1e-9);
        assert!((checker.compute_sat_degree(1, &f) - 0.0).abs() < 1e-9);

        // Not: degree = 1 - inner
        let f_not = CTLFormula::not(CTLFormula::atom("a"));
        checker.check(&f_not);
        assert!((checker.compute_sat_degree(0, &f_not) - 0.0).abs() < 1e-9);
        assert!((checker.compute_sat_degree(1, &f_not) - 1.0).abs() < 1e-9);

        // And: min
        let model2 = cycle_model();
        let mut checker2 = default_checker(model2);
        let f_and = CTLFormula::and(CTLFormula::atom("a"), CTLFormula::atom("b"));
        checker2.check(&f_and);
        // s1 has both a and b → min(1,1) = 1
        assert!((checker2.compute_sat_degree(1, &f_and) - 1.0).abs() < 1e-9);
        // s0 has a but not b → min(1,0) = 0
        assert!((checker2.compute_sat_degree(0, &f_and) - 0.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 11. Witness generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_witness_generation() {
        let model = simple_chain();
        let mut checker = witness_checker(model);

        // Atom witness
        let f = CTLFormula::atom("a");
        let result = checker.check(&f);
        assert!(result.witness.is_some());
        let w = result.witness.unwrap();
        assert_eq!(w.state, 0);
        assert_eq!(w.witness_type, WitnessType::Atomic);

        // EX witness
        let f_ex = CTLFormula::ex(CTLFormula::atom("b"));
        let result_ex = checker.check(&f_ex);
        assert!(result_ex.witness.is_some());
        let w_ex = result_ex.witness.unwrap();
        assert_eq!(w_ex.state, 0);
        assert_eq!(w_ex.witness_type, WitnessType::ExistentialNext);
        assert!(w_ex.trace.is_some());
        let trace = w_ex.trace.unwrap();
        assert_eq!(trace, vec![0, 1]);
    }

    // -----------------------------------------------------------------------
    // 12. Counterexample generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_counterexample_generation() {
        let model = simple_chain();
        let mut checker = witness_checker(model);

        // s0 does not satisfy atom "c"
        let f = CTLFormula::atom("c");
        let result = checker.check(&f);
        // initial state s0 doesn't satisfy c, so counterexample should be generated
        assert!(result.counterexample.is_some());
        let ce = result.counterexample.unwrap();
        assert_eq!(ce.state, 0);
        assert!(ce.explanation.contains("does not have"));

        // AG(a) fails on chain: s1 doesn't have a
        let f_ag = CTLFormula::ag(CTLFormula::atom("a"));
        let result_ag = checker.check(&f_ag);
        assert!(result_ag.counterexample.is_some());
    }

    // -----------------------------------------------------------------------
    // 13. Dead-end states
    // -----------------------------------------------------------------------

    #[test]
    fn test_dead_end_states() {
        let model = deadend_model();
        let mut checker = default_checker(model);

        // s1 is terminal. AX(anything) should be vacuously true at s1.
        let f_ax = CTLFormula::ax(CTLFormula::atom("goal"));
        let result = checker.check(&f_ax);
        assert!(result.satisfaction_map[1]); // terminal → vacuously true
        assert!(!result.satisfaction_map[0]); // s0->s1 doesn't have goal

        // EX at terminal state should be false (no successors).
        let f_ex = CTLFormula::ex(CTLFormula::atom("goal"));
        let result2 = checker.check(&f_ex);
        assert!(!result2.satisfaction_map[1]); // no successors

        // EG at terminal: terminal state has no successors, so it cannot
        // have a successor in sat → EG should be false.
        let f_eg = CTLFormula::eg(CTLFormula::atom("dead"));
        let result3 = checker.check(&f_eg);
        assert!(!result3.satisfaction_map[1]);
    }

    // -----------------------------------------------------------------------
    // 14. Caching behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_caching() {
        let model = simple_chain();
        let mut checker = default_checker(model);

        let f = CTLFormula::atom("a");
        checker.check(&f);
        let misses_after_first = checker.complexity().cache_misses;
        let hits_after_first = checker.complexity().cache_hits;

        // Second check of same formula should hit cache.
        checker.check(&f);
        assert!(checker.complexity().cache_hits > hits_after_first);

        // Check with caching disabled.
        let model2 = simple_chain();
        let mut checker2 = QCTLFModelChecker::new(
            model2,
            ModelCheckConfig {
                cache_results: false,
                ..Default::default()
            },
        );
        checker2.check(&f);
        checker2.check(&f);
        assert_eq!(checker2.complexity().cache_hits, 0);
        // Without cache, each call is a cache miss (but we skip cache entirely).
        assert_eq!(checker2.complexity().cache_misses, 0);
    }

    // -----------------------------------------------------------------------
    // 15. Complex nested formula
    // -----------------------------------------------------------------------

    #[test]
    fn test_complex_nested_formula() {
        let model = cycle_model(); // s0->s1->s0, s0->s2
        let mut checker = default_checker(model);

        // EF(a ∧ b): can reach a state with both a and b → s1
        // From s0: EF(a∧b) → yes (s0->s1)
        // From s1: already has a∧b → yes
        // From s2: terminal, no a∧b → no
        let f = CTLFormula::ef(CTLFormula::and(
            CTLFormula::atom("a"),
            CTLFormula::atom("b"),
        ));
        let result = checker.check(&f);
        assert!(result.satisfaction_map[0]);
        assert!(result.satisfaction_map[1]);
        assert!(!result.satisfaction_map[2]);

        // AG(a → EX(a)): on all paths, whenever a holds, some successor has a.
        // s0 has a and succ s1 has a → ok at s0 locally.
        // But s0 also has succ s2 which has c, not a. AG requires all paths...
        // s0 can reach s2 which doesn't have a, but a doesn't hold at s2,
        // so a→EX(a) is vacuously true at s2. So we need to check carefully.
        // s2 is terminal: a→EX(a) at s2: a is false so implication true.
        // s1 has a, succ is s0 which has a, so EX(a) true at s1.
        // s0 has a, succs s1(a) and s2(c). EX(a) true at s0 (s1 has a).
        // So a→EX(a) holds everywhere. AG(a→EX(a)) should hold from every state.
        let f2 = CTLFormula::ag(CTLFormula::implies(
            CTLFormula::atom("a"),
            CTLFormula::ex(CTLFormula::atom("a")),
        ));
        let result2 = checker.check(&f2);
        assert!(result2.satisfaction_map[0]);
        assert!(result2.satisfaction_map[1]);
        assert!(result2.satisfaction_map[2]);
    }

    // -----------------------------------------------------------------------
    // 16. Formula methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_formula_methods() {
        let f = CTLFormula::eu(
            CTLFormula::and(CTLFormula::atom("a"), CTLFormula::atom("b")),
            CTLFormula::not(CTLFormula::atom("c")),
        );

        assert_eq!(f.depth(), 2);
        assert_eq!(f.size(), 6);

        let mut atoms = f.atoms();
        atoms.sort();
        assert_eq!(atoms, vec!["a", "b", "c"]);

        let rendered = f.render();
        assert!(rendered.contains("U"));
    }

    // -----------------------------------------------------------------------
    // 17. KripkeModel methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_kripke_model_methods() {
        let model = simple_chain();

        assert_eq!(model.successors(0), &[(1, 1.0)]);
        assert!(model.is_terminal(2));
        assert!(!model.is_terminal(0));

        let preds = model.predecessors(1);
        assert_eq!(preds, vec![(0, 1.0)]);

        assert!(model.has_label(0, "a"));
        assert!(!model.has_label(0, "b"));

        let reachable = model.reachable_from(0);
        assert_eq!(reachable.len(), 3);
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
        assert!(reachable.contains(&2));

        let reachable2 = model.reachable_from(2);
        assert_eq!(reachable2, vec![2]);

        assert_eq!(model.states_with_label("b"), vec![1]);
    }

    // -----------------------------------------------------------------------
    // 18. ModelCheckResult summary and ComplexityTracker
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_summary_and_complexity() {
        let model = simple_chain();
        let mut checker = default_checker(model);

        let result = checker.check(&CTLFormula::atom("a"));
        assert_eq!(result.num_satisfying(), 1);
        assert!((result.fraction_satisfying() - 1.0 / 3.0).abs() < 1e-9);

        let summary = result.summary();
        assert!(summary.contains("1/3"));

        let comp = checker.complexity();
        assert!(comp.states_visited > 0);
        assert!(comp.formulas_checked > 0);

        let comp_summary = comp.summary();
        assert!(comp_summary.contains("states visited"));
    }

    // -----------------------------------------------------------------------
    // 19. State-level check
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_state() {
        let model = simple_chain();
        let mut checker = witness_checker(model);

        let sr = checker.check_state(0, &CTLFormula::atom("a"));
        assert!(sr.satisfied);
        assert!((sr.degree - 1.0).abs() < 1e-9);

        let sr2 = checker.check_state(1, &CTLFormula::atom("a"));
        assert!(!sr2.satisfied);
        assert!((sr2.degree - 0.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 20. Witness / counterexample rendering
    // -----------------------------------------------------------------------

    #[test]
    fn test_witness_counterexample_rendering() {
        let w = CheckerWitness {
            state: 0,
            formula: "EX(b)".to_string(),
            witness_type: WitnessType::ExistentialNext,
            children: vec![CheckerWitness {
                state: 1,
                formula: "b".to_string(),
                witness_type: WitnessType::Atomic,
                children: vec![],
                trace: None,
            }],
            trace: Some(vec![0, 1]),
        };
        let rendered = w.render();
        assert!(rendered.contains("s0"));
        assert!(rendered.contains("EX(b)"));
        assert_eq!(w.depth(), 1);

        let ce = CheckerCounterexample {
            state: 2,
            formula: "a".to_string(),
            trace: vec![2],
            explanation: "s2 lacks a".to_string(),
        };
        let ce_rendered = ce.render();
        assert!(ce_rendered.contains("s2"));
        assert_eq!(ce.length(), 1);
    }

    // -----------------------------------------------------------------------
    // 21. CompOp evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn test_comp_op() {
        assert!(CompOp::Ge.eval(0.7, 0.5));
        assert!(!CompOp::Ge.eval(0.3, 0.5));
        assert!(CompOp::Gt.eval(0.6, 0.5));
        assert!(!CompOp::Gt.eval(0.5, 0.5));
        assert!(CompOp::Le.eval(0.3, 0.5));
        assert!(CompOp::Lt.eval(0.4, 0.5));
        assert!(!CompOp::Lt.eval(0.5, 0.5));
        assert!(CompOp::Eq.eval(0.5, 0.5));
        assert!(!CompOp::Eq.eval(0.5, 0.6));
    }
}
