//! Denotational semantics of QCTL_F on finite probabilistic coalgebras.
//!
//! Provides quantitative satisfaction degree computation on [0,1],
//! fixed-point iteration for temporal operators (μ-calculus style),
//! and semantic equivalence checking.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use ordered_float::OrderedFloat;

use super::syntax::{
    BoolOp, ComparisonOp, Formula, PathQuantifier, TemporalOp,
};

// ───────────────────────────────────────────────────────────────────────────────
// Local types (will be replaced by coalgebra module types later)
// ───────────────────────────────────────────────────────────────────────────────

/// A state identifier (local alias).
pub type StateId = String;

/// Quantitative satisfaction degree in [0,1].
pub type SatisfactionDegree = f64;

/// Maps state → satisfaction degree.
pub type SatMap = BTreeMap<StateId, SatisfactionDegree>;

// ───────────────────────────────────────────────────────────────────────────────
// Kripke structure — finite probabilistic transition system
// ───────────────────────────────────────────────────────────────────────────────

/// A state in a finite Kripke structure with probabilistic transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KripkeState {
    /// State identifier
    pub id: StateId,
    /// Atomic propositions true at this state
    pub labels: BTreeSet<String>,
    /// Quantitative labeling: for each proposition, a value in [0,1]
    pub quant_labels: BTreeMap<String, f64>,
}

impl KripkeState {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            labels: BTreeSet::new(),
            quant_labels: BTreeMap::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        let l = label.into();
        self.labels.insert(l.clone());
        self.quant_labels.insert(l, 1.0);
        self
    }

    pub fn with_quant_label(mut self, label: impl Into<String>, value: f64) -> Self {
        let l = label.into();
        self.quant_labels.insert(l.clone(), value.clamp(0.0, 1.0));
        if value >= 0.5 {
            self.labels.insert(l);
        }
        self
    }

    /// Boolean satisfaction of an atomic proposition.
    pub fn satisfies(&self, prop: &str) -> bool {
        self.labels.contains(prop)
    }

    /// Quantitative satisfaction: returns the quantitative label, or 1.0/0.0 for boolean.
    pub fn quant_satisfies(&self, prop: &str) -> f64 {
        if let Some(&v) = self.quant_labels.get(prop) {
            v
        } else if self.labels.contains(prop) {
            1.0
        } else {
            0.0
        }
    }
}

/// A probabilistic transition between states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub from: StateId,
    pub to: StateId,
    /// Transition probability in [0,1]
    pub probability: f64,
    /// Optional action label
    pub action: Option<String>,
}

impl Transition {
    pub fn new(from: impl Into<String>, to: impl Into<String>, prob: f64) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            probability: prob.clamp(0.0, 1.0),
            action: None,
        }
    }

    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }
}

/// A finite Kripke structure with probabilistic transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KripkeStructure {
    /// States indexed by ID
    pub states: BTreeMap<StateId, KripkeState>,
    /// Transitions (from → list of (to, probability))
    pub transitions: BTreeMap<StateId, Vec<(StateId, f64)>>,
    /// Initial states
    pub initial: Vec<StateId>,
}

impl KripkeStructure {
    pub fn new() -> Self {
        Self {
            states: BTreeMap::new(),
            transitions: BTreeMap::new(),
            initial: Vec::new(),
        }
    }

    /// Add a state. If initial is true, mark it as an initial state.
    pub fn add_state(&mut self, state: KripkeState, initial: bool) {
        let id = state.id.clone();
        self.states.insert(id.clone(), state);
        if initial {
            self.initial.push(id);
        }
    }

    /// Add a probabilistic transition.
    pub fn add_transition(&mut self, from: impl Into<String>, to: impl Into<String>, prob: f64) {
        let f = from.into();
        let t = to.into();
        self.transitions.entry(f).or_default().push((t, prob.clamp(0.0, 1.0)));
    }

    /// Get successors with probabilities.
    pub fn successors(&self, state: &str) -> &[(StateId, f64)] {
        match self.transitions.get(state) {
            Some(succs) => succs.as_slice(),
            None => &[],
        }
    }

    /// Get all state IDs.
    pub fn state_ids(&self) -> Vec<&StateId> {
        self.states.keys().collect()
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Number of transitions.
    pub fn num_transitions(&self) -> usize {
        self.transitions.values().map(|v| v.len()).sum()
    }

    /// Check that all transition probabilities from each state sum to ≤ 1.
    pub fn is_well_formed(&self) -> bool {
        for (_, succs) in &self.transitions {
            let total: f64 = succs.iter().map(|(_, p)| p).sum();
            if total > 1.0 + 1e-9 {
                return false;
            }
        }
        true
    }

    /// Normalize transition probabilities so they sum to exactly 1 from each state.
    pub fn normalize(&mut self) {
        for (_, succs) in self.transitions.iter_mut() {
            let total: f64 = succs.iter().map(|(_, p)| p).sum();
            if total > 1e-12 {
                for (_, p) in succs.iter_mut() {
                    *p /= total;
                }
            }
        }
    }

    /// Get the predecessors of a state (states that have transitions to it).
    pub fn predecessors(&self, state: &str) -> Vec<(StateId, f64)> {
        let mut preds = Vec::new();
        for (from, succs) in &self.transitions {
            for (to, p) in succs {
                if to == state {
                    preds.push((from.clone(), *p));
                }
            }
        }
        preds
    }

    /// Check if a state is a deadlock (no outgoing transitions).
    pub fn is_deadlock(&self, state: &str) -> bool {
        self.transitions.get(state).map_or(true, |s| s.is_empty())
    }
}

impl Default for KripkeStructure {
    fn default() -> Self { Self::new() }
}

// ───────────────────────────────────────────────────────────────────────────────
// SemanticEvaluator — quantitative QCTL_F model checking
// ───────────────────────────────────────────────────────────────────────────────

/// Configuration for semantic evaluation.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Maximum number of fixed-point iterations.
    pub max_iterations: usize,
    /// Convergence threshold for fixed-point iteration.
    pub epsilon: f64,
    /// Whether to use quantitative (fuzzy) semantics vs boolean.
    pub quantitative: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            epsilon: 1e-10,
            quantitative: true,
        }
    }
}

/// Evaluates QCTL_F formulas on Kripke structures.
pub struct SemanticEvaluator {
    pub config: EvalConfig,
}

impl SemanticEvaluator {
    pub fn new() -> Self {
        Self { config: EvalConfig::default() }
    }

    pub fn with_config(config: EvalConfig) -> Self {
        Self { config }
    }

    /// Evaluate a formula on the Kripke structure, returning a satisfaction map.
    /// For each state, the map gives the quantitative satisfaction degree in [0,1].
    pub fn evaluate(&self, ks: &KripkeStructure, formula: &Formula) -> SatMap {
        self.eval_rec(ks, formula, &HashMap::new())
    }

    /// Evaluate and return the satisfaction degree at a specific state.
    pub fn evaluate_at(&self, ks: &KripkeStructure, formula: &Formula, state: &str) -> SatisfactionDegree {
        let sat_map = self.evaluate(ks, formula);
        sat_map.get(state).copied().unwrap_or(0.0)
    }

    /// Evaluate and return the minimum satisfaction degree across initial states.
    pub fn evaluate_initial(&self, ks: &KripkeStructure, formula: &Formula) -> SatisfactionDegree {
        let sat_map = self.evaluate(ks, formula);
        ks.initial.iter()
            .map(|s| sat_map.get(s).copied().unwrap_or(0.0))
            .fold(f64::INFINITY, f64::min)
            .max(0.0)
    }

    /// Recursive formula evaluation.
    fn eval_rec(
        &self,
        ks: &KripkeStructure,
        formula: &Formula,
        env: &HashMap<String, SatMap>,
    ) -> SatMap {
        match formula {
            Formula::Bool(true) => self.constant_map(ks, 1.0),
            Formula::Bool(false) => self.constant_map(ks, 0.0),
            Formula::QVal(v) => self.constant_map(ks, v.into_inner()),

            Formula::Atom(prop) => {
                let mut map = BTreeMap::new();
                for (id, state) in &ks.states {
                    let val = if self.config.quantitative {
                        state.quant_satisfies(prop)
                    } else {
                        if state.satisfies(prop) { 1.0 } else { 0.0 }
                    };
                    map.insert(id.clone(), val);
                }
                map
            }

            Formula::Var(v) => {
                env.get(v).cloned().unwrap_or_else(|| self.constant_map(ks, 0.0))
            }

            Formula::Not(inner) => {
                let inner_map = self.eval_rec(ks, inner, env);
                inner_map.into_iter().map(|(s, v)| (s, 1.0 - v)).collect()
            }

            Formula::BoolBin { op, lhs, rhs } => {
                let l_map = self.eval_rec(ks, lhs, env);
                let r_map = self.eval_rec(ks, rhs, env);
                self.apply_bool_op(ks, *op, &l_map, &r_map)
            }

            Formula::Next { quantifier, inner } => {
                let inner_map = self.eval_rec(ks, inner, env);
                self.eval_next(ks, *quantifier, &inner_map)
            }

            Formula::Globally { quantifier, inner } => {
                let inner_map = self.eval_rec(ks, inner, env);
                self.eval_globally(ks, *quantifier, &inner_map)
            }

            Formula::Finally { quantifier, inner } => {
                let inner_map = self.eval_rec(ks, inner, env);
                self.eval_finally(ks, *quantifier, &inner_map)
            }

            Formula::Until { quantifier, hold, goal } => {
                let hold_map = self.eval_rec(ks, hold, env);
                let goal_map = self.eval_rec(ks, goal, env);
                self.eval_until(ks, *quantifier, &hold_map, &goal_map)
            }

            Formula::Release { quantifier, trigger, invariant } => {
                let trig_map = self.eval_rec(ks, trigger, env);
                let inv_map = self.eval_rec(ks, invariant, env);
                self.eval_release(ks, *quantifier, &trig_map, &inv_map)
            }

            Formula::BoundedUntil { quantifier, hold, goal, bound } => {
                let hold_map = self.eval_rec(ks, hold, env);
                let goal_map = self.eval_rec(ks, goal, env);
                self.eval_bounded_until(ks, *quantifier, &hold_map, &goal_map, *bound)
            }

            Formula::ProbBound { op, threshold, inner } => {
                // P[⊳p](φ) — the probabilistic bound.
                // Evaluate the inner formula, then check the probabilistic bound
                // at each state. For existential path formulas inside, this is the
                // probability of the set of paths satisfying φ.
                // For non-path inner formulas, we treat this as:
                // "the probability of φ holding in the next state satisfies the bound"
                let inner_map = self.eval_rec(ks, inner, env);
                let mut result = BTreeMap::new();
                for (id, _) in &ks.states {
                    // The inner_map gives the satisfaction degree of the inner formula.
                    // The probability bound is checked against this degree.
                    let sat_degree = inner_map.get(id).copied().unwrap_or(0.0);
                    let val = if op.evaluate(sat_degree, *threshold) {
                        1.0
                    } else {
                        // Quantitative: how close is sat_degree to satisfying the bound?
                        if self.config.quantitative {
                            match op {
                                ComparisonOp::Ge => (sat_degree / threshold.max(1e-12)).min(1.0),
                                ComparisonOp::Gt => (sat_degree / (threshold + 1e-12)).min(1.0),
                                ComparisonOp::Le => ((1.0 - sat_degree) / (1.0 - threshold).max(1e-12)).min(1.0),
                                ComparisonOp::Lt => ((1.0 - sat_degree) / (1.0 - threshold + 1e-12)).min(1.0),
                                ComparisonOp::Eq => 1.0 - (sat_degree - threshold).abs(),
                            }
                        } else {
                            0.0
                        }
                    };
                    result.insert(id.clone(), val);
                }
                result
            }

            Formula::ExpBound { op, threshold, temporal, inner } => {
                // E[⊳v] temporal φ — expectation bound on temporal operator
                let inner_map = self.eval_rec(ks, inner, env);
                let temporal_map = match temporal {
                    TemporalOp::X => self.eval_expected_next(ks, &inner_map),
                    TemporalOp::F => self.eval_expected_finally(ks, &inner_map),
                    TemporalOp::G => self.eval_expected_globally(ks, &inner_map),
                    _ => inner_map.clone(),
                };
                let mut result = BTreeMap::new();
                for (id, _) in &ks.states {
                    let expected = temporal_map.get(id).copied().unwrap_or(0.0);
                    let val = if op.evaluate(expected, *threshold) { 1.0 } else { 0.0 };
                    result.insert(id.clone(), val);
                }
                result
            }

            Formula::GradedModality { grade, inner } => {
                // ⟨k⟩φ: at least k successors satisfy φ
                let inner_map = self.eval_rec(ks, inner, env);
                let mut result = BTreeMap::new();
                for (id, _) in &ks.states {
                    let succs = ks.successors(id);
                    let count = succs.iter()
                        .filter(|(s, _)| inner_map.get(s).copied().unwrap_or(0.0) > 0.5)
                        .count();
                    let val = if count >= *grade as usize { 1.0 } else {
                        count as f64 / (*grade as f64).max(1.0)
                    };
                    result.insert(id.clone(), val.min(1.0));
                }
                result
            }

            Formula::FixedPoint { is_least, variable, body } => {
                self.eval_fixedpoint(ks, *is_least, variable, body, env)
            }
        }
    }

    // ── helpers ──

    fn constant_map(&self, ks: &KripkeStructure, val: f64) -> SatMap {
        ks.states.keys().map(|id| (id.clone(), val)).collect()
    }

    fn apply_bool_op(
        &self,
        ks: &KripkeStructure,
        op: BoolOp,
        l: &SatMap,
        r: &SatMap,
    ) -> SatMap {
        ks.states.keys().map(|id| {
            let lv = l.get(id).copied().unwrap_or(0.0);
            let rv = r.get(id).copied().unwrap_or(0.0);
            let val = match op {
                BoolOp::And => lv.min(rv),   // t-norm (Gödel)
                BoolOp::Or => lv.max(rv),    // t-conorm
                BoolOp::Implies => (1.0 - lv).max(rv), // Gödel implication
                BoolOp::Iff => 1.0 - (lv - rv).abs(),  // agreement
            };
            (id.clone(), val)
        }).collect()
    }

    /// EX φ: max over successors weighted by probability.
    /// AX φ: min over successors weighted by probability.
    /// Quantitative: expected value over successor distribution.
    fn eval_next(
        &self,
        ks: &KripkeStructure,
        quantifier: PathQuantifier,
        inner: &SatMap,
    ) -> SatMap {
        let mut result = BTreeMap::new();
        for (id, _) in &ks.states {
            let succs = ks.successors(id);
            if succs.is_empty() {
                // Deadlock state: AX is vacuously true (1.0), EX is false (0.0)
                let val = match quantifier {
                    PathQuantifier::All => 1.0,
                    PathQuantifier::Exists => 0.0,
                };
                result.insert(id.clone(), val);
            } else {
                let val = match quantifier {
                    PathQuantifier::All => {
                        // For probabilistic: expected value (weighted min over paths)
                        let mut expected = 0.0;
                        let mut total_p = 0.0;
                        for (s, p) in succs {
                            let sv = inner.get(s).copied().unwrap_or(0.0);
                            expected += p * sv;
                            total_p += p;
                        }
                        if total_p > 1e-12 { expected / total_p } else { 1.0 }
                    }
                    PathQuantifier::Exists => {
                        // Maximum satisfaction among successors
                        succs.iter()
                            .map(|(s, _p)| inner.get(s).copied().unwrap_or(0.0))
                            .fold(0.0_f64, f64::max)
                    }
                };
                result.insert(id.clone(), val);
            }
        }
        result
    }

    /// AG φ = νX.(φ ∧ AX X) — greatest fixed point
    /// EG φ = νX.(φ ∧ EX X) — greatest fixed point
    fn eval_globally(
        &self,
        ks: &KripkeStructure,
        quantifier: PathQuantifier,
        inner: &SatMap,
    ) -> SatMap {
        // Start with all 1s (greatest fixed point) and iterate down
        let mut current: SatMap = ks.states.keys()
            .map(|id| (id.clone(), inner.get(id).copied().unwrap_or(0.0)))
            .collect();

        for _ in 0..self.config.max_iterations {
            let next_map = self.eval_next(ks, quantifier, &current);
            let mut new_current = BTreeMap::new();
            let mut changed = false;

            for (id, _) in &ks.states {
                let phi_val = inner.get(id).copied().unwrap_or(0.0);
                let next_val = next_map.get(id).copied().unwrap_or(0.0);
                let new_val = phi_val.min(next_val);
                let old_val = current.get(id).copied().unwrap_or(0.0);

                if (new_val - old_val).abs() > self.config.epsilon {
                    changed = true;
                }
                new_current.insert(id.clone(), new_val);
            }

            current = new_current;
            if !changed { break; }
        }

        current
    }

    /// AF φ = μX.(φ ∨ AX X) — least fixed point
    /// EF φ = μX.(φ ∨ EX X) — least fixed point
    fn eval_finally(
        &self,
        ks: &KripkeStructure,
        quantifier: PathQuantifier,
        inner: &SatMap,
    ) -> SatMap {
        // Start with all 0s (least fixed point) and iterate up
        let mut current: SatMap = ks.states.keys()
            .map(|id| (id.clone(), 0.0))
            .collect();

        for _ in 0..self.config.max_iterations {
            let next_map = self.eval_next(ks, quantifier, &current);
            let mut new_current = BTreeMap::new();
            let mut changed = false;

            for (id, _) in &ks.states {
                let phi_val = inner.get(id).copied().unwrap_or(0.0);
                let next_val = next_map.get(id).copied().unwrap_or(0.0);
                let new_val = phi_val.max(next_val);
                let old_val = current.get(id).copied().unwrap_or(0.0);

                if (new_val - old_val).abs() > self.config.epsilon {
                    changed = true;
                }
                new_current.insert(id.clone(), new_val);
            }

            current = new_current;
            if !changed { break; }
        }

        current
    }

    /// A[φ U ψ] = μX.(ψ ∨ (φ ∧ AX X))
    /// E[φ U ψ] = μX.(ψ ∨ (φ ∧ EX X))
    fn eval_until(
        &self,
        ks: &KripkeStructure,
        quantifier: PathQuantifier,
        hold: &SatMap,
        goal: &SatMap,
    ) -> SatMap {
        let mut current: SatMap = ks.states.keys()
            .map(|id| (id.clone(), 0.0))
            .collect();

        for _ in 0..self.config.max_iterations {
            let next_map = self.eval_next(ks, quantifier, &current);
            let mut new_current = BTreeMap::new();
            let mut changed = false;

            for (id, _) in &ks.states {
                let goal_val = goal.get(id).copied().unwrap_or(0.0);
                let hold_val = hold.get(id).copied().unwrap_or(0.0);
                let next_val = next_map.get(id).copied().unwrap_or(0.0);
                let new_val = goal_val.max(hold_val.min(next_val));
                let old_val = current.get(id).copied().unwrap_or(0.0);

                if (new_val - old_val).abs() > self.config.epsilon {
                    changed = true;
                }
                new_current.insert(id.clone(), new_val);
            }

            current = new_current;
            if !changed { break; }
        }

        current
    }

    /// A[φ R ψ] = νX.(ψ ∧ (φ ∨ AX X))
    /// E[φ R ψ] = νX.(ψ ∧ (φ ∨ EX X))
    fn eval_release(
        &self,
        ks: &KripkeStructure,
        quantifier: PathQuantifier,
        trigger: &SatMap,
        invariant: &SatMap,
    ) -> SatMap {
        let mut current: SatMap = ks.states.keys()
            .map(|id| (id.clone(), 1.0))
            .collect();

        for _ in 0..self.config.max_iterations {
            let next_map = self.eval_next(ks, quantifier, &current);
            let mut new_current = BTreeMap::new();
            let mut changed = false;

            for (id, _) in &ks.states {
                let inv_val = invariant.get(id).copied().unwrap_or(0.0);
                let trig_val = trigger.get(id).copied().unwrap_or(0.0);
                let next_val = next_map.get(id).copied().unwrap_or(0.0);
                let new_val = inv_val.min(trig_val.max(next_val));
                let old_val = current.get(id).copied().unwrap_or(0.0);

                if (new_val - old_val).abs() > self.config.epsilon {
                    changed = true;
                }
                new_current.insert(id.clone(), new_val);
            }

            current = new_current;
            if !changed { break; }
        }

        current
    }

    /// Bounded until: φ U≤n ψ
    fn eval_bounded_until(
        &self,
        ks: &KripkeStructure,
        quantifier: PathQuantifier,
        hold: &SatMap,
        goal: &SatMap,
        bound: u32,
    ) -> SatMap {
        // Base case: at bound 0, only goal states satisfy
        let mut current: SatMap = goal.clone();

        // Iterate exactly `bound` times
        for _ in 0..bound {
            let next_map = self.eval_next(ks, quantifier, &current);
            let mut new_current = BTreeMap::new();

            for (id, _) in &ks.states {
                let goal_val = goal.get(id).copied().unwrap_or(0.0);
                let hold_val = hold.get(id).copied().unwrap_or(0.0);
                let next_val = next_map.get(id).copied().unwrap_or(0.0);
                let new_val = goal_val.max(hold_val.min(next_val));
                new_current.insert(id.clone(), new_val);
            }

            current = new_current;
        }

        current
    }

    /// General fixed-point computation: μX.body or νX.body
    fn eval_fixedpoint(
        &self,
        ks: &KripkeStructure,
        is_least: bool,
        variable: &str,
        body: &Formula,
        env: &HashMap<String, SatMap>,
    ) -> SatMap {
        let init_val = if is_least { 0.0 } else { 1.0 };
        let mut current: SatMap = ks.states.keys()
            .map(|id| (id.clone(), init_val))
            .collect();

        for _ in 0..self.config.max_iterations {
            let mut new_env = env.clone();
            new_env.insert(variable.to_string(), current.clone());

            let new_current = self.eval_rec(ks, body, &new_env);

            let mut changed = false;
            for (id, &new_val) in &new_current {
                let old_val = current.get(id).copied().unwrap_or(init_val);
                if (new_val - old_val).abs() > self.config.epsilon {
                    changed = true;
                    break;
                }
            }

            current = new_current;
            if !changed { break; }
        }

        current
    }

    /// Expected value of inner formula at next step (weighted by transition probabilities).
    fn eval_expected_next(&self, ks: &KripkeStructure, inner: &SatMap) -> SatMap {
        let mut result = BTreeMap::new();
        for (id, _) in &ks.states {
            let succs = ks.successors(id);
            let mut expected = 0.0;
            for (s, p) in succs {
                expected += p * inner.get(s).copied().unwrap_or(0.0);
            }
            result.insert(id.clone(), expected);
        }
        result
    }

    /// Expected finally: expected number of steps to reach a state satisfying inner.
    /// Returns a map of "probability of eventually reaching" (clamped to [0,1]).
    fn eval_expected_finally(&self, ks: &KripkeStructure, inner: &SatMap) -> SatMap {
        // Compute the probability of eventually reaching a state where inner > 0.5
        // using iterative computation: prob(s) = inner(s) + (1-inner(s)) * sum_t p(s,t)*prob(t)
        let mut current: SatMap = ks.states.keys()
            .map(|id| (id.clone(), inner.get(id).copied().unwrap_or(0.0)))
            .collect();

        for _ in 0..self.config.max_iterations {
            let mut new_current = BTreeMap::new();
            let mut changed = false;

            for (id, _) in &ks.states {
                let base = inner.get(id).copied().unwrap_or(0.0);
                if base >= 1.0 - 1e-12 {
                    new_current.insert(id.clone(), 1.0);
                    continue;
                }

                let succs = ks.successors(id);
                let mut succ_prob = 0.0;
                for (s, p) in succs {
                    succ_prob += p * current.get(s).copied().unwrap_or(0.0);
                }
                let new_val = base.max(succ_prob).min(1.0);
                let old_val = current.get(id).copied().unwrap_or(0.0);

                if (new_val - old_val).abs() > self.config.epsilon {
                    changed = true;
                }
                new_current.insert(id.clone(), new_val);
            }

            current = new_current;
            if !changed { break; }
        }

        current
    }

    /// Expected globally: expected long-run average of inner formula.
    fn eval_expected_globally(&self, ks: &KripkeStructure, inner: &SatMap) -> SatMap {
        // Compute the probability that inner holds globally (at all future states).
        // prob_g(s) = inner(s) * sum_t p(s,t) * prob_g(t)
        let mut current: SatMap = ks.states.keys()
            .map(|id| (id.clone(), inner.get(id).copied().unwrap_or(0.0)))
            .collect();

        for _ in 0..self.config.max_iterations {
            let mut new_current = BTreeMap::new();
            let mut changed = false;

            for (id, _) in &ks.states {
                let base = inner.get(id).copied().unwrap_or(0.0);
                if base < 1e-12 {
                    new_current.insert(id.clone(), 0.0);
                    continue;
                }

                let succs = ks.successors(id);
                if succs.is_empty() {
                    new_current.insert(id.clone(), base);
                    continue;
                }

                let mut succ_val = 0.0;
                for (s, p) in succs {
                    succ_val += p * current.get(s).copied().unwrap_or(0.0);
                }
                let new_val = base.min(succ_val);
                let old_val = current.get(id).copied().unwrap_or(0.0);

                if (new_val - old_val).abs() > self.config.epsilon {
                    changed = true;
                }
                new_current.insert(id.clone(), new_val);
            }

            current = new_current;
            if !changed { break; }
        }

        current
    }
}

impl Default for SemanticEvaluator {
    fn default() -> Self { Self::new() }
}

// ───────────────────────────────────────────────────────────────────────────────
// FixedPointComputer — standalone fixed-point iteration engine
// ───────────────────────────────────────────────────────────────────────────────

/// Standalone fixed-point computation engine.
pub struct FixedPointComputer {
    pub max_iterations: usize,
    pub epsilon: f64,
}

impl FixedPointComputer {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            epsilon: 1e-10,
        }
    }

    /// Compute least fixed point: start from bottom (all 0s), iterate up.
    pub fn least_fixed_point<F>(&self, states: &[StateId], step: F) -> SatMap
    where
        F: Fn(&SatMap) -> SatMap,
    {
        let mut current: SatMap = states.iter()
            .map(|s| (s.clone(), 0.0))
            .collect();

        for _ in 0..self.max_iterations {
            let next = step(&current);
            if self.converged(&current, &next) {
                return next;
            }
            current = next;
        }
        current
    }

    /// Compute greatest fixed point: start from top (all 1s), iterate down.
    pub fn greatest_fixed_point<F>(&self, states: &[StateId], step: F) -> SatMap
    where
        F: Fn(&SatMap) -> SatMap,
    {
        let mut current: SatMap = states.iter()
            .map(|s| (s.clone(), 1.0))
            .collect();

        for _ in 0..self.max_iterations {
            let next = step(&current);
            if self.converged(&current, &next) {
                return next;
            }
            current = next;
        }
        current
    }

    /// Compute a fixed point with a given initial value.
    pub fn fixed_point_from<F>(&self, initial: SatMap, step: F) -> SatMap
    where
        F: Fn(&SatMap) -> SatMap,
    {
        let mut current = initial;
        for _ in 0..self.max_iterations {
            let next = step(&current);
            if self.converged(&current, &next) {
                return next;
            }
            current = next;
        }
        current
    }

    fn converged(&self, a: &SatMap, b: &SatMap) -> bool {
        for (key, &av) in a {
            let bv = b.get(key).copied().unwrap_or(0.0);
            if (av - bv).abs() > self.epsilon {
                return false;
            }
        }
        true
    }

    /// Compute the number of iterations needed for convergence.
    pub fn iterations_to_converge<F>(&self, states: &[StateId], is_least: bool, step: F) -> usize
    where
        F: Fn(&SatMap) -> SatMap,
    {
        let init_val = if is_least { 0.0 } else { 1.0 };
        let mut current: SatMap = states.iter()
            .map(|s| (s.clone(), init_val))
            .collect();

        for i in 0..self.max_iterations {
            let next = step(&current);
            if self.converged(&current, &next) {
                return i;
            }
            current = next;
        }
        self.max_iterations
    }
}

impl Default for FixedPointComputer {
    fn default() -> Self { Self::new() }
}

// ───────────────────────────────────────────────────────────────────────────────
// Semantic equivalence
// ───────────────────────────────────────────────────────────────────────────────

/// Check if two formulas are semantically equivalent on a given Kripke structure.
/// Two formulas are equivalent if their satisfaction maps are identical (within epsilon).
pub fn semantically_equivalent(
    ks: &KripkeStructure,
    f1: &Formula,
    f2: &Formula,
    epsilon: f64,
) -> bool {
    let eval = SemanticEvaluator::new();
    let m1 = eval.evaluate(ks, f1);
    let m2 = eval.evaluate(ks, f2);

    for id in ks.states.keys() {
        let v1 = m1.get(id).copied().unwrap_or(0.0);
        let v2 = m2.get(id).copied().unwrap_or(0.0);
        if (v1 - v2).abs() > epsilon {
            return false;
        }
    }
    true
}

/// Compute the maximum semantic distance between two formulas across all states.
pub fn semantic_distance(
    ks: &KripkeStructure,
    f1: &Formula,
    f2: &Formula,
) -> f64 {
    let eval = SemanticEvaluator::new();
    let m1 = eval.evaluate(ks, f1);
    let m2 = eval.evaluate(ks, f2);

    let mut max_dist = 0.0_f64;
    for id in ks.states.keys() {
        let v1 = m1.get(id).copied().unwrap_or(0.0);
        let v2 = m2.get(id).copied().unwrap_or(0.0);
        max_dist = max_dist.max((v1 - v2).abs());
    }
    max_dist
}

// ───────────────────────────────────────────────────────────────────────────────
// Model builder utility
// ───────────────────────────────────────────────────────────────────────────────

/// Builder for constructing Kripke structures for testing.
pub struct KripkeBuilder {
    ks: KripkeStructure,
}

impl KripkeBuilder {
    pub fn new() -> Self {
        Self { ks: KripkeStructure::new() }
    }

    pub fn state(mut self, id: impl Into<String>, labels: &[&str], initial: bool) -> Self {
        let mut state = KripkeState::new(id);
        for &l in labels {
            state = state.with_label(l);
        }
        self.ks.add_state(state, initial);
        self
    }

    pub fn quant_state(mut self, id: impl Into<String>, labels: &[(&str, f64)], initial: bool) -> Self {
        let mut state = KripkeState::new(id);
        for &(l, v) in labels {
            state = state.with_quant_label(l, v);
        }
        self.ks.add_state(state, initial);
        self
    }

    pub fn transition(mut self, from: impl Into<String>, to: impl Into<String>, prob: f64) -> Self {
        self.ks.add_transition(from, to, prob);
        self
    }

    pub fn build(self) -> KripkeStructure {
        self.ks
    }
}

impl Default for KripkeBuilder {
    fn default() -> Self { Self::new() }
}

// ───────────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple two-state Kripke structure:
    /// s0 (safe) --1.0--> s1 (safe) --1.0--> s1 (self-loop)
    fn two_state_safe() -> KripkeStructure {
        KripkeBuilder::new()
            .state("s0", &["safe"], true)
            .state("s1", &["safe"], false)
            .transition("s0", "s1", 1.0)
            .transition("s1", "s1", 1.0)
            .build()
    }

    /// Build a three-state Kripke structure:
    /// s0 (safe) --0.8--> s1 (safe) --1.0--> s2 (safe, done)
    /// s0 --0.2--> s2
    /// s2 --1.0--> s2 (self-loop)
    fn three_state_model() -> KripkeStructure {
        KripkeBuilder::new()
            .state("s0", &["safe"], true)
            .state("s1", &["safe"], false)
            .state("s2", &["safe", "done"], false)
            .transition("s0", "s1", 0.8)
            .transition("s0", "s2", 0.2)
            .transition("s1", "s2", 1.0)
            .transition("s2", "s2", 1.0)
            .build()
    }

    /// Refusal model: s0 refuses, s1 might break
    /// s0 (refusal) --0.95--> s0, --0.05--> s1
    /// s1 () --1.0--> s1
    fn refusal_model() -> KripkeStructure {
        KripkeBuilder::new()
            .state("s0", &["refusal"], true)
            .state("s1", &[], false)
            .transition("s0", "s0", 0.95)
            .transition("s0", "s1", 0.05)
            .transition("s1", "s1", 1.0)
            .build()
    }

    /// Unsafe model: s0 transitions to unsafe state
    fn unsafe_model() -> KripkeStructure {
        KripkeBuilder::new()
            .state("s0", &["safe"], true)
            .state("s1", &["unsafe"], false)
            .transition("s0", "s1", 1.0)
            .transition("s1", "s1", 1.0)
            .build()
    }

    // ── KripkeStructure ──

    #[test]
    fn test_kripke_well_formed() {
        let ks = two_state_safe();
        assert!(ks.is_well_formed());
    }

    #[test]
    fn test_kripke_normalize() {
        let mut ks = KripkeBuilder::new()
            .state("s0", &[], true)
            .state("s1", &[], false)
            .state("s2", &[], false)
            .transition("s0", "s1", 3.0)
            .transition("s0", "s2", 7.0)
            .build();
        ks.normalize();
        let succs = ks.successors("s0");
        let total: f64 = succs.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_kripke_predecessors() {
        let ks = three_state_model();
        let preds = ks.predecessors("s2");
        assert_eq!(preds.len(), 3); // s0, s1, s2
    }

    #[test]
    fn test_kripke_deadlock() {
        let ks = KripkeBuilder::new()
            .state("s0", &[], true)
            .build();
        assert!(ks.is_deadlock("s0"));
    }

    #[test]
    fn test_kripke_counts() {
        let ks = three_state_model();
        assert_eq!(ks.num_states(), 3);
        assert_eq!(ks.num_transitions(), 4);
    }

    // ── Atom evaluation ──

    #[test]
    fn test_eval_atom() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::atom("safe"));
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
        assert!((sat["s1"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_atom_absent() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::atom("toxic"));
        assert!((sat["s0"] - 0.0).abs() < 1e-9);
    }

    // ── Boolean connectives ──

    #[test]
    fn test_eval_and() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::and(Formula::atom("safe"), Formula::atom("done"));
        let sat = eval.evaluate(&ks, &f);
        assert!((sat["s0"] - 0.0).abs() < 1e-9); // safe but not done
        assert!((sat["s2"] - 1.0).abs() < 1e-9); // safe and done
    }

    #[test]
    fn test_eval_or() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::or(Formula::atom("safe"), Formula::atom("done"));
        let sat = eval.evaluate(&ks, &f);
        assert!((sat["s0"] - 1.0).abs() < 1e-9); // safe
        assert!((sat["s2"] - 1.0).abs() < 1e-9); // both
    }

    #[test]
    fn test_eval_not() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::not(Formula::atom("safe")));
        assert!((sat["s0"] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_implies() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::implies(Formula::atom("done"), Formula::atom("safe"));
        let sat = eval.evaluate(&ks, &f);
        // done → safe should be true at all states (done is false at s0,s1; true at s2 where safe)
        for (_, v) in &sat {
            assert!(*v >= 1.0 - 1e-9);
        }
    }

    // ── Temporal operators ──

    #[test]
    fn test_eval_ax() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let f = Formula::ax(Formula::atom("safe"));
        let sat = eval.evaluate(&ks, &f);
        // s0 --1.0--> s1(safe), so AX safe at s0 = 1.0
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_ex() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::ex(Formula::atom("done"));
        let sat = eval.evaluate(&ks, &f);
        // s0 has successor s2 which has "done"
        assert!(sat["s0"] > 0.5);
    }

    #[test]
    fn test_eval_ag_all_safe() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let f = Formula::ag(Formula::atom("safe"));
        let sat = eval.evaluate(&ks, &f);
        // All states are safe, so AG safe should be 1.0 everywhere
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
        assert!((sat["s1"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_ag_not_everywhere() {
        let ks = unsafe_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::ag(Formula::atom("safe"));
        let sat = eval.evaluate(&ks, &f);
        // s0 leads to unsafe s1, so AG safe at s0 should be < 1.0
        assert!(sat["s0"] < 1.0);
    }

    #[test]
    fn test_eval_ef() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::ef(Formula::atom("done"));
        let sat = eval.evaluate(&ks, &f);
        // All states can reach s2 (done), so EF done should be 1.0
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
        assert!((sat["s2"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_af() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::af(Formula::atom("done"));
        let sat = eval.evaluate(&ks, &f);
        // All paths from s0 eventually reach s2 (done)
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_eu() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::eu(Formula::atom("safe"), Formula::atom("done"));
        let sat = eval.evaluate(&ks, &f);
        // There exists a path from s0 where safe holds until done
        assert!(sat["s0"] > 0.5);
        // s2 has done, so it trivially satisfies
        assert!((sat["s2"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_au() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::au(Formula::atom("safe"), Formula::atom("done"));
        let sat = eval.evaluate(&ks, &f);
        // On all paths from s0, safe holds until done is reached
        assert!(sat["s0"] > 0.5);
    }

    // ── Quantitative semantics ──

    #[test]
    fn test_eval_quantitative_labels() {
        let ks = KripkeBuilder::new()
            .quant_state("s0", &[("safe", 0.8)], true)
            .quant_state("s1", &[("safe", 0.3)], false)
            .transition("s0", "s1", 1.0)
            .transition("s1", "s1", 1.0)
            .build();

        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::atom("safe"));
        assert!((sat["s0"] - 0.8).abs() < 1e-9);
        assert!((sat["s1"] - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_eval_prob_bound() {
        let ks = refusal_model();
        let eval = SemanticEvaluator::new();
        // P[>=0.9](AX refusal): probability of next state being refusal ≥ 0.9
        let f = Formula::prob_ge(0.9, Formula::ax(Formula::atom("refusal")));
        let sat = eval.evaluate(&ks, &f);
        // At s0: AX refusal = 0.95 (prob of staying in s0) ≥ 0.9, so P holds
        assert!(sat["s0"] > 0.5);
    }

    #[test]
    fn test_eval_prob_bound_fails() {
        let ks = refusal_model();
        let eval = SemanticEvaluator::new();
        // P[>=0.99](AX refusal): threshold too high
        let f = Formula::prob_ge(0.99, Formula::ax(Formula::atom("refusal")));
        let sat = eval.evaluate(&ks, &f);
        // At s0: AX refusal = 0.95 < 0.99, so P should be < 1.0
        assert!(sat["s0"] < 1.0);
    }

    // ── Bounded until ──

    #[test]
    fn test_eval_bounded_until() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::bounded_until(
            PathQuantifier::Exists,
            Formula::atom("safe"),
            Formula::atom("done"),
            3,
        );
        let sat = eval.evaluate(&ks, &f);
        assert!(sat["s0"] > 0.5);
    }

    #[test]
    fn test_eval_bounded_until_zero() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        let f = Formula::bounded_until(
            PathQuantifier::All,
            Formula::atom("safe"),
            Formula::atom("done"),
            0,
        );
        let sat = eval.evaluate(&ks, &f);
        // Bound 0 means only goal states satisfy
        assert!((sat["s0"] - 0.0).abs() < 1e-9);
        assert!((sat["s2"] - 1.0).abs() < 1e-9);
    }

    // ── Fixed point ──

    #[test]
    fn test_eval_mu_ef() {
        let ks = three_state_model();
        let eval = SemanticEvaluator::new();
        // μX.(done ∨ EX X) should be equivalent to EF done
        let mu_ef = Formula::mu("X", Formula::or(
            Formula::atom("done"),
            Formula::ex(Formula::var("X")),
        ));
        let ef = Formula::ef(Formula::atom("done"));
        let sat_mu = eval.evaluate(&ks, &mu_ef);
        let sat_ef = eval.evaluate(&ks, &ef);

        for id in ks.states.keys() {
            assert!(
                (sat_mu[id] - sat_ef[id]).abs() < 0.1,
                "Mismatch at {}: mu={}, ef={}", id, sat_mu[id], sat_ef[id]
            );
        }
    }

    #[test]
    fn test_eval_nu_ag() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        // νX.(safe ∧ AX X) should be equivalent to AG safe
        let nu_ag = Formula::nu("X", Formula::and(
            Formula::atom("safe"),
            Formula::ax(Formula::var("X")),
        ));
        let ag = Formula::ag(Formula::atom("safe"));
        let sat_nu = eval.evaluate(&ks, &nu_ag);
        let sat_ag = eval.evaluate(&ks, &ag);

        for id in ks.states.keys() {
            assert!(
                (sat_nu[id] - sat_ag[id]).abs() < 0.1,
                "Mismatch at {}: nu={}, ag={}", id, sat_nu[id], sat_ag[id]
            );
        }
    }

    // ── Graded modality ──

    #[test]
    fn test_eval_graded() {
        let ks = KripkeBuilder::new()
            .state("s0", &[], true)
            .state("s1", &["p"], false)
            .state("s2", &["p"], false)
            .state("s3", &[], false)
            .transition("s0", "s1", 0.4)
            .transition("s0", "s2", 0.3)
            .transition("s0", "s3", 0.3)
            .transition("s1", "s1", 1.0)
            .transition("s2", "s2", 1.0)
            .transition("s3", "s3", 1.0)
            .build();

        let eval = SemanticEvaluator::new();
        // ⟨2⟩p: at least 2 successors satisfy p
        let f = Formula::graded(2, Formula::atom("p"));
        let sat = eval.evaluate(&ks, &f);
        // s0 has 2 successors (s1, s2) satisfying p
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_graded_insufficient() {
        let ks = KripkeBuilder::new()
            .state("s0", &[], true)
            .state("s1", &["p"], false)
            .state("s2", &[], false)
            .transition("s0", "s1", 0.5)
            .transition("s0", "s2", 0.5)
            .transition("s1", "s1", 1.0)
            .transition("s2", "s2", 1.0)
            .build();

        let eval = SemanticEvaluator::new();
        let f = Formula::graded(2, Formula::atom("p"));
        let sat = eval.evaluate(&ks, &f);
        // Only 1 successor satisfies p, need 2
        assert!(sat["s0"] < 1.0);
    }

    // ── Expected value operators ──

    #[test]
    fn test_eval_exp_bound_next() {
        let ks = KripkeBuilder::new()
            .quant_state("s0", &[("q", 0.0)], true)
            .quant_state("s1", &[("q", 0.8)], false)
            .quant_state("s2", &[("q", 0.4)], false)
            .transition("s0", "s1", 0.5)
            .transition("s0", "s2", 0.5)
            .transition("s1", "s1", 1.0)
            .transition("s2", "s2", 1.0)
            .build();

        let eval = SemanticEvaluator::new();
        // E[>=0.5] X q: expected value of q at next step ≥ 0.5
        // Expected = 0.5*0.8 + 0.5*0.4 = 0.6 ≥ 0.5
        let f = Formula::exp_bound(ComparisonOp::Ge, 0.5, TemporalOp::X, Formula::atom("q"));
        let sat = eval.evaluate(&ks, &f);
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
    }

    // ── Semantic equivalence ──

    #[test]
    fn test_semantic_equivalence_trivial() {
        let ks = two_state_safe();
        let f = Formula::atom("safe");
        assert!(semantically_equivalent(&ks, &f, &f, 1e-9));
    }

    #[test]
    fn test_semantic_equivalence_double_neg() {
        let ks = two_state_safe();
        let f1 = Formula::atom("safe");
        let f2 = Formula::not(Formula::not(Formula::atom("safe")));
        assert!(semantically_equivalent(&ks, &f1, &f2, 1e-9));
    }

    #[test]
    fn test_semantic_distance() {
        let ks = two_state_safe();
        let f1 = Formula::atom("safe");
        let f2 = Formula::atom("toxic");
        let d = semantic_distance(&ks, &f1, &f2);
        assert!((d - 1.0).abs() < 1e-9); // max distance
    }

    #[test]
    fn test_semantic_distance_same() {
        let ks = two_state_safe();
        let f = Formula::atom("safe");
        let d = semantic_distance(&ks, &f, &f);
        assert!(d < 1e-9);
    }

    // ── Constants ──

    #[test]
    fn test_eval_true() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::top());
        for v in sat.values() { assert!((*v - 1.0).abs() < 1e-9); }
    }

    #[test]
    fn test_eval_false() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::bot());
        for v in sat.values() { assert!(v.abs() < 1e-9); }
    }

    // ── evaluate_at / evaluate_initial ──

    #[test]
    fn test_evaluate_at() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let val = eval.evaluate_at(&ks, &Formula::atom("safe"), "s0");
        assert!((val - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_initial() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let val = eval.evaluate_initial(&ks, &Formula::atom("safe"));
        assert!((val - 1.0).abs() < 1e-9);
    }

    // ── Fixed point computer ──

    #[test]
    fn test_lfp_constant() {
        let fpc = FixedPointComputer::new();
        let states: Vec<StateId> = vec!["s0".to_string(), "s1".to_string()];
        let result = fpc.least_fixed_point(&states, |_current| {
            vec![("s0".to_string(), 0.5), ("s1".to_string(), 0.7)].into_iter().collect()
        });
        assert!((result["s0"] - 0.5).abs() < 1e-9);
        assert!((result["s1"] - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_gfp_constant() {
        let fpc = FixedPointComputer::new();
        let states: Vec<StateId> = vec!["s0".to_string()];
        let result = fpc.greatest_fixed_point(&states, |_| {
            vec![("s0".to_string(), 0.3)].into_iter().collect()
        });
        assert!((result["s0"] - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_iterations_to_converge() {
        let fpc = FixedPointComputer::new();
        let states: Vec<StateId> = vec!["s0".to_string()];
        let iters = fpc.iterations_to_converge(&states, true, |_| {
            vec![("s0".to_string(), 0.5)].into_iter().collect()
        });
        assert!(iters <= 2);
    }

    // ── KripkeBuilder ──

    #[test]
    fn test_builder_basic() {
        let ks = KripkeBuilder::new()
            .state("a", &["p", "q"], true)
            .state("b", &["q"], false)
            .transition("a", "b", 1.0)
            .build();
        assert_eq!(ks.num_states(), 2);
        assert!(ks.states["a"].satisfies("p"));
        assert!(!ks.states["b"].satisfies("p"));
    }

    // ── Release operator ──

    #[test]
    fn test_eval_release() {
        let ks = KripkeBuilder::new()
            .state("s0", &["q"], true)
            .state("s1", &["p", "q"], false)
            .transition("s0", "s1", 1.0)
            .transition("s1", "s1", 1.0)
            .build();

        let eval = SemanticEvaluator::new();
        // A[p R q]: q must hold until p releases it
        let f = Formula::ar(Formula::atom("p"), Formula::atom("q"));
        let sat = eval.evaluate(&ks, &f);
        // q holds at s0, then p releases at s1 where q also holds
        assert!(sat["s0"] > 0.5);
    }

    // ── Complex formula ──

    #[test]
    fn test_eval_complex_refusal() {
        let ks = refusal_model();
        let eval = SemanticEvaluator::new();
        // AG(refusal → AX refusal): if refusing, next state also refuses
        // This is only approximately true (95% of the time)
        let f = Formula::ag(Formula::implies(
            Formula::atom("refusal"),
            Formula::ax(Formula::atom("refusal")),
        ));
        let sat = eval.evaluate(&ks, &f);
        // AG makes this fail because eventually refusal breaks
        assert!(sat["s0"] < 1.0);
    }

    // ── Deadlock handling ──

    #[test]
    fn test_eval_deadlock_ax() {
        let ks = KripkeBuilder::new()
            .state("s0", &[], true)
            .build();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::ax(Formula::atom("p")));
        // AX is vacuously true at deadlock
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_deadlock_ex() {
        let ks = KripkeBuilder::new()
            .state("s0", &[], true)
            .build();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::ex(Formula::atom("p")));
        // EX is false at deadlock (no successors)
        assert!((sat["s0"]).abs() < 1e-9);
    }

    // ── QVal ──

    #[test]
    fn test_eval_qval() {
        let ks = two_state_safe();
        let eval = SemanticEvaluator::new();
        let sat = eval.evaluate(&ks, &Formula::qval(0.42));
        for v in sat.values() { assert!((v - 0.42).abs() < 1e-9); }
    }

    // ── Boolean config ──

    #[test]
    fn test_boolean_mode() {
        let ks = KripkeBuilder::new()
            .quant_state("s0", &[("safe", 0.7)], true)
            .transition("s0", "s0", 1.0)
            .build();

        let eval = SemanticEvaluator::with_config(EvalConfig {
            quantitative: false,
            ..Default::default()
        });
        let sat = eval.evaluate(&ks, &Formula::atom("safe"));
        // In boolean mode, 0.7 (above 0.5 threshold in quant_label) → 1.0
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
    }

    // ── Iff ──

    #[test]
    fn test_eval_iff() {
        let ks = KripkeBuilder::new()
            .state("s0", &["a", "b"], true)
            .state("s1", &["a"], false)
            .transition("s0", "s1", 1.0)
            .transition("s1", "s1", 1.0)
            .build();

        let eval = SemanticEvaluator::new();
        let f = Formula::iff(Formula::atom("a"), Formula::atom("b"));
        let sat = eval.evaluate(&ks, &f);
        // s0: both true → 1.0; s1: a=1, b=0 → 0.0
        assert!((sat["s0"] - 1.0).abs() < 1e-9);
        assert!((sat["s1"]).abs() < 1e-9);
    }
}
