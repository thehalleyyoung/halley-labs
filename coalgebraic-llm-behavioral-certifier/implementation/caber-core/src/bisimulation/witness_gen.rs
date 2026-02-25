// Bisimulation witness generation module for CABER.
// Provides construction and validation of bisimulation witnesses,
// distinguishing formulas (Hennessy-Milner Logic), distinguishing traces,
// characteristic formulas, and witness minimization.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Local types – no imports from other CABER modules
// ---------------------------------------------------------------------------

/// A single transition in a labelled transition system.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LTSTransition {
    pub source: usize,
    pub action: String,
    pub target: usize,
}

/// Labelled transition system used for witness generation.
#[derive(Clone, Debug)]
pub struct WitnessLTS {
    pub num_states: usize,
    pub actions: Vec<String>,
    pub transitions: Vec<LTSTransition>,
    pub state_labels: Vec<Vec<String>>,
}

impl WitnessLTS {
    pub fn new(num_states: usize) -> Self {
        WitnessLTS {
            num_states,
            actions: Vec::new(),
            transitions: Vec::new(),
            state_labels: vec![Vec::new(); num_states],
        }
    }

    pub fn add_transition(&mut self, source: usize, action: &str, target: usize) {
        let act = action.to_string();
        if !self.actions.contains(&act) {
            self.actions.push(act.clone());
        }
        self.transitions.push(LTSTransition {
            source,
            action: act,
            target,
        });
    }

    pub fn add_label(&mut self, state: usize, label: &str) {
        if state < self.num_states {
            let l = label.to_string();
            if !self.state_labels[state].contains(&l) {
                self.state_labels[state].push(l);
            }
        }
    }

    /// All successor states reachable from `state` via `action`.
    pub fn successors(&self, state: usize, action: &str) -> Vec<usize> {
        self.transitions
            .iter()
            .filter(|t| t.source == state && t.action == action)
            .map(|t| t.target)
            .collect()
    }

    /// All predecessor states that can reach `state` via `action`.
    pub fn predecessors(&self, state: usize, action: &str) -> Vec<usize> {
        self.transitions
            .iter()
            .filter(|t| t.target == state && t.action == action)
            .map(|t| t.source)
            .collect()
    }

    /// Set of actions enabled at `state`.
    pub fn enabled_actions(&self, state: usize) -> Vec<String> {
        let mut acts: Vec<String> = self
            .transitions
            .iter()
            .filter(|t| t.source == state)
            .map(|t| t.action.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        acts.sort();
        acts
    }
}

// ---------------------------------------------------------------------------
// WitnessStep – one proof step inside a BisimulationWitness
// ---------------------------------------------------------------------------

/// A single step in a bisimulation proof: records which pair was checked,
/// the action considered, the matchings found, and a justification string.
#[derive(Clone, Debug)]
pub struct WitnessStep {
    pub pair: (usize, usize),
    pub action: String,
    pub matching: Vec<((usize, usize), (usize, usize))>,
    pub justification: String,
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub violations: Vec<String>,
}

impl ValidationResult {
    fn ok() -> Self {
        ValidationResult {
            valid: true,
            violations: Vec::new(),
        }
    }

    fn fail(violations: Vec<String>) -> Self {
        ValidationResult {
            valid: false,
            violations,
        }
    }
}

// ---------------------------------------------------------------------------
// BisimulationWitness
// ---------------------------------------------------------------------------

/// A witness that two states are bisimilar, consisting of a relation R and
/// proof steps showing the bisimulation conditions hold.
#[derive(Clone, Debug)]
pub struct BisimulationWitness {
    pub relation: Vec<(usize, usize)>,
    pub states_a: usize,
    pub states_b: usize,
    pub proof_steps: Vec<WitnessStep>,
}

impl BisimulationWitness {
    /// Try to construct a bisimulation relation containing `(s1, s2)`.
    ///
    /// Algorithm:
    ///   R ← {(s1, s2)}
    ///   worklist ← [(s1, s2)]
    ///   while worklist not empty:
    ///     pop (s, t)
    ///     for each action a enabled at s or t:
    ///       for each s' with s -a-> s':
    ///         find some t' with t -a-> t' such that we can recursively match
    ///         if no match found → return None
    ///         add (s', t') to R if new
    ///       for each t' with t -a-> t':
    ///         find some s' with s -a-> s' such that (s', t') ∈ R
    ///         if no match found → return None
    ///   return the witness
    pub fn construct(system: &WitnessLTS, s1: usize, s2: usize) -> Option<BisimulationWitness> {
        let mut relation_set: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();
        let mut proof_steps: Vec<WitnessStep> = Vec::new();

        relation_set.insert((s1, s2));
        worklist.push_back((s1, s2));

        // Collect all actions
        let all_actions: Vec<String> = system.actions.clone();

        while let Some((s, t)) = worklist.pop_front() {
            for action in &all_actions {
                let s_succs = system.successors(s, action);
                let t_succs = system.successors(t, action);

                let mut step_matchings: Vec<((usize, usize), (usize, usize))> = Vec::new();

                // Forward: for each s -a-> s', find matching t -a-> t'
                for &s_prime in &s_succs {
                    let matched = Self::find_match(
                        system,
                        s_prime,
                        &t_succs,
                        &relation_set,
                    );
                    match matched {
                        Some(t_prime) => {
                            step_matchings.push(((s, s_prime), (t, t_prime)));
                            if !relation_set.contains(&(s_prime, t_prime)) {
                                relation_set.insert((s_prime, t_prime));
                                worklist.push_back((s_prime, t_prime));
                            }
                        }
                        None => return None,
                    }
                }

                // Backward: for each t -a-> t', find matching s -a-> s'
                for &t_prime in &t_succs {
                    let matched = Self::find_match(
                        system,
                        t_prime,
                        &s_succs,
                        // We need to check that (s', t') is in R, so we look
                        // for s' such that (s', t_prime) ∈ relation_set
                        &relation_set,
                    );
                    match matched {
                        Some(s_prime) => {
                            // (s_prime, t_prime) should already be in R or we add it
                            if !relation_set.contains(&(s_prime, t_prime)) {
                                relation_set.insert((s_prime, t_prime));
                                worklist.push_back((s_prime, t_prime));
                            }
                        }
                        None => return None,
                    }
                }

                if !s_succs.is_empty() || !t_succs.is_empty() {
                    proof_steps.push(WitnessStep {
                        pair: (s, t),
                        action: action.clone(),
                        matching: step_matchings,
                        justification: format!(
                            "Pair ({},{}) checked for action '{}': {} forward, {} backward transitions matched",
                            s, t, action, s_succs.len(), t_succs.len()
                        ),
                    });
                }
            }
        }

        let relation: Vec<(usize, usize)> = relation_set.into_iter().collect();
        Some(BisimulationWitness {
            relation,
            states_a: system.num_states,
            states_b: system.num_states,
            proof_steps,
        })
    }

    /// Helper: given a state `target_state`, find some state in `candidates`
    /// such that the pair `(target_state, candidate)` is already in `relation`
    /// OR both states have compatible labels (same atomic propositions) and
    /// the same set of enabled actions (a heuristic for potential bisimilarity).
    fn find_match(
        system: &WitnessLTS,
        target_state: usize,
        candidates: &[usize],
        relation: &BTreeSet<(usize, usize)>,
    ) -> Option<usize> {
        // First try: already in relation
        for &c in candidates {
            if relation.contains(&(target_state, c)) {
                return Some(c);
            }
        }
        // Second try: same labels and compatible structure (optimistic)
        let target_labels: BTreeSet<&String> =
            system.state_labels[target_state].iter().collect();
        let target_actions: BTreeSet<String> =
            system.enabled_actions(target_state).into_iter().collect();

        for &c in candidates {
            let c_labels: BTreeSet<&String> = system.state_labels[c].iter().collect();
            let c_actions: BTreeSet<String> =
                system.enabled_actions(c).into_iter().collect();
            if target_labels == c_labels && target_actions == c_actions {
                return Some(c);
            }
        }
        // No match found
        if candidates.is_empty() {
            // Both sides have no successors – vacuously true, no pair needed
            // but we shouldn't reach here because the caller only iterates
            // over existing successors
            None
        } else {
            None
        }
    }

    /// Validate that `self.relation` is a bisimulation on `system`.
    ///
    /// For every (s, t) ∈ R and every action a:
    ///   - For every s -a-> s', there exists t -a-> t' with (s', t') ∈ R
    ///   - For every t -a-> t', there exists s -a-> s' with (s', t') ∈ R
    pub fn validate(&self, system: &WitnessLTS) -> ValidationResult {
        let relation_set: HashSet<(usize, usize)> =
            self.relation.iter().cloned().collect();
        let mut violations: Vec<String> = Vec::new();

        for &(s, t) in &self.relation {
            // Check labels match
            let s_labels: BTreeSet<&String> = if s < system.state_labels.len() {
                system.state_labels[s].iter().collect()
            } else {
                BTreeSet::new()
            };
            let t_labels: BTreeSet<&String> = if t < system.state_labels.len() {
                system.state_labels[t].iter().collect()
            } else {
                BTreeSet::new()
            };
            if s_labels != t_labels {
                violations.push(format!(
                    "States {} and {} have different labels: {:?} vs {:?}",
                    s, t, s_labels, t_labels
                ));
            }

            for action in &system.actions {
                let s_succs = system.successors(s, action);
                let t_succs = system.successors(t, action);

                // Forward condition
                for &s_prime in &s_succs {
                    let has_match = t_succs
                        .iter()
                        .any(|&t_prime| relation_set.contains(&(s_prime, t_prime)));
                    if !has_match {
                        violations.push(format!(
                            "Forward violation: ({},{}) action '{}': s'={} has no matching t'",
                            s, t, action, s_prime
                        ));
                    }
                }

                // Backward condition
                for &t_prime in &t_succs {
                    let has_match = s_succs
                        .iter()
                        .any(|&s_prime| relation_set.contains(&(s_prime, t_prime)));
                    if !has_match {
                        violations.push(format!(
                            "Backward violation: ({},{}) action '{}': t'={} has no matching s'",
                            s, t, action, t_prime
                        ));
                    }
                }
            }
        }

        if violations.is_empty() {
            ValidationResult::ok()
        } else {
            ValidationResult::fail(violations)
        }
    }

    pub fn relation(&self) -> &[(usize, usize)] {
        &self.relation
    }

    pub fn size(&self) -> usize {
        self.relation.len()
    }

    /// True if every pair in the relation is of the form (s, s).
    pub fn is_identity(&self) -> bool {
        self.relation.iter().all(|&(a, b)| a == b)
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("Bisimulation Witness\n");
        out.push_str(&format!("  States A: {}\n", self.states_a));
        out.push_str(&format!("  States B: {}\n", self.states_b));
        out.push_str(&format!("  Relation size: {}\n", self.relation.len()));
        out.push_str("  Pairs: {");
        let pairs: Vec<String> = self
            .relation
            .iter()
            .map(|(a, b)| format!("({},{})", a, b))
            .collect();
        out.push_str(&pairs.join(", "));
        out.push_str("}\n");
        out.push_str(&format!("  Proof steps: {}\n", self.proof_steps.len()));
        for (i, step) in self.proof_steps.iter().enumerate() {
            out.push_str(&format!(
                "    Step {}: pair ({},{}) action '{}' – {}\n",
                i, step.pair.0, step.pair.1, step.action, step.justification
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Hennessy-Milner Logic (HML) formulas
// ---------------------------------------------------------------------------

/// HML formula for distinguishing / characterizing states up to bisimilarity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HMLFormula {
    True,
    False,
    Atom(String),
    Not(Box<HMLFormula>),
    And(Box<HMLFormula>, Box<HMLFormula>),
    Or(Box<HMLFormula>, Box<HMLFormula>),
    /// Diamond modality `<a>φ`: there exists an a-transition to a φ-state.
    Diamond(String, Box<HMLFormula>),
    /// Box modality `[a]φ`: all a-transitions lead to a φ-state.
    Box(String, Box<HMLFormula>),
}

impl HMLFormula {
    /// Evaluate this formula at `state` in `system`.
    pub fn evaluate(&self, system: &WitnessLTS, state: usize) -> bool {
        match self {
            HMLFormula::True => true,
            HMLFormula::False => false,
            HMLFormula::Atom(prop) => {
                if state < system.state_labels.len() {
                    system.state_labels[state].contains(prop)
                } else {
                    false
                }
            }
            HMLFormula::Not(inner) => !inner.evaluate(system, state),
            HMLFormula::And(lhs, rhs) => {
                lhs.evaluate(system, state) && rhs.evaluate(system, state)
            }
            HMLFormula::Or(lhs, rhs) => {
                lhs.evaluate(system, state) || rhs.evaluate(system, state)
            }
            HMLFormula::Diamond(action, inner) => {
                let succs = system.successors(state, action);
                succs.iter().any(|&s| inner.evaluate(system, s))
            }
            HMLFormula::Box(action, inner) => {
                let succs = system.successors(state, action);
                succs.iter().all(|&s| inner.evaluate(system, s))
            }
        }
    }

    /// Modal depth of the formula.
    pub fn depth(&self) -> usize {
        match self {
            HMLFormula::True | HMLFormula::False | HMLFormula::Atom(_) => 0,
            HMLFormula::Not(inner) => inner.depth(),
            HMLFormula::And(lhs, rhs) | HMLFormula::Or(lhs, rhs) => {
                lhs.depth().max(rhs.depth())
            }
            HMLFormula::Diamond(_, inner) | HMLFormula::Box(_, inner) => {
                1 + inner.depth()
            }
        }
    }

    /// Number of sub-formula nodes (including self).
    pub fn size(&self) -> usize {
        match self {
            HMLFormula::True | HMLFormula::False | HMLFormula::Atom(_) => 1,
            HMLFormula::Not(inner) => 1 + inner.size(),
            HMLFormula::And(lhs, rhs) | HMLFormula::Or(lhs, rhs) => {
                1 + lhs.size() + rhs.size()
            }
            HMLFormula::Diamond(_, inner) | HMLFormula::Box(_, inner) => {
                1 + inner.size()
            }
        }
    }

    /// Logical negation (pushes negation inward where possible).
    pub fn negate(&self) -> HMLFormula {
        match self {
            HMLFormula::True => HMLFormula::False,
            HMLFormula::False => HMLFormula::True,
            HMLFormula::Atom(p) => HMLFormula::Not(Box::new(HMLFormula::Atom(p.clone()))),
            HMLFormula::Not(inner) => (**inner).clone(),
            HMLFormula::And(lhs, rhs) => {
                HMLFormula::Or(Box::new(lhs.negate()), Box::new(rhs.negate()))
            }
            HMLFormula::Or(lhs, rhs) => {
                HMLFormula::And(Box::new(lhs.negate()), Box::new(rhs.negate()))
            }
            HMLFormula::Diamond(a, inner) => {
                HMLFormula::Box(a.clone(), Box::new(inner.negate()))
            }
            HMLFormula::Box(a, inner) => {
                HMLFormula::Diamond(a.clone(), Box::new(inner.negate()))
            }
        }
    }

    /// Pretty-print the formula.
    pub fn render(&self) -> String {
        match self {
            HMLFormula::True => "tt".to_string(),
            HMLFormula::False => "ff".to_string(),
            HMLFormula::Atom(p) => p.clone(),
            HMLFormula::Not(inner) => format!("¬({})", inner.render()),
            HMLFormula::And(lhs, rhs) => {
                format!("({} ∧ {})", lhs.render(), rhs.render())
            }
            HMLFormula::Or(lhs, rhs) => {
                format!("({} ∨ {})", lhs.render(), rhs.render())
            }
            HMLFormula::Diamond(a, inner) => {
                format!("<{}>{}",  a, inner.render())
            }
            HMLFormula::Box(a, inner) => {
                format!("[{}]{}", a, inner.render())
            }
        }
    }

    // ---- helpers used internally ----

    /// Build a conjunction from an iterator of formulas.
    fn conjoin(mut formulas: Vec<HMLFormula>) -> HMLFormula {
        if formulas.is_empty() {
            return HMLFormula::True;
        }
        if formulas.len() == 1 {
            return formulas.pop().unwrap();
        }
        let mut acc = formulas.pop().unwrap();
        while let Some(f) = formulas.pop() {
            acc = HMLFormula::And(Box::new(f), Box::new(acc));
        }
        acc
    }

    /// Build a disjunction from an iterator of formulas.
    fn disjoin(mut formulas: Vec<HMLFormula>) -> HMLFormula {
        if formulas.is_empty() {
            return HMLFormula::False;
        }
        if formulas.len() == 1 {
            return formulas.pop().unwrap();
        }
        let mut acc = formulas.pop().unwrap();
        while let Some(f) = formulas.pop() {
            acc = HMLFormula::Or(Box::new(f), Box::new(acc));
        }
        acc
    }

    /// Simplify trivial cases (True/False absorption).
    fn simplify(&self) -> HMLFormula {
        match self {
            HMLFormula::And(lhs, rhs) => {
                let l = lhs.simplify();
                let r = rhs.simplify();
                match (&l, &r) {
                    (HMLFormula::True, _) => r,
                    (_, HMLFormula::True) => l,
                    (HMLFormula::False, _) | (_, HMLFormula::False) => HMLFormula::False,
                    _ => HMLFormula::And(Box::new(l), Box::new(r)),
                }
            }
            HMLFormula::Or(lhs, rhs) => {
                let l = lhs.simplify();
                let r = rhs.simplify();
                match (&l, &r) {
                    (HMLFormula::False, _) => r,
                    (_, HMLFormula::False) => l,
                    (HMLFormula::True, _) | (_, HMLFormula::True) => HMLFormula::True,
                    _ => HMLFormula::Or(Box::new(l), Box::new(r)),
                }
            }
            HMLFormula::Not(inner) => {
                let s = inner.simplify();
                match s {
                    HMLFormula::True => HMLFormula::False,
                    HMLFormula::False => HMLFormula::True,
                    HMLFormula::Not(inner2) => *inner2,
                    other => HMLFormula::Not(Box::new(other)),
                }
            }
            HMLFormula::Diamond(a, inner) => {
                HMLFormula::Diamond(a.clone(), Box::new(inner.simplify()))
            }
            HMLFormula::Box(a, inner) => {
                HMLFormula::Box(a.clone(), Box::new(inner.simplify()))
            }
            other => other.clone(),
        }
    }
}

impl fmt::Display for HMLFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ---------------------------------------------------------------------------
// DistinguishingFormula
// ---------------------------------------------------------------------------

/// A formula that is satisfied by one state and violated by another,
/// witnessing their non-bisimilarity.
#[derive(Clone, Debug)]
pub struct DistinguishingFormula {
    pub formula: HMLFormula,
    pub depth: usize,
    pub satisfied_by: usize,
    pub violated_by: usize,
}

impl DistinguishingFormula {
    /// Compute a minimal-depth HML formula distinguishing `s` from `t`.
    ///
    /// Uses iterative deepening: at depth 0 try atomic propositions,
    /// at depth d try diamond and box modalities with depth-(d-1) sub-formulas.
    ///
    /// Returns `None` if the states are bisimilar (no distinguishing formula
    /// exists by the Hennessy-Milner theorem for image-finite systems).
    pub fn compute(
        system: &WitnessLTS,
        s: usize,
        t: usize,
    ) -> Option<DistinguishingFormula> {
        // We'll iteratively increase the depth bound.
        let max_depth_limit = system.num_states * 2 + 5;
        for max_d in 0..=max_depth_limit {
            if let Some(f) = Self::distinguish_at_depth(system, s, t, max_d, &mut HashMap::new()) {
                return Some(DistinguishingFormula {
                    depth: f.depth(),
                    formula: f,
                    satisfied_by: s,
                    violated_by: t,
                });
            }
        }
        None
    }

    /// Try to build a distinguishing formula of modal depth ≤ `max_depth`.
    /// `memo` caches results for (s, t, max_depth) triples.
    fn distinguish_at_depth(
        system: &WitnessLTS,
        s: usize,
        t: usize,
        max_depth: usize,
        memo: &mut HashMap<(usize, usize, usize), Option<HMLFormula>>,
    ) -> Option<HMLFormula> {
        if s == t {
            return None;
        }

        let key = (s, t, max_depth);
        if let Some(cached) = memo.get(&key) {
            return cached.clone();
        }

        // Prevent infinite recursion by inserting None placeholder
        memo.insert(key, None);

        // Depth 0: try atomic propositions
        let s_labels: BTreeSet<&String> = if s < system.state_labels.len() {
            system.state_labels[s].iter().collect()
        } else {
            BTreeSet::new()
        };
        let t_labels: BTreeSet<&String> = if t < system.state_labels.len() {
            system.state_labels[t].iter().collect()
        } else {
            BTreeSet::new()
        };

        // Atom present in s but not t
        for label in &s_labels {
            if !t_labels.contains(*label) {
                let f = HMLFormula::Atom((*label).clone());
                memo.insert(key, Some(f.clone()));
                return Some(f);
            }
        }
        // Atom present in t but not s → negate
        for label in &t_labels {
            if !s_labels.contains(*label) {
                let f = HMLFormula::Not(Box::new(HMLFormula::Atom((*label).clone())));
                memo.insert(key, Some(f.clone()));
                return Some(f);
            }
        }

        if max_depth == 0 {
            memo.insert(key, None);
            return None;
        }

        // Collect all actions
        let mut all_actions: BTreeSet<String> = BTreeSet::new();
        for a in system.enabled_actions(s) {
            all_actions.insert(a);
        }
        for a in system.enabled_actions(t) {
            all_actions.insert(a);
        }

        // Try diamond modality: <a>φ
        // s satisfies <a>φ but t does not ⟺
        // ∃ s' with s-a->s' such that ∀ t' with t-a->t', φ distinguishes s' from t'
        for action in &all_actions {
            let s_succs = system.successors(s, action);
            let t_succs = system.successors(t, action);

            // If s has an a-transition but t has none, <a>tt distinguishes
            if !s_succs.is_empty() && t_succs.is_empty() {
                let f = HMLFormula::Diamond(action.clone(), Box::new(HMLFormula::True));
                memo.insert(key, Some(f.clone()));
                return Some(f);
            }

            // If t has an a-transition but s has none, [a]ff distinguishes
            // (s satisfies [a]ff vacuously, t does not)
            if s_succs.is_empty() && !t_succs.is_empty() {
                let f = HMLFormula::Box(action.clone(), Box::new(HMLFormula::False));
                memo.insert(key, Some(f.clone()));
                return Some(f);
            }

            // Try: find s' such that for all t', can distinguish s' from t'
            for &s_prime in &s_succs {
                let mut sub_formulas: Vec<HMLFormula> = Vec::new();
                let mut all_distinguished = true;

                for &t_prime in &t_succs {
                    if let Some(sub) =
                        Self::distinguish_at_depth(system, s_prime, t_prime, max_depth - 1, memo)
                    {
                        sub_formulas.push(sub);
                    } else {
                        all_distinguished = false;
                        break;
                    }
                }

                if all_distinguished && !t_succs.is_empty() {
                    let inner = HMLFormula::conjoin(sub_formulas);
                    let f = HMLFormula::Diamond(action.clone(), Box::new(inner));
                    memo.insert(key, Some(f.clone()));
                    return Some(f);
                }
            }

            // Try box modality: [a]φ
            // s satisfies [a]φ but t does not ⟺
            // ∀ s' with s-a->s', s' satisfies φ
            // ∃ t' with t-a->t', t' does not satisfy φ
            // That is, find t' such that for all s', can distinguish t' from s'
            // Then use [a](negate that formula)
            for &t_prime in &t_succs {
                let mut sub_formulas: Vec<HMLFormula> = Vec::new();
                let mut all_distinguished = true;

                for &s_prime in &s_succs {
                    // We need φ such that s' satisfies φ but t' does not
                    if let Some(sub) =
                        Self::distinguish_at_depth(system, s_prime, t_prime, max_depth - 1, memo)
                    {
                        sub_formulas.push(sub);
                    } else {
                        all_distinguished = false;
                        break;
                    }
                }

                if all_distinguished && !s_succs.is_empty() {
                    // All s-successors satisfy each sub-formula,
                    // t_prime violates at least one (any single one suffices for the box)
                    // We want [a]ψ where all s' satisfy ψ and t' doesn't.
                    // Take the disjunction: since each sub_formula is satisfied by
                    // the corresponding s', and they all distinguish from t_prime,
                    // we can use any single one.
                    let inner = sub_formulas[0].clone();
                    let f = HMLFormula::Box(action.clone(), Box::new(inner));
                    // Verify quickly
                    if f.evaluate(system, s) && !f.evaluate(system, t) {
                        memo.insert(key, Some(f.clone()));
                        return Some(f);
                    }
                }
            }
        }

        memo.insert(key, None);
        None
    }

    /// Validate that `satisfied_by` satisfies the formula and `violated_by` does not.
    pub fn validate(&self, system: &WitnessLTS) -> bool {
        let sat = self.formula.evaluate(system, self.satisfied_by);
        let viol = self.formula.evaluate(system, self.violated_by);
        sat && !viol
    }

    pub fn render(&self) -> String {
        format!(
            "Distinguishing formula: {}\n  Depth: {}\n  Satisfied by state {}\n  Violated by state {}",
            self.formula.render(),
            self.depth,
            self.satisfied_by,
            self.violated_by,
        )
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn size(&self) -> usize {
        self.formula.size()
    }
}

// ---------------------------------------------------------------------------
// DistinguishingTrace
// ---------------------------------------------------------------------------

/// A trace (sequence of actions) that produces observably different behaviour
/// when executed from two given states.
#[derive(Clone, Debug)]
pub struct DistinguishingTrace {
    pub actions: Vec<String>,
    pub observations_s: Vec<Vec<String>>,
    pub observations_t: Vec<Vec<String>>,
    pub divergence_point: usize,
}

impl DistinguishingTrace {
    /// Find a shortest action sequence witnessing different behaviour from `s` vs `t`.
    ///
    /// BFS over pairs of states `(s', t')` reachable by the same action prefix.
    /// A divergence is detected when:
    ///   - The labels of s' and t' differ, or
    ///   - An action is enabled at s' but not at t' (or vice versa).
    pub fn generate(
        system: &WitnessLTS,
        s: usize,
        t: usize,
        max_length: usize,
    ) -> Option<DistinguishingTrace> {
        if s == t {
            return None;
        }

        // BFS state: (current_s, current_t, action_trace, obs_s, obs_t)
        struct BfsEntry {
            state_s: usize,
            state_t: usize,
            actions: Vec<String>,
            obs_s: Vec<Vec<String>>,
            obs_t: Vec<Vec<String>>,
        }

        let labels_of = |st: usize| -> Vec<String> {
            if st < system.state_labels.len() {
                let mut l = system.state_labels[st].clone();
                l.sort();
                l
            } else {
                Vec::new()
            }
        };

        let init_obs_s = labels_of(s);
        let init_obs_t = labels_of(t);

        // Check initial divergence
        if init_obs_s != init_obs_t {
            return Some(DistinguishingTrace {
                actions: Vec::new(),
                observations_s: vec![init_obs_s],
                observations_t: vec![init_obs_t],
                divergence_point: 0,
            });
        }

        let mut queue: VecDeque<BfsEntry> = VecDeque::new();
        let mut visited: HashSet<(usize, usize)> = HashSet::new();

        visited.insert((s, t));
        queue.push_back(BfsEntry {
            state_s: s,
            state_t: t,
            actions: Vec::new(),
            obs_s: vec![init_obs_s],
            obs_t: vec![init_obs_t],
        });

        while let Some(entry) = queue.pop_front() {
            if entry.actions.len() >= max_length {
                continue;
            }

            let cur_s = entry.state_s;
            let cur_t = entry.state_t;

            // Collect enabled actions at both states
            let s_actions: BTreeSet<String> =
                system.enabled_actions(cur_s).into_iter().collect();
            let t_actions: BTreeSet<String> =
                system.enabled_actions(cur_t).into_iter().collect();

            // Check if enabled action sets differ
            if s_actions != t_actions {
                // Find the differing action
                let diff_act = s_actions
                    .symmetric_difference(&t_actions)
                    .next()
                    .unwrap()
                    .clone();
                let mut actions = entry.actions.clone();
                actions.push(diff_act);
                return Some(DistinguishingTrace {
                    divergence_point: entry.actions.len(),
                    actions,
                    observations_s: entry.obs_s,
                    observations_t: entry.obs_t,
                });
            }

            // Explore each action
            for action in &s_actions {
                let s_succs = system.successors(cur_s, action);
                let t_succs = system.successors(cur_t, action);

                // Try all combinations of successors
                for &s_next in &s_succs {
                    for &t_next in &t_succs {
                        if visited.contains(&(s_next, t_next)) {
                            continue;
                        }

                        let obs_s_next = labels_of(s_next);
                        let obs_t_next = labels_of(t_next);

                        let mut new_actions = entry.actions.clone();
                        new_actions.push(action.clone());

                        let mut new_obs_s = entry.obs_s.clone();
                        new_obs_s.push(obs_s_next.clone());
                        let mut new_obs_t = entry.obs_t.clone();
                        new_obs_t.push(obs_t_next.clone());

                        // Check if observations differ
                        if obs_s_next != obs_t_next {
                            return Some(DistinguishingTrace {
                                divergence_point: new_actions.len(),
                                actions: new_actions,
                                observations_s: new_obs_s,
                                observations_t: new_obs_t,
                            });
                        }

                        // Check if enabled actions differ at the new states
                        let s_next_acts: BTreeSet<String> =
                            system.enabled_actions(s_next).into_iter().collect();
                        let t_next_acts: BTreeSet<String> =
                            system.enabled_actions(t_next).into_iter().collect();
                        if s_next_acts != t_next_acts {
                            let diff_act = s_next_acts
                                .symmetric_difference(&t_next_acts)
                                .next()
                                .unwrap()
                                .clone();
                            new_actions.push(diff_act);
                            return Some(DistinguishingTrace {
                                divergence_point: new_actions.len() - 1,
                                actions: new_actions,
                                observations_s: new_obs_s,
                                observations_t: new_obs_t,
                            });
                        }

                        visited.insert((s_next, t_next));
                        queue.push_back(BfsEntry {
                            state_s: s_next,
                            state_t: t_next,
                            actions: new_actions,
                            obs_s: new_obs_s,
                            obs_t: new_obs_t,
                        });
                    }
                }
            }
        }

        None
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("Distinguishing Trace\n");
        out.push_str(&format!("  Length: {}\n", self.actions.len()));
        out.push_str(&format!("  Divergence at step: {}\n", self.divergence_point));
        out.push_str("  Actions: ");
        if self.actions.is_empty() {
            out.push_str("(empty)");
        } else {
            out.push_str(&self.actions.join(" → "));
        }
        out.push('\n');
        for (i, (os, ot)) in self
            .observations_s
            .iter()
            .zip(self.observations_t.iter())
            .enumerate()
        {
            out.push_str(&format!(
                "  Step {}: s={:?}  t={:?}{}\n",
                i,
                os,
                ot,
                if i == self.divergence_point {
                    " ← DIVERGE"
                } else {
                    ""
                }
            ));
        }
        out
    }

    pub fn length(&self) -> usize {
        self.actions.len()
    }
}

// ---------------------------------------------------------------------------
// WitnessMinimizer
// ---------------------------------------------------------------------------

/// Utilities for minimizing bisimulation witnesses and distinguishing formulas.
pub struct WitnessMinimizer;

impl WitnessMinimizer {
    /// Remove redundant pairs from a bisimulation relation while keeping it valid.
    ///
    /// Greedy approach: try removing each pair in turn; if the remaining relation
    /// is still a valid bisimulation, keep it removed.
    pub fn minimize_relation(
        witness: &BisimulationWitness,
        system: &WitnessLTS,
    ) -> BisimulationWitness {
        let mut current: Vec<(usize, usize)> = witness.relation.clone();

        // Try removing pairs from the end toward the front so that index
        // bookkeeping stays simple.
        let mut i = current.len();
        while i > 0 {
            i -= 1;
            let removed = current.remove(i);
            let candidate = BisimulationWitness {
                relation: current.clone(),
                states_a: witness.states_a,
                states_b: witness.states_b,
                proof_steps: Vec::new(),
            };
            let result = candidate.validate(system);
            if !result.valid {
                // Put it back
                current.insert(i, removed);
            }
        }

        BisimulationWitness {
            relation: current,
            states_a: witness.states_a,
            states_b: witness.states_b,
            proof_steps: witness.proof_steps.clone(),
        }
    }

    /// Simplify a distinguishing formula while preserving distinguishing power.
    ///
    /// Strategy:
    ///   1. Apply algebraic simplifications (True/False absorption).
    ///   2. Try removing conjuncts/disjuncts; keep removals that preserve the
    ///      distinguishing property.
    pub fn minimize_formula(
        formula: &DistinguishingFormula,
        system: &WitnessLTS,
    ) -> DistinguishingFormula {
        let simplified = formula.formula.simplify();

        // Try further structural minimization
        let minimized = Self::minimize_hml(
            &simplified,
            system,
            formula.satisfied_by,
            formula.violated_by,
        );

        DistinguishingFormula {
            formula: minimized.clone(),
            depth: minimized.depth(),
            satisfied_by: formula.satisfied_by,
            violated_by: formula.violated_by,
        }
    }

    /// Recursively attempt to simplify an HML formula while preserving
    /// that `sat` satisfies it and `viol` does not.
    fn minimize_hml(
        formula: &HMLFormula,
        system: &WitnessLTS,
        sat: usize,
        viol: usize,
    ) -> HMLFormula {
        match formula {
            HMLFormula::And(lhs, rhs) => {
                // Try dropping one conjunct
                if lhs.evaluate(system, sat) && !lhs.evaluate(system, viol) {
                    return Self::minimize_hml(lhs, system, sat, viol);
                }
                if rhs.evaluate(system, sat) && !rhs.evaluate(system, viol) {
                    return Self::minimize_hml(rhs, system, sat, viol);
                }
                // Can't drop either – minimize recursively
                let l = Self::minimize_hml(lhs, system, sat, viol);
                let r = Self::minimize_hml(rhs, system, sat, viol);
                HMLFormula::And(Box::new(l), Box::new(r))
            }
            HMLFormula::Or(lhs, rhs) => {
                if lhs.evaluate(system, sat) && !lhs.evaluate(system, viol) {
                    return Self::minimize_hml(lhs, system, sat, viol);
                }
                if rhs.evaluate(system, sat) && !rhs.evaluate(system, viol) {
                    return Self::minimize_hml(rhs, system, sat, viol);
                }
                let l = Self::minimize_hml(lhs, system, sat, viol);
                let r = Self::minimize_hml(rhs, system, sat, viol);
                HMLFormula::Or(Box::new(l), Box::new(r))
            }
            HMLFormula::Diamond(a, inner) => {
                let inner_min = inner.simplify();
                HMLFormula::Diamond(a.clone(), Box::new(inner_min))
            }
            HMLFormula::Box(a, inner) => {
                let inner_min = inner.simplify();
                HMLFormula::Box(a.clone(), Box::new(inner_min))
            }
            other => other.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// CharacteristicFormula
// ---------------------------------------------------------------------------

/// Builds a characteristic formula for a state up to bisimilarity.
///
/// The characteristic formula is the "most specific" HML formula satisfied
/// by a state: another state satisfies the formula iff it is bisimilar.
pub struct CharacteristicFormula;

impl CharacteristicFormula {
    /// Compute the characteristic formula for `state` up to modal depth `max_depth`.
    ///
    /// Depth 0: conjunction of all atomic propositions (and negations of absent ones).
    /// Depth d: conjunction of:
    ///   - For each action `a` and each `a`-successor `s'`:
    ///       `<a>(char(s', d-1))`
    ///   - For each action `a`:
    ///       `[a](disjunction of char(s', d-1) for each `a`-successor `s'`)`
    ///       (if no `a`-successors, this becomes `[a]ff`)
    pub fn compute(system: &WitnessLTS, state: usize, max_depth: usize) -> HMLFormula {
        Self::compute_with_memo(system, state, max_depth, &mut HashMap::new())
    }

    fn compute_with_memo(
        system: &WitnessLTS,
        state: usize,
        max_depth: usize,
        memo: &mut HashMap<(usize, usize), HMLFormula>,
    ) -> HMLFormula {
        let key = (state, max_depth);
        if let Some(cached) = memo.get(&key) {
            return cached.clone();
        }

        // Placeholder to handle recursion
        memo.insert(key, HMLFormula::True);

        let mut conjuncts: Vec<HMLFormula> = Vec::new();

        // Atomic propositions at this state
        if state < system.state_labels.len() {
            for label in &system.state_labels[state] {
                conjuncts.push(HMLFormula::Atom(label.clone()));
            }
        }

        if max_depth == 0 {
            let result = HMLFormula::conjoin(conjuncts);
            memo.insert(key, result.clone());
            return result;
        }

        // For each action, build diamond and box parts
        let all_actions = &system.actions;

        for action in all_actions {
            let succs = system.successors(state, action);

            // Diamond part: <a>(char(s')) for each successor s'
            for &s_prime in &succs {
                let sub =
                    Self::compute_with_memo(system, s_prime, max_depth - 1, memo);
                conjuncts.push(HMLFormula::Diamond(action.clone(), Box::new(sub)));
            }

            // Box part: [a](disjunction of char(s') for each successor s')
            if succs.is_empty() {
                // [a]ff – no a-successors, so the box is vacuously about false
                conjuncts.push(HMLFormula::Box(
                    action.clone(),
                    Box::new(HMLFormula::False),
                ));
            } else {
                let disj_parts: Vec<HMLFormula> = succs
                    .iter()
                    .map(|&sp| {
                        Self::compute_with_memo(system, sp, max_depth - 1, memo)
                    })
                    .collect();
                let disj = HMLFormula::disjoin(disj_parts);
                conjuncts.push(HMLFormula::Box(action.clone(), Box::new(disj)));
            }
        }

        let result = HMLFormula::conjoin(conjuncts);
        memo.insert(key, result.clone());
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helper builders ----

    /// Two-state LTS where states 0 and 1 are bisimilar.
    ///   0 --a--> 0
    ///   1 --a--> 1
    fn bisimilar_pair() -> WitnessLTS {
        let mut lts = WitnessLTS::new(2);
        lts.add_transition(0, "a", 0);
        lts.add_transition(1, "a", 1);
        lts
    }

    /// Two-state LTS where 0 and 1 are NOT bisimilar.
    ///   0 --a--> 0,  0 --b--> 0
    ///   1 --a--> 1
    fn non_bisimilar_pair() -> WitnessLTS {
        let mut lts = WitnessLTS::new(2);
        lts.add_transition(0, "a", 0);
        lts.add_transition(0, "b", 0);
        lts.add_transition(1, "a", 1);
        lts
    }

    /// Classic CCS coffee/tea example (non-bisimilar).
    ///  0 --coin--> 1, 0 --coin--> 2
    ///  1 --coffee--> 3
    ///  2 --tea--> 3
    ///
    ///  4 --coin--> 5
    ///  5 --coffee--> 3, 5 --tea--> 3
    fn coffee_tea_lts() -> WitnessLTS {
        let mut lts = WitnessLTS::new(6);
        lts.add_transition(0, "coin", 1);
        lts.add_transition(0, "coin", 2);
        lts.add_transition(1, "coffee", 3);
        lts.add_transition(2, "tea", 3);
        lts.add_transition(4, "coin", 5);
        lts.add_transition(5, "coffee", 3);
        lts.add_transition(5, "tea", 3);
        lts
    }

    /// States with different labels.
    fn labeled_lts() -> WitnessLTS {
        let mut lts = WitnessLTS::new(3);
        lts.add_label(0, "red");
        lts.add_label(1, "blue");
        lts.add_label(2, "red");
        lts.add_transition(0, "a", 1);
        lts.add_transition(2, "a", 1);
        lts
    }

    /// Simple diamond-shaped LTS.
    ///  0 --a--> 1, 0 --a--> 2
    ///  1 --b--> 3
    ///  2 --b--> 3
    ///  4 --a--> 5
    ///  5 --b--> 3
    fn diamond_lts() -> WitnessLTS {
        let mut lts = WitnessLTS::new(6);
        lts.add_transition(0, "a", 1);
        lts.add_transition(0, "a", 2);
        lts.add_transition(1, "b", 3);
        lts.add_transition(2, "b", 3);
        lts.add_transition(4, "a", 5);
        lts.add_transition(5, "b", 3);
        lts
    }

    // ---- tests ----

    #[test]
    fn test_witness_construction_bisimilar() {
        let lts = bisimilar_pair();
        let w = BisimulationWitness::construct(&lts, 0, 1);
        assert!(w.is_some(), "Should find bisimulation witness");
        let w = w.unwrap();
        assert!(w.relation.contains(&(0, 1)));
        let v = w.validate(&lts);
        assert!(v.valid, "Witness should be valid: {:?}", v.violations);
    }

    #[test]
    fn test_witness_failure_non_bisimilar() {
        let lts = non_bisimilar_pair();
        let w = BisimulationWitness::construct(&lts, 0, 1);
        assert!(w.is_none(), "States should not be bisimilar");
    }

    #[test]
    fn test_witness_identity() {
        let lts = bisimilar_pair();
        let w = BisimulationWitness::construct(&lts, 0, 0);
        assert!(w.is_some());
        let w = w.unwrap();
        assert!(w.is_identity());
    }

    #[test]
    fn test_witness_validation_valid() {
        let lts = diamond_lts();
        // 0 and 4: 0 --a--> {1,2}, 4 --a--> {5}
        // 1 --b--> 3, 2 --b--> 3, 5 --b--> 3
        // So 1 ~ 5 and 2 ~ 5, which means 0 ~ 4
        let w = BisimulationWitness::construct(&lts, 0, 4);
        assert!(w.is_some());
        let w = w.unwrap();
        let v = w.validate(&lts);
        assert!(v.valid, "Diamond LTS witness should be valid: {:?}", v.violations);
    }

    #[test]
    fn test_witness_validation_invalid() {
        // Manually create a bad witness
        let lts = non_bisimilar_pair();
        let bad_witness = BisimulationWitness {
            relation: vec![(0, 1)],
            states_a: 2,
            states_b: 2,
            proof_steps: Vec::new(),
        };
        let v = bad_witness.validate(&lts);
        assert!(!v.valid, "Bad witness should be invalid");
        assert!(!v.violations.is_empty());
    }

    #[test]
    fn test_witness_render() {
        let lts = bisimilar_pair();
        let w = BisimulationWitness::construct(&lts, 0, 1).unwrap();
        let rendered = w.render();
        assert!(rendered.contains("Bisimulation Witness"));
        assert!(rendered.contains("(0,1)"));
    }

    #[test]
    fn test_distinguishing_formula_generation() {
        let lts = non_bisimilar_pair();
        let df = DistinguishingFormula::compute(&lts, 0, 1);
        assert!(df.is_some(), "Should find distinguishing formula");
        let df = df.unwrap();
        assert!(df.validate(&lts), "Formula should distinguish the states");
    }

    #[test]
    fn test_distinguishing_formula_none_for_bisimilar() {
        let lts = bisimilar_pair();
        let df = DistinguishingFormula::compute(&lts, 0, 1);
        assert!(df.is_none(), "Bisimilar states should have no distinguishing formula");
    }

    #[test]
    fn test_hml_formula_evaluate_atom() {
        let lts = labeled_lts();
        let f = HMLFormula::Atom("red".to_string());
        assert!(f.evaluate(&lts, 0));
        assert!(!f.evaluate(&lts, 1));
        assert!(f.evaluate(&lts, 2));
    }

    #[test]
    fn test_hml_formula_evaluate_diamond() {
        let lts = bisimilar_pair();
        let f = HMLFormula::Diamond("a".to_string(), Box::new(HMLFormula::True));
        assert!(f.evaluate(&lts, 0));
        assert!(f.evaluate(&lts, 1));

        let f2 = HMLFormula::Diamond("b".to_string(), Box::new(HMLFormula::True));
        assert!(!f2.evaluate(&lts, 0));
    }

    #[test]
    fn test_hml_formula_evaluate_box() {
        let lts = non_bisimilar_pair();
        // [b]ff: all b-successors satisfy ff.
        // State 0 has b-successor 0, so 0 does not satisfy [b]ff.
        // State 1 has no b-successors, so vacuously true.
        let f = HMLFormula::Box("b".to_string(), Box::new(HMLFormula::False));
        assert!(!f.evaluate(&lts, 0));
        assert!(f.evaluate(&lts, 1));
    }

    #[test]
    fn test_hml_depth_and_size() {
        let f = HMLFormula::Diamond(
            "a".to_string(),
            Box::new(HMLFormula::And(
                Box::new(HMLFormula::Atom("p".to_string())),
                Box::new(HMLFormula::Box(
                    "b".to_string(),
                    Box::new(HMLFormula::True),
                )),
            )),
        );
        assert_eq!(f.depth(), 2);
        // Diamond(And(Atom, Box(True))) → 1 + 1 + 1 + 1 + 1 = 5
        assert_eq!(f.size(), 5);
    }

    #[test]
    fn test_hml_negate() {
        let f = HMLFormula::Diamond("a".to_string(), Box::new(HMLFormula::True));
        let neg = f.negate();
        // ¬<a>tt  =  [a]ff
        assert_eq!(neg, HMLFormula::Box("a".to_string(), Box::new(HMLFormula::False)));
    }

    #[test]
    fn test_hml_render() {
        let f = HMLFormula::Diamond(
            "a".to_string(),
            Box::new(HMLFormula::Atom("p".to_string())),
        );
        assert_eq!(f.render(), "<a>p");

        let g = HMLFormula::Box("b".to_string(), Box::new(HMLFormula::True));
        assert_eq!(g.render(), "[b]tt");
    }

    #[test]
    fn test_distinguishing_trace_generation() {
        let lts = non_bisimilar_pair();
        let dt = DistinguishingTrace::generate(&lts, 0, 1, 10);
        assert!(dt.is_some(), "Should find distinguishing trace");
        let dt = dt.unwrap();
        assert!(dt.length() <= 10);
    }

    #[test]
    fn test_distinguishing_trace_none_for_bisimilar() {
        let lts = bisimilar_pair();
        let dt = DistinguishingTrace::generate(&lts, 0, 1, 20);
        assert!(dt.is_none(), "Bisimilar states should have no distinguishing trace");
    }

    #[test]
    fn test_distinguishing_trace_render() {
        let lts = non_bisimilar_pair();
        let dt = DistinguishingTrace::generate(&lts, 0, 1, 10).unwrap();
        let rendered = dt.render();
        assert!(rendered.contains("Distinguishing Trace"));
    }

    #[test]
    fn test_witness_minimizer_relation() {
        let lts = diamond_lts();
        let w = BisimulationWitness::construct(&lts, 0, 4).unwrap();
        let minimized = WitnessMinimizer::minimize_relation(&w, &lts);
        let v = minimized.validate(&lts);
        assert!(v.valid, "Minimized witness must remain valid: {:?}", v.violations);
        assert!(minimized.size() <= w.size());
    }

    #[test]
    fn test_witness_minimizer_formula() {
        let lts = non_bisimilar_pair();
        let df = DistinguishingFormula::compute(&lts, 0, 1).unwrap();
        let minimized = WitnessMinimizer::minimize_formula(&df, &lts);
        assert!(minimized.validate(&lts), "Minimized formula must still distinguish");
        assert!(minimized.size() <= df.size());
    }

    #[test]
    fn test_characteristic_formula_depth0() {
        let lts = labeled_lts();
        let cf = CharacteristicFormula::compute(&lts, 0, 0);
        // At depth 0, should be the atom "red"
        assert!(cf.evaluate(&lts, 0));
        // State 1 has label "blue", not "red"
        assert!(!cf.evaluate(&lts, 1));
        // State 2 also has "red"
        assert!(cf.evaluate(&lts, 2));
    }

    #[test]
    fn test_characteristic_formula_depth1() {
        let lts = labeled_lts();
        let cf = CharacteristicFormula::compute(&lts, 0, 1);
        assert!(cf.evaluate(&lts, 0));
        // State 2 is bisimilar to 0 at depth 1 (same labels, same transitions)
        assert!(cf.evaluate(&lts, 2));
    }

    #[test]
    fn test_coffee_tea_non_bisimilar() {
        let lts = coffee_tea_lts();
        // States 0 and 4 are classically non-bisimilar
        let w = BisimulationWitness::construct(&lts, 0, 4);
        assert!(w.is_none(), "Coffee/tea vending machines should not be bisimilar");

        let df = DistinguishingFormula::compute(&lts, 0, 4);
        assert!(df.is_some(), "Should find distinguishing formula for coffee/tea");
        let df = df.unwrap();
        assert!(df.validate(&lts));
    }

    #[test]
    fn test_lts_basic_operations() {
        let mut lts = WitnessLTS::new(3);
        lts.add_transition(0, "a", 1);
        lts.add_transition(0, "a", 2);
        lts.add_transition(1, "b", 2);
        lts.add_label(0, "start");

        assert_eq!(lts.successors(0, "a"), vec![1, 2]);
        assert_eq!(lts.predecessors(2, "a"), vec![0]);
        assert_eq!(lts.predecessors(2, "b"), vec![1]);

        let ea = lts.enabled_actions(0);
        assert_eq!(ea, vec!["a".to_string()]);

        assert!(lts.state_labels[0].contains(&"start".to_string()));
    }

    #[test]
    fn test_distinguishing_formula_with_labels() {
        let lts = labeled_lts();
        // State 0 has label "red", state 1 has label "blue"
        let df = DistinguishingFormula::compute(&lts, 0, 1);
        assert!(df.is_some());
        let df = df.unwrap();
        assert!(df.validate(&lts));
        assert_eq!(df.depth(), 0, "Label difference needs depth 0");
    }

    #[test]
    fn test_witness_size() {
        let lts = bisimilar_pair();
        let w = BisimulationWitness::construct(&lts, 0, 1).unwrap();
        assert!(w.size() >= 1);
    }

    #[test]
    fn test_distinguishing_trace_labeled() {
        let lts = labeled_lts();
        let dt = DistinguishingTrace::generate(&lts, 0, 1, 5);
        assert!(dt.is_some());
        let dt = dt.unwrap();
        // Divergence at step 0 since labels differ immediately
        assert_eq!(dt.divergence_point, 0);
    }

    #[test]
    fn test_hml_simplify() {
        let f = HMLFormula::And(
            Box::new(HMLFormula::True),
            Box::new(HMLFormula::Atom("p".to_string())),
        );
        let s = f.simplify();
        assert_eq!(s, HMLFormula::Atom("p".to_string()));

        let g = HMLFormula::Or(
            Box::new(HMLFormula::False),
            Box::new(HMLFormula::Atom("q".to_string())),
        );
        let sg = g.simplify();
        assert_eq!(sg, HMLFormula::Atom("q".to_string()));
    }

    #[test]
    fn test_validation_result_construction() {
        let ok = ValidationResult::ok();
        assert!(ok.valid);
        assert!(ok.violations.is_empty());

        let fail = ValidationResult::fail(vec!["oops".to_string()]);
        assert!(!fail.valid);
        assert_eq!(fail.violations.len(), 1);
    }
}
