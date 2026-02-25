// Graded (quantitative) model checking module for CABER.
// Implements satisfaction-degree computation over graded Kripke structures
// using [0,1]-valued semantics for CTL formulas.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// CriticalDirection
// ---------------------------------------------------------------------------

/// Direction of a critical state relative to a threshold.
#[derive(Debug, Clone, PartialEq)]
pub enum CriticalDirection {
    AboveThreshold,
    BelowThreshold,
    AtThreshold,
}

impl fmt::Display for CriticalDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CriticalDirection::AboveThreshold => write!(f, "above"),
            CriticalDirection::BelowThreshold => write!(f, "below"),
            CriticalDirection::AtThreshold => write!(f, "at"),
        }
    }
}

// ---------------------------------------------------------------------------
// GradedFormula
// ---------------------------------------------------------------------------

/// CTL formula for graded model checking.
#[derive(Debug, Clone, PartialEq)]
pub enum GradedFormula {
    True,
    False,
    Atom(String),
    Not(Box<GradedFormula>),
    And(Box<GradedFormula>, Box<GradedFormula>),
    Or(Box<GradedFormula>, Box<GradedFormula>),
    EX(Box<GradedFormula>),
    AX(Box<GradedFormula>),
    EU(Box<GradedFormula>, Box<GradedFormula>),
    AU(Box<GradedFormula>, Box<GradedFormula>),
    EG(Box<GradedFormula>),
    AG(Box<GradedFormula>),
    EF(Box<GradedFormula>),
    AF(Box<GradedFormula>),
}

impl GradedFormula {
    /// Render the formula as a human-readable string.
    pub fn render(&self) -> String {
        match self {
            GradedFormula::True => "true".to_string(),
            GradedFormula::False => "false".to_string(),
            GradedFormula::Atom(p) => p.clone(),
            GradedFormula::Not(f) => format!("¬({})", f.render()),
            GradedFormula::And(l, r) => format!("({} ∧ {})", l.render(), r.render()),
            GradedFormula::Or(l, r) => format!("({} ∨ {})", l.render(), r.render()),
            GradedFormula::EX(f) => format!("EX({})", f.render()),
            GradedFormula::AX(f) => format!("AX({})", f.render()),
            GradedFormula::EU(l, r) => format!("E[{} U {}]", l.render(), r.render()),
            GradedFormula::AU(l, r) => format!("A[{} U {}]", l.render(), r.render()),
            GradedFormula::EG(f) => format!("EG({})", f.render()),
            GradedFormula::AG(f) => format!("AG({})", f.render()),
            GradedFormula::EF(f) => format!("EF({})", f.render()),
            GradedFormula::AF(f) => format!("AF({})", f.render()),
        }
    }

    /// Maximum nesting depth of the formula.
    pub fn depth(&self) -> usize {
        match self {
            GradedFormula::True | GradedFormula::False | GradedFormula::Atom(_) => 0,
            GradedFormula::Not(f)
            | GradedFormula::EX(f)
            | GradedFormula::AX(f)
            | GradedFormula::EG(f)
            | GradedFormula::AG(f)
            | GradedFormula::EF(f)
            | GradedFormula::AF(f) => 1 + f.depth(),
            GradedFormula::And(l, r)
            | GradedFormula::Or(l, r)
            | GradedFormula::EU(l, r)
            | GradedFormula::AU(l, r) => 1 + l.depth().max(r.depth()),
        }
    }

    /// Collect all atomic proposition names referenced in the formula.
    pub fn atoms(&self) -> Vec<String> {
        let mut set = HashSet::new();
        self.collect_atoms(&mut set);
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    fn collect_atoms(&self, out: &mut HashSet<String>) {
        match self {
            GradedFormula::Atom(p) => {
                out.insert(p.clone());
            }
            GradedFormula::Not(f)
            | GradedFormula::EX(f)
            | GradedFormula::AX(f)
            | GradedFormula::EG(f)
            | GradedFormula::AG(f)
            | GradedFormula::EF(f)
            | GradedFormula::AF(f) => f.collect_atoms(out),
            GradedFormula::And(l, r)
            | GradedFormula::Or(l, r)
            | GradedFormula::EU(l, r)
            | GradedFormula::AU(l, r) => {
                l.collect_atoms(out);
                r.collect_atoms(out);
            }
            GradedFormula::True | GradedFormula::False => {}
        }
    }

    // Helper constructors
    pub fn atom(name: &str) -> Self {
        GradedFormula::Atom(name.to_string())
    }

    pub fn not(f: GradedFormula) -> Self {
        GradedFormula::Not(Box::new(f))
    }

    pub fn and(l: GradedFormula, r: GradedFormula) -> Self {
        GradedFormula::And(Box::new(l), Box::new(r))
    }

    pub fn or(l: GradedFormula, r: GradedFormula) -> Self {
        GradedFormula::Or(Box::new(l), Box::new(r))
    }

    pub fn ex(f: GradedFormula) -> Self {
        GradedFormula::EX(Box::new(f))
    }

    pub fn ax(f: GradedFormula) -> Self {
        GradedFormula::AX(Box::new(f))
    }

    pub fn eu(phi: GradedFormula, psi: GradedFormula) -> Self {
        GradedFormula::EU(Box::new(phi), Box::new(psi))
    }

    pub fn au(phi: GradedFormula, psi: GradedFormula) -> Self {
        GradedFormula::AU(Box::new(phi), Box::new(psi))
    }

    pub fn eg(f: GradedFormula) -> Self {
        GradedFormula::EG(Box::new(f))
    }

    pub fn ag(f: GradedFormula) -> Self {
        GradedFormula::AG(Box::new(f))
    }

    pub fn ef(f: GradedFormula) -> Self {
        GradedFormula::EF(Box::new(f))
    }

    pub fn af(f: GradedFormula) -> Self {
        GradedFormula::AF(Box::new(f))
    }
}

impl fmt::Display for GradedFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ---------------------------------------------------------------------------
// GradedKripke
// ---------------------------------------------------------------------------

/// A Kripke structure with probabilistic/weighted transitions and optional
/// graded atomic propositions.
#[derive(Debug, Clone)]
pub struct GradedKripke {
    pub num_states: usize,
    /// transitions[s] = vec of (target, probability/weight)
    pub transitions: Vec<Vec<(usize, f64)>>,
    /// Boolean labels per state
    pub labels: Vec<Vec<String>>,
    /// Optional graded (fuzzy) labels: (state, label) -> degree in [0,1]
    pub graded_labels: HashMap<(usize, String), f64>,
}

impl GradedKripke {
    pub fn new(num_states: usize) -> Self {
        GradedKripke {
            num_states,
            transitions: vec![Vec::new(); num_states],
            labels: vec![Vec::new(); num_states],
            graded_labels: HashMap::new(),
        }
    }

    pub fn add_transition(&mut self, source: usize, target: usize, probability: f64) {
        assert!(source < self.num_states, "source out of range");
        assert!(target < self.num_states, "target out of range");
        self.transitions[source].push((target, probability));
    }

    pub fn add_label(&mut self, state: usize, label: &str) {
        assert!(state < self.num_states, "state out of range");
        if !self.labels[state].contains(&label.to_string()) {
            self.labels[state].push(label.to_string());
        }
    }

    pub fn add_graded_label(&mut self, state: usize, label: &str, degree: f64) {
        assert!(state < self.num_states, "state out of range");
        assert!(
            (0.0..=1.0).contains(&degree),
            "degree must be in [0,1]"
        );
        self.graded_labels
            .insert((state, label.to_string()), degree);
    }

    pub fn successors(&self, state: usize) -> &[(usize, f64)] {
        assert!(state < self.num_states, "state out of range");
        &self.transitions[state]
    }

    /// Return the satisfaction degree of an atomic proposition at a state.
    /// Checks graded_labels first; falls back to Boolean labels (1.0 / 0.0).
    pub fn label_degree(&self, state: usize, label: &str) -> f64 {
        if let Some(&deg) = self.graded_labels.get(&(state, label.to_string())) {
            return deg;
        }
        if self.labels[state].contains(&label.to_string()) {
            1.0
        } else {
            0.0
        }
    }

    pub fn has_label(&self, state: usize, label: &str) -> bool {
        if let Some(&deg) = self.graded_labels.get(&(state, label.to_string())) {
            return deg > 0.0;
        }
        self.labels[state].contains(&label.to_string())
    }
}

// ---------------------------------------------------------------------------
// GradedConfig
// ---------------------------------------------------------------------------

/// Configuration for the graded model checker.
#[derive(Debug, Clone)]
pub struct GradedConfig {
    pub max_iterations: usize,
    pub epsilon: f64,
    pub discount_factor: f64,
}

impl Default for GradedConfig {
    fn default() -> Self {
        GradedConfig {
            max_iterations: 10000,
            epsilon: 1e-10,
            discount_factor: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GradedSatisfaction
// ---------------------------------------------------------------------------

/// Result of graded model checking: a satisfaction degree in [0,1] per state.
#[derive(Debug, Clone)]
pub struct GradedSatisfaction {
    pub degrees: Vec<f64>,
    pub formula_description: String,
    pub num_states: usize,
}

impl GradedSatisfaction {
    pub fn new(num_states: usize, formula: &str) -> Self {
        GradedSatisfaction {
            degrees: vec![0.0; num_states],
            formula_description: formula.to_string(),
            num_states,
        }
    }

    pub fn set_degree(&mut self, state: usize, degree: f64) {
        assert!(state < self.num_states, "state out of range");
        let clamped = degree.clamp(0.0, 1.0);
        self.degrees[state] = clamped;
    }

    pub fn degree(&self, state: usize) -> f64 {
        assert!(state < self.num_states, "state out of range");
        self.degrees[state]
    }

    pub fn max_degree(&self) -> f64 {
        self.degrees
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_degree(&self) -> f64 {
        self.degrees
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    pub fn average_degree(&self) -> f64 {
        if self.num_states == 0 {
            return 0.0;
        }
        self.degrees.iter().sum::<f64>() / self.num_states as f64
    }

    /// Return indices of states whose degree >= threshold.
    pub fn satisfying_states(&self, threshold: f64) -> Vec<usize> {
        self.degrees
            .iter()
            .enumerate()
            .filter(|(_, &d)| d >= threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Convert graded result to Boolean by thresholding.
    pub fn to_boolean(&self, threshold: f64) -> Vec<bool> {
        self.degrees.iter().map(|&d| d >= threshold).collect()
    }

    /// Check agreement with a Boolean result on extreme values.
    /// degree == 1.0 ⟹ boolean == true, degree == 0.0 ⟹ boolean == false.
    pub fn agrees_with_boolean(&self, boolean_result: &[bool]) -> bool {
        if boolean_result.len() != self.num_states {
            return false;
        }
        for (i, &b) in boolean_result.iter().enumerate() {
            let d = self.degrees[i];
            if (d - 1.0).abs() < f64::EPSILON && !b {
                return false;
            }
            if d.abs() < f64::EPSILON && b {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for GradedSatisfaction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GradedSatisfaction for: {}", self.formula_description)?;
        for (i, &d) in self.degrees.iter().enumerate() {
            writeln!(f, "  state {}: {:.6}", i, d)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SensitivityResult / StateSensitivity
// ---------------------------------------------------------------------------

/// Sensitivity of a single state to label perturbations.
#[derive(Debug, Clone)]
pub struct StateSensitivity {
    pub state: usize,
    pub sensitivity: f64,
    pub affected_labels: Vec<String>,
}

/// Result of sensitivity analysis across all states.
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    pub per_state: Vec<StateSensitivity>,
    pub most_sensitive_state: usize,
    pub max_sensitivity: f64,
}

impl SensitivityResult {
    pub fn render(&self) -> String {
        let mut out = format!(
            "Sensitivity analysis: most sensitive state = {} (sensitivity = {:.6})\n",
            self.most_sensitive_state, self.max_sensitivity
        );
        for ss in &self.per_state {
            out.push_str(&format!(
                "  state {}: sensitivity = {:.6}, affected labels = [{}]\n",
                ss.state,
                ss.sensitivity,
                ss.affected_labels.join(", ")
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// CriticalState
// ---------------------------------------------------------------------------

/// A state whose satisfaction degree is close to a threshold.
#[derive(Debug, Clone)]
pub struct CriticalState {
    pub state: usize,
    pub degree: f64,
    pub margin: f64,
    pub direction: CriticalDirection,
}

// ---------------------------------------------------------------------------
// GradedModelChecker
// ---------------------------------------------------------------------------

/// The main graded model checker. Computes [0,1]-valued satisfaction degrees
/// for CTL formulas over a weighted/probabilistic Kripke structure.
pub struct GradedModelChecker {
    pub model: GradedKripke,
    pub config: GradedConfig,
}

impl GradedModelChecker {
    pub fn new(model: GradedKripke, config: GradedConfig) -> Self {
        GradedModelChecker { model, config }
    }

    /// Compute the satisfaction degree for every state.
    pub fn check(&mut self, formula: &GradedFormula) -> GradedSatisfaction {
        let degs = self.eval(formula);
        let mut sat = GradedSatisfaction::new(self.model.num_states, &formula.render());
        for (i, d) in degs.into_iter().enumerate() {
            sat.set_degree(i, d);
        }
        sat
    }

    /// Check a single state.
    pub fn check_state(&self, state: usize, formula: &GradedFormula) -> f64 {
        let degs = self.eval(formula);
        degs[state]
    }

    /// How far is the degree from the threshold?  Positive ⟹ satisfies.
    pub fn robustness_margin(
        &self,
        state: usize,
        formula: &GradedFormula,
        threshold: f64,
    ) -> f64 {
        let d = self.check_state(state, formula);
        d - threshold
    }

    /// Sensitivity analysis: for each state, perturb labels and measure change.
    pub fn sensitivity_analysis(
        &mut self,
        formula: &GradedFormula,
    ) -> SensitivityResult {
        let baseline = self.eval(formula);
        let n = self.model.num_states;
        let mut per_state: Vec<StateSensitivity> = Vec::with_capacity(n);

        // Collect all labels in use
        let mut all_labels: HashSet<String> = HashSet::new();
        for s in 0..n {
            for l in &self.model.labels[s] {
                all_labels.insert(l.clone());
            }
        }
        for ((_, l), _) in &self.model.graded_labels {
            all_labels.insert(l.clone());
        }
        let all_labels: Vec<String> = all_labels.into_iter().collect();

        for s in 0..n {
            let mut max_change: f64 = 0.0;
            let mut affected: Vec<String> = Vec::new();

            for label in &all_labels {
                // Save original state
                let had_label = self.model.labels[s].contains(label);
                let had_graded = self
                    .model
                    .graded_labels
                    .get(&(s, label.clone()))
                    .copied();

                // Perturb: toggle the Boolean label
                if had_label {
                    self.model.labels[s].retain(|l| l != label);
                } else {
                    self.model.labels[s].push(label.clone());
                }

                // Also perturb graded label if present
                if let Some(old_deg) = had_graded {
                    let new_deg = 1.0 - old_deg;
                    self.model
                        .graded_labels
                        .insert((s, label.clone()), new_deg);
                }

                let perturbed = self.eval(formula);

                // Measure maximum change across all states
                let change: f64 = baseline
                    .iter()
                    .zip(perturbed.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);

                if change > 1e-12 {
                    affected.push(label.clone());
                }
                max_change = max_change.max(change);

                // Restore original state
                if had_label {
                    if !self.model.labels[s].contains(label) {
                        self.model.labels[s].push(label.clone());
                    }
                } else {
                    self.model.labels[s].retain(|l| l != label);
                }
                match had_graded {
                    Some(d) => {
                        self.model
                            .graded_labels
                            .insert((s, label.clone()), d);
                    }
                    None => {
                        self.model
                            .graded_labels
                            .remove(&(s, label.clone()));
                    }
                }
            }

            affected.sort();
            affected.dedup();
            per_state.push(StateSensitivity {
                state: s,
                sensitivity: max_change,
                affected_labels: affected,
            });
        }

        let (most_sensitive_state, max_sensitivity) = per_state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.sensitivity
                    .partial_cmp(&b.sensitivity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, ss)| (i, ss.sensitivity))
            .unwrap_or((0, 0.0));

        SensitivityResult {
            per_state,
            most_sensitive_state,
            max_sensitivity,
        }
    }

    /// Find states whose degree is close to the threshold.
    pub fn critical_states(
        &mut self,
        formula: &GradedFormula,
        threshold: f64,
    ) -> Vec<CriticalState> {
        let sat = self.check(formula);
        let n = self.model.num_states;
        let mut result = Vec::new();
        // "close" is defined as within 0.1 of the threshold, or we include all
        // and let the caller filter.  We include all states and sort by margin.
        for s in 0..n {
            let d = sat.degree(s);
            let margin = d - threshold;
            let direction = if margin.abs() < 1e-12 {
                CriticalDirection::AtThreshold
            } else if margin > 0.0 {
                CriticalDirection::AboveThreshold
            } else {
                CriticalDirection::BelowThreshold
            };
            result.push(CriticalState {
                state: s,
                degree: d,
                margin,
                direction,
            });
        }
        // Sort by absolute margin (most critical first)
        result.sort_by(|a, b| {
            a.margin
                .abs()
                .partial_cmp(&b.margin.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    // -----------------------------------------------------------------------
    // Core evaluation: bottom-up over formula structure
    // -----------------------------------------------------------------------

    fn eval(&self, formula: &GradedFormula) -> Vec<f64> {
        let n = self.model.num_states;
        match formula {
            GradedFormula::True => vec![1.0; n],
            GradedFormula::False => vec![0.0; n],

            GradedFormula::Atom(p) => {
                (0..n)
                    .map(|s| self.model.label_degree(s, p))
                    .collect()
            }

            GradedFormula::Not(f) => {
                let inner = self.eval(f);
                inner.iter().map(|&d| 1.0 - d).collect()
            }

            GradedFormula::And(l, r) => {
                let dl = self.eval(l);
                let dr = self.eval(r);
                dl.iter().zip(dr.iter()).map(|(&a, &b)| a.min(b)).collect()
            }

            GradedFormula::Or(l, r) => {
                let dl = self.eval(l);
                let dr = self.eval(r);
                dl.iter().zip(dr.iter()).map(|(&a, &b)| a.max(b)).collect()
            }

            GradedFormula::EX(f) => {
                let inner = self.eval(f);
                self.eval_ex(&inner)
            }

            GradedFormula::AX(f) => {
                let inner = self.eval(f);
                self.eval_ax(&inner)
            }

            GradedFormula::EU(phi, psi) => {
                let d_phi = self.eval(phi);
                let d_psi = self.eval(psi);
                self.eval_eu(&d_phi, &d_psi)
            }

            GradedFormula::AU(phi, psi) => {
                let d_phi = self.eval(phi);
                let d_psi = self.eval(psi);
                self.eval_au(&d_phi, &d_psi)
            }

            GradedFormula::EG(f) => {
                let inner = self.eval(f);
                self.eval_eg(&inner)
            }

            GradedFormula::AG(f) => {
                let inner = self.eval(f);
                self.eval_ag(&inner)
            }

            // EF(φ) = E[true U φ]
            GradedFormula::EF(f) => {
                let d_true = vec![1.0; n];
                let d_f = self.eval(f);
                self.eval_eu(&d_true, &d_f)
            }

            // AF(φ) = A[true U φ]
            GradedFormula::AF(f) => {
                let d_true = vec![1.0; n];
                let d_f = self.eval(f);
                self.eval_au(&d_true, &d_f)
            }
        }
    }

    /// EX(φ): max over successors of (prob * degree(φ, successor))
    fn eval_ex(&self, inner: &[f64]) -> Vec<f64> {
        let n = self.model.num_states;
        let gamma = self.config.discount_factor;
        (0..n)
            .map(|s| {
                let succs = self.model.successors(s);
                if succs.is_empty() {
                    0.0
                } else {
                    succs
                        .iter()
                        .map(|&(t, p)| p * inner[t] * gamma)
                        .fold(0.0_f64, f64::max)
                        .clamp(0.0, 1.0)
                }
            })
            .collect()
    }

    /// AX(φ): min over successors of degree(φ, successor), weighted by probability.
    fn eval_ax(&self, inner: &[f64]) -> Vec<f64> {
        let n = self.model.num_states;
        let gamma = self.config.discount_factor;
        (0..n)
            .map(|s| {
                let succs = self.model.successors(s);
                if succs.is_empty() {
                    // No successors: AX is vacuously true
                    1.0
                } else {
                    succs
                        .iter()
                        .map(|&(t, p)| {
                            // Weight the successor degree by probability
                            (p * inner[t] * gamma).clamp(0.0, 1.0)
                        })
                        .fold(f64::INFINITY, f64::min)
                        .clamp(0.0, 1.0)
                }
            })
            .collect()
    }

    /// EU(φ,ψ): least fixpoint
    ///   d_0(s)     = degree_ψ(s)
    ///   d_{n+1}(s) = max(degree_ψ(s), min(degree_φ(s), max_{s'∈succ(s)} prob(s,s') * d_n(s')))
    fn eval_eu(&self, d_phi: &[f64], d_psi: &[f64]) -> Vec<f64> {
        let n = self.model.num_states;
        let gamma = self.config.discount_factor;
        let mut d: Vec<f64> = d_psi.to_vec();

        for _ in 0..self.config.max_iterations {
            let mut d_new = vec![0.0; n];
            let mut converged = true;

            for s in 0..n {
                let succs = self.model.successors(s);
                let ex_val = if succs.is_empty() {
                    0.0
                } else {
                    succs
                        .iter()
                        .map(|&(t, p)| p * d[t] * gamma)
                        .fold(0.0_f64, f64::max)
                };

                d_new[s] = d_psi[s].max(d_phi[s].min(ex_val)).clamp(0.0, 1.0);

                if (d_new[s] - d[s]).abs() > self.config.epsilon {
                    converged = false;
                }
            }

            d = d_new;
            if converged {
                break;
            }
        }
        d
    }

    /// AU(φ,ψ): least fixpoint
    ///   d_0(s)     = degree_ψ(s)
    ///   d_{n+1}(s) = max(degree_ψ(s), min(degree_φ(s), min_{s'∈succ(s)} d_n(s')))
    fn eval_au(&self, d_phi: &[f64], d_psi: &[f64]) -> Vec<f64> {
        let n = self.model.num_states;
        let mut d: Vec<f64> = d_psi.to_vec();

        for _ in 0..self.config.max_iterations {
            let mut d_new = vec![0.0; n];
            let mut converged = true;

            for s in 0..n {
                let succs = self.model.successors(s);
                let ax_val = if succs.is_empty() {
                    // No successors: can't continue the path, so AU cannot
                    // be satisfied via the "continue" branch.
                    0.0
                } else {
                    succs
                        .iter()
                        .map(|&(t, _)| d[t])
                        .fold(f64::INFINITY, f64::min)
                };

                d_new[s] = d_psi[s].max(d_phi[s].min(ax_val)).clamp(0.0, 1.0);

                if (d_new[s] - d[s]).abs() > self.config.epsilon {
                    converged = false;
                }
            }

            d = d_new;
            if converged {
                break;
            }
        }
        d
    }

    /// EG(φ): greatest fixpoint
    ///   d_0(s)     = degree_φ(s)
    ///   d_{n+1}(s) = min(degree_φ(s), max_{s'∈succ(s)} prob(s,s') * d_n(s'))
    fn eval_eg(&self, d_phi: &[f64]) -> Vec<f64> {
        let n = self.model.num_states;
        let gamma = self.config.discount_factor;
        let mut d: Vec<f64> = d_phi.to_vec();

        for _ in 0..self.config.max_iterations {
            let mut d_new = vec![0.0; n];
            let mut converged = true;

            for s in 0..n {
                let succs = self.model.successors(s);
                let ex_val = if succs.is_empty() {
                    // No successors: EG requires an infinite path, so if
                    // there are no successors the path terminates — degree 0.
                    0.0
                } else {
                    succs
                        .iter()
                        .map(|&(t, p)| p * d[t] * gamma)
                        .fold(0.0_f64, f64::max)
                };

                d_new[s] = d_phi[s].min(ex_val).clamp(0.0, 1.0);

                if (d_new[s] - d[s]).abs() > self.config.epsilon {
                    converged = false;
                }
            }

            d = d_new;
            if converged {
                break;
            }
        }
        d
    }

    /// AG(φ): greatest fixpoint
    ///   d_0(s)     = degree_φ(s)
    ///   d_{n+1}(s) = min(degree_φ(s), min_{s'∈succ(s)} d_n(s'))
    fn eval_ag(&self, d_phi: &[f64]) -> Vec<f64> {
        let n = self.model.num_states;
        let mut d: Vec<f64> = d_phi.to_vec();

        for _ in 0..self.config.max_iterations {
            let mut d_new = vec![0.0; n];
            let mut converged = true;

            for s in 0..n {
                let succs = self.model.successors(s);
                let ax_val = if succs.is_empty() {
                    // No successors: AG on a deadlock — the current state
                    // is the only one on the path, so AG(φ) = φ.
                    d_phi[s]
                } else {
                    succs
                        .iter()
                        .map(|&(t, _)| d[t])
                        .fold(f64::INFINITY, f64::min)
                };

                d_new[s] = d_phi[s].min(ax_val).clamp(0.0, 1.0);

                if (d_new[s] - d[s]).abs() > self.config.epsilon {
                    converged = false;
                }
            }

            d = d_new;
            if converged {
                break;
            }
        }
        d
    }
}

// ---------------------------------------------------------------------------
// DistanceFromSatisfaction
// ---------------------------------------------------------------------------

/// Utility for computing how far each state's degree is from a threshold.
pub struct DistanceFromSatisfaction;

impl DistanceFromSatisfaction {
    /// For each state, compute |degree - threshold|.
    pub fn compute(graded: &GradedSatisfaction, threshold: f64) -> Vec<f64> {
        graded.degrees.iter().map(|&d| (d - threshold).abs()).collect()
    }

    /// Average distance from threshold across all states.
    pub fn average_distance(graded: &GradedSatisfaction, threshold: f64) -> f64 {
        if graded.num_states == 0 {
            return 0.0;
        }
        let distances = Self::compute(graded, threshold);
        distances.iter().sum::<f64>() / graded.num_states as f64
    }

    /// States whose degree is within `margin` of the threshold.
    pub fn states_within_margin(
        graded: &GradedSatisfaction,
        threshold: f64,
        margin: f64,
    ) -> Vec<usize> {
        graded
            .degrees
            .iter()
            .enumerate()
            .filter(|(_, &d)| (d - threshold).abs() <= margin)
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ComparisonResult / GradedBooleanComparison
// ---------------------------------------------------------------------------

/// Result of comparing graded and Boolean model-checking results.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub agreements: usize,
    pub disagreements: usize,
    pub agreement_rate: f64,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
}

impl ComparisonResult {
    pub fn summary(&self) -> String {
        format!(
            "Agreement: {}/{} ({:.1}%), FP={}, FN={}, Precision={:.4}, Recall={:.4}",
            self.agreements,
            self.agreements + self.disagreements,
            self.agreement_rate * 100.0,
            self.false_positives,
            self.false_negatives,
            self.precision,
            self.recall,
        )
    }
}

/// Compare graded satisfaction with Boolean results.
pub struct GradedBooleanComparison;

impl GradedBooleanComparison {
    /// Compare graded result (thresholded at 0.5) against a Boolean result.
    /// Agreement: degree >= 0.5 ⟺ boolean == true.
    /// Also check extreme-value agreement: degree==1 ⟹ true, degree==0 ⟹ false.
    /// Compute precision, recall, agreement rate.
    pub fn compare(
        graded: &GradedSatisfaction,
        boolean: &[bool],
    ) -> ComparisonResult {
        assert_eq!(
            graded.num_states,
            boolean.len(),
            "size mismatch between graded and boolean results"
        );
        let n = graded.num_states;

        let mut agreements: usize = 0;
        let mut disagreements: usize = 0;
        let mut true_positives: usize = 0;
        let mut false_positives: usize = 0;
        let mut false_negatives: usize = 0;

        for i in 0..n {
            let d = graded.degrees[i];
            let b = boolean[i];
            // Threshold the graded result at 0.5
            let graded_bool = d >= 0.5;

            if graded_bool == b {
                agreements += 1;
                if b {
                    true_positives += 1;
                }
            } else {
                disagreements += 1;
                if graded_bool && !b {
                    false_positives += 1;
                }
                if !graded_bool && b {
                    false_negatives += 1;
                }
            }
        }

        let total = agreements + disagreements;
        let agreement_rate = if total > 0 {
            agreements as f64 / total as f64
        } else {
            1.0
        };

        let predicted_positives = true_positives + false_positives;
        let precision = if predicted_positives > 0 {
            true_positives as f64 / predicted_positives as f64
        } else {
            1.0
        };

        let actual_positives = true_positives + false_negatives;
        let recall = if actual_positives > 0 {
            true_positives as f64 / actual_positives as f64
        } else {
            1.0
        };

        ComparisonResult {
            agreements,
            disagreements,
            agreement_rate,
            false_positives,
            false_negatives,
            precision,
            recall,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple 3-state chain:  0 --(1.0)--> 1 --(1.0)--> 2 (self-loop)
    /// Labels: state 0 has "a", state 1 has "b", state 2 has "a","b"
    fn simple_chain() -> GradedKripke {
        let mut k = GradedKripke::new(3);
        k.add_transition(0, 1, 1.0);
        k.add_transition(1, 2, 1.0);
        k.add_transition(2, 2, 1.0);
        k.add_label(0, "a");
        k.add_label(1, "b");
        k.add_label(2, "a");
        k.add_label(2, "b");
        k
    }

    /// Helper: build a branching model with probabilistic transitions.
    ///   0 --(0.7)--> 1
    ///   0 --(0.3)--> 2
    ///   1 --(1.0)--> 1  (self-loop)
    ///   2 --(1.0)--> 2  (self-loop)
    ///   Labels: 1 has "p", 2 has "q"
    fn branching_model() -> GradedKripke {
        let mut k = GradedKripke::new(3);
        k.add_transition(0, 1, 0.7);
        k.add_transition(0, 2, 0.3);
        k.add_transition(1, 1, 1.0);
        k.add_transition(2, 2, 1.0);
        k.add_label(1, "p");
        k.add_label(2, "q");
        k
    }

    fn default_checker(model: GradedKripke) -> GradedModelChecker {
        GradedModelChecker::new(model, GradedConfig::default())
    }

    // -----------------------------------------------------------------------
    // 1. Graded atom check
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_atom() {
        let k = simple_chain();
        let mut mc = default_checker(k);
        let sat = mc.check(&GradedFormula::atom("a"));
        assert!((sat.degree(0) - 1.0).abs() < 1e-9);
        assert!(sat.degree(1).abs() < 1e-9);
        assert!((sat.degree(2) - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 2. Graded negation
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_negation() {
        let k = simple_chain();
        let mut mc = default_checker(k);
        let sat = mc.check(&GradedFormula::not(GradedFormula::atom("a")));
        assert!(sat.degree(0).abs() < 1e-9);       // ¬a at state 0 (has a) = 0
        assert!((sat.degree(1) - 1.0).abs() < 1e-9); // ¬a at state 1 (no a) = 1
        assert!(sat.degree(2).abs() < 1e-9);       // ¬a at state 2 (has a) = 0
    }

    // -----------------------------------------------------------------------
    // 3. Graded And / Or
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_and_or() {
        let k = simple_chain();
        let mut mc = default_checker(k);

        // And: a ∧ b — only state 2 has both
        let sat_and = mc.check(&GradedFormula::and(
            GradedFormula::atom("a"),
            GradedFormula::atom("b"),
        ));
        assert!(sat_and.degree(0).abs() < 1e-9);
        assert!(sat_and.degree(1).abs() < 1e-9);
        assert!((sat_and.degree(2) - 1.0).abs() < 1e-9);

        // Or: a ∨ b — states 0, 1, 2 all have at least one
        let sat_or = mc.check(&GradedFormula::or(
            GradedFormula::atom("a"),
            GradedFormula::atom("b"),
        ));
        assert!((sat_or.degree(0) - 1.0).abs() < 1e-9);
        assert!((sat_or.degree(1) - 1.0).abs() < 1e-9);
        assert!((sat_or.degree(2) - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 4. Graded EX
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_ex() {
        let k = branching_model();
        let mut mc = default_checker(k);
        // EX(p): max over successors of (prob * degree(p, succ))
        // state 0: max(0.7*1.0, 0.3*0.0) = 0.7
        // state 1: max(1.0*1.0) = 1.0  (self-loop with label p)
        // state 2: max(1.0*0.0) = 0.0
        let sat = mc.check(&GradedFormula::ex(GradedFormula::atom("p")));
        assert!((sat.degree(0) - 0.7).abs() < 1e-9);
        assert!((sat.degree(1) - 1.0).abs() < 1e-9);
        assert!(sat.degree(2).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 5. Graded AX
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_ax() {
        let k = branching_model();
        let mut mc = default_checker(k);
        // AX(p): min over successors of (prob * degree(p, succ))
        // state 0: min(0.7*1.0, 0.3*0.0) = 0.0
        // state 1: min(1.0*1.0) = 1.0
        // state 2: min(1.0*0.0) = 0.0
        let sat = mc.check(&GradedFormula::ax(GradedFormula::atom("p")));
        assert!(sat.degree(0).abs() < 1e-9);
        assert!((sat.degree(1) - 1.0).abs() < 1e-9);
        assert!(sat.degree(2).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 6. Graded EU fixpoint
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_eu() {
        let k = simple_chain();
        let mut mc = default_checker(k);
        // E[a U b]: eventually b while a holds on the path
        // state 2: has b ⟹ degree = 1.0
        // state 1: has b ⟹ degree = 1.0
        // state 0: has a, succ is state 1 with degree 1.0 ⟹ min(1.0, 1.0*1.0) = 1.0
        //          max(0.0, 1.0) = 1.0  (d_psi(0) = 0, but iteration gives 1.0)
        let sat = mc.check(&GradedFormula::eu(
            GradedFormula::atom("a"),
            GradedFormula::atom("b"),
        ));
        assert!((sat.degree(2) - 1.0).abs() < 1e-9);
        assert!((sat.degree(1) - 1.0).abs() < 1e-9);
        assert!((sat.degree(0) - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 7. Graded AU fixpoint
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_au() {
        let k = simple_chain();
        let mut mc = default_checker(k);
        // A[a U b]:
        // state 2: has b ⟹ 1.0
        // state 1: has b ⟹ 1.0
        // state 0: has a, min over succs {state 1} of d[1]=1.0 ⟹ min(1.0, 1.0) =1.0
        //          max(0.0, 1.0)=1.0
        let sat = mc.check(&GradedFormula::au(
            GradedFormula::atom("a"),
            GradedFormula::atom("b"),
        ));
        assert!((sat.degree(2) - 1.0).abs() < 1e-9);
        assert!((sat.degree(1) - 1.0).abs() < 1e-9);
        assert!((sat.degree(0) - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 8. Graded EG fixpoint
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_eg() {
        let k = simple_chain();
        let mut mc = default_checker(k);
        // EG(a): greatest fixpoint
        // state 2: has a, self-loop with prob 1.0 ⟹ converges to 1.0
        // state 1: no a ⟹ d_phi = 0, so min(0, ...) = 0
        // state 0: has a, succ = {state 1}, prob*d[1] = 1.0*0.0 = 0.0
        //          min(1.0, 0.0) = 0.0
        let sat = mc.check(&GradedFormula::eg(GradedFormula::atom("a")));
        assert!((sat.degree(2) - 1.0).abs() < 1e-9);
        assert!(sat.degree(1).abs() < 1e-9);
        assert!(sat.degree(0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 9. Graded AG fixpoint
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_ag() {
        let k = simple_chain();
        let mut mc = default_checker(k);
        // AG(b):
        // state 0: no b ⟹ 0.0
        // state 1: has b, succ = {state 2 with b} ⟹ min(1.0, d[2])
        // state 2: has b, self-loop ⟹ converges to 1.0
        // state 1: min(1.0, 1.0) = 1.0
        let sat = mc.check(&GradedFormula::ag(GradedFormula::atom("b")));
        assert!(sat.degree(0).abs() < 1e-9);
        assert!((sat.degree(1) - 1.0).abs() < 1e-9);
        assert!((sat.degree(2) - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 10. Robustness margin
    // -----------------------------------------------------------------------
    #[test]
    fn test_robustness_margin() {
        let k = branching_model();
        let mc = default_checker(k);
        // EX(p) at state 0 = 0.7
        let m = mc.robustness_margin(0, &GradedFormula::ex(GradedFormula::atom("p")), 0.5);
        assert!((m - 0.2).abs() < 1e-9); // 0.7 - 0.5 = 0.2

        let m2 = mc.robustness_margin(2, &GradedFormula::ex(GradedFormula::atom("p")), 0.5);
        assert!((m2 - (-0.5)).abs() < 1e-9); // 0.0 - 0.5 = -0.5
    }

    // -----------------------------------------------------------------------
    // 11. Sensitivity analysis
    // -----------------------------------------------------------------------
    #[test]
    fn test_sensitivity_analysis() {
        let k = simple_chain();
        let mut mc = default_checker(k);
        let formula = GradedFormula::atom("a");
        let result = mc.sensitivity_analysis(&formula);
        assert_eq!(result.per_state.len(), 3);
        // Perturbing state 0's "a" label changes degree(0) from 1 to 0
        assert!(result.per_state[0].sensitivity > 0.5);
        assert!(result.max_sensitivity > 0.0);
    }

    // -----------------------------------------------------------------------
    // 12. Critical states
    // -----------------------------------------------------------------------
    #[test]
    fn test_critical_states() {
        let k = branching_model();
        let mut mc = default_checker(k);
        let formula = GradedFormula::ex(GradedFormula::atom("p"));
        let crits = mc.critical_states(&formula, 0.5);
        // We should get 3 CriticalState entries (one per state)
        assert_eq!(crits.len(), 3);

        // State 0 has degree 0.7, margin = 0.2
        let s0 = crits.iter().find(|c| c.state == 0).unwrap();
        assert!((s0.degree - 0.7).abs() < 1e-9);
        assert!((s0.margin - 0.2).abs() < 1e-9);
        assert_eq!(s0.direction, CriticalDirection::AboveThreshold);

        // State 2 has degree 0.0, margin = -0.5
        let s2 = crits.iter().find(|c| c.state == 2).unwrap();
        assert!((s2.degree - 0.0).abs() < 1e-9);
        assert_eq!(s2.direction, CriticalDirection::BelowThreshold);

        // Sorted by |margin|: state 0 (0.2) should be first
        assert_eq!(crits[0].state, 0);
    }

    // -----------------------------------------------------------------------
    // 13. Distance from satisfaction
    // -----------------------------------------------------------------------
    #[test]
    fn test_distance_from_satisfaction() {
        let mut sat = GradedSatisfaction::new(4, "test");
        sat.set_degree(0, 0.0);
        sat.set_degree(1, 0.3);
        sat.set_degree(2, 0.5);
        sat.set_degree(3, 1.0);

        let distances = DistanceFromSatisfaction::compute(&sat, 0.5);
        assert!((distances[0] - 0.5).abs() < 1e-9);
        assert!((distances[1] - 0.2).abs() < 1e-9);
        assert!(distances[2].abs() < 1e-9);
        assert!((distances[3] - 0.5).abs() < 1e-9);

        let avg = DistanceFromSatisfaction::average_distance(&sat, 0.5);
        assert!((avg - 0.3).abs() < 1e-9); // (0.5+0.2+0.0+0.5)/4 = 0.3

        let within = DistanceFromSatisfaction::states_within_margin(&sat, 0.5, 0.25);
        // States with |d - 0.5| <= 0.25: state 1 (0.2), state 2 (0.0)
        assert!(within.contains(&1));
        assert!(within.contains(&2));
        assert!(!within.contains(&0));
        assert!(!within.contains(&3));
    }

    // -----------------------------------------------------------------------
    // 14. Graded-Boolean comparison
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_boolean_comparison() {
        let mut sat = GradedSatisfaction::new(4, "test");
        sat.set_degree(0, 0.0);
        sat.set_degree(1, 0.3);
        sat.set_degree(2, 0.7);
        sat.set_degree(3, 1.0);

        let boolean = vec![false, false, true, true];
        let result = GradedBooleanComparison::compare(&sat, &boolean);
        // At threshold 0.5: graded bools = [false, false, true, true]
        // All agree with the boolean vector
        assert_eq!(result.agreements, 4);
        assert_eq!(result.disagreements, 0);
        assert!((result.agreement_rate - 1.0).abs() < 1e-9);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.false_negatives, 0);
        assert!((result.precision - 1.0).abs() < 1e-9);
        assert!((result.recall - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 15. agrees_with_boolean
    // -----------------------------------------------------------------------
    #[test]
    fn test_agrees_with_boolean() {
        let mut sat = GradedSatisfaction::new(3, "test");
        sat.set_degree(0, 1.0);
        sat.set_degree(1, 0.0);
        sat.set_degree(2, 0.5);

        // degree=1 ⟹ true, degree=0 ⟹ false, degree=0.5 is unconstrained
        assert!(sat.agrees_with_boolean(&[true, false, true]));
        assert!(sat.agrees_with_boolean(&[true, false, false]));

        // Violation: degree=1.0 but boolean=false
        assert!(!sat.agrees_with_boolean(&[false, false, true]));

        // Violation: degree=0.0 but boolean=true
        assert!(!sat.agrees_with_boolean(&[true, true, true]));
    }

    // -----------------------------------------------------------------------
    // 16. Graded labels
    // -----------------------------------------------------------------------
    #[test]
    fn test_graded_labels() {
        let mut k = GradedKripke::new(2);
        k.add_transition(0, 1, 1.0);
        k.add_transition(1, 1, 1.0);
        k.add_graded_label(0, "p", 0.6);
        k.add_graded_label(1, "p", 0.9);

        let mut mc = default_checker(k);
        let sat = mc.check(&GradedFormula::atom("p"));
        assert!((sat.degree(0) - 0.6).abs() < 1e-9);
        assert!((sat.degree(1) - 0.9).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 17. Complex formula
    // -----------------------------------------------------------------------
    #[test]
    fn test_complex_formula() {
        let k = simple_chain();
        let mut mc = default_checker(k);

        // EF(a ∧ b) = E[true U (a ∧ b)]
        // state 2 has both a, b ⟹ 1.0
        // state 1 can reach state 2 ⟹ 1.0
        // state 0 can reach via 1 → 2 ⟹ 1.0
        let formula = GradedFormula::ef(GradedFormula::and(
            GradedFormula::atom("a"),
            GradedFormula::atom("b"),
        ));
        let sat = mc.check(&formula);
        assert!((sat.degree(0) - 1.0).abs() < 1e-9);
        assert!((sat.degree(1) - 1.0).abs() < 1e-9);
        assert!((sat.degree(2) - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 18. Formula utilities
    // -----------------------------------------------------------------------
    #[test]
    fn test_formula_utilities() {
        let f = GradedFormula::eu(
            GradedFormula::atom("a"),
            GradedFormula::and(GradedFormula::atom("b"), GradedFormula::atom("c")),
        );
        assert_eq!(f.depth(), 2);
        let atoms = f.atoms();
        assert_eq!(atoms, vec!["a", "b", "c"]);
        let rendered = f.render();
        assert!(rendered.contains("U"));
    }

    // -----------------------------------------------------------------------
    // 19. GradedSatisfaction utilities
    // -----------------------------------------------------------------------
    #[test]
    fn test_satisfaction_utilities() {
        let mut sat = GradedSatisfaction::new(5, "test");
        sat.set_degree(0, 0.1);
        sat.set_degree(1, 0.4);
        sat.set_degree(2, 0.6);
        sat.set_degree(3, 0.9);
        sat.set_degree(4, 1.0);

        assert!((sat.max_degree() - 1.0).abs() < 1e-9);
        assert!((sat.min_degree() - 0.1).abs() < 1e-9);
        assert!((sat.average_degree() - 0.6).abs() < 1e-9);

        let above = sat.satisfying_states(0.5);
        assert_eq!(above, vec![2, 3, 4]);

        let bools = sat.to_boolean(0.5);
        assert_eq!(bools, vec![false, false, true, true, true]);
    }

    // -----------------------------------------------------------------------
    // 20. EF / AF
    // -----------------------------------------------------------------------
    #[test]
    fn test_ef_af() {
        // 0 → 1 → 2 (self-loop), label "goal" only at state 2
        let mut k = GradedKripke::new(3);
        k.add_transition(0, 1, 1.0);
        k.add_transition(1, 2, 1.0);
        k.add_transition(2, 2, 1.0);
        k.add_label(2, "goal");

        let mut mc = default_checker(k);

        // EF(goal): all states can reach state 2
        let sat_ef = mc.check(&GradedFormula::ef(GradedFormula::atom("goal")));
        assert!((sat_ef.degree(0) - 1.0).abs() < 1e-9);
        assert!((sat_ef.degree(1) - 1.0).abs() < 1e-9);
        assert!((sat_ef.degree(2) - 1.0).abs() < 1e-9);

        // AF(goal): on all paths from any state, goal is eventually reached
        let sat_af = mc.check(&GradedFormula::af(GradedFormula::atom("goal")));
        assert!((sat_af.degree(0) - 1.0).abs() < 1e-9);
        assert!((sat_af.degree(1) - 1.0).abs() < 1e-9);
        assert!((sat_af.degree(2) - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 21. Comparison with disagreements
    // -----------------------------------------------------------------------
    #[test]
    fn test_comparison_disagreements() {
        let mut sat = GradedSatisfaction::new(4, "test");
        sat.set_degree(0, 0.0);
        sat.set_degree(1, 0.8); // graded says true (>= 0.5)
        sat.set_degree(2, 0.2); // graded says false (< 0.5)
        sat.set_degree(3, 1.0);

        // boolean disagrees on states 1 and 2
        let boolean = vec![false, false, true, true];
        let result = GradedBooleanComparison::compare(&sat, &boolean);
        assert_eq!(result.agreements, 2); // states 0, 3 agree
        assert_eq!(result.disagreements, 2); // states 1, 2 disagree
        assert_eq!(result.false_positives, 1); // state 1: graded true, boolean false
        assert_eq!(result.false_negatives, 1); // state 2: graded false, boolean true
    }

    // -----------------------------------------------------------------------
    // 22. ComparisonResult summary
    // -----------------------------------------------------------------------
    #[test]
    fn test_comparison_summary() {
        let result = ComparisonResult {
            agreements: 8,
            disagreements: 2,
            agreement_rate: 0.8,
            false_positives: 1,
            false_negatives: 1,
            precision: 0.75,
            recall: 0.75,
        };
        let s = result.summary();
        assert!(s.contains("80.0%"));
        assert!(s.contains("FP=1"));
        assert!(s.contains("FN=1"));
    }

    // -----------------------------------------------------------------------
    // 23. SensitivityResult render
    // -----------------------------------------------------------------------
    #[test]
    fn test_sensitivity_render() {
        let result = SensitivityResult {
            per_state: vec![
                StateSensitivity {
                    state: 0,
                    sensitivity: 0.5,
                    affected_labels: vec!["a".into()],
                },
                StateSensitivity {
                    state: 1,
                    sensitivity: 0.1,
                    affected_labels: vec![],
                },
            ],
            most_sensitive_state: 0,
            max_sensitivity: 0.5,
        };
        let rendered = result.render();
        assert!(rendered.contains("most sensitive state = 0"));
        assert!(rendered.contains("0.5"));
    }

    // -----------------------------------------------------------------------
    // 24. Discount factor
    // -----------------------------------------------------------------------
    #[test]
    fn test_discount_factor() {
        let mut k = GradedKripke::new(2);
        k.add_transition(0, 1, 1.0);
        k.add_transition(1, 1, 1.0);
        k.add_label(1, "p");

        let config = GradedConfig {
            discount_factor: 0.5,
            ..GradedConfig::default()
        };
        let mut mc = GradedModelChecker::new(k, config);
        // EX(p) at state 0 with discount 0.5: prob=1.0, d_p=1.0, gamma=0.5 ⟹ 0.5
        let sat = mc.check(&GradedFormula::ex(GradedFormula::atom("p")));
        assert!((sat.degree(0) - 0.5).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 25. True / False formulas
    // -----------------------------------------------------------------------
    #[test]
    fn test_true_false() {
        let k = GradedKripke::new(3);
        let mut mc = default_checker(k);

        let sat_t = mc.check(&GradedFormula::True);
        for i in 0..3 {
            assert!((sat_t.degree(i) - 1.0).abs() < 1e-9);
        }

        let sat_f = mc.check(&GradedFormula::False);
        for i in 0..3 {
            assert!(sat_f.degree(i).abs() < 1e-9);
        }
    }
}
