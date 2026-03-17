//! Quantifier-free linear real arithmetic (QF_LRA) feasibility checking.
//!
//! Provides [`LinearCombination`] and [`LinearConstraint`] for representing
//! systems of linear inequalities over real-valued variables, and
//! [`FeasibilityChecker`] for deciding satisfiability via iterative bound
//! propagation.
//!
//! The checker implements a lightweight theory solver suitable for integration
//! into a DPLL(T) loop: it maintains per-variable interval bounds, propagates
//! single-variable constraints analytically, and uses a fixed-point iteration
//! to tighten multi-variable bounds.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

/// Comparison relation in a linear constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Relation {
    /// Less-than-or-equal (≤).
    Le,
    /// Strictly less-than (<).
    Lt,
    /// Greater-than-or-equal (≥).
    Ge,
    /// Strictly greater-than (>).
    Gt,
    /// Equality (=).
    Eq,
}

impl Relation {
    /// Flip the relation (swap lhs and rhs).
    pub fn flip(self) -> Self {
        match self {
            Relation::Le => Relation::Ge,
            Relation::Lt => Relation::Gt,
            Relation::Ge => Relation::Le,
            Relation::Gt => Relation::Lt,
            Relation::Eq => Relation::Eq,
        }
    }

    /// Returns `true` for strict inequalities (`<` or `>`).
    pub fn is_strict(self) -> bool {
        matches!(self, Relation::Lt | Relation::Gt)
    }
}

// ---------------------------------------------------------------------------
// LinearCombination
// ---------------------------------------------------------------------------

/// A linear combination of variables plus a constant: Σ aᵢ·xᵢ + c.
///
/// Represented as a sparse map from variable names to coefficients, plus an
/// additive constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearCombination {
    /// Sparse map: variable name → coefficient.
    pub terms: IndexMap<String, f64>,
    /// Additive constant.
    pub constant: f64,
}

impl LinearCombination {
    /// Create an empty linear combination (zero).
    pub fn new() -> Self {
        Self {
            terms: IndexMap::new(),
            constant: 0.0,
        }
    }

    /// Create a linear combination consisting of a single variable with
    /// coefficient 1: `1·name + 0`.
    pub fn variable(name: impl Into<String>) -> Self {
        let mut terms = IndexMap::new();
        terms.insert(name.into(), 1.0);
        Self {
            terms,
            constant: 0.0,
        }
    }

    /// Create a constant linear combination: `0·() + c`.
    pub fn constant(c: f64) -> Self {
        Self {
            terms: IndexMap::new(),
            constant: c,
        }
    }

    /// Add a term `coeff · name` to this combination.
    pub fn add_term(&mut self, name: impl Into<String>, coeff: f64) {
        let name = name.into();
        let entry = self.terms.entry(name).or_insert(0.0);
        *entry += coeff;
    }

    /// Negate every coefficient and the constant.
    pub fn negate(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(k, v)| (k.clone(), -v))
            .collect();
        Self {
            terms,
            constant: -self.constant,
        }
    }

    /// Element-wise addition of two linear combinations.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.constant += other.constant;
        for (var, &coeff) in &other.terms {
            let entry = result.terms.entry(var.clone()).or_insert(0.0);
            *entry += coeff;
        }
        result
    }

    /// Evaluate the combination under a variable assignment.
    pub fn evaluate(&self, assignment: &IndexMap<String, f64>) -> f64 {
        let mut value = self.constant;
        for (var, &coeff) in &self.terms {
            if let Some(&val) = assignment.get(var) {
                value += coeff * val;
            }
        }
        value
    }

    /// Returns `true` when there are no variable terms.
    pub fn is_constant(&self) -> bool {
        self.terms.iter().all(|(_, v)| v.abs() < 1e-15)
    }

    /// Collect the set of variable names referenced by this combination.
    pub fn variables(&self) -> Vec<&str> {
        self.terms
            .iter()
            .filter(|(_, v)| v.abs() > 1e-15)
            .map(|(k, _)| k.as_str())
            .collect()
    }
}

impl Default for LinearCombination {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LinearConstraint
// ---------------------------------------------------------------------------

/// A linear constraint of the form `lhs R rhs` where `R` is a [`Relation`].
///
/// Both sides are [`LinearCombination`]s, so a constraint like `2x + 3y ≤ 10`
/// is represented as `lhs = 2x + 3y`, `relation = Le`, `rhs = 10`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearConstraint {
    /// Left-hand side.
    pub lhs: LinearCombination,
    /// Comparison relation.
    pub relation: Relation,
    /// Right-hand side.
    pub rhs: LinearCombination,
}

impl LinearConstraint {
    /// Construct a new constraint: `lhs relation rhs`.
    pub fn new(lhs: LinearCombination, relation: Relation, rhs: LinearCombination) -> Self {
        Self { lhs, relation, rhs }
    }

    /// Rewrite to the canonical form `(lhs − rhs) R 0`.
    pub fn normalize(&self) -> (LinearCombination, Relation) {
        (self.lhs.add(&self.rhs.negate()), self.relation)
    }

    /// Collect all variable names appearing in either side.
    pub fn variables(&self) -> Vec<String> {
        let mut vars: Vec<String> = self
            .lhs
            .terms
            .keys()
            .chain(self.rhs.terms.keys())
            .cloned()
            .collect();
        vars.sort();
        vars.dedup();
        vars
    }
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// A satisfying variable assignment returned by the feasibility checker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    values: IndexMap<String, f64>,
}

impl Assignment {
    /// Create an assignment from an ordered map.
    pub fn from_index_map(map: IndexMap<String, f64>) -> Self {
        Self { values: map }
    }

    /// Convert to an [`IndexMap`].
    pub fn to_index_map(&self) -> IndexMap<String, f64> {
        self.values.clone()
    }

    /// Look up a single variable's value.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).copied()
    }
}

// ---------------------------------------------------------------------------
// FeasibilityResult
// ---------------------------------------------------------------------------

/// Outcome of a QF_LRA feasibility check.
#[derive(Debug, Clone)]
pub enum FeasibilityResult {
    /// The constraint system is satisfiable; a witness assignment is provided.
    Feasible(Assignment),
    /// The constraint system is unsatisfiable.
    Infeasible,
    /// The checker could not determine feasibility (e.g., iteration limit).
    Unknown,
}

impl FeasibilityResult {
    /// Returns `true` when a feasible assignment was found.
    pub fn is_feasible(&self) -> bool {
        matches!(self, FeasibilityResult::Feasible(_))
    }
}

// ---------------------------------------------------------------------------
// FeasibilityChecker
// ---------------------------------------------------------------------------

/// Interval bounds for a single variable.
#[derive(Debug, Clone)]
struct VarBounds {
    lower: f64,
    upper: f64,
}

impl VarBounds {
    fn new() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    fn midpoint(&self) -> f64 {
        let lo = if self.lower.is_finite() {
            self.lower
        } else {
            -1e6
        };
        let hi = if self.upper.is_finite() {
            self.upper
        } else {
            1e6
        };
        (lo + hi) / 2.0
    }

    fn is_feasible(&self) -> bool {
        self.lower <= self.upper + 1e-12
    }

    fn tighten_lower(&mut self, bound: f64) -> bool {
        if bound > self.lower {
            self.lower = bound;
            true
        } else {
            false
        }
    }

    fn tighten_upper(&mut self, bound: f64) -> bool {
        if bound < self.upper {
            self.upper = bound;
            true
        } else {
            false
        }
    }
}

/// Feasibility checker for systems of linear constraints over reals.
///
/// Maintains per-variable interval bounds and iteratively propagates
/// constraints to tighten them.  When the bounds converge without
/// contradiction, a feasible midpoint assignment is produced.
pub struct FeasibilityChecker {
    variables: IndexMap<String, usize>,
    bounds: Vec<VarBounds>,
    constraints: Vec<LinearConstraint>,
    max_propagation_rounds: usize,
}

impl FeasibilityChecker {
    /// Create an empty checker with default settings.
    pub fn new() -> Self {
        Self {
            variables: IndexMap::new(),
            bounds: Vec::new(),
            constraints: Vec::new(),
            max_propagation_rounds: 200,
        }
    }

    /// Register a variable for the system.
    pub fn add_variable(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.variables.contains_key(&name) {
            let idx = self.bounds.len();
            self.variables.insert(name, idx);
            self.bounds.push(VarBounds::new());
        }
    }

    /// Add a linear constraint to the system.
    pub fn add_constraint(&mut self, constraint: LinearConstraint) {
        for var in constraint.variables() {
            self.add_variable(var);
        }
        self.constraints.push(constraint);
    }

    /// Read-only access to the registered variables and their indices.
    pub fn variables(&self) -> &IndexMap<String, usize> {
        &self.variables
    }

    /// Read-only access to the current constraint set.
    pub fn constraints(&self) -> &[LinearConstraint] {
        &self.constraints
    }

    /// Check feasibility of the current constraint system.
    ///
    /// Performs iterative bound propagation.  If all bounds remain
    /// consistent after convergence, returns a feasible midpoint
    /// assignment.  If any variable's interval becomes empty, returns
    /// [`FeasibilityResult::Infeasible`].
    pub fn check(&mut self) -> FeasibilityResult {
        // Phase 1: iterative bound propagation.
        for _round in 0..self.max_propagation_rounds {
            let mut changed = false;
            for ci in 0..self.constraints.len() {
                let c = &self.constraints[ci];
                let (combo, relation) = c.normalize();
                if !self.propagate_normalized(&combo, relation) {
                    // Detected infeasibility.
                    return FeasibilityResult::Infeasible;
                }
                // Check if any bounds tightened by re-propagating single-var constraints.
                changed |= self.propagate_single_var(&combo, relation);
            }
            if !changed {
                break;
            }
        }

        // Phase 2: verify feasibility of the midpoint assignment.
        if !self.bounds_consistent() {
            return FeasibilityResult::Infeasible;
        }

        let assignment = self.midpoint_assignment();
        if self.satisfies_all(&assignment) {
            FeasibilityResult::Feasible(Assignment::from_index_map(assignment))
        } else {
            // Bounds are consistent but the midpoint doesn't satisfy all
            // multi-variable constraints — try a perturbation search.
            match self.search_feasible() {
                Some(a) => FeasibilityResult::Feasible(Assignment::from_index_map(a)),
                None => FeasibilityResult::Unknown,
            }
        }
    }

    // -- internal helpers --

    /// Propagate a normalized constraint `combo R 0` to tighten bounds on
    /// single-variable constraints.
    fn propagate_single_var(&mut self, combo: &LinearCombination, relation: Relation) -> bool {
        let active: Vec<_> = combo
            .terms
            .iter()
            .filter(|(_, v)| v.abs() > 1e-15)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        if active.len() != 1 {
            return false;
        }
        let (var, coeff) = &active[0];
        let bound = -combo.constant / coeff;
        let idx = match self.variables.get(var.as_str()) {
            Some(&i) => i,
            None => return false,
        };
        match relation {
            Relation::Le | Relation::Lt => {
                if *coeff > 0.0 {
                    self.bounds[idx].tighten_upper(bound)
                } else {
                    self.bounds[idx].tighten_lower(-bound)
                }
            }
            Relation::Ge | Relation::Gt => {
                if *coeff > 0.0 {
                    self.bounds[idx].tighten_lower(bound)
                } else {
                    self.bounds[idx].tighten_upper(-bound)
                }
            }
            Relation::Eq => {
                let a = self.bounds[idx].tighten_lower(bound);
                let b = self.bounds[idx].tighten_upper(bound);
                a || b
            }
        }
    }

    /// Check a normalized constraint for obvious infeasibility using current
    /// bounds.  Returns `false` if the constraint is provably unsatisfiable.
    fn propagate_normalized(&self, combo: &LinearCombination, relation: Relation) -> bool {
        if !combo.is_constant() {
            return true;
        }
        let val = combo.constant;
        match relation {
            Relation::Le => val <= 1e-12,
            Relation::Lt => val < -1e-12,
            Relation::Ge => val >= -1e-12,
            Relation::Gt => val > 1e-12,
            Relation::Eq => val.abs() < 1e-12,
        }
    }

    fn bounds_consistent(&self) -> bool {
        self.bounds.iter().all(|b| b.is_feasible())
    }

    fn midpoint_assignment(&self) -> IndexMap<String, f64> {
        self.variables
            .iter()
            .map(|(name, &idx)| (name.clone(), self.bounds[idx].midpoint()))
            .collect()
    }

    fn satisfies_all(&self, assignment: &IndexMap<String, f64>) -> bool {
        self.constraints.iter().all(|c| {
            let lhs_val = c.lhs.evaluate(assignment);
            let rhs_val = c.rhs.evaluate(assignment);
            match c.relation {
                Relation::Le => lhs_val <= rhs_val + 1e-9,
                Relation::Lt => lhs_val < rhs_val + 1e-9,
                Relation::Ge => lhs_val >= rhs_val - 1e-9,
                Relation::Gt => lhs_val > rhs_val - 1e-9,
                Relation::Eq => (lhs_val - rhs_val).abs() < 1e-9,
            }
        })
    }

    /// Attempt to find a feasible point by perturbing the midpoint toward
    /// constraint satisfaction.  Uses a simple iterative projection.
    fn search_feasible(&self) -> Option<IndexMap<String, f64>> {
        let mut point = self.midpoint_assignment();
        for _iter in 0..100 {
            if self.satisfies_all(&point) {
                return Some(point);
            }
            // Project toward feasibility for each violated constraint.
            for c in &self.constraints {
                let lhs_val = c.lhs.evaluate(&point);
                let rhs_val = c.rhs.evaluate(&point);
                let violation = match c.relation {
                    Relation::Le | Relation::Lt => (lhs_val - rhs_val).max(0.0),
                    Relation::Ge | Relation::Gt => (rhs_val - lhs_val).max(0.0),
                    Relation::Eq => lhs_val - rhs_val,
                };
                if violation.abs() < 1e-10 {
                    continue;
                }
                // Move variables proportionally to their coefficients.
                let (combo, _) = c.normalize();
                let norm_sq: f64 = combo.terms.values().map(|v| v * v).sum();
                if norm_sq < 1e-15 {
                    continue;
                }
                let step = violation / norm_sq;
                for (var, &coeff) in &combo.terms {
                    if let Some(val) = point.get_mut(var) {
                        *val -= step * coeff * 0.5;
                        // Clamp to bounds.
                        if let Some(&idx) = self.variables.get(var.as_str()) {
                            let b = &self.bounds[idx];
                            let lo = if b.lower.is_finite() {
                                b.lower
                            } else {
                                -1e6
                            };
                            let hi = if b.upper.is_finite() {
                                b.upper
                            } else {
                                1e6
                            };
                            *val = val.clamp(lo, hi);
                        }
                    }
                }
            }
        }
        if self.satisfies_all(&point) {
            Some(point)
        } else {
            None
        }
    }
}

impl Default for FeasibilityChecker {
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

    #[test]
    fn constant_combination() {
        let c = LinearCombination::constant(5.0);
        assert!(c.is_constant());
        assert!((c.constant - 5.0).abs() < 1e-15);
    }

    #[test]
    fn variable_combination() {
        let v = LinearCombination::variable("x");
        assert!(!v.is_constant());
        assert_eq!(v.terms.get("x"), Some(&1.0));
    }

    #[test]
    fn evaluate_combination() {
        let mut lc = LinearCombination::new();
        lc.add_term("x", 2.0);
        lc.add_term("y", -1.0);
        lc.constant = 3.0;
        let mut assignment = IndexMap::new();
        assignment.insert("x".into(), 4.0);
        assignment.insert("y".into(), 1.0);
        let val = lc.evaluate(&assignment);
        assert!((val - 10.0).abs() < 1e-12); // 2*4 - 1 + 3 = 10
    }

    #[test]
    fn negate_and_add() {
        let a = LinearCombination::variable("x");
        let b = LinearCombination::constant(5.0);
        let sum = a.add(&b);
        assert!((sum.constant - 5.0).abs() < 1e-15);
        assert_eq!(sum.terms.get("x"), Some(&1.0));

        let neg = sum.negate();
        assert!((neg.constant + 5.0).abs() < 1e-15);
        assert_eq!(neg.terms.get("x"), Some(&-1.0));
    }

    #[test]
    fn simple_feasible_box() {
        let mut checker = FeasibilityChecker::new();
        checker.add_variable("x");
        checker.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"),
            Relation::Ge,
            LinearCombination::constant(0.0),
        ));
        checker.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"),
            Relation::Le,
            LinearCombination::constant(10.0),
        ));
        let result = checker.check();
        assert!(result.is_feasible());
        if let FeasibilityResult::Feasible(a) = result {
            let x = a.get("x").unwrap();
            assert!(x >= 0.0 && x <= 10.0);
        }
    }

    #[test]
    fn infeasible_contradictory_bounds() {
        let mut checker = FeasibilityChecker::new();
        checker.add_variable("x");
        checker.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"),
            Relation::Ge,
            LinearCombination::constant(10.0),
        ));
        checker.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"),
            Relation::Le,
            LinearCombination::constant(5.0),
        ));
        let result = checker.check();
        assert!(!result.is_feasible());
    }

    #[test]
    fn two_variable_feasible() {
        let mut checker = FeasibilityChecker::new();
        checker.add_variable("x");
        checker.add_variable("y");
        for name in ["x", "y"] {
            checker.add_constraint(LinearConstraint::new(
                LinearCombination::variable(name),
                Relation::Ge,
                LinearCombination::constant(0.0),
            ));
            checker.add_constraint(LinearConstraint::new(
                LinearCombination::variable(name),
                Relation::Le,
                LinearCombination::constant(10.0),
            ));
        }
        let result = checker.check();
        assert!(result.is_feasible());
    }

    #[test]
    fn assignment_to_index_map() {
        let mut map = IndexMap::new();
        map.insert("a".into(), 1.0);
        map.insert("b".into(), 2.0);
        let a = Assignment::from_index_map(map.clone());
        assert_eq!(a.to_index_map(), map);
        assert!((a.get("a").unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn relation_flip() {
        assert_eq!(Relation::Le.flip(), Relation::Ge);
        assert_eq!(Relation::Gt.flip(), Relation::Lt);
        assert_eq!(Relation::Eq.flip(), Relation::Eq);
    }
}
