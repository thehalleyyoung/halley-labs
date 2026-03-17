// Theory propagation interfaces and implementations for SMT solving.
// Includes: TheoryPropagator trait, LinearArithmeticTheory, EqualityTheory, CombinedTheory.

use crate::variable::{Assignment, Literal, LiteralVec, Variable};
use smallvec::smallvec;
use std::collections::HashMap;
use std::fmt;

// ── TheoryResult ──────────────────────────────────────────────────────────────

/// Result of a theory consistency check.
#[derive(Debug, Clone)]
pub enum TheoryResult {
    /// The assignment is consistent with the theory.
    Consistent,
    /// The assignment is inconsistent; explanation is a set of literals that form a conflict.
    Inconsistent(Vec<Literal>),
    /// Theory can propagate a new literal with an explanation.
    Propagation(Literal, Vec<Literal>),
}

// ── TheoryLemma ───────────────────────────────────────────────────────────────

/// A clause learned from theory reasoning.
#[derive(Debug, Clone)]
pub struct TheoryLemma {
    pub literals: LiteralVec,
    pub source: String,
}

impl TheoryLemma {
    pub fn new(literals: LiteralVec, source: impl Into<String>) -> Self {
        TheoryLemma {
            literals,
            source: source.into(),
        }
    }
}

// ── TheoryPropagator ──────────────────────────────────────────────────────────

/// Interface for SMT theory solvers.
pub trait TheoryPropagator: fmt::Debug {
    /// Check consistency of the current assignment with the theory.
    fn check_consistency(&mut self, assignment: &Assignment) -> TheoryResult;

    /// Propagate theory consequences. Returns new lemmas to add.
    fn propagate(&mut self, assignment: &Assignment) -> Vec<TheoryLemma>;

    /// Explain why a literal was propagated by the theory.
    /// Returns the set of literals that imply the given literal.
    fn explain(&self, literal: Literal) -> Vec<Literal>;

    /// Notify the theory that a new variable has been assigned.
    fn on_assign(&mut self, _var: Variable, _value: bool) {}

    /// Notify the theory that a variable has been unassigned (backtrack).
    fn on_unassign(&mut self, _var: Variable) {}

    /// Name of this theory (for debugging).
    fn name(&self) -> &str;
}

// ── Bound / LinearArithmeticTheory ────────────────────────────────────────────

/// A bound on an integer variable: var <= value or var >= value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundKind {
    Upper,
    Lower,
}

/// A constraint of the form: sum(coeffs[i] * int_vars[i]) <op> rhs,
/// guarded by a Boolean literal (when literal is true, constraint is active).
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// The Boolean literal that activates this constraint.
    pub guard: Literal,
    /// Coefficients.
    pub coeffs: Vec<i64>,
    /// Integer variable indices.
    pub int_vars: Vec<u32>,
    /// Right-hand side.
    pub rhs: i64,
    /// Comparison operator: Le (<=), Lt (<), Eq (=), Ge (>=), Gt (>).
    pub op: ComparisonOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Le,
    Lt,
    Eq,
    Ge,
    Gt,
}

/// Variable bounds tracker for the simplex-based theory.
#[derive(Debug, Clone)]
struct VarBounds {
    lower: Option<i64>,
    upper: Option<i64>,
    lower_reason: Option<Literal>,
    upper_reason: Option<Literal>,
}

impl VarBounds {
    fn new() -> Self {
        VarBounds {
            lower: None,
            upper: None,
            lower_reason: None,
            upper_reason: None,
        }
    }

    fn is_feasible(&self) -> bool {
        match (self.lower, self.upper) {
            (Some(lo), Some(hi)) => lo <= hi,
            _ => true,
        }
    }

    fn set_lower(&mut self, val: i64, reason: Literal) -> bool {
        match self.lower {
            Some(lo) if lo >= val => false,
            _ => {
                self.lower = Some(val);
                self.lower_reason = Some(reason);
                true
            }
        }
    }

    fn set_upper(&mut self, val: i64, reason: Literal) -> bool {
        match self.upper {
            Some(hi) if hi <= val => false,
            _ => {
                self.upper = Some(val);
                self.upper_reason = Some(reason);
                true
            }
        }
    }
}

/// Linear arithmetic theory solver using bound propagation and simplex feasibility checking.
#[derive(Debug, Clone)]
pub struct LinearArithmeticTheory {
    /// Constraints indexed by their guard literal's variable.
    constraints: Vec<LinearConstraint>,
    /// Bounds for each integer variable.
    bounds: Vec<VarBounds>,
    /// Number of integer variables.
    num_int_vars: u32,
    /// Propagation explanations cache: literal -> explaining literals.
    explanations: HashMap<Literal, Vec<Literal>>,
}

impl LinearArithmeticTheory {
    pub fn new(num_int_vars: u32) -> Self {
        LinearArithmeticTheory {
            constraints: Vec::new(),
            bounds: (0..num_int_vars).map(|_| VarBounds::new()).collect(),
            num_int_vars,
            explanations: HashMap::new(),
        }
    }

    /// Add a linear constraint.
    pub fn add_constraint(&mut self, constraint: LinearConstraint) {
        self.constraints.push(constraint);
    }

    /// Reset all bounds.
    pub fn reset_bounds(&mut self) {
        for b in &mut self.bounds {
            *b = VarBounds::new();
        }
        self.explanations.clear();
    }

    /// Try to update bounds from a single active constraint.
    /// Returns theory lemmas if bounds become infeasible or if propagation is possible.
    fn process_constraint(
        &mut self,
        idx: usize,
        assignment: &Assignment,
    ) -> Option<TheoryResult> {
        let constraint = &self.constraints[idx];

        // Check if the guard literal is assigned true.
        if assignment.eval_literal(constraint.guard) != Some(true) {
            return None;
        }

        // For single-variable constraints, directly update bounds.
        if constraint.coeffs.len() == 1 && constraint.int_vars.len() == 1 {
            let coeff = constraint.coeffs[0];
            let var_idx = constraint.int_vars[0] as usize;
            let rhs = constraint.rhs;

            if var_idx >= self.bounds.len() {
                return None;
            }

            match constraint.op {
                ComparisonOp::Le => {
                    // coeff * x <= rhs
                    if coeff > 0 {
                        // x <= rhs / coeff
                        let bound = rhs / coeff;
                        self.bounds[var_idx].set_upper(bound, constraint.guard);
                    } else if coeff < 0 {
                        // x >= rhs / coeff (division flips)
                        let bound = (rhs + (-coeff) - 1) / (-coeff); // ceiling division
                        self.bounds[var_idx].set_lower(bound, constraint.guard);
                    }
                }
                ComparisonOp::Ge => {
                    if coeff > 0 {
                        let bound = (rhs + coeff - 1) / coeff;
                        self.bounds[var_idx].set_lower(bound, constraint.guard);
                    } else if coeff < 0 {
                        let bound = rhs / (-coeff);
                        self.bounds[var_idx].set_upper(bound, constraint.guard);
                    }
                }
                ComparisonOp::Eq => {
                    if coeff != 0 {
                        if rhs % coeff != 0 {
                            // No integer solution: conflict.
                            return Some(TheoryResult::Inconsistent(vec![constraint.guard]));
                        }
                        let val = rhs / coeff;
                        self.bounds[var_idx].set_lower(val, constraint.guard);
                        self.bounds[var_idx].set_upper(val, constraint.guard);
                    }
                }
                ComparisonOp::Lt => {
                    if coeff > 0 {
                        let bound = (rhs - 1) / coeff;
                        self.bounds[var_idx].set_upper(bound, constraint.guard);
                    } else if coeff < 0 {
                        let bound = (rhs - 1 + (-coeff) - 1) / (-coeff) + 1;
                        self.bounds[var_idx].set_lower(bound, constraint.guard);
                    }
                }
                ComparisonOp::Gt => {
                    if coeff > 0 {
                        let bound = rhs / coeff + 1;
                        self.bounds[var_idx].set_lower(bound, constraint.guard);
                    } else if coeff < 0 {
                        let bound = (rhs) / (-coeff) - 1;
                        self.bounds[var_idx].set_upper(bound, constraint.guard);
                    }
                }
            }

            // Check feasibility.
            if !self.bounds[var_idx].is_feasible() {
                let mut explanation = vec![constraint.guard];
                if let Some(lo_reason) = self.bounds[var_idx].lower_reason {
                    if lo_reason != constraint.guard {
                        explanation.push(lo_reason);
                    }
                }
                if let Some(hi_reason) = self.bounds[var_idx].upper_reason {
                    if hi_reason != constraint.guard && !explanation.contains(&hi_reason) {
                        explanation.push(hi_reason);
                    }
                }
                return Some(TheoryResult::Inconsistent(explanation));
            }
        }

        None
    }

    /// Simplex-based feasibility check for multi-variable constraints.
    /// Simplified version: evaluate the constraint to check if it's satisfied
    /// given the current bounds.
    fn check_multi_var_feasibility(
        &self,
        constraint: &LinearConstraint,
    ) -> bool {
        // Compute min and max of the linear expression given current bounds.
        let mut min_val: i64 = 0;
        let mut max_val: i64 = 0;
        let mut all_bounded = true;

        for (i, &var_idx) in constraint.int_vars.iter().enumerate() {
            let coeff = constraint.coeffs[i];
            let idx = var_idx as usize;
            if idx >= self.bounds.len() {
                all_bounded = false;
                break;
            }
            let b = &self.bounds[idx];
            match (b.lower, b.upper) {
                (Some(lo), Some(hi)) => {
                    if coeff >= 0 {
                        min_val += coeff * lo;
                        max_val += coeff * hi;
                    } else {
                        min_val += coeff * hi;
                        max_val += coeff * lo;
                    }
                }
                _ => {
                    all_bounded = false;
                    break;
                }
            }
        }

        if !all_bounded {
            return true; // Can't determine infeasibility.
        }

        match constraint.op {
            ComparisonOp::Le => min_val <= constraint.rhs,
            ComparisonOp::Lt => min_val < constraint.rhs,
            ComparisonOp::Ge => max_val >= constraint.rhs,
            ComparisonOp::Gt => max_val > constraint.rhs,
            ComparisonOp::Eq => min_val <= constraint.rhs && max_val >= constraint.rhs,
        }
    }
}

impl TheoryPropagator for LinearArithmeticTheory {
    fn check_consistency(&mut self, assignment: &Assignment) -> TheoryResult {
        self.reset_bounds();

        for idx in 0..self.constraints.len() {
            if let Some(result) = self.process_constraint(idx, assignment) {
                return result;
            }
        }

        // Check multi-variable constraints.
        for constraint in &self.constraints {
            if assignment.eval_literal(constraint.guard) != Some(true) {
                continue;
            }
            if constraint.int_vars.len() > 1 && !self.check_multi_var_feasibility(constraint) {
                return TheoryResult::Inconsistent(vec![constraint.guard]);
            }
        }

        TheoryResult::Consistent
    }

    fn propagate(&mut self, assignment: &Assignment) -> Vec<TheoryLemma> {
        let mut lemmas = Vec::new();

        // For constraints whose guard is unassigned, check if the constraint
        // is already implied or contradicted by current bounds.
        for constraint in &self.constraints {
            if assignment.eval_literal(constraint.guard).is_some() {
                continue; // Already assigned.
            }
            if constraint.int_vars.len() == 1 && constraint.coeffs.len() == 1 {
                let coeff = constraint.coeffs[0];
                let var_idx = constraint.int_vars[0] as usize;
                if var_idx >= self.bounds.len() {
                    continue;
                }
                let b = &self.bounds[var_idx];

                // Check if bounds already satisfy the constraint.
                let implied = match constraint.op {
                    ComparisonOp::Le if coeff > 0 => {
                        b.upper.map_or(false, |hi| coeff * hi <= constraint.rhs)
                    }
                    ComparisonOp::Ge if coeff > 0 => {
                        b.lower.map_or(false, |lo| coeff * lo >= constraint.rhs)
                    }
                    _ => false,
                };

                if implied {
                    // The guard can be forced true.
                    let mut explanation = Vec::new();
                    if let Some(reason) = b.lower_reason {
                        explanation.push(reason);
                    }
                    if let Some(reason) = b.upper_reason {
                        if !explanation.contains(&reason) {
                            explanation.push(reason);
                        }
                    }
                    // Lemma: explanation → guard.
                    let mut lits: LiteralVec = explanation
                        .iter()
                        .map(|l| l.negated())
                        .collect();
                    lits.push(constraint.guard);
                    lemmas.push(TheoryLemma::new(lits, "LA bound propagation"));
                }
            }
        }

        lemmas
    }

    fn explain(&self, literal: Literal) -> Vec<Literal> {
        self.explanations
            .get(&literal)
            .cloned()
            .unwrap_or_default()
    }

    fn name(&self) -> &str {
        "LinearArithmetic"
    }
}

// ── EqualityTheory ────────────────────────────────────────────────────────────

/// Node in the union-find for congruence closure.
#[derive(Debug, Clone)]
struct UFNode {
    parent: u32,
    rank: u32,
}

/// Equality atom: guard literal activates (term_a == term_b) or (term_a != term_b).
#[derive(Debug, Clone)]
pub struct EqualityAtom {
    pub guard: Literal,
    pub term_a: u32,
    pub term_b: u32,
    pub is_equality: bool, // true = equality, false = disequality
}

/// Congruence closure for uninterpreted functions.
#[derive(Debug, Clone)]
pub struct EqualityTheory {
    nodes: Vec<UFNode>,
    num_terms: u32,
    atoms: Vec<EqualityAtom>,
    /// Trail for backtracking union-find operations.
    merge_trail: Vec<(u32, u32, u32)>, // (term, old_parent, old_rank)
    explanations: HashMap<Literal, Vec<Literal>>,
}

impl EqualityTheory {
    pub fn new(num_terms: u32) -> Self {
        let nodes = (0..num_terms)
            .map(|i| UFNode { parent: i, rank: 0 })
            .collect();
        EqualityTheory {
            nodes,
            num_terms,
            atoms: Vec::new(),
            merge_trail: Vec::new(),
            explanations: HashMap::new(),
        }
    }

    pub fn add_atom(&mut self, atom: EqualityAtom) {
        self.atoms.push(atom);
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.nodes[x as usize].parent != x {
            // Path halving.
            let p = self.nodes[x as usize].parent;
            let gp = self.nodes[p as usize].parent;
            self.nodes[x as usize].parent = gp;
            x = gp;
        }
        x
    }

    fn find_immutable(&self, mut x: u32) -> u32 {
        while self.nodes[x as usize].parent != x {
            x = self.nodes[x as usize].parent;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        // Union by rank.
        if self.nodes[ra as usize].rank < self.nodes[rb as usize].rank {
            self.merge_trail
                .push((ra, self.nodes[ra as usize].parent, self.nodes[ra as usize].rank));
            self.nodes[ra as usize].parent = rb;
        } else if self.nodes[ra as usize].rank > self.nodes[rb as usize].rank {
            self.merge_trail
                .push((rb, self.nodes[rb as usize].parent, self.nodes[rb as usize].rank));
            self.nodes[rb as usize].parent = ra;
        } else {
            self.merge_trail
                .push((rb, self.nodes[rb as usize].parent, self.nodes[rb as usize].rank));
            self.merge_trail
                .push((ra, self.nodes[ra as usize].parent, self.nodes[ra as usize].rank));
            self.nodes[rb as usize].parent = ra;
            self.nodes[ra as usize].rank += 1;
        }
        true
    }

    fn reset(&mut self) {
        for i in 0..self.num_terms {
            self.nodes[i as usize] = UFNode { parent: i, rank: 0 };
        }
        self.merge_trail.clear();
        self.explanations.clear();
    }
}

impl TheoryPropagator for EqualityTheory {
    fn check_consistency(&mut self, assignment: &Assignment) -> TheoryResult {
        self.reset();

        // First pass: process all equalities.
        let mut equality_guards = Vec::new();
        for atom in &self.atoms {
            if !atom.is_equality {
                continue;
            }
            if assignment.eval_literal(atom.guard) == Some(true) {
                equality_guards.push((atom.term_a, atom.term_b, atom.guard));
            }
        }

        for &(a, b, _guard) in &equality_guards {
            self.union(a, b);
        }

        // Second pass: check disequalities.
        for atom in &self.atoms {
            if atom.is_equality {
                continue;
            }
            if assignment.eval_literal(atom.guard) != Some(true) {
                continue;
            }
            let ra = self.find_immutable(atom.term_a);
            let rb = self.find_immutable(atom.term_b);
            if ra == rb {
                // Conflict: a = b but disequality says a ≠ b.
                let mut explanation = vec![atom.guard];
                // Collect equality guards that caused the merge.
                for &(a, b, guard) in &equality_guards {
                    let fa = self.find_immutable(a);
                    let fb = self.find_immutable(b);
                    // Simple heuristic: include all equalities in the explanation.
                    if fa == fb {
                        if !explanation.contains(&guard) {
                            explanation.push(guard);
                        }
                    }
                }
                return TheoryResult::Inconsistent(explanation);
            }
        }

        TheoryResult::Consistent
    }

    fn propagate(&mut self, assignment: &Assignment) -> Vec<TheoryLemma> {
        let mut lemmas = Vec::new();

        // Check if any unassigned equality/disequality can be deduced.
        for atom in &self.atoms {
            if assignment.eval_literal(atom.guard).is_some() {
                continue;
            }
            let ra = self.find_immutable(atom.term_a);
            let rb = self.find_immutable(atom.term_b);

            if atom.is_equality && ra == rb {
                // The terms are already equal; force the guard true.
                lemmas.push(TheoryLemma::new(
                    smallvec![atom.guard],
                    "equality propagation",
                ));
            } else if !atom.is_equality && ra == rb {
                // The terms are equal, but this is a disequality: force guard false.
                lemmas.push(TheoryLemma::new(
                    smallvec![atom.guard.negated()],
                    "disequality conflict propagation",
                ));
            }
        }

        lemmas
    }

    fn explain(&self, literal: Literal) -> Vec<Literal> {
        self.explanations
            .get(&literal)
            .cloned()
            .unwrap_or_default()
    }

    fn name(&self) -> &str {
        "Equality"
    }
}

// ── CombinedTheory ────────────────────────────────────────────────────────────

/// Nelson-Oppen style theory combination.
#[derive(Debug)]
pub struct CombinedTheory {
    theories: Vec<Box<dyn TheoryPropagator>>,
}

impl CombinedTheory {
    pub fn new() -> Self {
        CombinedTheory {
            theories: Vec::new(),
        }
    }

    pub fn add_theory(&mut self, theory: Box<dyn TheoryPropagator>) {
        self.theories.push(theory);
    }
}

impl Default for CombinedTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl TheoryPropagator for CombinedTheory {
    fn check_consistency(&mut self, assignment: &Assignment) -> TheoryResult {
        // Check each theory independently (simplified Nelson-Oppen).
        for theory in &mut self.theories {
            let result = theory.check_consistency(assignment);
            match result {
                TheoryResult::Consistent => continue,
                other => return other,
            }
        }
        TheoryResult::Consistent
    }

    fn propagate(&mut self, assignment: &Assignment) -> Vec<TheoryLemma> {
        let mut all_lemmas = Vec::new();
        // Run a fixed-point loop: propagate until no new lemmas.
        let mut changed = true;
        let mut iterations = 0;
        while changed && iterations < 10 {
            changed = false;
            iterations += 1;
            for theory in &mut self.theories {
                let lemmas = theory.propagate(assignment);
                if !lemmas.is_empty() {
                    changed = true;
                    all_lemmas.extend(lemmas);
                }
            }
        }
        all_lemmas
    }

    fn explain(&self, literal: Literal) -> Vec<Literal> {
        for theory in &self.theories {
            let explanation = theory.explain(literal);
            if !explanation.is_empty() {
                return explanation;
            }
        }
        Vec::new()
    }

    fn on_assign(&mut self, var: Variable, value: bool) {
        for theory in &mut self.theories {
            theory.on_assign(var, value);
        }
    }

    fn on_unassign(&mut self, var: Variable) {
        for theory in &mut self.theories {
            theory.on_unassign(var);
        }
    }

    fn name(&self) -> &str {
        "Combined"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::{Reason, Variable};

    fn lit(v: i32) -> Literal {
        Literal::from_dimacs(v)
    }

    #[test]
    fn test_la_theory_consistent() {
        let mut la = LinearArithmeticTheory::new(2);
        // x0 <= 10, guarded by literal 1.
        la.add_constraint(LinearConstraint {
            guard: lit(1),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 10,
            op: ComparisonOp::Le,
        });
        // x0 >= 5, guarded by literal 2.
        la.add_constraint(LinearConstraint {
            guard: lit(2),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 5,
            op: ComparisonOp::Ge,
        });

        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(2), true, 0, Reason::Decision);

        let result = la.check_consistency(&asgn);
        assert!(matches!(result, TheoryResult::Consistent));
    }

    #[test]
    fn test_la_theory_inconsistent() {
        let mut la = LinearArithmeticTheory::new(1);
        // x0 <= 3, guarded by literal 1.
        la.add_constraint(LinearConstraint {
            guard: lit(1),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 3,
            op: ComparisonOp::Le,
        });
        // x0 >= 5, guarded by literal 2.
        la.add_constraint(LinearConstraint {
            guard: lit(2),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 5,
            op: ComparisonOp::Ge,
        });

        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(2), true, 0, Reason::Decision);

        let result = la.check_consistency(&asgn);
        assert!(matches!(result, TheoryResult::Inconsistent(_)));
    }

    #[test]
    fn test_la_theory_inactive_guard() {
        let mut la = LinearArithmeticTheory::new(1);
        la.add_constraint(LinearConstraint {
            guard: lit(1),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 3,
            op: ComparisonOp::Le,
        });
        la.add_constraint(LinearConstraint {
            guard: lit(2),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 5,
            op: ComparisonOp::Ge,
        });

        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        // Literal 2 is NOT assigned → constraint x0>=5 inactive.

        let result = la.check_consistency(&asgn);
        assert!(matches!(result, TheoryResult::Consistent));
    }

    #[test]
    fn test_la_equality_constraint() {
        let mut la = LinearArithmeticTheory::new(1);
        // x0 = 7, guarded by literal 1.
        la.add_constraint(LinearConstraint {
            guard: lit(1),
            coeffs: vec![2],
            int_vars: vec![0],
            rhs: 7,
            op: ComparisonOp::Eq,
        });

        let mut asgn = Assignment::new(1);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);

        // 2 * x0 = 7 has no integer solution → inconsistent.
        let result = la.check_consistency(&asgn);
        assert!(matches!(result, TheoryResult::Inconsistent(_)));
    }

    #[test]
    fn test_equality_theory_consistent() {
        let mut eq = EqualityTheory::new(3);
        // a = b (guard: lit(1)), b = c (guard: lit(2))
        eq.add_atom(EqualityAtom {
            guard: lit(1),
            term_a: 0,
            term_b: 1,
            is_equality: true,
        });
        eq.add_atom(EqualityAtom {
            guard: lit(2),
            term_a: 1,
            term_b: 2,
            is_equality: true,
        });

        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(2), true, 0, Reason::Decision);

        let result = eq.check_consistency(&asgn);
        assert!(matches!(result, TheoryResult::Consistent));
    }

    #[test]
    fn test_equality_theory_inconsistent() {
        let mut eq = EqualityTheory::new(3);
        // a = b (guard: lit(1)), b = c (guard: lit(2)), a ≠ c (guard: lit(3))
        eq.add_atom(EqualityAtom {
            guard: lit(1),
            term_a: 0,
            term_b: 1,
            is_equality: true,
        });
        eq.add_atom(EqualityAtom {
            guard: lit(2),
            term_a: 1,
            term_b: 2,
            is_equality: true,
        });
        eq.add_atom(EqualityAtom {
            guard: lit(3),
            term_a: 0,
            term_b: 2,
            is_equality: false,
        });

        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(2), true, 0, Reason::Decision);
        asgn.set(Variable::new(3), true, 0, Reason::Decision);

        let result = eq.check_consistency(&asgn);
        assert!(matches!(result, TheoryResult::Inconsistent(_)));
    }

    #[test]
    fn test_combined_theory() {
        let la = LinearArithmeticTheory::new(2);
        let eq = EqualityTheory::new(2);

        let mut combined = CombinedTheory::new();
        combined.add_theory(Box::new(la));
        combined.add_theory(Box::new(eq));

        let asgn = Assignment::new(0);
        let result = combined.check_consistency(&asgn);
        assert!(matches!(result, TheoryResult::Consistent));
    }

    #[test]
    fn test_la_theory_propagate() {
        let mut la = LinearArithmeticTheory::new(1);
        // x0 <= 3 guarded by lit(1) (assigned true).
        la.add_constraint(LinearConstraint {
            guard: lit(1),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 3,
            op: ComparisonOp::Le,
        });
        // x0 <= 10 guarded by lit(2) (unassigned); should be propagated since x0 <= 3 implies x0 <= 10.
        la.add_constraint(LinearConstraint {
            guard: lit(2),
            coeffs: vec![1],
            int_vars: vec![0],
            rhs: 10,
            op: ComparisonOp::Le,
        });

        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);

        // First check consistency to set up bounds.
        let _ = la.check_consistency(&asgn);
        let lemmas = la.propagate(&asgn);
        // The propagation should suggest that lit(2) is implied.
        assert!(
            lemmas.iter().any(|l| l.literals.contains(&lit(2))),
            "Expected propagation of lit(2)"
        );
    }

    #[test]
    fn test_equality_propagation() {
        let mut eq = EqualityTheory::new(3);
        eq.add_atom(EqualityAtom {
            guard: lit(1),
            term_a: 0,
            term_b: 1,
            is_equality: true,
        });
        eq.add_atom(EqualityAtom {
            guard: lit(2),
            term_a: 1,
            term_b: 2,
            is_equality: true,
        });
        // a = c (guard: lit(3)) — should be propagated after a=b and b=c.
        eq.add_atom(EqualityAtom {
            guard: lit(3),
            term_a: 0,
            term_b: 2,
            is_equality: true,
        });

        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(2), true, 0, Reason::Decision);

        let _ = eq.check_consistency(&asgn);
        let lemmas = eq.propagate(&asgn);
        assert!(
            lemmas.iter().any(|l| l.literals.contains(&lit(3))),
            "Expected propagation of lit(3)"
        );
    }

    #[test]
    fn test_theory_lemma_creation() {
        let lemma = TheoryLemma::new(smallvec![lit(1), lit(-2)], "test");
        assert_eq!(lemma.literals.len(), 2);
        assert_eq!(lemma.source, "test");
    }

    #[test]
    fn test_combined_theory_default() {
        let combined = CombinedTheory::default();
        assert_eq!(combined.name(), "Combined");
    }
}
