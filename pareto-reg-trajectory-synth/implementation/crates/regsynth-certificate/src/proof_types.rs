//! Proof data structures used across all certificate types.
//!
//! Provides resolution steps/proofs, satisfaction witnesses, dominance proofs,
//! proof-node trees, and utilities for validation and compaction.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ─── Identifiers ────────────────────────────────────────────────────────────

pub type ClauseId = usize;

// ─── Literals & Clauses ─────────────────────────────────────────────────────

/// A propositional literal: a variable name with polarity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Literal {
    Pos(String),
    Neg(String),
}

impl Literal {
    pub fn var(&self) -> &str {
        match self {
            Literal::Pos(v) | Literal::Neg(v) => v,
        }
    }

    pub fn is_positive(&self) -> bool {
        matches!(self, Literal::Pos(_))
    }

    pub fn negate(&self) -> Literal {
        match self {
            Literal::Pos(v) => Literal::Neg(v.clone()),
            Literal::Neg(v) => Literal::Pos(v.clone()),
        }
    }

    /// Evaluate the literal under a Boolean assignment.
    pub fn evaluate(&self, assignment: &HashMap<String, bool>) -> Option<bool> {
        assignment.get(self.var()).map(|&v| {
            if self.is_positive() { v } else { !v }
        })
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Pos(v) => write!(f, "{}", v),
            Literal::Neg(v) => write!(f, "¬{}", v),
        }
    }
}

/// A disjunctive clause (set of literals).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    pub fn empty() -> Self {
        Self { literals: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }

    pub fn variables(&self) -> HashSet<&str> {
        self.literals.iter().map(|l| l.var()).collect()
    }

    /// Evaluate the clause under a Boolean assignment.
    /// Returns `Some(true)` if any literal is satisfied, `Some(false)` if
    /// all are falsified, `None` if some are unassigned.
    pub fn evaluate(&self, assignment: &HashMap<String, bool>) -> Option<bool> {
        if self.is_empty() {
            return Some(false);
        }
        let mut all_evaluated = true;
        for lit in &self.literals {
            match lit.evaluate(assignment) {
                Some(true) => return Some(true),
                None => all_evaluated = false,
                Some(false) => {}
            }
        }
        if all_evaluated { Some(false) } else { None }
    }

    /// Resolve this clause with `other` on the pivot variable.
    /// Returns `None` if no complementary pivot exists.
    pub fn resolve(&self, other: &Clause, pivot: &str) -> Option<Clause> {
        let self_pos = self.literals.iter().any(|l| l.var() == pivot && l.is_positive());
        let self_neg = self.literals.iter().any(|l| l.var() == pivot && !l.is_positive());
        let other_pos = other.literals.iter().any(|l| l.var() == pivot && l.is_positive());
        let other_neg = other.literals.iter().any(|l| l.var() == pivot && !l.is_positive());

        let complementary = (self_pos && other_neg) || (self_neg && other_pos);
        if !complementary {
            return None;
        }

        let mut seen = HashSet::new();
        let mut result_lits = Vec::new();
        for lit in self.literals.iter().chain(other.literals.iter()) {
            if lit.var() == pivot {
                continue;
            }
            let key = format!("{}:{}", lit.var(), lit.is_positive());
            if seen.insert(key) {
                result_lits.push(lit.clone());
            }
        }
        Some(Clause::new(result_lits))
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "⊥");
        }
        let parts: Vec<String> = self.literals.iter().map(|l| format!("{}", l)).collect();
        write!(f, "({})", parts.join(" ∨ "))
    }
}

// ─── Resolution Step ────────────────────────────────────────────────────────

/// A single resolution step in a proof: resolve clause `parent1` with
/// `parent2` on `pivot_variable` to produce `resolvent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStep {
    pub step_id: ClauseId,
    pub parent1: ClauseId,
    pub parent2: ClauseId,
    pub pivot_variable: String,
    pub resolvent: Clause,
}

impl ResolutionStep {
    /// Validate that the resolvent is the correct result of resolving
    /// the two parent clauses on the pivot.
    pub fn validate(&self, clause_store: &HashMap<ClauseId, Clause>) -> bool {
        let p1 = match clause_store.get(&self.parent1) {
            Some(c) => c,
            None => return false,
        };
        let p2 = match clause_store.get(&self.parent2) {
            Some(c) => c,
            None => return false,
        };
        match p1.resolve(p2, &self.pivot_variable) {
            Some(expected) => {
                let mut expected_set: Vec<String> =
                    expected.literals.iter().map(|l| format!("{}", l)).collect();
                expected_set.sort();
                let mut actual_set: Vec<String> =
                    self.resolvent.literals.iter().map(|l| format!("{}", l)).collect();
                actual_set.sort();
                expected_set == actual_set
            }
            None => false,
        }
    }
}

// ─── Resolution Proof ───────────────────────────────────────────────────────

/// A complete resolution proof of unsatisfiability.
///
/// Starts from `initial_clauses`, applies `steps`, and derives the empty
/// clause at `empty_clause_step`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionProof {
    pub initial_clauses: Vec<(ClauseId, Clause)>,
    pub steps: Vec<ResolutionStep>,
    pub empty_clause_step: Option<ClauseId>,
    pub mus_constraint_ids: Vec<String>,
}

impl ResolutionProof {
    pub fn new() -> Self {
        Self {
            initial_clauses: Vec::new(),
            steps: Vec::new(),
            empty_clause_step: None,
            mus_constraint_ids: Vec::new(),
        }
    }

    pub fn add_initial_clause(&mut self, id: ClauseId, clause: Clause) {
        self.initial_clauses.push((id, clause));
    }

    /// Add a resolution step and return its id.
    pub fn add_step(
        &mut self,
        parent1: ClauseId,
        parent2: ClauseId,
        pivot: &str,
        resolvent: Clause,
    ) -> ClauseId {
        let step_id = self.initial_clauses.len() + self.steps.len();
        let is_empty = resolvent.is_empty();
        self.steps.push(ResolutionStep {
            step_id,
            parent1,
            parent2,
            pivot_variable: pivot.to_string(),
            resolvent,
        });
        if is_empty {
            self.empty_clause_step = Some(step_id);
        }
        step_id
    }

    pub fn is_complete(&self) -> bool {
        self.empty_clause_step.is_some()
    }

    pub fn proof_length(&self) -> usize {
        self.steps.len()
    }

    /// Build a clause store mapping clause ids to their clauses.
    fn clause_store(&self) -> HashMap<ClauseId, Clause> {
        let mut store: HashMap<ClauseId, Clause> = HashMap::new();
        for (id, clause) in &self.initial_clauses {
            store.insert(*id, clause.clone());
        }
        for step in &self.steps {
            store.insert(step.step_id, step.resolvent.clone());
        }
        store
    }

    /// Validate the entire resolution proof chain.
    pub fn validate(&self) -> bool {
        if !self.is_complete() {
            return false;
        }
        let store = self.clause_store();
        for step in &self.steps {
            if step.parent1 >= step.step_id || step.parent2 >= step.step_id {
                return false;
            }
            if !step.validate(&store) {
                return false;
            }
        }
        if let Some(empty_id) = self.empty_clause_step {
            if let Some(c) = store.get(&empty_id) {
                return c.is_empty();
            }
            return false;
        }
        false
    }

    /// Validate a single step given the clause store.
    pub fn validate_step(&self, step_idx: usize) -> bool {
        if step_idx >= self.steps.len() {
            return false;
        }
        let store = self.clause_store();
        self.steps[step_idx].validate(&store)
    }

    /// Validate the full chain and return per-step results.
    pub fn validate_chain(&self) -> Vec<bool> {
        let store = self.clause_store();
        self.steps.iter().map(|s| s.validate(&store)).collect()
    }

    /// Total size of the proof in literals.
    pub fn proof_size(&self) -> usize {
        let initial: usize = self.initial_clauses.iter().map(|(_, c)| c.len()).sum();
        let steps: usize = self.steps.iter().map(|s| s.resolvent.len()).sum();
        initial + steps
    }

    /// Remove redundant steps that are not ancestors of the empty clause.
    pub fn compact(&mut self) {
        if self.empty_clause_step.is_none() {
            return;
        }
        let target = self.empty_clause_step.unwrap();
        let mut needed: HashSet<ClauseId> = HashSet::new();
        needed.insert(target);

        // Walk backwards collecting ancestors
        for step in self.steps.iter().rev() {
            if needed.contains(&step.step_id) {
                needed.insert(step.parent1);
                needed.insert(step.parent2);
            }
        }

        self.steps.retain(|s| needed.contains(&s.step_id));
        self.initial_clauses
            .retain(|(id, _)| needed.contains(id));
    }
}

impl Default for ResolutionProof {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Satisfaction Witness ───────────────────────────────────────────────────

/// A value in a satisfaction witness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WitnessValue {
    Bool(bool),
    Real(f64),
    Int(i64),
}

impl fmt::Display for WitnessValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WitnessValue::Bool(b) => write!(f, "{}", b),
            WitnessValue::Real(v) => write!(f, "{:.6}", v),
            WitnessValue::Int(v) => write!(f, "{}", v),
        }
    }
}

/// A witness that demonstrates constraint satisfaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionWitness {
    pub variable_values: HashMap<String, WitnessValue>,
    pub constraint_results: HashMap<String, bool>,
}

impl SatisfactionWitness {
    pub fn new() -> Self {
        Self {
            variable_values: HashMap::new(),
            constraint_results: HashMap::new(),
        }
    }

    pub fn set_bool(&mut self, var: &str, val: bool) {
        self.variable_values
            .insert(var.to_string(), WitnessValue::Bool(val));
    }

    pub fn set_real(&mut self, var: &str, val: f64) {
        self.variable_values
            .insert(var.to_string(), WitnessValue::Real(val));
    }

    pub fn set_int(&mut self, var: &str, val: i64) {
        self.variable_values
            .insert(var.to_string(), WitnessValue::Int(val));
    }

    pub fn record_constraint(&mut self, id: &str, satisfied: bool) {
        self.constraint_results.insert(id.to_string(), satisfied);
    }

    pub fn all_satisfied(&self) -> bool {
        self.constraint_results.values().all(|&v| v)
    }

    pub fn satisfaction_ratio(&self) -> f64 {
        if self.constraint_results.is_empty() {
            return 1.0;
        }
        let sat = self.constraint_results.values().filter(|&&v| v).count();
        sat as f64 / self.constraint_results.len() as f64
    }

    /// Extract a Boolean assignment from the witness.
    pub fn bool_assignment(&self) -> HashMap<String, bool> {
        self.variable_values
            .iter()
            .filter_map(|(k, v)| match v {
                WitnessValue::Bool(b) => Some((k.clone(), *b)),
                _ => None,
            })
            .collect()
    }

    /// Extract a real-valued assignment from the witness.
    pub fn real_assignment(&self) -> HashMap<String, f64> {
        self.variable_values
            .iter()
            .filter_map(|(k, v)| match v {
                WitnessValue::Real(r) => Some((k.clone(), *r)),
                WitnessValue::Int(i) => Some((k.clone(), *i as f64)),
                _ => None,
            })
            .collect()
    }
}

impl Default for SatisfactionWitness {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Dominance Proof ────────────────────────────────────────────────────────

/// Proof that no feasible point can improve on a Pareto point in a given
/// dimension without worsening another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionInfeasibilityProof {
    pub dimension_name: String,
    pub dimension_index: usize,
    pub current_value: f64,
    pub proven_lower_bound: f64,
    pub proof_method: String,
    pub witness_constraints: Vec<String>,
}

impl DimensionInfeasibilityProof {
    /// Returns true if the proof is valid: the lower bound is at or above
    /// the current value (meaning no improvement is possible).
    pub fn is_valid(&self) -> bool {
        self.proven_lower_bound >= self.current_value - 1e-9
    }
}

/// Proof that a point on the Pareto frontier is non-dominated: for each
/// dimension, we prove that any feasible point with a strictly better value
/// in that dimension must be worse in at least one other dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominanceProof {
    pub point_costs: Vec<f64>,
    pub dimension_proofs: Vec<DimensionInfeasibilityProof>,
    pub strategy_assignment: HashMap<String, bool>,
}

impl DominanceProof {
    pub fn new(costs: Vec<f64>) -> Self {
        Self {
            point_costs: costs,
            dimension_proofs: Vec::new(),
            strategy_assignment: HashMap::new(),
        }
    }

    pub fn add_dimension_proof(&mut self, proof: DimensionInfeasibilityProof) {
        self.dimension_proofs.push(proof);
    }

    /// A dominance proof is complete if every dimension has a proof.
    pub fn is_complete(&self) -> bool {
        !self.point_costs.is_empty()
            && self.dimension_proofs.len() >= self.point_costs.len()
    }

    /// Validate all dimension proofs.
    pub fn validate(&self) -> bool {
        if !self.is_complete() {
            return false;
        }
        self.dimension_proofs.iter().all(|dp| dp.is_valid())
    }
}

// ─── Proof Node Tree ────────────────────────────────────────────────────────

/// A tree-structured proof node for hierarchical proof representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    pub label: String,
    pub description: String,
    pub children: Vec<ProofNode>,
    pub is_leaf: bool,
    pub valid: Option<bool>,
}

impl ProofNode {
    pub fn leaf(label: &str, description: &str, valid: bool) -> Self {
        Self {
            label: label.to_string(),
            description: description.to_string(),
            children: Vec::new(),
            is_leaf: true,
            valid: Some(valid),
        }
    }

    pub fn internal(label: &str, description: &str, children: Vec<ProofNode>) -> Self {
        let all_valid = children.iter().all(|c| c.is_valid());
        Self {
            label: label.to_string(),
            description: description.to_string(),
            children,
            is_leaf: false,
            valid: Some(all_valid),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.valid.unwrap_or(false)
    }

    /// Count the total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Maximum depth of the tree.
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Collect all leaf labels.
    pub fn leaf_labels(&self) -> Vec<&str> {
        if self.is_leaf {
            vec![&self.label]
        } else {
            self.children
                .iter()
                .flat_map(|c| c.leaf_labels())
                .collect()
        }
    }
}

impl fmt::Display for ProofNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indented(f, 0)
    }
}

impl ProofNode {
    fn fmt_indented(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let prefix = "  ".repeat(indent);
        let status = match self.valid {
            Some(true) => "✓",
            Some(false) => "✗",
            None => "?",
        };
        writeln!(f, "{}[{}] {} — {}", prefix, status, self.label, self.description)?;
        for child in &self.children {
            child.fmt_indented(f, indent + 1)?;
        }
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_evaluate() {
        let mut env = HashMap::new();
        env.insert("x".to_string(), true);
        env.insert("y".to_string(), false);

        assert_eq!(Literal::Pos("x".into()).evaluate(&env), Some(true));
        assert_eq!(Literal::Neg("x".into()).evaluate(&env), Some(false));
        assert_eq!(Literal::Pos("y".into()).evaluate(&env), Some(false));
        assert_eq!(Literal::Neg("y".into()).evaluate(&env), Some(true));
        assert_eq!(Literal::Pos("z".into()).evaluate(&env), None);
    }

    #[test]
    fn clause_resolve() {
        let c1 = Clause::new(vec![Literal::Pos("x".into()), Literal::Pos("y".into())]);
        let c2 = Clause::new(vec![Literal::Neg("x".into()), Literal::Pos("z".into())]);
        let resolved = c1.resolve(&c2, "x").unwrap();
        assert_eq!(resolved.len(), 2);
        assert!(resolved.literals.iter().any(|l| l.var() == "y"));
        assert!(resolved.literals.iter().any(|l| l.var() == "z"));
    }

    #[test]
    fn clause_resolve_to_empty() {
        let c1 = Clause::new(vec![Literal::Pos("x".into())]);
        let c2 = Clause::new(vec![Literal::Neg("x".into())]);
        let resolved = c1.resolve(&c2, "x").unwrap();
        assert!(resolved.is_empty());
    }

    #[test]
    fn clause_resolve_no_pivot() {
        let c1 = Clause::new(vec![Literal::Pos("x".into())]);
        let c2 = Clause::new(vec![Literal::Pos("y".into())]);
        assert!(c1.resolve(&c2, "x").is_none());
    }

    #[test]
    fn resolution_proof_build_and_validate() {
        let mut proof = ResolutionProof::new();
        // {x, y}
        proof.add_initial_clause(0, Clause::new(vec![
            Literal::Pos("x".into()), Literal::Pos("y".into()),
        ]));
        // {¬x, y}
        proof.add_initial_clause(1, Clause::new(vec![
            Literal::Neg("x".into()), Literal::Pos("y".into()),
        ]));
        // {x, ¬y}
        proof.add_initial_clause(2, Clause::new(vec![
            Literal::Pos("x".into()), Literal::Neg("y".into()),
        ]));
        // {¬x, ¬y}
        proof.add_initial_clause(3, Clause::new(vec![
            Literal::Neg("x".into()), Literal::Neg("y".into()),
        ]));

        // Resolve 0,1 on x -> {y}
        let s4 = proof.add_step(0, 1, "x", Clause::new(vec![Literal::Pos("y".into())]));
        assert_eq!(s4, 4);

        // Resolve 2,3 on x -> {¬y}
        let s5 = proof.add_step(2, 3, "x", Clause::new(vec![Literal::Neg("y".into())]));
        assert_eq!(s5, 5);

        // Resolve 4,5 on y -> {}
        let s6 = proof.add_step(4, 5, "y", Clause::empty());
        assert_eq!(s6, 6);

        assert!(proof.is_complete());
        assert!(proof.validate());
        assert_eq!(proof.proof_length(), 3);
    }

    #[test]
    fn resolution_proof_compact() {
        let mut proof = ResolutionProof::new();
        proof.add_initial_clause(0, Clause::new(vec![Literal::Pos("a".into())]));
        proof.add_initial_clause(1, Clause::new(vec![Literal::Neg("a".into())]));
        // Unused clause
        proof.add_initial_clause(2, Clause::new(vec![Literal::Pos("b".into())]));

        proof.add_step(0, 1, "a", Clause::empty());
        assert_eq!(proof.initial_clauses.len(), 3);
        proof.compact();
        // Clause 2 should be removed
        assert_eq!(proof.initial_clauses.len(), 2);
    }

    #[test]
    fn satisfaction_witness_basics() {
        let mut w = SatisfactionWitness::new();
        w.set_bool("x", true);
        w.set_real("cost", 100.5);
        w.record_constraint("c1", true);
        w.record_constraint("c2", true);
        assert!(w.all_satisfied());
        assert!((w.satisfaction_ratio() - 1.0).abs() < 1e-10);

        w.record_constraint("c3", false);
        assert!(!w.all_satisfied());
        assert!((w.satisfaction_ratio() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn dominance_proof_validation() {
        let mut dp = DominanceProof::new(vec![1.0, 2.0]);
        assert!(!dp.is_complete());
        dp.add_dimension_proof(DimensionInfeasibilityProof {
            dimension_name: "cost".into(),
            dimension_index: 0,
            current_value: 1.0,
            proven_lower_bound: 1.0,
            proof_method: "ilp_bound".into(),
            witness_constraints: vec!["c1".into()],
        });
        dp.add_dimension_proof(DimensionInfeasibilityProof {
            dimension_name: "time".into(),
            dimension_index: 1,
            current_value: 2.0,
            proven_lower_bound: 2.5,
            proof_method: "ilp_bound".into(),
            witness_constraints: vec!["c2".into()],
        });
        assert!(dp.is_complete());
        assert!(dp.validate());
    }

    #[test]
    fn proof_node_tree() {
        let leaf1 = ProofNode::leaf("step1", "constraint ok", true);
        let leaf2 = ProofNode::leaf("step2", "bound ok", true);
        let root = ProofNode::internal("root", "all checks", vec![leaf1, leaf2]);
        assert!(root.is_valid());
        assert_eq!(root.node_count(), 3);
        assert_eq!(root.depth(), 2);
        assert_eq!(root.leaf_labels(), vec!["step1", "step2"]);
    }

    #[test]
    fn proof_node_invalid_child() {
        let leaf1 = ProofNode::leaf("ok", "good", true);
        let leaf2 = ProofNode::leaf("bad", "fail", false);
        let root = ProofNode::internal("root", "checks", vec![leaf1, leaf2]);
        assert!(!root.is_valid());
    }

    #[test]
    fn clause_evaluate() {
        let mut env = HashMap::new();
        env.insert("x".to_string(), false);
        env.insert("y".to_string(), true);

        let c = Clause::new(vec![Literal::Pos("x".into()), Literal::Pos("y".into())]);
        assert_eq!(c.evaluate(&env), Some(true));

        let c2 = Clause::new(vec![Literal::Pos("x".into()), Literal::Neg("y".into())]);
        assert_eq!(c2.evaluate(&env), Some(false));

        let empty = Clause::empty();
        assert_eq!(empty.evaluate(&env), Some(false));
    }

    #[test]
    fn proof_size() {
        let mut proof = ResolutionProof::new();
        proof.add_initial_clause(0, Clause::new(vec![Literal::Pos("x".into()), Literal::Pos("y".into())]));
        proof.add_initial_clause(1, Clause::new(vec![Literal::Neg("x".into())]));
        proof.add_step(0, 1, "x", Clause::new(vec![Literal::Pos("y".into())]));
        assert_eq!(proof.proof_size(), 4); // 2 + 1 + 1
    }
}
