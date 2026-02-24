//! Axiomatic memory model framework.
//!
//! Provides formal axiom specifications, evaluation against execution graphs,
//! consistency checking, entailment testing, and model comparison.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use super::execution::{BitMatrix, ExecutionGraph, Event, EventId, OpType};
use super::memory_model::{
    MemoryModel, RelationExpr, PredicateExpr, Constraint,
    BuiltinModel, DerivedRelation,
};

// ---------------------------------------------------------------------------
// Axiom — formal axiom specification
// ---------------------------------------------------------------------------

/// A formal axiom in a memory model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Axiom {
    /// The relation must be acyclic.
    Acyclic {
        name: String,
        relation: RelationExpr,
    },
    /// The relation must be irreflexive.
    Irreflexive {
        name: String,
        relation: RelationExpr,
    },
    /// The relation must be empty.
    Empty {
        name: String,
        relation: RelationExpr,
    },
    /// The relation must be total on matching events.
    Total {
        name: String,
        relation: RelationExpr,
        filter: PredicateExpr,
    },
    /// Custom: a first-order-logic formula over relations.
    Custom {
        name: String,
        formula: FOLFormula,
    },
}

impl Axiom {
    pub fn acyclic(name: &str, rel: RelationExpr) -> Self {
        Self::Acyclic { name: name.to_string(), relation: rel }
    }

    pub fn irreflexive(name: &str, rel: RelationExpr) -> Self {
        Self::Irreflexive { name: name.to_string(), relation: rel }
    }

    pub fn empty(name: &str, rel: RelationExpr) -> Self {
        Self::Empty { name: name.to_string(), relation: rel }
    }

    pub fn total(name: &str, rel: RelationExpr, filter: PredicateExpr) -> Self {
        Self::Total { name: name.to_string(), relation: rel, filter }
    }

    pub fn custom(name: &str, formula: FOLFormula) -> Self {
        Self::Custom { name: name.to_string(), formula }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Acyclic { name, .. } |
            Self::Irreflexive { name, .. } |
            Self::Empty { name, .. } |
            Self::Total { name, .. } |
            Self::Custom { name, .. } => name,
        }
    }
}

impl fmt::Display for Axiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acyclic { name, relation } =>
                write!(f, "axiom {} : acyclic({})", name, relation),
            Self::Irreflexive { name, relation } =>
                write!(f, "axiom {} : irreflexive({})", name, relation),
            Self::Empty { name, relation } =>
                write!(f, "axiom {} : empty({})", name, relation),
            Self::Total { name, relation, filter } =>
                write!(f, "axiom {} : total({}) over [{}]", name, relation, filter),
            Self::Custom { name, formula } =>
                write!(f, "axiom {} : {}", name, formula),
        }
    }
}

// ---------------------------------------------------------------------------
// FOLFormula — first-order logic over relations
// ---------------------------------------------------------------------------

/// First-order logic formula over relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FOLFormula {
    /// (x, y) ∈ R
    InRelation {
        relation: RelationExpr,
    },
    /// ∀ x. φ(x)
    ForAll {
        var: String,
        body: Box<FOLFormula>,
    },
    /// ∃ x. φ(x)
    Exists {
        var: String,
        body: Box<FOLFormula>,
    },
    /// φ ∧ ψ
    And(Box<FOLFormula>, Box<FOLFormula>),
    /// φ ∨ ψ
    Or(Box<FOLFormula>, Box<FOLFormula>),
    /// ¬φ
    Not(Box<FOLFormula>),
    /// φ → ψ
    Implies(Box<FOLFormula>, Box<FOLFormula>),
    /// Relation R is acyclic.
    IsAcyclic(RelationExpr),
    /// Relation R is irreflexive.
    IsIrreflexive(RelationExpr),
    /// Relation R is empty.
    IsEmpty(RelationExpr),
    /// Relation R is total over events matching predicate.
    IsTotal(RelationExpr, PredicateExpr),
    /// Relation R1 is a subset of R2.
    Subset(RelationExpr, RelationExpr),
    /// True constant.
    True,
    /// False constant.
    False,
}

impl FOLFormula {
    pub fn and(a: Self, b: Self) -> Self { Self::And(Box::new(a), Box::new(b)) }
    pub fn or(a: Self, b: Self) -> Self { Self::Or(Box::new(a), Box::new(b)) }
    pub fn not(a: Self) -> Self { Self::Not(Box::new(a)) }
    pub fn implies(a: Self, b: Self) -> Self { Self::Implies(Box::new(a), Box::new(b)) }
    pub fn for_all(var: &str, body: Self) -> Self {
        Self::ForAll { var: var.to_string(), body: Box::new(body) }
    }
    pub fn exists(var: &str, body: Self) -> Self {
        Self::Exists { var: var.to_string(), body: Box::new(body) }
    }
}

impl fmt::Display for FOLFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InRelation { relation } => write!(f, "∈ {}", relation),
            Self::ForAll { var, body } => write!(f, "∀{}. {}", var, body),
            Self::Exists { var, body } => write!(f, "∃{}. {}", var, body),
            Self::And(a, b) => write!(f, "({} ∧ {})", a, b),
            Self::Or(a, b) => write!(f, "({} ∨ {})", a, b),
            Self::Not(a) => write!(f, "¬{}", a),
            Self::Implies(a, b) => write!(f, "({} → {})", a, b),
            Self::IsAcyclic(r) => write!(f, "acyclic({})", r),
            Self::IsIrreflexive(r) => write!(f, "irreflexive({})", r),
            Self::IsEmpty(r) => write!(f, "empty({})", r),
            Self::IsTotal(r, p) => write!(f, "total({}, [{}])", r, p),
            Self::Subset(a, b) => write!(f, "{} ⊆ {}", a, b),
            Self::True => write!(f, "⊤"),
            Self::False => write!(f, "⊥"),
        }
    }
}

// ---------------------------------------------------------------------------
// AxiomaticModel — a model defined by axioms
// ---------------------------------------------------------------------------

/// A complete axiomatic memory model with formal axiom specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomaticModel {
    pub name: String,
    pub description: String,
    pub derived_relations: Vec<DerivedRelation>,
    pub axioms: Vec<Axiom>,
}

impl AxiomaticModel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            derived_relations: Vec::new(),
            axioms: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn add_derived(&mut self, name: &str, expr: RelationExpr, desc: &str) {
        self.derived_relations.push(DerivedRelation::new(name, expr, desc));
    }

    pub fn add_axiom(&mut self, axiom: Axiom) {
        self.axioms.push(axiom);
    }

    /// Convert to a standard MemoryModel.
    pub fn to_memory_model(&self) -> MemoryModel {
        let mut model = MemoryModel::new(&self.name);
        for dr in &self.derived_relations {
            model.add_derived(&dr.name, dr.expr.clone(), &dr.description);
        }
        for axiom in &self.axioms {
            match axiom {
                Axiom::Acyclic { relation, .. } => {
                    model.add_acyclic(relation.clone());
                }
                Axiom::Irreflexive { relation, .. } => {
                    model.add_irreflexive(relation.clone());
                }
                Axiom::Empty { relation, .. } => {
                    model.add_empty(relation.clone());
                }
                _ => {} // Custom/Total not directly representable.
            }
        }
        model
    }
}

impl fmt::Display for AxiomaticModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "model \"{}\"", self.name)?;
        if !self.description.is_empty() {
            writeln!(f, "  // {}", self.description)?;
        }
        for dr in &self.derived_relations {
            writeln!(f, "  {}", dr)?;
        }
        for ax in &self.axioms {
            writeln!(f, "  {}", ax)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AxiomEvaluator — evaluate axioms against execution graphs
// ---------------------------------------------------------------------------

/// Result of evaluating a single axiom.
#[derive(Debug, Clone)]
pub struct AxiomEvalResult {
    pub axiom_name: String,
    pub satisfied: bool,
    pub witness: Option<Vec<EventId>>,
}

impl fmt::Display for AxiomEvalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.satisfied {
            write!(f, "✓ {}", self.axiom_name)
        } else {
            write!(f, "✗ {}", self.axiom_name)?;
            if let Some(witness) = &self.witness {
                write!(f, " (witness: {:?})", witness)?;
            }
            Ok(())
        }
    }
}

/// Evaluate axioms against execution graphs.
pub struct AxiomEvaluator<'a> {
    model: &'a AxiomaticModel,
}

impl<'a> AxiomEvaluator<'a> {
    pub fn new(model: &'a AxiomaticModel) -> Self {
        Self { model }
    }

    /// Evaluate all axioms against an execution graph.
    pub fn evaluate_all(&self, exec: &ExecutionGraph) -> Vec<AxiomEvalResult> {
        let mm = self.model.to_memory_model();
        let env = mm.compute_derived(exec);

        self.model.axioms.iter().map(|axiom| {
            self.evaluate_axiom(axiom, exec, &mm, &env)
        }).collect()
    }

    /// Check if the execution is consistent with all axioms.
    pub fn is_consistent(&self, exec: &ExecutionGraph) -> bool {
        self.evaluate_all(exec).iter().all(|r| r.satisfied)
    }

    /// Evaluate a single axiom.
    fn evaluate_axiom(
        &self,
        axiom: &Axiom,
        exec: &ExecutionGraph,
        mm: &MemoryModel,
        env: &HashMap<String, BitMatrix>,
    ) -> AxiomEvalResult {
        match axiom {
            Axiom::Acyclic { name, relation } => {
                let mat = mm.eval_expr(relation, exec, env);
                let satisfied = mat.is_acyclic();
                let witness = if !satisfied { mat.find_cycle() } else { None };
                AxiomEvalResult { axiom_name: name.clone(), satisfied, witness }
            }
            Axiom::Irreflexive { name, relation } => {
                let mat = mm.eval_expr(relation, exec, env);
                let satisfied = mat.is_irreflexive();
                let witness = if !satisfied {
                    let n = mat.dim();
                    (0..n).find(|&i| mat.get(i, i)).map(|i| vec![i])
                } else { None };
                AxiomEvalResult { axiom_name: name.clone(), satisfied, witness }
            }
            Axiom::Empty { name, relation } => {
                let mat = mm.eval_expr(relation, exec, env);
                let satisfied = mat.is_empty();
                let witness = if !satisfied {
                    mat.edges().first().map(|&(i, j)| vec![i, j])
                } else { None };
                AxiomEvalResult { axiom_name: name.clone(), satisfied, witness }
            }
            Axiom::Total { name, relation, filter } => {
                let mat = mm.eval_expr(relation, exec, env);
                let matching: Vec<usize> = exec.events.iter()
                    .filter(|e| filter.eval(e))
                    .map(|e| e.id)
                    .collect();
                let satisfied = is_total_on(&mat, &matching);
                AxiomEvalResult { axiom_name: name.clone(), satisfied, witness: None }
            }
            Axiom::Custom { name, formula } => {
                let satisfied = self.evaluate_formula(formula, exec, mm, env);
                AxiomEvalResult { axiom_name: name.clone(), satisfied, witness: None }
            }
        }
    }

    /// Evaluate a FOL formula.
    fn evaluate_formula(
        &self,
        formula: &FOLFormula,
        exec: &ExecutionGraph,
        mm: &MemoryModel,
        env: &HashMap<String, BitMatrix>,
    ) -> bool {
        match formula {
            FOLFormula::IsAcyclic(rel) => {
                mm.eval_expr(rel, exec, env).is_acyclic()
            }
            FOLFormula::IsIrreflexive(rel) => {
                mm.eval_expr(rel, exec, env).is_irreflexive()
            }
            FOLFormula::IsEmpty(rel) => {
                mm.eval_expr(rel, exec, env).is_empty()
            }
            FOLFormula::IsTotal(rel, pred) => {
                let mat = mm.eval_expr(rel, exec, env);
                let matching: Vec<usize> = exec.events.iter()
                    .filter(|e| pred.eval(e))
                    .map(|e| e.id)
                    .collect();
                is_total_on(&mat, &matching)
            }
            FOLFormula::Subset(a, b) => {
                let ma = mm.eval_expr(a, exec, env);
                let mb = mm.eval_expr(b, exec, env);
                ma.difference(&mb).is_empty()
            }
            FOLFormula::And(a, b) => {
                self.evaluate_formula(a, exec, mm, env) &&
                self.evaluate_formula(b, exec, mm, env)
            }
            FOLFormula::Or(a, b) => {
                self.evaluate_formula(a, exec, mm, env) ||
                self.evaluate_formula(b, exec, mm, env)
            }
            FOLFormula::Not(a) => {
                !self.evaluate_formula(a, exec, mm, env)
            }
            FOLFormula::Implies(a, b) => {
                !self.evaluate_formula(a, exec, mm, env) ||
                self.evaluate_formula(b, exec, mm, env)
            }
            FOLFormula::True => true,
            FOLFormula::False => false,
            FOLFormula::InRelation { relation } => {
                !mm.eval_expr(relation, exec, env).is_empty()
            }
            FOLFormula::ForAll { body, .. } => {
                // Approximate: treat as the body formula.
                self.evaluate_formula(body, exec, mm, env)
            }
            FOLFormula::Exists { body, .. } => {
                self.evaluate_formula(body, exec, mm, env)
            }
        }
    }
}

/// Check if a relation is total on the given set of elements.
fn is_total_on(mat: &BitMatrix, elements: &[usize]) -> bool {
    for i in 0..elements.len() {
        for j in i + 1..elements.len() {
            let a = elements[i];
            let b = elements[j];
            if !mat.get(a, b) && !mat.get(b, a) {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Axiom consistency checking
// ---------------------------------------------------------------------------

/// Check if a set of axioms is consistent (can all be satisfied simultaneously).
pub struct AxiomConsistencyChecker;

impl AxiomConsistencyChecker {
    /// Check if axioms are consistent by verifying they don't produce contradictions
    /// on small example executions.
    pub fn check_consistency(model: &AxiomaticModel) -> ConsistencyResult {
        // Generate small test executions and check if any satisfies all axioms.
        let test_sizes = [2, 3, 4];
        let mut tested = 0;
        let mut satisfiable = false;

        for &n in &test_sizes {
            let exec = Self::make_trivial_exec(n);
            let evaluator = AxiomEvaluator::new(model);
            tested += 1;
            if evaluator.is_consistent(&exec) {
                satisfiable = true;
                break;
            }
        }

        ConsistencyResult {
            likely_consistent: satisfiable,
            executions_tested: tested,
        }
    }

    /// Make a trivial execution with n sequential events on one thread.
    fn make_trivial_exec(n: usize) -> ExecutionGraph {
        use super::execution::Event;
        let events: Vec<Event> = (0..n).map(|i| {
            let op = if i % 2 == 0 { OpType::Write } else { OpType::Read };
            Event::new(i, 0, op, 0x100, if i % 2 == 0 { (i + 1) as u64 } else { 0 })
                .with_po_index(i)
        }).collect();
        let mut graph = ExecutionGraph::new(events);
        // Reads read from preceding writes.
        for i in 0..n {
            if i % 2 == 1 && i > 0 {
                graph.rf.set(i - 1, i, true);
            }
        }
        graph.derive_fr();
        graph
    }
}

/// Result of consistency checking.
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    pub likely_consistent: bool,
    pub executions_tested: usize,
}

impl fmt::Display for ConsistencyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Consistency: {} (tested {} executions)",
            if self.likely_consistent { "likely consistent" } else { "possibly inconsistent" },
            self.executions_tested)
    }
}

// ---------------------------------------------------------------------------
// Axiom entailment
// ---------------------------------------------------------------------------

/// Check whether one set of axioms entails another.
pub struct AxiomEntailmentChecker;

impl AxiomEntailmentChecker {
    /// Check if model A's axioms entail a specific axiom.
    /// This is done by checking: for all executions, if A's axioms hold, then the axiom holds.
    /// We approximate by testing on generated executions.
    pub fn entails(
        model_a: &AxiomaticModel,
        axiom: &Axiom,
    ) -> EntailmentResult {
        let test_model = AxiomaticModel {
            name: "test".to_string(),
            description: String::new(),
            derived_relations: model_a.derived_relations.clone(),
            axioms: vec![axiom.clone()],
        };

        let eval_a = AxiomEvaluator::new(model_a);
        let eval_test = AxiomEvaluator::new(&test_model);

        let mut tested = 0;
        let mut counterexample_found = false;

        // Generate test executions.
        for n in 2..=4 {
            let exec = AxiomConsistencyChecker::make_trivial_exec(n);
            tested += 1;
            if eval_a.is_consistent(&exec) && !eval_test.is_consistent(&exec) {
                counterexample_found = true;
                break;
            }
        }

        EntailmentResult {
            likely_entails: !counterexample_found,
            executions_tested: tested,
        }
    }
}

/// Result of entailment checking.
#[derive(Debug, Clone)]
pub struct EntailmentResult {
    pub likely_entails: bool,
    pub executions_tested: usize,
}

impl fmt::Display for EntailmentResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entailment: {} (tested {} executions)",
            if self.likely_entails { "likely entailed" } else { "not entailed" },
            self.executions_tested)
    }
}

// ---------------------------------------------------------------------------
// Model comparison
// ---------------------------------------------------------------------------

/// Compare two memory models to determine relative strength.
#[derive(Debug, Clone)]
pub struct ModelComparisonResult {
    pub model_a: String,
    pub model_b: String,
    pub a_strictly_weaker: bool,
    pub b_strictly_weaker: bool,
    pub equivalent: bool,
    pub incomparable: bool,
    pub executions_tested: usize,
}

impl fmt::Display for ModelComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.equivalent {
            write!(f, "{} ≡ {}", self.model_a, self.model_b)
        } else if self.a_strictly_weaker {
            write!(f, "{} ⊂ {} (A is strictly weaker)", self.model_a, self.model_b)
        } else if self.b_strictly_weaker {
            write!(f, "{} ⊃ {} (B is strictly weaker)", self.model_a, self.model_b)
        } else {
            write!(f, "{} ⋈ {} (incomparable)", self.model_a, self.model_b)
        }
    }
}

/// Compare two axiomatic models.
pub struct ModelComparator;

impl ModelComparator {
    /// Compare two models by checking which allows more behaviors.
    /// Model A is weaker than B if A allows all behaviors that B allows, plus more.
    pub fn compare(a: &AxiomaticModel, b: &AxiomaticModel) -> ModelComparisonResult {
        let eval_a = AxiomEvaluator::new(a);
        let eval_b = AxiomEvaluator::new(b);

        let mut a_allows_but_not_b = false;
        let mut b_allows_but_not_a = false;
        let mut tested = 0;

        for n in 2..=5 {
            let exec = AxiomConsistencyChecker::make_trivial_exec(n);
            tested += 1;
            let a_consistent = eval_a.is_consistent(&exec);
            let b_consistent = eval_b.is_consistent(&exec);

            if a_consistent && !b_consistent {
                a_allows_but_not_b = true;
            }
            if b_consistent && !a_consistent {
                b_allows_but_not_a = true;
            }
        }

        let equivalent = !a_allows_but_not_b && !b_allows_but_not_a;
        let a_strictly_weaker = a_allows_but_not_b && !b_allows_but_not_a;
        let b_strictly_weaker = b_allows_but_not_a && !a_allows_but_not_b;
        let incomparable = a_allows_but_not_b && b_allows_but_not_a;

        ModelComparisonResult {
            model_a: a.name.clone(),
            model_b: b.name.clone(),
            a_strictly_weaker,
            b_strictly_weaker,
            equivalent,
            incomparable,
            executions_tested: tested,
        }
    }
}

// ---------------------------------------------------------------------------
// Standard axiomatic model builders
// ---------------------------------------------------------------------------

/// Build the SC (Sequential Consistency) axiomatic model.
pub fn build_sc_axiomatic() -> AxiomaticModel {
    let mut model = AxiomaticModel::new("SC")
        .with_description("Sequential Consistency");

    model.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication = rf ∪ co ∪ fr",
    );

    model.add_axiom(Axiom::acyclic("sc",
        RelationExpr::union(RelationExpr::base("po"), RelationExpr::base("com")),
    ));

    model
}

/// Build the TSO (Total Store Order) axiomatic model.
pub fn build_tso_axiomatic() -> AxiomaticModel {
    let mut model = AxiomaticModel::new("TSO")
        .with_description("Total Store Order (x86-TSO)");

    model.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    // TSO preserves all program order except W→R.
    // ppo = po \ (W × R \ same-loc)
    model.add_derived("ppo",
        RelationExpr::diff(
            RelationExpr::base("po"),
            RelationExpr::diff(
                RelationExpr::seq(
                    RelationExpr::filter(PredicateExpr::IsWrite),
                    RelationExpr::seq(
                        RelationExpr::base("po"),
                        RelationExpr::filter(PredicateExpr::IsRead),
                    ),
                ),
                RelationExpr::base("po-loc"),
            ),
        ),
        "preserved program order",
    );

    model.add_axiom(Axiom::acyclic("tso",
        RelationExpr::union(RelationExpr::base("ppo"), RelationExpr::base("com")),
    ));

    model
}

/// Build the ARMv8 axiomatic model.
pub fn build_arm_axiomatic() -> AxiomaticModel {
    let mut model = AxiomaticModel::new("ARMv8")
        .with_description("ARMv8 memory model");

    model.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    model.add_axiom(Axiom::acyclic("internal",
        RelationExpr::union(RelationExpr::base("po-loc"), RelationExpr::base("com")),
    ));

    model.add_axiom(Axiom::acyclic("external",
        RelationExpr::union(RelationExpr::base("rfe"), RelationExpr::base("fre")),
    ));

    model
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::execution::{Event, OpType, Scope as ExecScope};

    fn make_sb_exec() -> ExecutionGraph {
        // Store Buffer test: T0: W(x,1); R(y,0) | T1: W(y,1); R(x,0)
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 0x200, 0).with_po_index(1),
            Event::new(2, 1, OpType::Write, 0x200, 1).with_po_index(0),
            Event::new(3, 1, OpType::Read, 0x100, 0).with_po_index(1),
        ];
        let mut graph = ExecutionGraph::new(events);
        // Both reads see initial value (0), so no rf from writes.
        graph.co.set(0, 0, false); // No co needed for single writer per addr.
        graph.derive_fr();
        // FR: read(1) reads 0 at 0x200, but write(2) writes 1 at 0x200 → fr(1,2)
        graph.fr.set(1, 2, true);
        // FR: read(3) reads 0 at 0x100, but write(0) writes 1 at 0x100 → fr(3,0)
        graph.fr.set(3, 0, true);
        graph
    }

    fn make_simple_exec() -> ExecutionGraph {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 0x100, 1).with_po_index(1),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 1, true);
        graph.derive_fr();
        graph
    }

    #[test]
    fn test_axiom_display() {
        let ax = Axiom::acyclic("sc", RelationExpr::base("po"));
        let s = format!("{}", ax);
        assert!(s.contains("acyclic"));
        assert!(s.contains("po"));
    }

    #[test]
    fn test_fol_formula_display() {
        let f = FOLFormula::IsAcyclic(RelationExpr::base("po"));
        let s = format!("{}", f);
        assert!(s.contains("acyclic"));
    }

    #[test]
    fn test_axiomatic_model_creation() {
        let model = build_sc_axiomatic();
        assert_eq!(model.name, "SC");
        assert!(!model.axioms.is_empty());
    }

    #[test]
    fn test_sc_axiom_evaluation() {
        let model = build_sc_axiomatic();
        let evaluator = AxiomEvaluator::new(&model);
        let exec = make_simple_exec();
        let results = evaluator.evaluate_all(&exec);
        assert!(!results.is_empty());
        assert!(results[0].satisfied);
    }

    #[test]
    fn test_sc_consistency() {
        let model = build_sc_axiomatic();
        let evaluator = AxiomEvaluator::new(&model);
        let exec = make_simple_exec();
        assert!(evaluator.is_consistent(&exec));
    }

    #[test]
    fn test_tso_axiomatic_creation() {
        let model = build_tso_axiomatic();
        assert_eq!(model.name, "TSO");
        assert!(!model.axioms.is_empty());
    }

    #[test]
    fn test_arm_axiomatic_creation() {
        let model = build_arm_axiomatic();
        assert_eq!(model.name, "ARMv8");
        assert!(!model.axioms.is_empty());
    }

    #[test]
    fn test_to_memory_model() {
        let am = build_sc_axiomatic();
        let mm = am.to_memory_model();
        assert_eq!(mm.name, "SC");
        assert!(!mm.constraints.is_empty());
    }

    #[test]
    fn test_axiom_consistency_check() {
        let model = build_sc_axiomatic();
        let result = AxiomConsistencyChecker::check_consistency(&model);
        assert!(result.likely_consistent);
        assert!(result.executions_tested > 0);
    }

    #[test]
    fn test_axiom_entailment() {
        let sc = build_sc_axiomatic();
        let test_axiom = Axiom::acyclic("test", RelationExpr::base("po"));
        let result = AxiomEntailmentChecker::entails(&sc, &test_axiom);
        assert!(result.executions_tested > 0);
    }

    #[test]
    fn test_model_comparison_sc_tso() {
        let sc = build_sc_axiomatic();
        let tso = build_tso_axiomatic();
        let result = ModelComparator::compare(&sc, &tso);
        assert!(result.executions_tested > 0);
        // SC should be at least as strong as (or equivalent to) TSO on trivial tests.
    }

    #[test]
    fn test_fol_formula_and() {
        let f = FOLFormula::and(
            FOLFormula::IsAcyclic(RelationExpr::base("po")),
            FOLFormula::IsIrreflexive(RelationExpr::base("co")),
        );
        let s = format!("{}", f);
        assert!(s.contains("∧"));
    }

    #[test]
    fn test_fol_formula_implies() {
        let f = FOLFormula::implies(
            FOLFormula::IsAcyclic(RelationExpr::base("po")),
            FOLFormula::IsAcyclic(RelationExpr::base("co")),
        );
        let s = format!("{}", f);
        assert!(s.contains("→"));
    }

    #[test]
    fn test_fol_formula_subset() {
        let f = FOLFormula::Subset(RelationExpr::base("po"), RelationExpr::base("co"));
        let model = AxiomaticModel::new("test");
        let evaluator = AxiomEvaluator::new(&model);
        let mm = model.to_memory_model();
        let exec = make_simple_exec();
        let env = mm.compute_derived(&exec);
        let result = evaluator.evaluate_formula(&f, &exec, &mm, &env);
        // po is not a subset of co for a single-thread execution.
        assert!(!result);
    }

    #[test]
    fn test_axiom_total() {
        let axiom = Axiom::total("co-total", RelationExpr::base("co"), PredicateExpr::IsWrite);
        let s = format!("{}", axiom);
        assert!(s.contains("total"));
    }

    #[test]
    fn test_axiom_name() {
        let ax = Axiom::acyclic("test-axiom", RelationExpr::base("po"));
        assert_eq!(ax.name(), "test-axiom");
    }

    #[test]
    fn test_custom_axiom() {
        let formula = FOLFormula::and(
            FOLFormula::IsAcyclic(RelationExpr::base("po")),
            FOLFormula::IsIrreflexive(RelationExpr::base("fr")),
        );
        let axiom = Axiom::custom("custom-test", formula);
        let s = format!("{}", axiom);
        assert!(s.contains("custom-test"));
    }

    #[test]
    fn test_consistency_result_display() {
        let result = ConsistencyResult {
            likely_consistent: true,
            executions_tested: 5,
        };
        let s = format!("{}", result);
        assert!(s.contains("likely consistent"));
    }

    #[test]
    fn test_entailment_result_display() {
        let result = EntailmentResult {
            likely_entails: true,
            executions_tested: 3,
        };
        let s = format!("{}", result);
        assert!(s.contains("likely entailed"));
    }

    #[test]
    fn test_comparison_result_display() {
        let result = ModelComparisonResult {
            model_a: "SC".to_string(),
            model_b: "TSO".to_string(),
            a_strictly_weaker: false,
            b_strictly_weaker: false,
            equivalent: true,
            incomparable: false,
            executions_tested: 4,
        };
        let s = format!("{}", result);
        assert!(s.contains("≡"));
    }

    #[test]
    fn test_is_total_on() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(0, 2, true);
        assert!(is_total_on(&m, &[0, 1, 2]));
    }

    #[test]
    fn test_is_not_total_on() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        assert!(!is_total_on(&m, &[0, 1, 2]));
    }

    #[test]
    fn test_fol_for_all() {
        let f = FOLFormula::for_all("x", FOLFormula::True);
        let s = format!("{}", f);
        assert!(s.contains("∀"));
    }

    #[test]
    fn test_fol_exists() {
        let f = FOLFormula::exists("x", FOLFormula::True);
        let s = format!("{}", f);
        assert!(s.contains("∃"));
    }

    #[test]
    fn test_fol_not() {
        let f = FOLFormula::not(FOLFormula::True);
        let s = format!("{}", f);
        assert!(s.contains("¬"));
    }

    #[test]
    fn test_model_with_description() {
        let model = AxiomaticModel::new("test")
            .with_description("A test model");
        assert_eq!(model.description, "A test model");
    }
}
