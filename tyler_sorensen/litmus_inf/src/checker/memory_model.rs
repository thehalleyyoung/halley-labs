//! Memory model specification via relation expressions and constraints.
//!
//! Provides an AST for defining axiomatic memory models (SC, TSO, PSO, ARM,
//! RISC-V, PTX, WebGPU) as collections of derived relations and acyclicity /
//! irreflexivity constraints. Includes a small DSL parser.

use std::fmt;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use super::execution::{BitMatrix, ExecutionGraph, OpType, Scope};

// ---------------------------------------------------------------------------
// RelationExpr — AST for relation expressions
// ---------------------------------------------------------------------------

/// AST node for a relation expression in model definitions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationExpr {
    /// A base (named) relation: po, rf, co, fr, etc.
    Base(String),
    /// Sequential composition: R1 ; R2.
    Seq(Box<RelationExpr>, Box<RelationExpr>),
    /// Union: R1 | R2.
    Union(Box<RelationExpr>, Box<RelationExpr>),
    /// Intersection: R1 & R2.
    Inter(Box<RelationExpr>, Box<RelationExpr>),
    /// Set difference: R1 \ R2.
    Diff(Box<RelationExpr>, Box<RelationExpr>),
    /// Inverse (transpose): R^{-1}.
    Inverse(Box<RelationExpr>),
    /// Transitive closure: R+.
    Plus(Box<RelationExpr>),
    /// Reflexive-transitive closure: R*.
    Star(Box<RelationExpr>),
    /// Optional (reflexive closure): R?.
    Optional(Box<RelationExpr>),
    /// Identity relation restricted to all events.
    Identity,
    /// Filter / guard: [pred] — identity restricted to events matching predicate.
    Filter(PredicateExpr),
    /// Empty relation.
    Empty,
}

impl RelationExpr {
    // Convenience constructors.
    pub fn base(name: &str) -> Self { Self::Base(name.to_string()) }
    pub fn seq(a: Self, b: Self) -> Self { Self::Seq(Box::new(a), Box::new(b)) }
    pub fn union(a: Self, b: Self) -> Self { Self::Union(Box::new(a), Box::new(b)) }
    pub fn inter(a: Self, b: Self) -> Self { Self::Inter(Box::new(a), Box::new(b)) }
    pub fn diff(a: Self, b: Self) -> Self { Self::Diff(Box::new(a), Box::new(b)) }
    pub fn inverse(a: Self) -> Self { Self::Inverse(Box::new(a)) }
    pub fn plus(a: Self) -> Self { Self::Plus(Box::new(a)) }
    pub fn star(a: Self) -> Self { Self::Star(Box::new(a)) }
    pub fn optional(a: Self) -> Self { Self::Optional(Box::new(a)) }
    pub fn filter(p: PredicateExpr) -> Self { Self::Filter(p) }

    /// Multi-way union of relation expressions.
    pub fn union_many(exprs: Vec<Self>) -> Self {
        exprs.into_iter().reduce(|a, b| Self::union(a, b)).unwrap_or(Self::Empty)
    }

    /// Multi-way sequence.
    pub fn seq_many(exprs: Vec<Self>) -> Self {
        exprs.into_iter().reduce(|a, b| Self::seq(a, b)).unwrap_or(Self::Identity)
    }

    /// Collect all base relation names referenced.
    pub fn referenced_bases(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.collect_bases(&mut names);
        names.sort();
        names.dedup();
        names
    }

    fn collect_bases(&self, out: &mut Vec<String>) {
        match self {
            Self::Base(n) => out.push(n.clone()),
            Self::Seq(a, b) | Self::Union(a, b) | Self::Inter(a, b) | Self::Diff(a, b) => {
                a.collect_bases(out);
                b.collect_bases(out);
            }
            Self::Inverse(a) | Self::Plus(a) | Self::Star(a) | Self::Optional(a) => {
                a.collect_bases(out);
            }
            Self::Identity | Self::Filter(_) | Self::Empty => {}
        }
    }
}

impl fmt::Display for RelationExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base(n) => write!(f, "{}", n),
            Self::Seq(a, b) => write!(f, "({} ; {})", a, b),
            Self::Union(a, b) => write!(f, "({} | {})", a, b),
            Self::Inter(a, b) => write!(f, "({} & {})", a, b),
            Self::Diff(a, b) => write!(f, "({} \\ {})", a, b),
            Self::Inverse(a) => write!(f, "{}^-1", a),
            Self::Plus(a) => write!(f, "{}+", a),
            Self::Star(a) => write!(f, "{}*", a),
            Self::Optional(a) => write!(f, "{}?", a),
            Self::Identity => write!(f, "id"),
            Self::Filter(p) => write!(f, "[{}]", p),
            Self::Empty => write!(f, "0"),
        }
    }
}

// ---------------------------------------------------------------------------
// PredicateExpr
// ---------------------------------------------------------------------------

/// Predicate on events (used in [P] guard expressions).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredicateExpr {
    /// All read events (including RMW reads).
    IsRead,
    /// All write events (including RMW writes).
    IsWrite,
    /// Fence events.
    IsFence,
    /// RMW events.
    IsRMW,
    /// Events at a specific address.
    AtAddress(u64),
    /// Events on a specific thread.
    OnThread(usize),
    /// Events with a specific scope.
    HasScope(ScopeFilter),
    /// Conjunction.
    And(Box<PredicateExpr>, Box<PredicateExpr>),
    /// Disjunction.
    Or(Box<PredicateExpr>, Box<PredicateExpr>),
    /// Negation.
    Not(Box<PredicateExpr>),
    /// All events (tautology).
    True,
}

impl PredicateExpr {
    pub fn and(a: Self, b: Self) -> Self { Self::And(Box::new(a), Box::new(b)) }
    pub fn or(a: Self, b: Self) -> Self { Self::Or(Box::new(a), Box::new(b)) }
    pub fn not(a: Self) -> Self { Self::Not(Box::new(a)) }

    /// Evaluate predicate on an event.
    pub fn eval(&self, event: &super::execution::Event) -> bool {
        match self {
            Self::IsRead => event.is_read(),
            Self::IsWrite => event.is_write(),
            Self::IsFence => event.is_fence(),
            Self::IsRMW => event.is_rmw(),
            Self::AtAddress(a) => event.address == *a,
            Self::OnThread(t) => event.thread == *t,
            Self::HasScope(s) => match s {
                ScopeFilter::CTA => event.scope == Scope::CTA,
                ScopeFilter::GPU => event.scope == Scope::GPU,
                ScopeFilter::System => event.scope == Scope::System,
            },
            Self::And(a, b) => a.eval(event) && b.eval(event),
            Self::Or(a, b) => a.eval(event) || b.eval(event),
            Self::Not(a) => !a.eval(event),
            Self::True => true,
        }
    }
}

impl fmt::Display for PredicateExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IsRead => write!(f, "R"),
            Self::IsWrite => write!(f, "W"),
            Self::IsFence => write!(f, "F"),
            Self::IsRMW => write!(f, "RMW"),
            Self::AtAddress(a) => write!(f, "addr={:#x}", a),
            Self::OnThread(t) => write!(f, "thread={}", t),
            Self::HasScope(s) => write!(f, "scope={:?}", s),
            Self::And(a, b) => write!(f, "({} & {})", a, b),
            Self::Or(a, b) => write!(f, "({} | {})", a, b),
            Self::Not(a) => write!(f, "!{}", a),
            Self::True => write!(f, "true"),
        }
    }
}

/// Scope filter for GPU models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScopeFilter {
    CTA,
    GPU,
    System,
}

// ---------------------------------------------------------------------------
// RelationType / RelationDef / DerivedRelation
// ---------------------------------------------------------------------------

/// Whether a relation is a base primitive or derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    Base,
    Derived,
}

/// Definition of a named relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationDef {
    pub name: String,
    pub rel_type: RelationType,
    pub description: String,
}

impl RelationDef {
    pub fn base(name: &str, desc: &str) -> Self {
        Self { name: name.into(), rel_type: RelationType::Base, description: desc.into() }
    }
    pub fn derived(name: &str, desc: &str) -> Self {
        Self { name: name.into(), rel_type: RelationType::Derived, description: desc.into() }
    }
}

/// A derived relation: name = expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedRelation {
    pub name: String,
    pub expr: RelationExpr,
    pub description: String,
}

impl DerivedRelation {
    pub fn new(name: &str, expr: RelationExpr, desc: &str) -> Self {
        Self { name: name.into(), expr, description: desc.into() }
    }
}

impl fmt::Display for DerivedRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "let {} = {}", self.name, self.expr)
    }
}

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------

/// A model constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// The relation must be acyclic: ¬(∃ cycle in R).
    Acyclic(RelationExpr, String),
    /// The relation must be irreflexive: ¬(∃ x. (x,x) ∈ R).
    Irreflexive(RelationExpr, String),
    /// The relation must be empty.
    Empty(RelationExpr, String),
}

impl Constraint {
    pub fn acyclic(expr: RelationExpr) -> Self {
        let desc = format!("acyclic({})", expr);
        Self::Acyclic(expr, desc)
    }
    pub fn irreflexive(expr: RelationExpr) -> Self {
        let desc = format!("irreflexive({})", expr);
        Self::Irreflexive(expr, desc)
    }
    pub fn empty(expr: RelationExpr) -> Self {
        let desc = format!("empty({})", expr);
        Self::Empty(expr, desc)
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Acyclic(_, n) | Self::Irreflexive(_, n) | Self::Empty(_, n) => n,
        }
    }

    pub fn expr(&self) -> &RelationExpr {
        match self {
            Self::Acyclic(e, _) | Self::Irreflexive(e, _) | Self::Empty(e, _) => e,
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acyclic(e, _) => write!(f, "acyclic {}", e),
            Self::Irreflexive(e, _) => write!(f, "irreflexive {}", e),
            Self::Empty(e, _) => write!(f, "empty {}", e),
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryModel
// ---------------------------------------------------------------------------

/// A complete axiomatic memory model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryModel {
    pub name: String,
    pub base_relations: Vec<RelationDef>,
    pub derived_relations: Vec<DerivedRelation>,
    pub constraints: Vec<Constraint>,
}

impl MemoryModel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            base_relations: vec![
                RelationDef::base("po", "program order"),
                RelationDef::base("rf", "reads-from"),
                RelationDef::base("co", "coherence order"),
                RelationDef::base("fr", "from-reads"),
            ],
            derived_relations: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a derived relation.
    pub fn add_derived(&mut self, name: &str, expr: RelationExpr, desc: &str) {
        self.derived_relations.push(DerivedRelation::new(name, expr, desc));
    }

    /// Add an acyclicity constraint.
    pub fn add_acyclic(&mut self, expr: RelationExpr) {
        self.constraints.push(Constraint::acyclic(expr));
    }

    /// Add an irreflexivity constraint.
    pub fn add_irreflexive(&mut self, expr: RelationExpr) {
        self.constraints.push(Constraint::irreflexive(expr));
    }

    /// Add an emptiness constraint.
    pub fn add_empty(&mut self, expr: RelationExpr) {
        self.constraints.push(Constraint::empty(expr));
    }

    /// Validate the model definition (all referenced relations are defined).
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let mut known: Vec<String> = self.base_relations.iter().map(|r| r.name.clone()).collect();

        // Built-in relations always available.
        for builtin in &["po", "rf", "co", "fr", "id", "ext", "int",
                          "po-loc", "com", "rfe", "rfi", "coe", "coi",
                          "fre", "fri", "addr", "data", "ctrl",
                          "rmw", "amo", "same-loc"] {
            if !known.contains(&builtin.to_string()) {
                known.push(builtin.to_string());
            }
        }

        for dr in &self.derived_relations {
            let refs = dr.expr.referenced_bases();
            for r in &refs {
                if !known.contains(r) {
                    errors.push(format!("derived relation '{}' references unknown '{}'", dr.name, r));
                }
            }
            known.push(dr.name.clone());
        }

        for c in &self.constraints {
            let refs = c.expr().referenced_bases();
            for r in &refs {
                if !known.contains(r) {
                    errors.push(format!("constraint '{}' references unknown '{}'", c.name(), r));
                }
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Evaluate a relation expression against an execution graph.
    pub fn eval_expr(&self, expr: &RelationExpr, exec: &ExecutionGraph,
                     env: &HashMap<String, BitMatrix>) -> BitMatrix {
        let n = exec.len();
        match expr {
            RelationExpr::Base(name) => {
                if let Some(m) = env.get(name.as_str()) {
                    m.clone()
                } else {
                    match name.as_str() {
                        "po" => exec.po.clone(),
                        "rf" => exec.rf.clone(),
                        "co" => exec.co.clone(),
                        "fr" => exec.fr.clone(),
                        "id" => BitMatrix::identity(n),
                        "po-loc" => {
                            let sa = exec.same_address();
                            exec.po.intersection(&sa)
                        }
                        "ext" => {
                            let st = exec.same_thread();
                            BitMatrix::universal(n).difference(&st)
                        }
                        "int" => exec.same_thread(),
                        "com" => exec.rf.union(&exec.co).union(&exec.fr),
                        "rfe" => exec.external(&exec.rf),
                        "rfi" => exec.internal(&exec.rf),
                        "coe" => exec.external(&exec.co),
                        "coi" => exec.internal(&exec.co),
                        "fre" => exec.external(&exec.fr),
                        "fri" => exec.internal(&exec.fr),
                        "same-loc" => exec.same_address(),
                        _ => {
                            if let Some(m) = exec.get_relation(name) {
                                m.clone()
                            } else {
                                BitMatrix::new(n)
                            }
                        }
                    }
                }
            }
            RelationExpr::Seq(a, b) => {
                let ma = self.eval_expr(a, exec, env);
                let mb = self.eval_expr(b, exec, env);
                ma.compose(&mb)
            }
            RelationExpr::Union(a, b) => {
                let ma = self.eval_expr(a, exec, env);
                let mb = self.eval_expr(b, exec, env);
                ma.union(&mb)
            }
            RelationExpr::Inter(a, b) => {
                let ma = self.eval_expr(a, exec, env);
                let mb = self.eval_expr(b, exec, env);
                ma.intersection(&mb)
            }
            RelationExpr::Diff(a, b) => {
                let ma = self.eval_expr(a, exec, env);
                let mb = self.eval_expr(b, exec, env);
                ma.difference(&mb)
            }
            RelationExpr::Inverse(a) => {
                self.eval_expr(a, exec, env).inverse()
            }
            RelationExpr::Plus(a) => {
                self.eval_expr(a, exec, env).transitive_closure()
            }
            RelationExpr::Star(a) => {
                self.eval_expr(a, exec, env).reflexive_transitive_closure()
            }
            RelationExpr::Optional(a) => {
                self.eval_expr(a, exec, env).optional()
            }
            RelationExpr::Identity => BitMatrix::identity(n),
            RelationExpr::Filter(pred) => {
                let bools: Vec<bool> = exec.events.iter().map(|e| pred.eval(e)).collect();
                BitMatrix::identity_filter(n, &bools)
            }
            RelationExpr::Empty => BitMatrix::new(n),
        }
    }

    /// Compute all derived relations and return the environment.
    pub fn compute_derived(&self, exec: &ExecutionGraph) -> HashMap<String, BitMatrix> {
        let mut env: HashMap<String, BitMatrix> = HashMap::new();
        for dr in &self.derived_relations {
            let m = self.eval_expr(&dr.expr, exec, &env);
            env.insert(dr.name.clone(), m);
        }
        env
    }
}

impl fmt::Display for MemoryModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "model \"{}\"", self.name)?;
        for dr in &self.derived_relations {
            writeln!(f, "  {}", dr)?;
        }
        for c in &self.constraints {
            writeln!(f, "  {}", c)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BuiltinModel — factory for standard models
// ---------------------------------------------------------------------------

/// Enumeration of built-in memory models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuiltinModel {
    SC,
    TSO,
    PSO,
    ARM,
    RISCV,
    PTX,
    WebGPU,
}

impl BuiltinModel {
    /// All built-in models.
    pub fn all() -> Vec<Self> {
        vec![Self::SC, Self::TSO, Self::PSO, Self::ARM, Self::RISCV, Self::PTX, Self::WebGPU]
    }

    /// Build the memory model.
    pub fn build(&self) -> MemoryModel {
        match self {
            Self::SC    => build_sc(),
            Self::TSO   => build_tso(),
            Self::PSO   => build_pso(),
            Self::ARM   => build_arm(),
            Self::RISCV => build_riscv(),
            Self::PTX   => build_ptx(),
            Self::WebGPU => build_webgpu(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::SC    => "SC",
            Self::TSO   => "TSO",
            Self::PSO   => "PSO",
            Self::ARM   => "ARMv8",
            Self::RISCV => "RISC-V",
            Self::PTX   => "PTX",
            Self::WebGPU => "WebGPU",
        }
    }
}

impl fmt::Display for BuiltinModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// SC — Sequential Consistency
// ---------------------------------------------------------------------------

fn build_sc() -> MemoryModel {
    let mut m = MemoryModel::new("SC");

    // SC: all memory accesses appear in a single total order consistent with
    // program order. Equivalently: acyclic(po ∪ com).
    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication = rf ∪ co ∪ fr",
    );

    m.add_acyclic(
        RelationExpr::union(RelationExpr::base("po"), RelationExpr::base("com")),
    );

    m
}

// ---------------------------------------------------------------------------
// TSO — Total Store Order (x86)
// ---------------------------------------------------------------------------

fn build_tso() -> MemoryModel {
    let mut m = MemoryModel::new("TSO");

    // TSO relaxes W→R ordering within the same thread (store buffer).
    // Preserved program order (ppo): po \ (W×R \ same-loc).
    // Equivalently: po ∩ (R×M ∪ W×W ∪ same-loc).

    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    // Reads: [R] ; po — po edges starting from a read.
    // ppo for TSO: po ∩ (R×M ∪ W×W) ∪ po-loc
    m.add_derived("mfence",
        RelationExpr::seq(
            RelationExpr::seq(
                RelationExpr::filter(PredicateExpr::IsWrite),
                RelationExpr::base("po"),
            ),
            RelationExpr::filter(PredicateExpr::IsRead),
        ),
        "W;po;R (the pairs TSO relaxes unless fenced)",
    );

    // implied: the ppo is everything except unfenced W→R.
    // For TSO, we define:
    //   ppo = po \ (([W];po;[R]) \ po-loc)
    // Actually, simpler: TSO = acyclic(po-loc ∪ com ∪ [R];po;[M] ∪ [W];po;[W] ∪ [F];po ∪ po;[F])
    // We use the standard cat definition.

    // Preserved program order for TSO:
    //   ppo = [R];po;[R] | [R];po;[W] | [W];po;[W] | po-loc
    m.add_derived("ppo",
        RelationExpr::union_many(vec![
            // R → R
            RelationExpr::seq_many(vec![
                RelationExpr::filter(PredicateExpr::IsRead),
                RelationExpr::base("po"),
                RelationExpr::filter(PredicateExpr::IsRead),
            ]),
            // R → W
            RelationExpr::seq_many(vec![
                RelationExpr::filter(PredicateExpr::IsRead),
                RelationExpr::base("po"),
                RelationExpr::filter(PredicateExpr::IsWrite),
            ]),
            // W → W
            RelationExpr::seq_many(vec![
                RelationExpr::filter(PredicateExpr::IsWrite),
                RelationExpr::base("po"),
                RelationExpr::filter(PredicateExpr::IsWrite),
            ]),
            // po-loc
            RelationExpr::base("po-loc"),
        ]),
        "preserved program order for TSO",
    );

    // Fence-induced ordering: MFENCE orders everything.
    m.add_derived("fence-order",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::IsFence),
            RelationExpr::base("po"),
        ]),
        "ordering induced by fences",
    );

    m.add_derived("ghb",
        RelationExpr::union_many(vec![
            RelationExpr::base("ppo"),
            RelationExpr::base("fence-order"),
            RelationExpr::base("rfe"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "global happens-before",
    );

    m.add_acyclic(RelationExpr::base("ghb"));

    // Internal visibility: uniproc / coherence per location.
    m.add_irreflexive(
        RelationExpr::seq(RelationExpr::base("fre"), RelationExpr::base("rfe")),
    );

    m
}

// ---------------------------------------------------------------------------
// PSO — Partial Store Order (SPARC)
// ---------------------------------------------------------------------------

fn build_pso() -> MemoryModel {
    let mut m = MemoryModel::new("PSO");

    // PSO further relaxes W→W ordering (different addresses).
    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    // ppo = R→R ∪ R→W ∪ po-loc (but NOT W→W to different addresses)
    m.add_derived("ppo",
        RelationExpr::union_many(vec![
            RelationExpr::seq_many(vec![
                RelationExpr::filter(PredicateExpr::IsRead),
                RelationExpr::base("po"),
                RelationExpr::filter(PredicateExpr::IsRead),
            ]),
            RelationExpr::seq_many(vec![
                RelationExpr::filter(PredicateExpr::IsRead),
                RelationExpr::base("po"),
                RelationExpr::filter(PredicateExpr::IsWrite),
            ]),
            RelationExpr::base("po-loc"),
        ]),
        "preserved program order for PSO",
    );

    m.add_derived("fence-order",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::IsFence),
            RelationExpr::base("po"),
        ]),
        "fence-induced ordering",
    );

    m.add_derived("ghb",
        RelationExpr::union_many(vec![
            RelationExpr::base("ppo"),
            RelationExpr::base("fence-order"),
            RelationExpr::base("rfe"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "global happens-before",
    );

    m.add_acyclic(RelationExpr::base("ghb"));

    m
}

// ---------------------------------------------------------------------------
// ARM (ARMv8 / AArch64)
// ---------------------------------------------------------------------------

fn build_arm() -> MemoryModel {
    let mut m = MemoryModel::new("ARMv8");

    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    // Observed-by: ob = obs ∪ dob ∪ aob ∪ bob
    // obs = rfe ∪ fre ∪ coe
    m.add_derived("obs",
        RelationExpr::union_many(vec![
            RelationExpr::base("rfe"),
            RelationExpr::base("fre"),
            RelationExpr::base("coe"),
        ]),
        "observation",
    );

    // Dependency-ordered-before (simplified):
    // dob = addr ∪ data ∪ ctrl;[W] ∪ addr;po;[W]
    m.add_derived("dob",
        RelationExpr::union_many(vec![
            RelationExpr::base("addr"),
            RelationExpr::base("data"),
            RelationExpr::seq(RelationExpr::base("ctrl"), RelationExpr::filter(PredicateExpr::IsWrite)),
            RelationExpr::seq_many(vec![
                RelationExpr::base("addr"),
                RelationExpr::base("po"),
                RelationExpr::filter(PredicateExpr::IsWrite),
            ]),
        ]),
        "dependency-ordered-before",
    );

    // Atomic-ordered-before:
    // aob = rmw ∪ [range(rmw)];rfi
    m.add_derived("aob",
        RelationExpr::union_many(vec![
            RelationExpr::base("rmw"),
            RelationExpr::seq(
                RelationExpr::filter(PredicateExpr::IsRMW),
                RelationExpr::base("rfi"),
            ),
        ]),
        "atomic-ordered-before",
    );

    // Barrier-ordered-before (simplified):
    // bob = po;[F];po
    m.add_derived("bob",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::IsFence),
            RelationExpr::base("po"),
        ]),
        "barrier-ordered-before",
    );

    // ob = (obs ∪ dob ∪ aob ∪ bob)+
    m.add_derived("ob",
        RelationExpr::plus(RelationExpr::union_many(vec![
            RelationExpr::base("obs"),
            RelationExpr::base("dob"),
            RelationExpr::base("aob"),
            RelationExpr::base("bob"),
        ])),
        "ordered-before",
    );

    // Internal visibility.
    m.add_irreflexive(RelationExpr::base("ob"));

    // Coherence: acyclic(po-loc ∪ com)
    m.add_acyclic(
        RelationExpr::union(RelationExpr::base("po-loc"), RelationExpr::base("com")),
    );

    // Atomic: empty(rmw ∩ (fre;coe))
    m.add_empty(
        RelationExpr::inter(
            RelationExpr::base("rmw"),
            RelationExpr::seq(RelationExpr::base("fre"), RelationExpr::base("coe")),
        ),
    );

    m
}

// ---------------------------------------------------------------------------
// RISC-V (RVWMO)
// ---------------------------------------------------------------------------

fn build_riscv() -> MemoryModel {
    let mut m = MemoryModel::new("RISC-V");

    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    // Preserved program order for RVWMO.
    // ppo = (addr | data | ctrl;[W] | po-loc) ∪ fence-order
    m.add_derived("deps",
        RelationExpr::union_many(vec![
            RelationExpr::base("addr"),
            RelationExpr::base("data"),
            RelationExpr::seq(RelationExpr::base("ctrl"), RelationExpr::filter(PredicateExpr::IsWrite)),
        ]),
        "dependencies",
    );

    m.add_derived("fence-order",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::IsFence),
            RelationExpr::base("po"),
        ]),
        "fence-induced ordering",
    );

    m.add_derived("ppo",
        RelationExpr::union_many(vec![
            RelationExpr::base("deps"),
            RelationExpr::base("po-loc"),
            RelationExpr::base("fence-order"),
            RelationExpr::base("rmw"),
        ]),
        "preserved program order",
    );

    m.add_derived("ghb",
        RelationExpr::union_many(vec![
            RelationExpr::base("ppo"),
            RelationExpr::base("rfe"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "global happens-before",
    );

    m.add_acyclic(RelationExpr::base("ghb"));

    // Coherence per location.
    m.add_acyclic(
        RelationExpr::union(RelationExpr::base("po-loc"), RelationExpr::base("com")),
    );

    // Atomicity.
    m.add_empty(
        RelationExpr::inter(
            RelationExpr::base("rmw"),
            RelationExpr::seq(RelationExpr::base("fre"), RelationExpr::base("coe")),
        ),
    );

    m
}

// ---------------------------------------------------------------------------
// PTX (GPU)
// ---------------------------------------------------------------------------

fn build_ptx() -> MemoryModel {
    let mut m = MemoryModel::new("PTX");

    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    // Scoped synchronization: scope inclusion determines visibility.
    // For PTX, we have CTA / GPU / System scope fences and operations.

    // CTA-visible ordering.
    m.add_derived("cta-fence",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::And(
                Box::new(PredicateExpr::IsFence),
                Box::new(PredicateExpr::HasScope(ScopeFilter::CTA)),
            )),
            RelationExpr::base("po"),
        ]),
        "CTA-scope fence ordering",
    );

    // GPU-visible ordering.
    m.add_derived("gpu-fence",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::And(
                Box::new(PredicateExpr::IsFence),
                Box::new(PredicateExpr::Or(
                    Box::new(PredicateExpr::HasScope(ScopeFilter::GPU)),
                    Box::new(PredicateExpr::HasScope(ScopeFilter::System)),
                )),
            )),
            RelationExpr::base("po"),
        ]),
        "GPU-scope fence ordering",
    );

    // System-visible ordering.
    m.add_derived("sys-fence",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::And(
                Box::new(PredicateExpr::IsFence),
                Box::new(PredicateExpr::HasScope(ScopeFilter::System)),
            )),
            RelationExpr::base("po"),
        ]),
        "System-scope fence ordering",
    );

    // Scoped ppo: dependencies ∪ fence-order ∪ po-loc.
    m.add_derived("ppo",
        RelationExpr::union_many(vec![
            RelationExpr::base("po-loc"),
            RelationExpr::base("cta-fence"),
            RelationExpr::base("gpu-fence"),
            RelationExpr::base("sys-fence"),
        ]),
        "preserved program order (scoped)",
    );

    m.add_derived("ghb",
        RelationExpr::union_many(vec![
            RelationExpr::base("ppo"),
            RelationExpr::base("rfe"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "global happens-before",
    );

    m.add_acyclic(RelationExpr::base("ghb"));

    // Coherence.
    m.add_acyclic(
        RelationExpr::union(RelationExpr::base("po-loc"), RelationExpr::base("com")),
    );

    m
}

// ---------------------------------------------------------------------------
// WebGPU
// ---------------------------------------------------------------------------

fn build_webgpu() -> MemoryModel {
    let mut m = MemoryModel::new("WebGPU");

    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );

    // WebGPU has workgroup / device / queue-family scopes (mapped to CTA / GPU / System).
    // Barrier synchronization within workgroups.
    m.add_derived("barrier-order",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::And(
                Box::new(PredicateExpr::IsFence),
                Box::new(PredicateExpr::HasScope(ScopeFilter::CTA)),
            )),
            RelationExpr::base("po"),
        ]),
        "workgroup barrier ordering",
    );

    m.add_derived("device-fence",
        RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::filter(PredicateExpr::And(
                Box::new(PredicateExpr::IsFence),
                Box::new(PredicateExpr::Or(
                    Box::new(PredicateExpr::HasScope(ScopeFilter::GPU)),
                    Box::new(PredicateExpr::HasScope(ScopeFilter::System)),
                )),
            )),
            RelationExpr::base("po"),
        ]),
        "device-scope fence ordering",
    );

    m.add_derived("ppo",
        RelationExpr::union_many(vec![
            RelationExpr::base("po-loc"),
            RelationExpr::base("barrier-order"),
            RelationExpr::base("device-fence"),
        ]),
        "preserved program order",
    );

    m.add_derived("ghb",
        RelationExpr::union_many(vec![
            RelationExpr::base("ppo"),
            RelationExpr::base("rfe"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "global happens-before",
    );

    m.add_acyclic(RelationExpr::base("ghb"));

    m.add_acyclic(
        RelationExpr::union(RelationExpr::base("po-loc"), RelationExpr::base("com")),
    );

    m
}

// ---------------------------------------------------------------------------
// DSL Parser
// ---------------------------------------------------------------------------

/// Simple DSL parser for memory model definitions.
///
/// Syntax:
/// ```text
/// model "Name"
/// let rel = expr
/// acyclic expr
/// irreflexive expr
/// empty expr
/// ```
///
/// Expression syntax:
/// ```text
/// expr ::= name | expr ';' expr | expr '|' expr | expr '&' expr
///        | expr '^-1' | expr '+' | expr '*' | expr '?'
///        | '[' pred ']' | 'id' | '0' | '(' expr ')'
/// pred ::= 'R' | 'W' | 'F' | 'RMW'
/// ```
pub struct ModelParser;

impl ModelParser {
    /// Parse a model definition from the DSL.
    pub fn parse(input: &str) -> Result<MemoryModel, String> {
        let lines: Vec<&str> = input.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("//"))
            .collect();

        if lines.is_empty() {
            return Err("empty model definition".into());
        }

        // Parse model name.
        let name = if lines[0].starts_with("model") {
            let rest = lines[0].strip_prefix("model").unwrap().trim();
            rest.trim_matches('"').to_string()
        } else {
            "unnamed".to_string()
        };

        let mut model = MemoryModel::new(&name);

        for line in &lines[1..] {
            if line.starts_with("let ") {
                let rest = &line[4..];
                if let Some(eq_pos) = rest.find('=') {
                    let name = rest[..eq_pos].trim();
                    let expr_str = rest[eq_pos + 1..].trim();
                    let expr = Self::parse_expr(expr_str)?;
                    model.add_derived(name, expr, "");
                } else {
                    return Err(format!("invalid let statement: {}", line));
                }
            } else if line.starts_with("acyclic ") {
                let expr_str = &line[8..].trim();
                let expr = Self::parse_expr(expr_str)?;
                model.add_acyclic(expr);
            } else if line.starts_with("irreflexive ") {
                let expr_str = &line[12..].trim();
                let expr = Self::parse_expr(expr_str)?;
                model.add_irreflexive(expr);
            } else if line.starts_with("empty ") {
                let expr_str = &line[6..].trim();
                let expr = Self::parse_expr(expr_str)?;
                model.add_empty(expr);
            } else if !line.starts_with("model") {
                return Err(format!("unrecognized statement: {}", line));
            }
        }

        Ok(model)
    }

    /// Parse a relation expression.
    fn parse_expr(input: &str) -> Result<RelationExpr, String> {
        let input = input.trim();
        if input.is_empty() {
            return Err("empty expression".into());
        }

        // Handle union (lowest precedence).
        if let Some(pos) = Self::find_operator(input, '|') {
            let left = Self::parse_expr(&input[..pos])?;
            let right = Self::parse_expr(&input[pos + 1..])?;
            return Ok(RelationExpr::union(left, right));
        }

        // Handle intersection.
        if let Some(pos) = Self::find_operator(input, '&') {
            let left = Self::parse_expr(&input[..pos])?;
            let right = Self::parse_expr(&input[pos + 1..])?;
            return Ok(RelationExpr::inter(left, right));
        }

        // Handle sequence.
        if let Some(pos) = Self::find_operator(input, ';') {
            let left = Self::parse_expr(&input[..pos])?;
            let right = Self::parse_expr(&input[pos + 1..])?;
            return Ok(RelationExpr::seq(left, right));
        }

        // Handle postfix operators.
        if input.ends_with("^-1") {
            let inner = Self::parse_expr(&input[..input.len() - 3])?;
            return Ok(RelationExpr::inverse(inner));
        }
        if input.ends_with('+') {
            let inner = Self::parse_expr(&input[..input.len() - 1])?;
            return Ok(RelationExpr::plus(inner));
        }
        if input.ends_with('*') {
            let inner = Self::parse_expr(&input[..input.len() - 1])?;
            return Ok(RelationExpr::star(inner));
        }
        if input.ends_with('?') {
            let inner = Self::parse_expr(&input[..input.len() - 1])?;
            return Ok(RelationExpr::optional(inner));
        }

        // Handle parentheses.
        if input.starts_with('(') && input.ends_with(')') {
            return Self::parse_expr(&input[1..input.len() - 1]);
        }

        // Handle filter [pred].
        if input.starts_with('[') && input.ends_with(']') {
            let pred_str = &input[1..input.len() - 1];
            let pred = Self::parse_predicate(pred_str)?;
            return Ok(RelationExpr::filter(pred));
        }

        // Atoms.
        match input {
            "id" => Ok(RelationExpr::Identity),
            "0"  => Ok(RelationExpr::Empty),
            _    => Ok(RelationExpr::base(input)),
        }
    }

    /// Parse a predicate expression.
    fn parse_predicate(input: &str) -> Result<PredicateExpr, String> {
        let input = input.trim();
        match input {
            "R" | "Read" => Ok(PredicateExpr::IsRead),
            "W" | "Write" => Ok(PredicateExpr::IsWrite),
            "F" | "Fence" => Ok(PredicateExpr::IsFence),
            "RMW" => Ok(PredicateExpr::IsRMW),
            "true" | "*" => Ok(PredicateExpr::True),
            _ => {
                if input.starts_with('!') {
                    let inner = Self::parse_predicate(&input[1..])?;
                    Ok(PredicateExpr::not(inner))
                } else {
                    // Treat as opaque — default to True.
                    Ok(PredicateExpr::True)
                }
            }
        }
    }

    /// Find operator at the top level (not inside parentheses or brackets).
    fn find_operator(input: &str, op: char) -> Option<usize> {
        let mut depth = 0i32;
        let bytes = input.as_bytes();
        // Scan right to left for left-associativity.
        for i in (0..bytes.len()).rev() {
            match bytes[i] {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                c if c == op as u8 && depth == 0 => return Some(i),
                _ => {}
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::execution::ExecutionGraphBuilder;

    #[test]
    fn test_relation_expr_display() {
        let e = RelationExpr::seq(RelationExpr::base("po"), RelationExpr::base("rf"));
        assert_eq!(format!("{}", e), "(po ; rf)");

        let e2 = RelationExpr::union(
            RelationExpr::base("rf"),
            RelationExpr::inverse(RelationExpr::base("co")),
        );
        assert_eq!(format!("{}", e2), "(rf | co^-1)");
    }

    #[test]
    fn test_relation_expr_referenced_bases() {
        let e = RelationExpr::union(
            RelationExpr::seq(RelationExpr::base("po"), RelationExpr::base("rf")),
            RelationExpr::plus(RelationExpr::base("co")),
        );
        let bases = e.referenced_bases();
        assert_eq!(bases, vec!["co", "po", "rf"]);
    }

    #[test]
    fn test_predicate_eval() {
        use super::super::execution::{Event, OpType};
        let r = Event::new(0, 0, OpType::Read, 0x100, 1);
        let w = Event::new(1, 0, OpType::Write, 0x100, 2);

        assert!(PredicateExpr::IsRead.eval(&r));
        assert!(!PredicateExpr::IsRead.eval(&w));
        assert!(PredicateExpr::IsWrite.eval(&w));
        assert!(PredicateExpr::True.eval(&r));
    }

    #[test]
    fn test_predicate_compound() {
        use super::super::execution::{Event, OpType};
        let r = Event::new(0, 0, OpType::Read, 0x100, 1);
        let p = PredicateExpr::and(PredicateExpr::IsRead, PredicateExpr::AtAddress(0x100));
        assert!(p.eval(&r));
        let p2 = PredicateExpr::and(PredicateExpr::IsRead, PredicateExpr::AtAddress(0x200));
        assert!(!p2.eval(&r));
    }

    #[test]
    fn test_sc_model() {
        let sc = BuiltinModel::SC.build();
        assert_eq!(sc.name, "SC");
        assert!(!sc.constraints.is_empty());
        assert!(sc.validate().is_ok());
    }

    #[test]
    fn test_tso_model() {
        let tso = BuiltinModel::TSO.build();
        assert_eq!(tso.name, "TSO");
        assert!(tso.validate().is_ok());
    }

    #[test]
    fn test_pso_model() {
        let pso = BuiltinModel::PSO.build();
        assert_eq!(pso.name, "PSO");
        assert!(pso.validate().is_ok());
    }

    #[test]
    fn test_arm_model() {
        let arm = BuiltinModel::ARM.build();
        assert_eq!(arm.name, "ARMv8");
        assert!(arm.validate().is_ok());
    }

    #[test]
    fn test_riscv_model() {
        let rv = BuiltinModel::RISCV.build();
        assert_eq!(rv.name, "RISC-V");
        assert!(rv.validate().is_ok());
    }

    #[test]
    fn test_ptx_model() {
        let ptx = BuiltinModel::PTX.build();
        assert_eq!(ptx.name, "PTX");
        assert!(ptx.validate().is_ok());
    }

    #[test]
    fn test_webgpu_model() {
        let wgpu = BuiltinModel::WebGPU.build();
        assert_eq!(wgpu.name, "WebGPU");
        assert!(wgpu.validate().is_ok());
    }

    #[test]
    fn test_model_display() {
        let sc = BuiltinModel::SC.build();
        let s = format!("{}", sc);
        assert!(s.contains("SC"));
    }

    #[test]
    fn test_all_models() {
        for m in BuiltinModel::all() {
            let model = m.build();
            assert!(model.validate().is_ok(), "model {} failed validation", m.name());
        }
    }

    #[test]
    fn test_eval_base_relations() {
        let mut b = ExecutionGraphBuilder::new();
        let w0 = b.add_write(0, 0x100, 1);
        let r0 = b.add_read(0, 0x100, 1);
        let w1 = b.add_write(1, 0x100, 2);
        let mut g = b.build();
        g.add_rf(w0, r0);
        g.add_co(w0, w1);
        g.derive_fr();

        let sc = BuiltinModel::SC.build();
        let env = HashMap::new();
        let po = sc.eval_expr(&RelationExpr::base("po"), &g, &env);
        assert!(po.get(0, 1));

        let rf = sc.eval_expr(&RelationExpr::base("rf"), &g, &env);
        assert!(rf.get(0, 1));

        let id = sc.eval_expr(&RelationExpr::Identity, &g, &env);
        assert!(id.get(0, 0));
    }

    #[test]
    fn test_eval_derived_relations() {
        let mut b = ExecutionGraphBuilder::new();
        let w0 = b.add_write(0, 0x100, 1);
        let r0 = b.add_read(0, 0x100, 1);
        let w1 = b.add_write(1, 0x100, 2);
        let mut g = b.build();
        g.add_rf(w0, r0);
        g.add_co(w0, w1);
        g.derive_fr();

        let sc = BuiltinModel::SC.build();
        let env = sc.compute_derived(&g);
        assert!(env.contains_key("com"));
        let com = &env["com"];
        // com = rf ∪ co ∪ fr
        assert!(com.get(w0, r0)); // rf
        assert!(com.get(w0, w1)); // co
        assert!(com.get(r0, w1)); // fr
    }

    #[test]
    fn test_eval_filter() {
        let mut b = ExecutionGraphBuilder::new();
        b.add_write(0, 0x100, 1);
        b.add_read(0, 0x100, 1);
        b.add_write(1, 0x100, 2);
        let g = b.build();

        let sc = BuiltinModel::SC.build();
        let env = HashMap::new();
        let read_filter = sc.eval_expr(&RelationExpr::filter(PredicateExpr::IsRead), &g, &env);
        assert!(!read_filter.get(0, 0)); // write
        assert!(read_filter.get(1, 1));  // read
        assert!(!read_filter.get(2, 2)); // write
    }

    #[test]
    fn test_eval_sequence() {
        let mut b = ExecutionGraphBuilder::new();
        let w0 = b.add_write(0, 0x100, 1);
        let r0 = b.add_read(0, 0x100, 1);
        let w1 = b.add_write(1, 0x100, 2);
        let mut g = b.build();
        g.add_rf(w0, r0);
        g.add_co(w0, w1);
        g.derive_fr();

        let sc = BuiltinModel::SC.build();
        let env = HashMap::new();
        // rf;co should not exist (r0 is not a write so no co from it)
        let expr = RelationExpr::seq(RelationExpr::base("rf"), RelationExpr::base("co"));
        let result = sc.eval_expr(&expr, &g, &env);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dsl_parse_simple() {
        let input = r#"
model "TestModel"
let com = rf | co | fr
acyclic po | com
"#;
        let model = ModelParser::parse(input).unwrap();
        assert_eq!(model.name, "TestModel");
        assert_eq!(model.derived_relations.len(), 1);
        assert_eq!(model.constraints.len(), 1);
    }

    #[test]
    fn test_dsl_parse_sc() {
        let input = r#"
model "SC"
let com = rf | co | fr
acyclic po | com
"#;
        let model = ModelParser::parse(input).unwrap();
        assert_eq!(model.name, "SC");
    }

    #[test]
    fn test_dsl_parse_filter() {
        let input = r#"
model "Test"
let ppo = [R] ; po ; [W]
acyclic ppo | rf
"#;
        let model = ModelParser::parse(input).unwrap();
        assert_eq!(model.derived_relations.len(), 1);
    }

    #[test]
    fn test_dsl_parse_closures() {
        let input = r#"
model "Test"
let hb = (po | rf)+
irreflexive hb ; fr
"#;
        let model = ModelParser::parse(input).unwrap();
        assert_eq!(model.derived_relations.len(), 1);
        assert_eq!(model.constraints.len(), 1);
    }

    #[test]
    fn test_dsl_parse_empty_constraint() {
        let input = r#"
model "Test"
empty rmw & (fre ; coe)
"#;
        let model = ModelParser::parse(input).unwrap();
        assert_eq!(model.constraints.len(), 1);
    }

    #[test]
    fn test_model_validate_bad_ref() {
        let mut m = MemoryModel::new("Bad");
        m.add_derived("foo",
            RelationExpr::base("nonexistent_relation"),
            "bad ref",
        );
        let result = m.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_display() {
        let c = Constraint::acyclic(RelationExpr::union(
            RelationExpr::base("po"),
            RelationExpr::base("com"),
        ));
        let s = format!("{}", c);
        assert!(s.contains("acyclic"));
    }

    #[test]
    fn test_relation_def() {
        let rd = RelationDef::base("po", "program order");
        assert_eq!(rd.rel_type, RelationType::Base);

        let dd = RelationDef::derived("ppo", "preserved program order");
        assert_eq!(dd.rel_type, RelationType::Derived);
    }

    #[test]
    fn test_builtin_model_display() {
        assert_eq!(format!("{}", BuiltinModel::SC), "SC");
        assert_eq!(format!("{}", BuiltinModel::ARM), "ARMv8");
    }

    #[test]
    fn test_relation_expr_union_many() {
        let e = RelationExpr::union_many(vec![
            RelationExpr::base("a"),
            RelationExpr::base("b"),
            RelationExpr::base("c"),
        ]);
        let bases = e.referenced_bases();
        assert_eq!(bases, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_relation_expr_seq_many() {
        let e = RelationExpr::seq_many(vec![
            RelationExpr::base("a"),
            RelationExpr::base("b"),
        ]);
        let s = format!("{}", e);
        assert!(s.contains(";"));
    }

    #[test]
    fn test_scope_filter() {
        let sf = ScopeFilter::CTA;
        assert_eq!(format!("{:?}", sf), "CTA");
    }

    #[test]
    fn test_eval_po_loc() {
        let mut b = ExecutionGraphBuilder::new();
        b.add_write(0, 0x100, 1);
        b.add_read(0, 0x100, 1);
        b.add_write(0, 0x200, 2);
        let g = b.build();

        let sc = BuiltinModel::SC.build();
        let env = HashMap::new();
        let po_loc = sc.eval_expr(&RelationExpr::base("po-loc"), &g, &env);
        assert!(po_loc.get(0, 1)); // same address
        assert!(!po_loc.get(0, 2)); // different address
    }

    #[test]
    fn test_eval_ext_int() {
        let mut b = ExecutionGraphBuilder::new();
        b.add_write(0, 0x100, 1);
        b.add_read(0, 0x100, 1);
        b.add_write(1, 0x100, 2);
        let g = b.build();

        let sc = BuiltinModel::SC.build();
        let env = HashMap::new();

        let int = sc.eval_expr(&RelationExpr::base("int"), &g, &env);
        assert!(int.get(0, 1));
        assert!(!int.get(0, 2));

        let ext = sc.eval_expr(&RelationExpr::base("ext"), &g, &env);
        assert!(!ext.get(0, 1));
        assert!(ext.get(0, 2));
    }

    #[test]
    fn test_eval_rfe_rfi() {
        let mut b = ExecutionGraphBuilder::new();
        let w0 = b.add_write(0, 0x100, 1);
        let r0 = b.add_read(0, 0x100, 1);
        let r1 = b.add_read(1, 0x100, 1);
        let mut g = b.build();
        g.add_rf(w0, r0);
        g.add_rf(w0, r1);

        let sc = BuiltinModel::SC.build();
        let env = HashMap::new();

        let rfi = sc.eval_expr(&RelationExpr::base("rfi"), &g, &env);
        assert!(rfi.get(w0, r0));
        assert!(!rfi.get(w0, r1));

        let rfe = sc.eval_expr(&RelationExpr::base("rfe"), &g, &env);
        assert!(!rfe.get(w0, r0));
        assert!(rfe.get(w0, r1));
    }

    #[test]
    fn test_derived_relation_display() {
        let dr = DerivedRelation::new("ppo",
            RelationExpr::base("po"),
            "preserved program order",
        );
        let s = format!("{}", dr);
        assert!(s.contains("ppo"));
        assert!(s.contains("po"));
    }
}
