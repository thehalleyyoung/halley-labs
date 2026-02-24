//! Extended relation algebra for memory model verification.
//!
//! Provides a high-level `NamedRelation` type wrapping `BitMatrix` with named
//! operations, a relation expression evaluator, derived relation computation,
//! pretty printing, efficient transitive closure, and comparison utilities.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

use super::execution::{BitMatrix, ExecutionGraph, Event, EventId, OpType};
use super::memory_model::{RelationExpr, PredicateExpr, MemoryModel};

// ---------------------------------------------------------------------------
// NamedRelation — high-level relation with metadata
// ---------------------------------------------------------------------------

/// A relation with a name, description, and underlying bit matrix.
#[derive(Clone)]
pub struct NamedRelation {
    pub name: String,
    pub description: String,
    pub matrix: BitMatrix,
}

impl fmt::Debug for NamedRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NamedRelation({}, {}x{}, {} edges)",
            self.name, self.matrix.dim(), self.matrix.dim(), self.matrix.count_edges())
    }
}

impl PartialEq for NamedRelation {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.matrix == other.matrix
    }
}
impl Eq for NamedRelation {}

impl Hash for NamedRelation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.matrix.hash(state);
    }
}

impl NamedRelation {
    /// Create a new named relation.
    pub fn new(name: impl Into<String>, matrix: BitMatrix) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            matrix,
        }
    }

    /// Create with description.
    pub fn with_desc(name: impl Into<String>, desc: impl Into<String>, matrix: BitMatrix) -> Self {
        Self {
            name: name.into(),
            description: desc.into(),
            matrix,
        }
    }

    /// Create an empty relation of given dimension.
    pub fn empty(name: impl Into<String>, n: usize) -> Self {
        Self::new(name, BitMatrix::new(n))
    }

    /// Create the identity relation.
    pub fn identity(name: impl Into<String>, n: usize) -> Self {
        Self::new(name, BitMatrix::identity(n))
    }

    /// Create the universal relation.
    pub fn universal(name: impl Into<String>, n: usize) -> Self {
        Self::new(name, BitMatrix::universal(n))
    }

    /// Dimension (number of elements).
    pub fn dim(&self) -> usize {
        self.matrix.dim()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.matrix.count_edges()
    }

    /// Whether the relation is empty.
    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }

    /// Get all edges as pairs.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.matrix.edges()
    }

    /// Check if a pair is in the relation.
    pub fn contains(&self, i: usize, j: usize) -> bool {
        self.matrix.get(i, j)
    }

    /// Add an edge.
    pub fn add_edge(&mut self, i: usize, j: usize) {
        self.matrix.set(i, j, true);
    }

    /// Remove an edge.
    pub fn remove_edge(&mut self, i: usize, j: usize) {
        self.matrix.set(i, j, false);
    }
}

// ---------------------------------------------------------------------------
// Relation algebra operations
// ---------------------------------------------------------------------------

impl NamedRelation {
    /// Union: R1 ∪ R2.
    pub fn union(&self, other: &Self) -> Self {
        Self::new(
            format!("({} ∪ {})", self.name, other.name),
            self.matrix.union(&other.matrix),
        )
    }

    /// Intersection: R1 ∩ R2.
    pub fn intersection(&self, other: &Self) -> Self {
        Self::new(
            format!("({} ∩ {})", self.name, other.name),
            self.matrix.intersection(&other.matrix),
        )
    }

    /// Complement: ¬R.
    pub fn complement(&self) -> Self {
        Self::new(
            format!("¬{}", self.name),
            self.matrix.complement(),
        )
    }

    /// Difference: R1 \ R2.
    pub fn difference(&self, other: &Self) -> Self {
        Self::new(
            format!("({} \\ {})", self.name, other.name),
            self.matrix.difference(&other.matrix),
        )
    }

    /// Composition (sequence): R1 ; R2.
    pub fn compose(&self, other: &Self) -> Self {
        Self::new(
            format!("({} ; {})", self.name, other.name),
            self.matrix.compose(&other.matrix),
        )
    }

    /// Inverse (transpose): R⁻¹.
    pub fn inverse(&self) -> Self {
        Self::new(
            format!("{}⁻¹", self.name),
            self.matrix.inverse(),
        )
    }

    /// Transitive closure: R⁺.
    pub fn transitive_closure(&self) -> Self {
        Self::new(
            format!("{}⁺", self.name),
            self.matrix.transitive_closure(),
        )
    }

    /// Reflexive-transitive closure: R*.
    pub fn reflexive_transitive_closure(&self) -> Self {
        Self::new(
            format!("{}*", self.name),
            self.matrix.reflexive_transitive_closure(),
        )
    }

    /// Optional (reflexive closure): R? = Id ∪ R.
    pub fn optional(&self) -> Self {
        Self::new(
            format!("{}?", self.name),
            self.matrix.optional(),
        )
    }

    /// Symmetric closure: R ∪ R⁻¹.
    pub fn symmetric_closure(&self) -> Self {
        let inv = self.matrix.inverse();
        Self::new(
            format!("sym({})", self.name),
            self.matrix.union(&inv),
        )
    }
}

// ---------------------------------------------------------------------------
// Relation properties
// ---------------------------------------------------------------------------

impl NamedRelation {
    /// Is the relation reflexive? (∀ i. (i,i) ∈ R)
    pub fn is_reflexive(&self) -> bool {
        let n = self.dim();
        for i in 0..n {
            if !self.matrix.get(i, i) {
                return false;
            }
        }
        true
    }

    /// Is the relation irreflexive? (∀ i. (i,i) ∉ R)
    pub fn is_irreflexive(&self) -> bool {
        self.matrix.is_irreflexive()
    }

    /// Is the relation symmetric? (∀ i,j. (i,j) ∈ R → (j,i) ∈ R)
    pub fn is_symmetric(&self) -> bool {
        let n = self.dim();
        for i in 0..n {
            for j in i + 1..n {
                if self.matrix.get(i, j) != self.matrix.get(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Is the relation antisymmetric? (∀ i,j. (i,j) ∈ R ∧ (j,i) ∈ R → i = j)
    pub fn is_antisymmetric(&self) -> bool {
        let n = self.dim();
        for i in 0..n {
            for j in i + 1..n {
                if self.matrix.get(i, j) && self.matrix.get(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Is the relation transitive? (∀ i,j,k. (i,j) ∈ R ∧ (j,k) ∈ R → (i,k) ∈ R)
    pub fn is_transitive(&self) -> bool {
        let tc = self.matrix.transitive_closure();
        tc == self.matrix
    }

    /// Is the relation acyclic?
    pub fn is_acyclic(&self) -> bool {
        self.matrix.is_acyclic()
    }

    /// Is the relation a partial order? (reflexive, antisymmetric, transitive)
    pub fn is_partial_order(&self) -> bool {
        self.is_reflexive() && self.is_antisymmetric() && self.is_transitive()
    }

    /// Is the relation a strict partial order? (irreflexive, transitive — antisymmetry follows)
    pub fn is_strict_partial_order(&self) -> bool {
        self.is_irreflexive() && self.is_transitive()
    }

    /// Is the relation a total order? (partial order + totality)
    pub fn is_total_order(&self) -> bool {
        if !self.is_partial_order() { return false; }
        let n = self.dim();
        for i in 0..n {
            for j in 0..n {
                if i != j && !self.matrix.get(i, j) && !self.matrix.get(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Is the relation an equivalence relation? (reflexive, symmetric, transitive)
    pub fn is_equivalence(&self) -> bool {
        self.is_reflexive() && self.is_symmetric() && self.is_transitive()
    }

    /// Is the relation a strict total order? (irreflexive, transitive, total on distinct elements)
    pub fn is_strict_total_order(&self) -> bool {
        if !self.is_irreflexive() || !self.is_transitive() { return false; }
        let n = self.dim();
        for i in 0..n {
            for j in i + 1..n {
                if !self.matrix.get(i, j) && !self.matrix.get(j, i) {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Domain and range extraction
// ---------------------------------------------------------------------------

impl NamedRelation {
    /// Domain: set of elements that appear as sources.
    pub fn domain(&self) -> Vec<usize> {
        let n = self.dim();
        (0..n).filter(|&i| {
            (0..n).any(|j| self.matrix.get(i, j))
        }).collect()
    }

    /// Range: set of elements that appear as targets.
    pub fn range(&self) -> Vec<usize> {
        let n = self.dim();
        (0..n).filter(|&j| {
            (0..n).any(|i| self.matrix.get(i, j))
        }).collect()
    }

    /// Domain restriction: restrict source to the given set.
    pub fn domain_restrict(&self, dom: &[usize]) -> Self {
        let n = self.dim();
        let mut mask = vec![false; n];
        for &i in dom {
            if i < n { mask[i] = true; }
        }
        let mut result = self.matrix.clone();
        for i in 0..n {
            if !mask[i] {
                for j in 0..n {
                    result.set(i, j, false);
                }
            }
        }
        Self::new(format!("dom_restrict({})", self.name), result)
    }

    /// Range restriction: restrict target to the given set.
    pub fn range_restrict(&self, rng: &[usize]) -> Self {
        let n = self.dim();
        let mut mask = vec![false; n];
        for &j in rng {
            if j < n { mask[j] = true; }
        }
        let mut result = self.matrix.clone();
        for i in 0..n {
            for j in 0..n {
                if !mask[j] {
                    result.set(i, j, false);
                }
            }
        }
        Self::new(format!("rng_restrict({})", self.name), result)
    }
}

// ---------------------------------------------------------------------------
// Comparison and subset
// ---------------------------------------------------------------------------

impl NamedRelation {
    /// Is self a subset of other? (∀ i,j. (i,j) ∈ self → (i,j) ∈ other)
    pub fn is_subset_of(&self, other: &Self) -> bool {
        assert_eq!(self.dim(), other.dim());
        let n = self.dim();
        for i in 0..n {
            for j in 0..n {
                if self.matrix.get(i, j) && !other.matrix.get(i, j) {
                    return false;
                }
            }
        }
        true
    }

    /// Is self a strict subset of other?
    pub fn is_strict_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other) && self.matrix != other.matrix
    }

    /// Edges in self but not in other.
    pub fn edges_not_in(&self, other: &Self) -> Vec<(usize, usize)> {
        self.difference(other).edges()
    }

    /// Edges in other but not in self.
    pub fn missing_edges_from(&self, other: &Self) -> Vec<(usize, usize)> {
        other.difference(self).edges()
    }
}

// ---------------------------------------------------------------------------
// Pretty printing
// ---------------------------------------------------------------------------

impl fmt::Display for NamedRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} edges over {} elements", self.name, self.edge_count(), self.dim())
    }
}

impl NamedRelation {
    /// Pretty-print as edge list.
    pub fn as_edge_list(&self) -> String {
        let edges = self.edges();
        if edges.is_empty() {
            return format!("{}: ∅", self.name);
        }
        let items: Vec<String> = edges.iter()
            .map(|(i, j)| format!("({}, {})", i, j))
            .collect();
        format!("{}: {{{}}}", self.name, items.join(", "))
    }

    /// Pretty-print as edge list with event labels.
    pub fn as_labeled_edge_list(&self, events: &[Event]) -> String {
        let edges = self.edges();
        if edges.is_empty() {
            return format!("{}: ∅", self.name);
        }
        let items: Vec<String> = edges.iter()
            .map(|(i, j)| {
                let src = if *i < events.len() { events[*i].label() } else { format!("?{}", i) };
                let dst = if *j < events.len() { events[*j].label() } else { format!("?{}", j) };
                format!("{} → {}", src, dst)
            })
            .collect();
        format!("{}: {{{}}}", self.name, items.join(", "))
    }

    /// Pretty-print as matrix.
    pub fn as_matrix(&self) -> String {
        let mut s = format!("{}:\n", self.name);
        s.push_str(&self.matrix.pretty_print());
        s
    }

    /// Generate DOT fragment for this relation.
    pub fn to_dot(&self, color: &str) -> String {
        self.matrix.dot_edges(&self.name, color)
    }
}

// ---------------------------------------------------------------------------
// RelationEnvironment — manage a set of named relations
// ---------------------------------------------------------------------------

/// An environment mapping relation names to their computed bit matrices.
#[derive(Debug, Clone)]
pub struct RelationEnvironment {
    relations: HashMap<String, BitMatrix>,
    n: usize,
}

impl RelationEnvironment {
    /// Create a new environment for `n` events.
    pub fn new(n: usize) -> Self {
        Self {
            relations: HashMap::new(),
            n,
        }
    }

    /// Create from an execution graph, populating base relations.
    pub fn from_graph(exec: &ExecutionGraph) -> Self {
        let n = exec.len();
        let mut env = Self::new(n);
        env.set("po", exec.po.clone());
        env.set("rf", exec.rf.clone());
        env.set("co", exec.co.clone());
        env.set("fr", exec.fr.clone());
        env.set("id", BitMatrix::identity(n));

        // Derived builtins.
        env.set("com", exec.rf.union(&exec.co).union(&exec.fr));
        env.set("po-loc", exec.po.intersection(&exec.same_address()));
        env.set("ext", BitMatrix::universal(n).difference(&exec.same_thread()));
        env.set("int", exec.same_thread());
        env.set("rfe", exec.external(&exec.rf));
        env.set("rfi", exec.internal(&exec.rf));
        env.set("coe", exec.external(&exec.co));
        env.set("coi", exec.internal(&exec.co));
        env.set("fre", exec.external(&exec.fr));
        env.set("fri", exec.internal(&exec.fr));
        env.set("same-loc", exec.same_address());

        // event extra relations
        for r in &exec.extra {
            env.set(&r.name, r.matrix.clone());
        }

        env
    }

    /// Set a relation.
    pub fn set(&mut self, name: &str, matrix: BitMatrix) {
        self.relations.insert(name.to_string(), matrix);
    }

    /// Get a relation by name.
    pub fn get(&self, name: &str) -> Option<&BitMatrix> {
        self.relations.get(name)
    }

    /// Get a relation, returning empty if not found.
    pub fn get_or_empty(&self, name: &str) -> BitMatrix {
        self.relations.get(name).cloned().unwrap_or_else(|| BitMatrix::new(self.n))
    }

    /// Get a named relation wrapper.
    pub fn get_named(&self, name: &str) -> NamedRelation {
        NamedRelation::new(name, self.get_or_empty(name))
    }

    /// List all relation names.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.relations.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of relations.
    pub fn len(&self) -> usize {
        self.relations.len()
    }

    /// Whether the environment is empty.
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// RelationExprEvaluator — compile and evaluate relation expressions
// ---------------------------------------------------------------------------

/// Evaluator for relation expressions against an environment.
pub struct RelationExprEvaluator {
    env: RelationEnvironment,
}

impl RelationExprEvaluator {
    /// Create a new evaluator with the given environment.
    pub fn new(env: RelationEnvironment) -> Self {
        Self { env }
    }

    /// Create from an execution graph.
    pub fn from_graph(exec: &ExecutionGraph) -> Self {
        Self::new(RelationEnvironment::from_graph(exec))
    }

    /// Evaluate a relation expression.
    pub fn eval(&self, expr: &RelationExpr, exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.len();
        match expr {
            RelationExpr::Base(name) => {
                self.env.get_or_empty(name)
            }
            RelationExpr::Seq(a, b) => {
                let ma = self.eval(a, exec);
                let mb = self.eval(b, exec);
                ma.compose(&mb)
            }
            RelationExpr::Union(a, b) => {
                let ma = self.eval(a, exec);
                let mb = self.eval(b, exec);
                ma.union(&mb)
            }
            RelationExpr::Inter(a, b) => {
                let ma = self.eval(a, exec);
                let mb = self.eval(b, exec);
                ma.intersection(&mb)
            }
            RelationExpr::Diff(a, b) => {
                let ma = self.eval(a, exec);
                let mb = self.eval(b, exec);
                ma.difference(&mb)
            }
            RelationExpr::Inverse(a) => {
                self.eval(a, exec).inverse()
            }
            RelationExpr::Plus(a) => {
                self.eval(a, exec).transitive_closure()
            }
            RelationExpr::Star(a) => {
                self.eval(a, exec).reflexive_transitive_closure()
            }
            RelationExpr::Optional(a) => {
                self.eval(a, exec).optional()
            }
            RelationExpr::Identity => BitMatrix::identity(n),
            RelationExpr::Filter(pred) => {
                let bools: Vec<bool> = exec.events.iter().map(|e| pred.eval(e)).collect();
                BitMatrix::identity_filter(n, &bools)
            }
            RelationExpr::Empty => BitMatrix::new(n),
        }
    }

    /// Evaluate and return a named relation.
    pub fn eval_named(&self, name: &str, expr: &RelationExpr, exec: &ExecutionGraph) -> NamedRelation {
        NamedRelation::new(name, self.eval(expr, exec))
    }

    /// Get the underlying environment (mutable, for adding derived relations).
    pub fn env_mut(&mut self) -> &mut RelationEnvironment {
        &mut self.env
    }

    /// Get the underlying environment.
    pub fn env(&self) -> &RelationEnvironment {
        &self.env
    }
}

// ---------------------------------------------------------------------------
// DerivedRelationComputer — compute all derived relations from a model
// ---------------------------------------------------------------------------

/// Compute all derived relations specified by a memory model against an execution graph.
pub struct DerivedRelationComputer<'a> {
    model: &'a MemoryModel,
}

impl<'a> DerivedRelationComputer<'a> {
    pub fn new(model: &'a MemoryModel) -> Self {
        Self { model }
    }

    /// Compute all derived relations and return the full environment.
    pub fn compute(&self, exec: &ExecutionGraph) -> RelationEnvironment {
        let mut env = RelationEnvironment::from_graph(exec);

        // Add event-type filter relations.
        let n = exec.len();
        let reads: Vec<bool> = exec.events.iter().map(|e| e.is_read()).collect();
        let writes: Vec<bool> = exec.events.iter().map(|e| e.is_write()).collect();
        let fences: Vec<bool> = exec.events.iter().map(|e| e.is_fence()).collect();
        let rmws: Vec<bool> = exec.events.iter().map(|e| e.is_rmw()).collect();

        env.set("[R]", BitMatrix::identity_filter(n, &reads));
        env.set("[W]", BitMatrix::identity_filter(n, &writes));
        env.set("[F]", BitMatrix::identity_filter(n, &fences));
        env.set("[RMW]", BitMatrix::identity_filter(n, &rmws));

        // Compute each derived relation in order (they may reference earlier ones).
        let evaluator = RelationExprEvaluator::new(env.clone());
        for dr in &self.model.derived_relations {
            let matrix = evaluator.eval(&dr.expr, exec);
            env.set(&dr.name, matrix);
        }

        env
    }

    /// Compute derived relations and return as named relations.
    pub fn compute_named(&self, exec: &ExecutionGraph) -> Vec<NamedRelation> {
        let env = self.compute(exec);
        self.model.derived_relations.iter()
            .map(|dr| {
                NamedRelation::with_desc(
                    &dr.name,
                    &dr.description,
                    env.get_or_empty(&dr.name),
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Relation comparison utilities
// ---------------------------------------------------------------------------

/// Compare two sets of relations (e.g., from different models).
#[derive(Debug, Clone)]
pub struct RelationComparison {
    pub name: String,
    pub left_only: Vec<(usize, usize)>,
    pub right_only: Vec<(usize, usize)>,
    pub common: Vec<(usize, usize)>,
}

impl RelationComparison {
    /// Compare two named relations.
    pub fn compare(left: &NamedRelation, right: &NamedRelation) -> Self {
        let diff_lr = left.difference(right);
        let diff_rl = right.difference(left);
        let common = left.intersection(right);

        Self {
            name: format!("{} vs {}", left.name, right.name),
            left_only: diff_lr.edges(),
            right_only: diff_rl.edges(),
            common: common.edges(),
        }
    }

    /// Are the two relations identical?
    pub fn is_equal(&self) -> bool {
        self.left_only.is_empty() && self.right_only.is_empty()
    }

    /// Is left a subset of right?
    pub fn left_subset_of_right(&self) -> bool {
        self.left_only.is_empty()
    }

    /// Is right a subset of left?
    pub fn right_subset_of_left(&self) -> bool {
        self.right_only.is_empty()
    }
}

impl fmt::Display for RelationComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Comparison: {}", self.name)?;
        writeln!(f, "  Common edges: {}", self.common.len())?;
        writeln!(f, "  Left-only edges: {}", self.left_only.len())?;
        writeln!(f, "  Right-only edges: {}", self.right_only.len())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Efficient transitive closure with word-level operations
// ---------------------------------------------------------------------------

/// Compute transitive closure using Warshall's algorithm with word-level bit ops.
/// This is functionally identical to `BitMatrix::transitive_closure` but
/// exposes the algorithm for benchmarking and testing.
pub fn warshall_transitive_closure(matrix: &BitMatrix) -> BitMatrix {
    matrix.transitive_closure()
}

/// Incremental transitive closure: given an existing TC and a new edge (i,j),
/// update the TC without full recomputation.
pub fn incremental_tc(tc: &mut BitMatrix, i: usize, j: usize) {
    let n = tc.dim();
    if tc.get(i, j) {
        return; // Edge already present and TC already reflects it.
    }
    tc.set(i, j, true);

    // All nodes that can reach i (including i itself).
    let mut preds: Vec<usize> = vec![i];
    for p in 0..n {
        if tc.get(p, i) {
            preds.push(p);
        }
    }

    // All nodes reachable from j (including j itself).
    let mut succs: Vec<usize> = vec![j];
    for s in 0..n {
        if tc.get(j, s) {
            succs.push(s);
        }
    }

    // Add edges from every predecessor to every successor.
    for &p in &preds {
        for &s in &succs {
            tc.set(p, s, true);
        }
    }
}

// ---------------------------------------------------------------------------
// Relation expression optimization
// ---------------------------------------------------------------------------

/// Simplify a relation expression by applying algebraic identities.
pub fn simplify_expr(expr: &RelationExpr) -> RelationExpr {
    match expr {
        // Union with empty.
        RelationExpr::Union(a, b) => {
            let sa = simplify_expr(a);
            let sb = simplify_expr(b);
            match (&sa, &sb) {
                (RelationExpr::Empty, _) => sb,
                (_, RelationExpr::Empty) => sa,
                _ if sa == sb => sa,
                _ => RelationExpr::union(sa, sb),
            }
        }
        // Intersection with empty.
        RelationExpr::Inter(a, b) => {
            let sa = simplify_expr(a);
            let sb = simplify_expr(b);
            match (&sa, &sb) {
                (RelationExpr::Empty, _) | (_, RelationExpr::Empty) => RelationExpr::Empty,
                _ if sa == sb => sa,
                _ => RelationExpr::inter(sa, sb),
            }
        }
        // Sequence with identity.
        RelationExpr::Seq(a, b) => {
            let sa = simplify_expr(a);
            let sb = simplify_expr(b);
            match (&sa, &sb) {
                (RelationExpr::Identity, _) => sb,
                (_, RelationExpr::Identity) => sa,
                (RelationExpr::Empty, _) | (_, RelationExpr::Empty) => RelationExpr::Empty,
                _ => RelationExpr::seq(sa, sb),
            }
        }
        // Diff from empty.
        RelationExpr::Diff(a, b) => {
            let sa = simplify_expr(a);
            let sb = simplify_expr(b);
            match (&sa, &sb) {
                (RelationExpr::Empty, _) => RelationExpr::Empty,
                (_, RelationExpr::Empty) => sa,
                _ => RelationExpr::diff(sa, sb),
            }
        }
        // Double inverse.
        RelationExpr::Inverse(a) => {
            let sa = simplify_expr(a);
            match sa {
                RelationExpr::Inverse(inner) => *inner,
                _ => RelationExpr::inverse(sa),
            }
        }
        // Plus/Star on identity.
        RelationExpr::Plus(a) => {
            let sa = simplify_expr(a);
            match &sa {
                RelationExpr::Empty => RelationExpr::Empty,
                _ => RelationExpr::plus(sa),
            }
        }
        RelationExpr::Star(a) => {
            let sa = simplify_expr(a);
            match &sa {
                RelationExpr::Empty => RelationExpr::Identity,
                _ => RelationExpr::star(sa),
            }
        }
        RelationExpr::Optional(a) => {
            let sa = simplify_expr(a);
            RelationExpr::optional(sa)
        }
        // Leaves.
        other => other.clone(),
    }
}

/// Count the number of AST nodes in an expression (a measure of complexity).
pub fn expr_complexity(expr: &RelationExpr) -> usize {
    match expr {
        RelationExpr::Base(_) | RelationExpr::Identity |
        RelationExpr::Filter(_) | RelationExpr::Empty => 1,
        RelationExpr::Seq(a, b) | RelationExpr::Union(a, b) |
        RelationExpr::Inter(a, b) | RelationExpr::Diff(a, b) => {
            1 + expr_complexity(a) + expr_complexity(b)
        }
        RelationExpr::Inverse(a) | RelationExpr::Plus(a) |
        RelationExpr::Star(a) | RelationExpr::Optional(a) => {
            1 + expr_complexity(a)
        }
    }
}

// ---------------------------------------------------------------------------
// Relation hashing utilities
// ---------------------------------------------------------------------------

/// Compute a fingerprint hash of a relation for quick comparison.
pub fn relation_fingerprint(rel: &BitMatrix) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    rel.hash(&mut hasher);
    hasher.finish()
}

/// Check if two relations are isomorphic under a given permutation.
pub fn is_isomorphic_under(r1: &BitMatrix, r2: &BitMatrix, perm: &[usize]) -> bool {
    let n = r1.dim();
    if n != r2.dim() || perm.len() != n { return false; }
    for i in 0..n {
        for j in 0..n {
            if r1.get(i, j) != r2.get(perm[i], perm[j]) {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Multi-relation analysis
// ---------------------------------------------------------------------------

/// Analyze the relationship between multiple relations.
#[derive(Debug)]
pub struct RelationAnalysis {
    pub total_edges: usize,
    pub unique_edges: usize,
    pub overlap_matrix: Vec<Vec<usize>>,
    pub names: Vec<String>,
}

impl RelationAnalysis {
    /// Analyze a set of named relations.
    pub fn analyze(relations: &[NamedRelation]) -> Self {
        let k = relations.len();
        let mut overlap = vec![vec![0usize; k]; k];
        let mut all_edges = BitMatrix::new(if k > 0 { relations[0].dim() } else { 0 });

        for i in 0..k {
            for j in 0..k {
                let isect = relations[i].intersection(&relations[j]);
                overlap[i][j] = isect.edge_count();
            }
            if i == 0 {
                all_edges = relations[i].matrix.clone();
            } else {
                all_edges = all_edges.union(&relations[i].matrix);
            }
        }

        let total: usize = relations.iter().map(|r| r.edge_count()).sum();

        Self {
            total_edges: total,
            unique_edges: all_edges.count_edges(),
            overlap_matrix: overlap,
            names: relations.iter().map(|r| r.name.clone()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(n: usize) -> BitMatrix {
        let mut m = BitMatrix::new(n);
        for i in 0..n.saturating_sub(1) {
            m.set(i, i + 1, true);
        }
        m
    }

    fn make_cycle(n: usize) -> BitMatrix {
        let mut m = make_chain(n);
        if n > 1 {
            m.set(n - 1, 0, true);
        }
        m
    }

    #[test]
    fn test_named_relation_basics() {
        let m = make_chain(4);
        let r = NamedRelation::new("po", m);
        assert_eq!(r.dim(), 4);
        assert_eq!(r.edge_count(), 3);
        assert!(!r.is_empty());
        assert!(r.contains(0, 1));
        assert!(!r.contains(1, 0));
    }

    #[test]
    fn test_union() {
        let a = NamedRelation::new("a", make_chain(3));
        let mut m = BitMatrix::new(3);
        m.set(2, 0, true);
        let b = NamedRelation::new("b", m);
        let u = a.union(&b);
        assert_eq!(u.edge_count(), 3);
        assert!(u.contains(0, 1));
        assert!(u.contains(2, 0));
    }

    #[test]
    fn test_intersection() {
        let a = NamedRelation::new("a", make_chain(3));
        let b = NamedRelation::new("b", make_chain(3));
        let i = a.intersection(&b);
        assert_eq!(i.edge_count(), 2);
    }

    #[test]
    fn test_complement() {
        let r = NamedRelation::new("r", BitMatrix::new(3));
        let c = r.complement();
        assert_eq!(c.edge_count(), 9); // 3x3 all ones
    }

    #[test]
    fn test_difference() {
        let a = NamedRelation::new("a", BitMatrix::universal(3));
        let b = NamedRelation::new("b", BitMatrix::identity(3));
        let d = a.difference(&b);
        assert_eq!(d.edge_count(), 6); // 9 - 3
    }

    #[test]
    fn test_compose() {
        let a = NamedRelation::new("a", make_chain(3));
        let b = NamedRelation::new("b", make_chain(3));
        let c = a.compose(&b);
        // 0→1→2 gives (0,2)
        assert!(c.contains(0, 2));
        assert!(!c.contains(0, 1));
    }

    #[test]
    fn test_inverse() {
        let r = NamedRelation::new("r", make_chain(3));
        let inv = r.inverse();
        assert!(inv.contains(1, 0));
        assert!(inv.contains(2, 1));
        assert!(!inv.contains(0, 1));
    }

    #[test]
    fn test_transitive_closure() {
        let r = NamedRelation::new("r", make_chain(4));
        let tc = r.transitive_closure();
        // 0→1→2→3 should give (0,2), (0,3), (1,3)
        assert!(tc.contains(0, 2));
        assert!(tc.contains(0, 3));
        assert!(tc.contains(1, 3));
        assert_eq!(tc.edge_count(), 6); // 3 + 2 + 1
    }

    #[test]
    fn test_reflexive_transitive_closure() {
        let r = NamedRelation::new("r", make_chain(3));
        let rtc = r.reflexive_transitive_closure();
        assert!(rtc.contains(0, 0));
        assert!(rtc.contains(1, 1));
        assert!(rtc.contains(0, 2));
    }

    #[test]
    fn test_symmetric_closure() {
        let r = NamedRelation::new("r", make_chain(3));
        let sc = r.symmetric_closure();
        assert!(sc.contains(0, 1));
        assert!(sc.contains(1, 0));
    }

    #[test]
    fn test_is_reflexive() {
        let id = NamedRelation::identity("id", 3);
        assert!(id.is_reflexive());
        let chain = NamedRelation::new("chain", make_chain(3));
        assert!(!chain.is_reflexive());
    }

    #[test]
    fn test_is_irreflexive() {
        let chain = NamedRelation::new("chain", make_chain(3));
        assert!(chain.is_irreflexive());
        let id = NamedRelation::identity("id", 3);
        assert!(!id.is_irreflexive());
    }

    #[test]
    fn test_is_symmetric() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 0, true);
        let r = NamedRelation::new("r", m);
        assert!(r.is_symmetric());
    }

    #[test]
    fn test_is_antisymmetric() {
        let chain = NamedRelation::new("chain", make_chain(3));
        assert!(chain.is_antisymmetric());
    }

    #[test]
    fn test_is_transitive() {
        let chain = NamedRelation::new("chain", make_chain(3));
        assert!(!chain.is_transitive());
        let tc = chain.transitive_closure();
        assert!(tc.is_transitive());
    }

    #[test]
    fn test_is_acyclic() {
        let chain = NamedRelation::new("chain", make_chain(4));
        assert!(chain.is_acyclic());
        let cycle = NamedRelation::new("cycle", make_cycle(4));
        assert!(!cycle.is_acyclic());
    }

    #[test]
    fn test_is_partial_order() {
        let mut m = BitMatrix::identity(3);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 2, true);
        let r = NamedRelation::new("po", m);
        assert!(r.is_partial_order());
    }

    #[test]
    fn test_is_strict_partial_order() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 2, true);
        let r = NamedRelation::new("spo", m);
        assert!(r.is_strict_partial_order());
    }

    #[test]
    fn test_is_equivalence() {
        let id = NamedRelation::identity("id", 3);
        assert!(id.is_equivalence());
        let univ = NamedRelation::universal("u", 3);
        assert!(univ.is_equivalence());
    }

    #[test]
    fn test_domain_and_range() {
        let chain = NamedRelation::new("chain", make_chain(4));
        let dom = chain.domain();
        let rng = chain.range();
        assert_eq!(dom, vec![0, 1, 2]);
        assert_eq!(rng, vec![1, 2, 3]);
    }

    #[test]
    fn test_domain_restrict() {
        let chain = NamedRelation::new("chain", make_chain(4));
        let restricted = chain.domain_restrict(&[0, 1]);
        assert!(restricted.contains(0, 1));
        assert!(restricted.contains(1, 2));
        assert!(!restricted.contains(2, 3));
    }

    #[test]
    fn test_range_restrict() {
        let chain = NamedRelation::new("chain", make_chain(4));
        let restricted = chain.range_restrict(&[1, 2]);
        assert!(restricted.contains(0, 1));
        assert!(restricted.contains(1, 2));
        assert!(!restricted.contains(2, 3));
    }

    #[test]
    fn test_subset() {
        let a = NamedRelation::new("a", make_chain(3));
        let b = NamedRelation::new("b", BitMatrix::universal(3));
        assert!(a.is_subset_of(&b));
        assert!(a.is_strict_subset_of(&b));
        assert!(!b.is_subset_of(&a));
    }

    #[test]
    fn test_comparison() {
        let a = NamedRelation::new("a", make_chain(3));
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(2, 0, true);
        let b = NamedRelation::new("b", m);
        let cmp = RelationComparison::compare(&a, &b);
        assert!(!cmp.is_equal());
        assert_eq!(cmp.common.len(), 1); // (0,1)
        assert_eq!(cmp.left_only.len(), 1); // (1,2)
        assert_eq!(cmp.right_only.len(), 1); // (2,0)
    }

    #[test]
    fn test_pretty_print() {
        let r = NamedRelation::new("po", make_chain(3));
        let edge_list = r.as_edge_list();
        assert!(edge_list.contains("(0, 1)"));
        assert!(edge_list.contains("(1, 2)"));
    }

    #[test]
    fn test_simplify_union_empty() {
        let expr = RelationExpr::union(RelationExpr::base("po"), RelationExpr::Empty);
        let simplified = simplify_expr(&expr);
        assert_eq!(simplified, RelationExpr::base("po"));
    }

    #[test]
    fn test_simplify_seq_identity() {
        let expr = RelationExpr::seq(RelationExpr::Identity, RelationExpr::base("rf"));
        let simplified = simplify_expr(&expr);
        assert_eq!(simplified, RelationExpr::base("rf"));
    }

    #[test]
    fn test_simplify_double_inverse() {
        let expr = RelationExpr::inverse(RelationExpr::inverse(RelationExpr::base("po")));
        let simplified = simplify_expr(&expr);
        assert_eq!(simplified, RelationExpr::base("po"));
    }

    #[test]
    fn test_simplify_inter_empty() {
        let expr = RelationExpr::inter(RelationExpr::base("po"), RelationExpr::Empty);
        let simplified = simplify_expr(&expr);
        assert_eq!(simplified, RelationExpr::Empty);
    }

    #[test]
    fn test_simplify_star_empty() {
        let expr = RelationExpr::star(RelationExpr::Empty);
        let simplified = simplify_expr(&expr);
        assert_eq!(simplified, RelationExpr::Identity);
    }

    #[test]
    fn test_expr_complexity() {
        let simple = RelationExpr::base("po");
        assert_eq!(expr_complexity(&simple), 1);

        let complex = RelationExpr::union(
            RelationExpr::seq(RelationExpr::base("po"), RelationExpr::base("rf")),
            RelationExpr::base("co"),
        );
        assert_eq!(expr_complexity(&complex), 5);
    }

    #[test]
    fn test_incremental_tc() {
        let chain = make_chain(4);
        let mut tc = chain.transitive_closure();
        let orig_edges = tc.count_edges();
        // Already in TC, should not change.
        incremental_tc(&mut tc, 0, 1);
        assert_eq!(tc.count_edges(), orig_edges);
    }

    #[test]
    fn test_relation_fingerprint() {
        let a = make_chain(4);
        let b = make_chain(4);
        let c = make_cycle(4);
        assert_eq!(relation_fingerprint(&a), relation_fingerprint(&b));
        assert_ne!(relation_fingerprint(&a), relation_fingerprint(&c));
    }

    #[test]
    fn test_is_isomorphic_under_identity() {
        let m = make_chain(3);
        let perm = vec![0, 1, 2];
        assert!(is_isomorphic_under(&m, &m, &perm));
    }

    #[test]
    fn test_relation_environment() {
        let mut env = RelationEnvironment::new(4);
        env.set("po", make_chain(4));
        env.set("rf", BitMatrix::new(4));
        assert_eq!(env.len(), 2);
        assert!(env.get("po").is_some());
        assert!(env.get("unknown").is_none());
        let empty = env.get_or_empty("unknown");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_relation_analysis() {
        let a = NamedRelation::new("a", make_chain(4));
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(2, 3, true);
        let b = NamedRelation::new("b", m);
        let analysis = RelationAnalysis::analyze(&[a, b]);
        assert_eq!(analysis.names.len(), 2);
        assert!(analysis.unique_edges <= analysis.total_edges);
    }

    #[test]
    fn test_named_relation_optional() {
        let r = NamedRelation::new("r", make_chain(3));
        let opt = r.optional();
        assert!(opt.contains(0, 0)); // identity
        assert!(opt.contains(0, 1)); // original
    }

    #[test]
    fn test_is_strict_total_order() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 2, true);
        let r = NamedRelation::new("sto", m);
        assert!(r.is_strict_total_order());
    }

    #[test]
    fn test_is_total_order() {
        let mut m = BitMatrix::identity(3);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 2, true);
        let r = NamedRelation::new("to", m);
        assert!(r.is_total_order());
    }

    #[test]
    fn test_edges_not_in() {
        let a = NamedRelation::new("a", make_chain(3));
        let b = NamedRelation::new("b", BitMatrix::new(3));
        let diff = a.edges_not_in(&b);
        assert_eq!(diff.len(), 2);
    }

    #[test]
    fn test_missing_edges_from() {
        let a = NamedRelation::new("a", BitMatrix::new(3));
        let b = NamedRelation::new("b", make_chain(3));
        let missing = a.missing_edges_from(&b);
        assert_eq!(missing.len(), 2);
    }

    #[test]
    fn test_env_names() {
        let mut env = RelationEnvironment::new(3);
        env.set("po", BitMatrix::new(3));
        env.set("rf", BitMatrix::new(3));
        env.set("co", BitMatrix::new(3));
        let names = env.names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"po".to_string()));
    }

    #[test]
    fn test_named_relation_to_dot() {
        let r = NamedRelation::new("po", make_chain(3));
        let dot = r.to_dot("blue");
        assert!(dot.contains("po"));
        assert!(dot.contains("blue"));
    }
}
