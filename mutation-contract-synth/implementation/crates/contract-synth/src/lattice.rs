//! # Specification Lattice
//!
//! Mathematical foundation for the MutSpec contract synthesis algorithm.
//!
//! The specification lattice orders logical formulas by entailment (logical strength).
//! The sigma function σ: P(M_kill) → Spec sends a subset S of killed mutants
//! to ∧_{m ∈ S} ¬E(m), where E(m) is the error predicate of mutant m.
//! This is a lattice homomorphism from the powerset lattice (ordered by ⊆) to the
//! specification lattice (ordered by ⊨).
//!
//! ## Lattice Structure
//!
//! - **Top** = `True` (weakest specification — everything satisfies it)
//! - **Bottom** = `False` (strongest specification — nothing satisfies it)
//! - **Meet** = conjunction (∧) — strengthens the spec
//! - **Join** = disjunction (∨) — weakens the spec
//! - **Order** = entailment: `a ≤ b` iff `a ⊨ b` (a is stronger than b)

use std::collections::{BTreeSet, HashSet};
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use shared_types::{
    Contract, ContractClause, ContractProvenance, ContractStrength, Formula, MutantId, Predicate,
    Relation, SynthesisTier, Term,
};

// ---------------------------------------------------------------------------
// LatticeElement
// ---------------------------------------------------------------------------

/// An element in the specification lattice: a formula together with the set of
/// mutant identifiers whose error predicates contributed to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeElement {
    /// The logical formula this element represents.
    formula: Formula,
    /// Mutant IDs whose negated error predicates were conjoined to form this element.
    provenance: BTreeSet<MutantId>,
    /// Cached formula size (AST node count).
    cached_size: Option<usize>,
    /// Cached formula depth.
    cached_depth: Option<usize>,
    /// Optional human-readable label for debugging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    label: Option<String>,
}

impl LatticeElement {
    // -- constructors -------------------------------------------------------

    /// Create a lattice element from a formula with no provenance.
    pub fn new(formula: Formula) -> Self {
        Self {
            formula,
            provenance: BTreeSet::new(),
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

    /// Create a lattice element with a single mutant provenance.
    pub fn from_mutant(formula: Formula, mutant_id: MutantId) -> Self {
        let mut provenance = BTreeSet::new();
        provenance.insert(mutant_id);
        Self {
            formula,
            provenance,
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

    /// Create a lattice element with multiple mutant provenances.
    pub fn from_mutants(formula: Formula, mutant_ids: impl IntoIterator<Item = MutantId>) -> Self {
        Self {
            formula,
            provenance: mutant_ids.into_iter().collect(),
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

    /// The lattice top: `True` (weakest spec).
    pub fn top() -> Self {
        Self::new(Formula::True)
    }

    /// The lattice bottom: `False` (strongest spec, unsatisfiable).
    pub fn bottom() -> Self {
        Self::new(Formula::False)
    }

    /// Create the element for a single negated error predicate ¬E(m).
    pub fn from_error_predicate(error_pred: Formula, mutant_id: MutantId) -> Self {
        let negated = Formula::not(error_pred);
        Self::from_mutant(negated, mutant_id)
    }

    // -- accessors ----------------------------------------------------------

    pub fn formula(&self) -> &Formula {
        &self.formula
    }

    pub fn into_formula(self) -> Formula {
        self.formula
    }

    pub fn provenance(&self) -> &BTreeSet<MutantId> {
        &self.provenance
    }

    pub fn mutant_count(&self) -> usize {
        self.provenance.len()
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    // -- formula metrics ----------------------------------------------------

    /// AST node count of the formula.
    pub fn size(&mut self) -> usize {
        if let Some(s) = self.cached_size {
            return s;
        }
        let s = self.formula.size();
        self.cached_size = Some(s);
        s
    }

    /// Size without mutating (recomputes if not cached).
    pub fn size_hint(&self) -> usize {
        self.cached_size.unwrap_or_else(|| self.formula.size())
    }

    /// Maximum nesting depth of the formula.
    pub fn depth(&mut self) -> usize {
        if let Some(d) = self.cached_depth {
            return d;
        }
        let d = self.formula.depth();
        self.cached_depth = Some(d);
        d
    }

    pub fn depth_hint(&self) -> usize {
        self.cached_depth.unwrap_or_else(|| self.formula.depth())
    }

    /// Free variables appearing in the formula.
    pub fn free_vars(&self) -> BTreeSet<String> {
        self.formula.free_vars()
    }

    /// Complexity score: weighted combination of size, depth, and variable count.
    pub fn complexity(&self) -> f64 {
        let sz = self.formula.size() as f64;
        let dp = self.formula.depth() as f64;
        let vars = self.formula.free_vars().len() as f64;
        sz + 2.0 * dp + 1.5 * vars
    }

    pub fn is_top(&self) -> bool {
        self.formula.is_true()
    }

    pub fn is_bottom(&self) -> bool {
        self.formula.is_false()
    }

    /// True if this element is an atomic predicate (single literal).
    pub fn is_atomic(&self) -> bool {
        matches!(&self.formula, Formula::Atom(_) | Formula::Not(box Formula::Atom(_)))
    }

    // -- lattice operations -------------------------------------------------

    /// Meet (conjunction) of two elements.  Merges provenance.
    pub fn meet(&self, other: &LatticeElement) -> LatticeElement {
        if self.is_bottom() || other.is_bottom() {
            let mut bot = LatticeElement::bottom();
            bot.provenance = self.provenance.union(&other.provenance).cloned().collect();
            return bot;
        }
        if self.is_top() {
            return other.clone();
        }
        if other.is_top() {
            return self.clone();
        }

        let formula = Formula::and(vec![self.formula.clone(), other.formula.clone()]);
        let provenance = self.provenance.union(&other.provenance).cloned().collect();
        LatticeElement {
            formula,
            provenance,
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

    /// Join (disjunction) of two elements.  Intersects provenance.
    pub fn join(&self, other: &LatticeElement) -> LatticeElement {
        if self.is_top() || other.is_top() {
            let mut top = LatticeElement::top();
            top.provenance = self
                .provenance
                .intersection(&other.provenance)
                .cloned()
                .collect();
            return top;
        }
        if self.is_bottom() {
            return other.clone();
        }
        if other.is_bottom() {
            return self.clone();
        }

        let formula = Formula::or(vec![self.formula.clone(), other.formula.clone()]);
        let provenance = self
            .provenance
            .intersection(&other.provenance)
            .cloned()
            .collect();
        LatticeElement {
            formula,
            provenance,
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

    /// Negate the formula (logical complement).
    pub fn negate(&self) -> LatticeElement {
        let formula = Formula::not(self.formula.clone());
        LatticeElement {
            formula,
            provenance: self.provenance.clone(),
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

    /// Simplify the underlying formula (syntactic simplification).
    pub fn simplify(&self) -> LatticeElement {
        let simplified = self.formula.simplify();
        LatticeElement {
            formula: simplified,
            provenance: self.provenance.clone(),
            cached_size: None,
            cached_depth: None,
            label: self.label.clone(),
        }
    }

    /// Add mutant provenance to this element.
    pub fn add_provenance(&mut self, id: MutantId) {
        self.provenance.insert(id);
    }

    /// Merge provenance from another element.
    pub fn merge_provenance(&mut self, other: &LatticeElement) {
        self.provenance.extend(other.provenance.iter().cloned());
    }

    /// Convert this lattice element to a postcondition contract clause.
    pub fn to_ensures_clause(&self) -> ContractClause {
        ContractClause::Ensures(self.formula.clone())
    }

    /// Convert this lattice element to a precondition contract clause.
    pub fn to_requires_clause(&self) -> ContractClause {
        ContractClause::Requires(self.formula.clone())
    }

    /// Build a [`Contract`] from this element for a given function.
    pub fn to_contract(&self, function_name: &str) -> Contract {
        let clause = self.to_ensures_clause();
        let prov = ContractProvenance {
            targeted_mutants: self.provenance.iter().cloned().collect(),
            tier: SynthesisTier::Tier1LatticeWalk,
            solver_queries: 0,
            synthesis_time_ms: 0.0,
        };
        Contract {
            function_name: function_name.to_string(),
            clauses: vec![clause],
            provenance: vec![prov],
            strength: ContractStrength::Weak,
            verified: false,
        }
    }
}

impl PartialEq for LatticeElement {
    fn eq(&self, other: &Self) -> bool {
        self.formula == other.formula
    }
}

impl Eq for LatticeElement {}

impl fmt::Display for LatticeElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref lbl) = self.label {
            write!(f, "{lbl}: ")?;
        }
        write!(f, "{:?}", self.formula)?;
        if !self.provenance.is_empty() {
            let ids: Vec<String> = self.provenance.iter().map(|m| m.short()).collect();
            write!(f, " [mutants: {}]", ids.join(", "))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SpecLattice
// ---------------------------------------------------------------------------

/// The specification lattice ordered by logical entailment.
///
/// Elements are logical formulas; the ordering `a ≤ b` means `a ⊨ b`
/// (a is at least as strong as b).
///
/// - Top = `True`  (weakest)
/// - Bottom = `False` (strongest / inconsistent)
/// - Meet = ∧  (conjunction, strengthening)
/// - Join = ∨  (disjunction, weakening)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecLattice {
    /// All elements that have been materialised in this lattice.
    elements: Vec<LatticeElement>,
    /// Optional upper bound on formula complexity for elements we keep.
    max_complexity: Option<f64>,
    /// Whether to apply syntactic simplification after every lattice op.
    auto_simplify: bool,
    /// Counter for SMT queries issued for entailment checks.
    entailment_queries: u64,
}

impl SpecLattice {
    /// Create a new, empty specification lattice.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            max_complexity: None,
            auto_simplify: true,
            entailment_queries: 0,
        }
    }

    /// Create a lattice with a complexity bound.
    pub fn with_max_complexity(max_complexity: f64) -> Self {
        Self {
            elements: Vec::new(),
            max_complexity: Some(max_complexity),
            auto_simplify: true,
            entailment_queries: 0,
        }
    }

    /// Enable or disable automatic simplification after lattice operations.
    pub fn set_auto_simplify(&mut self, enable: bool) {
        self.auto_simplify = enable;
    }

    // -- top / bottom -------------------------------------------------------

    /// The top element: `True`.
    pub fn top(&self) -> LatticeElement {
        LatticeElement::top()
    }

    /// The bottom element: `False`.
    pub fn bottom(&self) -> LatticeElement {
        LatticeElement::bottom()
    }

    pub fn is_top(elem: &LatticeElement) -> bool {
        elem.is_top()
    }

    pub fn is_bottom(elem: &LatticeElement) -> bool {
        elem.is_bottom()
    }

    // -- lattice operations -------------------------------------------------

    /// Meet (∧) of two lattice elements.  Returns a (possibly simplified) conjunction.
    pub fn meet(&self, a: &LatticeElement, b: &LatticeElement) -> LatticeElement {
        let result = a.meet(b);
        self.maybe_simplify(result)
    }

    /// Meet of an arbitrary collection of elements.
    pub fn meet_all(&self, elems: impl IntoIterator<Item = LatticeElement>) -> LatticeElement {
        let mut acc = LatticeElement::top();
        for e in elems {
            acc = self.meet(&acc, &e);
            if acc.is_bottom() {
                break;
            }
        }
        acc
    }

    /// Join (∨) of two lattice elements.
    pub fn join(&self, a: &LatticeElement, b: &LatticeElement) -> LatticeElement {
        let result = a.join(b);
        self.maybe_simplify(result)
    }

    /// Join of an arbitrary collection of elements.
    pub fn join_all(&self, elems: impl IntoIterator<Item = LatticeElement>) -> LatticeElement {
        let mut acc = LatticeElement::bottom();
        for e in elems {
            acc = self.join(&acc, &e);
            if acc.is_top() {
                break;
            }
        }
        acc
    }

    /// Negate a lattice element.
    pub fn complement(&self, elem: &LatticeElement) -> LatticeElement {
        let result = elem.negate();
        self.maybe_simplify(result)
    }

    /// Syntactically simplify a lattice element.
    pub fn simplify(&self, elem: &LatticeElement) -> LatticeElement {
        elem.simplify()
    }

    // -- entailment (semantic ordering) -------------------------------------

    /// Syntactic entailment approximation: returns `Some(true)` when `a ⊨ b`
    /// can be decided purely by formula structure, `None` when an SMT query
    /// would be needed.
    ///
    /// Rules applied:
    /// - ⊥ ⊨ anything
    /// - anything ⊨ ⊤
    /// - a ⊨ a (syntactic equality)
    /// - a ∧ c ⊨ a (conjunction weakening)
    pub fn entails_syntactic(a: &LatticeElement, b: &LatticeElement) -> Option<bool> {
        // ⊥ entails everything.
        if a.is_bottom() {
            return Some(true);
        }
        // Everything entails ⊤.
        if b.is_top() {
            return Some(true);
        }
        // ⊤ does not entail non-⊤.
        if a.is_top() && !b.is_top() {
            return Some(false);
        }
        // Syntactic equality.
        if a.formula() == b.formula() {
            return Some(true);
        }
        // Conjunction weakening: if b is one of the top-level conjuncts of a.
        if let Formula::And(conjuncts) = a.formula() {
            if conjuncts.iter().any(|c| c == b.formula()) {
                return Some(true);
            }
        }
        // If a appears as one of the disjuncts of b, then a entails b.
        if let Formula::Or(disjuncts) = b.formula() {
            if disjuncts.iter().any(|d| d == a.formula()) {
                return Some(true);
            }
        }
        None
    }

    /// Full entailment check: first tries syntactic rules, then falls back to
    /// SMT-based validity checking of `a → b`.
    ///
    /// Since we do not have an SMT solver wired in at the lattice level, this
    /// method uses syntactic checking plus some sound structural rules.  The
    /// [`entailment_queries`] counter tracks how many times the SMT fallback
    /// *would* have been invoked (useful for profiling).
    pub fn entails(&mut self, a: &LatticeElement, b: &LatticeElement) -> EntailmentResult {
        if let Some(result) = Self::entails_syntactic(a, b) {
            return if result {
                EntailmentResult::Entailed
            } else {
                EntailmentResult::NotEntailed
            };
        }

        // Structural implication rules for QF-LIA predicates.
        if let Some(result) = self.entails_structural(a, b) {
            return result;
        }

        self.entailment_queries += 1;
        EntailmentResult::Unknown
    }

    /// Structural entailment for atomic predicates over the same terms.
    fn entails_structural(
        &self,
        a: &LatticeElement,
        b: &LatticeElement,
    ) -> Option<EntailmentResult> {
        let fa = a.formula();
        let fb = b.formula();

        match (fa, fb) {
            // x == c  ⊨  x != d  when c ≠ d
            (Formula::Atom(pa), Formula::Atom(pb))
                if pa.left == pb.left && pa.right == pb.right =>
            {
                match (&pa.relation, &pb.relation) {
                    (Relation::Eq, Relation::Eq) => Some(EntailmentResult::Entailed),
                    (Relation::Eq, Relation::Le) | (Relation::Eq, Relation::Ge) => {
                        Some(EntailmentResult::Entailed)
                    }
                    (Relation::Lt, Relation::Le) | (Relation::Gt, Relation::Ge) => {
                        Some(EntailmentResult::Entailed)
                    }
                    (Relation::Lt, Relation::Ne) | (Relation::Gt, Relation::Ne) => {
                        Some(EntailmentResult::Entailed)
                    }
                    (Relation::Eq, Relation::Ne) => Some(EntailmentResult::NotEntailed),
                    (Relation::Ne, Relation::Eq) => Some(EntailmentResult::NotEntailed),
                    _ => None,
                }
            }
            // a ∧ b ⊨ c if any conjunct of a entails c
            (Formula::And(conjuncts), _) => {
                for conjunct in conjuncts {
                    let elem = LatticeElement::new(conjunct.clone());
                    if let Some(true) = Self::entails_syntactic(&elem, b) {
                        return Some(EntailmentResult::Entailed);
                    }
                }
                None
            }
            // a ⊨ b ∨ c if a entails any disjunct of b
            (_, Formula::Or(disjuncts)) => {
                for disjunct in disjuncts {
                    let elem = LatticeElement::new(disjunct.clone());
                    if let Some(true) = Self::entails_syntactic(a, &elem) {
                        return Some(EntailmentResult::Entailed);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Number of SMT-level entailment queries that were attempted.
    pub fn entailment_query_count(&self) -> u64 {
        self.entailment_queries
    }

    // -- element management -------------------------------------------------

    /// Register an element in the lattice and return its index.
    pub fn add(&mut self, elem: LatticeElement) -> usize {
        let idx = self.elements.len();
        self.elements.push(elem);
        idx
    }

    /// Retrieve an element by index.
    pub fn get(&self, idx: usize) -> Option<&LatticeElement> {
        self.elements.get(idx)
    }

    /// Number of elements currently in the lattice.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Iterate over all elements.
    pub fn elements(&self) -> &[LatticeElement] {
        &self.elements
    }

    /// Check whether an element exceeds the complexity bound (if set).
    pub fn exceeds_complexity(&self, elem: &LatticeElement) -> bool {
        if let Some(max) = self.max_complexity {
            elem.complexity() > max
        } else {
            false
        }
    }

    // -- internal helpers ---------------------------------------------------

    fn maybe_simplify(&self, elem: LatticeElement) -> LatticeElement {
        if self.auto_simplify {
            elem.simplify()
        } else {
            elem
        }
    }
}

impl Default for SpecLattice {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SpecLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SpecLattice ({} elements):", self.elements.len())?;
        for (i, elem) in self.elements.iter().enumerate() {
            writeln!(f, "  [{i}] {elem}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EntailmentResult
// ---------------------------------------------------------------------------

/// Result of an entailment query `a ⊨ b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntailmentResult {
    /// `a` logically entails `b`.
    Entailed,
    /// `a` does not entail `b`.
    NotEntailed,
    /// Could not be decided without an SMT query.
    Unknown,
}

impl EntailmentResult {
    pub fn is_entailed(self) -> bool {
        self == Self::Entailed
    }

    pub fn is_not_entailed(self) -> bool {
        self == Self::NotEntailed
    }

    pub fn is_unknown(self) -> bool {
        self == Self::Unknown
    }
}

// ---------------------------------------------------------------------------
// DiscriminationLattice
// ---------------------------------------------------------------------------

/// Specialised lattice for the mutation–discrimination problem.
///
/// The **sigma function** σ: P(M_kill) → Spec maps a set S of killed mutants
/// to the specification ∧_{m ∈ S} ¬E(m).
///
/// This struct maintains:
/// - A mapping from each mutant to its error predicate E(m).
/// - A cache of already-computed lattice points (keyed by mutant set).
/// - The underlying [`SpecLattice`] for meet/join/entailment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminationLattice {
    /// Error predicates for each mutant: mutant_id → E(m).
    error_predicates: IndexMap<MutantId, Formula>,
    /// Cache of computed lattice points: sorted mutant-set key → element index
    /// in `lattice.elements`.
    #[serde(skip)]
    cache: IndexMap<Vec<MutantId>, usize>,
    /// The underlying specification lattice.
    lattice: SpecLattice,
    /// Statistics: how many sigma evaluations were served from cache.
    cache_hits: u64,
    /// Statistics: how many sigma evaluations required fresh computation.
    cache_misses: u64,
}

impl DiscriminationLattice {
    /// Create a new discrimination lattice with no mutants registered.
    pub fn new() -> Self {
        Self {
            error_predicates: IndexMap::new(),
            cache: IndexMap::new(),
            lattice: SpecLattice::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Create with a pre-configured spec lattice.
    pub fn with_lattice(lattice: SpecLattice) -> Self {
        Self {
            error_predicates: IndexMap::new(),
            cache: IndexMap::new(),
            lattice,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    // -- mutant registration ------------------------------------------------

    /// Register the error predicate E(m) for a mutant.
    pub fn register_mutant(&mut self, id: MutantId, error_predicate: Formula) {
        self.error_predicates.insert(id, error_predicate);
        // Invalidate cache — a new mutant changes possible sigma values.
        self.cache.clear();
    }

    /// Register multiple mutants at once.
    pub fn register_mutants(
        &mut self,
        mutants: impl IntoIterator<Item = (MutantId, Formula)>,
    ) {
        for (id, ep) in mutants {
            self.error_predicates.insert(id, ep);
        }
        self.cache.clear();
    }

    /// Get the error predicate for a mutant, if registered.
    pub fn error_predicate(&self, id: &MutantId) -> Option<&Formula> {
        self.error_predicates.get(id)
    }

    /// All registered mutant IDs.
    pub fn mutant_ids(&self) -> Vec<MutantId> {
        self.error_predicates.keys().cloned().collect()
    }

    /// Number of registered mutants.
    pub fn mutant_count(&self) -> usize {
        self.error_predicates.len()
    }

    // -- sigma function -----------------------------------------------------

    /// σ(S) = ∧_{m ∈ S} ¬E(m)
    ///
    /// Computes the specification for a subset of killed mutants.  Results are
    /// cached so repeated queries for the same set are O(1).
    pub fn sigma(&mut self, mutant_set: &[MutantId]) -> LatticeElement {
        let key = Self::cache_key(mutant_set);

        if let Some(&idx) = self.cache.get(&key) {
            self.cache_hits += 1;
            return self
                .lattice
                .get(idx)
                .cloned()
                .unwrap_or_else(LatticeElement::top);
        }

        self.cache_misses += 1;

        if mutant_set.is_empty() {
            return LatticeElement::top();
        }

        // Build ∧_{m ∈ S} ¬E(m)
        let negated: Vec<Formula> = mutant_set
            .iter()
            .filter_map(|id| {
                self.error_predicates
                    .get(id)
                    .map(|ep| Formula::not(ep.clone()))
            })
            .collect();

        if negated.is_empty() {
            return LatticeElement::top();
        }

        let formula = if negated.len() == 1 {
            negated.into_iter().next().unwrap()
        } else {
            Formula::and(negated)
        };

        let elem =
            LatticeElement::from_mutants(formula, mutant_set.iter().cloned()).simplify();

        let idx = self.lattice.add(elem.clone());
        self.cache.insert(key, idx);
        elem
    }

    /// Compute sigma for a single mutant (convenience wrapper).
    pub fn sigma_single(&mut self, id: &MutantId) -> LatticeElement {
        self.sigma(&[id.clone()])
    }

    /// Compute sigma for *all* registered mutants — the strongest possible spec.
    pub fn sigma_all(&mut self) -> LatticeElement {
        let all_ids: Vec<MutantId> = self.error_predicates.keys().cloned().collect();
        self.sigma(&all_ids)
    }

    // -- lattice walk helpers -----------------------------------------------

    /// Compute the *incremental* meet: given the current spec and one new mutant,
    /// return current ∧ ¬E(new_mutant).
    pub fn extend_with_mutant(
        &mut self,
        current: &LatticeElement,
        new_mutant: &MutantId,
    ) -> LatticeElement {
        let single = self.sigma_single(new_mutant);
        self.lattice.meet(current, &single)
    }

    /// Compute the join of specs for each individual mutant (weakest useful spec).
    pub fn join_individuals(&mut self) -> LatticeElement {
        let ids: Vec<MutantId> = self.error_predicates.keys().cloned().collect();
        let individuals: Vec<LatticeElement> =
            ids.iter().map(|id| self.sigma_single(id)).collect();
        self.lattice.join_all(individuals)
    }

    /// Check entailment between two lattice elements.
    pub fn entails(
        &mut self,
        a: &LatticeElement,
        b: &LatticeElement,
    ) -> EntailmentResult {
        self.lattice.entails(a, b)
    }

    /// Iterate over lattice points in a walk-friendly order: from individual
    /// mutants to progressively larger subsets.  Returns pairs of (mutant set,
    /// lattice element).
    ///
    /// The iterator is *lazy*: each element is computed (and cached) on demand.
    pub fn walk_order(&mut self) -> Vec<(Vec<MutantId>, LatticeElement)> {
        let ids: Vec<MutantId> = self.error_predicates.keys().cloned().collect();
        let n = ids.len();
        let mut results = Vec::new();

        // Level 1: individual mutants
        for id in &ids {
            let elem = self.sigma_single(id);
            results.push((vec![id.clone()], elem));
        }

        // Level 2: pairwise combinations (if manageable)
        if n <= 64 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let pair = vec![ids[i].clone(), ids[j].clone()];
                    let elem = self.sigma(&pair);
                    results.push((pair, elem));
                }
            }
        }

        // Level n: all mutants together
        if n > 1 {
            let all = ids.clone();
            let elem = self.sigma(&all);
            results.push((all, elem));
        }

        results
    }

    /// Get the set of mutant IDs that a lattice element *covers* — i.e.
    /// for each registered mutant m, if `elem ⊨ ¬E(m)` then m is covered.
    pub fn covered_mutants(&mut self, elem: &LatticeElement) -> Vec<MutantId> {
        let mut covered = Vec::new();
        for (id, _ep) in &self.error_predicates {
            let single = LatticeElement::from_mutant(
                Formula::not(_ep.clone()),
                id.clone(),
            );
            match self.lattice.entails(elem, &single) {
                EntailmentResult::Entailed => covered.push(id.clone()),
                _ => {}
            }
        }
        covered
    }

    /// Build a [`Contract`] from the strongest spec that covers all mutants.
    pub fn synthesize_contract(&mut self, function_name: &str) -> Contract {
        let spec = self.sigma_all();
        let mut contract = spec.to_contract(function_name);
        contract.strength = strength_from_coverage(
            self.covered_mutants(&spec).len(),
            self.error_predicates.len(),
        );
        contract
    }

    // -- accessors ----------------------------------------------------------

    /// Access the underlying `SpecLattice`.
    pub fn spec_lattice(&self) -> &SpecLattice {
        &self.lattice
    }

    /// Mutable access to the underlying `SpecLattice`.
    pub fn spec_lattice_mut(&mut self) -> &mut SpecLattice {
        &mut self.lattice
    }

    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    pub fn cache_misses(&self) -> u64 {
        self.cache_misses
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Fraction of sigma calls served from cache.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Clear the lattice point cache (e.g. after mutant registration changes).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    // -- internal -----------------------------------------------------------

    /// Produce a canonical cache key from an unordered mutant set.
    fn cache_key(mutant_set: &[MutantId]) -> Vec<MutantId> {
        let mut key: Vec<MutantId> = mutant_set.to_vec();
        key.sort_by(|a, b| a.0.cmp(&b.0));
        key.dedup();
        key
    }
}

impl Default for DiscriminationLattice {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DiscriminationLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DiscriminationLattice: {} mutants, {} cached points",
            self.error_predicates.len(),
            self.cache.len(),
        )?;
        for (id, ep) in &self.error_predicates {
            writeln!(f, "  E({}) = {:?}", id.short(), ep)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a coverage ratio to a `ContractStrength`.
fn strength_from_coverage(covered: usize, total: usize) -> ContractStrength {
    if total == 0 {
        return ContractStrength::Trivial;
    }
    let ratio = covered as f64 / total as f64;
    ContractStrength::from_kill_ratio(ratio)
}

/// Utility: build a simple predicate `var rel const`.
pub fn var_rel_const(var: &str, rel: Relation, c: i64) -> Formula {
    Formula::Atom(Predicate {
        relation: rel,
        left: Term::Var(var.to_string()),
        right: Term::Const(c),
    })
}

/// Utility: build `var == const`.
pub fn var_eq(var: &str, c: i64) -> Formula {
    var_rel_const(var, Relation::Eq, c)
}

/// Utility: build `var != const`.
pub fn var_ne(var: &str, c: i64) -> Formula {
    var_rel_const(var, Relation::Ne, c)
}

/// Utility: build `var < const`.
pub fn var_lt(var: &str, c: i64) -> Formula {
    var_rel_const(var, Relation::Lt, c)
}

/// Utility: build `var >= const`.
pub fn var_ge(var: &str, c: i64) -> Formula {
    var_rel_const(var, Relation::Ge, c)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id() -> MutantId {
        MutantId::new()
    }

    fn pred_x_gt_0() -> Formula {
        var_rel_const("x", Relation::Gt, 0)
    }

    fn pred_y_eq_1() -> Formula {
        var_eq("y", 1)
    }

    fn pred_x_lt_10() -> Formula {
        var_lt("x", 10)
    }

    // -- LatticeElement tests -----------------------------------------------

    #[test]
    fn test_element_top_and_bottom() {
        let top = LatticeElement::top();
        let bot = LatticeElement::bottom();
        assert!(top.is_top());
        assert!(!top.is_bottom());
        assert!(bot.is_bottom());
        assert!(!bot.is_top());
    }

    #[test]
    fn test_element_from_mutant() {
        let id = make_id();
        let elem = LatticeElement::from_mutant(pred_x_gt_0(), id.clone());
        assert_eq!(elem.mutant_count(), 1);
        assert!(elem.provenance().contains(&id));
    }

    #[test]
    fn test_element_from_error_predicate() {
        let id = make_id();
        let elem = LatticeElement::from_error_predicate(pred_x_gt_0(), id.clone());
        // Should be ¬(x > 0)
        assert!(matches!(elem.formula(), Formula::Not(_)));
        assert_eq!(elem.mutant_count(), 1);
    }

    #[test]
    fn test_element_meet_merges_provenance() {
        let id1 = make_id();
        let id2 = make_id();
        let a = LatticeElement::from_mutant(pred_x_gt_0(), id1.clone());
        let b = LatticeElement::from_mutant(pred_y_eq_1(), id2.clone());
        let m = a.meet(&b);
        assert_eq!(m.mutant_count(), 2);
        assert!(m.provenance().contains(&id1));
        assert!(m.provenance().contains(&id2));
    }

    #[test]
    fn test_element_meet_with_top() {
        let a = LatticeElement::from_mutant(pred_x_gt_0(), make_id());
        let top = LatticeElement::top();
        let m = a.meet(&top);
        assert_eq!(m.formula(), a.formula());
    }

    #[test]
    fn test_element_meet_with_bottom() {
        let a = LatticeElement::from_mutant(pred_x_gt_0(), make_id());
        let bot = LatticeElement::bottom();
        let m = a.meet(&bot);
        assert!(m.is_bottom());
    }

    #[test]
    fn test_element_join_intersects_provenance() {
        let id1 = make_id();
        let id2 = make_id();
        let a = LatticeElement::from_mutants(pred_x_gt_0(), vec![id1.clone(), id2.clone()]);
        let b = LatticeElement::from_mutants(pred_y_eq_1(), vec![id2.clone()]);
        let j = a.join(&b);
        assert_eq!(j.mutant_count(), 1);
        assert!(j.provenance().contains(&id2));
    }

    #[test]
    fn test_element_join_with_bottom() {
        let a = LatticeElement::from_mutant(pred_x_gt_0(), make_id());
        let bot = LatticeElement::bottom();
        let j = a.join(&bot);
        assert_eq!(j.formula(), a.formula());
    }

    #[test]
    fn test_element_negate() {
        let a = LatticeElement::new(pred_x_gt_0());
        let neg = a.negate();
        assert!(matches!(neg.formula(), Formula::Not(_)));
    }

    #[test]
    fn test_element_complexity_increases_with_size() {
        let small = LatticeElement::new(pred_x_gt_0());
        let big = LatticeElement::new(Formula::and(vec![
            pred_x_gt_0(),
            pred_y_eq_1(),
            pred_x_lt_10(),
        ]));
        assert!(big.complexity() > small.complexity());
    }

    #[test]
    fn test_element_to_ensures_clause() {
        let a = LatticeElement::new(pred_x_gt_0());
        let clause = a.to_ensures_clause();
        assert!(matches!(clause, ContractClause::Ensures(_)));
    }

    #[test]
    fn test_element_display_with_label() {
        let a = LatticeElement::new(pred_x_gt_0()).with_label("test_spec");
        let s = format!("{a}");
        assert!(s.contains("test_spec"));
    }

    // -- SpecLattice tests --------------------------------------------------

    #[test]
    fn test_lattice_meet_all_empty() {
        let lat = SpecLattice::new();
        let result = lat.meet_all(std::iter::empty());
        assert!(result.is_top());
    }

    #[test]
    fn test_lattice_join_all_empty() {
        let lat = SpecLattice::new();
        let result = lat.join_all(std::iter::empty());
        assert!(result.is_bottom());
    }

    #[test]
    fn test_lattice_entails_bottom_anything() {
        let mut lat = SpecLattice::new();
        let bot = LatticeElement::bottom();
        let a = LatticeElement::new(pred_x_gt_0());
        assert!(lat.entails(&bot, &a).is_entailed());
    }

    #[test]
    fn test_lattice_entails_anything_top() {
        let mut lat = SpecLattice::new();
        let top = LatticeElement::top();
        let a = LatticeElement::new(pred_x_gt_0());
        assert!(lat.entails(&a, &top).is_entailed());
    }

    #[test]
    fn test_lattice_entails_self() {
        let mut lat = SpecLattice::new();
        let a = LatticeElement::new(pred_x_gt_0());
        assert!(lat.entails(&a, &a).is_entailed());
    }

    #[test]
    fn test_lattice_entails_conjunction_weakening() {
        let mut lat = SpecLattice::new();
        let conj = LatticeElement::new(Formula::and(vec![pred_x_gt_0(), pred_y_eq_1()]));
        let single = LatticeElement::new(pred_x_gt_0());
        assert!(lat.entails(&conj, &single).is_entailed());
    }

    #[test]
    fn test_lattice_entails_structural_lt_implies_le() {
        let mut lat = SpecLattice::new();
        let lt = LatticeElement::new(var_lt("x", 10));
        let le = LatticeElement::new(var_rel_const("x", Relation::Le, 10));
        assert!(lat.entails(&lt, &le).is_entailed());
    }

    #[test]
    fn test_lattice_complexity_bound() {
        let lat = SpecLattice::with_max_complexity(5.0);
        let big = LatticeElement::new(Formula::and(vec![
            pred_x_gt_0(),
            pred_y_eq_1(),
            pred_x_lt_10(),
        ]));
        assert!(lat.exceeds_complexity(&big));
        let small = LatticeElement::new(Formula::True);
        assert!(!lat.exceeds_complexity(&small));
    }

    // -- DiscriminationLattice tests ----------------------------------------

    #[test]
    fn test_disc_sigma_empty_is_top() {
        let mut dl = DiscriminationLattice::new();
        let elem = dl.sigma(&[]);
        assert!(elem.is_top());
    }

    #[test]
    fn test_disc_sigma_single() {
        let mut dl = DiscriminationLattice::new();
        let id = make_id();
        dl.register_mutant(id.clone(), pred_x_gt_0());
        let elem = dl.sigma_single(&id);
        // Should be ¬(x > 0), with provenance {id}
        assert!(matches!(elem.formula(), Formula::Not(_)));
        assert!(elem.provenance().contains(&id));
    }

    #[test]
    fn test_disc_sigma_pair() {
        let mut dl = DiscriminationLattice::new();
        let id1 = make_id();
        let id2 = make_id();
        dl.register_mutant(id1.clone(), pred_x_gt_0());
        dl.register_mutant(id2.clone(), pred_y_eq_1());
        let elem = dl.sigma(&[id1.clone(), id2.clone()]);
        assert_eq!(elem.mutant_count(), 2);
    }

    #[test]
    fn test_disc_sigma_caching() {
        let mut dl = DiscriminationLattice::new();
        let id = make_id();
        dl.register_mutant(id.clone(), pred_x_gt_0());

        let _ = dl.sigma_single(&id);
        assert_eq!(dl.cache_misses(), 1);
        assert_eq!(dl.cache_hits(), 0);

        let _ = dl.sigma_single(&id);
        assert_eq!(dl.cache_hits(), 1);
    }

    #[test]
    fn test_disc_sigma_all() {
        let mut dl = DiscriminationLattice::new();
        let id1 = make_id();
        let id2 = make_id();
        dl.register_mutant(id1.clone(), pred_x_gt_0());
        dl.register_mutant(id2.clone(), pred_y_eq_1());
        let all = dl.sigma_all();
        assert_eq!(all.mutant_count(), 2);
    }

    #[test]
    fn test_disc_extend_with_mutant() {
        let mut dl = DiscriminationLattice::new();
        let id1 = make_id();
        let id2 = make_id();
        dl.register_mutant(id1.clone(), pred_x_gt_0());
        dl.register_mutant(id2.clone(), pred_y_eq_1());

        let spec1 = dl.sigma_single(&id1);
        let spec12 = dl.extend_with_mutant(&spec1, &id2);
        // Extended spec should be at least as strong.
        assert!(spec12.size_hint() >= spec1.size_hint() || spec12.is_bottom());
    }

    #[test]
    fn test_disc_synthesize_contract() {
        let mut dl = DiscriminationLattice::new();
        let id1 = make_id();
        dl.register_mutant(id1.clone(), pred_x_gt_0());
        let contract = dl.synthesize_contract("foo");
        assert_eq!(contract.function_name, "foo");
        assert!(!contract.clauses.is_empty());
    }

    #[test]
    fn test_disc_cache_key_order_independent() {
        let id1 = MutantId::from_uuid(Uuid::from_u128(1));
        let id2 = MutantId::from_uuid(Uuid::from_u128(2));
        let key_a = DiscriminationLattice::cache_key(&[id1.clone(), id2.clone()]);
        let key_b = DiscriminationLattice::cache_key(&[id2.clone(), id1.clone()]);
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn test_disc_walk_order_includes_individuals_and_all() {
        let mut dl = DiscriminationLattice::new();
        let id1 = make_id();
        let id2 = make_id();
        dl.register_mutant(id1.clone(), pred_x_gt_0());
        dl.register_mutant(id2.clone(), pred_y_eq_1());

        let walk = dl.walk_order();
        // Should include: 2 individuals + 1 pair + 1 all = 4
        assert_eq!(walk.len(), 4);
    }

    #[test]
    fn test_disc_clear_cache() {
        let mut dl = DiscriminationLattice::new();
        let id = make_id();
        dl.register_mutant(id.clone(), pred_x_gt_0());
        let _ = dl.sigma_single(&id);
        assert_eq!(dl.cache_size(), 1);
        dl.clear_cache();
        assert_eq!(dl.cache_size(), 0);
    }

    #[test]
    fn test_disc_display() {
        let mut dl = DiscriminationLattice::new();
        dl.register_mutant(make_id(), pred_x_gt_0());
        let s = format!("{dl}");
        assert!(s.contains("1 mutants"));
    }

    #[test]
    fn test_entailment_result_methods() {
        assert!(EntailmentResult::Entailed.is_entailed());
        assert!(EntailmentResult::NotEntailed.is_not_entailed());
        assert!(EntailmentResult::Unknown.is_unknown());
    }

    #[test]
    fn test_lattice_top_does_not_entail_nontop() {
        let mut lat = SpecLattice::new();
        let top = LatticeElement::top();
        let a = LatticeElement::new(pred_x_gt_0());
        assert!(lat.entails(&top, &a).is_not_entailed());
    }

    #[test]
    fn test_element_free_vars() {
        let elem = LatticeElement::new(Formula::and(vec![pred_x_gt_0(), pred_y_eq_1()]));
        let vars = elem.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_lattice_add_and_get() {
        let mut lat = SpecLattice::new();
        let elem = LatticeElement::new(pred_x_gt_0());
        let idx = lat.add(elem.clone());
        assert_eq!(lat.len(), 1);
        assert_eq!(lat.get(idx).unwrap().formula(), elem.formula());
    }

    #[test]
    fn test_element_size_and_depth_caching() {
        let mut elem = LatticeElement::new(Formula::and(vec![pred_x_gt_0(), pred_y_eq_1()]));
        let s1 = elem.size();
        let s2 = elem.size();
        assert_eq!(s1, s2);
        let d1 = elem.depth();
        let d2 = elem.depth();
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_helper_var_eq_ne_lt_ge() {
        assert!(matches!(var_eq("x", 0), Formula::Atom(p) if p.relation == Relation::Eq));
        assert!(matches!(var_ne("x", 0), Formula::Atom(p) if p.relation == Relation::Ne));
        assert!(matches!(var_lt("x", 0), Formula::Atom(p) if p.relation == Relation::Lt));
        assert!(matches!(var_ge("x", 0), Formula::Atom(p) if p.relation == Relation::Ge));
    }

    #[test]
    fn test_disc_join_individuals() {
        let mut dl = DiscriminationLattice::new();
        let id1 = make_id();
        let id2 = make_id();
        dl.register_mutant(id1.clone(), pred_x_gt_0());
        dl.register_mutant(id2.clone(), pred_y_eq_1());
        let joined = dl.join_individuals();
        // Join of ¬(x>0) and ¬(y==1) should be a disjunction
        assert!(
            matches!(joined.formula(), Formula::Or(_))
                || joined.is_top()
        );
    }
}
