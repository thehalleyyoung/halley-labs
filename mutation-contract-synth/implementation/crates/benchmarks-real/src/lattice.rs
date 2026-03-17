//! Inlined lattice + lattice-walk code from contract-synth.
//!
//! We inline this because contract-synth depends on the broken program-analysis
//! crate.  All logic faithfully mirrors crates/contract-synth/src/{lattice.rs,
//! lattice_walk.rs}.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use shared_types::{Contract, ContractClause, ContractStrength, Formula, MutantId};

// =========================================================================
// LatticeElement
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeElement {
    formula: Formula,
    provenance: HashSet<MutantId>,
    cached_size: Option<usize>,
    cached_depth: Option<usize>,
    label: Option<String>,
}

impl LatticeElement {
    pub fn new(formula: Formula) -> Self {
        Self {
            formula,
            provenance: BTreeSet::new(),
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

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

    pub fn top() -> Self {
        Self::new(Formula::True)
    }

    pub fn bottom() -> Self {
        Self::new(Formula::False)
    }

    pub fn from_error_predicate(error_pred: Formula, mutant_id: MutantId) -> Self {
        let negated = Formula::not(error_pred);
        Self::from_mutant(negated, mutant_id)
    }

    pub fn formula(&self) -> &Formula {
        &self.formula
    }
    pub fn provenance(&self) -> &HashSet<MutantId> {
        &self.provenance
    }
    pub fn mutant_count(&self) -> usize {
        self.provenance.len()
    }

    pub fn size(&mut self) -> usize {
        if let Some(s) = self.cached_size {
            return s;
        }
        let s = self.formula.size();
        self.cached_size = Some(s);
        s
    }

    pub fn size_hint(&self) -> usize {
        self.cached_size.unwrap_or_else(|| self.formula.size())
    }

    pub fn depth(&mut self) -> usize {
        if let Some(d) = self.cached_depth {
            return d;
        }
        let d = self.formula.depth();
        self.cached_depth = Some(d);
        d
    }

    pub fn free_vars(&self) -> BTreeSet<String> {
        self.formula.free_vars()
    }

    pub fn is_top(&self) -> bool {
        matches!(self.formula, Formula::True)
    }
    pub fn is_bottom(&self) -> bool {
        matches!(self.formula, Formula::False)
    }

    /// Lattice meet: conjunction.
    pub fn meet(&self, other: &LatticeElement) -> LatticeElement {
        if self.is_top() {
            return other.clone();
        }
        if other.is_top() {
            return self.clone();
        }
        if self.is_bottom() || other.is_bottom() {
            return LatticeElement::bottom();
        }

        let formula = Formula::and(vec![self.formula.clone(), other.formula.clone()]);
        let mut provenance = self.provenance.clone();
        provenance.extend(other.provenance.iter().cloned());
        LatticeElement {
            formula,
            provenance,
            cached_size: None,
            cached_depth: None,
            label: None,
        }
    }

    pub fn simplify(&self) -> LatticeElement {
        LatticeElement {
            formula: self.formula.simplify(),
            provenance: self.provenance.clone(),
            cached_size: None,
            cached_depth: None,
            label: self.label.clone(),
        }
    }

    pub fn to_ensures_clause(&self) -> ContractClause {
        ContractClause::ensures(self.formula.clone())
    }
}

impl fmt::Display for LatticeElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formula)
    }
}

// =========================================================================
// DiscriminationLattice
// =========================================================================

pub struct DiscriminationLattice {
    mutants: HashMap<MutantId, Formula>,
    cache: HashMap<Vec<MutantId>, LatticeElement>,
}

impl DiscriminationLattice {
    pub fn new() -> Self {
        Self {
            mutants: BTreeMap::new(),
            cache: BTreeMap::new(),
        }
    }

    pub fn register_mutant(&mut self, id: MutantId, error_predicate: Formula) {
        self.mutants.insert(id, error_predicate);
    }

    pub fn error_predicate(&self, id: &MutantId) -> Option<&Formula> {
        self.mutants.get(id)
    }

    pub fn mutant_ids(&self) -> Vec<MutantId> {
        self.mutants.keys().cloned().collect()
    }

    pub fn mutant_count(&self) -> usize {
        self.mutants.len()
    }

    /// σ: P(M_kill) → Spec.  Maps a subset of killed mutants to their
    /// conjoined negated error predicates.
    pub fn sigma(&mut self, mutant_set: &[MutantId]) -> LatticeElement {
        let mut sorted: Vec<MutantId> = mutant_set.to_vec();
        sorted.sort();

        if let Some(cached) = self.cache.get(&sorted) {
            return cached.clone();
        }

        if sorted.is_empty() {
            return LatticeElement::top();
        }

        let mut conjuncts = Vec::new();
        let mut provenance = BTreeSet::new();

        for mid in &sorted {
            if let Some(ep) = self.mutants.get(mid) {
                conjuncts.push(Formula::not(ep.clone()));
                provenance.insert(mid.clone());
            }
        }

        let formula = if conjuncts.is_empty() {
            Formula::True
        } else if conjuncts.len() == 1 {
            conjuncts.into_iter().next().unwrap()
        } else {
            Formula::and(conjuncts)
        };

        let elem = LatticeElement {
            formula,
            provenance,
            cached_size: None,
            cached_depth: None,
            label: None,
        };
        self.cache.insert(sorted, elem.clone());
        elem
    }

    pub fn sigma_single(&mut self, id: &MutantId) -> LatticeElement {
        self.sigma(&[id.clone()])
    }

    pub fn sigma_all(&mut self) -> LatticeElement {
        let ids: Vec<MutantId> = self.mutant_ids();
        self.sigma(&ids)
    }

    /// Build a contract from all registered mutants.
    pub fn synthesize_contract(&mut self, function_name: &str) -> Contract {
        let all = self.sigma_all();
        let mut contract = Contract::new(function_name.to_string());
        // Decompose conjunction into individual ensures clauses
        match all.formula() {
            Formula::And(conjuncts) => {
                for c in conjuncts {
                    contract.add_clause(ContractClause::ensures(c.clone()));
                }
            }
            Formula::True => {}
            other => {
                contract.add_clause(ContractClause::ensures(other.clone()));
            }
        }
        contract
    }
}

// =========================================================================
// WalkConfig
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkConfig {
    pub max_formula_size: usize,
    pub max_formula_depth: usize,
    pub max_steps: usize,
    pub timeout_ms: u64,
    pub simplify_eagerly: bool,
    pub decompose_clauses: bool,
    pub size_weight: f64,
    pub var_weight: f64,
    pub depth_weight: f64,
    pub enable_subsumption: bool,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            max_formula_size: 200,
            max_formula_depth: 15,
            max_steps: 1000,
            timeout_ms: 30_000,
            simplify_eagerly: true,
            decompose_clauses: true,
            size_weight: 1.0,
            var_weight: 2.0,
            depth_weight: 1.5,
            enable_subsumption: true,
        }
    }
}

// =========================================================================
// WalkStatistics
// =========================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalkStatistics {
    pub total_steps: u64,
    pub accepted_steps: u64,
    pub skipped_steps: u64,
    pub entailment_checks: u64,
    pub total_time_ms: f64,
}

// =========================================================================
// Internals: priority ordering
// =========================================================================

#[derive(Debug)]
struct MutantPriority {
    id: MutantId,
    score: f64,
}

// =========================================================================
// LatticeWalkSynthesizer
// =========================================================================

pub struct LatticeWalkSynthesizer {
    config: WalkConfig,
    statistics: Option<WalkStatistics>,
}

impl LatticeWalkSynthesizer {
    pub fn new(config: WalkConfig) -> Self {
        Self {
            config,
            statistics: None,
        }
    }

    pub fn statistics(&self) -> Option<&WalkStatistics> {
        self.statistics.as_ref()
    }

    /// Synthesize a contract by walking the discrimination lattice.
    ///
    /// Algorithm A2 from the paper:
    /// 1. Start at Top (True).
    /// 2. Order mutants by priority (fewer free vars, smaller AST → first).
    /// 3. For each mutant, conjoin ¬E(m) if the result is consistent and
    ///    within complexity budget.
    /// 4. Decompose final conjunction into per-clause ensures.
    pub fn synthesize(
        &mut self,
        disc: &mut DiscriminationLattice,
        function_name: &str,
    ) -> Contract {
        let start = Instant::now();
        let mutant_ids = disc.mutant_ids();
        let total = mutant_ids.len();

        if total == 0 {
            self.statistics = Some(WalkStatistics::default());
            return Contract::new(function_name.to_string());
        }

        // 1. Compute priority ordering.
        let ordered = self.compute_order(disc, &mutant_ids);

        // 2. Walk.
        let mut current = LatticeElement::top();
        let mut accepted: u64 = 0;
        let mut skipped: u64 = 0;
        let mut entailment_checks: u64 = 0;
        let mut steps: u64 = 0;

        for pri in &ordered {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            if elapsed_ms >= self.config.timeout_ms {
                break;
            }
            if steps >= self.config.max_steps as u64 {
                break;
            }

            steps += 1;

            let ep = match disc.error_predicate(&pri.id) {
                Some(ep) => ep.clone(),
                None => {
                    skipped += 1;
                    continue;
                }
            };

            let negated_ep = LatticeElement::from_error_predicate(ep.clone(), pri.id.clone());

            // Subsumption check: is ¬E(m) already implied by current?
            if self.config.enable_subsumption {
                entailment_checks += 1;
                if syntactic_entails(&current, &negated_ep) {
                    skipped += 1;
                    continue;
                }
            }

            // Compute candidate = current ∧ ¬E(m).
            let candidate = current.meet(&negated_ep);

            // Check complexity budget.
            let mut candidate_mut = candidate.clone();
            if candidate_mut.size() > self.config.max_formula_size {
                skipped += 1;
                continue;
            }
            if candidate_mut.depth() > self.config.max_formula_depth {
                skipped += 1;
                continue;
            }

            // Check consistency (not Bottom).  We do a quick syntactic check.
            if is_syntactically_false(&candidate.formula()) {
                skipped += 1;
                continue;
            }

            // Accept.
            current = if self.config.simplify_eagerly {
                candidate.simplify()
            } else {
                candidate
            };
            accepted += 1;
        }

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.statistics = Some(WalkStatistics {
            total_steps: steps,
            accepted_steps: accepted,
            skipped_steps: skipped,
            entailment_checks,
            total_time_ms,
        });

        // 3. Build contract from final specification.
        let mut contract = Contract::new(function_name.to_string());
        if self.config.decompose_clauses {
            decompose_into_clauses(&current, &mut contract);
        } else {
            if !current.is_top() {
                contract.add_clause(current.to_ensures_clause());
            }
        }

        contract.set_strength(classify_strength(accepted, total as u64));
        contract
    }
}

impl LatticeWalkSynthesizer {
    fn compute_order(&self, disc: &DiscriminationLattice, ids: &[MutantId]) -> Vec<MutantPriority> {
        let mut priorities: Vec<MutantPriority> = ids
            .iter()
            .filter_map(|id| {
                let ep = disc.error_predicate(id)?;
                let fv_count = ep.free_vars().len() as f64;
                let size = ep.size() as f64;
                let depth = ep.depth() as f64;
                // Lower score = higher priority (fewer vars, smaller formula).
                let score = self.config.var_weight * fv_count
                    + self.config.size_weight * size
                    + self.config.depth_weight * depth;
                Some(MutantPriority {
                    id: id.clone(),
                    score,
                })
            })
            .collect();

        priorities.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        priorities
    }
}

// --- Helpers ---

/// Quick syntactic entailment: checks if `a` already contains `b`'s formula
/// as a conjunct (no SMT calls).
fn syntactic_entails(a: &LatticeElement, b: &LatticeElement) -> bool {
    if b.is_top() {
        return true;
    }
    if a.is_top() {
        return false;
    }

    let a_conjuncts = collect_conjuncts(a.formula());
    let b_conjuncts = collect_conjuncts(b.formula());

    // b is entailed if every conjunct in b appears in a.
    b_conjuncts
        .iter()
        .all(|bc| a_conjuncts.iter().any(|ac| formula_eq(ac, bc)))
}

fn collect_conjuncts(f: &Formula) -> Vec<&Formula> {
    match f {
        Formula::And(cs) => cs.iter().flat_map(|c| collect_conjuncts(c)).collect(),
        other => vec![other],
    }
}

fn formula_eq(a: &Formula, b: &Formula) -> bool {
    format!("{}", a) == format!("{}", b)
}

fn is_syntactically_false(f: &Formula) -> bool {
    matches!(f, Formula::False)
}

fn decompose_into_clauses(elem: &LatticeElement, contract: &mut Contract) {
    match elem.formula() {
        Formula::And(conjuncts) => {
            for c in conjuncts {
                if !matches!(c, Formula::True) {
                    contract.add_clause(ContractClause::ensures(c.clone()));
                }
            }
        }
        Formula::True => {}
        other => {
            contract.add_clause(ContractClause::ensures(other.clone()));
        }
    }
}

fn classify_strength(accepted: u64, total: u64) -> ContractStrength {
    if total == 0 {
        return ContractStrength::Trivial;
    }
    let ratio = accepted as f64 / total as f64;
    if ratio >= 0.8 {
        ContractStrength::Strong
    } else if ratio >= 0.5 {
        ContractStrength::Moderate
    } else if ratio > 0.0 {
        ContractStrength::Weak
    } else {
        ContractStrength::Trivial
    }
}
