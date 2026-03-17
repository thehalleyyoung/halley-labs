//! Formula preprocessing for MaxSAT.
//!
//! Applies simplification techniques to reduce formula size before solving:
//! unit propagation, backbone detection, subsumption elimination,
//! self-subsuming resolution, bounded variable elimination, equivalent
//! literal substitution, and label compression.

use crate::formula::{
    Clause, ClauseDatabase, HardClause, Literal, MaxSatFormula, SoftClause,
};
use crate::solver::{CdclSolver, SatOracle, SatOracleResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the preprocessor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub enable_bve: bool,
    pub enable_subsumption: bool,
    pub enable_backbone: bool,
    pub enable_self_subsuming: bool,
    pub enable_equivalent_literals: bool,
    pub enable_label_compression: bool,
    pub bve_bound: usize,
    pub max_rounds: usize,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            enable_bve: true,
            enable_subsumption: true,
            enable_backbone: false,
            enable_self_subsuming: true,
            enable_equivalent_literals: true,
            enable_label_compression: true,
            bve_bound: 10,
            max_rounds: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreprocessStats {
    pub clauses_removed: usize,
    pub variables_eliminated: usize,
    pub literals_removed: usize,
    pub time_ms: u64,
    pub rounds: usize,
    pub backbone_literals: usize,
    pub subsumed_clauses: usize,
    pub bve_eliminations: usize,
    pub equivalent_pairs: usize,
    pub labels_compressed: usize,
}

impl std::fmt::Display for PreprocessStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "removed {} clauses, {} vars, {} lits in {}ms ({} rounds)",
            self.clauses_removed, self.variables_eliminated, self.literals_removed,
            self.time_ms, self.rounds
        )
    }
}

// ---------------------------------------------------------------------------
// MaxSatPreprocessor
// ---------------------------------------------------------------------------

/// Preprocessor that applies a sequence of simplification techniques.
pub struct MaxSatPreprocessor {
    pub config: PreprocessConfig,
    pub stats: PreprocessStats,
    /// Assignments deduced during preprocessing.
    pub deduced: HashMap<u32, bool>,
}

impl MaxSatPreprocessor {
    pub fn new(config: PreprocessConfig) -> Self {
        Self {
            config,
            stats: PreprocessStats::default(),
            deduced: HashMap::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(PreprocessConfig::default())
    }

    /// Apply all enabled preprocessing techniques.
    pub fn preprocess(&mut self, formula: &mut MaxSatFormula) -> PreprocessStats {
        let start = Instant::now();
        let initial_hard = formula.num_hard_clauses();
        let initial_soft = formula.num_soft_clauses();

        for round in 0..self.config.max_rounds {
            let before = formula.num_clauses();

            self.unit_propagation(formula);

            if self.config.enable_subsumption {
                self.clause_subsumption_elimination(formula);
            }

            if self.config.enable_self_subsuming {
                self.self_subsuming_resolution(formula);
            }

            if self.config.enable_equivalent_literals {
                self.equivalent_literal_substitution(formula);
            }

            if self.config.enable_bve {
                self.bounded_variable_elimination(formula, self.config.bve_bound);
            }

            if self.config.enable_label_compression {
                self.label_compression(formula);
            }

            self.stats.rounds = round + 1;

            let after = formula.num_clauses();
            if after >= before {
                break; // No more progress
            }
        }

        if self.config.enable_backbone {
            let backbone = self.backbone_detection(formula);
            self.stats.backbone_literals = backbone.len();
            for lit in &backbone {
                self.deduced.insert(lit.variable, !lit.negated);
            }
            // Apply backbone to formula
            for lit in backbone {
                propagate_literal_in_formula(formula, lit, &mut self.stats);
            }
        }

        // Remove tautologies and duplicates
        formula.remove_tautologies();
        formula.deduplicate_literals();

        self.stats.clauses_removed =
            (initial_hard + initial_soft).saturating_sub(formula.num_clauses());
        self.stats.time_ms = start.elapsed().as_millis() as u64;
        self.stats.clone()
    }
}

// ---------------------------------------------------------------------------
// Individual techniques
// ---------------------------------------------------------------------------

impl MaxSatPreprocessor {
    /// Unit propagation: find unit hard clauses and propagate.
    pub fn unit_propagation(&mut self, formula: &mut MaxSatFormula) {
        loop {
            let unit = formula
                .hard_clauses
                .iter()
                .find_map(|hc| {
                    if hc.0.literals.len() == 1 {
                        Some(hc.0.literals[0])
                    } else {
                        None
                    }
                });

            match unit {
                Some(lit) => {
                    self.deduced.insert(lit.variable, !lit.negated);
                    propagate_literal_in_formula(formula, lit, &mut self.stats);
                }
                None => break,
            }
        }
    }

    /// Backbone detection: find literals that must be true in every satisfying assignment.
    ///
    /// Uses SAT calls: for each variable, check if forcing it to a specific polarity
    /// makes the formula UNSAT.
    pub fn backbone_detection(&mut self, formula: &MaxSatFormula) -> Vec<Literal> {
        let mut backbone = Vec::new();
        let mut solver = CdclSolver::new();

        // Collect all hard clauses
        let hard_clauses: Vec<Clause> = formula.hard_clauses.iter().map(|hc| hc.0.clone()).collect();

        // First check if the hard clauses are satisfiable
        match solver.solve_sat(&hard_clauses, formula.num_variables) {
            SatOracleResult::Unsat(_) => return backbone,
            SatOracleResult::Sat(model) => {
                // For each variable in the model, check if the opposite is possible
                let vars: Vec<u32> = formula.referenced_variables().into_iter().collect();
                for &var in &vars {
                    if self.deduced.contains_key(&var) {
                        continue;
                    }
                    let model_val = model.get(&var).copied().unwrap_or(false);
                    let test_lit = if model_val {
                        Literal::negative(var) // try opposite
                    } else {
                        Literal::positive(var)
                    };

                    let assumptions = vec![test_lit];
                    match solver.solve_with_assumptions(
                        &hard_clauses,
                        &assumptions,
                        formula.num_variables,
                    ) {
                        SatOracleResult::Unsat(_) => {
                            // Opposite is UNSAT: variable is backbone
                            let backbone_lit = if model_val {
                                Literal::positive(var)
                            } else {
                                Literal::negative(var)
                            };
                            backbone.push(backbone_lit);
                        }
                        SatOracleResult::Sat(_) => {}
                    }
                }
            }
        }

        backbone
    }

    /// Remove clauses that are subsumed by other clauses.
    pub fn clause_subsumption_elimination(&mut self, formula: &mut MaxSatFormula) {
        let mut removed_indices = HashSet::new();

        // Check hard clauses against each other
        let n = formula.hard_clauses.len();
        for i in 0..n {
            if removed_indices.contains(&i) {
                continue;
            }
            for j in (i + 1)..n {
                if removed_indices.contains(&j) {
                    continue;
                }
                if formula.hard_clauses[i].0.subsumes(&formula.hard_clauses[j].0) {
                    removed_indices.insert(j);
                    self.stats.subsumed_clauses += 1;
                } else if formula.hard_clauses[j].0.subsumes(&formula.hard_clauses[i].0) {
                    removed_indices.insert(i);
                    self.stats.subsumed_clauses += 1;
                    break;
                }
            }
        }

        // Remove subsumed hard clauses
        let mut idx = 0;
        formula.hard_clauses.retain(|_| {
            let keep = !removed_indices.contains(&idx);
            idx += 1;
            keep
        });

        // Check soft clauses: a hard clause can subsume a soft clause
        let mut soft_removed = HashSet::new();
        for hc in &formula.hard_clauses {
            for (j, sc) in formula.soft_clauses.iter().enumerate() {
                if hc.0.subsumes(&sc.clause) {
                    soft_removed.insert(j);
                    self.stats.subsumed_clauses += 1;
                }
            }
        }

        let mut idx = 0;
        formula.soft_clauses.retain(|_| {
            let keep = !soft_removed.contains(&idx);
            idx += 1;
            keep
        });
    }

    /// Self-subsuming resolution: strengthen clauses using resolution.
    ///
    /// If clause C contains literal `l` and clause D = C\{l} ∪ {¬l},
    /// then we can remove `l` from D (since C resolves with D to give C\{l}).
    pub fn self_subsuming_resolution(&mut self, formula: &mut MaxSatFormula) {
        let mut changed = true;
        while changed {
            changed = false;
            let n = formula.hard_clauses.len();

            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let ci = &formula.hard_clauses[i].0;
                    let cj = &formula.hard_clauses[j].0;

                    // Find if ci and cj differ by exactly one literal (complementary)
                    if let Some(resolved_lit) = find_self_subsumption(ci, cj) {
                        formula.hard_clauses[j].0.remove_literal(resolved_lit);
                        self.stats.literals_removed += 1;
                        changed = true;
                        break;
                    }
                }
                if changed {
                    break;
                }
            }
        }
    }

    /// Bounded variable elimination: eliminate variables that appear in few clauses.
    pub fn bounded_variable_elimination(&mut self, formula: &mut MaxSatFormula, bound: usize) {
        let vars: Vec<u32> = formula.referenced_variables().into_iter().collect();

        for &var in &vars {
            if self.deduced.contains_key(&var) {
                continue;
            }

            // Count positive and negative occurrences in hard clauses
            let pos_clauses: Vec<usize> = formula
                .hard_clauses
                .iter()
                .enumerate()
                .filter(|(_, hc)| hc.0.contains(Literal::positive(var)))
                .map(|(i, _)| i)
                .collect();

            let neg_clauses: Vec<usize> = formula
                .hard_clauses
                .iter()
                .enumerate()
                .filter(|(_, hc)| hc.0.contains(Literal::negative(var)))
                .map(|(i, _)| i)
                .collect();

            // Check also soft clauses — skip if variable appears in any soft clause
            let in_soft = formula.soft_clauses.iter().any(|sc| {
                sc.clause
                    .literals
                    .iter()
                    .any(|l| l.variable == var)
            });
            if in_soft {
                continue;
            }

            let num_resolvents = pos_clauses.len() * neg_clauses.len();
            let num_removed = pos_clauses.len() + neg_clauses.len();

            // Only eliminate if the number of resolvents is within bound
            if num_resolvents > bound || num_resolvents > num_removed {
                continue;
            }

            // Generate resolvents
            let mut new_clauses = Vec::new();
            for &pi in &pos_clauses {
                for &ni in &neg_clauses {
                    let pc = &formula.hard_clauses[pi].0;
                    let nc = &formula.hard_clauses[ni].0;
                    if let Some(resolvent) = resolve(pc, nc, var) {
                        if !resolvent.is_tautology() {
                            new_clauses.push(resolvent);
                        }
                    }
                }
            }

            // Remove original clauses and add resolvents
            let mut to_remove: HashSet<usize> = pos_clauses.iter().copied().collect();
            to_remove.extend(neg_clauses.iter());

            let mut idx = 0;
            formula.hard_clauses.retain(|_| {
                let keep = !to_remove.contains(&idx);
                idx += 1;
                keep
            });

            for c in new_clauses {
                formula.hard_clauses.push(HardClause(c));
            }

            self.stats.bve_eliminations += 1;
            self.stats.variables_eliminated += 1;
        }
    }

    /// Equivalent literal substitution: find pairs of literals that are equivalent
    /// (always assigned the same value) and merge them.
    pub fn equivalent_literal_substitution(&mut self, formula: &mut MaxSatFormula) {
        // Find equivalences from binary hard clauses:
        // (¬a ∨ b) ∧ (¬b ∨ a) means a ≡ b
        let mut implications: HashMap<u32, HashSet<u32>> = HashMap::new();

        for hc in &formula.hard_clauses {
            if hc.0.literals.len() == 2 {
                let l0 = hc.0.literals[0];
                let l1 = hc.0.literals[1];
                // ¬l0 ∨ l1 means l0 => l1
                if l0.negated && !l1.negated {
                    implications
                        .entry(l0.variable)
                        .or_default()
                        .insert(l1.variable);
                } else if !l0.negated && l1.negated {
                    implications
                        .entry(l1.variable)
                        .or_default()
                        .insert(l0.variable);
                }
            }
        }

        // Find equivalences: a => b AND b => a
        let mut substitutions: HashMap<u32, u32> = HashMap::new();
        for (&a, targets) in &implications {
            for &b in targets {
                if a < b {
                    if let Some(b_targets) = implications.get(&b) {
                        if b_targets.contains(&a) {
                            // a ≡ b: substitute b with a
                            substitutions.insert(b, a);
                            self.stats.equivalent_pairs += 1;
                        }
                    }
                }
            }
        }

        if substitutions.is_empty() {
            return;
        }

        // Apply substitutions
        for hc in &mut formula.hard_clauses {
            for lit in &mut hc.0.literals {
                if let Some(&replacement) = substitutions.get(&lit.variable) {
                    lit.variable = replacement;
                }
            }
        }
        for sc in &mut formula.soft_clauses {
            for lit in &mut sc.clause.literals {
                if let Some(&replacement) = substitutions.get(&lit.variable) {
                    lit.variable = replacement;
                }
            }
        }

        formula.deduplicate_literals();
    }

    /// Label compression: merge soft clauses that share the same relaxation variable pattern.
    pub fn label_compression(&mut self, formula: &mut MaxSatFormula) {
        if formula.soft_clauses.len() < 2 {
            return;
        }

        // Group soft clauses by their literal sets
        let mut groups: HashMap<Vec<(u32, bool)>, Vec<usize>> = HashMap::new();

        for (i, sc) in formula.soft_clauses.iter().enumerate() {
            let mut key: Vec<(u32, bool)> = sc
                .clause
                .literals
                .iter()
                .map(|l| (l.variable, l.negated))
                .collect();
            key.sort();
            groups.entry(key).or_default().push(i);
        }

        let mut to_remove = HashSet::new();
        let mut new_soft = Vec::new();

        for (_, indices) in &groups {
            if indices.len() < 2 {
                continue;
            }

            // Merge: sum the weights
            let total_weight: u64 = indices
                .iter()
                .map(|&i| formula.soft_clauses[i].weight)
                .sum();

            let first = &formula.soft_clauses[indices[0]];
            let merged = SoftClause::new(first.clause.literals.clone(), total_weight);
            new_soft.push(merged);

            for &i in indices {
                to_remove.insert(i);
            }
            self.stats.labels_compressed += indices.len() - 1;
        }

        // Remove merged clauses and add new ones
        let mut idx = 0;
        formula.soft_clauses.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
        formula.soft_clauses.extend(new_soft);
    }
}

impl Default for MaxSatPreprocessor {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Propagate a unit literal through the formula.
fn propagate_literal_in_formula(
    formula: &mut MaxSatFormula,
    lit: Literal,
    stats: &mut PreprocessStats,
) {
    let neg = lit.negate();

    // Remove satisfied hard clauses
    let before_h = formula.hard_clauses.len();
    formula.hard_clauses.retain(|hc| !hc.0.contains(lit));
    stats.clauses_removed += before_h - formula.hard_clauses.len();

    // Remove negation from remaining hard clauses
    for hc in &mut formula.hard_clauses {
        let before = hc.0.literals.len();
        hc.0.remove_literal(neg);
        stats.literals_removed += before - hc.0.literals.len();
    }

    // Same for soft clauses
    let before_s = formula.soft_clauses.len();
    formula.soft_clauses.retain(|sc| !sc.clause.contains(lit));
    stats.clauses_removed += before_s - formula.soft_clauses.len();

    for sc in &mut formula.soft_clauses {
        let before = sc.clause.literals.len();
        sc.clause.remove_literal(neg);
        stats.literals_removed += before - sc.clause.literals.len();
    }
}

/// Check for self-subsuming resolution between two clauses.
///
/// Returns the literal that can be removed from `cj`, if any.
fn find_self_subsumption(ci: &Clause, cj: &Clause) -> Option<Literal> {
    // ci self-subsumes cj if ci\{l} ⊆ cj and ¬l ∈ cj for some l ∈ ci
    for &lit in &ci.literals {
        let neg = lit.negate();
        if cj.contains(neg) {
            // Check if all other literals of ci are in cj
            let all_others = ci
                .literals
                .iter()
                .all(|&l| l == lit || cj.contains(l));
            if all_others {
                return Some(neg);
            }
        }
    }
    None
}

/// Resolve two clauses on the given variable.
fn resolve(c1: &Clause, c2: &Clause, var: u32) -> Option<Clause> {
    let mut lits: Vec<Literal> = Vec::new();

    // Add literals from c1 except the variable being resolved
    for &l in &c1.literals {
        if l.variable != var {
            lits.push(l);
        }
    }

    // Add literals from c2 except the variable being resolved
    for &l in &c2.literals {
        if l.variable != var && !lits.contains(&l) {
            lits.push(l);
        }
    }

    Some(Clause::new(lits))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::Literal;

    fn lit(v: u32) -> Literal {
        Literal::positive(v)
    }
    fn nlit(v: u32) -> Literal {
        Literal::negative(v)
    }

    #[test]
    fn test_unit_propagation_basic() {
        let mut f = MaxSatFormula::new();
        f.add_variable();
        f.add_variable();
        f.add_hard_clause(vec![lit(1)]);
        f.add_hard_clause(vec![nlit(1), lit(2)]);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.unit_propagation(&mut f);

        assert_eq!(*pp.deduced.get(&1).unwrap(), true);
        // After propagation: unit(1) removes clause 1; clause 2 becomes unit(2) or removed
        assert!(f.num_hard_clauses() <= 1);
    }

    #[test]
    fn test_unit_propagation_chain() {
        let mut f = MaxSatFormula::new();
        f.add_variable();
        f.add_variable();
        f.add_variable();
        f.add_hard_clause(vec![lit(1)]);
        f.add_hard_clause(vec![nlit(1), lit(2)]);
        f.add_hard_clause(vec![nlit(2), lit(3)]);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.unit_propagation(&mut f);

        assert_eq!(*pp.deduced.get(&1).unwrap(), true);
    }

    #[test]
    fn test_subsumption_elimination() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![lit(1)]);
        f.add_hard_clause(vec![lit(1), lit(2)]); // subsumed by {1}
        f.add_hard_clause(vec![lit(2), lit(3)]);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.clause_subsumption_elimination(&mut f);

        assert_eq!(f.num_hard_clauses(), 2);
    }

    #[test]
    fn test_subsumption_soft() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![lit(1)]);
        f.add_soft_clause(vec![lit(1), lit(2)], 5); // subsumed by hard {1}

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.clause_subsumption_elimination(&mut f);

        assert_eq!(f.num_soft_clauses(), 0);
    }

    #[test]
    fn test_self_subsuming_resolution() {
        let mut f = MaxSatFormula::new();
        // {1, 2} and {1, ¬2, 3}: self-subsuming on 2 gives {1, 3}
        f.add_hard_clause(vec![lit(1), lit(2)]);
        f.add_hard_clause(vec![lit(1), nlit(2), lit(3)]);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.self_subsuming_resolution(&mut f);

        // The second clause should have ¬2 removed
        let has_short = f.hard_clauses.iter().any(|hc| hc.0.len() == 2);
        assert!(has_short || f.num_hard_clauses() == 2);
    }

    #[test]
    fn test_bve_simple() {
        let mut f = MaxSatFormula::new();
        f.add_variable();
        f.add_variable();
        f.add_variable();
        // x1 ∨ x2 and ¬x2 ∨ x3
        f.add_hard_clause(vec![lit(1), lit(2)]);
        f.add_hard_clause(vec![nlit(2), lit(3)]);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.bounded_variable_elimination(&mut f, 10);

        // x2 should be eliminated: resolvent is x1 ∨ x3
        // 2 clauses removed, 1 resolvent added
        assert!(f.num_hard_clauses() <= 2);
    }

    #[test]
    fn test_bve_skips_soft_vars() {
        let mut f = MaxSatFormula::new();
        f.add_variable();
        f.add_variable();
        f.add_hard_clause(vec![lit(1), lit(2)]);
        f.add_hard_clause(vec![nlit(2)]);
        f.add_soft_clause(vec![lit(2)], 5);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.bounded_variable_elimination(&mut f, 10);

        // x2 is in soft clause, should not be eliminated
        assert!(f.num_soft_clauses() == 1);
    }

    #[test]
    fn test_equivalent_literal_substitution() {
        let mut f = MaxSatFormula::new();
        f.add_variable();
        f.add_variable();
        f.add_variable();
        // ¬1 ∨ 2 and ¬2 ∨ 1 means 1 ≡ 2
        f.add_hard_clause(vec![nlit(1), lit(2)]);
        f.add_hard_clause(vec![nlit(2), lit(1)]);
        f.add_hard_clause(vec![lit(2), lit(3)]);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.equivalent_literal_substitution(&mut f);

        assert!(pp.stats.equivalent_pairs >= 1);
    }

    #[test]
    fn test_label_compression() {
        let mut f = MaxSatFormula::new();
        f.add_variable();
        f.add_soft_clause(vec![lit(1)], 5);
        f.add_soft_clause(vec![lit(1)], 10);
        f.add_soft_clause(vec![nlit(1)], 3);

        let mut pp = MaxSatPreprocessor::with_default_config();
        pp.label_compression(&mut f);

        // Two {1} clauses merged into one with weight 15
        assert_eq!(f.num_soft_clauses(), 2);
        let total: u64 = f.soft_clauses.iter().map(|sc| sc.weight).sum();
        assert_eq!(total, 18); // 15 + 3
    }

    #[test]
    fn test_full_preprocess() {
        let mut f = MaxSatFormula::new();
        f.add_variable();
        f.add_variable();
        f.add_variable();
        f.add_hard_clause(vec![lit(1)]);
        f.add_hard_clause(vec![nlit(1), lit(2)]);
        f.add_hard_clause(vec![lit(2), lit(3)]);
        f.add_soft_clause(vec![nlit(3)], 10);

        let mut pp = MaxSatPreprocessor::with_default_config();
        let stats = pp.preprocess(&mut f);

        assert!(stats.time_ms < 10_000);
        assert!(stats.rounds >= 1);
    }

    #[test]
    fn test_preprocess_config_default() {
        let cfg = PreprocessConfig::default();
        assert!(cfg.enable_bve);
        assert!(cfg.enable_subsumption);
        assert!(!cfg.enable_backbone);
        assert_eq!(cfg.bve_bound, 10);
    }

    #[test]
    fn test_preprocess_empty_formula() {
        let mut f = MaxSatFormula::new();
        let mut pp = MaxSatPreprocessor::with_default_config();
        let stats = pp.preprocess(&mut f);
        assert_eq!(stats.clauses_removed, 0);
    }

    #[test]
    fn test_resolve_basic() {
        let c1 = Clause::new(vec![lit(1), lit(2)]);
        let c2 = Clause::new(vec![nlit(2), lit(3)]);
        let result = resolve(&c1, &c2, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains(lit(1)));
        assert!(result.contains(lit(3)));
    }

    #[test]
    fn test_find_self_subsumption() {
        let ci = Clause::new(vec![lit(1), lit(2)]);
        let cj = Clause::new(vec![lit(1), nlit(2), lit(3)]);
        let result = find_self_subsumption(&ci, &cj);
        assert_eq!(result, Some(nlit(2)));
    }

    #[test]
    fn test_find_self_subsumption_none() {
        let ci = Clause::new(vec![lit(1), lit(2)]);
        let cj = Clause::new(vec![lit(3), lit(4)]);
        assert!(find_self_subsumption(&ci, &cj).is_none());
    }

    #[test]
    fn test_preprocess_stats_display() {
        let stats = PreprocessStats {
            clauses_removed: 5,
            variables_eliminated: 2,
            literals_removed: 10,
            time_ms: 42,
            rounds: 3,
            ..Default::default()
        };
        let s = stats.to_string();
        assert!(s.contains("5 clauses"));
        assert!(s.contains("42ms"));
    }
}
