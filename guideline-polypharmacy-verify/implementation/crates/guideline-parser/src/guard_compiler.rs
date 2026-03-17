//! Guard expression compilation and optimisation.
//!
//! Converts [`GuidelineGuard`] trees into PTA [`Guard`] trees and then
//! applies a series of simplification / optimisation passes.

use crate::format::{ComparisonOp, GuidelineGuard};
use crate::pta_builder::{Guard, PtaBuilder};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GuardCompiler
// ---------------------------------------------------------------------------

/// Compiles guideline guards into PTA guards and optimises them.
#[derive(Debug, Clone)]
pub struct GuardCompiler {
    /// Optimisation passes to run after compilation.
    pub passes: Vec<OptimisationPass>,
    /// Maximum depth allowed after compilation (for safety).
    pub max_depth: usize,
}

/// Named optimisation passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimisationPass {
    /// Remove True / False leaves from conjunctions / disjunctions.
    SimplifyIdentities,
    /// Flatten nested And(And(...)) / Or(Or(...)).
    Flatten,
    /// Remove tautologies (e.g. x > 5 AND x > 3 ⟹ x > 5).
    RemoveTautologies,
    /// Merge overlapping range constraints on the same variable.
    MergeRanges,
    /// De-duplicate identical sub-guards.
    Deduplicate,
    /// Push negations inward (De Morgan).
    PushNegation,
}

impl Default for GuardCompiler {
    fn default() -> Self {
        Self {
            passes: vec![
                OptimisationPass::SimplifyIdentities,
                OptimisationPass::Flatten,
                OptimisationPass::RemoveTautologies,
                OptimisationPass::MergeRanges,
                OptimisationPass::Deduplicate,
                OptimisationPass::PushNegation,
            ],
            max_depth: 50,
        }
    }
}

impl GuardCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a compiler with no optimisation passes.
    pub fn unoptimised() -> Self {
        Self {
            passes: vec![],
            max_depth: 50,
        }
    }

    /// Create a compiler with only the specified passes.
    pub fn with_passes(passes: Vec<OptimisationPass>) -> Self {
        Self { passes, max_depth: 50 }
    }

    /// Compile a `GuidelineGuard` to a PTA `Guard` and apply all optimisation
    /// passes.
    pub fn compile(&self, gg: &GuidelineGuard) -> Guard {
        let builder = PtaBuilder::new();
        let mut guard = builder.compile_guideline_guard(gg);

        for pass in &self.passes {
            guard = self.apply_pass(guard, *pass);
        }

        guard
    }

    /// Compile without optimisation.
    pub fn compile_raw(&self, gg: &GuidelineGuard) -> Guard {
        let builder = PtaBuilder::new();
        builder.compile_guideline_guard(gg)
    }

    /// Run a single optimisation pass.
    pub fn apply_pass(&self, guard: Guard, pass: OptimisationPass) -> Guard {
        match pass {
            OptimisationPass::SimplifyIdentities => simplify_identities(guard),
            OptimisationPass::Flatten => flatten(guard),
            OptimisationPass::RemoveTautologies => remove_tautologies(guard),
            OptimisationPass::MergeRanges => merge_ranges(guard),
            OptimisationPass::Deduplicate => deduplicate(guard),
            OptimisationPass::PushNegation => push_negation(guard),
        }
    }

    /// Apply all configured passes in sequence.
    pub fn optimise(&self, guard: Guard) -> Guard {
        let mut g = guard;
        for pass in &self.passes {
            g = self.apply_pass(g, *pass);
        }
        g
    }
}

// ---------------------------------------------------------------------------
// GuardOptimizer — standalone optimiser working on PTA Guard trees
// ---------------------------------------------------------------------------

/// Standalone guard optimiser (operates on already-compiled PTA guards).
#[derive(Debug, Clone)]
pub struct GuardOptimizer {
    passes: Vec<OptimisationPass>,
}

impl Default for GuardOptimizer {
    fn default() -> Self {
        Self {
            passes: vec![
                OptimisationPass::SimplifyIdentities,
                OptimisationPass::Flatten,
                OptimisationPass::RemoveTautologies,
                OptimisationPass::MergeRanges,
                OptimisationPass::Deduplicate,
            ],
        }
    }
}

impl GuardOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_passes(passes: Vec<OptimisationPass>) -> Self {
        Self { passes }
    }

    pub fn optimize(&self, guard: Guard) -> Guard {
        let mut g = guard;
        for pass in &self.passes {
            g = match pass {
                OptimisationPass::SimplifyIdentities => simplify_identities(g),
                OptimisationPass::Flatten => flatten(g),
                OptimisationPass::RemoveTautologies => remove_tautologies(g),
                OptimisationPass::MergeRanges => merge_ranges(g),
                OptimisationPass::Deduplicate => deduplicate(g),
                OptimisationPass::PushNegation => push_negation(g),
            };
        }
        g
    }

    /// Run a fixed-point loop until no more changes occur (up to `max_iter`).
    pub fn optimize_fixed_point(&self, guard: Guard, max_iter: usize) -> Guard {
        let mut current = guard;
        for _ in 0..max_iter {
            let next = self.optimize(current.clone());
            let cur_json = serde_json::to_string(&current).unwrap_or_default();
            let nxt_json = serde_json::to_string(&next).unwrap_or_default();
            if cur_json == nxt_json {
                return next;
            }
            current = next;
        }
        current
    }
}

// ---------------------------------------------------------------------------
// Pass implementations
// ---------------------------------------------------------------------------

/// Remove `True` from `And(...)` and `False` from `Or(...)`.
/// Collapse single-element wrappers.
fn simplify_identities(guard: Guard) -> Guard {
    match guard {
        Guard::And(parts) => {
            let simplified: Vec<Guard> = parts
                .into_iter()
                .map(simplify_identities)
                .filter(|g| !matches!(g, Guard::True))
                .collect();
            if simplified.iter().any(|g| matches!(g, Guard::False)) {
                return Guard::False;
            }
            match simplified.len() {
                0 => Guard::True,
                1 => simplified.into_iter().next().unwrap(),
                _ => Guard::And(simplified),
            }
        }
        Guard::Or(parts) => {
            let simplified: Vec<Guard> = parts
                .into_iter()
                .map(simplify_identities)
                .filter(|g| !matches!(g, Guard::False))
                .collect();
            if simplified.iter().any(|g| matches!(g, Guard::True)) {
                return Guard::True;
            }
            match simplified.len() {
                0 => Guard::False,
                1 => simplified.into_iter().next().unwrap(),
                _ => Guard::Or(simplified),
            }
        }
        Guard::Not(inner) => {
            let inner = simplify_identities(*inner);
            match inner {
                Guard::True => Guard::False,
                Guard::False => Guard::True,
                Guard::Not(double) => *double,
                other => Guard::Not(Box::new(other)),
            }
        }
        other => other,
    }
}

/// Flatten nested `And(And(...))` and `Or(Or(...))`.
fn flatten(guard: Guard) -> Guard {
    match guard {
        Guard::And(parts) => {
            let mut flat = Vec::new();
            for p in parts {
                let p = flatten(p);
                match p {
                    Guard::And(inner) => flat.extend(inner),
                    other => flat.push(other),
                }
            }
            match flat.len() {
                0 => Guard::True,
                1 => flat.into_iter().next().unwrap(),
                _ => Guard::And(flat),
            }
        }
        Guard::Or(parts) => {
            let mut flat = Vec::new();
            for p in parts {
                let p = flatten(p);
                match p {
                    Guard::Or(inner) => flat.extend(inner),
                    other => flat.push(other),
                }
            }
            match flat.len() {
                0 => Guard::False,
                1 => flat.into_iter().next().unwrap(),
                _ => Guard::Or(flat),
            }
        }
        Guard::Not(inner) => {
            let inner = flatten(*inner);
            Guard::Not(Box::new(inner))
        }
        other => other,
    }
}

/// Key for grouping variable constraints.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VarKey {
    variable: String,
}

/// In an `And(...)`, if we have `x >= 5` and `x >= 3` keep only `x >= 5`.
/// Similarly `x < 10` and `x < 7` keep only `x < 7`.
fn remove_tautologies(guard: Guard) -> Guard {
    match guard {
        Guard::And(parts) => {
            let parts: Vec<Guard> = parts.into_iter().map(remove_tautologies).collect();
            // Group variable constraints by variable name
            let mut lower_bounds: HashMap<String, f64> = HashMap::new();
            let mut upper_bounds: HashMap<String, f64> = HashMap::new();
            let mut kept = Vec::new();
            let mut handled_vars: HashMap<String, Vec<usize>> = HashMap::new();

            for (i, p) in parts.iter().enumerate() {
                match p {
                    Guard::VariableConstraint { variable, op, value } => {
                        match op {
                            ComparisonOp::Ge | ComparisonOp::Gt => {
                                let entry = lower_bounds.entry(variable.clone()).or_insert(f64::NEG_INFINITY);
                                if *value > *entry {
                                    *entry = *value;
                                }
                                handled_vars.entry(variable.clone()).or_default().push(i);
                            }
                            ComparisonOp::Le | ComparisonOp::Lt => {
                                let entry = upper_bounds.entry(variable.clone()).or_insert(f64::INFINITY);
                                if *value < *entry {
                                    *entry = *value;
                                }
                                handled_vars.entry(variable.clone()).or_default().push(i);
                            }
                            _ => {
                                kept.push(p.clone());
                            }
                        }
                    }
                    Guard::ClockConstraint { clock, op, value } => {
                        match op {
                            ComparisonOp::Ge | ComparisonOp::Gt => {
                                let key = format!("__clock_{}", clock);
                                let entry = lower_bounds.entry(key.clone()).or_insert(f64::NEG_INFINITY);
                                if *value > *entry {
                                    *entry = *value;
                                }
                                handled_vars.entry(key).or_default().push(i);
                            }
                            ComparisonOp::Le | ComparisonOp::Lt => {
                                let key = format!("__clock_{}", clock);
                                let entry = upper_bounds.entry(key.clone()).or_insert(f64::INFINITY);
                                if *value < *entry {
                                    *entry = *value;
                                }
                                handled_vars.entry(key).or_default().push(i);
                            }
                            _ => {
                                kept.push(p.clone());
                            }
                        }
                    }
                    other => {
                        kept.push(other.clone());
                    }
                }
            }

            // Reconstruct tightened constraints
            let all_vars: std::collections::HashSet<String> = lower_bounds
                .keys()
                .chain(upper_bounds.keys())
                .cloned()
                .collect();

            for var in all_vars {
                if var.starts_with("__clock_") {
                    let clock = var.trim_start_matches("__clock_").to_string();
                    if let Some(&lb) = lower_bounds.get(&var) {
                        if lb > f64::NEG_INFINITY {
                            kept.push(Guard::ClockConstraint {
                                clock: clock.clone(),
                                op: ComparisonOp::Ge,
                                value: lb,
                            });
                        }
                    }
                    if let Some(&ub) = upper_bounds.get(&var) {
                        if ub < f64::INFINITY {
                            kept.push(Guard::ClockConstraint {
                                clock,
                                op: ComparisonOp::Le,
                                value: ub,
                            });
                        }
                    }
                } else {
                    if let Some(&lb) = lower_bounds.get(&var) {
                        if lb > f64::NEG_INFINITY {
                            kept.push(Guard::VariableConstraint {
                                variable: var.clone(),
                                op: ComparisonOp::Ge,
                                value: lb,
                            });
                        }
                    }
                    if let Some(&ub) = upper_bounds.get(&var) {
                        if ub < f64::INFINITY {
                            kept.push(Guard::VariableConstraint {
                                variable: var.clone(),
                                op: ComparisonOp::Le,
                                value: ub,
                            });
                        }
                    }
                }
            }

            match kept.len() {
                0 => Guard::True,
                1 => kept.into_iter().next().unwrap(),
                _ => Guard::And(kept),
            }
        }
        Guard::Or(parts) => {
            let parts: Vec<Guard> = parts.into_iter().map(remove_tautologies).collect();
            Guard::Or(parts)
        }
        Guard::Not(inner) => {
            let inner = remove_tautologies(*inner);
            Guard::Not(Box::new(inner))
        }
        other => other,
    }
}

/// Merge adjacent `VariableConstraint` on the same variable into a
/// `RangeConstraint` when we have both lower and upper bound in an `And`.
fn merge_ranges(guard: Guard) -> Guard {
    match guard {
        Guard::And(parts) => {
            let parts: Vec<Guard> = parts.into_iter().map(merge_ranges).collect();

            // Collect lower/upper bounds per variable
            let mut lowers: HashMap<String, f64> = HashMap::new();
            let mut uppers: HashMap<String, f64> = HashMap::new();
            let mut other = Vec::new();

            for p in &parts {
                match p {
                    Guard::VariableConstraint { variable, op, value } => match op {
                        ComparisonOp::Ge | ComparisonOp::Gt => {
                            lowers
                                .entry(variable.clone())
                                .and_modify(|v| {
                                    if *value > *v {
                                        *v = *value;
                                    }
                                })
                                .or_insert(*value);
                        }
                        ComparisonOp::Le | ComparisonOp::Lt => {
                            uppers
                                .entry(variable.clone())
                                .and_modify(|v| {
                                    if *value < *v {
                                        *v = *value;
                                    }
                                })
                                .or_insert(*value);
                        }
                        _ => other.push(p.clone()),
                    },
                    _ => other.push(p.clone()),
                }
            }

            // For variables with both bounds, produce a RangeConstraint
            let both_vars: Vec<String> = lowers
                .keys()
                .filter(|k| uppers.contains_key(*k))
                .cloned()
                .collect();

            for var in &both_vars {
                let lo = lowers.remove(var).unwrap();
                let hi = uppers.remove(var).unwrap();
                other.push(Guard::RangeConstraint {
                    variable: var.clone(),
                    min: lo,
                    max: hi,
                });
            }

            // Remaining single-sided constraints
            for (var, val) in lowers {
                other.push(Guard::VariableConstraint {
                    variable: var,
                    op: ComparisonOp::Ge,
                    value: val,
                });
            }
            for (var, val) in uppers {
                other.push(Guard::VariableConstraint {
                    variable: var,
                    op: ComparisonOp::Le,
                    value: val,
                });
            }

            match other.len() {
                0 => Guard::True,
                1 => other.into_iter().next().unwrap(),
                _ => Guard::And(other),
            }
        }
        Guard::Or(parts) => {
            let parts: Vec<Guard> = parts.into_iter().map(merge_ranges).collect();
            Guard::Or(parts)
        }
        Guard::Not(inner) => Guard::Not(Box::new(merge_ranges(*inner))),
        other => other,
    }
}

/// Remove duplicate sub-guards in `And` / `Or`.
fn deduplicate(guard: Guard) -> Guard {
    match guard {
        Guard::And(parts) => {
            let parts: Vec<Guard> = parts.into_iter().map(deduplicate).collect();
            let mut unique = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for p in parts {
                let key = serde_json::to_string(&p).unwrap_or_default();
                if seen.insert(key) {
                    unique.push(p);
                }
            }
            match unique.len() {
                0 => Guard::True,
                1 => unique.into_iter().next().unwrap(),
                _ => Guard::And(unique),
            }
        }
        Guard::Or(parts) => {
            let parts: Vec<Guard> = parts.into_iter().map(deduplicate).collect();
            let mut unique = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for p in parts {
                let key = serde_json::to_string(&p).unwrap_or_default();
                if seen.insert(key) {
                    unique.push(p);
                }
            }
            match unique.len() {
                0 => Guard::False,
                1 => unique.into_iter().next().unwrap(),
                _ => Guard::Or(unique),
            }
        }
        Guard::Not(inner) => Guard::Not(Box::new(deduplicate(*inner))),
        other => other,
    }
}

/// Push negations inward using De Morgan's laws.
fn push_negation(guard: Guard) -> Guard {
    match guard {
        Guard::Not(inner) => {
            let inner = push_negation(*inner);
            match inner {
                Guard::And(parts) => {
                    Guard::Or(parts.into_iter().map(|p| Guard::Not(Box::new(p))).collect())
                }
                Guard::Or(parts) => {
                    Guard::And(parts.into_iter().map(|p| Guard::Not(Box::new(p))).collect())
                }
                Guard::Not(double) => *double,
                Guard::True => Guard::False,
                Guard::False => Guard::True,
                Guard::VariableConstraint { variable, op, value } => {
                    Guard::VariableConstraint {
                        variable,
                        op: op.negate(),
                        value,
                    }
                }
                Guard::ClockConstraint { clock, op, value } => Guard::ClockConstraint {
                    clock,
                    op: op.negate(),
                    value,
                },
                other => Guard::Not(Box::new(other)),
            }
        }
        Guard::And(parts) => Guard::And(parts.into_iter().map(push_negation).collect()),
        Guard::Or(parts) => Guard::Or(parts.into_iter().map(push_negation).collect()),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Additional utilities
// ---------------------------------------------------------------------------

/// Count the total number of nodes in a guard tree.
pub fn guard_node_count(guard: &Guard) -> usize {
    match guard {
        Guard::And(parts) | Guard::Or(parts) => {
            1 + parts.iter().map(guard_node_count).sum::<usize>()
        }
        Guard::Not(inner) => 1 + guard_node_count(inner),
        _ => 1,
    }
}

/// Collect all variable names referenced by the guard.
pub fn guard_variables(guard: &Guard) -> Vec<String> {
    let mut vars = std::collections::HashSet::new();
    collect_vars(guard, &mut vars);
    let mut out: Vec<String> = vars.into_iter().collect();
    out.sort();
    out
}

fn collect_vars(guard: &Guard, acc: &mut std::collections::HashSet<String>) {
    match guard {
        Guard::VariableConstraint { variable, .. }
        | Guard::BooleanVariable { variable, .. }
        | Guard::RangeConstraint { variable, .. } => {
            acc.insert(variable.clone());
        }
        Guard::ClockConstraint { clock, .. } => {
            acc.insert(clock.clone());
        }
        Guard::And(parts) | Guard::Or(parts) => {
            for p in parts {
                collect_vars(p, acc);
            }
        }
        Guard::Not(inner) => collect_vars(inner, acc),
        _ => {}
    }
}

/// Pretty-print a guard expression as a human-readable string.
pub fn guard_to_string(guard: &Guard) -> String {
    match guard {
        Guard::ClockConstraint { clock, op, value } => {
            format!("{} {} {}", clock, op, value)
        }
        Guard::VariableConstraint { variable, op, value } => {
            format!("{} {} {}", variable, op, value)
        }
        Guard::BooleanVariable { variable, expected } => {
            if *expected {
                variable.clone()
            } else {
                format!("!{}", variable)
            }
        }
        Guard::RangeConstraint { variable, min, max } => {
            format!("{} <= {} <= {}", min, variable, max)
        }
        Guard::And(parts) => {
            let strs: Vec<String> = parts.iter().map(guard_to_string).collect();
            format!("({})", strs.join(" ∧ "))
        }
        Guard::Or(parts) => {
            let strs: Vec<String> = parts.iter().map(guard_to_string).collect();
            format!("({})", strs.join(" ∨ "))
        }
        Guard::Not(inner) => format!("¬({})", guard_to_string(inner)),
        Guard::True => "true".into(),
        Guard::False => "false".into(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ComparisonOp, GuidelineGuard};

    #[test]
    fn test_compile_simple() {
        let compiler = GuardCompiler::new();
        let gg = GuidelineGuard::LabThreshold {
            test_name: "HbA1c".into(),
            operator: ComparisonOp::Ge,
            value: 7.0,
            unit: None,
        };
        let guard = compiler.compile(&gg);
        match &guard {
            Guard::VariableConstraint { variable, op, value } => {
                assert_eq!(variable, "HbA1c");
                assert_eq!(*op, ComparisonOp::Ge);
                assert_eq!(*value, 7.0);
            }
            _ => panic!("Expected VariableConstraint"),
        }
    }

    #[test]
    fn test_simplify_true_and() {
        let guard = Guard::And(vec![
            Guard::True,
            Guard::VariableConstraint {
                variable: "x".into(),
                op: ComparisonOp::Gt,
                value: 5.0,
            },
            Guard::True,
        ]);
        let simplified = simplify_identities(guard);
        match &simplified {
            Guard::VariableConstraint { variable, .. } => assert_eq!(variable, "x"),
            _ => panic!("Expected single VariableConstraint, got {:?}", simplified),
        }
    }

    #[test]
    fn test_simplify_false_or() {
        let guard = Guard::Or(vec![
            Guard::False,
            Guard::VariableConstraint {
                variable: "y".into(),
                op: ComparisonOp::Lt,
                value: 10.0,
            },
        ]);
        let simplified = simplify_identities(guard);
        match &simplified {
            Guard::VariableConstraint { variable, .. } => assert_eq!(variable, "y"),
            _ => panic!("Expected VariableConstraint"),
        }
    }

    #[test]
    fn test_flatten_nested_and() {
        let guard = Guard::And(vec![
            Guard::VariableConstraint {
                variable: "a".into(),
                op: ComparisonOp::Gt,
                value: 1.0,
            },
            Guard::And(vec![
                Guard::VariableConstraint {
                    variable: "b".into(),
                    op: ComparisonOp::Lt,
                    value: 2.0,
                },
                Guard::VariableConstraint {
                    variable: "c".into(),
                    op: ComparisonOp::Eq,
                    value: 3.0,
                },
            ]),
        ]);
        let flat = flatten(guard);
        if let Guard::And(parts) = &flat {
            assert_eq!(parts.len(), 3);
        } else {
            panic!("Expected flat And with 3 parts");
        }
    }

    #[test]
    fn test_remove_tautologies() {
        // x >= 5 AND x >= 3 should keep only x >= 5
        let guard = Guard::And(vec![
            Guard::VariableConstraint {
                variable: "x".into(),
                op: ComparisonOp::Ge,
                value: 5.0,
            },
            Guard::VariableConstraint {
                variable: "x".into(),
                op: ComparisonOp::Ge,
                value: 3.0,
            },
        ]);
        let result = remove_tautologies(guard);
        match &result {
            Guard::VariableConstraint { variable, op, value } => {
                assert_eq!(variable, "x");
                assert_eq!(*op, ComparisonOp::Ge);
                assert_eq!(*value, 5.0);
            }
            _ => panic!("Expected single constraint, got {:?}", result),
        }
    }

    #[test]
    fn test_merge_ranges() {
        // x >= 5 AND x <= 10 should merge into 5 <= x <= 10
        let guard = Guard::And(vec![
            Guard::VariableConstraint {
                variable: "x".into(),
                op: ComparisonOp::Ge,
                value: 5.0,
            },
            Guard::VariableConstraint {
                variable: "x".into(),
                op: ComparisonOp::Le,
                value: 10.0,
            },
        ]);
        let result = merge_ranges(guard);
        match &result {
            Guard::RangeConstraint { variable, min, max } => {
                assert_eq!(variable, "x");
                assert_eq!(*min, 5.0);
                assert_eq!(*max, 10.0);
            }
            _ => panic!("Expected RangeConstraint, got {:?}", result),
        }
    }

    #[test]
    fn test_deduplicate() {
        let g = Guard::VariableConstraint {
            variable: "x".into(),
            op: ComparisonOp::Gt,
            value: 5.0,
        };
        let guard = Guard::And(vec![g.clone(), g.clone(), g.clone()]);
        let result = deduplicate(guard);
        match &result {
            Guard::VariableConstraint { .. } => {} // collapsed to single
            _ => panic!("Expected single constraint after dedup"),
        }
    }

    #[test]
    fn test_push_negation() {
        // NOT(a AND b) should become (NOT a) OR (NOT b)
        let guard = Guard::Not(Box::new(Guard::And(vec![
            Guard::VariableConstraint {
                variable: "a".into(),
                op: ComparisonOp::Gt,
                value: 1.0,
            },
            Guard::VariableConstraint {
                variable: "b".into(),
                op: ComparisonOp::Lt,
                value: 2.0,
            },
        ])));
        let result = push_negation(guard);
        match &result {
            Guard::Or(parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("Expected Or after De Morgan, got {:?}", result),
        }
    }

    #[test]
    fn test_guard_node_count() {
        let guard = Guard::And(vec![
            Guard::VariableConstraint {
                variable: "x".into(),
                op: ComparisonOp::Gt,
                value: 5.0,
            },
            Guard::Or(vec![
                Guard::True,
                Guard::ClockConstraint {
                    clock: "c".into(),
                    op: ComparisonOp::Le,
                    value: 10.0,
                },
            ]),
        ]);
        assert_eq!(guard_node_count(&guard), 5);
    }

    #[test]
    fn test_guard_to_string() {
        let guard = Guard::And(vec![
            Guard::VariableConstraint {
                variable: "HbA1c".into(),
                op: ComparisonOp::Ge,
                value: 7.0,
            },
            Guard::ClockConstraint {
                clock: "t".into(),
                op: ComparisonOp::Le,
                value: 90.0,
            },
        ]);
        let s = guard_to_string(&guard);
        assert!(s.contains("HbA1c"));
        assert!(s.contains("∧"));
    }

    #[test]
    fn test_compile_complex() {
        let compiler = GuardCompiler::new();
        let gg = GuidelineGuard::And(vec![
            GuidelineGuard::LabThreshold {
                test_name: "HbA1c".into(),
                operator: ComparisonOp::Ge,
                value: 7.5,
                unit: None,
            },
            GuidelineGuard::LabThreshold {
                test_name: "HbA1c".into(),
                operator: ComparisonOp::Lt,
                value: 9.0,
                unit: None,
            },
            GuidelineGuard::TimeElapsed {
                clock: "treatment".into(),
                operator: ComparisonOp::Ge,
                days: 90.0,
            },
        ]);
        let guard = compiler.compile(&gg);
        let vars = guard_variables(&guard);
        assert!(vars.contains(&"HbA1c".to_string()) || vars.contains(&"treatment".to_string()));
    }

    #[test]
    fn test_optimizer_fixed_point() {
        let optimizer = GuardOptimizer::new();
        let guard = Guard::And(vec![
            Guard::True,
            Guard::And(vec![
                Guard::VariableConstraint {
                    variable: "x".into(),
                    op: ComparisonOp::Ge,
                    value: 5.0,
                },
                Guard::VariableConstraint {
                    variable: "x".into(),
                    op: ComparisonOp::Ge,
                    value: 3.0,
                },
            ]),
        ]);
        let result = optimizer.optimize_fixed_point(guard, 10);
        // Should converge to a single constraint
        let count = guard_node_count(&result);
        assert!(count <= 3, "Expected simplified guard, got {} nodes", count);
    }

    #[test]
    fn test_unoptimised_compiler() {
        let compiler = GuardCompiler::unoptimised();
        let gg = GuidelineGuard::And(vec![GuidelineGuard::True, GuidelineGuard::True]);
        let guard = compiler.compile(&gg);
        // Without optimisation, the True leaves remain
        match &guard {
            Guard::And(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected And with 2 True parts without optimisation"),
        }
    }
}
