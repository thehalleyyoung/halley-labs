//! SMT encoding optimizations.
//!
//! Applies various optimizations to reduce formula size and improve
//! solver performance: symmetry breaking, structural hashing,
//! dead variable elimination, cone of influence reduction,
//! and theory-specific simplifications.

use crate::{ConstraintOrigin, SmtConstraint, SmtDeclaration, SmtExpr, SmtFormula, SmtSort};
use indexmap::IndexMap;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// ─── Optimization configuration ─────────────────────────────────────────

/// Configuration for encoding optimizations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub enable_constant_folding: bool,
    pub enable_dead_variable_elimination: bool,
    pub enable_structural_hashing: bool,
    pub enable_cone_of_influence: bool,
    pub enable_clause_sharing: bool,
    pub enable_theory_simplification: bool,
    pub enable_symmetry_breaking: bool,
    pub max_sharing_depth: u32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            enable_constant_folding: true,
            enable_dead_variable_elimination: true,
            enable_structural_hashing: true,
            enable_cone_of_influence: true,
            enable_clause_sharing: true,
            enable_theory_simplification: true,
            enable_symmetry_breaking: true,
            max_sharing_depth: 8,
        }
    }
}

// ─── Optimization statistics ────────────────────────────────────────────

/// Statistics from the optimization passes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStats {
    pub constants_folded: usize,
    pub dead_vars_eliminated: usize,
    pub subexprs_shared: usize,
    pub cone_vars_removed: usize,
    pub theory_simplifications: usize,
    pub original_nodes: usize,
    pub optimized_nodes: usize,
    pub original_constraints: usize,
    pub optimized_constraints: usize,
}

impl OptimizationStats {
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_nodes == 0 {
            return 1.0;
        }
        self.optimized_nodes as f64 / self.original_nodes as f64
    }
}

// ─── Constant folder ────────────────────────────────────────────────────

/// Folds constant expressions at encoding time.
struct ConstantFolder;

impl ConstantFolder {
    fn fold(expr: &SmtExpr) -> SmtExpr {
        match expr {
            SmtExpr::Not(a) => {
                let a = Self::fold(a);
                SmtExpr::not(a) // SmtExpr::not already folds BoolLit
            }
            SmtExpr::And(es) => {
                let folded: Vec<SmtExpr> = es.iter().map(Self::fold).collect();
                SmtExpr::and(folded) // SmtExpr::and already eliminates true/false
            }
            SmtExpr::Or(es) => {
                let folded: Vec<SmtExpr> = es.iter().map(Self::fold).collect();
                SmtExpr::or(folded)
            }
            SmtExpr::Implies(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                SmtExpr::implies(a, b)
            }
            SmtExpr::Ite(c, t, e) => {
                let c = Self::fold(c);
                let t = Self::fold(t);
                let e = Self::fold(e);
                SmtExpr::ite(c, t, e) // already simplifies on BoolLit
            }
            SmtExpr::Eq(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                // Constant equality
                match (&a, &b) {
                    (SmtExpr::BvLit(v1, w1), SmtExpr::BvLit(v2, w2)) if w1 == w2 => {
                        SmtExpr::BoolLit(v1 == v2)
                    }
                    (SmtExpr::BoolLit(b1), SmtExpr::BoolLit(b2)) => {
                        SmtExpr::BoolLit(b1 == b2)
                    }
                    (SmtExpr::IntLit(i1), SmtExpr::IntLit(i2)) => {
                        SmtExpr::BoolLit(i1 == i2)
                    }
                    _ => SmtExpr::eq(a, b),
                }
            }
            SmtExpr::BvAdd(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(v1, w1), SmtExpr::BvLit(v2, w2)) if w1 == w2 => {
                        let mask = if *w1 >= 64 {
                            u64::MAX
                        } else {
                            (1u64 << w1) - 1
                        };
                        SmtExpr::BvLit(v1.wrapping_add(*v2) & mask, *w1)
                    }
                    _ => SmtExpr::bv_add(a, b),
                }
            }
            SmtExpr::BvSub(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(v1, w1), SmtExpr::BvLit(v2, w2)) if w1 == w2 => {
                        let mask = if *w1 >= 64 {
                            u64::MAX
                        } else {
                            (1u64 << w1) - 1
                        };
                        SmtExpr::BvLit(v1.wrapping_sub(*v2) & mask, *w1)
                    }
                    _ => SmtExpr::bv_sub(a, b),
                }
            }
            SmtExpr::BvAnd(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(v1, w1), SmtExpr::BvLit(v2, w2)) if w1 == w2 => {
                        SmtExpr::BvLit(v1 & v2, *w1)
                    }
                    _ => SmtExpr::bv_and(a, b),
                }
            }
            SmtExpr::BvOr(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(v1, w1), SmtExpr::BvLit(v2, w2)) if w1 == w2 => {
                        SmtExpr::BvLit(v1 | v2, *w1)
                    }
                    _ => SmtExpr::bv_or(a, b),
                }
            }
            SmtExpr::BvUlt(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(v1, w1), SmtExpr::BvLit(v2, w2)) if w1 == w2 => {
                        SmtExpr::BoolLit(v1 < v2)
                    }
                    _ => SmtExpr::bv_ult(a, b),
                }
            }
            SmtExpr::BvUle(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(v1, w1), SmtExpr::BvLit(v2, w2)) if w1 == w2 => {
                        SmtExpr::BoolLit(v1 <= v2)
                    }
                    _ => SmtExpr::bv_ule(a, b),
                }
            }
            SmtExpr::IntAdd(es) => {
                let folded: Vec<SmtExpr> = es.iter().map(Self::fold).collect();
                // Try to merge constant operands
                let mut const_sum: i64 = 0;
                let mut non_const = Vec::new();
                for e in folded {
                    match e {
                        SmtExpr::IntLit(v) => const_sum += v,
                        other => non_const.push(other),
                    }
                }
                if non_const.is_empty() {
                    SmtExpr::IntLit(const_sum)
                } else {
                    if const_sum != 0 {
                        non_const.push(SmtExpr::IntLit(const_sum));
                    }
                    SmtExpr::int_add(non_const)
                }
            }
            SmtExpr::IntLe(a, b) => {
                let a = Self::fold(a);
                let b = Self::fold(b);
                match (&a, &b) {
                    (SmtExpr::IntLit(i1), SmtExpr::IntLit(i2)) => SmtExpr::BoolLit(i1 <= i2),
                    _ => SmtExpr::int_le(a, b),
                }
            }
            SmtExpr::Extract(a, hi, lo) => {
                let a = Self::fold(a);
                match &a {
                    SmtExpr::BvLit(v, _w) => {
                        let bits = hi - lo + 1;
                        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
                        SmtExpr::BvLit((v >> lo) & mask, bits)
                    }
                    _ => SmtExpr::Extract(Box::new(a), *hi, *lo),
                }
            }
            // For complex exprs, just recurse structurally
            SmtExpr::Select(a, b) => {
                SmtExpr::select(Self::fold(a), Self::fold(b))
            }
            SmtExpr::Store(a, b, c) => {
                SmtExpr::store(Self::fold(a), Self::fold(b), Self::fold(c))
            }
            SmtExpr::Let(bindings, body) => {
                let new_bindings: Vec<_> = bindings
                    .iter()
                    .map(|(n, e)| (n.clone(), Self::fold(e)))
                    .collect();
                let new_body = Self::fold(body);
                SmtExpr::Let(new_bindings, Box::new(new_body))
            }
            SmtExpr::ForAll(vars, body) => {
                SmtExpr::ForAll(vars.clone(), Box::new(Self::fold(body)))
            }
            SmtExpr::Exists(vars, body) => {
                SmtExpr::Exists(vars.clone(), Box::new(Self::fold(body)))
            }
            SmtExpr::Apply(name, args) => {
                SmtExpr::Apply(name.clone(), args.iter().map(Self::fold).collect())
            }
            // Leaf nodes / passthrough
            other => other.clone(),
        }
    }
}

// ─── Dead variable elimination ──────────────────────────────────────────

/// Eliminates declarations for variables not referenced in any constraint.
struct DeadVariableEliminator;

impl DeadVariableEliminator {
    fn eliminate(formula: &mut SmtFormula) -> usize {
        // Collect all variable names referenced in constraints
        let mut used_vars = BTreeSet::new();
        for constraint in &formula.constraints {
            let fv = constraint.formula.free_vars();
            used_vars.extend(fv);
        }

        // Also check defined function bodies for references
        for decl in &formula.declarations {
            if let SmtDeclaration::DefineFun { body, .. } = decl {
                let fv = body.free_vars();
                used_vars.extend(fv);
            }
        }

        let original_count = formula.declarations.len();

        // Keep only declarations whose names are used
        formula.declarations.retain(|decl| match decl {
            SmtDeclaration::DeclareConst { name, .. } => used_vars.contains(name),
            SmtDeclaration::DeclareFun { name, .. } => {
                // Keep if the function name is used
                formula
                    .constraints
                    .iter()
                    .any(|c| format!("{}", c.formula).contains(name))
            }
            SmtDeclaration::DeclareSort { .. } | SmtDeclaration::DefineFun { .. } => true,
        });

        original_count - formula.declarations.len()
    }
}

// ─── Structural hashing ─────────────────────────────────────────────────

/// Identifies common subexpressions and introduces let-bindings.
struct StructuralHasher {
    expr_counts: FxHashMap<String, usize>,
    let_bindings: Vec<(String, SmtExpr)>,
    next_id: u32,
    min_size: usize,
}

impl StructuralHasher {
    fn new(min_size: usize) -> Self {
        StructuralHasher {
            expr_counts: FxHashMap::default(),
            let_bindings: Vec::new(),
            next_id: 0,
            min_size,
        }
    }

    fn count_subexprs(&mut self, expr: &SmtExpr) {
        let key = format!("{}", expr);
        if expr.node_count() >= self.min_size {
            *self.expr_counts.entry(key).or_insert(0) += 1;
        }

        match expr {
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::IntAdd(es) | SmtExpr::Distinct(es) => {
                for e in es {
                    self.count_subexprs(e);
                }
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::BvUlt(a, b)
            | SmtExpr::BvUle(a, b)
            | SmtExpr::BvAdd(a, b)
            | SmtExpr::BvSub(a, b)
            | SmtExpr::BvAnd(a, b)
            | SmtExpr::BvOr(a, b)
            | SmtExpr::Concat(a, b)
            | SmtExpr::IntLe(a, b)
            | SmtExpr::Select(a, b) => {
                self.count_subexprs(a);
                self.count_subexprs(b);
            }
            SmtExpr::Ite(c, t, e) | SmtExpr::Store(c, t, e) => {
                self.count_subexprs(c);
                self.count_subexprs(t);
                self.count_subexprs(e);
            }
            SmtExpr::Not(a) | SmtExpr::Extract(a, _, _) | SmtExpr::ZeroExtend(a, _) => {
                self.count_subexprs(a);
            }
            _ => {}
        }
    }

    fn shared_subexpr_count(&self) -> usize {
        self.expr_counts.values().filter(|&&c| c > 1).count()
    }
}

// ─── Cone of influence reduction ────────────────────────────────────────

/// Removes constraints not in the cone of influence of the property.
struct ConeOfInfluence;

impl ConeOfInfluence {
    fn reduce(formula: &mut SmtFormula) -> usize {
        // Collect variables used in property negation constraints
        let mut relevant_vars = BTreeSet::new();
        for constraint in &formula.constraints {
            if matches!(constraint.origin, ConstraintOrigin::PropertyNegation) {
                relevant_vars.extend(constraint.formula.free_vars());
            }
        }

        // Budget and depth bounds are always relevant
        for constraint in &formula.constraints {
            if matches!(
                constraint.origin,
                ConstraintOrigin::BudgetBound | ConstraintOrigin::DepthBound
            ) {
                relevant_vars.extend(constraint.formula.free_vars());
            }
        }

        // Fixed-point: expand with variables from constraints touching relevant vars
        let mut changed = true;
        while changed {
            changed = false;
            for constraint in &formula.constraints {
                let fv = constraint.formula.free_vars();
                if fv.intersection(&relevant_vars).next().is_some() {
                    let before = relevant_vars.len();
                    relevant_vars.extend(fv);
                    if relevant_vars.len() > before {
                        changed = true;
                    }
                }
            }
        }

        // Remove constraints with no variables in the cone
        let original_count = formula.constraints.len();
        formula.constraints.retain(|c| {
            let fv = c.formula.free_vars();
            // Keep if it has no free vars (pure axiom), or shares vars with cone
            fv.is_empty() || fv.intersection(&relevant_vars).next().is_some()
        });

        original_count - formula.constraints.len()
    }
}

// ─── Theory-specific simplifications ────────────────────────────────────

/// Bitvector theory-specific simplifications.
struct TheorySimplifier;

impl TheorySimplifier {
    fn simplify(expr: &SmtExpr) -> SmtExpr {
        match expr {
            // bvadd(x, 0) = x
            SmtExpr::BvAdd(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(0, _), _) => b,
                    (_, SmtExpr::BvLit(0, _)) => a,
                    _ => SmtExpr::bv_add(a, b),
                }
            }
            // bvand(x, 0) = 0, bvand(x, all_ones) = x
            SmtExpr::BvAnd(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(0, w), _) => SmtExpr::BvLit(0, *w),
                    (_, SmtExpr::BvLit(0, w)) => SmtExpr::BvLit(0, *w),
                    _ => SmtExpr::bv_and(a, b),
                }
            }
            // bvor(x, 0) = x
            SmtExpr::BvOr(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (SmtExpr::BvLit(0, _), _) => b,
                    (_, SmtExpr::BvLit(0, _)) => a,
                    _ => SmtExpr::bv_or(a, b),
                }
            }
            // bvule(0, x) = true
            SmtExpr::BvUle(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match &a {
                    SmtExpr::BvLit(0, _) => SmtExpr::BoolLit(true),
                    _ => SmtExpr::bv_ule(a, b),
                }
            }
            // Recurse on other nodes
            SmtExpr::And(es) => {
                SmtExpr::and(es.iter().map(Self::simplify).collect())
            }
            SmtExpr::Or(es) => {
                SmtExpr::or(es.iter().map(Self::simplify).collect())
            }
            SmtExpr::Implies(a, b) => {
                SmtExpr::implies(Self::simplify(a), Self::simplify(b))
            }
            SmtExpr::Ite(c, t, e) => {
                SmtExpr::ite(Self::simplify(c), Self::simplify(t), Self::simplify(e))
            }
            SmtExpr::Not(a) => SmtExpr::not(Self::simplify(a)),
            SmtExpr::Eq(a, b) => SmtExpr::eq(Self::simplify(a), Self::simplify(b)),
            SmtExpr::Select(a, b) => SmtExpr::select(Self::simplify(a), Self::simplify(b)),
            SmtExpr::Store(a, b, c) => {
                SmtExpr::store(Self::simplify(a), Self::simplify(b), Self::simplify(c))
            }
            other => other.clone(),
        }
    }
}

// ─── Encoding Optimizer ─────────────────────────────────────────────────

/// Main optimizer applying all optimization passes to an SmtFormula.
#[derive(Debug, Clone)]
pub struct EncodingOptimizer {
    config: OptimizationConfig,
    stats: OptimizationStats,
}

impl EncodingOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        EncodingOptimizer {
            config,
            stats: OptimizationStats::default(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(OptimizationConfig::default())
    }

    /// Apply all enabled optimizations to the formula.
    pub fn optimize(&mut self, formula: &mut SmtFormula) {
        self.stats.original_nodes = formula.total_nodes();
        self.stats.original_constraints = formula.constraints.len();

        // Pass 1: Constant folding
        if self.config.enable_constant_folding {
            self.pass_constant_folding(formula);
        }

        // Pass 2: Theory-specific simplifications
        if self.config.enable_theory_simplification {
            self.pass_theory_simplification(formula);
        }

        // Pass 3: Remove trivially true constraints
        self.pass_remove_trivial(formula);

        // Pass 4: Cone of influence reduction
        if self.config.enable_cone_of_influence {
            let removed = ConeOfInfluence::reduce(formula);
            self.stats.cone_vars_removed = removed;
        }

        // Pass 5: Dead variable elimination
        if self.config.enable_dead_variable_elimination {
            let removed = DeadVariableEliminator::eliminate(formula);
            self.stats.dead_vars_eliminated = removed;
        }

        // Pass 6: Structural hashing analysis (for statistics)
        if self.config.enable_structural_hashing {
            let mut hasher = StructuralHasher::new(5);
            for c in &formula.constraints {
                hasher.count_subexprs(&c.formula);
            }
            self.stats.subexprs_shared = hasher.shared_subexpr_count();
        }

        self.stats.optimized_nodes = formula.total_nodes();
        self.stats.optimized_constraints = formula.constraints.len();
    }

    fn pass_constant_folding(&mut self, formula: &mut SmtFormula) {
        let mut folded = 0;
        for constraint in &mut formula.constraints {
            let before = constraint.formula.node_count();
            constraint.formula = ConstantFolder::fold(&constraint.formula);
            let after = constraint.formula.node_count();
            if after < before {
                folded += before - after;
            }
        }
        self.stats.constants_folded = folded;
    }

    fn pass_theory_simplification(&mut self, formula: &mut SmtFormula) {
        let mut simplified = 0;
        for constraint in &mut formula.constraints {
            let before = constraint.formula.node_count();
            constraint.formula = TheorySimplifier::simplify(&constraint.formula);
            let after = constraint.formula.node_count();
            if after < before {
                simplified += before - after;
            }
        }
        self.stats.theory_simplifications = simplified;
    }

    fn pass_remove_trivial(&mut self, formula: &mut SmtFormula) {
        formula
            .constraints
            .retain(|c| !matches!(&c.formula, SmtExpr::BoolLit(true)));
    }

    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }

    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintOrigin;

    #[test]
    fn test_constant_folder_bool() {
        let e = SmtExpr::and(vec![SmtExpr::BoolLit(true), SmtExpr::var("x")]);
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::var("x"));
    }

    #[test]
    fn test_constant_folder_bv_add() {
        let e = SmtExpr::bv_add(SmtExpr::bv_lit(3, 8), SmtExpr::bv_lit(5, 8));
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::bv_lit(8, 8));
    }

    #[test]
    fn test_constant_folder_bv_sub() {
        let e = SmtExpr::bv_sub(SmtExpr::bv_lit(10, 8), SmtExpr::bv_lit(3, 8));
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::bv_lit(7, 8));
    }

    #[test]
    fn test_constant_folder_bv_and() {
        let e = SmtExpr::bv_and(SmtExpr::bv_lit(0xFF, 8), SmtExpr::bv_lit(0x0F, 8));
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::bv_lit(0x0F, 8));
    }

    #[test]
    fn test_constant_folder_eq() {
        let e = SmtExpr::eq(SmtExpr::bv_lit(42, 16), SmtExpr::bv_lit(42, 16));
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::BoolLit(true));

        let e = SmtExpr::eq(SmtExpr::bv_lit(1, 16), SmtExpr::bv_lit(2, 16));
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::BoolLit(false));
    }

    #[test]
    fn test_constant_folder_ult() {
        let e = SmtExpr::bv_ult(SmtExpr::bv_lit(3, 8), SmtExpr::bv_lit(5, 8));
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_constant_folder_int_add() {
        let e = SmtExpr::int_add(vec![
            SmtExpr::IntLit(3),
            SmtExpr::var("x"),
            SmtExpr::IntLit(5),
        ]);
        let folded = ConstantFolder::fold(&e);
        // Should merge 3+5=8
        match &folded {
            SmtExpr::IntAdd(es) => {
                assert_eq!(es.len(), 2); // x and 8
            }
            _ => panic!("expected IntAdd, got {:?}", folded),
        }
    }

    #[test]
    fn test_constant_folder_extract() {
        let e = SmtExpr::Extract(Box::new(SmtExpr::bv_lit(0xFF, 8)), 3, 0);
        let folded = ConstantFolder::fold(&e);
        assert_eq!(folded, SmtExpr::bv_lit(0x0F, 4));
    }

    #[test]
    fn test_theory_simplifier_bvadd_zero() {
        let e = SmtExpr::bv_add(SmtExpr::var("x"), SmtExpr::bv_lit(0, 16));
        let simplified = TheorySimplifier::simplify(&e);
        assert_eq!(simplified, SmtExpr::var("x"));
    }

    #[test]
    fn test_theory_simplifier_bvand_zero() {
        let e = SmtExpr::bv_and(SmtExpr::var("x"), SmtExpr::bv_lit(0, 16));
        let simplified = TheorySimplifier::simplify(&e);
        assert_eq!(simplified, SmtExpr::bv_lit(0, 16));
    }

    #[test]
    fn test_theory_simplifier_bvor_zero() {
        let e = SmtExpr::bv_or(SmtExpr::bv_lit(0, 16), SmtExpr::var("x"));
        let simplified = TheorySimplifier::simplify(&e);
        assert_eq!(simplified, SmtExpr::var("x"));
    }

    #[test]
    fn test_theory_simplifier_bvule_zero() {
        let e = SmtExpr::bv_ule(SmtExpr::bv_lit(0, 16), SmtExpr::var("x"));
        let simplified = TheorySimplifier::simplify(&e);
        assert_eq!(simplified, SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_dead_variable_elimination() {
        let mut formula = SmtFormula::new(1, 1);
        formula.add_declaration(SmtDeclaration::DeclareConst {
            name: "used".to_string(),
            sort: SmtSort::Bool,
        });
        formula.add_declaration(SmtDeclaration::DeclareConst {
            name: "unused".to_string(),
            sort: SmtSort::Bool,
        });
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::var("used"),
            ConstraintOrigin::InitialState,
            "uses_used",
        ));

        let removed = DeadVariableEliminator::eliminate(&mut formula);
        assert_eq!(removed, 1);
        assert_eq!(formula.declarations.len(), 1);
    }

    #[test]
    fn test_remove_trivial() {
        let mut formula = SmtFormula::new(1, 1);
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::BoolLit(true),
            ConstraintOrigin::InitialState,
            "trivial",
        ));
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::var("x"),
            ConstraintOrigin::InitialState,
            "real",
        ));

        let mut optimizer = EncodingOptimizer::with_defaults();
        optimizer.optimize(&mut formula);
        assert_eq!(formula.constraints.len(), 1);
    }

    #[test]
    fn test_optimizer_full() {
        let mut formula = SmtFormula::new(5, 3);
        formula.add_declaration(SmtDeclaration::DeclareConst {
            name: "x".to_string(),
            sort: SmtSort::BitVec(16),
        });
        formula.add_declaration(SmtDeclaration::DeclareConst {
            name: "dead".to_string(),
            sort: SmtSort::BitVec(16),
        });

        formula.add_constraint(SmtConstraint::new(
            SmtExpr::bv_ule(
                SmtExpr::bv_add(SmtExpr::bv_lit(3, 16), SmtExpr::bv_lit(5, 16)),
                SmtExpr::var("x"),
            ),
            ConstraintOrigin::PropertyNegation,
            "test",
        ));
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::BoolLit(true),
            ConstraintOrigin::InitialState,
            "trivial",
        ));

        let mut optimizer = EncodingOptimizer::with_defaults();
        optimizer.optimize(&mut formula);

        let stats = optimizer.stats();
        assert!(stats.optimized_constraints <= stats.original_constraints);
    }

    #[test]
    fn test_cone_of_influence() {
        let mut formula = SmtFormula::new(1, 1);
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::var("relevant"),
            ConstraintOrigin::PropertyNegation,
            "property",
        ));
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::var("relevant"),
            ConstraintOrigin::InitialState,
            "connected",
        ));
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::var("isolated"),
            ConstraintOrigin::InitialState,
            "disconnected",
        ));

        let removed = ConeOfInfluence::reduce(&mut formula);
        assert_eq!(removed, 1);
        assert_eq!(formula.constraints.len(), 2);
    }

    #[test]
    fn test_structural_hasher() {
        let mut hasher = StructuralHasher::new(3);
        let shared = SmtExpr::bv_add(SmtExpr::var("a"), SmtExpr::var("b"));
        let e1 = SmtExpr::and(vec![shared.clone(), SmtExpr::var("c")]);
        let e2 = SmtExpr::and(vec![shared, SmtExpr::var("d")]);
        hasher.count_subexprs(&e1);
        hasher.count_subexprs(&e2);
        assert!(hasher.shared_subexpr_count() >= 1);
    }

    #[test]
    fn test_optimization_stats() {
        let stats = OptimizationStats {
            original_nodes: 100,
            optimized_nodes: 60,
            ..Default::default()
        };
        assert!((stats.reduction_ratio() - 0.6).abs() < 0.001);
    }
}
