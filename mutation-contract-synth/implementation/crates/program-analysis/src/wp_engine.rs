//! Weakest Precondition computation engine.
//!
//! Implements WP calculus for the loop-free imperative language:
//! - wp(x := e, Q) = Q[e/x]
//! - wp(S1;S2, Q) = wp(S1, wp(S2, Q))
//! - wp(if b then S1 else S2, Q) = (b => wp(S1,Q)) /\ (!b => wp(S2,Q))
//! - wp(assert b, Q) = b /\ Q
//! - wp(return e, Q) = Q[e/ret]
//!
//! Also supports batch WP computation for mutant analysis and formula simplification.

use std::collections::HashMap;

use shared_types::{
    AnalysisError, AnalysisResult, ArithOp, BasicBlock, BlockId, Formula, IrExpr, IrFunction,
    IrStatement, LogicOp, RelOp, Terminator, UnaryOp, VarId,
};

// ---------------------------------------------------------------------------
// WP configuration and results
// ---------------------------------------------------------------------------

/// Configuration for the WP engine.
#[derive(Debug, Clone)]
pub struct WpConfig {
    /// Maximum formula size before simplification is triggered.
    pub simplify_threshold: usize,
    /// Whether to use array theory (uninterpreted functions).
    pub use_array_theory: bool,
    /// Whether to cache intermediate WP results.
    pub enable_caching: bool,
    /// Maximum depth for recursive WP computation.
    pub max_depth: usize,
}

impl Default for WpConfig {
    fn default() -> Self {
        WpConfig {
            simplify_threshold: 1000,
            use_array_theory: true,
            enable_caching: true,
            max_depth: 100,
        }
    }
}

/// Result of a WP computation.
#[derive(Debug, Clone)]
pub struct WpResult {
    /// The computed weakest precondition formula.
    pub formula: Formula,
    /// Statistics about the computation.
    pub stats: WpStats,
}

/// Statistics for WP computation.
#[derive(Debug, Clone, Default)]
pub struct WpStats {
    pub substitutions: usize,
    pub simplifications: usize,
    pub cache_hits: usize,
    pub formula_size: usize,
    pub max_depth_reached: usize,
}

// ---------------------------------------------------------------------------
// WpEngine
// ---------------------------------------------------------------------------

/// Weakest precondition computation engine.
pub struct WpEngine {
    config: WpConfig,
    cache: HashMap<(BlockId, String), Formula>,
    stats: WpStats,
}

impl WpEngine {
    pub fn new(config: WpConfig) -> Self {
        WpEngine {
            config,
            cache: HashMap::new(),
            stats: WpStats::default(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(WpConfig::default())
    }

    /// Compute the weakest precondition of a function with respect to a postcondition.
    pub fn compute_wp(
        &mut self,
        func: &IrFunction,
        postcondition: &Formula,
    ) -> AnalysisResult<WpResult> {
        self.cache.clear();
        self.stats = WpStats::default();

        let formula = self.wp_function(func, postcondition, 0)?;
        let simplified = self.simplify_formula(&formula);

        self.stats.formula_size = simplified.size();
        Ok(WpResult {
            formula: simplified,
            stats: self.stats.clone(),
        })
    }

    /// Compute WP for a function: process blocks from entry following the CFG.
    fn wp_function(
        &mut self,
        func: &IrFunction,
        post: &Formula,
        depth: usize,
    ) -> AnalysisResult<Formula> {
        if depth > self.config.max_depth {
            return Err(AnalysisError::WpError {
                message: "Maximum WP computation depth exceeded".into(),
            });
        }
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);

        self.wp_block(func, func.entry_block, post, depth)
    }

    /// Compute WP for a single basic block.
    fn wp_block(
        &mut self,
        func: &IrFunction,
        block_id: BlockId,
        post: &Formula,
        depth: usize,
    ) -> AnalysisResult<Formula> {
        // Check cache
        let cache_key = (block_id, format!("{}", post));
        if self.config.enable_caching {
            if let Some(cached) = self.cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        let block = func.block(block_id).ok_or_else(|| AnalysisError::WpError {
            message: format!("Block {} not found", block_id),
        })?;

        // First compute WP for the terminator
        let term_wp = self.wp_terminator(func, &block.terminator, post, depth)?;

        // Then compute WP backwards through statements
        let mut current = term_wp;
        for stmt in block.stmts.iter().rev() {
            current = self.wp_statement(stmt, &current)?;
            if current.size() > self.config.simplify_threshold {
                current = self.simplify_formula(&current);
            }
        }

        // Cache result
        if self.config.enable_caching {
            self.cache.insert(cache_key, current.clone());
        }

        Ok(current)
    }

    /// Compute WP for a terminator.
    fn wp_terminator(
        &mut self,
        func: &IrFunction,
        term: &Terminator,
        post: &Formula,
        depth: usize,
    ) -> AnalysisResult<Formula> {
        match term {
            Terminator::Goto(target) => self.wp_block(func, *target, post, depth + 1),
            Terminator::Branch {
                condition,
                true_target,
                false_target,
            } => {
                let cond_formula = self.ir_expr_to_formula(condition);
                let true_wp = self.wp_block(func, *true_target, post, depth + 1)?;
                let false_wp = self.wp_block(func, *false_target, post, depth + 1)?;

                // wp(if b then S1 else S2, Q) = (b => wp(S1,Q)) /\ (!b => wp(S2,Q))
                let true_branch = Formula::implies(cond_formula.clone(), true_wp);
                let false_branch = Formula::implies(Formula::not(cond_formula), false_wp);
                Ok(Formula::and(true_branch, false_branch))
            }
            Terminator::Return(Some(expr)) => {
                // wp(return e, Q) = Q[e/ret]
                let ret_formula = self.ir_expr_to_formula(expr);
                let result = post.substitute("__ret", &ret_formula);
                self.stats.substitutions += 1;
                Ok(result)
            }
            Terminator::Return(None) => Ok(post.clone()),
            Terminator::Unreachable => Ok(Formula::True),
        }
    }

    /// Compute WP for a single statement.
    pub fn wp_statement(&mut self, stmt: &IrStatement, post: &Formula) -> AnalysisResult<Formula> {
        match stmt {
            IrStatement::Assign { target, value } => {
                // wp(x := e, Q) = Q[e/x]
                let val_formula = self.ir_expr_to_formula(value);
                let result = post.substitute(target, &val_formula);
                self.stats.substitutions += 1;
                Ok(result)
            }
            IrStatement::ArrayWrite {
                array,
                index,
                value,
            } => {
                if self.config.use_array_theory {
                    // Use array theory: store(a, i, v)
                    let idx_formula = self.ir_expr_to_formula(index);
                    let val_formula = self.ir_expr_to_formula(value);
                    let new_array = format!("{}'", array);
                    let store = Formula::ArrayWrite {
                        base_array: array.clone(),
                        index: Box::new(idx_formula),
                        value: Box::new(val_formula),
                        result_array: new_array.clone(),
                    };
                    let result = post.substitute(array, &Formula::IntVar(new_array));
                    self.stats.substitutions += 1;
                    Ok(Formula::and(store, result))
                } else {
                    Ok(post.clone())
                }
            }
            IrStatement::Assert { condition } => {
                // wp(assert b, Q) = b /\ Q
                let cond_formula = self.ir_expr_to_formula(condition);
                Ok(Formula::and(cond_formula, post.clone()))
            }
        }
    }

    /// Compute WP for a sequence of statements.
    pub fn wp_sequence(
        &mut self,
        stmts: &[IrStatement],
        post: &Formula,
    ) -> AnalysisResult<Formula> {
        let mut current = post.clone();
        for stmt in stmts.iter().rev() {
            current = self.wp_statement(stmt, &current)?;
        }
        Ok(current)
    }

    // -- IR expression to Formula conversion --------------------------------

    pub fn ir_expr_to_formula(&self, expr: &IrExpr) -> Formula {
        match expr {
            IrExpr::IntConst(v) => Formula::IntConst(*v),
            IrExpr::BoolConst(true) => Formula::True,
            IrExpr::BoolConst(false) => Formula::False,
            IrExpr::Var(name) => Formula::IntVar(name.clone()),
            IrExpr::BinaryArith { left, op, right } => {
                let l = self.ir_expr_to_formula(left);
                let r = self.ir_expr_to_formula(right);
                match op {
                    ArithOp::Add => Formula::add(l, r),
                    ArithOp::Sub => Formula::sub(l, r),
                    ArithOp::Mul => Formula::mul(l, r),
                    ArithOp::Div => Formula::Div(Box::new(l), Box::new(r)),
                    ArithOp::Mod => Formula::Mod(Box::new(l), Box::new(r)),
                }
            }
            IrExpr::BinaryRel { left, op, right } => {
                let l = self.ir_expr_to_formula(left);
                let r = self.ir_expr_to_formula(right);
                match op {
                    RelOp::Eq => Formula::eq(l, r),
                    RelOp::Ne => Formula::ne(l, r),
                    RelOp::Lt => Formula::Lt(Box::new(l), Box::new(r)),
                    RelOp::Le => Formula::Le(Box::new(l), Box::new(r)),
                    RelOp::Gt => Formula::Gt(Box::new(l), Box::new(r)),
                    RelOp::Ge => Formula::Ge(Box::new(l), Box::new(r)),
                }
            }
            IrExpr::BinaryLogic { left, op, right } => {
                let l = self.ir_expr_to_formula(left);
                let r = self.ir_expr_to_formula(right);
                match op {
                    LogicOp::And => Formula::and(l, r),
                    LogicOp::Or => Formula::or(l, r),
                    LogicOp::Implies => Formula::implies(l, r),
                }
            }
            IrExpr::Unary { op, operand } => {
                let inner = self.ir_expr_to_formula(operand);
                match op {
                    UnaryOp::Neg => Formula::Neg(Box::new(inner)),
                    UnaryOp::Not => Formula::not(inner),
                    UnaryOp::BitwiseNot => Formula::Not(Box::new(inner)),
                }
            }
            IrExpr::ArrayRead { array, index } => {
                let idx = self.ir_expr_to_formula(index);
                Formula::ArrayRead {
                    array: array.clone(),
                    index: Box::new(idx),
                }
            }
            IrExpr::FunctionCall { name, args } => {
                // Uninterpreted function: just use the name as a variable
                Formula::IntVar(format!(
                    "{}({})",
                    name,
                    args.iter()
                        .map(|a| format!("{}", self.ir_expr_to_formula(a)))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
        }
    }

    // -- Formula simplification ---------------------------------------------

    /// Simplify a formula by applying algebraic identities.
    pub fn simplify_formula(&mut self, formula: &Formula) -> Formula {
        self.stats.simplifications += 1;
        self.simplify_rec(formula, 0)
    }

    fn simplify_rec(&self, formula: &Formula, depth: usize) -> Formula {
        if depth > 50 {
            return formula.clone();
        }
        match formula {
            Formula::True
            | Formula::False
            | Formula::IntConst(_)
            | Formula::BoolVar(_)
            | Formula::IntVar(_) => formula.clone(),

            Formula::And(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::and(sa, sb)
            }
            Formula::Or(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::or(sa, sb)
            }
            Formula::Not(a) => {
                let sa = self.simplify_rec(a, depth + 1);
                Formula::not(sa)
            }
            Formula::Implies(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::implies(sa, sb)
            }
            Formula::Eq(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::eq(sa, sb)
            }
            Formula::Ne(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::ne(sa, sb)
            }
            Formula::Add(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::add(sa, sb)
            }
            Formula::Sub(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::sub(sa, sb)
            }
            Formula::Mul(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                Formula::mul(sa, sb)
            }
            Formula::Neg(a) => {
                let sa = self.simplify_rec(a, depth + 1);
                match sa {
                    Formula::IntConst(v) => Formula::IntConst(-v),
                    Formula::Neg(inner) => *inner,
                    other => Formula::Neg(Box::new(other)),
                }
            }
            Formula::Ite(c, t, e) => {
                let sc = self.simplify_rec(c, depth + 1);
                let st = self.simplify_rec(t, depth + 1);
                let se = self.simplify_rec(e, depth + 1);
                Formula::ite(sc, st, se)
            }
            Formula::Lt(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                match (&sa, &sb) {
                    (Formula::IntConst(x), Formula::IntConst(y)) => {
                        if x < y {
                            Formula::True
                        } else {
                            Formula::False
                        }
                    }
                    _ => Formula::Lt(Box::new(sa), Box::new(sb)),
                }
            }
            Formula::Le(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                match (&sa, &sb) {
                    (Formula::IntConst(x), Formula::IntConst(y)) => {
                        if x <= y {
                            Formula::True
                        } else {
                            Formula::False
                        }
                    }
                    _ => Formula::Le(Box::new(sa), Box::new(sb)),
                }
            }
            Formula::Gt(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                match (&sa, &sb) {
                    (Formula::IntConst(x), Formula::IntConst(y)) => {
                        if x > y {
                            Formula::True
                        } else {
                            Formula::False
                        }
                    }
                    _ => Formula::Gt(Box::new(sa), Box::new(sb)),
                }
            }
            Formula::Ge(a, b) => {
                let sa = self.simplify_rec(a, depth + 1);
                let sb = self.simplify_rec(b, depth + 1);
                match (&sa, &sb) {
                    (Formula::IntConst(x), Formula::IntConst(y)) => {
                        if x >= y {
                            Formula::True
                        } else {
                            Formula::False
                        }
                    }
                    _ => Formula::Ge(Box::new(sa), Box::new(sb)),
                }
            }
            _ => formula.clone(),
        }
    }

    // -- Batch WP computation -----------------------------------------------

    /// Compute WP for a function and all its mutants sharing a common prefix.
    /// Returns the original WP and a map of mutant WPs.
    pub fn batch_wp(
        &mut self,
        original: &IrFunction,
        mutants: &[(String, IrFunction)],
        postcondition: &Formula,
    ) -> AnalysisResult<BatchWpResult> {
        let original_wp = self.compute_wp(original, postcondition)?;

        let mut mutant_wps = HashMap::new();
        for (name, mutant_func) in mutants {
            let wp = self.compute_wp(mutant_func, postcondition)?;
            mutant_wps.insert(name.clone(), wp);
        }

        Ok(BatchWpResult {
            original: original_wp,
            mutants: mutant_wps,
        })
    }

    /// Compute the divergence formula: inputs where original and mutant differ.
    /// divergence = wp(original, true) /\ !wp(mutant, true)
    pub fn wp_difference(
        &mut self,
        original: &IrFunction,
        mutant: &IrFunction,
    ) -> AnalysisResult<Formula> {
        let post = Formula::True;
        let orig_wp = self.compute_wp(original, &post)?.formula;
        let mut_wp = self.compute_wp(mutant, &post)?.formula;

        let diff = Formula::and(orig_wp, Formula::not(mut_wp));
        Ok(self.simplify_formula(&diff))
    }
}

/// Result of batch WP computation.
#[derive(Debug, Clone)]
pub struct BatchWpResult {
    pub original: WpResult,
    pub mutants: HashMap<String, WpResult>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_lowering::IrLowering;
    use crate::parser::Parser;

    fn get_ir_fn(src: &str) -> IrFunction {
        let prog = Parser::parse_source(src).unwrap();
        let ir = IrLowering::new().lower_program(&prog).unwrap();
        ir.functions.into_iter().next().unwrap()
    }

    fn compute_wp(src: &str, post: &Formula) -> WpResult {
        let func = get_ir_fn(src);
        let mut engine = WpEngine::with_defaults();
        engine.compute_wp(&func, post).unwrap()
    }

    #[test]
    fn test_wp_return_const() {
        let result = compute_wp(
            "fn f() -> int { return 42; }",
            &Formula::Eq(
                Box::new(Formula::IntVar("__ret".into())),
                Box::new(Formula::IntConst(0)),
            ),
        );
        // wp should substitute 42 for __ret: (42 == 0) => False
        assert_ne!(result.formula, Formula::True);
    }

    #[test]
    fn test_wp_assign_return() {
        let result = compute_wp(
            "fn f(x: int) -> int { let y: int = x + 1; return y; }",
            &Formula::Gt(
                Box::new(Formula::IntVar("__ret".into())),
                Box::new(Formula::IntConst(0)),
            ),
        );
        // wp should be: x + 1 > 0
        assert_ne!(result.formula, Formula::True);
        assert!(result.stats.substitutions > 0);
    }

    #[test]
    fn test_wp_assert() {
        let func = get_ir_fn("fn f(x: int) -> void { assert(x > 0); }");
        let mut engine = WpEngine::with_defaults();
        let result = engine.compute_wp(&func, &Formula::True).unwrap();
        // WP of assert(x > 0) with postcondition True is: x > 0
        assert_ne!(result.formula, Formula::True);
    }

    #[test]
    fn test_wp_if_else() {
        let result = compute_wp(
            "fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }",
            &Formula::Ge(
                Box::new(Formula::IntVar("__ret".into())),
                Box::new(Formula::IntConst(0)),
            ),
        );
        // Should compute a formula involving both branches
        assert!(result.formula.size() > 1);
    }

    #[test]
    fn test_wp_sequence() {
        let stmts = vec![
            IrStatement::Assign {
                target: "x".into(),
                value: IrExpr::IntConst(5),
            },
            IrStatement::Assign {
                target: "y".into(),
                value: IrExpr::BinaryArith {
                    left: Box::new(IrExpr::Var("x".into())),
                    op: ArithOp::Add,
                    right: Box::new(IrExpr::IntConst(1)),
                },
            },
        ];
        let post = Formula::Eq(
            Box::new(Formula::IntVar("y".into())),
            Box::new(Formula::IntConst(6)),
        );
        let mut engine = WpEngine::with_defaults();
        let result = engine.wp_sequence(&stmts, &post).unwrap();
        // After substitution: (5 + 1 == 6) => True
        let simplified = engine.simplify_formula(&result);
        assert_eq!(simplified, Formula::True);
    }

    #[test]
    fn test_wp_simplification() {
        let mut engine = WpEngine::with_defaults();
        let f = Formula::and(
            Formula::True,
            Formula::Gt(
                Box::new(Formula::IntVar("x".into())),
                Box::new(Formula::IntConst(0)),
            ),
        );
        let s = engine.simplify_formula(&f);
        assert!(matches!(s, Formula::Gt(..)));
    }

    #[test]
    fn test_wp_simplify_double_neg() {
        let mut engine = WpEngine::with_defaults();
        let f = Formula::Not(Box::new(Formula::Not(Box::new(Formula::IntVar(
            "x".into(),
        )))));
        let s = engine.simplify_formula(&f);
        assert_eq!(s, Formula::IntVar("x".into()));
    }

    #[test]
    fn test_wp_simplify_const_arith() {
        let mut engine = WpEngine::with_defaults();
        let f = Formula::Add(
            Box::new(Formula::IntConst(3)),
            Box::new(Formula::IntConst(4)),
        );
        let s = engine.simplify_formula(&f);
        assert_eq!(s, Formula::IntConst(7));
    }

    #[test]
    fn test_wp_simplify_const_comparison() {
        let mut engine = WpEngine::with_defaults();
        let f = Formula::Lt(
            Box::new(Formula::IntConst(3)),
            Box::new(Formula::IntConst(5)),
        );
        let s = engine.simplify_formula(&f);
        assert_eq!(s, Formula::True);
    }

    #[test]
    fn test_wp_caching() {
        let func =
            get_ir_fn("fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }");
        let mut engine = WpEngine::with_defaults();
        let post = Formula::Ge(
            Box::new(Formula::IntVar("__ret".into())),
            Box::new(Formula::IntConst(0)),
        );

        engine.compute_wp(&func, &post).unwrap();
        let first_stats = engine.stats.clone();

        engine.cache.clear();
        engine.stats = WpStats::default();
        engine.compute_wp(&func, &post).unwrap();
        let _ = first_stats;
    }

    #[test]
    fn test_wp_ir_to_formula() {
        let engine = WpEngine::with_defaults();
        let expr = IrExpr::BinaryArith {
            left: Box::new(IrExpr::Var("x".into())),
            op: ArithOp::Add,
            right: Box::new(IrExpr::IntConst(1)),
        };
        let f = engine.ir_expr_to_formula(&expr);
        assert!(matches!(f, Formula::Add(..)));
    }

    #[test]
    fn test_wp_array_read() {
        let engine = WpEngine::with_defaults();
        let expr = IrExpr::ArrayRead {
            array: "a".into(),
            index: Box::new(IrExpr::IntConst(0)),
        };
        let f = engine.ir_expr_to_formula(&expr);
        assert!(matches!(f, Formula::ArrayRead { .. }));
    }

    #[test]
    fn test_wp_complex_function() {
        let result = compute_wp(
            r#"
            fn clamp(x: int, lo: int, hi: int) -> int {
                if (x < lo) { return lo; }
                else if (x > hi) { return hi; }
                else { return x; }
            }
        "#,
            &Formula::And(
                Box::new(Formula::Ge(
                    Box::new(Formula::IntVar("__ret".into())),
                    Box::new(Formula::IntVar("lo".into())),
                )),
                Box::new(Formula::Le(
                    Box::new(Formula::IntVar("__ret".into())),
                    Box::new(Formula::IntVar("hi".into())),
                )),
            ),
        );
        assert!(result.formula.size() > 1);
    }

    #[test]
    fn test_wp_stats() {
        let result = compute_wp(
            "fn f(x: int) -> int { let y: int = x + 1; return y; }",
            &Formula::Gt(
                Box::new(Formula::IntVar("__ret".into())),
                Box::new(Formula::IntConst(0)),
            ),
        );
        assert!(result.stats.substitutions > 0);
    }

    #[test]
    fn test_wp_difference() {
        let orig = get_ir_fn("fn f(x: int) -> int { return x + 1; }");
        let mutant = get_ir_fn("fn f(x: int) -> int { return x - 1; }");
        let mut engine = WpEngine::with_defaults();
        let diff = engine.wp_difference(&orig, &mutant).unwrap();
        assert_ne!(diff, Formula::False);
    }

    #[test]
    fn test_wp_batch() {
        let orig = get_ir_fn("fn f(x: int) -> int { return x + 1; }");
        let m1 = get_ir_fn("fn f(x: int) -> int { return x - 1; }");
        let m2 = get_ir_fn("fn f(x: int) -> int { return x * 1; }");
        let mut engine = WpEngine::with_defaults();
        let post = Formula::Gt(
            Box::new(Formula::IntVar("__ret".into())),
            Box::new(Formula::IntConst(0)),
        );
        let result = engine
            .batch_wp(&orig, &[("m1".into(), m1), ("m2".into(), m2)], &post)
            .unwrap();
        assert_eq!(result.mutants.len(), 2);
    }

    #[test]
    fn test_wp_return_void() {
        let result = compute_wp("fn f() -> void { return; }", &Formula::True);
        assert_eq!(result.formula, Formula::True);
    }

    #[test]
    fn test_wp_nested_if() {
        let result = compute_wp(
            r#"
            fn f(x: int, y: int) -> int {
                if (x > 0) {
                    if (y > 0) { return x + y; }
                    else { return x; }
                } else {
                    return 0;
                }
            }
        "#,
            &Formula::Ge(
                Box::new(Formula::IntVar("__ret".into())),
                Box::new(Formula::IntConst(0)),
            ),
        );
        assert!(result.formula.size() > 1);
    }
}
