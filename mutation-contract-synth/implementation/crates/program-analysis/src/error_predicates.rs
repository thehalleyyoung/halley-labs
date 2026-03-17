//! Error predicate extraction from mutations.
//!
//! Computes E(m)(x) = (m(x) != f(x)) symbolically for various mutation operators.
//! For AOR: compute difference formula based on operator change.
//! For ROR: compute boundary predicate.
//! For LCR: compute XOR-like predicate.
//! For UOI: compute non-zero predicate.

use std::collections::HashMap;

use shared_types::{
    AnalysisError, AnalysisResult, ArithOp, Formula, IrExpr, IrFunction, LogicOp, MutantId,
    Mutation, MutationOperator, RelOp, UnaryOp,
};

use crate::wp_engine::WpEngine;

// ---------------------------------------------------------------------------
// ErrorPredicate
// ---------------------------------------------------------------------------

/// An error predicate: a formula characterizing when a mutant diverges from the original.
#[derive(Debug, Clone)]
pub struct ErrorPredicate {
    pub mutant_id: MutantId,
    pub formula: Formula,
    pub operator: MutationOperator,
    pub description: String,
}

impl ErrorPredicate {
    pub fn new(mutant_id: MutantId, formula: Formula, operator: MutationOperator) -> Self {
        let desc = format!("E({}) for {}", mutant_id, operator);
        ErrorPredicate {
            mutant_id,
            formula,
            operator,
            description: desc,
        }
    }

    /// Check if this predicate is trivially true (always diverges).
    pub fn is_trivial_true(&self) -> bool {
        matches!(self.formula, Formula::True)
    }

    /// Check if this predicate is trivially false (never diverges).
    pub fn is_trivial_false(&self) -> bool {
        matches!(self.formula, Formula::False)
    }

    /// Return the size (complexity) of this predicate.
    pub fn complexity(&self) -> usize {
        self.formula.size()
    }
}

// ---------------------------------------------------------------------------
// ErrorPredicateSet
// ---------------------------------------------------------------------------

/// A collection of error predicates for a function.
#[derive(Debug, Clone)]
pub struct ErrorPredicateSet {
    predicates: Vec<ErrorPredicate>,
    by_mutant: HashMap<String, usize>,
}

impl ErrorPredicateSet {
    pub fn new() -> Self {
        ErrorPredicateSet {
            predicates: Vec::new(),
            by_mutant: HashMap::new(),
        }
    }

    pub fn add(&mut self, pred: ErrorPredicate) {
        let idx = self.predicates.len();
        self.by_mutant.insert(pred.mutant_id.0.clone(), idx);
        self.predicates.push(pred);
    }

    pub fn get(&self, mutant_id: &str) -> Option<&ErrorPredicate> {
        self.by_mutant.get(mutant_id).map(|&i| &self.predicates[i])
    }

    pub fn all(&self) -> &[ErrorPredicate] {
        &self.predicates
    }

    pub fn len(&self) -> usize {
        self.predicates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Filter out trivially false predicates.
    pub fn non_trivial(&self) -> Vec<&ErrorPredicate> {
        self.predicates
            .iter()
            .filter(|p| !p.is_trivial_false())
            .collect()
    }

    /// Return predicates sorted by complexity (simplest first).
    pub fn sorted_by_complexity(&self) -> Vec<&ErrorPredicate> {
        let mut sorted: Vec<_> = self.predicates.iter().collect();
        sorted.sort_by_key(|p| p.complexity());
        sorted
    }
}

impl Default for ErrorPredicateSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ErrorPredicateExtractor
// ---------------------------------------------------------------------------

/// Extracts error predicates from mutations.
pub struct ErrorPredicateExtractor {
    wp_engine: WpEngine,
    cache: HashMap<String, Formula>,
}

impl ErrorPredicateExtractor {
    pub fn new() -> Self {
        ErrorPredicateExtractor {
            wp_engine: WpEngine::with_defaults(),
            cache: HashMap::new(),
        }
    }

    /// Extract error predicate for a single mutation.
    pub fn extract(
        &mut self,
        original: &IrFunction,
        mutant: &IrFunction,
        mutation: &Mutation,
    ) -> AnalysisResult<ErrorPredicate> {
        let formula = match &mutation.operator {
            MutationOperator::AOR {
                original: orig_op,
                replacement: rep_op,
            } => self.extract_aor(original, mutant, *orig_op, *rep_op)?,
            MutationOperator::ROR {
                original: orig_op,
                replacement: rep_op,
            } => self.extract_ror(original, mutant, *orig_op, *rep_op)?,
            MutationOperator::LCR {
                original: orig_op,
                replacement: rep_op,
            } => self.extract_lcr(original, mutant, *orig_op, *rep_op)?,
            MutationOperator::UOI { op } => self.extract_uoi(original, mutant, *op)?,
            MutationOperator::SDL => self.extract_sdl(original, mutant)?,
        };

        let simplified = self.wp_engine.simplify_formula(&formula);
        Ok(ErrorPredicate::new(
            mutation.id.clone(),
            simplified,
            mutation.operator.clone(),
        ))
    }

    /// Batch extract error predicates for all mutations of a function.
    pub fn extract_batch(
        &mut self,
        original: &IrFunction,
        mutants: &[(Mutation, IrFunction)],
    ) -> AnalysisResult<ErrorPredicateSet> {
        let mut set = ErrorPredicateSet::new();
        for (mutation, mutant_func) in mutants {
            let pred = self.extract(original, mutant_func, mutation)?;
            set.add(pred);
        }
        Ok(set)
    }

    // -- AOR extraction -----------------------------------------------------

    /// For AOR (Arithmetic Operator Replacement):
    /// E = original_result != mutant_result
    /// e.g., (x + y) != (x - y) => 2*y != 0 => y != 0
    fn extract_aor(
        &mut self,
        original: &IrFunction,
        mutant: &IrFunction,
        orig_op: ArithOp,
        rep_op: ArithOp,
    ) -> AnalysisResult<Formula> {
        // Use WP difference computation
        let diff = self.wp_engine.wp_difference(original, mutant)?;
        if diff != Formula::False {
            return Ok(diff);
        }

        // Fallback: construct difference formula directly
        // For op1 -> op2: the divergence is when op1(a,b) != op2(a,b)
        let a = Formula::IntVar("__lhs".into());
        let b = Formula::IntVar("__rhs".into());
        let orig_result = self.apply_arith_op(orig_op, a.clone(), b.clone());
        let mut_result = self.apply_arith_op(rep_op, a, b);
        Ok(Formula::ne(orig_result, mut_result))
    }

    fn apply_arith_op(&self, op: ArithOp, a: Formula, b: Formula) -> Formula {
        match op {
            ArithOp::Add => Formula::Add(Box::new(a), Box::new(b)),
            ArithOp::Sub => Formula::Sub(Box::new(a), Box::new(b)),
            ArithOp::Mul => Formula::Mul(Box::new(a), Box::new(b)),
            ArithOp::Div => Formula::Div(Box::new(a), Box::new(b)),
            ArithOp::Mod => Formula::Mod(Box::new(a), Box::new(b)),
        }
    }

    // -- ROR extraction -----------------------------------------------------

    /// For ROR (Relational Operator Replacement):
    /// E = original_cond XOR mutant_cond (they differ on the boundary)
    fn extract_ror(
        &mut self,
        original: &IrFunction,
        mutant: &IrFunction,
        orig_op: RelOp,
        rep_op: RelOp,
    ) -> AnalysisResult<Formula> {
        let diff = self.wp_engine.wp_difference(original, mutant)?;
        if diff != Formula::False {
            return Ok(diff);
        }

        // Construct boundary predicate directly
        let a = Formula::IntVar("__lhs".into());
        let b = Formula::IntVar("__rhs".into());
        let orig_cond = self.apply_rel_op(orig_op, a.clone(), b.clone());
        let mut_cond = self.apply_rel_op(rep_op, a, b);
        // XOR: (orig && !mut) || (!orig && mut)
        Ok(Formula::or(
            Formula::and(orig_cond.clone(), Formula::not(mut_cond.clone())),
            Formula::and(Formula::not(orig_cond), mut_cond),
        ))
    }

    fn apply_rel_op(&self, op: RelOp, a: Formula, b: Formula) -> Formula {
        match op {
            RelOp::Eq => Formula::Eq(Box::new(a), Box::new(b)),
            RelOp::Ne => Formula::Ne(Box::new(a), Box::new(b)),
            RelOp::Lt => Formula::Lt(Box::new(a), Box::new(b)),
            RelOp::Le => Formula::Le(Box::new(a), Box::new(b)),
            RelOp::Gt => Formula::Gt(Box::new(a), Box::new(b)),
            RelOp::Ge => Formula::Ge(Box::new(a), Box::new(b)),
        }
    }

    // -- LCR extraction -----------------------------------------------------

    /// For LCR (Logical Connector Replacement):
    /// E = original_logic XOR mutant_logic
    fn extract_lcr(
        &mut self,
        original: &IrFunction,
        mutant: &IrFunction,
        orig_op: LogicOp,
        rep_op: LogicOp,
    ) -> AnalysisResult<Formula> {
        let diff = self.wp_engine.wp_difference(original, mutant)?;
        if diff != Formula::False {
            return Ok(diff);
        }

        let a = Formula::BoolVar("__lhs".into());
        let b = Formula::BoolVar("__rhs".into());
        let orig_result = self.apply_logic_op(orig_op, a.clone(), b.clone());
        let mut_result = self.apply_logic_op(rep_op, a, b);
        // XOR
        Ok(Formula::or(
            Formula::and(orig_result.clone(), Formula::not(mut_result.clone())),
            Formula::and(Formula::not(orig_result), mut_result),
        ))
    }

    fn apply_logic_op(&self, op: LogicOp, a: Formula, b: Formula) -> Formula {
        match op {
            LogicOp::And => Formula::And(Box::new(a), Box::new(b)),
            LogicOp::Or => Formula::Or(Box::new(a), Box::new(b)),
            LogicOp::Implies => Formula::Implies(Box::new(a), Box::new(b)),
        }
    }

    // -- UOI extraction -----------------------------------------------------

    /// For UOI (Unary Operator Insertion):
    /// Neg: E = (x != -x) i.e. x != 0
    /// Not: E = true (always differs)
    fn extract_uoi(
        &mut self,
        original: &IrFunction,
        mutant: &IrFunction,
        op: UnaryOp,
    ) -> AnalysisResult<Formula> {
        let diff = self.wp_engine.wp_difference(original, mutant)?;
        if diff != Formula::False {
            return Ok(diff);
        }

        match op {
            UnaryOp::Neg => {
                // x != -x  <=>  x != 0
                Ok(Formula::Ne(
                    Box::new(Formula::IntVar("__operand".into())),
                    Box::new(Formula::IntConst(0)),
                ))
            }
            UnaryOp::Not => {
                // !b != b is always true
                Ok(Formula::True)
            }
            UnaryOp::BitwiseNot => Ok(Formula::True),
        }
    }

    // -- SDL extraction -----------------------------------------------------

    /// For SDL (Statement Deletion):
    /// The error predicate is the condition under which the deleted statement matters.
    fn extract_sdl(
        &mut self,
        original: &IrFunction,
        mutant: &IrFunction,
    ) -> AnalysisResult<Formula> {
        self.wp_engine.wp_difference(original, mutant)
    }

    // -- Predicate strength comparison --------------------------------------

    /// Check if predicate `a` implies predicate `b` (a is stronger).
    /// Uses syntactic approximation (not full SMT check).
    pub fn implies_syntactic(a: &Formula, b: &Formula) -> bool {
        if a == b {
            return true;
        }
        if matches!(a, Formula::False) {
            return true;
        }
        if matches!(b, Formula::True) {
            return true;
        }

        // a => a || c
        if let Formula::Or(l, r) = b {
            if l.as_ref() == a || r.as_ref() == a {
                return true;
            }
        }
        // a && c => a
        if let Formula::And(l, r) = a {
            if l.as_ref() == b || r.as_ref() == b {
                return true;
            }
        }

        false
    }

    /// Normalize a predicate for comparison.
    pub fn normalize(formula: &Formula) -> Formula {
        let mut engine = WpEngine::with_defaults();
        engine.simplify_formula(formula)
    }
}

impl Default for ErrorPredicateExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_lowering::IrLowering;
    use crate::parser::Parser;
    use shared_types::{MutantId, Span};

    fn get_ir(src: &str) -> IrFunction {
        let prog = Parser::parse_source(src).unwrap();
        let ir = IrLowering::new().lower_program(&prog).unwrap();
        ir.functions.into_iter().next().unwrap()
    }

    fn make_mutation(id: &str, op: MutationOperator) -> Mutation {
        Mutation {
            id: MutantId(id.into()),
            operator: op,
            function_name: "f".into(),
            location: Span::dummy(),
            original_text: "".into(),
            replacement_text: "".into(),
        }
    }

    #[test]
    fn test_aor_extraction() {
        let orig = get_ir("fn f(x: int, y: int) -> int { return x + y; }");
        let mutant = get_ir("fn f(x: int, y: int) -> int { return x - y; }");
        let mutation = make_mutation(
            "aor1",
            MutationOperator::AOR {
                original: ArithOp::Add,
                replacement: ArithOp::Sub,
            },
        );
        let mut ext = ErrorPredicateExtractor::new();
        let pred = ext.extract(&orig, &mutant, &mutation).unwrap();
        assert!(!pred.is_trivial_false());
    }

    #[test]
    fn test_ror_extraction() {
        let orig = get_ir("fn f(x: int) -> int { if (x >= 0) { return x; } else { return -x; } }");
        let mutant = get_ir("fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }");
        let mutation = make_mutation(
            "ror1",
            MutationOperator::ROR {
                original: RelOp::Ge,
                replacement: RelOp::Gt,
            },
        );
        let mut ext = ErrorPredicateExtractor::new();
        let pred = ext.extract(&orig, &mutant, &mutation).unwrap();
        assert!(!pred.is_trivial_false());
    }

    #[test]
    fn test_lcr_extraction() {
        let orig = get_ir("fn f(a: bool, b: bool) -> bool { let r: bool = a && b; return r; }");
        let mutant = get_ir("fn f(a: bool, b: bool) -> bool { let r: bool = a || b; return r; }");
        let mutation = make_mutation(
            "lcr1",
            MutationOperator::LCR {
                original: LogicOp::And,
                replacement: LogicOp::Or,
            },
        );
        let mut ext = ErrorPredicateExtractor::new();
        let pred = ext.extract(&orig, &mutant, &mutation).unwrap();
        assert!(!pred.is_trivial_false());
    }

    #[test]
    fn test_uoi_extraction() {
        let orig = get_ir("fn f(x: int) -> int { return x; }");
        let mutant = get_ir("fn f(x: int) -> int { return -x; }");
        let mutation = make_mutation("uoi1", MutationOperator::UOI { op: UnaryOp::Neg });
        let mut ext = ErrorPredicateExtractor::new();
        let pred = ext.extract(&orig, &mutant, &mutation).unwrap();
        assert!(!pred.is_trivial_false());
    }

    #[test]
    fn test_sdl_extraction() {
        let orig = get_ir("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        let mutant = get_ir("fn f(x: int) -> int { return x; }");
        let mutation = make_mutation("sdl1", MutationOperator::SDL);
        let mut ext = ErrorPredicateExtractor::new();
        let pred = ext.extract(&orig, &mutant, &mutation).unwrap();
        // Deletion of y=x+1 and return y vs return x: diverges when x+1 != x (always)
        let _ = pred;
    }

    #[test]
    fn test_batch_extraction() {
        let orig = get_ir("fn f(x: int, y: int) -> int { return x + y; }");
        let m1_func = get_ir("fn f(x: int, y: int) -> int { return x - y; }");
        let m2_func = get_ir("fn f(x: int, y: int) -> int { return x * y; }");
        let m1 = make_mutation(
            "m1",
            MutationOperator::AOR {
                original: ArithOp::Add,
                replacement: ArithOp::Sub,
            },
        );
        let m2 = make_mutation(
            "m2",
            MutationOperator::AOR {
                original: ArithOp::Add,
                replacement: ArithOp::Mul,
            },
        );
        let mut ext = ErrorPredicateExtractor::new();
        let set = ext
            .extract_batch(&orig, &[(m1, m1_func), (m2, m2_func)])
            .unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_error_predicate_set() {
        let mut set = ErrorPredicateSet::new();
        set.add(ErrorPredicate::new(
            MutantId("m1".into()),
            Formula::True,
            MutationOperator::SDL,
        ));
        set.add(ErrorPredicate::new(
            MutantId("m2".into()),
            Formula::False,
            MutationOperator::SDL,
        ));
        assert_eq!(set.len(), 2);
        assert_eq!(set.non_trivial().len(), 1);
        assert!(set.get("m1").unwrap().is_trivial_true());
        assert!(set.get("m2").unwrap().is_trivial_false());
    }

    #[test]
    fn test_sorted_by_complexity() {
        let mut set = ErrorPredicateSet::new();
        set.add(ErrorPredicate::new(
            MutantId("m1".into()),
            Formula::And(
                Box::new(Formula::Gt(
                    Box::new(Formula::IntVar("x".into())),
                    Box::new(Formula::IntConst(0)),
                )),
                Box::new(Formula::Lt(
                    Box::new(Formula::IntVar("y".into())),
                    Box::new(Formula::IntConst(10)),
                )),
            ),
            MutationOperator::SDL,
        ));
        set.add(ErrorPredicate::new(
            MutantId("m2".into()),
            Formula::True,
            MutationOperator::SDL,
        ));
        let sorted = set.sorted_by_complexity();
        assert!(sorted[0].complexity() <= sorted[1].complexity());
    }

    #[test]
    fn test_implies_syntactic() {
        assert!(ErrorPredicateExtractor::implies_syntactic(
            &Formula::False,
            &Formula::True
        ));
        assert!(ErrorPredicateExtractor::implies_syntactic(
            &Formula::True,
            &Formula::True
        ));
        let a = Formula::IntVar("x".into());
        let b = Formula::Or(Box::new(a.clone()), Box::new(Formula::IntVar("y".into())));
        assert!(ErrorPredicateExtractor::implies_syntactic(&a, &b));
    }

    #[test]
    fn test_normalize() {
        let f = Formula::and(
            Formula::True,
            Formula::Gt(
                Box::new(Formula::IntVar("x".into())),
                Box::new(Formula::IntConst(0)),
            ),
        );
        let n = ErrorPredicateExtractor::normalize(&f);
        assert!(matches!(n, Formula::Gt(..)));
    }

    #[test]
    fn test_predicate_properties() {
        let p = ErrorPredicate::new(
            MutantId("test".into()),
            Formula::Gt(
                Box::new(Formula::IntVar("x".into())),
                Box::new(Formula::IntConst(0)),
            ),
            MutationOperator::SDL,
        );
        assert!(!p.is_trivial_true());
        assert!(!p.is_trivial_false());
        assert!(p.complexity() > 0);
    }
}
