use crate::{SmtExpr, SmtSort};

/// Constraint simplifier: applies algebraic simplification rules to SMT expressions.
/// Rules: double negation elimination, identity elements, absorption, constant folding.
pub struct ConstraintSimplifier {
    max_iterations: usize,
    stats: SimplificationStats,
}

#[derive(Debug, Clone, Default)]
pub struct SimplificationStats {
    pub rules_applied: usize,
    pub constant_folds: usize,
    pub double_negations: usize,
    pub identity_eliminations: usize,
    pub absorption_rules: usize,
    pub trivial_eliminations: usize,
    pub size_before: usize,
    pub size_after: usize,
}

impl ConstraintSimplifier {
    pub fn new() -> Self {
        ConstraintSimplifier { max_iterations: 10, stats: SimplificationStats::default() }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn simplify(&mut self, expr: &SmtExpr) -> SmtExpr {
        self.stats = SimplificationStats::default();
        self.stats.size_before = Self::expr_size(expr);

        let mut current = expr.clone();
        for _ in 0..self.max_iterations {
            let prev_rules = self.stats.rules_applied;
            current = self.simplify_pass(&current);
            if self.stats.rules_applied == prev_rules {
                break; // Fixed point
            }
        }

        self.stats.size_after = Self::expr_size(&current);
        current
    }

    fn simplify_pass(&mut self, expr: &SmtExpr) -> SmtExpr {
        match expr {
            SmtExpr::Not(inner) => self.simplify_not(inner),
            SmtExpr::And(exprs) => self.simplify_and(exprs),
            SmtExpr::Or(exprs) => self.simplify_or(exprs),
            SmtExpr::Implies(a, b) => self.simplify_implies(a, b),
            SmtExpr::Ite(c, t, e) => self.simplify_ite(c, t, e),
            _ => expr.clone(),
        }
    }

    fn simplify_not(&mut self, inner: &SmtExpr) -> SmtExpr {
        let simplified_inner = self.simplify_pass(inner);
        match simplified_inner {
            // Double negation: ¬¬A = A
            SmtExpr::Not(ref e) => {
                self.stats.double_negations += 1;
                self.stats.rules_applied += 1;
                (**e).clone()
            }
            // Constant: ¬true = false, ¬false = true
            SmtExpr::BoolLit(v) => {
                self.stats.constant_folds += 1;
                self.stats.rules_applied += 1;
                SmtExpr::BoolLit(!v)
            }
            other => SmtExpr::Not(Box::new(other)),
        }
    }

    fn simplify_and(&mut self, exprs: &[SmtExpr]) -> SmtExpr {
        let mut simplified: Vec<SmtExpr> = Vec::new();

        for e in exprs {
            let s = self.simplify_pass(e);
            match s {
                // Identity: A ∧ true = A
                SmtExpr::BoolLit(true) => {
                    self.stats.identity_eliminations += 1;
                    self.stats.rules_applied += 1;
                    continue;
                }
                // Annihilator: A ∧ false = false
                SmtExpr::BoolLit(false) => {
                    self.stats.absorption_rules += 1;
                    self.stats.rules_applied += 1;
                    return SmtExpr::BoolLit(false);
                }
                // Flatten nested And
                SmtExpr::And(inner) => {
                    simplified.extend(inner);
                    self.stats.rules_applied += 1;
                }
                other => simplified.push(other),
            }
        }

        // Remove duplicates
        let before_len = simplified.len();
        simplified.dedup_by(|a, b| format!("{:?}", a) == format!("{:?}", b));
        if simplified.len() < before_len {
            self.stats.rules_applied += 1;
        }

        match simplified.len() {
            0 => SmtExpr::BoolLit(true),
            1 => simplified.into_iter().next().unwrap(),
            _ => SmtExpr::And(simplified),
        }
    }

    fn simplify_or(&mut self, exprs: &[SmtExpr]) -> SmtExpr {
        let mut simplified: Vec<SmtExpr> = Vec::new();

        for e in exprs {
            let s = self.simplify_pass(e);
            match s {
                // Identity: A ∨ false = A
                SmtExpr::BoolLit(false) => {
                    self.stats.identity_eliminations += 1;
                    self.stats.rules_applied += 1;
                    continue;
                }
                // Annihilator: A ∨ true = true
                SmtExpr::BoolLit(true) => {
                    self.stats.absorption_rules += 1;
                    self.stats.rules_applied += 1;
                    return SmtExpr::BoolLit(true);
                }
                // Flatten nested Or
                SmtExpr::Or(inner) => {
                    simplified.extend(inner);
                    self.stats.rules_applied += 1;
                }
                other => simplified.push(other),
            }
        }

        let before_len = simplified.len();
        simplified.dedup_by(|a, b| format!("{:?}", a) == format!("{:?}", b));
        if simplified.len() < before_len {
            self.stats.rules_applied += 1;
        }

        match simplified.len() {
            0 => SmtExpr::BoolLit(false),
            1 => simplified.into_iter().next().unwrap(),
            _ => SmtExpr::Or(simplified),
        }
    }

    fn simplify_implies(&mut self, a: &SmtExpr, b: &SmtExpr) -> SmtExpr {
        let sa = self.simplify_pass(a);
        let sb = self.simplify_pass(b);
        match (&sa, &sb) {
            // false → B = true
            (SmtExpr::BoolLit(false), _) => {
                self.stats.constant_folds += 1;
                self.stats.rules_applied += 1;
                SmtExpr::BoolLit(true)
            }
            // A → true = true
            (_, SmtExpr::BoolLit(true)) => {
                self.stats.constant_folds += 1;
                self.stats.rules_applied += 1;
                SmtExpr::BoolLit(true)
            }
            // true → B = B
            (SmtExpr::BoolLit(true), _) => {
                self.stats.identity_eliminations += 1;
                self.stats.rules_applied += 1;
                sb
            }
            // A → false = ¬A
            (_, SmtExpr::BoolLit(false)) => {
                self.stats.rules_applied += 1;
                SmtExpr::Not(Box::new(sa))
            }
            _ => SmtExpr::Implies(Box::new(sa), Box::new(sb)),
        }
    }

    fn simplify_ite(&mut self, cond: &SmtExpr, then_br: &SmtExpr, else_br: &SmtExpr) -> SmtExpr {
        let sc = self.simplify_pass(cond);
        let st = self.simplify_pass(then_br);
        let se = self.simplify_pass(else_br);
        match &sc {
            SmtExpr::BoolLit(true) => {
                self.stats.constant_folds += 1;
                self.stats.rules_applied += 1;
                st
            }
            SmtExpr::BoolLit(false) => {
                self.stats.constant_folds += 1;
                self.stats.rules_applied += 1;
                se
            }
            _ => {
                // If both branches are the same, just return one
                if format!("{:?}", st) == format!("{:?}", se) {
                    self.stats.trivial_eliminations += 1;
                    self.stats.rules_applied += 1;
                    st
                } else {
                    SmtExpr::Ite(Box::new(sc), Box::new(st), Box::new(se))
                }
            }
        }
    }

    fn expr_size(expr: &SmtExpr) -> usize {
        match expr {
            SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_) | SmtExpr::Var(..) => 1,
            SmtExpr::Not(e) | SmtExpr::Neg(e) => 1 + Self::expr_size(e),
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Mul(es) => {
                1 + es.iter().map(|e| Self::expr_size(e)).sum::<usize>()
            }
            SmtExpr::Implies(a, b) | SmtExpr::Eq(a, b) | SmtExpr::Sub(a, b)
            | SmtExpr::Lt(a, b) | SmtExpr::Le(a, b) | SmtExpr::Gt(a, b) | SmtExpr::Ge(a, b) => {
                1 + Self::expr_size(a) + Self::expr_size(b)
            }
            SmtExpr::Ite(c, t, e) => 1 + Self::expr_size(c) + Self::expr_size(t) + Self::expr_size(e),
            SmtExpr::Apply(_, args) => 1 + args.iter().map(|e| Self::expr_size(e)).sum::<usize>(),
        }
    }

    pub fn stats(&self) -> &SimplificationStats {
        &self.stats
    }

    /// Detect obvious conflicts (e.g., x AND NOT x)
    pub fn detect_obvious_conflicts(expr: &SmtExpr) -> Vec<String> {
        let mut conflicts = Vec::new();
        if let SmtExpr::And(ref exprs) = expr {
            for (i, e1) in exprs.iter().enumerate() {
                for e2 in exprs.iter().skip(i + 1) {
                    if Self::is_negation_pair(e1, e2) {
                        conflicts.push(format!("Direct contradiction: {:?} and {:?}", e1, e2));
                    }
                }
            }
        }
        conflicts
    }

    fn is_negation_pair(a: &SmtExpr, b: &SmtExpr) -> bool {
        match (a, b) {
            (SmtExpr::Var(x, _), SmtExpr::Not(inner)) | (SmtExpr::Not(inner), SmtExpr::Var(x, _)) => {
                if let SmtExpr::Var(y, _) = inner.as_ref() { x == y } else { false }
            }
            _ => false,
        }
    }
}

impl Default for ConstraintSimplifier {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_negation() {
        let mut s = ConstraintSimplifier::new();
        let expr = SmtExpr::Not(Box::new(SmtExpr::Not(Box::new(SmtExpr::Var("x".to_string(), SmtSort::Bool)))));
        let result = s.simplify(&expr);
        assert!(matches!(result, SmtExpr::Var(..)));
    }

    #[test]
    fn test_and_identity() {
        let mut s = ConstraintSimplifier::new();
        let expr = SmtExpr::And(vec![SmtExpr::BoolLit(true), SmtExpr::Var("x".to_string(), SmtSort::Bool)]);
        let result = s.simplify(&expr);
        assert!(matches!(result, SmtExpr::Var(..)));
    }

    #[test]
    fn test_and_annihilator() {
        let mut s = ConstraintSimplifier::new();
        let expr = SmtExpr::And(vec![SmtExpr::BoolLit(false), SmtExpr::Var("x".to_string(), SmtSort::Bool)]);
        let result = s.simplify(&expr);
        assert!(matches!(result, SmtExpr::BoolLit(false)));
    }

    #[test]
    fn test_or_annihilator() {
        let mut s = ConstraintSimplifier::new();
        let expr = SmtExpr::Or(vec![SmtExpr::BoolLit(true), SmtExpr::Var("x".to_string(), SmtSort::Bool)]);
        let result = s.simplify(&expr);
        assert!(matches!(result, SmtExpr::BoolLit(true)));
    }

    #[test]
    fn test_implies_simplification() {
        let mut s = ConstraintSimplifier::new();
        let expr = SmtExpr::Implies(
            Box::new(SmtExpr::BoolLit(false)),
            Box::new(SmtExpr::Var("x".to_string(), SmtSort::Bool)),
        );
        let result = s.simplify(&expr);
        assert!(matches!(result, SmtExpr::BoolLit(true)));
    }

    #[test]
    fn test_detect_conflicts() {
        let expr = SmtExpr::And(vec![
            SmtExpr::Var("x".to_string(), SmtSort::Bool),
            SmtExpr::Not(Box::new(SmtExpr::Var("x".to_string(), SmtSort::Bool))),
        ]);
        let conflicts = ConstraintSimplifier::detect_obvious_conflicts(&expr);
        assert_eq!(conflicts.len(), 1);
    }
}
