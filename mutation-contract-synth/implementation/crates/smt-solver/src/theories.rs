//! SMT theory-specific utilities and formula builders.
//!
//! Provides helper functions for constructing SMT expressions within
//! specific theories: core (Bool), QF_LIA (quantifier-free linear integer
//! arithmetic), and arrays.

/// Core theory (Bool connectives, equality, ITE).
pub mod core {
    use crate::ast::SmtExpr;

    /// Build `(and a b)`.
    pub fn mk_and(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::and(a, b)
    }

    /// Build a conjunction of many formulas.
    pub fn mk_and_many(fs: Vec<SmtExpr>) -> SmtExpr {
        SmtExpr::and_many(fs)
    }

    /// Build `(or a b)`.
    pub fn mk_or(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::or(a, b)
    }

    /// Build a disjunction of many formulas.
    pub fn mk_or_many(fs: Vec<SmtExpr>) -> SmtExpr {
        SmtExpr::or_many(fs)
    }

    /// Build `(not e)`.
    pub fn mk_not(e: SmtExpr) -> SmtExpr {
        SmtExpr::not(e)
    }

    /// Build `(=> a b)`.
    pub fn mk_implies(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::implies(a, b)
    }

    /// Build `(= a b)`.
    pub fn mk_eq(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::eq(a, b)
    }

    /// Build `(distinct a b)`.
    pub fn mk_distinct(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::Distinct(vec![a, b])
    }

    /// Build `(ite c t e)`.
    pub fn mk_ite(c: SmtExpr, t: SmtExpr, e: SmtExpr) -> SmtExpr {
        SmtExpr::ite(c, t, e)
    }

    /// Build `(iff a b)` as `(and (=> a b) (=> b a))`.
    pub fn mk_iff(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::and(
            SmtExpr::implies(a.clone(), b.clone()),
            SmtExpr::implies(b, a),
        )
    }

    /// Build `(xor a b)` as `(not (= a b))`.
    pub fn mk_xor(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::not(SmtExpr::eq(a, b))
    }

    /// Build `(and p (not q))` — p but not q.
    pub fn mk_and_not(p: SmtExpr, q: SmtExpr) -> SmtExpr {
        SmtExpr::and(p, SmtExpr::not(q))
    }

    /// Negate a formula, applying simple simplifications.
    pub fn negate(e: SmtExpr) -> SmtExpr {
        match e {
            SmtExpr::True => SmtExpr::False,
            SmtExpr::False => SmtExpr::True,
            SmtExpr::Not(inner) => *inner,
            other => SmtExpr::not(other),
        }
    }

    /// Build a chain of equalities: `(and (= a b) (= b c) ...)`.
    pub fn mk_eq_chain(exprs: &[SmtExpr]) -> SmtExpr {
        if exprs.len() < 2 {
            return SmtExpr::True;
        }
        let eqs: Vec<SmtExpr> = exprs
            .windows(2)
            .map(|w| SmtExpr::eq(w[0].clone(), w[1].clone()))
            .collect();
        SmtExpr::and_many(eqs)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_mk_and() {
            let r = mk_and(SmtExpr::sym("p"), SmtExpr::sym("q"));
            assert_eq!(r.to_string(), "(and p q)");
        }

        #[test]
        fn test_mk_not() {
            let r = mk_not(SmtExpr::sym("p"));
            assert_eq!(r.to_string(), "(not p)");
        }

        #[test]
        fn test_negate_double() {
            let e = SmtExpr::not(SmtExpr::sym("p"));
            let result = negate(e);
            assert_eq!(result, SmtExpr::sym("p"));
        }

        #[test]
        fn test_mk_iff() {
            let r = mk_iff(SmtExpr::sym("p"), SmtExpr::sym("q"));
            let s = r.to_string();
            assert!(s.contains("=>"));
        }

        #[test]
        fn test_eq_chain() {
            let r = mk_eq_chain(&[SmtExpr::sym("a"), SmtExpr::sym("b"), SmtExpr::sym("c")]);
            let s = r.to_string();
            assert!(s.contains("= a b"));
            assert!(s.contains("= b c"));
        }

        #[test]
        fn test_eq_chain_single() {
            let r = mk_eq_chain(&[SmtExpr::sym("a")]);
            assert_eq!(r, SmtExpr::True);
        }
    }
}

/// QF_LIA (Quantifier-Free Linear Integer Arithmetic) theory utilities.
pub mod qf_lia {
    use crate::ast::{SmtExpr, SmtSort};
    use crate::context::SmtContext;

    /// Build an integer constant.
    pub fn mk_int(v: i64) -> SmtExpr {
        SmtExpr::int(v)
    }

    /// Build a variable reference.
    pub fn mk_var(name: &str) -> SmtExpr {
        SmtExpr::sym(name)
    }

    /// Build `(+ a b)`.
    pub fn mk_add(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::add(a, b)
    }

    /// Build `(- a b)`.
    pub fn mk_sub(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::sub(a, b)
    }

    /// Build `(* c x)` where c is a constant.
    pub fn mk_scale(coeff: i64, x: SmtExpr) -> SmtExpr {
        if coeff == 0 {
            SmtExpr::int(0)
        } else if coeff == 1 {
            x
        } else if coeff == -1 {
            SmtExpr::Sub(vec![SmtExpr::int(0), x])
        } else {
            SmtExpr::mul(SmtExpr::int(coeff), x)
        }
    }

    /// Build a linear combination: c0 + c1*x1 + c2*x2 + ...
    pub fn mk_linear_combination(constant: i64, terms: &[(i64, &str)]) -> SmtExpr {
        let mut parts = Vec::new();
        if constant != 0 || terms.is_empty() {
            parts.push(SmtExpr::int(constant));
        }
        for (coeff, var) in terms {
            parts.push(mk_scale(*coeff, SmtExpr::sym(*var)));
        }
        if parts.len() == 1 {
            parts.into_iter().next().unwrap()
        } else {
            SmtExpr::Add(parts)
        }
    }

    /// Build `(< a b)`.
    pub fn mk_lt(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::lt(a, b)
    }

    /// Build `(<= a b)`.
    pub fn mk_le(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::le(a, b)
    }

    /// Build `(> a b)`.
    pub fn mk_gt(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::gt(a, b)
    }

    /// Build `(>= a b)`.
    pub fn mk_ge(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::ge(a, b)
    }

    /// Build `(= a b)`.
    pub fn mk_eq(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::eq(a, b)
    }

    /// Build `(div a b)`.
    pub fn mk_div(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::div(a, b)
    }

    /// Build `(mod a b)`.
    pub fn mk_mod(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::modulo(a, b)
    }

    /// Build `(abs e)`.
    pub fn mk_abs(e: SmtExpr) -> SmtExpr {
        SmtExpr::abs(e)
    }

    /// Build a range constraint: `lo <= x <= hi`.
    pub fn mk_range(x: SmtExpr, lo: i64, hi: i64) -> SmtExpr {
        SmtExpr::and(
            SmtExpr::ge(x.clone(), SmtExpr::int(lo)),
            SmtExpr::le(x, SmtExpr::int(hi)),
        )
    }

    /// Build `x = clamp(x, lo, hi)` — the clamp postcondition.
    pub fn mk_clamp_post(x: &str, lo: i64, hi: i64) -> SmtExpr {
        SmtExpr::and(
            SmtExpr::ge(SmtExpr::sym(x), SmtExpr::int(lo)),
            SmtExpr::le(SmtExpr::sym(x), SmtExpr::int(hi)),
        )
    }

    /// Build an absolute-value definition: `result = (ite (>= x 0) x (- 0 x))`.
    pub fn mk_abs_def(x: &str, result: &str) -> SmtExpr {
        SmtExpr::eq(
            SmtExpr::sym(result),
            SmtExpr::ite(
                SmtExpr::ge(SmtExpr::sym(x), SmtExpr::int(0)),
                SmtExpr::sym(x),
                SmtExpr::Sub(vec![SmtExpr::int(0), SmtExpr::sym(x)]),
            ),
        )
    }

    /// Build max(a, b) = (ite (>= a b) a b).
    pub fn mk_max(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::ite(SmtExpr::ge(a.clone(), b.clone()), a, b)
    }

    /// Build min(a, b) = (ite (<= a b) a b).
    pub fn mk_min(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::ite(SmtExpr::le(a.clone(), b.clone()), a, b)
    }

    /// Build a sign function: sign(x) = (ite (> x 0) 1 (ite (< x 0) -1 0)).
    pub fn mk_sign(x: SmtExpr) -> SmtExpr {
        SmtExpr::ite(
            SmtExpr::gt(x.clone(), SmtExpr::int(0)),
            SmtExpr::int(1),
            SmtExpr::ite(
                SmtExpr::lt(x, SmtExpr::int(0)),
                SmtExpr::int(-1),
                SmtExpr::int(0),
            ),
        )
    }

    /// Build a divisibility constraint: `(= (mod x d) 0)`.
    pub fn mk_divides(d: i64, x: SmtExpr) -> SmtExpr {
        SmtExpr::eq(SmtExpr::modulo(x, SmtExpr::int(d)), SmtExpr::int(0))
    }

    /// Create a QF_LIA context with common variable declarations.
    pub fn standard_context(vars: &[&str]) -> SmtContext {
        let mut ctx = SmtContext::qf_lia();
        for v in vars {
            ctx.declare_int(v);
        }
        ctx
    }

    /// Build a "bounded model checking" unrolling constraint:
    /// variables x_0, x_1, ..., x_n with transition relation T(x_i, x_{i+1}).
    pub fn mk_bmc_unroll(
        base: &str,
        steps: usize,
        transition: impl Fn(SmtExpr, SmtExpr) -> SmtExpr,
    ) -> (Vec<SmtExpr>, SmtExpr) {
        let vars: Vec<SmtExpr> = (0..=steps)
            .map(|i| SmtExpr::sym(format!("{}_{}", base, i)))
            .collect();
        let transitions: Vec<SmtExpr> = vars
            .windows(2)
            .map(|w| transition(w[0].clone(), w[1].clone()))
            .collect();
        let conj = SmtExpr::and_many(transitions);
        (vars, conj)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_mk_int() {
            assert_eq!(mk_int(42).to_string(), "42");
            assert_eq!(mk_int(-1).to_string(), "(- 1)");
        }

        #[test]
        fn test_mk_linear_combination() {
            let e = mk_linear_combination(5, &[(2, "x"), (-1, "y")]);
            let s = e.to_string();
            assert!(s.contains("+"));
            assert!(s.contains("5"));
        }

        #[test]
        fn test_mk_range() {
            let e = mk_range(SmtExpr::sym("x"), 0, 100);
            let s = e.to_string();
            assert!(s.contains(">= x 0"));
            assert!(s.contains("<= x 100"));
        }

        #[test]
        fn test_mk_scale_identity() {
            let e = mk_scale(1, SmtExpr::sym("x"));
            assert_eq!(e, SmtExpr::sym("x"));
        }

        #[test]
        fn test_mk_scale_zero() {
            let e = mk_scale(0, SmtExpr::sym("x"));
            assert_eq!(e, SmtExpr::int(0));
        }

        #[test]
        fn test_mk_abs_def() {
            let e = mk_abs_def("x", "r");
            let s = e.to_string();
            assert!(s.contains("ite"));
            assert!(s.contains(">= x 0"));
        }

        #[test]
        fn test_mk_max() {
            let e = mk_max(SmtExpr::sym("a"), SmtExpr::sym("b"));
            let s = e.to_string();
            assert!(s.contains("ite"));
            assert!(s.contains(">= a b"));
        }

        #[test]
        fn test_mk_min() {
            let e = mk_min(SmtExpr::sym("a"), SmtExpr::sym("b"));
            let s = e.to_string();
            assert!(s.contains("<= a b"));
        }

        #[test]
        fn test_mk_divides() {
            let e = mk_divides(3, SmtExpr::sym("x"));
            let s = e.to_string();
            assert!(s.contains("mod"));
            assert!(s.contains("= "));
        }

        #[test]
        fn test_standard_context() {
            let ctx = standard_context(&["x", "y", "z"]);
            assert!(ctx.is_declared("x"));
            assert!(ctx.is_declared("y"));
            assert!(ctx.is_declared("z"));
        }

        #[test]
        fn test_bmc_unroll() {
            let (vars, constraint) = mk_bmc_unroll("x", 3, |cur, next| {
                SmtExpr::eq(next, SmtExpr::add(cur, SmtExpr::int(1)))
            });
            assert_eq!(vars.len(), 4);
            let s = constraint.to_string();
            assert!(s.contains("x_0"));
            assert!(s.contains("x_3"));
        }
    }
}

/// Array theory utilities (select/store).
pub mod arrays {
    use crate::ast::{SmtExpr, SmtSort};
    use crate::context::SmtContext;

    /// Build `(select arr idx)`.
    pub fn mk_select(arr: SmtExpr, idx: SmtExpr) -> SmtExpr {
        SmtExpr::select(arr, idx)
    }

    /// Build `(store arr idx val)`.
    pub fn mk_store(arr: SmtExpr, idx: SmtExpr, val: SmtExpr) -> SmtExpr {
        SmtExpr::store(arr, idx, val)
    }

    /// Build a chain of stores: `(store (store (store arr i0 v0) i1 v1) i2 v2)`.
    pub fn mk_stores(arr: SmtExpr, updates: &[(SmtExpr, SmtExpr)]) -> SmtExpr {
        let mut result = arr;
        for (idx, val) in updates {
            result = SmtExpr::store(result, idx.clone(), val.clone());
        }
        result
    }

    /// Build an array equality over a range of indices:
    /// `(and (= (select a 0) (select b 0)) (= (select a 1) (select b 1)) ...)`.
    pub fn mk_array_eq_range(a: &SmtExpr, b: &SmtExpr, lo: i64, hi: i64) -> SmtExpr {
        let eqs: Vec<SmtExpr> = (lo..=hi)
            .map(|i| {
                SmtExpr::eq(
                    SmtExpr::select(a.clone(), SmtExpr::int(i)),
                    SmtExpr::select(b.clone(), SmtExpr::int(i)),
                )
            })
            .collect();
        SmtExpr::and_many(eqs)
    }

    /// Build a "sorted" constraint for an array over [lo, hi):
    /// `(and (<= (select a 0) (select a 1)) (<= (select a 1) (select a 2)) ...)`.
    pub fn mk_sorted(arr: &SmtExpr, lo: i64, hi: i64) -> SmtExpr {
        if hi <= lo + 1 {
            return SmtExpr::True;
        }
        let constraints: Vec<SmtExpr> = (lo..hi - 1)
            .map(|i| {
                SmtExpr::le(
                    SmtExpr::select(arr.clone(), SmtExpr::int(i)),
                    SmtExpr::select(arr.clone(), SmtExpr::int(i + 1)),
                )
            })
            .collect();
        SmtExpr::and_many(constraints)
    }

    /// Build a bounded sum: `result = a[lo] + a[lo+1] + ... + a[hi-1]`.
    pub fn mk_array_sum(arr: &SmtExpr, lo: i64, hi: i64) -> SmtExpr {
        let terms: Vec<SmtExpr> = (lo..hi)
            .map(|i| SmtExpr::select(arr.clone(), SmtExpr::int(i)))
            .collect();
        if terms.is_empty() {
            SmtExpr::int(0)
        } else if terms.len() == 1 {
            terms.into_iter().next().unwrap()
        } else {
            SmtExpr::Add(terms)
        }
    }

    /// Build a "permutation" constraint (weak form): same sum and same count of elements.
    /// This is an incomplete check but useful for bounded verification.
    pub fn mk_permutation_sum_check(a: &SmtExpr, b: &SmtExpr, lo: i64, hi: i64) -> SmtExpr {
        let sum_a = mk_array_sum(a, lo, hi);
        let sum_b = mk_array_sum(b, lo, hi);
        SmtExpr::eq(sum_a, sum_b)
    }

    /// Declare an integer array in the context.
    pub fn declare_int_array(ctx: &mut SmtContext, name: &str) {
        ctx.declare_const(name, SmtSort::int_array());
    }

    /// Build "all elements in range" constraint: lo <= a[i] <= hi for i in [start, end).
    pub fn mk_all_in_range(arr: &SmtExpr, start: i64, end: i64, lo: i64, hi: i64) -> SmtExpr {
        let constraints: Vec<SmtExpr> = (start..end)
            .map(|i| {
                let elem = SmtExpr::select(arr.clone(), SmtExpr::int(i));
                SmtExpr::and(
                    SmtExpr::ge(elem.clone(), SmtExpr::int(lo)),
                    SmtExpr::le(elem, SmtExpr::int(hi)),
                )
            })
            .collect();
        SmtExpr::and_many(constraints)
    }

    /// Build a swap constraint: b = store(store(a, i, a[j]), j, a[i]).
    pub fn mk_swap(a: &SmtExpr, b: &SmtExpr, i: SmtExpr, j: SmtExpr) -> SmtExpr {
        let ai = SmtExpr::select(a.clone(), i.clone());
        let aj = SmtExpr::select(a.clone(), j.clone());
        let step1 = SmtExpr::store(a.clone(), i.clone(), aj);
        let step2 = SmtExpr::store(step1, j, ai);
        SmtExpr::eq(b.clone(), step2)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_mk_select() {
            let e = mk_select(SmtExpr::sym("a"), SmtExpr::int(0));
            assert_eq!(e.to_string(), "(select a 0)");
        }

        #[test]
        fn test_mk_store() {
            let e = mk_store(SmtExpr::sym("a"), SmtExpr::int(0), SmtExpr::int(42));
            assert_eq!(e.to_string(), "(store a 0 42)");
        }

        #[test]
        fn test_mk_stores_chain() {
            let e = mk_stores(
                SmtExpr::sym("a"),
                &[
                    (SmtExpr::int(0), SmtExpr::int(10)),
                    (SmtExpr::int(1), SmtExpr::int(20)),
                ],
            );
            let s = e.to_string();
            assert!(s.contains("store"));
        }

        #[test]
        fn test_mk_sorted() {
            let e = mk_sorted(&SmtExpr::sym("a"), 0, 3);
            let s = e.to_string();
            assert!(s.contains("<="));
            assert!(s.contains("select"));
        }

        #[test]
        fn test_mk_array_sum() {
            let e = mk_array_sum(&SmtExpr::sym("a"), 0, 3);
            let s = e.to_string();
            assert!(s.contains("+"));
            assert!(s.contains("select a 0"));
        }

        #[test]
        fn test_mk_array_eq_range() {
            let e = mk_array_eq_range(&SmtExpr::sym("a"), &SmtExpr::sym("b"), 0, 1);
            let s = e.to_string();
            assert!(s.contains("= (select a"));
        }

        #[test]
        fn test_mk_all_in_range() {
            let e = mk_all_in_range(&SmtExpr::sym("a"), 0, 2, 0, 100);
            let s = e.to_string();
            assert!(s.contains(">="));
            assert!(s.contains("<="));
        }

        #[test]
        fn test_mk_swap() {
            let e = mk_swap(
                &SmtExpr::sym("a"),
                &SmtExpr::sym("b"),
                SmtExpr::int(0),
                SmtExpr::int(1),
            );
            let s = e.to_string();
            assert!(s.contains("store"));
        }
    }
}
