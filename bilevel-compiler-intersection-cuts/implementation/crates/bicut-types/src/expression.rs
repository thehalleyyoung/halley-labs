//! Linear and quadratic expression types with full arithmetic support.
//!
//! Expressions represent mathematical functions of optimization variables.
//! They support building, evaluation at a point, canonicalization (sorting
//! and merging duplicate terms), and Rust operator overloading.

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use serde::{Deserialize, Serialize};

use crate::variable::VariableId;

// ── Expression term ────────────────────────────────────────────────

/// A single term in an expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprTerm {
    /// A constant value.
    Constant(f64),
    /// coefficient * variable.
    Linear { coeff: f64, var: VariableId },
    /// coefficient * var_i * var_j.
    Quadratic {
        coeff: f64,
        var_i: VariableId,
        var_j: VariableId,
    },
}

impl ExprTerm {
    pub fn constant(val: f64) -> Self {
        ExprTerm::Constant(val)
    }

    pub fn linear(coeff: f64, var: VariableId) -> Self {
        ExprTerm::Linear { coeff, var }
    }

    pub fn quadratic(coeff: f64, var_i: VariableId, var_j: VariableId) -> Self {
        // Canonicalize so var_i <= var_j
        if var_i.0 <= var_j.0 {
            ExprTerm::Quadratic {
                coeff,
                var_i,
                var_j,
            }
        } else {
            ExprTerm::Quadratic {
                coeff,
                var_i: var_j,
                var_j: var_i,
            }
        }
    }

    /// Evaluate this term given variable values.
    pub fn evaluate(&self, values: &[f64]) -> f64 {
        match self {
            ExprTerm::Constant(c) => *c,
            ExprTerm::Linear { coeff, var } => coeff * values[var.0],
            ExprTerm::Quadratic {
                coeff,
                var_i,
                var_j,
            } => coeff * values[var_i.0] * values[var_j.0],
        }
    }

    /// Scale this term by a constant.
    pub fn scale(&self, factor: f64) -> Self {
        match self {
            ExprTerm::Constant(c) => ExprTerm::Constant(c * factor),
            ExprTerm::Linear { coeff, var } => ExprTerm::Linear {
                coeff: coeff * factor,
                var: *var,
            },
            ExprTerm::Quadratic {
                coeff,
                var_i,
                var_j,
            } => ExprTerm::Quadratic {
                coeff: coeff * factor,
                var_i: *var_i,
                var_j: *var_j,
            },
        }
    }

    /// Whether the coefficient is effectively zero.
    pub fn is_zero(&self, tol: f64) -> bool {
        match self {
            ExprTerm::Constant(c) => c.abs() < tol,
            ExprTerm::Linear { coeff, .. } => coeff.abs() < tol,
            ExprTerm::Quadratic { coeff, .. } => coeff.abs() < tol,
        }
    }
}

impl fmt::Display for ExprTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprTerm::Constant(c) => write!(f, "{}", c),
            ExprTerm::Linear { coeff, var } => write!(f, "{}*{}", coeff, var),
            ExprTerm::Quadratic {
                coeff,
                var_i,
                var_j,
            } => {
                if var_i == var_j {
                    write!(f, "{}*{}^2", coeff, var_i)
                } else {
                    write!(f, "{}*{}*{}", coeff, var_i, var_j)
                }
            }
        }
    }
}

// ── Linear expression ──────────────────────────────────────────────

/// A linear expression: constant + Σ aᵢxᵢ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearExpr {
    /// Constant (offset) term.
    pub constant: f64,
    /// Sparse representation: (variable_id, coefficient).
    pub terms: Vec<(VariableId, f64)>,
}

impl LinearExpr {
    pub fn zero() -> Self {
        Self {
            constant: 0.0,
            terms: Vec::new(),
        }
    }

    pub fn constant(val: f64) -> Self {
        Self {
            constant: val,
            terms: Vec::new(),
        }
    }

    pub fn single(var: VariableId, coeff: f64) -> Self {
        Self {
            constant: 0.0,
            terms: vec![(var, coeff)],
        }
    }

    pub fn from_terms(constant: f64, terms: Vec<(VariableId, f64)>) -> Self {
        let mut expr = Self { constant, terms };
        expr.canonicalize();
        expr
    }

    /// Add a term c*x to this expression.
    pub fn add_term(&mut self, var: VariableId, coeff: f64) {
        self.terms.push((var, coeff));
    }

    /// Add a constant to this expression.
    pub fn add_constant(&mut self, val: f64) {
        self.constant += val;
    }

    /// Scale all coefficients (including constant) by a factor.
    pub fn scale(&mut self, factor: f64) {
        self.constant *= factor;
        for (_, c) in &mut self.terms {
            *c *= factor;
        }
    }

    /// Return a scaled copy.
    pub fn scaled(&self, factor: f64) -> Self {
        let mut copy = self.clone();
        copy.scale(factor);
        copy
    }

    /// Merge duplicate variable entries and sort by variable id.
    pub fn canonicalize(&mut self) {
        let mut map: HashMap<usize, f64> = HashMap::new();
        for &(var, coeff) in &self.terms {
            *map.entry(var.0).or_insert(0.0) += coeff;
        }
        self.terms = map
            .into_iter()
            .filter(|(_, c)| c.abs() > 1e-15 || c.is_nan())
            .map(|(id, c)| (VariableId(id), c))
            .collect();
        self.terms.sort_by_key(|(v, _)| v.0);
    }

    /// Evaluate at a point.
    pub fn evaluate(&self, values: &[f64]) -> f64 {
        let mut result = self.constant;
        for &(var, coeff) in &self.terms {
            result += coeff * values[var.0];
        }
        result
    }

    /// Number of nonzero terms (excluding constant).
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Whether this expression has no variable terms.
    pub fn is_constant(&self) -> bool {
        self.terms.is_empty()
    }

    /// Set of variable ids referenced.
    pub fn variable_ids(&self) -> Vec<VariableId> {
        self.terms.iter().map(|(v, _)| *v).collect()
    }

    /// Get the coefficient of a specific variable (0 if absent).
    pub fn coeff_of(&self, var: VariableId) -> f64 {
        self.terms
            .iter()
            .filter(|(v, _)| *v == var)
            .map(|(_, c)| c)
            .sum()
    }

    /// Convert to dense coefficient vector of given dimension.
    pub fn to_dense(&self, dim: usize) -> Vec<f64> {
        let mut dense = vec![0.0; dim];
        for &(var, coeff) in &self.terms {
            if var.0 < dim {
                dense[var.0] += coeff;
            }
        }
        dense
    }

    /// Create from a dense coefficient vector.
    pub fn from_dense(constant: f64, coeffs: &[f64]) -> Self {
        let terms: Vec<(VariableId, f64)> = coeffs
            .iter()
            .enumerate()
            .filter(|(_, &c)| c.abs() > 1e-15)
            .map(|(i, &c)| (VariableId(i), c))
            .collect();
        Self { constant, terms }
    }

    /// Dot product of coefficient vector with a value vector (ignores constant).
    pub fn dot(&self, values: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|&(var, coeff)| coeff * values[var.0])
            .sum()
    }
}

impl Default for LinearExpr {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for LinearExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        if self.constant.abs() > 1e-15 {
            write!(f, "{}", self.constant)?;
            first = false;
        }
        for &(var, coeff) in &self.terms {
            if first {
                if (coeff - 1.0).abs() < 1e-15 {
                    write!(f, "{}", var)?;
                } else if (coeff + 1.0).abs() < 1e-15 {
                    write!(f, "-{}", var)?;
                } else {
                    write!(f, "{}*{}", coeff, var)?;
                }
                first = false;
            } else if coeff > 0.0 {
                if (coeff - 1.0).abs() < 1e-15 {
                    write!(f, " + {}", var)?;
                } else {
                    write!(f, " + {}*{}", coeff, var)?;
                }
            } else {
                if (coeff + 1.0).abs() < 1e-15 {
                    write!(f, " - {}", var)?;
                } else {
                    write!(f, " - {}*{}", -coeff, var)?;
                }
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl Add for LinearExpr {
    type Output = LinearExpr;

    fn add(self, rhs: LinearExpr) -> LinearExpr {
        let mut result = self;
        result.constant += rhs.constant;
        result.terms.extend(rhs.terms);
        result.canonicalize();
        result
    }
}

impl Sub for LinearExpr {
    type Output = LinearExpr;

    fn sub(self, rhs: LinearExpr) -> LinearExpr {
        let mut result = self;
        result.constant -= rhs.constant;
        for (var, coeff) in rhs.terms {
            result.terms.push((var, -coeff));
        }
        result.canonicalize();
        result
    }
}

impl Neg for LinearExpr {
    type Output = LinearExpr;

    fn neg(self) -> LinearExpr {
        LinearExpr {
            constant: -self.constant,
            terms: self.terms.into_iter().map(|(v, c)| (v, -c)).collect(),
        }
    }
}

impl Mul<f64> for LinearExpr {
    type Output = LinearExpr;

    fn mul(self, rhs: f64) -> LinearExpr {
        self.scaled(rhs)
    }
}

impl Mul<LinearExpr> for f64 {
    type Output = LinearExpr;

    fn mul(self, rhs: LinearExpr) -> LinearExpr {
        rhs.scaled(self)
    }
}

// ── Quadratic expression ───────────────────────────────────────────

/// A quadratic expression: constant + Σ aᵢxᵢ + Σ qᵢⱼ xᵢ xⱼ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuadraticExpr {
    /// The linear part.
    pub linear: LinearExpr,
    /// Quadratic terms: ((var_i, var_j), coefficient). Always var_i <= var_j.
    pub quad_terms: Vec<(VariableId, VariableId, f64)>,
}

impl QuadraticExpr {
    pub fn zero() -> Self {
        Self {
            linear: LinearExpr::zero(),
            quad_terms: Vec::new(),
        }
    }

    pub fn from_linear(linear: LinearExpr) -> Self {
        Self {
            linear,
            quad_terms: Vec::new(),
        }
    }

    pub fn constant(val: f64) -> Self {
        Self {
            linear: LinearExpr::constant(val),
            quad_terms: Vec::new(),
        }
    }

    /// Add a quadratic term c * x_i * x_j.
    pub fn add_quad_term(&mut self, var_i: VariableId, var_j: VariableId, coeff: f64) {
        let (vi, vj) = if var_i.0 <= var_j.0 {
            (var_i, var_j)
        } else {
            (var_j, var_i)
        };
        self.quad_terms.push((vi, vj, coeff));
    }

    /// Add a linear term.
    pub fn add_linear_term(&mut self, var: VariableId, coeff: f64) {
        self.linear.add_term(var, coeff);
    }

    /// Add a constant.
    pub fn add_constant(&mut self, val: f64) {
        self.linear.add_constant(val);
    }

    /// Scale all terms.
    pub fn scale(&mut self, factor: f64) {
        self.linear.scale(factor);
        for (_, _, c) in &mut self.quad_terms {
            *c *= factor;
        }
    }

    /// Return a scaled copy.
    pub fn scaled(&self, factor: f64) -> Self {
        let mut copy = self.clone();
        copy.scale(factor);
        copy
    }

    /// Canonicalize: merge duplicate quadratic terms, sort, remove near-zeros.
    pub fn canonicalize(&mut self) {
        self.linear.canonicalize();

        let mut map: HashMap<(usize, usize), f64> = HashMap::new();
        for &(vi, vj, coeff) in &self.quad_terms {
            *map.entry((vi.0, vj.0)).or_insert(0.0) += coeff;
        }
        self.quad_terms = map
            .into_iter()
            .filter(|(_, c)| c.abs() > 1e-15)
            .map(|((i, j), c)| (VariableId(i), VariableId(j), c))
            .collect();
        self.quad_terms.sort_by_key(|(vi, vj, _)| (vi.0, vj.0));
    }

    /// Evaluate at a point.
    pub fn evaluate(&self, values: &[f64]) -> f64 {
        let mut result = self.linear.evaluate(values);
        for &(vi, vj, coeff) in &self.quad_terms {
            result += coeff * values[vi.0] * values[vj.0];
        }
        result
    }

    /// Whether this is actually a linear expression (no quadratic terms).
    pub fn is_linear(&self) -> bool {
        self.quad_terms.is_empty()
    }

    /// Number of quadratic terms.
    pub fn num_quad_terms(&self) -> usize {
        self.quad_terms.len()
    }

    /// All variable ids referenced (linear and quadratic).
    pub fn variable_ids(&self) -> Vec<VariableId> {
        let mut ids: Vec<VariableId> = self.linear.variable_ids();
        for &(vi, vj, _) in &self.quad_terms {
            ids.push(vi);
            ids.push(vj);
        }
        ids.sort_by_key(|v| v.0);
        ids.dedup();
        ids
    }

    /// Extract the linear part, discarding quadratic terms.
    pub fn to_linear(&self) -> Option<LinearExpr> {
        if self.is_linear() {
            Some(self.linear.clone())
        } else {
            None
        }
    }

    /// Whether this quadratic is (likely) convex. A rough check: diagonal terms must be non-negative.
    /// This is a necessary condition for convexity but not sufficient for general Q.
    pub fn is_diagonal_convex(&self) -> bool {
        for &(vi, vj, coeff) in &self.quad_terms {
            if vi == vj && coeff < -1e-10 {
                return false;
            }
        }
        true
    }
}

impl Default for QuadraticExpr {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for QuadraticExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.linear)?;
        for &(vi, vj, coeff) in &self.quad_terms {
            if coeff > 0.0 {
                if vi == vj {
                    write!(f, " + {}*{}^2", coeff, vi)?;
                } else {
                    write!(f, " + {}*{}*{}", coeff, vi, vj)?;
                }
            } else {
                if vi == vj {
                    write!(f, " - {}*{}^2", -coeff, vi)?;
                } else {
                    write!(f, " - {}*{}*{}", -coeff, vi, vj)?;
                }
            }
        }
        Ok(())
    }
}

impl Add for QuadraticExpr {
    type Output = QuadraticExpr;

    fn add(self, rhs: QuadraticExpr) -> QuadraticExpr {
        let mut result = QuadraticExpr {
            linear: self.linear + rhs.linear,
            quad_terms: self.quad_terms,
        };
        result.quad_terms.extend(rhs.quad_terms);
        result.canonicalize();
        result
    }
}

impl Sub for QuadraticExpr {
    type Output = QuadraticExpr;

    fn sub(self, rhs: QuadraticExpr) -> QuadraticExpr {
        let mut result = QuadraticExpr {
            linear: self.linear - rhs.linear,
            quad_terms: self.quad_terms,
        };
        for (vi, vj, c) in rhs.quad_terms {
            result.quad_terms.push((vi, vj, -c));
        }
        result.canonicalize();
        result
    }
}

impl Neg for QuadraticExpr {
    type Output = QuadraticExpr;

    fn neg(self) -> QuadraticExpr {
        QuadraticExpr {
            linear: -self.linear,
            quad_terms: self
                .quad_terms
                .into_iter()
                .map(|(vi, vj, c)| (vi, vj, -c))
                .collect(),
        }
    }
}

impl Mul<f64> for QuadraticExpr {
    type Output = QuadraticExpr;

    fn mul(self, rhs: f64) -> QuadraticExpr {
        self.scaled(rhs)
    }
}

impl Mul<QuadraticExpr> for f64 {
    type Output = QuadraticExpr;

    fn mul(self, rhs: QuadraticExpr) -> QuadraticExpr {
        rhs.scaled(self)
    }
}

/// Multiply two linear expressions to produce a quadratic expression.
impl Mul for LinearExpr {
    type Output = QuadraticExpr;

    fn mul(self, rhs: LinearExpr) -> QuadraticExpr {
        let mut result = QuadraticExpr::zero();

        // constant * constant
        result.linear.constant = self.constant * rhs.constant;

        // constant * rhs_linear + lhs_linear * constant
        for &(var, coeff) in &rhs.terms {
            result.linear.add_term(var, self.constant * coeff);
        }
        for &(var, coeff) in &self.terms {
            result.linear.add_term(var, coeff * rhs.constant);
        }

        // lhs_linear * rhs_linear → quadratic
        for &(vi, ci) in &self.terms {
            for &(vj, cj) in &rhs.terms {
                result.add_quad_term(vi, vj, ci * cj);
            }
        }

        result.canonicalize();
        result
    }
}

// ── Conversion helpers ─────────────────────────────────────────────

impl From<LinearExpr> for QuadraticExpr {
    fn from(linear: LinearExpr) -> Self {
        QuadraticExpr::from_linear(linear)
    }
}

impl From<f64> for LinearExpr {
    fn from(val: f64) -> Self {
        LinearExpr::constant(val)
    }
}

impl From<f64> for QuadraticExpr {
    fn from(val: f64) -> Self {
        QuadraticExpr::constant(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_expr_evaluate() {
        let expr = LinearExpr::from_terms(5.0, vec![(VariableId(0), 2.0), (VariableId(1), -3.0)]);
        let vals = vec![1.0, 2.0];
        // 5 + 2*1 + (-3)*2 = 5 + 2 - 6 = 1
        assert!((expr.evaluate(&vals) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_expr_canonicalize() {
        let mut expr = LinearExpr {
            constant: 0.0,
            terms: vec![
                (VariableId(0), 3.0),
                (VariableId(1), 2.0),
                (VariableId(0), -1.0),
            ],
        };
        expr.canonicalize();
        assert_eq!(expr.num_terms(), 2);
        assert!((expr.coeff_of(VariableId(0)) - 2.0).abs() < 1e-10);
        assert!((expr.coeff_of(VariableId(1)) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_expr_add() {
        let a = LinearExpr::from_terms(1.0, vec![(VariableId(0), 2.0)]);
        let b = LinearExpr::from_terms(3.0, vec![(VariableId(0), 4.0), (VariableId(1), 5.0)]);
        let c = a + b;
        assert!((c.constant - 4.0).abs() < 1e-10);
        assert!((c.coeff_of(VariableId(0)) - 6.0).abs() < 1e-10);
        assert!((c.coeff_of(VariableId(1)) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_expr_sub() {
        let a = LinearExpr::from_terms(10.0, vec![(VariableId(0), 5.0)]);
        let b = LinearExpr::from_terms(3.0, vec![(VariableId(0), 2.0)]);
        let c = a - b;
        assert!((c.constant - 7.0).abs() < 1e-10);
        assert!((c.coeff_of(VariableId(0)) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_expr_neg() {
        let a = LinearExpr::from_terms(5.0, vec![(VariableId(0), -3.0)]);
        let b = -a;
        assert!((b.constant - (-5.0)).abs() < 1e-10);
        assert!((b.coeff_of(VariableId(0)) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_expr_scale() {
        let a = LinearExpr::from_terms(2.0, vec![(VariableId(0), 3.0)]);
        let b = a * 4.0;
        assert!((b.constant - 8.0).abs() < 1e-10);
        assert!((b.coeff_of(VariableId(0)) - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_expr_to_dense() {
        let expr = LinearExpr::from_terms(0.0, vec![(VariableId(1), 3.0), (VariableId(3), -1.0)]);
        let dense = expr.to_dense(5);
        assert_eq!(dense, vec![0.0, 3.0, 0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_linear_expr_from_dense() {
        let expr = LinearExpr::from_dense(2.0, &[0.0, 3.0, 0.0, -1.0]);
        assert!((expr.constant - 2.0).abs() < 1e-10);
        assert_eq!(expr.num_terms(), 2);
    }

    #[test]
    fn test_quadratic_expr_evaluate() {
        let mut qe = QuadraticExpr::zero();
        qe.linear = LinearExpr::from_terms(1.0, vec![(VariableId(0), 2.0)]);
        qe.add_quad_term(VariableId(0), VariableId(0), 3.0);
        // 1 + 2*x0 + 3*x0^2 at x0=2 => 1 + 4 + 12 = 17
        let val = qe.evaluate(&[2.0]);
        assert!((val - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_expr_add() {
        let mut a = QuadraticExpr::zero();
        a.linear = LinearExpr::from_terms(1.0, vec![(VariableId(0), 2.0)]);
        a.add_quad_term(VariableId(0), VariableId(1), 3.0);

        let mut b = QuadraticExpr::zero();
        b.linear = LinearExpr::from_terms(2.0, vec![(VariableId(0), 1.0)]);
        b.add_quad_term(VariableId(0), VariableId(1), -1.0);

        let c = a + b;
        assert!((c.linear.constant - 3.0).abs() < 1e-10);
        assert!((c.linear.coeff_of(VariableId(0)) - 3.0).abs() < 1e-10);
        assert_eq!(c.quad_terms.len(), 1);
        assert!((c.quad_terms[0].2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_mul_produces_quadratic() {
        // (1 + 2*x0) * (3 + x1) = 3 + x1 + 6*x0 + 2*x0*x1
        let a = LinearExpr::from_terms(1.0, vec![(VariableId(0), 2.0)]);
        let b = LinearExpr::from_terms(3.0, vec![(VariableId(1), 1.0)]);
        let q = a * b;
        // constant = 1*3 = 3
        assert!((q.linear.constant - 3.0).abs() < 1e-10);
        // Evaluate at x0=1, x1=2: (1+2)*(3+2) = 15
        let val = q.evaluate(&[1.0, 2.0]);
        assert!((val - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_is_linear() {
        let q = QuadraticExpr::from_linear(LinearExpr::single(VariableId(0), 5.0));
        assert!(q.is_linear());
        assert!(q.to_linear().is_some());
    }

    #[test]
    fn test_expr_term_display() {
        let t = ExprTerm::quadratic(2.0, VariableId(0), VariableId(0));
        assert!(t.to_string().contains("^2"));

        let t2 = ExprTerm::quadratic(3.0, VariableId(0), VariableId(1));
        assert!(t2.to_string().contains("v0*v1"));
    }

    #[test]
    fn test_display_zero_expr() {
        let e = LinearExpr::zero();
        assert_eq!(e.to_string(), "0");
    }
}
