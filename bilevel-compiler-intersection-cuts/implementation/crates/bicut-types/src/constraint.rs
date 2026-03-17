//! Constraint types for bilevel optimization problems.
//!
//! Supports linear, quadratic, indicator, and SOS1 constraints with
//! full metadata, evaluation, and violation checking.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{BicutError, BicutResult, ValidationError};
use crate::expression::{LinearExpr, QuadraticExpr};
use crate::variable::VariableId;

// ── Constraint sense ───────────────────────────────────────────────

/// The sense (direction) of a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintSense {
    /// a^T x ≤ b
    Le,
    /// a^T x ≥ b
    Ge,
    /// a^T x = b
    Eq,
}

impl ConstraintSense {
    /// Flip the sense (Le ↔ Ge, Eq stays).
    pub fn flip(&self) -> Self {
        match self {
            ConstraintSense::Le => ConstraintSense::Ge,
            ConstraintSense::Ge => ConstraintSense::Le,
            ConstraintSense::Eq => ConstraintSense::Eq,
        }
    }

    /// Whether a given lhs value satisfies this sense w.r.t. rhs.
    pub fn is_satisfied(&self, lhs: f64, rhs: f64, tol: f64) -> bool {
        match self {
            ConstraintSense::Le => lhs <= rhs + tol,
            ConstraintSense::Ge => lhs >= rhs - tol,
            ConstraintSense::Eq => (lhs - rhs).abs() <= tol,
        }
    }

    /// The violation amount (positive means violated).
    pub fn violation(&self, lhs: f64, rhs: f64) -> f64 {
        match self {
            ConstraintSense::Le => (lhs - rhs).max(0.0),
            ConstraintSense::Ge => (rhs - lhs).max(0.0),
            ConstraintSense::Eq => (lhs - rhs).abs(),
        }
    }
}

impl fmt::Display for ConstraintSense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintSense::Le => write!(f, "<="),
            ConstraintSense::Ge => write!(f, ">="),
            ConstraintSense::Eq => write!(f, "="),
        }
    }
}

// ── Constraint type tag ────────────────────────────────────────────

/// Classification of constraint type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintType {
    Linear,
    Quadratic,
    Indicator,
    Sos1,
}

impl fmt::Display for ConstraintType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintType::Linear => write!(f, "linear"),
            ConstraintType::Quadratic => write!(f, "quadratic"),
            ConstraintType::Indicator => write!(f, "indicator"),
            ConstraintType::Sos1 => write!(f, "SOS1"),
        }
    }
}

// ── Constraint id ──────────────────────────────────────────────────

/// Unique id for a constraint within a problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConstraintId(pub usize);

impl ConstraintId {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

impl fmt::Display for ConstraintId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "c{}", self.0)
    }
}

impl From<usize> for ConstraintId {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

// ── Linear constraint ──────────────────────────────────────────────

/// A linear constraint: expr sense rhs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearConstraint {
    pub id: ConstraintId,
    pub name: String,
    pub expr: LinearExpr,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

impl LinearConstraint {
    pub fn new(
        id: ConstraintId,
        name: impl Into<String>,
        expr: LinearExpr,
        sense: ConstraintSense,
        rhs: f64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            expr,
            sense,
            rhs,
        }
    }

    /// Evaluate lhs at a point and return (lhs_value, rhs).
    pub fn evaluate(&self, values: &[f64]) -> (f64, f64) {
        (self.expr.evaluate(values), self.rhs)
    }

    /// Check feasibility at a point.
    pub fn is_satisfied(&self, values: &[f64], tol: f64) -> bool {
        let lhs = self.expr.evaluate(values);
        self.sense.is_satisfied(lhs, self.rhs, tol)
    }

    /// Compute violation (0 if satisfied).
    pub fn violation(&self, values: &[f64]) -> f64 {
        let lhs = self.expr.evaluate(values);
        self.sense.violation(lhs, self.rhs)
    }

    /// Convert to standard form: expr ≤ rhs (flip if needed).
    pub fn to_le_form(&self) -> LinearConstraint {
        match self.sense {
            ConstraintSense::Le => self.clone(),
            ConstraintSense::Ge => LinearConstraint {
                id: self.id,
                name: self.name.clone(),
                expr: -self.expr.clone(),
                sense: ConstraintSense::Le,
                rhs: -self.rhs,
            },
            ConstraintSense::Eq => self.clone(), // caller must handle eq separately
        }
    }

    /// Convert to Ax ≤ b form, moving the constant from expr to rhs.
    pub fn normalize(&self) -> LinearConstraint {
        let adjusted_rhs = self.rhs - self.expr.constant;
        let mut new_expr = self.expr.clone();
        new_expr.constant = 0.0;
        LinearConstraint {
            id: self.id,
            name: self.name.clone(),
            expr: new_expr,
            sense: self.sense,
            rhs: adjusted_rhs,
        }
    }

    /// Variable ids appearing in this constraint.
    pub fn variable_ids(&self) -> Vec<VariableId> {
        self.expr.variable_ids()
    }

    /// Number of nonzero coefficients.
    pub fn nnz(&self) -> usize {
        self.expr.num_terms()
    }

    pub fn validate(&self) -> BicutResult<()> {
        if self.rhs.is_nan() {
            return Err(BicutError::Validation(ValidationError::StructuralError {
                detail: format!("constraint '{}' has NaN rhs", self.name),
            }));
        }
        for &(_, coeff) in &self.expr.terms {
            if coeff.is_nan() || coeff.is_infinite() {
                return Err(BicutError::Validation(ValidationError::StructuralError {
                    detail: format!("constraint '{}' has invalid coefficient", self.name),
                }));
            }
        }
        Ok(())
    }
}

impl fmt::Display for LinearConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} {} {}",
            self.name, self.expr, self.sense, self.rhs
        )
    }
}

// ── Quadratic constraint ───────────────────────────────────────────

/// A quadratic constraint: quadratic_expr sense rhs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuadraticConstraint {
    pub id: ConstraintId,
    pub name: String,
    pub expr: QuadraticExpr,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

impl QuadraticConstraint {
    pub fn new(
        id: ConstraintId,
        name: impl Into<String>,
        expr: QuadraticExpr,
        sense: ConstraintSense,
        rhs: f64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            expr,
            sense,
            rhs,
        }
    }

    pub fn evaluate(&self, values: &[f64]) -> (f64, f64) {
        (self.expr.evaluate(values), self.rhs)
    }

    pub fn is_satisfied(&self, values: &[f64], tol: f64) -> bool {
        let lhs = self.expr.evaluate(values);
        self.sense.is_satisfied(lhs, self.rhs, tol)
    }

    pub fn violation(&self, values: &[f64]) -> f64 {
        let lhs = self.expr.evaluate(values);
        self.sense.violation(lhs, self.rhs)
    }

    /// Whether the quadratic part is (diagonally) convex.
    pub fn is_convex(&self) -> bool {
        match self.sense {
            ConstraintSense::Le => self.expr.is_diagonal_convex(),
            ConstraintSense::Ge => {
                // For ≥, the negated expression should be convex
                let neg = self.expr.scaled(-1.0);
                neg.is_diagonal_convex()
            }
            ConstraintSense::Eq => false, // Equality with quadratic is generally nonconvex
        }
    }

    pub fn variable_ids(&self) -> Vec<VariableId> {
        self.expr.variable_ids()
    }

    pub fn validate(&self) -> BicutResult<()> {
        if self.rhs.is_nan() {
            return Err(BicutError::Validation(ValidationError::StructuralError {
                detail: format!("quadratic constraint '{}' has NaN rhs", self.name),
            }));
        }
        Ok(())
    }
}

impl fmt::Display for QuadraticConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} {} {}",
            self.name, self.expr, self.sense, self.rhs
        )
    }
}

// ── Indicator constraint ───────────────────────────────────────────

/// An indicator constraint: if z = val then expr sense rhs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndicatorConstraint {
    pub id: ConstraintId,
    pub name: String,
    /// The binary indicator variable.
    pub indicator_var: VariableId,
    /// The value (0 or 1) that activates the constraint.
    pub indicator_value: bool,
    /// The implied linear constraint.
    pub expr: LinearExpr,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

impl IndicatorConstraint {
    pub fn new(
        id: ConstraintId,
        name: impl Into<String>,
        indicator_var: VariableId,
        indicator_value: bool,
        expr: LinearExpr,
        sense: ConstraintSense,
        rhs: f64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            indicator_var,
            indicator_value,
            expr,
            sense,
            rhs,
        }
    }

    /// Whether the constraint is active (indicator matches).
    pub fn is_active(&self, values: &[f64], tol: f64) -> bool {
        let z = values[self.indicator_var.0];
        if self.indicator_value {
            (z - 1.0).abs() <= tol
        } else {
            z.abs() <= tol
        }
    }

    /// Check satisfaction: if inactive, trivially true; if active, check linear part.
    pub fn is_satisfied(&self, values: &[f64], tol: f64) -> bool {
        if !self.is_active(values, tol) {
            return true;
        }
        let lhs = self.expr.evaluate(values);
        self.sense.is_satisfied(lhs, self.rhs, tol)
    }

    pub fn violation(&self, values: &[f64], tol: f64) -> f64 {
        if !self.is_active(values, tol) {
            return 0.0;
        }
        let lhs = self.expr.evaluate(values);
        self.sense.violation(lhs, self.rhs)
    }

    pub fn validate(&self) -> BicutResult<()> {
        if self.rhs.is_nan() {
            return Err(BicutError::Validation(ValidationError::StructuralError {
                detail: format!("indicator constraint '{}' has NaN rhs", self.name),
            }));
        }
        Ok(())
    }
}

impl fmt::Display for IndicatorConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = if self.indicator_value { 1 } else { 0 };
        write!(
            f,
            "{}: if {} = {} then {} {} {}",
            self.name, self.indicator_var, val, self.expr, self.sense, self.rhs
        )
    }
}

// ── SOS1 constraint ────────────────────────────────────────────────

/// A Special Ordered Set type 1: at most one variable in the set is nonzero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sos1Constraint {
    pub id: ConstraintId,
    pub name: String,
    /// Variables in the SOS1 set with priority weights.
    pub members: Vec<(VariableId, f64)>,
}

impl Sos1Constraint {
    pub fn new(id: ConstraintId, name: impl Into<String>, members: Vec<(VariableId, f64)>) -> Self {
        Self {
            id,
            name: name.into(),
            members,
        }
    }

    /// Check if the SOS1 condition is satisfied: at most one member is nonzero.
    pub fn is_satisfied(&self, values: &[f64], tol: f64) -> bool {
        let nonzero_count = self
            .members
            .iter()
            .filter(|(var, _)| values[var.0].abs() > tol)
            .count();
        nonzero_count <= 1
    }

    /// Which members are nonzero?
    pub fn nonzero_members(&self, values: &[f64], tol: f64) -> Vec<VariableId> {
        self.members
            .iter()
            .filter(|(var, _)| values[var.0].abs() > tol)
            .map(|(var, _)| *var)
            .collect()
    }

    pub fn variable_ids(&self) -> Vec<VariableId> {
        self.members.iter().map(|(v, _)| *v).collect()
    }

    pub fn validate(&self) -> BicutResult<()> {
        if self.members.len() < 2 {
            return Err(BicutError::Validation(ValidationError::StructuralError {
                detail: format!(
                    "SOS1 constraint '{}' needs at least 2 members, got {}",
                    self.name,
                    self.members.len()
                ),
            }));
        }
        for &(_, weight) in &self.members {
            if weight.is_nan() || weight.is_infinite() {
                return Err(BicutError::Validation(ValidationError::StructuralError {
                    detail: format!("SOS1 constraint '{}' has invalid weight", self.name),
                }));
            }
        }
        Ok(())
    }
}

impl fmt::Display for Sos1Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: SOS1(", self.name)?;
        for (i, (var, weight)) in self.members.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}:{:.2}", var, weight)?;
        }
        write!(f, ")")
    }
}

// ── Generic constraint wrapper ─────────────────────────────────────

/// A constraint of any supported type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    Linear(LinearConstraint),
    Quadratic(QuadraticConstraint),
    Indicator(IndicatorConstraint),
    Sos1(Sos1Constraint),
}

impl Constraint {
    pub fn id(&self) -> ConstraintId {
        match self {
            Constraint::Linear(c) => c.id,
            Constraint::Quadratic(c) => c.id,
            Constraint::Indicator(c) => c.id,
            Constraint::Sos1(c) => c.id,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Constraint::Linear(c) => &c.name,
            Constraint::Quadratic(c) => &c.name,
            Constraint::Indicator(c) => &c.name,
            Constraint::Sos1(c) => &c.name,
        }
    }

    pub fn constraint_type(&self) -> ConstraintType {
        match self {
            Constraint::Linear(_) => ConstraintType::Linear,
            Constraint::Quadratic(_) => ConstraintType::Quadratic,
            Constraint::Indicator(_) => ConstraintType::Indicator,
            Constraint::Sos1(_) => ConstraintType::Sos1,
        }
    }

    pub fn is_linear(&self) -> bool {
        matches!(self, Constraint::Linear(_))
    }

    pub fn as_linear(&self) -> Option<&LinearConstraint> {
        match self {
            Constraint::Linear(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_quadratic(&self) -> Option<&QuadraticConstraint> {
        match self {
            Constraint::Quadratic(c) => Some(c),
            _ => None,
        }
    }

    pub fn variable_ids(&self) -> Vec<VariableId> {
        match self {
            Constraint::Linear(c) => c.variable_ids(),
            Constraint::Quadratic(c) => c.variable_ids(),
            Constraint::Indicator(c) => {
                let mut ids = c.expr.variable_ids();
                ids.push(c.indicator_var);
                ids.sort_by_key(|v| v.0);
                ids.dedup();
                ids
            }
            Constraint::Sos1(c) => c.variable_ids(),
        }
    }

    pub fn validate(&self) -> BicutResult<()> {
        match self {
            Constraint::Linear(c) => c.validate(),
            Constraint::Quadratic(c) => c.validate(),
            Constraint::Indicator(c) => c.validate(),
            Constraint::Sos1(c) => c.validate(),
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::Linear(c) => write!(f, "{}", c),
            Constraint::Quadratic(c) => write!(f, "{}", c),
            Constraint::Indicator(c) => write!(f, "{}", c),
            Constraint::Sos1(c) => write!(f, "{}", c),
        }
    }
}

// ── Constraint set ─────────────────────────────────────────────────

/// An ordered collection of constraints with lookup by name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintSet {
    constraints: Vec<Constraint>,
    name_index: IndexMap<String, ConstraintId>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            name_index: IndexMap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            constraints: Vec::with_capacity(cap),
            name_index: IndexMap::with_capacity(cap),
        }
    }

    /// Add a constraint (id is reassigned to the next index).
    pub fn add(&mut self, mut constraint: Constraint) -> BicutResult<ConstraintId> {
        let name = constraint.name().to_owned();
        if self.name_index.contains_key(&name) {
            return Err(BicutError::Validation(ValidationError::DuplicateName {
                name,
                context: "constraint set".into(),
            }));
        }
        let id = ConstraintId(self.constraints.len());
        // update the id inside the constraint
        match &mut constraint {
            Constraint::Linear(c) => c.id = id,
            Constraint::Quadratic(c) => c.id = id,
            Constraint::Indicator(c) => c.id = id,
            Constraint::Sos1(c) => c.id = id,
        }
        self.name_index.insert(name, id);
        self.constraints.push(constraint);
        Ok(id)
    }

    /// Convenience: add a linear constraint.
    pub fn add_linear(
        &mut self,
        name: impl Into<String>,
        expr: LinearExpr,
        sense: ConstraintSense,
        rhs: f64,
    ) -> BicutResult<ConstraintId> {
        let c = LinearConstraint::new(ConstraintId(0), name, expr, sense, rhs);
        self.add(Constraint::Linear(c))
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn get(&self, id: ConstraintId) -> Option<&Constraint> {
        self.constraints.get(id.0)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&Constraint> {
        self.name_index.get(name).and_then(|id| self.get(*id))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter()
    }

    /// Only linear constraints.
    pub fn linear_constraints(&self) -> Vec<&LinearConstraint> {
        self.constraints
            .iter()
            .filter_map(|c| c.as_linear())
            .collect()
    }

    /// Only quadratic constraints.
    pub fn quadratic_constraints(&self) -> Vec<&QuadraticConstraint> {
        self.constraints
            .iter()
            .filter_map(|c| c.as_quadratic())
            .collect()
    }

    /// Filter by constraint type.
    pub fn filter_by_type(&self, ctype: ConstraintType) -> Vec<&Constraint> {
        self.constraints
            .iter()
            .filter(|c| c.constraint_type() == ctype)
            .collect()
    }

    /// Number of linear constraints.
    pub fn num_linear(&self) -> usize {
        self.constraints.iter().filter(|c| c.is_linear()).count()
    }

    /// Total number of nonzeros across all linear constraints.
    pub fn total_nnz(&self) -> usize {
        self.constraints
            .iter()
            .filter_map(|c| c.as_linear())
            .map(|c| c.nnz())
            .sum()
    }

    /// Validate all constraints.
    pub fn validate(&self) -> BicutResult<()> {
        for c in &self.constraints {
            c.validate()?;
        }
        Ok(())
    }

    /// Check if all constraints are satisfied at a point.
    pub fn all_satisfied(&self, values: &[f64], tol: f64) -> bool {
        self.constraints.iter().all(|c| match c {
            Constraint::Linear(lc) => lc.is_satisfied(values, tol),
            Constraint::Quadratic(qc) => qc.is_satisfied(values, tol),
            Constraint::Indicator(ic) => ic.is_satisfied(values, tol),
            Constraint::Sos1(sc) => sc.is_satisfied(values, tol),
        })
    }

    /// Maximum violation over all constraints.
    pub fn max_violation(&self, values: &[f64]) -> f64 {
        self.constraints
            .iter()
            .map(|c| match c {
                Constraint::Linear(lc) => lc.violation(values),
                Constraint::Quadratic(qc) => qc.violation(values),
                Constraint::Indicator(ic) => ic.violation(values, 1e-8),
                Constraint::Sos1(_) => 0.0, // SOS1 violation is discrete
            })
            .fold(0.0_f64, f64::max)
    }
}

impl Default for ConstraintSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConstraintSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ConstraintSet ({} constraints):", self.len())?;
        for c in &self.constraints {
            writeln!(f, "  {}", c)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_expr(pairs: &[(usize, f64)]) -> LinearExpr {
        LinearExpr::from_terms(
            0.0,
            pairs.iter().map(|&(i, c)| (VariableId(i), c)).collect(),
        )
    }

    #[test]
    fn test_constraint_sense_satisfied() {
        assert!(ConstraintSense::Le.is_satisfied(3.0, 5.0, 1e-8));
        assert!(!ConstraintSense::Le.is_satisfied(6.0, 5.0, 1e-8));
        assert!(ConstraintSense::Ge.is_satisfied(5.0, 3.0, 1e-8));
        assert!(ConstraintSense::Eq.is_satisfied(5.0, 5.0, 1e-8));
        assert!(!ConstraintSense::Eq.is_satisfied(5.0, 6.0, 1e-8));
    }

    #[test]
    fn test_constraint_sense_violation() {
        assert!((ConstraintSense::Le.violation(7.0, 5.0) - 2.0).abs() < 1e-10);
        assert!((ConstraintSense::Le.violation(3.0, 5.0)).abs() < 1e-10);
        assert!((ConstraintSense::Ge.violation(3.0, 5.0) - 2.0).abs() < 1e-10);
        assert!((ConstraintSense::Eq.violation(7.0, 5.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_constraint_evaluate() {
        // 2*x0 + 3*x1 <= 10
        let expr = make_expr(&[(0, 2.0), (1, 3.0)]);
        let c = LinearConstraint::new(ConstraintId(0), "c1", expr, ConstraintSense::Le, 10.0);
        let vals = vec![1.0, 2.0]; // 2 + 6 = 8 <= 10 ✓
        assert!(c.is_satisfied(&vals, 1e-8));
        assert!((c.violation(&vals)).abs() < 1e-10);

        let vals2 = vec![3.0, 2.0]; // 6 + 6 = 12 > 10 ✗
        assert!(!c.is_satisfied(&vals2, 1e-8));
        assert!((c.violation(&vals2) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_constraint_normalize() {
        // (3 + 2*x0) <= 10  →  2*x0 <= 7
        let mut expr = make_expr(&[(0, 2.0)]);
        expr.constant = 3.0;
        let c = LinearConstraint::new(ConstraintId(0), "c1", expr, ConstraintSense::Le, 10.0);
        let norm = c.normalize();
        assert!((norm.expr.constant).abs() < 1e-10);
        assert!((norm.rhs - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_constraint() {
        // x0^2 + x1 <= 5
        let mut qe = QuadraticExpr::zero();
        qe.add_quad_term(VariableId(0), VariableId(0), 1.0);
        qe.add_linear_term(VariableId(1), 1.0);
        let c = QuadraticConstraint::new(ConstraintId(0), "qc1", qe, ConstraintSense::Le, 5.0);
        let vals = vec![1.0, 2.0]; // 1 + 2 = 3 <= 5 ✓
        assert!(c.is_satisfied(&vals, 1e-8));
        let vals2 = vec![2.0, 2.0]; // 4 + 2 = 6 > 5 ✗
        assert!(!c.is_satisfied(&vals2, 1e-8));
    }

    #[test]
    fn test_indicator_constraint() {
        // if z=1 then 2*x0 <= 5
        let expr = make_expr(&[(0, 2.0)]);
        let ic = IndicatorConstraint::new(
            ConstraintId(0),
            "ind1",
            VariableId(1),
            true,
            expr,
            ConstraintSense::Le,
            5.0,
        );
        // z=0, x0=10 → inactive, satisfied
        assert!(ic.is_satisfied(&[10.0, 0.0], 1e-8));
        // z=1, x0=2 → active, 4 <= 5 ✓
        assert!(ic.is_satisfied(&[2.0, 1.0], 1e-8));
        // z=1, x0=3 → active, 6 > 5 ✗
        assert!(!ic.is_satisfied(&[3.0, 1.0], 1e-8));
    }

    #[test]
    fn test_sos1_constraint() {
        let sos = Sos1Constraint::new(
            ConstraintId(0),
            "sos1",
            vec![
                (VariableId(0), 1.0),
                (VariableId(1), 2.0),
                (VariableId(2), 3.0),
            ],
        );
        // only x0 nonzero
        assert!(sos.is_satisfied(&[1.0, 0.0, 0.0], 1e-8));
        // two nonzero
        assert!(!sos.is_satisfied(&[1.0, 2.0, 0.0], 1e-8));
        // all zero
        assert!(sos.is_satisfied(&[0.0, 0.0, 0.0], 1e-8));
    }

    #[test]
    fn test_constraint_set_add_and_lookup() {
        let mut cs = ConstraintSet::new();
        let expr = make_expr(&[(0, 1.0)]);
        let id = cs
            .add_linear("row1", expr, ConstraintSense::Le, 5.0)
            .unwrap();
        assert_eq!(id, ConstraintId(0));
        assert_eq!(cs.len(), 1);

        let c = cs.get_by_name("row1").unwrap();
        assert_eq!(c.name(), "row1");
        assert!(c.is_linear());
    }

    #[test]
    fn test_constraint_set_duplicate_name() {
        let mut cs = ConstraintSet::new();
        let expr = make_expr(&[(0, 1.0)]);
        cs.add_linear("row1", expr.clone(), ConstraintSense::Le, 5.0)
            .unwrap();
        let result = cs.add_linear("row1", expr, ConstraintSense::Ge, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_set_all_satisfied() {
        let mut cs = ConstraintSet::new();
        cs.add_linear("c1", make_expr(&[(0, 1.0)]), ConstraintSense::Le, 5.0)
            .unwrap();
        cs.add_linear("c2", make_expr(&[(1, 1.0)]), ConstraintSense::Ge, 0.0)
            .unwrap();

        assert!(cs.all_satisfied(&[3.0, 1.0], 1e-8));
        assert!(!cs.all_satisfied(&[6.0, 1.0], 1e-8));
    }

    #[test]
    fn test_constraint_validate() {
        let expr = make_expr(&[(0, f64::NAN)]);
        let c = LinearConstraint::new(ConstraintId(0), "bad", expr, ConstraintSense::Le, 1.0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_constraint_sense_flip() {
        assert_eq!(ConstraintSense::Le.flip(), ConstraintSense::Ge);
        assert_eq!(ConstraintSense::Ge.flip(), ConstraintSense::Le);
        assert_eq!(ConstraintSense::Eq.flip(), ConstraintSense::Eq);
    }
}
