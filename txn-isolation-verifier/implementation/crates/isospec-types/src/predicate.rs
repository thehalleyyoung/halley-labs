//! Predicate types for SQL WHERE clause modeling.
//!
//! Implements the conjunctive inequality fragment from M5:
//! p = AND_i(col_i op_i const_i) where op in {=, !=, <, <=, >, >=}
use serde::{Deserialize, Serialize};
use crate::value::Value;
use std::fmt;

/// A predicate over database columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Predicate {
    /// Always true - matches all rows.
    True,
    /// Always false - matches no rows.
    False,
    /// Comparison: column op value.
    Comparison(ComparisonPredicate),
    /// Conjunction of predicates.
    And(Vec<Predicate>),
    /// Disjunction of predicates.
    Or(Vec<Predicate>),
    /// Negation of a predicate.
    Not(Box<Predicate>),
    /// Column IS NULL.
    IsNull(ColumnRef),
    /// Column IS NOT NULL.
    IsNotNull(ColumnRef),
    /// Column BETWEEN low AND high.
    Between(ColumnRef, Value, Value),
    /// Column IN (values).
    In(ColumnRef, Vec<Value>),
    /// EXISTS subquery (opaque, conservatively treated as universal conflict).
    Exists(String),
    /// LIKE pattern (undecidable, treated as universal conflict).
    Like(ColumnRef, String),
}

/// A comparison predicate: column op value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonPredicate {
    pub column: ColumnRef,
    pub op: ComparisonOp,
    pub value: Value,
}

/// Reference to a column, possibly qualified.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnRef {
    pub table: Option<String>,
    pub column: String,
    pub nullable: bool,
}

impl ColumnRef {
    pub fn new(column: impl Into<String>) -> Self {
        Self {
            table: None,
            column: column.into(),
            nullable: false,
        }
    }

    pub fn qualified(table: impl Into<String>, column: impl Into<String>) -> Self {
        Self {
            table: Some(table.into()),
            column: column.into(),
            nullable: false,
        }
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    pub fn full_name(&self) -> String {
        match &self.table {
            Some(t) => format!("{}.{}", t, self.column),
            None => self.column.clone(),
        }
    }
}

impl From<&str> for ColumnRef {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for ColumnRef {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.full_name())
    }
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl ComparisonOp {
    pub fn negate(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
            Self::Ge => Self::Lt,
        }
    }

    pub fn flip(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Gt => Self::Lt,
            Self::Ge => Self::Le,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }

    pub fn evaluate(&self, left: &Value, right: &Value) -> Option<bool> {
        let cmp = left.compare(right)?;
        Some(match self {
            Self::Eq => cmp == std::cmp::Ordering::Equal,
            Self::Ne => cmp != std::cmp::Ordering::Equal,
            Self::Lt => cmp == std::cmp::Ordering::Less,
            Self::Le => cmp != std::cmp::Ordering::Greater,
            Self::Gt => cmp == std::cmp::Ordering::Greater,
            Self::Ge => cmp != std::cmp::Ordering::Less,
        })
    }
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

impl Predicate {
    pub fn eq(col: impl Into<ColumnRef>, val: Value) -> Self {
        Self::Comparison(ComparisonPredicate {
            column: col.into(),
            op: ComparisonOp::Eq,
            value: val,
        })
    }

    pub fn lt(col: impl Into<ColumnRef>, val: Value) -> Self {
        Self::Comparison(ComparisonPredicate {
            column: col.into(),
            op: ComparisonOp::Lt,
            value: val,
        })
    }

    pub fn le(col: impl Into<ColumnRef>, val: Value) -> Self {
        Self::Comparison(ComparisonPredicate {
            column: col.into(),
            op: ComparisonOp::Le,
            value: val,
        })
    }

    pub fn gt(col: impl Into<ColumnRef>, val: Value) -> Self {
        Self::Comparison(ComparisonPredicate {
            column: col.into(),
            op: ComparisonOp::Gt,
            value: val,
        })
    }

    pub fn ge(col: impl Into<ColumnRef>, val: Value) -> Self {
        Self::Comparison(ComparisonPredicate {
            column: col.into(),
            op: ComparisonOp::Ge,
            value: val,
        })
    }

    pub fn between(col: impl Into<ColumnRef>, low: Value, high: Value) -> Self {
        Self::Between(col.into(), low, high)
    }

    pub fn and(preds: Vec<Predicate>) -> Self {
        let mut flattened = Vec::new();
        for p in preds {
            match p {
                Predicate::And(inner) => flattened.extend(inner),
                Predicate::True => {}
                Predicate::False => return Predicate::False,
                other => flattened.push(other),
            }
        }
        if flattened.is_empty() {
            Predicate::True
        } else if flattened.len() == 1 {
            flattened.into_iter().next().unwrap()
        } else {
            Predicate::And(flattened)
        }
    }

    pub fn or(preds: Vec<Predicate>) -> Self {
        let mut flattened = Vec::new();
        for p in preds {
            match p {
                Predicate::Or(inner) => flattened.extend(inner),
                Predicate::True => return Predicate::True,
                Predicate::False => {}
                other => flattened.push(other),
            }
        }
        if flattened.is_empty() {
            Predicate::False
        } else if flattened.len() == 1 {
            flattened.into_iter().next().unwrap()
        } else {
            Predicate::Or(flattened)
        }
    }

    pub fn not(pred: Predicate) -> Self {
        match pred {
            Predicate::True => Predicate::False,
            Predicate::False => Predicate::True,
            Predicate::Not(inner) => *inner,
            other => Predicate::Not(Box::new(other)),
        }
    }

    /// Check if this predicate is in the conjunctive inequality (CI) fragment.
    pub fn is_ci_fragment(&self) -> bool {
        match self {
            Self::True | Self::False => true,
            Self::Comparison(_) => true,
            Self::IsNull(_) | Self::IsNotNull(_) => true,
            Self::Between(_, _, _) => true,
            Self::And(preds) => preds.iter().all(|p| p.is_ci_fragment()),
            Self::Or(_) | Self::Not(_) => false,
            Self::In(_, _) => true,
            Self::Exists(_) | Self::Like(_, _) => false,
        }
    }

    /// Check if all referenced columns are NOT NULL (decidable fragment).
    pub fn all_columns_not_null(&self) -> bool {
        self.referenced_columns().iter().all(|c| !c.nullable)
    }

    /// Get all columns referenced by this predicate.
    pub fn referenced_columns(&self) -> Vec<&ColumnRef> {
        let mut cols = Vec::new();
        self.collect_columns(&mut cols);
        cols
    }

    fn collect_columns<'a>(&'a self, cols: &mut Vec<&'a ColumnRef>) {
        match self {
            Self::Comparison(c) => cols.push(&c.column),
            Self::And(preds) | Self::Or(preds) => {
                for p in preds {
                    p.collect_columns(cols);
                }
            }
            Self::Not(inner) => inner.collect_columns(cols),
            Self::IsNull(c) | Self::IsNotNull(c) => cols.push(c),
            Self::Between(c, _, _) => cols.push(c),
            Self::In(c, _) => cols.push(c),
            Self::Like(c, _) => cols.push(c),
            Self::True | Self::False | Self::Exists(_) => {}
        }
    }

    /// Evaluate this predicate against a row.
    pub fn evaluate(&self, row: &crate::value::Row) -> Option<bool> {
        match self {
            Self::True => Some(true),
            Self::False => Some(false),
            Self::Comparison(c) => {
                let col_val = row.get(&c.column.full_name())?;
                c.op.evaluate(col_val, &c.value)
            }
            Self::And(preds) => {
                let mut result = true;
                for p in preds {
                    match p.evaluate(row) {
                        Some(false) => return Some(false),
                        None => result = false,
                        Some(true) => {}
                    }
                }
                if result { Some(true) } else { None }
            }
            Self::Or(preds) => {
                let mut all_false = true;
                for p in preds {
                    match p.evaluate(row) {
                        Some(true) => return Some(true),
                        None => all_false = false,
                        Some(false) => {}
                    }
                }
                if all_false { Some(false) } else { None }
            }
            Self::Not(inner) => inner.evaluate(row).map(|b| !b),
            Self::IsNull(c) => {
                let val = row.get(&c.full_name());
                Some(val.map_or(true, |v| v.is_null()))
            }
            Self::IsNotNull(c) => {
                let val = row.get(&c.full_name());
                Some(val.map_or(false, |v| !v.is_null()))
            }
            Self::Between(c, low, high) => {
                let val = row.get(&c.full_name())?;
                let ge_low = ComparisonOp::Ge.evaluate(val, low)?;
                let le_high = ComparisonOp::Le.evaluate(val, high)?;
                Some(ge_low && le_high)
            }
            Self::In(c, vals) => {
                let col_val = row.get(&c.full_name())?;
                Some(vals.iter().any(|v| col_val == v))
            }
            Self::Exists(_) => None,
            Self::Like(_, _) => None,
        }
    }

    /// Check if two predicates may conflict (their satisfying sets overlap).
    /// Sound over-approximation: returns true when unsure.
    pub fn may_conflict_with(&self, other: &Predicate) -> bool {
        if !self.is_ci_fragment() || !other.is_ci_fragment() {
            return true;
        }
        let self_constraints = self.to_interval_constraints();
        let other_constraints = other.to_interval_constraints();
        for (col, self_interval) in &self_constraints {
            if let Some(other_interval) = other_constraints.get(col) {
                if !self_interval.overlaps(other_interval) {
                    return false;
                }
            }
        }
        true
    }

    /// Convert CI predicate to interval constraints per column.
    pub fn to_interval_constraints(&self) -> indexmap::IndexMap<String, Interval> {
        let mut intervals = indexmap::IndexMap::new();
        self.collect_intervals(&mut intervals);
        intervals
    }

    fn collect_intervals(&self, intervals: &mut indexmap::IndexMap<String, Interval>) {
        match self {
            Self::Comparison(c) => {
                let col = c.column.full_name();
                let interval = intervals.entry(col).or_insert_with(Interval::unbounded);
                interval.constrain(c.op, &c.value);
            }
            Self::And(preds) => {
                for p in preds {
                    p.collect_intervals(intervals);
                }
            }
            Self::Between(c, low, high) => {
                let col = c.full_name();
                let interval = intervals.entry(col).or_insert_with(Interval::unbounded);
                interval.constrain(ComparisonOp::Ge, low);
                interval.constrain(ComparisonOp::Le, high);
            }
            _ => {}
        }
    }

    /// Count the number of atomic predicates.
    pub fn atom_count(&self) -> usize {
        match self {
            Self::True | Self::False => 0,
            Self::Comparison(_) | Self::IsNull(_) | Self::IsNotNull(_)
            | Self::Between(_, _, _) | Self::In(_, _) | Self::Like(_, _)
            | Self::Exists(_) => 1,
            Self::And(preds) | Self::Or(preds) => preds.iter().map(|p| p.atom_count()).sum(),
            Self::Not(inner) => inner.atom_count(),
        }
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::True => write!(f, "TRUE"),
            Self::False => write!(f, "FALSE"),
            Self::Comparison(c) => write!(f, "{} {} {}", c.column, c.op, c.value),
            Self::And(preds) => {
                let parts: Vec<String> = preds.iter().map(|p| format!("{}", p)).collect();
                write!(f, "({})", parts.join(" AND "))
            }
            Self::Or(preds) => {
                let parts: Vec<String> = preds.iter().map(|p| format!("{}", p)).collect();
                write!(f, "({})", parts.join(" OR "))
            }
            Self::Not(inner) => write!(f, "NOT ({})", inner),
            Self::IsNull(c) => write!(f, "{} IS NULL", c),
            Self::IsNotNull(c) => write!(f, "{} IS NOT NULL", c),
            Self::Between(c, low, high) => write!(f, "{} BETWEEN {} AND {}", c, low, high),
            Self::In(c, vals) => {
                let parts: Vec<String> = vals.iter().map(|v| format!("{}", v)).collect();
                write!(f, "{} IN ({})", c, parts.join(", "))
            }
            Self::Exists(s) => write!(f, "EXISTS ({})", s),
            Self::Like(c, pat) => write!(f, "{} LIKE '{}'", c, pat),
        }
    }
}

/// An interval constraint on a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interval {
    pub lower: Bound,
    pub upper: Bound,
    pub excluded_values: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Bound {
    Unbounded,
    Inclusive(Value),
    Exclusive(Value),
}

impl Interval {
    pub fn unbounded() -> Self {
        Self {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
            excluded_values: Vec::new(),
        }
    }

    pub fn point(value: Value) -> Self {
        Self {
            lower: Bound::Inclusive(value.clone()),
            upper: Bound::Inclusive(value),
            excluded_values: Vec::new(),
        }
    }

    pub fn constrain(&mut self, op: ComparisonOp, value: &Value) {
        match op {
            ComparisonOp::Eq => {
                self.lower = Bound::Inclusive(value.clone());
                self.upper = Bound::Inclusive(value.clone());
            }
            ComparisonOp::Ne => {
                self.excluded_values.push(value.clone());
            }
            ComparisonOp::Lt => {
                self.upper = self.tighten_upper(Bound::Exclusive(value.clone()));
            }
            ComparisonOp::Le => {
                self.upper = self.tighten_upper(Bound::Inclusive(value.clone()));
            }
            ComparisonOp::Gt => {
                self.lower = self.tighten_lower(Bound::Exclusive(value.clone()));
            }
            ComparisonOp::Ge => {
                self.lower = self.tighten_lower(Bound::Inclusive(value.clone()));
            }
        }
    }

    fn tighten_upper(&self, new: Bound) -> Bound {
        match (&self.upper, &new) {
            (Bound::Unbounded, _) => new,
            (_, Bound::Unbounded) => self.upper.clone(),
            (Bound::Inclusive(a), Bound::Inclusive(b)) => {
                if let Some(std::cmp::Ordering::Greater) = a.compare(b) {
                    new
                } else {
                    self.upper.clone()
                }
            }
            (Bound::Exclusive(a), Bound::Exclusive(b)) => {
                if let Some(std::cmp::Ordering::Greater) = a.compare(b) {
                    new
                } else {
                    self.upper.clone()
                }
            }
            (Bound::Inclusive(a), Bound::Exclusive(b)) => {
                if let Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal) = a.compare(b) {
                    new
                } else {
                    self.upper.clone()
                }
            }
            (Bound::Exclusive(a), Bound::Inclusive(b)) => {
                if let Some(std::cmp::Ordering::Greater) = a.compare(b) {
                    new
                } else {
                    self.upper.clone()
                }
            }
        }
    }

    fn tighten_lower(&self, new: Bound) -> Bound {
        match (&self.lower, &new) {
            (Bound::Unbounded, _) => new,
            (_, Bound::Unbounded) => self.lower.clone(),
            (Bound::Inclusive(a), Bound::Inclusive(b)) => {
                if let Some(std::cmp::Ordering::Less) = a.compare(b) {
                    new
                } else {
                    self.lower.clone()
                }
            }
            (Bound::Exclusive(a), Bound::Exclusive(b)) => {
                if let Some(std::cmp::Ordering::Less) = a.compare(b) {
                    new
                } else {
                    self.lower.clone()
                }
            }
            (Bound::Inclusive(a), Bound::Exclusive(b)) => {
                if let Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal) = a.compare(b) {
                    new
                } else {
                    self.lower.clone()
                }
            }
            (Bound::Exclusive(a), Bound::Inclusive(b)) => {
                if let Some(std::cmp::Ordering::Less) = a.compare(b) {
                    new
                } else {
                    self.lower.clone()
                }
            }
        }
    }

    pub fn overlaps(&self, other: &Interval) -> bool {
        let lower_ok = match (&self.lower, &other.upper) {
            (Bound::Unbounded, _) | (_, Bound::Unbounded) => true,
            (Bound::Inclusive(a), Bound::Inclusive(b)) => {
                a.compare(b).map_or(true, |o| o != std::cmp::Ordering::Greater)
            }
            (Bound::Inclusive(a), Bound::Exclusive(b))
            | (Bound::Exclusive(a), Bound::Inclusive(b)) => {
                a.compare(b).map_or(true, |o| o == std::cmp::Ordering::Less)
            }
            (Bound::Exclusive(a), Bound::Exclusive(b)) => {
                a.compare(b).map_or(true, |o| o == std::cmp::Ordering::Less)
            }
        };
        let upper_ok = match (&other.lower, &self.upper) {
            (Bound::Unbounded, _) | (_, Bound::Unbounded) => true,
            (Bound::Inclusive(a), Bound::Inclusive(b)) => {
                a.compare(b).map_or(true, |o| o != std::cmp::Ordering::Greater)
            }
            (Bound::Inclusive(a), Bound::Exclusive(b))
            | (Bound::Exclusive(a), Bound::Inclusive(b)) => {
                a.compare(b).map_or(true, |o| o == std::cmp::Ordering::Less)
            }
            (Bound::Exclusive(a), Bound::Exclusive(b)) => {
                a.compare(b).map_or(true, |o| o == std::cmp::Ordering::Less)
            }
        };
        lower_ok && upper_ok
    }

    pub fn is_empty(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Bound::Unbounded, _) | (_, Bound::Unbounded) => false,
            (Bound::Inclusive(a), Bound::Inclusive(b)) => {
                a.compare(b).map_or(false, |o| o == std::cmp::Ordering::Greater)
            }
            (Bound::Inclusive(a), Bound::Exclusive(b))
            | (Bound::Exclusive(a), Bound::Inclusive(b)) => {
                a.compare(b).map_or(false, |o| o != std::cmp::Ordering::Less)
            }
            (Bound::Exclusive(a), Bound::Exclusive(b)) => {
                a.compare(b).map_or(false, |o| o != std::cmp::Ordering::Less)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Row, Value};

    #[test]
    fn test_simple_comparison() {
        let pred = Predicate::eq(ColumnRef::new("x"), Value::Integer(5));
        let row = Row::new().with_column("x", Value::Integer(5));
        assert_eq!(pred.evaluate(&row), Some(true));
        let row2 = Row::new().with_column("x", Value::Integer(3));
        assert_eq!(pred.evaluate(&row2), Some(false));
    }

    #[test]
    fn test_conjunction() {
        let pred = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(1)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(10)),
        ]);
        let row = Row::new().with_column("x", Value::Integer(5));
        assert_eq!(pred.evaluate(&row), Some(true));
        let row2 = Row::new().with_column("x", Value::Integer(15));
        assert_eq!(pred.evaluate(&row2), Some(false));
    }

    #[test]
    fn test_ci_fragment_detection() {
        let ci = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(1)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(10)),
        ]);
        assert!(ci.is_ci_fragment());

        let non_ci = Predicate::or(vec![
            Predicate::eq(ColumnRef::new("x"), Value::Integer(1)),
            Predicate::eq(ColumnRef::new("x"), Value::Integer(2)),
        ]);
        assert!(!non_ci.is_ci_fragment());
    }

    #[test]
    fn test_interval_overlap() {
        let i1 = Interval {
            lower: Bound::Inclusive(Value::Integer(1)),
            upper: Bound::Inclusive(Value::Integer(10)),
            excluded_values: vec![],
        };
        let i2 = Interval {
            lower: Bound::Inclusive(Value::Integer(5)),
            upper: Bound::Inclusive(Value::Integer(15)),
            excluded_values: vec![],
        };
        assert!(i1.overlaps(&i2));

        let i3 = Interval {
            lower: Bound::Exclusive(Value::Integer(10)),
            upper: Bound::Inclusive(Value::Integer(20)),
            excluded_values: vec![],
        };
        assert!(!i1.overlaps(&i3));
    }

    #[test]
    fn test_predicate_conflict() {
        let p1 = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(1)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(10)),
        ]);
        let p2 = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(5)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(15)),
        ]);
        assert!(p1.may_conflict_with(&p2));

        let p3 = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(20)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(30)),
        ]);
        assert!(!p1.may_conflict_with(&p3));
    }

    #[test]
    fn test_between_predicate() {
        let pred = Predicate::between(
            ColumnRef::new("x"),
            Value::Integer(10),
            Value::Integer(20),
        );
        let row = Row::new().with_column("x", Value::Integer(15));
        assert_eq!(pred.evaluate(&row), Some(true));
        let row2 = Row::new().with_column("x", Value::Integer(25));
        assert_eq!(pred.evaluate(&row2), Some(false));
    }

    #[test]
    fn test_predicate_simplification() {
        let p = Predicate::and(vec![Predicate::True, Predicate::eq(ColumnRef::new("x"), Value::Integer(1))]);
        match p {
            Predicate::Comparison(_) => {} // simplified to single comparison
            _ => panic!("Expected simplified predicate"),
        }
    }

    #[test]
    fn test_not_simplification() {
        let p = Predicate::not(Predicate::not(Predicate::True));
        assert!(matches!(p, Predicate::True));

        let p2 = Predicate::not(Predicate::True);
        assert!(matches!(p2, Predicate::False));
    }
}
