//! Predicate-level conflict theory (M5).
//!
//! Extends Adya's item-level DSG theory to SQL WHERE clauses, range scans,
//! INSERT/DELETE on predicate-defined sets.
use isospec_types::predicate::*;
use isospec_types::value::Value;
use isospec_types::constraint::*;
use isospec_types::error::IsoSpecResult;
use std::collections::HashMap;

/// Result of predicate conflict analysis.
#[derive(Debug, Clone)]
pub enum ConflictResult {
    NoConflict,
    Conflict(ConflictInfo),
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ConflictInfo {
    pub conflicting_columns: Vec<String>,
    pub overlap_description: String,
    pub is_exact: bool,
}

/// Predicate conflict detector implementing M5 theory.
pub struct PredicateConflictDetector {
    three_valued_logic: bool,
    cache: HashMap<(u64, u64), ConflictResult>,
}

impl PredicateConflictDetector {
    pub fn new(three_valued_logic: bool) -> Self {
        Self { three_valued_logic, cache: HashMap::new() }
    }

    /// Check if two predicates conflict (their satisfying sets overlap).
    pub fn check_conflict(&mut self, p1: &Predicate, p2: &Predicate) -> ConflictResult {
        if !p1.is_ci_fragment() || !p2.is_ci_fragment() {
            return ConflictResult::Unknown;
        }

        let intervals1 = p1.to_interval_constraints();
        let intervals2 = p2.to_interval_constraints();

        if self.three_valued_logic {
            let has_nullable = p1.referenced_columns().iter().any(|c| c.nullable)
                || p2.referenced_columns().iter().any(|c| c.nullable);
            if has_nullable {
                return self.check_conflict_three_valued(&intervals1, &intervals2);
            }
        }

        self.check_conflict_two_valued(&intervals1, &intervals2)
    }

    fn check_conflict_two_valued(
        &self,
        i1: &indexmap::IndexMap<String, Interval>,
        i2: &indexmap::IndexMap<String, Interval>,
    ) -> ConflictResult {
        let mut conflicting_cols = Vec::new();
        for (col, interval1) in i1 {
            if let Some(interval2) = i2.get(col) {
                if !interval1.overlaps(interval2) {
                    return ConflictResult::NoConflict;
                }
                conflicting_cols.push(col.clone());
            }
        }
        if conflicting_cols.is_empty() {
            ConflictResult::Conflict(ConflictInfo {
                conflicting_columns: vec!["*".to_string()],
                overlap_description: "No common columns constrained - potential full overlap".into(),
                is_exact: false,
            })
        } else {
            ConflictResult::Conflict(ConflictInfo {
                conflicting_columns: conflicting_cols,
                overlap_description: "Interval overlap detected".into(),
                is_exact: true,
            })
        }
    }

    fn check_conflict_three_valued(
        &self,
        i1: &indexmap::IndexMap<String, Interval>,
        i2: &indexmap::IndexMap<String, Interval>,
    ) -> ConflictResult {
        // Under 3VL, NULL-possible conflicts conservatively resolve to CONFLICT
        for (col, interval1) in i1 {
            if let Some(interval2) = i2.get(col) {
                if !interval1.overlaps(interval2) {
                    // Even with NULLs, non-overlapping ranges on NOT NULL cols are safe
                    return ConflictResult::NoConflict;
                }
            }
        }
        ConflictResult::Conflict(ConflictInfo {
            conflicting_columns: i1.keys().cloned().collect(),
            overlap_description: "3VL conservative conflict resolution".into(),
            is_exact: false,
        })
    }

    /// Encode predicate conflict as SMT constraint.
    pub fn encode_conflict_smt(
        &self,
        p1: &Predicate,
        p2: &Predicate,
        prefix: &str,
    ) -> SmtConstraintSet {
        let mut cs = SmtConstraintSet::new("QF_LIA");
        let columns = self.collect_all_columns(p1, p2);

        for (i, col) in columns.iter().enumerate() {
            let var_name = format!("{}_{}", prefix, col.replace('.', "_"));
            cs.declare(&var_name, SmtSort::Int);
        }

        let p1_constraints = self.predicate_to_smt(p1, prefix);
        let p2_constraints = self.predicate_to_smt(p2, prefix);

        cs.assert(SmtExpr::and(vec![p1_constraints, p2_constraints]));
        cs
    }

    fn collect_all_columns(&self, p1: &Predicate, p2: &Predicate) -> Vec<String> {
        let mut cols: Vec<String> = Vec::new();
        for c in p1.referenced_columns().into_iter().chain(p2.referenced_columns()) {
            let name = c.full_name();
            if !cols.contains(&name) {
                cols.push(name);
            }
        }
        cols
    }

    fn predicate_to_smt(&self, pred: &Predicate, prefix: &str) -> SmtExpr {
        match pred {
            Predicate::True => SmtExpr::BoolLit(true),
            Predicate::False => SmtExpr::BoolLit(false),
            Predicate::Comparison(c) => {
                let var = SmtExpr::int_var(format!("{}_{}", prefix, c.column.full_name().replace('.', "_")));
                let val = match &c.value {
                    Value::Integer(i) => SmtExpr::IntLit(*i),
                    Value::Float(f) => SmtExpr::IntLit(*f as i64),
                    _ => SmtExpr::IntLit(0),
                };
                match c.op {
                    ComparisonOp::Eq => SmtExpr::eq(var, val),
                    ComparisonOp::Ne => SmtExpr::not(SmtExpr::eq(var, val)),
                    ComparisonOp::Lt => SmtExpr::lt(var, val),
                    ComparisonOp::Le => SmtExpr::le(var, val),
                    ComparisonOp::Gt => SmtExpr::Gt(Box::new(var), Box::new(val)),
                    ComparisonOp::Ge => SmtExpr::Ge(Box::new(var), Box::new(val)),
                }
            }
            Predicate::And(preds) => {
                SmtExpr::and(preds.iter().map(|p| self.predicate_to_smt(p, prefix)).collect())
            }
            Predicate::Or(preds) => {
                SmtExpr::or(preds.iter().map(|p| self.predicate_to_smt(p, prefix)).collect())
            }
            Predicate::Not(inner) => SmtExpr::not(self.predicate_to_smt(inner, prefix)),
            Predicate::Between(col, low, high) => {
                let var = SmtExpr::int_var(format!("{}_{}", prefix, col.full_name().replace('.', "_")));
                let low_val = match low { Value::Integer(i) => SmtExpr::IntLit(*i), _ => SmtExpr::IntLit(0) };
                let high_val = match high { Value::Integer(i) => SmtExpr::IntLit(*i), _ => SmtExpr::IntLit(0) };
                SmtExpr::and(vec![
                    SmtExpr::Ge(Box::new(var.clone()), Box::new(low_val)),
                    SmtExpr::Le(Box::new(var), Box::new(high_val)),
                ])
            }
            _ => SmtExpr::BoolLit(true), // over-approximate undecidable predicates
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Predicate footprint analysis for a transaction.
#[derive(Debug, Clone)]
pub struct PredicateFootprint {
    pub read_predicates: Vec<(String, Predicate)>,
    pub write_predicates: Vec<(String, Predicate)>,
    pub insert_tables: Vec<String>,
    pub delete_predicates: Vec<(String, Predicate)>,
}

impl PredicateFootprint {
    pub fn new() -> Self {
        Self {
            read_predicates: Vec::new(),
            write_predicates: Vec::new(),
            insert_tables: Vec::new(),
            delete_predicates: Vec::new(),
        }
    }

    pub fn add_read(&mut self, table: impl Into<String>, pred: Predicate) {
        self.read_predicates.push((table.into(), pred));
    }

    pub fn add_write(&mut self, table: impl Into<String>, pred: Predicate) {
        self.write_predicates.push((table.into(), pred));
    }

    pub fn add_insert(&mut self, table: impl Into<String>) {
        self.insert_tables.push(table.into());
    }

    pub fn add_delete(&mut self, table: impl Into<String>, pred: Predicate) {
        self.delete_predicates.push((table.into(), pred));
    }

    pub fn tables_touched(&self) -> Vec<String> {
        let mut tables = Vec::new();
        for (t, _) in &self.read_predicates {
            if !tables.contains(t) { tables.push(t.clone()); }
        }
        for (t, _) in &self.write_predicates {
            if !tables.contains(t) { tables.push(t.clone()); }
        }
        for t in &self.insert_tables {
            if !tables.contains(t) { tables.push(t.clone()); }
        }
        for (t, _) in &self.delete_predicates {
            if !tables.contains(t) { tables.push(t.clone()); }
        }
        tables
    }

    pub fn conflicts_with(&self, other: &PredicateFootprint) -> bool {
        let mut detector = PredicateConflictDetector::new(true);
        // Check write-write conflicts
        for (t1, p1) in &self.write_predicates {
            for (t2, p2) in &other.write_predicates {
                if t1 == t2 {
                    if let ConflictResult::Conflict(_) = detector.check_conflict(p1, p2) {
                        return true;
                    }
                }
            }
        }
        // Check write-read conflicts
        for (t1, p1) in &self.write_predicates {
            for (t2, p2) in &other.read_predicates {
                if t1 == t2 {
                    if let ConflictResult::Conflict(_) = detector.check_conflict(p1, p2) {
                        return true;
                    }
                }
            }
        }
        for (t1, p1) in &self.read_predicates {
            for (t2, p2) in &other.write_predicates {
                if t1 == t2 {
                    if let ConflictResult::Conflict(_) = detector.check_conflict(p1, p2) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Default for PredicateFootprint {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::predicate::ColumnRef;

    #[test]
    fn test_no_conflict() {
        let mut det = PredicateConflictDetector::new(false);
        let p1 = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(1)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(10)),
        ]);
        let p2 = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(20)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(30)),
        ]);
        assert!(matches!(det.check_conflict(&p1, &p2), ConflictResult::NoConflict));
    }

    #[test]
    fn test_conflict_detected() {
        let mut det = PredicateConflictDetector::new(false);
        let p1 = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(1)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(10)),
        ]);
        let p2 = Predicate::and(vec![
            Predicate::ge(ColumnRef::new("x"), Value::Integer(5)),
            Predicate::le(ColumnRef::new("x"), Value::Integer(15)),
        ]);
        assert!(matches!(det.check_conflict(&p1, &p2), ConflictResult::Conflict(_)));
    }

    #[test]
    fn test_smt_encoding() {
        let det = PredicateConflictDetector::new(false);
        let p1 = Predicate::ge(ColumnRef::new("x"), Value::Integer(1));
        let p2 = Predicate::le(ColumnRef::new("x"), Value::Integer(10));
        let cs = det.encode_conflict_smt(&p1, &p2, "test");
        assert!(cs.variable_count() > 0);
        assert!(cs.constraint_count() > 0);
    }

    #[test]
    fn test_predicate_footprint() {
        let mut fp1 = PredicateFootprint::new();
        fp1.add_read("t", Predicate::ge(ColumnRef::new("x"), Value::Integer(1)));
        fp1.add_write("t", Predicate::eq(ColumnRef::new("x"), Value::Integer(5)));

        let mut fp2 = PredicateFootprint::new();
        fp2.add_read("t", Predicate::le(ColumnRef::new("x"), Value::Integer(10)));

        assert!(fp1.conflicts_with(&fp2));
    }

    #[test]
    fn test_unknown_for_non_ci() {
        let mut det = PredicateConflictDetector::new(false);
        let p1 = Predicate::Like(ColumnRef::new("name"), "%test%".into());
        let p2 = Predicate::eq(ColumnRef::new("x"), Value::Integer(1));
        assert!(matches!(det.check_conflict(&p1, &p2), ConflictResult::Unknown));
    }
}
