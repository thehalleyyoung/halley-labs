//! Mutant representation, sets, filtering, diff, and equivalence.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use shared_types::ast::Function;
use shared_types::formula::Formula;
use shared_types::operators::{
    KillInfo, MutantDescriptor, MutantId, MutantStatus, MutationOperator, MutationSite,
};

// ---------------------------------------------------------------------------
// Mutant
// ---------------------------------------------------------------------------

/// A concrete mutant: the original function plus one syntactic change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutant {
    pub id: MutantId,
    pub descriptor: MutantDescriptor,
    pub original_function: Function,
    pub mutated_function: Function,
    pub status: MutantStatus,
    pub error_predicate: Option<Formula>,
    pub kill_info: Option<KillInfo>,
    /// Operator family name, e.g. "AOR".
    pub operator_name: String,
    /// Creation timestamp (milliseconds since UNIX epoch).
    pub created_at: u64,
}

impl Mutant {
    pub fn new(
        descriptor: MutantDescriptor,
        original_function: Function,
        mutated_function: Function,
        operator_name: impl Into<String>,
    ) -> Self {
        Self {
            id: descriptor.id.clone(),
            descriptor,
            original_function,
            mutated_function,
            status: MutantStatus::Alive,
            error_predicate: None,
            kill_info: None,
            operator_name: operator_name.into(),
            created_at: now_millis(),
        }
    }

    pub fn is_killed(&self) -> bool {
        self.status == MutantStatus::Killed
    }

    pub fn is_alive(&self) -> bool {
        self.status == MutantStatus::Alive
    }

    pub fn is_equivalent(&self) -> bool {
        self.status == MutantStatus::Equivalent
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            MutantStatus::Killed | MutantStatus::Equivalent | MutantStatus::Error(_)
        )
    }

    pub fn mark_killed(&mut self, kill_info: KillInfo) {
        self.status = MutantStatus::Killed;
        self.kill_info = Some(kill_info);
    }

    pub fn mark_equivalent(&mut self) {
        self.status = MutantStatus::Equivalent;
    }

    pub fn mark_timeout(&mut self) {
        self.status = MutantStatus::Timeout;
    }

    pub fn mark_error(&mut self) {
        self.status = MutantStatus::Error(String::new());
    }

    pub fn set_error_predicate(&mut self, pred: Formula) {
        self.error_predicate = Some(pred);
    }

    /// Compute the syntactic diff between original and mutated function.
    pub fn diff(&self) -> MutantDiff {
        MutantDiff::compute(&self.original_function, &self.mutated_function)
    }

    /// Check syntactic equivalence with another mutant (same mutation applied).
    pub fn syntactically_equivalent(&self, other: &Mutant) -> bool {
        self.mutated_function == other.mutated_function
    }
}

impl fmt::Display for Mutant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mutant[{}] ({}) – {} – {}",
            self.id,
            self.operator_name,
            self.descriptor.site.summary(),
            self.status
        )
    }
}

// ---------------------------------------------------------------------------
// MutantDiff
// ---------------------------------------------------------------------------

/// Records what changed between original and mutated functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutantDiff {
    /// Index of the first statement that differs.
    pub changed_stmt_indices: Vec<usize>,
    /// Human-readable summary.
    pub summary: String,
    /// Original expression (serialised).
    pub original_expr: Option<String>,
    /// Mutated expression (serialised).
    pub mutated_expr: Option<String>,
}

impl MutantDiff {
    pub fn compute(original: &Function, mutated: &Function) -> Self {
        let changed = if original.body != mutated.body {
            vec![0]
        } else {
            Vec::new()
        };

        let summary = if changed.is_empty() {
            "No changes detected".into()
        } else {
            format!("{} statement(s) differ", changed.len())
        };

        Self {
            changed_stmt_indices: changed,
            summary,
            original_expr: None,
            mutated_expr: None,
        }
    }

    pub fn with_exprs(mut self, original: &str, mutated: &str) -> Self {
        self.original_expr = Some(original.into());
        self.mutated_expr = Some(mutated.into());
        self
    }

    pub fn is_empty(&self) -> bool {
        self.changed_stmt_indices.is_empty()
    }

    pub fn num_changes(&self) -> usize {
        self.changed_stmt_indices.len()
    }
}

impl fmt::Display for MutantDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary)?;
        if let (Some(orig), Some(mutd)) = (&self.original_expr, &self.mutated_expr) {
            write!(f, " ({} → {})", orig, mutd)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MutantFilter
// ---------------------------------------------------------------------------

/// Trait for filtering mutants.
pub trait MutantFilter: Send + Sync {
    fn matches(&self, mutant: &Mutant) -> bool;
    fn description(&self) -> String;
}

/// Filter by operator name.
#[derive(Debug, Clone)]
pub struct ByOperator {
    pub operator_name: String,
}

impl ByOperator {
    pub fn new(op: impl Into<String>) -> Self {
        Self {
            operator_name: op.into(),
        }
    }
}

impl MutantFilter for ByOperator {
    fn matches(&self, mutant: &Mutant) -> bool {
        mutant.operator_name == self.operator_name
    }

    fn description(&self) -> String {
        format!("operator == {}", self.operator_name)
    }
}

/// Filter by status.
#[derive(Debug, Clone)]
pub struct ByStatus {
    pub status: MutantStatus,
}

impl ByStatus {
    pub fn new(status: MutantStatus) -> Self {
        Self { status }
    }
}

impl MutantFilter for ByStatus {
    fn matches(&self, mutant: &Mutant) -> bool {
        mutant.status == self.status
    }

    fn description(&self) -> String {
        format!("status == {}", self.status)
    }
}

/// Filter by function name.
#[derive(Debug, Clone)]
pub struct ByFunction {
    pub function_name: String,
}

impl ByFunction {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            function_name: name.into(),
        }
    }
}

impl MutantFilter for ByFunction {
    fn matches(&self, mutant: &Mutant) -> bool {
        mutant.original_function.name == self.function_name
    }

    fn description(&self) -> String {
        format!("function == {}", self.function_name)
    }
}

/// Composite AND filter.
pub struct AndFilter {
    filters: Vec<Box<dyn MutantFilter>>,
}

impl AndFilter {
    pub fn new(filters: Vec<Box<dyn MutantFilter>>) -> Self {
        Self { filters }
    }
}

impl MutantFilter for AndFilter {
    fn matches(&self, mutant: &Mutant) -> bool {
        self.filters.iter().all(|f| f.matches(mutant))
    }

    fn description(&self) -> String {
        let descs: Vec<_> = self.filters.iter().map(|f| f.description()).collect();
        format!("({})", descs.join(" AND "))
    }
}

/// Composite OR filter.
pub struct OrFilter {
    filters: Vec<Box<dyn MutantFilter>>,
}

impl OrFilter {
    pub fn new(filters: Vec<Box<dyn MutantFilter>>) -> Self {
        Self { filters }
    }
}

impl MutantFilter for OrFilter {
    fn matches(&self, mutant: &Mutant) -> bool {
        self.filters.iter().any(|f| f.matches(mutant))
    }

    fn description(&self) -> String {
        let descs: Vec<_> = self.filters.iter().map(|f| f.description()).collect();
        format!("({})", descs.join(" OR "))
    }
}

// ---------------------------------------------------------------------------
// MutantSet
// ---------------------------------------------------------------------------

/// A collection of mutants with indexing by operator, function, and status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MutantSet {
    mutants: Vec<Mutant>,
    /// Index: operator_name → Vec of positions in `mutants`.
    by_operator: HashMap<String, Vec<usize>>,
    /// Index: function_name → Vec of positions in `mutants`.
    by_function: HashMap<String, Vec<usize>>,
}

impl MutantSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, mutant: Mutant) {
        let idx = self.mutants.len();
        self.by_operator
            .entry(mutant.operator_name.clone())
            .or_default()
            .push(idx);
        self.by_function
            .entry(mutant.original_function.name.clone())
            .or_default()
            .push(idx);
        self.mutants.push(mutant);
    }

    pub fn len(&self) -> usize {
        self.mutants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mutants.is_empty()
    }

    pub fn get(&self, id: &MutantId) -> Option<&Mutant> {
        self.mutants.iter().find(|m| m.id == *id)
    }

    pub fn get_mut(&mut self, id: &MutantId) -> Option<&mut Mutant> {
        self.mutants.iter_mut().find(|m| m.id == *id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Mutant> {
        self.mutants.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Mutant> {
        self.mutants.iter_mut()
    }

    pub fn all(&self) -> &[Mutant] {
        &self.mutants
    }

    /// Get mutants by operator name.
    pub fn by_operator(&self, name: &str) -> Vec<&Mutant> {
        self.by_operator
            .get(name)
            .map(|indices| indices.iter().map(|&i| &self.mutants[i]).collect())
            .unwrap_or_default()
    }

    /// Get mutants for a specific function.
    pub fn by_function(&self, name: &str) -> Vec<&Mutant> {
        self.by_function
            .get(name)
            .map(|indices| indices.iter().map(|&i| &self.mutants[i]).collect())
            .unwrap_or_default()
    }

    /// Filter mutants using a MutantFilter.
    pub fn filter(&self, filter: &dyn MutantFilter) -> Vec<&Mutant> {
        self.mutants.iter().filter(|m| filter.matches(m)).collect()
    }

    /// Get all alive mutants.
    pub fn alive(&self) -> Vec<&Mutant> {
        self.mutants.iter().filter(|m| m.is_alive()).collect()
    }

    /// Get all killed mutants.
    pub fn killed(&self) -> Vec<&Mutant> {
        self.mutants.iter().filter(|m| m.is_killed()).collect()
    }

    /// Get all equivalent mutants.
    pub fn equivalent(&self) -> Vec<&Mutant> {
        self.mutants.iter().filter(|m| m.is_equivalent()).collect()
    }

    /// Compute basic statistics.
    pub fn stats(&self) -> MutantSetStats {
        let total = self.mutants.len();
        let killed = self.killed().len();
        let alive = self.alive().len();
        let equivalent = self.equivalent().len();
        let timed_out = self
            .mutants
            .iter()
            .filter(|m| m.status == MutantStatus::Timeout)
            .count();
        let errored = self.mutants.iter().filter(|m| m.status.is_error()).count();

        let denom = total.saturating_sub(equivalent);
        let mutation_score = if denom == 0 {
            1.0
        } else {
            killed as f64 / denom as f64
        };

        let mut by_operator = HashMap::new();
        for (op, indices) in &self.by_operator {
            let op_killed = indices
                .iter()
                .filter(|&&i| self.mutants[i].is_killed())
                .count();
            let op_total = indices.len();
            by_operator.insert(op.clone(), (op_killed, op_total));
        }

        MutantSetStats {
            total,
            killed,
            alive,
            equivalent,
            timed_out,
            errored,
            mutation_score,
            by_operator,
        }
    }

    /// Remove duplicate mutants (same mutated function body).
    pub fn deduplicate(&mut self) {
        let mut seen = std::collections::HashSet::new();
        let mut keep = Vec::new();

        for (i, m) in self.mutants.iter().enumerate() {
            let key = format!("{:?}", m.mutated_function.body);
            if seen.insert(key) {
                keep.push(i);
            }
        }

        let new_mutants: Vec<Mutant> = keep.iter().map(|&i| self.mutants[i].clone()).collect();
        *self = MutantSet::new();
        for m in new_mutants {
            self.add(m);
        }
    }

    /// Serialise to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.mutants)
    }

    /// Deserialise from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let mutants: Vec<Mutant> = serde_json::from_str(json)?;
        let mut set = MutantSet::new();
        for m in mutants {
            set.add(m);
        }
        Ok(set)
    }

    /// Merge another MutantSet into this one.
    pub fn merge(&mut self, other: MutantSet) {
        for m in other.mutants {
            self.add(m);
        }
    }
}

impl fmt::Display for MutantSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        write!(
            f,
            "MutantSet({} total, {} killed, {} alive, {} equiv, score={:.1}%)",
            stats.total,
            stats.killed,
            stats.alive,
            stats.equivalent,
            stats.mutation_score * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// MutantSetStats
// ---------------------------------------------------------------------------

/// Summary statistics for a MutantSet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutantSetStats {
    pub total: usize,
    pub killed: usize,
    pub alive: usize,
    pub equivalent: usize,
    pub timed_out: usize,
    pub errored: usize,
    pub mutation_score: f64,
    pub by_operator: HashMap<String, (usize, usize)>, // (killed, total)
}

impl fmt::Display for MutantSetStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Mutation Testing Statistics:")?;
        writeln!(f, "  Total mutants:   {}", self.total)?;
        writeln!(f, "  Killed:          {}", self.killed)?;
        writeln!(f, "  Alive:           {}", self.alive)?;
        writeln!(f, "  Equivalent:      {}", self.equivalent)?;
        writeln!(f, "  Timed out:       {}", self.timed_out)?;
        writeln!(f, "  Errors:          {}", self.errored)?;
        writeln!(f, "  Mutation score:  {:.1}%", self.mutation_score * 100.0)?;
        if !self.by_operator.is_empty() {
            writeln!(f, "  By operator:")?;
            let mut ops: Vec<_> = self.by_operator.iter().collect();
            ops.sort_by_key(|(k, _)| k.clone());
            for (op, (killed, total)) in ops {
                let score = if *total == 0 {
                    0.0
                } else {
                    *killed as f64 / *total as f64 * 100.0
                };
                writeln!(f, "    {}: {}/{} ({:.1}%)", op, killed, total, score)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::ast::{ArithOp, Expression, Statement};
    use shared_types::types::*;
    use shared_types::SpanInfo;

    fn dummy_site() -> MutationSite {
        MutationSite {
            location: SpanInfo::unknown(),
            operator: MutationOperator::Aor,
            original: "+".into(),
            replacement: "-".into(),
            function_name: None,
            expression_type: None,
        }
    }

    fn dummy_descriptor() -> MutantDescriptor {
        MutantDescriptor {
            id: MutantId::new(),
            operator: MutationOperator::Aor,
            site: dummy_site(),
            status: MutantStatus::Alive,
            kill_info: None,
            equivalent_reason: None,
        }
    }

    fn dummy_func(name: &str) -> Function {
        Function::new(
            name,
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::ret(Some(Expression::BinaryArith {
                op: ArithOp::Add,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::IntLiteral(1)),
            })),
        )
    }

    fn dummy_mutated_func(name: &str) -> Function {
        Function::new(
            name,
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::ret(Some(Expression::BinaryArith {
                op: ArithOp::Sub,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::IntLiteral(1)),
            })),
        )
    }

    fn make_mutant(_id_str: &str, op_name: &str, func_name: &str) -> Mutant {
        let desc = MutantDescriptor {
            id: MutantId::new(),
            operator: MutationOperator::Aor,
            site: dummy_site(),
            status: MutantStatus::Alive,
            kill_info: None,
            equivalent_reason: None,
        };
        Mutant::new(
            desc,
            dummy_func(func_name),
            dummy_mutated_func(func_name),
            op_name,
        )
    }

    #[test]
    fn test_mutant_creation() {
        let m = make_mutant("m1", "AOR", "f");
        assert!(!m.id.0.is_nil());
        assert_eq!(m.operator_name, "AOR");
        assert!(m.is_alive());
        assert!(!m.is_killed());
    }

    #[test]
    fn test_mutant_mark_killed() {
        let mut m = make_mutant("m1", "AOR", "f");
        m.mark_killed(KillInfo::new("test_1", 0.0));
        assert!(m.is_killed());
        assert!(!m.is_alive());
    }

    #[test]
    fn test_mutant_mark_equivalent() {
        let mut m = make_mutant("m1", "AOR", "f");
        m.mark_equivalent();
        assert!(m.is_equivalent());
    }

    #[test]
    fn test_mutant_diff() {
        let m = make_mutant("m1", "AOR", "f");
        let diff = m.diff();
        assert!(!diff.is_empty());
        assert_eq!(diff.num_changes(), 1);
    }

    #[test]
    fn test_mutant_display() {
        let m = make_mutant("m1", "AOR", "f");
        let s = format!("{}", m);
        assert!(s.contains("AOR"));
        assert!(s.contains("m1"));
    }

    #[test]
    fn test_mutant_set_basic() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        set.add(make_mutant("m2", "ROR", "f"));
        set.add(make_mutant("m3", "AOR", "g"));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_mutant_set_by_operator() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        set.add(make_mutant("m2", "ROR", "f"));
        set.add(make_mutant("m3", "AOR", "g"));
        assert_eq!(set.by_operator("AOR").len(), 2);
        assert_eq!(set.by_operator("ROR").len(), 1);
        assert_eq!(set.by_operator("LCR").len(), 0);
    }

    #[test]
    fn test_mutant_set_by_function() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        set.add(make_mutant("m2", "ROR", "f"));
        set.add(make_mutant("m3", "AOR", "g"));
        assert_eq!(set.by_function("f").len(), 2);
        assert_eq!(set.by_function("g").len(), 1);
    }

    #[test]
    fn test_mutant_set_filter() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        set.add(make_mutant("m2", "ROR", "f"));
        let filter = ByOperator::new("AOR");
        assert_eq!(set.filter(&filter).len(), 1);
    }

    #[test]
    fn test_mutant_set_stats() {
        let mut set = MutantSet::new();
        let mut m1 = make_mutant("m1", "AOR", "f");
        m1.mark_killed(KillInfo::new("t1", 0.0));
        let mut m2 = make_mutant("m2", "ROR", "f");
        m2.mark_equivalent();
        set.add(m1);
        set.add(m2);
        set.add(make_mutant("m3", "AOR", "f"));

        let stats = set.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.killed, 1);
        assert_eq!(stats.equivalent, 1);
        // score = 1 / (3 - 1) = 0.5
        assert!((stats.mutation_score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_mutant_set_alive_killed() {
        let mut set = MutantSet::new();
        let mut m1 = make_mutant("m1", "AOR", "f");
        m1.mark_killed(KillInfo::new("t1", 0.0));
        set.add(m1);
        set.add(make_mutant("m2", "AOR", "f"));

        assert_eq!(set.alive().len(), 1);
        assert_eq!(set.killed().len(), 1);
    }

    #[test]
    fn test_mutant_set_json_roundtrip() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        set.add(make_mutant("m2", "ROR", "g"));

        let json = set.to_json().unwrap();
        let restored = MutantSet::from_json(&json).unwrap();
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn test_mutant_set_merge() {
        let mut set1 = MutantSet::new();
        set1.add(make_mutant("m1", "AOR", "f"));
        let mut set2 = MutantSet::new();
        set2.add(make_mutant("m2", "ROR", "g"));
        set1.merge(set2);
        assert_eq!(set1.len(), 2);
    }

    #[test]
    fn test_mutant_diff_no_changes() {
        let f = dummy_func("f");
        let diff = MutantDiff::compute(&f, &f);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_mutant_diff_display() {
        let diff = MutantDiff {
            changed_stmt_indices: vec![0],
            summary: "1 change".into(),
            original_expr: Some("x + y".into()),
            mutated_expr: Some("x - y".into()),
        };
        let s = format!("{}", diff);
        assert!(s.contains("x + y"));
        assert!(s.contains("x - y"));
    }

    #[test]
    fn test_by_operator_filter() {
        let f = ByOperator::new("AOR");
        let m = make_mutant("m1", "AOR", "f");
        assert!(f.matches(&m));
        let m2 = make_mutant("m2", "ROR", "f");
        assert!(!f.matches(&m2));
    }

    #[test]
    fn test_by_status_filter() {
        let f = ByStatus::new(MutantStatus::Alive);
        let m = make_mutant("m1", "AOR", "f");
        assert!(f.matches(&m));
    }

    #[test]
    fn test_by_function_filter() {
        let f = ByFunction::new("f");
        let m = make_mutant("m1", "AOR", "f");
        assert!(f.matches(&m));
        let m2 = make_mutant("m2", "AOR", "g");
        assert!(!f.matches(&m2));
    }

    #[test]
    fn test_and_filter() {
        let f = AndFilter::new(vec![
            Box::new(ByOperator::new("AOR")),
            Box::new(ByFunction::new("f")),
        ]);
        assert!(f.matches(&make_mutant("m1", "AOR", "f")));
        assert!(!f.matches(&make_mutant("m2", "ROR", "f")));
        assert!(!f.matches(&make_mutant("m3", "AOR", "g")));
    }

    #[test]
    fn test_or_filter() {
        let f = OrFilter::new(vec![
            Box::new(ByOperator::new("AOR")),
            Box::new(ByOperator::new("ROR")),
        ]);
        assert!(f.matches(&make_mutant("m1", "AOR", "f")));
        assert!(f.matches(&make_mutant("m2", "ROR", "f")));
        assert!(!f.matches(&make_mutant("m3", "LCR", "f")));
    }

    #[test]
    fn test_mutant_set_deduplicate() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        set.add(make_mutant("m2", "AOR", "f")); // same mutated body
        set.deduplicate();
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_syntactically_equivalent() {
        let m1 = make_mutant("m1", "AOR", "f");
        let m2 = make_mutant("m2", "AOR", "f");
        assert!(m1.syntactically_equivalent(&m2));
    }

    #[test]
    fn test_mutant_set_get() {
        let mut set = MutantSet::new();
        let m = make_mutant("m1", "AOR", "f");
        let id = m.id.clone();
        set.add(m);
        assert!(set.get(&id).is_some());
        assert!(set.get(&MutantId::new()).is_none());
    }

    #[test]
    fn test_mutant_set_display() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        let s = format!("{}", set);
        assert!(s.contains("MutantSet"));
    }

    #[test]
    fn test_stats_display() {
        let mut set = MutantSet::new();
        set.add(make_mutant("m1", "AOR", "f"));
        let stats = set.stats();
        let s = format!("{}", stats);
        assert!(s.contains("Mutation Testing Statistics"));
    }
}
