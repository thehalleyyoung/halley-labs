//! Witness validation.
//!
//! Validates generated SQL witness scripts for correctness: schema consistency,
//! predicate validity, value consistency, and structural integrity.

use std::collections::{HashMap, HashSet};
use std::fmt;

use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::{ItemId, TransactionId};
use isospec_types::isolation::AnomalyClass;
use isospec_types::operation::OpKind;
use isospec_types::schedule::{Schedule, ScheduleStep};
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

/// A single validation issue found in a witness.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub message: String,
    /// Optional step/position reference.
    pub position: Option<u64>,
    /// Optional transaction reference.
    pub transaction: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueCategory {
    Schema,
    Predicate,
    ValueConsistency,
    StructuralIntegrity,
    SqlSyntax,
    TransactionBoundary,
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sev = match self.severity {
            IssueSeverity::Error => "ERROR",
            IssueSeverity::Warning => "WARN",
            IssueSeverity::Info => "INFO",
        };
        write!(f, "[{}][{:?}] {}", sev, self.category, self.message)
    }
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// Aggregated validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
        }
    }

    pub fn add(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect()
    }

    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .collect()
    }

    pub fn is_valid(&self) -> bool {
        self.errors().is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors().len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings().len()
    }

    pub fn merge(&mut self, other: &ValidationResult) {
        self.issues.extend(other.issues.iter().cloned());
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Validation: {} errors, {} warnings, {} info",
            self.error_count(),
            self.warning_count(),
            self.issues.len() - self.error_count() - self.warning_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// WitnessValidator
// ---------------------------------------------------------------------------

/// Validates witness schedules and SQL scripts.
pub struct WitnessValidator {
    /// Known data items and their initial values.
    initial_values: HashMap<u64, Value>,
    /// Maximum allowed item ID.
    max_item_id: u64,
}

impl WitnessValidator {
    pub fn new(initial_values: HashMap<u64, Value>, max_item_id: u64) -> Self {
        Self {
            initial_values,
            max_item_id,
        }
    }

    pub fn with_defaults() -> Self {
        let mut initial = HashMap::new();
        for i in 0..8 {
            initial.insert(i, Value::Integer(0));
        }
        Self {
            initial_values: initial,
            max_item_id: 100,
        }
    }

    /// Run all validations on a schedule.
    pub fn validate_schedule(&self, schedule: &Schedule) -> ValidationResult {
        let mut result = ValidationResult::new();
        result.merge(&self.validate_structure(schedule));
        result.merge(&self.validate_item_references(schedule));
        result.merge(&self.validate_value_consistency(schedule));
        result.merge(&self.validate_transaction_boundaries(schedule));
        result
    }

    /// Validate structural integrity of the schedule.
    pub fn validate_structure(&self, schedule: &Schedule) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Must have at least 2 transactions for an anomaly
        let txn_ids: HashSet<TransactionId> = schedule
            .steps
            .iter()
            .map(|s| s.operation.txn_id)
            .collect();
        if txn_ids.len() < 2 {
            result.add(ValidationIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::StructuralIntegrity,
                message: format!(
                    "schedule has {} transaction(s), need at least 2 for an anomaly",
                    txn_ids.len()
                ),
                position: None,
                transaction: None,
            });
        }

        // All listed transactions must have operations
        for txn in &schedule.transaction_order {
            let has_ops = schedule
                .steps
                .iter()
                .any(|s| s.operation.txn_id == *txn);
            if !has_ops {
                result.add(ValidationIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::StructuralIntegrity,
                    message: format!("transaction {} listed but has no operations", txn),
                    position: None,
                    transaction: None,
                });
            }
        }

        // Positions should be unique
        let mut seen_positions: HashSet<usize> = HashSet::new();
        for step in &schedule.steps {
            if !seen_positions.insert(step.position) {
                result.add(ValidationIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::StructuralIntegrity,
                    message: format!("duplicate position {} in schedule", step.position),
                    position: Some(step.position as u64),
                    transaction: None,
                });
            }
        }

        // Operations within the same transaction should have increasing positions
        let mut txn_positions: HashMap<TransactionId, Vec<usize>> = HashMap::new();
        for step in &schedule.steps {
            txn_positions
                .entry(step.operation.txn_id)
                .or_default()
                .push(step.position);
        }
        for (txn, positions) in &txn_positions {
            let mut sorted = positions.clone();
            sorted.sort();
            for i in 0..sorted.len().saturating_sub(1) {
                if sorted[i] >= sorted[i + 1] {
                    result.add(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::StructuralIntegrity,
                        message: format!(
                            "transaction {} has non-increasing positions: {} >= {}",
                            txn,
                            sorted[i],
                            sorted[i + 1]
                        ),
                        position: Some(sorted[i] as u64),
                        transaction: None,
                    });
                }
            }
        }

        result
    }

    /// Validate that all item references are within bounds.
    pub fn validate_item_references(&self, schedule: &Schedule) -> ValidationResult {
        let mut result = ValidationResult::new();

        for step in &schedule.steps {
            let item_id_opt = extract_item_id(&step.operation.kind);
            if let Some(item_id) = item_id_opt {
                let id_val = item_id.as_u64();
                if id_val > self.max_item_id {
                    result.add(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::Schema,
                        message: format!(
                            "item ID {} exceeds maximum {} at position {}",
                            id_val, self.max_item_id, step.position
                        ),
                        position: Some(step.position as u64),
                        transaction: None,
                    });
                }
            }
        }

        result
    }

    /// Validate value consistency: reads should see previously written values.
    pub fn validate_value_consistency(&self, schedule: &Schedule) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Simulate execution to track current values
        let mut current_values: HashMap<u64, Value> = self.initial_values.clone();
        let mut sorted_steps = schedule.steps.clone();
        sorted_steps.sort_by_key(|s| s.position);

        for step in &sorted_steps {
            match &step.operation.kind {
                OpKind::Write(w) => {
                    let id = w.item.as_u64();
                    current_values.insert(id, w.new_value.clone());
                }
                OpKind::Read(r) => {
                    if let Some(ref expected) = r.value_read {
                        let id = r.item.as_u64();
                        if let Some(current) = current_values.get(&id) {
                            if current != expected {
                                result.add(ValidationIssue {
                                    severity: IssueSeverity::Warning,
                                    category: IssueCategory::ValueConsistency,
                                    message: format!(
                                        "read of item {} at position {} expects {:?} but current is {:?} (may be correct under non-serializable isolation)",
                                        id,
                                        step.position,
                                        expected,
                                        current,
                                    ),
                                    position: Some(step.position as u64),
                                    transaction: None,
                                });
                            }
                        }
                    }
                }
                OpKind::Insert(i) => {
                    let id = i.item.as_u64();
                    if current_values.contains_key(&id) {
                        result.add(ValidationIssue {
                            severity: IssueSeverity::Warning,
                            category: IssueCategory::ValueConsistency,
                            message: format!(
                                "insert on existing item {} at position {}",
                                id, step.position
                            ),
                            position: Some(step.position as u64),
                            transaction: None,
                        });
                    }
                    if let Some(val) = i.values.values().next() {
                        current_values.insert(id, val.clone());
                    }
                }
                OpKind::Delete(d) => {
                    for item_id in &d.deleted_items {
                        current_values.remove(&item_id.as_u64());
                    }
                }
                _ => {}
            }
        }

        result
    }

    /// Validate transaction boundary correctness.
    pub fn validate_transaction_boundaries(&self, schedule: &Schedule) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Group steps by transaction
        let mut txn_steps: HashMap<TransactionId, Vec<&ScheduleStep>> = HashMap::new();
        for step in &schedule.steps {
            txn_steps
                .entry(step.operation.txn_id)
                .or_default()
                .push(step);
        }

        for (txn_id, steps) in &txn_steps {
            let mut sorted = steps.clone();
            sorted.sort_by_key(|s| s.position);

            // Check that if there's a Begin, it's first
            for (idx, step) in sorted.iter().enumerate() {
                if matches!(step.operation.kind, OpKind::Begin(_)) && idx != 0 {
                    result.add(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::TransactionBoundary,
                        message: format!(
                            "BEGIN in transaction {} not at start (pos {})",
                            txn_id, step.position
                        ),
                        position: Some(step.position as u64),
                        transaction: None,
                    });
                }
            }

            // Check that Commit/Abort, if present, is last
            for (idx, step) in sorted.iter().enumerate() {
                let is_terminal = matches!(
                    step.operation.kind,
                    OpKind::Commit(_) | OpKind::Abort(_)
                );
                if is_terminal && idx != sorted.len() - 1 {
                    result.add(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::TransactionBoundary,
                        message: format!(
                            "COMMIT/ABORT in transaction {} not at end (pos {})",
                            txn_id, step.position
                        ),
                        position: Some(step.position as u64),
                        transaction: None,
                    });
                }
            }
        }

        result
    }

    /// Validate SQL scripts for basic syntax.
    pub fn validate_sql_scripts(
        &self,
        scripts: &HashMap<usize, Vec<String>>,
    ) -> ValidationResult {
        let mut result = ValidationResult::new();

        for (txn_idx, stmts) in scripts {
            for (stmt_idx, stmt) in stmts.iter().enumerate() {
                let trimmed = stmt.trim();
                if trimmed.is_empty() || trimmed.starts_with("--") {
                    continue;
                }
                // Basic SQL syntax: should end with semicolon
                if !trimmed.ends_with(';') {
                    result.add(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        category: IssueCategory::SqlSyntax,
                        message: format!(
                            "statement {} in txn {} does not end with semicolon: '{}'",
                            stmt_idx,
                            txn_idx,
                            truncate(trimmed, 50),
                        ),
                        position: None,
                        transaction: Some(*txn_idx),
                    });
                }

                // Check for common SQL keywords
                let upper = trimmed.to_uppercase();
                let has_keyword = upper.starts_with("SELECT")
                    || upper.starts_with("INSERT")
                    || upper.starts_with("UPDATE")
                    || upper.starts_with("DELETE")
                    || upper.starts_with("BEGIN")
                    || upper.starts_with("COMMIT")
                    || upper.starts_with("ROLLBACK")
                    || upper.starts_with("SET")
                    || upper.starts_with("CREATE")
                    || upper.starts_with("DROP")
                    || upper.starts_with("START")
                    || upper.starts_with("EXEC");
                if !has_keyword {
                    result.add(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        category: IssueCategory::SqlSyntax,
                        message: format!(
                            "statement {} in txn {} does not start with known SQL keyword: '{}'",
                            stmt_idx,
                            txn_idx,
                            truncate(trimmed, 50),
                        ),
                        position: None,
                        transaction: Some(*txn_idx),
                    });
                }

                // Check for balanced parentheses
                let open = trimmed.chars().filter(|c| *c == '(').count();
                let close = trimmed.chars().filter(|c| *c == ')').count();
                if open != close {
                    result.add(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::SqlSyntax,
                        message: format!(
                            "unbalanced parentheses in statement {} of txn {}: {} open, {} close",
                            stmt_idx, txn_idx, open, close
                        ),
                        position: None,
                        transaction: Some(*txn_idx),
                    });
                }
            }
        }

        result
    }
}

/// Extract the item ID from an operation kind, if applicable.
fn extract_item_id(kind: &OpKind) -> Option<ItemId> {
    match kind {
        OpKind::Read(r) => Some(r.item),
        OpKind::Write(w) => Some(w.item),
        OpKind::Insert(i) => Some(i.item),
        OpKind::Delete(d) => d.deleted_items.first().copied(),
        OpKind::Lock(l) => l.item,
        _ => None,
    }
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::OperationId;
    use isospec_types::operation::{Operation, ReadOp, WriteOp};
    use isospec_types::schedule::ScheduleMetadata;

    fn make_valid_schedule() -> Schedule {
        let t0 = TransactionId::new(0);
        let t1 = TransactionId::new(1);
        let mut s = Schedule::new();
        s.add_step(t0, Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0));
        s.add_step(t1, Operation::write(OperationId::new(1), t1, isospec_types::identifier::TableId::new(0), ItemId::new(0), Value::Integer(1), 1));
        s
    }

    #[test]
    fn test_validate_valid_schedule() {
        let validator = WitnessValidator::with_defaults();
        let schedule = make_valid_schedule();
        let result = validator.validate_schedule(&schedule);
        assert!(result.is_valid(), "expected valid: {:?}", result.issues);
    }

    #[test]
    fn test_validate_single_transaction() {
        let validator = WitnessValidator::with_defaults();
        let t0 = TransactionId::new(0);
        let mut schedule = Schedule::new();
        schedule.add_step(t0, Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0));
        let result = validator.validate_schedule(&schedule);
        assert!(!result.is_valid());
        assert!(result.errors().len() >= 1);
    }

    #[test]
    fn test_validate_duplicate_positions() {
        let validator = WitnessValidator::with_defaults();
        let t0 = TransactionId::new(0);
        let t1 = TransactionId::new(1);
        let mut schedule = Schedule::new();
        // Manually create steps with duplicate positions
        schedule.steps.push(ScheduleStep {
            id: isospec_types::identifier::ScheduleStepId::new(0),
            txn_id: t0,
            operation: Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0),
            position: 0,
        });
        schedule.steps.push(ScheduleStep {
            id: isospec_types::identifier::ScheduleStepId::new(1),
            txn_id: t1,
            operation: Operation::read(OperationId::new(1), t1, isospec_types::identifier::TableId::new(0), ItemId::new(1), 0),
            position: 0, // duplicate!
        });
        schedule.transaction_order = vec![t0, t1];
        let result = validator.validate_structure(&schedule);
        assert!(result.errors().len() >= 1);
    }

    #[test]
    fn test_validate_item_out_of_bounds() {
        let validator = WitnessValidator::new(HashMap::new(), 10);
        let t0 = TransactionId::new(0);
        let t1 = TransactionId::new(1);
        let mut schedule = Schedule::new();
        schedule.add_step(t0, Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(999), 0));
        schedule.add_step(t1, Operation::read(OperationId::new(1), t1, isospec_types::identifier::TableId::new(0), ItemId::new(0), 1));
        let result = validator.validate_item_references(&schedule);
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_validate_sql_scripts() {
        let validator = WitnessValidator::with_defaults();
        let mut scripts = HashMap::new();
        scripts.insert(
            0,
            vec![
                "SELECT val FROM t WHERE id = 0;".to_string(),
                "UPDATE t SET val = 1 WHERE id = 0;".to_string(),
            ],
        );
        let result = validator.validate_sql_scripts(&scripts);
        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_sql_missing_semicolon() {
        let validator = WitnessValidator::with_defaults();
        let mut scripts = HashMap::new();
        scripts.insert(0, vec!["SELECT 1".to_string()]);
        let result = validator.validate_sql_scripts(&scripts);
        assert!(result.warning_count() >= 1);
    }

    #[test]
    fn test_validate_sql_unbalanced_parens() {
        let validator = WitnessValidator::with_defaults();
        let mut scripts = HashMap::new();
        scripts.insert(0, vec!["SELECT (1;".to_string()]);
        let result = validator.validate_sql_scripts(&scripts);
        assert!(result.error_count() >= 1);
    }

    #[test]
    fn test_validation_result_merge() {
        let mut a = ValidationResult::new();
        a.add(ValidationIssue {
            severity: IssueSeverity::Error,
            category: IssueCategory::Schema,
            message: "test".into(),
            position: None,
            transaction: None,
        });
        let mut b = ValidationResult::new();
        b.add(ValidationIssue {
            severity: IssueSeverity::Warning,
            category: IssueCategory::SqlSyntax,
            message: "warn".into(),
            position: None,
            transaction: None,
        });
        a.merge(&b);
        assert_eq!(a.issues.len(), 2);
    }

    #[test]
    fn test_validation_result_display() {
        let result = ValidationResult::new();
        let display = format!("{}", result);
        assert!(display.contains("0 errors"));
    }

    #[test]
    fn test_truncate_helper() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
    }
}
