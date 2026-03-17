//! Core traits for engine models.
use isospec_types::identifier::TransactionId;
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::Operation;
use isospec_types::schedule::Schedule;
use isospec_types::config::EngineKind;
use isospec_types::dependency::Dependency;
use isospec_types::constraint::SmtConstraintSet;
use isospec_types::error::IsoSpecResult;
use std::collections::HashMap;

/// A Labeled Transition System (LTS) for an engine's concurrency control.
pub trait EngineModel: Send + Sync {
    fn engine_kind(&self) -> EngineKind;
    fn supported_isolation_levels(&self) -> Vec<IsolationLevel>;
    fn create_state(&self, isolation_level: IsolationLevel) -> Box<dyn EngineState>;
    fn encode_constraints(&self, isolation_level: IsolationLevel, txn_count: usize,
                          op_count: usize) -> IsoSpecResult<SmtConstraintSet>;
    fn version_string(&self) -> &str;
    fn validate_schedule(&self, schedule: &Schedule, level: IsolationLevel) -> IsoSpecResult<ValidationResult>;
}

/// Mutable engine state during schedule execution.
pub trait EngineState: Send {
    fn begin_transaction(&mut self, txn_id: TransactionId, level: IsolationLevel) -> IsoSpecResult<()>;
    fn execute_operation(&mut self, op: &Operation) -> IsoSpecResult<OperationOutcome>;
    fn commit_transaction(&mut self, txn_id: TransactionId) -> IsoSpecResult<CommitOutcome>;
    fn abort_transaction(&mut self, txn_id: TransactionId) -> IsoSpecResult<()>;
    fn extract_dependencies(&self) -> Vec<Dependency>;
    fn active_transactions(&self) -> Vec<TransactionId>;
    fn transaction_status(&self, txn_id: TransactionId) -> Option<isospec_types::transaction::TransactionStatus>;
    fn snapshot_info(&self, txn_id: TransactionId) -> Option<SnapshotInfo>;
    fn reset(&mut self);
}

#[derive(Debug, Clone)]
pub struct OperationOutcome {
    pub success: bool,
    pub values_read: Vec<isospec_types::value::Value>,
    pub locks_acquired: Vec<isospec_types::identifier::LockId>,
    pub blocked: bool,
    pub blocked_by: Option<TransactionId>,
    pub message: Option<String>,
}

impl OperationOutcome {
    pub fn success() -> Self {
        Self { success: true, values_read: Vec::new(), locks_acquired: Vec::new(),
               blocked: false, blocked_by: None, message: None }
    }
    pub fn blocked(by: TransactionId) -> Self {
        Self { success: false, values_read: Vec::new(), locks_acquired: Vec::new(),
               blocked: true, blocked_by: Some(by), message: None }
    }
    pub fn with_value(mut self, v: isospec_types::value::Value) -> Self {
        self.values_read.push(v);
        self
    }
}

#[derive(Debug, Clone)]
pub struct CommitOutcome {
    pub committed: bool,
    pub abort_reason: Option<String>,
    pub commit_timestamp: u64,
    pub dependencies: Vec<Dependency>,
}

impl CommitOutcome {
    pub fn success(ts: u64) -> Self {
        Self { committed: true, abort_reason: None, commit_timestamp: ts, dependencies: Vec::new() }
    }
    pub fn aborted(reason: impl Into<String>) -> Self {
        Self { committed: false, abort_reason: Some(reason.into()), commit_timestamp: 0, dependencies: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    pub snapshot_time: u64,
    pub active_txns: Vec<TransactionId>,
    pub committed_before_snapshot: Vec<TransactionId>,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub violations: Vec<String>,
    pub schedule_accepted: bool,
    pub transactions_aborted: Vec<TransactionId>,
    pub dependencies_found: Vec<Dependency>,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self { valid: true, violations: Vec::new(), schedule_accepted: true,
               transactions_aborted: Vec::new(), dependencies_found: Vec::new() }
    }
    pub fn invalid(violations: Vec<String>) -> Self {
        Self { valid: false, violations, schedule_accepted: false,
               transactions_aborted: Vec::new(), dependencies_found: Vec::new() }
    }
}

/// Registry of engine models.
pub struct EngineRegistry {
    engines: HashMap<EngineKind, Box<dyn EngineModel>>,
}

impl EngineRegistry {
    pub fn new() -> Self { Self { engines: HashMap::new() } }

    pub fn register(&mut self, model: Box<dyn EngineModel>) {
        let kind = model.engine_kind();
        self.engines.insert(kind, model);
    }

    pub fn get(&self, kind: EngineKind) -> Option<&dyn EngineModel> {
        self.engines.get(&kind).map(|m| m.as_ref())
    }

    pub fn registered_engines(&self) -> Vec<EngineKind> {
        self.engines.keys().copied().collect()
    }
}

impl Default for EngineRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_operation_outcome() {
        let outcome = OperationOutcome::success();
        assert!(outcome.success);
        assert!(!outcome.blocked);
    }
    #[test]
    fn test_commit_outcome() {
        let ok = CommitOutcome::success(100);
        assert!(ok.committed);
        let fail = CommitOutcome::aborted("serialization failure");
        assert!(!fail.committed);
    }
    #[test]
    fn test_engine_registry() {
        let registry = EngineRegistry::new();
        assert!(registry.registered_engines().is_empty());
    }
}
