//! Total Store Order (TSO) memory model for LITMUS∞.
//!
//! TSO (x86-style) relaxes W→R ordering within the same thread via store
//! buffers. All other orderings are preserved. This module provides TSO
//! axiom definitions, store buffer semantics, and violation detection.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::{
    LitmusTest, Outcome, LitmusOutcome,
    MemoryModel, RelationExpr, BuiltinModel,
    ExecutionGraph, BitMatrix,
};
use crate::checker::execution::{EventId, OpType};
use crate::checker::memory_model::PredicateExpr;

// ---------------------------------------------------------------------------
// TsoAxiom
// ---------------------------------------------------------------------------

/// Axioms of Total Store Order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TsoAxiom {
    /// FIFO store buffer: stores drain in order.
    StoreBufferFIFO,
    /// Store forwarding: a load can read from the local store buffer.
    StoreForwarding,
    /// Total ordering of stores to the same location.
    TotalStoreOrder,
    /// Preserved program order (everything except W→R).
    PreservedProgramOrder,
    /// Fence ordering: MFENCE restores W→R ordering.
    FenceOrdering,
}

impl fmt::Display for TsoAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoreBufferFIFO => write!(f, "tso-fifo"),
            Self::StoreForwarding => write!(f, "tso-forwarding"),
            Self::TotalStoreOrder => write!(f, "tso-total-order"),
            Self::PreservedProgramOrder => write!(f, "tso-ppo"),
            Self::FenceOrdering => write!(f, "tso-fence"),
        }
    }
}

impl TsoAxiom {
    /// All TSO axioms.
    pub fn all() -> Vec<Self> {
        vec![
            Self::StoreBufferFIFO,
            Self::StoreForwarding,
            Self::TotalStoreOrder,
            Self::PreservedProgramOrder,
            Self::FenceOrdering,
        ]
    }
}

// ---------------------------------------------------------------------------
// TsoViolation
// ---------------------------------------------------------------------------

/// A violation of a TSO axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsoViolation {
    /// Which axiom was violated.
    pub axiom: TsoAxiom,
    /// Events involved.
    pub events: Vec<EventId>,
    /// Description of the violation.
    pub description: String,
}

impl fmt::Display for TsoViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TSO violation [{}]: {} [events: {:?}]",
            self.axiom, self.description, self.events)
    }
}

// ---------------------------------------------------------------------------
// StoreBuffer
// ---------------------------------------------------------------------------

/// Per-thread store buffer for TSO simulation.
#[derive(Debug, Clone)]
pub struct StoreBuffer {
    entries: VecDeque<StoreBufferEntry>,
    max_size: usize,
}

/// An entry in the store buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreBufferEntry {
    pub addr: u64,
    pub value: u64,
    pub event_id: Option<EventId>,
}

impl StoreBuffer {
    /// Create a new store buffer with a maximum size.
    pub fn new(max_size: usize) -> Self {
        Self { entries: VecDeque::new(), max_size }
    }

    /// Push a store into the buffer.
    pub fn push(&mut self, addr: u64, val: u64) {
        if self.entries.len() >= self.max_size {
            self.entries.pop_front();
        }
        self.entries.push_back(StoreBufferEntry {
            addr,
            value: val,
            event_id: None,
        });
    }

    /// Push a store with an event ID.
    pub fn push_with_id(&mut self, addr: u64, val: u64, id: EventId) {
        if self.entries.len() >= self.max_size {
            self.entries.pop_front();
        }
        self.entries.push_back(StoreBufferEntry {
            addr,
            value: val,
            event_id: Some(id),
        });
    }

    /// Lookup the most recent store to `addr` in the buffer (forwarding).
    pub fn lookup(&self, addr: u64) -> Option<u64> {
        self.entries.iter().rev().find(|e| e.addr == addr).map(|e| e.value)
    }

    /// Flush the oldest entry (drain to memory).
    pub fn flush_one(&mut self) -> Option<StoreBufferEntry> {
        self.entries.pop_front()
    }

    /// Flush all entries for a given address.
    pub fn flush_addr(&mut self, addr: u64) -> Vec<StoreBufferEntry> {
        let mut flushed = Vec::new();
        self.entries.retain(|e| {
            if e.addr == addr {
                flushed.push(e.clone());
                false
            } else {
                true
            }
        });
        flushed
    }

    /// Flush all entries (e.g., on MFENCE).
    pub fn flush_all(&mut self) -> Vec<StoreBufferEntry> {
        self.entries.drain(..).collect()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Number of entries.
    pub fn len(&self) -> usize { self.entries.len() }

    /// Check if any entry targets the given address.
    pub fn contains_addr(&self, addr: u64) -> bool {
        self.entries.iter().any(|e| e.addr == addr)
    }

    /// Get all entries (for inspection).
    pub fn entries(&self) -> &VecDeque<StoreBufferEntry> {
        &self.entries
    }
}

// ---------------------------------------------------------------------------
// TsoModel
// ---------------------------------------------------------------------------

/// Total Store Order model checker.
#[derive(Debug, Clone)]
pub struct TsoModel {
    pub name: String,
    pub max_buffer_size: usize,
    model: MemoryModel,
}

impl TsoModel {
    /// Create a new TSO model.
    pub fn new() -> Self {
        Self {
            name: "TSO".to_string(),
            max_buffer_size: 32,
            model: BuiltinModel::TSO.build(),
        }
    }

    /// Create with custom buffer size.
    pub fn with_buffer_size(max_size: usize) -> Self {
        Self {
            max_buffer_size: max_size,
            ..Self::new()
        }
    }

    /// Get the underlying memory model.
    pub fn model(&self) -> &MemoryModel {
        &self.model
    }

    /// Build the TSO memory model from scratch.
    pub fn build_model() -> MemoryModel {
        let mut m = MemoryModel::new("TSO");

        m.add_derived("com",
            RelationExpr::union_many(vec![
                RelationExpr::base("rf"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "communication",
        );

        // ppo = R→R ∪ R→W ∪ W→W ∪ po-loc
        m.add_derived("ppo",
            RelationExpr::union_many(vec![
                RelationExpr::seq_many(vec![
                    RelationExpr::filter(PredicateExpr::IsRead),
                    RelationExpr::base("po"),
                    RelationExpr::filter(PredicateExpr::IsRead),
                ]),
                RelationExpr::seq_many(vec![
                    RelationExpr::filter(PredicateExpr::IsRead),
                    RelationExpr::base("po"),
                    RelationExpr::filter(PredicateExpr::IsWrite),
                ]),
                RelationExpr::seq_many(vec![
                    RelationExpr::filter(PredicateExpr::IsWrite),
                    RelationExpr::base("po"),
                    RelationExpr::filter(PredicateExpr::IsWrite),
                ]),
                RelationExpr::base("po-loc"),
            ]),
            "preserved program order for TSO",
        );

        m.add_derived("fence-order",
            RelationExpr::seq_many(vec![
                RelationExpr::base("po"),
                RelationExpr::filter(PredicateExpr::IsFence),
                RelationExpr::base("po"),
            ]),
            "ordering induced by fences",
        );

        m.add_derived("ghb",
            RelationExpr::union_many(vec![
                RelationExpr::base("ppo"),
                RelationExpr::base("fence-order"),
                RelationExpr::base("rfe"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "global happens-before",
        );

        m.add_acyclic(RelationExpr::base("ghb"));
        m.add_irreflexive(
            RelationExpr::seq(RelationExpr::base("fre"), RelationExpr::base("rfe")),
        );

        m
    }

    /// Check whether an execution is TSO-consistent.
    pub fn check(&self, exec: &ExecutionGraph) -> Result<(), Vec<TsoViolation>> {
        let mut violations = Vec::new();
        violations.extend(self.check_ppo(exec));
        violations.extend(self.check_coherence(exec));
        if violations.is_empty() { Ok(()) } else { Err(violations) }
    }

    /// Check preserved program order.
    pub fn check_ppo(&self, exec: &ExecutionGraph) -> Vec<TsoViolation> {
        let env = self.model.compute_derived(exec);
        let ghb = env.get("ghb").cloned().unwrap_or_else(|| BitMatrix::new(exec.len()));
        let tc = ghb.transitive_closure();
        let n = exec.len();
        let mut violations = Vec::new();
        for i in 0..n {
            if tc.get(i, i) {
                violations.push(TsoViolation {
                    axiom: TsoAxiom::PreservedProgramOrder,
                    events: vec![i],
                    description: format!("ghb cycle involving e{}", i),
                });
                break;
            }
        }
        violations
    }

    /// Check per-location coherence.
    pub fn check_coherence(&self, exec: &ExecutionGraph) -> Vec<TsoViolation> {
        let fre = exec.external(&exec.fr);
        let rfe = exec.external(&exec.rf);
        let fre_rfe = fre.compose(&rfe);
        let n = exec.len();
        let mut violations = Vec::new();
        for i in 0..n {
            if fre_rfe.get(i, i) {
                violations.push(TsoViolation {
                    axiom: TsoAxiom::TotalStoreOrder,
                    events: vec![i],
                    description: format!("fre;rfe reflexivity violation at e{}", i),
                });
                break;
            }
        }
        violations
    }

    /// Check a litmus test against TSO.
    pub fn check_test(&self, test: &LitmusTest) -> TsoTestResult {
        let executions = test.enumerate_executions();
        let mut consistent_outcomes: Vec<Outcome> = Vec::new();
        let mut inconsistent_count = 0;

        for (exec, regs, mem) in &executions {
            match self.check(exec) {
                Ok(()) => {
                    let outcome = Outcome {
                        registers: regs.clone(),
                        memory: mem.clone(),
                    };
                    if !consistent_outcomes.iter().any(|o| o == &outcome) {
                        consistent_outcomes.push(outcome);
                    }
                }
                Err(_) => { inconsistent_count += 1; }
            }
        }

        let mut forbidden_observed = false;
        for (outcome, kind) in &test.expected_outcomes {
            if *kind == LitmusOutcome::Forbidden {
                if consistent_outcomes.iter().any(|o| outcome.matches(&o.registers, &o.memory)) {
                    forbidden_observed = true;
                }
            }
        }

        TsoTestResult {
            test_name: test.name.clone(),
            total_executions: executions.len(),
            consistent_executions: executions.len() - inconsistent_count,
            inconsistent_executions: inconsistent_count,
            consistent_outcomes,
            forbidden_observed,
        }
    }

    /// Create per-thread store buffers.
    pub fn create_buffers(&self, n_threads: usize) -> Vec<StoreBuffer> {
        (0..n_threads).map(|_| StoreBuffer::new(self.max_buffer_size)).collect()
    }

    /// Determine if a W→R reordering would be visible (TSO allows this).
    pub fn allows_wr_reorder(&self, exec: &ExecutionGraph, w: EventId, r: EventId) -> bool {
        let ew = &exec.events[w];
        let er = &exec.events[r];
        // TSO allows W→R reordering if same thread, different addresses, no fence between.
        if ew.thread != er.thread { return false; }
        if ew.address == er.address { return false; }
        if !ew.is_write() || !er.is_read() { return false; }
        // Check no fence between w and r in po.
        let n = exec.len();
        for f in 0..n {
            if exec.events[f].is_fence()
                && exec.events[f].thread == ew.thread
                && exec.po.get(w, f)
                && exec.po.get(f, r)
            {
                return false;
            }
        }
        true
    }
}

impl Default for TsoModel {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// TsoTestResult
// ---------------------------------------------------------------------------

/// Result of checking a test against TSO.
#[derive(Debug, Clone)]
pub struct TsoTestResult {
    pub test_name: String,
    pub total_executions: usize,
    pub consistent_executions: usize,
    pub inconsistent_executions: usize,
    pub consistent_outcomes: Vec<Outcome>,
    pub forbidden_observed: bool,
}

impl fmt::Display for TsoTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TSO check for '{}': {}/{} consistent",
            self.test_name,
            self.consistent_executions,
            self.total_executions,
        )?;
        writeln!(f, "  {} distinct outcomes", self.consistent_outcomes.len())?;
        for (i, o) in self.consistent_outcomes.iter().enumerate() {
            writeln!(f, "    [{}] {}", i, o)?;
        }
        if self.forbidden_observed {
            writeln!(f, "  ⚠ forbidden outcome observed!")?;
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::execution::ExecutionGraphBuilder;

    #[test]
    fn test_tso_axiom_all() {
        let axioms = TsoAxiom::all();
        assert_eq!(axioms.len(), 5);
    }

    #[test]
    fn test_tso_axiom_display() {
        assert!(format!("{}", TsoAxiom::StoreBufferFIFO).contains("fifo"));
    }

    #[test]
    fn test_tso_model_creation() {
        let model = TsoModel::new();
        assert_eq!(model.name, "TSO");
    }

    #[test]
    fn test_tso_with_buffer_size() {
        let model = TsoModel::with_buffer_size(16);
        assert_eq!(model.max_buffer_size, 16);
    }

    #[test]
    fn test_tso_build_model() {
        let model = TsoModel::build_model();
        assert_eq!(model.name, "TSO");
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_tso_violation_display() {
        let v = TsoViolation {
            axiom: TsoAxiom::StoreBufferFIFO,
            events: vec![0],
            description: "test".into(),
        };
        assert!(format!("{}", v).contains("TSO violation"));
    }

    #[test]
    fn test_store_buffer_new() {
        let buf = StoreBuffer::new(8);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_store_buffer_push() {
        let mut buf = StoreBuffer::new(8);
        buf.push(0x100, 42);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_store_buffer_lookup() {
        let mut buf = StoreBuffer::new(8);
        buf.push(0x100, 42);
        buf.push(0x200, 7);
        assert_eq!(buf.lookup(0x100), Some(42));
        assert_eq!(buf.lookup(0x200), Some(7));
        assert_eq!(buf.lookup(0x300), None);
    }

    #[test]
    fn test_store_buffer_forwarding_latest() {
        let mut buf = StoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x100, 2);
        assert_eq!(buf.lookup(0x100), Some(2));
    }

    #[test]
    fn test_store_buffer_flush_one() {
        let mut buf = StoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x200, 2);
        let entry = buf.flush_one().unwrap();
        assert_eq!(entry.addr, 0x100);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_store_buffer_flush_all() {
        let mut buf = StoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x200, 2);
        let entries = buf.flush_all();
        assert_eq!(entries.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_store_buffer_flush_addr() {
        let mut buf = StoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x200, 2);
        buf.push(0x100, 3);
        let flushed = buf.flush_addr(0x100);
        assert_eq!(flushed.len(), 2);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_store_buffer_overflow() {
        let mut buf = StoreBuffer::new(2);
        buf.push(0x100, 1);
        buf.push(0x200, 2);
        buf.push(0x300, 3);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.lookup(0x100), None);
    }

    #[test]
    fn test_store_buffer_contains_addr() {
        let mut buf = StoreBuffer::new(8);
        buf.push(0x100, 1);
        assert!(buf.contains_addr(0x100));
        assert!(!buf.contains_addr(0x200));
    }

    #[test]
    fn test_store_buffer_push_with_id() {
        let mut buf = StoreBuffer::new(8);
        buf.push_with_id(0x100, 1, 42);
        let entry = buf.entries().front().unwrap();
        assert_eq!(entry.event_id, Some(42));
    }

    #[test]
    fn test_tso_check_empty() {
        let model = TsoModel::new();
        let builder = ExecutionGraphBuilder::new();
        let exec = builder.build();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_tso_check_simple_rf() {
        let model = TsoModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        exec.derive_fr();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_tso_check_test_sb() {
        let model = TsoModel::new();
        let test = crate::checker::litmus::sb_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
    }

    #[test]
    fn test_tso_check_test_mp() {
        let model = TsoModel::new();
        let test = crate::checker::litmus::mp_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
    }

    #[test]
    fn test_tso_create_buffers() {
        let model = TsoModel::new();
        let buffers = model.create_buffers(4);
        assert_eq!(buffers.len(), 4);
        for buf in &buffers {
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn test_tso_test_result_display() {
        let result = TsoTestResult {
            test_name: "test".into(),
            total_executions: 10,
            consistent_executions: 8,
            inconsistent_executions: 2,
            consistent_outcomes: vec![],
            forbidden_observed: false,
        };
        let s = format!("{}", result);
        assert!(s.contains("TSO check"));
    }

    #[test]
    fn test_tso_default() {
        let model = TsoModel::default();
        assert_eq!(model.name, "TSO");
        assert_eq!(model.max_buffer_size, 32);
    }

    #[test]
    fn test_tso_model_ref() {
        let model = TsoModel::new();
        assert_eq!(model.model().name, "TSO");
    }
}
