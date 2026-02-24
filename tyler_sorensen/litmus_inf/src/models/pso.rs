//! Partial Store Order (PSO) memory model for LITMUS∞.
//!
//! PSO (SPARC-style) extends TSO by also relaxing W→W ordering to
//! different addresses. Each address has its own store buffer, so
//! stores to different locations can be reordered. This module provides
//! PSO axiom definitions, per-variable store buffer semantics, and
//! violation detection.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::{
    LitmusTest, Outcome, LitmusOutcome,
    MemoryModel, RelationExpr, BuiltinModel,
    ExecutionGraph, BitMatrix,
};
use crate::checker::execution::{EventId, Address, OpType};
use crate::checker::memory_model::PredicateExpr;

// ---------------------------------------------------------------------------
// PsoAxiom
// ---------------------------------------------------------------------------

/// Axioms of Partial Store Order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PsoAxiom {
    /// Per-address store ordering.
    PerAddressStoreOrder,
    /// Store forwarding from per-address buffer.
    StoreForwarding,
    /// Read-to-read ordering preserved.
    ReadReadOrder,
    /// Read-to-write ordering preserved.
    ReadWriteOrder,
    /// Po-loc (program order same location) preserved.
    PoLocOrder,
    /// Fence ordering.
    FenceOrdering,
}

impl fmt::Display for PsoAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PerAddressStoreOrder => write!(f, "pso-per-addr"),
            Self::StoreForwarding => write!(f, "pso-forwarding"),
            Self::ReadReadOrder => write!(f, "pso-rr"),
            Self::ReadWriteOrder => write!(f, "pso-rw"),
            Self::PoLocOrder => write!(f, "pso-po-loc"),
            Self::FenceOrdering => write!(f, "pso-fence"),
        }
    }
}

impl PsoAxiom {
    /// All PSO axioms.
    pub fn all() -> Vec<Self> {
        vec![
            Self::PerAddressStoreOrder,
            Self::StoreForwarding,
            Self::ReadReadOrder,
            Self::ReadWriteOrder,
            Self::PoLocOrder,
            Self::FenceOrdering,
        ]
    }
}

// ---------------------------------------------------------------------------
// PsoViolation
// ---------------------------------------------------------------------------

/// A violation of a PSO axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsoViolation {
    /// Which axiom was violated.
    pub axiom: PsoAxiom,
    /// Events involved.
    pub events: Vec<EventId>,
    /// Description of the violation.
    pub description: String,
}

impl fmt::Display for PsoViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PSO violation [{}]: {} [events: {:?}]",
            self.axiom, self.description, self.events)
    }
}

// ---------------------------------------------------------------------------
// PerAddressStoreBuffer
// ---------------------------------------------------------------------------

/// Per-address store buffer (PSO has one buffer per address per thread).
#[derive(Debug, Clone)]
pub struct PerAddressStoreBuffer {
    /// Per-address queues: addr → FIFO queue of values.
    buffers: HashMap<Address, VecDeque<u64>>,
    max_size_per_addr: usize,
}

impl PerAddressStoreBuffer {
    /// Create with a maximum buffer size per address.
    pub fn new(max_size_per_addr: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            max_size_per_addr,
        }
    }

    /// Push a store into the per-address buffer.
    pub fn push(&mut self, addr: Address, val: u64) {
        let buf = self.buffers.entry(addr).or_insert_with(VecDeque::new);
        if buf.len() >= self.max_size_per_addr {
            buf.pop_front();
        }
        buf.push_back(val);
    }

    /// Lookup the most recent store to `addr` (forwarding).
    pub fn lookup(&self, addr: Address) -> Option<u64> {
        self.buffers.get(&addr)
            .and_then(|buf| buf.back().copied())
    }

    /// Flush the oldest store for a specific address.
    pub fn flush_one(&mut self, addr: Address) -> Option<u64> {
        self.buffers.get_mut(&addr)
            .and_then(|buf| buf.pop_front())
    }

    /// Flush all stores for a specific address.
    pub fn flush_addr(&mut self, addr: Address) -> Vec<u64> {
        self.buffers.remove(&addr)
            .map(|buf| buf.into_iter().collect())
            .unwrap_or_default()
    }

    /// Flush all stores for all addresses (e.g., on STBAR/fence).
    pub fn flush_all(&mut self) -> Vec<(Address, u64)> {
        let mut result = Vec::new();
        for (addr, buf) in self.buffers.drain() {
            for val in buf {
                result.push((addr, val));
            }
        }
        result
    }

    /// Whether all buffers are empty.
    pub fn is_empty(&self) -> bool {
        self.buffers.values().all(|b| b.is_empty())
    }

    /// Total number of entries across all buffers.
    pub fn total_entries(&self) -> usize {
        self.buffers.values().map(|b| b.len()).sum()
    }

    /// Number of active address buffers.
    pub fn active_addresses(&self) -> usize {
        self.buffers.values().filter(|b| !b.is_empty()).count()
    }

    /// Check if a specific address has buffered stores.
    pub fn has_pending(&self, addr: Address) -> bool {
        self.buffers.get(&addr).map_or(false, |b| !b.is_empty())
    }
}

// ---------------------------------------------------------------------------
// PsoModel
// ---------------------------------------------------------------------------

/// Partial Store Order model checker.
#[derive(Debug, Clone)]
pub struct PsoModel {
    pub name: String,
    pub max_buffer_size: usize,
    model: MemoryModel,
}

impl PsoModel {
    /// Create a new PSO model.
    pub fn new() -> Self {
        Self {
            name: "PSO".to_string(),
            max_buffer_size: 16,
            model: BuiltinModel::PSO.build(),
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

    /// Build the PSO memory model from scratch.
    pub fn build_model() -> MemoryModel {
        let mut m = MemoryModel::new("PSO");

        m.add_derived("com",
            RelationExpr::union_many(vec![
                RelationExpr::base("rf"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "communication",
        );

        // PSO ppo: R→R ∪ R→W ∪ po-loc (no W→W to different addresses)
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
                RelationExpr::base("po-loc"),
            ]),
            "preserved program order for PSO",
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

        m
    }

    /// Check whether an execution is PSO-consistent.
    pub fn check(&self, exec: &ExecutionGraph) -> Result<(), Vec<PsoViolation>> {
        let mut violations = Vec::new();
        violations.extend(self.check_ppo(exec));
        if violations.is_empty() { Ok(()) } else { Err(violations) }
    }

    /// Check preserved program order.
    pub fn check_ppo(&self, exec: &ExecutionGraph) -> Vec<PsoViolation> {
        let env = self.model.compute_derived(exec);
        let ghb = env.get("ghb").cloned().unwrap_or_else(|| BitMatrix::new(exec.len()));
        let tc = ghb.transitive_closure();
        let n = exec.len();
        let mut violations = Vec::new();
        for i in 0..n {
            if tc.get(i, i) {
                violations.push(PsoViolation {
                    axiom: PsoAxiom::PerAddressStoreOrder,
                    events: vec![i],
                    description: format!("ghb cycle involving e{}", i),
                });
                break;
            }
        }
        violations
    }

    /// Check a litmus test against PSO.
    pub fn check_test(&self, test: &LitmusTest) -> PsoTestResult {
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

        PsoTestResult {
            test_name: test.name.clone(),
            total_executions: executions.len(),
            consistent_executions: executions.len() - inconsistent_count,
            inconsistent_executions: inconsistent_count,
            consistent_outcomes,
            forbidden_observed,
        }
    }

    /// Create per-thread per-address store buffers.
    pub fn create_buffers(&self, n_threads: usize) -> Vec<PerAddressStoreBuffer> {
        (0..n_threads).map(|_| PerAddressStoreBuffer::new(self.max_buffer_size)).collect()
    }

    /// Determine if a W→W reorder to different addresses is allowed.
    pub fn allows_ww_reorder(&self, exec: &ExecutionGraph, w1: EventId, w2: EventId) -> bool {
        let e1 = &exec.events[w1];
        let e2 = &exec.events[w2];
        if e1.thread != e2.thread { return false; }
        if !e1.is_write() || !e2.is_write() { return false; }
        // PSO allows W→W reorder if different addresses and no fence.
        if e1.address == e2.address { return false; }
        let n = exec.len();
        for f in 0..n {
            if exec.events[f].is_fence()
                && exec.events[f].thread == e1.thread
                && exec.po.get(w1, f)
                && exec.po.get(f, w2)
            {
                return false;
            }
        }
        true
    }
}

impl Default for PsoModel {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// PsoTestResult
// ---------------------------------------------------------------------------

/// Result of checking a test against PSO.
#[derive(Debug, Clone)]
pub struct PsoTestResult {
    pub test_name: String,
    pub total_executions: usize,
    pub consistent_executions: usize,
    pub inconsistent_executions: usize,
    pub consistent_outcomes: Vec<Outcome>,
    pub forbidden_observed: bool,
}

impl fmt::Display for PsoTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PSO check for '{}': {}/{} consistent",
            self.test_name,
            self.consistent_executions,
            self.total_executions,
        )?;
        writeln!(f, "  {} distinct outcomes", self.consistent_outcomes.len())?;
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
    fn test_pso_axiom_all() {
        let axioms = PsoAxiom::all();
        assert_eq!(axioms.len(), 6);
    }

    #[test]
    fn test_pso_axiom_display() {
        assert!(format!("{}", PsoAxiom::PerAddressStoreOrder).contains("per-addr"));
    }

    #[test]
    fn test_pso_model_creation() {
        let model = PsoModel::new();
        assert_eq!(model.name, "PSO");
    }

    #[test]
    fn test_pso_with_buffer_size() {
        let model = PsoModel::with_buffer_size(8);
        assert_eq!(model.max_buffer_size, 8);
    }

    #[test]
    fn test_pso_build_model() {
        let model = PsoModel::build_model();
        assert_eq!(model.name, "PSO");
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_pso_violation_display() {
        let v = PsoViolation {
            axiom: PsoAxiom::PerAddressStoreOrder,
            events: vec![0],
            description: "test".into(),
        };
        assert!(format!("{}", v).contains("PSO violation"));
    }

    #[test]
    fn test_per_addr_buffer_new() {
        let buf = PerAddressStoreBuffer::new(8);
        assert!(buf.is_empty());
        assert_eq!(buf.total_entries(), 0);
    }

    #[test]
    fn test_per_addr_buffer_push() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 42);
        assert_eq!(buf.total_entries(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_per_addr_buffer_lookup() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 42);
        buf.push(0x200, 7);
        assert_eq!(buf.lookup(0x100), Some(42));
        assert_eq!(buf.lookup(0x200), Some(7));
        assert_eq!(buf.lookup(0x300), None);
    }

    #[test]
    fn test_per_addr_buffer_latest() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x100, 2);
        assert_eq!(buf.lookup(0x100), Some(2));
    }

    #[test]
    fn test_per_addr_buffer_flush_one() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x100, 2);
        let val = buf.flush_one(0x100).unwrap();
        assert_eq!(val, 1);
        assert_eq!(buf.total_entries(), 1);
    }

    #[test]
    fn test_per_addr_buffer_flush_addr() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x100, 2);
        buf.push(0x200, 3);
        let vals = buf.flush_addr(0x100);
        assert_eq!(vals, vec![1, 2]);
        assert_eq!(buf.total_entries(), 1);
    }

    #[test]
    fn test_per_addr_buffer_flush_all() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x200, 2);
        let entries = buf.flush_all();
        assert_eq!(entries.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_per_addr_buffer_active_addresses() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 1);
        buf.push(0x200, 2);
        assert_eq!(buf.active_addresses(), 2);
    }

    #[test]
    fn test_per_addr_buffer_has_pending() {
        let mut buf = PerAddressStoreBuffer::new(8);
        buf.push(0x100, 1);
        assert!(buf.has_pending(0x100));
        assert!(!buf.has_pending(0x200));
    }

    #[test]
    fn test_per_addr_buffer_overflow() {
        let mut buf = PerAddressStoreBuffer::new(2);
        buf.push(0x100, 1);
        buf.push(0x100, 2);
        buf.push(0x100, 3);
        assert_eq!(buf.total_entries(), 2);
    }

    #[test]
    fn test_pso_check_empty() {
        let model = PsoModel::new();
        let builder = ExecutionGraphBuilder::new();
        let exec = builder.build();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_pso_check_simple_rf() {
        let model = PsoModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        exec.derive_fr();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_pso_check_test_sb() {
        let model = PsoModel::new();
        let test = crate::checker::litmus::sb_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
    }

    #[test]
    fn test_pso_check_test_mp() {
        let model = PsoModel::new();
        let test = crate::checker::litmus::mp_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
    }

    #[test]
    fn test_pso_create_buffers() {
        let model = PsoModel::new();
        let buffers = model.create_buffers(4);
        assert_eq!(buffers.len(), 4);
    }

    #[test]
    fn test_pso_test_result_display() {
        let result = PsoTestResult {
            test_name: "test".into(),
            total_executions: 10,
            consistent_executions: 8,
            inconsistent_executions: 2,
            consistent_outcomes: vec![],
            forbidden_observed: false,
        };
        let s = format!("{}", result);
        assert!(s.contains("PSO check"));
    }

    #[test]
    fn test_pso_default() {
        let model = PsoModel::default();
        assert_eq!(model.name, "PSO");
    }

    #[test]
    fn test_pso_model_ref() {
        let model = PsoModel::new();
        assert_eq!(model.model().name, "PSO");
    }
}
